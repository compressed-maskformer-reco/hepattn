"""Batched linear assignment solved on whichever device the costs already live on.

The CLIC training step hands the matcher ~10k independent assignment problems at once
(batch x the 4 decoder layers plus the final head), each at most a few hundred rows square.
The host solvers in :mod:`hepattn.models.matcher` need that whole stack copied device->host
first, and CPU-side parallelism over it saturates at ~16 threads. This module solves the same
problems in place on the GPU so the copy and the host stall disappear.

The algorithm is Bertsekas' auction with epsilon-scaling, in the Jacobi (all-unassigned-bid-at-
once) form, which is what makes it data-parallel. Rows bid for columns; a column goes to its
highest bidder and its price rises to the winning bid, which forces the evicted row to look
elsewhere. Bidding by the gap between a row's best and second-best column, plus ``eps``,
guarantees termination and leaves the assignment within ``n_rows * eps`` of optimal.

Costs are affine-normalised per problem before solving. This is exact -- every permutation
sums exactly ``n_rows`` entries, so ``(c - min) / range`` is a monotone map on total cost --
and it is what keeps the auction numerically sane: prices accumulate in units of the cost
range, so the raw sentinel that :class:`~hepattn.models.matcher.Matcher` uses for forbidden
assignments (``float32_max / 10``) would otherwise destroy the fp32 dynamic range.

Termination at ``eps > 0`` means the result is epsilon-optimal rather than provably optimal,
so :func:`batched_auction` reports which problems it actually solved and the caller is
expected to send the rest to an exact host solver.

Note on epsilon-scaling, which is not used by default and is the one real trap here. Auction
is usually run as a sequence of phases with a shrinking ``eps``, carrying the prices forward
and restarting the assignment. **That is wrong for the rectangular problems the matcher
actually poses.** The dual of the rectangular assignment problem requires ``p_j > 0`` only for
columns that are assigned; a forward auction maintains this within one phase, because a column
is priced only by being won and is never released. Carrying prices across a restart breaks it:
a column can enter a phase priced from the last one and end this one unassigned. Measured on
uniform random costs, scaling-with-carry is exact at ``num_rows == num_cols`` and badly wrong
below it -- gaps of ~1 on costs normalised to [0, 1], not the ``num_rows * eps`` the theory
promises. Resetting the prices each phase restores exactness but costs more rounds than a
single phase does, and a single phase is *also* the fastest option whenever the problem is
rectangular, which is the matcher's case (~50 valid targets into 150 query slots). Hence the
default of one phase, and the price reset for anyone who sets a schedule anyway.
"""

import torch
from torch import Tensor

__all__ = ["assignment_to_permutation", "batched_auction", "batched_jv", "require_jv"]


def _normalise(costs: Tensor, allowed: Tensor, forbidden_cost: float) -> Tensor:
    """Map each problem's allowed costs onto [0, 1] and its forbidden ones onto a sentinel.

    Args:
        costs: [batch, num_rows, num_cols] cost matrices.
        allowed: [batch, num_rows, num_cols] bool, False where a row may not take a column.
        forbidden_cost: Value to write into the disallowed entries. Must exceed the largest
            achievable total cost of a feasible assignment so that a forbidden pairing is
            never preferred to a feasible one.

    Returns:
        The normalised costs, same shape and dtype as ``costs``.
    """
    inf = torch.inf
    hi = torch.where(allowed, costs, torch.full_like(costs, -inf)).amax(dim=(1, 2))
    lo = torch.where(allowed, costs, torch.full_like(costs, inf)).amin(dim=(1, 2))

    # A problem with no allowed entry at all (e.g. no valid targets) leaves hi/lo infinite;
    # it has nothing to solve, so any finite affine map will do.
    degenerate = ~torch.isfinite(hi) | ~torch.isfinite(lo)
    lo = torch.where(degenerate, torch.zeros_like(lo), lo)
    scale = torch.where(degenerate, torch.ones_like(hi), hi - lo).clamp_min(torch.finfo(costs.dtype).tiny)

    normalised = (costs - lo[:, None, None]) / scale[:, None, None]
    return torch.where(allowed, normalised, torch.full_like(normalised, forbidden_cost))


@torch.no_grad()
def batched_auction(
    costs: Tensor,
    row_valid: Tensor,
    col_allowed: Tensor | None = None,
    eps_start: float = 1e-6,
    eps_final: float = 1e-6,
    eps_decay: float = 0.2,
    max_iters: int = 10_000,
    check_interval: int = 8,
) -> tuple[Tensor, Tensor]:
    """Solve a batch of rectangular linear assignment problems by auction.

    Every valid row is assigned a distinct column, minimising the total cost. There must be at
    least as many columns as valid rows in each problem, which is guaranteed here because the
    matcher only ever matches targets (rows) to a larger pool of queries (columns).

    Args:
        costs: [batch, num_rows, num_cols] cost matrices. Non-finite entries are treated as
            forbidden assignments rather than propagating NaN.
        row_valid: [batch, num_rows] bool marking the rows that need an assignment. Padded
            rows never bid and come back as -1.
        col_allowed: Optional [batch, num_cols] bool marking the columns that may be assigned.
            Columns that are False are left for the unmatched remainder.
        eps_start: Bidding increment for the first epsilon-scaling phase, in units of the
            per-problem cost range. Equal to ``eps_final`` by default, i.e. a single phase;
            see the module docstring for why scaling rarely pays here.
        eps_final: Final bidding increment. The returned assignment is within
            ``num_valid_rows * eps_final`` of optimal in normalised cost units.
        eps_decay: Factor by which ``eps`` shrinks between scaling phases.
        max_iters: Cap on bidding rounds per phase, after which the problem is reported
            unsolved rather than looping forever on a degenerate cost matrix.
        check_interval: How often, in rounds, to test for completion. Each test is a device
            sync, so this trades a little wasted work against a lot of latency.

    Returns:
        Tuple of the [batch, num_rows] column assigned to each row (-1 where unassigned) and a
        [batch] bool marking the problems in which every valid row got a column. Problems that
        are False must be re-solved by an exact host solver.

    Raises:
        ValueError: If any problem has more valid rows than assignable columns.
    """
    if costs.ndim != 3:
        raise ValueError(f"Expected costs of shape [batch, num_rows, num_cols], got {tuple(costs.shape)}")

    batch, num_rows, num_cols = costs.shape
    device = costs.device
    dtype = costs.dtype if costs.dtype.is_floating_point else torch.float32
    costs = costs.to(dtype)

    row_valid = row_valid.to(device=device, dtype=torch.bool)
    n_valid_rows = row_valid.sum(dim=1)

    if col_allowed is None:
        col_allowed = torch.ones(batch, num_cols, dtype=torch.bool, device=device)
    else:
        col_allowed = col_allowed.to(device=device, dtype=torch.bool)

    if bool((n_valid_rows > col_allowed.sum(dim=1)).any()):
        raise ValueError("Some assignment problems have more valid rows than assignable columns")

    assigned = torch.full((batch, num_rows), -1, dtype=torch.long, device=device)
    if num_rows == 0 or num_cols == 0:
        return assigned, n_valid_rows == 0

    # A forbidden pairing must beat every feasible assignment, and a feasible one costs at most
    # num_rows in normalised units, so num_rows + 1 is sufficient and stays small enough to keep
    # the auction's prices well conditioned.
    allowed = col_allowed[:, None, :] & row_valid[:, :, None] & torch.isfinite(costs)
    benefit = -_normalise(costs, allowed, forbidden_cost=float(num_rows + 1))

    # A row that has no allowed column cannot be assigned at all; report those problems as
    # unsolved up front rather than letting them spin to the iteration cap.
    row_feasible = allowed.any(dim=-1) | ~row_valid

    row_idx = torch.arange(num_rows, device=device)
    col_idx = torch.arange(num_cols, device=device)
    # `assigned` and `owner` are written by scatter, and rows/columns that must not be touched
    # in a given round are aimed at a trailing scratch slot instead of being branched on.
    row_base = torch.arange(batch, device=device)[:, None] * (num_rows + 1)
    scratch_row = row_base + num_rows
    neg_inf = torch.finfo(dtype).min

    prices = torch.zeros(batch, num_cols, dtype=dtype, device=device)
    slots = torch.full((batch, num_rows + 1), -1, dtype=torch.long, device=device)
    flat_slots = slots.view(-1)
    owner = torch.full((batch, num_cols), -1, dtype=torch.long, device=device)
    clear = torch.full((batch * num_cols,), -1, dtype=torch.long, device=device)

    eps_values = []
    eps = max(eps_start, eps_final)
    while eps > eps_final:
        eps_values.append(eps)
        eps *= eps_decay
    eps_values.append(eps_final)

    for eps in eps_values:
        # Restart each phase from zero prices as well as an empty assignment. Carrying the
        # prices is the textbook version and is what makes scaling fast, but it is only valid
        # when every column ends up assigned -- see the module docstring.
        prices.zero_()
        slots.fill_(-1)
        owner.fill_(-1)

        for it in range(max_iters):
            unassigned = row_valid & row_feasible & (slots[:, :num_rows] < 0)
            if it % check_interval == 0 and not bool(unassigned.any()):
                break

            # Each unassigned row bids for its best column by the margin over its second best,
            # which is what stops a row from being outbid on a column it barely preferred.
            value = torch.where(unassigned[:, :, None], benefit - prices[:, None, :], torch.full_like(benefit, neg_inf))
            if num_cols > 1:
                best = value.topk(2, dim=-1)
                target, first, second = best.indices[..., 0], best.values[..., 0], best.values[..., 1]
            else:
                first, target = value.max(dim=-1)
                second = torch.full_like(first, neg_inf)

            margin = torch.where(torch.isfinite(second), first - second, torch.zeros_like(first)).clamp_min(0.0)
            bid = torch.where(unassigned, prices.gather(1, target) + margin + eps, torch.full_like(first, neg_inf))

            # Resolve the round: highest bid per column wins, ties broken by row index so that
            # exactly one row is declared the winner even on perfectly degenerate costs.
            col_bid = torch.full((batch, num_cols), neg_inf, dtype=dtype, device=device)
            col_bid.scatter_reduce_(1, target, bid, reduce="amax", include_self=True)
            is_top = unassigned & (bid >= col_bid.gather(1, target))
            contender = torch.where(is_top, row_idx.expand(batch, num_rows), torch.full_like(target, -1))
            winner = torch.full((batch, num_cols), -1, dtype=torch.long, device=device)
            winner.scatter_reduce_(1, target, contender, reduce="amax", include_self=True)
            won = winner >= 0

            prices = torch.where(won, col_bid, prices)

            # Evict the previous holders before seating the winners. A winner was unassigned
            # this round, so it can never also be an evictee.
            evicted = won & (owner >= 0)
            flat_slots.scatter_(0, torch.where(evicted, row_base + owner, scratch_row).reshape(-1), clear)
            flat_slots.scatter_(
                0,
                torch.where(won, row_base + winner, scratch_row).reshape(-1),
                torch.where(won, col_idx.expand(batch, num_cols), torch.full_like(winner, -1)).reshape(-1),
            )
            owner = torch.where(won, winner, owner)

    assigned = torch.where(row_valid, slots[:, :num_rows], torch.full_like(slots[:, :num_rows], -1))
    solved = ~(row_valid & (assigned < 0)).any(dim=1)
    return assigned, solved


def require_jv(device: torch.device | None = None):
    """Import the batched Jonker-Volgenant backend, or explain how to get one.

    Deliberately loud, and called at ``Matcher`` construction rather than mid-training. A
    missing build must not quietly leave the caller on the host solver, and a CPU-only build
    must not quietly solve device costs on the host -- ``batch_linear_assignment`` merely warns
    in that case. A silent degradation of exactly that kind (a ``lap1015`` extension built
    without ``FORCE_CUDA``) cost this study a day; see the study notes under
    ``experiments/clic/studies/b200_utilization/cuda_matcher``.

    Args:
        device: If given and on CUDA, also require that the extension was built with CUDA.

    Returns:
        The ``batch_linear_assignment`` entry point.

    Raises:
        RuntimeError: If the package is not importable, or was built without CUDA support and
            the costs are on a CUDA device.
    """
    try:
        # The private backend, for has_cuda(): a build-time property with no public accessor.
        import torch_linear_assignment._backend as backend  # noqa: PLC0415, PLC2701  # ty: ignore[unresolved-import]
        from torch_linear_assignment import batch_linear_assignment  # noqa: PLC0415  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise RuntimeError(
            "The 'jv' device solver needs torch-linear-assignment, which is not importable. "
            "Build it with FORCE_CUDA=1 (setup.py gates on torch.cuda.is_available(), so a "
            "build on a login node silently produces a CPU-only extension) and "
            "TORCH_CUDA_ARCH_LIST set for the target GPU, then put the built tree on "
            "PYTHONPATH with LD_LIBRARY_PATH pointing at this environment's lib. "
            "setup/build_torch_linear_assignment.sh does all of this: run it inside the "
            "environment (e.g. `pixi run -e clic bash setup/build_torch_linear_assignment.sh`) "
            "and follow what it prints. "
            f"Original error: {exc}"
        ) from exc

    if device is not None and torch.device(device).type == "cuda" and not backend.has_cuda():
        raise RuntimeError(
            "torch-linear-assignment was built without CUDA support, so the 'jv' device solver "
            "would solve GPU costs on the host and quietly undo the point of the option. "
            "Rebuild with FORCE_CUDA=1 and TORCH_CUDA_ARCH_LIST set for this GPU, which "
            "setup/build_torch_linear_assignment.sh does."
        )
    return batch_linear_assignment


@torch.no_grad()
def batched_jv(
    costs: Tensor,
    row_valid: Tensor,
    col_allowed: Tensor | None = None,
    **_auction_options,
) -> tuple[Tensor, None]:
    """Solve a batch of rectangular linear assignment problems exactly, by Jonker-Volgenant.

    A drop-in replacement for :func:`batched_auction` that is exact rather than epsilon-optimal.
    ``torch_linear_assignment`` implements Crouse (2016) -- the algorithm scipy itself uses --
    batched over problems on the GPU. It is strongly polynomial, so its cost depends on the
    shape of a problem and not on the values in it; the auction is pseudo-polynomial, which is
    what killed it on the real matcher costs (89.0% exact, 21.6% non-convergence, 350x slower than
    the host path) where uniform-random costs had shown no such trouble.

    Two preparation decisions carry the correctness of this path, and neither is obvious from
    the host solvers:

    * **Forbidden entries** go to ``num_rows + 1`` after the per-problem affine normalisation,
      as they do for the auction, and *not* to the ``float32_max / 10`` sentinel
      :meth:`~hepattn.models.matcher.Matcher._prepare_costs` hands the host solvers. JV works
      with fp32 duals (``cost - u - v``); a 3.4e37 entry destroys them.
    * **Padded rows** get a constant cost across every column. A batched solver assigns every
      row it is given, so padded rows cannot simply abstain the way they do in the auction.
      Constant rows contribute the same total whichever columns they take, so the solver parks
      them wherever suits the real rows and the optimum over the real rows is unchanged. This
      is the standard dummy-row reduction and it is exact -- padding with a large sentinel would
      not be, since the padded rows would then compete for the cheap columns.

    Args:
        costs: [batch, num_rows, num_cols] cost matrices. Non-finite entries are treated as
            forbidden assignments rather than propagating NaN.
        row_valid: [batch, num_rows] bool marking the rows that need an assignment. Padded rows
            come back as -1.
        col_allowed: Optional [batch, num_cols] bool marking the columns that may be assigned.
        **_auction_options: Accepted and ignored, so both device solvers share one call
            signature. JV has no bidding increment and no iteration cap to tune.

    Returns:
        Tuple of the [batch, num_rows] column assigned to each valid row (-1 for padded rows)
        and ``None``, which tells the caller the solver cannot come back short -- there is
        nothing to check and so no reason to sync the host on the result.

    Raises:
        ValueError: If the costs are not 3-dimensional, if a problem has more rows than
            columns, or if one has more valid rows than assignable columns.
    """
    if costs.ndim != 3:
        raise ValueError(f"Expected costs of shape [batch, num_rows, num_cols], got {tuple(costs.shape)}")

    batch, num_rows, num_cols = costs.shape
    device = costs.device
    dtype = costs.dtype if costs.dtype.is_floating_point else torch.float32
    costs = costs.to(dtype)

    row_valid = row_valid.to(device=device, dtype=torch.bool)
    if col_allowed is None:
        col_allowed = torch.ones(batch, num_cols, dtype=torch.bool, device=device)
    else:
        col_allowed = col_allowed.to(device=device, dtype=torch.bool)

    if bool((row_valid.sum(dim=1) > col_allowed.sum(dim=1)).any()):
        raise ValueError("Some assignment problems have more valid rows than assignable columns")
    # Every row is handed to the solver, padding included, so there must be a column for each.
    # The auction has no such requirement, since its padded rows simply never bid.
    if num_rows > num_cols:
        raise ValueError(f"Batched JV needs at least as many columns as rows, got {num_rows} rows into {num_cols} columns")

    unassigned = torch.full((batch, num_rows), -1, dtype=torch.long, device=device)
    if batch == 0 or num_rows == 0 or num_cols == 0:
        return unassigned, None

    solve = require_jv(device)
    allowed = torch.isfinite(costs) & row_valid[:, :, None] & col_allowed[:, None, :]
    prepared = _normalise(costs, allowed, forbidden_cost=float(num_rows + 1))
    prepared = torch.where(row_valid[:, :, None], prepared, torch.zeros_like(prepared))

    matching = solve(prepared.contiguous()).to(device=device, dtype=torch.long)
    return torch.where(row_valid, matching, unassigned), None


@torch.no_grad()
def assignment_to_permutation(assigned: Tensor, n_valid_rows: Tensor, num_cols: int) -> Tensor:
    """Expand a partial row->column assignment into the full column permutation the matcher wants.

    Position ``i`` of the result holds the column matched to row ``i``, for the valid rows; the
    remaining positions are filled with the unmatched columns in ascending order, so the result
    is always a permutation of ``range(num_cols)`` and can index the prediction tensors directly.

    Args:
        assigned: [batch, num_rows] column assigned to each row, -1 where unassigned.
        n_valid_rows: [batch] number of valid rows per problem, i.e. how many leading positions
            of the output are real matches.
        num_cols: Width of the permutation to produce.

    Returns:
        [batch, num_cols] int64 permutation. Rows of problems that were not fully solved are
        meaningless and must be overwritten by an exact solver.
    """
    batch, num_rows = assigned.shape
    device = assigned.device
    matched = assigned >= 0

    # Scatter through a trailing scratch column so that unassigned rows cannot race with real
    # writes: every valid row holds a distinct column, and everything else is aimed at `num_cols`.
    trash_col = torch.full_like(assigned, num_cols)
    used = torch.zeros(batch, num_cols + 1, dtype=torch.bool, device=device)
    used.scatter_(1, torch.where(matched, assigned, trash_col), torch.ones_like(assigned, dtype=torch.bool))

    # Unmatched columns keep their natural order and start where the matched rows leave off.
    free = ~used[:, :num_cols]
    slot = n_valid_rows[:, None] + free.long().cumsum(1) - 1
    col_idx = torch.arange(num_cols, device=device).expand(batch, num_cols)
    trash_slot = torch.full_like(slot, num_cols)

    perm = torch.zeros(batch, num_cols + 1, dtype=torch.long, device=device)
    perm.scatter_(1, torch.where(free, slot.clamp(0, num_cols), trash_slot), col_idx)
    row_idx = torch.arange(num_rows, device=device).expand(batch, num_rows)
    perm.scatter_(1, torch.where(matched, row_idx, torch.full_like(row_idx, num_cols)), assigned.clamp_min(0))
    return perm[:, :num_cols]
