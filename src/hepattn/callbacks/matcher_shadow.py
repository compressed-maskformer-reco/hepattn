"""Run both matchers on the same costs, every N steps, and record how different they are.

The CUDA matching study has two kinds of evidence that the device solver is a safe substitute
for the host one, and a gap between them.

At the bottom there is :mod:`bench_device_matcher`: on one dumped step's real cost tensors, does
the device solver reach the same *total cost* as scipy's float64 optimum? That is a single
scalar on a single step, and it says nothing about *which* pairing was chosen. At the top there
is the Phase-4 A/B: two full trainings compared on val loss, jet-E IQR and efficiency/purity.
That is what anyone actually cares about, but it is a terminal measurement -- it sits downstream
of ~10^5 optimiser steps, so any difference is entangled with SGD nondeterminism, data order and
seed noise. Overlapping IQR curves are evidence of neutrality, but weak evidence: they can hide
a real solver difference under training noise, and they can equally manufacture one that has
nothing to do with the matcher.

This callback is the missing middle layer. Given *identical* cost matrices, how often do the two
solvers return the same assignment, and when they do not, how much worse is the one the GPU
picked? It measures that inline, on the live model, so the answer comes as a function of
training step rather than from a single frozen snapshot -- which is the one question a replay
of a dumped cost tensor cannot answer: does agreement drift as the model sharpens and its costs
become less degenerate?

Three layers of metric, in increasing order of how much they mean and how much they cost:

1. **Optimality gap.** The matcher minimises total assignment cost, so "quality" is
   ``cost(device permutation) - cost(host optimum)``, per problem, evaluated in float64 on the
   same host cost array for both so the comparison is fair. This is a *solver-correctness*
   check and nothing more. The cost matrix is detached before it reaches the matcher
   (``MaskFormer.loss``, and ``Task.cost`` again inside it), and its only
   consumer is the assignment problem, so no gradient ever flows through a cost. A zero gap
   says the device solver found an optimal assignment, which makes any disagreement with the
   host a disagreement between *equally optimal* assignments rather than a degradation of the
   matching objective. It does not say the two runs train the same -- that is layer 2's job.
   For the exact ``jv`` solver this is a null check and is expected to be identically zero; for
   the epsilon-optimal auction it is bounded by ``num_valid_targets * eps``.

2. **Assignment agreement / IoU.** This is the layer that bears on training, because the
   permutation is the *only* channel through which the matcher reaches the loss: it selects
   which query's outputs are scored against which target, and everything downstream follows
   from that. An identical permutation gives an identical loss and an identical gradient; a
   different one changes the gradient even when the two assignments cost exactly the same.
   Equal cost does not imply equal permutation, and it does not imply equal loss either --
   the cost is a deliberate *proxy* for the loss rather than the loss (the CLIC v6 config
   scores matching with ``mask_dice`` alone, while the loss it trains on is
   ``mask_bce * 5 + mask_dice``). Both solvers return a set of (query slot, target)
   pairs; the intersection-over-union of those two sets is ``agree / (2n - agree)`` where
   ``agree`` is the number of shared pairs and ``n`` the number of valid targets. Real
   mask-BCE costs are highly degenerate -- many near-equal entries, ties within fp32 -- so two
   *exact* solvers routinely disagree on the permutation while agreeing exactly on the total,
   which layer 1 cannot see. ``tie_gap`` records how degenerate the swaps are: for each
   disagreeing target, the cost difference between the entry the device solver took and the one
   the host solver took. If those are ~0 the disagreement is a coin flip between equals -- by
   the matcher's criterion. Whether the model cares is layer 3.

3. **Mask IoU of the disagreements.** Layer 2 establishes that the gradient differs; it does
   not say whether the difference matters. So for the pairs where the two solvers disagree, compare
   the mask IoU of the *assigned prediction against its truth particle* under each assignment.
   If those two distributions coincide, the two matchings are physically interchangeable even
   though the permutations differ -- which is a far stronger statement than overlapping jet-E
   IQR curves, and it is made at the level of the object the matcher actually pairs up. An
   equal-sized sample of agreeing pairs is scored alongside as a control, to give the numbers a
   scale: it is the typical mask IoU of a matched pair in this model at this step.

**The measurement cannot alter the training it rides on.** The production ``Matcher.forward``
call is left completely untouched and still produces the permutation the loss uses; the shadow
solve happens beside it, on the same inputs, and its result is thrown away. The device path is
re-solved rather than reused for exactly this reason. Nothing here touches a torch RNG, and the
matcher's own ``step`` and ``device_fallbacks`` counters are restored afterwards, so a run with
this callback on is still a valid arm of a physics A/B.

Cost: on shadow steps only, one extra host solve (threaded, the same ``match_parallel`` the host
arm uses), one extra device solve, and a few thousand mask IoUs. At ``interval: 100`` on the
CLIC B200 geometry that is well under 1% of wall time.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from lightning import Callback

from hepattn.models.matcher import SOLVERS, Matcher, match_parallel
from hepattn.models.task import ObjectHitMaskTask


def _quantiles(values: np.ndarray, keys=(0.0, 0.01, 0.5, 0.99, 1.0)) -> dict:
    """Summarise a distribution by quantiles, so a shadow step costs a handful of floats."""
    if values.size == 0:
        return {f"q{k:g}": float("nan") for k in keys} | {"mean": float("nan"), "n": 0}
    qs = np.quantile(values.astype(np.float64), keys)
    return {f"q{k:g}": float(q) for k, q in zip(keys, qs, strict=True)} | {"mean": float(values.mean()), "n": int(values.size)}


class MatcherShadow(Callback):
    """Solve every matching problem twice -- device and host -- and record the difference.

    Args:
        interval: Shadow every this many training batches. The comparison is exact per problem,
            so it needs no averaging over steps; the interval is set by how finely the *drift*
            over training should be sampled, not by statistics.
        warmup: First batch index eligible for a shadow step. Default is past ``torch.compile``
            warmup, so the recorded step times are steady-state ones.
        reference_solver: Host solver taken as the optimum, from ``matcher.SOLVERS``. ``scipy``
            by default: it is the reference the rest of the study scores against, and unlike
            ``lap1015_late`` it has no GIL caveat when threaded.
        n_jobs: Threads for the reference host solve. Only affects the shadow's own cost.
        device_solver: Device solver to score, or ``None`` to use whatever the matcher in the
            model is configured with. Set it explicitly to run this callback on a *host* arm,
            where the matcher has no device solver of its own.
        max_mask_pairs: Cap on how many disagreeing pairs are scored for mask IoU per shadow
            step, and on the size of the agreeing-pair control sample drawn beside them.
            ``0`` disables layer 3.
        raw_dump_steps: Dump the PER-PAIR arrays -- not just their quantiles -- for this many
            shadow steps, as ``<output_name>_raw_step<N>.npz`` beside the .jsonl. The quantile
            summary cannot answer questions about the *joint* distribution of the two solvers'
            choices; the 2026-08-31 finding that the two IoU multisets are bitwise identical at
            every stored quantile is exactly such a question. A few hundred pairs per step, so
            the files are kilobytes. ``0`` (default) writes nothing.
        output_name: Basename of the ``.jsonl`` written to ``trainer.log_dir``. One line per
            shadow step, appended as it goes, so a killed job still leaves everything measured
            up to the point it died.
        verbose: Print a one-line summary per shadow step to stdout.
    """

    def __init__(
        self,
        interval: int = 100,
        warmup: int = 50,
        reference_solver: str = "scipy",
        n_jobs: int = 16,
        device_solver: str | None = None,
        max_mask_pairs: int = 4096,
        raw_dump_steps: int = 0,
        output_name: str = "matcher_shadow",
        verbose: bool = True,
    ):
        super().__init__()
        if reference_solver not in SOLVERS:
            raise ValueError(f"Unknown reference solver: {reference_solver}. Available: {list(SOLVERS)}")

        self.interval = interval
        self.warmup = warmup
        self.reference_solver = reference_solver
        self.n_jobs = n_jobs
        self.device_solver = device_solver
        self.max_mask_pairs = max_mask_pairs
        self.raw_dump_steps = raw_dump_steps
        self.output_name = output_name
        self.verbose = verbose

        self._matchers: list[tuple[Matcher, object]] = []
        self._models: list[tuple[object, object]] = []
        self._mask_task: ObjectHitMaskTask | None = None
        self._ctx: tuple | None = None
        self._active = False
        self._records: list[dict] = []
        self._layer_names: list[str] = []
        self._raw: dict | None = None
        self._raw_written = False
        self._raw_remaining = raw_dump_steps
        self._seen_steps = 0
        self.output_path: Path | None = None

    # ------------------------------------------------------------------ instrumentation

    def _shadowing(self, matcher: Matcher, fn):
        """Wrap ``Matcher.forward``: compare first, then hand the untouched call through."""

        def wrapper(costs, object_valid_mask=None, query_valid_mask=None, *args, **kwargs):
            if self._active:
                # ``MaskFormer.loss`` stacks every decoder layer's costs along the batch axis and
                # calls the matcher once, layer-major in the order of its outputs. Split the call
                # back into its layers so each record names the layer it belongs to.
                batch_size = self._ctx[1][self._target_valid_key].shape[0] if self._ctx is not None else costs.shape[0]
                num_layers = max(1, costs.shape[0] // batch_size)
                for i in range(num_layers):
                    layer = self._layer_names[i] if i < len(self._layer_names) else f"call{i}"
                    rows = slice(i * batch_size, (i + 1) * batch_size)
                    try:
                        record = self._compare(
                            matcher,
                            costs[rows],
                            None if object_valid_mask is None else object_valid_mask[rows],
                            None if query_valid_mask is None else query_valid_mask[rows],
                            layer,
                        )
                    except Exception as exc:  # noqa: BLE001
                        # A measurement must never be able to kill a 13-hour training run.
                        print(f"MatcherShadow: comparison failed at step {self._seen_steps} ({layer}): {exc!r}", flush=True)
                        record = {"error": repr(exc)}
                    self._records.append({"layer": layer} | record)
            return fn(costs, object_valid_mask, query_valid_mask, *args, **kwargs)

        return wrapper

    def _contextualising(self, model, fn):
        """Wrap ``MaskFormer.loss`` to expose the un-permuted masks.

        Layer 3 needs the predicted masks as they were when the matcher chose, and ``loss``
        permutes them in place (it replaces the entries of the nested outputs dict). The matcher
        call happens inside it, once for all decoder layers stacked, so a shallow copy of the
        outputs taken here and read from the matcher wrapper sees the masks before the permutation.
        """

        def wrapper(outputs, targets, *args, **kwargs):
            if self._active:
                unpermuted = {layer: {task: dict(out) for task, out in layer_outputs.items()} for layer, layer_outputs in outputs.items()}
                self._ctx = (unpermuted, targets)
                self._target_valid_key = f"{model.target_object}_valid"
                # The layers the matcher will be called for: every layer has a cost unless no
                # task contributes an intermediate loss, in which case only "final" does.
                intermediate = any(getattr(task, "has_intermediate_loss", True) for task in model.tasks)
                self._layer_names = [layer for layer in outputs if layer == "final" or intermediate]
            try:
                return fn(outputs, targets, *args, **kwargs)
            finally:
                self._ctx = None

        return wrapper

    def on_train_start(self, trainer, pl_module) -> None:
        if self._matchers:
            return

        matchers = [m for m in pl_module.modules() if isinstance(m, Matcher)]
        if not matchers:
            print("MatcherShadow: no Matcher found in the model; nothing will be measured.")
            return

        solver = self.device_solver or matchers[0].device_solver
        if solver is None:
            raise ValueError(
                "MatcherShadow: the model's matcher has no device_solver and none was given. "
                "Set device_solver on the callback to name the solver to score against the host."
            )
        self.device_solver = solver

        for matcher in matchers:
            self._matchers.append((matcher, matcher.forward))
            matcher.forward = self._shadowing(matcher, matcher.forward)

        for module in pl_module.modules():
            if hasattr(module, "matcher") and hasattr(module, "tasks") and callable(getattr(module, "loss", None)):
                self._models.append((module, module.loss))
                module.loss = self._contextualising(module, module.loss)
                for task in module.tasks:
                    if isinstance(task, ObjectHitMaskTask) and self._mask_task is None:
                        self._mask_task = task

        out_dir = Path(getattr(trainer, "log_dir", None) or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = out_dir / f"{self.output_name}.jsonl"

        layer3 = self._mask_task.name if (self._mask_task is not None and self.max_mask_pairs) else "off"
        print(
            f"MatcherShadow: {len(self._matchers)} matcher(s), scoring '{self.device_solver}' against "
            f"'{self.reference_solver}' every {self.interval} steps after step {self.warmup}; "
            f"mask-IoU layer: {layer3}; writing {self.output_path}",
            flush=True,
        )

    def _restore(self) -> None:
        for matcher, fn in self._matchers:
            matcher.forward = fn
        for module, fn in self._models:
            module.loss = fn
        self._matchers = []
        self._models = []

    def on_train_end(self, trainer, pl_module) -> None:
        self._restore()

    def teardown(self, trainer, pl_module, stage) -> None:
        self._restore()

    # ------------------------------------------------------------------ lightning hooks

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        self._records = []
        self._raw_written = False
        self._active = bool(self._matchers) and self._seen_steps >= self.warmup and self._seen_steps % self.interval == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        records, self._active = self._records, False
        self._records = []
        self._seen_steps += 1
        if self._raw_written:
            self._raw_remaining -= 1
            self._raw_written = False
        if not records:
            return

        # One record per decoder layer, since the matcher is called once per layer here; each
        # is written as its own line, and the logged scalars are the mean over the layers.
        head = {"step": int(trainer.global_step), "epoch": int(trainer.current_epoch), "batch_idx": int(batch_idx)}
        records = [head | record for record in records]

        if trainer.global_rank == 0 and self.output_path is not None:
            with self.output_path.open("a") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

        good = [record for record in records if "error" not in record]
        if good:

            def mean(*keys):
                values = []
                for record in good:
                    value = record
                    for key in keys:
                        value = value[key]
                    values.append(value)
                return float(np.mean(values))

            pl_module.log_dict(
                {
                    "shadow/agreement": mean("agreement", "mean"),
                    "shadow/iou": mean("assignment_iou", "mean"),
                    "shadow/problems_identical": mean("problems_identical"),
                    "shadow/cost_gap_mean": mean("cost_gap", "mean"),
                    "shadow/cost_gap_max": mean("cost_gap", "q1"),
                    "shadow/mask_iou_delta": mean("mask_iou", "delta", "mean"),
                },
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
            if self.verbose:
                for record in good:
                    self._print(record)

    def _print(self, record: dict) -> None:
        mask = record["mask_iou"]
        print(
            f"MatcherShadow step {record['step']} {record['layer']}: "
            f"agreement {record['agreement']['mean']:.4f}  IoU {record['assignment_iou']['mean']:.4f}  "
            f"identical problems {record['problems_identical']:.3%}  "
            f"cost gap mean {record['cost_gap']['mean']:.3e} max {record['cost_gap']['q1']:.3e}  "
            f"tie gap median {record['tie_gap']['q0.5']:.3e}  "
            f"mask IoU dev {mask['device']['mean']:.4f} host {mask['host']['mean']:.4f} "
            f"(control {mask['control']['mean']:.4f}, n={mask['device']['n']})  "
            f"[{record['host_solve_s']:.2f}s host / {record['device_solve_s']:.3f}s device]",
            flush=True,
        )

    # ------------------------------------------------------------------ the comparison

    def _compare(self, matcher: Matcher, costs, object_valid_mask, query_valid_mask, layer: str) -> dict:
        """Solve one layer's problems both ways and reduce the difference to a record."""
        num_pred = costs.shape[1]

        # Device assignment. The matcher's own counters are restored so that a shadowed run is
        # bit-identical in state to an unshadowed one.
        prev_solver, prev_fallbacks, prev_step = matcher.device_solver, matcher.device_fallbacks, matcher.step
        matcher.device_solver = self.device_solver
        if costs.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            idx_dev = matcher._match_on_device(costs, object_valid_mask, query_valid_mask)  # noqa: SLF001
        finally:
            if costs.is_cuda:
                torch.cuda.synchronize()
            device_solve_s = time.perf_counter() - t0
            device_fallbacks = matcher.device_fallbacks - prev_fallbacks
            matcher.device_solver, matcher.device_fallbacks, matcher.step = prev_solver, prev_fallbacks, prev_step

        # Host assignment, on the identical sanitised cost array both sides are then scored on.
        costs_t, lengths = matcher._prepare_costs(costs, object_valid_mask, query_valid_mask)  # noqa: SLF001
        t0 = time.perf_counter()
        idx_host = match_parallel(SOLVERS[self.reference_solver], costs_t, lengths, num_pred, n_jobs=self.n_jobs)
        host_solve_s = time.perf_counter() - t0

        # Both permutations are [batch, num_pred]; the cost array is [batch, max_true, num_pred].
        # They are compared over the target rows, which is min(max_true, num_pred) columns --
        # equal in the CLIC geometry, clamped here so an unequal one cannot broadcast wrongly.
        max_true = min(costs_t.shape[1], num_pred)
        costs_t = np.asarray(costs_t[:, :max_true], dtype=np.float64)
        perm_dev = idx_dev[:, :max_true].cpu().numpy().astype(np.int64)
        perm_host = idx_host[:, :max_true].cpu().numpy().astype(np.int64)

        # [batch, max_true] mask of the target rows that are real rather than padding.
        valid = np.arange(max_true)[None, :] < lengths[:, None]
        n_valid = valid.sum(axis=1)

        self._raw = {} if self._raw_remaining > 0 else None

        # Layer 1 -- optimality gap, in float64 on the same array for both permutations.
        gathered_dev = np.take_along_axis(costs_t, perm_dev[:, :, None], axis=2)[..., 0]
        gathered_host = np.take_along_axis(costs_t, perm_host[:, :, None], axis=2)[..., 0]
        cost_dev = (gathered_dev * valid).sum(axis=1)
        cost_host = (gathered_host * valid).sum(axis=1)
        gap = cost_dev - cost_host
        scale = np.maximum(np.abs(cost_host), 1e-12)

        # Layer 2 -- assignment agreement and IoU. Both permutations cover the same n_valid
        # targets, so |A n B| is the shared-pair count and |A u B| = 2n - |A n B|.
        same = (perm_dev == perm_host) & valid
        agree = same.sum(axis=1)
        nonempty = n_valid > 0
        agreement = np.divide(agree, np.maximum(n_valid, 1), dtype=np.float64)[nonempty]
        assignment_iou = np.divide(agree, np.maximum(2 * n_valid - agree, 1), dtype=np.float64)[nonempty]

        # How degenerate the swaps are: the cost difference between what the device solver took
        # for a target and what the host solver took for the same target.
        disagree = valid & ~same
        tie_gap = np.abs(gathered_dev - gathered_host)[disagree]

        mask_iou = self._mask_iou(disagree, same, perm_dev, perm_host, layer)
        if self._raw:
            self._save_raw(costs_t, layer)

        return {
            "num_problems": int(costs_t.shape[0]),
            "targets_median": int(np.median(n_valid)),
            "device_solver": self.device_solver,
            "device_fallbacks": int(device_fallbacks),
            "device_solve_s": device_solve_s,
            "host_solve_s": host_solve_s,
            "cost_gap": _quantiles(gap),
            "cost_gap_relative": _quantiles(gap / scale),
            "problems_worse": float((gap > 1e-9 * scale).mean()),
            "problems_identical": float((agree[nonempty] == n_valid[nonempty]).mean()) if nonempty.any() else float("nan"),
            "agreement": _quantiles(agreement),
            "assignment_iou": _quantiles(assignment_iou),
            "pairs_disagreeing": float(disagree.sum() / max(n_valid.sum(), 1)),
            "tie_gap": _quantiles(tie_gap),
            "mask_iou": mask_iou,
        }

    @staticmethod
    def _raw_chunk(k_sel, j_sel, perm_dev, perm_host, b, j, truth_mask, targets) -> dict:
        """One layer's worth of raw columns: who was paired, and what the two targets ARE.

        The queries and IoUs say the two solvers disagreed; they cannot say whether the swap
        mattered. That needs the targets themselves -- the truth constituent mask (bit-packed,
        so two targets can be compared for exact equality offline) and every per-particle
        scalar the data module supplies, which is where a difference in energy or class would
        show up. Any ``[batch, num_targets]`` entry in ``targets`` qualifies; the mask itself is
        ``[batch, num_targets, num_constituents]`` and is excluded by the ndim test.
        """
        sel = (k_sel, j_sel)
        n_targets = truth_mask.shape[0]
        chunk = {
            "problem": k_sel,
            "target": j_sel,
            "query_device": perm_dev[sel],
            "query_host": perm_host[sel],
            "truth_mask": np.packbits(truth_mask.cpu().numpy(), axis=-1),
            "truth_size": truth_mask.sum(-1).cpu().numpy(),
        }
        for key, val in targets.items():
            if torch.is_tensor(val) and val.ndim == 2 and val.shape[0] > int(b.max()) and val.shape[1] > int(j.max()):
                picked = val[b, j]
                if picked.shape[0] == n_targets:
                    chunk[f"target_{key}"] = picked.float().cpu().numpy()
        return chunk

    def _save_raw(self, costs_t: np.ndarray, layer: str) -> None:
        """Write one layer's per-pair arrays, so the joint distribution can be examined offline."""
        r = self._raw
        k, j, qd, qh = (r[c] for c in ("problem", "target", "query_device", "query_host"))
        path = self.output_path.with_name(f"{self.output_name}_raw_step{self._seen_steps}_{layer}.npz")
        np.savez_compressed(
            path,
            cost_device=costs_t[k, j, qd],
            cost_host=costs_t[k, j, qh],
            **r,
        )
        self._raw_written = True
        print(f"MatcherShadow: wrote {k.size} raw disagreeing pairs to {path}", flush=True)

    # ------------------------------------------------------------------ layer 3

    def _mask_iou(self, disagree: np.ndarray, same: np.ndarray, perm_dev: np.ndarray, perm_host: np.ndarray, layer: str) -> dict:
        """Score the disagreeing pairs, and a control sample of agreeing ones, by mask IoU.

        Returns the IoU distribution of (assigned prediction, truth particle) under each
        solver's choice, plus their per-pair difference. Two coincident distributions mean the
        matchings are physically interchangeable however much the permutations differ.
        """
        empty = {k: _quantiles(np.empty(0)) for k in ("device", "host", "delta", "control")}
        if self._mask_task is None or self.max_mask_pairs <= 0 or self._ctx is None:
            return empty

        decoder_outputs, targets = self._ctx
        task = self._mask_task
        if task.name not in decoder_outputs.get(layer, {}):
            return empty
        logit_key = task.output_object_hit + "_logit"
        truth = targets[task.target_object_hit + "_" + task.target_field]
        hit_valid = targets[task.input_hit + "_valid"]
        logits = decoder_outputs[layer][task.name][logit_key]

        rng = np.random.default_rng(self._seen_steps)
        picks = {"disagree": self._sample(disagree, rng), "control": self._sample(same, rng)}

        out = {}
        for tag, (k_idx, j_idx) in picks.items():
            dev_iou, host_iou, keys = [], [], []
            if k_idx.size:
                b = torch.as_tensor(k_idx, device=logits.device)
                j = torch.as_tensor(j_idx, device=logits.device)
                truth_mask = truth[b, j].bool() & hit_valid[b].bool()
                if self._raw is not None and tag == "disagree":
                    keys.append(self._raw_chunk(k_idx, j_idx, perm_dev, perm_host, b, j, truth_mask, targets))
                for perm, sink in ((perm_dev, dev_iou), (perm_host, host_iou)):
                    q = torch.as_tensor(perm[k_idx, j_idx], device=logits.device)
                    pred_mask = (logits[b, q].float().sigmoid() >= task.pred_threshold) & hit_valid[b].bool()
                    inter = (pred_mask & truth_mask).sum(-1)
                    union = (pred_mask | truth_mask).sum(-1)
                    sink.append((inter / union.clamp(min=1)).float().cpu().numpy())

            dev = np.concatenate(dev_iou) if dev_iou else np.empty(0)
            host = np.concatenate(host_iou) if host_iou else np.empty(0)
            if tag == "control":
                # Same query on both sides by construction, so one distribution is the answer:
                # the typical mask IoU of a matched pair, i.e. the scale the others are read on.
                out["control"] = _quantiles(dev)
            else:
                if self._raw is not None and keys:
                    self._raw |= {c: np.concatenate([chunk[c] for chunk in keys]) for c in keys[0]}
                    self._raw |= {"iou_device": dev, "iou_host": host}
                out["device"], out["host"], out["delta"] = _quantiles(dev), _quantiles(host), _quantiles(dev - host)
                out["interchangeable"] = float((np.abs(dev - host) < 0.01).mean()) if dev.size else float("nan")
        return empty | out

    def _sample(self, selected: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
        """Draw at most ``max_mask_pairs`` (problem, target) pairs out of a boolean mask."""
        k_idx, j_idx = np.nonzero(selected)
        if k_idx.size > self.max_mask_pairs:
            keep = rng.choice(k_idx.size, size=self.max_mask_pairs, replace=False)
            k_idx, j_idx = k_idx[keep], j_idx[keep]
        return k_idx, j_idx
