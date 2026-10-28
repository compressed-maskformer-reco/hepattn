import math

import pytest
import torch
import torch.nn.functional as F

from hepattn.models.encoder import Encoder, GeometricMessagePassing, gmp_neighbourhood

DIM, HEADS, SEQ, P = 32, 4, 24, 4
GEOMETRY = {"kernel_size": 3, "eta_cell": 0.25, "phi_cells": 16, "eta_max": 5.0}
TWO_PI = 2 * math.pi


def gmp_module(seed=0, **geometry):
    torch.manual_seed(seed)
    gmp = GeometricMessagePassing(DIM, **(GEOMETRY | geometry)).double()
    with torch.no_grad():
        gmp.dw.bias.normal_()  # nonzero, so the bias path is tested too
    return gmp


def run_gmp(gmp, x, eta, phi, valid):
    neighbours, in_grid = gmp_neighbourhood(eta, phi, valid, **gmp.geometry)
    return gmp(x, neighbours.to(x.dtype), in_grid)


def cell_centre(u, v, geometry=GEOMETRY):
    # (eta, phi) at the centre of cell (u, v), away from the floor boundaries
    eta = -geometry["eta_max"] + (u + 0.5) * geometry["eta_cell"]
    phi = (v + 0.5) * TWO_PI / geometry["phi_cells"]
    return eta, torch.remainder(phi + math.pi, TWO_PI) - math.pi


def dense_gmp(gmp, x, eta, phi, valid):
    """The grid form: scatter-sum into (B, H, W, D), circular pad in phi, zero pad in eta, depthwise conv2d, gather."""
    batch_size, num_tokens, dim = x.shape
    k, r = gmp.kernel_size, gmp.kernel_size // 2
    num_eta, num_phi = math.floor(2 * gmp.eta_max / gmp.eta_cell) + 1, gmp.phi_cells
    in_grid = valid & eta.isfinite() & phi.isfinite() & (eta.abs() <= gmp.eta_max)
    u = torch.floor((torch.where(in_grid, eta, 0.0) + gmp.eta_max) / gmp.eta_cell).long()
    v = torch.remainder(torch.floor(torch.where(in_grid, phi, 0.0) * num_phi / TWO_PI), num_phi).long()
    flat = (torch.arange(batch_size)[:, None] * num_eta + u) * num_phi + v
    src = torch.where(in_grid[..., None], x, 0.0).reshape(-1, dim)
    grid = torch.zeros(batch_size * num_eta * num_phi, dim, dtype=x.dtype).index_add(0, flat.reshape(-1), src)
    grid = grid.view(batch_size, num_eta, num_phi, dim).permute(0, 3, 1, 2)
    grid = F.pad(F.pad(grid, (r, r, 0, 0), mode="circular"), (0, 0, r, r))
    grid = F.conv2d(grid, gmp.dw.weight, gmp.dw.bias, groups=dim)
    assert grid.shape[-2:] == (num_eta, num_phi) and k == gmp.dw.kernel_size[0]
    z = grid.permute(0, 2, 3, 1).reshape(-1, dim)[flat.reshape(-1)].view(batch_size, num_tokens, dim)
    return torch.where(in_grid[..., None], x + gmp.norm(gmp.pointwise(z)), x)


def reference_gmp(gmp, x, eta, phi, grid_size):
    """Line-by-line torch translation of the reference GeometricMessagePassing.call (PHAT-JeT 28f4e59,
    models/PHAT_JeT.py:41-78): per-event min-shifted cells, batch-global H x W, sum scatter of every slot, Keras
    "same" zero padding, gather, Dense, LayerNorm, residual."""
    batch_size, num_tokens, dim = x.shape
    gu = ((eta - eta.amin(1, keepdim=True)) / grid_size).long()
    gv = ((phi - phi.amin(1, keepdim=True)) / grid_size).long()
    num_eta, num_phi = int(gu.max()) + 1, int(gv.max()) + 1
    flat = (torch.arange(batch_size)[:, None] * num_eta + gu) * num_phi + gv
    grid = torch.zeros(batch_size * num_eta * num_phi, dim, dtype=x.dtype).index_add(0, flat.reshape(-1), x.reshape(-1, dim))
    grid = grid.view(batch_size, num_eta, num_phi, dim).permute(0, 3, 1, 2)
    k = gmp.kernel_size
    lo, hi = (k - 1) // 2, k - 1 - (k - 1) // 2
    grid = F.conv2d(F.pad(grid, (lo, hi, lo, hi)), gmp.dw.weight, gmp.dw.bias, groups=dim)
    z = grid.permute(0, 2, 3, 1).reshape(-1, dim)[flat.reshape(-1)].view(batch_size, num_tokens, dim)
    return x + gmp.norm(gmp.pointwise(z))


def random_batch(lengths, n=SEQ, seed=1, eta_scale=4.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(lengths), n, DIM, generator=g, dtype=torch.float64)
    eta = (torch.rand(len(lengths), n, generator=g, dtype=torch.float64) * 2 - 1) * eta_scale
    phi = (torch.rand(len(lengths), n, generator=g, dtype=torch.float64) * 2 - 1) * math.pi
    valid = torch.arange(n)[None] < torch.tensor(lengths)[:, None]
    # two tokens in one cell, and two either side of phi = +-pi
    eta[:, 1], phi[:, 1] = eta[:, 0], phi[:, 0]
    eta[:, 2], phi[:, 2], eta[:, 3], phi[:, 3] = 0.1, math.pi - 0.01, 0.1, -math.pi + 0.01
    return x, eta, phi, valid


# ---------------------------------------------------------------- the GMP operator


@pytest.mark.parametrize("lengths", [[SEQ], [SEQ, 7, 1]])
@pytest.mark.parametrize("kernel_size", [3, 5])
def test_matches_dense_grid_convolution(lengths, kernel_size):
    # the pair form is the grid form (scatter, periodic conv, gather) exactly, in values and gradients
    gmp = gmp_module(kernel_size=kernel_size)
    x, eta, phi, valid = random_batch(lengths)
    eta[:, 4] = 6.0  # off the grid
    xp, xd = x.clone().requires_grad_(), x.clone().requires_grad_()
    out_p, out_d = run_gmp(gmp, xp, eta, phi, valid), dense_gmp(gmp, xd, eta, phi, valid)
    assert out_p.shape == x.shape
    torch.testing.assert_close(out_p, out_d)
    grads_p = torch.autograd.grad(out_p.square().sum(), [xp, *gmp.parameters()])
    grads_d = torch.autograd.grad(out_d.square().sum(), [xd, *gmp.parameters()])
    for gp, gd in zip(grads_p, grads_d, strict=True):
        torch.testing.assert_close(gp, gd)


def test_matches_reference_where_the_grids_agree():
    # Reference parity: away from the phi seam and the eta edges, with no padding and the tokens at cell centres, the
    # reference's min-shifted grid and ours give the same cell offsets, so the outputs are the same. (The reference
    # cuts phi at its per-event minimum, scatters padded slots and sizes the grid by the batch: see the next tests.)
    gmp = gmp_module(kernel_size=3)
    g = torch.Generator().manual_seed(5)
    u = torch.randint(10, 16, (2, SEQ), generator=g)
    v = torch.randint(2, 8, (2, SEQ), generator=g)
    eta, phi = cell_centre(u.double(), v.double())
    x = torch.randn(2, SEQ, DIM, generator=g, dtype=torch.float64)
    # the reference's own grid size is arbitrary here (0.25, exact in binary, so its int() truncation is exact)
    ref = reference_gmp(gmp, x, (u + 0.5) * 0.25, (v + 0.5) * 0.25, grid_size=0.25)
    torch.testing.assert_close(run_gmp(gmp, x, eta, phi, torch.ones_like(u, dtype=torch.bool)), ref)


def test_reference_cuts_phi_where_ours_wraps():
    # an intentional difference: across phi = +-pi the reference grid has no neighbours, ours does
    gmp = gmp_module()
    x, eta, phi, valid = random_batch([SEQ])
    x2 = x.clone()
    x2[0, 3] += 1.0  # the token just above -pi; token 2 is just below +pi
    delta = (run_gmp(gmp, x2, eta, phi, valid) - run_gmp(gmp, x, eta, phi, valid))[0, 2].abs().max()
    assert delta > 1e-3
    ref_delta = (reference_gmp(gmp, x2, eta, phi, 0.25) - reference_gmp(gmp, x, eta, phi, 0.25))[0, 2].abs().max()
    assert ref_delta == 0


def test_phi_is_periodic():
    # rotating every token by whole phi cells, across the seam, only relabels the cells: same outputs
    gmp = gmp_module()
    g = torch.Generator().manual_seed(6)
    u = torch.randint(15, 25, (2, SEQ), generator=g).double()
    v = torch.randint(0, 16, (2, SEQ), generator=g).double()
    x = torch.randn(2, SEQ, DIM, generator=g, dtype=torch.float64)
    valid = torch.ones(2, SEQ, dtype=torch.bool)
    ref = run_gmp(gmp, x, *cell_centre(u, v), valid)
    for shift in (1, 5, 8, 15, -3):
        eta, phi = cell_centre(u, v + shift)
        torch.testing.assert_close(run_gmp(gmp, x, eta, phi, valid), ref)
    # and any 2pi branch of phi is the same azimuth
    eta, phi = cell_centre(u, v)
    torch.testing.assert_close(run_gmp(gmp, x, eta, phi + TWO_PI, valid), ref)
    torch.testing.assert_close(run_gmp(gmp, x, eta, phi - 2 * TWO_PI, valid), ref)


def test_eta_is_not_periodic_and_out_of_range_tokens_skip_gmp():
    # In CLIC calorimeter coordinates the off-grid tokens are tracks whose extrapolation meets the endcap plane near
    # the beam axis: |eta_int| up to ~9.6 (e.g. -9.6 below), against |eta| <= 5 for every topo cluster
    gmp = gmp_module()
    x = torch.randn(1, 4, DIM, dtype=torch.float64)
    valid = torch.ones(1, 4, dtype=torch.bool)
    eta = torch.tensor([[-4.9, 4.9, 5.01, -9.6]], dtype=torch.float64)
    phi = torch.zeros(1, 4, dtype=torch.float64)
    neighbours, in_grid = gmp_neighbourhood(eta, phi, valid, **gmp.geometry)
    assert in_grid.tolist() == [[True, True, False, False]]
    # the two edges of the eta range are not neighbours; the off-grid tokens have none, not even themselves
    adjacency = neighbours.view(1, 9, 4, 4).any(1)[0]
    assert adjacency.tolist() == [[True, False, False, False], [False, True, False, False], [False] * 4, [False] * 4]
    # off-grid tokens are left exactly as they were (not clamped onto the edge cells), and reach nobody
    out = run_gmp(gmp, x, eta, phi, valid)
    assert torch.equal(out[0, 2:], x[0, 2:])
    x2 = x.clone()
    x2[0, 2:] += 1.0
    assert torch.equal(run_gmp(gmp, x2, eta, phi, valid)[0, :2], out[0, :2])


def test_nonfinite_coordinates_of_real_tokens_skip_gmp():
    gmp = gmp_module()
    x, eta, phi, valid = random_batch([SEQ])
    keep = torch.arange(SEQ) != 5
    ref = run_gmp(gmp, x[:, keep], eta[:, keep], phi[:, keep], valid[:, keep])
    for bad in (float("nan"), float("inf"), -float("inf")):
        for coord in ("eta", "phi"):
            e, p = eta.clone(), phi.clone()
            (e if coord == "eta" else p)[0, 5] = bad
            out = run_gmp(gmp, x, e, p, valid)
            assert out.isfinite().all()
            assert torch.equal(out[0, 5], x[0, 5])
            torch.testing.assert_close(out[:, keep], ref)


def test_tokens_sharing_a_cell():
    # tokens in one cell get the same message, and the cell sums them (as the reference's scatter_nd)
    gmp = gmp_module()
    x = torch.randn(1, 3, DIM, dtype=torch.float64)
    eta = torch.tensor([[0.1, 0.12, 0.13]], dtype=torch.float64)
    phi = torch.tensor([[0.1, 0.11, 0.12]], dtype=torch.float64)
    valid = torch.ones(1, 3, dtype=torch.bool)
    neighbours, in_grid = gmp_neighbourhood(eta, phi, valid, **gmp.geometry)
    z_offsets = neighbours.view(1, 9, 3, 3).double()
    assert torch.equal(z_offsets[0, 4], torch.ones(3, 3))  # the centre offset: all three, themselves included
    assert z_offsets.sum() == 9  # and no other offset
    out = gmp(x, neighbours.double(), in_grid)
    msg = out - x
    torch.testing.assert_close(msg[0, 0], msg[0, 1])
    torch.testing.assert_close(msg[0, 0], msg[0, 2])
    # summing: two copies of a token in one cell send the message of one token with twice the features
    x_two = x[:, :1].expand(1, 2, DIM).clone()
    msg_two = run_gmp(gmp, x_two, eta[:, :2], phi[:, :2], valid[:, :2]) - x_two
    msg_double = run_gmp(gmp, 2 * x[:, :1], eta[:, :1], phi[:, :1], valid[:, :1]) - 2 * x[:, :1]
    torch.testing.assert_close(msg_two[0, 0], msg_double[0, 0])
    torch.testing.assert_close(msg_two[0, 1], msg_double[0, 0])


@pytest.mark.parametrize("fill", [1e4, float("inf"), float("nan")])
def test_padding_contributes_nothing(fill):
    gmp = gmp_module()
    x, eta, phi, valid = random_batch([SEQ, 7, 1])
    ref = run_gmp(gmp, x, eta, phi, valid)
    garbage = [torch.where(valid[..., None], x, fill), torch.where(valid, eta, fill), torch.where(valid, phi, fill)]
    out = run_gmp(gmp, *garbage, valid)
    assert torch.equal(out[valid], ref[valid])
    assert out[valid].isfinite().all()
    # padded slots skip GMP: they keep whatever they held
    torch.testing.assert_close(out[~valid], garbage[0][~valid], equal_nan=True)
    # adding padded slots changes nothing either
    pad = 5
    longer = [torch.cat([t, torch.full((3, pad, *t.shape[2:]), fill, dtype=t.dtype)], dim=1) for t in (x, eta, phi)]
    longer_valid = torch.cat([valid, torch.zeros(3, pad, dtype=torch.bool)], dim=1)
    torch.testing.assert_close(run_gmp(gmp, *longer, longer_valid)[:, :SEQ][valid], ref[valid])


def test_all_padded_events_stay_finite():
    gmp = gmp_module()
    x, eta, phi, valid = random_batch([0, 5])
    out = run_gmp(gmp, x, torch.where(valid, eta, float("nan")), phi, valid)
    assert out.isfinite().all()
    assert torch.equal(out[0], x[0])


def test_permutation_equivariant():
    gmp = gmp_module()
    x, eta, phi, valid = random_batch([SEQ, 13])
    ref = run_gmp(gmp, x, eta, phi, valid)
    perm = torch.randperm(SEQ)
    out = run_gmp(gmp, x[:, perm], eta[:, perm], phi[:, perm], valid[:, perm])
    torch.testing.assert_close(out, ref[:, perm])


def test_gradients_reach_tokens_and_parameters():
    gmp = gmp_module().train()
    x, eta, phi, valid = random_batch([SEQ, 7])
    x = torch.where(valid[..., None], x, float("nan")).requires_grad_(True)
    out = run_gmp(gmp, x, eta, phi, valid)
    out[valid].square().sum().backward()
    assert x.grad[valid].isfinite().all()
    assert (x.grad[~valid] == 0).all()
    for name, p in gmp.named_parameters():
        assert p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0, name


def test_track_and_cluster_in_neighbouring_cells_exchange_information():
    # A tiny example: a "track" (at its calorimeter-front-face (eta_int, phi_int)) and a "cluster" (at its barycentre)
    # in neighbouring (eta, phi) cells, and a third token three cells away. GMP passes information between the first
    # two and not to the third. It is only a geometric neighbourhood, not an association: all tokens in a neighbourhood
    # are treated alike, with no truth or matching information.
    gmp = gmp_module()
    eta = torch.tensor([[0.10, 0.30, 0.10]], dtype=torch.float64)  # cells 20, 21, 20 (cell 0.25 from -5)
    phi = torch.tensor([[0.10, 0.10, 1.30]], dtype=torch.float64)  # cells 0, 0, 3 (cell 2pi / 16 = 0.39)
    valid = torch.ones(1, 3, dtype=torch.bool)
    x = torch.zeros(1, 3, DIM, dtype=torch.float64)
    x[0, 1, 0] = 1.0  # only the cluster carries a signal
    base = run_gmp(gmp, torch.zeros_like(x), eta, phi, valid)
    out = run_gmp(gmp, x, eta, phi, valid)
    assert (out[0, 0] - base[0, 0]).abs().max() > 1e-3  # the track sees the cluster
    assert torch.equal(out[0, 2], base[0, 2])  # the distant token does not


def test_bad_geometry_refused():
    with pytest.raises(ValueError, match="odd"):
        GeometricMessagePassing(DIM, kernel_size=4)
    with pytest.raises(ValueError, match="phi_cells"):
        GeometricMessagePassing(DIM, kernel_size=5, phi_cells=4)
    with pytest.raises(ValueError, match="positive"):
        GeometricMessagePassing(DIM, eta_cell=0.0)


def test_parameter_count():
    # depthwise k x k + bias, Linear, LayerNorm: 68,864 per layer at dim 256, kernel 3
    gmp = GeometricMessagePassing(256, kernel_size=3)
    assert sum(p.numel() for p in gmp.parameters()) == 256 * 9 + 256 + 256 * 256 + 256 + 2 * 256 == 68_864


# ---------------------------------------------------------------- in the patch encoder


def patch_encoder(gmp=GEOMETRY, num_layers=2, patch_size=P, patch_global=True, num_register_tokens=8, seed=0):
    torch.manual_seed(seed)
    kwargs = {} if gmp is None else {"gmp": gmp}
    enc = Encoder(
        num_layers=num_layers,
        dim=DIM,
        attn_type="patch",
        patch_size=patch_size,
        patch_global=patch_global,
        num_register_tokens=num_register_tokens,
        value_residual=True,
        hybrid_norm=True,
        attn_kwargs={"num_heads": HEADS},
        dense_kwargs={"activation": "SwiGLU"},
        **kwargs,
    )
    return enc.double().eval()


def coords(eta, phi):
    return torch.stack([eta, phi], dim=-1)


def test_gmp_off_is_the_patch_encoder_unchanged():
    # gmp=None builds exactly the PHAT16 encoder: same parameters, same random init, bit-identical outputs
    phat, off = patch_encoder(gmp=None), patch_encoder(gmp=None)
    off_explicit = Encoder(
        num_layers=2, dim=DIM, attn_type="patch", patch_size=P, num_register_tokens=8, value_residual=True, hybrid_norm=True,
        attn_kwargs={"num_heads": HEADS}, dense_kwargs={"activation": "SwiGLU"}, gmp=None,
    )  # fmt: skip
    assert list(off_explicit.state_dict()) == list(phat.state_dict())
    assert all(layer.gmp is None for layer in off.layers)
    x, eta, phi, valid = random_batch([SEQ, 7])
    assert torch.equal(off(x, x_sort_value=phi, kv_mask=valid), phat(x, x_sort_value=phi, kv_mask=valid))
    # and the GMP encoder is the PHAT16 encoder plus the gmp.* parameters of each layer
    gmp_keys = set(patch_encoder().state_dict()) - set(phat.state_dict())
    assert set(phat.state_dict()) < set(patch_encoder().state_dict())
    assert gmp_keys == {f"layers.{i}.gmp.{n}" for i in range(2) for n in ("dw.weight", "dw.bias", "pointwise.weight", "pointwise.bias", "norm.weight", "norm.bias")}


def test_coordinates_needed_exactly_with_gmp():
    x, eta, phi, valid = random_batch([SEQ])
    with pytest.raises(ValueError, match="x_coords"):
        patch_encoder()(x, kv_mask=valid)
    with pytest.raises(ValueError, match="x_coords"):
        patch_encoder(gmp=None)(x, kv_mask=valid, x_coords=coords(eta, phi))
    with pytest.raises(ValueError, match="only used with attn_type 'patch'"):
        Encoder(num_layers=1, dim=DIM, attn_type="torch", gmp=GEOMETRY)
    with pytest.raises(TypeError):
        patch_encoder(gmp=GEOMETRY | {"grid_size": 0.05})


@pytest.mark.parametrize("lengths", [[SEQ], [SEQ, 9, 1], [0, 5]])
def test_encoder_shapes_and_finite(lengths):
    enc = patch_encoder()
    x, eta, phi, valid = random_batch(lengths)
    out = enc(x, x_sort_value=phi, kv_mask=valid, x_coords=coords(eta, phi))
    assert out.shape == x.shape
    assert out.isfinite().all()
    assert (out[~valid] == 0).all()


def test_gmp_changes_the_encoder():
    enc = patch_encoder()
    phat = patch_encoder(gmp=None)
    phat.load_state_dict(enc.state_dict(), strict=False)
    x, eta, phi, valid = random_batch([SEQ, 9])
    out = enc(x, kv_mask=valid, x_coords=coords(eta, phi))
    assert (out[valid] - phat(x, kv_mask=valid)[valid]).abs().max() > 1e-3
    # moving one token's coordinates changes the others' outputs: the geometry is used
    eta2 = eta.clone()
    eta2[0, 0] += 1.0
    assert (enc(x, kv_mask=valid, x_coords=coords(eta2, phi))[0, 1:] - out[0, 1:]).abs().max() > 1e-6


def test_coordinates_follow_their_tokens_through_the_sort():
    # Sorting is internal: presorting the tokens, mask and coordinates by hand gives the same per-token outputs as
    # letting the encoder sort. If the encoder sorted x but not x_coords, GMP would see the wrong geometry here.
    enc = patch_encoder()
    x, eta, phi, valid = random_batch([SEQ, 13, 2])
    sort_value = torch.rand(3, SEQ, dtype=torch.float64)
    out = enc(x, x_sort_value=sort_value, kv_mask=valid, x_coords=coords(eta, phi))
    idx = torch.argsort(sort_value.masked_fill(~valid, float("inf")), dim=-1)
    gather = lambda t: t.gather(1, idx.view(*idx.shape, *[1] * (t.ndim - 2)).expand_as(t))  # noqa: E731
    out_pre = enc(gather(x), kv_mask=gather(valid), x_coords=gather(coords(eta, phi)))
    torch.testing.assert_close(gather(out), out_pre)
    # control: scrambling the coordinates relative to the tokens does change the outputs
    scrambled = coords(eta, phi)[:, torch.randperm(SEQ)]
    assert (enc(x, x_sort_value=sort_value, kv_mask=valid, x_coords=scrambled)[valid] - out[valid]).abs().max() > 1e-6


def test_encoder_follows_input_order():
    # permuting tokens, sort values, coordinates and mask together permutes the outputs (in the input slot order)
    enc = patch_encoder()
    x, eta, phi, valid = random_batch([SEQ, 20, 9])
    sort_value = torch.rand(3, SEQ, dtype=torch.float64)
    out = enc(x, x_sort_value=sort_value, kv_mask=valid, x_coords=coords(eta, phi))
    perm = torch.stack([torch.randperm(SEQ) for _ in range(3)])
    g2 = lambda t: t.gather(1, perm)  # noqa: E731
    out_p = enc(x.gather(1, perm[..., None].expand_as(x)), x_sort_value=g2(sort_value), kv_mask=g2(valid), x_coords=coords(g2(eta), g2(phi)))
    out_back = torch.empty_like(out_p).scatter_(1, perm[..., None].expand_as(out_p), out_p)
    torch.testing.assert_close(out_back[valid], out[valid])


def test_interleaved_padding_without_sort():
    # without a sort the real tokens are compacted before patching, and their coordinates with them
    enc = patch_encoder()
    x, eta, phi, _ = random_batch([SEQ])
    valid = torch.rand(1, SEQ, generator=torch.Generator().manual_seed(2)) < 0.5
    order = torch.argsort((~valid).int(), dim=-1, stable=True)[0]
    out = enc(x, kv_mask=valid, x_coords=coords(eta, phi))
    out_c = enc(x[:, order], kv_mask=valid[:, order], x_coords=coords(eta, phi)[:, order])
    torch.testing.assert_close(out[valid], out_c[valid[:, order]])


@pytest.mark.parametrize("fill", [1e4, float("inf"), float("nan")])
def test_encoder_ignores_padded_tokens_and_coordinates(fill):
    enc = patch_encoder()
    x, eta, phi, valid = random_batch([7, SEQ, 1])
    sort_value = torch.rand(3, SEQ, dtype=torch.float64)
    out = enc(x, x_sort_value=sort_value, kv_mask=valid, x_coords=coords(eta, phi))
    g = lambda t: torch.where(valid if t.ndim == 2 else valid[..., None], t, fill)  # noqa: E731
    out_g = enc(g(x), x_sort_value=sort_value, kv_mask=valid, x_coords=coords(g(eta), g(phi)))
    assert torch.equal(out_g[valid], out[valid])
    assert out_g.isfinite().all()


def test_no_mask_with_one_event():
    # MaskFormer passes kv_mask=None for a single event with no padding
    enc = patch_encoder()
    x, eta, phi, valid = random_batch([SEQ])
    torch.testing.assert_close(enc(x, x_coords=coords(eta, phi)), enc(x, kv_mask=valid, x_coords=coords(eta, phi)))


def test_encoder_gradients_complete():
    enc = patch_encoder().train()
    x, eta, phi, valid = random_batch([7, SEQ, 1])
    x = torch.where(valid[..., None], x, float("nan")).requires_grad_(True)
    enc(x, x_sort_value=phi, kv_mask=valid, x_coords=coords(eta, phi))[valid].square().sum().backward()
    assert x.grad[valid].isfinite().all()
    assert (x.grad[~valid] == 0).all()
    no_grad = [n for n, p in enc.named_parameters() if p.grad is None or not p.grad.isfinite().all() or not p.grad.abs().sum() > 0]
    assert not no_grad, no_grad


def test_state_dict_round_trip(tmp_path):
    enc = patch_encoder(seed=0)
    x, eta, phi, valid = random_batch([SEQ, 9])
    out = enc(x, kv_mask=valid, x_coords=coords(eta, phi))
    torch.save(enc.state_dict(), tmp_path / "gmp.pt")
    loaded = patch_encoder(seed=1)
    loaded.load_state_dict(torch.load(tmp_path / "gmp.pt"))
    assert torch.equal(loaded(x, kv_mask=valid, x_coords=coords(eta, phi)), out)
    # a PHAT16 (no GMP) state dict loads into a GMP encoder only non-strictly, missing exactly the gmp.* tensors
    missing, unexpected = patch_encoder().load_state_dict(patch_encoder(gmp=None).state_dict(), strict=False)
    assert not unexpected
    assert missing and all(".gmp." in name for name in missing)


def test_bf16_autocast_finite():
    enc = patch_encoder().float()
    x, eta, phi, valid = random_batch([SEQ, 5])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = enc(x.float(), x_sort_value=phi.float(), kv_mask=valid, x_coords=coords(eta, phi).float())
    assert out.isfinite().all()
    ref = enc(x.float(), x_sort_value=phi.float(), kv_mask=valid, x_coords=coords(eta, phi).float())
    assert (out.float() - ref)[valid].abs().max() < 0.1 * ref[valid].abs().max()


def test_compiles():
    # torch.compile(dynamic=True), as the Compile callback, on CPU: same outputs as eager, and a different batch and
    # length run too
    enc = patch_encoder().float()
    compiled = torch.compile(enc, dynamic=True)
    for lengths, n in (([SEQ, 7], SEQ), ([10, 3, 1], SEQ + 4)):
        x, eta, phi, valid = (t.float() if t.is_floating_point() else t for t in random_batch(lengths, n=n))
        args = {"x_sort_value": phi, "kv_mask": valid, "x_coords": coords(eta, phi)}
        torch.testing.assert_close(compiled(x, **args), enc(x, **args), atol=1e-4, rtol=1e-4)
