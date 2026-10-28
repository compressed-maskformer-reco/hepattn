import pytest
import torch

from hepattn.models.encoder import Encoder, PatchEncoderLayer, periodic_sort_value, sort_index

DIM, HEADS, SEQ, P = 32, 4, 24, 4


def patch_encoder(patch_size=P, patch_global=True, num_register_tokens=8, value_residual=True, num_layers=2, **kwargs):
    torch.manual_seed(0)
    enc = Encoder(
        num_layers=num_layers,
        dim=DIM,
        attn_type="patch",
        patch_size=patch_size,
        patch_global=patch_global,
        num_register_tokens=num_register_tokens,
        value_residual=value_residual,
        hybrid_norm=True,
        attn_kwargs={"num_heads": HEADS},
        dense_kwargs={"activation": "SwiGLU"},
        **kwargs,
    )
    return enc.double().eval()


def padded_batch(lengths, n=SEQ):
    torch.manual_seed(1)
    x = torch.randn(len(lengths), n, DIM, dtype=torch.float64)
    mask = torch.arange(n)[None] < torch.tensor(lengths)[:, None]
    return x, mask


def test_layers_are_patch_layers():
    enc = patch_encoder()
    assert enc.attn_type == "patch"
    assert all(isinstance(layer, PatchEncoderLayer) for layer in enc.layers)
    # within a patch the layers run the torch backend
    assert all(layer.attn.fn.attn_type == "torch" for layer in enc.layers)
    assert all(layer.patch_attn.attn_type == "torch" for layer in enc.layers)
    # the global stage exists, and the register tokens are kept
    assert all(hasattr(layer, "patch_attn") for layer in enc.layers)
    assert enc.register_tokens.shape == (1, 8, DIM)


@pytest.mark.parametrize("seq_len", [SEQ, SEQ + 1, 7])
def test_output_shape(seq_len):
    # one embedding per constituent, also when the length is not a multiple of the patch size
    enc = patch_encoder()
    x, mask = padded_batch([seq_len, 3], n=seq_len)
    out = enc(x, kv_mask=mask)
    assert out.shape == x.shape
    assert out.isfinite().all()


def test_matches_full_attention_with_one_patch():
    # with a single patch covering the sequence and no global stage, patch attention is ordinary attention
    full = Encoder(
        num_layers=2,
        dim=DIM,
        attn_type="torch",
        value_residual=True,
        hybrid_norm=True,
        attn_kwargs={"num_heads": HEADS},
        dense_kwargs={"activation": "SwiGLU"},
    )
    full = full.double().eval()
    patch = patch_encoder(patch_size=SEQ, patch_global=False, num_register_tokens=None)
    patch.load_state_dict(full.state_dict())
    x, mask = padded_batch([7, 24, 13])
    torch.testing.assert_close(patch(x, kv_mask=mask)[mask], full(x, kv_mask=mask)[mask])


def test_follows_input_order():
    # sorting is internal: permuting the inputs permutes the outputs the same way
    enc = patch_encoder()
    x, mask = padded_batch([20, 24, 9])
    phi = torch.rand(3, SEQ, dtype=torch.float64) * 2 - 1
    out = enc(x, x_sort_value=phi, kv_mask=mask)
    perm = torch.stack([torch.randperm(SEQ) for _ in range(3)])
    out_p = enc(
        x.gather(1, perm[..., None].expand_as(x)),
        x_sort_value=phi.gather(1, perm),
        kv_mask=mask.gather(1, perm),
    )
    out_back = torch.empty_like(out_p).scatter_(1, perm[..., None].expand_as(out_p), out_p)
    torch.testing.assert_close(out_back[mask], out[mask])


def test_sort_groups_neighbours_into_patches():
    # patches are cut from the sorted order: with no global stage, a constituent only sees the
    # constituents of its own patch, i.e. its neighbours in the sort value
    enc = patch_encoder(patch_global=False, num_register_tokens=None)
    x, mask = padded_batch([SEQ])
    phi = torch.rand(1, SEQ, dtype=torch.float64)
    out = enc(x, x_sort_value=phi, kv_mask=mask)
    rank = torch.argsort(torch.argsort(phi, dim=-1), dim=-1)[0]
    lowest, highest = int(torch.argmin(phi)), int(torch.argmax(phi))
    x2 = x.clone()
    x2[0, highest] += 1.0
    out2 = enc(x2, x_sort_value=phi, kv_mask=mask)
    same_patch = rank // P == rank[lowest] // P
    torch.testing.assert_close(out2[0, same_patch], out[0, same_patch])
    assert (out2[0, highest] - out[0, highest]).abs().max() > 1e-3


def test_global_stage_connects_patches():
    x, mask = padded_batch([SEQ])
    x2 = x.clone()
    x2[0, -1] += 1.0  # last patch
    first = slice(0, P)
    for patch_global, changes in ((False, False), (True, True)):
        enc = patch_encoder(patch_global=patch_global, num_register_tokens=8 if patch_global else None)
        delta = (enc(x2, kv_mask=mask)[0, first] - enc(x, kv_mask=mask)[0, first]).abs().max()
        assert (delta > 1e-6) == changes, (patch_global, delta)


@pytest.mark.parametrize("fill", [1e3, -1e4, float("inf"), float("nan")])
def test_padded_constituents_do_not_leak(fill):
    # lengths leave a partly padded patch and several fully padded ones
    enc = patch_encoder()
    x, mask = padded_batch([7, 24, 1])
    out = enc(x, kv_mask=mask)
    garbage = torch.where(mask[..., None], x, fill)
    out_g = enc(garbage, kv_mask=mask)
    assert torch.equal(out_g[mask], out[mask])
    assert out_g.isfinite().all()
    # control: a real constituent does change the others
    x2 = x.clone()
    x2[0, 3] += 1.0
    assert (enc(x2, kv_mask=mask)[mask] - out[mask]).abs().max() > 1e-3


def test_interleaved_padding():
    # without a sort the real constituents are compacted in their input order before patching
    enc = patch_encoder()
    x, _ = padded_batch([SEQ])
    torch.manual_seed(2)
    mask = torch.rand(1, SEQ) < 0.5
    order = torch.argsort((~mask).int(), dim=-1, stable=True)
    out = enc(x, kv_mask=mask)
    out_c = enc(x[:, order[0]], kv_mask=mask[:, order[0]])
    torch.testing.assert_close(out[mask], out_c[mask[:, order[0]]])


def test_no_padding_mask_and_empty_events():
    enc = patch_encoder()
    x, mask = padded_batch([SEQ, SEQ])
    torch.testing.assert_close(enc(x), enc(x, kv_mask=mask))
    # an event with no real constituent at all stays finite, with and without registers
    for registers in (8, None):
        enc = patch_encoder(num_register_tokens=registers)
        x, mask = padded_batch([0, 5])
        assert enc(x, kv_mask=mask).isfinite().all()


def test_registers_take_part():
    # the register tokens are not returned, but they do reach the constituents through the global stage
    enc = patch_encoder()
    x, mask = padded_batch([10, 24])
    out = enc(x, kv_mask=mask)
    assert out.shape == x.shape
    with torch.no_grad():
        enc.register_tokens.add_(1.0)
    assert (enc(x, kv_mask=mask)[mask] - out[mask]).abs().max() > 1e-3


@pytest.mark.parametrize("value_residual", [False, True])
def test_value_residual(value_residual):
    enc = patch_encoder(value_residual=value_residual, num_layers=3)
    names = [n for n, _ in enc.named_parameters() if "value_residual_mix" in n]
    if not value_residual:
        assert not names
        return
    # every layer after the first mixes in the first layer's values, in the local and the global attention
    assert sorted({n.split(".")[1] for n in names}) == ["1", "2"]
    assert any(".attn." in n for n in names)
    assert any(".patch_attn." in n for n in names)
    enc.train()
    x, mask = padded_batch([10, 24])
    enc(x, kv_mask=mask)[mask].square().sum().backward()
    for n, p in enc.named_parameters():
        if "value_residual_mix" in n:
            assert p.grad is not None
            assert p.grad.abs().sum() > 0, n


@pytest.mark.parametrize("patch_global", [True, False])
def test_gradients_finite_and_complete(patch_global):
    # every parameter is trained, in both modes
    enc = patch_encoder(patch_global=patch_global, num_register_tokens=8 if patch_global else None).train()
    x, mask = padded_batch([7, 24, 1])
    x = torch.where(mask[..., None], x, float("nan")).requires_grad_(True)
    out = enc(x, kv_mask=mask)
    out[mask].square().sum().backward()
    assert x.grad[mask].isfinite().all()
    assert (x.grad[~mask] == 0).all()
    no_grad = [n for n, p in enc.named_parameters() if p.grad is None or not p.grad.isfinite().all() or not p.grad.abs().sum() > 0]
    assert not no_grad, no_grad


def test_bad_configurations_refused():
    # patch_size only means something with attn_type patch, which needs it
    with pytest.raises(ValueError, match="only used with attn_type 'patch'"):
        Encoder(num_layers=1, dim=DIM, attn_type="flash-varlen", patch_size=P)
    with pytest.raises(ValueError, match="only used with attn_type 'patch'"):
        Encoder(num_layers=1, dim=DIM, attn_type="torch", patch_size=P)
    for patch_size in (None, 0):
        with pytest.raises(ValueError, match="positive patch_size"):
            Encoder(num_layers=1, dim=DIM, attn_type="patch", patch_size=patch_size)
    with pytest.raises(ValueError, match="window"):
        Encoder(num_layers=1, dim=DIM, attn_type="patch", patch_size=P, window_size=4)
    # without the global stage the registers would never reach a constituent
    with pytest.raises(ValueError, match="Register tokens"):
        Encoder(num_layers=1, dim=DIM, attn_type="patch", patch_size=P, patch_global=False, num_register_tokens=8)
    enc = patch_encoder()
    with pytest.raises(ValueError, match="backends"):
        enc.set_backend("flash")


def test_set_backend_keeps_patch_attention():
    enc = patch_encoder()
    x, mask = padded_batch([7, 24])
    out = enc(x, kv_mask=mask)
    for backend in ("torch", "patch"):
        enc.set_backend(backend)
        assert enc.attn_type == "patch"
        torch.testing.assert_close(enc(x, kv_mask=mask), out)


def sorted_positions(value, mask, period=None):
    # where each token lands in the order the encoder uses (padding last)
    if period is not None:
        value = periodic_sort_value(value, mask, period)
    return torch.argsort(sort_index(value, mask), dim=-1)


def test_periodic_sort_keeps_wrap_neighbours_together():
    # five tokens either side of phi = +-pi and five near 0: the raw order puts the +-pi group at both ends
    phi = torch.tensor([[3.10, -3.12, 3.05, -3.0, 3.13, 0.0, 0.1, -0.1, 0.2, -0.2]], dtype=torch.float64)
    mask = torch.ones_like(phi, dtype=torch.bool)
    group = torch.tensor([0, 1, 2, 3, 4])
    raw = sorted_positions(phi, mask)[0, group]
    assert raw.max() - raw.min() == 9
    periodic = sorted_positions(phi, mask, 2 * torch.pi)[0, group]
    assert periodic.max() - periodic.min() == 4
    # and within the group the order follows the angle across the wrap point: 3.05, 3.10, 3.13, -3.12, -3.0
    assert torch.equal(torch.argsort(periodic), torch.tensor([2, 0, 4, 1, 3]))


def test_periodic_sort_is_rotation_invariant():
    # the cut is the event's largest empty gap, so rotating every phi leaves the order unchanged
    torch.manual_seed(3)
    phi = torch.rand(4, SEQ, dtype=torch.float64) * 2 * torch.pi - torch.pi
    mask = torch.arange(SEQ)[None] < torch.tensor([[SEQ], [13], [2], [1]])
    ref = sorted_positions(phi, mask, 2 * torch.pi)

    def rotate(angle):
        return torch.remainder(phi + angle + torch.pi, 2 * torch.pi) - torch.pi

    for angle in (0.5, 2.0, -3.0, 10.0):
        assert torch.equal(sorted_positions(rotate(angle), mask, 2 * torch.pi)[mask], ref[mask])
    # the raw order does change under rotation, where tokens cross phi = +-pi
    assert not torch.equal(sorted_positions(rotate(2.0), mask)[mask], sorted_positions(phi, mask)[mask])


@pytest.mark.parametrize("fill", [0.0, 1e4, float("inf"), float("nan")])
def test_periodic_sort_ignores_padding(fill):
    torch.manual_seed(4)
    phi = torch.rand(3, SEQ, dtype=torch.float64) * 2 * torch.pi - torch.pi
    mask = torch.rand(3, SEQ) < 0.6
    mask[2] = False  # an event without real tokens
    ref = sorted_positions(phi, mask, 2 * torch.pi)
    got = sorted_positions(torch.where(mask, phi, fill), mask, 2 * torch.pi)
    assert torch.equal(got[mask], ref[mask])
    # real tokens take the first slots, padding the rest
    assert torch.equal(torch.sort(got[0][mask[0]]).values, torch.arange(int(mask[0].sum())))
    assert periodic_sort_value(torch.where(mask, phi, fill), mask, 2 * torch.pi)[mask].isfinite().all()


def test_other_attn_types_unchanged():
    # without attn_type patch the encoder builds plain EncoderLayers, as before
    enc = Encoder(num_layers=2, dim=DIM, attn_type="torch", attn_kwargs={"num_heads": HEADS})
    assert enc.patch_size is None
    assert not any(isinstance(layer, PatchEncoderLayer) for layer in enc.layers)
