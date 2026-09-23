"""Padding and normalisation contracts of the Linformer backend, through `Attention`.

Linformer projects keys and values along the sequence axis, so a padded row that survives to
the projection is blended into every projected column and no later mask can remove it. These
tests pin the two consequences that a training run cannot see: padded rows must contribute
nothing, and the q/k/v norms must run BEFORE the zeroing (a LayerNorm bias turns a zero row
into a non-zero one). Reference: the same input truncated to its valid length.
"""

import torch

from hepattn.models.attention import Attention

DIM, HEADS, B, N = 32, 4, 2, 12
REAL = 8  # valid tokens per event; the remaining N - REAL are padding


def _attention(qkv_norm: bool, seed: int = 0) -> Attention:
    torch.manual_seed(seed)
    a = Attention(dim=DIM, num_heads=HEADS, attn_type="linformer", linformer_seq_len=N, linformer_proj_dim=N, qkv_norm=qkv_norm)
    if qkv_norm:
        # The repo's LayerNorm has no bias, and LayerNorm(0) == 0 without one, so the order of
        # "normalise" and "zero the padding" is invisible with it. Swap in a biased norm: if the
        # zeroing ran first, every padded row would come out equal to the bias and be blended
        # into the projection. This is the case the order guard exists for.
        width = a.attn.k_norm.normalized_shape if hasattr(a.attn.k_norm, "normalized_shape") else a.attn.k_norm.weight.shape
        for name in ("k_norm", "v_norm"):
            biased = torch.nn.LayerNorm(width)
            with torch.no_grad():
                biased.bias.fill_(0.1)
            setattr(a.attn, name, biased)
    return a.eval()


def _inputs(seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(B, N, DIM)
    kv_mask = torch.zeros(B, N, dtype=torch.bool)
    kv_mask[:, :REAL] = True
    return x, kv_mask


def test_padded_equals_truncated_without_norm():
    a = _attention(qkv_norm=False)
    x, kv_mask = _inputs()
    with torch.no_grad():
        padded = a(x, kv_mask=kv_mask)[:, :REAL]
        truncated = a(x[:, :REAL])
    torch.testing.assert_close(padded, truncated, atol=1e-6, rtol=1e-6)


def test_padded_equals_truncated_with_norm_and_nonzero_bias():
    """The order-of-operations guard: norm first, then zero the padding.

    With the k/v LayerNorm bias at 0.1, zeroing before the norm would leave every padded row at
    the bias value and blend it into the projection; the truncated reference exposes that.
    """
    a = _attention(qkv_norm=True)
    x, kv_mask = _inputs()
    with torch.no_grad():
        padded = a(x, kv_mask=kv_mask)[:, :REAL]
        truncated = a(x[:, :REAL])
    torch.testing.assert_close(padded, truncated, atol=1e-6, rtol=1e-6)


def test_garbage_in_padding_cannot_reach_real_rows():
    a = _attention(qkv_norm=True)
    x, kv_mask = _inputs()
    y = x.clone()
    y[:, REAL:] = 1e3 * torch.randn(B, N - REAL, DIM)  # arbitrary content in the padded rows
    with torch.no_grad():
        o1 = a(x, kv_mask=kv_mask)[:, :REAL]
        o2 = a(y, kv_mask=kv_mask)[:, :REAL]
    torch.testing.assert_close(o1, o2, atol=1e-6, rtol=1e-6)
    # non-vacuous: without the mask the same garbage does move the real rows
    with torch.no_grad():
        u1 = a(x)[:, :REAL]
        u2 = a(y)[:, :REAL]
    assert not torch.allclose(u1, u2, atol=1e-3)


def test_qkv_norms_receive_gradient():
    a = _attention(qkv_norm=True).train()
    x, kv_mask = _inputs()
    a(x, kv_mask=kv_mask).square().mean().backward()
    for name in ("q_norm", "k_norm", "v_norm"):
        for pname, p in getattr(a.attn, name).named_parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0, f"{name}.{pname} received no gradient"


def test_no_dead_parameters_with_qkv_norm():
    """Every parameter of a Linformer Attention with qkv_norm must receive gradient: the norms
    that Attention itself owns must not be duplicated as dead weights beside the Linformer's.
    """
    a = _attention(qkv_norm=True).train()
    x, kv_mask = _inputs()
    a(x, kv_mask=kv_mask).square().mean().backward()
    dead = [n for n, p in a.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"parameters with no gradient: {dead}"
