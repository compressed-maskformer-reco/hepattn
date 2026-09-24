"""Padding and normalisation contracts of the Linformer backend, through `Attention`.

Linformer projects keys and values along the sequence axis, so a padded row that survives to
the projection is blended into every projected column and no later mask can remove it. These
tests pin the two consequences a training run cannot see: padded rows must contribute nothing,
and the q/k/v norms must run BEFORE the zeroing (a LayerNorm bias turns a zero row into a
non-zero one). Reference: the same input truncated to its valid length.
"""

import pytest
import torch

from hepattn.models.attention import Attention

DIM, HEADS, B, N = 32, 4, 2, 12
REAL = 8  # valid tokens per event; the rest is padding


def _attention(qkv_norm: bool, seed: int = 0) -> Attention:
    torch.manual_seed(seed)
    kwargs = {"qkv_norm": True, "norm": "LayerNorm"} if qkv_norm else {}
    a = Attention(dim=DIM, num_heads=HEADS, attn_type="linformer", linformer_seq_len=N, linformer_proj_dim=N, **kwargs)
    if qkv_norm:
        # A zero bias would hide the ordering bug (LayerNorm(0) == bias), so set it away from zero.
        with torch.no_grad():
            a.k_norm.bias.fill_(0.1)
            a.v_norm.bias.fill_(0.1)
    return a.eval()


def _inputs(seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(B, N, DIM)
    kv_mask = torch.zeros(B, N, dtype=torch.bool)
    kv_mask[:, :REAL] = True
    return x, kv_mask


@pytest.mark.parametrize("qkv_norm", [False, True])
def test_padded_equals_truncated(qkv_norm):
    """With qkv_norm and a biased norm this is the order guard: zeroing before the norm would
    leave every padded row at the bias and blend it into the projection.
    """
    a = _attention(qkv_norm)
    x, kv_mask = _inputs()
    with torch.no_grad():
        padded = a(x, x, x, kv_mask=kv_mask)[:, :REAL]
        truncated = a(x[:, :REAL], x[:, :REAL], x[:, :REAL])
    torch.testing.assert_close(padded, truncated, atol=1e-6, rtol=1e-6)


def test_garbage_in_padding_cannot_reach_real_rows():
    a = _attention(qkv_norm=True)
    x, kv_mask = _inputs()
    y = x.clone()
    y[:, REAL:] = 1e3 * torch.randn(B, N - REAL, DIM)
    with torch.no_grad():
        o1 = a(x, x, x, kv_mask=kv_mask)[:, :REAL]
        o2 = a(y, y, y, kv_mask=kv_mask)[:, :REAL]
    torch.testing.assert_close(o1, o2, atol=1e-6, rtol=1e-6)
    # non-vacuous: without the mask the same garbage does move the real rows
    with torch.no_grad():
        u1 = a(x, x, x)[:, :REAL]
        u2 = a(y, y, y)[:, :REAL]
    assert not torch.allclose(u1, u2, atol=1e-3)


def test_qkv_norms_receive_gradient_and_nothing_is_dead():
    a = _attention(qkv_norm=True).train()
    x, kv_mask = _inputs()
    a(x, x, x, kv_mask=kv_mask).square().mean().backward()
    dead = [n for n, p in a.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"parameters with no gradient: {dead}"


def test_attn_mask_is_refused():
    a = _attention(qkv_norm=False)
    x, kv_mask = _inputs()
    with pytest.raises((ValueError, AssertionError)):
        a(x, x, x, kv_mask=kv_mask, attn_mask=torch.ones(B, N, N, dtype=torch.bool))


def test_self_attention_call_without_k_v():
    a = _attention(qkv_norm=False)
    x, kv_mask = _inputs()
    with torch.no_grad():
        torch.testing.assert_close(a.attn(x, kv_mask=kv_mask), a.attn(x, x, x, kv_mask=kv_mask))


def test_value_residual_is_rejected_with_linformer():
    with pytest.raises(ValueError, match="value_residual"):
        Attention(dim=DIM, num_heads=HEADS, attn_type="linformer", linformer_seq_len=N, linformer_proj_dim=N, value_residual=True)


def test_set_backend_keeps_weights_and_refuses_switch():
    a = _attention(qkv_norm=False)
    before = {n: p.detach().clone() for n, p in a.named_parameters()}
    a.set_backend("linformer")
    for n, p in a.named_parameters():
        torch.testing.assert_close(p, before[n])
    with pytest.raises(ValueError, match="switch"):
        a.set_backend("torch")
    a.reset_parameters()  # must be a no-op for linformer, not an AttributeError
