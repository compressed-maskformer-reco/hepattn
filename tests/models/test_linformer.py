import pytest
import torch

from hepattn.models.attention import Attention
from hepattn.models.linformer import LinformerAttention

torch.manual_seed(42)


@pytest.mark.parametrize("kv_len", [None, 96, 160])
def test_linformer_shapes_and_padding(kv_len):
    batch_size, q_len, dim, num_heads = 3, 150, 64, 8
    attn = Attention(dim=dim, num_heads=num_heads, attn_type="linformer", linformer_seq_len=160, linformer_proj_dim=16)
    assert isinstance(attn.attn, LinformerAttention)
    assert not hasattr(attn, "in_proj_weight")

    q = torch.randn(batch_size, q_len, dim)
    kv = torch.randn(batch_size, kv_len, dim) if kv_len else None
    kv_mask = torch.rand(batch_size, kv_len or q_len) > 0.3
    kv_mask[:, 0] = True

    out = attn(q, kv, kv_mask=kv_mask)
    assert out.shape == (batch_size, q_len, dim)
    assert torch.isfinite(out).all()

    # Padded key/value slots must not influence the output: replace them with garbage and compare
    if kv is not None:
        kv_garbage = torch.where(kv_mask.unsqueeze(-1), kv, 1e3 * torch.randn_like(kv))
        out_garbage = attn(q, kv_garbage, kv_mask=kv_mask)
        torch.testing.assert_close(out, out_garbage)


def test_linformer_rejects_attention_mask_and_backend_switch():
    attn = Attention(dim=32, num_heads=4, attn_type="linformer", linformer_seq_len=8, linformer_proj_dim=4)
    q = torch.randn(2, 8, 32)
    with pytest.raises(AssertionError):
        attn(q, attn_mask=torch.ones(2, 8, 8, dtype=torch.bool))
    with pytest.raises(ValueError, match="linformer"):
        attn.set_backend("torch")
    with pytest.raises(ValueError, match="linformer"):
        Attention(dim=32, num_heads=4, attn_type="torch").set_backend("linformer")


def test_linformer_set_backend_keeps_weights():
    attn = Attention(dim=32, num_heads=4, attn_type="linformer", linformer_seq_len=8, linformer_proj_dim=4)
    proj = attn.attn.proj_k.detach().clone()
    attn.set_backend("linformer")
    torch.testing.assert_close(attn.attn.proj_k, proj)


def test_linformer_sequence_too_long():
    attn = Attention(dim=32, num_heads=4, attn_type="linformer", linformer_seq_len=8, linformer_proj_dim=4)
    with pytest.raises(AssertionError, match="at most 8"):
        attn(torch.randn(1, 9, 32))


def test_linformer_honours_bias():
    """Bias is a config option like any other; the backend used to hardcode it off."""
    kwargs = {"dim": 16, "num_heads": 2, "attn_type": "linformer", "linformer_seq_len": 8, "linformer_proj_dim": 4}
    assert Attention(bias=True, **kwargs).attn.to_q.bias is not None
    assert Attention(bias=False, **kwargs).attn.to_q.bias is None


def test_linformer_honours_qkv_norm():
    """The norms act on the real tokens, before the sequence projection."""
    kwargs = {"dim": 16, "num_heads": 2, "attn_type": "linformer", "linformer_seq_len": 8, "linformer_proj_dim": 4}
    plain = Attention(qkv_norm=False, **kwargs)
    normed = Attention(qkv_norm=True, **kwargs)
    assert not hasattr(plain.attn, "q_norm")
    assert hasattr(normed.attn, "q_norm")

    # The norms must actually be applied, not merely built
    normed.attn.load_state_dict(plain.attn.state_dict(), strict=False)
    x = torch.randn(2, 8, 16)
    assert not torch.allclose(plain(x), normed(x), atol=1e-6)


def test_linformer_value_residual():
    """The first layer supplies the values the later layers mix in, as for the other backends."""
    kwargs = {"dim": 16, "num_heads": 2, "attn_type": "linformer", "linformer_seq_len": 8, "linformer_proj_dim": 4}
    first = Attention(value_residual=True, is_first_layer=True, **kwargs)
    later = Attention(value_residual=True, is_first_layer=False, **kwargs)

    assert not hasattr(first.attn, "value_residual_mix"), "the first layer supplies values, it does not mix them"
    assert hasattr(later.attn, "value_residual_mix")

    x = torch.randn(2, 8, 16)
    initial_values = {}
    first(x, initial_values=initial_values)

    # (B, H, M, Dh): the values of the real tokens, before the sequence projection
    assert initial_values["v"].shape == (2, 2, 8, 8)

    mixed = later(x, initial_values=initial_values)
    unmixed = later(x, initial_values=None)
    assert not torch.allclose(mixed, unmixed, atol=1e-6)

    # and the mix is trained, so it has to receive gradient
    mixed.sum().backward()
    assert later.attn.value_residual_mix[0].weight.grad is not None


def test_linformer_value_residual_needs_self_attention():
    """The mix is a per-query weight applied to the values, so the two sequences must be the same length."""
    attn = Attention(dim=16, num_heads=2, attn_type="linformer", linformer_seq_len=8, linformer_proj_dim=4, value_residual=True, is_first_layer=False)
    with pytest.raises(AssertionError, match="self-attention"):
        attn(torch.randn(2, 8, 16), torch.randn(2, 6, 16), initial_values={"v": torch.randn(2, 2, 6, 8)})


def test_linformer_honours_query_masks():
    """A query mask reaches the other backends through merge_masks, which this path never calls.

    The query axis is neither compressed nor mixed, so the mask cannot change a valid row; it only
    says which rows are real. The backend zeroes the padded ones, which is what the torch backend
    returns for a row that is masked everywhere -- checked against it here rather than asserted.
    """
    batch_size, q_len, kv_len, dim = 2, 10, 8, 16
    attn = Attention(dim=dim, num_heads=2, attn_type="linformer", linformer_seq_len=kv_len, linformer_proj_dim=4)
    q = torch.randn(batch_size, q_len, dim)
    kv = torch.randn(batch_size, kv_len, dim)
    q_mask = torch.ones(batch_size, q_len, dtype=torch.bool)
    q_mask[:, -3:] = False

    out = attn(q, kv, q_mask=q_mask)
    unmasked = attn(q, kv)

    # Valid rows are untouched; padded rows carry the output projection's bias and nothing else
    torch.testing.assert_close(out[q_mask], unmasked[q_mask])
    torch.testing.assert_close(out[~q_mask], attn.attn.to_out.bias.expand_as(out[~q_mask]))

    # Which is what scaled_dot_product_attention gives a row it is told to mask everywhere
    zeroed = attn.attn.to_out(torch.zeros(1, attn.attn.dim_head * attn.attn.heads))
    torch.testing.assert_close(out[~q_mask][:1], zeroed)
