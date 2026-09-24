"""Linformer attention (Wang et al., arXiv:2006.04768) in the shape hepattn's Attention expects.

Trivial rewriting of https://github.com/lucidrains/linformer. Two things matter here and are
easy to get wrong:

* Keys and values are projected ALONG THE SEQUENCE AXIS (``E[n, k]``), so every projected
  column is a mixture of every original position. Padding therefore has to be removed
  BEFORE the projection (a score-space mask cannot undo the blend afterwards), and a
  per-query ``attn_mask`` cannot be honoured at all: there is no projected column that
  "is" hit j. This module zeroes padded rows pre-projection and refuses ``attn_mask``.
* The q/k/v norms that hepattn's ``Attention`` owns (``hybrid_norm``) have to be applied on
  this path too, and BEFORE the zeroing: a LayerNorm with a bias maps a zero row to the
  bias, which would leak into the projection.
"""

import math

import torch
from torch import Tensor, nn


def _init_projection(tensor: Tensor) -> Tensor:
    std = 1 / math.sqrt(tensor.shape[-1])
    tensor.uniform_(-std, std)
    return tensor


class LinformerAttention(nn.Module):
    def __init__(self, dim: int, seq_len: int, k: int = 256, heads: int = 8, dim_head: int | None = None, dropout: float = 0.0) -> None:
        """Linformer self/cross attention with a learned sequence-axis projection.

        Args:
            dim: model dimension.
            seq_len: maximum key/value sequence length; the projection has this many rows and
                is sliced for shorter inputs.
            k: projected (compressed) sequence length.
            heads: number of attention heads.
            dim_head: per-head dimension; ``dim // heads`` if None.
            dropout: dropout on the attention weights.
        """
        super().__init__()
        assert dim % heads == 0, "dimension must be divisible by the number of heads"
        self.seq_len = seq_len
        self.k = k
        self.heads = heads
        self.dim_head = dim_head if dim_head is not None else dim // heads

        inner = self.dim_head * heads
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.proj_k = nn.Parameter(_init_projection(torch.zeros(seq_len, k)))
        self.proj_v = nn.Parameter(_init_projection(torch.zeros(seq_len, k)))
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner, dim)

    def forward(
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        attn_mask: Tensor | None = None,
        kv_mask: Tensor | None = None,
        qkv_norms: tuple[nn.Module, nn.Module, nn.Module] | None = None,
        **_ignored,
    ) -> Tensor:
        """Attend from ``q`` (B, N, D) to ``k``/``v`` (B, M, D); self-attention when both are None.

        ``kv_mask`` (B, M) is True for real key/value slots; padded slots are zeroed before the
        sequence projection so they contribute exactly nothing. ``qkv_norms`` are Attention's
        q/k/v norm modules, applied here to the projected q/k/v before that zeroing.
        ``attn_mask`` is rejected: see the module docstring.

        Raises:
            ValueError: if ``attn_mask`` is given, or if only one of ``k``/``v`` is given.
        """
        if attn_mask is not None:
            raise ValueError("LinformerAttention cannot apply a per-query attn_mask: keys are mixed along the sequence axis before scoring")
        if (k is None) != (v is None):
            raise ValueError("pass both k and v, or neither (self-attention)")
        if k is None:
            k, v = q, q
        assert v is not None
        b, n, _ = q.shape
        kv_len = k.shape[1]
        assert kv_len <= self.seq_len, f"key/value length {kv_len} exceeds seq_len {self.seq_len}"

        queries = self.to_q(q)
        keys = self.to_k(k)
        values = self.to_v(v)

        if qkv_norms is not None:
            q_norm, k_norm, v_norm = qkv_norms
            queries = q_norm(queries)
            keys = k_norm(keys)
            values = v_norm(values)

        if kv_mask is not None:
            keep = kv_mask[..., None].to(keys.dtype)
            keys = keys * keep
            values = values * keep

        # Project along the sequence axis: (B, M, inner) -> (B, k, inner). Slice the projection
        # to the actual length so shorter sequences use the leading rows.
        keys = torch.einsum("bnd,nk->bkd", keys, self.proj_k[:kv_len])
        values = torch.einsum("bnd,nk->bkd", values, self.proj_v[:kv_len])

        h, d_h, k_num = self.heads, self.dim_head, self.k
        queries = queries.reshape(b, n, h, d_h).transpose(1, 2)
        keys = keys.reshape(b, k_num, h, d_h).transpose(1, 2)
        values = values.reshape(b, k_num, h, d_h).transpose(1, 2)

        scores = torch.einsum("bhnd,bhkd->bhnk", queries, keys) * (d_h**-0.5)
        attn = self.dropout(scores.softmax(dim=-1))
        out = torch.einsum("bhnk,bhkd->bhnd", attn, values)
        return self.to_out(out.transpose(1, 2).reshape(b, n, -1))
