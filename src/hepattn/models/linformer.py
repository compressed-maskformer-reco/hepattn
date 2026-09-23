"""Linformer attention: linear-complexity attention via a low-rank projection of the keys and values.

A trivial rewriting of https://github.com/lucidrains/linformer (arXiv:2006.04768) into a form that
fits the hepattn attention/decoder setup. Ported from lgray/compression-work (76f2078).

The keys and values of a sequence of length ``n`` are projected along the sequence axis down to
``k`` "virtual" tokens with a learned ``(seq_len, k)`` matrix, so the attention matrix is ``(n, k)``
instead of ``(n, n)``. The projection mixes sequence positions, which is why per-position attention
masks cannot be honoured: padded keys/values are zeroed before the projection instead, and an
explicit attention mask is refused. A *query* padding mask is a different thing and is honoured:
the query axis is neither compressed nor mixed, so it only decides which output rows are real.
"""

import math

import torch
from torch import Tensor, nn

from hepattn.models.norm import LayerNorm


def _init_projection(tensor: Tensor) -> Tensor:
    std = 1 / math.sqrt(tensor.shape[-1])
    tensor.uniform_(-std, std)
    return tensor


class LinformerAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        k: int = 256,
        heads: int = 8,
        dim_head: int | None = None,
        one_kv_head: bool = False,
        share_kv: bool = False,
        dropout: float = 0.0,
        bias: bool = True,
        qkv_norm: bool = False,
        value_residual: bool = False,
        is_first_layer: bool = False,
    ) -> None:
        """Linformer attention module with its own input and output projections.

        Parameters
        ----------
        dim : int
            Embedding dimension of the inputs and the output.
        seq_len : int
            Maximum key/value sequence length. Shorter sequences use a slice of the projection.
        k : int
            Projected sequence length (the low rank). Values at or above ``seq_len`` compress nothing.
        heads : int
            Number of attention heads.
        dim_head : int | None
            Dimension per head; defaults to ``dim // heads``.
        one_kv_head : bool
            Share a single key/value head across all query heads.
        share_kv : bool
            Use the key projection for the values too.
        dropout : float
            Dropout on the attention weights.
        bias : bool
            Whether the input and output projections carry a bias.
        qkv_norm : bool
            Normalise the queries, keys and values. Applied to the real tokens, before the sequence
            projection, so it means the same thing as it does for the other backends.
        value_residual : bool
            Mix this layer's values with the first layer's, as in the other backends. Also applied
            before the sequence projection, and only meaningful for self-attention.
        is_first_layer : bool
            Whether this is the first layer, which supplies the values the later layers mix in.
        """
        super().__init__()
        assert dim % heads == 0, "dimension must be divisible by the number of heads"

        self.seq_len = seq_len
        self.k = k
        self.heads = heads
        self.dim_head = dim_head if dim_head is not None else dim // heads
        self.share_kv = share_kv
        self.qkv_norm = qkv_norm
        self.value_residual = value_residual
        self.is_first_layer = is_first_layer

        self.to_q = nn.Linear(dim, self.dim_head * heads, bias=bias)

        kv_dim = self.dim_head if one_kv_head else self.dim_head * heads
        self.to_k = nn.Linear(dim, kv_dim, bias=bias)
        self.proj_k = nn.Parameter(_init_projection(torch.zeros(seq_len, k)))

        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=bias)
            self.proj_v = nn.Parameter(_init_projection(torch.zeros(seq_len, k)))

        if qkv_norm:
            self.q_norm = LayerNorm(self.dim_head * heads)
            self.k_norm = LayerNorm(kv_dim)
            self.v_norm = LayerNorm(kv_dim)

        if value_residual and not is_first_layer:
            self.value_residual_mix = nn.Sequential(nn.Linear(dim, heads), nn.Sigmoid())

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(self.dim_head * heads, dim)

    def forward(
        self,
        q: Tensor,
        kv: Tensor | None = None,
        q_mask: Tensor | None = None,
        kv_mask: Tensor | None = None,
        initial_values: dict | None = None,
    ) -> Tensor:
        """Attend from ``q`` to ``kv`` (or to ``q`` itself for self-attention).

        Parameters
        ----------
        q : Tensor
            Queries of shape (B, N, D).
        kv : Tensor | None
            Keys/values of shape (B, M, D); ``q`` if None.
        q_mask : Tensor | None
            Boolean (B, N) mask, True for valid query slots. Padded queries get a zeroed attention
            output, which is what the other backends return for a row that is masked everywhere.
        kv_mask : Tensor | None
            Boolean (B, M) mask, True for valid key/value slots. Padded slots are zeroed before the
            sequence projection so they contribute nothing to the projected keys and values.
        initial_values : dict | None
            Value-residual store, as for the other backends: the first layer writes its values into
            it and the later layers mix them in. Ignored unless ``value_residual`` is set.
        """
        if kv is None:
            kv = q
        batch_size, num_queries, _ = q.shape
        kv_len = kv.shape[1]
        assert kv_len <= self.seq_len, f"the key/value sequence length must be at most {self.seq_len}, got {kv_len}"

        queries = self.to_q(q)
        keys = self.to_k(kv)
        values = self.to_v(kv) if not self.share_kv else keys

        # Normalise the real tokens, before the sequence projection and before the padded slots are
        # zeroed: LayerNorm has a bias, so a zeroed slot would not stay zero if the order were swapped
        if self.qkv_norm:
            queries = self.q_norm(queries)
            keys = self.k_norm(keys)
            values = self.v_norm(values) if not self.share_kv else keys

        if kv_mask is not None:
            valid = kv_mask.unsqueeze(-1).to(keys.dtype)
            keys = keys * valid
            values = values * valid

        # Split heads: queries (B, H, N, Dh); keys/values (B, H, M, Dh), broadcast if a single kv head
        queries = queries.reshape(batch_size, num_queries, self.heads, self.dim_head).transpose(1, 2)
        keys = keys.reshape(batch_size, kv_len, -1, self.dim_head).transpose(1, 2).expand(-1, self.heads, -1, -1)
        values = values.reshape(batch_size, kv_len, -1, self.dim_head).transpose(1, 2).expand(-1, self.heads, -1, -1)

        # Mix in the first layer's values, still on the real tokens so that the residual means the
        # same thing it does for the other backends: the compression is a later, separate step
        if self.value_residual and initial_values is not None:
            if self.is_first_layer:
                initial_values["v"] = values
            else:
                assert num_queries == kv_len, "value_residual mixes a per-query weight into the values, so it needs self-attention"
                mix = self.value_residual_mix(q).unsqueeze(-1).transpose(-2, -3)
                values = values * mix + initial_values["v"] * (1.0 - mix)

        # Allow variable sequence lengths (up to the maximum) by slicing the projections
        proj_k = self.proj_k[:kv_len]
        proj_v = self.proj_v[:kv_len] if not self.share_kv else proj_k

        # Project keys and values along the sequence axis down to k virtual tokens
        keys = torch.einsum("bhnd,nk->bhkd", keys, proj_k)
        values = torch.einsum("bhnd,nk->bhkd", values, proj_v)

        scores = torch.einsum("bhnd,bhkd->bhnk", queries, keys) * (self.dim_head**-0.5)
        attn = self.dropout(scores.softmax(dim=-1))
        out = torch.einsum("bhnk,bhkd->bhnd", attn, values)

        # Nothing here mixes the query axis -- a row of `out` is built from its own query and from
        # the keys and values alone -- so a query padding mask cannot change a valid row, and
        # zeroing the padded ones reproduces exactly what `scaled_dot_product_attention` returns for
        # a row whose mask is False everywhere. The compressed KEY axis is the one Linformer cannot
        # mask per position; that restriction is untouched, and `kv_mask` still goes in before the
        # projection.
        if q_mask is not None:
            out = out * q_mask.unsqueeze(-2).unsqueeze(-1).to(out.dtype)

        out = out.transpose(1, 2).reshape(batch_size, num_queries, -1)
        return self.to_out(out)
