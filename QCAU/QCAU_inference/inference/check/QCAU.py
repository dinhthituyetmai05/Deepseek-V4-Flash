"""Query-Core Attention Unit (QCAU).

This module owns exactly the shaded block of the QCAU dataflow diagram:

    c_q --> [Query Up (W_UQ)] --> [per-head RMSNorm] --> [RoPE forward] --> q
    q, kv, topk_idxs, attn_sink --> [Sparse MQA, online softmax] --> o_raw
    o_raw --> [RoPE inverse] --> o

Everything before it (Query Down / wq_a / q_norm, KV build, top-k selection)
and everything after it (the grouped output projection wo_a / wo_b) stays in
model.py.

Reference semantics are taken from `kernel.sparse_attn_kernel` (TileLang):

  * tiles of `tile_size` (kernel block = 64) columns,
  * FlashAttention-style online softmax with a running max in FP32,
  * invalid (-1) addresses contribute a zeroed KV row and a -inf score,
  * the attention-sink term is added to the denominator only, after the loop,
  * the P @ V product uses BF16 probabilities (`acc_s_cast` in the kernel)
    with FP32 accumulation -- see `prob_dtype` below.
"""

from typing import Callable, Optional, Union

import torch


WeightOrProjector = Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply the QCAU query-up weight: q = c_Q @ W_UQ.T."""
    if weight.ndim != 2:
        raise ValueError("2D W_UQ is required for the linear projection")
    return torch.matmul(x, weight.transpose(-1, -2))


def _rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """Per-head RMSNorm used after Query Up.

    Kept in the input dtype on purpose: model_origin.py normalises q inline
    with `q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)`, i.e. in
    BF16, not through the FP32 `RMSNorm` module.
    """
    dtype = x.dtype
    rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    y = x * rms
    if weight is not None:
        y = y * weight.to(dtype)
    return y.to(dtype)


def _apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    rope_head_dim: int,
    start_pos: int = 0,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply RoPE to the final rope_head_dim values of q or o."""
    if rope_head_dim == 0:
        return x
    if rope_head_dim % 2 != 0:
        raise ValueError("rope_head_dim must be even")
    if x.ndim != 4:
        raise ValueError("RoPE input must have shape [B,S,H,D]")
    if x.size(-1) < rope_head_dim:
        raise ValueError("last dimension is smaller than rope_head_dim")
    if freqs_cis.ndim != 2:
        raise ValueError("freqs_cis must have shape [S, rope_head_dim // 2]")

    n_pairs = rope_head_dim // 2
    if freqs_cis.size(-1) != n_pairs:
        raise ValueError("freqs_cis has the wrong number of rotary pairs")
    if freqs_cis.size(0) == x.size(1):
        position_freqs = freqs_cis
    elif freqs_cis.size(0) >= start_pos + x.size(1):
        position_freqs = freqs_cis[start_pos:start_pos + x.size(1)]
    else:
        raise ValueError("freqs_cis does not cover the QCAU query positions")

    out = x.clone()
    tail = out[..., -rope_head_dim:]
    tail_complex = torch.view_as_complex(
        tail.float().contiguous().unflatten(-1, (n_pairs, 2))
    )
    freqs = position_freqs.conj() if inverse else position_freqs
    freqs = freqs.view(1, x.size(1), 1, n_pairs)
    rotated = torch.view_as_real(tail_complex * freqs).flatten(-2)
    out[..., -rope_head_dim:] = rotated.to(out.dtype)
    return out


def _query_up(
    c_q: torch.Tensor,
    w_uq: WeightOrProjector,
    *,
    n_local_heads: Optional[int],
    head_dim: int,
    q_head_norm_weight: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    rope_head_dim: int,
    start_pos: int,
    norm_eps: float,
) -> torch.Tensor:
    """QCAU Query Up, per-head RMSNorm and forward RoPE."""
    if c_q.ndim != 3:
        raise ValueError("c_q must have shape [B,S,D_C]")

    if callable(w_uq):
        q_projected = w_uq(c_q)
    elif w_uq.ndim == 2:
        q_projected = _linear(c_q, w_uq)
    elif w_uq.ndim == 3:
        if w_uq.size(0) == c_q.size(-1):
            q_projected = torch.einsum("bsd,dhc->bshc", c_q, w_uq)
        elif w_uq.size(1) == c_q.size(-1):
            q_projected = torch.einsum("bsd,hdc->bshc", c_q, w_uq)
        else:
            raise ValueError("3D W_UQ must be [D_C,H,Dh] or [H,D_C,Dh]")
    else:
        raise ValueError("W_UQ must be a tensor or callable projector")

    if q_projected.ndim == 3:
        if q_projected.size(-1) % head_dim != 0:
            raise ValueError("W_UQ output is not divisible by head_dim")
        inferred_heads = q_projected.size(-1) // head_dim
        if n_local_heads is not None and inferred_heads != n_local_heads:
            raise ValueError("W_UQ output does not match n_local_heads")
        n_local_heads = inferred_heads
        q = q_projected.unflatten(-1, (n_local_heads, head_dim))
    elif q_projected.ndim == 4:
        if q_projected.size(-1) != head_dim:
            raise ValueError("W_UQ output head dimension does not match head_dim")
        if n_local_heads is not None and q_projected.size(-2) != n_local_heads:
            raise ValueError("W_UQ output does not match n_local_heads")
        q = q_projected
    else:
        raise ValueError("W_UQ must produce [B,S,H*Dh] or [B,S,H,Dh]")

    if q_head_norm_weight is not None:
        expected = (q.size(-2), q.size(-1))
        if tuple(q_head_norm_weight.shape) != expected:
            raise ValueError(f"q_head_norm_weight must have shape {expected}")
    q = _rms_norm(q, q_head_norm_weight, norm_eps)
    return _apply_rotary_emb(q, freqs_cis, rope_head_dim, start_pos)


def _gather_kv_tile(kv: torch.Tensor, safe_idxs: torch.Tensor) -> torch.Tensor:
    """Gather kv[b, safe_idxs[b,s,t]] -> [B,S,T,Dh] without a [B,S,T,Dh] index.

    torch.gather would need the address tensor broadcast over Dh and
    materialised as int64, which is an order of magnitude more memory than the
    gathered rows themselves at S=4096 / Dh=512. Advanced indexing keeps the
    address tensor at [B,S,T].
    """
    batch_index = torch.arange(kv.size(0), device=kv.device).view(-1, 1, 1)
    return kv[batch_index, safe_idxs]


def _sparse_attention_online(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    tile_size: int,
    prob_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> torch.Tensor:
    """QCAU core MQA: tiled QK, online softmax, sink and weighted V."""
    if q.ndim != 4:
        raise ValueError("q must have shape [B,S,H,Dh]")
    if kv.ndim != 3:
        raise ValueError("kv must have shape [B,N,Dh]")
    if topk_idxs.ndim != 3:
        raise ValueError("topk_idxs must have shape [B,S,K]")
    if kv.size(1) == 0:
        raise ValueError("kv must contain at least one latent row")
    if q.size(0) != kv.size(0) or q.size(0) != topk_idxs.size(0):
        raise ValueError("q, kv and topk_idxs batch dimensions must match")
    if q.size(1) != topk_idxs.size(1):
        raise ValueError("q and topk_idxs sequence dimensions must match")
    if q.size(-1) != kv.size(-1):
        raise ValueError("q and kv head dimensions must match")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    bsz, seqlen, n_heads, head_dim = q.shape
    if attn_sink.numel() != n_heads:
        raise ValueError("attn_sink must have one value per local query head")

    q_float = q.float()
    idxs = topk_idxs.to(device=q.device, dtype=torch.long)
    acc_o = torch.zeros(
        bsz, seqlen, n_heads, head_dim,
        device=q.device,
        dtype=torch.float32,
    )
    sum_exp = torch.zeros(
        bsz, seqlen, n_heads,
        device=q.device,
        dtype=torch.float32,
    )
    scores_max = torch.full(
        (bsz, seqlen, n_heads),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )

    for tile_base in range(0, idxs.size(-1), tile_size):
        tile_idxs = idxs[..., tile_base:tile_base + tile_size]
        valid = (tile_idxs >= 0) & (tile_idxs < kv.size(1))
        safe_idxs = tile_idxs.clamp(0, kv.size(1) - 1)

        # Gather in the storage dtype first, then widen: this mirrors
        # kv_shared (BF16) in the kernel and keeps the FP32 copy per tile.
        kv_tile = _gather_kv_tile(kv, safe_idxs)
        kv_tile = torch.where(
            valid.unsqueeze(-1),
            kv_tile,
            torch.zeros_like(kv_tile),
        ).float()

        scores = torch.einsum("bshd,bstd->bsht", q_float, kv_tile)
        scores = scores * softmax_scale
        scores = torch.where(
            valid.unsqueeze(2),
            scores,
            torch.full_like(scores, float("-inf")),
        )

        previous_max = scores_max
        tile_has_valid = valid.any(dim=-1).unsqueeze(-1)
        tile_max = scores.amax(dim=-1)
        next_max = torch.maximum(previous_max, tile_max)
        scores_max = torch.where(tile_has_valid, next_max, previous_max)

        old_scale = torch.where(
            tile_has_valid,
            torch.exp(previous_max - scores_max),
            torch.ones_like(scores_max),
        )
        safe_max = torch.where(
            torch.isfinite(scores_max),
            scores_max,
            torch.zeros_like(scores_max),
        )
        probs = torch.exp(scores - safe_max.unsqueeze(-1))
        probs = torch.where(
            valid.unsqueeze(2),
            probs,
            torch.zeros_like(probs),
        )

        # The kernel reduces the FP32 probabilities into sum_exp and only then
        # casts them to BF16 for the P @ V GEMM, so keep the two apart.
        sum_exp = sum_exp * old_scale + probs.sum(dim=-1)
        probs_pv = probs if prob_dtype is None else probs.to(prob_dtype).float()
        acc_o = (
            acc_o * old_scale.unsqueeze(-1)
            + torch.einsum("bsht,bstd->bshd", probs_pv, kv_tile)
        )

    sink = attn_sink.to(device=q.device, dtype=torch.float32).view(1, 1, n_heads)
    finite_scores = torch.isfinite(scores_max)
    finite_sink = torch.isfinite(sink)
    sink_term = torch.where(
        finite_scores,
        torch.exp(sink - scores_max),
        finite_sink.to(sum_exp.dtype),
    )
    sum_exp = sum_exp + sink_term
    return (
        acc_o
        / sum_exp.clamp_min(torch.finfo(sum_exp.dtype).tiny).unsqueeze(-1)
    ).to(q.dtype)


def QCAU(
    *,
    c_q: torch.Tensor,
    w_uq: WeightOrProjector,
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    attn_sink: torch.Tensor,
    freqs_cis: torch.Tensor,
    q_head_norm_weight: Optional[torch.Tensor] = None,
    n_local_heads: Optional[int] = None,
    head_dim: int = 512,
    rope_head_dim: int = 64,
    start_pos: int = 0,
    norm_eps: float = 1e-6,
    softmax_scale: Optional[float] = None,
    tile_size: int = 64,
    prob_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> torch.Tensor:
    """Run exactly the Query-Core Attention Unit.

    Inputs are produced by the preceding ACU/LIU stages:
      c_q:       [B,S,D_C] latent query from Query Down
      w_uq:      Query Up weight/projector
      kv:        [B,N,Dh] selected compressed KV plus sliding-window KV
      topk_idxs: [B,S,K] addresses into kv; invalid entries are -1
      attn_sink: [H_local] attention-sink scalar for each query head
      freqs_cis: [S, rope_head_dim // 2] table, or a longer table with
                  start_pos selecting the QCAU query positions

    prob_dtype rounds the softmax probabilities before the P @ V product.
    torch.bfloat16 reproduces `acc_s_cast` in kernel.sparse_attn_kernel and is
    what the RTL should be checked against; None keeps the product in FP32 for
    an infinite-precision golden model.

    The output is `o` with shape [B,S,H_local,Dh]. GOPU is downstream.
    """
    if c_q.ndim != 3:
        raise ValueError("c_q must have shape [B,S,D_C]")
    if kv.ndim != 3:
        raise ValueError("kv must have shape [B,N,Dh]")
    if topk_idxs.ndim != 3:
        raise ValueError("topk_idxs must have shape [B,S,K]")
    if c_q.size(0) != kv.size(0) or c_q.size(0) != topk_idxs.size(0):
        raise ValueError("c_q, kv and topk_idxs batch dimensions must match")
    if c_q.size(1) != topk_idxs.size(1):
        raise ValueError("c_q and topk_idxs sequence dimensions must match")

    q = _query_up(
        c_q,
        w_uq,
        n_local_heads=n_local_heads,
        head_dim=head_dim,
        q_head_norm_weight=q_head_norm_weight,
        freqs_cis=freqs_cis,
        rope_head_dim=rope_head_dim,
        start_pos=start_pos,
        norm_eps=norm_eps,
    )
    if q.size(-1) != kv.size(-1):
        raise ValueError("Query Up head dimension must match KV width")
    if attn_sink.numel() != q.size(-2):
        raise ValueError("attn_sink must match the number of local query heads")

    if softmax_scale is None:
        softmax_scale = q.size(-1) ** -0.5

    o = _sparse_attention_online(
        q,
        kv,
        attn_sink,
        topk_idxs,
        softmax_scale,
        tile_size,
        prob_dtype,
    )
    return _apply_rotary_emb(
        o,
        freqs_cis,
        rope_head_dim,
        start_pos,
        inverse=True,
    )
