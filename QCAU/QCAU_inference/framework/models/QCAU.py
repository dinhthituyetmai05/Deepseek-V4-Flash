from typing import Callable, Optional, Union

import torch
import tilelang
import tilelang.language as T


tilelang.set_log_level("WARNING")

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


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
    """RMSNorm used after Query Up, with an optional per-head gamma."""
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


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _qcau_sparse_attn_kernel(h: int, d: int, scale=None):
    b = T.symbolic("b")
    m = T.symbolic("m")
    n = T.symbolic("n")
    topk = T.symbolic("topk")
    if scale is None:
        scale = (1.0 / d) ** 0.5

    num_stages = 2
    threads = 256
    block = 64
    num_blocks = tilelang.cdiv(topk, block)

    @T.prim_func
    def _qcau_sparse_attn_kernel_(
        q: T.Tensor[(b, m, h, d), BF16],
        kv: T.Tensor[(b, n, d), BF16],
        o: T.Tensor[(b, m, h, d), BF16],
        attn_sink: T.Tensor[(h,), FP32],
        topk_idxs: T.Tensor[(b, m, topk), INT32],
    ):
        with T.Kernel(m, b, threads=threads) as (bx, by):
            q_shared = T.alloc_shared((h, d), BF16)
            kv_shared = T.alloc_shared((block, d), BF16)
            o_shared = T.alloc_shared((h, d), BF16)
            acc_s_cast = T.alloc_shared((h, block), BF16)

            idxs = T.alloc_fragment(block, INT32)
            acc_s = T.alloc_fragment((h, block), FP32)
            acc_o = T.alloc_fragment((h, d), FP32)
            scores_max = T.alloc_fragment(h, FP32)
            scores_max_prev = T.alloc_fragment(h, FP32)
            scores_scale = T.alloc_fragment(h, FP32)
            scores_sum = T.alloc_fragment(h, FP32)
            sum_exp = T.alloc_fragment(h, FP32)

            T.clear(acc_o)
            T.clear(sum_exp)
            T.fill(scores_max, -T.infinity(FP32))
            T.copy(q[by, bx, :, :], q_shared)

            for t in T.Pipelined(num_blocks, num_stages=num_stages):
                for i in T.Parallel(block):
                    idxs[i] = T.if_then_else(
                        t * block + i < topk,
                        topk_idxs[by, bx, t * block + i],
                        -1,
                    )
                for i, j in T.Parallel(block, d):
                    kv_shared[i, j] = T.if_then_else(
                        idxs[i] != -1,
                        kv[by, idxs[i], j],
                        0,
                    )
                for i, j in T.Parallel(h, block):
                    acc_s[i, j] = T.if_then_else(
                        idxs[j] != -1,
                        0,
                        -T.infinity(FP32),
                    )
                T.gemm(
                    q_shared,
                    kv_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, j in T.Parallel(h, block):
                    acc_s[i, j] *= scale
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(h):
                    scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                for i, j in T.Parallel(h, block):
                    acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(h):
                    sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)
                for i, j in T.Parallel(h, d):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(
                    acc_s_cast,
                    kv_shared,
                    acc_o,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            for i in T.Parallel(h):
                sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
            for i, j in T.Parallel(h, d):
                acc_o[i, j] /= sum_exp[i]
            T.copy(acc_o, o_shared)
            T.copy(o_shared, o[by, bx, :, :])

    return _qcau_sparse_attn_kernel_


# def _sparse_attention_online_torch_reference(
#     q: torch.Tensor,
#     kv: torch.Tensor,
#     attn_sink: torch.Tensor,
#     topk_idxs: torch.Tensor,
#     softmax_scale: float,
#     tile_size: int,
# ) -> torch.Tensor:
#     """QCAU core MQA: tiled QK, online softmax, sink and weighted V."""
#     if q.ndim != 4:
#         raise ValueError("q must have shape [B,S,H,Dh]")
#     if kv.ndim != 3:
#         raise ValueError("kv must have shape [B,N,Dh]")
#     if topk_idxs.ndim != 3:
#         raise ValueError("topk_idxs must have shape [B,S,K]")
#     if kv.size(1) == 0:
#         raise ValueError("kv must contain at least one latent row")
#     if q.size(0) != kv.size(0) or q.size(0) != topk_idxs.size(0):
#         raise ValueError("q, kv and topk_idxs batch dimensions must match")
#     if q.size(1) != topk_idxs.size(1):
#         raise ValueError("q and topk_idxs sequence dimensions must match")
#     if q.size(-1) != kv.size(-1):
#         raise ValueError("q and kv head dimensions must match")
#     if tile_size <= 0:
#         raise ValueError("tile_size must be positive")

#     bsz, seqlen, n_heads, head_dim = q.shape
#     if attn_sink.numel() != n_heads:
#         raise ValueError("attn_sink must have one value per local query head")

#     q_float = q.float()
#     kv_float = kv.float()
#     idxs = topk_idxs.to(device=q.device, dtype=torch.long)
#     acc_o = torch.zeros(
#         bsz, seqlen, n_heads, head_dim,
#         device=q.device,
#         dtype=torch.float32,
#     )
#     sum_exp = torch.zeros(
#         bsz, seqlen, n_heads,
#         device=q.device,
#         dtype=torch.float32,
#     )
#     scores_max = torch.full(
#         (bsz, seqlen, n_heads),
#         float("-inf"),
#         device=q.device,
#         dtype=torch.float32,
#     )

#     for tile_base in range(0, idxs.size(-1), tile_size):
#         tile_idxs = idxs[..., tile_base:tile_base + tile_size]
#         valid = (tile_idxs >= 0) & (tile_idxs < kv.size(1))
#         safe_idxs = tile_idxs.clamp(0, kv.size(1) - 1)

#         gather_index = safe_idxs.unsqueeze(-1).expand(*safe_idxs.shape, head_dim)
#         kv_tile = torch.gather(
#             kv_float,
#             1,
#             gather_index.reshape(bsz, -1, head_dim),
#         ).view(bsz, seqlen, -1, head_dim)
#         kv_tile = torch.where(
#             valid.unsqueeze(-1),
#             kv_tile,
#             torch.zeros_like(kv_tile),
#         )

#         scores = torch.einsum("bshd,bstd->bsht", q_float, kv_tile)
#         scores = scores * softmax_scale
#         scores = torch.where(
#             valid.unsqueeze(2),
#             scores,
#             torch.full_like(scores, float("-inf")),
#         )

#         previous_max = scores_max
#         tile_has_valid = valid.any(dim=-1).unsqueeze(-1)
#         tile_max = scores.amax(dim=-1)
#         next_max = torch.maximum(previous_max, tile_max)
#         scores_max = torch.where(tile_has_valid, next_max, previous_max)

#         old_scale = torch.where(
#             tile_has_valid,
#             torch.exp(previous_max - scores_max),
#             torch.ones_like(scores_max),
#         )
#         safe_max = torch.where(
#             torch.isfinite(scores_max),
#             scores_max,
#             torch.zeros_like(scores_max),
#         )
#         probs = torch.exp(scores - safe_max.unsqueeze(-1))
#         probs = torch.where(
#             valid.unsqueeze(2),
#             probs,
#             torch.zeros_like(probs),
#         )

#         sum_exp = sum_exp * old_scale + probs.sum(dim=-1)
#         # TileLang sparse_attn copies the unnormalized exp scores through a
#         # BF16 shared buffer before the value GEMM (acc_s -> acc_s_cast).
#         # Keep the reference path numerically aligned with that kernel.
#         probs_cast = probs.to(q.dtype).float()
#         acc_o = (
#             acc_o * old_scale.unsqueeze(-1)
#             + torch.einsum("bsht,bstd->bshd", probs_cast, kv_tile)
#         )

#     sink = attn_sink.to(device=q.device, dtype=torch.float32).view(1, 1, n_heads)
#     finite_scores = torch.isfinite(scores_max)
#     finite_sink = torch.isfinite(sink)
#     sink_term = torch.where(
#         finite_scores,
#         torch.exp(sink - scores_max),
#         finite_sink.to(sum_exp.dtype),
#     )
#     sum_exp = sum_exp + sink_term
#     return (
#         acc_o
#         / sum_exp.clamp_min(torch.finfo(sum_exp.dtype).tiny).unsqueeze(-1)
#     ).to(q.dtype)


def _sparse_attention_online(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    head_chunk: Optional[int],
) -> torch.Tensor:
    """QCAU sparse attention implemented locally with the model.py kernel math.

    This is intentionally TileLang, not a PyTorch einsum reference, because the
    original model's BF16 GEMM accumulation order is part of the golden output.
    Head splitting is exact because sparse attention has no cross-head dependency.
    """
    if q.ndim != 4:
        raise ValueError("q must have shape [B,S,H,Dh]")
    if kv.ndim != 3:
        raise ValueError("kv must have shape [B,N,Dh]")
    if topk_idxs.ndim != 3:
        raise ValueError("topk_idxs must have shape [B,S,K]")
    if q.size(0) != kv.size(0) or q.size(0) != topk_idxs.size(0):
        raise ValueError("q, kv and topk_idxs batch dimensions must match")
    if q.size(1) != topk_idxs.size(1):
        raise ValueError("q and topk_idxs sequence dimensions must match")
    if q.size(-1) != kv.size(-1):
        raise ValueError("q and kv head dimensions must match")
    if attn_sink.numel() != q.size(2):
        raise ValueError("attn_sink must have one value per local query head")

    def run_kernel(q_part: torch.Tensor, sink_part: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, n_heads, head_dim = q_part.size()
        q_kernel = q_part
        sink_kernel = sink_part
        padded_heads = n_heads
        if n_heads < 16:
            padded_heads = 16
            q_kernel = torch.cat(
                [q_part, q_part.new_zeros(bsz, seqlen, padded_heads - n_heads, head_dim)],
                dim=2,
            )
            sink_kernel = torch.cat(
                [sink_part, sink_part.new_zeros(padded_heads - n_heads)],
                dim=0,
            )
        out = torch.empty_like(q_kernel)
        kernel = _qcau_sparse_attn_kernel(padded_heads, head_dim, softmax_scale)
        kernel(q_kernel, kv, out, sink_kernel, topk_idxs.int())
        if n_heads < 16:
            out = out.narrow(2, 0, n_heads).contiguous()
        return out

    if head_chunk is None or head_chunk <= 0 or head_chunk >= q.size(2):
        return run_kernel(q, attn_sink)

    chunks = []
    for start in range(0, q.size(2), head_chunk):
        end = min(start + head_chunk, q.size(2))
        chunks.append(
            run_kernel(
                q[:, :, start:end].contiguous(),
                attn_sink[start:end].contiguous(),
            )
        )
    return torch.cat(chunks, dim=2)


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
    use_tilelang_sparse: bool = True,
    head_chunk: Optional[int] = 16,
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

    if use_tilelang_sparse:
        o = _sparse_attention_online(
            q,
            kv,
            attn_sink,
            topk_idxs,
            softmax_scale,
            head_chunk,
        )
    else:
        o = _sparse_attention_online_torch_reference(
            q,
            kv,
            attn_sink,
            topk_idxs,
            softmax_scale,
            tile_size,
        )
    return _apply_rotary_emb(
        o,
        freqs_cis,
        rope_head_dim,
        start_pos,
        inverse=True,
    )
