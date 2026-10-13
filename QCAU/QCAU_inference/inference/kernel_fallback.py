"""PyTorch fallback for inference kernels.

This module keeps the inference graph usable when TileLang cannot be imported,
for example because the installed TVM-FFI wrapper is incompatible with the
bundled TVM runtime. It is a correctness backend, not a performance backend.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def _block_scales(
    x: torch.Tensor,
    block_size: int,
    max_value: float,
    scale_fmt: Optional[str] = None,
) -> torch.Tensor:
    if x.size(-1) % block_size != 0:
        raise ValueError(
            f"last dimension {x.size(-1)} must be divisible by block_size={block_size}"
        )
    blocks = x.float().reshape(*x.shape[:-1], -1, block_size)
    amax = blocks.abs().amax(dim=-1).clamp_min(1e-4)
    scales = amax / max_value
    if scale_fmt is not None:
        scales = torch.exp2(torch.ceil(torch.log2(scales)))
    return scales


def _quantize_dequantize(
    x: torch.Tensor,
    block_size: int,
    max_value: float,
    scale_fmt: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    scales = _block_scales(x, block_size, max_value, scale_fmt)
    expanded = scales.unsqueeze(-1).expand(
        *scales.shape, block_size
    ).reshape_as(x)
    q = torch.clamp(x.float() / expanded, -max_value, max_value)
    return q, scales


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    """Reference block-wise FP8 quantization/dequantization."""
    q, scales = _quantize_dequantize(x, block_size, 448.0, scale_fmt)
    if inplace:
        expanded = scales.unsqueeze(-1).expand(
            *scales.shape, block_size
        ).reshape_as(x)
        dequantized = q.to(torch.float8_e4m3fn).float() * expanded
        x.copy_(dequantized.to(x.dtype))
        return x
    return q.to(torch.float8_e4m3fn), scales.to(scale_dtype)


def _decode_fp4_packed(weight: torch.Tensor) -> torch.Tensor:
    """Decode PyTorch's two-values-per-byte E2M1 FP4 storage."""
    packed = weight.view(torch.uint8)
    low = packed & 0x0F
    high = packed >> 4
    codes = torch.cat((low, high), dim=-1).long()
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=weight.device,
        dtype=torch.float32,
    )
    return values[codes]


def _expand_activation_scales(
    scales: torch.Tensor, width: int, block_size: int
) -> torch.Tensor:
    return scales.float().repeat_interleave(block_size, dim=-1)[..., :width]


def _expand_weight_scales(
    scales: torch.Tensor, rows: int, width: int, row_block: int, col_block: int
) -> torch.Tensor:
    return scales.float().repeat_interleave(row_block, dim=0).repeat_interleave(
        col_block, dim=1
    )[:rows, :width]


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute A @ B.T after applying block scales in FP32."""
    del scale_dtype
    k = a.size(-1)
    a2 = a.float().reshape(-1, k)
    a_scale = _expand_activation_scales(a_s, k, 128).reshape(-1, k)
    b_float = b.float()
    b_scale = _expand_weight_scales(b_s, b.size(0), k, 128, 128)
    b_float = b_float * b_scale
    out = a2.mul(a_scale).matmul(b_float.transpose(0, 1))
    out = out.reshape(*a.shape[:-1], b.size(0))
    return out.to(torch.get_default_dtype())


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
):
    """Reference block-wise FP4 quantization/dequantization.

    The fallback uses FP32 values internally because CPU PyTorch does not
    provide arithmetic kernels for the packed float4 storage type.
    """
    q, scales = _quantize_dequantize(x, block_size, 6.0, "ue8m0")
    if inplace:
        x.copy_((q * scales.unsqueeze(-1).expand(
            *scales.shape, block_size
        ).reshape_as(x)).to(x.dtype))
        return x
    raise RuntimeError("non-inplace fp4_act_quant requires the TileLang backend")


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute FP8-activation x packed-FP4-weight GEMM in FP32."""
    del scale_dtype
    k = a.size(-1)
    a2 = a.float().reshape(-1, k)
    a_scale = _expand_activation_scales(a_s, k, 128).reshape(-1, k)
    b_float = _decode_fp4_packed(b)
    b_scale = _expand_weight_scales(b_s, b.size(0), k, 1, 32)
    out = a2.mul(a_scale).matmul((b_float * b_scale).transpose(0, 1))
    out = out.reshape(*a.shape[:-1], b.size(0))
    return out.to(torch.get_default_dtype())


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """PyTorch equivalent of hc_split_sinkhorn_kernel."""
    mix_hc = (2 + hc_mult) * hc_mult
    if mixes.size(-1) != mix_hc:
        raise ValueError(f"expected mixes[..., {mix_hc}], got {mixes.shape}")
    scale = hc_scale.float().reshape(-1)
    base = hc_base.float().reshape(-1)
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc_mult] * scale[0] + base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        m[..., hc_mult:2 * hc_mult] * scale[1]
        + base[hc_mult:2 * hc_mult]
    )
    comb = m[..., 2 * hc_mult:].reshape(*m.shape[:-1], hc_mult, hc_mult)
    comb_base = base[2 * hc_mult:].reshape(hc_mult, hc_mult)
    comb = comb * scale[2] + comb_base
    # Match hc_split_sinkhorn_kernel: one row-softmax followed by column
    # normalization, then alternating row/column normalization.
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Compatibility wrapper for callers that still use sparse_attn directly."""
    try:
        from .QCAU import _sparse_attention_online
    except ImportError:
        from dinhanh.QCAU.inference.QCAU import _sparse_attention_online

    return _sparse_attention_online(
        q,
        kv,
        attn_sink,
        topk_idxs,
        softmax_scale,
        tile_size=64,
    )
