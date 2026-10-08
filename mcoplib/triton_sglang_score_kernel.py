# SPDX-License-Identifier: Apache-2.0
# Kimi-K3 Attention Residual: snapshot bank + aggregation.
#
# The public API is the AttnResidual class (constructed once per forward pass).
# It owns the frozen snapshot bank [T, NB, H] and the valid-row counter, and
# dispatches each aggregation point (score rows → softmax → weighted sum →
# RMSNorm) by hardware capability:
#   fast  — warp-specialized TMA kernel: cp.async.bulk producer +
#           online-softmax consumers over a double-buffered chunk ring, out
#           norm fused, per-nvb tuned launch config, one persistent CTA per
#           SM. Taken on SM100+ with H=7168.
#   hip   — single Triton kernel, everything in one launch; taken on ROCm
#           within its register budget.
#   fused — Triton 2-kernel pipeline with full H-parallelism; the fallback
#           everywhere the fast kernel does not apply.
# aggregate_stream_torch is the eager reference (tests and the
# H % _BLOCK_H != 0 shape fallback of aggregate_stream).

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.utils import is_hip, is_npu

_BLOCK_H: int = 1024  # H = 7168 = 7 x 1024
_SCORE_BLOCK_H: int = 256  # _score_kernel H-chunk (C600-U tuned: NW=2, BH=256)
_MAX_ROWS: int = 16  # next_pow2(8 + 1), K3 has <= 8 snapshots

_FAST_SUPPORTED = None
_HIP_SHAPE_GATE = None


def _use_fast(hidden_size: int) -> bool:
    """The TMA kernel needs SM100+ (tcgen05, cp.async.bulk) and its H=7168
    template instantiation; everything else takes the triton pipeline."""
    global _FAST_SUPPORTED
    if is_npu():
        return False
    if _FAST_SUPPORTED is None:
        major, _ = torch.cuda.get_device_capability()
        _FAST_SUPPORTED = major >= 10
    return _FAST_SUPPORTED and hidden_size == 7168

def get_cw(
    proj: ReplicatedLinear,
    norm: RMSNorm,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Cached product norm_weight ⊙ proj_weight (both [H]) in `dtype`.

    Cached per dtype: the fast kernel consumes bf16 while the triton path
    consumes fp32, and a shared slot would hand one path the other's dtype."""
    cache = getattr(proj, "_attn_res_cw_cache", None)
    if cache is None:
        cache = {}
        proj._attn_res_cw_cache = cache
    cw = cache.get(dtype)
    if cw is None:
        cw = (norm.weight.float() * proj.weight.squeeze().float()).contiguous()
        cw = cache[dtype] = cw.to(dtype)
    return cw
    
    
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 128}, num_warps=1),
        triton.Config({'BLOCK_H': 128}, num_warps=2),
        triton.Config({'BLOCK_H': 128}, num_warps=4),
        triton.Config({'BLOCK_H': 256}, num_warps=1),
        triton.Config({'BLOCK_H': 256}, num_warps=2),
        triton.Config({'BLOCK_H': 256}, num_warps=4),
        triton.Config({'BLOCK_H': 256}, num_warps=8),
        triton.Config({'BLOCK_H': 512}, num_warps=1),
        triton.Config({'BLOCK_H': 512}, num_warps=2),
        triton.Config({'BLOCK_H': 512}, num_warps=4),
        triton.Config({'BLOCK_H': 512}, num_warps=8),
        triton.Config({'BLOCK_H': 1024}, num_warps=1),
        triton.Config({'BLOCK_H': 1024}, num_warps=2),
        triton.Config({'BLOCK_H': 1024}, num_warps=4),
        triton.Config({'BLOCK_H': 1024}, num_warps=8),
    ],
    key=['NVB'], 
)

@triton.jit
def _score_kernel(
    prefix_ptr,  # [T, H]
    bank_ptr,  # [T, NB_total, H]
    cw_ptr,  # [H] fp32
    scores_ptr,  # [T, MAX_ROWS] fp32
    NVB,
    eps,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MAXR: tl.constexpr,
):
    """One CTA per token: score all (nvb+1) rows in a single pass.

    The (token, row) grid of the original re-read the shared cw[H] weight once
    per row (T*(nvb+1) times), and cw — identical for every row — became the L2
    bottleneck (probe: bf16 rows alone reach ~1.2 TB/s, adding the fp32 cw
    halves it). Here one program owns a token and loops H in BLOCK_H chunks,
    loading each cw chunk ONCE and reusing it across all rows held in a
    [MAXR, BLOCK_H] register tile. Rows past nvb are masked off. The two
    products accumulate elementwise and reduce once along H at the end (a single
    barrier reduction per token, not 2*(H/BLOCK_H) per row).
    """
    pid_t = tl.program_id(0)
    rj = tl.arange(0, MAXR)  # [MAXR] logical row ids
    active = rj <= NVB
    # Per-row base pointer: bank row j for j < NVB, else the prefix row.
    row_base = tl.where(
        rj < NVB,
        bank_ptr + pid_t * stride_bm + rj * stride_bb,
        prefix_ptr + pid_t * stride_pm,
    )  # [MAXR]
    offs0 = tl.arange(0, BLOCK_H)
    sumsq = tl.zeros([MAXR, BLOCK_H], tl.float32)
    dotv = tl.zeros([MAXR, BLOCK_H], tl.float32)
    for h0 in tl.static_range(0, H, BLOCK_H):
        offs_h = h0 + offs0
        cw = tl.load(cw_ptr + offs_h)  # [BLOCK_H], loaded once, reused
        ptrs = row_base[:, None] + offs_h[None, :]  # [MAXR, BLOCK_H]
        v = tl.load(ptrs, mask=active[:, None], other=0.0).to(tl.float32)
        sumsq += v * v
        dotv += v * cw[None, :]
    ssum = tl.sum(sumsq, axis=1)  # [MAXR]
    dsum = tl.sum(dotv, axis=1)  # [MAXR]
    rrms = 1.0 / tl.sqrt(ssum / H + eps)
    tl.store(scores_ptr + pid_t * stride_sm + rj, dsum * rrms, mask=active)

@triton.jit
def _combine_kernel(
    prefix_ptr,
    bank_ptr,
    scores_ptr,  # [T, MAX_ROWS] fp32
    out_ptr,  # [T, H]
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    stride_om,
    NVB: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    """One CTA per (token, H-chunk): softmax(scores) → weighted sum → write chunk.

    Softmax is redundantly computed by each H-chunk CTA (≤16 elements, trivial).
    This gives full H-parallelism: 7 CTAs for H=7168/1024.

    NVB is a compile-time constant so the (NVB+1)-row loop fully unrolls: the
    independent bf16 row loads are then issued back-to-back and the memory
    system pipelines them, instead of the baseline's runtime `range(NVB+1)`
    which serialised each load behind its own accumulate (nvb=2 was *slower per
    byte* than nvb=1 there). The softmax probabilities are hoisted into scalars
    once, so the per-row `tl.sum(where(...))` reduction is gone from the loop.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    h0 = pid_h * BLOCK_H

    # Softmax over the NVB+1 valid scores (compile-time count).
    offs_b = tl.arange(0, MAX_ROWS)
    mask_b = offs_b < (NVB + 1)
    raw = tl.load(
        scores_ptr + pid_t * stride_sm + offs_b, mask=mask_b, other=float("-inf")
    )
    m = tl.max(raw, axis=0)
    e = tl.where(mask_b, tl.exp(raw - m), 0.0)
    p = e / tl.sum(e, axis=0)

    offs_h = h0 + tl.arange(0, BLOCK_H)
    base_b = bank_ptr + pid_t * stride_bm + offs_h
    # Prefix row prob (row index == NVB) times the prefix row.
    acc = tl.load(prefix_ptr + pid_t * stride_pm + offs_h).to(tl.float32) * tl.sum(
        tl.where(offs_b == NVB, p, 0.0), axis=0
    )
    # NVB is constexpr, so this loop unrolls: the independent bf16 row loads are
    # issued back-to-back and the memory system pipelines them.
    for j in range(0, NVB):
        p_j = tl.sum(tl.where(offs_b == j, p, 0.0), axis=0)
        acc += p_j * tl.load(base_b + j * stride_bb).to(tl.float32)
    tl.store(
        out_ptr + pid_t * stride_om + offs_h,
        acc.to(out_ptr.dtype.element_ty),
    )

def _mix_fused(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
) -> torch.Tensor:
    """Triton score + combine pair: returns the pre-norm mixture."""
    T, H = prefix_sum.shape
    if T == 0:
        return prefix_sum
    cw = get_cw(score_proj, score_norm)
    if is_npu():
        from sgl_kernel_npu.kimi_k3.attn_residual import mix_fused

        return mix_fused(
            prefix_sum,
            bank,
            nvb,
            cw,
            score_norm.variance_epsilon,
        )
    n_h_blocks = H // _BLOCK_H

    # Step 1: score every row of each token in one program (cw reused across
    # rows). Grid is (T,); the [MAXR, BLOCK_H] register tile holds all rows, so
    # MAXR = next_pow2(nvb+1) and a small BLOCK_H keeps the tile in registers.
    scores = torch.empty((T, _MAX_ROWS), dtype=torch.float32, device=prefix_sum.device)
    maxr = 1 << (nvb + 1 - 1).bit_length()  # next_pow2(nvb+1)
    _score_kernel[(T,)](
        prefix_sum,
        bank,
        cw,
        scores,
        nvb,
        score_norm.variance_epsilon,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        H=H,
        MAXR=maxr,
    )

    # Step 2: softmax + weighted sum (2D grid, full H-parallelism)
    out = torch.empty_like(prefix_sum)
    _combine_kernel[(T, n_h_blocks)](
        prefix_sum,
        bank,
        scores,
        out,
        nvb,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        out.stride(0),
        BLOCK_H=_BLOCK_H,
        MAX_ROWS=_MAX_ROWS,
        num_warps=4,
    )
    return out

