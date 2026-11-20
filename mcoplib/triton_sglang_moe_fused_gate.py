from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
# Cross-platform compatibility guards (MACA / C600-U vs full sglang CUDA build)
# --------------------------------------------------------------------------- #
try:
    from triton.language.extra import libdevice  # type: ignore

    _HAS_LIBDEVICE = True
except Exception:  # noqa: BLE001
    libdevice = None  # type: ignore
    _HAS_LIBDEVICE = False

try:  # pragma: no cover - present only in the full sglang build
    from sglang.kernels.ops.moe import moe_route_radix  # type: ignore

    _HAS_RADIX = True
except Exception:  # noqa: BLE001
    moe_route_radix = None  # type: ignore
    _HAS_RADIX = False

try:  # pragma: no cover
    from sglang.kernels.jit.utils import (
        get_jit_cuda_arch,
    )
except Exception:  # noqa: BLE001
    
    def is_arch_support_pdl() -> bool:
        """C600-U / MACA fallback: no PDL (programmatic dependent launch)."""
        return False


try:  # pragma: no cover
    from sglang.kernels.kernel_api_logging import debug_kernel_api
except Exception:  # noqa: BLE001

    def debug_kernel_api(fn):  # type: ignore
        return fn


if TYPE_CHECKING:
    from tvm_ffi.module import Module

_SCORING_FUNC_MAP = {
    "sigmoid": 0,
    "sqrtsoftplus": 1,
    "softmax": 2,
}







def _selected_sum_config(rows: int, routed: int):
    """Return (V, W, H) for the source-derived continuous FP32 reducer.

    The reference gather must materialize contiguous FP32 [rows, routed].
    Vendor continuous reduction and split-warp settings must be at defaults.
    K<=4 retains the separately proven small trees in this variant.
    """
    if not 1 <= routed <= 4096:
        raise NotImplementedError(
            "Source-derived selected sum supports routed K in [1, 4096]; "
            "larger K needs a real eager selected-buffer sum fallback"
        )
    if rows < 1 or rows * routed * 4 >= (1 << 31):
        raise NotImplementedError(
            "Source-derived selected sum requires nonempty rows and "
            "a contiguous FP32 selected buffer within signed 32-bit indexing"
        )
    vector = 1
    if routed > 128:
        vector = 4 if routed % 4 == 0 else (2 if routed % 2 == 0 else 1)
    dim0 = routed // vector
    p = min(1 << (dim0.bit_length() - 1), 512)
    q = min(1 << (rows.bit_length() - 1), 512)
    initial_width = min(p, 16 if routed <= 256 else 64)
    height = min(q, 512 // initial_width)
    width = min(p, 512 // height)
    if routed > 4:
        # ReduceConfigMaca::values_per_thread uses scalar routed K, not K/V.
        values_per_thread = (routed + width - 1) // width
        if values_per_thread >= 16 * height or values_per_thread >= 256:
            raise NotImplementedError(
                "Selected sum needs unimplemented block-y/global reduction"
            )
    return vector, width, height


@triton.jit
def _torch_small_selected_sum(values, K_ROUTED: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr):
    # Fold partial rows, then reduce adjacent pairs in Torch's vendor order.
    if K_ROUTED < 2:
        accum = tl.sum(tl.reshape(values, (BLOCK_M, BLOCK_K // 1, 1)), axis=1)
        return tl.reshape(accum, (BLOCK_M, 1))
    elif K_ROUTED < 4:
        accum = tl.sum(tl.reshape(values, (BLOCK_M, BLOCK_K // 2, 2)), axis=1)
        accum = tl.sum(tl.reshape(accum, (BLOCK_M, 1, 2)), axis=2)
        return tl.reshape(accum, (BLOCK_M, 1))
    else:
        accum = tl.sum(tl.reshape(values, (BLOCK_M, BLOCK_K // 4, 4)), axis=1)
        accum = tl.sum(tl.reshape(accum, (BLOCK_M, 2, 2)), axis=2)
        accum = tl.sum(tl.reshape(accum, (BLOCK_M, 1, 2)), axis=2)
        return tl.reshape(accum, (BLOCK_M, 1))


@triton.jit
def _selected_contiguous_chunk(values, BLOCK_M: tl.constexpr,
                               WIDTH: tl.constexpr, CHUNK: tl.constexpr,
                               INDEX: tl.constexpr):
    # Select a compile-time contiguous slice by data movement alone. Avoid
    # floating reductions for extraction and avoid this backend's tl.gather.
    if WIDTH == CHUNK:
        return tl.reshape(values, (BLOCK_M, CHUNK))
    else:
        low, high = tl.split(tl.permute(
            tl.reshape(values, (BLOCK_M, 2, WIDTH // 2)), (0, 2, 1)))
        if INDEX < WIDTH // (2 * CHUNK):
            return _selected_contiguous_chunk(
                low, BLOCK_M, WIDTH // 2, CHUNK, INDEX)
        else:
            return _selected_contiguous_chunk(
                high, BLOCK_M, WIDTH // 2, CHUNK,
                INDEX - WIDTH // (2 * CHUNK))


@triton.jit
def _selected_adjacent_reduce(values, BLOCK_M: tl.constexpr,
                              WIDTH: tl.constexpr):
    if WIDTH == 1:
        return tl.reshape(values, (BLOCK_M, 1))
    else:
        even, odd = tl.split(tl.reshape(values, (BLOCK_M, WIDTH // 2, 2)))
        return _selected_adjacent_reduce(
            libdevice.add_rn(even, odd), BLOCK_M, WIDTH // 2)


@triton.jit
def _selected_upper_reduce(values, BLOCK_M: tl.constexpr,
                           WIDTH: tl.constexpr):
    if WIDTH > 64:
        low, high = tl.split(tl.permute(
            tl.reshape(values, (BLOCK_M, 2, WIDTH // 2)), (0, 2, 1)))
        return _selected_upper_reduce(
            libdevice.add_rn(low, high), BLOCK_M, WIDTH // 2)
    else:
        # Ascending shfl_down 1,2,4,... is an adjacent balanced tree for lane0.
        return _selected_adjacent_reduce(values, BLOCK_M, WIDTH)


@triton.jit
def _torch_selected_sum(values, K_ROUTED: tl.constexpr, BLOCK_K: tl.constexpr,
                        BLOCK_M: tl.constexpr, SUM_VECTOR: tl.constexpr,
                        SUM_WIDTH: tl.constexpr):
    if K_ROUTED <= 4:
        return _torch_small_selected_sum(values, K_ROUTED, BLOCK_K, BLOCK_M)
    else:
        tl.static_assert(K_ROUTED <= 4096)
        tl.static_assert(SUM_VECTOR == 1 or SUM_VECTOR == 2 or SUM_VECTOR == 4)
        tl.static_assert(BLOCK_K >= SUM_WIDTH * SUM_VECTOR)
        lanes = tl.arange(0, SUM_WIDTH)
        accum = tl.full((BLOCK_M, SUM_WIDTH), 0.0, tl.float32)
        # Scalar RN left fold in the same vector/lane order as the vendor
        # continuous reducer. Missing lanes perform no addition at all.
        for i in tl.static_range((K_ROUTED + SUM_WIDTH * SUM_VECTOR - 1) // (SUM_WIDTH * SUM_VECTOR)):
            chunk = _selected_contiguous_chunk(
                values, BLOCK_M, BLOCK_K, SUM_WIDTH * SUM_VECTOR, i)
            active = i * SUM_WIDTH + lanes < K_ROUTED // SUM_VECTOR
            if SUM_VECTOR == 4:
                even, odd = tl.split(tl.reshape(chunk, (BLOCK_M, SUM_WIDTH, 2, 2)))
                v0, v2 = tl.split(even)
                v1, v3 = tl.split(odd)
                next_accum = libdevice.add_rn(accum, v0)
                next_accum = libdevice.add_rn(next_accum, v1)
                next_accum = libdevice.add_rn(next_accum, v2)
                next_accum = libdevice.add_rn(next_accum, v3)
            elif SUM_VECTOR == 2:
                v0, v1 = tl.split(tl.reshape(chunk, (BLOCK_M, SUM_WIDTH, 2)))
                next_accum = libdevice.add_rn(accum, v0)
                next_accum = libdevice.add_rn(next_accum, v1)
            else:
                next_accum = libdevice.add_rn(accum, chunk)
            accum = tl.where(active[None, :], next_accum, accum)
        return _selected_upper_reduce(accum, BLOCK_M, SUM_WIDTH)


@triton.jit
def _softmax_vendor_sum(e, BLOCK_N: tl.constexpr, VECTOR4: tl.constexpr):
    # Installed PersistentSoftmax FP32: optimized 512/1024 use float4 packs;
    # generic widths visit lane + 64*j. Preserve each lane's RN left fold.
    acc = tl.full((1, 64), 0.0, tl.float32)
    if VECTOR4:
        for block in tl.static_range(BLOCK_N // 256):
            chunk = _selected_contiguous_chunk(e, 1, BLOCK_N, 256, block)
            even, odd = tl.split(tl.reshape(chunk, (1, 64, 2, 2)))
            v0, v2 = tl.split(even)
            v1, v3 = tl.split(odd)
            acc = libdevice.add_rn(acc, v0)
            acc = libdevice.add_rn(acc, v1)
            acc = libdevice.add_rn(acc, v2)
            acc = libdevice.add_rn(acc, v3)
    else:
        for block in tl.static_range(BLOCK_N // 64):
            chunk = _selected_contiguous_chunk(e, 1, BLOCK_N, 64, block)
            acc = libdevice.add_rn(acc, chunk)
    return tl.reshape(tl.sum(acc, axis=1), ())


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm}, num_warps=nw, num_stages=ns)
        for bm in (1, 2, 4, 8)
        for nw in (1, 2, 4, 8)
        for ns in (1, 2)
    ],
    key=["N", "K", "SCORING_FUNC", "N_GROUP"],
)

@triton.jit
def _router_triton_kernel(
    scores_ptr,  # [M, N] raw logits, fp32/fp16/bf16 (upcast to fp32 on load)
    bias_ptr,  # [N]    fp32/fp16/bf16 (upcast to fp32 on load)
    bias_alt_ptr,
    input_ids_ptr,
    num_token_non_padded_ptr,
    out_weights_ptr,  # [M, K] fp32
    out_indices_ptr,  # [M, K] int32
    out_packed_ptr,  # [M, K] int32 (HAS_PACKED)
    M,
    routed_scaling_factor,
    shared_inverse_scale,
    moe_softcapping,
    softcap_inverse,
    N: tl.constexpr,
    K: tl.constexpr,  # total topk (includes fused shared experts)
    K_ROUTED: tl.constexpr,  # K - num_fused_shared_experts
    SUM_VECTOR: tl.constexpr,
    SUM_WIDTH: tl.constexpr,
    BLOCK_N: tl.constexpr,  # >= N, power of 2
    BLOCK_K: tl.constexpr,  # >= K, power of 2
    N_GROUP: tl.constexpr,  # expert groups (1 = ungrouped)
    TOPK_GROUP: tl.constexpr,  # groups kept per token (grouped routing)
    EXPERTS_PER_GROUP: tl.constexpr,  # N // N_GROUP
    BLOCK_G: tl.constexpr,  # >= N_GROUP, power of 2
    SCORING_FUNC: tl.constexpr,  # 0 = sigmoid, 1 = sqrtsoftplus, 2 = softmax
    SQRTSOFTPLUS_LOG1P: tl.constexpr,  # sqrtsoftplus via log1p (V4.1 numerics)
    USE_LIBDEVICE: tl.constexpr,  # whether triton.language.extra.libdevice is available
    HAS_SOFTCAP: tl.constexpr,  # tanh softcapping (softmax only)
    RENORMALIZE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,  # apply_routed_scaling_factor_on_output
    HAS_BIAS: tl.constexpr,
    HAS_TOKEN_BIAS: tl.constexpr,
    BIAS_ALT_TOKEN_ID: tl.constexpr,
    HAS_PADDING: tl.constexpr,
    HAS_PACKED: tl.constexpr,
    RENORMALIZE_EPSILON: tl.constexpr,
    USE_PDL: tl.constexpr,
    USE_BITONIC: tl.constexpr,  # parallel bitonic top-k vs serial argmax passes
    LOGN: tl.constexpr,  # log2(BLOCK_N), bitonic stage count
    stride_bias,
    stride_bias_alt,
    stride_input_ids,
    stride_sm,
    stride_sn,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    stride_pm,
    stride_pk,
    BLOCK_M: tl.constexpr,  # rows processed per program (autotuned; ILP over the
    #                         serial top-k reduction chain -> hides its latency)
) -> None:
    # Row-tiled: each program handles BLOCK_M rows; all reductions run along the
    # expert (N) axis. Tiling rows keeps CTAs large enough to stay occupancy-bound
    # rather than launch-bound at small N.
    pid = tl.program_id(0)
    offs_m = pid.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_m = offs_m < M
    mask_n = offs_n < N

    # PDL may start this grid before prior kernel stores are visible. Bias can
    # be produced by a preceding cast or fill kernel, so wait before loading
    # either bias or scores.
    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    # Plain softmax routing has no bias, so keep the zero value in registers
    # rather than materializing and clearing a device tensor per call.
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n.to(tl.int64) * stride_bias, mask=mask_n, other=0.0).to(
            tl.float32
        )
    else:
        bias = tl.zeros([BLOCK_N], dtype=tl.float32)

    if HAS_TOKEN_BIAS:
        bias_alt = tl.load(
            bias_alt_ptr + offs_n.to(tl.int64) * stride_bias_alt, mask=mask_n, other=0.0
        ).to(tl.float32)

    live_m = mask_m
    if HAS_PADDING:
        live_m = live_m & (offs_m < tl.load(num_token_non_padded_ptr))

    row_bias = bias[None, :]
    if HAS_TOKEN_BIAS:
        input_ids = tl.load(
            input_ids_ptr + offs_m * stride_input_ids, mask=live_m, other=0
        )
        row_bias = tl.where(
            (input_ids == BIAS_ALT_TOKEN_ID)[:, None], bias_alt[None, :], row_bias
        )

    row_ptr = scores_ptr + offs_m[:, None] * stride_sm + offs_n[None, :].to(tl.int64) * stride_sn
    mask2d = live_m[:, None] & mask_n[None, :]
    scores = tl.load(row_ptr, mask=mask2d, other=0.0).to(
        tl.float32
    )  # [BLOCK_M, BLOCK_N]

    if SCORING_FUNC == 0:
        # sigmoid(x) = 1 / (1 + exp(-x)); bias is for ranking only, weight is bias-free.
        activated = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-scores))
        biased = activated + row_bias
    elif SCORING_FUNC == 1:
        sp = tl.where(scores > 20.0, scores, libdevice.log1p(libdevice.exp(scores)))
        activated = libdevice.sqrt_rn(sp)
        biased = activated + row_bias
    else:
        # A row-vector division lowers to the same hardware reciprocal multiply
        # as the installed Torch softmax, unlike a broadcast 2-D division.
        tl.static_assert(BLOCK_M == 1)
        logit = tl.reshape(scores, (BLOCK_N,))
        if HAS_SOFTCAP:
            logit = libdevice.mul_rn(moe_softcapping, libdevice.tanh(logit * softcap_inverse))
        biased_row = tl.where(mask_n, logit + tl.reshape(row_bias, (BLOCK_N,)), -float("inf"))
        exp_row = tl.exp(biased_row - tl.max(biased_row, 0))
        if N >= 512 and N <= 1024 and N % 4 == 0:
            row_sum = _softmax_vendor_sum(exp_row, BLOCK_N, True)
        elif BLOCK_N >= 64 and N != 128 and N != 256:
            row_sum = _softmax_vendor_sum(exp_row, BLOCK_N, False)
        else:
            row_sum = tl.sum(exp_row, 0)
        activated = tl.reshape(exp_row / row_sum, (1, BLOCK_N))
        biased = tl.reshape(biased_row, (1, BLOCK_N))

    biased = tl.where(mask_n[None, :], biased, -float("inf"))  # [BLOCK_M, BLOCK_N]

    if SCORING_FUNC == 1 and SQRTSOFTPLUS_LOG1P:
        # Rank NaNs above finite scores, matching torch.topk (DeepSeek-V4.1).
        biased = tl.where(biased == biased, biased, float("inf"))
    else:
        # Map NaN -> a finite floor so NaNs never beat valid experts.
        biased = tl.where(biased == biased, biased, -1e30)

    # Grouped routing (DeepSeek-V3 noaux_tc): per-group score = sum of the top-2
    # biased values; keep TOPK_GROUP groups (lowest group id wins ties); mask the
    # experts of dropped groups to -inf before the top-k below. Weight is still the
    # bias-free `activated`. Constexpr N_GROUP <= 1 skips this entirely (ungrouped).
    if N_GROUP > 1:
        offs_g = tl.arange(0, BLOCK_G)  # [BLOCK_G]
        group_of_n = offs_n // EXPERTS_PER_GROUP  # [BLOCK_N]
        group_score = tl.full([BLOCK_M, BLOCK_G], -float("inf"), dtype=tl.float32)
        for g in tl.static_range(N_GROUP):
            in_g = (group_of_n[None, :] == g) & mask_n[None, :]
            vals = tl.where(in_g, biased, -float("inf"))
            top1 = tl.max(vals, axis=1)[:, None]  # [BLOCK_M, 1]
            first_lane = tl.argmax(vals, axis=1)[:, None]
            # Remove one winner, not every tied maximum: top-2 counts multiplicity.
            vals2 = tl.where(offs_n[None, :] == first_lane, -float("inf"), vals)
            top2 = tl.max(vals2, axis=1)[:, None]  # [BLOCK_M, 1]
            group_score = tl.where(offs_g[None, :] == g, top1 + top2, group_score)

        # Opposite infinities can make top1 + top2 NaN. Apply the same finite
        # floor as expert ranking so group selection always yields a valid ID.
        group_score = tl.where(group_score == group_score, group_score, -1e30)
        gcur = group_score
        groups_left = tl.broadcast_to((offs_g < N_GROUP)[None, :], (BLOCK_M, BLOCK_G))
        keep = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for _i in tl.static_range(TOPK_GROUP):
            gmax = tl.max(gcur, axis=1)[:, None]  # [BLOCK_M, 1]
            glane = tl.where(groups_left & (gcur == gmax), offs_g[None, :], N_GROUP + 1)
            win_g = tl.min(glane, axis=1)[:, None]  # [BLOCK_M, 1] lowest-id on ties
            keep = tl.where(group_of_n[None, :] == win_g, 1.0, keep)
            groups_left = groups_left & (offs_g[None, :] != win_g)
            gcur = tl.where(groups_left, gcur, -float("inf"))
        biased = tl.where(keep > 0.0, biased, -float("inf"))

    offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]
    mask_k_total = offs_k < K
    mask_k_routed = offs_k < K_ROUTED
    selected_vals = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    selected_idx = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.int32)

    remaining = tl.broadcast_to(mask_n[None, :], (BLOCK_M, BLOCK_N))
    if N_GROUP > 1:
        remaining = remaining & (keep > 0.0)
    # Numeric FP32 order followed by inverse expert ID gives stable ties.
    # Canonicalize signed zero because both signs compare numerically equal.
    numeric = tl.where(biased == 0.0, 0.0, biased)
    score_bits = numeric.to(tl.int32, bitcast=True)
    ordered = (score_bits ^ ((score_bits >> 31) | (-2147483648))).to(tl.uint32)
    inverse_id = 4294967295 - offs_n[None, :].to(tl.uint64)
    keys = (ordered.to(tl.uint64) << 32) | inverse_id
    keys = tl.where(remaining, keys, 0)
    for k in tl.range(0, K_ROUTED, loop_unroll_factor=8):
        winner = tl.max(keys, axis=1)
        idx = (4294967295 - (winner & 4294967295)).to(tl.int32)
        idx = idx[:, None]
        slot = offs_k[None, :] == k
        selected_idx = tl.where(slot, idx, selected_idx)
        if SCORING_FUNC == 2:
            bits = tl.where(offs_n[None, :] == idx, activated.to(tl.uint32, bitcast=True), 0)
            value = tl.sum(bits, axis=1).to(tl.float32, bitcast=True)[:, None]
            selected_vals = tl.where(slot, value, selected_vals)
        remaining = remaining & (offs_n[None, :] != idx)
        keys = tl.where(remaining, keys, 0)

    if SCORING_FUNC != 2:
        picked = tl.load(scores_ptr + offs_m[:, None] * stride_sm + selected_idx.to(tl.int64) * stride_sn,
                         mask=live_m[:, None] & mask_k_routed[None, :] & (selected_idx >= 0) & (selected_idx < N), other=0.0).to(tl.float32)
        if SCORING_FUNC == 0:
            selected_vals = libdevice.div_rn(1.0, 1.0 + libdevice.exp(-picked))
        elif SCORING_FUNC == 1:
            sp = tl.where(picked > 20.0, picked, libdevice.log1p(libdevice.exp(picked)))
            selected_vals = libdevice.sqrt_rn(sp)
    selected_vals = tl.where(mask_k_routed[None, :], selected_vals, 0.0)

    routed_sum = _torch_selected_sum(
        selected_vals, K_ROUTED, BLOCK_K, BLOCK_M, SUM_VECTOR, SUM_WIDTH)

    # Fill fused-shared-expert slots: weight = routed_sum / routed_scaling_factor,
    # id = num_experts + (slot - K_ROUTED).
    if K_ROUTED < K:
        is_shared = (offs_k[None, :] >= K_ROUTED) & mask_k_total[None, :]
        shared_weight = routed_sum * shared_inverse_scale  # [BLOCK_M, 1]
        shared_idx = (N + (offs_k - K_ROUTED)).to(tl.int32)[None, :]  # [1, BLOCK_K]
        selected_vals = tl.where(is_shared, shared_weight, selected_vals)
        selected_idx = tl.where(is_shared, shared_idx, selected_idx)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    if RENORMALIZE:
        if RENORMALIZE_EPSILON > 0.0:
            norm = routed_sum + RENORMALIZE_EPSILON
        else:
            norm = tl.where(routed_sum > 0.0, routed_sum, 1.0)  # [BLOCK_M, 1]
        selected_vals = libdevice.div_rn(selected_vals, norm)
    if APPLY_SCALE:
        selected_vals = selected_vals * routed_scaling_factor
    if HAS_PADDING:
        selected_vals = tl.where(live_m[:, None], selected_vals, 0.0)
        selected_idx = tl.where(live_m[:, None], selected_idx, -1)

    out_w_ptr = (
        out_weights_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk
    )
    out_i_ptr = (
        out_indices_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    )
    store_mask = mask_m[:, None] & mask_k_total[None, :]
    tl.store(out_w_ptr, selected_vals, mask=store_mask)
    tl.store(out_i_ptr, selected_idx, mask=store_mask)
    if HAS_PACKED:
        # Must stay bitwise identical to fused_pack_topk.
        w_bits = selected_vals.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
        packed = (selected_idx << 16) | (w_bits & 0xFFFF)
        out_p_ptr = (
            out_packed_ptr + offs_m[:, None] * stride_pm + offs_k[None, :] * stride_pk
        )
        tl.store(out_p_ptr, packed, mask=store_mask)


_DUMMY_I32: Dict[torch.device, torch.Tensor] = {}


def _dummy_i32(device: torch.device) -> torch.Tensor:
    # Placeholder pointer for kernel args whose constexpr flag is off.
    t = _DUMMY_I32.get(device)
    if t is None:
        t = _DUMMY_I32[device] = torch.empty(1, dtype=torch.int32, device=device)
    return t


@debug_kernel_api
def moe_fused_gate(
    scores: torch.Tensor,
    bias: Optional[torch.Tensor],
    topk: int,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
    moe_softcapping: float = 0.0,
    num_expert_group: int = 1,
    topk_group: int = 1,
    *,
    bias_alt: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    bias_alt_token_id: Optional[int] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    renormalize_epsilon: float = 0.0,
    packed_out: Optional[torch.Tensor] = None,
    sqrtsoftplus_log1p: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP32 standalone router with stable expert IDs and optional fused outputs.

    Supports SGLang 0.5.21 call signatures without the native CUDA JIT loader.
    NaN ranking and NaN group sums map to -1e30 (sqrtsoftplus_log1p maps
    expert NaNs to +inf). Exact reduction order targets Torch 2.10/MACA 3.8;
    native SGLang integration and other backend versions require validation.
    """
    if not _HAS_LIBDEVICE:
        raise RuntimeError("The exact MACA router requires triton.language.extra.libdevice")
    scoring_func_int = _SCORING_FUNC_MAP.get(scoring_func.lower())
    assert scoring_func_int is not None, (
        f"Unknown scoring_func '{scoring_func}', must be one of {list(_SCORING_FUNC_MAP.keys())}"
    )
    assert scores.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ), "scores must be float32/float16/bfloat16"
    assert scores.ndim == 2, "scores must be 2D"
    if bias is None:
        assert scoring_func.lower() == "softmax", (
            "bias is required for non-softmax routing"
        )
    else:
        assert bias.dtype in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ), "bias must be float32/float16/bfloat16"
        assert bias.ndim == 1, "bias must be 1D"
        assert scores.size(1) == bias.size(0), (
            "scores and bias must have same num_experts"
        )
    assert scores.is_cuda, "scores must be a CUDA/MACA tensor"
    assert scores.size(1) > 0, "num_experts must be positive"
    assert isinstance(topk, int) and not isinstance(topk, bool), "topk must be an integer"
    assert isinstance(num_fused_shared_experts, int) and num_fused_shared_experts >= 0
    assert topk > num_fused_shared_experts, "topk must be > num_fused_shared_experts"
    assert topk - num_fused_shared_experts <= scores.size(1), "routed topk exceeds num_experts"
    assert isinstance(num_expert_group, int) and num_expert_group >= 1
    assert isinstance(topk_group, int) and 1 <= topk_group <= num_expert_group
    assert scores.size(1) % num_expert_group == 0
    if num_expert_group > 1:
        assert scores.size(1) // num_expert_group >= 2, "groups need at least two experts"
        assert topk - num_fused_shared_experts <= topk_group * (scores.size(1) // num_expert_group)
    if bias is not None:
        assert bias.device == scores.device, "bias must be on the scores device"
    assert math.isfinite(moe_softcapping) and moe_softcapping >= 0
    assert math.isfinite(renormalize_epsilon) and renormalize_epsilon >= 0
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0
    assert math.isfinite(routed_scaling_factor)
    if num_fused_shared_experts:
        assert routed_scaling_factor > 0, "shared expert normalization needs positive scale"
    if bias_alt is not None:
        assert input_ids is not None, "bias_alt requires input_ids"
        assert bias_alt.device == scores.device and bias_alt.ndim == 1
        assert bias_alt.dtype in (torch.float32, torch.float16, torch.bfloat16)
    if input_ids is not None:
        assert input_ids.device == scores.device and input_ids.dtype in (torch.int32, torch.int64)
        assert isinstance(bias_alt_token_id, int)
    if num_token_non_padded is not None:
        assert num_token_non_padded.device == scores.device
        assert num_token_non_padded.numel() == 1 and num_token_non_padded.dtype in (torch.int32, torch.int64)
    if input_ids is not None:
        assert bias_alt is not None and bias_alt_token_id is not None
        assert bias is not None and bias_alt.shape == bias.shape
        assert input_ids.shape == (scores.size(0),)
    if packed_out is not None:
        assert packed_out.device == scores.device, "packed_out must be on the scores device"
        assert packed_out.dtype == torch.int32, "packed_out must be int32"
        assert scores.size(1) + num_fused_shared_experts <= 65536, "packed expert IDs must fit in 16 bits"
        assert packed_out.shape == (scores.size(0), topk), (
            "packed_out must be [M, topk]"
        )
        # A contiguous destination excludes overlapping writes and broadcast views.
        assert packed_out.is_contiguous(), "packed_out must be contiguous"
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    # K3 radix-select fast path (guarded for standalone / MACA builds)
    if (
        _HAS_RADIX
        and scoring_func.lower() == "sigmoid"
        and num_fused_shared_experts == 0
        and num_expert_group <= 1
        and moe_softcapping == 0.0
        and input_ids is None
        and num_token_non_padded is None
        and renormalize_epsilon == 0.0
        and packed_out is None
        and bias is not None
        and bias.stride(0) == 1
    ):
        radix_args = (
            scores,
            bias,
            topk,
            renormalize,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
        )
        # Keep disabled on C600-U/MACA if native radix kernel is not desired:
        if False and moe_route_radix.covered(scores, bias, topk):
            return moe_route_radix.route_radix(*radix_args, sorted=False)

    # The scoring contract is FP32. Separating conversion avoids excessive
    # private-memory allocation in the MACA fused BF16 softmax lowering.
    if scoring_func_int == 2 and scores.dtype != torch.float32:
        scores = scores.float()

    M, N = scores.shape
    K = topk
    K_routed = topk - num_fused_shared_experts
    if num_expert_group > 1:
        assert N % num_expert_group == 0, "num_experts must be divisible by group count"
        assert 1 <= topk_group <= num_expert_group, "invalid topk_group"
    experts_per_group = N // num_expert_group
    BLOCK_G = triton.next_power_of_2(num_expert_group)

    weights = torch.empty((M, K), dtype=torch.float32, device=scores.device)
    indices = torch.empty((M, K), dtype=torch.int32, device=scores.device)
    if M == 0:
        return weights, indices

    sum_vector, sum_width, _sum_height = _selected_sum_config(M, K_routed)

    BLOCK_N = triton.next_power_of_2(N)  # 256 -> 256, 384 -> 512
    BLOCK_K = triton.next_power_of_2(K)  # 6 -> 8, 8 -> 8
    use_bitonic = False
    log_n = BLOCK_N.bit_length() - 1  # exact: BLOCK_N is a power of two

    # Autotuned grid over BLOCK_M
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    use_pdl = False
    if use_pdl and scoring_func_int == 1 and N == 384 and K == 6 and M <= 8:
        arch = get_jit_cuda_arch()
        if arch is not None:
            use_pdl = (arch.major, arch.minor) != (10, 3)
    extra = {"launch_pdl": True} if use_pdl else {}

    _unused_i32 = _dummy_i32(scores.device)
    _router_triton_kernel[grid](
        scores,
        bias if bias is not None else scores,
        bias_alt if bias_alt is not None else _unused_i32,
        input_ids if input_ids is not None else _unused_i32,
        num_token_non_padded if num_token_non_padded is not None else _unused_i32,
        weights,
        indices,
        packed_out if packed_out is not None else _unused_i32,
        M,
        float(routed_scaling_factor),
        1.0 / float(routed_scaling_factor) if num_fused_shared_experts else 1.0,
        float(moe_softcapping),
        1.0 / float(moe_softcapping) if moe_softcapping else 1.0,
        N=N,
        K=K,
        K_ROUTED=K_routed,
        SUM_VECTOR=sum_vector,
        SUM_WIDTH=sum_width,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        N_GROUP=num_expert_group,
        TOPK_GROUP=topk_group,
        EXPERTS_PER_GROUP=experts_per_group,
        BLOCK_G=BLOCK_G,
        SCORING_FUNC=scoring_func_int,
        SQRTSOFTPLUS_LOG1P=bool(sqrtsoftplus_log1p),
        USE_LIBDEVICE=_HAS_LIBDEVICE,
        HAS_SOFTCAP=bool(moe_softcapping != 0.0),
        RENORMALIZE=bool(renormalize),
        APPLY_SCALE=bool(apply_routed_scaling_factor_on_output),
        HAS_BIAS=bias is not None,
        HAS_TOKEN_BIAS=input_ids is not None,
        BIAS_ALT_TOKEN_ID=bias_alt_token_id if bias_alt_token_id is not None else 0,
        HAS_PADDING=num_token_non_padded is not None,
        HAS_PACKED=packed_out is not None,
        RENORMALIZE_EPSILON=float(renormalize_epsilon),
        USE_PDL=use_pdl,
        USE_BITONIC=use_bitonic,
        LOGN=log_n,
        stride_bias=bias.stride(0) if bias is not None else 0,
        stride_bias_alt=bias_alt.stride(0) if bias_alt is not None else 0,
        stride_input_ids=input_ids.stride(0) if input_ids is not None else 0,
        stride_sm=scores.stride(0),
        stride_sn=scores.stride(1),
        stride_wm=weights.stride(0),
        stride_wk=weights.stride(1),
        stride_im=indices.stride(0),
        stride_ik=indices.stride(1),
        stride_pm=packed_out.stride(0) if packed_out is not None else 0,
        stride_pk=packed_out.stride(1) if packed_out is not None else 0,
        enable_fp_fusion=False,
        **extra,
    )
    return weights, indices