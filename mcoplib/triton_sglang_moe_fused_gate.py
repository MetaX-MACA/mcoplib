from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import torch
import triton
import triton.language as tl


if TYPE_CHECKING:
    from tvm_ffi.module import Module

# --------------------------------------------------------------------------- #
# Standalone fallbacks (MACA / C600-U build has no sglang): the native-CUDA
# radix-select fast path and PDL helper live in sglang and are unavailable here,
# so routing always lands on the pure-Triton kernel below. These guards keep the
# public wrapper runnable without sglang while preserving identical semantics.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - present only in the full sglang build
    from sglang.kernels.ops.moe import moe_route_radix  # type: ignore

    _HAS_RADIX = True
except Exception:  # noqa: BLE001
    moe_route_radix = None  # type: ignore
    _HAS_RADIX = False


def is_arch_support_pdl() -> bool:
    """C600-U / MACA has no PDL (programmatic dependent launch)."""
    return False


_SCORING_FUNC_MAP = {
    "sigmoid": 0,
    "sqrtsoftplus": 1,
    "softmax": 2,
}


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
    scores_ptr,  # [M, N] fp32, GEMM output (raw logits)
    bias_ptr,  # [N]    fp32
    out_weights_ptr,  # [M, K] fp32
    out_indices_ptr,  # [M, K] int32
    M,
    routed_scaling_factor,
    moe_softcapping,
    N: tl.constexpr,
    K: tl.constexpr,  # total topk (includes fused shared experts)
    K_ROUTED: tl.constexpr,  # K - num_fused_shared_experts
    BLOCK_N: tl.constexpr,  # >= N, power of 2
    BLOCK_K: tl.constexpr,  # >= K, power of 2
    N_GROUP: tl.constexpr,  # expert groups (1 = ungrouped)
    TOPK_GROUP: tl.constexpr,  # groups kept per token (grouped routing)
    EXPERTS_PER_GROUP: tl.constexpr,  # N // N_GROUP
    BLOCK_G: tl.constexpr,  # >= N_GROUP, power of 2
    SCORING_FUNC: tl.constexpr,  # 0 = sigmoid, 1 = sqrtsoftplus, 2 = softmax
    HAS_SOFTCAP: tl.constexpr,  # tanh softcapping (softmax only)
    RENORMALIZE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,  # apply_routed_scaling_factor_on_output
    USE_PDL: tl.constexpr,
    USE_BITONIC: tl.constexpr,  # parallel bitonic top-k vs serial argmax passes
    LOGN: tl.constexpr,  # log2(BLOCK_N), bitonic stage count
    stride_sm,
    stride_sn,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
    BLOCK_M: tl.constexpr,  # rows processed per program (autotuned; ILP over the
    #                         serial top-k reduction chain -> hides its latency)
) -> None:
    # Row-tiled: each program handles BLOCK_M rows; all reductions run along the
    # expert (N) axis. Tiling rows keeps CTAs large enough to stay occupancy-bound
    # rather than launch-bound at small N (many tiny 1-warp CTAs otherwise).
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_m = offs_m < M
    mask_n = offs_n < N

    # prefetch bias before PDL wait
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(
        tl.float32
    )  # [BLOCK_N]

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    row_ptr = scores_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    mask2d = mask_m[:, None] & mask_n[None, :]
    scores = tl.load(row_ptr, mask=mask2d, other=0.0).to(
        tl.float32
    )  # [BLOCK_M, BLOCK_N]

    if SCORING_FUNC == 0:
        # sigmoid(x) = 1 / (1 + exp(-x)); bias is for ranking only, weight is bias-free.
        activated = tl.sigmoid(scores)
        biased = activated + bias[None, :]
    elif SCORING_FUNC == 1:
        # sqrt(softplus(x)) = sqrt(log1p(exp(x))); guard against overflow when x is large.
        sp = tl.where(scores > 20.0, scores, tl.log(1.0 + tl.exp(scores)))
        activated = tl.sqrt(sp)
        biased = activated + bias[None, :]
    else:
        # softmax over the row: weight is the softmax probability (bias kept), with
        # optional tanh softcapping. Ranking by the (softcapped, biased) logit is
        # monotonic with the softmax prob, so the topk loop below ranks on `biased`.
        logit = scores
        if HAS_SOFTCAP:
            # tanh(z) = 2*sigmoid(2z) - 1 (avoids relying on tl.math.tanh availability).
            z = logit / moe_softcapping
            logit = moe_softcapping * (2.0 * tl.sigmoid(2.0 * z) - 1.0)
        biased = logit + bias[None, :]
        biased = tl.where(mask_n[None, :], biased, -float("inf"))
        row_max = tl.max(biased, axis=1)[:, None]  # [BLOCK_M, 1]
        exp_row = tl.where(mask_n[None, :], tl.exp(biased - row_max), 0.0)
        row_sum = tl.sum(exp_row, axis=1)[:, None]  # [BLOCK_M, 1]
        activated = exp_row / row_sum

    biased = tl.where(mask_n[None, :], biased, -float("inf"))  # [BLOCK_M, BLOCK_N]

    # Map NaN -> a finite floor
    biased = tl.where(biased == biased, biased, -1e30)  # [BLOCK_M, BLOCK_N]

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
            vals2 = tl.where(vals >= top1, -float("inf"), vals)
            top2 = tl.max(vals2, axis=1)[:, None]  # [BLOCK_M, 1]
            group_score = tl.where(offs_g[None, :] == g, top1 + top2, group_score)

        gcur = group_score
        keep = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for _i in tl.static_range(TOPK_GROUP):
            gmax = tl.max(gcur, axis=1)[:, None]  # [BLOCK_M, 1]
            glane = tl.where(gcur == gmax, offs_g[None, :], N_GROUP + 1)
            win_g = tl.min(glane, axis=1)[:, None]  # [BLOCK_M, 1] lowest-id on ties
            keep = tl.where(group_of_n[None, :] == win_g, 1.0, keep)
            gcur = tl.where(offs_g[None, :] == win_g, -float("inf"), gcur)
        biased = tl.where(keep > 0.0, biased, -float("inf"))

    offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]
    mask_k_total = offs_k < K
    mask_k_routed = offs_k < K_ROUTED
    selected_vals = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    selected_idx = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.int32)

    cur = biased  # [BLOCK_M, BLOCK_N]
    if USE_BITONIC:
        # Parallel top-k via bitonic sort (replaces the K serial argmax passes,
        # which are ~85% of the kernel and form a latency-bound dependency
        # chain). Pack (value desc, id asc) into one int64 key so the network is
        # a total order and the lowest-id tie-break is automatic:
        #   key32 = monotone_uint32(biased);  packed = (~key32 << 11) | expert_id
        # Ascending sort then puts the best (largest value, smallest id) first,
        # bit-identical to iterative argmax (verified: random / all-tied / NaN).
        u = biased.to(tl.int32, bitcast=True)  # [BLOCK_M, BLOCK_N]
        key32 = u ^ ((u >> 31) | (-2147483648))  # ^0x80000000 -> monotone in value
        key64 = key32.to(tl.int64) & 0xFFFFFFFF  # 0..2^32-1, monotone in value
        packed = ((4294967295 - key64) << 11) | offs_n.to(tl.int64)[None, :]
        for s in tl.static_range(1, LOGN + 1):
            kk = 1 << s
            for t in tl.static_range(0, s):
                j = kk >> (t + 1)
                partner = tl.broadcast_to((offs_n ^ j)[None, :], [BLOCK_M, BLOCK_N])
                xp = tl.gather(packed, partner, axis=1)
                mn = tl.minimum(packed, xp)
                mx = tl.maximum(packed, xp)
                is_lower = ((offs_n & j) == 0)[None, :]
                asc = ((offs_n & kk) == 0)[None, :]  # ascending sub-block
                packed = tl.where(is_lower == asc, mn, mx)
        sorted_ids = (packed & 2047).to(tl.int32)  # [BLOCK_M, BLOCK_N] best-first
        # First BLOCK_K columns are the winners in order; gather their weights
        # in a single parallel pass (no per-slot serial gather).
        col = tl.broadcast_to(offs_k[None, :], [BLOCK_M, BLOCK_K])
        sel_idx = tl.gather(sorted_ids, col, axis=1)  # [BLOCK_M, BLOCK_K]
        sel_val = tl.gather(activated, sel_idx, axis=1)  # activated at winners
        routed_slot = mask_k_routed[None, :]
        selected_idx = tl.where(routed_slot, sel_idx, selected_idx)
        selected_vals = tl.where(routed_slot, sel_val, selected_vals)
    else:
        for k in tl.static_range(K_ROUTED):
            # tl.argmax fuses the max + lowest-id tie-break into ONE reduction
            # (verified: argmax returns the smallest index on ties == lowest
            # expert id wins). This replaces the former max + min(lane_id) pair.
            win_lane = tl.argmax(cur, axis=1)[:, None].to(tl.int32)  # [BLOCK_M, 1]
            win_activated = tl.sum(
                tl.where(offs_n[None, :] == win_lane, activated, 0.0), axis=1
            )[
                :, None
            ]  # [BLOCK_M, 1]
            slot = offs_k[None, :] == k  # [1, BLOCK_K]
            selected_vals = tl.where(slot, win_activated, selected_vals)
            selected_idx = tl.where(slot, win_lane, selected_idx)
            cur = tl.where(offs_n[None, :] == win_lane, -float("inf"), cur)

    routed_sum = tl.sum(tl.where(mask_k_routed[None, :], selected_vals, 0.0), axis=1)[
        :, None
    ]  # [BLOCK_M, 1]

    # Fill fused-shared-expert slots: weight = routed_sum / routed_scaling_factor,
    # id = num_experts + (slot - K_ROUTED).
    if K_ROUTED < K:
        is_shared = (offs_k[None, :] >= K_ROUTED) & mask_k_total[None, :]
        shared_weight = routed_sum / routed_scaling_factor  # [BLOCK_M, 1]
        shared_idx = (N + (offs_k - K_ROUTED)).to(tl.int32)[None, :]  # [1, BLOCK_K]
        selected_vals = tl.where(is_shared, shared_weight, selected_vals)
        selected_idx = tl.where(is_shared, shared_idx, selected_idx)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    if RENORMALIZE:
        norm = tl.where(routed_sum > 0.0, routed_sum, 1.0)  # [BLOCK_M, 1]
        selected_vals = selected_vals / norm
    if APPLY_SCALE:
        selected_vals = selected_vals * routed_scaling_factor

    out_w_ptr = (
        out_weights_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk
    )
    out_i_ptr = (
        out_indices_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    )
    store_mask = mask_m[:, None] & mask_k_total[None, :]
    tl.store(out_w_ptr, selected_vals, mask=store_mask)
    tl.store(out_i_ptr, selected_idx, mask=store_mask)


def moe_fused_gate(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
    moe_softcapping: float = 0.0,
    num_expert_group: int = 1,
    topk_group: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton fused router: scoring + bias + topk + (optional) renorm/scale.

    Mirrors the semantics of :func:`moe_fused_gate_jit` (the CUDA JIT kernel).
    With ``num_expert_group > 1`` it performs DeepSeek-V3 grouped routing
    (per-group top-2-sum group scores, keep ``topk_group`` groups, then top-k
    within). The first argument is named ``scores`` (raw GEMM logits) to match
    the existing call sites.
    """
    scoring_func_int = _SCORING_FUNC_MAP.get(scoring_func.lower())
    assert (
        scoring_func_int is not None
    ), f"Unknown scoring_func '{scoring_func}', must be one of {list(_SCORING_FUNC_MAP.keys())}"
    assert scores.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ), "scores must be float32/float16/bfloat16"
    assert bias.dtype == torch.float32, "bias must be float32"
    assert scores.ndim == 2, "scores must be 2D"
    assert bias.ndim == 1, "bias must be 1D"
    assert scores.size(1) == bias.size(0), "scores and bias must have same num_experts"
    assert topk > num_fused_shared_experts, "topk must be > num_fused_shared_experts"
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    # K3 radix-select fast path: native-CUDA radix-select replaces the 16
    # dependent argmax rounds (single CTA per token; ids bit-identical to this
    # triton kernel incl. ties).
    # The radix kernel keeps keys register-resident and returns winners in
    # expert-id order (skipping the biased-descending sort; downstream MoE
    # kernels are order-insensitive). It is 3.1-3.5x faster than the Triton
    # kernel at [1..8192, 896] top-16 on B200.
    if (
        _HAS_RADIX
        and scoring_func.lower() == "sigmoid"
        and num_fused_shared_experts == 0
        and num_expert_group <= 1
        and moe_softcapping == 0.0
    ):
        radix_args = (
            scores,
            bias,
            topk,
            renormalize,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
        )
        #if moe_route_radix.covered(scores, bias, topk):
        if False:
            return moe_route_radix.route_radix(*radix_args, sorted=False)

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

    BLOCK_N = triton.next_power_of_2(N)  # 256 -> 256, 384 -> 512
    BLOCK_K = triton.next_power_of_2(K)  # 6 -> 8, 8 -> 8
    # Parallel bitonic top-k (USE_BITONIC path below) was implemented and
    # verified bit-exact, but measured ~5x SLOWER on C600-U/MACA: tl.gather is
    # lowered to shared-memory traffic (not warp shuffles), int64 keys double the
    # tile movement, and a full 55-step sort does far more work than the top-16
    # needs -- whereas tl.max/tl.argmax already hit the fast subgroup-reduction
    # path. So the serial-argmax top-k (well-tuned reductions + row-tiling ILP)
    # wins here; bitonic is kept, gated off, for hardware where gather is cheap.
    use_bitonic = False
    log_n = BLOCK_N.bit_length() - 1  # exact: BLOCK_N is a power of two
    # BLOCK_M / num_warps / num_stages are chosen by @triton.autotune (row-tiling
    # for ILP overlaps independent rows; the autotuner searches per shape).
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
    use_pdl = is_arch_support_pdl()
    extra = {"launch_pdl": True} if use_pdl else {}
    _router_triton_kernel[grid](
        scores,
        bias,
        weights,
        indices,
        M,
        float(routed_scaling_factor),
        float(moe_softcapping),
        N=N,
        K=K,
        K_ROUTED=K_routed,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        N_GROUP=num_expert_group,
        TOPK_GROUP=topk_group,
        EXPERTS_PER_GROUP=experts_per_group,
        BLOCK_G=BLOCK_G,
        SCORING_FUNC=scoring_func_int,
        HAS_SOFTCAP=bool(moe_softcapping != 0.0),
        RENORMALIZE=bool(renormalize),
        APPLY_SCALE=bool(apply_routed_scaling_factor_on_output),
        USE_PDL=use_pdl,
        USE_BITONIC=use_bitonic,
        LOGN=log_n,
        stride_sm=scores.stride(0),
        stride_sn=scores.stride(1),
        stride_wm=weights.stride(0),
        stride_wk=weights.stride(1),
        stride_im=indices.stride(0),
        stride_ik=indices.stride(1),
        **extra,
    )
    return weights, indices
