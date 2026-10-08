

import torch
import triton
import triton.language as tl
import itertools
"""Standalone, dependency-free reimplementation of
`xpu_perf.micro_perf.core.utils.get_moe_tokens_info`.

It reproduces the reference algorithm exactly (same round-robin expert
assignment, same per-rank dispatch grouping, same 15-tuple return order), so the
unit test no longer needs to import xpu_perf.

Return tuple (index -> meaning), matching the original:
   0 num_scatter_tokens              = num_tokens * topk
   1 num_scatter_tokens_per_rank     = num_scatter_tokens // ep_size
   2 num_experts_per_rank            = num_experts // ep_size
   3 experts_start_idx               = ep_rank * num_experts_per_rank
   4 experts_end_idx                 = experts_start_idx + num_experts_per_rank
   5 all_select_experts              list[num_tokens][topk]  (global expert ids)
   6 all_select_weights              list[num_tokens][topk]  (1/topk each)
   7 dispatch_tokens                 int  (# routed rows landing on this rank)
   8 used_src_tokens                 int  (# distinct source tokens on this rank)
   9 expert_dispatch_tokens          list[epr][...]  token ids per local expert
  10 expert_dispatch_weights         list[epr][...]  weights per local expert
  11 scatter_token_id                list[dispatch_tokens]  (grouped by expert)
  12 scatter_token_weight            list[dispatch_tokens]
  13 expert_dispatch_token_count     list[epr]
  14 expert_dispatch_token_offset    list[epr]  (exclusive prefix sum of count)
"""
def get_moe_tokens_info(num_tokens, num_experts, topk, ep_size=1, ep_rank=0):
    # --- split tokens / experts across expert-parallel ranks -----------------
    num_scatter_tokens = num_tokens * topk
    num_scatter_tokens_per_rank = num_scatter_tokens // ep_size
    num_experts_per_rank = num_experts // ep_size

    experts_start_idx = ep_rank * num_experts_per_rank
    experts_end_idx = experts_start_idx + num_experts_per_rank

    # Round-robin expert order: lay experts of every rank out column-major so a
    # token's topk consecutive picks spread across ranks evenly.
    experts_idx_for_each_rank = [
        list(range(r * num_experts_per_rank, r * num_experts_per_rank + num_experts_per_rank))
        for r in range(ep_size)
    ]
    transpose_experts = [list(row) for row in zip(*experts_idx_for_each_rank)]
    experts_array = [num for row in transpose_experts for num in row]

    # --- each input token picks topk experts (cyclic global cursor) ----------
    all_select_experts = []
    all_select_weights = []
    cur_expert = 0
    for _ in range(num_tokens):
        cur_token_selections = []
        for _ in range(topk):
            cur_token_selections.append(experts_array[cur_expert])
            cur_expert += 1
            if cur_expert >= num_experts:
                cur_expert = 0
        all_select_experts.append(cur_token_selections)
        all_select_weights.append([1 / topk for _ in range(topk)])

    # --- keep only routes landing on THIS rank -------------------------------
    cur_rank_tokens = {}
    cur_rank_weights = {}
    dispatch_tokens = 0
    for token_idx in range(num_tokens):
        cur_token_dispatch_experts = []
        cur_token_dispatch_weights = []
        for expert_idx, expert_weight in zip(
            all_select_experts[token_idx], all_select_weights[token_idx]
        ):
            if experts_start_idx <= expert_idx < experts_end_idx:
                cur_token_dispatch_experts.append(expert_idx)
                cur_token_dispatch_weights.append(expert_weight)
        if cur_token_dispatch_experts:
            cur_rank_tokens[token_idx] = cur_token_dispatch_experts
            cur_rank_weights[token_idx] = cur_token_dispatch_weights
            dispatch_tokens += len(cur_token_dispatch_experts)

    used_src_tokens = len(cur_rank_tokens)

    # --- group routed rows by local expert -----------------------------------
    expert_dispatch_tokens = [[] for _ in range(num_experts_per_rank)]
    expert_dispatch_weights = [[] for _ in range(num_experts_per_rank)]
    expert_dispatch_token_count = [0 for _ in range(num_experts_per_rank)]

    for token_idx in cur_rank_tokens:
        for expert_idx, weight in zip(cur_rank_tokens[token_idx], cur_rank_weights[token_idx]):
            local = expert_idx - experts_start_idx
            expert_dispatch_tokens[local].append(token_idx)
            expert_dispatch_weights[local].append(weight)
            expert_dispatch_token_count[local] += 1

    expert_dispatch_token_offset = (
        [0] + list(itertools.accumulate(expert_dispatch_token_count))
    )[:num_experts_per_rank]

    # --- flatten to scatter arrays (rows ordered by expert) ------------------
    scatter_token_id = []
    scatter_token_weight = []
    for local, tokens in enumerate(expert_dispatch_tokens):
        for target_token, target_weight in zip(tokens, expert_dispatch_weights[local]):
            scatter_token_id.append(target_token)
            scatter_token_weight.append(target_weight)

    return (
        num_scatter_tokens,
        num_scatter_tokens_per_rank,
        num_experts_per_rank,
        experts_start_idx,
        experts_end_idx,
        all_select_experts,
        all_select_weights,
        dispatch_tokens,
        used_src_tokens,
        expert_dispatch_tokens,
        expert_dispatch_weights,
        scatter_token_id,
        scatter_token_weight,
        expert_dispatch_token_count,
        expert_dispatch_token_offset,
    )


@triton.jit
def _group_gemm_combine_kernel(
    scatter_tokens,
    per_token_scale,
    experts_weight,
    experts_scale,
    experts_token_count,
    experts_token_offset,
    scatter_token_id,
    scatter_token_weight,
    convergent_tokens,
    hidden_size: tl.constexpr,
    new_hidden_size: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    expert_idx = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    expert_token_count = tl.load(experts_token_count + expert_idx)
    expert_token_offset = tl.load(experts_token_offset + expert_idx)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    dispatch_idx = expert_token_offset + offs_m
    valid_m = offs_m < expert_token_count

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    num_pid_k = tl.cdiv(hidden_size, BLOCK_SIZE_K)
    w_base = experts_weight + expert_idx * new_hidden_size * hidden_size
    for k_start in range(0, num_pid_k):
        cur_k = k_start * BLOCK_SIZE_K + offs_k
        a = tl.load(
            scatter_tokens + dispatch_idx[:, None] * hidden_size + cur_k[None, :],
            mask=valid_m[:, None] & (cur_k[None, :] < hidden_size),
            other=0,
        )
        # Load weight tile directly as [K, N] (weight is stored [N, K]) so the
        # MMA is a plain A[M,K] x W[K,N] with NO in-loop tl.trans / shared transpose.
        w = tl.load(
            w_base + cur_k[:, None] + offs_n[None, :] * hidden_size,
            mask=(cur_k[:, None] < hidden_size)
            & (offs_n[None, :] < new_hidden_size),
            other=0,
        )
        # int8 x int8 -> int32 accumulate engages the INT8 tensor cores.
        acc += tl.dot(a, w, out_dtype=tl.int32)

    acc_f = acc.to(tl.float32)
    per_token_scale_value = tl.load(
        per_token_scale + dispatch_idx,
        mask=valid_m,
        other=0.0,
    ).to(tl.float32)
    weight_scale = tl.load(
        experts_scale + expert_idx * new_hidden_size + offs_n,
        mask=offs_n < new_hidden_size,
        other=0.0,
    ).to(tl.float32)
    router_weight = tl.load(
        scatter_token_weight + dispatch_idx,
        mask=valid_m,
        other=0.0,
    ).to(tl.float32)
    token_id = tl.load(scatter_token_id + dispatch_idx, mask=valid_m, other=0)
    token_scale = per_token_scale_value * router_weight
    acc_f *= token_scale[:, None] * weight_scale[None, :]

    out_ptrs = (
        convergent_tokens
        + token_id[:, None] * new_hidden_size
        + offs_n[None, :]
    )
    tl.atomic_add(
        out_ptrs,
        acc_f,
        sem="relaxed",
        mask=valid_m[:, None] & (offs_n[None, :] < new_hidden_size),
    )


# Best-known launch config for the C600-U (MACA) INT8 group-GEMM + combine.
# Tuned for hidden=1536, new_hidden=4096, per-expert M~640 (see unit test).
DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 128,
    "num_warps": 8,
    "num_stages": 2,
}


def group_gemm_combine(
    scatter_tokens,
    per_token_scale,
    experts_weight,
    experts_scale,
    experts_token_count,
    experts_token_offset,
    scatter_token_id,
    scatter_token_weight,
    convergent_tokens,
    hidden_size,
    new_hidden_size,
    num_experts_per_rank,
    max_expert_tokens,
    config=None,
):
    """Launch the grouped-GEMM + combine kernel.

    This is the single source of truth for the kernel launch; both the
    benchmark op class and the standalone unit test go through here so the
    triton directory always reflects the best-known version.
    """
    cfg = dict(DEFAULT_CONFIG if config is None else config)
    block_m = cfg["BLOCK_SIZE_M"]
    block_n = cfg["BLOCK_SIZE_N"]
    grid = (
        num_experts_per_rank,
        triton.cdiv(max_expert_tokens, block_m),
        triton.cdiv(new_hidden_size, block_n),
    )
    _group_gemm_combine_kernel[grid](
        scatter_tokens,
        per_token_scale,
        experts_weight,
        experts_scale,
        experts_token_count,
        experts_token_offset,
        scatter_token_id,
        scatter_token_weight,
        convergent_tokens,
        hidden_size,
        new_hidden_size,
        **cfg,
    )
    return convergent_tokens

