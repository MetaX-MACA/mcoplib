

import torch
import triton
import triton.language as tl


@triton.jit
def _quant_group_gemm_reduce_sum_kernel(
    hidden_states,
    per_token_scale,
    weight,
    weight_scale,
    output,
    sp_size: tl.constexpr,
    num_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    new_hidden_size: tl.constexpr,
    trans_w: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # -- L2-friendly program swizzle (group GROUP_SIZE_M M-tiles per N sweep) --
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(num_tokens, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(new_hidden_size, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_pid_k = tl.cdiv(hidden_size, BLOCK_SIZE_K)
    for sp_idx in range(0, sp_size):
        # INT8 matrix-core path: accumulate the raw int8xint8 products in int32
        # (bit-exact, and maps to the hardware int8 MMA instead of an fp32-out
        #  upconvert path). Scales are applied once, after the K-reduction.
        sp_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
        for k_start in range(0, num_pid_k):
            cur_k = k_start * BLOCK_SIZE_K + offs_k
            a = tl.load(
                hidden_states
                + sp_idx * num_tokens * hidden_size
                + offs_m[:, None] * hidden_size
                + cur_k[None, :],
                mask=(offs_m[:, None] < num_tokens)
                & (cur_k[None, :] < hidden_size),
                other=0,
            )

            if trans_w:
                    w = tl.load(
                        weight
                        + sp_idx * new_hidden_size * hidden_size
                        + cur_k[:, None]
                        + offs_n[None, :] * hidden_size,
                        mask=(cur_k[:, None] < hidden_size)
                        & (offs_n[None, :] < new_hidden_size),
                        other=0,
                    )
            else:
                w = tl.load(
                    weight
                    + sp_idx * hidden_size * new_hidden_size
                    + cur_k[:, None] * new_hidden_size
                    + offs_n[None, :],
                    mask=(cur_k[:, None] < hidden_size)
                    & (offs_n[None, :] < new_hidden_size),
                    other=0,
                )
            sp_acc += tl.dot(a, w, out_dtype=tl.int32)

        token_scale = tl.load(
            per_token_scale + sp_idx * num_tokens + offs_m,
            mask=offs_m < num_tokens,
            other=0.0,
        ).to(tl.float32)
        cur_weight_scale = tl.load(
            weight_scale + sp_idx * new_hidden_size + offs_n,
            mask=offs_n < new_hidden_size,
            other=0.0,
        ).to(tl.float32)
        acc += sp_acc.to(tl.float32) * token_scale[:, None] * cur_weight_scale[None, :]

    tl.store(
        output + offs_m[:, None] * new_hidden_size + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < num_tokens)
        & (offs_n[None, :] < new_hidden_size),
    )

