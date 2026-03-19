import importlib
import time
import os
from typing import Optional

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

import copy
import argparse
from datetime import datetime
from typing import Any, Dict, List, Tuple, TypedDict
import vllm_metax.patch
from mcoplib.triton_fused_moe import fused_moe_triton_kernel,fused_moe_triton_kernel_gptq_awq
from mcoplib.triton_utils import tl, triton

# ---- parameters (DeepSeek-R1 ratios, scaled down for safety) ----
SCALE = float(os.environ.get("DEEPSEEK_UNITTEST_SCALE", "1.0"))

K = max(1, int(4096 * SCALE))      # hidden size
N = max(1, int(11008 * SCALE))     # intermediate / output
E = max(1, int(64 * SCALE))        # number of experts
TOP_K = 6
M = max(1, int(256 * SCALE))       # tokens
DTYPE = torch.float16

WARMUP = 5
ITERS = 20
COS_THRESHOLD = 0.999  # Keep original threshold, but 0.9993 is already very good

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- helper: reference compute (pure PyTorch) ----
def reference_moe(A: torch.Tensor, B: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
    """
    A: (M, K)
    B: (E, N, K)  (so B[e].T is (K,N))
    expert_ids: (M, top_k)
    returns: (M, N)
    """
    M_loc, K_loc = A.shape
    E_loc, N_loc, K2 = B.shape
    assert K_loc == K2
    out = torch.zeros((M_loc, N_loc), dtype=A.dtype, device=A.device)
    for t in range(M_loc):
        a = A[t]  # (K,)
        for j in range(expert_ids.size(1)):
            e = int(expert_ids[t, j].item())
            out_piece = a @ B[e].t()  # (N,)
            out[t] += out_piece
    return out

def reference_moe_with_padding(A: torch.Tensor, B: torch.Tensor,
                              sorted_token_ids: torch.Tensor,
                              expert_ids: torch.Tensor,
                              TOP_K: int) -> torch.Tensor:
    """
    Exact reference implementation that matches kernel behavior including padding handling.
    This replicates how the Triton kernel processes sorted_token_ids.
    """
    M, K = A.shape
    E, N, K2 = B.shape
    assert K == K2

    # Initialize output buffer like kernel (M, TOP_K, N)
    C = torch.zeros((M, TOP_K, N), dtype=A.dtype, device=A.device)

    # Process each token in sorted_token_ids like the kernel
    for i, token_id in enumerate(sorted_token_ids):
        # Convert to actual token index (same as kernel: offs_token // top_k)
        actual_token_idx = token_id // TOP_K
        expert_idx_in_block = i % TOP_K  # Which expert position this corresponds to

        # Get expert ID for this block (same as kernel: expert_ids[pid_m])
        block_idx = i // TOP_K  # Which block we're in
        if block_idx < expert_ids.numel():
            expert_id = expert_ids[block_idx].item()

            # Only process valid experts and valid tokens
            if expert_id >= 0 and expert_id < E and actual_token_idx < M:
                # Compute A @ B[expert_id].T like the kernel
                a = A[actual_token_idx]  # (K,)
                b_expert = B[expert_id]  # (N, K)
                result = a @ b_expert.t()  # (N,)

                # Store in C at the correct position
                C[actual_token_idx, expert_idx_in_block, :] = result

    # Sum across TOP_K dimension like vLLM does
    return torch.sum(C, dim=1)  # Shape: (M, N)

def invoke_fused_moe_wna16_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_shape: list[int] | None,
):
    assert B_scale is not None and B_scale.ndim == 3
    assert B_zp is None or B_zp.ndim == 3
    assert block_shape is not None and block_shape[0] == 0

    M = A.size(0)
    num_tokens = M * top_k

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique,
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )

    fused_moe_triton_kernel_gptq_awq(
        grid,
        A,
        B,
        C,
        B_scale,
        B_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        A.size(1),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        B_scale.stride(0),
        B_scale.stride(2),
        B_scale.stride(1),
        B_zp.stride(0) if B_zp is not None else 0,
        B_zp.stride(2) if B_zp is not None else 0,
        B_zp.stride(1) if B_zp is not None else 0,
        block_k_diviable=A.size(1) % config["BLOCK_SIZE_K"] == 0,
        group_size=block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        has_zp=B_zp is not None,
        use_int4_w4a16=use_int4_w4a16,
        use_int8_w8a16=use_int8_w8a16,
        **config,
    )


def invoke_fused_moe_triton_kernel(
                            A: torch.Tensor,
                            B: torch.Tensor,
                            C: torch.Tensor,
                            A_scale: torch.Tensor | None,
                            B_scale: torch.Tensor | None,
                            topk_weights: torch.Tensor | None,
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool,
                            top_k: int,
                            config: dict[str, Any],
                            compute_type: tl.dtype,
                            use_fp8_w8a8: bool,
                            use_int8_w8a8: bool,
                            use_int8_w8a16: bool,
                            use_int4_w4a16: bool,
                            per_channel_quant: bool,
                            block_shape: list[int] | None = None,
                            B_bias: torch.Tensor | None = None) -> None:

    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    
    if use_fp8_w8a8 or use_int8_w8a8:
        assert B_scale is not None
        assert block_shape is None or triton.cdiv(
            B.size(-2), block_shape[0]
        ) == B_scale.size(-2)
        assert block_shape is None or triton.cdiv(
            B.size(-1), block_shape[1]
        ) == B_scale.size(-1)

    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    M = A.size(0)
    num_tokens = M * top_k

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique,
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) 
        * triton.cdiv(
            # ┌------------------------  Metax Modification -------------------------┐
            B.shape[1],
            META['BLOCK_SIZE_N']),
            META['SPLIT_K'])
            # └------------------------- Metax Modification -------------------------┘
    HAS_BIAS = B_bias is not None

    config = config.copy()
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
    if block_shape is not None:
        BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0], block_shape[1]))
    
            # ┌─────────────────────────────────────────────────────────────────────┐
        # │ 通过 fused_moe_triton_kernel Python接口调用 fused_moe_kernel         │
        # │ 支持 **config 方式传递配置参数                                       │
        # └─────────────────────────────────────────────────────────────────────┘
    fused_moe_triton_kernel(
            grid=grid,
            a_ptr=A,
            b_ptr=B,
            c_ptr=C,
            b_bias_ptr=B_bias,
            a_scale_ptr=A_scale,
            b_scale_ptr=B_scale,
            topk_weights_ptr=topk_weights,
            sorted_token_ids_ptr=sorted_token_ids,
            expert_ids_ptr=expert_ids,
            num_tokens_post_padded_ptr=num_tokens_post_padded,
            N=B.size(1),
            K=B.size(2),
            EM=EM,
            num_valid_tokens=num_tokens,
            stride_am=A.stride(0),
            stride_ak=A.stride(1),
            stride_be=B.stride(0),
            stride_bk=B.stride(2),
            stride_bn=B.stride(1),
            stride_cm=C.stride(1),
            stride_cn=C.stride(2),
            stride_asm=A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
            stride_ask=A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
            stride_bse=B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
            stride_bsk=B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
            stride_bsn=B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
            stride_bbe=B_bias.stride(0) if B_bias is not None else 0,
            stride_bbn=B_bias.stride(1) if B_bias is not None else 0,
            group_n=0 if block_shape is None else block_shape[0],
            group_k=0 if block_shape is None else block_shape[1],
            naive_block_assignment=(sorted_token_ids is None),
            # ┌──────────────────────────────────────────────────────────────────┐
            # │ 通过 **kwargs 传递的配置参数                                       │
            # └──────────────────────────────────────────────────────────────────┘
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            per_channel_quant=per_channel_quant,
            HAS_BIAS=HAS_BIAS,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            # ┌------------------------  Metax Modification -------------------------┐
            E=B.size(0),
            FAST_F32_TO_BF16=True,
            # └------------------------- Metax Modification -------------------------┘
            **config,
        )


def test_invoke_fused_moe_kernel_only():
    print("=== invoke_fused_moe_kernel final test ===")
    print(f"Device: {DEVICE}; dtype: {DTYPE}")
    print(f"Scaled shapes: M={M}, K={K}, N={N}, E={E}, TOP_K={TOP_K}")

    # import module object robustly


    # import the function under test
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size  # still OK

    # Prepare tensors
    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=DTYPE, device=DEVICE)
    # B stored as (E, N, K) - this is the correct layout as verified
    # Each expert weight matrix B[e] has shape (N, K) and is used as B[e].T in matmul
    B = torch.randn((E, N, K), dtype=DTYPE, device=DEVICE)
    # C must be (M, TOP_K, N) to match vLLM's actual usage pattern
    # This follows the exact allocation pattern used in vLLM's fused_moe.py
    C = torch.empty((M, TOP_K, N), dtype=DTYPE, device=DEVICE)

    # Generate correct MoE routing data
    topk_ids = torch.randint(0, E, (M, TOP_K), dtype=torch.int32, device=DEVICE)

    # Use the proper MoE token alignment function
    block_size = 128  # Should match config['BLOCK_SIZE_M']
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size, E, pad_sorted_ids=False
    )

    # CRITICAL FIX: Proper handling of padding tokens in sorted_token_ids
    # Follow vLLM's pattern from moe_permute_unpermute.py more precisely
    num_tokens = M * TOP_K  # = 256 * 6 = 1536
    original_max = sorted_token_ids.max().item()

    # Count how many tokens are valid vs padding
    valid_tokens = (sorted_token_ids < num_tokens).sum().item()
    padding_tokens = (sorted_token_ids >= num_tokens).sum().item()

    print(f"Token analysis:")
    print(f"  Original max token ID: {original_max}")
    print(f"  Valid tokens: {valid_tokens}")
    print(f"  Padding tokens: {padding_tokens}")
    print(f"  Total tokens: {sorted_token_ids.numel()}")

    # Only clamp if there are problematic padding tokens
    if original_max >= num_tokens:
        sorted_token_ids = sorted_token_ids.clamp(max=num_tokens - 1)
        print(f"  Applied clamp to prevent out-of-bounds access")
    else:
        print(f"  No clamp needed - all tokens are already in valid range")

    # preconditions for invoke_fused_moe_kernel: ensure topk_weights None and mul_routed_weight False
    topk_weights = None
    mul_routed_weight = False

    # Print debug information
    print(f"Generated MoE routing data:")
    print(f"  topk_ids shape: {topk_ids.shape}")
    print(f"  sorted_token_ids shape: {sorted_token_ids.shape}")
    print(f"  expert_ids shape: {expert_ids.shape}")
    print(f"  num_tokens_post_padded: {num_tokens_post_padded.item()}")
    print(f"  max token ID in sorted_token_ids: {sorted_token_ids.max().item() if sorted_token_ids.numel() > 0 else 'empty'}")
    print(f"  valid token range for kernel: [0, {M*TOP_K-1}] (will be divided by top_k)")
    print(f"  actual token indices after //top_k: [0, {(M*TOP_K-1)//TOP_K}]")

    # Validate data integrity
    max_token_id = sorted_token_ids.max().item()
    max_actual_token = max_token_id // TOP_K
    assert max_actual_token < M, f"Token {max_actual_token} exceeds input tensor size {M}"

    # Check expert IDs for out-of-bounds access
    max_expert_id = expert_ids.max().item()
    min_expert_id = expert_ids.min().item()
    print(f"  expert ID range: [{min_expert_id}, {max_expert_id}] (should be [0, {E-1}] or -1)")
    assert max_expert_id < E or max_expert_id == -1, f"Expert ID {max_expert_id} exceeds expert count {E}"

    # Check tensor B dimensions
    print(f"  Input tensor shapes:")
    print(f"    A: {A.shape}")
    print(f"    B: {B.shape} (should be [{E}, {N}, {K}])")
    print(f"    C: {C.shape}")

    # Validate expert access to B tensor
    if max_expert_id >= 0:  # Skip if all expert_ids are -1 (expert parallel case)
        assert max_expert_id < B.size(0), f"Expert {max_expert_id} exceeds B.size(0) {B.size(0)}"

    # avoid quantization paths
    A_scale = None
    B_scale = None
    B_zp = None
    B_bias = None

    # compute_type ignored by fake; try to set if triton available (optional)
    compute_type = None
    try:
        import triton.language as tl  # type: ignore
        compute_type = tl.float16
    except Exception:
        compute_type = None

    try:
        # Warmup runs
        for _ in range(WARMUP):
            
            print(f"invoke_fused_moe_wna16_triton_kernel C tensor:{C.shape}")

            invoke_fused_moe_triton_kernel(
                A, B, C,
                A_scale, B_scale,
                topk_weights,
                sorted_token_ids, expert_ids,
                num_tokens_post_padded,
                mul_routed_weight,
                TOP_K,
                config={"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "SPLIT_K":1, "GROUP_SIZE_M":8},
                compute_type=compute_type,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
                B_bias=B_bias,
            )
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter_ns()
        # Go back to original reference using topk_ids - it was correct!
        ref = reference_moe(A, B, topk_ids)
        t1 = time.perf_counter_ns()
        cpu_us = (t1 - t0) / 1000.0
        print(f"Single CPU call time: {cpu_us:.2f} us")
        # Time single CPU call (wall-clock)
        
        invoke_fused_moe_triton_kernel(
            A, B, C,
            A_scale, B_scale, 
            topk_weights,
            sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            TOP_K,
            config={"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "SPLIT_K":1, "GROUP_SIZE_M":8},
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
            B_bias=B_bias,
        )

        
        
        # Get kernel output from C and sum across top_k dimension (following vLLM's pattern)
        # vLLM uses ops.moe_sum to combine outputs from all top-k experts
        try:
            from vllm import _custom_ops as ops
            final_output = torch.empty((M, N), dtype=DTYPE, device=DEVICE)
            ops.moe_sum(C, final_output)
            out = final_output
        except Exception:
            # Fallback to manual sum if vLLM ops not available
            out = torch.sum(C, dim=1)  # Shape: (M, N)

        # Compute cosine similarity
        cos_sim = F.cosine_similarity(ref.reshape(-1).float(), out.reshape(-1).float(), dim=0).item()

        # Detailed precision analysis
        max_diff = torch.max(torch.abs(ref - out)).item()
        mean_diff = torch.mean(torch.abs(ref - out)).item()
        ref_norm = torch.norm(ref).item()
        out_norm = torch.norm(out).item()

        print(f"Precision analysis:")
        print(f"  Max absolute difference: {max_diff:.8f}")
        print(f"  Mean absolute difference: {mean_diff:.8f}")
        print(f"  Reference norm: {ref_norm:.6f}")
        print(f"  Kernel norm: {out_norm:.6f}")
        print(f"  Relative norm difference: {abs(ref_norm - out_norm) / ref_norm:.8f}")

        # GPU timing average (if CUDA)
        gpu_us_avg = None
        if DEVICE.type == "cuda":
            evt_s = torch.cuda.Event(enable_timing=True)
            evt_e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            evt_s.record()
            for _ in range(ITERS):
                invoke_fused_moe_triton_kernel(
                    A, B, C,
                    A_scale, B_scale,
                    topk_weights,
                    sorted_token_ids, expert_ids,
                    num_tokens_post_padded,
                    mul_routed_weight,
                    TOP_K,
                    config={"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "SPLIT_K":1, "GROUP_SIZE_M":8},
                    compute_type=compute_type,
                    use_fp8_w8a8=False,
                    use_int8_w8a8=False,
                    use_int8_w8a16=False,
                    use_int4_w4a16=False,
                    per_channel_quant=False,
                    block_shape=None,
                    B_bias=B_bias,
                )
            evt_e.record()
            torch.cuda.synchronize()
            gpu_ms_total = evt_s.elapsed_time(evt_e)
            gpu_us_avg = (gpu_ms_total * 1000.0) / ITERS

        # Estimate FLOPs & bytes (rough)
        flops = 2 * K * N * M * TOP_K  # conservative
        itemsize = torch.tensor([], dtype=DTYPE).element_size()
        bytes_moved = (A.numel() + B.numel() + C.numel() + expert_ids.numel() + sorted_token_ids.numel()) * itemsize
        compute_intensity = flops / bytes_moved if bytes_moved > 0 else float("inf")

        # Print summary and assert
        print("\n--- Summary ---")
        print(f"A:{tuple(A.shape)} B:{tuple(B.shape)} C:{tuple(C.shape)} expert_ids:{tuple(expert_ids.shape)}")
        print(f"Reference output shape: {tuple(ref.shape)}")
        print(f"Kernel output shape (after sum): {tuple(out.shape)}")
        print(f"Cosine similarity (ref vs kernel): {cos_sim:.8f}")
        print(f"Single-call CPU time: {cpu_us:.2f} μs")
        if gpu_us_avg is not None:
            print(f"Avg GPU kernel time (over {ITERS} iters): {gpu_us_avg:.2f} μs")
        print(f"Estimated FLOPs: {flops:,}, bytes moved (est): {bytes_moved:,}, F/B: {compute_intensity:.3f}")

        # Print summary and assert
        print("\n--- Summary ---")
        print(f"A:{tuple(A.shape)} B:{tuple(B.shape)} C:{tuple(C.shape)} expert_ids:{tuple(expert_ids.shape)}")
        print(f"Reference output shape: {tuple(ref.shape)}")
        print(f"Kernel output shape (after sum): {tuple(out.shape)}")
        print(f"Cosine similarity (ref vs kernel): {cos_sim:.8f}")
        print(f"Single-call CPU time: {cpu_us:.2f} μs")
        if gpu_us_avg is not None:
            print(f"Avg GPU kernel time (over {ITERS} iters): {gpu_us_avg:.2f} μs")
        print(f"Estimated FLOPs: {flops:,}, bytes moved (est): {bytes_moved:,}, F/B: {compute_intensity:.3f}")

        assert cos_sim >= COS_THRESHOLD, f"Cosine similarity {cos_sim:.6f} below threshold {COS_THRESHOLD}"

    finally:
        print("finish!!!!")


if __name__ == "__main__":
    test_invoke_fused_moe_kernel_only()
    print("Done.")