import importlib
import time
import os
import gc
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

import copy
import argparse
from datetime import datetime
from typing import Any, Dict, List, Tuple, TypedDict
from mcoplib.triton_fused_moe import fused_moe_triton_kernel,fused_moe_triton_kernel_gptq_awq
from mcoplib.triton_utils import tl, triton

# ---- parameters (DeepSeek-R1 ratios, scaled down for safety) ----
SCALE = float(os.environ.get("DEEPSEEK_UNITTEST_SCALE", "1.0"))

K = max(1, int(4096 * SCALE))      # hidden size
N = max(1, int(11008 * SCALE))     # intermediate / output
E = max(1, int(64 * SCALE))        # number of experts
TOP_K = 6
M = max(1, int(256 * SCALE))       # tokens

# Supported data types for testing
DTYPE_OPTIONS = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,  # For int8 quantization testing
}

# WNA16 quantization modes
WNA16_MODES = {
    "int8_w8a16": {"use_int8_w8a16": True, "use_int4_w4a16": False},
    "int4_w4a16": {"use_int8_w8a16": False, "use_int4_w4a16": True},
}

WARMUP = 5
ITERS = 20
COS_THRESHOLD = 0.995  # Adjusted for fp16/bf16 precision and int8 quantization tolerance
COS_THRESHOLD_FP = 0.997  # Threshold for float16/bfloat16
COS_THRESHOLD_INT8 = 0.99  # Threshold for int8 quantized (lower due to quantization)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- helper: MoE token alignment (pure Python implementation) ----
def moe_align_block_size_py(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    pad_sorted_ids: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Python implementation of MoE token alignment that matches vLLM's behavior.

    Key requirement: Each block must contain tokens for exactly ONE expert.
    This means we need to pad each expert's tokens to be aligned to block_size.

    Args:
        topk_ids: [M, top_k] tensor of expert IDs for each token
        block_size: Block size for kernel
        num_experts: Number of experts
        pad_sorted_ids: Whether to pad sorted_token_ids

    Returns:
        sorted_token_ids: Token indices sorted and padded by expert
        expert_ids: Expert ID for each block
        num_tokens_post_padded: Number of tokens after padding
    """
    M, top_k = topk_ids.shape
    device = topk_ids.device

    # Create token IDs in the format: token_idx * top_k + k
    token_ids = torch.arange(M, device=device).unsqueeze(1) * top_k
    token_ids = token_ids + torch.arange(top_k, device=device).unsqueeze(0)
    token_ids = token_ids.flatten()  # [M * top_k]

    # Get expert assignment for each position
    experts = topk_ids.flatten()  # [M * top_k]

    # Count tokens per expert
    expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    for e in range(num_experts):
        expert_counts[e] = (experts == e).sum().item()

    # For each expert, collect their tokens and pad to block_size
    sorted_token_ids_list = []
    expert_ids_list = []
    num_tokens = M * top_k

    for expert_id in range(num_experts):
        count = expert_counts[expert_id].item()
        if count == 0:
            continue

        # Get tokens for this expert
        mask = experts == expert_id
        expert_tokens = token_ids[mask]

        # Pad to block_size alignment
        if count % block_size != 0:
            pad_size = block_size - (count % block_size)
            expert_tokens = torch.cat([
                expert_tokens,
                torch.full((pad_size,), num_tokens, dtype=expert_tokens.dtype, device=device)
            ])

        # Add to lists
        sorted_token_ids_list.append(expert_tokens)

        # Number of blocks for this expert
        num_blocks_for_expert = expert_tokens.numel() // block_size
        for _ in range(num_blocks_for_expert):
            expert_ids_list.append(expert_id)

    # Concatenate all
    sorted_token_ids = torch.cat(sorted_token_ids_list) if sorted_token_ids_list else torch.tensor([], dtype=torch.int32, device=device)
    expert_ids = torch.tensor(expert_ids_list, dtype=torch.int32, device=device) if expert_ids_list else torch.tensor([], dtype=torch.int32, device=device)

    num_tokens_post_padded = torch.tensor(sorted_token_ids.numel(), dtype=torch.int32, device=device)

    return sorted_token_ids, expert_ids, num_tokens_post_padded


# ---- helper: quantization functions for int8 testing ----
def per_token_quant_int8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token INT8 quantization for activations.

    Args:
        tensor: Input tensor of shape [M, K] in float16/bfloat16

    Returns:
        quantized: INT8 tensor of shape [M, K]
        scales: Float32 tensor of shape [M] - per-token scales
    """
    abs_max = tensor.abs().max(dim=-1)[0]  # [M]
    scales = abs_max / 127.0
    # Avoid division by zero
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quantized = torch.round(tensor / scales.unsqueeze(-1))
    quantized = torch.clamp(quantized, -127, 127).to(torch.int8)
    return quantized, scales.float()


def per_channel_quant_int8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-channel INT8 quantization for weights.

    Args:
        tensor: Input tensor of shape [E, N, K] in float16/bfloat16

    Returns:
        quantized: INT8 tensor of shape [E, N, K]
        scales: Float32 tensor of shape [E, N] - per-channel scales
    """
    E, N, K = tensor.shape
    abs_max = tensor.abs().max(dim=-1)[0]  # [E, N]
    scales = abs_max / 127.0
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quantized = torch.round(tensor / scales.unsqueeze(-1))
    quantized = torch.clamp(quantized, -127, 127).to(torch.int8)
    return quantized, scales.float()


def dequantize_int8(quantized: torch.Tensor, scales: torch.Tensor, original_dtype: torch.dtype) -> torch.Tensor:
    """
    Dequantize INT8 tensor back to original dtype for reference computation.

    Args:
        quantized: INT8 tensor
        scales: Float32 scale tensor
        original_dtype: Target dtype (float16/bfloat16)

    Returns:
        Dequantized tensor in original_dtype
    """
    return (quantized.float() * scales.unsqueeze(-1) if scales.ndim == 2
            else quantized.float() * scales.unsqueeze(-1).unsqueeze(-1)).to(original_dtype)


# ---- helper: WNA16 quantization functions ----
def quantize_weight_int8_w8a16(
    weight: torch.Tensor,
    group_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weights to INT8 for W8A16 (activation is fp16/bf16, weight is int8).

    Uses symmetric quantization with zero point at 128 (for unsigned storage).

    Args:
        weight: Float tensor of shape [E, N, K] in float32/float16/bfloat16
        group_size: Group size for quantization along K dimension

    Returns:
        quantized: INT8 tensor of shape [E, N, K]
        scales: Float32 tensor of shape [E, N, K//group_size]
    """
    E, N, K = weight.shape
    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"

    device = weight.device
    num_groups = K // group_size

    # Initialize output tensors
    quantized = torch.empty((E, N, K), dtype=torch.int8, device=device)
    scales = torch.empty((E, N, num_groups), dtype=torch.float32, device=device)

    # Process in chunks to avoid OOM
    for g in range(num_groups):
        k_start = g * group_size
        k_end = (g + 1) * group_size

        # Get weight chunk
        w_chunk = weight[:, :, k_start:k_end].float()

        # Compute scale for this group
        abs_max = w_chunk.abs().max(dim=-1)[0]  # [E, N]
        scale = abs_max / 127.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        scales[:, :, g] = scale

        # Quantize
        w_q = torch.round(w_chunk / scale.unsqueeze(-1))
        w_q = torch.clamp(w_q, -127, 127)
        quantized[:, :, k_start:k_end] = w_q.to(torch.int8)

        # Free memory
        del w_chunk, w_q, abs_max, scale

    return quantized, scales


def quantize_weight_int4_w4a16(
    weight: torch.Tensor,
    group_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weights to INT4 (packed) for W4A16 (activation is fp16/bf16, weight is int4).

    Uses symmetric quantization with zero point at 8 (for unsigned storage).
    INT4 values are packed: 2 values per int8.

    Args:
        weight: Float tensor of shape [E, N, K] in float32/float16/bfloat16
        group_size: Group size for quantization along K dimension

    Returns:
        quantized: INT8 tensor of shape [E, N, K//2] (packed int4)
        scales: Float32 tensor of shape [E, N, K//group_size]
    """
    E, N, K = weight.shape
    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"
    assert K % 2 == 0, f"K ({K}) must be even for int4 packing"

    device = weight.device
    num_groups = K // group_size

    # Initialize output tensors
    scales = torch.empty((E, N, num_groups), dtype=torch.float32, device=device)

    # First pass: compute scales
    for g in range(num_groups):
        k_start = g * group_size
        k_end = (g + 1) * group_size
        w_chunk = weight[:, :, k_start:k_end].float()
        abs_max = w_chunk.abs().max(dim=-1)[0]
        scale = abs_max / 7.0
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        scales[:, :, g] = scale
        del w_chunk, abs_max, scale

    # Second pass: quantize and pack using group-wise processing
    # Process each group separately to manage memory
    quantized_even = []  # Even K indices
    quantized_odd = []   # Odd K indices

    for g in range(num_groups):
        k_start = g * group_size
        k_end = (g + 1) * group_size

        w_chunk = weight[:, :, k_start:k_end].float()
        w_q = torch.round(w_chunk / scales[:, :, g:g+1])
        w_q = torch.clamp(w_q, -7, 7).to(torch.int8)

        # Separate even and odd positions within this group
        even_indices = torch.arange(0, group_size, 2, device=device)
        odd_indices = torch.arange(1, group_size, 2, device=device)

        quantized_even.append(w_q[:, :, even_indices] if even_indices.numel() > 0 else None)
        quantized_odd.append(w_q[:, :, odd_indices] if odd_indices.numel() > 0 else None)

        del w_chunk, w_q

    # Concatenate all groups
    quantized_even = torch.cat([q for q in quantized_even if q is not None], dim=2)  # [E, N, K//2]
    quantized_odd = torch.cat([q for q in quantized_odd if q is not None], dim=2)    # [E, N, K//2]

    # Pack: low nibble = even K, high nibble = odd K
    low = (quantized_even + 8).to(torch.uint8)
    high = (quantized_odd + 8).to(torch.uint8)
    packed = (high.to(torch.int8) << 4) | low.to(torch.int8)

    del quantized_even, quantized_odd, low, high

    return packed, scales


def reference_moe_wna16(
    A: torch.Tensor,
    B_quant: torch.Tensor,
    B_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    group_size: int,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    original_dtype: torch.dtype
) -> torch.Tensor:
    """
    Reference MoE computation for WNA16 quantization (W8A16 or W4A16).

    Args:
        A: Activation tensor [M, K] in float16/bfloat16
        B_quant: Quantized weights [E, N, K] or [E, N, K//2] for int4
        B_scale: Per-group scales [E, N, K//group_size]
        expert_ids: Expert assignment [M, top_k]
        group_size: Group size for dequantization
        use_int8_w8a16: True for int8 weights
        use_int4_w4a16: True for int4 packed weights
        original_dtype: Output dtype

    Returns:
        Output tensor [M, N]
    """
    M, K = A.shape
    E, N = B_quant.shape[:2]

    # Dequantize weights
    if use_int8_w8a16:
        # B_quant shape: [E, N, K]
        # Dequantize: (b - 128) * scale
        num_groups = K // group_size
        B_dequant = torch.zeros(E, N, K, dtype=torch.float32, device=A.device)

        for g in range(num_groups):
            k_start = g * group_size
            k_end = (g + 1) * group_size
            scale = B_scale[:, :, g:g+1]  # [E, N, 1]
            B_dequant[:, :, k_start:k_end] = (B_quant[:, :, k_start:k_end].float() - 128) * scale

    elif use_int4_w4a16:
        # B_quant shape: [E, N, K//2] - packed int4
        # Unpack and dequantize
        B_dequant = torch.zeros(E, N, K, dtype=torch.float32, device=A.device)

        # Unpack: low nibble is k=even, high nibble is k=odd
        for k in range(K):
            if k % 2 == 0:
                # Low nibble
                packed_idx = k // 2
                b_val = (B_quant[:, :, packed_idx].to(torch.uint8) & 0xF).float()
            else:
                # High nibble
                packed_idx = k // 2
                b_val = ((B_quant[:, :, packed_idx].to(torch.uint8) >> 4) & 0xF).float()

            # Dequantize: (b - 8) * scale
            g = k // group_size
            b_dequant = (b_val - 8) * B_scale[:, :, g]
            B_dequant[:, :, k] = b_dequant

    # Compute MoE
    out = torch.zeros((M, N), dtype=torch.float32, device=A.device)
    for t in range(M):
        a = A[t].float()  # [K]
        for j in range(expert_ids.size(1)):
            e = int(expert_ids[t, j].item())
            out_piece = a @ B_dequant[e].t()  # [N]
            out[t] += out_piece

    return out.to(original_dtype)


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


def reference_moe_int8_w8a8(
    A_quant: torch.Tensor,
    A_scale: torch.Tensor,
    B_quant: torch.Tensor,
    B_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    original_dtype: torch.dtype
) -> torch.Tensor:
    """
    Reference MoE computation for INT8 W8A8 quantization.

    Args:
        A_quant: INT8 quantized activations [M, K]
        A_scale: Per-token activation scales [M]
        B_quant: INT8 quantized weights [E, N, K]
        B_scale: Per-channel weight scales [E, N]
        expert_ids: Expert assignment [M, top_k]
        original_dtype: Output dtype

    Returns:
        Output tensor [M, N] in original_dtype
    """
    M, K = A_quant.shape
    E, N, _ = B_quant.shape

    # Dequantize for reference computation
    A_dequant = A_quant.float() * A_scale.unsqueeze(-1)  # [M, K]
    B_dequant = B_quant.float() * B_scale.unsqueeze(-1)  # [E, N, K]

    out = torch.zeros((M, N), dtype=torch.float32, device=A_quant.device)
    for t in range(M):
        a = A_dequant[t]  # [K]
        for j in range(expert_ids.size(1)):
            e = int(expert_ids[t, j].item())
            out_piece = a @ B_dequant[e].t()  # [N]
            out[t] += out_piece
    return out.to(original_dtype)

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


def test_invoke_fused_moe_kernel_dtype(dtype_str: str = "float16"):
    """
    Test fused_moe_kernel for a specific data type.

    Args:
        dtype_str: Data type string, one of "float16", "bfloat16", "int8"
    """
    assert dtype_str in DTYPE_OPTIONS, f"Unsupported dtype: {dtype_str}"

    use_int8_w8a8 = dtype_str == "int8"
    compute_dtype = DTYPE_OPTIONS[dtype_str] if not use_int8_w8a8 else torch.bfloat16

    print(f"\n{'='*90}")
    print(f"Testing fused_moe_kernel with dtype: {dtype_str}")
    print(f"{'='*90}")
    print(f"Device: {DEVICE}; compute_dtype: {compute_dtype}; use_int8_w8a8: {use_int8_w8a8}")
    print(f"Scaled shapes: M={M}, K={K}, N={N}, E={E}, TOP_K={TOP_K}")

    # Clear GPU cache before test
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # Prepare tensors
    torch.manual_seed(0)

    if use_int8_w8a8:
        # INT8 W8A8 quantization path
        # Generate float tensors first, then quantize
        A_float = torch.randn((M, K), dtype=torch.bfloat16, device=DEVICE)
        B_float = torch.randn((E, N, K), dtype=torch.bfloat16, device=DEVICE)

        # Quantize A (per-token) and B (per-channel)
        A, A_scale = per_token_quant_int8(A_float)
        B, B_scale = per_channel_quant_int8(B_float)

        # C output is in compute_dtype (bfloat16 for int8 path)
        C = torch.empty((M, TOP_K, N), dtype=compute_dtype, device=DEVICE)

        print(f"INT8 quantization info:")
        print(f"  A quantized shape: {A.shape}, A_scale shape: {A_scale.shape}")
        print(f"  B quantized shape: {B.shape}, B_scale shape: {B_scale.shape}")
    else:
        # Float16/Bfloat16 path
        A = torch.randn((M, K), dtype=compute_dtype, device=DEVICE)
        B = torch.randn((E, N, K), dtype=compute_dtype, device=DEVICE)
        C = torch.empty((M, TOP_K, N), dtype=compute_dtype, device=DEVICE)
        A_scale = None
        B_scale = None

    # Generate correct MoE routing data
    topk_ids = torch.randint(0, E, (M, TOP_K), dtype=torch.int32, device=DEVICE)

    # Use the proper MoE token alignment function
    block_size = 128  # Should match config['BLOCK_SIZE_M']
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size_py(
        topk_ids, block_size, E, pad_sorted_ids=False
    )

    # CRITICAL FIX: Proper handling of padding tokens in sorted_token_ids
    num_tokens = M * TOP_K
    original_max = sorted_token_ids.max().item()

    valid_tokens = (sorted_token_ids < num_tokens).sum().item()
    padding_tokens = (sorted_token_ids >= num_tokens).sum().item()

    print(f"Token analysis:")
    print(f"  Original max token ID: {original_max}")
    print(f"  Valid tokens: {valid_tokens}")
    print(f"  Padding tokens: {padding_tokens}")
    print(f"  Total tokens: {sorted_token_ids.numel()}")

    if original_max >= num_tokens:
        sorted_token_ids = sorted_token_ids.clamp(max=num_tokens - 1)
        print(f"  Applied clamp to prevent out-of-bounds access")
    else:
        print(f"  No clamp needed - all tokens are already in valid range")

    topk_weights = None
    mul_routed_weight = False

    print(f"Generated MoE routing data:")
    print(f"  topk_ids shape: {topk_ids.shape}")
    print(f"  sorted_token_ids shape: {sorted_token_ids.shape}")
    print(f"  expert_ids shape: {expert_ids.shape}")
    print(f"  num_tokens_post_padded: {num_tokens_post_padded.item()}")

    # Validate data integrity
    max_token_id = sorted_token_ids.max().item()
    max_actual_token = max_token_id // TOP_K
    assert max_actual_token < M, f"Token {max_actual_token} exceeds input tensor size {M}"

    max_expert_id = expert_ids.max().item()
    min_expert_id = expert_ids.min().item()
    print(f"  expert ID range: [{min_expert_id}, {max_expert_id}] (should be [0, {E-1}] or -1)")
    assert max_expert_id < E or max_expert_id == -1, f"Expert ID {max_expert_id} exceeds expert count {E}"

    print(f"  Input tensor shapes:")
    print(f"    A: {A.shape}")
    print(f"    B: {B.shape} (should be [{E}, {N}, {K}])")
    print(f"    C: {C.shape}")
    if use_int8_w8a8:
        print(f"    A_scale: {A_scale.shape}")
        print(f"    B_scale: {B_scale.shape}")

    if max_expert_id >= 0:
        assert max_expert_id < B.size(0), f"Expert {max_expert_id} exceeds B.size(0) {B.size(0)}"

    B_zp = None
    B_bias = None

    # Set compute_type based on dtype
    compute_type = None
    try:
        import triton.language as tl
        if use_int8_w8a8:
            compute_type = tl.bfloat16
        elif compute_dtype == torch.float16:
            compute_type = tl.float16
        elif compute_dtype == torch.bfloat16:
            compute_type = tl.bfloat16
    except Exception:
        compute_type = None

    try:
        # Warmup runs
        for _ in range(WARMUP):
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
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=use_int8_w8a8,
                block_shape=None,
                B_bias=B_bias,
            )
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter_ns()
        # Reference computation
        if use_int8_w8a8:
            ref = reference_moe_int8_w8a8(A, A_scale, B, B_scale, topk_ids, compute_dtype)
        else:
            ref = reference_moe(A, B, topk_ids)
        t1 = time.perf_counter_ns()
        cpu_us = (t1 - t0) / 1000.0
        print(f"Single CPU call time: {cpu_us:.2f} us")

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
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=use_int8_w8a8,
            block_shape=None,
            B_bias=B_bias,
        )

        # Get kernel output from C and sum across top_k dimension
        try:
            from vllm import _custom_ops as ops
            final_output = torch.empty((M, N), dtype=compute_dtype, device=DEVICE)
            ops.moe_sum(C, final_output)
            out = final_output
        except Exception:
            out = torch.sum(C, dim=1)

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
                    use_int8_w8a8=use_int8_w8a8,
                    use_int8_w8a16=False,
                    use_int4_w4a16=False,
                    per_channel_quant=use_int8_w8a8,
                    block_shape=None,
                    B_bias=B_bias,
                )
            evt_e.record()
            torch.cuda.synchronize()
            gpu_ms_total = evt_s.elapsed_time(evt_e)
            gpu_us_avg = (gpu_ms_total * 1000.0) / ITERS

        # Estimate FLOPs & bytes
        flops = 2 * K * N * M * TOP_K
        itemsize = torch.tensor([], dtype=compute_dtype).element_size()
        if use_int8_w8a8:
            bytes_moved = (A.numel() + B.numel() + C.numel() +
                          A_scale.numel() * 4 + B_scale.numel() * 4 +
                          expert_ids.numel() + sorted_token_ids.numel())
        else:
            bytes_moved = (A.numel() + B.numel() + C.numel() + expert_ids.numel() + sorted_token_ids.numel()) * itemsize
        compute_intensity = flops / bytes_moved if bytes_moved > 0 else float("inf")

        print("\n--- Summary ---")
        print(f"A:{tuple(A.shape)} B:{tuple(B.shape)} C:{tuple(C.shape)} expert_ids:{tuple(expert_ids.shape)}")
        if use_int8_w8a8:
            print(f"A_scale:{tuple(A_scale.shape)} B_scale:{tuple(B_scale.shape)}")
        print(f"Reference output shape: {tuple(ref.shape)}")
        print(f"Kernel output shape (after sum): {tuple(out.shape)}")
        print(f"Cosine similarity (ref vs kernel): {cos_sim:.8f}")
        print(f"Single-call CPU time: {cpu_us:.2f} μs")
        if gpu_us_avg is not None:
            print(f"Avg GPU kernel time (over {ITERS} iters): {gpu_us_avg:.2f} μs")
        print(f"Estimated FLOPs: {flops:,}, bytes moved (est): {bytes_moved:,}, F/B: {compute_intensity:.3f}")

        # Use appropriate threshold based on dtype
        threshold = COS_THRESHOLD_INT8 if use_int8_w8a8 else COS_THRESHOLD_FP
        assert cos_sim >= threshold, f"Cosine similarity {cos_sim:.6f} below threshold {threshold}"
        print(f"✓ Test passed for dtype={dtype_str}")

        return True

    finally:
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Finish test for dtype={dtype_str}")


def test_invoke_fused_moe_wna16_kernel(mode_str: str = "int8_w8a16"):
    """
    Test invoke_fused_moe_wna16_triton_kernel for WNA16 quantization modes.

    Args:
        mode_str: Quantization mode, one of "int8_w8a16" or "int4_w4a16"
    """
    assert mode_str in WNA16_MODES, f"Unsupported WNA16 mode: {mode_str}"

    mode_config = WNA16_MODES[mode_str]
    use_int8_w8a16 = mode_config["use_int8_w8a16"]
    use_int4_w4a16 = mode_config["use_int4_w4a16"]

    # Group size for quantization (common values: 64, 128)
    GROUP_SIZE = 128

    # Use smaller scale for WNA16 tests to avoid OOM
    WNA16_SCALE = 0.5  # Reduce tensor sizes for WNA16 tests
    wna16_M = max(1, int(M * WNA16_SCALE))
    wna16_K = max(1, int(K * WNA16_SCALE))
    wna16_N = max(1, int(N * WNA16_SCALE))
    wna16_E = max(1, int(E * WNA16_SCALE))

    compute_dtype = torch.bfloat16  # WNA16 uses bf16/fp16 activation

    print(f"\n{'='*90}")
    print(f"Testing invoke_fused_moe_wna16_triton_kernel with mode: {mode_str}")
    print(f"{'='*90}")
    print(f"Device: {DEVICE}; compute_dtype: {compute_dtype}")
    print(f"GROUP_SIZE: {GROUP_SIZE}")
    print(f"Config: use_int8_w8a16={use_int8_w8a16}, use_int4_w4a16={use_int4_w4a16}")
    print(f"Scaled shapes: M={wna16_M}, K={wna16_K}, N={wna16_N}, E={wna16_E}, TOP_K={TOP_K}")

    # Clear GPU cache before test
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # Ensure K is divisible by GROUP_SIZE
    actual_K = (wna16_K // GROUP_SIZE) * GROUP_SIZE
    if actual_K != wna16_K:
        print(f"Note: Adjusted K from {wna16_K} to {actual_K} for group alignment")

    # Prepare tensors
    torch.manual_seed(0)

    # Create activation tensor (float16/bfloat16)
    A = torch.randn((wna16_M, actual_K), dtype=compute_dtype, device=DEVICE)

    # Create weight tensor and quantize
    B_float = torch.randn((wna16_E, wna16_N, actual_K), dtype=compute_dtype, device=DEVICE)

    if use_int8_w8a16:
        B, B_scale = quantize_weight_int8_w8a16(B_float.float(), GROUP_SIZE)
        print(f"INT8 W8A16 quantization info:")
        print(f"  B quantized shape: {B.shape}, dtype: {B.dtype}")
        print(f"  B_scale shape: {B_scale.shape}")
    elif use_int4_w4a16:
        B, B_scale = quantize_weight_int4_w4a16(B_float.float(), GROUP_SIZE)
        print(f"INT4 W4A16 quantization info:")
        print(f"  B quantized shape: {B.shape}, dtype: {B.dtype} (packed int4)")
        print(f"  B_scale shape: {B_scale.shape}")

    # Output tensor
    C = torch.empty((wna16_M, TOP_K, wna16_N), dtype=compute_dtype, device=DEVICE)

    # Generate MoE routing data
    topk_ids = torch.randint(0, wna16_E, (wna16_M, TOP_K), dtype=torch.int32, device=DEVICE)

    # Token alignment
    block_size = 128
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size_py(
        topk_ids, block_size, wna16_E, pad_sorted_ids=False
    )

    # Handle padding tokens
    num_tokens = wna16_M * TOP_K
    original_max = sorted_token_ids.max().item()

    valid_tokens = (sorted_token_ids < num_tokens).sum().item()
    padding_tokens = (sorted_token_ids >= num_tokens).sum().item()

    print(f"Token analysis:")
    print(f"  Original max token ID: {original_max}")
    print(f"  Valid tokens: {valid_tokens}")
    print(f"  Padding tokens: {padding_tokens}")
    print(f"  Total tokens: {sorted_token_ids.numel()}")

    if original_max >= num_tokens:
        sorted_token_ids = sorted_token_ids.clamp(max=num_tokens - 1)
        print(f"  Applied clamp to prevent out-of-bounds access")

    topk_weights = None
    mul_routed_weight = False
    B_zp = None

    # Set compute_type
    compute_type = None
    try:
        import triton.language as tl
        compute_type = tl.bfloat16
    except Exception:
        compute_type = None

    try:
        # Config for WNA16 kernel
        config = {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "SPLIT_K": 1  # Required by fused_moe_kernel_gptq_awq
        }

        # block_shape for WNA16: [0, group_size] - only K dimension grouped
        block_shape = [0, GROUP_SIZE]

        # Reshape B_scale to 3D: [E, K//group_size, N]
        # The kernel expects B_scale with ndim == 3
        B_scale_3d = B_scale.transpose(1, 2)  # [E, K//group_size, N]

        print(f"  B_scale_3d shape for kernel: {B_scale_3d.shape}")

        # Warmup runs
        for _ in range(WARMUP):
            invoke_fused_moe_wna16_triton_kernel(
                A, B, C,
                B_scale_3d, B_zp,
                topk_weights,
                sorted_token_ids, expert_ids,
                num_tokens_post_padded,
                mul_routed_weight,
                TOP_K,
                config=config,
                compute_type=compute_type,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                block_shape=block_shape,
            )
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        # Reference computation
        t0 = time.perf_counter_ns()
        ref = reference_moe_wna16(
            A, B, B_scale, topk_ids, GROUP_SIZE,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            original_dtype=compute_dtype
        )
        t1 = time.perf_counter_ns()
        cpu_us = (t1 - t0) / 1000.0
        print(f"Single CPU call time: {cpu_us:.2f} us")

        # Final kernel execution
        invoke_fused_moe_wna16_triton_kernel(
            A, B, C,
            B_scale_3d, B_zp,
            topk_weights,
            sorted_token_ids, expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            TOP_K,
            config=config,
            compute_type=compute_type,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            block_shape=block_shape,
        )

        # Sum across top_k dimension
        out = torch.sum(C, dim=1)

        # Compute cosine similarity
        cos_sim = F.cosine_similarity(ref.reshape(-1).float(), out.reshape(-1).float(), dim=0).item()

        # Precision analysis
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

        # GPU timing
        gpu_us_avg = None
        if DEVICE.type == "cuda":
            evt_s = torch.cuda.Event(enable_timing=True)
            evt_e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            evt_s.record()
            for _ in range(ITERS):
                invoke_fused_moe_wna16_triton_kernel(
                    A, B, C,
                    B_scale_3d, B_zp,
                    topk_weights,
                    sorted_token_ids, expert_ids,
                    num_tokens_post_padded,
                    mul_routed_weight,
                    TOP_K,
                    config=config,
                    compute_type=compute_type,
                    use_int8_w8a16=use_int8_w8a16,
                    use_int4_w4a16=use_int4_w4a16,
                    block_shape=block_shape,
                )
            evt_e.record()
            torch.cuda.synchronize()
            gpu_ms_total = evt_s.elapsed_time(evt_e)
            gpu_us_avg = (gpu_ms_total * 1000.0) / ITERS

        # FLOPs and bandwidth estimation
        flops = 2 * actual_K * wna16_N * wna16_M * TOP_K
        if use_int4_w4a16:
            weight_bytes = wna16_E * wna16_N * (actual_K // 2)  # int4 is packed
        else:
            weight_bytes = wna16_E * wna16_N * actual_K  # int8
        bytes_moved = (wna16_M * actual_K * 2 +  # activation
                       weight_bytes +  # weights
                       wna16_M * TOP_K * wna16_N * 2 +  # output
                       B_scale.numel() * 4)  # scales

        print("\n--- Summary ---")
        print(f"A:{tuple(A.shape)} B:{tuple(B.shape)} C:{tuple(C.shape)}")
        print(f"B_scale:{tuple(B_scale_3d.shape)} GROUP_SIZE:{GROUP_SIZE}")
        print(f"Reference output shape: {tuple(ref.shape)}")
        print(f"Kernel output shape (after sum): {tuple(out.shape)}")
        print(f"Cosine similarity (ref vs kernel): {cos_sim:.8f}")
        print(f"Single-call CPU time: {cpu_us:.2f} μs")
        if gpu_us_avg is not None:
            print(f"Avg GPU kernel time (over {ITERS} iters): {gpu_us_avg:.2f} μs")
        print(f"Estimated FLOPs: {flops:,}")

        # Threshold for WNA16 is lower due to quantization
        # int4 has lower precision than int8, so use a lower threshold
        threshold = 0.97 if use_int4_w4a16 else 0.98
        assert cos_sim >= threshold, f"Cosine similarity {cos_sim:.6f} below threshold {threshold}"
        print(f"✓ Test passed for mode={mode_str}")

        return True

    finally:
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Finish test for mode={mode_str}")


def test_invoke_fused_moe_kernel_only():
    """
    Run tests for all supported data types and quantization modes.
    """
    print("=" * 90)
    print("fused_moe Triton Kernel Multi-Dtype Unit Test")
    print("=" * 90)
    print(f"Testing dtypes: {list(DTYPE_OPTIONS.keys())}")
    print(f"Testing WNA16 modes: {list(WNA16_MODES.keys())}")

    all_passed = True
    results = {}

    # Test float16, bfloat16, int8 (W8A8)
    for dtype_str in DTYPE_OPTIONS.keys():
        try:
            # Clear GPU memory before each test
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            passed = test_invoke_fused_moe_kernel_dtype(dtype_str)
            results[dtype_str] = "PASS" if passed else "FAIL"
        except Exception as e:
            all_passed = False
            results[dtype_str] = f"ERROR: {e}"
            print(f"✗ Test failed for dtype={dtype_str}: {e}")
        finally:
            # Clear GPU memory after each test
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    # Test WNA16 modes (int8_w8a16, int4_w4a16)
    for mode_str in WNA16_MODES.keys():
        try:
            # Clear GPU memory before each test
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            passed = test_invoke_fused_moe_wna16_kernel(mode_str)
            results[mode_str] = "PASS" if passed else "FAIL"
        except Exception as e:
            all_passed = False
            results[mode_str] = f"ERROR: {e}"
            import traceback
            traceback.print_exc()
            print(f"✗ Test failed for mode={mode_str}: {e}")
        finally:
            # Clear GPU memory after each test
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 90)
    print("Test Summary")
    print("=" * 90)
    for name, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"{status} {name}: {result}")

    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed, please check output!")

    return all_passed


if __name__ == "__main__":
    success = test_invoke_fused_moe_kernel_only()
    print("Done.")