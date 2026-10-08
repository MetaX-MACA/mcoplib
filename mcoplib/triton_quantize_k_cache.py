"""Triton kernels for GLM-4.2 FP8 MLA K-cache quantization."""

from typing import Optional, Tuple

import torch

from .triton_utils import HAS_TRITON, tl, triton


DIM_NOPE = 512
DIM_ROPE = 64
GROUP_SIZE = 128
NUM_NOPE_GROUPS = DIM_NOPE // GROUP_SIZE
NOPE_PART_BYTES = DIM_NOPE + NUM_NOPE_GROUPS * 4
ROPE_PART_BYTES = DIM_ROPE * 2


def _fp8_dtype() -> torch.dtype:
    """Return the FP8 format used by the MLA cache byte layout."""
    return torch.float8_e4m3fn


@triton.jit
def _quantize_k_cache_fast_kernel(
    output_nope_q_ptr,
    output_nope_s_ptr,
    output_rope_ptr,
    k_nope_ptr,
    k_rope_ptr,
    output_nope_q_stride_0: int,
    output_nope_s_stride_0: int,
    output_rope_stride_0: int,
    k_nope_stride_0: int,
    k_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    # two dimensions: axis 0: token dim
    # axis 1: block dim for the dimensions.
    token_id = tl.program_id(0)
    raw_block_id = tl.program_id(1)

    # The blocks will be divided into the nope and rope.
    if raw_block_id < NUM_NOPE_BLOCKS:
        # a. quant nope
        effective_block_id = raw_block_id
        # get the offsets for the nope part in the hidden dim
        # [0..127] [128..255]
        # 0*128 + [0..127], 1*128 + [0..127],
        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_NOPE
        # token_id * k_nope_stride_0  : is the token dim
        ptr = k_nope_ptr + token_id * k_nope_stride_0 + offs

        y = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        # the ref impl do not have a `tl.maximum(... eps)`, so we remove it here
        y_s = tl.max(tl.abs(y)) / FP8_MAX
        y_s_inv = 1.0 / y_s
        y_q = tl.clamp(y * y_s_inv, FP8_MIN, FP8_MAX).to(
            output_nope_q_ptr.dtype.element_ty
        )
        # store the fp8 scaled part and the scales.
        dst_q_ptr = output_nope_q_ptr + token_id * output_nope_q_stride_0 + offs
        dst_s_ptr = (
            output_nope_s_ptr + token_id * output_nope_s_stride_0 + effective_block_id
        )

        tl.store(
            dst_q_ptr, y_q, mask=mask
        )
        tl.store(dst_s_ptr, y_s)
    else:
        # b. copy rope
        # pass the nope blocks
        effective_block_id = raw_block_id - NUM_NOPE_BLOCKS

        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE

        src_ptr = k_rope_ptr + token_id * k_rope_stride_0 + offs
        dst_ptr = output_rope_ptr + token_id * output_rope_stride_0 + offs

        data = tl.load(src_ptr, mask=mask)
        tl.store(dst_ptr, data, mask=mask)


@triton.jit
def _quantize_k_cache_fast_kernel_optimized(
    output_nope_q_ptr,
    output_nope_s_ptr,
    output_rope_ptr,
    k_nope_ptr,
    k_rope_ptr,
    num_tokens,
    output_nope_q_stride_0: int,
    output_nope_s_stride_0: int,
    output_rope_stride_0: int,
    k_nope_stride_0: int,
    k_rope_stride_0: int,
    BLOCK_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    token_block_id = tl.program_id(0)
    group_id = tl.program_id(1)

    token_offsets = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    group_offsets = group_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    token_mask = token_offsets < num_tokens

    input_offsets = (
        token_offsets[:, None] * k_nope_stride_0 + group_offsets[None, :]
    )
    values = tl.load(
        k_nope_ptr + input_offsets,
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    max_abs = tl.max(tl.abs(values), axis=1)
    scales = max_abs / FP8_MAX
    safe_scales = tl.where(max_abs > 0.0, scales, 1.0)
    quantized = tl.clamp(
        values / safe_scales[:, None], FP8_MIN, FP8_MAX
    ).to(output_nope_q_ptr.dtype.element_ty)

    output_q_offsets = (
        token_offsets[:, None] * output_nope_q_stride_0
        + group_offsets[None, :]
    )
    output_s_offsets = (
        token_offsets * output_nope_s_stride_0 + group_id
    )
    tl.store(
        output_nope_q_ptr + output_q_offsets,
        quantized,
        mask=token_mask[:, None],
    )
    tl.store(
        output_nope_s_ptr + output_s_offsets,
        safe_scales,
        mask=token_mask,
    )

    if group_id == 0:
        rope_offsets = tl.arange(0, DIM_ROPE)
        rope_input_offsets = (
            token_offsets[:, None] * k_rope_stride_0
            + rope_offsets[None, :]
        )
        rope_output_offsets = (
            token_offsets[:, None] * output_rope_stride_0
            + rope_offsets[None, :]
        )
        rope = tl.load(
            k_rope_ptr + rope_input_offsets,
            mask=token_mask[:, None],
            other=0.0,
        )
        tl.store(
            output_rope_ptr + rope_output_offsets,
            rope,
            mask=token_mask[:, None],
        )


@triton.jit
def _quantize_k_cache_all_groups_kernel(
    output_nope_q_ptr,
    output_nope_s_ptr,
    output_rope_ptr,
    k_nope_ptr,
    k_rope_ptr,
    num_tokens,
    output_nope_q_stride_0: int,
    output_nope_s_stride_0: int,
    output_rope_stride_0: int,
    k_nope_stride_0: int,
    k_rope_stride_0: int,
    BLOCK_M: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    token_block_id = tl.program_id(0)
    token_offsets = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    nope_offsets = tl.arange(0, DIM_NOPE)
    token_mask = token_offsets < num_tokens

    input_offsets = (
        token_offsets[:, None] * k_nope_stride_0 + nope_offsets[None, :]
    )
    values = tl.load(
        k_nope_ptr + input_offsets,
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    values_grouped = tl.reshape(values, (BLOCK_M, NUM_GROUPS, GROUP_SIZE))
    max_abs = tl.max(tl.abs(values_grouped), axis=2)
    scales = max_abs / FP8_MAX
    safe_scales = tl.where(max_abs > 0.0, scales, 1.0)
    quantized_grouped = tl.clamp(
        values_grouped / safe_scales[:, :, None], FP8_MIN, FP8_MAX
    ).to(output_nope_q_ptr.dtype.element_ty)
    quantized = tl.reshape(quantized_grouped, (BLOCK_M, DIM_NOPE))

    output_q_offsets = (
        token_offsets[:, None] * output_nope_q_stride_0
        + nope_offsets[None, :]
    )
    group_offsets = tl.arange(0, NUM_GROUPS)
    output_s_offsets = (
        token_offsets[:, None] * output_nope_s_stride_0
        + group_offsets[None, :]
    )
    tl.store(
        output_nope_q_ptr + output_q_offsets,
        quantized,
        mask=token_mask[:, None],
    )
    tl.store(
        output_nope_s_ptr + output_s_offsets,
        safe_scales,
        mask=token_mask[:, None],
    )

    rope_offsets = tl.arange(0, DIM_ROPE)
    rope_input_offsets = (
        token_offsets[:, None] * k_rope_stride_0 + rope_offsets[None, :]
    )
    rope_output_offsets = (
        token_offsets[:, None] * output_rope_stride_0 + rope_offsets[None, :]
    )
    rope = tl.load(
        k_rope_ptr + rope_input_offsets,
        mask=token_mask[:, None],
        other=0.0,
    )
    tl.store(
        output_rope_ptr + rope_output_offsets,
        rope,
        mask=token_mask[:, None],
    )


def _allocate_outputs(
    k_nope: torch.Tensor, k_rope: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = k_nope.shape[0]
    nope_part = torch.empty(
        (num_tokens, NOPE_PART_BYTES), dtype=torch.uint8, device=k_nope.device
    )
    rope_part = torch.empty(
        (num_tokens, ROPE_PART_BYTES), dtype=torch.uint8, device=k_rope.device
    )
    nope_q = nope_part[:, :DIM_NOPE].view(_fp8_dtype())
    nope_s = nope_part[:, DIM_NOPE:].view(torch.float32)
    rope = rope_part.view(torch.bfloat16)
    return nope_part, rope_part, nope_q, nope_s, rope


def _launch_single_token(
    k_nope: torch.Tensor, k_rope: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    nope_part, rope_part, nope_q, nope_s, rope = _allocate_outputs(k_nope, k_rope)
    num_tokens = k_nope.shape[0]
    if num_tokens:
        _quantize_k_cache_fast_kernel[(num_tokens, NUM_NOPE_GROUPS + 1)](
            nope_q,
            nope_s,
            rope,
            k_nope,
            k_rope,
            nope_q.stride(0),
            nope_s.stride(0),
            rope.stride(0),
            k_nope.stride(0),
            k_rope.stride(0),
            NUM_NOPE_BLOCKS=NUM_NOPE_GROUPS,
            GROUP_SIZE=GROUP_SIZE,
            DIM_NOPE=DIM_NOPE,
            DIM_ROPE=DIM_ROPE,
            FP8_MIN=torch.finfo(_fp8_dtype()).min,
            FP8_MAX=torch.finfo(_fp8_dtype()).max,
        )
    return nope_part.unsqueeze(1), rope_part.unsqueeze(1)


def _launch_multi_token(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    block_m: Optional[int],
    num_warps: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    nope_part, rope_part, nope_q, nope_s, rope = _allocate_outputs(k_nope, k_rope)
    num_tokens = k_nope.shape[0]
    block_m = block_m if block_m is not None else (4 if num_tokens <= 4096 else 32)
    num_warps = num_warps if num_warps is not None else 1
    if num_tokens:
        grid = (triton.cdiv(num_tokens, block_m), NUM_NOPE_GROUPS)
        _quantize_k_cache_fast_kernel_optimized[grid](
            nope_q,
            nope_s,
            rope,
            k_nope,
            k_rope,
            num_tokens,
            nope_q.stride(0),
            nope_s.stride(0),
            rope.stride(0),
            k_nope.stride(0),
            k_rope.stride(0),
            BLOCK_M=block_m,
            GROUP_SIZE=GROUP_SIZE,
            DIM_ROPE=DIM_ROPE,
            FP8_MIN=torch.finfo(_fp8_dtype()).min,
            FP8_MAX=torch.finfo(_fp8_dtype()).max,
            num_warps=num_warps,
        )
    return nope_part.unsqueeze(1), rope_part.unsqueeze(1)


def _launch_all_groups(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    block_m: int = 4,
    num_warps: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    nope_part, rope_part, nope_q, nope_s, rope = _allocate_outputs(k_nope, k_rope)
    num_tokens = k_nope.shape[0]
    if num_tokens:
        _quantize_k_cache_all_groups_kernel[(triton.cdiv(num_tokens, block_m),)](
            nope_q,
            nope_s,
            rope,
            k_nope,
            k_rope,
            num_tokens,
            nope_q.stride(0),
            nope_s.stride(0),
            rope.stride(0),
            k_nope.stride(0),
            k_rope.stride(0),
            BLOCK_M=block_m,
            NUM_GROUPS=NUM_NOPE_GROUPS,
            GROUP_SIZE=GROUP_SIZE,
            DIM_NOPE=DIM_NOPE,
            DIM_ROPE=DIM_ROPE,
            FP8_MIN=torch.finfo(_fp8_dtype()).min,
            FP8_MAX=torch.finfo(_fp8_dtype()).max,
            num_warps=num_warps,
        )
    return nope_part.unsqueeze(1), rope_part.unsqueeze(1)


def _normalize_and_validate_inputs(
    k_nope: torch.Tensor, k_rope: torch.Tensor, tile_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for quantize_k_cache_separate_optimized")

    k_nope = k_nope.squeeze(1) if k_nope.ndim == 3 else k_nope
    k_rope = k_rope.squeeze(1) if k_rope.ndim == 3 else k_rope

    if k_nope.ndim != 2 or k_nope.shape[1] != DIM_NOPE:
        raise ValueError(
            f"Expected k_nope shape [N, {DIM_NOPE}] or [N, 1, {DIM_NOPE}], "
            f"got {tuple(k_nope.shape)}"
        )
    if k_rope.ndim != 2 or k_rope.shape[1] != DIM_ROPE:
        raise ValueError(
            f"Expected k_rope shape [N, {DIM_ROPE}] or [N, 1, {DIM_ROPE}], "
            f"got {tuple(k_rope.shape)}"
        )
    if k_nope.shape[0] != k_rope.shape[0]:
        raise ValueError(
            "k_nope and k_rope must have the same number of tokens, got "
            f"{k_nope.shape[0]} and {k_rope.shape[0]}"
        )
    if k_nope.device != k_rope.device:
        raise ValueError("k_nope and k_rope must be on the same device")
    if not k_nope.is_cuda:
        raise ValueError("k_nope and k_rope must be CUDA tensors")
    if k_nope.dtype != torch.bfloat16 or k_rope.dtype != torch.bfloat16:
        raise TypeError("k_nope and k_rope must both have dtype torch.bfloat16")
    if tile_size != GROUP_SIZE:
        raise ValueError(
            f"The optimized kernel requires tile_size={GROUP_SIZE}, got {tile_size}"
        )
    return k_nope.contiguous(), k_rope.contiguous()


def quantize_k_cache_separate_optimized(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    tile_size: int = GROUP_SIZE,
    block_m: Optional[int] = None,
    num_warps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize the GLM-4.2 MLA K cache into separate NOPE and RoPE buffers.

    Inputs have shapes ``[N, 512]``/``[N, 64]`` or include a singleton head
    dimension. The returned uint8 tensors have shapes ``[N, 1, 528]`` and
    ``[N, 1, 128]``. The NOPE buffer stores 512 FP8 bytes followed by four
    FP32 scales; the RoPE buffer stores the original 64 BF16 values as bytes.
    """
    if block_m is not None and block_m not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError(f"Unsupported block_m={block_m}")
    if num_warps is not None and num_warps not in (1, 2, 4):
        raise ValueError(f"Unsupported num_warps={num_warps}")

    k_nope, k_rope = _normalize_and_validate_inputs(k_nope, k_rope, tile_size)
    num_tokens = k_nope.shape[0]

    if block_m is None and num_warps is None and num_tokens <= 64:
        return _launch_single_token(k_nope, k_rope)
    if block_m is None and num_warps is None and num_tokens >= 8192:
        return _launch_all_groups(k_nope, k_rope)
    return _launch_multi_token(k_nope, k_rope, block_m, num_warps)


__all__ = ["quantize_k_cache_separate_optimized"]
