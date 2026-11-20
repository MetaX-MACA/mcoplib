"""GLM FP8 MLA cache gather/dequantization with flattened physical row IDs."""

from typing import Optional
from functools import lru_cache

import torch

from .triton_utils import HAS_TRITON, tl, triton
try:
    from .triton_dequantize_k_cache_c600u import (
        _dequantize_k_cache_c600u_rows2_kernel,
        _dequantize_k_cache_c600u_pairtiles_kernel,
    )
except ImportError:
    _dequantize_k_cache_c600u_rows2_kernel = None
    _dequantize_k_cache_c600u_pairtiles_kernel = None

try:
    from .triton_dequantize_k_cache_gluon import (
        _dequantize_k_cache_gluon_kernel, _dequantize_k_cache_gluon_masked_kernel,
        _dequantize_k_cache_gluon_masked_adaptive_kernel,
        _dequantize_k_cache_gluon_masked_prefetch_kernel,
        _dequantize_k_cache_gluon_rows2_kernel,
    )
except ImportError:
    _dequantize_k_cache_gluon_kernel = None
    _dequantize_k_cache_gluon_masked_kernel = None
    _dequantize_k_cache_gluon_masked_adaptive_kernel = None
    _dequantize_k_cache_gluon_masked_prefetch_kernel = None
    _dequantize_k_cache_gluon_rows2_kernel = None


DIM_NOPE = 512
DIM_ROPE = 64
GROUP_SIZE = 128
NUM_GROUPS = DIM_NOPE // GROUP_SIZE
NOPE_PART_BYTES = DIM_NOPE + NUM_GROUPS * 4


@triton.jit
def _dequantize_k_cache_paged_optimized_kernel(
    output_ptr,
    input_nope_q_ptr,
    input_nope_s_ptr,
    input_rope_ptr,
    page_table_ptr,
    row_mask_ptr,
    OUTPUT_STRIDE: tl.constexpr,
    INPUT_NOPE_Q_STRIDE: tl.constexpr,
    INPUT_NOPE_S_STRIDE: tl.constexpr,
    INPUT_ROPE_STRIDE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    USE_INT64_INPUT_INDEX: tl.constexpr,
    USE_INT64_OUTPUT_INDEX: tl.constexpr,
    HAS_ROW_MASK: tl.constexpr,
    EXACT_DECODE: tl.constexpr = False,
):
    token_id = tl.program_id(0)
    if HAS_ROW_MASK:
        if tl.load(row_mask_ptr + token_id) == 0:
            return

    # Masked rows return before reading even their physical row ID.
    paged_token_id = tl.load(page_table_ptr + token_id)
    if USE_INT64_INPUT_INDEX:
        paged_token_id = paged_token_id.to(tl.int64)
    else:
        paged_token_id = paged_token_id.to(tl.int32)
    output_token_id = token_id
    if USE_INT64_OUTPUT_INDEX:
        output_token_id = token_id.to(tl.int64)

    offsets = tl.arange(0, DIM_NOPE)
    if EXACT_DECODE:
        byte_ptr = input_nope_q_ptr.to(tl.pointer_type(tl.uint8))
        bits = tl.load(byte_ptr + paged_token_id * INPUT_NOPE_Q_STRIDE + offsets)
        quantized = _e4m3_to_fp32_fma(bits)
    else:
        quantized = tl.load(
            input_nope_q_ptr + paged_token_id * INPUT_NOPE_Q_STRIDE + offsets
        ).to(tl.float32)
    scales = tl.load(
        input_nope_s_ptr
        + paged_token_id * INPUT_NOPE_S_STRIDE
        + offsets // GROUP_SIZE
    )
    tl.store(
        output_ptr + output_token_id * OUTPUT_STRIDE + offsets,
        quantized * scales,
    )
    if DIM_ROPE > 0:
        rope_offsets = tl.arange(0, DIM_ROPE)
        rope = tl.load(
            input_rope_ptr + paged_token_id * INPUT_ROPE_STRIDE + rope_offsets
        )
        tl.store(
            output_ptr + output_token_id * OUTPUT_STRIDE + DIM_NOPE + rope_offsets,
            rope,
        )


@lru_cache(maxsize=None)
def _is_c500(device_index):
    return torch.cuda.get_device_properties(device_index).name.upper().startswith("METAX C500")


@lru_cache(maxsize=None)
def _is_c600u(device_index):
    return torch.cuda.get_device_properties(device_index).name.upper() == "METAX C600-U"


@triton.jit
def _e4m3_to_fp32_fma(bits):
    code = bits.to(tl.uint32)
    magnitude = code & 0x7f
    sign = (code & 0x80) << 24
    normal = ((magnitude << 20) + (120 << 23)).to(tl.float32, bitcast=True)
    # Exact E4M3FN exponent-zero correction, including signed zero below.
    subnormal = tl.fma(normal, 2.0, -0.015625)
    value = tl.where(magnitude < 8, subnormal, normal)
    result = value.to(tl.uint32, bitcast=True) | sign
    result = tl.where(magnitude == 0x7f, 0x7fc00000 | sign, result)
    return result.to(tl.float32, bitcast=True)


@triton.jit
def _e4m3_to_fp32(bits):
    # E4M3FN normals map exactly by rebiasing exponent 7 -> 127.
    # Exponent-zero codes are integer multiples of 2^-9; 0x7f is NaN.
    code = bits.to(tl.uint32)
    magnitude = code & 0x7f
    sign = (code & 0x80) << 24
    normal = ((magnitude << 20) + (120 << 23)) | sign
    subnormal = (magnitude.to(tl.float32) * 0.001953125).to(tl.uint32, bitcast=True) | sign
    result = tl.where(magnitude < 8, subnormal, normal)
    result = tl.where(magnitude == 0x7f, 0x7fc00000 | sign, result)
    return result.to(tl.float32, bitcast=True)


@triton.jit
def _dequantize_k_cache_rows_kernel(
    output_ptr, input_nope_q_ptr, input_nope_s_ptr, input_rope_ptr, page_table_ptr,
    M, OUTPUT_STRIDE: tl.constexpr, INPUT_NOPE_Q_STRIDE: tl.constexpr,
    INPUT_NOPE_S_STRIDE: tl.constexpr, INPUT_ROPE_STRIDE: tl.constexpr,
    DIM_ROPE: tl.constexpr, USE_INT64_INPUT_INDEX: tl.constexpr,
    USE_INT64_OUTPUT_INDEX: tl.constexpr, BLOCK_M: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    valid = rows < M
    ids = tl.load(page_table_ptr + rows, valid, 0)
    ids = ids.to(tl.int64) if USE_INT64_INPUT_INDEX else ids.to(tl.int32)
    output_rows = rows.to(tl.int64) if USE_INT64_OUTPUT_INDEX else rows
    groups = tl.arange(0, 4)
    elements = tl.arange(0, 128)
    offsets = groups[:, None] * 128 + elements[None, :]
    byte_ptr = input_nope_q_ptr.to(tl.pointer_type(tl.uint8))
    bits = tl.load(byte_ptr + ids[:, None, None] * INPUT_NOPE_Q_STRIDE
                   + offsets[None, :, :], valid[:, None, None], 0)
    values = _e4m3_to_fp32_fma(bits)
    scales = tl.load(input_nope_s_ptr + ids[:, None] * INPUT_NOPE_S_STRIDE
                     + groups[None, :], valid[:, None], 0)
    tl.store(output_ptr + output_rows[:, None, None] * OUTPUT_STRIDE
             + offsets[None, :, :], values * scales[:, :, None], valid[:, None, None])
    if DIM_ROPE > 0:
        rope_offsets = tl.arange(0, DIM_ROPE)
        rope = tl.load(input_rope_ptr + ids[:, None] * INPUT_ROPE_STRIDE
                       + rope_offsets[None, :], valid[:, None], 0)
        tl.store(output_ptr + output_rows[:, None] * OUTPUT_STRIDE + 512
                 + rope_offsets[None, :], rope, valid[:, None])


def dequantize_k_cache_paged_optimized(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    group_size: int = GROUP_SIZE,
    row_mask: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather packed FP8 rows and return ``[M, 1, 512 + rope_dim]`` BF16.

    Cache rows contain 512 E4M3FN bytes, four FP32 scales, and optionally
    64 BF16 RoPE values, for a total of 528 or 656 bytes. Leading cache
    dimensions are flattened without copying. This is the GLM row layout;
    the DSV4 page layout and its UE8M0 scales use a separate operator.

    The contiguous int32/int64 index vector contains physical row IDs, with
    arbitrary order and repetitions. Its length M determines the output
    length, including when M exceeds the cache capacity through sharing.
    Active IDs must be in ``[0, number_of_cache_rows)``. No synchronous
    device-side index scan is performed by this wrapper.

    A zero row_mask entry skips all reads/writes for that output row. Such
    rows retain their previous values with ``out=`` and are unspecified in
    a newly allocated result. Nonzero real numeric mask values enable a
    row. ``out`` must be contiguous BF16 on the cache device, with the exact
    output shape, and use separate storage from all inputs.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for dequantize_k_cache_paged_optimized")
    if not quant_k_cache.is_cuda:
        raise ValueError("quant_k_cache must be a CUDA tensor")
    if quant_k_cache.dtype != torch.float8_e4m3fn:
        raise TypeError("quant_k_cache must have dtype torch.float8_e4m3fn")
    if quant_k_cache.ndim == 0 or quant_k_cache.shape[-1] not in (528, 656):
        raise ValueError("quant_k_cache rows must contain 528 or 656 bytes")
    if group_size != GROUP_SIZE:
        raise ValueError("group_size must be 128")
    if quant_k_cache.stride(-1) != 1:
        raise ValueError("quant_k_cache rows must have unit column stride")

    dim_quant = quant_k_cache.shape[-1]
    dim_rope = (dim_quant - NOPE_PART_BYTES) // 2
    try:
        cache = quant_k_cache.view(-1, dim_quant)
    except RuntimeError as error:
        raise ValueError("cache leading dimensions must flatten without copying") from error
    if cache.stride(0) % 4 or cache.data_ptr() % 4:
        raise ValueError("cache rows and their FP32 scales must be 4-byte aligned")

    indices = page_table_1_flattened
    if indices.device != cache.device:
        raise ValueError("page_table_1_flattened must be on the cache device")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("page_table_1_flattened must have dtype int32 or int64")
    if indices.ndim != 1 or not indices.is_contiguous():
        raise ValueError("page_table_1_flattened must be a contiguous 1D tensor")
    num_tokens = indices.numel()
    if row_mask is not None:
        if row_mask.device != cache.device:
            raise ValueError("row_mask must be on the cache device")
        if row_mask.ndim != 1 or row_mask.numel() != num_tokens or not row_mask.is_contiguous():
            raise ValueError("row_mask must be contiguous with shape [M]")
        if row_mask.is_complex() or row_mask.is_quantized:
            raise TypeError("row_mask must have a real numeric or bool dtype")

    output_shape = (num_tokens, 1, DIM_NOPE + dim_rope)
    if out is None:
        out = torch.empty(output_shape, dtype=torch.bfloat16, device=cache.device)
    else:
        if out.device != cache.device:
            raise ValueError("out must be on the cache device")
        if out.dtype != torch.bfloat16:
            raise TypeError("out must have dtype torch.bfloat16")
        if tuple(out.shape) != output_shape or not out.is_contiguous():
            raise ValueError(f"out must be contiguous with shape {output_shape}")
        if num_tokens:
            inputs = [cache, indices] + ([row_mask] if row_mask is not None else [])
            if any(out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in inputs):
                raise ValueError("out must use separate storage from all inputs")
    if num_tokens == 0:
        return out

    input_nope_q = cache[:, :DIM_NOPE]
    input_nope_s = cache[:, DIM_NOPE:NOPE_PART_BYTES].view(torch.float32)
    input_rope = cache[:, NOPE_PART_BYTES:].view(torch.bfloat16)
    row_stride = cache.stride(0)
    use_int64_input = row_stride != 0 and cache.shape[0] > (2**31 - 1) // row_stride
    use_int64_output = num_tokens > (2**31 - 1) // (DIM_NOPE + dim_rope)
    if (num_tokens >= 2048 and row_mask is not None and _is_c500(cache.device.index)
            and _dequantize_k_cache_gluon_masked_kernel is not None):
        masked_kernel = _dequantize_k_cache_gluon_masked_kernel
        if num_tokens >= 8192 and _dequantize_k_cache_gluon_masked_adaptive_kernel is not None:
            masked_kernel = _dequantize_k_cache_gluon_masked_adaptive_kernel
            if (num_tokens >= 16384 and indices.dtype == torch.int32
                    and _dequantize_k_cache_gluon_masked_prefetch_kernel is not None):
                masked_kernel = _dequantize_k_cache_gluon_masked_prefetch_kernel
        masked_kernel[(triton.cdiv(num_tokens, 2),)](
            out, input_nope_q, input_nope_s, input_rope, indices, row_mask, num_tokens,
            OUTPUT_STRIDE=out.stride(0), INPUT_NOPE_Q_STRIDE=row_stride,
            INPUT_NOPE_S_STRIDE=input_nope_s.stride(0),
            INPUT_ROPE_STRIDE=input_rope.stride(0), DIM_ROPE=dim_rope,
            USE_INT64_INPUT_INDEX=use_int64_input,
            USE_INT64_OUTPUT_INDEX=use_int64_output, BLOCK_M=2,
            num_warps=1, num_stages=1,
        )
        return out
    if num_tokens >= 8192 and row_mask is None and _is_c500(cache.device.index):
        kernel = _dequantize_k_cache_rows_kernel
        block_m = 4
        if num_tokens < 65536 and _dequantize_k_cache_gluon_rows2_kernel is not None:
            kernel = _dequantize_k_cache_gluon_rows2_kernel
            block_m = 2
        elif num_tokens >= 12288 and _dequantize_k_cache_gluon_kernel is not None:
            kernel = _dequantize_k_cache_gluon_kernel
        kernel[(triton.cdiv(num_tokens, block_m),)](
            out, input_nope_q, input_nope_s, input_rope, indices, num_tokens,
            OUTPUT_STRIDE=out.stride(0), INPUT_NOPE_Q_STRIDE=row_stride,
            INPUT_NOPE_S_STRIDE=input_nope_s.stride(0),
            INPUT_ROPE_STRIDE=input_rope.stride(0), DIM_ROPE=dim_rope,
            USE_INT64_INPUT_INDEX=use_int64_input,
            USE_INT64_OUTPUT_INDEX=use_int64_output, BLOCK_M=block_m,
            num_warps=1, num_stages=1,
        )
        return out
    if (num_tokens >= 32768 and row_mask is None and _is_c600u(cache.device.index)
            and _dequantize_k_cache_c600u_rows2_kernel is not None):
        kernel = _dequantize_k_cache_c600u_rows2_kernel
        block_m = 2
        # Both two-row output widths are multiples of 256 bytes.
        if out.data_ptr() % 256 == 0 and _dequantize_k_cache_c600u_pairtiles_kernel is not None:
            kernel = _dequantize_k_cache_c600u_pairtiles_kernel
            block_m = 4
        kernel[(triton.cdiv(num_tokens, block_m),)](
            out, input_nope_q, input_nope_s, input_rope, indices, num_tokens,
            OUTPUT_STRIDE=out.stride(0), INPUT_NOPE_Q_STRIDE=row_stride,
            INPUT_NOPE_S_STRIDE=input_nope_s.stride(0),
            INPUT_ROPE_STRIDE=input_rope.stride(0), DIM_ROPE=dim_rope,
            USE_INT64_INPUT_INDEX=use_int64_input,
            USE_INT64_OUTPUT_INDEX=use_int64_output,
            num_warps=1, num_stages=1,
        )
        return out
    _dequantize_k_cache_paged_optimized_kernel[(num_tokens,)](
        out, input_nope_q, input_nope_s, input_rope, indices, row_mask,
        OUTPUT_STRIDE=out.stride(0),
        INPUT_NOPE_Q_STRIDE=row_stride,
        INPUT_NOPE_S_STRIDE=input_nope_s.stride(0),
        INPUT_ROPE_STRIDE=input_rope.stride(0),
        DIM_NOPE=DIM_NOPE,
        GROUP_SIZE=GROUP_SIZE,
        DIM_ROPE=dim_rope,
        USE_INT64_INPUT_INDEX=use_int64_input,
        USE_INT64_OUTPUT_INDEX=use_int64_output,
        HAS_ROW_MASK=row_mask is not None,
        EXACT_DECODE=num_tokens >= 1024 and row_mask is None and _is_c500(cache.device.index),
        num_warps=1,
        num_stages=1,
    )
    return out


__all__ = ["dequantize_k_cache_paged_optimized"]
