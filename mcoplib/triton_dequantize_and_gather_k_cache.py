
import torch
import triton
import triton.language as tl
from functools import lru_cache
import os
import inspect
from typing import Any, Callable, Dict, Literal, Optional, Tuple

V4_BYTES_PER_TOKEN = 584
V41_BYTES_PER_TOKEN = 528
V41_QUANT_BLOCK = 32
V41_NUM_SCALES = 512 // V41_QUANT_BLOCK  # 16
V41_NVFP4_BYTES_PER_TOKEN = 288
V41_NVFP4_QUANT_BLOCK = 16
V41_NVFP4_NUM_SCALES = 512 // V41_NVFP4_QUANT_BLOCK  # 32

@triton.jit
def _dequantize_and_gather_k_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    # Constants
    max_blocks_per_seq: tl.constexpr,
    fp8_dim: tl.constexpr,  # 448
    bf16_dim: tl.constexpr,  # 64
    scale_dim: tl.constexpr,  # 8
    quant_block: tl.constexpr,  # 64 (quantization block size)
    cache_block_size: tl.constexpr,  # 64 or 128 (paged cache block size)
    token_data_size: tl.constexpr,  # 576 bytes per token data
    block_stride: tl.constexpr,  # total bytes per block (padded) int32
    output_dim: tl.constexpr,  # 512
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,  # 7 real blocks
    use_fnuz: tl.constexpr = False,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        # Gather all tokens
        gather_len = seq_len
    start_pos = seq_len - gather_len

    for i in range(worker_id, gather_len, num_workers):
        # Calculate the actual token index in the sequence
        pos = start_pos + i

        # Calculate which block and position within block
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        # Get physical block index from block table
        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)  # int32

        # int64: physical_block_idx * block_stride can exceed 2^31 with many
        # KV-cache blocks (e.g. >= 57K at block_stride ~37K).
        cache_block_ptr = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride

        # Token data pointer
        token_data_ptr = cache_block_ptr + pos_in_block * token_data_size

        # Scale pointer: after all token data
        token_scale_ptr = (
            cache_block_ptr
            + cache_block_size * token_data_size
            + pos_in_block * scale_dim
        )

        # Token data layout: [0:448] fp8, [448:576] bf16
        token_fp8_ptr = token_data_ptr
        token_bf16_ptr = token_data_ptr + fp8_dim

        # Output pointer for this token (flattened)
        output_row_ptr = out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1

        # ========== Dequantize FP8 portion using UE8M0 ==========
        for qblock_idx in tl.static_range(n_quant_blocks):
            qblock_start = qblock_idx * quant_block

            if qblock_start < fp8_dim:
                offsets = qblock_start + tl.arange(0, quant_block)
                mask = offsets < fp8_dim

                # Load quantized fp8 values (stored as uint8)
                x_uint8 = tl.load(token_fp8_ptr + offsets, mask=mask, other=0)

                # Bitcast uint8 back to fp8 (FNUZ on gfx942, OCP elsewhere).
                if use_fnuz:
                    x_fp8 = x_uint8.to(tl.float8e4b8, bitcast=True)
                else:
                    x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)

                # Convert fp8 to float32 for computation
                x_float = x_fp8.to(tl.float32)

                # Load and decode UE8M0 scale
                # UE8M0: scale = 2^(stored_value - 127)
                encoded_scale = tl.load(token_scale_ptr + qblock_idx)
                exponent = encoded_scale.to(tl.float32) - 127.0
                scale = tl.exp2(exponent)

                # Dequantize: bf16_value = fp8_value * scale
                x_dequant = x_float * scale

                # Store as bf16
                tl.store(output_row_ptr + offsets, x_dequant.to(tl.bfloat16), mask=mask)

        # ========== Copy BF16 portion directly ==========
        bf16_output_offset = fp8_dim  # After 448 elements in output

        # Read bf16 from cache
        bf16_cache_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))

        # Process in chunks of 16
        for j in tl.static_range(bf16_dim // 16):
            chunk_offsets = j * 16 + tl.arange(0, 16)
            bf16_vals = tl.load(bf16_cache_ptr + chunk_offsets)
            tl.store(output_row_ptr + bf16_output_offset + chunk_offsets, bf16_vals)



@triton.jit
def _dequantize_and_gather_int8_k_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    # Constants
    max_blocks_per_seq: tl.constexpr,
    scale_dim: tl.constexpr,  # 64
    quant_block: tl.constexpr,  # 32 (quantization block size)
    cache_block_size: tl.constexpr,  # 64 or 128 (paged cache block size)
    token_data_size: tl.constexpr,  # 512 bytes per token data
    block_stride: tl.constexpr,  # total bytes per block (padded) int32
    output_dim: tl.constexpr,  # 512
    n_quant_blocks: tl.constexpr,  # 16 real blocks
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        # Gather all tokens
        gather_len = seq_len
    start_pos = seq_len - gather_len

    for i in range(worker_id, gather_len, num_workers):
        # Calculate the actual token index in the sequence
        pos = start_pos + i

        # Calculate which block and position within block
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        # Get physical block index from block table
        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)  # int32

        # int64: physical_block_idx * block_stride can exceed 2^31 with many
        # KV-cache blocks (e.g. >= 57K at block_stride ~37K).
        cache_block_ptr = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride

        # Token data pointer
        token_data_ptr = cache_block_ptr + pos_in_block * token_data_size

        # Scale pointer: after all token data
        token_scale_ptr = (
            cache_block_ptr
            + cache_block_size * token_data_size
            + pos_in_block * scale_dim
        )
        
        output_row_ptr = out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1
        for qblock_idx in tl.static_range(n_quant_blocks):
            qblock_start = qblock_idx * quant_block
            offsets = qblock_start + tl.arange(0, quant_block)
            mask = offsets < output_dim
            x_int8  = tl.load(token_data_ptr + offsets, mask=mask, other=0).to(tl.int8, bitcast=True)
            x_float = x_int8.to(tl.float32)
            scale_ptr_f32 = token_scale_ptr.to(tl.pointer_type(tl.float32))
            scale = tl.load(scale_ptr_f32 + qblock_idx)
            x_dequant = x_float * scale
            tl.store(output_row_ptr + offsets, x_dequant.to(tl.bfloat16), mask=mask)

def dequantize_and_gather_k_cache_triton(
    # [num_reqs, max_num_tokens, head_size]
    out: torch.Tensor,
    # [num_blocks, block_size, head_bytes]
    k_cache: torch.Tensor,
    # [num_reqs]
    seq_lens: torch.Tensor,
    # [num_reqs]
    gather_lens: torch.Tensor | None,
    # [num_reqs, max_blocks_per_seq]
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
    use_fnuz: bool = False,
    use_int8: bool = False,
) -> None:
    num_reqs = seq_lens.shape[0]
    NUM_WORKERS = 128
    if k_cache.shape[-1] == V41_NVFP4_BYTES_PER_TOKEN:
        return

    if k_cache.shape[-1] == V41_BYTES_PER_TOKEN:
        return

    if use_int8:
        TOKEN_DATA_SIZE = 512
        TOKEN_SCALE_DIM = 64
        QUANT_BLOCK_SIZE = 32
        N_QUANT_BLOCKS = 16

        _dequantize_and_gather_int8_k_kernel[(num_reqs, NUM_WORKERS)](
            out,
            out.stride(0),
            out.stride(1),
            k_cache,
            seq_lens,
            block_table,
            offset,
            gather_lens,
            max_blocks_per_seq=block_table.shape[-1],
            scale_dim=TOKEN_SCALE_DIM,
            quant_block=QUANT_BLOCK_SIZE,
            cache_block_size=block_size,
            token_data_size=TOKEN_DATA_SIZE,
            block_stride=k_cache.stride(0),
            output_dim=512,
            n_quant_blocks=N_QUANT_BLOCKS,
        )

    else:
        TOKEN_FP8_DIM = 448
        TOKEN_BF16_DIM = 64
        TOKEN_SCALE_DIM = 8
        QUANT_BLOCK_SIZE = 64
        FP8_MAX = 448.0
        N_QUANT_BLOCKS = 7
        TOKEN_DATA_SIZE = TOKEN_FP8_DIM + TOKEN_BF16_DIM * 2

        _dequantize_and_gather_k_kernel[(num_reqs, NUM_WORKERS)](
            out,
            out.stride(0),
            out.stride(1),
            k_cache,
            seq_lens,
            block_table,
            offset,
            gather_lens,
            max_blocks_per_seq=block_table.shape[-1],
            fp8_dim=TOKEN_FP8_DIM,
            bf16_dim=TOKEN_BF16_DIM,
            scale_dim=TOKEN_SCALE_DIM,
            quant_block=QUANT_BLOCK_SIZE,
            cache_block_size=block_size,
            token_data_size=TOKEN_DATA_SIZE,
            block_stride=k_cache.stride(0),
            output_dim=512,
            fp8_max=FP8_MAX,
            n_quant_blocks=N_QUANT_BLOCKS,
            use_fnuz=use_fnuz,
        )

    


def dequantize_and_gather_k_cache(
    # [num_reqs, max_num_tokens, head_size]
    out: torch.Tensor,
    # [num_blocks, block_size, head_bytes]
    k_cache: torch.Tensor,
    # [num_reqs]
    seq_lens: torch.Tensor,
    # [num_reqs]
    gather_lens: torch.Tensor | None,
    # [num_reqs, max_blocks_per_seq]
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
    use_fnuz: bool = False,
    use_int8: bool = False,
) -> None:
    """Dequantize and gather a paged DSv4 K cache.

    The record is read off ``k_cache.shape[-1]``; see the module header. Only
    the fp8 records have a CuteDSL gather, so NVFP4 always takes the Triton
    path.

    ``use_fnuz`` MUST match the encoder of the specific cache being read:
    ``False`` for ``compressed_k_cache`` (Triton encoder is OCP everywhere),
    ``current_platform.is_fp8_fnuz()`` for ``swa_k_cache`` (C++ encoder
    writes FNUZ on gfx942 and OCP on gfx950).
    """
    dequantize_and_gather_k_cache_triton(
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset,
        use_fnuz=use_fnuz,
        use_int8=use_int8,
    )
