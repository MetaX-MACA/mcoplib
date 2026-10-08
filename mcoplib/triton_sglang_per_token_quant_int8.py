# SPDX-License-Identifier: Apache-2.0
"""SGLang-compatible Triton per-token INT8 quantization kernels."""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _per_token_quant_int8(
    x_ptr,
    xq_ptr,
    scale_ptr,
    x_sum_ptr,
    stride_x,
    stride_xq,
    N,
    CAL_SUM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(
        x_ptr + row_id * stride_x + cols, mask=mask, other=0.0
    ).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / 127
    x_q = x * (127 / absmax)
    x_q = tl.extra.cuda.libdevice.round(x_q).to(tl.int8)
    if CAL_SUM:
        x_sum = tl.sum(x, axis=0)
        tl.store(
            x_sum_ptr + row_id,
            x_sum.to(x_sum_ptr.dtype.element_ty),
        )

    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x.to(scale_ptr.dtype.element_ty))


@triton.jit
def _per_token_quant_int8_multirow(
    x_ptr,
    xq_ptr,
    scale_ptr,
    x_sum_ptr,
    stride_x,
    stride_xq,
    n_rows,
    n_cols,
    CAL_SUM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)

    x_offsets = rows[:, None] * stride_x + cols[None, :]
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10)
    scale_x = absmax / 127.0
    x_q = x * (127.0 / absmax[:, None])
    x_q = tl.extra.cuda.libdevice.round(x_q).to(tl.int8)

    if CAL_SUM:
        x_sum = tl.sum(x, axis=1)
        tl.store(
            x_sum_ptr + rows,
            x_sum.to(x_sum_ptr.dtype.element_ty),
            mask=rows < n_rows,
        )

    xq_offsets = rows[:, None] * stride_xq + cols[None, :]
    tl.store(xq_ptr + xq_offsets, x_q, mask=mask)
    tl.store(
        scale_ptr + rows,
        scale_x.to(scale_ptr.dtype.element_ty),
        mask=rows < n_rows,
    )


# Frozen source-identical baselines used by the optimization harness.
_per_token_quant_int8_baseline = _per_token_quant_int8
_per_token_quant_int8_multirow_baseline = _per_token_quant_int8_multirow


@triton.jit
def _per_token_quant_int8_inline_round(
    x_ptr,
    xq_ptr,
    scale_ptr,
    x_sum_ptr,
    stride_x,
    stride_xq,
    N,
    CAL_SUM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(
        x_ptr + row_id * stride_x + cols, mask=mask, other=0.0
    ).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / 127
    x_q = x * (127 / absmax)
    x_q = (
        x_q + tl.where(x_q >= 0.0, 0.5, -0.5)
    ).to(tl.int8)
    if CAL_SUM:
        x_sum = tl.sum(x, axis=0)
        tl.store(
            x_sum_ptr + row_id,
            x_sum.to(x_sum_ptr.dtype.element_ty),
        )
    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x.to(scale_ptr.dtype.element_ty))


@triton.jit
def _per_token_quant_int8_multirow_inline_round(
    x_ptr,
    xq_ptr,
    scale_ptr,
    x_sum_ptr,
    stride_x,
    stride_xq,
    n_rows,
    n_cols,
    CAL_SUM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_start = tl.program_id(0) * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
    x_offsets = rows[:, None] * stride_x + cols[None, :]
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10)
    scale_x = absmax / 127.0
    x_q = x * (127.0 / absmax[:, None])
    x_q = (
        x_q + tl.where(x_q >= 0.0, 0.5, -0.5)
    ).to(tl.int8)
    if CAL_SUM:
        x_sum = tl.sum(x, axis=1)
        tl.store(
            x_sum_ptr + rows,
            x_sum.to(x_sum_ptr.dtype.element_ty),
            mask=rows < n_rows,
        )
    xq_offsets = rows[:, None] * stride_xq + cols[None, :]
    tl.store(xq_ptr + xq_offsets, x_q, mask=mask)
    tl.store(
        scale_ptr + rows,
        scale_x.to(scale_ptr.dtype.element_ty),
        mask=rows < n_rows,
    )


@triton.jit
def _per_token_quant_int8_multirow_persistent(
    x_ptr,
    xq_ptr,
    scale_ptr,
    x_sum_ptr,
    stride_x,
    stride_xq,
    n_rows,
    n_cols,
    CAL_SUM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    program_id = tl.program_id(0)
    program_count = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_N)
    for row_start in tl.range(
        program_id * BLOCK_M,
        n_rows,
        program_count * BLOCK_M,
        loop_unroll_factor=1,
        disable_licm=True,
    ):
        rows = row_start + tl.arange(0, BLOCK_M)
        mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
        x_offsets = rows[:, None] * stride_x + cols[None, :]
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-10)
        scale_x = absmax / 127.0
        x_q = x * (127.0 / absmax[:, None])
        x_q = (
            x_q + tl.where(x_q >= 0.0, 0.5, -0.5)
        ).to(tl.int8)
        if CAL_SUM:
            x_sum = tl.sum(x, axis=1)
            tl.store(
                x_sum_ptr + rows,
                x_sum.to(x_sum_ptr.dtype.element_ty),
                mask=rows < n_rows,
            )
        xq_offsets = rows[:, None] * stride_xq + cols[None, :]
        tl.store(xq_ptr + xq_offsets, x_q, mask=mask)
        tl.store(
            scale_ptr + rows,
            scale_x.to(scale_ptr.dtype.element_ty),
            mask=rows < n_rows,
        )


def get_per_token_quant_rows_per_program(n_rows: int, n_cols: int) -> int:
    block_n = triton.next_power_of_2(n_cols)
    max_rows_by_work = max(1, 4096 // block_n)
    rows_by_grid = max(1, n_rows // 512)
    rows_by_grid = 1 << (rows_by_grid.bit_length() - 1)
    return min(16, max_rows_by_work, rows_by_grid)


def _get_k384_persistent_program_count(n_rows: int) -> int:
    # C600-U: cap the grid at 32/48/64 programs per MP for 28 MPs.
    if n_rows >= 196608:
        return 1792
    if n_rows >= 98304:
        return 1344
    return 896


def launch_per_token_quant_int8(
    x: torch.Tensor,
    x_q: torch.Tensor,
    scales: torch.Tensor,
    x_sum: Optional[torch.Tensor] = None,
    *,
    rows_per_program: Optional[int] = None,
    num_warps: Optional[int] = None,
    use_baseline: bool = False,
) -> None:
    """Launch into preallocated outputs."""
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got shape {tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x_q.shape != x.shape or x_q.dtype != torch.int8:
        raise ValueError("x_q must be contiguous int8 with the same shape as x")
    if not x_q.is_contiguous():
        raise ValueError("x_q must be contiguous")
    if scales.shape != (x.shape[0], 1):
        raise ValueError("scales must have shape [M, 1]")
    if x_sum is not None and x_sum.shape != (x.shape[0],):
        raise ValueError("x_sum must have shape [M]")

    M, K = x.shape
    block_n = triton.next_power_of_2(K)
    if rows_per_program is None:
        rows_per_program = get_per_token_quant_rows_per_program(M, K)
    if rows_per_program <= 0 or rows_per_program & (rows_per_program - 1):
        raise ValueError("rows_per_program must be a positive power of two")
    if num_warps is None:
        default_warps = min(
            max(block_n * rows_per_program // 256, 1), 8
        )
        num_warps = (
            4
            if not use_baseline and K in (384, 7168)
            else default_warps
        )

    if rows_per_program > 1:
        if not use_baseline and K == 384:
            grid_size = triton.cdiv(M, rows_per_program)
            if (
                M >= 48000
                and rows_per_program == 8
                and num_warps == 4
            ):
                grid_size = min(
                    grid_size,
                    _get_k384_persistent_program_count(M),
                )
                _per_token_quant_int8_multirow_persistent[(grid_size,)](
                    x,
                    x_q,
                    scales,
                    x_sum,
                    stride_x=x.stride(0),
                    stride_xq=x_q.stride(0),
                    n_rows=M,
                    n_cols=K,
                    CAL_SUM=x_sum is not None,
                    BLOCK_M=rows_per_program,
                    BLOCK_N=block_n,
                    num_warps=num_warps,
                    num_stages=1,
                )
                return
            _per_token_quant_int8_multirow_inline_round[
                (grid_size,)
            ](
                x,
                x_q,
                scales,
                x_sum,
                stride_x=x.stride(0),
                stride_xq=x_q.stride(0),
                n_rows=M,
                n_cols=K,
                CAL_SUM=x_sum is not None,
                BLOCK_M=rows_per_program,
                BLOCK_N=block_n,
                num_warps=num_warps,
                num_stages=1,
            )
            return
        kernel = (
            _per_token_quant_int8_multirow_baseline
            if use_baseline
            else _per_token_quant_int8_multirow
        )
        kernel[(triton.cdiv(M, rows_per_program),)](
            x,
            x_q,
            scales,
            x_sum,
            stride_x=x.stride(0),
            stride_xq=x_q.stride(0),
            n_rows=M,
            n_cols=K,
            CAL_SUM=x_sum is not None,
            BLOCK_M=rows_per_program,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        if not use_baseline and K == 7168:
            _per_token_quant_int8_inline_round[(M,)](
                x,
                x_q,
                scales,
                x_sum,
                stride_x=x.stride(0),
                stride_xq=x_q.stride(0),
                N=K,
                CAL_SUM=x_sum is not None,
                BLOCK=block_n,
                num_warps=num_warps,
                num_stages=1,
            )
            return
        kernel = (
            _per_token_quant_int8_baseline
            if use_baseline
            else _per_token_quant_int8
        )
        kernel[(M,)](
            x,
            x_q,
            scales,
            x_sum,
            stride_x=x.stride(0),
            stride_xq=x_q.stride(0),
            N=K,
            CAL_SUM=x_sum is not None,
            BLOCK=block_n,
            num_warps=num_warps,
            num_stages=1,
        )


def per_token_quant_int8(
    x: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    cal_sum: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Allocate outputs and run SGLang-compatible per-token INT8 quantization."""
    x_2d = x.reshape(-1, x.shape[-1])
    x_q = torch.empty_like(x_2d, dtype=torch.int8)
    scales = torch.empty(
        (x_2d.shape[0], 1), device=x.device, dtype=scale_dtype
    )
    x_sum = (
        torch.empty(x_2d.shape[0], device=x.device, dtype=x.dtype)
        if cal_sum
        else None
    )
    launch_per_token_quant_int8(x_2d, x_q, scales, x_sum)
    output_shape = x.shape
    scale_shape = x.shape[:-1] + (1,)
    if cal_sum:
        return (
            x_q.reshape(output_shape),
            scales.reshape(scale_shape),
            x_sum.reshape(x.shape[:-1]),
        )
    return x_q.reshape(output_shape), scales.reshape(scale_shape)
