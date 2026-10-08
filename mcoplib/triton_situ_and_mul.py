"""Triton implementation of the gated SiTU-and-multiply activation."""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def situ_and_mul_baseline_kernel(
    input_ptr,
    output_ptr,
    num_rows,
    half_width: tl.constexpr,
    situ_beta,
    situ_linear_beta,
    BLOCK_SIZE: tl.constexpr,
    APPLY_UP_LIMIT: tl.constexpr,
):
    """Compute SiTU(gate) * up for one input row per Triton program.

    The input row is laid out as ``[gate, up]`` and has width
    ``2 * half_width``. Math is explicitly performed in FP32; ``tl.store``
    converts the result back to the output tensor's element type.
    """
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < half_width

    input_row = input_ptr + row * (2 * half_width)
    gate = tl.load(input_row + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(
        input_row + half_width + offsets, mask=mask, other=0.0
    ).to(tl.float32)

    beta = situ_beta.to(tl.float32)
    # Original implementation: tanh(z) = 2 * sigmoid(2z) - 1.
    scaled_gate = gate / beta
    gate_tanh = 2.0 * tl.sigmoid(2.0 * scaled_gate) - 1.0
    gate = beta * gate_tanh * tl.sigmoid(gate)

    if APPLY_UP_LIMIT:
        linear_beta = situ_linear_beta.to(tl.float32)
        scaled_up = up / linear_beta
        up_tanh = 2.0 * tl.sigmoid(2.0 * scaled_up) - 1.0
        up = linear_beta * up_tanh

    output_row = output_ptr + row * half_width
    tl.store(output_row + offsets, gate * up, mask=mask)


@triton.jit
def situ_and_mul_kernel(
    input_ptr,
    output_ptr,
    total_outputs,
    half_width: tl.constexpr,
    situ_beta,
    situ_beta_inv,
    situ_linear_beta,
    situ_linear_beta_inv,
    BLOCK_SIZE: tl.constexpr,
    APPLY_UP_LIMIT: tl.constexpr,
):
    """C500-optimized SiTU kernel using a linearized output tile.

    Linearizing across rows removes the 25% inactive lanes caused by padding
    D=384 to 512. The launch tile and warp count are tuned for C500.
    """
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_outputs
    row = offsets // half_width
    column = offsets - row * half_width
    input_offsets = row * (2 * half_width) + column

    gate = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(
        input_ptr + input_offsets + half_width, mask=mask, other=0.0
    ).to(tl.float32)

    beta = situ_beta.to(tl.float32)
    beta_inv = situ_beta_inv.to(tl.float32)
    gate_tanh = 2.0 * tl.sigmoid(2.0 * gate * beta_inv) - 1.0
    gate = beta * gate_tanh * tl.sigmoid(gate)

    if APPLY_UP_LIMIT:
        linear_beta = situ_linear_beta.to(tl.float32)
        linear_beta_inv = situ_linear_beta_inv.to(tl.float32)
        up_tanh = 2.0 * tl.sigmoid(2.0 * up * linear_beta_inv) - 1.0
        up = linear_beta * up_tanh

    tl.store(output_ptr + offsets, gate * up, mask=mask)


def _validate_and_allocate(
    x: torch.Tensor,
    situ_beta: float,
    situ_linear_beta: Optional[float],
    out: Optional[torch.Tensor],
) -> tuple[torch.Tensor, int, int]:
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.ndim == 0 or x.shape[-1] % 2 != 0:
        raise ValueError("the last dimension of x must be positive and even")
    if x.shape[-1] == 0:
        raise ValueError("the last dimension of x must be positive and even")
    if situ_beta <= 0:
        raise ValueError("situ_beta must be greater than zero")
    if situ_linear_beta is not None and situ_linear_beta <= 0:
        raise ValueError("situ_linear_beta must be greater than zero")

    half_width = x.shape[-1] // 2
    output_shape = (*x.shape[:-1], half_width)
    if out is None:
        out = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    else:
        if out.shape != output_shape:
            raise ValueError(f"out must have shape {output_shape}, got {out.shape}")
        if out.device != x.device or out.dtype != x.dtype:
            raise ValueError("out must have the same device and dtype as x")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    num_rows = x.numel() // x.shape[-1]
    return out, num_rows, half_width


def situ_and_mul_baseline(
    x: torch.Tensor,
    situ_beta: float = 4.0,
    situ_linear_beta: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the original one-program-per-row kernel for comparison."""
    out, num_rows, half_width = _validate_and_allocate(
        x, situ_beta, situ_linear_beta, out
    )
    if num_rows == 0:
        return out

    block_size = triton.next_power_of_2(half_width)
    if block_size > 65536:
        raise ValueError("half of the last dimension must not exceed 65536")
    num_warps = 4 if block_size <= 2048 else 8

    situ_and_mul_baseline_kernel[(num_rows,)](
        x,
        out,
        num_rows,
        half_width,
        situ_beta,
        1.0 if situ_linear_beta is None else situ_linear_beta,
        BLOCK_SIZE=block_size,
        APPLY_UP_LIMIT=situ_linear_beta is not None,
        num_warps=num_warps,
    )
    return out


def situ_and_mul(
    x: torch.Tensor,
    situ_beta: float = 4.0,
    situ_linear_beta: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    *,
    block_size: int = 1024,
    num_warps: int = 2,
) -> torch.Tensor:
    """Apply the C500-optimized gated SiTU activation.

    ``x`` is shaped ``[..., 2 * D]`` and the result is ``[..., D]``. The
    defaults are selected from measurements on C500 xcore1000.
    """
    out, num_rows, half_width = _validate_and_allocate(
        x, situ_beta, situ_linear_beta, out
    )
    if num_rows == 0:
        return out
    if block_size not in (64, 128, 256, 512, 1024, 2048, 4096):
        raise ValueError(
            "block_size must be one of 64, 128, 256, 512, 1024, 2048, 4096"
        )
    if num_warps not in (1, 2, 4, 8, 16):
        raise ValueError("num_warps must be one of 1, 2, 4, 8, 16")

    total_outputs = num_rows * half_width
    grid = (triton.cdiv(total_outputs, block_size),)
    situ_and_mul_kernel[grid](
        x,
        out,
        total_outputs,
        half_width,
        situ_beta,
        1.0 / situ_beta,
        1.0 if situ_linear_beta is None else situ_linear_beta,
        1.0 if situ_linear_beta is None else 1.0 / situ_linear_beta,
        BLOCK_SIZE=block_size,
        APPLY_UP_LIMIT=situ_linear_beta is not None,
        num_warps=num_warps,
    )
    return out


__all__ = [
    "situ_and_mul",
    "situ_and_mul_kernel",
    "situ_and_mul_baseline",
    "situ_and_mul_baseline_kernel",
]
