# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT,
)
from vllm.utils.math_utils import cdiv



def scales_shape_stride_dtype(
    E: int, T: int, G: int, quant_scale_fmt: DeepGemmQuantScaleFMT
) -> tuple[tuple[int, ...], tuple[int, ...], torch.dtype]:
    shape = (E, T, G)
    strides = (T * G, 1, T)
    if quant_scale_fmt in [
        DeepGemmQuantScaleFMT.FLOAT32,
        DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
    ]:
        return shape, strides, torch.float32

    assert quant_scale_fmt == DeepGemmQuantScaleFMT.UE8M0
    shape = (E, T, cdiv(G, 4))
    strides = (T * cdiv(G, 4), 1, T)
    return shape, strides, torch.int32


@triton.jit
def _swiglustep_mul_fp8_quant_deep_gemm(
    # Pointers ------------------------------------------------------------
    input_ptr,  # 16-bit activations (E, T, 2*H)
    y_q_ptr,  # fp8 quantized activations (E, T, H)
    y_s_ptr,  # 16-bit scales (E, T, G)
    counts_ptr,  # int32 num tokens per expert (E)
    # Sizes ---------------------------------------------------------------
    H: tl.constexpr,  # hidden dimension (per output)
    GROUP_SIZE: tl.constexpr,  # elements per group (usually 128)
    # Strides for input (elements) ---------------------------------------
    stride_i_e,
    stride_i_t,
    stride_i_h,
    # Strides for y_q (elements) -----------------------------------------
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,
    # Strides for y_s (elements) -----------------------------------------
    stride_ys_e,
    stride_ys_t,
    stride_ys_g,
    # Stride for counts (elements)
    stride_counts_e,
    # Numeric params ------------------------------------------------------
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    ceil_ue8m0: tl.constexpr,
    limit,  # SwiGLU-step clamp limit (scalar, passed by value via triton).
    # Meta ---------------------------------------------------------------
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """SwiGLU-step variant of _silu_mul_fp8_quant_deep_gemm.

    Activation math (matches the CUDA persistent_masked_m_swiglu_mul_quant
    reference, computed in fp32 then cast back to the activation dtype):

        Gate_act = min(SiLU(gate), limit)        # one-sided clamp
        Up_act   = clip(up, -limit, limit)        # symmetric clamp
        Result   = Gate_act * Up_act

    The result is then per-group (group_size=128) absmax quantized to FP8
    with optional UE8M0 ceil'd scales, exactly like the silu variant.
    """
    G = H // GROUP_SIZE

    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < BLOCK

    base_input_offset = e * stride_i_e + g * GROUP_SIZE * stride_i_h
    base_gate_offset = base_input_offset + cols * stride_i_h
    base_up_offset = base_input_offset + H * stride_i_h + cols * stride_i_h
    base_yq_offset = e * stride_yq_e + g * GROUP_SIZE * stride_yq_h + cols * stride_yq_h
    base_ys_offset = e * stride_ys_e + g * stride_ys_g

    # Compute SiLU(gate) in fp32 once per group, then apply the limit clamp.
    # SiLU(x) = x / (1 + exp(-x)) = x * sigmoid(x).
    for t in tl.range(0, n_tokens, num_stages=NUM_STAGES):
        gate = tl.load(
            input_ptr + base_gate_offset + t * stride_i_t, mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(input_ptr + base_up_offset + t * stride_i_t, mask=mask, other=0.0)

        silu_gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
        gate_act = tl.minimum(silu_gate, limit)
        up_act = tl.clamp(up.to(tl.float32), -limit, limit)
        y = gate_act * up_act

        # Use multiply-by-reciprocal to match PyTorch's tensor/scalar
        # division precision.
        y_s = tl.maximum(tl.max(tl.abs(y)), eps) * (1.0 / fp8_max)
        if ceil_ue8m0:
            y_s = tl.exp2(tl.ceil(tl.log2(y_s)))

        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + base_yq_offset + t * stride_yq_t, y_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset + t * stride_ys_t, y_s)


@triton.jit
def _silu_mul_fp8_quant_deep_gemm(
    # Pointers ------------------------------------------------------------
    input_ptr,  # 16-bit activations (E, T, 2*H)
    y_q_ptr,  # fp8 quantized activations (E, T, H)
    y_s_ptr,  # 16-bit scales (E, T, G)
    counts_ptr,  # int32 num tokens per expert (E)
    # Sizes ---------------------------------------------------------------
    H: tl.constexpr,  # hidden dimension (per output)
    GROUP_SIZE: tl.constexpr,  # elements per group (usually 128)
    # Strides for input (elements) ---------------------------------------
    stride_i_e,
    stride_i_t,
    stride_i_h,
    # Strides for y_q (elements) -----------------------------------------
    stride_yq_e,
    stride_yq_t,
    stride_yq_h,
    # Strides for y_s (elements) -----------------------------------------
    stride_ys_e,
    stride_ys_t,
    stride_ys_g,
    # Stride for counts (elements)
    stride_counts_e,
    # Numeric params ------------------------------------------------------
    eps: tl.constexpr,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    ceil_ue8m0: tl.constexpr,
    # Meta ---------------------------------------------------------------
    BLOCK: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    G = H // GROUP_SIZE

    # map program id -> (e, g)
    pid = tl.program_id(0)
    e = pid // G
    g = pid % G

    e = e.to(tl.int64)
    g = g.to(tl.int64)

    # number of valid tokens for this expert
    n_tokens = tl.load(counts_ptr + e * stride_counts_e).to(tl.int64)

    cols = tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < BLOCK

    base_input_offset = e * stride_i_e + g * GROUP_SIZE * stride_i_h
    base_gate_offset = base_input_offset + cols * stride_i_h
    base_up_offset = base_input_offset + H * stride_i_h + cols * stride_i_h
    base_yq_offset = e * stride_yq_e + g * GROUP_SIZE * stride_yq_h + cols * stride_yq_h
    base_ys_offset = e * stride_ys_e + g * stride_ys_g

    for t in tl.range(0, n_tokens, num_stages=NUM_STAGES):
        gate = tl.load(
            input_ptr + base_gate_offset + t * stride_i_t, mask=mask, other=0.0
        ).to(tl.float32)
        up = tl.load(input_ptr + base_up_offset + t * stride_i_t, mask=mask, other=0.0)

        gate = gate * (1.0 / (1.0 + tl.exp(-gate)))
        y = gate * up

        # Use multiply-by-reciprocal to match PyTorch's tensor/scalar
        # division precision (Triton GPU fast-division for constexpr
        # divisors can introduce 1-ULP error).
        y_s = tl.maximum(tl.max(tl.abs(y)), eps) * (1.0 / fp8_max)
        if ceil_ue8m0:
            y_s = tl.exp2(tl.ceil(tl.log2(y_s)))

        y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)

        tl.store(y_q_ptr + base_yq_offset + t * stride_yq_t, y_q, mask=mask)
        tl.store(y_s_ptr + base_ys_offset + t * stride_ys_t, y_s)


def persistent_masked_m_silu_mul_quant(
    y: torch.Tensor,  # (E, T, 2*H)
    tokens_per_expert: torch.Tensor,  # (E,) number of valid tokens per expert
    num_parallel_tokens=16,
    group_size: int = 128,
    quant_scale_fmt: DeepGemmQuantScaleFMT = DeepGemmQuantScaleFMT.FLOAT32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize silu(y[..., :H]) * y[..., H:] to FP8 with group per-token scales
    y has shape (E, T, 2*H). The first half of the last dimension is
    silu-activated, multiplied by the second half, then quantized into FP8.
    We launch a fixed grid of threads to accommodate CUDA graphs. Let `P2`
    be a parallelization factor for persistent_masked_m_silu_mul_quant over the
    hidden dimension.

    Let `expert_offsets = [0] + [num_tokens.cumsum()]` and
    `total_tokens = expert_offsets[-1]`.
    persistent_masked_m_silu_mul_quant launches `total_tokens x P2` number of
    thread blocks. Each thread block contains `NUM_WARPS` warps.

    Every thread block needs to find it's corresponding expert by warp-parallel scanning
    over the `expert_offsets` array.

    The i-th warp in the first thread block processes
    `[i * warp_chunk_size, (i + 1) * warp_chunk_size]` groups
    sequentially, where `warp_chunk_size = ((H / GROUP_SIZE) / P2) / NUM_WARPS`,
    pipelining loads and computes.

    The shared memory layout for 4 warps with a 2-stage pipeline for SiLU V2
    can is visualized like so:

                         stage0                              stage1
    ┌─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┬─────┬───┐
    │gate0│up0│gate1│up1│gate2│up2│gate3│up3│gate0│up0│gate1│up1│gate2│up2│gate3│up3│
    └─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┴─────┴───┘

    with the main difference between V1 and V2 being the global load
    stride between warps, and between half-warps. Regarding the latter stride,
    we assign the first half warp of every warp for `gate` loads and the second
    half-warp to `up` loads.

    Returns `(y_q, y_s)` where
    * `y_q`: FP8 tensor, shape (E, T, H), same layout as y[..., :H]
    * `y_s` depends on quant_scale_fmt,
      - quant_scale_fmt == FLOAT32,
         `y_s`: FP32 tensor, shape (E, T, H // group_size), strides (T*G, 1, T)
      - quant_scale_fmt == E8M0,
         `y_s`: Int32 tensor, shape (E, T, H // group_size // 4), strides (T*G, 1, T)
      - quant_scale_fmt == E8M0_FLOAT32_SPARSE
         `y_s`: FP32 tensor, shape (E, T, H // group_size), strides (T*G, 1, T)
    Let NUM_WARPS be the number of warps in a single thread block and
    `GROUP_SIZE = 128` be the size of the quantization group.
    """
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    G = (H + group_size - 1) // group_size
    assert H % 8 == 0, "H must be divisible by 8"
    assert group_size == 128, "H must be divisible by 8"
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E

    tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)

    fp8_dtype = current_platform.fp8_dtype()
    y_q = torch.empty((E, T, H), dtype=fp8_dtype, device=y.device)

    ys_shape, ys_strides, ys_dtype = scales_shape_stride_dtype(E, T, G, quant_scale_fmt)
    y_s = torch.empty_strided(
        ys_shape,
        ys_strides,
        dtype=ys_dtype,
        device=y.device,
    )

    ceil_ue8m0 = quant_scale_fmt in [
        DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
        DeepGemmQuantScaleFMT.UE8M0,
    ]

   
        # Triton fallback for ROCm -- the C++ kernel is guarded by
        # #ifndef USE_ROCM in activation_kernels.cu.
        # https://github.com/ROCm/aiter/issues/2420
    stride_cnt_e = tokens_per_expert.stride()[0]

    # Static grid over experts and H-groups.
    # A loop inside the kernel handles the token dim
    grid = (E * G,)
    # strides (elements)
    stride_i_e, stride_i_t, stride_i_h = y.stride()
    stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()

    fp8_min, fp8_max = get_fp8_min_max()
    eps: float = 1e-10
    assert y_s.dtype == torch.float32, (
        "_silu_mul_fp8_quant_deep_gemm Triton fallback does not "
        f"support {y_s.dtype} scales. Only torch.float32 supported."
    )
    _silu_mul_fp8_quant_deep_gemm[grid](
        y,
        y_q,
        y_s,
        tokens_per_expert,
        H,
        group_size,
        stride_i_e,
        stride_i_t,
        stride_i_h,
        stride_yq_e,
        stride_yq_t,
        stride_yq_h,
        ys_strides[0],
        ys_strides[1],
        ys_strides[2],
        stride_cnt_e,
        eps,
        fp8_min,
        fp8_max,
        ceil_ue8m0,
        BLOCK=group_size,
        NUM_STAGES=4,
        num_warps=1,
    )

    return y_q, y_s


def persistent_masked_m_swiglu_mul_quant(
    y: torch.Tensor,  # (E, T, 2*H)
    tokens_per_expert: torch.Tensor,  # (E,) number of valid tokens per expert
    limit: torch.Tensor,  # (1,) bf16/fp32 SwiGLU-step clamp limit
    num_parallel_tokens=16,
    group_size: int = 128,
    quant_scale_fmt: DeepGemmQuantScaleFMT = DeepGemmQuantScaleFMT.FLOAT32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize SwiGLU-step(y[..., :H], y[..., H:]) to FP8 with group per-token
    scales.  Math (matches the CUDA persistent_masked_m_swiglu_mul_quant):

        Gate_act = min(SiLU(gate), limit)        # one-sided clamp
        Up_act   = clip(up, -limit, limit)        # symmetric clamp
        Result   = Gate_act * Up_act

    The result is per-group absmax quantized to FP8 (group_size=128) with
    optional UE8M0 ceil'd scales, exactly like the silu variant.  `limit`
    is a 1-element tensor (bf16 or fp32) -- passing it as a tensor instead
    of a python scalar avoids the MACA cucc trailing-scalar ABI segfault
    documented for persistent masked-m kernels.
    """
    assert y.ndim == 3, "y must be (E, T, 2*H)"
    E, T, H2 = y.shape
    assert H2 % 2 == 0, "last dim of y must be even (2*H)"
    H = H2 // 2
    G = (H + group_size - 1) // group_size
    assert H % 8 == 0, "H must be divisible by 8"
    assert group_size == 128, "H must be divisible by 8"
    assert tokens_per_expert.ndim == 1 and tokens_per_expert.shape[0] == E

    tokens_per_expert = tokens_per_expert.to(device=y.device, dtype=torch.int32)

    fp8_dtype = current_platform.fp8_dtype()
    y_q = torch.empty((E, T, H), dtype=fp8_dtype, device=y.device)

    ys_shape, ys_strides, ys_dtype = scales_shape_stride_dtype(E, T, G, quant_scale_fmt)
    y_s = torch.empty_strided(
        ys_shape,
        ys_strides,
        dtype=ys_dtype,
        device=y.device,
    )

    ceil_ue8m0 = quant_scale_fmt in [
        DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0,
        DeepGemmQuantScaleFMT.UE8M0,
    ]

    stride_cnt_e = tokens_per_expert.stride()[0]

    grid = (E * G,)
    stride_i_e, stride_i_t, stride_i_h = y.stride()
    stride_yq_e, stride_yq_t, stride_yq_h = y_q.stride()

    fp8_min, fp8_max = get_fp8_min_max()
    eps: float = 1e-10
    assert y_s.dtype == torch.float32, (
        "_swiglustep_mul_fp8_quant_deep_gemm Triton fallback does not "
        f"support {y_s.dtype} scales. Only torch.float32 supported."
    )

    # Resolve `limit` to a python float for triton (constexpr-style scalar).
    # Triton passes scalars via the kernel argument ABI; on MACA cucc this
    # is fine because triton lowers scalars to stack-slotted args, not as
    # trailing by-value params to a persistent CUDA kernel.
    if isinstance(limit, torch.Tensor):
        assert limit.numel() == 1, "limit must be a 1-element tensor"
        limit_val = float(limit.detach().cpu().float().item())
    else:
        limit_val = float(limit)

    _swiglustep_mul_fp8_quant_deep_gemm[grid](
        y,
        y_q,
        y_s,
        tokens_per_expert,
        H,
        group_size,
        stride_i_e,
        stride_i_t,
        stride_i_h,
        stride_yq_e,
        stride_yq_t,
        stride_yq_h,
        ys_strides[0],
        ys_strides[1],
        ys_strides[2],
        stride_cnt_e,
        eps,
        fp8_min,
        fp8_max,
        ceil_ue8m0,
        limit_val,
        BLOCK=group_size,
        NUM_STAGES=4,
        num_warps=1,
    )

    return y_q, y_s