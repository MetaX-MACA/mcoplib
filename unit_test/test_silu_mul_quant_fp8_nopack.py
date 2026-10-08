import torch
from typing import Tuple, Optional, TYPE_CHECKING

import pytest
import triton
import triton.language as tl
from torch.profiler import profile, ProfilerActivity
from mcoplib.op import fused_silu_mul_dq_mask_quant_fp8_nopack
from mcoplib.profiler import profiler
# from sglang.jit_kernel.utils import (
#     cache_once,
#     is_arch_support_pdl,
#     load_jit,
#     make_cpp_args,
# )

@triton.jit
def _silu_and_mul_post_quant_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    output_scale_ptr,
    stride_output_scale_0,
    stride_output_scale_1,
    stride_output_scale_2,
    masked_m_ptr,
    size_n,
    fp8_max,
    fp8_min,
    swiglu_limit,
    BLOCK_N: tl.constexpr,
    NUM_STAGE: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)

    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)

    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    offs_in_d = hidden_dim_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    input_ptr_offs = input_ptr + expert_id * stride_input_0 + offs_in_d
    output_ptr_offs = output_ptr + expert_id * stride_output_0 + offs_in_d
    output_scale_offs = (
        output_scale_ptr
        + expert_id * stride_output_scale_0
        + hidden_dim_block_index * stride_output_scale_2
    )

    for token_index in tl.range(
        token_id, token_num_cur_expert, block_num_per_expert, num_stages=NUM_STAGE
    ):
        gate = tl.load(
            input_ptr_offs + token_index * stride_input_1,
            mask=offs_in_d < size_n,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            input_ptr_offs + token_index * stride_input_1 + size_n,
            mask=offs_in_d < size_n,
            other=0.0,
        ).to(tl.float32)
        if swiglu_limit!=0:
            limit_tensor = tl.full((BLOCK_N,), swiglu_limit, dtype=gate.dtype)
            gate = tl.minimum(gate, limit_tensor)
            up = tl.maximum(tl.minimum(up, limit_tensor), -limit_tensor)
        gate = gate / (1 + tl.exp(-gate))
        gate = gate.to(input_ptr.dtype.element_ty)
        gate_up = up * gate
        _absmax = tl.maximum(tl.max(tl.abs(gate_up)), 1e-10)
        output_s = _absmax / fp8_max
        if SCALE_UE8M0:
            output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))
        output_q = tl.clamp(gate_up / output_s, fp8_min, fp8_max).to(
            output_ptr.dtype.element_ty
        )
        tl.store(
            output_ptr_offs + token_index * stride_output_1,
            output_q,
            mask=offs_in_d < size_n,
        )
        tl.store(
            output_scale_offs + token_index * stride_output_scale_1,
            output_s,
        )


def silu_and_mul_masked_post_quant_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    scale_ue8m0: bool = False,
    swiglu_limit: float=None,
    
):
    """
    input shape [expert_num, token_num_padded, hidden_dim]
    output shape [expert_num, token_num_padded, hidden_dim // 2], dtype fp8
    output_scale [expert_num token_num_paddded, hidden_dim // 2 // 128] dtype float32
    quant_group_size  int,
    masked_m shape [expert_num],
    """

    assert input.is_contiguous()
    assert output.dtype == torch.float8_e4m3fn
    assert output.is_contiguous()
    assert len(input.shape) == 3
    assert input.shape[0] == masked_m.shape[0]
    assert input.shape[-1] % 2 == 0

    size_n = input.shape[-1] // 2
    assert size_n % quant_group_size == 0

    expert_num = len(masked_m)

    if expert_num < 4:
        BLOCK_NUM_PER_EXPERT = 64
    else:
        BLOCK_NUM_PER_EXPERT = 32

    BLOCK_N = quant_group_size
    num_warps = 1
    NUM_STAGES = 6
    hidden_dim_split_block_num = triton.cdiv(size_n, BLOCK_N)
    assert BLOCK_N % quant_group_size == 0

    grid = (
        hidden_dim_split_block_num,
        BLOCK_NUM_PER_EXPERT,
        expert_num,
    )

    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max = finfo.max
    fp8_min = -fp8_max

    _silu_and_mul_post_quant_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        output_scale,
        *output_scale.stride(),
        masked_m,
        size_n,
        fp8_max,
        fp8_min,
        swiglu_limit,
        BLOCK_N=BLOCK_N,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
        SCALE_UE8M0=scale_ue8m0,
    )
    return

def make_name(name: str) -> str:
    return f"dpsk_v4_{name}"

def calc_diff(x, y):
    x, y = x.double(), y.double()
    cos_sim = (x * y).sum() / (x.norm() * y.norm())
    return 1 - cos_sim  # 余弦距离

@profiler(output_dir="./profiles", warmup=2, repeat=3)
def _varlen_deep_gemm_silu_mul_quant_transpose(
    gateup_output: torch.Tensor,
    masked_m: Optional[torch.Tensor],
    topk: int,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    group_size = 128

    assert masked_m is not None
    hidden_states_device = gateup_output.device
    E, N, D_2 = gateup_output.shape
    D = D_2 // 2
    del D_2
    G = D // group_size
    gateup_output_clone = gateup_output.clone()
    down_input = torch.empty(
        (E, N, D),
        device=hidden_states_device,
        dtype=torch.float8_e4m3fn,
    )
    down_input_clone = down_input.clone()
    down_input_scale = torch.empty(
        (E, N, G),
        device=hidden_states_device,
        dtype=torch.float32,
    )
    value=0.0
    if swiglu_limit!=None:
        value=swiglu_limit
    
    down_input_scale_tmp = torch.empty(
        (E, G, N),                       # 真·连续列主序 buffer
        device=hidden_states_device,
        dtype=torch.float32,
    ).transpose(-1,-2)

    silu_and_mul_masked_post_quant_fwd(
        gateup_output,
        down_input,
        down_input_scale,
        group_size,
        masked_m,
        scale_ue8m0=False,
        swiglu_limit=value,
    )
    fused_silu_mul_dq_mask_quant_fp8_nopack(
        down_input_clone, 
        down_input_scale_tmp, 
        gateup_output_clone, 
        masked_m, 
        group_size,
        swiglu_limit,
        True
    )
    
    down_input_scale_clone = down_input_scale_tmp.contiguous()
    for j in range(num_groups):
        diff = calc_diff(
            down_input[j, :masked_m[j].item()], 
            down_input_clone[j, :masked_m[j].item()]
        )
        assert diff < 0.05, f'fp8 {m=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'

        diff = calc_diff(
            down_input_scale[j, :masked_m[j].item()], 
            down_input_scale_clone[j, :masked_m[j].item()]
        )

        assert diff < 0.05, f'scale {m=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'
        print("check successfuly")
    
@profiler(output_dir="./profiles", warmup=2, repeat=3)
def _varlen_deep_gemm_silu_mul_quant(
    gateup_output: torch.Tensor,
    masked_m: Optional[torch.Tensor],
    topk: int,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    group_size = 128

    assert masked_m is not None
    hidden_states_device = gateup_output.device
    E, N, D_2 = gateup_output.shape
    D = D_2 // 2
    del D_2
    G = D // group_size
    gateup_output_clone = gateup_output.clone()
    down_input = torch.empty(
        (E, N, D),
        device=hidden_states_device,
        dtype=torch.float8_e4m3fn,
    )
    down_input_clone = down_input.clone()
    down_input_scale = torch.empty(
        (E, N, G),
        device=hidden_states_device,
        dtype=torch.float32,
    )
    value=0.0
    if swiglu_limit!=None:
        value=swiglu_limit
    down_input_scale_clone = down_input_scale.clone()

    silu_and_mul_masked_post_quant_fwd(
        gateup_output,
        down_input,
        down_input_scale,
        group_size,
        masked_m,
        scale_ue8m0=False,
        swiglu_limit=value,
    )
    fused_silu_mul_dq_mask_quant_fp8_nopack(
        down_input_clone, 
        down_input_scale_clone, 
        gateup_output_clone, 
        masked_m, 
        group_size,
        swiglu_limit
    )

    for j in range(num_groups):
        diff = calc_diff(
            down_input[j, :masked_m[j].item()], 
            down_input_clone[j, :masked_m[j].item()]
        )
        assert diff < 0.05, f'fp8 {m=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'

        diff = calc_diff(
            down_input_scale[j, :masked_m[j].item()], 
            down_input_scale_clone[j, :masked_m[j].item()]
        )

        assert diff < 0.05, f'scale {m=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}'
        print("check successfuly")


# @pytest.mark.parametrize("num_groups", [8,16])
# @pytest.mark.parametrize("m", [4096,2048])
# @pytest.mark.parametrize("n", [4096,6144,7168])
# @pytest.mark.parametrize("topk", [8,6])
# @pytest.mark.parametrize("swizzle", [True,False])
# @pytest.mark.parametrize("mask_id", [32,64,128,256])
# @pytest.mark.parametrize("swiglu_limit", [10.0,None])
# def test_main(
#     num_groups: int,
#     m: int,
#     n: int,
#     topk: int,
#     swizzle: bool,
#     mask_id: int,
#     swiglu_limit: Optional[float],
# ):
#     masked_m = torch.full((num_groups,), mask_id, dtype=torch.int32, device='cuda')
#     gateup_output = torch.randn((num_groups, m, n), device='cuda', dtype=torch.bfloat16)
#     _varlen_deep_gemm_silu_mul_quant(gateup_output, masked_m, topk, swiglu_limit, swizzle)

if __name__ == "__main__":

    num_groups = 8
    m = 4096
    n = 7168
    topk = 8
    torch.manual_seed(10)
    for swiglu_limit in (10.0, None):
        for mask_id in [32,64,128,256]:
            masked_m = torch.full((num_groups,), mask_id, dtype=torch.int32, device='cuda')
            gateup_output = torch.randn((num_groups, m, n), device='cuda', dtype=torch.bfloat16)
            _varlen_deep_gemm_silu_mul_quant(gateup_output, masked_m, topk, swiglu_limit, False)
            _varlen_deep_gemm_silu_mul_quant_transpose(gateup_output, masked_m, topk, swiglu_limit, False)
