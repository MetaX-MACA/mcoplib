import torch
from mcoplib import op

import triton
import triton.language as tl
from typing import Optional, Tuple, Union
from mcoplib.profiler import profiler
FP8_DTYPE = 1
# 使用简化的profiler装饰器
@triton.jit
def _silu_and_mul_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    expert_num,
    size_n,
    BLOCK_N: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)
    # token_num_cur_expert = tl.load(masked_m_ptr + expert_id)

    # Convert strides to int64 for address calculation
    stride_input_0 = tl.cast(stride_input_0, dtype=tl.int64)
    stride_output_0 = tl.cast(stride_output_0, dtype=tl.int64)
    stride_input_1 = tl.cast(stride_input_1, dtype=tl.int64)
    stride_output_1 = tl.cast(stride_output_1, dtype=tl.int64)

    # Calculate base offsets
    offs_in_d = hidden_dim_block_index * BLOCK_N + tl.arange(0, BLOCK_N)
    input_ptr_offs = input_ptr + expert_id * stride_input_0 + offs_in_d
    output_ptr_offs = output_ptr + expert_id * stride_output_0 + offs_in_d

    # Main processing loop
    for token_index in tl.range(
        token_id, expert_num, block_num_per_expert, num_stages=NUM_STAGE
    ):
        # Load gate and up values
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

        # Compute SILU(gate) * up
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        gate_up = up * (gate * sigmoid)

        # Store BF16 result
        tl.store(
            output_ptr_offs + token_index * stride_output_1,
            gate_up.to(tl.bfloat16),
            mask=offs_in_d < size_n,
        )

def as_float32_tensor(x: Union[float, torch.tensor]) -> torch.tensor:
    return torch.as_tensor(x, dtype=torch.float32, device='cuda')

def ref_dynamic_per_token_quant(x: torch.tensor,
                                quant_dtype: torch.dtype,
                                scale_ub: Optional[torch.tensor] = None) \
        -> Tuple[torch.tensor, torch.tensor]:

    qtype_traits = torch.iinfo(quant_dtype) if quant_dtype == torch.int8 \
            else torch.finfo(torch.float8_e4m3fn)
    qtype_traits_max =  qtype_traits.max
    qtype_traits_min =  qtype_traits.min
    qtype_max = as_float32_tensor(qtype_traits_max)
    
    # For fp8, in order to match the cuda kernel output, we have to do exactly
    # the same operations as in the corresponding fp8 kernel to prevent
    # rounding errors.
    s_1 = as_float32_tensor(1.0)
    # Compute scales
    x_token_max, _ = x.abs().max(dim=-1)
    x_token_max = as_float32_tensor(x_token_max)
    
    scales = (x_token_max / qtype_max)

    # Quant
    if quant_dtype == torch.int8:
        iscales = as_float32_tensor(s_1 / scales)
        torch_out = as_float32_tensor(x) * iscales.unsqueeze(-1).expand(-1, x.shape[-1]) 
        torch_out = torch_out.round()
        torch_out = torch_out.clamp(qtype_traits_min,
                                    qtype_traits_max).to(quant_dtype)
    else:
        assert quant_dtype == FP8_DTYPE
        scales = scales.unsqueeze(-1).expand(-1, -1, x.shape[-1])   
        torch_out = as_float32_tensor(x) / scales
        torch_out = torch_out.to(torch.float8_e4m3fn)
        # scales = scales.expand(-1,-1,1)
        scales = scales[:,:,0]
        print(scales.shape)
    return torch_out, scales


def test_dq(input: torch.Tensor):
    assert input.is_contiguous()
    assert len(input.shape) == 3
    assert input.shape[-1] % 2 == 0
    output = torch.zeros(input.shape[0], input.shape[1], input.shape[2] // 2, dtype=input.dtype, device='cuda')
    size_n = input.shape[-1] // 2
    expert_num = input.shape[0]
    # Tuning parameters
    BLOCK_N = 128  
    block_num_per_expert = 64

    num_warps = 4
    NUM_STAGES = 3
    hidden_dim_split_block_num = triton.cdiv(size_n, BLOCK_N)

    grid = (
        hidden_dim_split_block_num,
        block_num_per_expert,
        expert_num,
    )

    _silu_and_mul_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        input.shape[1],
        size_n,
        BLOCK_N=BLOCK_N,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
    )

    output_quant = torch.zeros((output.shape[0], output.shape[1], output.shape[2]), dtype=torch.float8_e4m3fn, device='cuda')
    output_scale = torch.zeros(output.shape[0]*output.shape[1], 1, dtype=torch.float32, device='cuda')
    
    tmp_x, tmp_scale = ref_dynamic_per_token_quant(output, FP8_DTYPE)

    op.fused_silu_mul_dq_quant(output_quant, output_scale, input)

    assert(torch.allclose(tmp_x.to(float), output_quant.to(float), rtol=1, atol=1e-01, equal_nan=True))
    assert(torch.allclose(tmp_scale.reshape(-1), output_scale.reshape(-1), rtol=1e-04, atol=1e-03, equal_nan=True))

if __name__ == "__main__":
    ''''''
    x = torch.randn((8,1024,4096), device='cuda', dtype=torch.bfloat16)
    test_dq(x)
