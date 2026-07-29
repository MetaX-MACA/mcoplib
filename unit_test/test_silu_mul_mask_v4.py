import torch
import os
# from mcoplid.profiler import profiler
import mcoplid.op as op
# import mcoplid.sgl_kernel
from mcoplid.op import silu_mul_mask
import triton
import triton.language as tl
from typing import Optional, Tuple, Union
from mcoplid.profiler import profiler

FP8_DTYPE = 1

# 使用简化的profiler装饰器
@triton.jit
def _silu_and_mul_masked_kernel(
    input_ptr,
    stride_input_0,
    stride_input_1,
    stride_input_2,
    output_ptr,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    masked_m_ptr,
    size_n,
    swiglu_limit,
    BLOCK_N: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    expert_id = tl.program_id(2)
    token_id = tl.program_id(1)
    hidden_dim_block_index = tl.program_id(0)

    block_num_per_expert = tl.num_programs(1)
    token_num_cur_expert = tl.load(masked_m_ptr + expert_id)

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
    for token_index in tl.range(token_num_cur_expert, block_num_per_expert, num_stages=NUM_STAGE):
        token_id, token_num_cur_expert
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
        if swiglu_limit > 0.0:
            # up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            up = max(min(up, swiglu_limit), -swiglu_limit)
            # gate = torch.clamp(gate, max=self.swiglu_limit)
            gate = min(gate, swiglu_limit)

        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        gate_up = up * (gate * sigmoid)

        # Store BF16 result
        tl.store(
            output_ptr_offs + token_index * stride_output_1,
            gate_up.to(tl.bfloat16),
            mask=offs_in_d < size_n,
        )

ROCM_FP8_MAX = 224.0

def silu_and_mul_masked_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    masked_m: torch.Tensor,
    swiglu_limit: float
):
    """
    input shape [expert_num, token_num_padded, hidden_dim]
    output shape [expert_num, token_num_padded, hidden_dim // 2], dtype bf16
    masked_m shape [expert_num], indicates valid tokens per expert

    实现 silu_and_mul + quant + 打包
    """
    assert input.is_contiguous()
    assert output.dtype == torch.bfloat16
    assert output.is_contiguous()
    assert len(input.shape) == 3
    assert input.shape[0] == masked_m.shape[0]
    assert input.shape[-1] % 2 == 0

    size_n = input.shape[-1] // 2
    expert_num = len(masked_m)
    mask_value = masked_m[0]
    # Tuning parameters
    BLOCK_N = 128
    block_num_per_expert = 64
    swiglu_limit = 2.0
    num_warps = 4
    NUM_STAGES = 3

    hidden_dim_split_block_num = triton.cdiv(size_n, BLOCK_N)
    grid = (
        hidden_dim_split_block_num,
        block_num_per_expert,
        expert_num,
    )

    _silu_and_mul_masked_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        masked_m,
        size_n,
        swiglu_limit,
        BLOCK_N=BLOCK_N,
        NUM_STAGE=NUM_STAGES,
        num_warps=num_warps,
    )
    output_check = torch.zeros_like(output, device='cuda')
    silu_mul_mask(output_check, input, masked_m, swiglu_limit)
    # import pdb
    # pdb.set_trace()
    assert(torch.allclose(output, output_check, rtol=1e-3, atol=1e-3, equal_nan = True))
    print("check successfully")

def benchmark(func, args, warmup=2, rep=10):
    for _ in range(warmup):
        func(*args)
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    for i in range(rep):
        start_event[i].record()
        func(*args)
        end_event[i].record()
    torch.cuda.synchronize()
    dur = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float,
    )
    return dur

@profiler(output_dir="./profiles", warmup=2, repeat=3)
def test_silu_and_mul_masked_fwd(dtype,warmup=1, rep=1):
    torch.manual_seed(42)
    for n in (512, 1024, 2048, 4096, 8192, 16384):
        for num_groups, m, mask_id in ((1, 32768, 32), (2, 16384, 64),(2, 18432, 96) ,(4, 8192, 128), (5, 8192, 128), (6, 8192, 128)):
            print(f"{num_groups}, {m}, {mask_id},{n}")
            x = torch.randn((num_groups, m, n), device='cuda', dtype=torch.bfloat16)
            x_out = torch.zeros((num_groups, m, n//2), dtype=torch.bfloat16,device='cuda' )
            masked_m = torch.full((num_groups,), mask_id, dtype=torch.int32, device='cuda')
            silu_and_mul_masked_fwd(x, x_out, masked_m,2.0)

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("使用设备:", device)
    print("v4")
    test_silu_and_mul_masked_fwd(torch.bfloat16)
