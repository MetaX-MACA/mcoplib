import torch
import os
# from mcoplib.profiler import profiler
import mcoplib.op as op
# import mcoplib.sgl_kernel
from mcoplib.op import fused_situ_mul_dq_mask_quant_pack
import triton
import triton.language as tl
from typing import Optional, Tuple, Union
from mcoplib.profiler import profiler

FP8_DTYPE = 1

def situ_reference(
    input: torch.Tensor,
    beta: float,
    linear_beta: float,
):
    """
    input:
        [..., 2*d]

    output:
        [..., d]
    """

    gate, up = input.float().chunk(2, dim=-1)

    gate_out = (
        beta
        * torch.tanh(gate / beta)
        * torch.sigmoid(gate)
    )

    if linear_beta > 0:
        up = linear_beta * torch.tanh(up / linear_beta)

    out = gate_out * up

    return out.to(input.dtype)

ROCM_FP8_MAX = 224.0

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

    # Compute scales
    x_token_max, _ = x.abs().max(dim=-1)
    x_token_max = as_float32_tensor(x_token_max)
    
    scales = (x_token_max / qtype_max)
    s_1 = as_float32_tensor(1.0)
    # Quant
    if quant_dtype == torch.int8:
        iscales = as_float32_tensor(s_1 / scales)
        tmp = iscales.unsqueeze(-1).expand(-1, x.shape[-1]) 
        torch_out = as_float32_tensor(x) * tmp
        torch_out = torch_out.round()
        torch_out = torch_out.clamp(qtype_traits_min,
                                    qtype_traits_max).to(quant_dtype)
    else:
        assert quant_dtype == FP8_DTYPE
        scales = scales.unsqueeze(-1).expand(-1, x.shape[-1])   
        torch_out = as_float32_tensor(x) / scales
        torch_out = torch_out.to(torch.float8_e4m3fn)
        # scales = scales.expand(-1,-1,1)
        scales = scales[:,0]
        print(scales.shape)
    return torch_out, scales

def situ_and_mul_masked_fwd(
    input: torch.Tensor,
    output: torch.Tensor,
    masked_m: torch.Tensor,
    beta: float,
    linear_beta: float,
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
    
    output = situ_reference(input, beta, linear_beta)

    output_quant = torch.zeros((output.shape[0], output.shape[1], output.shape[2]), dtype = torch.int8, device='cuda')
    output_scale = torch.zeros(output.shape[0] * output.shape[1], dtype=torch.float32, device='cuda')
    for i in range(output.shape[0]):
        tmp_x, tmp_scale = ref_dynamic_per_token_quant(output[i][:masked_m[i],:], torch.int8)
        output_quant[i][:masked_m[i].item()]  = tmp_x
        output_scale[i*output.shape[1]: i*output.shape[1] + masked_m[i].item()] = tmp_scale

    output_scale =output_scale.reshape(output.shape[0], output.shape[1], 1)

    combine_tensor = torch.zeros(
            size=(output.shape[0], output.shape[1], (size_n//2 + 257)//256*256),
            dtype=output.dtype,
            device=output.device
    )

    hidden_size = output.shape[-1]
    a_bytes = combine_tensor.view(torch.uint8)
    b_bytes = output_quant.contiguous().view(torch.uint8)
    c_bytes = output_scale.contiguous().view(b_bytes.shape[0], b_bytes.shape[1], 1).view(torch.uint8)
    a_bytes[:,:, :hidden_size] = b_bytes
    a_bytes[:,:, hidden_size:hidden_size+4] = c_bytes
    out_stride = (input.shape[-1] // 4 + 257) // 256 * 256
    dst = torch.empty((output.shape[0], output.shape[1], out_stride), device='cuda', dtype=output.dtype)
    fused_situ_mul_dq_mask_quant_pack(dst, input, masked_m, beta, linear_beta, 1)
    
    int8_ref = combine_tensor.view(torch.uint8)
    int8_dst = dst.view(torch.uint8)
    float32_ref = int8_ref.view(torch.float32)
    float32_dst = int8_dst.view(torch.float32)
    ref=torch.zeros(int8_ref.shape[0], mask_value,1,dtype=torch.float32)
    dst = torch.zeros(int8_dst.shape[0], mask_value,1,dtype=torch.float32)
    # print(hidden_size)
    for i  in range(float32_ref.shape[0]):
        for j in range(mask_value):
            ref[i][j][0] = float32_ref[i][j][(hidden_size)//4]
            dst[i][j][0] = float32_dst[i][j][(hidden_size)//4]  

    assert(torch.allclose(ref, dst, rtol=1e-04, atol=1e-03, equal_nan=True))
    
    quant_ref = torch.zeros(int8_ref.shape[0], mask_value, hidden_size, dtype=torch.int8)
    quant_dst_ref = torch.zeros(int8_ref.shape[0], mask_value, hidden_size, dtype=torch.int8)
    for i in range(int8_ref.shape[0]):
        for j in range(mask_value):
            for k in range(hidden_size):
                quant_ref[i][j][k] = int8_ref[i][j][k]
                quant_dst_ref[i][j][k] = int8_dst[i][j][k]
                diff = quant_ref[i][j][k] - quant_dst_ref[i][j][k]
        
    assert(torch.allclose(quant_ref, quant_dst_ref, rtol=1e-3, atol=1, equal_nan = True))
    print("done")


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
def test_situ_and_mul_masked_fwd(m, n, dtype,warmup=1, rep=1):
    torch.manual_seed(42)

    for mask_id in [64,128,256,512]:
            # for num_groups, m, mask_id in ((1, 32768, 32), (2, 16384, 64),(2, 18432, 96) ,(4, 8192, 128), ):
            print(mask_id)
            num_groups = 2
            x = torch.randn((num_groups, m, n), device='cuda', dtype=torch.bfloat16)
            x_out = torch.zeros((num_groups, m, n//2), dtype=torch.bfloat16,device='cuda' )
            masked_m = torch.full((num_groups,), mask_id, dtype=torch.int32, device='cuda')
            # print(f"mask_m:{masked_m}")
            situ_and_mul_masked_fwd(x, x_out, masked_m, 1.4,1.5)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("使用设备:", device)
    test_situ_and_mul_masked_fwd(4096,4096,torch.bfloat16)
