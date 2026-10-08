import torch
import os
import mcoplib
import mcoplib.op as op
from mcoplib.op import per_token_quant_int8_pack
import triton
import triton.language as tl
from typing import Optional, Tuple, Union
from mcoplib.profiler import profiler

FP8_DTYPE = 1

def as_float32_tensor(x: Union[float, torch.Tensor]) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device='cuda')

def ref_dynamic_per_token_quant(x: torch.Tensor,
                                quant_dtype: torch.dtype,
                                scale_ub: Optional[torch.Tensor] = None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    if quant_dtype == torch.int8:
        qtype_traits = torch.iinfo(quant_dtype)
    else:
        qtype_traits = torch.finfo(torch.float8_e4m3fn)
    qtype_traits_max = qtype_traits.max
    qtype_traits_min = qtype_traits.min
    qtype_max = as_float32_tensor(qtype_traits_max)

    # Compute per-token max
    x_token_max, _ = x.abs().max(dim=-1)
    x_token_max = as_float32_tensor(x_token_max)

    scales = (x_token_max / qtype_max)
    s_1 = as_float32_tensor(1.0)

    if quant_dtype == torch.int8:
        iscales = as_float32_tensor(s_1 / scales)
        tmp = iscales.unsqueeze(-1)
        torch_out = as_float32_tensor(x) * tmp
        torch_out = torch_out.round()
        torch_out = torch_out.clamp(qtype_traits_min, qtype_traits_max).to(quant_dtype)
    else:
        # FP8分支，这里quant_dtype直接用torch.float8_e4m3fn，不再和数字1比较
        assert quant_dtype == torch.float8_e4m3fn
        scales = scales.unsqueeze(-1)
        torch_out = as_float32_tensor(x) / scales
        torch_out = torch_out.to(torch.float8_e4m3fn)
        scales = scales[:, 0]
    return torch_out, scales


def per_token_quant_pack(input: torch.Tensor):
    assert input.is_contiguous()
    g, m, n = input.shape
    size_n = n
    
    output_quant, output_scale = ref_dynamic_per_token_quant(input, torch.int8)
    # scale shape [g,m], expand to [g,m,1]
    output_scale = output_scale.reshape(g, m, 1)

    # 对齐padding size，kernel要求256对齐
    packed_last_dim = ((size_n // 2 + 257) // 256) * 256
    combine_tensor = torch.zeros(
        size=(g, m, packed_last_dim),
        dtype=input.dtype,
        device=input.device
    )
    hidden_size = size_n

    # 转uint8字节视图，写入int8量化数据
    a_bytes = combine_tensor.view(torch.uint8)
    b_bytes = output_quant.contiguous().view(torch.uint8)
    # scale是float32，4字节，[g,m,1]
    c_bytes = output_scale.contiguous().view(torch.uint8)
    # int8数据放在前n字节，后面4字节放scale
    a_bytes[:, :, :hidden_size] = b_bytes
    a_bytes[:, :, hidden_size:hidden_size + 4] = c_bytes

    # call mcoplib cuda kernel
    dst = torch.empty((g, m, packed_last_dim), device='cuda', dtype=input.dtype)
    per_token_quant_int8_pack(dst, input)

    # 转字节视图
    int8_ref = combine_tensor.view(torch.uint8)
    int8_dst = dst.view(torch.uint8)

    # 读取scale：scale从offset hidden_size开始，4个uint8组成float32
    # 取float32视图，每4字节1个float
    float32_ref_view = int8_ref.view(torch.float32)
    float32_dst_view = int8_dst.view(torch.float32)

    # scale的索引位置：hidden_size个uint8，对应hidden_size/4个float32元素
    scale_float_idx = hidden_size // 4
    ref_scale = float32_ref_view[:, :, scale_float_idx]
    dst_scale = float32_dst_view[:, :, scale_float_idx]

    # 校验scale
    assert torch.allclose(ref_scale, dst_scale, rtol=1e-04, atol=1e-03, equal_nan=True), "scale mismatch!"
    # 额外校验int8量化数据
    assert torch.allclose(int8_ref[:, :, :hidden_size].to(torch.int16), int8_dst[:, :, :hidden_size].to(torch.int16), rtol=1,atol=1,equal_nan=True), "quant int8 data mismatch!"
    print("✅ test passed")


def benchmark(func, args, warmup=2, rep=10):
    for _ in range(warmup):
        func(*args)
    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
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

# @profiler(output_dir="./profiles", warmup=2, repeat=3)
def test_per_token_pack(g, m, n, dtype, warmup=1, rep=1):
    torch.manual_seed(42)
    x = torch.randn((g, m, n), device='cuda', dtype=dtype)
    per_token_quant_pack(x)

if __name__ == "__main__":
    test_per_token_pack(4,1024, 4096, torch.bfloat16)
