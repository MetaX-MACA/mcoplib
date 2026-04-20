import torch
from mcoplib.custom_ops import fused_add_rms_norm_dynamic_per_token_quant_padding_output, fused_fp8_add_rms_norm_dynamic_per_token_quant_padding_output


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

def torch_func(x, residual, weight, eps):
    sum = x + residual
    rms_res = torch.rms_norm(sum, sum.shape, weight=weight.unsqueeze(0), eps=eps)
    return sum, rms_res


def test_rms_norm_custom(warmup=10, rep=100):
    eps = 1e-6
    dtype = torch.bfloat16
    batch_size = 1
    hidden_size = 7168
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")
    
    dur_forward = benchmark(fused_add_rms_norm_dynamic_per_token_quant_padding_output, (x, residual, weight, 3840, eps), warmup, rep)
    print(f"batch_size={batch_size}, hidden_size={hidden_size}, dtype={dtype}, infi_forward={dur_forward.mean().item()}")

    sum_torch, rms_torch = torch_func(x, residual, weight,eps)
    output_, residual_, output_rms_, output_quant_int8_, scales_ = fused_add_rms_norm_dynamic_per_token_quant_padding_output(x, residual, weight, 3840, eps)

    torch.allclose(sum_torch, residual_)
    torch.allclose(rms_torch, output_rms_)
    print("precision check pass")

def test_rms_norm_fp8_custom(warmup=10, rep=100):
    eps = 1e-6
    dtype = torch.bfloat16
    batch_size = 1
    hidden_size = 7168
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")
    
    dur_forward = benchmark(fused_fp8_add_rms_norm_dynamic_per_token_quant_padding_output, (x, residual, weight, 3840, eps), warmup, rep)
    print(f"batch_size={batch_size}, hidden_size={hidden_size}, dtype={dtype}, infi_forward={dur_forward.mean().item()}")

    sum_torch, rms_torch = torch_func(x, residual, weight,eps)
    output_, residual_, output_rms_, output_quant_int8_, scales_ = fused_fp8_add_rms_norm_dynamic_per_token_quant_padding_output(x, residual, weight, 3840, eps)

    torch.allclose(sum_torch, residual_)
    torch.allclose(rms_torch, output_rms_)
    print("precision check pass")

if __name__ == "__main__":

    # In this case, Megatron foward cost is 700 us , inficom forward cost is 200 us 
    # we assume the similar speedup in backward 
    test_rms_norm_custom(10, 100)
    test_rms_norm_fp8_custom(10,100) 
