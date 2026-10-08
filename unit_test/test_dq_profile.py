import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import mcoplib._C
from typing import Optional, Union, Tuple
import mcoplib.sgl_kernel

def as_float32_tensor(x: Union[float, torch.tensor]) -> torch.tensor:
    return torch.as_tensor(x, dtype=torch.float32, device='cuda')

FP8_DTYPE = torch.float8_e4m3fn

def ref_dynamic_per_token_quant(x: torch.tensor,
                                quant_dtype: torch.dtype,
                                scale_ub: Optional[torch.tensor] = None) \
        -> Tuple[torch.tensor, torch.tensor]:
    assert quant_dtype in [torch.int8]
    
    qtype_traits = torch.iinfo(quant_dtype) if quant_dtype == torch.int8 \
        else torch.finfo(quant_dtype)
    qtype_traits_max =  qtype_traits.max
    qtype_traits_min =  qtype_traits.min
    qtype_max = as_float32_tensor(qtype_traits_max)
    s_1 = as_float32_tensor(1.0)
    s_512 = as_float32_tensor(512.0)

    # Compute scales：在最后一维求abs max
    x_token_max, _ = x.abs().max(dim=-1)
    x_token_max = as_float32_tensor(x_token_max)
    if scale_ub is not None:
        x_token_max = x_token_max.clamp(max=scale_ub)
    # ========= FIX 这里！用 ... None 在最后一维扩维度 =========
    scales = (x_token_max / qtype_max)[..., None]

    # Quant
    if quant_dtype == torch.int8:
        iscales = as_float32_tensor(s_1 / scales)
        torch_out = as_float32_tensor(x) * iscales
        torch_out = torch_out.round()
        torch_out = torch_out.clamp(qtype_traits_min,
                                    qtype_traits_max).to(quant_dtype)
    else:
        assert quant_dtype == FP8_DTYPE
        min_scaling_factor = s_1 / (qtype_max * s_512)
        scales = scales.clamp(min=min_scaling_factor)
        torch_out = as_float32_tensor(x) / scales
        torch_out = torch_out.clamp(qtype_traits_min,
                                    qtype_traits_max).to(quant_dtype)
    return torch_out, scales


def scaled_int8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    azp: Optional[torch.Tensor] = None,
    symmetric: bool = True
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp.
    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).
    Returns:
     tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] : Output int8 tensor, scales, and optionally azp.
    """
    output = torch.empty_like(input, dtype=torch.int8)
    if scale is not None:
        # static-per-tensor quantization.
        assert symmetric == (
            azp
            is None), "azp must only be provided for asymmetric quantization."
        torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)
        return output, scale, azp
    # dynamic-per-token quantization.
    input_scales = torch.empty((input.numel() // input.shape[-1], 1),
                               device=input.device,
                               dtype=torch.float32)
    input_azp = None if symmetric else torch.empty_like(input_scales,
                                                        dtype=torch.int32)
    torch.ops.sgl_kernel.dynamic_scaled_int8_quant.default(output, input.contiguous(),
                                            input_scales, input_azp)
    return output, input_scales, input_azp

# ====================== 新增：反量化 & 精度评估函数 ======================
def dequant_int8_per_token(q_val: torch.Tensor, scales: torch.Tensor):
    """对称per-token反量化： x_recon = q_val * scale"""
    return q_val.to(torch.float32) * scales

def compute_quant_error(original_fp: torch.Tensor, recon_fp: torch.Tensor):
    """
    计算量化重建误差指标
    return: max_abs_err, rmse, mse, snr_db
    """
    orig_f32 = original_fp.to(torch.float32)
    recon_f32 = recon_fp.to(torch.float32)
    diff = orig_f32 - recon_f32
    mse = torch.mean(diff * diff).item()
    rmse = torch.sqrt(torch.mean(diff * diff)).item()
    max_abs_err = torch.max(torch.abs(diff)).item()
    # SNR dB
    signal_power = torch.mean(orig_f32 * orig_f32).item()
    noise_power = mse + 1e-12
    snr_db = 10 * torch.log10(torch.tensor(signal_power / noise_power)).item()
    return max_abs_err, rmse, mse, snr_db


def test_one_shape(shape, device="cuda", n_iter=100, warmup=5):
    """单shape：精度校验 + profiler测速"""
    print(f"\n===== Testing shape {shape} =====")
    # 造输入
    attn_output = torch.randn(*shape, device=device, dtype=torch.float16)
    x_input = attn_output.contiguous() # (seq,bs,hdim)
    print(f"input after transpose+contiguous: {x_input.shape}")

    # ===== 1. 参考实现 ref_dynamic_per_token_quant =====
    q_ref, scale_ref = ref_dynamic_per_token_quant(x_input, quant_dtype=torch.int8)
    recon_ref = dequant_int8_per_token(q_ref, scale_ref)

    # ===== 2. C++ kernel实现 scaled_int8_quant =====
    # 预热
    print(x_input.shape)
    for _ in range(warmup):
        q_cpp, scale_cpp, _ = scaled_int8_quant(x_input)
    torch.cuda.synchronize()

    # 精度校验：kernel vs ref
    recon_cpp = dequant_int8_per_token(q_cpp, scale_cpp)
    max_abs_err, rmse, mse, snr_db = compute_quant_error(x_input, recon_cpp)
    # kernel输出 和 ref输出比对
    q_diff = (q_cpp.to(torch.int32) - q_ref.to(torch.int32)).abs().max().item()
    scale_diff = (scale_cpp - scale_ref).abs().max().item()

    print(f"[Precision Check]")
    print(f"  q_val max diff (cpp vs ref): {q_diff}")
    print(f"  scale max diff (cpp vs ref): {scale_diff:.8e}")
    print(f"  Reconstruction MaxAbsErr: {max_abs_err:.6f}")
    print(f"  Reconstruction RMSE:      {rmse:.6f}")
    print(f"  Reconstruction MSE:       {mse:.6f}")
    print(f"  SNR(dB):                 {snr_db:.2f}")

    # ===== 3. Profiler 测速 =====
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(n_iter):
            q_cpp, scale_cpp, _ = scaled_int8_quant(x_input)
            torch.cuda.synchronize()
    print("\nProfiler result (sorted by cuda_time_total):")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return {
        "shape": shape,
        "q_max_diff": q_diff,
        "scale_max_diff": scale_diff,
        "max_abs_err": max_abs_err,
        "rmse": rmse,
        "snr_db": snr_db
    }

# ====================== 批量多shape测试列表（LLM常用attn输出shape） ======================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("This kernel only runs on CUDA")
    torch.manual_seed(42)

    # 你可以在这里增减shape，格式：(bs, seq_len, head_dim)
    test_shapes = [
        (1024, 4096),
        (2048, 4096),
        (3*1024, 4096),
        (4*1024, 4096),
        (5*1024, 4096),
        (6*1024, 4096),
        (7*1024, 4096),
        (8*1024, 4096),
    ]
    all_results = []
    for s in test_shapes:
        res = test_one_shape(s, device=device, n_iter=100, warmup=5)
        all_results.append(res)

    # 汇总打印所有shape精度总表
    print("\n" + "="*80)
    print("SUMMARY ALL SHAPE PRECISION RESULT")
    print(f"{'shape':<20} | q_diff | scale_diff | MaxAbsErr | RMSE | SNR(dB)")
    print("-"*80)
    for r in all_results:
        sh = str(r["shape"])
        print(f"{sh:<20} | {r['q_max_diff']:<7.0f} | {r['scale_max_diff']:.2e} | {r['max_abs_err']:.6f} | {r['rmse']:.6f} | {r['snr_db']:.2f}")
    print("="*80)
