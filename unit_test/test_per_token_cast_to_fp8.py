import random
import sys
from typing import Tuple

import torch
import torch.nn.functional as F
import time


def per_token_cast_to_fp8_ref(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference matching kernel's C600 path: float -> float16 -> FP8."""
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (128 - (n % 128)) % 128
    x = torch.nn.functional.pad(x, (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    # Match kernel C600 path: scale to [0,448] then round through float16 -> fp8
    scaled = x_view * (448.0 / x_amax.unsqueeze(2))
    scaled_f16 = scaled.float().to(torch.float16).float()
    fp8_data = scaled_f16.to(torch.float8_e4m3fn)
    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)


def benchmark_op(func, args, warmup=5, rep=20):
    """Benchmark a CUDA op, return avg latency (ms)."""
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
    avg_ms = dur.mean().item()

    return avg_ms


def dequantize_fp8(fp8_tensor, scale):
    """Dequantize FP8 E4M3 back to float32 using per-group scale."""
    m, n = fp8_tensor.shape
    scale_expanded = scale.unsqueeze(2).expand(m, n // 128, 128).reshape(m, n)
    return fp8_tensor.float() * scale_expanded * 448.0


def test_per_token_cast_to_fp8(dtype_str, dtype):
    """Test per_token_cast_to_fp8 with given input dtype.

    Validation criteria:
    1. Scale must match reference (atol=1e-6)
    2. Dequantized FP8 output cosine similarity >= 0.999
    """
    torch.manual_seed(42)
    device = "cuda"

    M, N = 6, 3072

    print(f"\n{'='*70}")
    print(f"  Test: per_token_cast_to_fp8  ({dtype_str} -> FP8 E4M3)")
    print(f"  Input shape:  [{M}, {N}], dtype={dtype}")
    print(f"  Scale shape:  [{M}, {N // 128}], dtype=torch.float32")
    print(f"  Output shape: [{M}, {N}], dtype=torch.float8_e4m3fn")
    print(f"{'='*70}")

    x = torch.randn(M, N, dtype=dtype, device=device)

    # Reference
    out_ref, scale_ref = per_token_cast_to_fp8_ref(x)

    # Kernel output
    out = torch.empty(M, N, dtype=torch.float8_e4m3fn, device=device)
    scale = torch.empty(M, N // 128, dtype=torch.float32, device=device)

    torch.ops.sgl_kernel.per_token_cast_to_fp8.default(out, scale, x)

    # ---- Accuracy check: scale ----
    scale_match = torch.allclose(scale_ref, scale, rtol=1e-5, atol=1e-6)
    max_diff_scale = (scale_ref - scale).abs().max().item()
    print(f"\n  Scale accuracy:")
    print(f"    match:     {scale_match}")
    print(f"    max_diff:  {max_diff_scale:.2e}")

    # ---- Accuracy check: FP8 output via cosine similarity ----
    out_dequant = dequantize_fp8(out, scale)
    out_ref_dequant = dequantize_fp8(out_ref, scale_ref)
    x_float = x.float()

    # Cosine similarity: kernel dequant vs original input
    cos_sim_kernel = F.cosine_similarity(
        out_dequant.flatten().unsqueeze(0),
        x_float.flatten().unsqueeze(0),
    ).item()

    # Cosine similarity: kernel dequant vs reference dequant
    cos_sim_vs_ref = F.cosine_similarity(
        out_dequant.flatten().unsqueeze(0),
        out_ref_dequant.flatten().unsqueeze(0),
    ).item()

    # Per-token cosine similarity
    per_token_cos = F.cosine_similarity(out_dequant, x_float, dim=1)
    min_token_cos = per_token_cos.min().item()

    fp8_pass = cos_sim_kernel >= 0.999

    print(f"\n  FP8 quantization accuracy (cosine similarity):")
    print(f"    kernel vs original:    {cos_sim_kernel:.6f}  (threshold: 0.999)")
    print(f"    kernel vs ref_dequant: {cos_sim_vs_ref:.6f}")
    print(f"    min per-token cos:     {min_token_cos:.6f}")
    print(f"    pass: {fp8_pass}")

    # ---- Bandwidth & Latency ----
    def run_kernel(out, scale, x):
        torch.ops.sgl_kernel.per_token_cast_to_fp8.default(out, scale, x)

    avg_ms = benchmark_op(run_kernel, (out, scale, x), warmup=5, rep=20)

    read_bytes = M * N * 2
    write_bytes = M * N * 1 + M * (N // 128) * 4
    total_bytes = read_bytes + write_bytes
    bandwidth_gb_s = total_bytes / (avg_ms * 1e-3) / 1e9

    print(f"\n  Performance:")
    print(f"    avg latency:  {avg_ms:.4f} ms")
    print(f"    read bytes:   {read_bytes}")
    print(f"    write bytes:  {write_bytes}")
    print(f"    total bytes:  {total_bytes}")
    print(f"    bandwidth:    {bandwidth_gb_s:.2f} GB/s")

    # ---- Verdict ----
    passed = scale_match and fp8_pass
    if passed:
        print(f"\n  [PASS] {dtype_str} -> FP8 test passed!")
    else:
        print(f"\n  [FAIL] Test failed!")
        if not scale_match:
            print(f"    Scale mismatch: max_diff={max_diff_scale:.2e}")
        if not fp8_pass:
            print(f"    FP8 cosine similarity {cos_sim_kernel:.6f} < 0.999")

    return passed


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is required")
        exit(1)

    import mcoplib.sgl_kernel

    result_fp16 = test_per_token_cast_to_fp8("FP16", torch.float16)
    result_bf16 = test_per_token_cast_to_fp8("BF16", torch.bfloat16)

    print(f"\n{'='*70}")
    print(f"  Summary:")
    print(f"    FP16 test: {'PASS' if result_fp16 else 'FAIL'}")
    print(f"    BF16 test: {'PASS' if result_bf16 else 'FAIL'}")
    print(f"{'='*70}")

    if not (result_fp16 and result_bf16):
        exit(1)
