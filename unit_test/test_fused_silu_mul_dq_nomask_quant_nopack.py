import torch
from typing import Optional, Tuple
import torch.nn.functional as F
import time
import math

import mcoplib.sgl_kernel


def ref_swiglu(x, limit):
    gate, up = x.chunk(2, dim=-1)
    gate = F.silu(gate)
    if limit is not None:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    return gate * up


def scaled_int8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    azp: Optional[torch.Tensor] = None,
    symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    output = torch.empty_like(input, dtype=torch.int8)
    if scale is not None:
        assert symmetric == (azp is None), \
            "azp must only be provided for asymmetric quantization."
        torch.ops.sgl_kernel.static_scaled_int8_quant.default(
            output, input, scale, azp)
        return output, scale, azp

    input_scales = torch.empty(
        (input.numel() // input.shape[-1], 1),
        device=input.device,
        dtype=torch.float32)
    input_azp = None if symmetric else torch.empty_like(
        input_scales, dtype=torch.int32)
    torch.ops.sgl_kernel.dynamic_scaled_int8_quant.default(
        output, input.contiguous(), input_scales, input_azp)
    return output, input_scales, input_azp


def ref_fused_swiglu_dq_quant(
    x: torch.Tensor,
    limit: Optional[float] = None,
    weight: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    gate_up = ref_swiglu(x, limit)
    if weight is not None:
        gate_up = gate_up * weight
    output, scale, _ = scaled_int8_quant(gate_up)
    return output, scale


def fused_silu_mul_dq_nomask_quant_nopack_torch(
    x: torch.Tensor,
    limit: Optional[float] = None,
    weight: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty(
        (x.shape[0], x.shape[1] // 2),
        device=x.device, dtype=torch.int8)
    scale = torch.empty(
        (x.shape[0], 1),
        device=x.device, dtype=torch.float32)
    torch.ops.sgl_kernel.fused_silu_mul_dq_nomask_quant_nopack.default(
        output, scale, x, limit, weight)
    return output, scale


def compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_float = a.float().flatten()
    b_float = b.float().flatten()
    dot = torch.dot(a_float, b_float)
    norm_a = torch.norm(a_float)
    norm_b = torch.norm(b_float)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return (dot / (norm_a * norm_b)).item()


def compute_bandwidth(
    seq_len: int,
    hidden_size: int,
    latency_ms: float
) -> float:
    """Compute effective bandwidth in GB/s for the fused kernel.

    The kernel reads input [seq_len, hidden_size] (bf16),
    writes out [seq_len, hidden_size//2] (int8),
    and writes scale [seq_len, 1] (float32).
    """
    input_bytes = seq_len * hidden_size * 2  # bf16 = 2 bytes
    output_bytes = seq_len * (hidden_size // 2) * 1  # int8 = 1 byte
    scale_bytes = seq_len * 1 * 4  # float32 = 4 bytes
    total_bytes = input_bytes + output_bytes + scale_bytes
    bandwidth_gbs = total_bytes / (latency_ms * 1e-3) / 1e9
    return bandwidth_gbs


def run_test(
    hidden_size: int,
    seq_len: int,
    limit: Optional[float],
    weight: Optional[torch.Tensor] = None,
    warmup_iters: int = 10,
    bench_iters: int = 100
):
    x = torch.randn(seq_len, hidden_size, device="cuda", dtype=torch.bfloat16)

    # ---- correctness test ----
    cuda_out, cuda_scale = fused_silu_mul_dq_nomask_quant_nopack_torch(
        x, limit, weight)

    # reference: dequantize both to bf16 for comparison
    ref_out, ref_scale = ref_fused_swiglu_dq_quant(x, limit, weight)

    # dequantize: float_val = int8_val * scale / 127
    cuda_deq = cuda_out.float() * cuda_scale.float() / 127.0
    ref_deq = ref_out.float() * ref_scale.float() / 127.0

    cosine_sim = compute_cosine_similarity(cuda_deq, ref_deq)

    # scale comparison (per-token absmax)
    scale_cosine = compute_cosine_similarity(cuda_scale, ref_scale)

    limit_str = f"{limit}" if limit is not None else "None"
    weight_str = "Yes" if weight is not None else "No"

    passed = cosine_sim > 0.999
    status = "PASS" if passed else "FAIL"

    print(f"[{status}] hidden_size={hidden_size}, seq_len={seq_len}, "
          f"limit={limit_str}, weight={weight_str} | "
          f"cosine_sim={cosine_sim:.6f} (threshold=0.999), "
          f"scale_cosine={scale_cosine:.6f}")

    if not passed:
        print(f"  ERROR: cosine similarity {cosine_sim:.6f} < 0.999")
        # Print some debug info
        print(f"  cuda_scale sample: {cuda_scale[:5].flatten()}")
        print(f"  ref_scale sample: {ref_scale[:5].flatten()}")
        print(f"  cuda_deq sample: {cuda_deq[0, :10]}")
        print(f"  ref_deq sample: {ref_deq[0, :10]}")

    # ---- performance benchmark ----
    # Warmup
    for _ in range(warmup_iters):
        fused_silu_mul_dq_nomask_quant_nopack_torch(x, limit, weight)
    torch.cuda.synchronize()

    # CUDA kernel timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(bench_iters):
        fused_silu_mul_dq_nomask_quant_nopack_torch(x, limit, weight)
    end.record()
    torch.cuda.synchronize()
    cuda_latency_ms = start.elapsed_time(end) / bench_iters

    # Reference timing
    for _ in range(warmup_iters):
        ref_fused_swiglu_dq_quant(x, limit, weight)
    torch.cuda.synchronize()

    start.record()
    for _ in range(bench_iters):
        ref_fused_swiglu_dq_quant(x, limit, weight)
    end.record()
    torch.cuda.synchronize()
    ref_latency_ms = start.elapsed_time(end) / bench_iters

    speedup = ref_latency_ms / cuda_latency_ms if cuda_latency_ms > 0 else float('inf')
    bandwidth = compute_bandwidth(seq_len, hidden_size, cuda_latency_ms)

    print(f"  Perf: CUDA={cuda_latency_ms:.4f}ms, Ref={ref_latency_ms:.4f}ms, "
          f"Speedup={speedup:.2f}x, Bandwidth={bandwidth:.2f} GB/s")

    return passed


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    all_passed = True
    total_tests = 0
    passed_tests = 0

    hidden_sizes = [4096, 5120, 6144, 7168]
    seq_lens = [1, 16, 32, 128, 4096, 8192]
    test_configs = []

    # Test 1: No clamp, no weight
    for hs in hidden_sizes:
        for sl in seq_lens:
            test_configs.append((hs, sl, None, None))

    # Test 2: With clamp (limit=10.0), no weight
    for hs in hidden_sizes:
        for sl in seq_lens:
            test_configs.append((hs, sl, 10.0, None))

    # Test 3: No clamp, with weight
    for hs in hidden_sizes:
        for sl in [1, 32, 4096]:
            w = torch.randn(sl, hs // 2, device="cuda", dtype=torch.bfloat16)
            test_configs.append((hs, sl, None, w))

    # Test 4: With clamp and weight
    for hs in hidden_sizes:
        for sl in [1, 32, 4096]:
            w = torch.randn(sl, hs // 2, device="cuda", dtype=torch.bfloat16)
            test_configs.append((hs, sl, 10.0, w))

    print("=" * 80)
    print("fused_silu_mul_dq_nomask_quant_nopack Unit Test")
    print("=" * 80)
    print()

    for hs, sl, limit, weight in test_configs:
        total_tests += 1
        try:
            passed = run_test(hs, sl, limit, weight)
            if passed:
                passed_tests += 1
            else:
                all_passed = False
        except Exception as e:
            print(f"[FAIL] hidden_size={hs}, seq_len={sl}, "
                  f"limit={limit}, weight={'Yes' if weight is not None else 'No'} | "
                  f"Exception: {e}")
            all_passed = False

    print()
    print("=" * 80)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
