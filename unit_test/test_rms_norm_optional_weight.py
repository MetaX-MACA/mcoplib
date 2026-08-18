"""
Unit test for rms_norm CUDA kernel with optional weight parameter.
Tests both scenarios: with weight and without weight.
Tests hidden_size: 7168, 5120, 6144, 4096.
"""

import torch
import torch.nn.functional as F
import math
import time

# Import the CUDA kernel
import mcoplib._C


def reference_rms_norm_with_weight(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float
) -> torch.Tensor:
    """
    Reference implementation of RMS norm with weight in PyTorch.

    Args:
        input: [num_tokens, hidden_size] - Input tensor
        weight: [hidden_size] - RMS norm weight
        epsilon: RMS norm epsilon

    Returns:
        output: [num_tokens, hidden_size] - RMS norm output
    """
    # Compute variance: mean(x^2)
    variance = input.float().pow(2).mean(dim=-1, keepdim=True)
    # RMS = sqrt(variance + epsilon)
    rms = torch.sqrt(variance + epsilon)
    # Normalized = x / rms * weight
    output = (input.float() / rms) * weight.float()
    return output.to(input.dtype)


def reference_rms_norm_without_weight(
    input: torch.Tensor,
    epsilon: float
) -> torch.Tensor:
    """
    Reference implementation of RMS norm without weight in PyTorch.
    Formula: q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)

    Args:
        input: [num_tokens, hidden_size] - Input tensor
        epsilon: RMS norm epsilon

    Returns:
        output: [num_tokens, hidden_size] - RMS norm output
    """
    # Compute variance: mean(x^2)
    variance = input.float().pow(2).mean(dim=-1, keepdim=True)
    # RMS = rsqrt(variance + epsilon)
    rms = torch.rsqrt(variance + epsilon)
    # Normalized = x * rms (no weight)
    output = input.float() * rms
    return output.to(input.dtype)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    cos_sim = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0), dim=1)
    return cos_sim.item()


def compute_bandwidth(hidden_size: int, num_tokens: int, time_ms: float, has_weight: bool) -> float:
    """
    Compute kernel bandwidth in GB/s.

    Total data read/written:
    - Read: input (bf16), weight (bf16) if present
    - Write: output (bf16)

    Each bf16 = 2 bytes
    """
    bytes_per_bf16 = 2

    total_bytes = 0

    # Reads
    total_bytes += num_tokens * hidden_size * bytes_per_bf16  # input
    if has_weight:
        total_bytes += hidden_size * bytes_per_bf16  # weight (shared across tokens)

    # Writes
    total_bytes += num_tokens * hidden_size * bytes_per_bf16  # output

    bandwidth = total_bytes / (time_ms * 1e-3) / 1e9  # GB/s
    return bandwidth


def benchmark(func, args, warmup=10, rep=100):
    """Benchmark function execution time."""
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()

    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        start_event[i].record()
        func(*args)
        end_event[i].record()

    torch.cuda.synchronize()
    durations = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float,
    )
    return durations


def test_single_hidden_size(hidden_size: int, has_weight: bool, num_tokens: int = 1, epsilon: float = 1e-6):
    """Test a single hidden_size configuration."""
    dtype = torch.bfloat16
    torch.manual_seed(42)

    # Create input tensors
    input = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
    out = torch.empty_like(input)

    if has_weight:
        weight = torch.randn(hidden_size, dtype=dtype, device="cuda")
    else:
        weight = None

    print(f"\n{'='*60}")
    print(f"Test: hidden_size={hidden_size}, has_weight={has_weight}, num_tokens={num_tokens}")
    print(f"{'='*60}")

    # Get reference results
    if has_weight:
        ref_out = reference_rms_norm_with_weight(input, weight, epsilon)
    else:
        ref_out = reference_rms_norm_without_weight(input, epsilon)

    # Call CUDA kernel
    torch.ops._C.rms_norm(out, input, weight, epsilon)

    torch.cuda.synchronize()

    # Verify precision using cosine similarity
    cos_sim = cosine_similarity(ref_out, out)
    print(f"Output cosine similarity: {cos_sim:.8f}")

    # Assert precision requirements
    assert cos_sim > 0.9999, f"Cosine similarity {cos_sim} < 0.9999"
    assert not math.isnan(cos_sim), "Cosine similarity is NaN"

    print("Precision verification PASSED!")

    # Performance benchmark
    # Create fresh tensors for benchmarking
    torch.manual_seed(42)
    input_bench = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
    out_bench = torch.empty_like(input_bench)
    weight_bench = torch.randn(hidden_size, dtype=dtype, device="cuda") if has_weight else None

    def cuda_kernel_func():
        torch.ops._C.rms_norm(out_bench, input_bench, weight_bench, epsilon)

    # Benchmark CUDA kernel
    dur_cuda = benchmark(cuda_kernel_func, (), warmup=10, rep=100)
    cuda_time_ms = dur_cuda.mean().item()
    print(f"CUDA kernel time: {cuda_time_ms:.4f} ms")

    # Benchmark PyTorch reference
    def torch_reference_func():
        if has_weight:
            reference_rms_norm_with_weight(input_bench, weight_bench, epsilon)
        else:
            reference_rms_norm_without_weight(input_bench, epsilon)

    dur_torch = benchmark(torch_reference_func, (), warmup=10, rep=100)
    torch_time_ms = dur_torch.mean().item()
    print(f"PyTorch reference time: {torch_time_ms:.4f} ms")

    # Performance ratio
    perf_ratio = torch_time_ms / cuda_time_ms
    print(f"Performance ratio (torch/cuda): {perf_ratio:.2f}x")

    # Compute bandwidth
    bandwidth = compute_bandwidth(hidden_size, num_tokens, cuda_time_ms, has_weight)
    print(f"CUDA kernel bandwidth: {bandwidth:.2f} GB/s")

    return True


def test_without_weight_param():
    """
    Test rms_norm kernel WITHOUT passing weight parameter at all.
    This tests that the weight parameter is truly optional.
    """
    print("\n" + "="*60)
    print("Test: rms_norm WITHOUT weight parameter (passing None)")
    print("="*60)

    hidden_size = 4096
    num_tokens = 1
    epsilon = 1e-6
    dtype = torch.bfloat16

    torch.manual_seed(42)
    input = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
    out = torch.empty_like(input)

    # Get reference results (without weight)
    ref_out = reference_rms_norm_without_weight(input, epsilon)

    # Call CUDA kernel with weight=None
    torch.ops._C.rms_norm(out, input, None, epsilon)

    torch.cuda.synchronize()

    # Verify precision
    cos_sim = cosine_similarity(ref_out, out)
    print(f"Output cosine similarity: {cos_sim:.8f}")

    assert cos_sim > 0.9999, f"Cosine similarity {cos_sim} < 0.9999"
    assert not math.isnan(cos_sim), "Cosine similarity is NaN"

    print("Test PASSED: Precision requirements met!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running rms_norm unit tests with optional weight")
    print("="*60)

    hidden_sizes = [7168, 5120, 6144, 4096]
    num_tokens = 1
    epsilon = 1e-6

    results = []

    for hidden_size in hidden_sizes:
        # Test with weight
        try:
            test_name = f"hidden_size={hidden_size}, with weight"
            test_single_hidden_size(hidden_size, has_weight=True, num_tokens=num_tokens, epsilon=epsilon)
            results.append((test_name, True))
        except Exception as e:
            print(f"Test FAILED: {e}")
            results.append((f"hidden_size={hidden_size}, with weight", False))

        # Test without weight
        try:
            test_name = f"hidden_size={hidden_size}, without weight"
            test_single_hidden_size(hidden_size, has_weight=False, num_tokens=num_tokens, epsilon=epsilon)
            results.append((test_name, True))
        except Exception as e:
            print(f"Test FAILED: {e}")
            results.append((f"hidden_size={hidden_size}, without weight", False))

    # Test passing None explicitly
    try:
        results.append(("Test without weight param (None)", test_without_weight_param()))
    except Exception as e:
        print(f"Test FAILED: {e}")
        results.append(("Test without weight param (None)", False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")

    return all_passed


if __name__ == "__main__":
    run_all_tests()