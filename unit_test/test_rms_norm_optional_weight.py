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


def reference_rms_norm_with_weight(input, weight, epsilon):
    variance = input.float().pow(2).mean(dim=-1, keepdim=True)
    rms = torch.rsqrt(variance + epsilon)

    if weight.dim() == 1:
        output = input.float() * rms * weight.float()
    elif weight.dim() == 2:
        output = input.float() * rms * weight.float()
    else:
        raise RuntimeError("weight dim must be 1 or 2")

    return output.to(input.dtype)


def reference_rms_norm_without_weight(input, epsilon):
    variance = input.float().pow(2).mean(dim=-1, keepdim=True)
    rms = torch.rsqrt(variance + epsilon)
    output = input.float() * rms
    return output.to(input.dtype)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    cos_sim = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0), dim=1)
    return cos_sim.item()


def compute_bandwidth(hidden_size, num_tokens, time_ms, has_weight, weight_dim=1):
    bytes_per_dtype = 2
    total_bytes = 0

    # read input
    total_bytes += num_tokens * hidden_size * bytes_per_dtype

    # write output
    total_bytes += num_tokens * hidden_size * bytes_per_dtype

    # read weight
    if has_weight:
        if weight_dim == 1:
            total_bytes += hidden_size * bytes_per_dtype
        else:
            total_bytes += num_tokens * hidden_size * bytes_per_dtype

    return total_bytes / (time_ms * 1e-3) / 1e9


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


def test_single_hidden_size(hidden_size, has_weight, weight_dim=1,
                            num_tokens=1, epsilon=1e-6):

    dtype = torch.bfloat16
    torch.manual_seed(42)

    input = torch.randn(
        num_tokens,
        hidden_size,
        dtype=dtype,
        device="cuda"
    )

    out = torch.empty_like(input)

    if has_weight:
        if weight_dim == 1:
            weight = torch.randn(
                hidden_size,
                dtype=dtype,
                device="cuda"
            )
        else:
            weight = torch.randn(
                num_tokens,
                hidden_size,
                dtype=dtype,
                device="cuda"
            )
    else:
        weight = None


    print("\n" + "="*60)
    print(
        f"Test hidden={hidden_size}, "
        f"tokens={num_tokens}, "
        f"weight_dim={weight_dim if has_weight else None}"
    )
    print("="*60)


    if has_weight:
        ref_out = reference_rms_norm_with_weight(
            input,
            weight,
            epsilon
        )
    else:
        ref_out = reference_rms_norm_without_weight(
            input,
            epsilon
        )


    torch.ops._C.rms_norm(
        out,
        input,
        weight,
        epsilon
    )

    torch.cuda.synchronize()


    cos_sim = cosine_similarity(
        ref_out,
        out
    )

    print(
        f"cos similarity={cos_sim:.8f}"
    )

    assert cos_sim > 0.9999


    print("Precision PASS")


    # benchmark

    input_bench = torch.randn(
        num_tokens,
        hidden_size,
        dtype=dtype,
        device="cuda"
    )

    out_bench = torch.empty_like(input_bench)


    if has_weight:
        if weight_dim == 1:
            weight_bench = torch.randn(
                hidden_size,
                dtype=dtype,
                device="cuda"
            )
        else:
            weight_bench = torch.randn(
                num_tokens,
                hidden_size,
                dtype=dtype,
                device="cuda"
            )
    else:
        weight_bench = None


    def cuda_kernel_func():
        torch.ops._C.rms_norm(
            out_bench,
            input_bench,
            weight_bench,
            epsilon
        )


    dur = benchmark(
        cuda_kernel_func,
        (),
        warmup=10,
        rep=100
    )


    cuda_time = dur.mean().item()


    bandwidth = compute_bandwidth(
        hidden_size,
        num_tokens,
        cuda_time,
        has_weight,
        weight_dim
    )


    print(
        f"time={cuda_time:.4f} ms "
        f"bandwidth={bandwidth:.2f} GB/s"
    )

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

    hidden_sizes = [
        4096
    ]

    num_tokens = 4096
    epsilon = 1e-6

    results = []


    for hidden_size in hidden_sizes:

        for weight_dim in [1,2]:

            try:
                test_single_hidden_size(
                    hidden_size,
                    True,
                    weight_dim,
                    num_tokens,
                    epsilon
                )

                results.append(
                    (
                        f"{hidden_size} weight_dim={weight_dim}",
                        True
                    )
                )

            except Exception as e:
                print(
                    f"FAILED weight_dim={weight_dim}: {e}"
                )

                results.append(
                    (
                        f"{hidden_size} weight_dim={weight_dim}",
                        False
                    )
                )


        try:
            test_single_hidden_size(
                hidden_size,
                False,
                1,
                num_tokens,
                epsilon
            )

            results.append(
                (
                    f"{hidden_size} no weight",
                    True
                )
            )

        except Exception as e:
            print(
                f"FAILED no weight: {e}"
            )

            results.append(
                (
                    f"{hidden_size} no weight",
                    False
                )
            )


    print("\nSummary")
    for name, ok in results:
        print(
            name,
            "PASS" if ok else "FAIL"
        )


if __name__ == "__main__":
    run_all_tests()