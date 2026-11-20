# SPDX-License-Identifier: Apache-2.0
"""GPTQ registration/quantizer regressions with per-case GPU performance.

Timing uses warmed-up CUDA Graph replay and CUDA events, not opcheck or
reference/quantization wall time. Effective GB/s counts each packed input
once plus the output (shuffle: one read + one write). It is a logical
I/O metric, not measured HBM traffic: cache reuse, repeated tile loads,
output initialization and internal FP32 scratch traffic are excluded.
"""

import math
from statistics import median

import pytest
import torch

import mcoplib._C  # noqa: F401
from vllm import _custom_ops as ops  # noqa: F401; registers fake implementations


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MetaX GPU is required"
)


def _shuffle_args(weight, bit):
    # mcoplib retains the optional activation-permutation tensor argument.
    if "q_perm" in str(torch.ops._C.gptq_shuffle.default._schema):
        return weight, torch.empty(0, device=weight.device, dtype=torch.int32), bit
    return weight, bit


def _measure_cuda_ms(fn, warmup=10, repeats=20, graph_ops=10, samples=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_ops):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings = []
    for _ in range(samples):
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / (repeats * graph_ops))
    ms = median(timings)
    assert math.isfinite(ms) and ms > 0, f"Invalid CUDA event duration: {ms}"
    return ms


def _print_gemm_perf(case, a, weight, zeros, scales, exllama, v2, bit):
    ms = _measure_cuda_ms(
        lambda: torch.ops._C.gptq_gemm(
            a, weight, zeros, scales, exllama, v2, bit
        )
    )
    read_bytes = sum(t.numel() * t.element_size()
                     for t in (a, weight, zeros, scales))
    write_bytes = a.shape[0] * weight.shape[1] * a.element_size()
    byte_count = read_bytes + write_bytes
    print(
        f"[PERF] {case} op=gptq_gemm dtype={a.dtype} bits={bit} "
        f"exllama={exllama} v2={v2} "
        f"M={a.shape[0]} K={a.shape[1]} N={weight.shape[1]} "
        f"group_size={a.shape[1] // scales.shape[0]} "
        f"GPU_time={ms * 1000:.3f} us ({ms:.6f} ms) "
        f"effective_bandwidth={byte_count / (ms * 1e6):.3f} GB/s "
        f"logical_io_bytes={byte_count} timing=cuda_graph_median",
        flush=True,
    )


def test_gptq_shuffle_opcheck():
    weight = torch.randint(
        -2000000, 2000000, (1792, 4096), device="cuda", dtype=torch.int32
    )
    torch.library.opcheck(torch.ops._C.gptq_shuffle, _shuffle_args(weight, 4))
    torch.cuda.synchronize()
    # Shuffle is in-place. Use disposable storage, never the opcheck input.
    # Its data-independent permutation has identical work on every replay.
    benchmark_weight = weight.clone()
    args = _shuffle_args(benchmark_weight, 4)
    ms = _measure_cuda_ms(lambda: torch.ops._C.gptq_shuffle(*args))
    byte_count = 2 * benchmark_weight.numel() * benchmark_weight.element_size()
    print(
        f"[PERF] test_gptq_shuffle_opcheck op=gptq_shuffle bits=4 "
        f"packed_shape={tuple(benchmark_weight.shape)} "
        f"GPU_time={ms * 1000:.3f} us ({ms:.6f} ms) "
        f"effective_bandwidth={byte_count / (ms * 1e6):.3f} GB/s "
        f"logical_io_bytes={byte_count} timing=cuda_graph_median",
        flush=True,
    )


@pytest.mark.parametrize("use_v2_format", [True, False], ids=["gptqv2", "gptqv1"])
def test_gptq_gemm_opcheck(use_v2_format):
    torch.manual_seed(20261116)
    a = torch.rand((240, 4096), device="cuda", dtype=torch.float16)
    weight = torch.randint(
        -2000000, 2000000, (512, 6144), device="cuda", dtype=torch.int32
    )
    # Registration coverage deliberately stays small; numerical/flag/projection
    # sweeps live in test_op_vllm_gptq_gemm.py. Use varying packed zeros so the
    # two format registrations do not benchmark an all-zero special case.
    zeros = torch.randint(-(1 << 31), 1 << 31, (32, 768), device="cuda", dtype=torch.int32)
    scales = torch.rand((32, 6144), device="cuda", dtype=torch.float16)
    torch.library.opcheck(
        torch.ops._C.gptq_gemm,
        (a, weight, zeros, scales, True, use_v2_format, 4),
        # Split-K FP16 atomics can differ by a few ulps across AOT launches.
        rtol=1e-3,
        atol=1e-3,
    )
    torch.cuda.synchronize()
    _print_gemm_perf(
        "test_gptq_gemm_opcheck", a, weight, zeros, scales, True, use_v2_format, 4
    )


@pytest.mark.parametrize("exllama", [False, True])
def test_gptq_gemm_vllm_quantizer(exllama):
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        gptq_quantize_weights,
    )

    torch.manual_seed(0)
    k, n, group_size = 256, 320, 128
    w = torch.randn(k, n, device="cuda", dtype=torch.float16) * 2
    w_ref, quantized, scales = gptq_quantize_weights(w, scalar_types.uint4b8, group_size)
    # The quantizer returns unpacked integers in [K, N] order.
    shifts = torch.arange(8, device="cuda", dtype=torch.int64).reshape(1, 8, 1) * 4
    qweight = (((quantized.to(torch.int64).reshape(k // 8, 8, n) & 15)
                << shifts).sum(dim=1)).to(torch.int32).contiguous()
    qzeros = torch.full((k // group_size, n // 8), 0x77777777,
                        device="cuda", dtype=torch.int32)
    if exllama:
        torch.ops._C.gptq_shuffle(*_shuffle_args(qweight, 4))
    a = torch.randn(1, k, device="cuda", dtype=torch.float16)
    actual = torch.ops._C.gptq_gemm(a, qweight, qzeros, scales, exllama, False, 4)
    # The native dequantizer rounds each scaled weight to the input dtype.
    expected = a.cpu().double() @ w_ref.to(a.dtype).cpu().double()
    actual_cpu = actual.cpu().double()
    error = (actual_cpu - expected).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(actual_cpu.reshape(-1), expected.reshape(-1), dim=0).item()
    print(f"[gptq_gemm] vllm_quantizer exllama={exllama} max_abs_err={error} cosine={cosine:.9f}")
    assert math.isfinite(cosine) and cosine >= 0.99999
    torch.testing.assert_close(actual_cpu, expected, atol=0.06, rtol=0.03)
    _print_gemm_perf(
        "test_gptq_gemm_vllm_quantizer", a, qweight, qzeros, scales, exllama, False, 4
    )


if __name__ == "__main__":
    import sys
    raise SystemExit(pytest.main([__file__, "-q", "-s", *sys.argv[1:]]))
