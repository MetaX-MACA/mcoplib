# SPDX-License-Identifier: Apache-2.0
"""Standalone correctness and logical-bandwidth tests for GPTQ GEMM.

Full Qwen72B TP1 BF16 projections are opt-in with MCOP_GPTQ_QWEN_FULL=1;
MCOP_GPTQ_QWEN_M=1,2,16,64 chooses batch sizes. Use pytest -k to select
projections/flags. All accuracy checks require cosine >= .99999, computed
in CPU FP64 against input-dtype-rounded dequantized weights. Effective GB/s
counts packed inputs once plus output, excludes scratch/internal traffic,
and measures warm-cache CUDA Graph replay rather than physical HBM traffic.
"""

import math
import os
from statistics import median

import pytest
import torch
import torch.nn.functional as F

import mcoplib
import mcoplib._C  # noqa: F401

DEVICE = "cuda"
BITS = (2, 3, 4, 8)
DTYPES = (torch.float16, torch.bfloat16)


def _pack(values, bits, use_exllama=False):
    """Pack whole blocks at once; the 3-bit loop has 32 iterations, never K."""
    k, n = values.shape
    mask = (1 << bits) - 1
    if bits == 3:
        assert k % 32 == 0
        values = values.to(torch.int64).reshape(k // 32, 32, n) & mask
        packed = torch.zeros((k // 32, 3, n), device=values.device, dtype=torch.int64)
        for lane in range(32):
            value = values[:, lane]
            if use_exllama:
                if lane < 30:
                    packed[:, lane // 10] |= value << ((lane % 10 // 2) * 3 + lane % 2 * 16)
                else:
                    for word in range(3):
                        packed[:, word] |= ((value >> word) & 1) << (15 if lane == 30 else 31)
            else:
                offset = lane * 3
                packed[:, offset // 32] |= (value << (offset % 32)) & 0xffffffff
                if offset % 32 > 29:
                    packed[:, offset // 32 + 1] |= value >> (32 - offset % 32)
        return packed.reshape(-1, n).to(torch.int32).contiguous()
    rows_per_word = 32 // bits
    assert k % rows_per_word == 0
    lane = torch.arange(rows_per_word, device=values.device)
    shift = ((lane // 2) * bits + (lane % 2) * 16
             if use_exllama and bits != 8 else lane * bits)
    values = values.to(torch.int64).reshape(-1, rows_per_word, n) & mask
    return (values << shift[None, :, None]).sum(dim=1).to(torch.int32).contiguous()


def _pack_zeros(zeros, bits):
    return _pack(zeros.T.contiguous(), bits).T.contiguous()


def _case(bits, dtype, exllama, v2, group_size, k, n, m, device):
    torch.manual_seed(2026 + bits + k)
    qmax = (1 << bits) - 1
    groups = k // group_size
    q = torch.randint(qmax + 1, (k, n), device=device, dtype=torch.int32)
    z = torch.randint(qmax + 1, (groups, n), device=device, dtype=torch.int32)
    scales = (torch.rand(groups, n, device=device) * 0.05 + 0.01).to(dtype)
    stored_z = z if v2 else (z - 1) % (qmax + 1)
    qweight = _pack(q, bits, exllama)
    qzeros = _pack_zeros(stored_z, bits)
    group = torch.arange(k, device=device) // group_size
    decoded_zero = stored_z + (0 if v2 else 1)
    w = (q.float() - decoded_zero[group].float()) * scales[group].float()
    a = torch.randn(m, k, device=device, dtype=dtype)
    return a, qweight, qzeros, scales, w.to(dtype)


def _run(a, qw, qz, scales, exllama, v2, bits):
    return torch.ops._C.gptq_gemm(a, qw, qz, scales, exllama, v2, bits)


def _reference(a, w):
    # W is rounded to the input dtype, matching native dequantization. CPU FP64
    # accumulation avoids backend TF32/reduced-precision GPU BLAS modes.
    return a.detach().cpu().double() @ w.detach().cpu().double()


def _cosine(x, y):
    return F.cosine_similarity(x.detach().cpu().double().reshape(-1),
                               y.detach().cpu().double().reshape(-1), dim=0).item()


def _check(actual, expected):
    actual = actual.detach().cpu().double()
    expected = expected.detach().cpu().double()
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    nonzero = expected.norm(dim=-1) > 0
    if nonzero.any():
        cosine = _cosine(actual[nonzero], expected[nonzero])
        row_cosine = F.cosine_similarity(actual[nonzero], expected[nonzero], dim=-1).min().item()
        assert min(cosine, row_cosine) >= 0.99999, (
            f"cosine={cosine:.9f}, minimum_row_cosine={row_cosine:.9f}")
    else:
        cosine = 1.0
    if (~nonzero).any():
        torch.testing.assert_close(actual[~nonzero], expected[~nonzero], atol=0.06, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0.06, rtol=0.03)
    return cosine


def _check_graph(a, qw, qz, scales, w, exllama, v2, bits):
    # Select a single parametrized node in a fresh pytest process for cold
    # capture. The reference never initializes a GPU BLAS handle.
    original = a.clone()
    expected = _reference(a, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run(a, qw, qz, scales, exllama, v2, bits)
    assert captured.dtype == a.dtype and captured.device == a.device
    try:
        for factor in (1.0, -0.5, 2.0, 1.0):
            a.copy_(original * factor)
            graph.replay()
            _check(captured, expected * factor)
    finally:
        a.copy_(original)


def _measure(fn, warmup=10, repeats=20, graph_ops=10, samples=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_ops):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
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


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("exllama", [False, True])
@pytest.mark.parametrize("v2", [False, True])
def test_gptq_gemm_correctness_and_graph(bits, dtype, exllama, v2):
    if not torch.cuda.is_available():
        pytest.skip("MetaX GPU is required")
    device = torch.device(DEVICE)
    group_size = 128
    k = 256
    n = 256
    m = 1
    a, qw, qz, scales, w = _case(bits, dtype, exllama, v2, group_size, k, n, m, device)
    _check_graph(a, qw, qz, scales, w, exllama, v2, bits)
    actual = _run(a, qw, qz, scales, exllama, v2, bits)
    expected = _reference(a, w)
    cosine = _check(actual, expected)

    ms = _measure(lambda: _run(a, qw, qz, scales, exllama, v2, bits))
    byte_count = a.numel() * a.element_size() + qw.numel() * qw.element_size()
    byte_count += qz.numel() * qz.element_size() + scales.numel() * scales.element_size()
    byte_count += actual.numel() * actual.element_size()
    print(f"[gptq_gemm] dtype={dtype} bits={bits} exllama={exllama} v2={v2} "
          f"shape=({m},{k})x({k},{n}) cos={cosine:.8f} {ms:.6f} ms "
          f"effective_logical={byte_count / (ms * 1e6):.1f} GB/s timing=cuda_graph_median cache=warm", flush=True)


def test_gptq_gemm_larger_m_and_cuda_graph():
    if not torch.cuda.is_available():
        pytest.skip("MetaX GPU is required")
    dtype, bits, k, n, m, group_size = torch.float16, 4, 512, 256, 4, 128
    a, qw, qz, scales, w = _case(bits, dtype, False, False, group_size, k, n, m, torch.device(DEVICE))
    _check_graph(a, qw, qz, scales, w, False, False, bits)
    expected = _reference(a, w)
    actual = _run(a, qw, qz, scales, False, False, bits)
    cosine = _check(actual, expected)
    print(f"[gptq_gemm] larger_m cos={cosine:.8f}", flush=True)


@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("k", [256, 384, 1024])
def test_gptq_gemm_fast_path_and_cold_graph(m, k):
    if not torch.cuda.is_available():
        pytest.skip("MetaX GPU is required")
    a, qw, qz, scales, w = _case(4, torch.float16, True, False, 128, k, 1024, m, torch.device(DEVICE))
    _check_graph(a, qw, qz, scales, w, True, False, 4)
    cosine = _check(_run(a, qw, qz, scales, True, False, 4), _reference(a, w))
    ms = _measure(lambda: _run(a, qw, qz, scales, True, False, 4))
    byte_count = sum(t.numel() * t.element_size() for t in (a, qw, qz, scales)) + m * 1024 * a.element_size()
    print(f"[gptq_gemm] fast shape=({m},{k},1024) cos={cosine:.8f} {ms:.6f} ms "
          f"effective_logical={byte_count / (ms * 1e6):.1f} GB/s timing=cuda_graph_median cache=warm", flush=True)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m,k,n,group_size,exllama,v2", [
    (3, 96, 160, 32, False, False),
    (17, 256, 96, 64, True, True),
    (64, 256, 320, 256, False, True),
    (2, 32, 64, 1, True, False),
    (3, 192, 160, 48, False, True),
    (3, 64, 96, 16, True, True),
])
def test_gptq_gemm_shapes_and_groups(bits, dtype, m, k, n, group_size, exllama, v2):
    if not torch.cuda.is_available():
        pytest.skip("MetaX GPU is required")
    a, qw, qz, scales, w = _case(bits, dtype, exllama, v2, group_size, k, n, m, torch.device(DEVICE))
    # Compute this small reference on the CPU, independently of the GPU BLAS.
    expected = _reference(a, w)
    cosine = _check(_run(a, qw, qz, scales, exllama, v2, bits), expected)
    ms = _measure(lambda: _run(a, qw, qz, scales, exllama, v2, bits))
    byte_count = sum(t.numel() * t.element_size() for t in (a, qw, qz, scales)) + m * n * a.element_size()
    print(f"[gptq_gemm] dtype={dtype} bits={bits} exllama={exllama} v2={v2} "
          f"shape=({m},{k},{n}) group_size={group_size} cos={cosine:.8f} {ms:.6f} ms "
          f"effective_logical={byte_count / (ms * 1e6):.1f} GB/s timing=cuda_graph_median cache=warm", flush=True)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("v2", [False, True])
def test_gptq_gemm_zero_point_boundaries(bits, v2):
    if not torch.cuda.is_available():
        pytest.skip("MetaX GPU is required")
    device = torch.device(DEVICE)
    mask = (1 << bits) - 1
    q = (torch.arange(64 * 32, device=device).reshape(64, 32) % (mask + 1)).int()
    stored = torch.tensor([0, mask], device=device, dtype=torch.int32).repeat(2, 16).reshape(2, 32)
    scales = torch.full((2, 32), 0.125, device=device, dtype=torch.float16)
    w = ((q.float() - (stored + (0 if v2 else 1)).repeat_interleave(32, dim=0)) * 0.125).half()
    a = torch.randn(3, 64, device=device, dtype=torch.float16)
    for exllama in (False, True):
        actual = _run(a, _pack(q, bits, exllama), _pack_zeros(stored, bits), scales, exllama, v2, bits)
        _check(actual, _reference(a, w))


def test_gptq_gemm_empty_batch_and_invalid_inputs():
    if not torch.cuda.is_available():
        pytest.skip("MetaX GPU is required")
    a, qw, qz, scales, _ = _case(4, torch.float16, False, False, 32, 64, 96, 0, torch.device(DEVICE))
    assert _run(a, qw, qz, scales, False, False, 4).shape == (0, 96)
    with pytest.raises(RuntimeError, match="bit"):
        _run(a, qw, qz, scales, False, False, 5)
    with pytest.raises(RuntimeError, match="packed int32"):
        _run(a, qw.long(), qz, scales, False, False, 4)
    with pytest.raises(RuntimeError, match="match a's dtype"):
        _run(a, qw, qz, scales.bfloat16(), False, False, 4)
    with pytest.raises(RuntimeError, match="packed weight/zero point shapes"):
        _run(a, qw[:-1], qz, scales, False, False, 4)
    invalid = [
        ((a.float(), qw, qz, scales), "FP16 or BF16"),
        ((a, qw, qz.long(), scales), "packed int32"),
        ((a, qw.T, qz, scales), "contiguous matrices"),
        ((a, qw.cpu(), qz, scales), "on a's device"),
        ((a, qw, qz.unsqueeze(0), scales), "contiguous matrices"),
        ((a, qw, qz, scales.T), "contiguous matrices"),
        ((a, qw, qz[:, :-1].contiguous(), scales), "packed weight/zero point shapes"),
        ((a, qw, qz, scales[:, :-1].contiguous()), "scale groups"),
        ((a[:, :-1].contiguous(), qw, qz, scales), "positive multiples of 32"),
        ((a, qw[:, :-1].contiguous(), qz, scales), "positive multiples of 32"),
        ((a, qw, qz[:0], scales[:0]), "scale groups"),
        ((a, qw, qz[:1].repeat(3, 1), scales[:1].repeat(3, 1)), "scale groups"),
    ]
    for inputs, message in invalid:
        with pytest.raises(RuntimeError, match=message):
            _run(*inputs, False, False, 4)
    for dtype in DTYPES:
        empty = _run(a.to(dtype), qw, qz, scales.to(dtype), False, False, 4)
        assert empty.shape == (0, 96) and empty.dtype == dtype and empty.device == a.device


@pytest.mark.parametrize("bits", [2, 3])
@pytest.mark.parametrize("v2", [False, True])
def test_gptq_gemm_small_kernel_fp32_accumulation(bits, v2):
    if not torch.cuda.is_available():
        pytest.skip("MetaX GPU is required")
    device = torch.device(DEVICE)
    k, n = 256, 64
    q = torch.full((k, n), 3, device=device, dtype=torch.int32)
    stored = torch.zeros((2, n), device=device, dtype=torch.int32)
    scales = torch.full((2, n), 2 ** -10, device=device, dtype=torch.float16)
    a = torch.full((1, k), 32768, device=device, dtype=torch.float16)
    w = ((q.float() - (0 if v2 else 1)) * 2 ** -10).half()
    actual = _run(a, _pack(q, bits, True), _pack_zeros(stored, bits),
                  scales, True, v2, bits)
    expected = a.cpu().float() @ w.cpu().float()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu().float(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("exllama,v2", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("m,k,n,group_size", [
    (15, 96, 160, 32),
    (16, 256, 320, 64),
    (17, 256, 96, 128),
    (31, 256, 96, 64),
    (32, 256, 96, 64),
    (33, 256, 96, 64),
    (47, 256, 96, 64),
    (48, 256, 96, 64),
    (49, 256, 96, 64),
    (63, 256, 320, 256),
    (64, 256, 96, 64),
    (65, 384, 160, 128),
    (129, 256, 320, 64),
    (240, 256, 6144, 128),
])
def test_gptq_gemm_native_boundaries(bits, dtype, exllama, v2, m, k, n, group_size):
    if not torch.cuda.is_available():
        pytest.skip("MetaX GPU is required")
    a, qw, qz, scales, w = _case(
        bits, dtype, exllama, v2, group_size, k, n, m, torch.device(DEVICE))
    _check_graph(a, qw, qz, scales, w, exllama, v2, bits)
    expected = _reference(a, w)
    cosine = _check(_run(a, qw, qz, scales, exllama, v2, bits), expected)
    print(f"[gptq_native] dtype={dtype} bits={bits} exllama={exllama} v2={v2} "
          f"shape=({m},{k},{n}) group_size={group_size} cos={cosine:.9f}", flush=True)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("exllama,v2", [(False, False), (False, True), (True, False), (True, True)])
def test_gptq_gemm_split_k_fp32_cancellation(bits, exllama, v2):
    if not torch.cuda.is_available():
        pytest.skip("MetaX GPU is required")
    device = torch.device(DEVICE)
    m, k, n, group_size = 64, 256, 96, 64
    zero = 1 << (bits - 1)
    q = torch.full((k, n), zero, device=device, dtype=torch.int32)
    q[:group_size] += 1
    q[2 * group_size:3 * group_size] -= 1
    stored = torch.full((k // group_size, n), zero - (0 if v2 else 1),
                        device=device, dtype=torch.int32)
    scales = torch.ones(k // group_size, n, device=device, dtype=torch.float16)
    a = torch.full((m, k), 1024, device=device, dtype=torch.float16)
    qw, qz = _pack(q, bits, exllama), _pack_zeros(stored, bits)
    w = (q - zero).half()
    # The +65536/-65536 split-K partials overflow half; their sum is finite.
    expected = a.cpu().float() @ w.cpu().float()
    actual = _run(a, qw, qz, scales, exllama, v2, bits)
    torch.testing.assert_close(actual.cpu().float(), expected, atol=0, rtol=0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run(a, qw, qz, scales, exllama, v2, bits)
    for factor in (1.0, -0.5, 2.0):
        a.mul_(factor)
        graph.replay()
        expected = a.cpu().float() @ w.cpu().float()
        torch.testing.assert_close(captured.cpu().float(), expected, atol=0, rtol=0)


# Real TP1 projections, expressed as A[M,K] @ W[K,N]. Q and output share
# a shape; K and V share a shape. Full cases are opt-in because 2/3-bit BF16
# intentionally exercises a slow grouped fallback on some builds.
QWEN72B_PROJECTIONS = {
    "q_o": (8192, 8192),
    "kv": (8192, 1024),
    "qkv": (8192, 10240),
    "gate_up_single": (8192, 29568),
    "gate_up_fused": (8192, 59136),
    "down": (29568, 8192),
}


def make_qwen_case(k, n, m=1, bits=4, exllama=True, v2=False,
                   dtype=torch.bfloat16, group_size=128, device=DEVICE,
                   zero_mode="asymmetric", seed=20261116):
    """Deterministic packed tensors plus an independent tiled CPU FP64 oracle.

    Same seed/shape produces identical logical weights for both layouts and
    formats. zero_mode='symmetric' is for the old fixed-zero BF16 baseline;
    correctness tests use channel/group-varying asymmetric zeros. No GPU BLAS
    or RNG is used, so the helper is safe before a process-cold capture.
    """
    assert bits in BITS and k % 32 == n % 32 == 0 and k % group_size == 0
    assert zero_mode in ("asymmetric", "symmetric")
    groups, mask = k // group_size, (1 << bits) - 1
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a_cpu = torch.randn(m, k, generator=generator).to(dtype)
    qw = torch.empty(k // 32 * bits, n, device=device, dtype=torch.int32)
    qz = torch.empty(groups, n // 32 * bits, device=device, dtype=torch.int32)
    scales = torch.empty(groups, n, device=device, dtype=dtype)
    expected = torch.empty(m, n, dtype=torch.float64)
    group = torch.arange(k) // group_size
    a64 = a_cpu.double()
    for start in range(0, n, 256):
        end = min(n, start + 256)
        width = end - start
        tile_generator = torch.Generator(device="cpu").manual_seed(seed + 1 + start)
        quant = torch.randint(mask + 1, (k, width), generator=tile_generator, dtype=torch.int32)
        if zero_mode == "symmetric":
            zero = torch.full((groups, width), 1 << (bits - 1), dtype=torch.int32)
        else:
            # Interior zero points have exactly the same decoded value in v1
            # and v2; dedicated boundary tests cover 0/mask and v1 mask+1.
            zero = torch.randint(1, mask + 1, (groups, width),
                                 generator=tile_generator, dtype=torch.int32)
        scale = (0.01 + 0.04 * torch.rand(groups, width, generator=tile_generator)).to(dtype)
        stored = zero if v2 else zero - 1
        qw[:, start:end].copy_(_pack(quant, bits, exllama))
        zstart, zend = start // 32 * bits, end // 32 * bits
        qz[:, zstart:zend].copy_(_pack_zeros(stored, bits))
        scales[:, start:end].copy_(scale)
        # Compute from unpacked logical integers, never from native decoding.
        weight = ((quant.float() - zero[group].float()) * scale[group].float()).to(dtype)
        expected[:, start:end] = a64 @ weight.double()
    return a_cpu.to(device), qw, qz, scales, expected


def _report_perf(label, a, qw, qz, scales, bits, exllama, v2):
    ms = _measure(lambda: _run(a, qw, qz, scales, exllama, v2, bits))
    byte_count = sum(t.numel() * t.element_size() for t in (a, qw, qz, scales))
    byte_count += a.shape[0] * qw.shape[1] * a.element_size()
    print(f"[PERF] {label} M={a.shape[0]} K={a.shape[1]} N={qw.shape[1]} "
          f"dtype={a.dtype} bits={bits} exllama={exllama} v2={v2} "
          f"group_size={a.shape[1] // scales.shape[0]} GPU_time={ms * 1000:.3f} us "
          f"logical_io_bytes={byte_count} effective_bandwidth={byte_count / (ms * 1e6):.3f} GB/s "
          "timing=cuda_graph_median traffic=logical_inputs_once_plus_output cache=warm", flush=True)


def _check_packed_graph(a, qw, qz, scales, expected, bits, exllama, v2):
    """Capture on a side stream and change both activation and scale storage."""
    stream = torch.cuda.Stream(device=a.device)
    stream.wait_stream(torch.cuda.current_stream(a.device))
    original_a, original_scales = a.clone(), scales.clone()
    stream.wait_stream(torch.cuda.current_stream(a.device))
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize(a.device)
    with torch.cuda.graph(graph, stream=stream):
        captured = _run(a, qw, qz, scales, exllama, v2, bits)
    assert captured.shape == expected.shape
    assert captured.dtype == a.dtype and captured.device == a.device
    try:
        for a_factor, scale_factor in ((1.0, 1.0), (-0.5, 2.0), (2.0, 0.5), (1.0, 1.0)):
            with torch.cuda.stream(stream):
                a.copy_(original_a * a_factor)
                scales.copy_(original_scales * scale_factor)
                graph.replay()
            stream.synchronize()
            _check(captured, expected * a_factor * scale_factor)
    finally:
        with torch.cuda.stream(stream):
            a.copy_(original_a)
            scales.copy_(original_scales)
        torch.cuda.current_stream(a.device).wait_stream(stream)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("exllama", [False, True])
@pytest.mark.parametrize("v2", [False, True])
def test_gptq_gemm_deterministic_bf16_graph(bits, exllama, v2):
    if not torch.cuda.is_available():
        pytest.skip("MetaX GPU is required")
    a, qw, qz, scales, expected = make_qwen_case(256, 96, m=3, bits=bits,
                                               exllama=exllama, v2=v2)
    _check_packed_graph(a, qw, qz, scales, expected, bits, exllama, v2)
    cosine = _check(_run(a, qw, qz, scales, exllama, v2, bits), expected)
    print(f"[ACCURACY] deterministic_bf16 cosine={cosine:.9f}", flush=True)


@pytest.mark.parametrize("bits", [2, 3])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m,group_size", [(7, 32), (8, 32), (9, 32), (3, 16), (3, 48)])
def test_gptq_gemm_specialized_and_grouped_boundaries(bits, dtype, m, group_size):
    if not torch.cuda.is_available():
        pytest.skip("MetaX GPU is required")
    a, qw, qz, scales, w = _case(bits, dtype, True, False, group_size,
                                 192, 96, m, torch.device(DEVICE))
    _check_graph(a, qw, qz, scales, w, True, False, bits)
    _check(_run(a, qw, qz, scales, True, False, bits), _reference(a, w))


@pytest.mark.parametrize("projection", QWEN72B_PROJECTIONS)
@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("exllama", [False, True])
@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("m", [int(value) for value in os.getenv("MCOP_GPTQ_QWEN_M", "1").split(",")])
def test_gptq_gemm_qwen72b_tp1_bf16(projection, bits, exllama, v2, m):
    if os.getenv("MCOP_GPTQ_QWEN_FULL") != "1":
        pytest.skip("Set MCOP_GPTQ_QWEN_FULL=1 for full Qwen72B TP1 projection coverage")
    if not torch.cuda.is_available():
        pytest.skip("MetaX GPU is required")
    k, n = QWEN72B_PROJECTIONS[projection]
    a, qw, qz, scales, expected = make_qwen_case(k, n, m, bits, exllama, v2)
    _check_packed_graph(a, qw, qz, scales, expected, bits, exllama, v2)
    cosine = _check(_run(a, qw, qz, scales, exllama, v2, bits), expected)
    print(f"[ACCURACY] qwen72b_tp1={projection} cosine={cosine:.9f} "
          "reference=cpu_fp64_dtype_rounded_weights zeros=asymmetric seed=20261116", flush=True)
    _report_perf(f"qwen72b_tp1={projection}", a, qw, qz, scales, bits, exllama, v2)


if __name__ == "__main__":
    import sys
    raise SystemExit(pytest.main([__file__, "-q", "-s", *sys.argv[1:]]))
