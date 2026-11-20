"""Indexer FP8 cache accuracy + CUDA Graph + effective-bandwidth regression.

Run after checking mx-smi and choosing an idle physical GPU:
  CUDA_VISIBLE_DEVICES=<idle GPU> python unit_test/test_indexer_k_quant_and_cache.py
  CUDA_VISIBLE_DEVICES=<idle GPU> pytest -v -s unit_test/test_indexer_k_quant_and_cache.py

The destination is always E4M3 FP8 (uint8 is its byte storage, NOT INT8).
Input -> dequantized FP8 cosine must be >= 0.999. Kernel -> independently
quantized CPU reference must be >= 0.99999, with scale and absolute-error
checks as well: cosine alone cannot detect an incorrect multiplicative scale.
Timing uses 10 warmups and a graph containing 100 launches; allocation,
reference calculation and gather/dequantization are excluded. Reported GB/s
is minimum logical traffic, not a hardware DRAM counter or a dual-die sum.
"""

import math
import os

import pytest
import torch
import torch.nn.functional as F

import mcoplib._C  # Register torch.ops._C_cache_ops without importing all of vLLM.


DTYPES = [torch.bfloat16, torch.float16, torch.float32]
FORMATS = ["fp32", "ue8m0", "mxfp8"]
TOKENS = [1, 4, 8, 17, 48, 128, 1024, 2048, 4096, 8192, 10240, 16384, 32768]
# Full token sweep plus multi-quant-block/grid-tail and numeric edge cases.
CASES = [
    (dtype, fmt, 128, bs, n, "normal", False)
    for dtype in DTYPES for fmt in FORMATS for bs in [16, 64] for n in TOKENS
] + [
    (dtype, fmt, dim, 32, 17, "normal", True)
    for dtype in DTYPES for fmt in ["fp32", "ue8m0"]
    for dim in [256, 384, 512, 640, 1024]
] + [
    (dtype, fmt, 128, 16, 17, pattern, True)
    for dtype in DTYPES for fmt in FORMATS
    for pattern in ["zero", "small", "large", "non_power2", "asymmetric", "boundary"]
]


def _op(k, cache, slots, fmt, quant_block=128):
    torch.ops._C_cache_ops.indexer_k_quant_and_cache(
        k, cache, slots, quant_block, fmt
    )


def _cosine(actual, expected):
    actual, expected = actual.double().flatten(), expected.double().flatten()
    if expected.count_nonzero() == 0:
        assert actual.count_nonzero() == 0, "nonzero output for zero input"
        return 1.0
    return F.cosine_similarity(actual, expected, dim=0, eps=1e-30).item()


def _reference(k, fmt):
    """CPU-only oracle, computed from rounded input values, never cache output."""
    values = k.float().cpu().reshape(k.shape[0], -1, 128)
    amax = values.abs().amax(-1).clamp_min(1e-4)
    scales = amax / 448.0
    if fmt in ("ue8m0", "mxfp8"):
        scales = torch.tensor([
            2.0 ** math.ceil(math.log2(float(s))) for s in scales.flatten()
        ], dtype=torch.float32).reshape(scales.shape)
    quant = (values / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quant.float().reshape(k.shape), scales


def _input(n, dim, dtype, pattern):
    generator = torch.Generator().manual_seed(20261119 + n + dim)
    values = torch.randn(n, dim, generator=generator)
    if pattern == "zero":
        values.zero_()
    elif pattern == "small":
        values *= 1e-4
    elif pattern == "large":
        values *= 4096
    elif pattern == "non_power2":
        # A non-power-of-two raw scale makes the original mxfp8 bug deterministic.
        values /= values.abs().amax(-1, keepdim=True)
        values *= 3.0
    elif pattern == "asymmetric":
        values[:, :64] *= 0.01
        values[:, 96:] *= 7
    elif pattern == "boundary":
        # Exact powers and adjacent representable scales test exponent boundaries.
        values /= values.abs().amax(-1, keepdim=True)
        peaks = torch.tensor([1.75, 1.7421875, 1.7578125])
        values *= peaks[torch.arange(n) % 3, None]
    return values.to(dtype).cuda()


def _decode(cache, slots, dim, block_size, fmt):
    """Read the actual planar cache layout without depending on another kernel."""
    pages = cache.cpu().reshape(cache.shape[0], -1)
    slots = slots.cpu()
    valid = slots >= 0
    selected = slots[valid]
    blocks, offsets = selected // block_size, selected % block_size
    data_idx = offsets[:, None] * dim + torch.arange(dim)[None, :]
    raw = pages[blocks[:, None], data_idx].contiguous()
    scale_idx = (block_size * dim + offsets[:, None] * (dim // 128 * 4)
                 + torch.arange(dim // 128 * 4)[None, :])
    scale_bytes = pages[blocks[:, None], scale_idx].contiguous()
    if fmt == "mxfp8":
        assert torch.equal(scale_bytes, scale_bytes[:, :1].expand_as(scale_bytes))
        scales = (scale_bytes[:, :1].to(torch.int32) << 23).view(torch.float32)
    else:
        scales = scale_bytes.view(torch.float32)
    return raw.view(torch.float8_e4m3fn).float(), scales, valid


def _check(k, cache, slots, block_size, fmt):
    dim = k.shape[1]
    quant, scales, valid = _decode(cache, slots, dim, block_size, fmt)
    source = k.cpu()[valid].float()
    ref_quant, ref_scales = _reference(k.cpu()[valid], fmt)
    torch.testing.assert_close(scales, ref_scales, rtol=2e-6, atol=0)
    recovered = (quant.reshape(-1, dim // 128, 128) * scales[..., None]).reshape(-1, dim)
    reference = (ref_quant.reshape(-1, dim // 128, 128)
                 * ref_scales[..., None]).reshape(-1, dim)
    ref_cos = _cosine(recovered, reference)
    cos = _cosine(recovered, source)
    assert ref_cos >= 0.99999, f"CPU reference cosine {ref_cos} < 0.99999"
    assert cos >= 0.999, f"FP8 reconstruction cosine {cos} < 0.999"
    # FP8's maximum half-ULP is 16 in the E4M3 finite range.
    diff = (recovered - source).abs().reshape(-1, dim // 128, 128)
    # Include FP32 division/multiply/subtract roundoff at the half-ULP bound.
    roundoff = 4 * torch.finfo(torch.float32).eps * source.abs().reshape(
        -1, dim // 128, 128
    ).amax(-1)
    assert torch.all(diff.amax(-1) <= 16 * ref_scales + roundoff + 1e-12)
    torch.testing.assert_close(recovered, reference, rtol=0, atol=0)
    # Unwritten slots (including negative/padded token slots) retain their sentinel.
    untouched = torch.ones(cache.shape[0] * block_size, dtype=torch.bool)
    untouched[slots.cpu()[valid]] = False
    pages = cache.cpu().reshape(cache.shape[0], -1)
    data = pages[:, :block_size * dim].reshape(-1, dim)
    meta = pages[:, block_size * dim:].reshape(-1, dim // 128 * 4)
    assert torch.all(data[untouched] == 123)
    assert torch.all(meta[untouched] == 123)
    return cos, ref_cos, diff.max().item()


def _run_case(dtype, fmt, dim, block_size, n, pattern, scattered):
    k = _input(n, dim, dtype, pattern)
    num_blocks = math.ceil((n + 7) / block_size) + 1
    cache = torch.full((num_blocks, block_size, dim + dim // 128 * 4), 123,
                       dtype=torch.uint8, device="cuda")
    if scattered:
        g = torch.Generator().manual_seed(n + dim)
        slots = torch.randperm(num_blocks * block_size, generator=g)[:n].cuda()
        slots[::7] = -1
    else:
        slots = torch.arange(n, dtype=torch.int64, device="cuda")

    def run():
        _op(k, cache, slots, fmt)

    run()
    cos, ref_cos, max_error = _check(k, cache, slots, block_size, fmt)
    expected = cache.clone()
    graph = torch.cuda.CUDAGraph()
    for _ in range(10):
        run()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(100):
            run()
    cache.fill_(123)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(cache, expected), "CUDA graph replay changed cache results"
    # Mutate inputs in place: graph must read current data, not capture-time values.
    original = k.clone()
    k.neg_()
    cache.fill_(123)
    graph.replay()
    _check(k, cache, slots, block_size, fmt)
    k.copy_(original)
    graph.replay()
    for _ in range(10):
        graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        graph.replay()
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / 1000
    valid = (slots >= 0).sum().item()
    # Each token's mapping is counted once as logical traffic (cache reuse ignored).
    traffic = n * 8 + valid * (dim * (k.element_size() + 1) + dim // 128 * 4)
    gbps = traffic / (ms * 1e6)
    print(f"[indexer] T={n:6d} D={dim:4d} B={block_size:2d} "
          f"{str(dtype).split('.')[-1]:8s}->fp8 fmt={fmt:5s} {pattern:10s} "
          f"cos_sim={cos:.7f} ref_cos={ref_cos:.7f} max_err={max_error:.6g} "
          f"OK {ms:.5f} ms {gbps:8.2f} GB/s", flush=True)
    return gbps


@pytest.mark.parametrize("dtype,fmt,dim,block_size,n,pattern,scattered", CASES)
def test_indexer_k_quant_and_cache(dtype, fmt, dim, block_size, n, pattern, scattered):
    _run_case(dtype, fmt, dim, block_size, n, pattern, scattered)


@pytest.mark.parametrize("quant_block", [0, 32, 64, 256])
def test_invalid_quant_block(quant_block):
    k = torch.zeros(1, 128, dtype=torch.bfloat16, device="cuda")
    cache = torch.zeros(1, 16, 144, dtype=torch.uint8, device="cuda")
    slots = torch.zeros(1, dtype=torch.int64, device="cuda")
    with pytest.raises(RuntimeError, match="quant_block_size"):
        _op(k, cache, slots, "ue8m0", quant_block)


def test_empty():
    k = torch.empty(0, 128, dtype=torch.bfloat16, device="cuda")
    cache = torch.full((1, 16, 132), 123, dtype=torch.uint8, device="cuda")
    _op(k, cache, torch.empty(0, dtype=torch.int64, device="cuda"), "ue8m0")
    torch.cuda.synchronize()
    assert torch.all(cache == 123)


if __name__ == "__main__":
    print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')!r} "
          f"device_count={torch.cuda.device_count()}", flush=True)
    print(f"Device: {torch.cuda.get_device_name(0)}; cases={len(CASES)}", flush=True)
    print("=" * 140)
    peak = 0.0
    for case in CASES:
        peak = max(peak, _run_case(*case))
    for quant_block in [0, 32, 64, 256]:
        test_invalid_quant_block(quant_block)
    test_empty()
    print("=" * 140)
    print(f"Accuracy: ALL PASS ({len(CASES)} numeric/graph cases + 5 validation cases); "
          "FP8 cosine >= 0.999; reference cosine >= 0.99999")
    print(f"Peak effective bandwidth: {peak:.2f} GB/s (single visible device; "
          "not aggregate dual-die/HBM-counter bandwidth)")
