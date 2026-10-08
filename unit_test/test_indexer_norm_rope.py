"""Official Step-4 Stage-0 parity tests for the native mcoplib callable.

Accuracy:
    python -m pytest -q unit_test/test_indexer_norm_rope.py

Bandwidth matrix (prints results; no fixed performance threshold):
    MCOP_RUN_PERF_TESTS=1 python -m pytest -s -q \
        unit_test/test_indexer_norm_rope.py -k bandwidth
"""

import math
import os
import statistics

import pytest
import torch

import mcoplib.op as ops


HEAD_DIM = 256
ROTARY_DIM = 16
BF16_RELATIVE_ULP = 2.0**-7
PERF_RESULTS = []


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(ops, "indexer_norm_rope"),
    reason="CUDA unavailable or Stage-0 op not built",
)


def make_inputs(tokens, q_heads, seed=20260812):
    device, dtype = "cuda", torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(seed + tokens + q_heads)

    def randn(*shape):
        return torch.randn(shape, generator=generator, device=device,
                           dtype=dtype).contiguous()

    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(ROTARY_DIM, device=device, dtype=torch.float32) / ROTARY_DIM)
    )
    angles = torch.arange(4096, device=device, dtype=torch.float32)[:, None]
    angles = angles * inv_freq[None, :]
    return {
        "q": randn(tokens, q_heads * HEAD_DIM),
        "k": randn(tokens, HEAD_DIM),
        "z": randn(tokens, HEAD_DIM),
        "qw": randn(HEAD_DIM),
        "kw": randn(HEAD_DIM),
        "kb": randn(HEAD_DIM),
        "cos": angles.cos().to(dtype).contiguous(),
        "sin": angles.sin().to(dtype).contiguous(),
        "positions": torch.randint(0, 4096, (tokens,), generator=generator,
                                   device=device, dtype=torch.int32),
    }


def reference(x, q_heads, q_weight_bias):
    tokens = x["q"].shape[0]
    q = x["q"].view(tokens, q_heads, HEAD_DIM).float()
    k = x["k"].view(tokens, 1, HEAD_DIM).float()
    q_rstd = torch.rsqrt(q.square().mean(-1, keepdim=True) + 1e-6)
    q = q * q_rstd * (x["qw"].float() + q_weight_bias)
    k_mean = k.mean(-1, keepdim=True)
    k_var = (k - k_mean).square().mean(-1, keepdim=True)
    k = (k - k_mean) * torch.rsqrt(k_var + 1e-6)
    k = k * x["kw"].float() + x["kb"].float()
    cos = x["cos"][x["positions"].long()].float().unsqueeze(1)
    sin = x["sin"][x["positions"].long()].float().unsqueeze(1)

    def rotate(value):
        result = value.clone()
        real = value[..., :ROTARY_DIM]
        imag = value[..., ROTARY_DIM : 2 * ROTARY_DIM]
        result[..., :ROTARY_DIM] = real * cos - imag * sin
        result[..., ROTARY_DIM : 2 * ROTARY_DIM] = real * sin + imag * cos
        return result.to(torch.bfloat16)

    return rotate(q).view_as(x["q"]), rotate(k).view_as(x["k"])


def assert_official_ulp(actual, expected):
    error = (actual.float() - expected.float()).abs()
    peak = float(expected.float().abs().max()) if expected.numel() else 0.0
    bound = 2.0 * BF16_RELATIVE_ULP * peak
    assert float(error.max()) <= bound, (
        f"max_abs={float(error.max()):.6e}, bound={bound:.6e}, peak={peak:.6e}"
    )


@pytest.mark.parametrize("q_heads", [4, 16])
@pytest.mark.parametrize("tokens", [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("q_weight_bias", [0.0, 1.0])
def test_separate_official_parity(tokens, q_heads, q_weight_bias):
    x = make_inputs(tokens, q_heads)
    q_ref, k_ref = reference(x, q_heads, q_weight_bias)
    q_out, k_out, z_out = ops.indexer_norm_rope(
        x["q"], x["k"], x["z"], x["qw"], x["kw"], x["kb"],
        x["cos"], x["sin"], x["positions"], HEAD_DIM, ROTARY_DIM,
        1e-6, q_weight_bias,
    )
    torch.cuda.synchronize()
    assert_official_ulp(q_out, q_ref)
    assert_official_ulp(k_out, k_ref)
    assert torch.equal(z_out, x["z"])
    assert z_out.data_ptr() == x["z"].data_ptr()


@pytest.mark.parametrize("q_heads", [4, 16])
@pytest.mark.parametrize("tokens", [1, 512, 4096])
def test_packed_matches_separate_and_preserves_z(tokens, q_heads):
    x = make_inputs(tokens, q_heads)
    separate = ops.indexer_norm_rope(
        x["q"], x["k"], x["z"], x["qw"], x["kw"], x["kb"],
        x["cos"], x["sin"], x["positions"],
    )
    packed = torch.cat((x["q"], x["k"], x["z"]), dim=-1)
    original_ptr = packed.data_ptr()
    z_before = packed[:, -HEAD_DIM:].clone()
    returned = ops.indexer_norm_rope_packed_(
        packed, x["qw"], x["kw"], x["kb"], x["cos"], x["sin"],
        x["positions"], q_heads,
    )
    q_width = q_heads * HEAD_DIM
    assert returned.data_ptr() == original_ptr
    assert torch.equal(packed[:, :q_width], separate[0])
    assert torch.equal(packed[:, q_width : q_width + HEAD_DIM], separate[1])
    assert torch.equal(packed[:, -HEAD_DIM:], z_before)


def test_accepts_three_dimensional_separate_inputs():
    x = make_inputs(7, 4)
    q, k, z = x["q"].view(7, 4, HEAD_DIM), x["k"].view(7, 1, HEAD_DIM), x["z"].view(7, 1, HEAD_DIM)
    outputs = ops.indexer_norm_rope(
        q, k, z, x["qw"], x["kw"], x["kb"], x["cos"], x["sin"], x["positions"]
    )
    assert outputs[0].shape == q.shape
    assert outputs[1].shape == k.shape
    assert outputs[2].data_ptr() == z.data_ptr()


def test_rejects_invalid_packed_width():
    x = make_inputs(1, 4)
    bad = torch.empty(1, 1535, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="packed width"):
        ops.indexer_norm_rope_packed_(
            bad, x["qw"], x["kw"], x["kb"], x["cos"], x["sin"],
            x["positions"], 4,
        )


@pytest.mark.skipif(
    os.getenv("MCOP_RUN_PERF_TESTS") != "1",
    reason="set MCOP_RUN_PERF_TESTS=1 to run the bandwidth matrix",
)
@pytest.mark.parametrize("q_heads", [4, 16])
@pytest.mark.parametrize("tokens", [1, 16, 128, 512, 4096])
def test_bandwidth(tokens, q_heads):
    """Report packed bandwidth and allocating separate-API latency."""
    x = make_inputs(tokens, q_heads, seed=20260902)
    packed_source = torch.cat((x["q"], x["k"], x["z"]), dim=-1)
    packed = packed_source.clone()
    iterations = 500 if tokens <= 16 else 300 if tokens <= 512 else 100
    batches = 15

    def packed_call():
        return ops.indexer_norm_rope_packed_(
            packed, x["qw"], x["kw"], x["kb"], x["cos"], x["sin"],
            x["positions"], q_heads,
        )

    def separate_call():
        return ops.indexer_norm_rope(
            x["q"], x["k"], x["z"], x["qw"], x["kw"], x["kb"],
            x["cos"], x["sin"], x["positions"],
        )

    for _ in range(100):
        packed_call()
    for _ in range(30):
        separate_call()
    torch.cuda.synchronize()

    def measure(call, restore_packed=False):
        samples = []
        for _ in range(batches):
            if restore_packed:
                packed.copy_(packed_source)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                call()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0 / iterations)
        return samples

    packed_samples = measure(packed_call, restore_packed=True)
    separate_samples = measure(separate_call)
    packed_median = statistics.median(packed_samples)
    separate_median = statistics.median(separate_samples)
    # Q/K input reads plus Q/K output writes. Z is alias-only passthrough.
    logical_bytes = tokens * (q_heads + 1) * HEAD_DIM * 2 * 2
    bandwidth_gbps = logical_bytes / (packed_median * 1e-6) / 1e9

    assert math.isfinite(packed_median) and packed_median > 0
    assert math.isfinite(separate_median) and separate_median > 0
    assert math.isfinite(bandwidth_gbps) and bandwidth_gbps > 0
    PERF_RESULTS.append(
        (tokens, q_heads, packed_median, separate_median, bandwidth_gbps)
    )
    print(
        f"tokens={tokens:5d} q_heads={q_heads:2d} "
        f"packed_median={packed_median:9.3f} us "
        f"packed_range=[{min(packed_samples):.3f}, {max(packed_samples):.3f}] us "
        f"bandwidth={bandwidth_gbps:8.2f} GB/s "
        f"separate_median={separate_median:9.3f} us "
        f"separate_range=[{min(separate_samples):.3f}, "
        f"{max(separate_samples):.3f}] us "
        f"packed_samples={packed_samples} separate_samples={separate_samples}"
    )


@pytest.mark.skipif(
    os.getenv("MCOP_RUN_PERF_TESTS") != "1",
    reason="set MCOP_RUN_PERF_TESTS=1 to run the bandwidth matrix",
)
def test_bandwidth_summary():
    """Print a compact summary after the detailed per-shape samples."""
    assert PERF_RESULTS, "bandwidth cases must run before the summary"
    print("\nindexer_norm_rope latency summary (median us)")
    print(" tokens  q_heads    packed   separate  packed_gain   bandwidth")
    for tokens, q_heads, packed, separate, bandwidth in sorted(PERF_RESULTS):
        packed_gain = (separate - packed) / separate * 100.0
        print(
            f"{tokens:7d} {q_heads:8d} {packed:9.3f} {separate:10.3f} "
            f"{packed_gain:11.2f}% {bandwidth:9.2f} GB/s"
        )
