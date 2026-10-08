"""Tests and benchmarks for ``silu_mul_quant_varlen``.

Run the following commands from the repository root.

Default normal path (``workspace=None``; ``pyproject.toml`` excludes ``full``):
    python -m pytest -sv unit_test/test_silu_mul_quant_varlen.py
    python -m pytest -sv unit_test/test_silu_mul_quant_varlen.py::test_silu_mul_quant_varlen_default

Full fixed-shape token sweep (tokens 1..64; compare normal and workspace paths):
    python -m pytest -sv -m full unit_test/test_silu_mul_quant_varlen.py::test_silu_mul_quant_varlen_full_token_range

One token-count case from the full sweep:
    python -m pytest -sv -m full 'unit_test/test_silu_mul_quant_varlen.py::test_silu_mul_quant_varlen_full_token_range[tokens64_active384]'

Fixed E=16, T=4096, H=2048 density comparison:
    python -m pytest -sv -m full unit_test/test_silu_mul_quant_varlen.py::test_silu_mul_quant_varlen_density_performance

Service-derived profiles:
    python -m pytest -sv -m full unit_test/test_silu_mul_quant_varlen.py::test_silu_mul_quant_varlen_service_profile

Feature and workspace-fallback checks:
    python -m pytest -sv -m full unit_test/test_silu_mul_quant_varlen.py::test_silu_mul_quant_varlen_transposed_ue8m0
    python -m pytest -sv -m full unit_test/test_silu_mul_quant_varlen.py::test_silu_mul_quant_varlen_undersized_workspace_falls_back
    python -m pytest -sv -m full unit_test/test_silu_mul_quant_varlen.py::test_silu_mul_quant_varlen_rejects_pdl

All extended tests:
    python -m pytest -sv -m full unit_test/test_silu_mul_quant_varlen.py
"""

import math
from dataclasses import dataclass

import pytest
import torch

import mcoplib.sgl_kernel  # noqa: F401 - registers torch.ops.sgl_kernel


GROUP_SIZE = 128
FP8_MAX = 448.0
DEFAULT_SWIGLU_LIMIT = 7.0
DEEPSEEK_HIDDEN = 7168
DEEPSEEK_EXPERTS = 256
DEEPSEEK_TOPK = 8
FP8_MISMATCH_RATE_THRESHOLD = 1e-5
MASKED_POST_QUANT_EXPERTS = 16
MASKED_POST_QUANT_HIDDEN = 2048
MASKED_POST_QUANT_TOKENS_PADDED = 2048
MASKED_POST_QUANT_TOPK = 6
MASKED_POST_QUANT_SWIGLU_LIMIT = 10.0
# MASKED_POST_QUANT_TOKEN_COUNTS = (*range(1, 65), 128)
# BENCHMARK_WARMUP = 2
# BENCHMARK_ITERS = 5

MASKED_POST_QUANT_TOKEN_COUNTS = ([128])
FULL_MASKED_POST_QUANT_TOKEN_COUNTS = tuple(range(1, 65))
BENCHMARK_WARMUP = 1
BENCHMARK_ITERS = 1
WORK_TABLE_HEADER_INT32 = 2
WORK_ITEM_INT32 = 2
PERSISTENT_BLOCKS_PER_SM = 4
DENSITY_TEST_EXPERTS = 16
DENSITY_TEST_TOKENS = 4096
DENSITY_TEST_HIDDEN = 2048
DENSITY_TEST_TOPK = 8
DENSITY_ACTIVE_ROUTES = (186, 512, 1024, 2452, 32768)


@dataclass(frozen=True)
class BenchmarkResult:
    token_count: int
    tokens_padded: int
    valid_routes: int
    latency_ms: float
    bandwidth_gbps: float
    persistent_latency_ms: float | None
    persistent_bandwidth_gbps: float | None


BENCHMARK_RESULTS: list[BenchmarkResult] = []


@dataclass(frozen=True)
class DensityBenchmarkResult:
    active_routes: int
    launch_routes: int
    normal_latency_ms: float
    persistent_latency_ms: float
    normal_bandwidth_gbps: float
    persistent_bandwidth_gbps: float


DENSITY_BENCHMARK_RESULTS: list[DensityBenchmarkResult] = []


@dataclass(frozen=True)
class ServiceProfile:
    name: str
    experts: int
    tokens_padded: int
    valid_routes: int
    source: str


@pytest.fixture(scope="module", autouse=True)
def _report_benchmark_summary(request: pytest.FixtureRequest):
    BENCHMARK_RESULTS.clear()
    yield

    if not BENCHMARK_RESULTS:
        return
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return

    reporter.write_sep("=", "silu_mul_quant_varlen performance summary")
    reporter.write_line(
        f"{'tokens':>6} {'T capacity':>11} {'routes':>9} "
        f"{'active ratio':>12} {'invalid CTA':>12} "
        f"{'normal(ms)':>11} {'persist(ms)':>11} {'speedup':>9} "
        f"{'normal GB/s':>11} {'persist GB/s':>12}"
    )
    for result in sorted(BENCHMARK_RESULTS, key=lambda item: item.token_count):
        launch_routes = result.tokens_padded * MASKED_POST_QUANT_TOPK
        active_ratio = result.valid_routes / launch_routes
        invalid_ctas = launch_routes - result.valid_routes
        if result.persistent_latency_ms is None:
            persistent_latency = "-"
            speedup = "-"
            persistent_bandwidth = "-"
        else:
            persistent_latency = f"{result.persistent_latency_ms:.4f}"
            speedup = f"{result.latency_ms / result.persistent_latency_ms:.3f}x"
            persistent_bandwidth = f"{result.persistent_bandwidth_gbps:.2f}"
        reporter.write_line(
            f"{result.token_count:>6} {result.tokens_padded:>11} "
            f"{result.valid_routes:>9} {active_ratio:>11.3%} "
            f"{invalid_ctas:>12} {result.latency_ms:>11.4f} "
            f"{persistent_latency:>11} {speedup:>9} "
            f"{result.bandwidth_gbps:>11.2f} "
            f"{persistent_bandwidth:>12}"
        )
    reporter.write_line(
        "payload = valid_routes * (BF16 input 2H + FP8 output H + "
        "FP32 scales H/128); excludes masked_m scan/cache traffic"
    )
    reporter.write_line(
        "token sweep keeps E=16, T=2048, H=2048 fixed; "
        "routes=sum(masked_m)=tokens*topk"
    )
    if any(
        result.persistent_latency_ms is not None for result in BENCHMARK_RESULTS
    ):
        reporter.write_line(
            "persistent timing includes work-table generation; "
            "workspace allocation is excluded"
        )
    else:
        reporter.write_line(
            "persistent columns are disabled in the default suite; run full tests"
        )


@pytest.fixture(scope="module", autouse=True)
def _report_density_benchmark_summary(request: pytest.FixtureRequest):
    DENSITY_BENCHMARK_RESULTS.clear()
    yield

    if not DENSITY_BENCHMARK_RESULTS:
        return
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return

    reporter.write_sep("=", "silu_mul_quant_varlen density performance summary")
    reporter.write_line(
        f"{'active':>8} {'active ratio':>12} {'invalid CTA':>12} "
        f"{'normal(ms)':>11} {'persist(ms)':>11} {'speedup':>9} "
        f"{'normal GB/s':>11} {'persist GB/s':>12}"
    )
    for result in sorted(
        DENSITY_BENCHMARK_RESULTS, key=lambda item: item.active_routes
    ):
        active_ratio = result.active_routes / result.launch_routes
        invalid_ctas = result.launch_routes - result.active_routes
        speedup = result.normal_latency_ms / result.persistent_latency_ms
        reporter.write_line(
            f"{result.active_routes:>8} {active_ratio:>11.3%} "
            f"{invalid_ctas:>12} {result.normal_latency_ms:>11.4f} "
            f"{result.persistent_latency_ms:>11.4f} {speedup:>8.3f}x "
            f"{result.normal_bandwidth_gbps:>11.2f} "
            f"{result.persistent_bandwidth_gbps:>12.2f}"
        )
    reporter.write_line(
        "shape: E=16, T=4096, H=2048, topk=8; persistent timing includes "
        "work-table generation"
    )


# Service data is converted to kernel-local shapes as follows:
#
# Prefill:
#   - short prompt: min(3072, chunk=8192) -> 3072 source tokens
#   - long prompt: seven full 8192 chunks plus a 7168-token tail
#   - routes are source_tokens * topk, spread across 256 experts
#   - an 8-expert slice preserves average expert occupancy without
#     allocating an impractical full [256, chunk, 2H] tensor
#   - tokens_padded adds 25% headroom for padding and routing skew
#
# Decode:
#   - local experts = 256 / EP
#   - candidate tokens per rank = concurrency / DP * MTP(3)
#   - valid local routes = candidate tokens * topk
#
# TTFT, TPOT and service throughput cover the entire serving stack and are
# intentionally reserved for a separate benchmark instead of correctness.
SERVICE_PROFILES = (
    ServiceProfile("prefill_input3072", 8, 120, 3072 * 8 * 8 // DEEPSEEK_EXPERTS,
                   "chunk=8192,input=3072,concurrency=16"),
    ServiceProfile("prefill_full_chunk8192", 8, 320, 8192 * 8 * 8 // DEEPSEEK_EXPERTS,
                   "chunk=8192,input=64512"),
    ServiceProfile("prefill_tail7168", 8, 280, 7168 * 8 * 8 // DEEPSEEK_EXPERTS,
                   "64512 % 8192 = 7168"),
    ServiceProfile("decode_h200_c256", DEEPSEEK_EXPERTS // 8, 256 // 8 * 3,
                   (256 // 8 * 3) * 8,
                   "DP8,EP8,MTP3,requests=2048,concurrency=256,input=3072,output=1024,rate=8"),
    ServiceProfile("decode_c600u_c256", DEEPSEEK_EXPERTS // 16, 256 // 16 * 3,
                   (256 // 16 * 3) * 8,
                   "DP16,EP16,MTP3,requests=2048,concurrency=256,input=3072,output=1024,rate=8"),
    ServiceProfile("decode_h200_c64", DEEPSEEK_EXPERTS // 8, 64 // 8 * 3,
                   (64 // 8 * 3) * 8,
                   "DP8,EP8,MTP3,requests=512,concurrency=64,input=64512,output=1024,rate=8"),
    ServiceProfile("decode_c600u_c64", DEEPSEEK_EXPERTS // 16, 64 // 16 * 3,
                   (64 // 16 * 3) * 8,
                   "DP16,EP16,MTP3,requests=512,concurrency=64,input=64512,output=1024,rate=8"),
)


def _build_masked_m(profile: ServiceProfile) -> torch.Tensor:
    """Build deterministic mildly-skewed counts with the requested sum."""
    base, remainder = divmod(profile.valid_routes, profile.experts)
    counts = torch.full((profile.experts,), base, dtype=torch.int32)
    if remainder:
        counts[:remainder] += 1

    skew = max(1, base // 4)
    for expert in range(0, profile.experts - 1, 4):
        moved = min(skew, int(counts[expert + 1]))
        counts[expert] += moved
        counts[expert + 1] -= moved

    assert int(counts.sum()) == profile.valid_routes
    assert int(counts.max()) <= profile.tokens_padded
    assert profile.valid_routes <= profile.tokens_padded * DEEPSEEK_TOPK
    return counts.to(device="cuda")


def _build_balanced_masked_m(
    experts: int, tokens_padded: int, active_routes: int
) -> torch.Tensor:
    """Distribute an explicit active-route count across fixed-capacity rows."""
    assert 0 <= active_routes <= experts * tokens_padded
    base, remainder = divmod(active_routes, experts)
    counts = torch.full((experts,), base, dtype=torch.int32)
    counts[:remainder] += 1
    assert int(counts.sum()) == active_routes
    assert int(counts.max()) <= tokens_padded
    return counts.to(device="cuda")


def _build_density_masked_m(active_routes: int) -> torch.Tensor:
    """Balance an explicit active-work count while keeping launch work fixed."""
    launch_routes = DENSITY_TEST_TOKENS * DENSITY_TEST_TOPK
    assert 0 <= active_routes <= launch_routes
    return _build_balanced_masked_m(
        DENSITY_TEST_EXPERTS,
        DENSITY_TEST_TOKENS,
        active_routes,
    )


def _reference(
    natural_input: torch.Tensor,
    masked_m: torch.Tensor,
    scale_ue8m0: bool,
    swiglu_limit: float | None,
):
    e, t, two_h = natural_input.shape
    h = two_h // 2
    g = h // GROUP_SIZE
    gate = natural_input[..., :h]
    up = natural_input[..., h:]

    if swiglu_limit is not None:
        limit = torch.tensor(
            swiglu_limit, dtype=torch.bfloat16, device=natural_input.device
        )
        gate = torch.minimum(gate, limit)
        up = torch.clamp(up, min=-limit, max=limit)

    gate_f = gate.float()
    values = gate_f / (1.0 + torch.exp(-gate_f)) * up.float()
    grouped = values.view(e, t, g, GROUP_SIZE)
    scales = grouped.abs().amax(dim=-1).clamp_min(1e-10) / FP8_MAX
    if scale_ue8m0:
        scales = torch.pow(2.0, torch.ceil(torch.log2(scales)))

    quantized = torch.clamp(
        grouped / scales.unsqueeze(-1), -FP8_MAX, FP8_MAX
    ).to(torch.float8_e4m3fn).view(e, t, h)

    valid = torch.arange(t, device=natural_input.device).unsqueeze(0)
    valid = valid < masked_m.unsqueeze(1)
    return quantized, scales.float(), valid


def _reference_valid(
    natural_input: torch.Tensor,
    masked_m: torch.Tensor,
    scale_ue8m0: bool,
    swiglu_limit: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the reference only for each expert's valid prefix."""
    h = natural_input.size(2) // 2
    g = h // GROUP_SIZE
    quantized_parts = []
    scale_parts = []
    for expert, count in enumerate(masked_m.cpu().tolist()):
        if not count:
            continue
        expert_mask = torch.tensor(
            [count], dtype=torch.int32, device=natural_input.device
        )
        quantized, scales, _ = _reference(
            natural_input[expert : expert + 1, :count],
            expert_mask,
            scale_ue8m0,
            swiglu_limit,
        )
        quantized_parts.append(quantized.view(count, h))
        scale_parts.append(scales.view(count, g))

    if quantized_parts:
        return torch.cat(quantized_parts), torch.cat(scale_parts)
    return (
        torch.empty(
            (0, h), dtype=torch.float8_e4m3fn, device=natural_input.device
        ),
        torch.empty((0, g), dtype=torch.float32, device=natural_input.device),
    )


def _ue8m0_exponents(scales: torch.Tensor) -> torch.Tensor:
    return ((scales.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)


def _assert_fp8_close(actual: torch.Tensor, expected: torch.Tensor):
    """Allow rare adjacent-bin differences from device exp/FP8 rounding."""
    actual_f = actual.float().flatten()
    expected_f = expected.float().flatten()
    mismatch = actual_f != expected_f
    mismatch_count = int(mismatch.sum().item())
    max_mismatches = max(
        1, math.ceil(actual_f.numel() * FP8_MISMATCH_RATE_THRESHOLD)
    )
    assert mismatch_count <= max_mismatches, (
        f"FP8 mismatch count {mismatch_count} exceeds {max_mismatches} "
        f"for {actual_f.numel()} elements"
    )

    if mismatch_count:
        actual_code = actual.contiguous().view(torch.uint8).flatten().to(torch.int16)
        expected_code = (
            expected.contiguous().view(torch.uint8).flatten().to(torch.int16)
        )
        code_distance = (actual_code[mismatch] - expected_code[mismatch]).abs()
        assert bool((code_distance == 1).all()), (
            "FP8 mismatches must be adjacent representable values; "
            f"maximum encoding distance is {int(code_distance.max().item())}"
        )


def _launch_default_kernel(
    natural_input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    masked_m: torch.Tensor,
    topk: int,
    swiglu_limit: float,
    workspace: torch.Tensor | None = None,
    persistent_grid: int = 0,
):
    torch.ops.sgl_kernel.silu_mul_quant_varlen(
        natural_input,
        output,
        output_scale,
        masked_m,
        topk,
        False,  # kScaleUE8M0
        False,  # kTransposed
        False,  # kSwizzle
        swiglu_limit,  # kApplySwigluLimit=True
        False,  # enable_pdl
        workspace,
        persistent_grid,
    )


def _payload_bytes(valid_routes: int, hidden_dim: int) -> int:
    input_bytes = valid_routes * 2 * hidden_dim * 2  # BF16 gate and up
    output_bytes = valid_routes * hidden_dim  # FP8 output
    scale_bytes = valid_routes * (hidden_dim // GROUP_SIZE) * 4
    return input_bytes + output_bytes + scale_bytes


def _measure_default_kernel(
    natural_input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    masked_m: torch.Tensor,
    topk: int,
    swiglu_limit: float,
    workspace: torch.Tensor | None = None,
    persistent_grid: int = 0,
) -> float:
    for _ in range(BENCHMARK_WARMUP):
        _launch_default_kernel(
            natural_input,
            output,
            output_scale,
            masked_m,
            topk,
            swiglu_limit,
            workspace,
            persistent_grid,
        )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(BENCHMARK_ITERS):
        _launch_default_kernel(
            natural_input,
            output,
            output_scale,
            masked_m,
            topk,
            swiglu_limit,
            workspace,
            persistent_grid,
        )
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / BENCHMARK_ITERS


def _assert_padding_untouched(
    output: torch.Tensor,
    output_scale: torch.Tensor,
    masked_m: torch.Tensor,
) -> None:
    """Check padding expert-by-expert to avoid a full FP32 temporary."""
    nan_code = torch.tensor(
        [math.nan], dtype=output.dtype, device=output.device
    ).view(torch.uint8)[0]
    for expert, count in enumerate(masked_m.cpu().tolist()):
        if count == output.size(1):
            continue
        padding_codes = output[expert, count:].contiguous().view(torch.uint8)
        assert bool((padding_codes == nan_code).all())
        assert torch.isnan(output_scale[expert, count:]).all()


def _run_density_case(active_routes: int) -> None:
    e = DENSITY_TEST_EXPERTS
    t = DENSITY_TEST_TOKENS
    h = DENSITY_TEST_HIDDEN
    topk = DENSITY_TEST_TOPK
    launch_routes = t * topk
    masked_m = _build_density_masked_m(active_routes)
    counts = masked_m.cpu().tolist()

    torch.manual_seed(23)
    natural_input = torch.empty(
        (e, t, 2 * h), dtype=torch.bfloat16, device="cuda"
    )
    for expert, count in enumerate(counts):
        if count:
            natural_input[expert, :count].normal_(mean=0.0, std=3.0)

    normal_output = torch.full(
        (e, t, h), math.nan, dtype=torch.float8_e4m3fn, device="cuda"
    )
    normal_scale = torch.full(
        (e, t, h // GROUP_SIZE),
        math.nan,
        dtype=torch.float32,
        device="cuda",
    )
    _launch_default_kernel(
        natural_input,
        normal_output,
        normal_scale,
        masked_m,
        topk,
        MASKED_POST_QUANT_SWIGLU_LIMIT,
    )

    valid = torch.arange(t, device="cuda").unsqueeze(0)
    valid = valid < masked_m.unsqueeze(1)
    reference_output = normal_output[valid].clone()
    reference_scale = normal_scale[valid].clone()
    _assert_padding_untouched(normal_output, normal_scale, masked_m)
    normal_latency_ms = _measure_default_kernel(
        natural_input,
        normal_output,
        normal_scale,
        masked_m,
        topk,
        MASKED_POST_QUANT_SWIGLU_LIMIT,
    )

    del normal_output
    del normal_scale
    persistent_output = torch.full(
        (e, t, h), math.nan, dtype=torch.float8_e4m3fn, device="cuda"
    )
    persistent_scale = torch.full(
        (e, t, h // GROUP_SIZE),
        math.nan,
        dtype=torch.float32,
        device="cuda",
    )
    workspace = torch.empty(
        WORK_TABLE_HEADER_INT32 + WORK_ITEM_INT32 * launch_routes,
        dtype=torch.int32,
        device="cuda",
    )
    sm_count = torch.cuda.get_device_properties(
        natural_input.device
    ).multi_processor_count
    persistent_grid = max(1, sm_count * PERSISTENT_BLOCKS_PER_SM)
    _launch_default_kernel(
        natural_input,
        persistent_output,
        persistent_scale,
        masked_m,
        topk,
        MASKED_POST_QUANT_SWIGLU_LIMIT,
        workspace,
        persistent_grid,
    )

    _assert_fp8_close(persistent_output[valid], reference_output)
    torch.testing.assert_close(
        persistent_scale[valid], reference_scale, rtol=1e-6, atol=0
    )
    _assert_padding_untouched(persistent_output, persistent_scale, masked_m)
    persistent_latency_ms = _measure_default_kernel(
        natural_input,
        persistent_output,
        persistent_scale,
        masked_m,
        topk,
        MASKED_POST_QUANT_SWIGLU_LIMIT,
        workspace,
        persistent_grid,
    )

    payload_bytes = _payload_bytes(active_routes, h)
    DENSITY_BENCHMARK_RESULTS.append(
        DensityBenchmarkResult(
            active_routes,
            launch_routes,
            normal_latency_ms,
            persistent_latency_ms,
            payload_bytes / normal_latency_ms / 1e6,
            payload_bytes / persistent_latency_ms / 1e6,
        )
    )


def _run_default_case(
    e: int,
    t: int,
    h: int,
    masked_m: torch.Tensor,
    topk: int,
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
    benchmark_token_count: int | None = None,
    test_workspace: bool = False,
):
    """Run kApplySwigluLimit=True with every other template flag false."""
    torch.manual_seed(7)
    counts = masked_m.cpu().tolist()
    natural_input = torch.empty(
        (e, t, 2 * h), dtype=torch.bfloat16, device="cuda"
    )
    for expert, count in enumerate(counts):
        if count:
            natural_input[expert, :count].normal_(mean=0.0, std=3.0)
    output = torch.full(
        (e, t, h), math.nan, dtype=torch.float8_e4m3fn, device="cuda"
    )
    output_scale = torch.full(
        (e, t, h // GROUP_SIZE), math.nan, dtype=torch.float32, device="cuda"
    )

    _launch_default_kernel(
        natural_input,
        output,
        output_scale,
        masked_m,
        topk,
        swiglu_limit,
    )

    valid = torch.arange(t, device=natural_input.device).unsqueeze(0)
    valid = valid < masked_m.unsqueeze(1)
    ref_q, ref_scale = _reference_valid(
        natural_input, masked_m, False, swiglu_limit
    )
    _assert_fp8_close(output[valid], ref_q)
    torch.testing.assert_close(
        output_scale[valid], ref_scale, rtol=1e-6, atol=0
    )
    _assert_padding_untouched(output, output_scale, masked_m)

    if benchmark_token_count is not None:
        valid_routes = int(masked_m.sum().item())

        def measure(
            measured_output: torch.Tensor,
            measured_scale: torch.Tensor,
            measured_workspace: torch.Tensor | None = None,
            measured_grid: int = 0,
        ) -> float:
            for _ in range(BENCHMARK_WARMUP):
                _launch_default_kernel(
                    natural_input,
                    measured_output,
                    measured_scale,
                    masked_m,
                    topk,
                    swiglu_limit,
                    measured_workspace,
                    measured_grid,
                )

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(BENCHMARK_ITERS):
                _launch_default_kernel(
                    natural_input,
                    measured_output,
                    measured_scale,
                    masked_m,
                    topk,
                    swiglu_limit,
                    measured_workspace,
                    measured_grid,
                )
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / BENCHMARK_ITERS

        latency_ms = measure(output, output_scale)
        persistent_latency_ms = None
        persistent_bandwidth_gbps = None

        if test_workspace:
            workspace = torch.empty(
                WORK_TABLE_HEADER_INT32 + WORK_ITEM_INT32 * t * topk,
                dtype=torch.int32,
                device=natural_input.device,
            )
            sm_count = torch.cuda.get_device_properties(
                natural_input.device
            ).multi_processor_count
            persistent_grid = max(1, sm_count * PERSISTENT_BLOCKS_PER_SM)
            persistent_output = torch.full(
                (e, t, h),
                math.nan,
                dtype=torch.float8_e4m3fn,
                device=natural_input.device,
            )
            persistent_scale = torch.full(
                (e, t, h // GROUP_SIZE),
                math.nan,
                dtype=torch.float32,
                device=natural_input.device,
            )

            _launch_default_kernel(
                natural_input,
                persistent_output,
                persistent_scale,
                masked_m,
                topk,
                swiglu_limit,
                workspace,
                persistent_grid,
            )
            _assert_fp8_close(persistent_output[valid], ref_q)
            torch.testing.assert_close(
                persistent_scale[valid], ref_scale, rtol=1e-6, atol=0
            )
            _assert_padding_untouched(
                persistent_output, persistent_scale, masked_m
            )

            persistent_latency_ms = measure(
                persistent_output,
                persistent_scale,
                workspace,
                persistent_grid,
            )
            persistent_bandwidth_gbps = (
                _payload_bytes(valid_routes, h) / persistent_latency_ms / 1e6
            )

        payload_bytes = _payload_bytes(valid_routes, h)
        bandwidth_gbps = payload_bytes / latency_ms / 1e6
        result = BenchmarkResult(
            benchmark_token_count,
            t,
            valid_routes,
            latency_ms,
            bandwidth_gbps,
            persistent_latency_ms,
            persistent_bandwidth_gbps,
        )
        if test_workspace:
            for index, existing in enumerate(BENCHMARK_RESULTS):
                same_shape = (
                    existing.token_count == benchmark_token_count
                    and existing.tokens_padded == t
                    and existing.valid_routes == valid_routes
                )
                if same_shape and existing.persistent_latency_ms is None:
                    BENCHMARK_RESULTS[index] = BenchmarkResult(
                        existing.token_count,
                        existing.tokens_padded,
                        existing.valid_routes,
                        existing.latency_ms,
                        existing.bandwidth_gbps,
                        persistent_latency_ms,
                        persistent_bandwidth_gbps,
                    )
                    break
            else:
                BENCHMARK_RESULTS.append(result)
        else:
            BENCHMARK_RESULTS.append(result)


@pytest.mark.parametrize(
    "token_count",
    MASKED_POST_QUANT_TOKEN_COUNTS,
    ids=lambda token_count: (
        f"tokens{token_count}_active"
        f"{token_count * MASKED_POST_QUANT_TOPK}"
    ),
)
def test_silu_mul_quant_varlen_default(token_count: int):
    """Default suite exercises the fixed-capacity normal path."""
    tokens_padded = MASKED_POST_QUANT_TOKENS_PADDED
    active_routes = token_count * MASKED_POST_QUANT_TOPK
    assert active_routes <= tokens_padded * MASKED_POST_QUANT_TOPK
    masked_m = _build_balanced_masked_m(
        MASKED_POST_QUANT_EXPERTS,
        tokens_padded,
        active_routes,
    )
    _run_default_case(
        MASKED_POST_QUANT_EXPERTS,
        tokens_padded,
        MASKED_POST_QUANT_HIDDEN,
        masked_m,
        MASKED_POST_QUANT_TOPK,
        MASKED_POST_QUANT_SWIGLU_LIMIT,
        benchmark_token_count=token_count,
    )


@pytest.mark.parametrize(
    "token_count",
    FULL_MASKED_POST_QUANT_TOKEN_COUNTS,
    ids=lambda token_count: (
        f"tokens{token_count}_active"
        f"{token_count * MASKED_POST_QUANT_TOPK}"
    ),
)
@pytest.mark.full
def test_silu_mul_quant_varlen_full_token_range(token_count: int):
    """Vary masked_m at fixed E/T/H and compare normal with persistent."""
    tokens_padded = MASKED_POST_QUANT_TOKENS_PADDED
    active_routes = token_count * MASKED_POST_QUANT_TOPK
    assert active_routes <= tokens_padded * MASKED_POST_QUANT_TOPK
    masked_m = _build_balanced_masked_m(
        MASKED_POST_QUANT_EXPERTS,
        tokens_padded,
        active_routes,
    )
    _run_default_case(
        MASKED_POST_QUANT_EXPERTS,
        tokens_padded,
        MASKED_POST_QUANT_HIDDEN,
        masked_m,
        MASKED_POST_QUANT_TOPK,
        MASKED_POST_QUANT_SWIGLU_LIMIT,
        benchmark_token_count=token_count,
        test_workspace=True,
    )


@pytest.mark.parametrize(
    "active_routes",
    DENSITY_ACTIVE_ROUTES,
    ids=lambda active: (
        f"active{active}_ratio"
        f"{active / (DENSITY_TEST_TOKENS * DENSITY_TEST_TOPK):.3%}"
    ),
)
@pytest.mark.full
def test_silu_mul_quant_varlen_density_performance(active_routes: int):
    """Compare normal and workspace paths at fixed launch density."""
    _run_density_case(active_routes)


@pytest.mark.full
def test_silu_mul_quant_varlen_small():
    masked_m = torch.tensor([6, 0, 3, 1], dtype=torch.int32, device="cuda")
    _run_default_case(4, 6, 512, masked_m, topk=3)


@pytest.mark.full
def test_silu_mul_quant_varlen_undersized_workspace_falls_back():
    torch.manual_seed(17)
    e, t, h, topk = 4, 6, 512, 3
    masked_m = torch.tensor([6, 0, 3, 1], dtype=torch.int32, device="cuda")
    natural_input = torch.randn(
        (e, t, 2 * h), dtype=torch.bfloat16, device="cuda"
    )
    output = torch.full(
        (e, t, h), math.nan, dtype=torch.float8_e4m3fn, device="cuda"
    )
    output_scale = torch.full(
        (e, t, h // GROUP_SIZE),
        math.nan,
        dtype=torch.float32,
        device="cuda",
    )
    undersized_workspace = torch.empty(1, dtype=torch.int32, device="cuda")

    _launch_default_kernel(
        natural_input,
        output,
        output_scale,
        masked_m,
        topk,
        DEFAULT_SWIGLU_LIMIT,
        undersized_workspace,
        persistent_grid=1,
    )

    ref_q, ref_scale, valid = _reference(
        natural_input, masked_m, False, DEFAULT_SWIGLU_LIMIT
    )
    _assert_fp8_close(output[valid], ref_q[valid])
    torch.testing.assert_close(
        output_scale[valid], ref_scale[valid], rtol=1e-6, atol=0
    )
    assert torch.isnan(output[~valid].float()).all()
    assert torch.isnan(output_scale[~valid]).all()


@pytest.mark.parametrize("profile", SERVICE_PROFILES, ids=lambda case: case.name)
@pytest.mark.full
def test_silu_mul_quant_varlen_service_profile(profile: ServiceProfile):
    masked_m = _build_masked_m(profile)
    _run_default_case(
        profile.experts,
        profile.tokens_padded,
        DEEPSEEK_HIDDEN,
        masked_m,
        DEEPSEEK_TOPK,
    )


@pytest.mark.full
def test_silu_mul_quant_varlen_transposed_ue8m0():
    """Explicit non-default feature check; service profiles keep flags false."""
    torch.manual_seed(11)
    e, t, h = 3, 5, 1024
    g = h // GROUP_SIZE
    masked_m = torch.tensor([5, 2, 0], dtype=torch.int32, device="cuda")
    natural_input = torch.randn(
        (e, t, 2 * h), dtype=torch.bfloat16, device="cuda"
    ) * 4

    output = torch.full(
        (e, t, h), math.nan, dtype=torch.float8_e4m3fn, device="cuda"
    )
    output_scale = torch.full(
        (e, g // 4, t), -1, dtype=torch.int32, device="cuda"
    )

    torch.ops.sgl_kernel.silu_mul_quant_varlen(
        natural_input.contiguous(),
        output,
        output_scale,
        masked_m,
        2,
        True,
        True,
        False,
        DEFAULT_SWIGLU_LIMIT,
        False,
    )

    ref_q, ref_scale, valid = _reference(
        natural_input, masked_m, True, DEFAULT_SWIGLU_LIMIT
    )
    _assert_fp8_close(output[valid], ref_q[valid])
    assert torch.isnan(output[~valid].float()).all()

    got_bytes = output_scale.view(torch.uint8).view(e, g // 4, t, 4)
    ref_exp = _ue8m0_exponents(ref_scale)
    for expert in range(e):
        count = int(masked_m[expert].item())
        for group in range(g):
            pack, slot = divmod(group, 4)
            assert torch.equal(
                got_bytes[expert, pack, :count, slot],
                ref_exp[expert, :count, group],
            )
            if count < t:
                assert torch.all(
                    got_bytes[expert, pack, count:, slot] == 0xFF
                )


@pytest.mark.full
def test_silu_mul_quant_varlen_rejects_pdl():
    e, t, h = 1, 1, 512
    input_tensor = torch.zeros(
        (e, t, 2 * h), dtype=torch.bfloat16, device="cuda"
    )
    output = torch.empty(
        (e, t, h), dtype=torch.float8_e4m3fn, device="cuda"
    )
    scales = torch.empty(
        (e, t, h // GROUP_SIZE), dtype=torch.float32, device="cuda"
    )
    masked_m = torch.ones(e, dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="enable_pdl=true"):
        torch.ops.sgl_kernel.silu_mul_quant_varlen(
            input_tensor,
            output,
            scales,
            masked_m,
            1,
            False,
            False,
            False,
            None,
            True,
        )
