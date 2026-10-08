from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from mcoplib.triton_causal_conv1d_fwd import causal_conv1d_fn, causal_conv1d_fwd


KERNEL_WIDTH = 4
BLOCK_M = 8
BLOCK_N = 256
REQUEST_TOKEN_CAP = 3072
NUM_CACHE_LINES = 212
WARMUP = int(os.getenv("CAUSAL_CONV_WARMUP", "10"))
REPEAT = int(os.getenv("CAUSAL_CONV_REPEAT", "50"))
OPT_BLOCK_M = os.getenv("CAUSAL_CONV_BLOCK_M")
OPT_BLOCK_N = os.getenv("CAUSAL_CONV_BLOCK_N")
OPT_NUM_WARPS = os.getenv("CAUSAL_CONV_NUM_WARPS")


@dataclass(frozen=True)
class Case:
    tp: int
    tokens: int

    @property
    def dim(self) -> int:
        # The model has 96 attention heads of size 128, sharded by TP.
        return (96 // self.tp) * 128


@dataclass(frozen=True)
class Result:
    tp: int
    tokens: int
    requests: int
    dim: int
    legacy_us: float
    optimized_us: float
    speedup: float
    legacy_gbps: float
    optimized_gbps: float
    max_abs_error: float


CASES = [
    Case(tp=tp, tokens=tokens)
    for tp in (8, 4)
    for tokens in (2048, 4096, 8192, 16384)
]


def _optimization_kwargs() -> dict[str, int]:
    """Optional launch overrides used while reproducing tuning experiments."""
    values = {
        "_block_m": OPT_BLOCK_M,
        "_block_n": OPT_BLOCK_N,
        "_num_warps": OPT_NUM_WARPS,
    }
    return {key: int(value) for key, value in values.items() if value is not None}


def _request_lengths(tokens: int) -> list[int]:
    """Split packed tokens into requests whose maximum length is 3072."""
    full_requests, remainder = divmod(tokens, REQUEST_TOKEN_CAP)
    lengths = [REQUEST_TOKEN_CAP] * full_requests
    if remainder:
        lengths.append(remainder)
    return lengths


def _make_strided_x(dim: int, tokens: int, device: torch.device) -> torch.Tensor:
    """Create (dim, tokens) BF16 input with stride (1, 4 * dim).

    This matches a transposed q/k/v view taken from a fused buffer whose
    per-token row contains four ``dim``-wide regions.
    """
    fused = torch.empty((tokens, 4 * dim), device=device, dtype=torch.bfloat16)
    fused.normal_(mean=0.0, std=0.5)
    return fused[:, :dim].transpose(0, 1)


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """Explicit causal convolution evaluated independently for each request."""
    outputs = []
    boundaries = query_start_loc.cpu().tolist()
    weight_fp32 = weight.float()
    bias_fp32 = None if bias is None else bias.float()

    for start, end in zip(boundaries, boundaries[1:]):
        sequence = x[:, start:end].float()
        output = torch.zeros_like(sequence)
        if bias_fp32 is not None:
            output += bias_fp32[:, None]
        for tap in range(KERNEL_WIDTH):
            history = KERNEL_WIDTH - 1 - tap
            output[:, history:] += (
                sequence[:, : sequence.shape[1] - history]
                * weight_fp32[:, tap, None]
            )
        outputs.append(F.silu(output))

    return torch.cat(outputs, dim=1).to(torch.bfloat16)


def _reference_with_state(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    conv_states: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    activation: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent reference covering initial-state and short-sequence edges."""
    outputs = []
    expected_states = conv_states.clone()
    boundaries = query_start_loc.cpu().tolist()
    for request_id, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        sequence = x[:, start:end].float()
        state_id = int(cache_indices[request_id])
        if bool(has_initial_state[request_id]):
            history = conv_states[state_id].float()
        else:
            history = torch.zeros(
                (x.shape[0], KERNEL_WIDTH - 1), device=x.device
            )
        padded = torch.cat((history, sequence), dim=1)
        output = torch.zeros_like(sequence)
        if bias is not None:
            output += bias.float()[:, None]
        for tap in range(KERNEL_WIDTH):
            output += (
                padded[:, tap : tap + sequence.shape[1]]
                * weight[:, tap, None].float()
            )
        if activation in ("silu", "swish"):
            output = F.silu(output)
        outputs.append(output)
        expected_states[state_id] = padded[:, -(KERNEL_WIDTH - 1) :]
    return torch.cat(outputs, dim=1).to(x.dtype), expected_states


def _percentile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    rank = math.ceil(q * len(ordered)) - 1
    return ordered[max(rank, 0)]


def _benchmark_ms(call) -> float:
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(REPEAT)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(REPEAT)]
    for start, end in zip(starts, ends):
        start.record()
        call()
        end.record()
    torch.cuda.synchronize()

    samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return _percentile(samples, 0.5)


def _effective_gbps(
    x: torch.Tensor,
    output: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    request_count: int,
    latency_ms: float,
) -> float:
    """Estimate useful tensor traffic, excluding implementation re-reads."""
    state_values = request_count * conv_states.shape[1] * conv_states.shape[2]
    traffic_bytes = (
        x.numel() * x.element_size()
        + output.numel() * output.element_size()
        + weight.numel() * weight.element_size()
        + state_values * conv_states.element_size()  # cache write
    )
    return traffic_bytes / (latency_ms * 1.0e6)


def _run_case(case: Case, device: torch.device) -> Result:
    torch.manual_seed(0)
    dim = case.dim
    seq_lens = _request_lengths(case.tokens)
    request_count = len(seq_lens)

    x = _make_strided_x(dim, case.tokens, device)
    weight = torch.randn(
        (dim, KERNEL_WIDTH), device=device, dtype=torch.float32
    ) * 0.25
    initial_conv_states = torch.randn(
        (NUM_CACHE_LINES, dim, KERNEL_WIDTH - 1),
        device=device,
        dtype=torch.bfloat16,
    )
    query_start_loc = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    cache_indices = torch.arange(
        request_count, device=device, dtype=torch.int32
    )

    assert x.shape == (dim, case.tokens)
    assert x.stride() == (1, 4 * dim)
    assert x.dtype == torch.bfloat16
    assert weight.shape == (dim, KERNEL_WIDTH)
    assert weight.dtype == torch.float32
    assert initial_conv_states.shape == (NUM_CACHE_LINES, dim, KERNEL_WIDTH - 1)
    assert initial_conv_states.dtype == torch.bfloat16
    assert query_start_loc.shape == (request_count + 1,)
    assert query_start_loc.dtype == torch.int32
    assert cache_indices.shape == (request_count,)
    assert cache_indices.dtype == torch.int32

    expected = _reference(x, weight, None, query_start_loc)
    legacy_states = initial_conv_states.clone()
    optimized_states = initial_conv_states.clone()
    legacy_output = causal_conv1d_fn(
        x=x,
        weight=weight,
        bias=None,
        conv_states=legacy_states,
        query_start_loc=query_start_loc,
        seq_lens_cpu=seq_lens,
        cache_indices=cache_indices,
        has_initial_state=None,
        activation="silu",
        validate_data=True,
    )
    optimized_output = causal_conv1d_fwd(
        x=x,
        weight=weight,
        bias=None,
        conv_states=optimized_states,
        query_start_loc=query_start_loc,
        seq_lens_cpu=seq_lens,
        cache_indices=cache_indices,
        has_initial_state=None,
        activation="silu",
        validate_data=True,
        **_optimization_kwargs(),
    )
    torch.cuda.synchronize()

    assert legacy_output.shape == optimized_output.shape == x.shape
    assert legacy_output.dtype == optimized_output.dtype == torch.bfloat16
    torch.testing.assert_close(legacy_output, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(optimized_output, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(optimized_output, legacy_output, rtol=2e-2, atol=2e-2)

    # With no initial state, every selected cache line receives the last K-1
    # raw input values of its corresponding request.
    boundaries = query_start_loc.cpu().tolist()
    for request_id, end in enumerate(boundaries[1:]):
        torch.testing.assert_close(
            legacy_states[cache_indices[request_id]],
            x[:, end - (KERNEL_WIDTH - 1) : end],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            optimized_states[cache_indices[request_id]],
            legacy_states[cache_indices[request_id]],
            rtol=0,
            atol=0,
        )

    output_fp32 = optimized_output.float()
    expected_fp32 = expected.float()
    abs_error = (output_fp32 - expected_fp32).abs()
    max_abs_error = abs_error.max().item()

    def run_legacy() -> torch.Tensor:
        return causal_conv1d_fn(
            x=x,
            weight=weight,
            bias=None,
            conv_states=legacy_states,
            query_start_loc=query_start_loc,
            seq_lens_cpu=seq_lens,
            cache_indices=cache_indices,
            has_initial_state=None,
            activation="silu",
        )

    def run_optimized() -> torch.Tensor:
        return causal_conv1d_fwd(
            x=x,
            weight=weight,
            bias=None,
            conv_states=optimized_states,
            query_start_loc=query_start_loc,
            seq_lens_cpu=seq_lens,
            cache_indices=cache_indices,
            has_initial_state=None,
            activation="silu",
            **_optimization_kwargs(),
        )

    legacy_ms = _benchmark_ms(run_legacy)
    optimized_ms = _benchmark_ms(run_optimized)
    legacy_gbps = _effective_gbps(
        x, legacy_output, weight, legacy_states, request_count, legacy_ms
    )
    optimized_gbps = _effective_gbps(
        x, optimized_output, weight, optimized_states, request_count, optimized_ms
    )
    return Result(
        tp=case.tp,
        tokens=case.tokens,
        requests=request_count,
        dim=dim,
        legacy_us=legacy_ms * 1e3,
        optimized_us=optimized_ms * 1e3,
        speedup=legacy_ms / optimized_ms,
        legacy_gbps=legacy_gbps,
        optimized_gbps=optimized_gbps,
        max_abs_error=max_abs_error,
    )


def test_causal_conv1d_fn_correctness_and_performance() -> None:
    if not torch.cuda.is_available():
        pytest.skip("causal_conv1d_fn requires a CUDA device")

    device = torch.device("cuda")
    torch.cuda.synchronize()
    total_start = time.perf_counter()
    results = [_run_case(case, device) for case in CASES]
    torch.cuda.synchronize()
    total_elapsed_s = time.perf_counter() - total_start

    print("\ncausal_conv1d_fn: legacy vs C600U-optimized (BF16, width=4)")
    print(
        f"{'TP':>3} {'tokens':>7} {'seqs':>5} {'dim':>5} "
        f"{'legacy(us)':>11} {'opt(us)':>9} {'speedup':>8} "
        f"{'legacy GB/s':>11} {'opt GB/s':>9} {'max_abs':>10}"
    )
    for result in results:
        print(
            f"{result.tp:>3} {result.tokens:>7} {result.requests:>5} "
            f"{result.dim:>5} {result.legacy_us:>11.2f} "
            f"{result.optimized_us:>9.2f} {result.speedup:>7.2f}x "
            f"{result.legacy_gbps:>11.2f} {result.optimized_gbps:>9.2f} "
            f"{result.max_abs_error:>10.6f}"
        )
    print(f"total elapsed time (including JIT compilation): {total_elapsed_s:.3f} s")


@pytest.mark.parametrize("activation", [None, "silu"])
def test_optimized_short_sequences_and_initial_state(activation: str | None) -> None:
    if not torch.cuda.is_available():
        pytest.skip("causal_conv1d_fn requires a CUDA device")

    torch.manual_seed(1)
    device = torch.device("cuda")
    dim = 320  # Exercises the channel-tail mask with BLOCK_N=256.
    seq_lens = [1, 2, 5, 33]
    tokens = sum(seq_lens)
    x = _make_strided_x(dim, tokens, device)
    weight = torch.randn((dim, KERNEL_WIDTH), device=device) * 0.25
    bias = torch.randn((dim,), device=device) * 0.1
    query_start_loc = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    cache_indices = torch.tensor([4, 2, 7, 1], device=device, dtype=torch.int32)
    has_initial_state = torch.tensor(
        [True, False, True, True], device=device, dtype=torch.bool
    )
    initial_states = torch.randn(
        (8, dim, KERNEL_WIDTH - 1), device=device, dtype=torch.bfloat16
    )
    expected, expected_states = _reference_with_state(
        x,
        weight,
        bias,
        query_start_loc,
        initial_states,
        cache_indices,
        has_initial_state,
        activation,
    )

    legacy_states = initial_states.clone()
    optimized_states = initial_states.clone()
    common = dict(
        x=x,
        weight=weight,
        bias=bias,
        query_start_loc=query_start_loc,
        seq_lens_cpu=seq_lens,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation=activation,
        validate_data=True,
        **_optimization_kwargs(),
    )
    legacy = causal_conv1d_fn(conv_states=legacy_states, **common)
    optimized = causal_conv1d_fwd(conv_states=optimized_states, **common)
    torch.cuda.synchronize()

    torch.testing.assert_close(legacy, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(optimized, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(optimized, legacy, rtol=2e-2, atol=2e-2)
    for state_id in cache_indices.tolist():
        torch.testing.assert_close(
            optimized_states[state_id], expected_states[state_id], rtol=0, atol=0
        )
        torch.testing.assert_close(
            optimized_states[state_id], legacy_states[state_id], rtol=0, atol=0
        )
