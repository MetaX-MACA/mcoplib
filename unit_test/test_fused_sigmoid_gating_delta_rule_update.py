"""Kimi-K3 fused sigmoid-gating delta-rule update shape sweep.

This is a single-device kernel test sweeping 6, 12, 24, and 48 heads on one GPU.

The default sweep covers the production Kimi-K3 DSpARK target-verify shapes
requested for heads={6,12,24,48}, batch_size={1,8,32,128,256}, and
tokens={1,8,32,128,256}.  Every case:

Examples::

    python unit_test/test_fused_sigmoid_gating_delta_rule_update.py
    python unit_test/test_fused_sigmoid_gating_delta_rule_update.py \
        --batch-sizes 8,16,32 --tokens 1,8,32 \
        --warmup 20 --iters 100
    python unit_test/test_fused_sigmoid_gating_delta_rule_update.py \
        --mode decode --batch-sizes 8 --no-check
"""

from __future__ import annotations

import argparse
import gc
import math
from dataclasses import dataclass
from typing import Iterable

import pytest
import torch

from mcoplib.triton_fused_sigmoid_gating_delta_rule_update import (
    _fused_sigmoid_gating_delta_rule_update_with_strategy,
    fused_sigmoid_gating_delta_rule_update,
)


DEFAULT_BATCH_SIZES = (1, 32, 128, 256)

TOTAL_HEADS = (6, 12, 24, 48)
HEAD_K_DIM = 128
HEAD_V_DIM = 128
LOWER_BOUND = -5.0
TOKENS = (1, 8,32, 128, 256)
DTYPE = torch.bfloat16
STATE_DTYPE = torch.float32


@dataclass
class KernelCase:
    mode: str
    num_steps: int
    batch_size: int
    local_heads: int
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    state: torch.Tensor
    state_indices: torch.Tensor
    cu_seqlens: torch.Tensor
    replayssm_rawv: torch.Tensor | None
    replayssm_rawk: torch.Tensor | None
    replayssm_g: torch.Tensor | None
    replayssm_beta: torch.Tensor | None


@dataclass
class CaseResult:
    kernel: str
    batch_size: int
    num_steps: int
    local_heads: int
    latency_us: float
    bandwidth_gbps: float
    max_abs_diff: float
    speedup: float = math.nan


def _scaled_randn(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    scale: float = 1.0,
    offset: float = 0.0,
) -> torch.Tensor:
    value = torch.empty(shape, device=device, dtype=dtype)
    value.normal_()
    if scale != 1.0:
        value.mul_(scale)
    if offset != 0.0:
        value.add_(offset)
    return value


def make_case(
    total_heads: int,
    batch_size: int,
    *,
    mode: str,
    num_steps: int,
    device: torch.device,
    seed: int,
) -> KernelCase:
    if mode not in ("decode", "verify"):
        raise ValueError(f"unsupported mode: {mode!r}")
    if mode == "decode":
        num_steps = 1
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    # q/k/v use the wrapper's packed-varlen layout: B=1,
    # T=batch_size*num_steps, and cu_seqlens divides the flat token axis into
    # one independent sequence per request.
    local_heads = total_heads
    total_tokens = batch_size * num_steps
    torch.manual_seed(seed)

    q = _scaled_randn(
        (1, total_tokens, local_heads, HEAD_K_DIM),
        device=device,
        dtype=DTYPE,
    )
    k = _scaled_randn(
        (1, total_tokens, local_heads, HEAD_K_DIM),
        device=device,
        dtype=DTYPE,
    )
    v = _scaled_randn(
        (1, total_tokens, local_heads, HEAD_V_DIM),
        device=device,
        dtype=DTYPE,
        scale=0.1,
    )
    # Raw per-K forget gate and raw per-head beta.  These are activated inside
    # fused_sigmoid_gating_delta_rule_update_kernel.
    gate_shape = (
        (batch_size, local_heads * HEAD_K_DIM)
        if mode == "decode"
        else (1, total_tokens, local_heads, HEAD_K_DIM)
    )
    a = _scaled_randn(
        gate_shape, device=device, dtype=DTYPE, scale=0.5, offset=-1.0
    )
    b = _scaled_randn(
        (1, total_tokens, local_heads),
        device=device,
        dtype=DTYPE,
        scale=0.5,
    )
    A_log = _scaled_randn(
        (1, 1, local_heads, 1), device=device, dtype=torch.float32, scale=0.2
    )
    dt_bias = _scaled_randn(
        (local_heads * HEAD_K_DIM,),
        device=device,
        dtype=torch.float32,
        scale=0.1,
    )

    # One state-pool slot per request is sufficient for a direct decode call.
    # 最大组合 BS256/48 heads 的 FP32 持久状态约占 768 MiB。
    state = _scaled_randn(
        (batch_size, local_heads, HEAD_V_DIM, HEAD_K_DIM),
        device=device,
        dtype=STATE_DTYPE,
        scale=0.01,
    )
    state_indices = torch.arange(batch_size, device=device, dtype=torch.int32)
    cu_seqlens = torch.arange(
        0, total_tokens + 1, num_steps, device=device, dtype=torch.int32
    )

    # DSpARK ReplaySSM target verify stores raw inputs and activated gates into
    # a compact per-request ring, then a separate exact-fold kernel commits the
    # accepted prefix. It intentionally does not write the persistent state.
    if mode == "verify":
        replayssm_rawv = torch.empty(
            batch_size,
            local_heads,
            num_steps,
            HEAD_V_DIM,
            device=device,
            dtype=DTYPE,
        )
        replayssm_rawk = torch.empty(
            batch_size,
            local_heads,
            num_steps,
            HEAD_K_DIM,
            device=device,
            dtype=DTYPE,
        )
        replayssm_g = torch.empty(
            batch_size,
            local_heads,
            num_steps,
            HEAD_K_DIM,
            device=device,
            dtype=torch.float32,
        )
        replayssm_beta = torch.empty(
            batch_size,
            local_heads,
            num_steps,
            device=device,
            dtype=torch.float32,
        )
    else:
        replayssm_rawv = None
        replayssm_rawk = None
        replayssm_g = None
        replayssm_beta = None

    return KernelCase(
        mode=mode,
        num_steps=num_steps,
        batch_size=batch_size,
        local_heads=local_heads,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        state=state,
        state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        replayssm_rawv=replayssm_rawv,
        replayssm_rawk=replayssm_rawk,
        replayssm_g=replayssm_g,
        replayssm_beta=replayssm_beta,
    )


def call_kernel(case: KernelCase, *, optimized: bool) -> torch.Tensor:
    kwargs = dict(
        A_log=case.A_log,
        a=case.a,
        dt_bias=case.dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=case.q,
        k=case.k,
        v=case.v,
        b=case.b,
        initial_state_source=case.state,
        initial_state_indices=case.state_indices,
        scale=HEAD_K_DIM**-0.5,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=case.cu_seqlens,
        is_kda=True,
        lower_bound=LOWER_BOUND,
        disable_state_update=case.mode == "verify",
        cache_ring=case.mode == "verify",
        replayssm_rawv=case.replayssm_rawv,
        replayssm_rawk=case.replayssm_rawk,
        replayssm_g=case.replayssm_g,
        replayssm_beta=case.replayssm_beta,
    )
    if optimized:
        return fused_sigmoid_gating_delta_rule_update(**kwargs)
    return _fused_sigmoid_gating_delta_rule_update_with_strategy(
        **kwargs, strategy="original"
    )


def _sample_indices(size: int) -> torch.Tensor:
    values = [0] if size == 1 else [0, size - 1]
    return torch.tensor(values, dtype=torch.long)


def check_case(case: KernelCase, *, optimized: bool) -> tuple[float, float, float]:
    """Check first/last request and head without cloning the full state pool."""

    device = case.state.device
    request_indices = _sample_indices(case.batch_size).to(device)
    head_indices = _sample_indices(case.local_heads).to(device)
    step_indices = torch.arange(case.num_steps, device=device, dtype=torch.long)
    token_indices = (
        request_indices[:, None] * case.num_steps + step_indices[None, :]
    ).flatten()
    num_sampled_requests = request_indices.numel()
    num_sampled_heads = head_indices.numel()

    # Save only the sampled state before the in-place kernel call.  This keeps
    # 即使 BS=256，也只复制抽样状态，避免精度检查额外占用大量显存。
    state_before = case.state.index_select(0, request_indices).index_select(
        1, head_indices
    ).clone()

    output = call_kernel(case, optimized=optimized)
    torch.cuda.synchronize(device)

    def select_token_heads(value: torch.Tensor) -> torch.Tensor:
        return (
            value.index_select(0, token_indices)
            .view(num_sampled_requests, case.num_steps, case.local_heads, -1)
            .index_select(2, head_indices)
            .float()
        )

    q = select_token_heads(case.q[0])
    k = select_token_heads(case.k[0])
    v = select_token_heads(case.v[0])
    gate_values = (
        case.a.view(-1, case.local_heads, HEAD_K_DIM)
        if case.mode == "verify"
        else case.a.view(case.batch_size, case.local_heads, HEAD_K_DIM)
    )
    raw_gate = (
        gate_values.index_select(0, token_indices)
        .view(
            num_sampled_requests,
            case.num_steps,
            case.local_heads,
            HEAD_K_DIM,
        )
        .index_select(2, head_indices)
        .float()
    )
    raw_beta = (
        case.b[0]
        .index_select(0, token_indices)
        .view(num_sampled_requests, case.num_steps, case.local_heads)
        .index_select(2, head_indices)
        .float()
    )
    A_log = case.A_log.reshape(-1).index_select(0, head_indices).float()
    dt_bias = (
        case.dt_bias.view(case.local_heads, HEAD_K_DIM)
        .index_select(0, head_indices)
        .float()
    )

    q = q / torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)
    gate = LOWER_BOUND * torch.sigmoid(
        torch.exp(A_log)[None, None, :, None]
        * (raw_gate + dt_bias[None, None, :, :])
    )
    beta = torch.sigmoid(raw_beta)

    expected_state = state_before
    expected_outputs = []
    for step in range(case.num_steps):
        expected_state = expected_state * torch.exp(gate[:, step]).unsqueeze(-2)
        delta = v[:, step] - torch.sum(
            expected_state * k[:, step].unsqueeze(-2), dim=-1
        )
        delta = delta * beta[:, step].unsqueeze(-1)
        expected_state = (
            expected_state + delta.unsqueeze(-1) * k[:, step].unsqueeze(-2)
        )
        scaled_q = q[:, step] * (HEAD_K_DIM**-0.5)
        expected_outputs.append(
            torch.sum(expected_state * scaled_q.unsqueeze(-2), dim=-1)
        )
    expected_output = torch.stack(expected_outputs, dim=1)

    actual_output = (
        output[0]
        .index_select(0, token_indices)
        .view(
            num_sampled_requests,
            case.num_steps,
            case.local_heads,
            HEAD_V_DIM,
        )
        .index_select(2, head_indices)
        .float()
    )
    actual_state = (
        case.state.index_select(0, request_indices)
        .index_select(1, head_indices)
        .float()
    )

    output_max_abs_diff = (actual_output - expected_output).abs().max().item()
    expected_persistent_state = (
        state_before if case.mode == "verify" else expected_state
    )
    state_max_abs_diff = (
        (actual_state - expected_persistent_state).abs().max().item()
    )

    ring_max_abs_diff = math.nan
    if case.mode == "verify":
        assert case.replayssm_rawv is not None
        assert case.replayssm_rawk is not None
        assert case.replayssm_g is not None
        assert case.replayssm_beta is not None

        ring_rawv = (
            case.replayssm_rawv.index_select(0, request_indices)
            .index_select(1, head_indices)
            .transpose(1, 2)
            .float()
        )
        ring_rawk = (
            case.replayssm_rawk.index_select(0, request_indices)
            .index_select(1, head_indices)
            .transpose(1, 2)
            .float()
        )
        ring_gate = (
            case.replayssm_g.index_select(0, request_indices)
            .index_select(1, head_indices)
            .transpose(1, 2)
            .float()
        )
        ring_beta = (
            case.replayssm_beta.index_select(0, request_indices)
            .index_select(1, head_indices)
            .transpose(1, 2)
            .float()
        )
        ring_diffs = (
            (ring_rawv - v).abs().max(),
            (ring_rawk - select_token_heads(case.k[0])).abs().max(),
            (ring_gate - gate).abs().max(),
            (ring_beta - beta).abs().max(),
        )
        ring_max_abs_diff = max(item.item() for item in ring_diffs)

        torch.testing.assert_close(ring_rawv, v, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            ring_rawk, select_token_heads(case.k[0]), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(ring_gate, gate, atol=2e-3, rtol=1e-2)
        torch.testing.assert_close(ring_beta, beta, atol=2e-3, rtol=1e-2)

    # BF16 output quantization and the Triton exp/sigmoid approximations account
    # for the tolerance.  State accumulation/storage itself is FP32.
    torch.testing.assert_close(
        actual_output, expected_output, atol=2e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_state, expected_persistent_state, atol=2e-3, rtol=1e-2
    )
    return output_max_abs_diff, state_max_abs_diff, ring_max_abs_diff


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def benchmark_case(
    case: KernelCase, *, optimized: bool, warmup: int, iterations: int
) -> float:
    if iterations <= 0:
        return math.nan

    output = None
    for _ in range(warmup):
        output = call_kernel(case, optimized=optimized)
    torch.cuda.synchronize(case.state.device)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        output = call_kernel(case, optimized=optimized)
        end.record()
    torch.cuda.synchronize(case.state.device)

    # Keep the final output alive until all queued launches have completed.
    assert output is not None
    times_us = [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]
    return _percentile(times_us, 0.5)


def run_case(
    total_heads: int,
    batch_size: int,
    *,
    mode: str,
    num_steps: int,
    device: torch.device,
    seed: int,
    warmup: int,
    iterations: int,
    check: bool,
    optimized: bool,
) -> CaseResult:
    case = make_case(
        total_heads,
        batch_size,
        mode=mode,
        num_steps=num_steps,
        device=device,
        seed=seed,
    )

    if check:
        output_diff, state_diff, ring_diff = check_case(
            case, optimized=optimized
        )
    else:
        # Compile and prove that the requested shape launches successfully.
        call_kernel(case, optimized=optimized)
        torch.cuda.synchronize(device)
        output_diff = math.nan
        state_diff = math.nan
        ring_diff = math.nan

    latency_us = benchmark_case(
        case, optimized=optimized, warmup=warmup, iterations=iterations
    )
    # Decode reads+writes one FP32 [V,K] state matrix per local head. ReplaySSM
    # verify reads it once and leaves the persistent state unchanged. This is a
    # state-traffic lower-bound metric; it intentionally excludes q/k/v and
    # verify-ring traffic.
    state_access_count = 2 if case.mode == "decode" else 1
    state_rw_bytes = (
        state_access_count
        * batch_size
        * case.local_heads
        * HEAD_V_DIM
        * HEAD_K_DIM
        * torch.tensor([], dtype=STATE_DTYPE).element_size()
    )
    # Logical tensor traffic for one kernel invocation.  This counts every
    # input/output element once (plus the persistent-state traffic above), so
    # it is an algorithmic effective-bandwidth metric rather than a claim about
    # physical HBM transactions.  In particular, the kernel's V tiles may
    # request q/k/g more than once and those requests can be served by cache.
    input_element_size = torch.empty((), dtype=DTYPE).element_size()
    state_element_size = torch.empty((), dtype=STATE_DTYPE).element_size()
    token_heads = batch_size * case.num_steps * case.local_heads
    logical_io_bytes = state_rw_bytes
    logical_io_bytes += token_heads * (
        # q read + k read + per-K raw gate read
        3 * HEAD_K_DIM * input_element_size
        # v read + output write
        + 2 * HEAD_V_DIM * input_element_size
        # raw beta read
        + input_element_size
    )
    # A_log and dt_bias are FP32 parameters. Count their logical reads once per
    # invocation, independent of cache behavior and V tiling.
    logical_io_bytes += case.local_heads * (1 + HEAD_K_DIM) * state_element_size
    if case.mode == "verify":
        # ReplaySSM ring writes: rawv, rawk, activated per-K gate, and beta.
        logical_io_bytes += token_heads * (
            (HEAD_V_DIM + HEAD_K_DIM) * input_element_size
            + (HEAD_K_DIM + 1) * state_element_size
        )
    if optimized and batch_size * case.local_heads >= 104:
        # 优化路径新增的中间张量流量：gate_decay 和 beta 各写一次、读一次；
        # verify 路径还需要保留 g，供递推 kernel 写入 ReplaySSM ring。
        logical_io_bytes += 2 * token_heads * (
            HEAD_K_DIM + 1
        ) * state_element_size
        if case.mode == "verify":
            logical_io_bytes += (
                2 * token_heads * HEAD_K_DIM * state_element_size
            )
    bandwidth_gbps = (
        logical_io_bytes / latency_us / 1000.0
        if math.isfinite(latency_us)
        else math.nan
    )
    finite_diffs = [
        diff for diff in (output_diff, state_diff, ring_diff) if math.isfinite(diff)
    ]
    max_abs_diff = max(finite_diffs, default=math.nan)

    return CaseResult(
        kernel="opt" if optimized else "base",
        batch_size=batch_size,
        num_steps=case.num_steps,
        local_heads=case.local_heads,
        latency_us=latency_us,
        bandwidth_gbps=bandwidth_gbps,
        max_abs_diff=max_abs_diff,
    )


def _parse_int_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list: {value!r}") from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("list must contain positive integers")
    return result


def _fmt(value: float, digits: int = 3) -> str:
    return "-" if not math.isfinite(value) else f"{value:.{digits}f}"


def _fmt_error(value: float) -> str:
    return "-" if not math.isfinite(value) else f"{value:.3e}"


def print_header() -> None:
    print(
        f"{'kernel':>7} {'H':>5} {'B':>4} {'T':>4} "
        f"{'耗时(us)':>10} {'带宽(GB/s)':>14} {'最大绝对误差':>10} "
        f"{'加速比':>6}"
    )


def print_result(result: CaseResult) -> None:
    print(
        f"{result.kernel:>7} {result.local_heads:>5} {result.batch_size:>4} "
        f"{result.num_steps:>4} "
        f"{_fmt(result.latency_us):>12} "
        f"{_fmt(result.bandwidth_gbps, 2):>16} "
        f"{_fmt_error(result.max_abs_diff):>14} "
        f"{_fmt(result.speedup, 2):>9}"
    )


def run_sweep(
    head_counts: Iterable[int],
    batch_sizes: Iterable[int],
    token_counts: Iterable[int],
    *,
    mode: str,
    device: torch.device,
    seed: int,
    warmup: int,
    iterations: int,
    check: bool,
    kernel: str,
) -> list[CaseResult]:
    results = []
    print_header()
    for total_heads in head_counts:
        for batch_size in batch_sizes:
            for requested_steps in token_counts:
                num_steps = requested_steps if mode == "verify" else 1
                case_seed = (
                    seed + total_heads * 1_000_000 + batch_size * 1_000 + num_steps
                )
                variants = (False, True) if kernel == "both" else (kernel == "opt",)
                pair = []
                for optimized in variants:
                    try:
                        result = run_case(
                            total_heads,
                            batch_size,
                            mode=mode,
                            num_steps=num_steps,
                            device=device,
                            seed=case_seed,
                            warmup=warmup,
                            iterations=iterations,
                            check=check,
                            optimized=optimized,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"测试失败：kernel={'opt' if optimized else 'base'}，"
                            f"H={total_heads}，B={batch_size}，T={num_steps}"
                        ) from exc
                    pair.append(result)
                    gc.collect()
                    torch.cuda.empty_cache()
                if len(pair) == 2:
                    pair[1].speedup = pair[0].latency_us / pair[1].latency_us
                for result in pair:
                    results.append(result)
                    print_result(result)
    return results


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA-compatible GPU"
)
@pytest.mark.parametrize(
    ("mode", "total_heads", "batch_size", "num_steps"),
    (
        ("decode", 6, 1, 1),       # original strategy
        ("verify", 12, 8, 4),      # single-kernel optimized strategy
        ("verify", 24, 8, 4),      # dual-kernel optimized strategy
    ),
)
def test_fused_sigmoid_gating_delta_rule_update(
    mode: str, total_heads: int, batch_size: int, num_steps: int
) -> None:
    """Validate outputs, state updates, and ReplaySSM ring writes."""
    device = torch.device("cuda:0")
    case = make_case(
        total_heads,
        batch_size,
        mode=mode,
        num_steps=num_steps,
        device=device,
        seed=42 + total_heads + batch_size + num_steps,
    )
    check_case(case, optimized=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Kimi-K3 fused_sigmoid_gating_delta_rule_update_kernel "
            "单卡精度与性能测试"
        )
    )
    parser.add_argument(
        "--total-heads",
        type=_parse_int_list,
        default=TOTAL_HEADS,
        help="head count，逗号分隔（默认：6,12,24,48）",
    )
    parser.add_argument(
        "--batch-sizes",
        type=_parse_int_list,
        default=DEFAULT_BATCH_SIZES,
        help="batch size，逗号分隔（默认：1,32,128,256）",
    )
    parser.add_argument(
        "--mode",
        choices=("verify", "decode"),
        default="verify",
        help=(
            "verify 测试 DSpARK ReplaySSM 验证路径；decode 测试单令牌状态写回路径"
        ),
    )
    parser.add_argument(
        "--tokens",
        type=_parse_int_list,
        default=TOKENS,
        help="token count，逗号分隔（默认：1,32,128,256）",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--kernel",
        choices=("both", "base", "opt"),
        default="both",
        help="对比原 kernel 与优化 kernel，或只运行其中一个（默认：both）",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="跳过 PyTorch 参考精度检查",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("该测试需要支持 CUDA 的 GPU")
    if args.warmup < 0 or args.iters < 0:
        raise ValueError("--warmup 和 --iters 不能为负数")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    print(
        "Kimi-K3 KDA 单卡测试："
        f"设备={torch.cuda.get_device_name(device)}，K=V={HEAD_K_DIM}，"
        f"输入类型={DTYPE}，状态类型={STATE_DTYPE}，"
        f"LOWER_BOUND={LOWER_BOUND}，模式={args.mode}，"
        f"H={args.total_heads}，"
        f"T={args.tokens if args.mode == 'verify' else (1,)}"
    )
    results = run_sweep(
        args.total_heads,
        args.batch_sizes,
        args.tokens if args.mode == "verify" else (1,),
        mode=args.mode,
        device=device,
        seed=args.seed,
        warmup=args.warmup,
        iterations=args.iters,
        check=not args.no_check,
        kernel=args.kernel,
    )
    print(f"通过：完成 {len(results)} 个单卡B/T组合")


if __name__ == "__main__":
    main()
