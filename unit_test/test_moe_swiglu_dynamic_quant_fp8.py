"""Correctness and performance baseline for FP8 moe_swiglu_dynamic_quantize.

The correctness suite intentionally uses the public ``mcoplib.op`` extension
and a PyTorch FP32 reference. It covers both supported floating input dtypes,
larger routed-token counts, uneven/empty experts, and the all-zero edge case.

The optional benchmark reports latency and two effective-bandwidth views and
can persist a JSON baseline for later speedup comparisons. It uses burst timing
so C600 DVFS and event overhead do not dominate the small shapes.

Examples:
  # Correctness only (no pytest dependency)
  python3 unit_test/test_moe_swiglu_dynamic_quant_fp8.py

  # Representative performance baseline
  python3 unit_test/test_moe_swiglu_dynamic_quant_fp8.py --benchmark-only \
      --output moe_swiglu_fp8_c600_baseline.json

  # Every num_tokens shape from the xpu-perf workload
  python3 unit_test/test_moe_swiglu_dynamic_quant_fp8.py --benchmark-only \
      --full-workload --output moe_swiglu_fp8_c600_full_baseline.json

  # Compare a later implementation with a saved baseline
  python3 unit_test/test_moe_swiglu_dynamic_quant_fp8.py --benchmark-only \
      --compare moe_swiglu_fp8_c600_baseline.json

  # Match the current xpu-sim workload: EP=1, 128 local experts.
  MOE_SWIGLU_TEST_EP_SIZE=1 \
  python3 unit_test/test_moe_swiglu_dynamic_quant_fp8.py --benchmark-only
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
FP8_MAX = 448.0
NUM_EXPERTS = 128
EP_SIZE = int(os.environ.get("MOE_SWIGLU_TEST_EP_SIZE", "8"))
if EP_SIZE <= 0 or NUM_EXPERTS % EP_SIZE != 0:
    raise ValueError(
        "MOE_SWIGLU_TEST_EP_SIZE must be a positive divisor of NUM_EXPERTS"
    )
EXPERTS_PER_RANK = NUM_EXPERTS // EP_SIZE
TOPK = 8
HIDDEN_SIZE = 1536

# Accuracy should remain stable on both C500 and C600. Raw FP8 bin agreement is
# diagnostic only; the user-visible contract is the dequantized value.
DEQUANT_COS_THRESHOLD = 0.999
SCALE_RTOL = 2e-3
SCALE_ATOL = 1e-6

# Larger than the original 16/64-token smoke test while staying practical for
# an ordinary correctness run.
CORRECTNESS_CASES = (
    (16, torch.bfloat16, "balanced"),
    (64, torch.float16, "balanced"),
    (128, torch.bfloat16, "sparse"),
    (512, torch.bfloat16, "imbalanced"),
    (1024, torch.float16, "imbalanced"),
    (4096, torch.bfloat16, "balanced"),
)

# Jira's explicitly prioritized acceptance shapes. --full-workload uses the
# complete num_tokens list from xpu-perf's xpu_sim workload for final sweeps.
PERF_NUM_TOKENS = (
    16, 64, 1024, 2048, 4096, 8192, 10240, 32768, 65536,
)
FULL_WORKLOAD_NUM_TOKENS = (
    16, 64, 128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792,
    2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 6144, 8192,
    10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
    28672, 30720, 32768, 65536,
)


def _load_moe_swiglu_dynamic_quantize():
    """Load the compiled extension or skip cleanly before mcoplib is built."""
    try:
        op_module = importlib.import_module("mcoplib.op")
    except (ImportError, OSError) as exc:
        raise unittest.SkipTest(f"mcoplib.op is not built or loadable: {exc}")

    op = getattr(op_module, "moe_swiglu_dynamic_quantize", None)
    if op is None:
        raise unittest.SkipTest(
            "mcoplib.op does not export moe_swiglu_dynamic_quantize"
        )
    return op


def _counts_from_weights(num_routed: int, weights: torch.Tensor) -> torch.Tensor:
    """Create deterministic integer expert counts that sum to num_routed."""
    weights = weights.to(torch.float64)
    raw = weights * (float(num_routed) / weights.sum().item())
    counts = torch.floor(raw).to(torch.int64)
    remainder = num_routed - int(counts.sum().item())
    if remainder:
        fractional = raw - counts
        order = torch.argsort(fractional, descending=True, stable=True)
        counts[order[:remainder]] += 1
    assert int(counts.sum().item()) == num_routed
    return counts.to(torch.int32)


def _build_routing(num_routed: int, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous expert counts/starts in the layout expected by the op."""
    if mode == "balanced":
        weights = torch.ones(EXPERTS_PER_RANK)
    elif mode == "imbalanced":
        # Deterministic long-tail distribution.
        weights = torch.arange(1, EXPERTS_PER_RANK + 1, dtype=torch.float64).square()
    elif mode == "sparse":
        # Exercise repeated offsets and empty experts.
        weights = torch.tensor(
            [1.0 if i % 3 == 0 else 0.0 for i in range(EXPERTS_PER_RANK)]
        )
    else:
        raise ValueError(f"unknown routing mode: {mode}")

    counts = _counts_from_weights(num_routed, weights)
    starts = torch.zeros_like(counts)
    starts[1:] = torch.cumsum(counts, dim=0)[:-1]
    return counts, starts


def _reference_fp8(
    scatter_tokens: torch.Tensor,
    smooth_scale: torch.Tensor,
    expert_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP32 reference for fused SwiGLU, expert smoothing and FP8 quantization."""
    gate, up = scatter_tokens.float().chunk(2, dim=-1)
    swiglu = F.silu(gate) * up

    expert_ids = torch.repeat_interleave(
        torch.arange(EXPERTS_PER_RANK, device=scatter_tokens.device),
        expert_counts.to(device=scatter_tokens.device, dtype=torch.int64),
    )
    smoothed = swiglu * smooth_scale[expert_ids]

    absmax = smoothed.abs().amax(dim=-1)
    per_token_scale = absmax / FP8_MAX
    inv_scale = torch.where(
        absmax > 0,
        FP8_MAX / absmax,
        torch.zeros_like(absmax),
    )

    # Current kernel uses __builtin_mxc_cvt_pk4_f16tof8. Model its FP16
    # intermediate rounding so it is not confused with a fused-op math error.
    scaled = (smoothed * inv_scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
    quantized = scaled.to(torch.float16).to(FP8_DTYPE)
    return quantized, per_token_scale, smoothed


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def _make_case(
    num_tokens: int,
    input_dtype: torch.dtype,
    routing: str,
    *,
    zero_input: bool = False,
) -> dict[str, Any]:
    """Allocate one reproducible device case."""
    device = torch.device("cuda")
    num_routed = num_tokens * TOPK
    seed = 20260820 + num_tokens + (17 if input_dtype == torch.float16 else 0)
    counts, starts = _build_routing(num_routed, routing)
    generator = torch.Generator(device=device).manual_seed(seed)

    if zero_input:
        scatter_tokens = torch.zeros(
            num_routed, 2 * HIDDEN_SIZE, dtype=input_dtype, device=device
        )
    else:
        scatter_tokens = torch.randn(
            num_routed,
            2 * HIDDEN_SIZE,
            dtype=input_dtype,
            device=device,
            generator=generator,
        )
    smooth_scale = (
        torch.rand(
            EXPERTS_PER_RANK,
            HIDDEN_SIZE,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        + 0.5
    ).contiguous()
    output = torch.empty(
        num_routed, HIDDEN_SIZE, dtype=FP8_DTYPE, device=device
    )
    per_token_scale = torch.empty(num_routed, dtype=torch.float32, device=device)

    return {
        "scatter": scatter_tokens,
        "smooth": smooth_scale,
        "counts_cpu": counts,
        "counts": counts.to(device),
        "starts": starts.to(device),
        "output": output,
        "scale": per_token_scale,
    }


def _run_correctness_case(
    op,
    num_tokens: int,
    input_dtype: torch.dtype,
    routing: str,
    *,
    zero_input: bool = False,
) -> dict[str, float]:
    case = _make_case(num_tokens, input_dtype, routing, zero_input=zero_input)
    op(
        case["scatter"], case["smooth"], case["starts"], case["counts"],
        case["output"], case["scale"], EXPERTS_PER_RANK,
    )
    torch.cuda.synchronize()

    ref_output, ref_scale, ref_smoothed = _reference_fp8(
        case["scatter"], case["smooth"], case["counts_cpu"]
    )

    assert case["output"].dtype == FP8_DTYPE
    assert torch.isfinite(case["output"].float()).all()
    assert torch.isfinite(case["scale"]).all()
    torch.testing.assert_close(
        case["scale"], ref_scale, rtol=SCALE_RTOL, atol=SCALE_ATOL
    )

    raw_output_cos = _cosine(case["output"], ref_output)
    dequantized = case["output"].float() * case["scale"].unsqueeze(-1)
    ref_dequantized = ref_output.float() * ref_scale.unsqueeze(-1)
    dequant_vs_ref_cos = _cosine(dequantized, ref_dequantized)
    dequant_vs_input_cos = _cosine(dequantized, ref_smoothed)
    relative_l2 = (
        torch.linalg.vector_norm(dequantized - ref_smoothed)
        / torch.linalg.vector_norm(ref_smoothed).clamp_min(1e-12)
    ).item()

    if zero_input:
        assert torch.count_nonzero(case["output"].float()).item() == 0
        assert torch.count_nonzero(case["scale"]).item() == 0
    else:
        metrics = (
            f"T={num_tokens}, dtype={input_dtype}, routing={routing}, "
            f"raw_output_cos={raw_output_cos}, "
            f"dequant_vs_ref_cos={dequant_vs_ref_cos}, "
            f"dequant_vs_input_cos={dequant_vs_input_cos}, "
            f"relative_l2={relative_l2}"
        )
        assert dequant_vs_input_cos >= DEQUANT_COS_THRESHOLD, metrics

    return {
        "raw_output_cos": raw_output_cos,
        "dequant_vs_ref_cos": dequant_vs_ref_cos,
        "dequant_vs_input_cos": dequant_vs_input_cos,
        "relative_l2": relative_l2,
    }


class TestMoeSwigluDynamicQuantFP8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if FP8_DTYPE is None:
            raise unittest.SkipTest(
                "this PyTorch build does not provide float8_e4m3fn"
            )
        if not torch.cuda.is_available():
            raise unittest.SkipTest("a CUDA/MACA device is required")
        cls.op = _load_moe_swiglu_dynamic_quantize()

    def test_correctness_matrix(self):
        for num_tokens, input_dtype, routing in CORRECTNESS_CASES:
            with self.subTest(
                num_tokens=num_tokens, input_dtype=input_dtype, routing=routing
            ):
                metrics = _run_correctness_case(
                    self.op, num_tokens, input_dtype, routing
                )
                print(
                    f"[accuracy] T={num_tokens:<5} dtype={str(input_dtype):<14} "
                    f"routing={routing:<10} "
                    f"raw_cos={metrics['raw_output_cos']:.7f} "
                    f"dequant_cos={metrics['dequant_vs_input_cos']:.7f} "
                    f"rel_l2={metrics['relative_l2']:.7f}"
                )

    def test_all_zero_input(self):
        _run_correctness_case(
            self.op, 16, torch.bfloat16, "sparse", zero_input=True
        )


def _auto_iterations(num_tokens: int) -> int:
    if num_tokens <= 256:
        return 200
    if num_tokens <= 4096:
        return 100
    if num_tokens <= 16384:
        return 40
    return 10


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _benchmark_shape(
    op,
    num_tokens: int,
    *,
    warmup: int,
    samples: int,
    iterations: int,
) -> dict[str, Any]:
    case = _make_case(num_tokens, torch.bfloat16, "balanced")

    def invoke():
        op(
            case["scatter"], case["smooth"], case["starts"], case["counts"],
            case["output"], case["scale"], EXPERTS_PER_RANK,
        )

    for _ in range(warmup):
        invoke()
    torch.cuda.synchronize()

    per_launch_ms = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            invoke()
        end.record()
        end.synchronize()
        per_launch_ms.append(start.elapsed_time(end) / iterations)

    latency_ms = statistics.median(per_launch_ms)
    latency_min_ms = min(per_launch_ms)
    latency_max_ms = max(per_launch_ms)
    num_routed = num_tokens * TOPK

    # Tensor-level xpu-perf convention: every input tensor once and every
    # output tensor once per invocation.
    logical_read_bytes = sum(
        _tensor_bytes(case[name])
        for name in ("scatter", "smooth", "starts", "counts")
    )
    logical_write_bytes = _tensor_bytes(case["output"]) + _tensor_bytes(
        case["scale"]
    )
    logical_io_bytes = logical_read_bytes + logical_write_bytes

    # Excludes small metadata and the L2-reused smooth table; this matches the
    # operator-specific effective-bandwidth convention.
    core_bytes = num_routed * (2 * HIDDEN_SIZE * 2 + HIDDEN_SIZE + 4)
    seconds = latency_ms * 1e-3

    return {
        "num_tokens": num_tokens,
        "num_routed": num_routed,
        "latency_us": latency_ms * 1e3,
        "latency_min_us": latency_min_ms * 1e3,
        "latency_max_us": latency_max_ms * 1e3,
        "noise_percent": (latency_max_ms - latency_min_ms) / latency_ms * 100.0,
        "logical_read_bytes": logical_read_bytes,
        "logical_write_bytes": logical_write_bytes,
        "logical_io_bytes": logical_io_bytes,
        "logical_io_gbps": logical_io_bytes / seconds / 1e9,
        "effective_core_gbps": core_bytes / seconds / 1e9,
        "routed_rows_per_second": num_routed / seconds,
        "warmup": warmup,
        "samples": samples,
        "iterations_per_sample": iterations,
    }


def _load_baseline(path: str | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(row["num_tokens"]): row for row in payload["results"]}


def _run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if FP8_DTYPE is None:
        raise RuntimeError("this PyTorch build does not provide float8_e4m3fn")
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA/MACA device is required")

    op = _load_moe_swiglu_dynamic_quantize()
    shapes = (
        tuple(args.shapes)
        if args.shapes
        else FULL_WORKLOAD_NUM_TOKENS
        if args.full_workload
        else PERF_NUM_TOKENS
    )
    baseline = _load_baseline(args.compare)
    results = []

    print(
        "\n[benchmark] bf16 -> float8_e4m3fn, "
        f"experts/rank={EXPERTS_PER_RANK}, topk={TOPK}, hidden={HIDDEN_SIZE}"
    )
    print(
        "  T(tokens)  routed    latency(us)  noise(%)  "
        "logical GB/s  core GB/s  speedup"
    )
    speedups = []
    for num_tokens in shapes:
        iterations = args.iterations or _auto_iterations(num_tokens)
        row = _benchmark_shape(
            op,
            num_tokens,
            warmup=args.warmup,
            samples=args.samples,
            iterations=iterations,
        )
        old = baseline.get(num_tokens)
        if old is not None:
            row["speedup_vs_baseline"] = old["latency_us"] / row["latency_us"]
            speedup = f"{row['speedup_vs_baseline']:.3f}x"
            speedups.append(row["speedup_vs_baseline"])
        else:
            speedup = "-"
        results.append(row)
        print(
            f"  {num_tokens:>9}  {row['num_routed']:>7}  "
            f"{row['latency_us']:>13.3f}  "
            f"{row['noise_percent']:>8.3f}  "
            f"{row['logical_io_gbps']:>12.3f}  "
            f"{row['effective_core_gbps']:>9.3f}  {speedup:>7}"
        )
        torch.cuda.empty_cache()

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload = {
        "schema_version": 1,
        "operator": "moe_swiglu_dynamic_quantize",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": {
            "name": props.name,
            "total_memory_bytes": props.total_memory,
            "multi_processor_count": props.multi_processor_count,
            "visible_device_count": torch.cuda.device_count(),
        },
        "software": {"torch_version": torch.__version__},
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "MOE_SWIGLU_WPB": os.environ.get("MOE_SWIGLU_WPB"),
            "MOE_SWIGLU_RPW": os.environ.get("MOE_SWIGLU_RPW"),
        },
        "config": {
            "input_dtype": "bfloat16",
            "output_dtype": "float8_e4m3fn",
            "num_experts": NUM_EXPERTS,
            "ep_size": EP_SIZE,
            "experts_per_rank": EXPERTS_PER_RANK,
            "topk": TOPK,
            "hidden_size": HIDDEN_SIZE,
            "routing": "balanced",
        },
        "results": results,
    }

    if speedups:
        geomean_speedup = math.exp(
            sum(math.log(value) for value in speedups) / len(speedups)
        )
        payload["comparison"] = {
            "baseline": str(Path(args.compare).resolve()),
            "matched_shapes": len(speedups),
            "geomean_speedup": geomean_speedup,
        }
        print(
            f"[benchmark] geometric-mean speedup across {len(speedups)} "
            f"matched shapes: {geomean_speedup:.4f}x"
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[benchmark] wrote baseline: {output_path.resolve()}")
    return payload


def _parse_cli() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--full-workload", action="store_true")
    parser.add_argument(
        "--shapes",
        type=lambda value: [int(item) for item in value.split(",") if item],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="launches per sample; zero selects a shape-dependent value",
    )
    parser.add_argument("--output")
    parser.add_argument("--compare")
    return parser.parse_known_args()


if __name__ == "__main__":
    cli_args, unittest_args = _parse_cli()
    test_ok = True
    if not cli_args.benchmark_only:
        program = unittest.main(
            argv=[sys.argv[0], *unittest_args], verbosity=2, exit=False
        )
        test_ok = program.result.wasSuccessful()
    if test_ok and (cli_args.benchmark or cli_args.benchmark_only):
        _run_benchmark(cli_args)
    if not test_ok:
        raise SystemExit(1)
