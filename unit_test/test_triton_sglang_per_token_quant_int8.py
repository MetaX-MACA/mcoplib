# SPDX-License-Identifier: Apache-2.0
"""Correctness and frozen baseline for SGLang per-token INT8 quantization."""

import argparse
import json
import math
import pathlib
import statistics
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mcoplib.triton_sglang_per_token_quant_int8 import (  # noqa: E402
    get_per_token_quant_rows_per_program,
    launch_per_token_quant_int8,
)


CASES = ((3072, 7168), (8192, 7168), (48152, 384), (131072, 384))
SUM_CASES = ((48152, 384), (131072, 384))
EDGE_CASES = (
    (1, 7168, None),
    (3, 7167, None),
    (17, 384, 8),
    (17, 385, 8),
)


def reference(x):
    x_f = x.float()
    absmax = x_f.abs().amax(dim=1).clamp_min(1e-10)
    scaled = x_f * (127.0 / absmax[:, None])
    rounded = torch.where(
        scaled >= 0, torch.floor(scaled + 0.5), torch.ceil(scaled - 0.5)
    )
    return rounded.to(torch.int8), (absmax / 127.0)[:, None]


def make_case(M, K, seed=20260910):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    x_q = torch.empty((M, K), device="cuda", dtype=torch.int8)
    scales = torch.empty((M, 1), device="cuda", dtype=torch.float32)
    return x, x_q, scales


@torch.inference_mode()
def check_case(M, K, rows_per_program=None, cal_sum=False):
    x, actual_q, actual_scales = make_case(M, K)
    baseline_q = torch.empty_like(actual_q)
    baseline_scales = torch.empty_like(actual_scales)
    actual_sum = (
        torch.empty((M,), device="cuda", dtype=x.dtype)
        if cal_sum
        else None
    )
    baseline_sum = torch.empty_like(actual_sum) if cal_sum else None
    launch_per_token_quant_int8(
        x,
        baseline_q,
        baseline_scales,
        baseline_sum,
        rows_per_program=rows_per_program,
        use_baseline=True,
    )
    launch_per_token_quant_int8(
        x,
        actual_q,
        actual_scales,
        actual_sum,
        rows_per_program=rows_per_program,
    )
    torch.cuda.synchronize()
    expected_q, expected_scales = reference(x)
    max_q_diff = int(
        (actual_q.int() - expected_q.int()).abs().max().item()
    )
    max_scale_diff = float(
        (actual_scales - expected_scales).abs().max().item()
    )
    sum_close = not cal_sum or torch.allclose(
        actual_sum,
        x.float().sum(dim=1).to(x.dtype),
        rtol=1e-2,
        atol=1e-2,
    )
    baseline_match = (
        torch.equal(actual_q, baseline_q)
        and torch.equal(actual_scales, baseline_scales)
        and (not cal_sum or torch.equal(actual_sum, baseline_sum))
    )
    scale_close = torch.allclose(
        actual_scales, expected_scales, rtol=1e-6, atol=1e-7
    )
    passed = max_q_diff <= 1 and scale_close and sum_close
    return {
        "passed": bool(passed),
        "baseline_match": bool(baseline_match),
        "max_q_diff": max_q_diff,
        "max_scale_diff": max_scale_diff,
        "scale_allclose": bool(scale_close),
        "sum_allclose": bool(sum_close),
    }


@torch.inference_mode()
def check_rounding_boundaries():
    values = torch.tensor(
        [
            -127.0,
            -126.5,
            -125.5,
            -2.5,
            -1.5,
            -0.50390625,
            -0.5,
            -0.49609375,
            0.0,
            0.49609375,
            0.5,
            0.50390625,
            1.5,
            2.5,
            125.5,
            126.5,
            127.0,
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected = torch.tensor(
        [
            -127,
            -127,
            -126,
            -3,
            -2,
            -1,
            -1,
            0,
            0,
            0,
            1,
            1,
            2,
            3,
            126,
            127,
            127,
        ],
        device="cuda",
        dtype=torch.int8,
    )
    cases = []
    for M, K, rows_per_program in ((3, 7168, 1), (8, 384, 8)):
        x = torch.zeros((M, K), device="cuda", dtype=torch.bfloat16)
        x[1, : values.numel()] = values
        x[2, :] = torch.linspace(
            -127, 127, K, device="cuda", dtype=torch.bfloat16
        )
        baseline_q = torch.empty_like(x, dtype=torch.int8)
        actual_q = torch.empty_like(baseline_q)
        baseline_scales = torch.empty(
            (M, 1), device="cuda", dtype=torch.float32
        )
        actual_scales = torch.empty_like(baseline_scales)
        baseline_sum = torch.empty((M,), device="cuda", dtype=x.dtype)
        actual_sum = torch.empty_like(baseline_sum)
        launch_per_token_quant_int8(
            x,
            baseline_q,
            baseline_scales,
            baseline_sum,
            rows_per_program=rows_per_program,
            use_baseline=True,
        )
        launch_per_token_quant_int8(
            x,
            actual_q,
            actual_scales,
            actual_sum,
            rows_per_program=rows_per_program,
        )
        torch.cuda.synchronize()
        expected_q, expected_scales = reference(x)
        expected_sum = x.float().sum(dim=1).to(x.dtype)
        max_q_diff = int(
            (actual_q.int() - expected_q.int()).abs().max().item()
        )
        scale_close = torch.allclose(
            actual_scales, expected_scales, rtol=1e-6, atol=1e-7
        )
        sum_close = torch.allclose(
            actual_sum, expected_sum, rtol=1e-2, atol=1e-2
        )
        baseline_match = (
            torch.equal(actual_q, baseline_q)
            and torch.equal(actual_scales, baseline_scales)
            and torch.equal(actual_sum, baseline_sum)
        )
        passed = (
            max_q_diff <= 1
            and scale_close
            and sum_close
            and torch.equal(actual_q[1, : values.numel()], expected)
        )
        cases.append(
            {
                "shape": [M, K],
                "passed": bool(passed),
                "baseline_match": bool(baseline_match),
                "max_q_diff": max_q_diff,
                "scale_allclose": bool(scale_close),
                "sum_allclose": bool(sum_close),
            }
        )
    return {
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (
        position - low
    )


@torch.inference_mode()
def benchmark_case(
    M, K, *, use_baseline, warmup=30, batches=11, launches=10
):
    x, x_q, scales = make_case(M, K, seed=20260911)
    for _ in range(warmup):
        launch_per_token_quant_int8(
            x, x_q, scales, use_baseline=use_baseline
        )
    torch.cuda.synchronize()

    samples = []
    for _ in range(batches):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(launches):
            launch_per_token_quant_int8(
                x, x_q, scales, use_baseline=use_baseline
            )
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / launches)

    median_ms = statistics.median(samples)
    bytes_per_call = M * K * (x.element_size() + x_q.element_size())
    bytes_per_call += M * scales.element_size()
    return {
        "rows_per_program": get_per_token_quant_rows_per_program(M, K),
        "samples_ms": samples,
        "median_ms": median_ms,
        "p10_ms": percentile(samples, 0.1),
        "p90_ms": percentile(samples, 0.9),
        "bytes": bytes_per_call,
        "bandwidth_gbps": bytes_per_call / (median_ms * 1e6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--json-out", type=pathlib.Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/MACA device is required")

    rounding_boundaries = check_rounding_boundaries()
    record = {
        "rounding_boundaries": rounding_boundaries,
        "edge_cases": [],
        "sum_cases": [],
        "cases": [],
        "all_passed": rounding_boundaries["passed"],
    }
    print(json.dumps({"rounding_boundaries": rounding_boundaries}))
    for M, K, rows_per_program in EDGE_CASES:
        correctness = check_case(M, K, rows_per_program)
        edge = {
            "shape": [M, K],
            "rows_per_program": rows_per_program,
            "correctness": correctness,
        }
        record["edge_cases"].append(edge)
        record["all_passed"] &= correctness["passed"]
        print(json.dumps(edge))

    for M, K in CASES:
        correctness = check_case(M, K)
        performance = None
        if not args.correctness_only:
            baseline = benchmark_case(M, K, use_baseline=True)
            optimized = benchmark_case(M, K, use_baseline=False)
            performance = {
                "baseline": baseline,
                "optimized": optimized,
                "speedup": baseline["median_ms"] / optimized["median_ms"],
            }
        record["cases"].append(
            {
                "shape": [M, K],
                "correctness": correctness,
                "performance": performance,
            }
        )
        record["all_passed"] &= correctness["passed"]
        print(json.dumps(record["cases"][-1]))

    for M, K in SUM_CASES:
        correctness = check_case(M, K, cal_sum=True)
        item = {
            "shape": [M, K],
            "cal_sum": True,
            "correctness": correctness,
        }
        record["sum_cases"].append(item)
        record["all_passed"] &= correctness["passed"]
        print(json.dumps(item))

    if args.json_out:
        args.json_out.write_text(json.dumps(record, indent=2) + "\n")
    return 0 if record["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
