#!/usr/bin/env python3
"""Correctness and bandwidth benchmark for vLLM silu_and_mul on MetaX C600-U.

For every output element:
    out[t, i] = silu(gate[t, i]) * up[t, i]

The input layout is [num_tokens, 2 * hidden], with gate and up stored in the
first and second halves. Effective HBM traffic counts two input reads and one
output write. Timings use one back-to-back CUDA-event interval around all
iterations so launch synchronization does not suppress C600-U DVFS.
"""

import argparse
import os
from typing import Iterable

import torch

import mcoplib._C  # noqa: F401  Registers torch.ops._C.silu_and_mul.


SHAPES = [
    (128, 1024),
    (1024, 2048),
    (4096, 4096),
    (32, 1536),
    (1, 1536),
    (128, 1536),
    (3, 1537),
    (64, 257),
    (1, 4096),
    (8, 4096),
    (128, 4096),
    (1, 6144),
    (16, 6144),
    (128, 8192),
    (1, 11008),
    (32, 11008),
    (1024, 11008),
    (1, 14336),
    (64, 14336),
    (4096, 14336),
    (12288, 384),
    (12288, 3584),
    (1024, 384),
    (1024, 3584),
    (2, 384),
    (2, 3584),
]

DTYPES = {
    "bf16": (torch.bfloat16, 0.9999),
    "fp16": (torch.float16, 0.9999),
    "fp32": (torch.float32, 0.9999),
}

SINGLE_DIE_GBPS = 1600.0
TARGET_85_GBPS = 0.85 * SINGLE_DIE_GBPS
DUAL_DIE_GBPS = 3200.0


def silu_and_mul_ref(x: torch.Tensor) -> torch.Tensor:
    hidden = x.shape[-1] // 2
    xf = x.float()
    return torch.nn.functional.silu(xf[..., :hidden]) * xf[..., hidden:]


def run_op(out: torch.Tensor, x: torch.Tensor) -> None:
    torch.ops._C.silu_and_mul(out, x)


def cosine_similarity(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_f = actual.float().flatten()
    expected_f = expected.float().flatten()
    denominator = (actual_f.norm() * expected_f.norm()).clamp_min(1e-20)
    return (actual_f.dot(expected_f) / denominator).item()


def representative_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[int, float]:
    flat_actual = actual.float().flatten()
    flat_expected = expected.float().flatten()
    count = min(8, flat_actual.numel())
    if count == 1:
        indices = [0]
    else:
        last = flat_actual.numel() - 1
        indices = [(i * last) // (count - 1) for i in range(count)]
    max_abs_error = max(
        abs(flat_actual[index].item() - flat_expected[index].item())
        for index in indices
    )
    return count, max_abs_error


def benchmark_case(
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype,
    threshold: float,
    warmup: int,
    iterations: int,
) -> tuple[bool, float, float]:
    x = torch.randn(num_tokens, 2 * hidden, dtype=dtype, device="cuda")
    out = torch.empty(num_tokens, hidden, dtype=dtype, device="cuda")

    run_op(out, x)
    torch.cuda.synchronize()
    expected = silu_and_mul_ref(x)
    similarity = cosine_similarity(out, expected)
    sample_count, sample_max_abs = representative_error(out, expected)
    finite = bool(torch.isfinite(out).all().item())

    for _ in range(warmup):
        run_op(out, x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        run_op(out, x)
    end.record()
    end.synchronize()
    average_ms = start.elapsed_time(end) / iterations

    effective_bytes = num_tokens * hidden * 3 * x.element_size()
    bandwidth_gbps = effective_bytes / (average_ms * 1e-3) / 1e9
    passed = finite and similarity >= threshold
    status = "OK" if passed else "FAIL"
    print(
        f"[silu_and_mul] T={num_tokens:5d} d={hidden:6d}  "
        f"cos_sim={similarity:.6f} {status:4s}  {average_ms:9.5f} ms  "
        f"{bandwidth_gbps:8.1f} GB/s  samples={sample_count} "
        f"sample_max_abs={sample_max_abs:.3e} finite={finite}"
    )
    return passed, average_ms, bandwidth_gbps


def selected_shapes(shape: str) -> Iterable[tuple[int, int]]:
    if shape == "all":
        return SHAPES
    try:
        tokens_text, hidden_text = shape.split(",", maxsplit=1)
        selected = (int(tokens_text), int(hidden_text))
    except (TypeError, ValueError) as error:
        raise ValueError("--shape must be 'all' or 'tokens,hidden'") from error
    if selected not in SHAPES:
        raise ValueError(f"shape {selected} is not in the required matrix")
    return [selected]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["all", *DTYPES], default="all")
    parser.add_argument("--shape", default="all")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be >= 0 and iterations must be > 0")

    torch.manual_seed(42)
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.cuda.manual_seed_all(42)

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    properties = torch.cuda.get_device_properties(0)
    print(
        f"CUDA_VISIBLE_DEVICES={visible_devices!r}  "
        f"device_count={torch.cuda.device_count()}"
    )
    print(
        f"device: {properties.name}  SMs={properties.multi_processor_count}  "
        f"cc={properties.major}.{properties.minor}  "
        f"L2={properties.L2_cache_size // (1024 * 1024)}MB"
    )
    print(f"timing: warmup={args.warmup}, iterations={args.iterations}, metric=average")
    print("=" * 120)

    dtype_names = DTYPES.keys() if args.dtype == "all" else [args.dtype]
    shapes = list(selected_shapes(args.shape))
    overall_pass = True
    summaries: dict[str, tuple[float, bool]] = {}

    for dtype_name in dtype_names:
        dtype, threshold = DTYPES[dtype_name]
        print(f"---- dtype={dtype_name}  threshold={threshold:.4f} " + "-" * 72)
        peak = 0.0
        dtype_pass = True
        for num_tokens, hidden in shapes:
            passed, _, bandwidth = benchmark_case(
                num_tokens,
                hidden,
                dtype,
                threshold,
                args.warmup,
                args.iterations,
            )
            peak = max(peak, bandwidth)
            dtype_pass &= passed
        summaries[dtype_name] = (peak, dtype_pass)
        overall_pass &= dtype_pass
        print(
            f"[{dtype_name}] peak={peak:.1f} GB/s  "
            f"accuracy={'ALL PASS' if dtype_pass else 'FAILURES PRESENT'}"
        )

    print("=" * 120)
    for dtype_name, (peak, dtype_pass) in summaries.items():
        if peak >= SINGLE_DIE_GBPS:
            target_status = "REACHED-1.6T"
        elif peak >= TARGET_85_GBPS:
            target_status = "REACHED-85%"
        else:
            target_status = "NOT-REACHED"
        print(
            f"{dtype_name:>4}: peak={peak:8.1f} GB/s  {target_status}  "
            f"single_die={SINGLE_DIE_GBPS:.0f}  target85={TARGET_85_GBPS:.0f}  "
            f"accuracy={'PASS' if dtype_pass else 'FAIL'}"
        )
    print(f"dual-die datasheet reference: {DUAL_DIE_GBPS:.0f} GB/s")
    if not overall_pass:
        raise AssertionError("accuracy, finite-output, or kernel execution check failed")
    print("Accuracy: ALL PASS (cosine similarity >= 0.9999); no kernel trap observed")


if __name__ == "__main__":
    main()
