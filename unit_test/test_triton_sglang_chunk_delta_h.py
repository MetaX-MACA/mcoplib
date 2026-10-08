# SPDX-License-Identifier: Apache-2.0
"""Correctness and reproducible baseline for the migrated chunk-delta kernel.

Default run covers the supplied ragged production case and the five main
H=8/N=1 production shapes:

    k/w/u/gk: [1, 8192, 12, 128]
    cu_seqlens: [0, 824, 3896, 6971, 8192]
    T: 8192, 34, 3072, 1, 7168; H=Hg=8; initial_state=float32

``--full`` additionally covers packed logical batch sizes 1..256, prefill
sizes 2K..16K, and H=12/24 (the local-head shapes assumed for TP8/TP4).
The physical packed batch dimension stays B=1.

Run on an idle device:

    source env_local.sh
    CUDA_VISIBLE_DEVICES=<physical-id> /opt/conda/bin/python \
        unit_test/test_triton_sglang_chunk_delta_h.py --json-out /tmp/baseline.json
"""

import argparse
import json
import math
import os
import pathlib
import statistics
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mcoplib.triton_sglang_chunk_delta_h import (  # noqa: E402
    CHUNK_SIZE,
    _is_metax_c500,
    _launch_chunk_gated_delta_rule_fwd_h,
    chunk_gated_delta_rule_fwd_h,
    chunk_gated_delta_rule_fwd_h_baseline,
    prepare_chunk_offsets,
)


K = 128
V = 128
HG_RATIO = 1
DTYPE = torch.bfloat16
COSINE_THRESHOLD = 0.9999
WARMUP = 10
SAMPLE_BATCHES = 11
LAUNCHES_PER_SAMPLE = 10
EXACT_CU_SEQLENS = (0, 824, 3896, 6971, 8192)


@dataclass
class Case:
    total_tokens: int
    logical_batch: int
    heads: int
    cu_values: Tuple[int, ...] = ()
    state_dtype: torch.dtype = DTYPE

    @property
    def label(self) -> str:
        return (
            f"T={self.total_tokens},N={self.logical_batch},H={self.heads},"
            f"state={self.state_dtype}"
        )


MAIN_H8_CASES = tuple(
    Case(T, 1, 8, (0, T), torch.float32)
    for T in (8192, 34, 3072, 1, 7168)
)


def _balanced_cu_seqlens(total_tokens: int, logical_batch: int) -> Tuple[int, ...]:
    if logical_batch > total_tokens:
        raise ValueError("logical_batch cannot exceed total_tokens")
    base, extra = divmod(total_tokens, logical_batch)
    lengths = [base + (i < extra) for i in range(logical_batch)]
    # Deterministically perturb adjacent lengths to exercise partial BT tails.
    for i in range(0, logical_batch - 1, 2):
        shift = min((i * 13 + 7) % CHUNK_SIZE, lengths[i + 1] - 1)
        lengths[i] += shift
        lengths[i + 1] -= shift
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return tuple(values)


def _cu_values(case: Case) -> Tuple[int, ...]:
    if case.cu_values:
        return case.cu_values
    return _balanced_cu_seqlens(case.total_tokens, case.logical_batch)


def _chunk_local_gk(
    total_tokens: int, heads: int, cu_values: Sequence[int]
) -> torch.Tensor:
    # Production gk is a chunk-local cumulative sum of non-positive gates.
    increments = -0.0125 * torch.rand(
        total_tokens, heads, K, device="cuda", dtype=torch.float32
    )
    gk = torch.empty_like(increments)
    for bos, eos in zip(cu_values[:-1], cu_values[1:]):
        for start in range(bos, eos, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, eos)
            gk[start:end] = increments[start:end].cumsum(0)
    return gk.unsqueeze(0)


def _make_inputs(case: Case, seed: int = 20260908) -> Dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    T, H = case.total_tokens, case.heads
    Hg = H // HG_RATIO
    cu_values = _cu_values(case)
    scale = 1.0 / math.sqrt(K)
    k = torch.randn(1, T, Hg, K, device="cuda", dtype=DTYPE) * scale
    w = torch.randn(1, T, H, K, device="cuda", dtype=DTYPE) * scale
    u = torch.randn(1, T, H, V, device="cuda", dtype=DTYPE)
    gk = _chunk_local_gk(T, H, cu_values)
    cu_seqlens = torch.tensor(cu_values, device="cuda", dtype=torch.int32)
    initial_state_indices = torch.arange(
        case.logical_batch, device="cuda", dtype=torch.int32
    )
    initial_state = (
        torch.randn(
            case.logical_batch,
            H,
            V,
            K,
            device="cuda",
            dtype=case.state_dtype,
        )
        * 0.01
    )
    return {
        "k": k,
        "w": w,
        "u": u,
        "gk": gk,
        "cu_seqlens": cu_seqlens,
        "initial_state_indices": initial_state_indices,
        "initial_state": initial_state,
    }


@torch.no_grad()
def _reference(
    inputs: Dict[str, torch.Tensor], case: Case
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k, w, u, gk = (inputs[name] for name in ("k", "w", "u", "gk"))
    initial_state = inputs["initial_state"].clone()
    cu_values = _cu_values(case)
    offsets = [0]
    for bos, eos in zip(cu_values[:-1], cu_values[1:]):
        offsets.append(offsets[-1] + math.ceil((eos - bos) / CHUNK_SIZE))
    total_chunks = offsets[-1]
    h = torch.empty(
        1, total_chunks, case.heads, V, K, device="cuda", dtype=DTYPE
    )
    v_new = torch.empty_like(u)
    head_map = torch.arange(case.heads, device="cuda") // HG_RATIO

    for seq, (bos, eos) in enumerate(zip(cu_values[:-1], cu_values[1:])):
        state = initial_state[seq].float()
        for local_chunk, start in enumerate(range(bos, eos, CHUNK_SIZE)):
            end = min(start + CHUNK_SIZE, eos)
            global_chunk = offsets[seq] + local_chunk
            h[0, global_chunk] = state.to(DTYPE)

            wc = w[0, start:end]
            # Triton casts the fp32 recurrent state to w.dtype before the dot.
            projected = torch.einsum(
                "thk,hvk->thv", wc.float(), state.to(DTYPE).float()
            )
            residual = u[0, start:end].float() - projected
            v_new[0, start:end] = residual.to(DTYPE)

            state *= torch.exp(gk[0, end - 1].float())[:, None, :]
            kc = k[0, start:end, head_map]
            residual_q = residual.to(DTYPE)
            state += torch.einsum(
                "thk,thv->hvk", kc.float(), residual_q.float()
            )
        initial_state[seq] = state.to(initial_state.dtype)
    return h, v_new, initial_state


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a, b = actual.double().flatten(), expected.double().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-30).item()


@torch.no_grad()
def _correctness(case: Case, variant: str = "baseline") -> Dict[str, object]:
    inputs = _make_inputs(case)
    expected_h, expected_v, expected_state = _reference(inputs, case)
    state = inputs["initial_state"].clone()
    op = (
        chunk_gated_delta_rule_fwd_h
        if variant == "optimized"
        else chunk_gated_delta_rule_fwd_h_baseline
    )
    actual_h, actual_v = op(
        k=inputs["k"],
        w=inputs["w"],
        u=inputs["u"],
        gk=inputs["gk"],
        initial_state=state,
        initial_state_indices=inputs["initial_state_indices"],
        cu_seqlens=inputs["cu_seqlens"],
    )
    torch.cuda.synchronize()
    metrics = {
        "h_cosine": _cosine(actual_h, expected_h),
        "v_new_cosine": _cosine(actual_v, expected_v),
        "state_cosine": _cosine(state, expected_state),
        "h_max_abs": (actual_h.float() - expected_h.float()).abs().max().item(),
        "v_new_max_abs": (actual_v.float() - expected_v.float()).abs().max().item(),
        "state_max_abs": (state.float() - expected_state.float()).abs().max().item(),
        "sample_values": actual_v.flatten()[:8].float().tolist(),
    }
    metrics["passed"] = all(
        metrics[name] >= COSINE_THRESHOLD
        for name in ("h_cosine", "v_new_cosine", "state_cosine")
    ) and all(
        torch.isfinite(tensor).all().item() for tensor in (actual_h, actual_v, state)
    )
    del inputs, expected_h, expected_v, expected_state, actual_h, actual_v, state
    torch.cuda.empty_cache()
    return metrics


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


@torch.no_grad()
def _benchmark(
    case: Case,
    warmup: int = WARMUP,
    sample_batches: int = SAMPLE_BATCHES,
    launches_per_sample: int = LAUNCHES_PER_SAMPLE,
    variant: str = "baseline",
) -> Dict[str, object]:
    inputs = _make_inputs(case, seed=20260909)
    cu = inputs["cu_seqlens"]
    offsets = prepare_chunk_offsets(cu, CHUNK_SIZE)
    total_chunks = int(offsets[-1].item())
    h = torch.empty(
        1, total_chunks, case.heads, V, K, device="cuda", dtype=DTYPE
    )
    v_new = torch.empty_like(inputs["u"])

    # Give every launch a pristine state slot. Restore all slots outside each
    # timed sample so state mutation never pollutes either data or latency.
    state_slots = torch.empty(
        launches_per_sample,
        *inputs["initial_state"].shape,
        device="cuda",
        dtype=inputs["initial_state"].dtype,
    )
    state_slots.copy_(inputs["initial_state"].unsqueeze(0))
    index_sets = [
        inputs["initial_state_indices"] + i * case.logical_batch
        for i in range(launches_per_sample)
    ]
    block_v, num_warps = (64, 4) if variant == "optimized" else (32, 4)

    def launch(i: int) -> None:
        _launch_chunk_gated_delta_rule_fwd_h(
            k=inputs["k"],
            w=inputs["w"],
            u=inputs["u"],
            h=h,
            v_new=v_new,
            gk=inputs["gk"],
            initial_state=state_slots.view(-1, case.heads, V, K),
            initial_state_indices=index_sets[i],
            cu_seqlens=cu,
            chunk_offsets=offsets,
            logical_batch_size=case.logical_batch,
            total_chunks=total_chunks,
            block_v=block_v,
            num_warps=num_warps,
            num_stages=1,
        )

    for i in range(warmup):
        launch(i % launches_per_sample)
    torch.cuda.synchronize()

    samples_ms: List[float] = []
    for _ in range(sample_batches):
        state_slots.copy_(inputs["initial_state"].unsqueeze(0))
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(launches_per_sample):
            launch(i)
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end) / launches_per_sample)

    # Two KxV GEMMs per token/head: W@H^T and K^T@V.
    flops = 4 * case.total_tokens * case.heads * K * V
    median_ms = statistics.median(samples_ms)

    # Tensor-byte model for reproducible bandwidth comparisons. CTA-issued
    # bytes count the repeated K/W/GK reads across V tiles; unique bytes count
    # each accessed tensor element once. Hardware HBM transactions may differ
    # because this model intentionally excludes cache-line and cache-hit effects.
    # On C500, the optimized packed H=12 path requests BV=64 but dispatches the
    # fused kernel with BV=32; use the actual tile size for the byte model.
    device_index = inputs["k"].device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    kernel_block_v = (
        32
        if _is_metax_c500(device_index)
        and variant == "optimized"
        and case.total_tokens == 8192
        and case.logical_batch == 4
        and case.heads == 12
        else block_v
    )
    v_tiles = math.ceil(V / kernel_block_v)
    k_head_reuse = case.heads // inputs["k"].shape[2]
    gk_endpoint_bytes = (
        total_chunks * case.heads * K * inputs["gk"].element_size()
    )
    unique_read_bytes = (
        inputs["k"].numel() * inputs["k"].element_size()
        + inputs["w"].numel() * inputs["w"].element_size()
        + inputs["u"].numel() * inputs["u"].element_size()
        + gk_endpoint_bytes
        + inputs["initial_state"].numel()
        * inputs["initial_state"].element_size()
    )
    issued_read_bytes = (
        inputs["k"].numel()
        * inputs["k"].element_size()
        * k_head_reuse
        * v_tiles
        + inputs["w"].numel() * inputs["w"].element_size() * v_tiles
        + inputs["u"].numel() * inputs["u"].element_size()
        + gk_endpoint_bytes * v_tiles
        + inputs["initial_state"].numel()
        * inputs["initial_state"].element_size()
    )
    write_bytes = (
        h.numel() * h.element_size()
        + v_new.numel() * v_new.element_size()
        + inputs["initial_state"].numel()
        * inputs["initial_state"].element_size()
    )
    unique_global_bytes = unique_read_bytes + write_bytes
    issued_global_bytes = issued_read_bytes + write_bytes

    result = {
        "samples_ms": samples_ms,
        "median_ms": median_ms,
        "p10_ms": _percentile(samples_ms, 0.10),
        "p90_ms": _percentile(samples_ms, 0.90),
        "tflops": flops / (median_ms * 1.0e9),
        "unique_global_bytes": unique_global_bytes,
        "issued_global_read_bytes": issued_read_bytes,
        "issued_global_write_bytes": write_bytes,
        "issued_global_bytes": issued_global_bytes,
        "unique_access_bandwidth_gbps": unique_global_bytes
        / (median_ms * 1.0e6),
        "issued_tensor_bandwidth_gbps": issued_global_bytes
        / (median_ms * 1.0e6),
        "bandwidth_model": (
            "tensor bytes; CTA-issued counts V-tile duplication and excludes "
            "cache-line/cache-hit effects"
        ),
        "total_chunks": total_chunks,
        "grid": [v_tiles, case.logical_batch * case.heads],
    }
    del inputs, h, v_new, state_slots, index_sets, offsets
    torch.cuda.empty_cache()
    return result


def _cases(full: bool) -> Iterable[Case]:
    yield Case(8192, 4, 12, EXACT_CU_SEQLENS)
    yield from MAIN_H8_CASES
    if not full:
        return
    for heads in (12, 24):
        for total_tokens in (2048, 4096, 8192, 16384):
            for logical_batch in (1, 4, 16, 64, 128, 256):
                yield Case(total_tokens, logical_batch, heads)


def _environment() -> Dict[str, object]:
    return {
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "logical_device": torch.cuda.current_device(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "dtype": str(DTYPE),
        "K": K,
        "V": V,
        "chunk_size": CHUNK_SIZE,
        "warmup": WARMUP,
        "sample_batches": SAMPLE_BATCHES,
        "launches_per_sample": LAUNCHES_PER_SAMPLE,
        "cosine_threshold": COSINE_THRESHOLD,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument(
        "--variant", choices=("baseline", "optimized"), default="baseline"
    )
    parser.add_argument("--json-out", type=pathlib.Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA/MACA device is required", file=sys.stderr)
        return 2

    record = {
        "environment": _environment(),
        "variant": args.variant,
        "cases": [],
    }
    print(json.dumps(record["environment"], ensure_ascii=False, indent=2))
    all_passed = True
    for case in _cases(args.full):
        print(f"\n[{case.label}] correctness")
        correctness = _correctness(case, variant=args.variant)
        print(json.dumps(correctness, ensure_ascii=False, indent=2))
        all_passed &= bool(correctness["passed"])
        performance = None
        if not args.correctness_only:
            performance = _benchmark(case, variant=args.variant)
            print(f"{args.variant}:", json.dumps(performance, ensure_ascii=False))
        record["cases"].append(
            {
                "shape": {
                    "physical_batch": 1,
                    "total_tokens": case.total_tokens,
                    "logical_batch": case.logical_batch,
                    "heads": case.heads,
                    "initial_state_dtype": str(case.state_dtype),
                    "cu_seqlens": list(_cu_values(case)),
                },
                "correctness": correctness,
                "performance": performance,
            }
        )

    record["all_passed"] = all_passed
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{args.variant} JSON: {args.json_out}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
