#!/usr/bin/env python3
"""Correctness and performance test for the compiled mcoplib KDA intra-chunk kernel."""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import statistics
import sys
from dataclasses import dataclass
from typing import Callable

import torch

# Ensure the repository containing this test is preferred over an older
# mcoplib installation in site-packages, especially when run from unit_test/.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class TestCase:
    name: str
    tokens: int
    heads: int
    cu_seqlens: tuple[int, ...] | None


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.split(",") if x)


def _parse_case(value: str) -> TestCase:
    fields = value.split(":")
    if len(fields) not in (3, 4):
        raise argparse.ArgumentTypeError(
            "case must be name:T:H or name:T:H:cu0,cu1,..."
        )
    name = fields[0]
    tokens = int(fields[1])
    heads = int(fields[2])
    cu = _parse_ints(fields[3]) if len(fields) == 4 else (0, tokens)
    if not cu or cu[0] != 0 or cu[-1] != tokens or any(b <= a for a, b in zip(cu, cu[1:])):
        raise argparse.ArgumentTypeError(
            "cu_seqlens must start at 0, end at T, and be strictly increasing"
        )
    return TestCase(name, tokens, heads, cu)


def _make_inputs(case: TestCase, batch: int, k: int, seed: int, bt: int):
    torch.manual_seed(seed)
    device = "cuda"
    q = torch.randn((batch, case.tokens, case.heads, k), device=device, dtype=torch.bfloat16)
    k_tensor = torch.randn_like(q)
    step = -torch.rand((case.tokens, case.heads, k), device=device, dtype=torch.float32) * 0.05
    g = torch.empty_like(step)
    bounds = case.cu_seqlens or (0, case.tokens)
    for bos, eos in zip(bounds[:-1], bounds[1:]):
        for start in range(bos, eos, bt):
            end = min(start + bt, eos)
            g[start:end] = step[start:end].cumsum(0)
    g = g.reshape(batch, case.tokens, case.heads, k)
    beta = torch.rand((batch, case.tokens, case.heads), device=device, dtype=torch.float32)
    cu = None
    if case.cu_seqlens is not None:
        cu = torch.tensor(case.cu_seqlens, device=device, dtype=torch.int32)
    return q, k_tensor, g, beta, cu


def _make_outputs(batch: int, case: TestCase, bt: int, bc: int):
    aqk = torch.full((batch, case.tokens, case.heads, bt), float("nan"), device="cuda", dtype=torch.bfloat16)
    akk = torch.full((batch, case.tokens, case.heads, bc), float("nan"), device="cuda", dtype=torch.float32)
    return aqk, akk


def _bench(fn: Callable[[], object], warmup: int, repeat: int, rounds: int) -> list[float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(rounds):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeat)
    return samples


def _summary(samples: list[float]) -> tuple[float, float]:
    return statistics.median(samples), min(samples)


def _load_baseline(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("kda_sglang_baseline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load baseline module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.chunk_kda_fwd_intra_token_parallel


def _reference(q, k, g, beta, scale, bt, bc):
    if q.size(0) != 1:
        raise ValueError("the optimized kernel currently requires batch=1")
    qf, kf, gf, bf = q.cpu().double(), k.cpu().double(), g.cpu().double(), beta.cpu().double()
    _, tokens, heads, _ = q.shape
    aqk_ref = torch.full((1, tokens, heads, bt), float("nan"), device="cpu", dtype=torch.float64)
    akk_ref = torch.full((1, tokens, heads, bc), float("nan"), device="cpu", dtype=torch.float64)

    for start in range(0, tokens, bc):
        count = min(bc, tokens - start)
        end = start + count
        q_tile = qf[0, start:end]
        k_tile = kf[0, start:end]
        g_tile = gf[0, start:end]
        beta_tile = bf[0, start:end]
        gate_diff = g_tile - g_tile[0:1]
        qg = q_tile * torch.exp2(gate_diff)
        kg = k_tile * torch.exp2(-gate_diff)
        kbg = k_tile * beta_tile.unsqueeze(-1) * torch.exp2(gate_diff)
        aqk_tile = scale * torch.einsum("ihk,jhk->ijh", qg, kg)
        akk_tile = torch.einsum("ihk,jhk->ijh", kbg, kg)
        lower = torch.tril(torch.ones((count, count), device="cpu", dtype=torch.bool))
        strict = torch.tril(torch.ones((count, count), device="cpu", dtype=torch.float64), -1)
        col0 = start % bt
        aqk_ref[0, start:end, :, col0:col0 + count] = torch.where(lower[:, None, :], aqk_tile.permute(0, 2, 1), torch.nan)
        akk_ref[0, start:end, :, :count] = torch.where(lower[:, None, :], (akk_tile * strict[:, :, None]).permute(0, 2, 1), torch.nan)
    return aqk_ref.to(q.device), akk_ref.to(q.device)


def _debug_candidate(q, k, g, beta, aqk, akk, scale, bt, bc):
    aqk_ref, akk_ref = _reference(q, k, g, beta, scale, bt, bc)
    valid_aqk = ~torch.isnan(aqk_ref)
    valid_akk = ~torch.isnan(akk_ref)
    print("write", int(valid_aqk.sum()), int((~torch.isnan(aqk)).sum()), int(valid_akk.sum()), int((~torch.isnan(akk)).sum()))
    print("aqk maxerr", (aqk[valid_aqk].double()-aqk_ref[valid_aqk]).abs().max().item())
    print("akk maxerr", (akk[valid_akk].double()-akk_ref[valid_akk]).abs().max().item())
    print("aqk close", torch.allclose(aqk[valid_aqk].double(), aqk_ref[valid_aqk], atol=0.002, rtol=0.01))
    print("akk close", torch.allclose(akk[valid_akk].double(), akk_ref[valid_akk], atol=0.002, rtol=0.01))

def _check_candidate(q, k, g, beta, aqk, akk, scale, bt, bc, atol, rtol):
    aqk_ref, akk_ref = _reference(q, k, g, beta, scale, bt, bc)
    valid_aqk = ~torch.isnan(aqk_ref)
    valid_akk = ~torch.isnan(akk_ref)
    if not torch.equal(valid_aqk, ~torch.isnan(aqk)) or not torch.equal(valid_akk, ~torch.isnan(akk)):
        return False
    return bool(
        torch.allclose(aqk[valid_aqk].double(), aqk_ref[valid_aqk], atol=atol, rtol=rtol)
        and torch.allclose(akk[valid_akk].double(), akk_ref[valid_akk], atol=atol, rtol=rtol)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Test correctness and benchmark compiled mcoplib KDA intra-chunk C++ kernel")
    parser.add_argument("--cases", type=_parse_case, nargs="+", default=[TestCase("production", 10240, 12, (0, 10240))])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--bt", type=int, default=64)
    parser.add_argument("--bc", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--baseline", type=pathlib.Path, help="optional original SGLang Python source")
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--rtol", type=float, default=0.01)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA/MACA device is visible")
    try:
        import mcoplib.sgl_kernel as sgl_kernel
        kernel = torch.ops.sgl_kernel.chunk_kda_fwd_intra_token_parallel
    except (ImportError, AttributeError, RuntimeError) as exc:
        raise SystemExit("mcoplib.sgl_kernel.chunk_kda_fwd_intra_token_parallel is unavailable; rebuild mcoplib first") from exc

    baseline = _load_baseline(args.baseline) if args.baseline else None
    scale = args.k**-0.5
    all_ok = True

    for case in args.cases:
        q, k, g, beta, cu = _make_inputs(case, args.batch, args.k, args.seed, args.bt)
        aqk, akk = _make_outputs(args.batch, case, args.bt, args.bc)
        candidate = lambda: kernel(q, k, g, beta, aqk, akk, scale)
        candidate()
        torch.cuda.synchronize()

        base_fn = None
        if baseline is not None:
            base_aqk, base_akk = _make_outputs(args.batch, case, args.bt, args.bc)
            base_fn = lambda: baseline(q, k, g, beta, base_aqk, base_akk, scale, cu, args.bt, args.bc)
            base_fn()
            torch.cuda.synchronize()
            valid = ~torch.isnan(base_aqk) & ~torch.isnan(aqk)
            okq = torch.allclose(base_aqk[valid], aqk[valid], atol=args.atol, rtol=args.rtol)
            valid = ~torch.isnan(base_akk) & ~torch.isnan(akk)
            okk = torch.allclose(base_akk[valid], akk[valid], atol=args.atol, rtol=args.rtol)
            write_set = bool(torch.equal(torch.isnan(base_aqk), torch.isnan(aqk)) and torch.equal(torch.isnan(base_akk), torch.isnan(akk)))
            precision = bool(okq and okk and write_set)
        else:
            precision = _check_candidate(q, k, g, beta, aqk, akk, scale, args.bt, args.bc, args.atol, args.rtol)
            _debug_candidate(q, k, g, beta, aqk, akk, scale, args.bt, args.bc)
        all_ok = all_ok and precision

        times = _bench(candidate, args.warmup, args.repeat, args.rounds)
        median_ms, min_ms = _summary(times)
        print(f"[{case.name}] B={args.batch} T={case.tokens} H={case.heads} K={args.k} BT={args.bt} BC={args.bc} cu={list(case.cu_seqlens or (0, case.tokens))}")

        if baseline is not None:
            base_times = _bench(base_fn, args.warmup, args.repeat, args.rounds)
            base_median, base_min = _summary(base_times)
            speedup = base_median / median_ms
            print(f"baseline_ms median={base_median:.6f} min={base_min:.6f} samples={[round(x, 6) for x in base_times]}")
            print(f"candidate_ms median={median_ms:.6f} min={min_ms:.6f} samples={[round(x, 6) for x in times]}")
            print("<time_before_opt>%.12f</time_before_opt>" % base_median)
            print("<time_after_opt>%.12f</time_after_opt>" % median_ms)
            print("<speedup>%.12f</speedup>" % speedup)
            print(f"baseline_precision={precision}")
            print(f"<precision>{precision}</precision>")
        else:
            print(f"candidate_ms median={median_ms:.6f} min={min_ms:.6f} samples={[round(x, 6) for x in times]}")
            print("<time_before_opt>null</time_before_opt>")
            print("<time_after_opt>%.12f</time_after_opt>" % median_ms)
            print("<speedup>null</speedup>")
            print(f"reference_precision={precision}")
            print(f"<precision>{precision}</precision>")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
