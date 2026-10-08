"""MHC pre big-fuse TileLang unit test, final configuration only.

Run accuracy and performance tests::

    pytest -q -s test_tilelang_mhc_pre.py

Run a small final benchmark::

    python unit_test/test_tilelang_mhc_pre.py
"""

import math
import os
import statistics
from functools import cache

import pytest
import torch

pytest.importorskip("tilelang")


from mcoplib import tilelang_mhc_pre


HC = 4
HIDDEN = 7168
RMS_EPS = 1e-6
HC_PRE_EPS = 1e-6
HC_SINKHORN_EPS = 1e-6
HC_POST_MULT_VALUE = 2.0
SINKHORN_REPEAT = 20


@cache
def compute_num_split2(num_tokens: int) -> int:
    return 16 if num_tokens >= 512 else 64


def torch_reference(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    residual: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, mhc_mult, hidden_size)

    rms = torch.rsqrt(
        gemm_out_sqrsum.sum(0) / (mhc_mult * hidden_size) + rms_eps
    )
    mixes = gemm_out_mul.sum(0) * rms.unsqueeze(-1)

    pre_mix = (
        torch.sigmoid(
            mixes[:, :mhc_mult] * mhc_scale[0] + mhc_base[:mhc_mult]
        )
        + mhc_pre_eps
    )
    post_mix = (
        torch.sigmoid(
            mixes[:, mhc_mult : 2 * mhc_mult] * mhc_scale[1]
            + mhc_base[mhc_mult : 2 * mhc_mult]
        )
        * mhc_post_mult_value
    )

    comb_mix = (
        mixes[:, 2 * mhc_mult :] * mhc_scale[2]
        + mhc_base[2 * mhc_mult :]
    ).view(-1, mhc_mult, mhc_mult)
    comb_mix = torch.softmax(comb_mix, dim=-1) + mhc_sinkhorn_eps
    comb_mix = comb_mix / (
        comb_mix.sum(dim=-2, keepdim=True) + mhc_sinkhorn_eps
    )
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (
            comb_mix.sum(dim=-1, keepdim=True) + mhc_sinkhorn_eps
        )
        comb_mix = comb_mix / (
            comb_mix.sum(dim=-2, keepdim=True) + mhc_sinkhorn_eps
        )

    layer_input = (
        residual_flat.float() * pre_mix.unsqueeze(-1)
    ).sum(dim=1).bfloat16()

    post_mix = post_mix.view(*outer_shape, mhc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, mhc_mult, mhc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)
    return post_mix, comb_mix, layer_input


def make_inputs(num_tokens: int, n_splits: int):
    residual = torch.randn(
        (num_tokens, HC, HIDDEN), device="cuda", dtype=torch.bfloat16
    )
    fn = torch.randn(
        (HC * (2 + HC), HC * HIDDEN), device="cuda", dtype=torch.float32
    ) / math.sqrt(HC * HIDDEN)
    hc_scale = torch.randn((3,), device="cuda", dtype=torch.float32) * 0.2 + 1.0
    hc_base = torch.randn(
        (HC * (2 + HC),), device="cuda", dtype=torch.float32
    ) * 0.2

    k_chunk = (HC * HIDDEN) // n_splits
    residual_2d = residual.view(num_tokens, HC * HIDDEN).float()
    gemm_out_mul = torch.empty(
        n_splits, num_tokens, HC * (2 + HC), device="cuda", dtype=torch.float32
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, device="cuda", dtype=torch.float32
    )
    for s in range(n_splits):
        a = residual_2d[:, s * k_chunk : (s + 1) * k_chunk]
        b = fn[:, s * k_chunk : (s + 1) * k_chunk]
        gemm_out_mul[s] = a @ b.t()
        gemm_out_sqrsum[s] = a.square().sum(-1)

    return residual, fn, hc_scale, hc_base, gemm_out_mul, gemm_out_sqrsum


def benchmark_ms(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def effective_bandwidth_gbps(
    tensors: tuple[torch.Tensor, ...], latency_ms: float
) -> float:
    total_bytes = sum(t.numel() * t.element_size() for t in tensors)
    return total_bytes / latency_ms / 1e6


DECODE_NUM_TOKENS = (
    4, 8, 12, 16, 20, 24, 28, 32,
    40, 48, 56, 64, 72, 80, 88, 96,
    104, 112, 120, 128,
)

PREFILL_CASES = (
    (1, 112),
    (6, 16),
    (2048, 16),
    (3072, 16),
    (4096, 736),
)

CASES = tuple(("decode", n, None) for n in DECODE_NUM_TOKENS) + tuple(
    ("prefill", n, count) for n, count in PREFILL_CASES
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
@pytest.mark.parametrize(
    "phase,num_tokens,workload_count",
    CASES,
    ids=[f"{phase}-n{n}" for phase, n, _ in CASES],
)
def test_mhc_pre_big_fuse_accuracy_and_performance(
    phase: str, num_tokens: int, workload_count: int | None
):
    n_splits = compute_num_split2(num_tokens)
    torch.manual_seed(20260817 + num_tokens)
    residual, fn, hc_scale, hc_base, gemm_out_mul, gemm_out_sqrsum = make_inputs(
        num_tokens, n_splits
    )

    def run():
        return tilelang_mhc_pre.mhc_pre(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual,
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            HC_POST_MULT_VALUE,
            SINKHORN_REPEAT,
            n_splits=n_splits,
        )

    actual = run()
    expected = torch_reference(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT_VALUE,
        SINKHORN_REPEAT,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual[0], expected[0], rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(actual[1], expected[1], rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(
        actual[2].float(), expected[2].float(), rtol=2e-2, atol=2e-2
    )

    max_abs_error = max(
        (actual[0] - expected[0]).abs().max().item(),
        (actual[1] - expected[1]).abs().max().item(),
        (actual[2].float() - expected[2].float()).abs().max().item(),
    )

    warmup = int(os.getenv("MHC_PRE_WARMUP", "20"))
    default_repeat = "200" if workload_count is None else str(workload_count)
    repeat = int(os.getenv(f"MHC_PRE_{phase.upper()}_REPEAT", default_repeat))
    rounds = int(os.getenv("MHC_PRE_BENCH_ROUNDS", "3"))

    times = [benchmark_ms(run, warmup=0, repeat=repeat) for _ in range(rounds)]
    latency_ms = statistics.median(times)
    tensors = (gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual, *actual)
    bandwidth = effective_bandwidth_gbps(tensors, latency_ms)

    phase_name = "decode" if phase == "decode" else "prefill"
    print(
        f"{phase_name} | token={num_tokens} | "
        f"误差={max_abs_error:.6g} | "
        f"latency={latency_ms:.4f} ms | "
        f"bandwidth={bandwidth:.1f} GB/s | "
        f"n_splits={n_splits} | "
        f"workload_count={workload_count or '-'}"
    )

    del residual, fn, hc_scale, hc_base, gemm_out_mul, gemm_out_sqrsum, actual, expected
    torch.cuda.empty_cache()



def final_benchmark_cli() -> None:
    warmup = int(os.getenv("MHC_PRE_WARMUP", "20"))
    rounds = int(os.getenv("MHC_PRE_BENCH_ROUNDS", "3"))

    for phase, num_tokens, workload_count in CASES:
        n_splits = compute_num_split2(num_tokens)
        torch.manual_seed(20260817 + num_tokens)
        residual, fn, hc_scale, hc_base, gemm_out_mul, gemm_out_sqrsum = make_inputs(
            num_tokens, n_splits
        )

        def run():
            return tilelang_mhc_pre.mhc_pre(
                gemm_out_mul,
                gemm_out_sqrsum,
                hc_scale,
                hc_base,
                residual,
                RMS_EPS,
                HC_PRE_EPS,
                HC_SINKHORN_EPS,
                HC_POST_MULT_VALUE,
                SINKHORN_REPEAT,
                n_splits=n_splits,
            )

        actual = run()
        torch.cuda.synchronize()
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()

        default_repeat = "200" if workload_count is None else str(workload_count)
        repeat = int(
            os.getenv(f"MHC_PRE_{phase.upper()}_REPEAT", default_repeat)
        )
        times = [
            benchmark_ms(run, warmup=0, repeat=repeat)
            for _ in range(rounds)
        ]
        latency_ms = statistics.median(times)
        tensors = (gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual, *actual)
        bandwidth = effective_bandwidth_gbps(tensors, latency_ms)

        print(
            f"final-{phase} | token={num_tokens} | "
            f"n_splits={n_splits} | "
            f"workload_count={workload_count or '-'} | "
            f"latency={latency_ms:.4f} ms | "
            f"bandwidth={bandwidth:.1f} GB/s"
        )

        del residual, fn, hc_scale, hc_base, gemm_out_mul, gemm_out_sqrsum, actual
        torch.cuda.empty_cache()


if __name__ == "__main__":
    final_benchmark_cli()
