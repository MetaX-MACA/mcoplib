"""MHC pre big-fuse TileLang unit test: accuracy (cosine) + latency + bandwidth.

Coverage:
  * the original regression suite (no norm): decode / prefill / prefill-4096
  * the required grid: hidden {4096, 7168} x num_tokens {0,1,6,8,17,32,64}
    x use_norm {False, True}
  * a >=100-case norm sweep over the deepseek-v4 / v4.1 / v4-flash shape
    families (hidden 4096 / 7168, bf16 residual, RMSNorm fused)

Every case prints cosine similarity (fp64, threshold 0.99999 for bf16/fp16
grade data), latency and effective bandwidth, in the style of
unit_test/test_op_moe_scatter_dynamic_quant.py.

Run::

    pytest -q -s unit_test/test_tilelang_mhc_pre.py

or the full printed report::

    export CUDA_VISIBLE_DEVICES=<idle gpu>
    python unit_test/test_tilelang_mhc_pre.py
"""

import math
import os
import statistics

import pytest
import torch

pytest.importorskip("tilelang")


from mcoplib import tilelang_mhc_pre


HC = 4
HIDDEN = 7168
HIDDEN_4096 = 4096
RMS_EPS = 1e-6
HC_PRE_EPS = 1e-6
HC_SINKHORN_EPS = 1e-6
HC_POST_MULT_VALUE = 2.0
SINKHORN_REPEAT = 20
NORM_EPS = 1e-6

COS_THRESHOLD = 0.99999
TARGET_GBPS = 1600.0
SINGLE_DIE_WALL_GBPS = 1400.0
DUAL_DIE_GBPS = 3400.0


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
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
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

    # fp32 weighted sum, mirroring the kernel's `ol` fragment math.
    layer_f32 = (
        residual_flat.float() * pre_mix.unsqueeze(-1)
    ).sum(dim=1)

    if norm_weight is not None:
        # Mirror the fused kernel: stash the bf16-rounded weighted sum
        # (this is what the second sweep reads back from shared memory),
        # but accumulate sum(x^2) from the pre-rounding fp32 values.
        stash = layer_f32.bfloat16()
        rstd = torch.rsqrt(
            layer_f32.square().sum(-1) / hidden_size + norm_eps
        )
        layer_input = (
            stash.float() * rstd.unsqueeze(-1) * norm_weight.float()
        ).bfloat16()
    else:
        layer_input = layer_f32.bfloat16()

    post_mix = post_mix.view(*outer_shape, mhc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, mhc_mult, mhc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)
    return post_mix, comb_mix, layer_input


def make_inputs(num_tokens: int, n_splits: int, hidden_size: int = HIDDEN):
    residual = torch.randn(
        (num_tokens, HC, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    fn = torch.randn(
        (HC * (2 + HC), HC * hidden_size), device="cuda", dtype=torch.float32
    ) / math.sqrt(HC * hidden_size)
    hc_scale = torch.randn((3,), device="cuda", dtype=torch.float32) * 0.2 + 1.0
    hc_base = torch.randn(
        (HC * (2 + HC),), device="cuda", dtype=torch.float32
    ) * 0.2

    k_chunk = (HC * hidden_size) // n_splits
    residual_2d = residual.view(num_tokens, HC * hidden_size).float()
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


def cosine_similarity(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """fp64 cosine similarity; scale-blind, so assert_close backs it up."""
    a = actual.double().flatten()
    b = expected.double().flatten()
    denom = (a.norm() * b.norm()).clamp_min(1e-30)
    return (torch.dot(a, b) / denom).item()


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


# --------------------------------------------------------------------------
# Case matrix
# --------------------------------------------------------------------------
# Original regression suite (norm off): decode / prefill at hidden 7168 and
# the prefill shape family from the serving log at hidden 4096.
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

PREFILL_4096_NUM_TOKENS = (
    8, 16, 24, 32, 40, 48, 56, 64, 334, 366, 384, 419, 512,
)

BASE_CASES = (
    tuple(("decode", n, HIDDEN, False) for n in DECODE_NUM_TOKENS)
    + tuple(("prefill", n, HIDDEN, False) for n, _ in PREFILL_CASES)
    + tuple(
        ("prefill4096", n, HIDDEN_4096, False) for n in PREFILL_4096_NUM_TOKENS
    )
)

# Required grid: hidden {4096, 7168} x num_tokens {0,1,6,8,17,32,64}
# x use_norm {False, True}.
GRID_NUM_TOKENS = (0, 1, 6, 8, 17, 32, 64)
GRID_CASES = tuple(
    ("grid", n, h, use_norm)
    for h in (HIDDEN_4096, HIDDEN)
    for n in GRID_NUM_TOKENS
    for use_norm in (False, True)
)

# Norm sweep over the deepseek-v4 / v4.1 / v4-flash shape families.
MODELS = ("dsv4", "dsv41", "dsv4flash")
SWEEP_NUM_TOKENS = (
    1, 6, 8, 17, 32, 64, 128, 257, 512, 1024,
    2048, 3072, 4096, 6144, 8192,
)
SWEEP_CASES = tuple(
    (model, n, h, True)
    for model in MODELS
    for h in (HIDDEN_4096, HIDDEN)
    for n in SWEEP_NUM_TOKENS
)

ALL_CASES = BASE_CASES + GRID_CASES + SWEEP_CASES
NORM_CASE_COUNT = sum(1 for case in ALL_CASES if case[3])
assert NORM_CASE_COUNT >= 100, f"norm cases {NORM_CASE_COUNT} < 100"


def case_id(case) -> str:
    label, num_tokens, hidden_size, use_norm = case
    suffix = "-norm" if use_norm else ""
    return f"{label}-n{num_tokens}-h{hidden_size}{suffix}"


def run_case(case, verbose: bool = True):
    """Run one (label, num_tokens, hidden_size, use_norm) case.

    Returns (ok, gbps or None). Prints the per-case report line when verbose.
    """
    label, num_tokens, hidden_size, use_norm = case
    n_splits = compute_num_split2(num_tokens)
    torch.manual_seed(
        20260817 + num_tokens + hidden_size + len(label)
        + (7 if use_norm else 0)
    )
    (
        residual,
        fn,
        hc_scale,
        hc_base,
        gemm_out_mul,
        gemm_out_sqrsum,
    ) = make_inputs(num_tokens, n_splits, hidden_size)

    norm_weight = None
    if use_norm:
        norm_weight = torch.randn(
            (hidden_size,), device="cuda", dtype=torch.bfloat16
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
            norm_weight=norm_weight,
            norm_eps=NORM_EPS if use_norm else None,
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
        norm_weight=norm_weight,
        norm_eps=NORM_EPS if use_norm else None,
    )
    torch.cuda.synchronize()

    if num_tokens == 0:
        # Empty launch is shape-checked only (the wrapper skips the kernel).
        assert actual[0].shape == (0, HC, 1)
        assert actual[1].shape == (0, HC, HC)
        assert actual[2].shape == (0, hidden_size)
        if verbose:
            print(
                f"[mhc_pre] {label:<11} T={num_tokens:6d} h={hidden_size} "
                f"norm={int(use_norm)}  cos_sim=    ---  OK   "
                f"(empty, shape-checked)"
            )
        del residual, fn, hc_scale, hc_base, gemm_out_mul, gemm_out_sqrsum
        torch.cuda.empty_cache()
        return True, None

    # Primary metric: cosine similarity over all three outputs (fp64).
    cos = min(
        cosine_similarity(actual[i].float(), expected[i].float())
        for i in range(3)
    )
    ok = cos >= COS_THRESHOLD
    if not ok:
        print(
            f"[mhc_pre] {label:<11} T={num_tokens:6d} h={hidden_size} "
            f"norm={int(use_norm)}  cos_sim={cos:.6f} FAIL"
        )

    # Secondary scale guard: cosine similarity cannot catch a wrong global
    # scale, so also check element-wise closeness with the calibrated
    # tolerances used by the original suite.
    try:
        torch.testing.assert_close(
            actual[0], expected[0], rtol=5e-3, atol=5e-3
        )
        torch.testing.assert_close(
            actual[1], expected[1], rtol=5e-3, atol=5e-3
        )
        torch.testing.assert_close(
            actual[2].float(), expected[2].float(), rtol=2e-2, atol=2e-2
        )
    except AssertionError as err:
        ok = False
        print(f"    assert_close failed for {case_id(case)}: {err}")

    warmup = int(os.getenv("MHC_PRE_WARMUP", "20"))
    if num_tokens < 512:
        repeat = int(os.getenv("MHC_PRE_REPEAT", "200"))
    elif num_tokens < 4096:
        repeat = int(os.getenv("MHC_PRE_REPEAT", "100"))
    else:
        repeat = int(os.getenv("MHC_PRE_REPEAT", "50"))
    rounds = int(os.getenv("MHC_PRE_BENCH_ROUNDS", "3"))

    times = [benchmark_ms(run, warmup=0, repeat=repeat) for _ in range(rounds)]
    latency_ms = statistics.median(times)

    tensors = (
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        *actual,
    )
    if use_norm:
        tensors = (norm_weight, *tensors)
    gbps = effective_bandwidth_gbps(tensors, latency_ms)

    ok_str = "OK" if ok else "FAIL"
    if verbose:
        print(
            f"[mhc_pre] {label:<11} T={num_tokens:6d} h={hidden_size} "
            f"norm={int(use_norm)}  cos_sim={cos:.6f} {ok_str:4s} "
            f"{latency_ms:8.4f} ms  {gbps:8.1f} GB/s"
        )

    del residual, fn, hc_scale, hc_base, gemm_out_mul, gemm_out_sqrsum
    del actual, expected
    torch.cuda.empty_cache()
    return ok, gbps


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
@pytest.mark.parametrize(
    "case",
    ALL_CASES,
    ids=[case_id(c) for c in ALL_CASES],
)
def test_mhc_pre_big_fuse_accuracy_and_performance(case):
    ok, _ = run_case(case)
    assert ok, f"accuracy check failed for {case_id(case)}"


def main() -> None:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES={vis!r}  device_count={torch.cuda.device_count()}")
    assert torch.cuda.is_available(), "CUDA not available"
    print(
        f"config: op=mhc_pre(tilelang) hidden={{4096,7168}} residual=bf16 | "
        f"norm: RMSNorm(weight=bf16, eps={NORM_EPS}) | "
        f"cases={len(ALL_CASES)} (norm cases={NORM_CASE_COUNT}) | "
        f"threshold cos_sim>={COS_THRESHOLD}"
    )
    print("=" * 100)

    all_ok = True
    peak = 0.0
    for case in ALL_CASES:
        ok, gbps = run_case(case)
        all_ok &= ok
        if gbps is not None:
            peak = max(peak, gbps)

    print("=" * 100)
    print(
        f"Accuracy: {'ALL PASS' if all_ok else 'FAILURES PRESENT'} "
        f"({len(ALL_CASES)} cases, threshold cos_sim >= {COS_THRESHOLD})"
    )
    reached = peak >= TARGET_GBPS
    print(
        f"Peak bandwidth: {peak:.1f} GB/s "
        f"(target {TARGET_GBPS:.0f} -> {'reached' if reached else 'NOT reached'}; "
        f"single-die practical wall ~{SINGLE_DIE_WALL_GBPS:.0f}, "
        f"dual-die datasheet ~{DUAL_DIE_GBPS:.0f})"
    )
    assert all_ok, "accuracy check failed"


if __name__ == "__main__":
    main()
