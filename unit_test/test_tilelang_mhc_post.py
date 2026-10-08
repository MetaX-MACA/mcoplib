"""MHC post TileLang 优化前后的精度与性能对比测试。

Run on a CUDA machine with::

    pytest -q -s test_mhc_post_tilelang.py

默认进行 20 次预热，每轮测量 decode 200 次、prefill 50 次，共取 3 轮中位数。
可通过 ``MHC_POST_WARMUP``、``MHC_POST_DECODE_REPEAT``、
``MHC_POST_PREFILL_REPEAT`` 和 ``MHC_POST_BENCH_ROUNDS`` 调整。
"""

import os
import statistics

import pytest
import torch

pytest.importorskip("tilelang")


from mcoplib.tilelang_mhc_post import mhc_post_baseline, mhc_post


HC = 4
HIDDEN_SIZES = (4096, 7168)

# 覆盖线上日志 shape，并补充实际 decode 范围 1～256 的边界、奇数和对齐点。
DECODE_NUM_TOKENS = (
    1,
    10,
    13,
    26,
    64,
    127,
    224,
    256,
)

# 覆盖 prefill 范围 1～8192：分派阈值、2 的幂前后、日志值及范围上界。
PREFILL_NUM_TOKENS = (
    1,
    6,
    127,
    128,
    255,
    256,
    257,
    511,
    512,
    513,
    1023,
    1024,
    1025,
    1280,
    2047,
    2048,
    2049,
    3072,
    3862,
    4085,
    4088,
    4089,
    4095,
    4096,
    4097,
    5120,
    6144,
    7168,
    8191,
    8192,
)

CASES = tuple(("decode", n) for n in DECODE_NUM_TOKENS) + tuple(
    ("prefill", n) for n in PREFILL_NUM_TOKENS
)


def torch_reference(
    comb_res_mix: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    layer_output: torch.Tensor,
) -> torch.Tensor:
    """FP32 reference for out[n,o,h] = c[n,o]*d[n,h] + sum_i a[n,i,o]*b[n,i,h]."""
    out_fp32 = torch.bmm(
        comb_res_mix.transpose(1, 2), residual.float()
    )
    out_fp32.add_(post_layer_mix * layer_output.float().unsqueeze(1))
    return out_fp32.to(torch.bfloat16)


def make_inputs(num_tokens: int, hidden: int):
    # Comb/post ranges resemble the Sinkhorn/post-sigmoid outputs that feed the
    # production kernel, while residual/layer output use typical BF16 activations.
    comb = torch.rand(
        (num_tokens, HC, HC), device="cuda", dtype=torch.float32
    )
    comb /= comb.sum(dim=1, keepdim=True)
    residual = torch.randn(
        (num_tokens, HC, hidden), device="cuda", dtype=torch.bfloat16
    )
    post = 2.0 * torch.rand(
        (num_tokens, HC, 1), device="cuda", dtype=torch.float32
    )
    layer_output = torch.randn(
        (num_tokens, hidden), device="cuda", dtype=torch.bfloat16
    )
    return comb, residual, post, layer_output


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


def benchmark_pair_ms(before_fn, after_fn, warmup: int, repeat: int, rounds: int):
    """交替测试并取中位数，降低动态频率和测试顺序带来的偏差。"""
    for _ in range(warmup):
        before_fn()
        after_fn()
    torch.cuda.synchronize()

    before_times = []
    after_times = []
    for index in range(rounds):
        if index % 2 == 0:
            before_times.append(benchmark_ms(before_fn, 0, repeat))
            after_times.append(benchmark_ms(after_fn, 0, repeat))
        else:
            after_times.append(benchmark_ms(after_fn, 0, repeat))
            before_times.append(benchmark_ms(before_fn, 0, repeat))
    return statistics.median(before_times), statistics.median(after_times)


def effective_bandwidth_gbps(
    tensors: tuple[torch.Tensor, ...], latency_ms: float
) -> float:
    """按所有输入各读取一次、输出写入一次估算有效带宽。"""
    total_bytes = sum(t.numel() * t.element_size() for t in tensors)
    return total_bytes / latency_ms / 1e6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
@pytest.mark.parametrize(
    "hidden,phase,num_tokens",
    tuple((hidden, phase, n) for hidden in HIDDEN_SIZES for phase, n in CASES),
    ids=[
        f"h{hidden}-{'decode' if phase == 'decode' else 'prefill'}-n{n}"
        for hidden in HIDDEN_SIZES
        for phase, n in CASES
    ],
)
def test_mhc_post_tilelang_accuracy_and_performance(
    hidden: int, phase: str, num_tokens: int
):
    torch.manual_seed(20260817 + num_tokens)
    comb, residual, post, layer_output = make_inputs(num_tokens, hidden)

    def run_before():
        return mhc_post_baseline(layer_output, residual, post, comb)

    def run_after():
        return mhc_post(layer_output, residual, post, comb)

    # This first launch also performs lazy JIT compilation; compilation is
    # intentionally excluded from the reported steady-state kernel latency.
    before = run_before()
    actual = run_after()
    expected = torch_reference(comb, residual, post, layer_output)
    torch.cuda.synchronize()

    torch.testing.assert_close(before, expected, rtol=2e-2, atol=2e-2)
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    abs_error = (actual_fp32 - expected_fp32).abs()
    max_abs_error = abs_error.max().item()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    warmup = int(os.getenv("MHC_POST_WARMUP", "20"))
    default_repeat = "200" if phase == "decode" else "50"
    repeat = int(os.getenv(f"MHC_POST_{phase.upper()}_REPEAT", default_repeat))
    rounds = int(os.getenv("MHC_POST_BENCH_ROUNDS", "3"))
    before_ms, after_ms = benchmark_pair_ms(
        run_before, run_after, warmup=warmup, repeat=repeat, rounds=rounds
    )
    tensors = (comb, residual, post, layer_output, actual)
    before_bandwidth = effective_bandwidth_gbps(tensors, before_ms)
    after_bandwidth = effective_bandwidth_gbps(tensors, after_ms)
    speedup = before_ms / after_ms

    phase_name = "decode" if phase == "decode" else "prefill"
    print(
        f"{phase_name} | hidden={hidden} | token={num_tokens} | "
        f"误差={max_abs_error:.6g} | "
        f"原版={before_ms:.4f} ms/{before_bandwidth:.1f} GB/s | "
        f"优化={after_ms:.4f} ms/{after_bandwidth:.1f} GB/s | "
        f"加速={speedup:.2f}x"
    )

    del comb, residual, post, layer_output, before, actual, expected
    torch.cuda.empty_cache()