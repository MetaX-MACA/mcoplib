"""使用完整生产 case 对比 MHC post 的 CUDA 与 TileLang 实现。

在 CUDA/MACA 机器上运行::

    pytest -q -s unit_test/test_mhc_post_cuda.py

测试沿用 ``test_tilelang_mhc_post.py`` 的 case、输入、预热、重复次数、
测量轮数和有效带宽口径。可通过 ``MHC_POST_WARMUP``、
``MHC_POST_DECODE_REPEAT``、``MHC_POST_PREFILL_REPEAT`` 和
``MHC_POST_BENCH_ROUNDS`` 调整测试参数。
"""

import os
import statistics
import unicodedata

import pytest
import torch

pytest.importorskip("tilelang")

import mcoplib.op as ops
from mcoplib.tilelang_mhc_post import mhc_post as mhc_post_tilelang


HC = 4
HIDDEN_SIZES = (4096, 7168)

DECODE_NUM_TOKENS = (1, 10, 13, 26, 64, 127, 224, 256)

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


def torch_reference(comb, residual, post, layer_output):
    """FP32 参考计算，最后转换为 BF16。"""
    output = torch.bmm(comb.transpose(1, 2), residual.float())
    output.add_(post * layer_output.float().unsqueeze(1))
    return output.to(torch.bfloat16)


def make_inputs(num_tokens: int, hidden: int):
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


def benchmark_ms(fn, repeat: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def benchmark_pair_ms(tilelang_fn, cuda_fn, warmup: int, repeat: int, rounds: int):
    """每轮交替执行顺序，返回耗时中位数。"""
    for _ in range(warmup):
        tilelang_fn()
        cuda_fn()
    torch.cuda.synchronize()

    tilelang_times = []
    cuda_times = []
    for index in range(rounds):
        if index % 2 == 0:
            tilelang_times.append(benchmark_ms(tilelang_fn, repeat))
            cuda_times.append(benchmark_ms(cuda_fn, repeat))
        else:
            cuda_times.append(benchmark_ms(cuda_fn, repeat))
            tilelang_times.append(benchmark_ms(tilelang_fn, repeat))
    return statistics.median(tilelang_times), statistics.median(cuda_times)


def effective_bandwidth_gbps(tensors, latency_ms: float) -> float:
    """按输入各读取一次、输出写入一次估算有效带宽。"""
    total_bytes = sum(t.numel() * t.element_size() for t in tensors)
    return total_bytes / latency_ms / 1e6


def display_width(text):
    """返回字符串在等宽终端中的显示宽度。"""
    return sum(
        2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        for char in str(text)
    )


def pad_cell(value, width, right=False):
    text = str(value)
    padding = " " * (width - display_width(text))
    return padding + text if right else text + padding


def print_table(rows):
    headers = (
        "阶段",
        "H",
        "N",
        "TileLang(ms)",
        "CUDA(ms)",
        "加速比",
        "TileLang(GB/s)",
        "CUDA(GB/s)",
        "带宽提升",
        "最大误差",
    )
    values = []
    for row in rows:
        values.append(
            (
                "decoder" if row["phase"] == "decode" else "prefill",
                str(row["hidden"]),
                str(row["tokens"]),
                f"{row['tilelang_ms']:.4f}",
                f"{row['cuda_ms']:.4f}",
                f"{row['speedup']:.2f}x",
                f"{row['tilelang_bw']:.1f}",
                f"{row['cuda_bw']:.1f}",
                f"{row['bw_uplift']:.2f}x",
                f"{row['max_abs_error']:.6g}",
            )
        )

    widths = [
        max(display_width(header), *(display_width(row[i]) for row in values))
        for i, header in enumerate(headers)
    ]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    print()
    print(separator)
    print("| " + " | ".join(pad_cell(value, widths[i]) for i, value in enumerate(headers)) + " |")
    print(separator)
    for row in values:
        cells = [
            pad_cell(value, widths[i], right=i > 0)
            for i, value in enumerate(row)
        ]
        print("| " + " | ".join(cells) + " |")
    print(separator)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA/MACA GPU")
def test_mhc_post_cuda_vs_tilelang():
    warmup = int(os.getenv("MHC_POST_WARMUP", "20"))
    rounds = int(os.getenv("MHC_POST_BENCH_ROUNDS", "3"))
    rows = []

    for hidden in HIDDEN_SIZES:
        for phase, num_tokens in CASES:
            torch.manual_seed(20260817 + num_tokens)
            comb, residual, post, layer_output = make_inputs(num_tokens, hidden)

            def run_tilelang():
                return mhc_post_tilelang(layer_output, residual, post, comb)

            def run_cuda():
                return ops.mhc_post_cuda(layer_output, residual, post, comb)

            # 首次调用包含动态加载/JIT，不计入稳态耗时。
            tilelang_output = run_tilelang()
            cuda_output = run_cuda()
            expected = torch_reference(comb, residual, post, layer_output)
            torch.cuda.synchronize()

            torch.testing.assert_close(
                tilelang_output, expected, rtol=2e-2, atol=2e-2
            )
            torch.testing.assert_close(
                cuda_output, expected, rtol=2e-2, atol=2e-2
            )
            torch.testing.assert_close(
                cuda_output, tilelang_output, rtol=2e-2, atol=2e-2
            )
            max_abs_error = (
                cuda_output.float() - tilelang_output.float()
            ).abs().max().item()

            default_repeat = "200" if phase == "decode" else "50"
            repeat = int(
                os.getenv(f"MHC_POST_{phase.upper()}_REPEAT", default_repeat)
            )
            tilelang_ms, cuda_ms = benchmark_pair_ms(
                run_tilelang,
                run_cuda,
                warmup=warmup,
                repeat=repeat,
                rounds=rounds,
            )
            tensors = (comb, residual, post, layer_output, cuda_output)
            tilelang_bw = effective_bandwidth_gbps(tensors, tilelang_ms)
            cuda_bw = effective_bandwidth_gbps(tensors, cuda_ms)
            rows.append(
                {
                    "phase": phase,
                    "hidden": hidden,
                    "tokens": num_tokens,
                    "tilelang_ms": tilelang_ms,
                    "cuda_ms": cuda_ms,
                    "speedup": tilelang_ms / cuda_ms,
                    "tilelang_bw": tilelang_bw,
                    "cuda_bw": cuda_bw,
                    "bw_uplift": cuda_bw / tilelang_bw,
                    "max_abs_error": max_abs_error,
                }
            )

            del comb, residual, post, layer_output
            del tilelang_output, cuda_output, expected
            torch.cuda.empty_cache()

    print_table(rows)
