# SPDX-License-Identifier: Apache-2.0
"""chunk_kda_fwd_kernel_intra_token_parallel 正确性与性能单测。需求来自[MC3-14536]

测试对象
========
直接调用 mcoplib.triton_sglang_chunk_intra_token_parallel 中的
chunk_kda_fwd_intra_token_parallel。本文件不导入、不运行 SGLang baseline，
性能结果仅表示当前 mcoplib 算子的稳定态单次耗时。

覆盖范围
========
* 正式 shape：T={2K,4K,8K,16K}，batch={1,7,32,128,256}，H={12,24}。
* 边界 shape：1、15/16/17、63/64/65、127/128/129、2047/2049 token，
  以及非整齐的多请求变长序列。
* varlen 输入保持 B=1，逻辑 batch 由 len(cu_seqlens)-1 表示。
* 额外检查少量 cu_seqlens=None 的 fixed-length 输入。
* K=128、BT=64、BC=16；q/k/Aqk=BF16，gk/beta/Akk=FP32，
  cu_seqlens=INT32，与当前生产 dump 的 dtype 元数据一致。

正确性
======
使用直接计算 2**(g_i-g_j) 的 FP64 PyTorch 公式作为独立参考，检查 Aqk、
Akk 的有效下三角区域、NaN/Inf、误差阈值以及未写区域是否被意外覆盖。

性能
====
正式 shape 完成预热后使用 CUDA Event 计时，默认 50 次 warmup、300 次重复、
3 次 trial，输出 trial 中位数。测试前必须通过 mx-smi 选择空闲 C600u；
GPU 被其他任务占用时得到的异常结果应作废重测。

运行
====
    cd /home/m01775/dev/mcoplib
    source env_local.sh
    mx-smi
    export CUDA_VISIBLE_DEVICES=<空闲卡号>
    python unit_test/test_chunk_kda_fwd_kernel_intra_token_parallel.py

H=24 是根据 TP8 改为 TP4 后本地 head 数可能翻倍加入的覆盖项，最终仍以
PP8TP4 真实运行 shape 为准。该脚本是单卡单算子测试，不验证双机整网测试。
"""

import argparse
import os
import pathlib
import statistics
import sys
import traceback

import torch


# 从源码树直接运行时，优先导入当前仓库内的 mcoplib。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mcoplib.triton_sglang_chunk_intra_token_parallel import (  # noqa: E402
    chunk_kda_fwd_intra_token_parallel,
)


K = 128
BT = 64
BC = 16
SCALE = K ** -0.5
MAIN_TOKENS = (2048, 4096, 8192, 16384)
MAIN_BATCHES = (1, 7, 32, 128, 256)
EDGE_VARLEN = (
    (1, 1),
    (15, 1), (16, 1), (17, 1),
    (63, 1), (64, 1), (65, 1),
    (127, 1), (128, 1), (129, 1),
    (2047, 1), (2049, 1),
    (17, 2), (33, 3), (65, 7), (129, 15), (257, 31), (2048, 255),
)
FIXED_CASES = ((2048, 1), (2048, 8), (2048, 256))
ATOL = 2e-3
RTOL = 1e-2


def _make_lengths(total_tokens, requests):
    """构造确定性的非等长请求，保证每个请求至少有一个 token。"""
    if not 1 <= requests <= total_tokens:
        raise ValueError("requests must be in [1, total_tokens]")
    weights = [(index * 17 % 31) + 1 for index in range(requests)]
    remaining = total_tokens - requests
    weight_sum = sum(weights)
    lengths = [1 + remaining * weight // weight_sum for weight in weights]
    lengths[-1] += total_tokens - sum(lengths)
    return lengths


def _bounds_from_lengths(lengths):
    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)
    return bounds


def _make_inputs(total_tokens, requests, heads, layout, seed):
    if layout == "fixed":
        if total_tokens % requests:
            raise ValueError("fixed layout requires total_tokens divisible by requests")
        lengths = [total_tokens // requests] * requests
        shape = (requests, lengths[0], heads, K)
        cu_seqlens = None
    else:
        lengths = _make_lengths(total_tokens, requests)
        shape = (1, total_tokens, heads, K)
        cu_seqlens = torch.tensor(
            _bounds_from_lengths(lengths), device="cuda", dtype=torch.int32
        )

    bounds = _bounds_from_lengths(lengths)
    torch.manual_seed(seed)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    beta = torch.rand(shape[:-1], device="cuda", dtype=torch.float32)

    # gk 是每个请求内、每个 BT chunk 重新开始的负 gate 增量累加。
    step = -torch.rand(
        total_tokens, heads, K, device="cuda", dtype=torch.float32
    ) * 0.05
    gk_flat = torch.empty_like(step)
    for bos, eos in zip(bounds[:-1], bounds[1:]):
        for start in range(bos, eos, BT):
            end = min(start + BT, eos)
            gk_flat[start:end] = step[start:end].cumsum(0)
    gk = gk_flat.reshape(shape)

    Aqk = torch.full(
        (*shape[:-1], BT), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    Akk = torch.full(
        (*shape[:-1], BC), float("nan"), device="cuda", dtype=torch.float32
    )
    return (q, k, gk, beta, Aqk, Akk, cu_seqlens), bounds


def _launch(inputs):
    q, k, gk, beta, Aqk, Akk, cu_seqlens = inputs
    return chunk_kda_fwd_intra_token_parallel(
        q=q,
        k=k,
        gk=gk,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=SCALE,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        sub_chunk_size=BC,
    )


def _update_error(stats, actual, expected):
    actual = actual.double()
    expected = expected.double()
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    stats["nonfinite"] += int((~finite).sum().item())
    diff = (actual - expected).abs()
    if diff.numel():
        stats["max_abs"] = max(
            stats["max_abs"],
            float(torch.nan_to_num(diff, nan=float("inf")).max().item()),
        )
        relative = diff / expected.abs().clamp_min(1e-8)
        stats["max_rel"] = max(
            stats["max_rel"],
            float(torch.nan_to_num(relative, nan=float("inf")).max().item()),
        )
    stats["mismatches"] += int(
        (~torch.isclose(actual, expected, atol=ATOL, rtol=RTOL)).sum().item()
    )


def _check_correctness(inputs, bounds):
    """逐 sub-chunk 与直接 FP64 pairwise 参考比较。"""
    q, k, gk, beta, Aqk, Akk, _ = inputs
    heads = q.shape[2]
    qf = q.reshape(-1, heads, K)
    kf = k.reshape(-1, heads, K)
    gf = gk.reshape(-1, heads, K)
    bf = beta.reshape(-1, heads)
    aqf = Aqk.reshape(-1, heads, BT)
    akf = Akk.reshape(-1, heads, BC)
    stats = {
        "Aqk": {"max_abs": 0.0, "max_rel": 0.0, "mismatches": 0, "nonfinite": 0},
        "Akk": {"max_abs": 0.0, "max_rel": 0.0, "mismatches": 0, "nonfinite": 0},
    }
    expected_written = 0

    for bos, eos in zip(bounds[:-1], bounds[1:]):
        for start in range(bos, eos, BC):
            end = min(start + BC, eos)
            count = end - start
            q_block = qf[start:end].double()
            k_block = kf[start:end].double()
            g_block = gf[start:end].double()
            beta_block = bf[start:end].double()

            decay = torch.exp2(g_block[:, None] - g_block[None, :])
            Aqk_ref = (
                (q_block[:, None] * k_block[None, :] * decay).sum(-1) * SCALE
            ).permute(0, 2, 1)
            Akk_ref = (
                (k_block[:, None] * k_block[None, :] * decay).sum(-1)
                * beta_block[:, None, :]
            )
            strict = torch.tril(
                torch.ones(count, count, device="cuda", dtype=torch.float64),
                diagonal=-1,
            )
            Akk_ref = (Akk_ref * strict[:, :, None]).permute(0, 2, 1)

            col0 = (start - bos) % BT
            Aqk_block = aqf[start:end, :, col0:col0 + count]
            Akk_block = akf[start:end, :, :count]
            lower = torch.tril(
                torch.ones(count, count, device="cuda", dtype=torch.bool)
            )[:, None, :].expand(count, heads, count)
            _update_error(stats["Aqk"], Aqk_block[lower], Aqk_ref[lower])
            _update_error(stats["Akk"], Akk_block[lower], Akk_ref[lower])
            expected_written += int(lower.sum().item())

    # 有效区域已检查；总写入数不等说明未写区域被覆盖。
    aq_written = int((~torch.isnan(Aqk)).sum().item())
    ak_written = int((~torch.isnan(Akk)).sum().item())
    write_set_ok = aq_written == expected_written and ak_written == expected_written
    passed = write_set_ok and all(
        item["mismatches"] == 0 and item["nonfinite"] == 0
        for item in stats.values()
    )
    return passed, stats, write_set_ok


def _benchmark(fn, warmup, repeats, trials):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    measurements = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end) * 1000.0 / repeats)
    return statistics.median(measurements), measurements


def _run_case(total_tokens, requests, heads, layout, performance,
              warmup, repeats, trials, seed):
    inputs, bounds = _make_inputs(total_tokens, requests, heads, layout, seed)
    _launch(inputs)  # 首次调用包含编译，不计入稳定态性能。
    torch.cuda.synchronize()
    passed, stats, write_set_ok = _check_correctness(inputs, bounds)
    latency_us = None
    trial_us = None
    if performance:
        latency_us, trial_us = _benchmark(
            lambda: _launch(inputs), warmup, repeats, trials
        )
    return passed, stats, write_set_ok, latency_us, trial_us


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        description="MCOPLIB KDA intra-token-parallel correctness/performance test"
    )
    parser.add_argument("--heads", type=int, nargs="+", default=[12, 24])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--skip-edge", action="store_true")
    parser.add_argument("--skip-fixed", action="store_true")
    args = parser.parse_args()
    args.heads = list(dict.fromkeys(args.heads))
    if not args.heads or any(heads < 1 for heads in args.heads):
        parser.error("--heads must contain positive values")
    if min(args.warmup, args.repeats, args.trials) < 1:
        parser.error("--warmup, --repeats and --trials must be positive")
    if not torch.cuda.is_available():
        parser.error("no visible CUDA/MACA device")

    print(
        f"device={torch.cuda.get_device_name()} "
        f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')!r}"
    )
    print(
        f"dtype=q/k/Aqk:bf16 gk/beta/Akk:fp32 cu:int32; "
        f"K={K} BT={BT} BC={BC}; heads={args.heads}"
    )
    print(
        f"performance: warmup={args.warmup} repeats={args.repeats} "
        f"trials={args.trials} (median)"
    )

    cases = [
        ("main", tokens, requests, heads, "varlen", True)
        for heads in args.heads
        for tokens in MAIN_TOKENS
        for requests in MAIN_BATCHES
    ]
    if not args.skip_edge:
        cases.extend(
            ("edge", tokens, requests, args.heads[0], "varlen", False)
            for tokens, requests in EDGE_VARLEN
        )
    if not args.skip_fixed:
        cases.extend(
            ("fixed", tokens, requests, heads, "fixed", False)
            for heads in args.heads
            for tokens, requests in FIXED_CASES
        )

    all_passed = True
    performance_results = []
    for index, (suite, tokens, requests, heads, layout, performance) in enumerate(cases, 1):
        tag = f"{suite} T={tokens} batch={requests} H={heads} {layout}"
        try:
            passed, stats, write_set_ok, latency_us, trial_us = _run_case(
                tokens, requests, heads, layout, performance,
                args.warmup, args.repeats, args.trials, seed=123 + index,
            )
            all_passed = all_passed and passed
            perf_text = ""
            if latency_us is not None:
                performance_results.append((tokens, requests, heads, latency_us))
                perf_text = f" latency={latency_us:.3f} us trials={trial_us}"
            print(
                f"[{index:02d}/{len(cases)}] {'PASS' if passed else 'FAIL'} {tag}"
                f" Aqk_abs={stats['Aqk']['max_abs']:.6g}"
                f" Akk_abs={stats['Akk']['max_abs']:.6g}"
                f" write_set={'OK' if write_set_ok else 'BAD'}{perf_text}",
                flush=True,
            )
        except Exception:
            all_passed = False
            print(f"[{index:02d}/{len(cases)}] ERROR {tag}", flush=True)
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()

    print("=" * 96)
    print(f"Correctness: {'ALL PASS' if all_passed else 'FAIL'}")
    if performance_results:
        fastest = min(performance_results, key=lambda item: item[3])
        slowest = max(performance_results, key=lambda item: item[3])
        print(
            f"Performance cases: {len(performance_results)}; "
            f"fastest=T{fastest[0]}/B{fastest[1]}/H{fastest[2]} "
            f"{fastest[3]:.3f} us; "
            f"slowest=T{slowest[0]}/B{slowest[1]}/H{slowest[2]} "
            f"{slowest[3]:.3f} us"
        )
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
