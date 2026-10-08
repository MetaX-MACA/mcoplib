# SPDX-License-Identifier: Apache-2.0
"""_combine_kernel 精度 + 带宽单元测试 (MetaX C600-U / MACA).

语义 (Kimi-K3 attention residual 聚合 Step 2):
  对每个 token, 对 nvb+1 个有效行的 scores 做 softmax, 再对
  (nvb 个 bank 行 + 1 个 prefix 行) 做加权和, 写出 [T, H] 输出。
  纯 streaming (triad 型: 多读 + 一写), memory-bound。

运行 (务必先设置环境变量并挑空闲卡):
    cd /home/yiyu/mcoplib/mcoplib_dev/mcoplib
    source /home/yiyu/mcoplib/mcoplib_dev/mcoplib/env_local.sh
    mx-smi                        # 找 GPU-Util 0% / Memory ~540MiB Available 的卡
    export CUDA_VISIBLE_DEVICES=0
    /opt/conda/bin/python unit_test/test_triton_combine_kernel.py

说明:
  * triton_sglang_score_kernel.py 顶层 import sglang(RMSNorm/ReplicatedLinear/utils),
    这些仅供包装函数使用, _combine_kernel 本身是纯 triton。为脱离 sglang 依赖
    (与 test_op_sglang_triton_score_kernel.py 相同做法), 这里在 import 前向
    sys.modules 注入桩模块, 从而只加载 triton kernel。
  * 精度: 余弦相似度 (cos_sim), 阈值 0.99999。
  * 带宽: 真实 HBM 流量 = 读入的 bf16 行 (T*(nvb+1)*H*2) + 输出行 (T*H*2)
    + scores 被 n_h_blocks 个 CTA 各读一次 (T*MAX_ROWS*4*n_h_blocks)。
    用连续 burst 计时撑住 DVFS(见 C600 优化指南 §2.2)。
  * 输出格式对齐 test_op_moe_scatter_dynamic_quant.py。
"""

import os
import pathlib
import sys
import time
import types

import torch
import triton

# --------------------------------------------------------------------------- #
# 让 `import triton_sglang_score_kernel` 从本文件同目录解析。
# --------------------------------------------------------------------------- #
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


# --------------------------------------------------------------------------- #
# 注入 sglang 桩模块 (只为让 triton_sglang_score_kernel 顶层 import 通过)。
# --------------------------------------------------------------------------- #
def _install_sglang_stubs() -> None:
    if "sglang" in sys.modules:
        return

    class _Stub:  # 占位, 本测试只用 _combine_kernel, 不实例化这些类
        pass

    def _false(*_a, **_k):
        return False

    sglang = types.ModuleType("sglang")
    srt = types.ModuleType("sglang.srt")
    layers = types.ModuleType("sglang.srt.layers")
    layernorm = types.ModuleType("sglang.srt.layers.layernorm")
    linear = types.ModuleType("sglang.srt.layers.linear")
    utils = types.ModuleType("sglang.srt.utils")

    layernorm.RMSNorm = _Stub
    linear.ReplicatedLinear = _Stub
    utils.is_hip = _false
    utils.is_npu = _false

    sglang.srt = srt
    srt.layers = layers
    layers.layernorm = layernorm
    layers.linear = linear
    srt.utils = utils

    for name, mod in {
        "sglang": sglang,
        "sglang.srt": srt,
        "sglang.srt.layers": layers,
        "sglang.srt.layers.layernorm": layernorm,
        "sglang.srt.layers.linear": linear,
        "sglang.srt.utils": utils,
    }.items():
        sys.modules[name] = mod


_install_sglang_stubs()

from mcoplib.triton_sglang_score_kernel import _combine_kernel, _BLOCK_H, _MAX_ROWS  # noqa: E402


# --------------------------------------------------------------------------- #
# 测试配置 (与需求一致)。
# --------------------------------------------------------------------------- #
HIDDEN_SIZE = 7168
BLOCK_H = _BLOCK_H          # 1024
MAX_ROWS = _MAX_ROWS        # 16
TOKEN_COUNTS = (2048, 4096, 8192, 16384)
# (allocated bank rows, valid bank rows); all pairs occur for every T.
BANK_ROWS_TO_NVB = (
    (2, (1, 2)),
    (8, (1, 2)),
)
SHAPES = tuple(
    (num_tokens, bank_rows, nvb)
    for num_tokens in TOKEN_COUNTS
    for bank_rows, nvb_values in BANK_ROWS_TO_NVB
    for nvb in nvb_values
)

COS_THRESHOLD = 0.99999
SINGLE_DIE_TARGET = 1300.0  # GB/s
DUAL_DIE_DATASHEET = 3480.0


def _launch(prefix_sum, bank, scores, out, nvb):
    num_tokens = prefix_sum.shape[0]
    n_h_blocks = HIDDEN_SIZE // BLOCK_H
    _combine_kernel[(num_tokens, n_h_blocks)](
        prefix_sum,
        bank,
        scores,
        out,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        out.stride(0),
        NVB=nvb,
        BLOCK_H=BLOCK_H,
        MAX_ROWS=MAX_ROWS,
        num_warps=int(os.getenv("NW", "1")),
    )


def _reference(prefix_sum, bank, scores, nvb):
    """softmax over the nvb+1 valid scores, then the weighted row sum (fp32 参考)。"""
    rows = torch.cat(
        [bank[:, :nvb].float(), prefix_sum[:, None].float()], dim=1
    )  # [T, nvb+1, H]
    probs = torch.softmax(scores[:, : nvb + 1].float(), dim=-1)  # [T, nvb+1]
    mixed = (probs.unsqueeze(-1) * rows).sum(dim=1)
    return mixed.to(prefix_sum.dtype)


def _cos_sim(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def _bench_burst(fn, warm=0.6, run=0.6):
    """连续 back-to-back 发射撑住 DVFS, 取每次 launch 的最优耗时 (ms)。"""
    torch.cuda.synchronize()
    t0 = time.time()
    n = 0
    while time.time() - t0 < warm:
        fn()
        n += 1
        if n % 64 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    reps = max(64, n)
    best_ms = float("inf")
    t0 = time.time()
    while time.time() - t0 < run:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record()
        e.synchronize()
        best_ms = min(best_ms, s.elapsed_time(e) / reps)
    return best_ms


def main():
    if not torch.cuda.is_available():
        print(
            "no visible CUDA/MACA device: 先 source env_local.sh 并 "
            "export CUDA_VISIBLE_DEVICES=<空闲卡号>",
            file=sys.stderr,
        )
        sys.exit(1)

    dev = torch.cuda.current_device()
    print(
        f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')!r}  "
        f"device={torch.cuda.get_device_name(dev)}  count={torch.cuda.device_count()}"
    )
    n_h_blocks = HIDDEN_SIZE // BLOCK_H
    print(
        f"config: H={HIDDEN_SIZE} BLOCK_H={BLOCK_H} MAX_ROWS={MAX_ROWS} "
        f"n_h_blocks={n_h_blocks} dtype=bf16(rows/out)/fp32(scores)"
    )
    print("=" * 100)

    torch.manual_seed(0)
    max_tokens = max(TOKEN_COUNTS)
    max_rows = max(br for br, _ in BANK_ROWS_TO_NVB)
    prefix_store = torch.randn(
        max_tokens, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    bank_store = torch.randn(
        max_tokens, max_rows, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    score_store = torch.randn(max_tokens, MAX_ROWS, device="cuda", dtype=torch.float32)
    out_store = torch.empty(
        max_tokens, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )

    all_pass = True
    peak_gbps = 0.0
    peak_shape = None

    for num_tokens, bank_rows, nvb in SHAPES:
        prefix_sum = prefix_store[:num_tokens]
        bank = bank_store[:num_tokens, :bank_rows, :].contiguous()
        scores = score_store[:num_tokens]
        out = out_store[:num_tokens]
        out.fill_(float("nan"))

        _launch(prefix_sum, bank, scores, out, nvb)  # 编译 + 结果
        torch.cuda.synchronize()

        expected = _reference(prefix_sum, bank, scores, nvb)
        cos = _cos_sim(out, expected)
        has_nan = bool(torch.isnan(out).any().item())
        ok = (cos >= COS_THRESHOLD) and not has_nan
        all_pass = all_pass and ok

        ms = _bench_burst(lambda: _launch(prefix_sum, bank, scores, out, nvb))
        rows_read = num_tokens * (nvb + 1) * HIDDEN_SIZE * 2  # bf16 行读
        out_write = num_tokens * HIDDEN_SIZE * 2  # bf16 行写
        score_read = num_tokens * MAX_ROWS * 4 * n_h_blocks  # scores 每 CTA 各读一次
        hbm_bytes = rows_read + out_write + score_read
        gbps = hbm_bytes / (ms * 1e6)

        if gbps > peak_gbps:
            peak_gbps = gbps
            peak_shape = (num_tokens, bank_rows, nvb)

        print(
            f"[combine] T={num_tokens:<6} rows={bank_rows} nvb={nvb}  "
            f"cos_sim={cos:.6f} {'OK  ' if ok else 'BAD '}  "
            f"{ms:8.4f} ms  {gbps:8.1f} GB/s"
        )

    print("=" * 100)
    print(
        f"Accuracy: {'ALL PASS' if all_pass else 'FAIL'} "
        f"(threshold cos_sim >= {COS_THRESHOLD})"
    )
    reached = "reached" if peak_gbps >= SINGLE_DIE_TARGET else "NOT reached"
    print(
        f"Peak bandwidth: {peak_gbps:.1f} GB/s @ T={peak_shape[0]} "
        f"rows={peak_shape[1]} nvb={peak_shape[2]}  "
        f"(single-die target {SINGLE_DIE_TARGET:.0f} -> {reached}; "
        f"dual-die datasheet {DUAL_DIE_DATASHEET:.0f})"
    )
    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
