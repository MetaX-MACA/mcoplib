# SPDX-License-Identifier: Apache-2.0
"""_score_kernel 精度 + 带宽单元测试 (MetaX C600-U / MACA).

运行 (务必先设置环境变量并挑空闲卡):
    cd /home/sw/yiyu/mcoplib/triton
    source /home/sw/yiyu/mcoplib/mcoplib/env_local.sh
    mx-smi                        # 找 GPU-Util 0% 且 Available 的卡
    export CUDA_VISIBLE_DEVICES=0,1
    /opt/conda/bin/python test_op_triton_score_kernel.py

说明:
  * triton_score_kernel.py 顶层 import sglang(RMSNorm/ReplicatedLinear/utils),
    这些仅供包装函数使用, _score_kernel 本身是纯 triton。为脱离 sglang 依赖
    (与本仓库 test_dsa_kpool_multi_pool.py 相同做法), 这里在 import 前向
    sys.modules 注入桩模块, 从而只加载 triton kernel。
  * 精度: 余弦相似度 (cos_sim), 阈值 0.99999。
  * 带宽: 真实 HBM 流量 = 读入的 bf16 行 (T*(nvb+1)*H*2) + cw(H*4,一次)
    + 标量输出(T*(nvb+1)*4)。用连续 burst 计时撑住 DVFS(见 C600 优化指南 §2.2)。
"""

import os
import pathlib
import sys
import time
import types

import torch
import triton

# --------------------------------------------------------------------------- #
# 让 `import triton_score_kernel` 从本文件同目录解析。
# --------------------------------------------------------------------------- #
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


# --------------------------------------------------------------------------- #
# 注入 sglang 桩模块 (只为让 triton_score_kernel 顶层 import 通过)。
# --------------------------------------------------------------------------- #
def _install_sglang_stubs() -> None:
    if "sglang" in sys.modules:
        return

    class _Stub:  # 占位, 本测试只用 _score_kernel, 不实例化这些类
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

from mcoplib.triton_sglang_score_kernel import _score_kernel  # noqa: E402


# --------------------------------------------------------------------------- #
# 测试配置 (与 test_bench_score_kernel.py / 需求一致)。
# --------------------------------------------------------------------------- #
HIDDEN_SIZE = 7168
BLOCK_H = int(os.getenv("BH", "256"))  # _score_kernel H-chunk (C600-U tuned)
EPS = 1e-6
TOKEN_COUNTS = (3960, 4096, 8192, 8200)
# (allocated bank rows, valid bank rows)
BANK_ROWS_TO_NVB = (
    (2, (1, 2)),
    (4, (2, 3, 4)),
    (6, (4, 5, 6)),
    (8, (6, 7, 8)),
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


def _launch(prefix_sum, bank, cw, scores, nvb):
    num_tokens = prefix_sum.shape[0]
    maxr = 1 << (nvb + 1 - 1).bit_length()  # next_pow2(nvb+1)
    _score_kernel[(num_tokens,)](
        prefix_sum,
        bank,
        cw,
        scores,
        nvb,
        EPS,
        prefix_sum.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        H=HIDDEN_SIZE,
        #BLOCK_H=BLOCK_H,
        MAXR=maxr,
        #num_warps=int(os.getenv("NW", "2")),
    )


def _reference(prefix_sum, bank, cw, nvb):
    """score[t, j] = <row, cw> * rsqrt(mean(row^2) + eps),  fp32 参考。"""
    rows = torch.cat(
        [bank[:, :nvb, :].float(), prefix_sum[:, None, :].float()], dim=1
    )  # [T, nvb+1, H]
    dot = (rows * cw).sum(-1)
    rms = torch.rsqrt(rows.square().mean(-1) + EPS)
    return dot * rms  # [T, nvb+1]


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
    print(
        f"config: H={HIDDEN_SIZE} BLOCK_H={BLOCK_H} eps={EPS} "
        f"dtype=bf16(rows)/fp32(cw,score)"
    )
    print("=" * 92)

    torch.manual_seed(0)
    max_tokens = max(TOKEN_COUNTS)
    max_rows = max(br for br, _ in BANK_ROWS_TO_NVB)
    prefix_store = torch.randn(
        max_tokens, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    bank_store = torch.randn(
        max_tokens, max_rows, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    cw = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.float32)
    score_store = torch.empty(max_tokens, 16, device="cuda", dtype=torch.float32)

    all_pass = True
    peak_gbps = 0.0
    peak_shape = None
    results = []

    for num_tokens, bank_rows, nvb in SHAPES:
        prefix_sum = prefix_store[:num_tokens]
        bank = bank_store[:num_tokens, :bank_rows, :].contiguous()
        scores = score_store[:num_tokens]
        scores.fill_(float("nan"))

        _launch(prefix_sum, bank, cw, scores, nvb)  # 编译 + 结果
        torch.cuda.synchronize()

        actual = scores[:, : nvb + 1]
        expected = _reference(prefix_sum, bank, cw, nvb)
        cos = _cos_sim(actual, expected)
        ok = cos >= COS_THRESHOLD
        all_pass = all_pass and ok

        ms = _bench_burst(lambda: _launch(prefix_sum, bank, cw, scores, nvb))
        rows_read = num_tokens * (nvb + 1) * HIDDEN_SIZE * 2  # bf16 行读
        cw_read = HIDDEN_SIZE * 4  # fp32, 一次
        out_write = num_tokens * (nvb + 1) * 4  # fp32 标量输出
        hbm_bytes = rows_read + cw_read + out_write
        gbps = hbm_bytes / (ms * 1e6)

        if gbps > peak_gbps:
            peak_gbps = gbps
            peak_shape = (num_tokens, bank_rows, nvb)
        results.append((num_tokens, bank_rows, nvb, cos, ok, ms, gbps))

        print(
            f"[score] T={num_tokens:<5} rows={bank_rows} nvb={nvb}  "
            f"cos_sim={cos:.6f} {'OK  ' if ok else 'BAD '}  "
            f"{ms:8.4f} ms  {gbps:8.1f} GB/s"
        )

    print("=" * 92)
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
    if os.getenv("SWEEP"):
        # quick peak-only sweep on representative shapes for tuning knobs
        torch.manual_seed(0)
        cw = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.float32)
        for (num_tokens, bank_rows, nvb) in [
            (3960, 8, 7), (8192, 8, 8), (4096, 4, 4),
        ]:
            prefix_sum = torch.randn(
                num_tokens, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
            )
            bank = torch.randn(
                num_tokens, bank_rows, HIDDEN_SIZE,
                device="cuda", dtype=torch.bfloat16,
            )
            scores = torch.empty(num_tokens, 16, device="cuda", dtype=torch.float32)
            _launch(prefix_sum, bank, cw, scores, nvb)
            torch.cuda.synchronize()
            ms = _bench_burst(lambda: _launch(prefix_sum, bank, cw, scores, nvb))
            hbm = num_tokens * (nvb + 1) * HIDDEN_SIZE * 2 + HIDDEN_SIZE * 4
            print(
                f"NW={os.getenv('NW','8')} BH={BLOCK_H} T={num_tokens} "
                f"rows={bank_rows} nvb={nvb}  {ms:.4f} ms  {hbm/(ms*1e6):8.1f} GB/s"
            )
        sys.exit(0)
    ok = main()
    sys.exit(0 if ok else 1)
