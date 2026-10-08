# SPDX-License-Identifier: Apache-2.0
"""_router_triton_kernel 精度 + 带宽单元测试 (MetaX C600-U / MACA).

算子: SGLang fused MoE router (triton_sglang_moe_fused_gate.py)。
K3 路由配置: N=896 experts, top-16, sigmoid scoring, 未分组 (N_GROUP=1),
无 fused shared experts, 无 softcapping。
每 token 语义:
    activated = sigmoid(scores)          # bias-free, 作为权重
    biased    = activated + bias         # 仅排序键 (NaN -> -1e30)
    ids       = biased 的迭代 top-K, 平局取最小 expert id
    weights   = activated[ids]; renormalize -> /sum; apply_scale -> *factor

Roofline (T=16384): scores 读 = M*N*4 = 58.7MB 占 HBM 流量 96.6%;
输出 weights/ids 写 = M*K*4*2 = 2.1MB。内存下界 @1.6T ≈ 38µs。
计算 = sigmoid + K=16 次串行 argmax (每次 ~3 个 1024 宽规约)。
属于 top-k 串行规约延迟 bound, 极限情况下逼近 scores 读的内存 bound。

运行 (先 source env_local.sh 并挑空闲卡):
    cd /home/yiyu/mcoplib/mcoplib_dev/mcoplib
    source env_local.sh
    mx-smi                              # 找 0% / Available 的卡
    export CUDA_VISIBLE_DEVICES=16
    /opt/conda/bin/python unit_test/test_triton_router_triton_kernel.py

说明:
  * kernel 顶层不再强依赖 sglang (radix 快路径与 PDL 已带 guard 回退到纯
    triton kernel); 若环境无 sglang, 本测试仍可独立运行。
  * 精度: 权重用余弦相似度 (cos_sim, 阈值 0.99999); ids 必须逐元素精确匹配
    (路由是离散决策, 单个 expert 不同即错), 一并计入 OK。
  * 带宽: 真实 HBM 流量 = scores 读 (M*N*4) + weights 写 (M*K*4)
    + indices 写 (M*K*4) + bias 读 (N*4)。burst 计时撑住 DVFS。
  * 输出格式对齐 test_op_moe_scatter_dynamic_quant.py。
"""

import os
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mcoplib.triton_sglang_moe_fused_gate import moe_fused_gate  # noqa: E402

# --------------------------------------------------------------------------- #
# 测试配置 (与需求一致)。
# --------------------------------------------------------------------------- #
N = 896                 # num_experts (K3 router GEMM width)
K = 16                  # topk (routed; num_fused_shared_experts = 0)
TOKEN_COUNTS = (2048, 4096, 8192, 16384)
ROUTED_SCALING_FACTOR = 2.5

COS_THRESHOLD = 0.99999
SINGLE_DIE_TARGET = 1600.0   # GB/s (需求目标 1.6T; 85% = 1360 可接受)
DUAL_DIE_DATASHEET = 3200.0


def _launch(scores, bias, renormalize=True, apply_scale=False):
    return moe_fused_gate(
        scores,
        bias,
        K,
        scoring_func="sigmoid",
        num_fused_shared_experts=0,
        renormalize=renormalize,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        apply_routed_scaling_factor_on_output=apply_scale,
        num_expert_group=1,
        topk_group=1,
    )


def _torch_reference(scores, bias, renormalize=True, apply_scale=False):
    """fp32 参考: 迭代 argmax, 平局取最小 expert id (torch.argmax 取首个 max, 一致)。"""
    activated = torch.sigmoid(scores.float())
    biased = activated + bias.float()[None, :]
    biased = torch.nan_to_num(biased, nan=-1e30)

    m = scores.shape[0]
    weights = torch.empty(m, K, dtype=torch.float32, device=scores.device)
    ids = torch.empty(m, K, dtype=torch.int32, device=scores.device)
    cur = biased.clone()
    rows = torch.arange(m, device=scores.device)
    for j in range(K):
        win = cur.argmax(dim=1)
        ids[:, j] = win.to(torch.int32)
        weights[:, j] = activated[rows, win]
        cur[rows, win] = -float("inf")

    if renormalize:
        s = weights.sum(dim=1, keepdim=True)
        weights = weights / torch.where(s > 0, s, torch.ones_like(s))
    if apply_scale:
        weights = weights * ROUTED_SCALING_FACTOR
    return weights, ids


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
        if n % 32 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    reps = max(32, n)
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
        f"device={torch.cuda.get_device_name(dev)}  device_count={torch.cuda.device_count()}"
    )
    print(
        f"config: num_experts={N} topk={K} scoring=sigmoid ungrouped "
        f"scaling={ROUTED_SCALING_FACTOR} dtype=fp32->fp32(w)/int32(id)"
    )
    print("=" * 92)

    torch.manual_seed(0)
    all_pass = True
    peak_gbps = 0.0
    peak_shape = None

    for num_tokens in TOKEN_COUNTS:
        for apply_scale in (False, True):
            scores = torch.randn(num_tokens, N, device="cuda", dtype=torch.float32)
            bias = torch.randn(N, device="cuda", dtype=torch.float32)

            weights, ids = _launch(scores, bias, apply_scale=apply_scale)
            torch.cuda.synchronize()

            exp_w, exp_i = _torch_reference(scores, bias, apply_scale=apply_scale)
            cos = _cos_sim(weights, exp_w)
            ids_ok = bool(torch.equal(ids, exp_i))
            has_nan = bool(torch.isnan(weights).any().item())
            ok = (cos >= COS_THRESHOLD) and ids_ok and not has_nan
            all_pass = all_pass and ok

            ms = _bench_burst(lambda: _launch(scores, bias, apply_scale=apply_scale))
            hbm_bytes = num_tokens * (N * 4 + K * 4 * 2) + N * 4
            gbps = hbm_bytes / (ms * 1e6)

            if gbps > peak_gbps:
                peak_gbps = gbps
                peak_shape = (num_tokens, apply_scale)

            tag = "OK  " if ok else ("BADid" if not ids_ok else "BAD ")
            print(
                f"[router] T={num_tokens:<6} scale={int(apply_scale)}  "
                f"cos_sim={cos:.6f} {tag}  {ms:8.4f} ms  {gbps:8.1f} GB/s"
            )

    print("=" * 92)
    print(
        f"Accuracy: {'ALL PASS' if all_pass else 'FAIL'} "
        f"(threshold cos_sim >= {COS_THRESHOLD}, ids exact)"
    )
    reached = "reached" if peak_gbps >= SINGLE_DIE_TARGET else "NOT reached"
    print(
        f"Peak bandwidth: {peak_gbps:.1f} GB/s @ T={peak_shape[0]} scale={int(peak_shape[1])}  "
        f"(single-die target {SINGLE_DIE_TARGET:.0f} -> {reached}; "
        f"dual-die datasheet {DUAL_DIE_DATASHEET:.0f})"
    )
    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
