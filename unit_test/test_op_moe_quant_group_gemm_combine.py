"""Unit test + perf benchmark for the triton `moe_quant_group_gemm_combine` kernel.

This version is FULLY STANDALONE — it depends on neither `xpu_perf` nor
`mcoplib`. Both helpers it needs are local modules in this directory:
  * `moe_tokens_info.get_moe_tokens_info`  (drop-in for xpu_perf's utility)
  * `moe_gemm_kernel.{group_gemm_combine, DEFAULT_CONFIG}`  (the optimized kernel)

Op semantics (MoE down-projection group-GEMM fused with expert-combine):
  For each routed (token -> local expert) pair (a "scatter row"):
      out_row = (int8_act @ int8_weight^T)               # int8 group GEMM, K=hidden_size
                * per_token_scale * expert_channel_scale  # dequant
                * router_weight                           # combine weight
  Rows belonging to the same original token are summed (index_add) into
  `convergent_tokens[token_id]` (fp32 accumulator, atomic_add in the kernel).

Roofline for the target shape (M=10240, N=4096, K=1536, INT8):
  FLOPs   = 2*M*K*N ~= 1.29e11  -> at 480 TOPS  = 268 us   (compute floor)
  arithmetic intensity ~= 1280 ops/byte  => strongly COMPUTE-BOUND.
So the primary metric is INT8 TOPS; target = 85% * 480 = 408 TOPS. GB/s printed too.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Standalone helpers (no xpu_perf / mcoplib import needed).
from mcoplib.triton_moe_quant_group_gemm_combine import group_gemm_combine, DEFAULT_CONFIG, get_moe_tokens_info 


# ---- fixed op config (from the workload json) -----------------------------
NUM_EXPERTS = 128
TOPK = 8
EP_SIZE = 8
HIDDEN = 1536          # K
NEW_HIDDEN = 4096      # N
COS_SIM_THRESHOLD = 0.999          # int8 -> relaxed 0.999
PEAK_INT8_TOPS = 480.0             # C600-U single-die INT8 peak
TARGET_TOPS = 0.85 * PEAK_INT8_TOPS  # 408 TOPS

# num_tokens sweep (10240 is the required config; others build the improvement table)
NUM_TOKENS_LIST = [1024, 2048, 4096, 8192, 10240]


def cos_sim(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def make_inputs(num_tokens, device):
    (_, _, num_experts_per_rank, _, _, _, _, dispatch_tokens, _, _, _,
     scatter_token_id, scatter_token_weight, cnt, off) = get_moe_tokens_info(
        num_tokens, NUM_EXPERTS, TOPK, ep_size=EP_SIZE, ep_rank=0)

    g = torch.Generator(device=device).manual_seed(1234)
    # random int8 activations / weights (full int8 range)
    scatter_tokens = torch.randint(-127, 128, (dispatch_tokens, HIDDEN),
                                   dtype=torch.int8, device=device, generator=g)
    experts_weight = torch.randint(-127, 128, (num_experts_per_rank, NEW_HIDDEN, HIDDEN),
                                   dtype=torch.int8, device=device, generator=g)
    # realistic small positive dequant scales
    per_token_scale = (torch.rand(dispatch_tokens, device=device, generator=g) * 0.02 + 0.001).to(torch.float32)
    experts_scale = (torch.rand(num_experts_per_rank, NEW_HIDDEN, device=device, generator=g) * 0.02 + 0.001).to(torch.float32)

    experts_token_count = torch.tensor(cnt, dtype=torch.int32, device=device)
    experts_token_offset = torch.tensor(off, dtype=torch.int32, device=device)
    scatter_token_id = torch.tensor(scatter_token_id, dtype=torch.int32, device=device)
    scatter_token_weight = torch.tensor(scatter_token_weight, dtype=torch.float32, device=device)

    convergent = torch.zeros((num_tokens, NEW_HIDDEN), dtype=torch.float32, device=device)

    return dict(
        scatter_tokens=scatter_tokens, per_token_scale=per_token_scale,
        experts_weight=experts_weight, experts_scale=experts_scale,
        experts_token_count=experts_token_count, experts_token_offset=experts_token_offset,
        scatter_token_id=scatter_token_id, scatter_token_weight=scatter_token_weight,
        convergent=convergent, num_experts_per_rank=num_experts_per_rank,
        dispatch_tokens=dispatch_tokens, cnt=cnt, off=off,
        max_expert_tokens=max(cnt),
    )


def torch_reference(t, num_tokens, device):
    """Exact int8-GEMM (fp64 accumulation) + dequant + weighted combine."""
    ref = torch.zeros((num_tokens, NEW_HIDDEN), dtype=torch.float32, device=device)
    scatter_tokens = t["scatter_tokens"]
    experts_weight = t["experts_weight"]
    for e in range(t["num_experts_per_rank"]):
        s = t["off"][e]
        n = t["cnt"][e]
        if n == 0:
            continue
        a = scatter_tokens[s:s + n].to(torch.float64)             # [n, K]
        w = experts_weight[e].to(torch.float64)                    # [N, K]
        acc = (a @ w.t()).to(torch.float32)                        # [n, N] exact int gemm (fp64 accum)
        acc *= t["per_token_scale"][s:s + n][:, None]
        acc *= t["experts_scale"][e][None, :]
        acc *= t["scatter_token_weight"][s:s + n][:, None]
        ref.index_add_(0, t["scatter_token_id"][s:s + n], acc)
    return ref


def run_kernel(t, config=None):
    t["convergent"].zero_()
    return group_gemm_combine(
        t["scatter_tokens"], t["per_token_scale"], t["experts_weight"], t["experts_scale"],
        t["experts_token_count"], t["experts_token_offset"], t["scatter_token_id"],
        t["scatter_token_weight"], t["convergent"], HIDDEN, NEW_HIDDEN,
        t["num_experts_per_rank"], t["max_expert_tokens"], config=config,
    )


def bench(t, config=None, warm_s=0.6, time_s=0.6):
    # warmup (also holds DVFS boost)
    torch.cuda.synchronize()
    t0 = time.time()
    n = 0
    while time.time() - t0 < warm_s:
        run_kernel(t, config)
        n += 1
        if n % 16 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    reps = max(16, n)
    best_ms = float("inf")
    t0 = time.time()
    while time.time() - t0 < time_s:
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(reps):
            run_kernel(t, config)
        e.record(); e.synchronize()
        best_ms = min(best_ms, s.elapsed_time(e) / reps)
    return best_ms


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES','')}'  device_count={torch.cuda.device_count()}")
    print(f"config: num_experts={NUM_EXPERTS} topk={TOPK} ep_size={EP_SIZE} "
          f"experts_per_rank={NUM_EXPERTS//EP_SIZE} hidden={HIDDEN} new_hidden={NEW_HIDDEN} "
          f"dtype=int8xint8->bf16  (COMPUTE-BOUND; peak INT8={PEAK_INT8_TOPS:.0f}T, target 85%={TARGET_TOPS:.0f}T)")
    print(f"launch cfg: {DEFAULT_CONFIG}")
    print("=" * 108)

    device = "cuda"
    peak_tops = 0.0
    all_pass = True
    rows = []
    for T in NUM_TOKENS_LIST:
        t = make_inputs(T, device)
        # ---- accuracy ----
        out = run_kernel(t).clone()
        ref = torch_reference(t, T, device)
        cs = cos_sim(out, ref)
        ok = cs >= COS_SIM_THRESHOLD
        all_pass &= ok
        # ---- perf ----
        ms = bench(t)
        flops = 2.0 * t["dispatch_tokens"] * HIDDEN * NEW_HIDDEN
        tops = flops / (ms * 1e-3) / 1e12
        # effective HBM bytes: read act + read weight + write fp32 accumulator (atomic RMW ~2x)
        rbytes = t["dispatch_tokens"] * HIDDEN + t["num_experts_per_rank"] * NEW_HIDDEN * HIDDEN
        wbytes = T * NEW_HIDDEN * 4
        gbps = (rbytes + wbytes) / (ms * 1e-3) / 1e9
        peak_tops = max(peak_tops, tops)
        rows.append((T, t["dispatch_tokens"], cs, ok, ms, tops, gbps))
        print(f"[gemm_combine] T={T:6d} routes={t['dispatch_tokens']:7d}  cos_sim={cs:.6f} "
              f"{'OK ' if ok else 'BAD'}  {ms:8.4f} ms  {tops:7.1f} TOPS  {gbps:7.1f} GB/s")

    print("=" * 108)
    print(f"Accuracy: {'ALL PASS' if all_pass else 'FAIL'} (threshold cos_sim >= {COS_SIM_THRESHOLD})")
    reached = peak_tops >= TARGET_TOPS
    print(f"Peak INT8 compute: {peak_tops:.1f} TOPS  "
          f"(target {TARGET_TOPS:.0f}T = 85%% of {PEAK_INT8_TOPS:.0f}T -> "
          f"{'REACHED' if reached else 'NOT reached'})")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
