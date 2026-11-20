"""Unit test + bandwidth benchmark for mcoplib.op.moe_scatter_dynamic_quant on MetaX C600-U.

Op (12-arg, in-place):
  moe_scatter_dynamic_quant(hidden_status, selected_experts, moe_weights, smooth_scale,
                            scatter_tokens, scatter_per_token_scale, scatter_tokens_offset,
                            experts_token_count, experts_token_start,
                            experts_per_rank, shared_experts_per_rank, shared_tokens_per_sp)

Semantics (fast path, hidden=4096, topk=8, experts_per_rank=16, shared=0):
  For each routed (token -> local expert e) pair one OUTPUT ROW is produced, rows grouped
  by expert. With v = hidden[token].float() * smooth_scale[e] (per channel), bm = max_c|v_c|:
    scatter_per_token_scale[out] = bm * |weight| / QMAX
    scatter_tokens[out]          = round( v * (QMAX/bm) * sign(weight) )
  => dequant  scatter_tokens[out] * scale[out]  ~=  hidden[token] * smooth[e] * weight.

Config (from request): dtype=bfloat16, dst_dtype in {int8, float8_e4m3fn},
  ep_size=8, num_experts=128, topk=8, hidden_size=4096, all 38 num_tokens.
Accuracy: cosine similarity, int8 >= 0.999, fp8 >= 0.999. Bandwidth printed per dtype+shape.

Run:
  export CUDA_VISIBLE_DEVICES=10,11
  python unit_test/test_op_moe_scatter_dynamic_quant.py
"""

import os
import torch

import mcoplib.op as ops

DEVICE = "cuda"

# ---- fixed op config ----
NUM_EXPERTS = 128
TOPK = 8
EP_SIZE = 8
EXPERTS_PER_RANK = NUM_EXPERTS // EP_SIZE          # 16 local experts / rank
HIDDEN = 4096
SHARED_EXPERTS_PER_RANK = 0
SHARED_TOKENS_PER_SP = 0

TARGET_GBPS = 1300.0                                # single-die gate
DUAL_DIE_GBPS = 3480.0                              # dual-die datasheet figure

# dst dtype -> (torch dtype, QMAX, out elem bytes, cos_sim threshold)
FLOAT8 = torch.float8_e4m3fn
DST_DTYPES = {
    "int8": (torch.int8, 127.0, 1, 0.999),
    "fp8":  (FLOAT8,     448.0, 1, 0.999),
}

NUM_TOKENS_CASES = [
    16, 48, 64, 128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792,
    2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 6144, 8192, 10240,
    12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624, 28672, 30720,
    32768, 65536,
]


def make_inputs(num_tokens, dst_torch_dtype, seed=1234):
    """Build op inputs. Each token routes to TOPK *distinct* local experts in [0,16),
    so every route lands on-rank -> total routes = num_tokens * TOPK, all output rows
    valid (exercises the full memory path)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    hidden = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=DEVICE, generator=g)

    rand = torch.rand(num_tokens, EXPERTS_PER_RANK, device=DEVICE, generator=g)
    sel = rand.argsort(dim=1)[:, :TOPK].to(torch.int32)                  # [T, 8] distinct
    selected_experts = sel.contiguous()

    moe_weights = torch.randn(num_tokens, TOPK, dtype=torch.float32, device=DEVICE, generator=g)

    smooth_scale = (torch.rand(EXPERTS_PER_RANK, HIDDEN, dtype=torch.float32,
                               device=DEVICE, generator=g) + 0.5).contiguous()

    capacity = num_tokens * TOPK
    scatter_tokens = torch.zeros(capacity, HIDDEN, dtype=dst_torch_dtype, device=DEVICE)
    scatter_per_token_scale = torch.zeros(capacity, dtype=torch.float32, device=DEVICE)
    scatter_tokens_offset = torch.full((capacity,), -1, dtype=torch.int32, device=DEVICE)
    experts_token_count = torch.zeros(EXPERTS_PER_RANK + 1, dtype=torch.int32, device=DEVICE)
    experts_token_start = torch.zeros(EXPERTS_PER_RANK + 1, dtype=torch.int32, device=DEVICE)

    return dict(
        hidden=hidden, selected_experts=selected_experts, moe_weights=moe_weights,
        smooth_scale=smooth_scale, scatter_tokens=scatter_tokens,
        scatter_per_token_scale=scatter_per_token_scale,
        scatter_tokens_offset=scatter_tokens_offset,
        experts_token_count=experts_token_count,
        experts_token_start=experts_token_start, capacity=capacity)


def run_op(t):
    ops.moe_scatter_dynamic_quant(
        t["hidden"], t["selected_experts"], t["moe_weights"], t["smooth_scale"],
        t["scatter_tokens"], t["scatter_per_token_scale"], t["scatter_tokens_offset"],
        t["experts_token_count"], t["experts_token_start"],
        EXPERTS_PER_RANK, SHARED_EXPERTS_PER_RANK, SHARED_TOKENS_PER_SP)


def streaming_cosine(t, chunk_rows=2048):
    """Chunked cosine similarity over valid output rows, fp64 accumulation, to avoid
    materializing a full [V, HIDDEN] fp32 matrix (OOM at large T).

    deq[out] = scatter_tokens[out] * scale[out]
    ref[out] = hidden[token] * smooth[expert] * weight  (token/expert from op outputs)
    """
    count = t["experts_token_count"][:EXPERTS_PER_RANK].long()
    total_valid = int(count.sum().item())
    start = t["experts_token_start"].long()
    offset = t["scatter_tokens_offset"].long()
    sel = t["selected_experts"].long()
    starts_hi = start[1:EXPERTS_PER_RANK + 1].contiguous()

    dot = torch.zeros((), dtype=torch.float64, device=DEVICE)
    na = torch.zeros((), dtype=torch.float64, device=DEVICE)
    nb = torch.zeros((), dtype=torch.float64, device=DEVICE)

    for lo in range(0, total_valid, chunk_rows):
        hi = min(lo + chunk_rows, total_valid)
        out_ids = torch.arange(lo, hi, device=DEVICE)
        out_expert = torch.searchsorted(starts_hi, out_ids, right=True).clamp_(0, EXPERTS_PER_RANK - 1)
        out_token = offset[lo:hi]
        match = (sel[out_token] == out_expert[:, None])
        slot = match.float().argmax(dim=1)
        weight = t["moe_weights"][out_token, slot]
        ref = (t["hidden"][out_token].float()
               * t["smooth_scale"][out_expert].float()
               * weight[:, None])
        deq = (t["scatter_tokens"][lo:hi].float()
               * t["scatter_per_token_scale"][lo:hi, None])
        dot += (deq * ref).double().sum()
        na += (deq * deq).double().sum()
        nb += (ref * ref).double().sum()

    denom = (na.sqrt() * nb.sqrt()).clamp_min(1e-30)
    return (dot / denom).item(), total_valid


def effective_bytes(routes, out_elem_bytes):
    # hidden read (bf16) + quant write, per route; per-row scale write (4B) counted
    return routes * (HIDDEN * 2 + HIDDEN * out_elem_bytes) + routes * 4


def bench_case(num_tokens, dst_name, warmup=10, rep=50):
    dst_dtype, qmax, out_bytes, thr = DST_DTYPES[dst_name]
    t = make_inputs(num_tokens, dst_dtype)

    # ---- accuracy ----
    run_op(t)
    torch.cuda.synchronize()
    sim, routes = streaming_cosine(t)

    # ---- bandwidth (back-to-back burst within one sync, sustains DVFS) ----
    for _ in range(warmup):
        run_op(t)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        run_op(t)
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    median_ms = times_ms[len(times_ms) // 2]

    gbps = effective_bytes(routes, out_bytes) / (median_ms * 1e-3) / 1e9
    ok = "OK" if sim >= thr else "FAIL"
    print(f"[{dst_name:>4}] T={num_tokens:6d} routes={routes:8d}  "
          f"cos_sim={sim:.6f} {ok:4s}  {median_ms:8.4f} ms  {gbps:8.1f} GB/s")
    return sim >= thr, gbps


def main():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES={vis!r}  device_count={torch.cuda.device_count()}")
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"config: num_experts={NUM_EXPERTS} topk={TOPK} ep_size={EP_SIZE} "
          f"experts_per_rank={EXPERTS_PER_RANK} hidden={HIDDEN} dtype=bf16")
    print("=" * 100)

    overall_pass = True
    peaks = {}
    for dst_name in DST_DTYPES:
        thr = DST_DTYPES[dst_name][3]
        print(f"---- dst_dtype = {dst_name} (cos_sim threshold {thr}) "
              + "-" * 40)
        peak, dpass = 0.0, True
        for T in NUM_TOKENS_CASES:
            ok, gbps = bench_case(T, dst_name)
            peak = max(peak, gbps)
            dpass &= ok
        peaks[dst_name] = (peak, dpass)
        print(f"     [{dst_name}] peak = {peak:.1f} GB/s   "
              f"accuracy {'ALL PASS' if dpass else 'FAILURES PRESENT'}")
        overall_pass &= dpass

    print("=" * 100)
    for dst_name, (peak, dpass) in peaks.items():
        print(f"{dst_name:>4}: peak {peak:8.1f} GB/s  "
              f"({'REACHED' if peak >= TARGET_GBPS else 'below'} single-die target "
              f"{TARGET_GBPS:.0f})  accuracy {'PASS' if dpass else 'FAIL'}")
    print(f"dual-die datasheet reference: {DUAL_DIE_GBPS:.0f} GB/s")
    assert overall_pass, "accuracy check failed"


if __name__ == "__main__":
    main()
