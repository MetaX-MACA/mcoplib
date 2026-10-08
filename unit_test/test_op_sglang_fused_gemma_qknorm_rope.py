"""Unit test + bandwidth benchmark for torch.ops.sgl_kernel.fused_gemma_qknorm_rope
on MetaX C600-U (AOT port of sglang_jit/fused_gemma_qknorm_rope.cuh).

Op (17-arg, in-place over qkv):
  fused_gemma_qknorm_rope(qkv, w0, w1, w2, w3, cos_sin_cache, positions,
                          off0, cnt0, off1, cnt1, off2, cnt2, off3, cnt3,
                          num_groups, eps)

Semantics (per token, per head; head_dim=128, rope_dim=64):
  * GemmaRMSNorm over the 128 dims (fp32 accum):  n = x * rsqrt(mean(x^2)+eps) * (1 + weight)
  * partial NeoX RoPE on the first 64 dims, paired (j, j+32) for j in [0,32):
        cos = cos_sin_cache[pos, 0:32],  sin = cos_sin_cache[pos, 32:64]
        o[j]    = n[j]*cos[j] - n[j+32]*sin[j]
        o[j+32] = n[j+32]*cos[j] + n[j]*sin[j]
    dims [64,128) pass through norm-only.
  A "group" is a contiguous head run sharing one norm weight (all roped). Up to 4
  groups per launch; heads outside every group (V / index-V) are left untouched.

Config (from request): dtype=bfloat16, cos_sin_cache=fp32, positions int64/int32.
Production shape (from real SGLang trace): qkv [chunked_size, 2304] (18 heads),
  groups (Q:0..15)+(K:16), V=head17 untouched, cos_sin_cache [~1.05e6, 64], int64
  positions. chunked_size is exercised over [2048, 4096, 8192, 16384, 32768].
Accuracy: cosine similarity over the touched region, threshold >= 0.9999 (bf16, no quant).
Bandwidth printed per shape+config (effective = qkv read + qkv write on touched heads).

Run:
  export CUDA_VISIBLE_DEVICES=<free>   # auto-selected below if unset
  python unit_test/test_op_sglang_fused_gemma_qknorm_rope.py
"""

import os
import subprocess


def _pick_free_gpus(max_gpus=2, mem_frac_free=0.90):
    """Select the freest GPUs via mx-smi before torch initializes CUDA.
    Honors a pre-set CUDA_VISIBLE_DEVICES; otherwise picks up to max_gpus with the
    most free memory."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return os.environ["CUDA_VISIBLE_DEVICES"]
    try:
        out = subprocess.check_output(["mx-smi"], stderr=subprocess.DEVNULL, text=True, timeout=30)
    except Exception:
        return None
    # Parse "used / total" MiB pairs per GPU from mx-smi tabular output.
    import re
    gpus = []  # (gpu_id, free_frac)
    gid = None
    for line in out.splitlines():
        m = re.search(r"\bGPU\s+(\d+)\b", line)
        if m:
            gid = int(m.group(1))
        # mem pattern like "1234/65536 MiB" or "1234 MiB / 65536 MiB"
        mm = re.search(r"(\d+)\s*(?:MiB|MB)?\s*/\s*(\d+)\s*(?:MiB|MB)", line)
        if mm and gid is not None:
            used, total = int(mm.group(1)), int(mm.group(2))
            if total > 0:
                gpus.append((gid, 1.0 - used / total))
                gid = None
    if not gpus:
        return None
    # de-dup by gpu id keeping max free
    best = {}
    for g, f in gpus:
        best[g] = max(best.get(g, 0.0), f)
    free = sorted([(g, f) for g, f in best.items() if f >= mem_frac_free], key=lambda x: -x[1])
    if not free:
        free = sorted(best.items(), key=lambda x: -x[1])
    chosen = ",".join(str(g) for g, _ in free[:max_gpus])
    if chosen:
        os.environ["CUDA_VISIBLE_DEVICES"] = chosen
    return chosen


_pick_free_gpus()

import torch
import mcoplib.sgl_kernel  # noqa: F401  registers torch.ops.sgl_kernel.*

DEVICE = "cuda"
HEAD_DIM = 128
ROPE_DIM = 64
HALF = ROPE_DIM // 2  # 32
EPS = 1e-6
MAX_GROUPS = 4
COS_THRESHOLD = 0.9999

TARGET_GBPS = 1360.0      # 85% of single-die 1.6 TB/s
SINGLE_DIE_GBPS = 1600.0
DUAL_DIE_GBPS = 3200.0

# ---- head configurations exercising all group paths + generalization ----
# row_heads = physical heads in the qkv row (incl. untouched V region);
# groups = list of (head_offset, head_count) sharing one norm weight, all roped.
CONFIGS = [
    # ---- production MiniMax-M3 layout (from real SGLang trace) ----
    # qkv row = 2304 = 18 heads; groups = (Q:0..15) + (K:16), head 17 = V (untouched).
    # Real trace shows off0=0 cnt0=16 off1=16 cnt1=1 num_groups=2 (the "-16" in the
    # dump is a signed-display artifact of the host-constant 16). cos_sin_cache rows
    # are large (>1e6) in production; positions int64. This is THE shipping shape.
    dict(name="prod_q16k1v1", row_heads=18, groups=[(0, 16), (16, 1)]),       # production
    # ---- generalization configs (all group paths, small + large rows) ----
    dict(name="q8k1v1",      row_heads=10, groups=[(0, 8), (8, 1)]),          # 2 groups (nq=8,nk=1,nv=1)
    dict(name="q16k2v2",     row_heads=20, groups=[(0, 16), (16, 2)]),        # 2 groups
    dict(name="q32k4v4",     row_heads=40, groups=[(0, 32), (32, 4)]),        # 2 groups, large row
    dict(name="q8_1grp",     row_heads=9,  groups=[(0, 8)]),                  # 1 group (+ untouched)
    dict(name="q8k1i8i1_4g", row_heads=18, groups=[(0, 8), (8, 1), (9, 8), (17, 1)]),  # 4 groups
]

# Production requires exercising chunked_size (== qkv/positions shape[0]) over the
# generalization range [2048, 4096, 8192, 16384, 32768]. We keep a small-shape
# sweep too so both extremes (latency-bound small + bandwidth-bound large) are covered.
CHUNKED_SIZE_CASES = [2048, 4096, 8192, 16384, 32768]
SMALL_TOKENS_CASES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
NUM_TOKENS_CASES = SMALL_TOKENS_CASES + CHUNKED_SIZE_CASES

POS_DTYPES = {"i64": torch.int64, "i32": torch.int32}


def make_cos_sin_cache(max_pos, base=10000.0):
    """Standard NeoX cos_sin_cache for rotary_dim=64 (32 freqs).
    Layout [max_pos, 64]: [:, 0:32]=cos, [:, 32:64]=sin."""
    inv_freq = 1.0 / (base ** (torch.arange(0, HALF, dtype=torch.float64) / HALF))  # [32]
    pos = torch.arange(max_pos, dtype=torch.float64)                                # [P]
    angle = torch.outer(pos, inv_freq)                                             # [P,32]
    cache = torch.empty(max_pos, ROPE_DIM, dtype=torch.float32)
    cache[:, :HALF] = torch.cos(angle).float()
    cache[:, HALF:] = torch.sin(angle).float()
    return cache.contiguous()


def torch_reference(qkv, group_weights, groups, cos_sin_cache, positions):
    """Reference GemmaRMSNorm + partial NeoX RoPE. Returns a new [T, row_heads*128]
    tensor; only the grouped heads are modified, V heads pass through."""
    T = qkv.shape[0]
    row_heads = qkv.shape[1] // HEAD_DIM
    x = qkv.float().view(T, row_heads, HEAD_DIM)
    out = x.clone()
    pos = positions.long()
    cos = cos_sin_cache[pos, :HALF].float()      # [T,32]
    sin = cos_sin_cache[pos, HALF:ROPE_DIM].float()  # [T,32]
    for (off, cnt), w in zip(groups, group_weights):
        if cnt == 0:
            continue
        seg = x[:, off:off + cnt, :]                            # [T,cnt,128]
        var = seg.pow(2).mean(dim=-1, keepdim=True)             # [T,cnt,1]
        inv_rms = torch.rsqrt(var + EPS)
        n = seg * inv_rms * (1.0 + w.float().view(1, 1, HEAD_DIM))  # gemma (1+w)
        n0 = n[..., :HALF]                                     # [T,cnt,32]
        n1 = n[..., HALF:ROPE_DIM]
        c = cos[:, None, :]                                     # [T,1,32]
        s = sin[:, None, :]
        o = n.clone()
        o[..., :HALF] = n0 * c - n1 * s
        o[..., HALF:ROPE_DIM] = n1 * c + n0 * s
        out[:, off:off + cnt, :] = o
    return out.view(T, row_heads * HEAD_DIM)


def cosine(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return (torch.dot(a, b).double()
            / (a.norm().double() * b.norm().double() + 1e-30)).item()


def build_groups_args(group_weights, groups):
    """Pad to 4 slots (dummy = group-0 weight, cnt=0) and flatten to op scalars."""
    weights = list(group_weights)
    offs = [g[0] for g in groups]
    cnts = [g[1] for g in groups]
    num_groups = len(groups)
    while len(weights) < MAX_GROUPS:
        weights.append(weights[0])
        offs.append(0)
        cnts.append(0)
    return weights, offs, cnts, num_groups


def bench_case(cfg, num_tokens, pos_name, warmup=15, rep=60):
    g = torch.Generator(device=DEVICE).manual_seed(1234 + num_tokens)
    row_heads = cfg["row_heads"]
    groups = cfg["groups"]
    pos_dtype = POS_DTYPES[pos_name]

    qkv = torch.randn(num_tokens, row_heads * HEAD_DIM, dtype=torch.bfloat16,
                      device=DEVICE, generator=g)
    group_weights = [
        (0.1 * torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=DEVICE, generator=g))
        for _ in groups
    ]
    # Production cos_sin_cache has ~1e6 rows; stress that large-cache indexing path
    # for the shipping config, keep it small (cheap) for the generalization configs.
    max_pos = 1048832 if cfg["name"] == "prod_q16k1v1" else 4096
    cos_sin_cache = make_cos_sin_cache(max_pos).to(DEVICE)
    positions = torch.randint(0, max_pos, (num_tokens,), device=DEVICE, generator=g).to(pos_dtype)

    weights, offs, cnts, num_groups = build_groups_args(group_weights, groups)

    # reference from a pristine copy
    ref = torch_reference(qkv, group_weights, groups, cos_sin_cache, positions)

    # in-place op on a working copy
    work = qkv.clone()
    torch.ops.sgl_kernel.fused_gemma_qknorm_rope(
        work, weights[0], weights[1], weights[2], weights[3],
        cos_sin_cache, positions,
        offs[0], cnts[0], offs[1], cnts[1], offs[2], cnts[2], offs[3], cnts[3],
        num_groups, EPS)
    torch.cuda.synchronize()

    # ---- accuracy over touched heads only (strict; V region equal in both) ----
    touched = torch.zeros(row_heads, dtype=torch.bool)
    for off, cnt in groups:
        touched[off:off + cnt] = True
    tmask = touched.repeat_interleave(HEAD_DIM).to(DEVICE)
    sim = cosine(work[:, tmask], ref[:, tmask])

    # verify untouched V heads are bit-identical (no stray writes)
    if (~touched).any():
        vmask = (~touched).repeat_interleave(HEAD_DIM).to(DEVICE)
        v_ok = torch.equal(work[:, vmask], qkv[:, vmask])
    else:
        v_ok = True

    # ---- bandwidth ----
    for _ in range(warmup):
        work.copy_(qkv)
        torch.ops.sgl_kernel.fused_gemma_qknorm_rope(
            work, weights[0], weights[1], weights[2], weights[3],
            cos_sin_cache, positions,
            offs[0], cnts[0], offs[1], cnts[1], offs[2], cnts[2], offs[3], cnts[3],
            num_groups, EPS)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        torch.ops.sgl_kernel.fused_gemma_qknorm_rope(
            work, weights[0], weights[1], weights[2], weights[3],
            cos_sin_cache, positions,
            offs[0], cnts[0], offs[1], cnts[1], offs[2], cnts[2], offs[3], cnts[3],
            num_groups, EPS)
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    best_ms = times_ms[0]  # best-of-N (memory-bound, quiet GPU)

    touched_heads = int(touched.sum().item())
    routes = num_tokens * touched_heads
    eff_bytes = routes * HEAD_DIM * 2 * 2  # bf16 read + bf16 write on touched region
    gbps = eff_bytes / (best_ms * 1e-3) / 1e9

    ok = (sim >= COS_THRESHOLD) and v_ok
    flag = "OK" if ok else "FAIL"
    vtag = "" if v_ok else " V-CORRUPT"
    print(f"[{pos_name}] {cfg['name']:>12s} T={num_tokens:6d} routes={routes:8d}  "
          f"cos={sim:.6f} {flag:4s}{vtag}  {best_ms:8.4f} ms  {gbps:8.1f} GB/s")
    return ok, gbps


def main():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES={vis!r}  device_count={torch.cuda.device_count()}")
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"device={torch.cuda.get_device_properties(0).name}  "
          f"head_dim={HEAD_DIM} rope_dim={ROPE_DIM} eps={EPS} cos_thr={COS_THRESHOLD}")
    print("=" * 104)

    overall_pass = True
    peak = 0.0
    for pos_name in POS_DTYPES:
        for cfg in CONFIGS:
            print(f"---- pos={pos_name}  config={cfg['name']}  "
                  f"row_heads={cfg['row_heads']} groups={cfg['groups']} " + "-" * 20)
            for T in NUM_TOKENS_CASES:
                ok, gbps = bench_case(cfg, T, pos_name)
                overall_pass &= ok
                peak = max(peak, gbps)
    print("=" * 104)
    print(f"peak {peak:8.1f} GB/s  "
          f"({'REACHED' if peak >= TARGET_GBPS else 'below'} target {TARGET_GBPS:.0f} "
          f"= 85% of single-die {SINGLE_DIE_GBPS:.0f})  "
          f"accuracy {'PASS' if overall_pass else 'FAIL'}")
    print(f"dual-die datasheet reference: {DUAL_DIE_GBPS:.0f} GB/s")
    assert overall_pass, "accuracy check failed"


if __name__ == "__main__":
    main()
