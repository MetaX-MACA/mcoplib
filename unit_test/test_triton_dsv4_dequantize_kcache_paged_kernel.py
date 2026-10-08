"""Unit test + bandwidth benchmark for the DSV4 paged K-cache dequantize op
(mcoplib.triton_dsv4_dequantize_k_cache_paged.dequantize_k_cache_paged) on MetaX C600-U.

The unit test goes through the public wrapper `dequantize_k_cache_paged` only;
it never calls the triton kernel `_dequantize_k_cache_paged_kernel` directly.

DSV4 paged K-cache layout (per token, see module header of the op file):
  [448 fp8(e4m3) nope | 64 bf16 rope] = 576 contiguous data bytes,
  plus 7 ue8m0 scales (one per 64-value nope tile) padded to 8 bytes that live
  in the page scale section.  Per page: [P*576 B token data][P*8 B scales],
  page byte size padded up to a multiple of 576.
  Dequant semantics:
      out[t, 0, c]       = fp8[t, c] * 2^(scale[t, c//64] - 127)    c < 448
      out[t, 0, 448 + r] = rope[t, r]                                (bf16 copy)

Tested DSV4 shapes (fixed dims: nope=448, rope=64, tile=64), page_size=64,
page_table=int32, sequential token locations (prefix fetch):
  num_tokens in {1024, 2048, 3072, 3500, 4096, 5120, 6144, 8024, 8192, 16384}

Accuracy: cosine similarity vs a torch reference built from the same cache
bytes — rope (bf16 passthrough) part >= 0.99999, nope (fp8-quantized) part
>= 0.999, full output >= 0.99999.  Bandwidth = logical bytes moved / time,
timed by CUDA-graph replay (K=32 back-to-back launches x R=10 replays) so
the measurement excludes Python launch overhead and sustains DVFS boost;
this also verifies the op is CUDA-graph capturable.

Run (pick idle GPUs with mx-smi first):
  export CUDA_VISIBLE_DEVICES=0
  cd /home/yiyu/mcoplib/release/mcoplib && python unit_test/test_triton_dsv4_dequantize_kcache_paged_kernel.py
Optional: --baseline <json produced by an earlier run> prints a per-shape
improvement table against that baseline.
"""

import argparse
import json
import os

import torch

from mcoplib.triton_dsv4_dequantize_k_cache_paged import (
    DIM_NOPE,
    DIM_ROPE,
    NOPE_ROPE_BYTES,
    PADDED_SCALE_PER_TOKEN,
    TILE_SIZE,
    dequantize_k_cache_paged,
)

DEVICE = "cuda"
FLOAT8 = torch.float8_e4m3fn

NUM_SCALE_TILES = DIM_NOPE // TILE_SIZE                      # 7
OUT_DIM = DIM_NOPE + DIM_ROPE                                # 512
TOKEN_DATA_BYTES = NOPE_ROPE_BYTES                           # 576

PAGE_SIZE = 64                                               # DSV4 page size
PAGE_TABLE_DTYPES = (torch.int32, torch.int64)

TARGET_GBPS = 1300.0                                         # single-die gate
DUAL_DIE_GBPS = 3480.0                                       # dual-die datasheet figure

NUM_TOKENS_CASES = [1024, 2048, 3072, 3500, 4096, 5120, 6144, 8024, 8192, 16384]

# cos_sim gates
THR_BF16 = 0.99999       # bf16 (rope passthrough) part
THR_FP8 = 0.999          # fp8-quantized (nope dequant) part
THR_FULL = 0.99999       # full output


def bytes_per_page(page_size):
    """Page byte size: token data + scale section, padded to a multiple of 576."""
    raw = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    return (raw + NOPE_ROPE_BYTES - 1) // NOPE_ROPE_BYTES * NOPE_ROPE_BYTES


def effective_bytes_per_token(table_dtype):
    """Logical HBM traffic per token (each tensor counted once):
    fp8 nope read 448 + scale read 8 (padded scale slot) + rope read 128
    + bf16 out write 1024 + page table entry read."""
    return (DIM_NOPE + PADDED_SCALE_PER_TOKEN + DIM_ROPE * 2 + OUT_DIM * 2
            + torch.empty(0, dtype=table_dtype).element_size())


def make_cache(num_tokens, page_size=PAGE_SIZE, table_dtype=torch.int32,
               sequential=True, seed=1234, scale_range=None):
    """Build a synthetic DSV4 paged K-cache plus its page table.

    By default each 64-value tile's ue8m0 scale is derived from the tile amax
    (scale = 2^ceil(log2(amax/448)), DeepSeek-style fp8 scaling), which
    guarantees the fp8 codes are encodable (|src|/scale <= 448, never NaN).
    With scale_range=(lo, hi) the ue8m0 bytes are forced into that range and
    the source values are scaled per tile to stay encodable (|src| <= 400*scale).
    rope values are plain bf16 N(0,1).  Returns the uint8 cache, the page
    table and the pre-quantization nope source (used only for the
    informational fp8-quantization-quality cos).
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    num_slots = (num_tokens + page_size - 1) // page_size * page_size
    num_pages = num_slots // page_size

    if sequential:
        locs = torch.arange(num_tokens, device=DEVICE, dtype=torch.int64)
    else:
        locs = torch.randperm(num_slots, device=DEVICE, generator=g)[:num_tokens].to(torch.int64)

    bpp = bytes_per_page(page_size)
    cache = torch.zeros(num_pages, bpp, dtype=torch.uint8, device=DEVICE)

    src_nope = torch.randn(num_tokens, DIM_NOPE, dtype=torch.bfloat16, device=DEVICE, generator=g)
    rope = torch.randn(num_tokens, DIM_ROPE, dtype=torch.bfloat16, device=DEVICE, generator=g)

    if scale_range is None:
        amax = src_nope.float().view(num_tokens, NUM_SCALE_TILES, TILE_SIZE).abs().amax(dim=2)
        e = torch.ceil(torch.log2(amax.clamp_min(1e-30) / 448.0))          # [T, 7]
        scale_u8 = (127 + e).clamp(1, 254).to(torch.uint8)
    else:
        lo, hi = scale_range
        scale_u8 = torch.randint(lo, hi + 1, (num_tokens, NUM_SCALE_TILES),
                                 dtype=torch.uint8, device=DEVICE, generator=g)

    scale_f = torch.exp2(scale_u8.float() - 127.0)                         # [T, 7]
    if scale_range is not None:
        unit = torch.randn(num_tokens, NUM_SCALE_TILES, TILE_SIZE, device=DEVICE, generator=g)
        src_nope = (unit.clamp(-1.0, 1.0) * 400.0 * scale_f[:, :, None]) \
            .view(num_tokens, DIM_NOPE).to(torch.bfloat16)

    fp8_vals = (src_nope.float() / scale_f.repeat_interleave(TILE_SIZE, dim=1)).to(FLOAT8)
    fp8_u8 = fp8_vals.view(torch.uint8)                                    # [T, 448]

    page_idx = locs // page_size
    in_page = locs % page_size
    data_base = page_idx * bpp + in_page * TOKEN_DATA_BYTES
    scale_base = page_idx * bpp + page_size * TOKEN_DATA_BYTES + in_page * PADDED_SCALE_PER_TOKEN

    flat_u8 = cache.view(-1)
    flat_bf16 = cache.view(torch.bfloat16).view(-1)
    offs64 = torch.arange(TILE_SIZE, device=DEVICE)
    for j in range(NUM_SCALE_TILES):
        flat_u8[(data_base[:, None] + j * TILE_SIZE) + offs64] = \
            fp8_u8[:, j * TILE_SIZE:(j + 1) * TILE_SIZE]
    flat_bf16[((data_base + DIM_NOPE) // 2)[:, None] + offs64] = rope
    pad = torch.zeros(num_tokens, 1, dtype=torch.uint8, device=DEVICE)
    flat_u8[scale_base[:, None] + torch.arange(PADDED_SCALE_PER_TOKEN, device=DEVICE)] = \
        torch.cat([scale_u8, pad], dim=1)

    return dict(cache=cache, page_table=locs.to(table_dtype), page_size=page_size,
                src_nope=src_nope)


def ref_dequantize(cache, page_table, page_size):
    """Torch reference: gathers the same bytes and applies the same math
    (fp8 -> fp32, * exp2(scale - 127) in fp32, -> bf16; rope copied as bf16)."""
    bpp = cache.shape[-1]
    num_tokens = page_table.shape[0]
    loc = page_table.long()
    data_base = (loc // page_size) * bpp + (loc % page_size) * TOKEN_DATA_BYTES
    scale_base = (loc // page_size) * bpp + page_size * TOKEN_DATA_BYTES \
        + (loc % page_size) * PADDED_SCALE_PER_TOKEN

    flat_u8 = cache.view(-1)
    flat_bf16 = cache.view(torch.bfloat16).view(-1)
    offs64 = torch.arange(TILE_SIZE, device=DEVICE)

    nope = torch.empty(num_tokens, DIM_NOPE, dtype=torch.float32, device=DEVICE)
    for j in range(NUM_SCALE_TILES):
        fp8 = flat_u8[(data_base[:, None] + j * TILE_SIZE) + offs64].view(FLOAT8)
        s_j = flat_u8[scale_base + j].float()
        nope[:, j * TILE_SIZE:(j + 1) * TILE_SIZE] = fp8.float() * torch.exp2(s_j - 127.0)[:, None]
    rope = flat_bf16[((data_base + DIM_NOPE) // 2)[:, None] + offs64]

    out = torch.empty(num_tokens, 1, OUT_DIM, dtype=torch.bfloat16, device=DEVICE)
    out[:, 0, :DIM_NOPE] = nope.to(torch.bfloat16)
    out[:, 0, DIM_NOPE:] = rope
    return out


def cos_sim(a, b):
    """Cosine similarity in fp64; non-finite inputs fail loudly."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    a = a.detach().flatten().double()
    b = b.detach().flatten().double()
    if a.numel() == 0:
        raise ValueError("empty input needs a dedicated edge-case check")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("non-finite values in reference or kernel output")
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return 1.0 if na == 0 and nb == 0 else 0.0
    return torch.dot(a / na, b / nb).item()


def run_op(t, out=None):
    return dequantize_k_cache_paged(t["cache"], t["page_table"], t["page_size"], out)


def bench_case(num_tokens, table_dtype=torch.int32, page_size=PAGE_SIZE,
               sequential=True, warmup=10, rep=100, seed=1234, with_timing=True,
               scale_range=None):
    t = make_cache(num_tokens, page_size=page_size, table_dtype=table_dtype,
                   sequential=sequential, seed=seed, scale_range=scale_range)

    # ---- accuracy (first call also warms the JIT / autotune) ----
    out = run_op(t)
    torch.cuda.synchronize()  # also surfaces kernel traps / launch errors
    ref = ref_dequantize(t["cache"], t["page_table"], page_size)

    cs_full = cos_sim(out.view(num_tokens, -1).float(), ref.view(num_tokens, -1).float())
    cs_fp8 = cos_sim(out[:, 0, :DIM_NOPE].float(), ref[:, 0, :DIM_NOPE].float())
    cs_bf16 = cos_sim(out[:, 0, DIM_NOPE:].float(), ref[:, 0, DIM_NOPE:].float())
    # informational: fp8 quantization quality vs the pre-quantization source
    cs_src = cos_sim(ref[:, 0, :DIM_NOPE].float(), t["src_nope"].float())
    ok = cs_fp8 >= THR_FP8 and cs_bf16 >= THR_BF16 and cs_full >= THR_FULL

    mean_ms = 0.0
    timing_mode = ""
    if with_timing:
        for _ in range(warmup):
            run_op(t)
        torch.cuda.synchronize()
        # Time via CUDA-graph replay: capture K back-to-back launches and
        # replay the graph R times. This removes Python-side launch overhead
        # (~30 us/launch, which otherwise dominates every case below ~8k
        # tokens), keeps the GPU continuously busy so DVFS boost is
        # sustained, and directly exercises the graph-capturability the op
        # is required to have. Falls back to plain back-to-back event timing
        # if graph capture is unavailable.
        K, R = 32, 10
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(K):
                    run_op(t)
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(R):
                g.replay()
            end.record()
            torch.cuda.synchronize()
            mean_ms = start.elapsed_time(end) / (K * R)
            timing_mode = "graph"
        except Exception:
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
            for i in range(rep):
                starts[i].record()
                run_op(t)
                ends[i].record()
            torch.cuda.synchronize()
            mean_ms = sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / rep
            timing_mode = "events"

    bpt = effective_bytes_per_token(table_dtype)
    gbps = num_tokens * bpt / (mean_ms * 1e-3) / 1e9 if mean_ms > 0 else 0.0
    return dict(ok=ok, cs_full=cs_full, cs_fp8=cs_fp8, cs_bf16=cs_bf16, cs_src=cs_src,
                mean_ms=mean_ms, gbps=gbps, num_pages=t["cache"].shape[0],
                page_size=page_size, table_dtype=str(table_dtype), sequential=sequential,
                timing_mode=timing_mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=str, default=None,
                    help="json file with a previous run's results (prints improvement table)")
    ap.add_argument("--save", type=str, default="/tmp/dq_kcache_results.json",
                    help="where to dump this run's per-shape results")
    args = ap.parse_args()

    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES={vis!r}  device_count={torch.cuda.device_count()}")
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"config: page_size={PAGE_SIZE} dims=nope{DIM_NOPE}(fp8)+rope{DIM_ROPE}(bf16) "
          f"scales=ue8m0x{NUM_SCALE_TILES} out=bf16[{OUT_DIM}] page_table=int32 "
          f"locs=sequential bytes/token={effective_bytes_per_token(torch.int32)}")
    print("=" * 100)

    results = {}
    peak, all_ok = 0.0, True
    for T in NUM_TOKENS_CASES:
        r = bench_case(T)
        results[str(T)] = r
        ok = "OK" if r["ok"] else "FAIL"
        print(f"[dq_kcache] T={T:6d} pages={r['num_pages']:5d}  "
              f"cos_sim={r['cs_full']:.6f} (fp8 {r['cs_fp8']:.6f} bf16 {r['cs_bf16']:.6f}) "
              f"{ok:4s} {r['mean_ms']:8.4f} ms {r['gbps']:8.1f} GB/s [{r['timing_mode']}]")
        peak = max(peak, r["gbps"])
        all_ok &= r["ok"]

    # ---- generality section: layouts/dtypes/page sizes the kernel must also
    #      handle correctly (accuracy only, no timing gate) ----
    print("-" * 100)
    print("generality checks (accuracy only):")
    gen_cases = [
        dict(num_tokens=77, page_size=PAGE_SIZE, note="partial last page"),
        dict(num_tokens=1, page_size=PAGE_SIZE, note="single token"),
        dict(num_tokens=100, page_size=16, note="page_size=16"),
        dict(num_tokens=513, page_size=PAGE_SIZE, table_dtype=torch.int64, note="int64 page table"),
        dict(num_tokens=4096, page_size=PAGE_SIZE, sequential=False, seed=7, note="shuffled locs"),
        dict(num_tokens=2048, page_size=PAGE_SIZE, scale_range=(100, 154), seed=11,
             note="wide ue8m0 scale range"),
    ]
    for gc in gen_cases:
        r = bench_case(gc["num_tokens"], table_dtype=gc.get("table_dtype", torch.int32),
                       page_size=gc["page_size"], sequential=gc.get("sequential", True),
                       seed=gc.get("seed", 1234), with_timing=False,
                       scale_range=gc.get("scale_range"))
        # out= provided as a strided slice of a larger workspace
        t = make_cache(gc["num_tokens"], page_size=gc["page_size"],
                       table_dtype=gc.get("table_dtype", torch.int32),
                       sequential=gc.get("sequential", True), seed=gc.get("seed", 1234),
                       scale_range=gc.get("scale_range"))
        big = torch.zeros(gc["num_tokens"], 2, OUT_DIM, dtype=torch.bfloat16, device=DEVICE)
        out_slice = big[:, 1:2, :]
        run_op(t, out=out_slice)
        torch.cuda.synchronize()
        ref = ref_dequantize(t["cache"], t["page_table"], gc["page_size"])
        ok_slice = (cos_sim(out_slice.float(), ref.float()) >= THR_FULL
                    and big[:, 0:1, :].abs().sum().item() == 0)
        all_ok &= r["ok"] and ok_slice
        print(f"  [gen] T={gc['num_tokens']:5d} page_size={gc['page_size']:3d} "
              f"{gc['note']:<24s} cos_sim={r['cs_full']:.6f} fp8={r['cs_fp8']:.6f} "
              f"bf16={r['cs_bf16']:.6f} out_slice={'OK' if ok_slice else 'FAIL'} "
              f"{'OK' if r['ok'] else 'FAIL'}")

    # ---- shuffled-locs bandwidth (informational; gather-heavy variant) ----
    r = bench_case(8024, sequential=False, seed=3)
    results["shuffled_8024"] = r
    print(f"  [dq_kcache shuffled locs] T=  8024  cos_sim={r['cs_full']:.6f} "
          f"{'OK' if r['ok'] else 'FAIL':4s} {r['mean_ms']:8.4f} ms {r['gbps']:8.1f} GB/s")
    all_ok &= r["ok"]

    print("=" * 100)
    print(f"Accuracy: {'ALL PASS' if all_ok else 'FAILURES PRESENT'} "
          f"(thresholds: cos_sim >= {THR_FULL} full / >= {THR_BF16} bf16 rope / >= {THR_FP8} fp8 nope)")
    print(f"Peak bandwidth: {peak:.1f} GB/s  "
          f"(single-die target {TARGET_GBPS:.0f} -> {'REACHED' if peak >= TARGET_GBPS else 'NOT reached'}; "
          f"dual-die datasheet {DUAL_DIE_GBPS:.0f})")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"results saved to {args.save}")

    if args.baseline and os.path.exists(args.baseline):
        with open(args.baseline) as f:
            base = json.load(f)
        print("-" * 100)
        print("per-shape improvement vs baseline:")
        print(f"{'T':>8s} {'base ms':>10s} {'best ms':>10s} {'base GB/s':>11s} "
              f"{'best GB/s':>11s} {'speedup':>9s} {'bw gain':>9s}")
        for T in NUM_TOKENS_CASES:
            b, c = base.get(str(T)), results.get(str(T))
            if not b or not c:
                continue
            sp = b["mean_ms"] / c["mean_ms"] if c["mean_ms"] > 0 else 0.0
            print(f"{T:8d} {b['mean_ms']:10.4f} {c['mean_ms']:10.4f} "
                  f"{b['gbps']:11.1f} {c['gbps']:11.1f} {sp:8.2f}x {(sp - 1) * 100:8.1f}%")
        bpk = base.get("shuffled_8024")
        cpk = results.get("shuffled_8024")
        if bpk and cpk and cpk["mean_ms"] > 0:
            sp = bpk["mean_ms"] / cpk["mean_ms"]
            print(f"{'shuf8024':>8s} {bpk['mean_ms']:10.4f} {cpk['mean_ms']:10.4f} "
                  f"{bpk['gbps']:11.1f} {cpk['gbps']:11.1f} {sp:8.2f}x {(sp - 1) * 100:8.1f}%")

    assert all_ok, "accuracy check failed"


if __name__ == "__main__":
    main()
