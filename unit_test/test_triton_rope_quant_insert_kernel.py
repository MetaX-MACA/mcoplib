"""Unit test + bandwidth benchmark for the Triton rope_quant_insert kernels on MetaX C600-U.

Covers BOTH output paths of mcoplib.triton_rope_quant_insert_kernel.rope_quant_insert
and BOTH quant types (fp8 + newly-added int8):

  * paged  (kv_cache dtype uint8 / int8, last-dim 584): fp8_ds_mla-style layout,
    576 value bytes (448 NoPE quant | 128 RoPE bf16) + 8 segregated scale bytes.
      - fp8 : per-group(8x64) UE8M0 power-of-two exponent scale, e4m3 values.
      - int8: per-token symmetric round-to-nearest, scale = amax/127 stored as fp32
              in the first 4 scale bytes.
  * plain  (kv_cache dtype bfloat16 / float8_e4m3fn / int8, last-dim 512):
    [448 NoPE | 64 RoPE] whole row.
      - bf16: identity store.
      - fp8 : per-tensor scale (fp8_scale arg).
      - int8: per-tensor symmetric round-to-nearest, scale = amax/127 (quant_scale out).

RoPE is GPT-J interleave on the last 64 dims (pairs (2i,2i+1)); NoPE dims are passed
through unrotated. compress_ratio in {1,2}; only positions with (pos+1)%CR==0 are
written; slot<0 rows are skipped.

Accuracy: cosine similarity in fp64 over the dequantized cache vs a torch reference;
fp8 and int8 thresholds are both 0.999. Bandwidth / latency printed per case; a
trap check runs via torch.cuda.synchronize().

Run:
  export CUDA_VISIBLE_DEVICES=<free gpu from mx-smi>
  python unit_test/test_triton_rope_quant_insert_kernel.py
"""

import os
import torch

from mcoplib.triton_rope_quant_insert_kernel import (
    _rope_quant_insert_kernel,  # noqa: F401  (import path required by the task)
    rope_quant_insert,
)

DEVICE = "cuda"

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_PAIRS = 64
FP8_BYTES = 584         # paged fp8:  512 val (448 fp8 | 128 rope-bf16... 576) + 8 scale
INT8_BYTES = 576        # paged int8: 512 int8 val + 64 scale (16 fp32, per-group 16x32)
INT8_GROUP = 32         # paged int8 per-group quant group size
INT8_NGROUP = HEAD_DIM // INT8_GROUP    # 16 groups over the full rotated 512 row

TARGET_GBPS = 1300.0    # single-die gate
DUAL_DIE_GBPS = 3480.0  # dual-die datasheet reference

FP8 = torch.float8_e4m3fn
COS_THRESH = 0.999

# num_tokens cases from the test_v41 reference (small + large shapes)
NUM_TOKENS_CASES = [1, 4, 17, 64, 256, 2048, 8192]
COMPRESS_RATIOS = [1, 2]


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def make_cos_sin_cache(max_pos, dtype=torch.float32):
    """cos_sin_cache[pos, 0:32] = cos(theta_i), [32:64] = sin(theta_i) for 32 freqs."""
    half = ROPE_PAIRS  # 64 pairs -> 64 freqs? layout uses 32 cos + 32 sin per row.
    # The kernel loads c from cs[pair] and s from cs[32+pair] for pair in [0,32),
    # i.e. 32 cos + 32 sin. So the rope segment covers 32 pairs (64 dims) but the
    # kernel indexes pair = arange(256)-224 -> valid pair in [0,32).
    dim = 64
    inv_freq = 1.0 / (10000 ** (torch.arange(0, 32, dtype=torch.float64) / 32))
    pos = torch.arange(max_pos, dtype=torch.float64)[:, None]
    ang = pos * inv_freq[None, :]           # [max_pos, 32]
    cache = torch.empty(max_pos, dim, dtype=dtype)
    cache[:, :32] = torch.cos(ang).to(dtype)
    cache[:, 32:] = torch.sin(ang).to(dtype)
    return cache.to(DEVICE)


def rope_gptj_ref(normed, position, cos_sin_cache, compress_ratio):
    """Torch reference matching the kernel's RoPE exactly.

    normed: [512] fp32. Kernel: even,odd = split(reshape(normed,(256,2)));
      pair = arange(256)-224  (valid pair in [0,32));
      cs = cos_sin_cache[(pos//CR)*CR];  c = cs[pair] (pair>=0 else 1), s = cs[32+pair] (else 0)
      rotated = interleave(even*c - odd*s, odd*c + even*s)  -> [512]
    Returns rotated[512] fp32 (this is the bf16 RoPE payload for the last 128 dims;
    the first 448 dims come back unchanged because their pair<0 -> c=1,s=0)."""
    even = normed[0::2].clone()            # [256]
    odd = normed[1::2].clone()             # [256]
    pair = torch.arange(256, device=DEVICE) - 224
    cs_row = cos_sin_cache[(position // compress_ratio) * compress_ratio]  # [64]
    c = torch.where(pair >= 0, cs_row[pair.clamp(min=0)], torch.ones_like(pair, dtype=torch.float32))
    s = torch.where(pair >= 0, cs_row[32 + pair.clamp(min=0)], torch.zeros_like(pair, dtype=torch.float32))
    ro_even = even * c - odd * s
    ro_odd = odd * c + even * s
    rotated = torch.empty(512, dtype=torch.float32, device=DEVICE)
    rotated[0::2] = ro_even
    rotated[1::2] = ro_odd
    return rotated


def cos_sim_fp64(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    na = a.norm()
    nb = b.norm()
    if na == 0 and nb == 0:
        return 1.0
    return (torch.dot(a, b) / (na * nb).clamp_min(1e-30)).item()


def make_common(num_tokens, compress_ratio, seed=1234):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    latent = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE, generator=g)
    # positions chosen so ALL rows are at a group boundary ((pos+1)%CR==0) -> full path
    if compress_ratio == 1:
        positions = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    else:
        positions = (torch.arange(num_tokens, dtype=torch.int64, device=DEVICE) * 2 + 1)
    max_pos = int(positions.max().item()) + 4
    cos_sin = make_cos_sin_cache(max_pos)
    return latent, positions, cos_sin


# --------------------------------------------------------------------------- #
# paged path (uint8 fp8 = 584B / int8 = 576B)                                  #
# --------------------------------------------------------------------------- #
def paged_alloc(num_tokens, quant):
    dtype = torch.uint8 if quant == "fp8" else torch.int8
    head_bytes = FP8_BYTES if quant == "fp8" else INT8_BYTES
    block_size = max(1, min(64, num_tokens))
    num_blocks = (num_tokens + block_size - 1) // block_size
    # 3D paged layout: [num_blocks, block_size, head_bytes]. The kernel reads
    # CACHE_BLOCK = shape[1]; last dim = 584 (fp8) or 576 (int8).
    kv = torch.zeros(num_blocks, block_size, head_bytes, dtype=dtype, device=DEVICE)
    slot = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    return kv, slot, block_size, num_blocks


def paged_dequant_ref_and_out(kv, slot, latent, positions, cos_sin, cr, block_size, quant):
    """Return (ref_rows, out_rows) both [num_tokens, 512] fp32, dequantized.

    Both layouts are SEGREGATED per-page (values region then scales region):
      fp8  (584B): page = [ block_size*576 value | block_size*8 scale ]
                   values(slot) = page + (slot%CB)*576  (448 fp8 | 128 rope-bf16)
                   scales(slot) = page + CB*576 + (slot%CB)*8   (7 UE8M0 + 1 pad)
      int8 (576B): page = [ block_size*512 value | block_size*64 scale ]
                   values(slot) = page + (slot%CB)*512  (full rotated 512 int8)
                   scales(slot) = page + CB*512 + (slot%CB)*64  (16 fp32, per 32 dims)
    """
    num_tokens = slot.numel()
    kv_flat = kv.reshape(kv.shape[0], -1)      # [num_blocks, block_size*head_bytes]
    ref = torch.empty(num_tokens, HEAD_DIM, dtype=torch.float32, device=DEVICE)
    out = torch.empty(num_tokens, HEAD_DIM, dtype=torch.float32, device=DEVICE)
    for t in range(num_tokens):
        sl = int(slot[t].item())
        blk, off = sl // block_size, sl % block_size
        page = kv_flat[blk]
        normed = latent[t].float()
        # ---- reference (identical for both quant types): NoPE identity | RoPE rotated
        rope_full = rope_gptj_ref(normed, int(positions[t].item()), cos_sin, cr)
        ref[t, :NOPE_DIM] = normed[:NOPE_DIM]
        ref[t, NOPE_DIM:] = rope_full[NOPE_DIM:]
        # ---- decode kernel output ----
        if quant == "fp8":
            vals = page[off * 576: off * 576 + 576].contiguous()        # [576]
            sc_base = block_size * 576 + off * 8
            sc_bytes = page[sc_base: sc_base + 8].contiguous()          # [8]
            # scales: 7 UE8M0 exponent bytes (groups 0..6 cover the 448 NoPE dims)
            exps = sc_bytes[:7].to(torch.float32) - 127.0
            deq = torch.empty(NOPE_DIM, dtype=torch.float32, device=DEVICE)
            fp8_vals = vals[:NOPE_DIM].view(FP8).float()
            for gi in range(7):
                deq[gi * 64:(gi + 1) * 64] = fp8_vals[gi * 64:(gi + 1) * 64] * (2.0 ** exps[gi])
            out[t, :NOPE_DIM] = deq
            # RoPE payload: bf16 in value bytes [448:576]
            out[t, NOPE_DIM:] = vals[NOPE_DIM:576].view(torch.bfloat16).float()  # [64]
        else:  # int8: full 512 int8 values + 16 fp32 per-group(32) scales
            vals = page[off * 512: off * 512 + 512].contiguous()        # [512] uint... int8
            sc_base = block_size * 512 + off * 64
            sc = page[sc_base: sc_base + 64].contiguous().view(torch.float32)  # [16]
            i8 = vals.view(torch.int8).float()                          # [512]
            deq = torch.empty(HEAD_DIM, dtype=torch.float32, device=DEVICE)
            for g in range(INT8_NGROUP):                                # 16 groups x 32
                deq[g * 32:(g + 1) * 32] = i8[g * 32:(g + 1) * 32] * sc[g]
            out[t] = deq                                                # groups 14/15 = rotated RoPE
    return ref, out


def bench_paged(num_tokens, cr, quant, warmup=10, rep=50):
    head_bytes = FP8_BYTES if quant == "fp8" else INT8_BYTES
    latent, positions, cos_sin = make_common(num_tokens, cr)
    kv, slot, block_size, num_blocks = paged_alloc(num_tokens, quant)

    rope_quant_insert(latent, positions, cos_sin, kv, slot, cr)
    torch.cuda.synchronize()
    ref, out = paged_dequant_ref_and_out(kv, slot, latent, positions, cos_sin, cr, block_size, quant)
    sim = cos_sim_fp64(out, ref)

    for _ in range(warmup):
        rope_quant_insert(latent, positions, cos_sin, kv, slot, cr)
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        st[i].record()
        rope_quant_insert(latent, positions, cos_sin, kv, slot, cr)
        en[i].record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in zip(st, en))[rep // 2]

    # traffic: read 512 bf16 latent + write head_bytes cache per token
    io = num_tokens * (HEAD_DIM * 2 + head_bytes)
    gbps = io / (ms * 1e-3) / 1e9
    ok = sim >= COS_THRESH
    print(f"[paged-{quant:>4} cr={cr}] T={num_tokens:6d}  cos_sim={sim:.6f} {'OK' if ok else 'FAIL':4s}  "
          f"{ms:8.4f} ms  {gbps:8.1f} GB/s")
    return ok, gbps


# --------------------------------------------------------------------------- #
# plain path (bf16 / fp8 / int8, 512 elems)                                    #
# --------------------------------------------------------------------------- #
def bench_plain(num_tokens, cr, quant, warmup=10, rep=50):
    if quant == "bf16":
        dtype, esz = torch.bfloat16, 2
    elif quant == "fp8":
        dtype, esz = FP8, 1
    else:
        dtype, esz = torch.int8, 1
    latent, positions, cos_sin = make_common(num_tokens, cr)
    block_size = max(1, min(64, num_tokens))
    num_blocks = (num_tokens + block_size - 1) // block_size
    kv = torch.zeros(num_blocks, block_size, HEAD_DIM, dtype=dtype, device=DEVICE)
    slot = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)

    fp8_scale = None
    quant_scale = None
    if quant == "fp8":
        fp8_scale = torch.tensor([latent.abs().float().max().item() / 448.0], dtype=torch.float32, device=DEVICE)
    elif quant == "int8":
        quant_scale = torch.zeros(num_tokens, dtype=torch.float32, device=DEVICE)

    rope_quant_insert(latent, positions, cos_sin, kv, slot, cr,
                      fp8_scale=fp8_scale, quant_scale=quant_scale)
    torch.cuda.synchronize()

    # reference: whole 512 row = [nope | rope] then quantize/dequantize
    ref = torch.empty(num_tokens, HEAD_DIM, dtype=torch.float32, device=DEVICE)
    for t in range(num_tokens):
        normed = latent[t].float()
        rope_full = rope_gptj_ref(normed, int(positions[t].item()), cos_sin, cr)
        row = normed.clone()
        row[NOPE_DIM:] = rope_full[NOPE_DIM:]
        ref[t] = row
    kv3 = kv.view(num_blocks, block_size, HEAD_DIM)
    out = torch.empty(num_tokens, HEAD_DIM, dtype=torch.float32, device=DEVICE)
    for t in range(num_tokens):
        sl = int(slot[t].item())
        blk, off = sl // block_size, sl % block_size
        r = kv3[blk, off]
        if quant == "bf16":
            out[t] = r.float()
        elif quant == "fp8":
            out[t] = r.float() * fp8_scale[0]
        else:
            out[t] = r.float() * quant_scale[t]
    sim = cos_sim_fp64(out, ref)

    for _ in range(warmup):
        rope_quant_insert(latent, positions, cos_sin, kv, slot, cr,
                          fp8_scale=fp8_scale, quant_scale=quant_scale)
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        st[i].record()
        rope_quant_insert(latent, positions, cos_sin, kv, slot, cr,
                          fp8_scale=fp8_scale, quant_scale=quant_scale)
        en[i].record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in zip(st, en))[rep // 2]

    io = num_tokens * (HEAD_DIM * 2 + HEAD_DIM * esz)
    gbps = io / (ms * 1e-3) / 1e9
    ok = sim >= COS_THRESH
    print(f"[plain-{quant:>4} cr={cr}] T={num_tokens:6d}  cos_sim={sim:.6f} {'OK' if ok else 'FAIL':4s}  "
          f"{ms:8.4f} ms  {gbps:8.1f} GB/s")
    return ok, gbps


def main():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES={vis!r}  device_count={torch.cuda.device_count()}")
    assert torch.cuda.is_available(), "CUDA not available"
    print("=" * 96)

    overall_pass = True
    peak = 0.0

    print("---- paged path (uint8 fp8 / int8, 584B)  cos_sim threshold 0.999 " + "-" * 20)
    for quant in ("fp8", "int8"):
        for cr in COMPRESS_RATIOS:
            for T in NUM_TOKENS_CASES:
                ok, gbps = bench_paged(T, cr, quant)
                overall_pass &= ok
                peak = max(peak, gbps)

    print("---- plain path (bf16 / fp8 / int8, 512)  cos_sim threshold 0.999 " + "-" * 20)
    for quant in ("bf16", "fp8", "int8"):
        for cr in COMPRESS_RATIOS:
            for T in NUM_TOKENS_CASES:
                ok, gbps = bench_plain(T, cr, quant)
                overall_pass &= ok
                peak = max(peak, gbps)

    print("=" * 96)
    print(f"peak = {peak:.1f} GB/s  ({'REACHED' if peak >= TARGET_GBPS else 'below'} "
          f"single-die target {TARGET_GBPS:.0f});  dual-die datasheet {DUAL_DIE_GBPS:.0f}")
    assert overall_pass, "accuracy check failed (some case cos_sim < 0.999)"
    print("ALL ACCURACY CHECKS PASSED")


if __name__ == "__main__":
    main()

