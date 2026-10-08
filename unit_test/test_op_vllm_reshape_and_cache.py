#!/usr/bin/env python3
"""Unit test + perf benchmark for the vLLM `reshape_and_cache` CUDA kernel on
MetaX C600-U.

Test matrix follows the store_kv_cache JSON config exactly:
  arg_type=llm, dtype=bfloat16, head_dim=128, q_head_num=8, kv_head_num=1,
  block_size=64, cache_dtype in {int8, float8, bfloat16}.
    decode : batch_size=16, q_len in {1,4},
             cache_len in {1024,2048,4096,8192,10240}
    prefill: batch_size=1,  q_len in {16,64,...,32768,65536}, cache_len=0

Op semantics (torch.ops._C_cache_ops.reshape_and_cache):
  key/value : [num_tokens, num_heads, head_size]   (bf16, STRIDED slices of QKV)
  key_cache : [num_blocks, num_heads, head_size/x, block_size, x]
  value_cache: [num_blocks, num_heads, head_size, block_size]
  For each token t with slot s = slot_mapping[t] (>=0):
     block = s // block_size ; off = s % block_size
     key_cache[block, h, hb, off, xx]  = Q(key[t, h, hb*x + xx])
     value_cache[block, h, d, off]     = Q(value[t, h, d])

  Q(.) depends on kv_cache_dtype:
    "auto" : identity copy, cache stored as bf16 (no quantization).
    "fp8"  : cache stored as float8_e4m3fn. q = clamp(x/scale, +-448) -> fp8.
    "int8" : cache stored as int8.          q = clamp(round(x/scale), +-127).
  Static per-tensor scale = absmax / qmax (qmax: fp8=448, int8=127), matching
  the kernel's inline CopyWithScaleOp. k_scale / v_scale computed on host; the
  torch reference reproduces the identical quantization so the kernel's
  scatter+quant is validated exactly.

Cosine-similarity thresholds: bf16 >= 0.9999, int8/fp8 >= 0.999.

Bandwidth (algorithmic lower bound, fused traffic):
  read  = num_tokens * kv_head * head_dim * 2 (key + value bf16 read)
        = 2 * num_tokens * kv_head * head_dim * 2B
  write = 2 * num_tokens * kv_head * head_dim * sizeof(cache_dtype)
  Only the WRITTEN tokens (slot>=0) count; slot_mapping here is dense so all
  num_tokens are written.

Run:
  export CUDA_VISIBLE_DEVICES=<idle gpu>
  /opt/conda/bin/python unit_test/test_op_vllm_reshape_and_cache.py
  /opt/conda/bin/python unit_test/test_op_vllm_reshape_and_cache.py --quick
"""

import argparse
import os
import torch

import mcoplib._C  # registers torch.ops._C_cache_ops.reshape_and_cache

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
SINGLE_DIE_GBPS = 1600.0     # single-die gate for this task
DUAL_DIE_GBPS = 3200.0       # dual-die datasheet figure
TARGET_GBPS = 1600.0

# Fixed config (JSON): q_head=8 kv_head=1 head_dim=128 block_size=64.
Q_HEAD = 8
KV_HEAD = 1
HEAD_DIM = 128
BLOCK_SIZE = 64
X = 16 // torch.tensor([], dtype=DTYPE).element_size()   # bf16 -> x = 8

# kv_cache_dtype -> (op string, torch cache dtype, qmax, cos_sim threshold)
CACHE_CONFIGS = {
    "bfloat16": ("auto", DTYPE,               None,  0.9999),
    "float8":   ("fp8",  torch.float8_e4m3fn, 448.0, 0.999),
    "int8":     ("int8", torch.int8,          127.0, 0.999),
}

# JSON decode: batch=16, q_len in {1,4}, cache_len sweep.
DECODE_BATCH = 16
DECODE_QLENS = [1, 4]
DECODE_CACHE_LENS = [1024, 2048, 4096, 8192, 10240]

# JSON prefill: batch=1, q_len sweep, cache_len=0.
PREFILL_BATCH = 1
PREFILL_QLENS = [16, 64, 128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536,
                 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096,
                 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528,
                 24576, 26624, 28672, 30720, 32768, 65536]

# Large-working-set sweep (batch=1, cache_len=0), beyond the JSON config's
# T=65536 ceiling, to expose the kernel's asymptotic bandwidth vs working set.
# Per-token fused traffic is 1 KiB (bf16 K+V read 512B + Kcache+Vcache write
# 512B), so working_set_MB == T / 1024. T=1048576 -> 1 GiB working set.
#   T=65536  -> 64MB     T=262144 -> 256MB    T=786432  -> 768MB
#   T=98304  -> 96MB     T=393216 -> 384MB    T=1048576 -> 1GB
#   T=131072 -> 128MB    T=524288 -> 512MB
#   T=196608 -> 192MB
LARGE_WS_QLENS = [65536, 98304, 131072, 196608, 262144, 393216, 524288,
                  786432, 1048576]


def cos_sim(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return 1.0
    return (torch.dot(a, b) / (na * nb)).item()


def quantize(x, scale, cache_dtype, qmax):
    """Reproduce the kernel's inline CopyWithScaleOp in torch."""
    if qmax is None:                       # "auto" -> identity
        return x.to(cache_dtype)
    f = x.float() / scale
    if cache_dtype == torch.int8:
        q = f.round().clamp(-127.0, 127.0)  # round-half-to-even, saturate
        return q.to(torch.int8)
    f = f.clamp(-448.0, 448.0)              # fp8 e4m3 saturate
    return f.to(torch.float8_e4m3fn)


def ref_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping,
                          block_size, x, k_scale, v_scale, cache_dtype, qmax):
    """Vectorized torch reference; quantizes exactly like the kernel."""
    T, H, D = key.shape
    hbc = D // x
    valid = slot_mapping >= 0
    slots = slot_mapping[valid]
    blocks = (slots // block_size).long()
    offs = (slots % block_size).long()
    key_r = key[valid].reshape(-1, H, hbc, x)
    key_cache[blocks, :, :, offs, :] = quantize(key_r, k_scale, cache_dtype, qmax)
    value_cache[blocks, :, :, offs] = quantize(value[valid], v_scale,
                                               cache_dtype, qmax)


def build_slot_mapping(mode, batch, q_len, cache_len, block_size, num_blocks):
    """Realistic slot mapping. Each sequence occupies a contiguous run of
    blocks; new tokens are appended starting at position `cache_len`.
      decode : each of `batch` sequences already holds cache_len tokens; the
               q_len new tokens go to slots [cache_len, cache_len+q_len).
      prefill: single sequence, cache_len=0, tokens fill [0, q_len).
    Sequences are placed in disjoint block ranges so slots never collide.
    """
    blocks_per_seq = (cache_len + q_len + block_size - 1) // block_size
    slots = []
    for b in range(batch):
        seq_block0 = b * blocks_per_seq
        for j in range(q_len):
            pos = cache_len + j
            slot = seq_block0 * block_size + pos
            slots.append(slot)
    return torch.tensor(slots, dtype=torch.int64, device=DEVICE)


def run_test(mode, batch, q_len, cache_len, cache_dtype_str):
    op_str, cache_torch_dtype, qmax, threshold = CACHE_CONFIGS[cache_dtype_str]
    x = int(X)
    hbc = HEAD_DIM // x
    num_tokens = batch * q_len

    blocks_per_seq = (cache_len + q_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = max(batch * blocks_per_seq + 2, 4)

    # Real packed QKV; key/value are STRIDED views, exactly as vLLM feeds it.
    total_heads = Q_HEAD + 2 * KV_HEAD
    qkv = torch.randn(num_tokens, total_heads, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    key = qkv[:, Q_HEAD:Q_HEAD + KV_HEAD, :]
    value = qkv[:, Q_HEAD + KV_HEAD:Q_HEAD + 2 * KV_HEAD, :]
    assert key.stride(0) == total_heads * HEAD_DIM

    if qmax is None:
        k_scale_v = v_scale_v = 1.0
    else:
        k_scale_v = (key.abs().max().float() / qmax).clamp_min(1e-8).item()
        v_scale_v = (value.abs().max().float() / qmax).clamp_min(1e-8).item()
    k_scale = torch.tensor([k_scale_v], dtype=torch.float32, device=DEVICE)
    v_scale = torch.tensor([v_scale_v], dtype=torch.float32, device=DEVICE)

    # Shared random init so unwritten cache slots match between ref and kernel.
    if cache_torch_dtype == DTYPE:
        base_kc = torch.randn(num_blocks, KV_HEAD, hbc, BLOCK_SIZE, x,
                              dtype=DTYPE, device=DEVICE)
        base_vc = torch.randn(num_blocks, KV_HEAD, HEAD_DIM, BLOCK_SIZE,
                              dtype=DTYPE, device=DEVICE)
    else:
        base_kc = torch.randn(num_blocks, KV_HEAD, hbc, BLOCK_SIZE, x,
                              dtype=torch.float32, device=DEVICE).to(cache_torch_dtype)
        base_vc = torch.randn(num_blocks, KV_HEAD, HEAD_DIM, BLOCK_SIZE,
                              dtype=torch.float32, device=DEVICE).to(cache_torch_dtype)

    slot_mapping = build_slot_mapping(mode, batch, q_len, cache_len,
                                      BLOCK_SIZE, num_blocks)
    assert slot_mapping.numel() == num_tokens
    assert int(slot_mapping.max()) < num_blocks * BLOCK_SIZE

    # reference
    kc_ref = base_kc.clone()
    vc_ref = base_vc.clone()
    ref_reshape_and_cache(key, value, kc_ref, vc_ref, slot_mapping, BLOCK_SIZE,
                          x, k_scale_v, v_scale_v, cache_torch_dtype, qmax)

    # kernel
    kc = base_kc.clone()
    vc = base_vc.clone()
    torch.ops._C_cache_ops.reshape_and_cache(
        key, value, kc, vc, slot_mapping, op_str, k_scale, v_scale)
    torch.cuda.synchronize()

    k_sim = cos_sim(kc_ref, kc)
    v_sim = cos_sim(vc_ref, vc)
    ok = (k_sim >= threshold) and (v_sim >= threshold)

    esize_in = qkv.element_size()                          # 2 (bf16)
    esize_out = torch.tensor([], dtype=cache_torch_dtype).element_size()
    elems = 2 * num_tokens * KV_HEAD * HEAD_DIM            # key + value
    moved = elems * esize_in + elems * esize_out

    for _ in range(20):
        torch.ops._C_cache_ops.reshape_and_cache(
            key, value, kc, vc, slot_mapping, op_str, k_scale, v_scale)
    torch.cuda.synchronize()

    N = 100
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(N):
        torch.ops._C_cache_ops.reshape_and_cache(
            key, value, kc, vc, slot_mapping, op_str, k_scale, v_scale)
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / N
    gbps = moved / (ms * 1e-3) / 1e9

    tag = "OK " if ok else "BAD"
    ws_mb = moved / (1024 * 1024)
    print(f"[{cache_dtype_str:8s}][{mode:7s}] bs={batch:<2d} qlen={q_len:<7d} "
          f"clen={cache_len:<5d} T={num_tokens:<7d} ws={ws_mb:7.1f}MB "
          f"k_cos={k_sim:.6f} v_cos={v_sim:.6f} {tag} "
          f"{ms:8.4f} ms  {gbps:8.1f} GB/s")
    return ok, gbps, num_tokens


def iter_cases(quick, large_ws=False):
    """Yield (mode, batch, q_len, cache_len)."""
    if large_ws:
        # Only the large-working-set prefill sweep (up to 1GB), to read the
        # kernel's asymptotic bandwidth ceiling vs working set.
        for q_len in LARGE_WS_QLENS:
            yield ("prefill", PREFILL_BATCH, q_len, 0)
        return
    d_clens = DECODE_CACHE_LENS if not quick else [1024, 10240]
    for q_len in DECODE_QLENS:
        for clen in d_clens:
            yield ("decode", DECODE_BATCH, q_len, clen)
    p_qlens = PREFILL_QLENS if not quick else [512, 4096, 65536]
    for q_len in p_qlens:
        yield ("prefill", PREFILL_BATCH, q_len, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="reduced shape sweep for fast ReAct iterations")
    ap.add_argument("--large-ws", action="store_true",
                    help="large working-set sweep up to 1GB (T=1048576) to "
                         "expose the asymptotic bandwidth ceiling")
    ap.add_argument("--cache-dtype", nargs="+",
                    choices=list(CACHE_CONFIGS.keys()), default=None)
    args = ap.parse_args()

    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    print(f"CUDA_VISIBLE_DEVICES='{dev}'  device_count={torch.cuda.device_count()}")
    print(f"dtype={DTYPE} q_head={Q_HEAD} kv_head={KV_HEAD} head_dim={HEAD_DIM} "
          f"block_size={BLOCK_SIZE} x={int(X)}")

    dtypes = args.cache_dtype or list(CACHE_CONFIGS.keys())
    overall_pass = True
    for cache_dtype_str in dtypes:
        _, _, _, threshold = CACHE_CONFIGS[cache_dtype_str]
        print("=" * 108)
        print(f"cache_dtype = {cache_dtype_str}  (cos_sim threshold >= {threshold})")
        print("-" * 108)
        all_pass = True
        peak = 0.0
        peak_desc = ""
        for mode, batch, q_len, clen in iter_cases(args.quick, args.large_ws):
            ok, gbps, T = run_test(mode, batch, q_len, clen, cache_dtype_str)
            all_pass = all_pass and ok
            # Peak measured on the largest-traffic prefill shapes.
            if mode == "prefill" and T >= 4096 and gbps > peak:
                peak = gbps
                peak_desc = f"{mode} T={T}"
        overall_pass = overall_pass and all_pass
        reached = "reached" if peak >= TARGET_GBPS else "NOT reached"
        print("-" * 108)
        print(f"  [{cache_dtype_str}] Accuracy: "
              f"{'ALL PASS' if all_pass else 'SOME FAILED'}   "
              f"Peak: {peak:.1f} GB/s [{peak_desc}] "
              f"(target {int(TARGET_GBPS)} -> {reached})")

    print("=" * 108)
    print(f"OVERALL: {'ALL PASS' if overall_pass else 'SOME FAILED'}  "
          f"(single-die {int(SINGLE_DIE_GBPS)}, dual-die {int(DUAL_DIE_GBPS)})")


if __name__ == "__main__":
    main()
