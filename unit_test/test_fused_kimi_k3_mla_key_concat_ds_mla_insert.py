# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import mcoplib._C


DEVICE = "cuda"
OP_NAME = "fused_kimi_k3_mla_key_concat_ds_mla_insert"

K_NOPE = 128
K_PE = 64
K_TOTAL = 192
KV_LORA = 512
CACHE_ROW = 656
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0  # matches kFp8ScaleDivisor on non-gfx942 targets

# 1000 randomized shape/seed cases x 2 dtypes = 2000 test cases, plus one
# peak-bandwidth case per common head count (test_ds_mla_insert_peak_bw).
NUM_CASES = 1000

# Reference-check chunk (tokens); scaled down for large head counts by
# _ref_chunk() so fp32 temporaries stay bounded (H=128 -> 8K tokens/chunk).
REF_CHUNK = 1 << 18  # 256K (head counts up to 4)

# Common production head counts (per MetaX feedback): 1 / 48 / 64 / 128 are
# the frequent TP-split values across models; 4 = Kimi K3 TP=24, 96 = Kimi K3
# unsplit. Random 1..16 coverage below still exercises odd-head boundaries.
COMMON_HEADS = (1, 4, 48, 64, 96, 128)


def _ref_chunk(num_heads):
    """Reference-check chunk (tokens): keeps chunk x H roughly constant so
    the fp32 temporaries of the reference check stay bounded for large head
    counts (H=128 -> 8K tokens, H<=4 -> 256K, same as the old fixed chunk)."""
    return min(REF_CHUNK, max(1 << 12, (1 << 20) // num_heads))

DTYPE_CASES = [
    pytest.param(torch.float16, id="fp16"),
    pytest.param(torch.bfloat16, id="bf16"),
]


def _require_op():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")

    if not hasattr(torch.ops._C, OP_NAME):
        pytest.skip(f"torch.ops._C.{OP_NAME} is not available")


def _make_rope_cache(max_position):
    # Standard rotary table: [max_position, 64] float32, cos[0:32] | sin[32:64].
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(0, K_PE, 2, dtype=torch.float32, device=DEVICE) / K_PE)
    )
    t = torch.arange(max_position, dtype=torch.float32, device=DEVICE)
    freqs = torch.outer(t, inv_freq)  # [max_position, 32]
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()


def _apply_gptj_rope_ref(x, cos_row, sin_row):
    # GPT-J interleaved rotation, matching the kernel's copyChunk8/rotateChunk8:
    # pairs (x[2i], x[2i+1]) with cos_row[i], sin_row[i]:
    #   out[2i]   = x[2i] * cos[i] - x[2i+1] * sin[i]
    #   out[2i+1] = x[2i] * sin[i] + x[2i+1] * cos[i]
    # x: [..., 64] float32; cos_row/sin_row: [..., 32] (broadcastable to x's
    # leading dims, e.g. [32], [T, 32] or [T, 1, 32]).
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos_row - x2 * sin_row
    out2 = x2 * cos_row + x1 * sin_row
    return torch.stack([out1, out2], dim=-1).flatten(-2)


def _make_inputs(num_tokens, num_heads, dtype, seed, apply_rope, max_position=2048):
    torch.manual_seed(seed)

    q = torch.randn((num_tokens, num_heads, K_TOTAL), dtype=dtype, device=DEVICE)
    k_nope = torch.randn((num_tokens, num_heads, K_NOPE), dtype=dtype, device=DEVICE)
    k_pe = torch.randn((num_tokens, K_PE), dtype=dtype, device=DEVICE)
    kv_c = torch.randn((num_tokens, KV_LORA), dtype=dtype, device=DEVICE)

    # Unordered slot mapping: random permutation of a range larger than
    # num_tokens -- cache rows are written out of order, some rows untouched.
    num_slots = num_tokens + max(16, num_tokens // 4)
    slots = torch.randperm(num_slots, device=DEVICE)[:num_tokens].to(torch.int64)

    block_size = 16
    nblk = (num_slots + block_size - 1) // block_size
    k_cache = torch.full(
        (nblk, block_size, CACHE_ROW), 0x7F, dtype=torch.uint8, device=DEVICE
    )

    position_ids = None
    cos_sin_cache = None
    if apply_rope:
        position_ids = torch.randint(
            0, max_position, (num_tokens,), dtype=torch.int64, device=DEVICE
        )
        cos_sin_cache = _make_rope_cache(max_position)

    return (
        q, k_nope, k_pe, kv_c, slots, k_cache, block_size,
        position_ids, cos_sin_cache,
    )


# L2 cache size fallback (bytes) if the device properties don't expose it;
# verify on the target box with `macainfo`.
_L2_BYTES_FALLBACK = 64 * 1024 * 1024
_L2_BYTES = None


def _l2_bytes():
    global _L2_BYTES
    if _L2_BYTES is None:
        try:
            _L2_BYTES = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).L2_cache_size
        except AttributeError:
            _L2_BYTES = _L2_BYTES_FALLBACK
    return _L2_BYTES


def _traffic_bytes(num_tokens, num_heads, apply_rope=True):
    """Estimated DRAM traffic (bytes).

    Two systematic over-counts of the old logical model are fixed:

    1. rope=off: the q rope segment is never touched (the NoPE checksum
       proves it), but the old model still charged a read + a write-back
       per head (H*64*2 + H*64*2 bytes/token).
    2. Small per-token tensors re-read by multiple warp orders hit L2, not
       DRAM: the ceil(H/4) light orders for one token are adjacent warps
       (same wave), so their k_pe/slots/position_ids re-reads hit L2; the
       later heavy-order re-read of k_pe misses only if k_pe (T*128 bytes)
       has been evicted, i.e. only when it exceeds L2. cos_sin_cache
       ([2048, 64] fp32 = 512 KB) is always L2-resident (~0 DRAM/token).

    Large streams (q rope rw, k_nope, kv_c, k_out, cache rows) are touched
    exactly once and always count. Write-allocate overhead on the scattered
    656B cache rows (<= 128B per 656B, < 1% of total) is ignored.
    """
    elem = 2  # fp16/bf16
    light_orders = (num_heads + 3) // 4  # ceil(H/4), HP=4 heads per warp

    per_token = num_heads * 128 * elem   # k_nope read (once)
    per_token += 512 * elem              # kv_c read (once)
    per_token += num_heads * 192 * elem  # k_out write
    per_token += 656                     # cache row write
    per_token += 8                       # slots read (light, once)
    if apply_rope:
        per_token += num_heads * 64 * elem * 2  # q rope: read + write-back
        per_token += 8                         # position_ids read (once)
    # k_pe: (light_orders + 1) logical reads collapse to 1 DRAM read
    # (light wave) + 1 more for the heavy wave iff k_pe > L2.
    k_pe_total = num_tokens * 64 * elem
    per_token += 64 * elem * (2 if k_pe_total > _l2_bytes() else 1)
    return num_tokens * per_token


def _case_bytes(num_tokens, num_heads):
    """Device bytes one test case needs, including the reference check's
    fp32 temporaries (bounded by REF_CHUNK)."""
    elem = 2  # fp16/bf16
    # persistent tensors: q, k_out, k_nope, q_pre(rope-only snapshot)
    persistent = (
        num_heads * (K_TOTAL + K_TOTAL + K_NOPE + K_PE) * elem
        + K_PE * elem          # k_pe
        + KV_LORA * elem       # kv_c
        + CACHE_ROW            # k_cache row
        + 8                    # slots
    )
    # reference-check peak per chunk: fp32 copies of q/k_nope/k_pe/kv_c
    # slices + intermediate rope tensors, ~4x the chunk's raw bytes.
    chunk = _ref_chunk(num_heads)
    chunk_peak = (
        num_heads * (K_TOTAL + K_NOPE + K_PE) * elem
        + K_PE * elem + KV_LORA * elem
    ) * 4 * chunk
    return num_tokens * persistent + chunk_peak


def _fit_tokens(num_tokens, num_heads, free_bytes):
    """Shrink num_tokens (huge-bucket cases) until the case fits in free
    memory. Returns the shrunk token count, or 0 if even 64K doesn't fit.
    The 64K floor (not 1M) matters for large head counts: H=128 needs
    ~147KB/token, so even 1M tokens won't fit -- but 64K tokens still give
    a huge grid (64K x 33 warps) with full coverage value."""
    while num_tokens > (1 << 16) and _case_bytes(num_tokens, num_heads) > free_bytes:
        num_tokens //= 2
    return num_tokens if _case_bytes(num_tokens, num_heads) <= free_bytes else 0


def _q_nope_checksum(q):
    """O(1)-memory checksum of q's NoPE segment: the kernel must leave it
    untouched. sum(dtype=fp32) accumulates in fp32 WITHOUT materializing an
    fp32 copy (q[...].float().sum() would allocate a full 2x-size temp)."""
    return q[:, :, :K_NOPE].sum(dtype=torch.float32).item()


def _run_op(q, k_nope, k_pe, kv_c, k_out, k_cache, slots, block_size,
            position_ids, cos_sin_cache):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.ops._C.fused_kimi_k3_mla_key_concat_ds_mla_insert(
        q, k_nope, k_pe, kv_c, k_out, k_cache, slots, block_size,
        position_ids, cos_sin_cache,
    )
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def _reference(q_pre_rope, k_nope, k_pe, kv_c, slots, k_cache, block_size,
               position_ids, cos_sin_cache, dtype):
    """Vectorized golden reference. q_pre_rope is the pre-op rope segment
    [T, H, 64]. Returns (q_rope_ref, k_out_ref, cache_fp8, cache_scales,
    cache_rope) with cache pieces as logical tensors."""
    num_tokens, num_heads, _ = k_nope.shape

    # ---- light path ----
    if position_ids is not None:
        cos_all = cos_sin_cache[position_ids, :32].float()  # [T, 32]
        sin_all = cos_sin_cache[position_ids, 32:].float()  # [T, 32]
        # q rope segment: [T, H, 64] -> rotate per token
        cos_q = cos_all.view(num_tokens, 1, 32)
        sin_q = sin_all.view(num_tokens, 1, 32)
        q_rope_ref = _apply_gptj_rope_ref(
            q_pre_rope.float(), cos_q, sin_q).to(dtype)
        # k_pe rotated once per token, broadcast to all heads
        pe_f = k_pe.float()  # [T, 64]
        pe_rot = _apply_gptj_rope_ref(pe_f, cos_all, sin_all).to(dtype)
        k_out_ref = torch.cat(
            [k_nope, pe_rot.unsqueeze(1).expand(num_tokens, num_heads, K_PE)],
            dim=-1,
        )
    else:
        q_rope_ref = q_pre_rope
        k_out_ref = torch.cat(
            [k_nope, k_pe.unsqueeze(1).expand(num_tokens, num_heads, K_PE)], dim=-1
        )
        pe_rot = k_pe

    # ---- heavy path: per-tile absmax fp8 quantization ----
    # 656B row = [fp8 latent 512B][4 x fp32 tile scales 16B][rope bf16 128B]
    latent = kv_c.float().view(num_tokens, 4, 128)  # [T, 4 tiles, 128]
    tile_max = latent.abs().amax(dim=-1)  # [T, 4]
    scales = (tile_max / FP8_MAX).clamp(min=torch.finfo(torch.float32).tiny)
    quant = (latent / scales.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)

    return q_rope_ref, k_out_ref, quant, scales, pe_rot


def _cos_sim(a, b):
    """Cosine similarity of two fp32 tensors, flattened. The acceptance
    metric for light-path (>= 0.99999) and quantization (>= 0.999)
    outputs."""
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0), dim=1
    ).item()


# Acceptance thresholds (per review requirement):
#   light path (rope outputs)  : cos_sim >= 0.99999  ("5 nines")
#   quantization (fp8 latent)  : cos_sim >= 0.999    ("3 nines")
LIGHT_PATH_COS_SIM_MIN = 0.99999
QUANT_COS_SIM_MIN = 0.999


def _check(q_pre_rope, q, k_nope, k_pe, kv_c, k_out, k_cache, slots, block_size,
           position_ids, cos_sin_cache, dtype):
    # q_pre_rope: pre-op snapshot of q's rope segment [T, H, 64] (the kernel
    # rotates q in place; only the rotated tail needs a pre-op copy).
    # q: the actual post-op tensor to verify.
    # The NoPE segment is verified separately by the caller via a checksum
    # (O(1) memory) -- see _q_nope_checksum.
    # Chunked over tokens so huge cases (>4M) don't materialize multiple
    # [T, 512] fp32 tensors at once.
    num_tokens = slots.shape[0]
    num_heads = k_nope.shape[1]
    chunk = _ref_chunk(num_heads)

    blk = slots // block_size
    off = slots % block_size

    for start in range(0, num_tokens, chunk):
        end = min(start + chunk, num_tokens)
        q_rope_ref, k_out_ref, quant_ref, scales_ref, pe_rot_ref = _reference(
            q_pre_rope[start:end], k_nope[start:end], k_pe[start:end],
            kv_c[start:end], slots[start:end], k_cache, block_size,
            position_ids[start:end] if position_ids is not None else None,
            cos_sin_cache, dtype,
        )

        # q rope segment: rotated, rounded. Acceptance: cos_sim >= 0.99999
        # (light path, 5 nines); the elementwise tolerance below is a
        # secondary guard against localized errors the cosine can average out.
        sim = _cos_sim(q[start:end, :, K_NOPE:], q_rope_ref)
        assert sim >= LIGHT_PATH_COS_SIM_MIN, (
            f"q rope cos_sim={sim:.7f} < {LIGHT_PATH_COS_SIM_MIN} "
            f"(tokens {start}:{end})"
        )
        torch.testing.assert_close(
            q[start:end, :, K_NOPE:].float(), q_rope_ref.float(),
            atol=2e-2, rtol=2e-2,
        )

        # k_out: NoPE exact copy, rope segment rounded
        assert torch.equal(k_out[start:end, :, :K_NOPE], k_out_ref[:, :, :K_NOPE])
        sim = _cos_sim(k_out[start:end, :, K_NOPE:], k_out_ref[:, :, K_NOPE:])
        assert sim >= LIGHT_PATH_COS_SIM_MIN, (
            f"k_out rope cos_sim={sim:.7f} < {LIGHT_PATH_COS_SIM_MIN} "
            f"(tokens {start}:{end})"
        )
        torch.testing.assert_close(
            k_out[start:end, :, K_NOPE:].float(), k_out_ref[:, :, K_NOPE:].float(),
            atol=2e-2, rtol=2e-2,
        )

        # cache rows written by the op (scatter via slots)
        actual_fp8 = k_cache[blk[start:end], off[start:end], :512].view(FP8_DTYPE).float()
        actual_scales = k_cache[blk[start:end], off[start:end], 512:528].view(torch.float32)
        actual_rope = k_cache[blk[start:end], off[start:end], 528:].view(dtype)

        # scales: fp32 exact math in kernel, but absmax over bf16-rounded inputs
        torch.testing.assert_close(
            actual_scales, scales_ref, atol=1e-6, rtol=1e-5,
        )
        # fp8 latent: dequantize and compare. Acceptance: cos_sim >= 0.999
        # (quantization, 3 nines) on the dequantized values; the elementwise
        # tolerance below stays as the per-element guard.
        actual_deq = actual_fp8.view(-1, 4, 128) * actual_scales.unsqueeze(-1)
        ref_deq = quant_ref.float() * scales_ref.unsqueeze(-1)
        sim = _cos_sim(actual_deq, ref_deq)
        assert sim >= QUANT_COS_SIM_MIN, (
            f"fp8 dequant cos_sim={sim:.7f} < {QUANT_COS_SIM_MIN} "
            f"(tokens {start}:{end})"
        )
        torch.testing.assert_close(
            actual_deq, ref_deq, atol=0.15, rtol=0.15,
        )
        # rope segment: bf16 copy of rotated pe -- same fp32 math as the kernel
        # but the cast to fp16/bf16 may differ by 1 ULP (FMA vs separate mul-add)
        sim = _cos_sim(actual_rope, pe_rot_ref)
        assert sim >= LIGHT_PATH_COS_SIM_MIN, (
            f"cache rope cos_sim={sim:.7f} < {LIGHT_PATH_COS_SIM_MIN} "
            f"(tokens {start}:{end})"
        )
        torch.testing.assert_close(
            actual_rope.float(), pe_rot_ref.float(),
            atol=2e-2, rtol=2e-2,
        )


# Peak-bandwidth shapes: common production head counts (per MetaX feedback:
# 1 / 48 / 64 / 128 are frequent; 4 = Kimi K3 TP=24, 96 = Kimi K3 unsplit).
# Token count scales as T = 2^24 / H so every case moves the same total
# element count (H=4 -> 4194304 tokens, the production scale).
# Defined before the randomized test so pytest (file order) runs these first.
PEAK_BW_HEADS = COMMON_HEADS
PEAK_BW_TOTAL_ELEMS = 1 << 24  # 16777216 = 4 heads x 4194304 tokens


def _peak_bw_tokens(num_heads):
    return PEAK_BW_TOTAL_ELEMS // num_heads


@pytest.mark.parametrize("dtype", DTYPE_CASES)
@pytest.mark.parametrize("num_heads", PEAK_BW_HEADS)
@torch.inference_mode()
def test_ds_mla_insert_peak_bw(dtype, num_heads):
    _require_op()

    num_tokens = _peak_bw_tokens(num_heads)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    if free < _case_bytes(num_tokens, num_heads):
        pytest.skip(
            f"not enough device memory for peak-bw case "
            f"(heads={num_heads}, tokens={num_tokens}, "
            f"need ~{_case_bytes(num_tokens, num_heads) / 2**30:.1f} GiB, "
            f"free {free / 2**30:.1f} GiB)"
        )

    apply_rope = True  # production path: rope always on for peak-bw runs
    seed = 40000 + num_heads

    (q, k_nope, k_pe, kv_c, slots, k_cache, block_size,
     position_ids, cos_sin_cache) = _make_inputs(
        num_tokens, num_heads, dtype, seed, apply_rope
    )
    k_out = torch.empty_like(q)
    # rope-only snapshot + NoPE checksum: the kernel must not touch NoPE.
    q_pre_rope = q[:, :, K_NOPE:].clone()
    q_nope_sum_pre = _q_nope_checksum(q)

    gpu_ms = _run_op(q, k_nope, k_pe, kv_c, k_out, k_cache, slots, block_size,
                     position_ids, cos_sin_cache)

    bw_gbs = _traffic_bytes(num_tokens, num_heads, apply_rope) / (gpu_ms * 1e-3) / 1e9
    print(
        f"[peak-bw heads={num_heads:2d}] tokens={num_tokens:8d} "
        f"dtype={str(dtype).split('.')[-1]:8s} gpu={gpu_ms:8.3f}ms "
        f"bw={bw_gbs:7.1f} GB/s"
    )

    # NoPE segment must be untouched (fp32 checksum, exact).
    assert _q_nope_checksum(q) == q_nope_sum_pre, (
        f"q NoPE segment was modified by the kernel (heads={num_heads})"
    )

    _check(q_pre_rope, q, k_nope, k_pe, kv_c, k_out, k_cache, slots, block_size,
           position_ids, cos_sin_cache, dtype)


@pytest.mark.parametrize("dtype", DTYPE_CASES)
@pytest.mark.parametrize("case_idx", range(NUM_CASES))
@torch.inference_mode()
def test_ds_mla_insert_randomized(dtype, case_idx):
    _require_op()

    # Deterministic per-case RNG: case_idx fully determines shape/seed.
    rng = torch.Generator()
    rng.manual_seed(20261016 + case_idx)

    # Token scale buckets: small / medium / large / huge (> 4M tokens).
    # Most cases stay small (reference-impl cost), a slice covers each
    # larger bucket so grid/scheduling boundaries get exercised.
    bucket = int(torch.randint(0, 100, (1,), generator=rng))
    if bucket < 60:
        num_tokens = int(torch.randint(1, 257, (1,), generator=rng))       # small
    elif bucket < 80:
        num_tokens = int(torch.randint(257, 4097, (1,), generator=rng))    # medium
    elif bucket < 95:
        num_tokens = int(torch.randint(4097, 65537, (1,), generator=rng))  # large
    else:
        num_tokens = int(torch.randint(4194305, 8388609, (1,), generator=rng))  # huge

    # Common production head counts dominate (per MetaX feedback: 1 / 48 /
    # 64 / 128 frequent, 4 = Kimi K3 TP=24, 96 = unsplit); a minority sweeps
    # 1..16 to cover the odd-head / lightOrders-boundary branches.
    if int(torch.randint(0, 10, (1,), generator=rng)) < 8:
        num_heads = COMMON_HEADS[case_idx % len(COMMON_HEADS)]
    else:
        num_heads = int(torch.randint(1, 17, (1,), generator=rng))

    apply_rope = bool(torch.randint(0, 2, (1,), generator=rng))
    seed = 30000 + case_idx

    # Memory guard: huge-bucket cases shrink to fit available memory so the
    # suite never OOMs (H=4, T=8M needs ~64 GiB of inputs alone).
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    num_tokens = _fit_tokens(num_tokens, num_heads, free)
    if num_tokens == 0:
        pytest.skip(
            f"not enough device memory even for 64K tokens "
            f"(heads={num_heads}, free {free / 2**30:.1f} GiB)"
        )

    (q, k_nope, k_pe, kv_c, slots, k_cache, block_size,
     position_ids, cos_sin_cache) = _make_inputs(
        num_tokens, num_heads, dtype, seed, apply_rope
    )
    k_out = torch.empty_like(q)
    # rope-only snapshot + NoPE checksum: the kernel must not touch NoPE.
    q_pre_rope = q[:, :, K_NOPE:].clone()
    q_nope_sum_pre = _q_nope_checksum(q)

    gpu_ms = _run_op(q, k_nope, k_pe, kv_c, k_out, k_cache, slots, block_size,
                     position_ids, cos_sin_cache)

    bw_gbs = _traffic_bytes(num_tokens, num_heads, apply_rope) / (gpu_ms * 1e-3) / 1e9
    print(
        f"[case {case_idx:4d}] tokens={num_tokens:4d} heads={num_heads:3d} "
        f"rope={'on ' if apply_rope else 'off'} dtype={str(dtype).split('.')[-1]:8s} "
        f"gpu={gpu_ms * 1e3:8.1f}us bw={bw_gbs:7.1f} GB/s"
    )

    # NoPE segment must be untouched (fp32 checksum, exact).
    assert _q_nope_checksum(q) == q_nope_sum_pre, (
        f"q NoPE segment was modified by the kernel "
        f"(case {case_idx}, heads={num_heads})"
    )

    _check(q_pre_rope, q, k_nope, k_pe, kv_c, k_out, k_cache, slots, block_size,
           position_ids, cos_sin_cache, dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
