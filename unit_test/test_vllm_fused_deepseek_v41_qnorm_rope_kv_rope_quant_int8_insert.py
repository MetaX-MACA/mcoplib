"""Unit test + bandwidth benchmark for the horizontally-fused DeepseekV4-MLA
kernels on MetaX C600-U.

Covers two decoupled KV quant formats, both driven by the same fused pipeline
(per-head RMSNorm(no weight) + GPT-J RoPE on Q; GPT-J RoPE + quant + paged
cache insert on KV):

  * FP8   -> torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
             (UE8M0 per-64-dim-block exponent, 7 blocks over 448 NoPE dims)
  * INT8  -> torch.ops._C.fused_deepseek_v41_qnorm_rope_kv_rope_int8_insert
             (per-32-dim-group symmetric int8, 14 groups, one fp32 scale/group)

MXFP4: the repository exposes no MXFP4 kernel path for this op, so MXFP4 cases
are reported as SKIPPED (see main()'s summary) rather than fabricated.

Accuracy: cosine similarity of the dequantized K-cache NoPE plane + bf16 RoPE
plane against a pure-torch reference (fp64-accumulated, chunked). Threshold for
both fp8 and int8 is > 0.999. Q output (RMSNorm+RoPE) is also cos-checked and a
handful of typical output values are spot-printed.

Bandwidth: back-to-back burst within a single sync (sustains DVFS), sorted
median over `rep` iterations, effective_bytes / median_time. Print format
follows unit_test/test_op_moe_scatter_dynamic_quant.py.

Run:
  mx-smi                         # pick an idle GPU
  export CUDA_VISIBLE_DEVICES=<idle>
  python unit_test/test_vllm_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.py
"""

import os
import torch
import mcoplib._C  # noqa: F401  registers torch.ops._C.*

DEVICE = "cuda"


def _detect_fp8_fnuz():
    """C600U/MACA is gfx942-family -> fp8 e4m3 is stored FNUZ (bias 8, no inf).
    The kernel emits FNUZ bytes into a float8_e4m3fn-typed tensor, so the NoPE
    plane must be reinterpreted under FNUZ before decoding. Detect at runtime
    without importing vllm: compare the device's fp8 round-trip of a known
    value under both encodings, or fall back to the ROCm-build heuristic."""
    # torch.version.hip is set on ROCm/MACA builds.
    is_hip = getattr(torch.version, "hip", None) is not None
    if is_hip and hasattr(torch, "float8_e4m3fnuz"):
        return True
    return False


FP8_FNUZ = _detect_fp8_fnuz()
FP8_STORE_DTYPE = (torch.float8_e4m3fnuz
                   if FP8_FNUZ and hasattr(torch, "float8_e4m3fnuz")
                   else torch.float8_e4m3fn)

# Mutable one-slot holder; calibrate_fp8_view() sets the encoding that actually
# decodes this build's fp8 bytes (MACA torch reports version.hip=None, so the
# static heuristic can be wrong -> pick empirically).
_FP8_VIEW = [FP8_STORE_DTYPE]

# ── Constants matching the kernels ──────────────────────────────────────────
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM  # 448

FP8_QUANT_BLOCK = 64
FP8_NUM_BLOCKS = NOPE_DIM // FP8_QUANT_BLOCK          # 7
FP8_SCALE_BYTES = FP8_NUM_BLOCKS + 1                  # 8 (7 exp + 1 pad)
FP8_MAX = 224.0 if FP8_FNUZ else 448.0  # kFp8Max: FNUZ e4m3 max on gfx942
FP8_HEAD_BYTES = NOPE_DIM + ROPE_DIM * 2 + FP8_SCALE_BYTES  # 584

INT8_QUANT_BLOCK = 32
# Full-head int8 quant: the ENTIRE 512-dim head (NoPE + rotated RoPE) is
# quantized to int8 in 16 groups of 32 dims, one fp32 scale per group.
INT8_NUM_BLOCKS = HEAD_DIM // INT8_QUANT_BLOCK        # 16
INT8_SCALE_BYTES = INT8_NUM_BLOCKS * 4               # 64
INT8_DATA_BYTES = HEAD_DIM                            # 512 (all int8)
INT8_HEAD_BYTES = INT8_DATA_BYTES + INT8_SCALE_BYTES  # 576

# FP8 keeps its own data plane (448 int8 + 128 bf16 RoPE); int8-g32 is all int8.
FP8_DATA_BYTES = NOPE_DIM + ROPE_DIM * 2             # 576

# The int8-g32 op under test. The availability probe and the runner both key
# off this single name so they can never drift apart (op-swap hardening).
INT8_OP_NAME = "fused_deepseek_v41_qnorm_rope_kv_rope_int8_insert"

TARGET_GBPS = 1300.0     # single-die gate
DUAL_DIE_GBPS = 3480.0   # dual-die datasheet reference (unreachable single-die)

COS_THR = 0.999

# padded head counts FlashMLA supports (compiled kernel instantiations)
HEAD_CASES = [
    (1, 8), (8, 8), (8, 16), (16, 16), (16, 32),
    (32, 32), (8, 64), (64, 64), (64, 128),
]
BLOCK_SIZES = [16, 64]
# correctness token counts (mirror the reference test) + DP padding coverage
CORRECTNESS_TOKENS = [1, 4, 17, 64, 2048]
# larger token counts for bandwidth headroom toward 1.3 TB/s
BW_TOKENS = [64, 512, 2048, 4096, 8192, 16384]


# ── PyTorch reference (self-contained; no vllm dependency) ───────────────────
def make_cos_sin_cache(max_pos, rope_dim=ROPE_DIM):
    base = 10000.0
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=DEVICE)
                 / rope_dim))
    t = torch.arange(max_pos, dtype=torch.float32, device=DEVICE)
    freqs = torch.einsum("i,j->ij", t, inv_freq)          # [max_pos, rope_dim/2]
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # [max_pos, rope_dim] fp32


def rmsnorm_no_weight(x, eps):
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(var + eps)


def apply_rope_gptj_last_k(x, positions, cos_sin_cache):
    """GPT-J (interleaved-pair) RoPE on the LAST rope_dim dims; fp32 in/out."""
    rope_dim = cos_sin_cache.shape[-1]
    half = rope_dim // 2
    nope = x.shape[-1] - rope_dim
    cs = cos_sin_cache[positions.long()].to(torch.float32)  # [..., rope_dim]
    cos = cs[..., :half]
    sin = cs[..., half:]
    rope = x[..., nope:].float()
    shape = rope.shape
    rope = rope.reshape(*shape[:-1], half, 2)
    even = rope[..., 0]
    odd = rope[..., 1]
    while cos.ndim < even.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    new_even = torch.addcmul(-odd * sin, even, cos)
    new_odd = torch.addcmul(odd * cos, even, sin)
    rot = torch.stack((new_even, new_odd), dim=-1).reshape(shape)
    out = x.clone().float()
    out[..., nope:] = rot
    return out


def _bf16_round(x):
    return x.to(torch.bfloat16).float()


def quant_int8_g32_sim(x):
    """Simulate the kernel's int8-g32 quantization exactly:
    rotate first (caller does GPT-J on last 64 dims), then per-32-dim-group
    symmetric quant: absmax (floored at 1e-4), scale = absmax/127 (fp32),
    q = round_half_away(x * 127/absmax), clamp to [-127, 127].
    Returns (deq, scales, codes) laid out like the cache: deq [...,512] fp32,
    scales [...,16] fp32, codes [...,512] int8."""
    shape = x.shape
    xf = x.float().reshape(-1, HEAD_DIM)
    g = xf.reshape(-1, INT8_NUM_BLOCKS, INT8_QUANT_BLOCK)
    absmax = g.abs().amax(dim=-1)                              # [N, 16]
    absmax = torch.maximum(absmax, torch.full_like(absmax, 1e-4))
    scales = absmax / 127.0                                    # [N, 16] fp32
    inv = 127.0 / absmax
    q = g * inv.unsqueeze(-1)
    # roundf(): half away from zero (torch.round is half-to-even instead)
    q = torch.where(q >= 0, torch.floor(q + 0.5), torch.ceil(q - 0.5))
    q = q.clamp(-127.0, 127.0)
    codes = q.to(torch.int8).reshape(-1, HEAD_DIM)
    deq = codes.float().reshape(-1, INT8_NUM_BLOCKS, INT8_QUANT_BLOCK) \
        * scales.unsqueeze(-1)
    deq = deq.reshape(-1, HEAD_DIM).reshape(shape)
    return deq, scales.reshape(*shape[:-1], INT8_NUM_BLOCKS), codes


# ── Cache decoders (match the kernel byte layouts exactly) ──────────────────
def _gather_token_bytes(k_cache_u8, slot_mapping, block_size, head_bytes,
                        num_tokens, token_data_bytes):
    """Return a [num_tokens, head_bytes] uint8 view of the per-token records
    addressed the way the kernel writes them: data plane is token-strided at
    token_data_bytes, scale trailer lives after all tokens' data planes."""
    num_blocks = k_cache_u8.shape[0]
    blk = k_cache_u8  # [num_blocks, block_size*head_bytes]
    slot = slot_mapping.long()
    block_idx = slot // block_size
    pos_in_block = slot % block_size
    data = torch.empty(num_tokens, token_data_bytes, dtype=blk.dtype,
                       device=DEVICE)
    scale_bytes = head_bytes - token_data_bytes
    scales = torch.empty(num_tokens, scale_bytes, dtype=blk.dtype,
                         device=DEVICE)
    for i in range(num_tokens):
        bi = int(block_idx[i])
        pb = int(pos_in_block[i])
        base = blk[bi]
        d0 = pb * token_data_bytes
        data[i] = base[d0:d0 + token_data_bytes]
        s0 = block_size * token_data_bytes + pb * scale_bytes
        scales[i] = base[s0:s0 + scale_bytes]
    return data, scales


def decode_fp8_cache(k_cache_u8, slot_mapping, block_size, num_tokens,
                     view_dtype=None):
    data, scale_b = _gather_token_bytes(
        k_cache_u8, slot_mapping, block_size, FP8_HEAD_BYTES, num_tokens,
        FP8_DATA_BYTES)
    # NoPE fp8 e4m3 (FNUZ on gfx942/MACA -> reinterpret under the stored dtype)
    vd = view_dtype if view_dtype is not None else FP8_STORE_DTYPE
    nope_fp8 = data[:, :NOPE_DIM].contiguous().view(vd).float()
    exps = scale_b[:, :FP8_NUM_BLOCKS].float() - 127.0        # [T, 7]
    factor = torch.pow(2.0, exps).repeat_interleave(FP8_QUANT_BLOCK, dim=1)
    nope_deq = nope_fp8 * factor                             # [T, 448]
    rope_bf16 = data[:, NOPE_DIM:NOPE_DIM + ROPE_DIM * 2].contiguous().view(
        torch.bfloat16).float()                             # [T, 64]
    return torch.cat([nope_deq, rope_bf16], dim=1)          # [T, 512]


def decode_int8_g32_cache(k_cache, slot_mapping, block_size, num_tokens):
    """Full-head int8 decode: all 512 dims are int8, 16 group scales, no bf16
    RoPE plane. Returns (deq [T,512] fp32, scales [T,16] fp32, codes [T,512]
    int8) gathered with the kernel's split layout (512B/token data plane, then
    the per-block 64B/token fp32 scale trailer)."""
    data, scale_b = _gather_token_bytes(
        k_cache, slot_mapping, block_size, INT8_HEAD_BYTES, num_tokens,
        INT8_DATA_BYTES)
    codes = data.contiguous().view(torch.int8)                 # [T, 512]
    scales = scale_b.contiguous().view(torch.float32)          # [T, 16]
    deq = codes.float() * scales.repeat_interleave(INT8_QUANT_BLOCK, dim=1)
    return deq, scales, codes


# ── Reference KV (RoPE on last 64, no quant) ──────────────────────────
def kv_reference(kv, positions, cos_sin_cache, fmt="int8"):
    roped = apply_rope_gptj_last_k(kv, positions, cos_sin_cache)  # [T, 512] fp32
    if fmt == "fp8":
        # FP8 path stores the RoPE plane as raw bf16 in the cache; round it for
        # a fair compare. NoPE quant error is absorbed by the cosine threshold.
        ref = roped.clone()
        ref[:, NOPE_DIM:] = _bf16_round(roped[:, NOPE_DIM:])
        return ref
    # int8-g32 quantizes the WHOLE rotated head to int8 -> compare against the
    # exact fp32 rotated head; quant error is absorbed by the cosine threshold.
    return roped


def q_reference(q_in, positions, cos_sin_cache, eps, apply_q_norm):
    """[T, H, 512] -> reference bf16-rounded Q output for the live heads."""
    x = q_in.float()
    if apply_q_norm:
        x = rmsnorm_no_weight(x, eps)
    x = apply_rope_gptj_last_k(x, positions, cos_sin_cache)
    return _bf16_round(x)


def cosine(a, b):
    a = a.double().reshape(-1)
    b = b.double().reshape(-1)
    denom = (a.norm() * b.norm()).clamp_min(1e-30)
    return (a @ b / denom).item()


# ── Input builders ──────────────────────────────────────────────────────────
def make_inputs(num_tokens, n_heads, padded_heads, block_size, pad=0, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(1234 + seed)
    dtype = torch.bfloat16
    q_in = torch.randn(num_tokens, n_heads, HEAD_DIM, dtype=dtype,
                       device=DEVICE, generator=g)
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=dtype, device=DEVICE,
                     generator=g)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    max_pos = max(num_tokens, 8) + 8
    cos_sin_cache = make_cos_sin_cache(max_pos)
    # DP padding: slot_mapping shorter than q rows (KV inserts only first rows).
    num_insert = num_tokens - pad
    slot_mapping = torch.arange(num_insert, dtype=torch.int64, device=DEVICE)
    num_blocks = (num_insert + block_size - 1) // block_size + 1
    return dict(q_in=q_in, kv=kv, positions=positions,
                cos_sin_cache=cos_sin_cache, slot_mapping=slot_mapping,
                num_insert=num_insert, num_blocks=num_blocks,
                block_size=block_size, padded_heads=padded_heads,
                n_heads=n_heads, num_tokens=num_tokens)


def run_fp8(t, eps, apply_q_norm):
    head_bytes = FP8_HEAD_BYTES
    k_cache = torch.zeros(t["num_blocks"], t["block_size"] * head_bytes,
                          dtype=torch.uint8, device=DEVICE)
    q_out = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        t["q_in"], t["kv"], k_cache, t["slot_mapping"], t["positions"],
        t["cos_sin_cache"], t["padded_heads"], eps, t["block_size"],
        apply_q_norm)
    return q_out, k_cache


def run_int8(t, eps, apply_q_norm):
    # int8-g32 op: k_cache is int8 (per token: 512 int8 codes + 16 fp32
    # group scales = 576 B). The op TORCH_CHECKs this dtype.
    head_bytes = INT8_HEAD_BYTES
    k_cache = torch.zeros(t["num_blocks"], t["block_size"] * head_bytes,
                          dtype=torch.int8, device=DEVICE)
    q_out = getattr(torch.ops._C, INT8_OP_NAME)(
        t["q_in"], t["kv"], k_cache, t["slot_mapping"], t["positions"],
        t["cos_sin_cache"], t["padded_heads"], eps, t["block_size"], apply_q_norm)
    return q_out, k_cache


# ── Bandwidth model ─────────────────────────────────────────────────────────
def effective_bytes(fmt, num_insert, num_tokens_full, n_heads, padded_heads):
    """Bytes actually moved by the fused op.
    KV rows: read 512 bf16 + write NoPE quant + RoPE bf16 + scale trailer.
    Q rows : read 512 bf16 (live) + write padded_heads*512 bf16 out.
    """
    if fmt == "fp8":
        kv_write = NOPE_DIM * 1 + ROPE_DIM * 2 + FP8_SCALE_BYTES
    else:
        # int8-g32: whole head is int8 (512 B) + 16 fp32 group scales (64 B).
        kv_write = HEAD_DIM * 1 + INT8_SCALE_BYTES
    kv_bytes = num_insert * (HEAD_DIM * 2 + kv_write)
    q_read = num_tokens_full * n_heads * HEAD_DIM * 2
    q_write = num_tokens_full * padded_heads * HEAD_DIM * 2
    return kv_bytes + q_read + q_write


# ── One correctness+bandwidth case ──────────────────────────────────────────
def bench_case(fmt, t, eps, apply_q_norm, warmup=10, rep=50, spot=False):
    runner = run_fp8 if fmt == "fp8" else run_int8

    # ---- correctness ----
    q_out, k_cache = runner(t, eps, apply_q_norm)
    torch.cuda.synchronize()

    ni = t["num_insert"]
    if fmt == "fp8":
        kv_deq = decode_fp8_cache(k_cache, t["slot_mapping"][:ni],
                                  t["block_size"], ni, view_dtype=_FP8_VIEW[0])
        kv_ref = kv_reference(t["kv"][:ni], t["positions"][:ni],
                              t["cos_sin_cache"], fmt="fp8")
        kv_sim = cosine(kv_deq, kv_ref)
        scale_ok, code_ok, quant_sim = True, True, 1.0
    else:
        kv_deq, kv_scales, kv_codes = decode_int8_g32_cache(
            k_cache, t["slot_mapping"][:ni], t["block_size"], ni)
        kv_ref = kv_reference(t["kv"][:ni], t["positions"][:ni],
                              t["cos_sin_cache"], fmt="int8")
        kv_sim = cosine(kv_deq, kv_ref)
        # Format-faithful reference: simulate the kernel's exact g32 quant
        # (rotate -> 32-dim group absmax -> scale=absmax/127 -> round/clamp)
        # and require the cache to match the SIMULATION, not merely a loose
        # end-to-end cosine. This pins the 32-dim granularity, the 16-scale
        # layout, and the fact that RoPE is quantized (groups 14/15).
        roped = apply_rope_gptj_last_k(t["kv"][:ni], t["positions"][:ni],
                                       t["cos_sin_cache"])
        sim_deq, sim_scales, sim_codes = quant_int8_g32_sim(roped)
        quant_sim = cosine(kv_deq, sim_deq)
        scale_ok = bool(torch.allclose(kv_scales, sim_scales,
                                       rtol=1e-5, atol=1e-9))
        code_bad = (kv_codes != sim_codes).float().mean().item()
        code_max = (kv_codes.int() - sim_codes.int()).abs().max().item()
        # Measured kernel-vs-sim noise: fp32 rotation differs by 1-2 ulp
        # (fma contraction), flipping roundf for ~0.14% of codes, always by
        # exactly +-1. A wrong granularity (64-dim groups) or wrong layout
        # produces diffs >= 2 on ~25% of dims, and trips scale_ok too.
        code_ok = bool(code_bad <= 1e-2 and code_max <= 1)

    # Q side: compare live heads (padded region is zero-filled by design).
    q_ref = q_reference(t["q_in"], t["positions"], t["cos_sin_cache"], eps,
                        apply_q_norm)
    q_got = q_out[:, :t["n_heads"], :].float()
    q_sim = cosine(q_got, q_ref)

    sim = min(kv_sim, q_sim)
    ok = (sim >= COS_THR) and scale_ok and code_ok

    if spot and not ok:
        print(f"      spot kv_deq[0,:6]={kv_deq[0,:6].tolist()}")
        print(f"      spot kv_ref[0,:6]={kv_ref[0,:6].tolist()}")
        print(f"      spot scale_ok={scale_ok} code_ok={code_ok} "
              f"quant_sim={quant_sim:.6f}")

    # ---- bandwidth (back-to-back burst within one sync) ----
    for _ in range(warmup):
        runner(t, eps, apply_q_norm)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        runner(t, eps, apply_q_norm)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    med = times[len(times) // 2]

    eb = effective_bytes(fmt, ni, t["num_tokens"], t["n_heads"],
                         t["padded_heads"])
    gbps = eb / (med * 1e-3) / 1e9
    return ok, sim, kv_sim, q_sim, med, gbps


# ── Test matrix build (>= 100 cases) ────────────────────────────────────────
def build_cases():
    cases = []  # (fmt, num_tokens, n_heads, padded_heads, block_size, pad, qnorm)
    for fmt in ("fp8", "int8"):
        for nt in CORRECTNESS_TOKENS:
            for (nh, ph) in HEAD_CASES:
                for bs in BLOCK_SIZES:
                    cases.append((fmt, nt, nh, ph, bs, 0, True))
                    cases.append((fmt, nt, nh, ph, bs, 0, False))
        # DP-padding coverage
        for nt in [17, 64, 2048]:
            for pad in (1, 5):
                cases.append((fmt, nt, 8, 64, 16, pad, True))
                cases.append((fmt, nt, 8, 64, 16, pad, False))
        # qnorm-off coverage (fp8 op supports the flag; int8-g32 always on ->
        # only meaningful for fp8, but keep the reference in sync)
        
            for nt in [4, 2048]:
                cases.append((fmt, nt, 8, 64, 16, 0, False))
    return cases


def calibrate_fp8_view(eps):
    """Decode a small FP8 run under both e4m3 encodings and keep whichever
    matches the reference best (MACA torch hides the FNUZ flag)."""
    candidates = [torch.float8_e4m3fn]
    if hasattr(torch, "float8_e4m3fnuz"):
        candidates.append(torch.float8_e4m3fnuz)
    t = make_inputs(64, 8, 8, 16, pad=0, seed=99)
    q_out, k_cache = run_fp8(t, eps, True)
    torch.cuda.synchronize()
    ni = t["num_insert"]
    kv_ref = kv_reference(t["kv"][:ni], t["positions"][:ni],
                          t["cos_sin_cache"], fmt="fp8")
    best, best_sim = candidates[0], -1.0
    for vd in candidates:
        deq = decode_fp8_cache(k_cache, t["slot_mapping"][:ni],
                               t["block_size"], ni, view_dtype=vd)
        s = cosine(deq, kv_ref)
        if s > best_sim:
            best, best_sim = vd, s
    _FP8_VIEW[0] = best
    print(f"fp8 encoding calibrated: {best} (cos={best_sim:.6f})")


def main():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES={vis!r}  device_count={torch.cuda.device_count()}")
    assert torch.cuda.is_available(), "CUDA not available"
    have_fp8 = hasattr(torch.ops._C,
                       "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert")
    have_int8 = hasattr(torch.ops._C, INT8_OP_NAME)
    print(f"ops available: fp8={have_fp8} int8_g32={have_int8}")
    print(f"config: HEAD_DIM={HEAD_DIM} ROPE_DIM={ROPE_DIM} "
          f"fp8_block={FP8_QUANT_BLOCK} int8_group={INT8_QUANT_BLOCK} "
          f"cos_thr={COS_THR}")
    if have_fp8:
        calibrate_fp8_view(1e-6)
    print("=" * 108)

    eps = 1e-6
    cases = build_cases()
    print(f"total correctness cases: {len(cases)} "
          f"(fp8 + int8_g32; MXFP4 = SKIPPED, no kernel path in repo)")

    overall = True
    peaks = {"fp8": 0.0, "int8": 0.0}
    npass = {"fp8": 0, "int8": 0}
    ntot = {"fp8": 0, "int8": 0}
    for (fmt, nt, nh, ph, bs, pad, qnorm) in cases:
        if fmt == "fp8" and not have_fp8:
            continue
        if fmt == "int8" and not have_int8:
            continue
        seed = hash((fmt, nt, nh, ph, bs, pad, qnorm)) & 0xffff
        t = make_inputs(nt, nh, ph, bs, pad=pad, seed=seed)
        ok, sim, kv_sim, q_sim, med, gbps = bench_case(fmt, t, eps, qnorm,
                                                       spot=True)
        peaks[fmt] = max(peaks[fmt], gbps)
        ntot[fmt] += 1
        npass[fmt] += int(ok)
        overall &= ok
        tag = "OK" if ok else "FAIL"
        print(f"[{fmt:>4}] T={nt:6d} h={nh:3d}/{ph:3d} bs={bs:2d} pad={pad} "
              f"qn={int(qnorm)}  cos={sim:.6f}(kv={kv_sim:.5f} q={q_sim:.5f}) "
              f"{tag:4s} {med:8.4f} ms {gbps:8.1f} GB/s")

    # ---- dedicated large-shape bandwidth sweep (peak hunt) ----
    print("-" * 108)
    print("bandwidth sweep (large shapes, padded_heads=64):")
    for fmt in ("fp8", "int8"):
        if fmt == "fp8" and not have_fp8:
            continue
        if fmt == "int8" and not have_int8:
            continue
        for nt in BW_TOKENS:
            t = make_inputs(nt, 1, 64, 64, pad=0, seed=7)
            ok, sim, kv_sim, q_sim, med, gbps = bench_case(fmt, t, eps, True)
            peaks[fmt] = max(peaks[fmt], gbps)
            tag = "OK" if ok else "FAIL"
            overall &= ok
            print(f"[{fmt:>4}] T={nt:6d} h=  1/ 64 bs=64        "
                  f"cos={sim:.6f}                 {tag:4s} "
                  f"{med:8.4f} ms {gbps:8.1f} GB/s")

    print("=" * 108)
    for fmt in ("fp8", "int8"):
        if ntot[fmt] == 0:
            print(f"{fmt:>4}: op not built in — skipped")
            continue
        reached = "REACHED" if peaks[fmt] >= TARGET_GBPS else "below"
        print(f"{fmt:>4}: peak {peaks[fmt]:8.1f} GB/s ({reached} single-die "
              f"target {TARGET_GBPS:.0f})  accuracy {npass[fmt]}/{ntot[fmt]} "
              f"{'PASS' if npass[fmt] == ntot[fmt] else 'FAIL'}")
    print("mxfp4: SKIPPED (repository exposes no MXFP4 kernel path for this op)")
    print(f"dual-die datasheet reference: {DUAL_DIE_GBPS:.0f} GB/s")
    assert overall, "accuracy check failed (cos_sim < 0.999 in >=1 case)"
    print("ALL ACCURACY CHECKS PASSED")


if __name__ == "__main__":
    main()
