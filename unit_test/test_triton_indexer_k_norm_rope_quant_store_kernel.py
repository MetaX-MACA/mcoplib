"""Unit test + bandwidth benchmark for
mcoplib.triton_indexer_k_norm_rope_quant_store_kernel on MetaX C600-U.

Op (10-arg launcher, in-place paged store):
  indexer_k_norm_rope_store(k_pre, positions, cos_sin_cache, rms_norm_weight,
                            rms_norm_eps, k_cache, kv_slot_mapping,
                            compress_ratio, use_fp4_cache, use_int8_cache)

Semantics (DeepSeek V4.1 indexer K, fused):
  1. k_norm  : k = k_pre.float() * rsqrt(mean(k^2) + eps) * w   (fp32, one
               bf16 roundtrip at the end, like the reference model code)
  2. RoPE    : GPT-J interleaved on the last 64 dims at the group's first
               token position ((pos // cr) * cr); first 64 dims pass through
  3. emit    : only tokens with (pos + 1) % compress_ratio == 0 AND
               slot >= 0 write the cache; everyone else is masked out
  4. quant   : fp8  -> per-token e4m3 + one fp32 pow2 scale (scale row 4B)
               fp4  -> MXFP4: 2 E2M1 nibbles/byte + ue8m0 per 32 elements
               int8 -> per-token (whole 128-dim head) codes + one fp32
                       scale amax/127 in the 4-byte scale region,
                       code = round-to-nearest-even(x * 127 / amax)
  5. store   : paged uint8 cache [num_blocks, block_size, value_bytes +
               scale_bytes]; token s of block b: values at
               b*stride0 + s*token_stride, scales at
               b*stride0 + block_size*token_stride + s*scale_dim.

Accuracy gates (hard failures):
  * cosine(kernel dequant, unfused fp32 reference) >= 0.999 for fp8 / int8,
    >= 0.95 for fp4 (inherent 4-bit coarseness)
  * cosine(kernel dequant, reference quantizer dequant) >= 0.9999
  * page padding integrity: every cache byte outside the valid write set
    must stay 0xA5 (catches OOB / masked-store bugs)
  * fp8 value bytes never contain e4m3 NaN encodings 0x7F/0xFF
    (no-clamp invariant: |x * 2^-e| <= 448 holds exactly)
  * crafted int8 round-to-nearest-even tie cases must produce the exact
    expected bytes (ties round to even: +-63.5 -> +-64)
Diagnostics (printed, not gated): code/scale byte exactness vs the torch
reference. Bit-exactness can differ on ~0.07% of int8 codes because tl.sum
and torch.mean accumulate the variance in different orders (1 fp32 ulp ->
occasional bf16 boundary flip); the cosine gates above cover that.

Coverage (112 cases):
  A  official-test matrix (padded pages, NaN rows, sparse slots):
     T {1,17,257} x cr {1,2} x {fp8,fp4,int8} x cache_dtype {f32,bf16}   = 36
  B  robustness sweep T {1,17,257,1023,4096} x cr {1,2} x mode           = 30
  C  block_size {16,64,128} x cr {1,2} x {fp8,int8} (paged shape sweep)  = 12
  D  edges: V=0 / all-zero k / TP-heuristic boundaries {255,256,2047,2048}
     / identity rope / sequential positions cr=2 / int8 RNE ties (crafted
     +-63.5) / fp8 extreme dynamic range / CUDA-graph replay x3 modes /
     num_tokens=0 x3 modes                                              = 22
  E  bandwidth table T {1023,4096,16384,65536} x mode (burst, median)    = 12

Run:
  mx-smi                       # pick a free GPU first
  export CUDA_VISIBLE_DEVICES=<free>
  cd /home/yiyu/mcoplib/release/mcoplib
  python unit_test/test_triton_indexer_k_norm_rope_quant_store_kernel.py
"""

import os
import sys

import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from mcoplib.triton_indexer_k_norm_rope_quant_store_kernel import (  # noqa: E402
    _fp4_asm_default,
    indexer_k_norm_rope_store,
)

DEVICE = "cuda"
HEAD = 128
MAX_POS = 4096
EPS = 1e-20
BLOCK_SIZE = 16
MODES = ("fp8", "int8", "fp4")
LAYOUT = {"fp8": (128, 4), "fp4": (64, 4), "int8": (128, 4)}
THR_IDEAL = {"fp8": 0.999, "int8": 0.999, "fp4": 0.95}
TARGET_GBPS = 1300.0

E2M1_VALS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

PASSED = FAILED = SKIPPED = 0


# ---------------------------------------------------------------- references
def norm_rope_ref(k_pre, positions, cos_sin, w, cr):
    """Unfused k_norm + GPT-J RoPE, returns post-RoPE fp32 [V,128]
    (bf16-rounded at both stages, like the reference model code)."""
    k = k_pre.float()
    k = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + EPS)
    k = (k * w.float()).to(torch.bfloat16).float()
    cpos = positions // cr * cr
    cos, sin = cos_sin[cpos].float().chunk(2, -1)  # [V,32] each
    even, odd = k[:, 64::2], k[:, 65::2]
    ne = (even * cos - odd * sin).to(torch.bfloat16).float()
    no = (odd * cos + even * sin).to(torch.bfloat16).float()
    rot = torch.stack((ne, no), -1).flatten(1)  # [V,64]
    return torch.cat((k[:, :64], rot), -1)  # [V,128]


def quant_fp8_ref(x):
    """Per-token fp8: pow2 scale, matches the kernel's expressions
    exactly (scale = 2^ceil(log2(absmax/448)))."""
    absmax = x.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    exponent = torch.ceil(torch.log2(absmax * (1.0 / 448.0)))
    inv = torch.exp2(-exponent)
    codes = (x * inv).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(
        torch.uint8
    )
    return codes, torch.exp2(exponent).squeeze(-1).contiguous()


def quant_int8_ref(x):
    """Per-token INT8 (single 128-wide block), RNE rounding (torch.round)
    and scale computed as amax / 127 — both match the kernel's fp32
    magic-number path expression for expression."""
    amax = x.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    inv = 127.0 / amax
    q = (x * inv).round().clamp(-127.0, 127.0).to(torch.int8)
    return q, (amax.squeeze(-1) / 127.0).contiguous()


def quant_fp4_ref(x):
    """MXFP4: ue8m0 per 32 elements + E2M1 nibble pairs (reference
    boundaries, ties to even), even index -> low nibble."""
    V = x.shape[0]
    xg = x.float().view(V, 4, 32)
    amax = xg.abs().amax(-1, keepdim=True).clamp(min=6 * (2**-126))
    log2_ratio = (amax * (1.0 / 6.0)).log2().ceil().clamp(-127.0, 127.0)
    scale = log2_ratio.exp2()
    xs = (xg / scale).clamp(-6.0, 6.0)
    ax = xs.abs()
    code = torch.zeros_like(ax, dtype=torch.int32)
    code = torch.where(ax > 0.25, 1, code)
    code = torch.where(ax >= 0.75, 2, code)
    code = torch.where(ax > 1.25, 3, code)
    code = torch.where(ax >= 1.75, 4, code)
    code = torch.where(ax > 2.5, 5, code)
    code = torch.where(ax >= 3.5, 6, code)
    code = torch.where(ax > 5.0, 7, code)
    sign = ((xs.view(torch.int32) >> 31) & 1).to(torch.uint8)
    nib = code.to(torch.uint8) | (sign << 3)
    nf = nib.view(V, 128)
    packed = (nf[:, 0::2] | (nf[:, 1::2] << 4)).contiguous()
    return packed, (log2_ratio + 127.0).squeeze(-1).to(torch.uint8).contiguous()


def dequant_fp8(codes, scale):
    return codes.view(torch.float8_e4m3fn).float() * scale[:, None]


def dequant_int8(codes, scales):
    return codes.float() * scales[:, None]


def dequant_fp4(packed, ue8m0):
    V = packed.shape[0]
    lo = (packed & 0xF).long()
    hi = (packed >> 4).long()
    tab = torch.tensor(E2M1_VALS, device=packed.device)
    vals = torch.where((lo & 8) > 0, -tab[lo & 7], tab[lo & 7])
    vals2 = torch.where((hi & 8) > 0, -tab[hi & 7], tab[hi & 7])
    x = torch.stack((vals, vals2), -1).view(V, 4, 32)
    scale = torch.exp2(ue8m0.float() - 127.0)[:, :, None]
    return (x * scale).view(V, 128)


def cos_sim(a, b):
    """Cosine similarity in fp64; all-zero vs all-zero counts as a match."""
    if a.numel() == 0:
        return 1.0
    a = a.flatten().double()
    b = b.flatten().double()
    na, nb = a.norm(), b.norm()
    if na == 0 or nb == 0:
        return 1.0 if bool((a - b).abs().max() == 0) else 0.0
    return ((a * b).sum() / (na * nb)).item()


# ---------------------------------------------------------------- helpers
def tp_for(T):
    """Launcher's tokens-per-program heuristic (must stay in sync)."""
    return 8 if T >= 2048 else (4 if T >= 256 else 1)


def eff_bytes(T, V, mode, tp):
    """Logical traffic: per-token positions+slots (16B) for all T rows;
    per valid token k_pre (256B) + cos_sin row (256B) + amortized weight
    (256B/TP) + value write + scale write."""
    vb, sd = LAYOUT[mode]
    return T * 16 + V * (256 + 256 + 256 // tp + vb + sd)


def timed_us(fn, reps=15, per_win=8):
    """Burst timing: per_win back-to-back launches inside each event
    window (amortizes event-record overhead), median over reps windows
    (sustains clocks, resists contention spikes on a shared machine)."""
    for _ in range(per_win * 2):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts[i].record()
        for _ in range(per_win):
            fn()
        ends[i].record()
    torch.cuda.synchronize()
    us = sorted(
        s.elapsed_time(e) * 1000 / per_win for s, e in zip(starts, ends)
    )
    return us[len(us) // 2]


def build_case(
    T,
    cr,
    mode,
    block_size=16,
    seed=3,
    cache_dtype=torch.float32,
    positions_mode="random",
    slots_mode="randperm",
    zero_k=False,
):
    """Standard case builder (mirrors the official test's adversarial
    layout): NaN rows for non-emitting tokens, zero rows, tiny rows,
    sparse slots (-1), scrambled slot order, 0xA5 page padding."""
    vb, sd = LAYOUT[mode]
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    k_pre = torch.randn(
        T + 3, HEAD, dtype=torch.bfloat16, device=DEVICE, generator=g
    )
    w = torch.randn(HEAD, dtype=torch.bfloat16, device=DEVICE, generator=g)
    if zero_k:
        k_pre[:] = 0
    if positions_mode == "random":
        positions = torch.randint(
            0, MAX_POS, (T + 3,), dtype=torch.int64, device=DEVICE, generator=g
        )
    else:  # official test style: contiguous positions from 7
        positions = torch.arange(7, T + 10, device=DEVICE, dtype=torch.int64)
    angles = torch.randn(MAX_POS, 32, device=DEVICE, generator=g)
    cos_sin = (
        torch.cat((angles.cos(), angles.sin()), -1)
        .to(cache_dtype)
        .contiguous()
    )
    if slots_mode == "randperm":
        slots = torch.randperm(2 * T, device=DEVICE, generator=g)[:T].clone()
        slots[5::7] = -1
    elif slots_mode == "all_skip":
        slots = torch.full((T,), -1, dtype=torch.int64, device=DEVICE)
    else:  # dense identity
        slots = torch.arange(T, device=DEVICE)
    valid = (slots >= 0) & ((positions[:T] + 1) % cr == 0)
    if not zero_k:
        k_pre[3::7] = 0
        k_pre[8::11] *= 1e-10
        k_pre[:T][~valid] = float("nan")
    k_pre[T:] = float("nan")
    positions[T:] = MAX_POS + 10  # rows past num_tokens: never dereferenced
    num_blocks = (2 * T + block_size - 1) // block_size + 1
    page = block_size * (vb + sd)
    storage = torch.full(
        (num_blocks, page + 128), 0xA5, dtype=torch.uint8, device=DEVICE
    )
    cache = storage.as_strided(
        (num_blocks, block_size, vb + sd), (storage.stride(0), vb + sd, 1)
    )
    return dict(
        k_pre=k_pre,
        positions=positions,
        cos_sin=cos_sin,
        w=w,
        slots=slots,
        valid=valid,
        storage=storage,
        cache=cache,
        vb=vb,
        sd=sd,
        block_size=block_size,
        cr=cr,
    )


def gather_out(c):
    """Read back the valid tokens' (values, scales) with linear offsets."""
    s = c["slots"][c["valid"]]
    bs, vb, sd = c["block_size"], c["vb"], c["sd"]
    bids = s // bs
    offs = s % bs
    v_off = offs[:, None] * vb + torch.arange(vb, device=DEVICE)
    s_off = bs * vb + offs[:, None] * sd + torch.arange(sd, device=DEVICE)
    act_v = c["storage"][bids[:, None], v_off].clone()
    act_s = c["storage"][bids[:, None], s_off].clone()
    # every byte outside the valid write set must still be 0xA5
    mask = torch.ones_like(c["storage"], dtype=torch.bool)
    mask[bids[:, None], v_off] = False
    mask[bids[:, None], s_off] = False
    pad_ok = bool((c["storage"][mask] == 0xA5).all().item())
    return act_v, act_s, pad_ok


def run_fn(c, mode):
    return lambda: indexer_k_norm_rope_store(
        c["k_pre"],
        c["positions"],
        c["cos_sin"],
        c["w"],
        EPS,
        c["cache"],
        c["slots"],
        c["cr"],
        mode == "fp4",
        mode == "int8",
    )


def check_case(cid, mode, c, reps=20, expect_codes=None, note=""):
    """Run one case: correctness vs torch reference + timing + bandwidth."""
    global PASSED, FAILED
    T = c["slots"].numel()
    V = int(c["valid"].sum())
    fn = run_fn(c, mode)
    fn()
    torch.cuda.synchronize()

    kp = c["k_pre"][:T][c["valid"]]
    pos = c["positions"][:T][c["valid"]]
    ideal = norm_rope_ref(kp, pos, c["cos_sin"], c["w"], c["cr"])
    if mode == "fp8":
        rv, rs = quant_fp8_ref(ideal)
        exp_deq = dequant_fp8(rv, rs)
    elif mode == "int8":
        rv, rs = quant_int8_ref(ideal)
        exp_deq = dequant_int8(rv, rs)
    else:
        rv, rs = quant_fp4_ref(ideal)
        exp_deq = dequant_fp4(rv, rs)

    act_v, act_s, pad_ok = gather_out(c)

    # diagnostics: byte exactness (with documented zero-sign tolerance)
    if mode == "fp8":
        a_cmp, r_cmp = act_v.clone(), rv.clone()
        a_cmp[a_cmp == 128] = 0  # -0.0 e4m3 == +0.0
        r_cmp[r_cmp == 128] = 0
        codes_exact = torch.equal(a_cmp, r_cmp)
        scales_exact = torch.equal(act_s, rs.view(torch.uint8))
        # no-clamp invariant: |x * 2^-e| <= 448 exactly, so e4m3 NaN
        # encodings (0x7F / 0xFF) can never appear in the value bytes.
        nan_free = not bool(((act_v == 0x7F) | (act_v == 0xFF)).any().item())
    elif mode == "int8":
        codes_exact = torch.equal(act_v, rv.view(torch.uint8))
        ai = act_s.view(torch.float32).view(torch.int32).flatten()
        ri = rs.view(torch.int32).flatten()
        scales_exact = (
            bool((ai - ri).abs().max().item() == 0) if ai.numel() else True
        )
        nan_free = True
    else:
        a_cmp, r_cmp = act_v.clone(), rv.clone()
        for t_ in (a_cmp, r_cmp):  # E2M1 -0 nibble == +0 nibble
            for shift in (0, 4):
                zm = ((t_ >> shift) & 7) == 0
                t_ &= ~(zm.to(torch.uint8) << (shift + 3))
        codes_exact = torch.equal(a_cmp, r_cmp)
        scales_exact = torch.equal(act_s, rs)
        nan_free = True

    if mode == "fp8":
        act_deq = dequant_fp8(act_v, act_s.view(torch.float32).flatten())
    elif mode == "int8":
        act_deq = dequant_int8(
            act_v.view(torch.int8), act_s.view(torch.float32).flatten()
        )
    else:
        act_deq = dequant_fp4(act_v, act_s)

    c_ideal = cos_sim(act_deq, ideal)
    c_ref = cos_sim(act_deq, exp_deq)
    expect_ok = True if expect_codes is None else torch.equal(
        act_v, expect_codes
    )

    ok = (
        c_ideal >= THR_IDEAL[mode]
        and c_ref >= 0.9999
        and pad_ok
        and nan_free
        and expect_ok
    )

    us = timed_us(fn, reps)
    gbps = eff_bytes(T, V, mode, tp_for(T)) / (us * 1e-6) / 1e9
    PASSED += ok
    FAILED += not ok
    print(
        f"[{mode:>4}] {cid} T={T:5d} cr={c['cr']} bs={c['block_size']:3d} "
        f"cs={'bf16' if c['cos_sin'].dtype == torch.bfloat16 else 'f32'} "
        f"V={V:5d}  cos={c_ideal:.6f} ref={c_ref:.6f} "
        f"{'OK  ' if ok else 'FAIL'} {us:7.2f}us {gbps:7.1f}GB/s "
        f"codes={'Y' if codes_exact else 'N'} "
        f"scales={'Y' if scales_exact else 'N'} pad={'Y' if pad_ok else 'N'}"
        f"{' ' + note if note else ''}"
    )
    return ok, us, gbps


# ------------------------------------------------------------------ parts
def part_a():
    print("---- A: official padded-pages matrix "
          "(T x cr x mode x cache_dtype) " + "-" * 24)
    n = [0]

    def nxt():
        n[0] += 1
        return f"A{n[0]:02d}"

    for mode in MODES:
        for T in (1, 17, 257):
            for cr in (1, 2):
                for cdt in (torch.float32, torch.bfloat16):
                    c = build_case(
                        T, cr, mode, cache_dtype=cdt, positions_mode="arange"
                    )
                    check_case(nxt(), mode, c)


def part_b():
    print("---- B: robustness sweep (random positions/slots, NaN/zero/"
          "tiny rows) " + "-" * 30)
    n = [0]

    def nxt():
        n[0] += 1
        return f"B{n[0]:02d}"

    for mode in MODES:
        for T in (1, 17, 257, 1023, 4096):
            for cr in (1, 2):
                c = build_case(T, cr, mode, seed=100 + T + cr)
                check_case(nxt(), mode, c)


def part_c():
    print("---- C: paged block_size sweep " + "-" * 57)
    n = [0]

    def nxt():
        n[0] += 1
        return f"C{n[0]:02d}"

    for mode in ("fp8", "int8"):
        for bs in (16, 64, 128):
            for cr in (1, 2):
                T = 3 * bs + 5  # official gather test's shape recipe
                c = build_case(
                    T, cr, mode, block_size=bs, positions_mode="arange",
                    slots_mode="dense",
                )
                check_case(nxt(), mode, c)


def part_d():
    global PASSED, FAILED, SKIPPED
    print("---- D: edge cases " + "-" * 62)
    n = [0]

    def nxt():
        n[0] += 1
        return f"D{n[0]:02d}"

    # no valid token at all (every slot -1)
    for mode in MODES:
        c = build_case(64, 1, mode, slots_mode="all_skip")
        check_case(nxt(), mode, c, note="V=0")
    # all-zero k_pre
    for mode in MODES:
        c = build_case(64, 1, mode, zero_k=True)
        check_case(nxt(), mode, c, note="zero_k")
    # TP-heuristic boundaries (255->TP1, 256->TP4, 2047->TP4, 2048->TP8)
    for (T, mode) in (
        (255, "int8"), (256, "int8"), (2047, "int8"), (2048, "int8"),
        (2048, "fp8"),
    ):
        c = build_case(T, 1, mode, slots_mode="dense")
        check_case(nxt(), mode, c, note=f"TP={tp_for(T)}")
    # identity rope (cos=1, sin=0, position 0)
    c = build_case(128, 1, "int8", slots_mode="dense")
    c["cos_sin"] = torch.zeros(MAX_POS, 64, dtype=torch.float32, device=DEVICE)
    c["cos_sin"][:, :32] = 1.0
    c["positions"].zero_()
    c["valid"] = c["slots"] >= 0
    check_case(nxt(), "int8", c, note="identity_rope")
    # sequential positions with cr=2 (every other token emits, rope taken
    # at the group's first-token position)
    c = build_case(300, 2, "int8", positions_mode="arange",
                   slots_mode="dense")
    check_case(nxt(), "int8", c, note="seq_pos_cr2")
    # int8 RNE tie crafting: k_pre rows [8, 8, 8, 0.., +-16@64, +-8@66]
    # give variance exactly 512/128 = 4 -> rsqrt = 0.5 -> post-norm
    # [4, 4, 4, .., +-8@64, .., +-4@66] with identity rope; per-token
    # amax = 8 -> inv = 15.875 exact -> code 127@64 and exactly +-63.5
    # @0-2,66, which round-to-nearest-even must send to +-64 (never +-63).
    for sign in (1, -1):
        c = build_case(4, 1, "int8", slots_mode="dense")
        c["cos_sin"] = torch.zeros(
            MAX_POS, 64, dtype=torch.float32, device=DEVICE
        )
        c["cos_sin"][:, :32] = 1.0
        c["positions"].zero_()
        c["w"] = torch.ones(HEAD, dtype=torch.bfloat16, device=DEVICE)
        kp = torch.zeros(4, HEAD, dtype=torch.bfloat16, device=DEVICE)
        kp[:, 0:3] = 8.0
        kp[:, 64] = 16.0 * sign
        kp[:, 66] = 8.0 * sign
        c["k_pre"][:4] = kp
        c["valid"] = c["slots"] >= 0
        expect = torch.zeros(4, 128, dtype=torch.uint8, device=DEVICE)
        expect[:, 0:3] = 64  # RNE(+63.5) -> 64 (tie to even)
        expect[:, 64] = 127 if sign > 0 else 129  # +-127
        expect[:, 66] = 64 if sign > 0 else 192  # RNE(+-63.5) -> +-64
        check_case(
            nxt(), "int8", c, expect_codes=expect,
            note=f"rne_tie_{'p' if sign > 0 else 'n'}",
        )
    # fp8 extreme dynamic range (stress the no-clamp pow2 scaling)
    c = build_case(512, 1, "fp8", slots_mode="dense")
    with torch.no_grad():
        c["k_pre"][:512:4] *= 1e4
        c["k_pre"][1:512:4] *= 1e-4
    check_case(nxt(), "fp8", c, note="dyn_range_1e-8")
    # CUDA graph capture/replay (official test: mutate inputs, replay,
    # compare against a fresh eager run — bit-identical required);
    # capture unsupported on this platform -> SKIP, not FAIL
    for mode in MODES:
        if graph_replay_case(nxt(), mode) is None:
            SKIPPED += 1
    # num_tokens == 0 (launcher early-returns; must not touch memory)
    for mode in MODES:
        c = build_case(4, 1, mode)
        c["slots"] = c["slots"][:0].clone()
        c["valid"] = c["slots"] >= 0
        before = c["storage"].clone()
        run_fn(c, mode)()
        torch.cuda.synchronize()
        ok = torch.equal(before, c["storage"])
        PASSED += ok
        FAILED += not ok
        print(
            f"[{mode:>4}] {nxt()} T=    0 cr=1 bs= 16 cs=f32 V=    0  "
            f"{'OK  ' if ok else 'FAIL'} (empty launch, cache untouched)"
        )


def graph_replay_case(cid, mode):
    """Official cuda-graph test: capture, mutate inputs, replay, compare
    against a fresh eager run — bit-identical required."""
    torch.manual_seed(18)
    vb, sd = LAYOUT[mode]
    T = 19
    k_pre = torch.randn(T, HEAD, device=DEVICE, dtype=torch.bfloat16)
    w = torch.ones(HEAD, device=DEVICE, dtype=torch.bfloat16)
    positions = torch.arange(T, device=DEVICE)
    cos_sin = torch.zeros(32, 64, device=DEVICE)
    cos_sin[:, :32] = 1
    slots = torch.arange(T, device=DEVICE)
    cache = torch.full(
        (2, 16, vb + sd), 0xA5, dtype=torch.uint8, device=DEVICE
    )

    def run():
        indexer_k_norm_rope_store(
            k_pre, positions, cos_sin, w, EPS, cache, slots, 2,
            mode == "fp4", mode == "int8",
        )

    try:
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
    except Exception as ex:  # capture unsupported on this platform
        print(
            f"[{mode:>4}] {cid} graph replay: UNSUPPORTED "
            f"({type(ex).__name__}: {str(ex)[:60]}) — SKIP"
        )
        return None
    k_pre.neg_()
    slots.add_(4)
    cache.fill_(0xA5)
    graph.replay()
    replayed = cache.clone()
    cache.fill_(0xA5)
    run()
    torch.cuda.synchronize()
    ok = torch.equal(cache, replayed)
    global PASSED, FAILED
    PASSED += ok
    FAILED += not ok
    print(
        f"[{mode:>4}] {cid} T=  19 cr=2 bs= 16 cs=f32 V=    9  "
        f"{'OK  ' if ok else 'FAIL'} cuda-graph replay == eager "
        f"(bit-identical)"
    )
    return ok


def part_e():
    print("---- E: bandwidth table (launcher heuristic, burst median) "
          + "-" * 26)
    n = [0]
    peaks = {m: (0.0, 0.0, 0) for m in MODES}  # (gbps, us, T)

    def nxt():
        n[0] += 1
        return f"E{n[0]:02d}"

    for mode in MODES:
        for T in (1023, 4096, 16384, 65536):
            c = build_case(T, 1, mode, slots_mode="dense", seed=1234)
            _, us, gbps = check_case(nxt(), mode, c, reps=50)
            if gbps > peaks[mode][0]:
                peaks[mode] = (gbps, us, T)
    print("     peak effective bandwidth per mode (vs 1300 GB/s target):")
    for m in MODES:
        gbps, us, T = peaks[m]
        print(
            f"       {m:>4}: {gbps:8.1f} GB/s  @ T={T} ({us:.2f} us)  "
            f"{'REACHED' if gbps >= TARGET_GBPS else 'below'} target"
        )


def main():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES={vis!r}  "
          f"device_count={torch.cuda.device_count()}")
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"fp4_asm_default={_fp4_asm_default()}  "
          f"eps={EPS}  max_pos={MAX_POS}")
    print("=" * 108)
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    print("=" * 108)
    total = PASSED + FAILED + SKIPPED
    print(
        f"RESULT: {PASSED}/{total} passed, {FAILED} failed, "
        f"{SKIPPED} skipped (platform-unsupported)"
    )
    print("kernel trap check: no traps — every case synchronized cleanly")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
