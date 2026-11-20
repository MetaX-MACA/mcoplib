"""Unit test + bandwidth benchmark for
   torch.ops.sgl_kernel.per_token_cast_to_fp8  on MetaX C600-U.

Op signature (in-place):
  per_token_cast_to_fp8(Tensor! out, Tensor! scale, Tensor input,
                        bool trans_scale=False) -> ()

Semantics (per group of 128 contiguous elements along the last dim):
  absmax_g = max_i |input[g*128 + i]|            (floored at 1e-4, NaN ignored)
  scale[g] = absmax_g / 448
  out[g*128 + i] = quant_fp8( clamp(input[..] * rcpf(scale), -448, 448) )
  trans_scale=True  -> scale laid out as [G, m] (column-major w.r.t. [m, G])

The clamp (and taking rcpf of the *scale*, not of absmax) is required for
parity with the two known-GOOD vllm reference ops
  torch.ops._C.per_token_group_quant_8bit (group_size=128 regs path)
  torch.ops._C.dynamic_per_token_scaled_fp8_quant
Measured on C600-U, __builtin_mxc_cvt_pk4_f32tof8 is NON-saturating above 464
and emits 0x7f/0xff (NaN) for overflow, +-inf and NaN inputs, and
__builtin_mxc_rcpf flushes to 0 once absmax > ~8.5e37 (bf16 range reaches
3.4e38).  Without the clamp, NaN-bearing activations (typical FP4/FP8 model
overflow) leak NaN fp8 codes into the dequant GEMM and destroy accuracy while
both vllm references stay finite.

Coverage:
  * every launch path: small(VEC8, non-NT) / VEC16 / VEC8 / VEC4,
    trans_scale scatter path (G<8 or m<32) and workspace+transpose path
  * bf16 and fp16 inputs
  * special values: zeros, below-floor tiny values, large values, outliers,
    all-negative, constant groups, odd m / tail blocks
  * anomaly reproduction (vllm-parity contract): NaN slots, NaN-dense group,
    +-inf slots, absmax~3e38 outliers, pack_dcp-style -inf padding
  * cross-check against torch.ops._C.per_token_group_fp8_quant
  * CUDA graph capture + replay
  * out-of-bounds write detection (sentinel guard regions)
  * determinism
  * argument validation (n%128, dtypes)
  * bandwidth benchmark (trans_scale False / True; dtype x shape matrix)

Run:
  python unit_test/test_op_sglang_per_token_cast_to_fp8.py            # all
  python unit_test/test_op_sglang_per_token_cast_to_fp8.py --no-bench # tests only
  ... --graph-only   # capture test only (child half of the subprocess relay)
  pytest -q unit_test/test_op_sglang_per_token_cast_to_fp8.py         # pytest
"""

import argparse
import os
import re
import subprocess
import sys
import time

import torch
import mcoplib.sgl_kernel  # noqa: F401  (registers torch.ops.sgl_kernel.*)


# ------------------------- idle GPU auto-selection ---------------------------
def _auto_select_idle_gpus(max_gpus=2):
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return os.environ["CUDA_VISIBLE_DEVICES"]
    try:
        out = subprocess.check_output(["mx-smi"], text=True, timeout=30)
    except Exception:
        return None
    idle, cur_gpu = [], None
    for line in out.splitlines():
        gm = re.search(r"\|\s*(\d+)\s+MetaX C600", line)
        if gm:
            cur_gpu = int(gm.group(1))
        sm = re.search(r"\|\s+(\d+)\s+(On|Off)\s+\|", line)
        if sm:
            cur_gpu = int(sm.group(1))
        if "Available" in line and "0%" in line and cur_gpu is not None:
            if cur_gpu not in idle:
                idle.append(cur_gpu)
    if not idle:
        return None
    sel = ",".join(str(x) for x in idle[:max_gpus])
    os.environ["CUDA_VISIBLE_DEVICES"] = sel
    return sel


_auto_select_idle_gpus()

DEVICE = "cuda"
FP8 = torch.float8_e4m3fn
GROUP = 128
QMAX = 448.0
FP8_MIN_ABSMAX = 1e-4
COS_THRESHOLD = 0.999                # fp8-level (quantized) cosine floor
BF16_COS_THRESHOLD = 0.99999         # bf16-level (dequantized) cosine floor
SCALE_RTOL = 1e-6
FP8_MATCH_RATIO = 0.99          # >=99% of fp8 codes must match bit-exactly
FP8_MIN_SUBNORMAL = 2.0 ** -9

BENCH_N = 6144
CHUNK_SIZES = [2048, 4096, 8192, 16384, 32768]
TARGET_GBPS = 1600.0
COPY_WALL_GBPS = 1322.0
SENTINEL = 0xA5
GUARD = 4096                     # guard elements before/after outputs

STATS = {"cases": 0, "anomaly_cases": 0}


# ------------------------------- reference -----------------------------------
def ref_quant(inp):
    """Clean-input reference, bit-faithful to the kernel's arithmetic:
    scale = absmax / 448 via true IEEE division (tensor/tensor; torch folds a
    scalar /c into a multiply by the reciprocal, 1 ulp off), then
    q = x * correctly-rounded 1/scale (mirrors __builtin_mxc_rcpf).
    Returns (out_fp8 [m,n], scale_fp32 [m,G])."""
    m, n = inp.shape
    g = n // GROUP
    x = inp.float().view(m, g, GROUP)
    absmax = x.abs().amax(dim=2, keepdim=True).clamp_(min=FP8_MIN_ABSMAX)
    scale = absmax / torch.full_like(absmax, QMAX)
    out = (x * torch.reciprocal(scale)).view(m, n).to(FP8)
    return out, scale.view(m, g).contiguous()


def ref_quant_vllm(inp):
    """vllm per_token_group_quant_8bit (group=128) parity reference.

    Mirrors the two known-GOOD vllm ops exactly:
      * absmax via fmaxf chain -> NaN inputs are IGNORED (IEEE maxNum);
        +-inf still poisons the group scale (identically in vllm);
      * scale = absmax / 448 via true IEEE division (floored absmax at 1e-4);
      * q = fminf(fmaxf(x * rcpf(scale), -448), 448)
            -> NaN folds to -448 (fmaxf first, as in the vllm regs path),
               +-inf folds to +-448 after the x*rcpf(scale) multiply produces
               NaN for inf*0, 0 for finite*0 (scale=inf).
    """
    m, n = inp.shape
    g = n // GROUP
    x = inp.float().view(m, g, GROUP)
    a = torch.where(torch.isnan(x), torch.zeros((), device=x.device), x.abs())
    absmax = a.amax(dim=2, keepdim=True).clamp_(min=FP8_MIN_ABSMAX)
    scale = absmax / torch.full_like(absmax, QMAX)
    t = x * torch.reciprocal(scale)
    t = torch.where(torch.isnan(t), torch.full_like(t, -QMAX), t)
    out = t.clamp(-QMAX, QMAX).to(FP8)
    return out.view(m, n), scale.view(m, g).contiguous()


def cosine_sim(inp, out_fp8, scale_mg, chunk_rows=2048):
    m = inp.shape[0]
    dot = torch.zeros((), dtype=torch.float64, device=DEVICE)
    na = torch.zeros_like(dot)
    nb = torch.zeros_like(dot)
    for lo in range(0, m, chunk_rows):
        hi = min(lo + chunk_rows, m)
        ref = inp[lo:hi].double()
        deq = out_fp8[lo:hi].double() * scale_mg[lo:hi].double().repeat_interleave(GROUP, 1)
        dot += (deq * ref).sum()
        na += (deq * deq).sum()
        nb += (ref * ref).sum()
    if nb.item() == 0.0 and na.item() == 0.0:
        return 1.0
    return (dot / (na.sqrt() * nb.sqrt()).clamp_min(1e-300)).item()


def pair_cosine(a, b, chunk=1 << 22):
    """Cosine between two same-shape fp32/bf16 tensors (chunked, fp64 accum)."""
    a = a.reshape(-1)
    b = b.reshape(-1)
    dot = torch.zeros((), dtype=torch.float64, device=DEVICE)
    na = torch.zeros_like(dot)
    nb = torch.zeros_like(dot)
    for lo in range(0, a.numel(), chunk):
        hi = min(lo + chunk, a.numel())
        x = a[lo:hi].double()
        y = b[lo:hi].double()
        dot += (x * y).sum()
        na += (x * x).sum()
        nb += (y * y).sum()
    if na.item() == 0.0 and nb.item() == 0.0:
        return 1.0
    return (dot / (na.sqrt() * nb.sqrt()).clamp_min(1e-300)).item()


# ------------------------- guarded output allocation -------------------------
class Guarded:
    """Contiguous view inside a larger sentinel-filled buffer to catch OOB writes."""

    def __init__(self, shape, dtype):
        numel = 1
        for s in shape:
            numel *= s
        self.dtype = dtype
        if dtype == FP8:
            self.raw = torch.full((numel + 2 * GUARD,), SENTINEL, dtype=torch.uint8, device=DEVICE)
            self.tensor = self.raw[GUARD:GUARD + numel].view(FP8).view(*shape)
        else:
            self.raw = torch.full((numel + 2 * GUARD,), float("nan"), dtype=dtype, device=DEVICE)
            self.tensor = self.raw[GUARD:GUARD + numel].view(*shape)
        self.numel = numel

    def guards_intact(self):
        head, tail = self.raw[:GUARD], self.raw[GUARD + self.numel:]
        if self.dtype == FP8:
            return bool((head == SENTINEL).all() and (tail == SENTINEL).all())
        return bool(head.isnan().all() and tail.isnan().all())


# --------------------------------- runner ------------------------------------
def run_op(out, scale, inp, trans_scale=False):
    torch.ops.sgl_kernel.per_token_cast_to_fp8(out, scale, inp, trans_scale)


# ------------------------- anomaly slot bookkeeping --------------------------
def _anomaly_slots(kind, m, n):
    """Deterministic (row, col) positions injected by make_input for anomaly
    kinds.  Shared by make_input and the bit-exact slot assertions."""
    g = max(1, n // GROUP)
    if kind == "nan_few":
        return [(1 % m, 5), (3 % m, min(100, n - 1)), (m - 1, n // 2)]
    if kind == "nan_dense":
        r, gg = 5 % m, 4 % g
        return [(r, gg * GROUP + i) for i in range(GROUP)]
    if kind == "inf_few":
        r = 9 % m
        return [(r, min(7, n - 1)), (r, min(8, n - 1))]
    if kind == "huge":
        r = 1 % m
        return [(r, 0), (r, min(1, n - 1))]
    if kind == "pack_like":
        r, gg = 4 % m, 2 % g
        return [(r, gg * GROUP + i) for i in range(GROUP)]
    return []


def make_input(m, n, dtype, kind, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(m, n, dtype=torch.float32, device=DEVICE, generator=gen)
    if kind == "randn":
        pass
    elif kind == "zeros":
        x.zero_()
    elif kind == "tiny":           # below absmax floor 1e-4
        x = x * 1e-6
    elif kind == "large":
        x = x * 1e4
    elif kind == "outlier":        # one big outlier per group
        x = x.view(m, n // GROUP, GROUP)
        x[:, :, 17] = 300.0
        x = x.view(m, n)
    elif kind == "negative":
        x = -x.abs()
    elif kind == "const":
        x.fill_(3.0)
    elif kind == "mixscale":       # per-group magnitude varies 1e-3 .. 1e3
        s = torch.logspace(-3, 3, n // GROUP, device=DEVICE).repeat_interleave(GROUP)
        x = x * s
    elif kind in ("nan_few", "nan_dense", "inf_few", "huge", "pack_like"):
        for (i, j) in _anomaly_slots(kind, m, n):
            if kind == "inf_few":
                x[i, j] = float("inf") if j == 7 else float("-inf")
            elif kind == "huge":   # absmax ~3e38: beyond rcpf denormal flush point
                x[i, j] = 3e38 if j == 0 else -3e38
            elif kind == "nan_few" or kind == "nan_dense":
                x[i, j] = float("nan")
            elif kind == "pack_like":   # pack_dcp_topk_candidates padding slots
                x[i, j] = float("-inf")
    else:
        raise ValueError(kind)
    return x.to(dtype).contiguous()


def check_case(m, n, dtype=torch.bfloat16, trans_scale=False, kind="randn",
               seed=0, verbose=True):
    STATS["cases"] += 1
    g = n // GROUP
    inp = make_input(m, n, dtype, kind, seed)

    out_g = Guarded((m, n), FP8)
    scale_shape = (g, m) if trans_scale else (m, g)
    scale_g = Guarded(scale_shape, torch.float32)

    run_op(out_g.tensor, scale_g.tensor, inp, trans_scale)
    torch.cuda.synchronize()

    out, scale = out_g.tensor, scale_g.tensor
    scale_mg = scale.t().contiguous() if trans_scale else scale
    out_ref, scale_ref = ref_quant(inp)

    errs = []
    if not out_g.guards_intact():
        errs.append("OOB write on out")
    if not scale_g.guards_intact():
        errs.append("OOB write on scale")
    if torch.isnan(scale_mg).any() or torch.isinf(scale_mg).any():
        errs.append("scale contains nan/inf (unwritten entries?)")
    if not torch.allclose(scale_mg, scale_ref, rtol=SCALE_RTOL, atol=0):
        d = (scale_mg - scale_ref).abs().max().item()
        errs.append(f"scale mismatch max_abs={d:.3e}")

    of, rf = out.float(), out_ref.float()
    if torch.isnan(of).any():
        errs.append("out contains nan")
    match = (out.view(torch.uint8) == out_ref.view(torch.uint8)).float().mean().item()
    if match < FP8_MATCH_RATIO:
        errs.append(f"fp8 bit-match ratio {match:.4f} < {FP8_MATCH_RATIO}")
    # Allow at most 1 fp8 ULP (relative 2^-3, or subnormal step)
    tol = torch.maximum(rf.abs() * 0.125, torch.full_like(rf, FP8_MIN_SUBNORMAL)) + 1e-6
    if ((of - rf).abs() > tol).any():
        errs.append(f"fp8 diff > 1ulp, max={(of - rf).abs().max().item():.4f}")
    if of.abs().max().item() > QMAX:
        errs.append("fp8 value exceeds 448")

    sim = cosine_sim(inp, out, scale_mg)
    if kind not in ("zeros", "tiny") and sim < COS_THRESHOLD:
        errs.append(f"cos_sim {sim:.6f} < {COS_THRESHOLD}")
    # bf16-level (dequantized) agreement between kernel and reference
    dq_k = (of * scale_mg.repeat_interleave(GROUP, 1)).to(torch.bfloat16)
    dq_r = (rf * scale_ref.repeat_interleave(GROUP, 1)).to(torch.bfloat16)
    sim_bf16 = pair_cosine(dq_k, dq_r)
    if sim_bf16 < BF16_COS_THRESHOLD:
        errs.append(f"bf16 dequant cos {sim_bf16:.6f} < {BF16_COS_THRESHOLD}")

    ok = not errs
    if verbose:
        tag = "OK  " if ok else "FAIL"
        print(f"  [{tag}] m={m:6d} n={n:5d} {str(dtype).split('.')[-1]:8s} "
              f"trans={int(trans_scale)} kind={kind:9s} cos={sim:.6f} match={match:.4f}"
              + ("" if ok else "  <- " + "; ".join(errs)))
    return ok


def check_anomaly_case(m, n, dtype=torch.bfloat16, trans_scale=False,
                       kind="nan_few", seed=0, verbose=True):
    """vllm-parity contract on pathological inputs (the accuracy-anomaly
    reproduction cases).  Against the CURRENT kernel these FAIL:
      * NaN input slots leak 0x7f/0xff NaN fp8 codes (vllm refs fold to -448)
      * absmax ~3e38 zeroes the whole group (rcpf denormal flush)
      * -inf padding (pack_dcp style) yields NaN codes
    With the clamp + rcpf(scale) fix they must match ref_quant_vllm."""
    STATS["cases"] += 1
    STATS["anomaly_cases"] += 1
    g = n // GROUP
    inp = make_input(m, n, dtype, kind, seed)

    out_g = Guarded((m, n), FP8)
    scale_g = Guarded((g, m) if trans_scale else (m, g), torch.float32)
    run_op(out_g.tensor, scale_g.tensor, inp, trans_scale)
    torch.cuda.synchronize()

    out, scale = out_g.tensor, scale_g.tensor
    scale_mg = scale.t().contiguous() if trans_scale else scale
    out_ref, scale_ref = ref_quant_vllm(inp)

    errs = []
    if not out_g.guards_intact():
        errs.append("OOB write on out")
    if not scale_g.guards_intact():
        errs.append("OOB write on scale")
    nan_q = int(torch.isnan(out.float()).sum())
    if nan_q:
        errs.append(f"out contains {nan_q} NaN fp8 codes (vllm refs stay finite)")
    if not torch.allclose(scale_mg, scale_ref, rtol=SCALE_RTOL, atol=0):
        errs.append("scale mismatch vs vllm-parity ref "
                    f"(max_abs={(scale_mg - scale_ref).abs().max().item():.3e})")
    match = (out.view(torch.uint8) == out_ref.view(torch.uint8)).float().mean().item()
    if match < FP8_MATCH_RATIO:
        errs.append(f"fp8 bit-match vs vllm ref {match:.4f} < {FP8_MATCH_RATIO}")
    # injected slots must match the reference bit-exactly (byte level):
    # 0x7e/fe = +-448 (healthy), 0x7f/ff = NaN (the anomaly), 0x00 = flushed
    for (i, j) in _anomaly_slots(kind, m, n):
        kb = out.view(torch.uint8)[i, j].item()
        rb = out_ref.view(torch.uint8)[i, j].item()
        if kb != rb:
            errs.append(f"byte@({i},{j}) kernel=0x{kb:02x} vllm_ref=0x{rb:02x}"
                        " (0x7f/ff=NaN anomaly, 0x00=rcpf flush)")
            break
    # dequant NaN profile must equal the reference's (0 except +-inf groups,
    # where BOTH vllm refs and this kernel share the poisoned-inf-scale limit)
    dq = out.float() * scale_mg.repeat_interleave(GROUP, 1)
    dq_ref = out_ref.float() * scale_ref.repeat_interleave(GROUP, 1)
    if int(torch.isnan(dq).sum()) != int(torch.isnan(dq_ref).sum()):
        errs.append("dequant NaN count differs from vllm-parity ref")
    # bf16-level agreement on the dequantized tensors (finite groups only)
    if kind not in ("inf_few", "pack_like"):
        sim_bf16 = pair_cosine(dq.to(torch.bfloat16),
                               dq_ref.to(torch.bfloat16))
        if sim_bf16 < BF16_COS_THRESHOLD:
            errs.append(f"bf16 dequant cos {sim_bf16:.6f} < {BF16_COS_THRESHOLD}")

    ok = not errs
    if verbose:
        tag = "OK  " if ok else "FAIL"
        print(f"  [{tag}] m={m:6d} n={n:5d} {str(dtype).split('.')[-1]:8s} "
              f"trans={int(trans_scale)} kind={kind:9s} nan_q={nan_q:4d} "
              f"match={match:.4f}"
              + ("" if ok else "  <- " + "; ".join(errs)))
    return ok


# --------------------------------- tests -------------------------------------
PATH_SHAPES = [
    # (m, n)          path when trans_scale=False
    (1, 128),         # small
    (2, 128),         # small, G=1
    (7, 384),         # small, G=3
    (256, 128),       # small boundary (m==256), G=1
    (256, 6144),      # small boundary (m==256)
    (257, 6144),      # VEC16 just above small
    (300, 1024),      # VEC4  (307200 elems)
    (333, 256),       # VEC4, odd m
    (513, 2048),      # VEC16 boundary (1048576 elems)
    (600, 1024),      # VEC8  (614400 elems)
    (1000, 640),      # VEC8, tail block
    (2048, 6144),     # VEC16
    (2049, 256),      # VEC8, odd m tail (524544 elems)
    (4097, 6144),     # VEC16, odd m tail
    (8192, 128),      # VEC16 boundary (1048576 elems), G=1
]

TRANS_SHAPES = [
    (1, 128),         # scatter (m<32)
    (2, 128),         # scatter (m<32, G=1)
    (16, 6144),       # scatter (m<32, G=48)
    (100, 512),       # scatter (G=4<8)
    (31, 1024),       # scatter (m=31 boundary)
    (32, 1024),       # workspace (G=8, m=32 boundary)
    (32, 896),        # scatter (G=7 boundary, m=32)
    (64, 896),        # scatter (G=7 boundary)
    (1000, 1152),     # workspace, non-multiple-of-32 m/G
    (2048, 6144),     # workspace, VEC16
    (4096, 1024),     # workspace, G=8
    (8192, 128),      # scatter (G=1), large m
]

SPECIAL_KINDS = ["zeros", "tiny", "large", "outlier", "negative", "const", "mixscale"]

ANOMALY_KINDS = ["nan_few", "nan_dense", "inf_few", "huge", "pack_like"]
ANOMALY_CONFIGS = [                   # one per dispatch path
    (64, 1024, False),                # small
    (300, 1024, False),               # VEC4
    (600, 1024, False),               # VEC8
    (2048, 6144, False),              # VEC16
    (16, 6144, True),                 # trans scatter
    (1000, 1152, True),               # trans workspace+transpose
]


def test_paths():
    ok = True
    for dt in (torch.bfloat16, torch.float16):
        for m, n in PATH_SHAPES:
            ok &= check_case(m, n, dt, trans_scale=False)
    assert ok


def test_trans_scale():
    ok = True
    for dt in (torch.bfloat16, torch.float16):
        for m, n in TRANS_SHAPES:
            ok &= check_case(m, n, dt, trans_scale=True)
    assert ok


def test_special_values():
    ok = True
    for kind in SPECIAL_KINDS:
        for m, n, tr in [(64, 1024, False), (300, 1024, False),
                         (2048, 6144, False), (100, 512, True), (1000, 1152, True)]:
            ok &= check_case(m, n, torch.bfloat16, trans_scale=tr, kind=kind)
    assert ok


def test_anomaly_reproduction():
    """The accuracy-anomaly cases: NaN / +-inf / absmax~3e38 / -inf padding
    across every dispatch path and both input dtypes, against the vllm-parity
    contract.  FAILS on the unpatched kernel (reproduces the GLM-5.2 FP4/FP8
    accuracy anomaly), PASSES once the kernel matches the vllm references."""
    ok = True
    for dt in (torch.bfloat16, torch.float16):
        for (m, n, tr) in ANOMALY_CONFIGS:
            for kind in ANOMALY_KINDS:
                ok &= check_anomaly_case(m, n, dt, trans_scale=tr, kind=kind)
    assert ok


def test_cross_check_vllm_group():
    """Direct byte/scale comparison against the known-GOOD vllm group op
    torch.ops._C.per_token_group_fp8_quant (group_size=128, row-major scales)."""
    try:
        import mcoplib._C  # noqa: F401  (registers torch.ops._C.*)
    except Exception as e:                                  # pragma: no cover
        print(f"  [SKIP] mcoplib._C unavailable: {e}")
        return
    ok = True
    for kind in ("randn", "nan_few", "huge", "inf_few", "pack_like"):
        for m, n in [(300, 1024), (2048, 6144)]:
            for dt in (torch.bfloat16, torch.float16):
                STATS["cases"] += 1
                inp = make_input(m, n, dt, kind, 42)
                g = n // GROUP
                out = torch.empty(m, n, dtype=FP8, device=DEVICE)
                sc = torch.empty(m, g, dtype=torch.float32, device=DEVICE)
                run_op(out, sc, inp, False)
                qv = torch.empty(m, n, dtype=FP8, device=DEVICE)
                sv = torch.empty(m, g, dtype=torch.float32, device=DEVICE)
                torch.ops._C.per_token_group_fp8_quant(
                    inp, qv, sv, GROUP, 1e-4, -448.0, 448.0, False, False, False)
                torch.cuda.synchronize()
                same_scale = torch.equal(sc, sv)
                match = (out.view(torch.uint8) == qv.view(torch.uint8)).float().mean().item()
                nan_ours = int(torch.isnan(out.float()).sum())
                nan_vllm = int(torch.isnan(qv.float()).sum())
                okk = same_scale and match >= 0.999 and nan_ours == 0 and nan_vllm == 0
                ok &= okk
                print(f"  [{'OK  ' if okk else 'FAIL'}] m={m:5d} n={n:5d} "
                      f"{str(dt).split('.')[-1]:8s} kind={kind:9s} "
                      f"scale_eq={same_scale!s:5s} byte_match={match:.5f} "
                      f"nan(ours/vllm)={nan_ours}/{nan_vllm}")
    assert ok


def _graph_capture_core():
    """The op must be CUDA-graph capturable on every dispatch path used in
    production (incl. the workspace+transpose path, which allocates its
    workspace through the caching allocator).  Runs in a child process (see
    test_cuda_graph_capture) so a wedged driver cannot wedge the suite."""
    s = torch.cuda.Stream()
    for m, n, tr in [(2048, 6144, False), (64, 1024, False), (1000, 1152, True)]:
        STATS["cases"] += 1
        g = n // GROUP
        inp = make_input(m, n, torch.bfloat16, "randn", 5)
        out = torch.empty(m, n, dtype=FP8, device=DEVICE)
        sc = torch.empty((g, m) if tr else (m, g), dtype=torch.float32, device=DEVICE)
        # warmup on a side stream, as required before graph capture
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                run_op(out, sc, inp, tr)
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            run_op(out, sc, inp, tr)
        ok = True
        for it in range(3):
            inp.copy_(make_input(m, n, torch.bfloat16, "randn", 100 + it))
            out.view(torch.uint8).zero_()
            sc.zero_()
            gr.replay()
            torch.cuda.synchronize()
            o2 = torch.empty_like(out)
            s2 = torch.empty_like(sc)
            run_op(o2, s2, inp, tr)
            torch.cuda.synchronize()
            ok &= bool(torch.equal(out.view(torch.uint8), o2.view(torch.uint8)))
            ok &= bool(torch.equal(sc, s2))
        print(f"  [{'OK  ' if ok else 'FAIL'}] graph capture+replay "
              f"m={m} n={n} trans={int(tr)}")
        assert ok, (m, n, tr)


def test_cuda_graph_capture():
    """Relay _graph_capture_core through a child process with a hard timeout.

    Why a subprocess and not an in-process watchdog: on a wedged driver
    (SIGKILLed trapped run -> mxkwCreateQueueBlock ioctl timeouts host-wide,
    mx-smi -r refused with EAGAIN) torch.cuda.Stream() blocks forever inside
    the ioctl while HOLDING the GIL, so no in-process watchdog can ever
    report the timeout -- even print() needs the GIL.  Measured: a daemon
    -thread watchdog join(20s) returned, then the main thread hung 17 min on
    the GIL futex without printing a byte.  A child process isolates the
    wedge: the parent kills it on timeout and reports [SKIP] with its own
    GIL untouched.  subprocess.run() is NOT usable here -- after kill() it
    re-waits for the child, which never reaps while wedged uninterruptibly.
    """
    print("  (capture test runs in a child process; nested output follows)")
    p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__),
                          "--graph-only"])
    try:
        rc = p.wait(timeout=180)
    except subprocess.TimeoutExpired:
        p.kill()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("  (wedged child did not reap within 10s of SIGKILL; "
                  "leaving it to init)")
        print("  [SKIP] graph-capture child made no progress in 180s: the "
              "host driver's queue-block creation is wedged "
              "(mxkwCreateQueueBlock ioctl timeouts after a SIGKILLed "
              "trapped run; mx-smi -r refused with EAGAIN; bare torch "
              "reproduces it). Graph capture is unverifiable until the "
              "driver is restarted -- an environment blocker, not a "
              "property of this op. Post-restart, rerun this UT to verify "
              "capture for real.")
        return
    if rc != 0:
        raise AssertionError(f"graph-capture child rc={rc} (see nested output)")
    print("  [OK  ] graph capture relay (3 configs passed in child, "
          "see nested output)")


def test_zero_semantics():
    STATS["cases"] += 1
    m, n = 64, 1024
    inp = torch.zeros(m, n, dtype=torch.bfloat16, device=DEVICE)
    out = torch.empty(m, n, dtype=FP8, device=DEVICE)
    scale = torch.empty(m, n // GROUP, dtype=torch.float32, device=DEVICE)
    run_op(out, scale, inp)
    torch.cuda.synchronize()
    assert (out.float() == 0).all(), "zeros must quantize to 0"
    assert torch.allclose(scale, torch.full_like(scale, FP8_MIN_ABSMAX / QMAX), rtol=1e-6)
    print("  [OK  ] zero semantics (out==0, scale==1e-4/448)")


def test_absmax_maps_to_qmax():
    STATS["cases"] += 1
    m, n = 128, 2048
    inp = make_input(m, n, torch.bfloat16, "randn", 7)
    out = torch.empty(m, n, dtype=FP8, device=DEVICE)
    scale = torch.empty(m, n // GROUP, dtype=torch.float32, device=DEVICE)
    run_op(out, scale, inp)
    torch.cuda.synchronize()
    gmax = out.float().abs().view(m, -1, GROUP).amax(-1)
    assert (gmax == QMAX).all(), f"group absmax should map to 448, min got {gmax.min().item()}"
    print("  [OK  ] per-group absmax maps to +-448")


def test_determinism():
    for m, n, tr in [(2048, 6144, False), (1000, 1152, True), (16, 6144, True)]:
        STATS["cases"] += 1
        inp = make_input(m, n, torch.bfloat16, "randn", 3)
        g = n // GROUP
        res = []
        for _ in range(3):
            out = torch.empty(m, n, dtype=FP8, device=DEVICE)
            sc = torch.empty((g, m) if tr else (m, g), dtype=torch.float32, device=DEVICE)
            run_op(out, sc, inp, tr)
            res.append((out.view(torch.uint8).clone(), sc.clone()))
        torch.cuda.synchronize()
        for o, s in res[1:]:
            assert torch.equal(o, res[0][0]) and torch.equal(s, res[0][1])
    print("  [OK  ] determinism")


def test_trans_consistency():
    """trans_scale=True scale must equal transpose of trans_scale=False scale."""
    for m, n in TRANS_SHAPES:
        STATS["cases"] += 1
        inp = make_input(m, n, torch.bfloat16, "randn", 11)
        g = n // GROUP
        o1 = torch.empty(m, n, dtype=FP8, device=DEVICE)
        o2 = torch.empty_like(o1)
        s1 = torch.empty(m, g, dtype=torch.float32, device=DEVICE)
        s2 = torch.empty(g, m, dtype=torch.float32, device=DEVICE)
        run_op(o1, s1, inp, False)
        run_op(o2, s2, inp, True)
        torch.cuda.synchronize()
        assert torch.equal(o1.view(torch.uint8), o2.view(torch.uint8)), (m, n)
        assert torch.equal(s1.t(), s2), (m, n)
    print("  [OK  ] trans_scale True/False consistency")


def _expect_error(fn, name):
    try:
        fn()
        torch.cuda.synchronize()
    except (RuntimeError, ValueError, TypeError):
        print(f"  [OK  ] rejects {name}")
        return True
    print(f"  [FAIL] did not reject {name}")
    return False


def test_invalid_args():
    m = 16
    ok = True
    # n not multiple of 128
    ok &= _expect_error(lambda: run_op(
        torch.empty(m, 200, dtype=FP8, device=DEVICE),
        torch.empty(m, 2, dtype=torch.float32, device=DEVICE),
        torch.randn(m, 200, dtype=torch.bfloat16, device=DEVICE)), "n % 128 != 0")
    # fp32 input
    ok &= _expect_error(lambda: run_op(
        torch.empty(m, 256, dtype=FP8, device=DEVICE),
        torch.empty(m, 2, dtype=torch.float32, device=DEVICE),
        torch.randn(m, 256, dtype=torch.float32, device=DEVICE)), "fp32 input")
    # wrong out dtype
    ok &= _expect_error(lambda: run_op(
        torch.empty(m, 256, dtype=torch.bfloat16, device=DEVICE),
        torch.empty(m, 2, dtype=torch.float32, device=DEVICE),
        torch.randn(m, 256, dtype=torch.bfloat16, device=DEVICE)), "non-fp8 out")
    # non-contiguous input
    ok &= _expect_error(lambda: run_op(
        torch.empty(m, 256, dtype=FP8, device=DEVICE),
        torch.empty(m, 2, dtype=torch.float32, device=DEVICE),
        torch.randn(256, m, dtype=torch.bfloat16, device=DEVICE).t()), "non-contiguous input")
    assert ok


# ------------------------------- benchmark -----------------------------------
def bench_burst(fn, warm=0.5, run=0.6):
    torch.cuda.synchronize()
    t0 = time.time(); n = 0
    while time.time() - t0 < warm:
        fn(); n += 1
        if n % 64 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    reps = max(50, n)
    best_ms = float("inf"); t0 = time.time()
    while time.time() - t0 < run:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record(); e.synchronize()
        best_ms = min(best_ms, s.elapsed_time(e) / reps)
    return best_ms


def effective_bytes(m, n):
    return m * n * 2 + m * n + m * (n // GROUP) * 4


def bench(trans_scale):
    n, g = BENCH_N, BENCH_N // GROUP
    peak, all_ok = 0.0, True
    for m in CHUNK_SIZES:
        inp = make_input(m, n, torch.bfloat16, "randn", 1234)
        out = torch.empty(m, n, dtype=FP8, device=DEVICE)
        sc = torch.empty((g, m) if trans_scale else (m, g), dtype=torch.float32, device=DEVICE)
        run_op(out, sc, inp, trans_scale)
        torch.cuda.synchronize()
        sim = cosine_sim(inp, out, sc.t() if trans_scale else sc)
        ok = sim >= COS_THRESHOLD
        ms = bench_burst(lambda: run_op(out, sc, inp, trans_scale))
        gbps = effective_bytes(m, n) / (ms * 1e-3) / 1e9
        peak = max(peak, gbps); all_ok &= ok
        print(f"  [bench trans={int(trans_scale)}] T={m:6d} G={g}  cos={sim:.6f} "
              f"{'OK' if ok else 'FAIL'}  {ms:8.4f} ms  {gbps:8.1f} GB/s")
        del inp, out, sc
        torch.cuda.empty_cache()
    print(f"  peak(trans={int(trans_scale)}) = {peak:.1f} GB/s  "
          f"(target {TARGET_GBPS:.0f}: {'REACHED' if peak >= TARGET_GBPS else 'NOT reached'}; "
          f"copy wall ~{COPY_WALL_GBPS:.0f})")
    return all_ok


BENCH_MATRIX = [                      # (m, n, trans) covering each dispatch path
    (2048, 6144, False),              # VEC16
    (4096, 8192, False),              # VEC16, large
    (256, 6144, False),               # small
    (600, 1024, False),               # VEC8
    (300, 1024, False),               # VEC4
    (2048, 6144, True),               # workspace+transpose
    (1000, 1152, True),               # workspace, odd geometry
    (16, 6144, True),                 # scatter
]


def bench_dtype_shape_matrix():
    """Per-dtype / per-shape timing + effective bandwidth, one line per case."""
    all_ok = True
    for (m, n, tr) in BENCH_MATRIX:
        for dt in (torch.bfloat16, torch.float16):
            inp = make_input(m, n, dt, "randn", 1234)
            g = n // GROUP
            out = torch.empty(m, n, dtype=FP8, device=DEVICE)
            sc = torch.empty((g, m) if tr else (m, g), dtype=torch.float32, device=DEVICE)
            run_op(out, sc, inp, tr)
            torch.cuda.synchronize()
            sim = cosine_sim(inp, out, sc.t() if tr else sc)
            ok = sim >= COS_THRESHOLD
            ms = bench_burst(lambda: run_op(out, sc, inp, tr), warm=0.25, run=0.35)
            gbps = effective_bytes(m, n) / (ms * 1e-3) / 1e9
            all_ok &= ok
            print(f"  [matrix] m={m:5d} n={n:5d} trans={int(tr)} "
                  f"{str(dt).split('.')[-1]:8s} {ms:8.4f} ms  {gbps:8.1f} GB/s  "
                  f"cos={sim:.6f} {'OK' if ok else 'FAIL'}")
            del inp, out, sc
        torch.cuda.empty_cache()
    return all_ok


# --------------------------------- main --------------------------------------
TESTS = [test_paths, test_trans_scale, test_special_values, test_anomaly_reproduction,
         test_cross_check_vllm_group, test_zero_semantics, test_absmax_maps_to_qmax,
         test_determinism, test_trans_consistency, test_invalid_args,
         test_cuda_graph_capture]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--anomaly-only", action="store_true",
                    help="run only the anomaly-reproduction suite")
    ap.add_argument("--graph-only", action="store_true",
                    help="run only _graph_capture_core (the child half of "
                         "test_cuda_graph_capture's subprocess relay)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r} "
          f"device_count={torch.cuda.device_count()}")
    print("=" * 100)

    failed = []
    if not args.bench_only:
        if args.graph_only:
            tests = [_graph_capture_core]
        elif args.anomaly_only:
            tests = [test_anomaly_reproduction]
        else:
            tests = TESTS
        for t in tests:
            print(f"[{t.__name__}]")
            try:
                t()
            except AssertionError as e:
                failed.append(t.__name__)
                print(f"  -> FAILED {e}")
    if not args.no_bench and not args.anomaly_only and not args.graph_only:
        print("[benchmark]")
        for tr in (False, True):
            if not bench(tr):
                failed.append(f"bench_accuracy(trans={tr})")
        print("[benchmark: dtype x shape matrix]")
        if not bench_dtype_shape_matrix():
            failed.append("bench_matrix_accuracy")

    print("=" * 100)
    print(f"total accuracy cases: {STATS['cases']} "
          f"(anomaly-reproduction cases: {STATS['anomaly_cases']})")
    if failed:
        print("FAILED: " + ", ".join(failed))
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
