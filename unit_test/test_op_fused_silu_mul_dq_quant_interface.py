"""Unit test + bandwidth benchmark for mcoplib.op.fused_silu_mul_dq_quant on MetaX C600-U.

Op (3-arg, in-place, backward compatible):
  fused_silu_mul_dq_quant(out, scale, input)

Semantics (per token row):
  input  : [T, hidden_size]  bf16          (hidden_size = FULL last dim)
  inner  = hidden_size // 2
  gate   = input[:, :inner]                 (first half)
  up     = input[:, inner:]                 (second half)
  silu(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
  gate_up    = up * silu(gate)              # [T, inner]  (fp32 math)
  bm  = max_c |gate_up[t, c]|               # per-token absmax
  scale[t] = bm / QMAX                       (QMAX = 127 int8 / 448 fp8)
  out[t]   = quant( gate_up[t] * (QMAX / bm) )   -> int8 (round) / fp8_e4m3 (RNE, sat +-448)
  => dequant  out[t] * scale[t]  ~=  gate_up[t]

  out    : [T, inner]  int8 or float8_e4m3fn
  scale  : [T, 1]      fp32

NEW (4-arg overload, same Python name):
  fused_silu_mul_dq_quant(out, scale, input, per_token_scale)

Quant mode is judged by the SHAPE of `scale`:
  * scale.numel() == num_tokens      -> per_token mode: identical to the 3-arg
    call; `per_token_scale` is NOT written (left untouched).
  * scale.numel() == inner           -> smooth_scale mode (new isolated kernel
    launch_silu_mul_smooth_quan_no_mask): scale is the per-channel fp32
    smooth_scale [inner], matching smooth_per_token_dynamic_quant applied to
    swiglu(input):
      smoothed[t,c]      = silu(x1[t,c]) * x2[t,c] * smooth_scale[c]   (fp32)
      absmax[t]          = max_c |smoothed[t,c]|
      per_token_scale[t] = absmax[t] / QMAX          (fp32, [T] OUTPUT)
      out[t,c]           = quant(smoothed * QMAX / absmax)
    => dequant out[t] * per_token_scale[t] ~= smoothed[t]

Config (from request): dtype=bfloat16, dst_dtype in {int8, float8_e4m3fn},
  hidden_size in {12800, 13824}, all 37 num_tokens.
Accuracy: cosine similarity (fp64 accumulate) of BOTH outputs vs fp32 reference:
  per_token mode: dequant vs gate_up;
  smooth mode:    dequant vs smoothed AND per_token_scale vs ref per_token_scale
                  (absmax_ref / QMAX).
  Thresholds: int8/fp8 >= 0.999 for both outputs. A dedicated value spot check
  verifies >= 8 typical output values per (hidden, dst) at quantum tolerance.

Extra smooth-mode checks: fp16/fp32 input dtype sweep (accuracy), 4-arg
per_token dispatch check (per_token_scale sentinel untouched), and CUDA Graph
capture/replay smoke tests for both modes (implementation must be capturable).

Effective bandwidth (2R + 1W):
  per_token: bytes = T * hidden_size * 2  +  T * inner * out_elem_bytes  (+ T*4 scale)
  smooth   : bytes = T * hidden_size * 2  +  T * inner * out_elem_bytes
                      + T*4 per_token_scale + inner*4 smooth_scale

GPU: mx-smi is parsed to auto-pick a free "Available" die with lowest memory use;
  CUDA_VISIBLE_DEVICES is set BEFORE torch import.

Run (inside container):
  /opt/conda/bin/python unit_test/test_op_fused_silu_mul_dq_quant_interface.py
"""

import os
import re
import subprocess


# ------------------------------------------------------------------ GPU select
def _pick_free_gpu():
    """Parse `mx-smi` and return the index of a free (Available) die with the
    lowest used memory. Falls back to '1' if parsing fails. Runs BEFORE torch import."""
    try:
        out = subprocess.check_output(["mx-smi"], stderr=subprocess.DEVNULL,
                                      timeout=30).decode("utf-8", "ignore")
    except Exception as e:
        print(f"[gpu-select] mx-smi failed ({e}); defaulting to CUDA_VISIBLE_DEVICES=1")
        return "1"

    lines = out.splitlines()
    cands = []          # (used_mib, index)
    prev_idx = None
    for ln in lines:
        parts = ln.split("|")
        if len(parts) < 6:
            continue
        # A device-id line: field[2] starts with the integer GPU index.
        m_idx = re.match(r"\s*(\d+)\s", parts[2])
        if m_idx and "MiB" not in ln:
            prev_idx = int(m_idx.group(1))
        # A memory/state line: field[3] has "used/total MiB", field[4] the state.
        if "MiB" in parts[3]:
            mm = re.search(r"(\d+)\s*/\s*(\d+)\s*MiB", parts[3])
            state = parts[4].strip()
            if mm and prev_idx is not None and state.lower() == "available":
                cands.append((int(mm.group(1)), prev_idx))
            prev_idx = None
    if not cands:
        print("[gpu-select] no Available die parsed; defaulting to CUDA_VISIBLE_DEVICES=1")
        return "1"
    cands.sort()
    chosen = cands[0][1]
    print(f"[gpu-select] {len(cands)} Available dies; picked index {chosen} "
          f"(used {cands[0][0]} MiB)")
    return str(chosen)


if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pick_free_gpu()

import torch  # noqa: E402  (import after CUDA_VISIBLE_DEVICES is fixed)
import mcoplib.op as ops  # noqa: E402

DEVICE = "cuda"

# ---- config (from request) ----
IN_DTYPE = torch.bfloat16
HIDDEN_SIZES = [12800, 13824, 25600, 27648]  # FULL input last dim; inner = hidden // 2

# dst dtype -> (torch dtype, QMAX, out elem bytes, cos_sim threshold)
FLOAT8 = torch.float8_e4m3fn
DST_DTYPES = {
    "int8": (torch.int8, 127.0, 1, 0.999),
    "fp8":  (FLOAT8,     448.0, 1, 0.999),
}

NUM_TOKENS_CASES = [
    16, 64, 128, 256, 768, 896, 1024,
    1280,  8192, 10240, 12288, 14336, 16384,  28672, 30720, 32768, 65536,163840,409600
]

TARGET_GBPS = 1600.0                      # per_token path: user single-die target
ACCEPT_GBPS = 0.85 * TARGET_GBPS          # 85% acceptable (=1360)
DUAL_DIE_GBPS = 3200.0                     # user dual-die reference
SMOOTH_TARGET_GBPS = 1200.0                # smooth_scale path requirement (>= 1.2 TB/s)


def make_input(num_tokens, hidden_size, seed=1234, dtype=IN_DTYPE):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    # randn*2 gives a realistic silu dynamic range without saturating.
    return (torch.randn(num_tokens, hidden_size, dtype=IN_DTYPE,
                        device=DEVICE, generator=g) * 2.0).to(dtype)


def make_smooth_scale(inner, seed=777):
    """Per-channel fp32 smooth_scale [inner]. Realistic smoothquant scales are
    positive with a spread, so use rand in [0.1, 1.0]."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return (torch.rand(inner, dtype=torch.float32, device=DEVICE, generator=g) * 0.9 + 0.1)


def run_op(inp, out, scale):
    ops.fused_silu_mul_dq_quant(out, scale, inp)


def run_op_smooth(inp, out, smooth, per_token_scale):
    ops.fused_silu_mul_dq_quant(out, smooth, inp, per_token_scale)


def streaming_cosine(inp, out, scale, inner, chunk_rows=4096):
    """Chunked cosine similarity of op-dequant vs fp32 reference gate_up.
    ref  = up * (gate * sigmoid(gate)); deq = out.float() * scale (per-token)."""
    T = inp.shape[0]
    dot = torch.zeros((), dtype=torch.float64, device=DEVICE)
    na = torch.zeros((), dtype=torch.float64, device=DEVICE)
    nb = torch.zeros((), dtype=torch.float64, device=DEVICE)
    for lo in range(0, T, chunk_rows):
        hi = min(lo + chunk_rows, T)
        gate = inp[lo:hi, :inner].float()
        up = inp[lo:hi, inner:].float()
        gate_up = up * (gate * torch.sigmoid(gate))       # [c, inner] fp32 reference
        deq = out[lo:hi].float() * scale[lo:hi].float()    # broadcast [c,1]
        dot += (deq * gate_up).double().sum()
        na += (deq * deq).double().sum()
        nb += (gate_up * gate_up).double().sum()
    denom = (na.sqrt() * nb.sqrt()).clamp_min(1e-30)
    return (dot / denom).item()


def streaming_cosine_smooth(inp, smooth, out, pts, inner, qmax, chunk_rows=4096):
    """Chunked cosine similarity of BOTH smooth-mode outputs vs the fp32
    reference. Each chunk holds FULL rows, so the per-row absmax is complete
    within a chunk and a single pass suffices:
      cos_q: out * per_token_scale  vs  smoothed = swiglu * smooth_scale
      cos_s: per_token_scale        vs  absmax_ref / QMAX"""
    T = inp.shape[0]
    dot = torch.zeros((), dtype=torch.float64, device=DEVICE)
    na = torch.zeros((), dtype=torch.float64, device=DEVICE)
    nb = torch.zeros((), dtype=torch.float64, device=DEVICE)
    sd = torch.zeros((), dtype=torch.float64, device=DEVICE)
    sa = torch.zeros((), dtype=torch.float64, device=DEVICE)
    sb = torch.zeros((), dtype=torch.float64, device=DEVICE)
    for lo in range(0, T, chunk_rows):
        hi = min(lo + chunk_rows, T)
        gate = inp[lo:hi, :inner].float()
        up = inp[lo:hi, inner:].float()
        gate_up = up * (gate * torch.sigmoid(gate))
        smoothed = gate_up * smooth.view(1, -1)                 # fp32 reference values
        deq = out[lo:hi].float() * pts[lo:hi].view(-1, 1).float()
        dot += (deq * smoothed).double().sum()
        na += (deq * deq).double().sum()
        nb += (smoothed * smoothed).double().sum()
        s_op = pts[lo:hi].double()
        s_ref = (smoothed.abs().amax(-1) / qmax).double()       # ref per_token_scale
        sd += (s_op * s_ref).sum()
        sa += (s_op * s_op).sum()
        sb += (s_ref * s_ref).sum()
    denom_q = (na.sqrt() * nb.sqrt()).clamp_min(1e-30)
    denom_s = (sa.sqrt() * sb.sqrt()).clamp_min(1e-30)
    return (dot / denom_q).item(), (sd / denom_s).item()


def smooth_reference(inp, smooth, dst_dtype, qmax, chunk_rows=2048):
    """fp32 chunked reference of swiglu + smooth_per_token_dynamic_quant.
    Returns (quant_ref [T, inner] dst_dtype, per_token_scale_ref [T] fp32)."""
    T, hidden = inp.shape
    inner = hidden // 2
    quant_ref = torch.empty(T, inner, dtype=dst_dtype, device=DEVICE)
    pts_ref = torch.empty(T, dtype=torch.float32, device=DEVICE)
    is_int8 = dst_dtype == torch.int8
    for lo in range(0, T, chunk_rows):
        hi = min(lo + chunk_rows, T)
        gate = inp[lo:hi, :inner].float()
        up = inp[lo:hi, inner:].float()
        gate_up = up * (gate * torch.sigmoid(gate))            # [c, inner] fp32
        smoothed = gate_up * smooth.view(1, -1)                 # fp32 per-channel
        absmax = smoothed.abs().amax(-1, keepdim=True)          # [c, 1]
        inv = absmax.reciprocal()                               # 1 / absmax
        pts = inv * qmax                                        # quant factor
        q = (smoothed * pts).clamp(-qmax, qmax)
        if is_int8:
            q = q.round()                                       # int8: round-half-even
        quant_ref[lo:hi] = q.to(dst_dtype)                      # fp8: RNE + saturate
        pts_ref[lo:hi] = pts.reciprocal().view(-1)              # absmax / QMAX
    return quant_ref, pts_ref


def effective_bytes(num_tokens, hidden_size, inner, out_bytes):
    return num_tokens * hidden_size * 2 + num_tokens * inner * out_bytes + num_tokens * 4


def effective_bytes_smooth(num_tokens, hidden_size, inner, out_bytes):
    # read: input bf16 [T, hidden] + smooth_scale fp32 [inner]
    # write: out [T, inner] * out_bytes + per_token_scale fp32 [T]
    return (num_tokens * hidden_size * 2 + inner * 4
            + num_tokens * inner * out_bytes + num_tokens * 4)


def bench_case(num_tokens, hidden_size, dst_name, warmup=10, rep=50):
    dst_dtype, qmax, out_bytes, thr = DST_DTYPES[dst_name]
    inner = hidden_size // 2
    inp = make_input(num_tokens, hidden_size)
    out = torch.empty(num_tokens, inner, dtype=dst_dtype, device=DEVICE)
    scale = torch.empty(num_tokens, 1, dtype=torch.float32, device=DEVICE)

    # ---- accuracy ----
    run_op(inp, out, scale)
    torch.cuda.synchronize()
    sim = streaming_cosine(inp, out, scale, inner)

    # ---- bandwidth (back-to-back burst within one sync -> sustains DVFS) ----
    for _ in range(warmup):
        run_op(inp, out, scale)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        run_op(inp, out, scale)
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    median_ms = times_ms[len(times_ms) // 2]

    gbps = effective_bytes(num_tokens, hidden_size, inner, out_bytes) / (median_ms * 1e-3) / 1e9
    ok = "OK" if sim >= thr else "FAIL"
    print(f"[{dst_name:>4}] T={num_tokens:6d} H={hidden_size:5d} inner={inner:5d}  "
          f"cos_sim={sim:.6f} {ok:4s}  {median_ms:8.4f} ms  {gbps:8.1f} GB/s")
    return sim >= thr, gbps


def bench_case_smooth(num_tokens, hidden_size, dst_name, inp=None, warmup=10, rep=50):
    """Accuracy (BOTH outputs, cosine >= threshold) + bandwidth, smooth mode."""
    dst_dtype, qmax, out_bytes, thr = DST_DTYPES[dst_name]
    inner = hidden_size // 2
    if inp is None:
        inp = make_input(num_tokens, hidden_size)
    out = torch.empty(num_tokens, inner, dtype=dst_dtype, device=DEVICE)
    smooth = make_smooth_scale(inner)
    pts = torch.empty(num_tokens, dtype=torch.float32, device=DEVICE)

    # ---- accuracy: cosine of BOTH outputs ----
    run_op_smooth(inp, out, smooth, pts)
    torch.cuda.synchronize()
    sim_q, sim_s = streaming_cosine_smooth(inp, smooth, out, pts, inner, qmax)

    # ---- bandwidth (back-to-back burst within one sync -> sustains DVFS) ----
    for _ in range(warmup):
        run_op_smooth(inp, out, smooth, pts)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        run_op_smooth(inp, out, smooth, pts)
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    median_ms = times_ms[len(times_ms) // 2]

    gbps = effective_bytes_smooth(num_tokens, hidden_size, inner, out_bytes) / (median_ms * 1e-3) / 1e9
    ok = sim_q >= thr and sim_s >= thr
    tag = "OK" if ok else "FAIL"
    print(f"[{dst_name:>4}] T={num_tokens:6d} H={hidden_size:5d} inner={inner:5d}  "
          f"cos_q={sim_q:.6f} cos_s={sim_s:.6f} {tag:4s}  "
          f"{median_ms:8.4f} ms  {gbps:8.1f} GB/s")
    return ok, gbps


def smooth_value_spot_check():
    """Dedicated >= 8-typical-value check per (hidden, dst): compares kernel
    outputs elementwise against smooth_reference at quantum tolerance.
      int8: max |op - ref| <= 2 quant steps, and <= 1 step for >= 99.9%
      fp8 : dequant relative diff <= 2 quanta (2^-3 each) everywhere,
            <= 1 quantum for >= 99.9%
      per_token_scale: max relative error < 1e-5 (ulp-level agreement)"""
    print("---- smooth value spot check (16 rows, quantum tolerance) " + "-" * 14)
    ok_all = True
    for hidden_size in HIDDEN_SIZES:
        inner = hidden_size // 2
        for T in [16, 1024]:
            inp = make_input(T, hidden_size, seed=314)
            smooth = make_smooth_scale(inner)
            for dst_name in DST_DTYPES:
                dst_dtype, qmax, out_bytes, thr = DST_DTYPES[dst_name]
                out = torch.empty(T, inner, dtype=dst_dtype, device=DEVICE)
                pts = torch.empty(T, dtype=torch.float32, device=DEVICE)
                run_op_smooth(inp, out, smooth, pts)
                torch.cuda.synchronize()
                quant_ref, pts_ref = smooth_reference(inp, smooth, dst_dtype, qmax)

                if dst_name == "int8":
                    diff = (out.int() - quant_ref.int()).abs()
                    max_diff = diff.max().item()
                    frac_gt1 = (diff > 1).float().mean().item()
                    ok = (max_diff <= 2) and (frac_gt1 <= 0.001)
                    info = f"max|d|={max_diff} frac(>1)={frac_gt1:.5f}"
                else:
                    deq_op = out.float() * pts.view(-1, 1)
                    deq_ref = quant_ref.float() * pts_ref.view(-1, 1)
                    rel = ((deq_op - deq_ref).abs() /
                           deq_ref.abs().clamp_min(1e-6)).max().item()
                    rel1 = (((deq_op - deq_ref).abs() /
                             deq_ref.abs().clamp_min(1e-6)) > 0.125).float().mean().item()
                    ok = (rel <= 0.25) and (rel1 <= 0.001)
                    info = f"max rel={rel:.4f} frac(>1q)={rel1:.5f}"
                pts_rel = ((pts - pts_ref).abs() / pts_ref.abs().clamp_min(1e-6)).max().item()
                ok = ok and pts_rel < 1e-5
                ok_all &= ok
                tag = "OK" if ok else "FAIL"
                print(f"     T={T:5d} H={hidden_size} {dst_name:>4}: {info}  "
                      f"pts max rel={pts_rel:.2e}  {tag}")
    assert ok_all, "smooth value spot check FAILED"
    print("     smooth value spot check PASS (all hidden x dst x 2 shapes)")


def check_4arg_per_token_dispatch():
    """scale.numel()==num_tokens must route to the per_token 3-arg path and
    leave per_token_scale untouched (it is a smooth-mode-only output)."""
    print("---- 4-arg per_token dispatch check " + "-" * 40)
    T, hidden_size = 128, 12800
    inner = hidden_size // 2
    inp = make_input(T, hidden_size, seed=42)

    # smooth-mode call fills per_token_scale
    out4 = torch.empty(T, inner, dtype=torch.int8, device=DEVICE)
    pts = torch.full((T,), float("nan"), dtype=torch.float32, device=DEVICE)
    run_op_smooth(inp, out4, make_smooth_scale(inner), pts)
    torch.cuda.synchronize()
    assert not torch.isnan(pts).any(), "smooth mode did not fill per_token_scale"

    # 3-arg reference
    out3 = torch.empty(T, inner, dtype=torch.int8, device=DEVICE)
    scale = torch.empty(T, 1, dtype=torch.float32, device=DEVICE)
    run_op(inp, out3, scale)
    torch.cuda.synchronize()

    # 4-arg call with scale numel == T -> per_token mode; sentinel untouched
    sentinel = torch.full((T,), -12345.0, dtype=torch.float32, device=DEVICE)
    out4b = torch.empty(T, inner, dtype=torch.int8, device=DEVICE)
    scale4 = torch.empty(T, 1, dtype=torch.float32, device=DEVICE)
    ops.fused_silu_mul_dq_quant(out4b, scale4, inp, sentinel)
    torch.cuda.synchronize()

    ok_o = torch.equal(out4b, out3)
    ok_s = torch.equal(scale4, scale)
    ok_p = torch.equal(sentinel, torch.full_like(sentinel, -12345.0))
    print(f"     out bitwise-equal to 3-arg: {ok_o}, scale equal: {ok_s}, "
          f"per_token_scale sentinel untouched: {ok_p}")
    assert ok_o and ok_s and ok_p, "4-arg per_token dispatch check FAILED"
    print("     4-arg per_token dispatch check PASS")


def graph_capture_smoke():
    """Both modes must be CUDA-Graph capturable and replay bitwise-identical."""
    print("---- CUDA Graph capture/replay smoke test " + "-" * 34)
    T = 2048
    for hidden_size in HIDDEN_SIZES:
        inner = hidden_size // 2
        inp = make_input(T, hidden_size, seed=99)
        smooth = make_smooth_scale(inner)
        for dst_name in DST_DTYPES:
            dst_dtype = DST_DTYPES[dst_name][0]

            # --- smooth mode ---
            out = torch.empty(T, inner, dtype=dst_dtype, device=DEVICE)
            pts = torch.empty(T, dtype=torch.float32, device=DEVICE)
            run_op_smooth(inp, out, smooth, pts)            # warmup (required)
            torch.cuda.synchronize()
            out_eager, pts_eager = out.clone(), pts.clone()
            out.zero_(); pts.zero_()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                run_op_smooth(inp, out, smooth, pts)        # side-stream warmup
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run_op_smooth(inp, out, smooth, pts)        # capture
            g.replay()
            torch.cuda.synchronize()
            ok_q = torch.equal(out.view(torch.int8), out_eager.view(torch.int8))
            ok_s = torch.equal(pts, pts_eager)

            # --- per_token mode (3-arg, backward compat) ---
            outp = torch.empty(T, inner, dtype=dst_dtype, device=DEVICE)
            scale = torch.empty(T, 1, dtype=torch.float32, device=DEVICE)
            run_op(inp, outp, scale)
            torch.cuda.synchronize()
            outp_eager = outp.clone()
            outp.zero_()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                run_op(inp, outp, scale)
            torch.cuda.current_stream().wait_stream(s)
            g2 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g2):
                run_op(inp, outp, scale)
            g2.replay()
            torch.cuda.synchronize()
            ok_p = torch.equal(outp.view(torch.int8), outp_eager.view(torch.int8))

            print(f"     H={hidden_size} {dst_name:>4}: smooth replay bitwise q={ok_q} "
                  f"s={ok_s}, per_token replay bitwise={ok_p}")
            assert ok_q and ok_s and ok_p, f"graph capture failed H={hidden_size} {dst_name}"
    print("     CUDA Graph capture/replay smoke test PASS (both modes, all dst dtypes)")


def smooth_input_dtype_sweep():
    """Accuracy-only sweep over the remaining MOE_DISPATCH_FLOATING_TYPES
    inputs (fp16, fp32) on representative token counts. bf16 is covered by the
    full smooth matrix. Returns True iff all cases pass."""
    print("---- smooth mode input dtype sweep (fp16/fp32) " + "-" * 26)
    ok_all = True
    for hidden_size in HIDDEN_SIZES:
        inner = hidden_size // 2
        for in_dtype, dname in [(torch.float16, "fp16"), (torch.float32, "fp32")]:
            for T in [16, 1024, 8192]:
                inp = make_input(T, hidden_size, seed=555, dtype=in_dtype)
                for dst_name in DST_DTYPES:
                    dst_dtype, qmax, out_bytes, thr = DST_DTYPES[dst_name]
                    out = torch.empty(T, inner, dtype=dst_dtype, device=DEVICE)
                    smooth = make_smooth_scale(inner)
                    pts = torch.empty(T, dtype=torch.float32, device=DEVICE)
                    run_op_smooth(inp, out, smooth, pts)
                    torch.cuda.synchronize()
                    sim_q, sim_s = streaming_cosine_smooth(inp, smooth, out, pts,
                                                           inner, qmax)
                    ok = sim_q >= thr and sim_s >= thr
                    ok_all &= ok
                    tag = "OK" if ok else "FAIL"
                    print(f"     in={dname} [{dst_name:>4}] T={T:5d} H={hidden_size}  "
                          f"cos_q={sim_q:.6f} cos_s={sim_s:.6f} {tag}")
    assert ok_all, "smooth input dtype sweep FAILED"
    print("     smooth input dtype sweep PASS (fp16/fp32 x both dst x both hidden)")
    return ok_all


def main():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES={vis!r}  device_count={torch.cuda.device_count()}")
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"config: dtype=bf16 dst={{int8,fp8}} hidden_sizes={HIDDEN_SIZES} "
          f"num_tokens={len(NUM_TOKENS_CASES)} cases")
    print(f"per_token target single-die={TARGET_GBPS:.0f} GB/s  85%-accept={ACCEPT_GBPS:.0f} GB/s  "
          f"dual-die-ref={DUAL_DIE_GBPS:.0f} GB/s")
    print(f"smooth_scale target={SMOOTH_TARGET_GBPS:.0f} GB/s (max bandwidth must exceed 1.2 TB/s)")
    print("=" * 118)

    # ---- phase 0: dispatch + graph + value spot checks (fast, early signal) ----
    check_4arg_per_token_dispatch()
    graph_capture_smoke()
    smooth_value_spot_check()
    print("=" * 118)

    overall_pass = True
    peaks = {}
    for hidden_size in HIDDEN_SIZES:
        for dst_name in DST_DTYPES:
            thr = DST_DTYPES[dst_name][3]
            print(f"---- per_token (3-arg, backward compat) hidden_size={hidden_size}  "
                  f"dst_dtype={dst_name} (cos_sim threshold {thr}) " + "-" * 10)
            peak, dpass = 0.0, True
            for T in NUM_TOKENS_CASES:
                ok, gbps = bench_case(T, hidden_size, dst_name)
                peak = max(peak, gbps)
                dpass &= ok
            key = (hidden_size, dst_name)
            peaks[key] = (peak, dpass)
            print(f"     [{dst_name} H={hidden_size}] per_token peak = {peak:.1f} GB/s   "
                  f"accuracy {'ALL PASS' if dpass else 'FAILURES PRESENT'}")
            overall_pass &= dpass

    print("=" * 118)
    smooth_peaks = {}
    smooth_largest = {}
    for hidden_size in HIDDEN_SIZES:
        for dst_name in DST_DTYPES:
            thr = DST_DTYPES[dst_name][3]
            print(f"---- smooth_scale (4-arg) hidden_size={hidden_size}  dst_dtype={dst_name} "
                  f"(both cos thresholds {thr}) " + "-" * 14)
            peak, dpass = 0.0, True
            largest_gbps = 0.0
            for T in NUM_TOKENS_CASES:
                ok, gbps = bench_case_smooth(T, hidden_size, dst_name)
                peak = max(peak, gbps)
                largest_gbps = gbps
                dpass &= ok
            key = (hidden_size, dst_name)
            smooth_peaks[key] = (peak, dpass)
            smooth_largest[key] = largest_gbps        # last (largest) T case
            print(f"     [{dst_name} H={hidden_size}] smooth peak = {peak:.1f} GB/s  "
                  f"largest-T ({NUM_TOKENS_CASES[-1]}) = {largest_gbps:.1f} GB/s   "
                  f"accuracy {'ALL PASS' if dpass else 'FAILURES PRESENT'}")
            overall_pass &= dpass

    overall_pass &= smooth_input_dtype_sweep()

    print("=" * 118)
    print("per_token (3-arg) summary:")
    gpeak = 0.0
    for (hidden_size, dst_name), (peak, dpass) in peaks.items():
        gpeak = max(gpeak, peak)
        tag = ("REACHED" if peak >= TARGET_GBPS else
               "accept" if peak >= ACCEPT_GBPS else "below")
        print(f"H={hidden_size} {dst_name:>4}: peak {peak:8.1f} GB/s  ({tag} "
              f"single-die target {TARGET_GBPS:.0f})  accuracy {'PASS' if dpass else 'FAIL'}")
    print(f"per_token GLOBAL peak = {gpeak:.1f} GB/s  (dual-die datasheet reference {DUAL_DIE_GBPS:.0f} GB/s)")
    print("-" * 118)
    print("smooth_scale (4-arg) summary:")
    sgpeak = 0.0
    for (hidden_size, dst_name), (peak, dpass) in smooth_peaks.items():
        sgpeak = max(sgpeak, peak)
        tag = "REACHED" if peak >= SMOOTH_TARGET_GBPS else "below"
        print(f"H={hidden_size} {dst_name:>4}: peak {peak:8.1f} GB/s  largest-T "
              f"{smooth_largest[(hidden_size, dst_name)]:8.1f} GB/s  ({tag} smooth target "
              f"{SMOOTH_TARGET_GBPS:.0f})  accuracy {'PASS' if dpass else 'FAIL'}")
    print(f"smooth GLOBAL peak = {sgpeak:.1f} GB/s  (requirement: max bandwidth > "
          f"{SMOOTH_TARGET_GBPS:.0f} GB/s)")
    assert overall_pass, "accuracy check failed"
    if sgpeak < SMOOTH_TARGET_GBPS:
        print(f"SMOOTH BANDWIDTH TARGET NOT MET: {sgpeak:.1f} < {SMOOTH_TARGET_GBPS:.0f} GB/s")
    else:
        print("SMOOTH BANDWIDTH TARGET MET")


if __name__ == "__main__":
    main()
