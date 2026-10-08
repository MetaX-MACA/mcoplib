"""Unit test + bandwidth benchmark for mcoplib.op.fused_silu_mul_dq_quant on MetaX C600-U.

Op (3-arg, in-place):
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

Config (from request): dtype=bfloat16, dst_dtype in {int8, float8_e4m3fn},
  hidden_size in {12800, 13824}, all 37 num_tokens.
Accuracy: cosine similarity of op-dequant vs fp32 reference gate_up,
  general >= 0.9999, int8/fp8 >= 0.999. Bandwidth printed per dtype+shape.

Effective bandwidth (2R + 1W):
  bytes = T * hidden_size * 2  +  T * inner * out_elem_bytes  (+ T*4 scale, negligible)

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
HIDDEN_SIZES = [12800, 13824]           # FULL input last dim; inner = hidden // 2

# dst dtype -> (torch dtype, QMAX, out elem bytes, cos_sim threshold)
FLOAT8 = torch.float8_e4m3fn
DST_DTYPES = {
    "int8": (torch.int8, 127.0, 1, 0.999),
    "fp8":  (FLOAT8,     448.0, 1, 0.999),
}

NUM_TOKENS_CASES = [
    16, 64, 128, 256, 384, 512, 640, 768, 896, 1024,
    1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584,
    3840, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480,
    22528, 24576, 26624, 28672, 30720, 32768, 65536,
]

TARGET_GBPS = 1600.0                      # user single-die target
ACCEPT_GBPS = 0.85 * TARGET_GBPS          # 85% acceptable (=1360)
DUAL_DIE_GBPS = 3200.0                     # user dual-die reference


def make_input(num_tokens, hidden_size, seed=1234):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    # randn*2 gives a realistic silu dynamic range without saturating.
    return (torch.randn(num_tokens, hidden_size, dtype=IN_DTYPE,
                        device=DEVICE, generator=g) * 2.0)


def run_op(inp, out, scale):
    ops.fused_silu_mul_dq_quant(out, scale, inp)


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


def effective_bytes(num_tokens, hidden_size, inner, out_bytes):
    return num_tokens * hidden_size * 2 + num_tokens * inner * out_bytes + num_tokens * 4


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


def main():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES={vis!r}  device_count={torch.cuda.device_count()}")
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"config: dtype=bf16 dst={{int8,fp8}} hidden_sizes={HIDDEN_SIZES} "
          f"num_tokens={len(NUM_TOKENS_CASES)} cases")
    print(f"target single-die={TARGET_GBPS:.0f} GB/s  85%-accept={ACCEPT_GBPS:.0f} GB/s  "
          f"dual-die-ref={DUAL_DIE_GBPS:.0f} GB/s")
    print("=" * 108)

    overall_pass = True
    peaks = {}
    for hidden_size in HIDDEN_SIZES:
        for dst_name in DST_DTYPES:
            thr = DST_DTYPES[dst_name][3]
            print(f"---- hidden_size={hidden_size}  dst_dtype={dst_name} "
                  f"(cos_sim threshold {thr}) " + "-" * 30)
            peak, dpass = 0.0, True
            for T in NUM_TOKENS_CASES:
                ok, gbps = bench_case(T, hidden_size, dst_name)
                peak = max(peak, gbps)
                dpass &= ok
            key = (hidden_size, dst_name)
            peaks[key] = (peak, dpass)
            print(f"     [{dst_name} H={hidden_size}] peak = {peak:.1f} GB/s   "
                  f"accuracy {'ALL PASS' if dpass else 'FAILURES PRESENT'}")
            overall_pass &= dpass

    print("=" * 108)
    gpeak = 0.0
    for (hidden_size, dst_name), (peak, dpass) in peaks.items():
        gpeak = max(gpeak, peak)
        tag = ("REACHED" if peak >= TARGET_GBPS else
               "accept" if peak >= ACCEPT_GBPS else "below")
        print(f"H={hidden_size} {dst_name:>4}: peak {peak:8.1f} GB/s  ({tag} "
              f"single-die target {TARGET_GBPS:.0f})  accuracy {'PASS' if dpass else 'FAIL'}")
    print(f"GLOBAL peak = {gpeak:.1f} GB/s")
    print(f"dual-die datasheet reference: {DUAL_DIE_GBPS:.0f} GB/s")
    assert overall_pass, "accuracy check failed"


if __name__ == "__main__":
    main()
