"""Unit test + bandwidth benchmark for
   torch.ops.sgl_kernel.per_token_cast_to_fp8  on MetaX C600-U.

Op signature (in-place):
  per_token_cast_to_fp8(Tensor! out, Tensor! scale, Tensor input,
                        bool trans_scale=False) -> ()

Semantics (per group of 128 contiguous elements along the last dim):
  absmax_g = max_i |input[g*128 + i]|            (floored at 1e-4)
  scale[g] = absmax_g / 448                       (fp8_e4m3fn qmax = 448)
  out[g*128 + i] = quant_fp8( input[..] * 448/absmax_g )
  dequant: out.float() * scale  ~=  input.float()

Config (from request):
  out   : [chunked_size, 6144]  torch.float8_e4m3fn
  scale : [chunked_size, 48]    torch.float32
  input : [chunked_size, 6144]  torch.bfloat16
  chunked_size in [2048, 4096, 8192, 16384, 32768]

Accuracy: cosine similarity, fp8 threshold >= 0.999.
Bandwidth: (read bf16 + write fp8 + write scale) / median_burst_time.

Run:
  export CUDA_VISIBLE_DEVICES=<idle GPU(s)>
  python unit_test/test_op_sglang_per_token_cast_to_fp8.py
"""

import os
import subprocess
import torch
import mcoplib.sgl_kernel  # noqa: F401  (registers torch.ops.sgl_kernel.*)


# ------------------------- idle GPU auto-selection ---------------------------
def _auto_select_idle_gpus(max_gpus=2):
    """If CUDA_VISIBLE_DEVICES is unset, parse `mx-smi` and pick idle cards
    (GPU-Util 0% and state Available). Sets the env var in-process."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return os.environ["CUDA_VISIBLE_DEVICES"]
    try:
        out = subprocess.check_output(["mx-smi"], text=True, timeout=30)
    except Exception:
        return None
    idle = []
    # Rows look like: "| 0  MetaX ... | 10  ... | 0%  Disabled |"  and the util
    # line "| 0%          Available |". We track the most recent GPU index seen.
    cur_gpu = None
    import re
    for line in out.splitlines():
        m = re.search(r"\|\s*(\d+)\s+MetaX", line)
        if not m:
            m2 = re.search(r"\|\s+(\d+)\s+\w+\s+On|\|\s+(\d+)\s+\w+\s+Off", line)
        m_idx = re.match(r"\|\s+(\d+)\s+", line)
        # detect a line that declares a physical GPU index followed by Bus-id
        gm = re.search(r"\|\s*(\d+)\s+MetaX C600", line)
        if gm:
            cur_gpu = int(gm.group(1))
        # sub-die index lines: "|                  | 11          Off | ..."
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
N = 6144                       # hidden dim (fixed by request)
GROUPS = N // GROUP            # 48
FP8_MIN_ABSMAX = 1e-4          # matches kernel floor
COS_THRESHOLD = 0.999          # fp8 quantization threshold

CHUNK_SIZES = [2048, 4096, 8192, 16384, 32768]

TARGET_GBPS = 1600.0           # requested single-die aspiration
COPY_WALL_GBPS = 1322.0        # measured C600-U single-die copy (1R+1W) wall


def per_token_cast_to_fp8_ref(input_bf16):
    """FP64-clean torch reference. Returns (out_fp8, scale_fp32)."""
    m = input_bf16.shape[0]
    x = input_bf16.float().view(m, GROUPS, GROUP)
    absmax = x.abs().amax(dim=2, keepdim=True).clamp_(min=FP8_MIN_ABSMAX)  # [m,G,1]
    scale = (absmax / QMAX).view(m, GROUPS).contiguous()
    q = (x * (QMAX / absmax)).view(m, N)
    out = q.to(FP8)
    return out, scale


def cosine_sim_chunked(input_bf16, out_fp8, scale, chunk_rows=2048):
    """Cosine similarity between dequant(out) and original input, fp64 accum,
    row-chunked to bound memory."""
    m = input_bf16.shape[0]
    dot = torch.zeros((), dtype=torch.float64, device=DEVICE)
    na = torch.zeros((), dtype=torch.float64, device=DEVICE)
    nb = torch.zeros((), dtype=torch.float64, device=DEVICE)
    for lo in range(0, m, chunk_rows):
        hi = min(lo + chunk_rows, m)
        ref = input_bf16[lo:hi].float()                                   # [r,N]
        sc = scale[lo:hi].repeat_interleave(GROUP, dim=1)                 # [r,N]
        deq = out_fp8[lo:hi].float() * sc
        dot += (deq * ref).double().sum()
        na += (deq * deq).double().sum()
        nb += (ref * ref).double().sum()
    denom = (na.sqrt() * nb.sqrt()).clamp_min(1e-30)
    return (dot / denom).item()


def run_op(out, scale, inp):
    torch.ops.sgl_kernel.per_token_cast_to_fp8(out, scale, inp)


def bench_burst(fn, warm=0.5, run=0.6):
    torch.cuda.synchronize()
    import time
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


def effective_bytes(m):
    return m * N * 2 + m * N * 1 + m * GROUPS * 4      # read bf16 + write fp8 + scale


def bench_case(m, seed=1234):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    inp = torch.randn(m, N, dtype=torch.bfloat16, device=DEVICE, generator=g)
    out = torch.empty(m, N, dtype=FP8, device=DEVICE)
    scale = torch.empty(m, GROUPS, dtype=torch.float32, device=DEVICE)

    run_op(out, scale, inp)
    torch.cuda.synchronize()

    out_ref, scale_ref = per_token_cast_to_fp8_ref(inp)
    sim = cosine_sim_chunked(inp, out, scale)

    ms = bench_burst(lambda: run_op(out, scale, inp))
    gbps = effective_bytes(m) / (ms * 1e-3) / 1e9
    ok = "OK" if sim >= COS_THRESHOLD else "FAIL"
    print(f"[per_token_fp8] T={m:6d} groups={GROUPS:3d}  cos_sim={sim:.6f} {ok:4s}  "
          f"{ms:8.4f} ms  {gbps:8.1f} GB/s")
    return sim >= COS_THRESHOLD, gbps


def main():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES={vis!r}  device_count={torch.cuda.device_count()}")
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"config: N={N} groups={GROUPS} dtype=bf16->fp8_e4m3fn  "
          f"cos_threshold={COS_THRESHOLD}")
    print("=" * 92)

    peak = 0.0
    all_pass = True
    for m in CHUNK_SIZES:
        ok, gbps = bench_case(m)
        peak = max(peak, gbps)
        all_pass &= ok
    print("=" * 92)
    print(f"Accuracy: {'ALL PASS' if all_pass else 'FAILURES PRESENT'} "
          f"(threshold cos_sim >= {COS_THRESHOLD})")
    print(f"Peak bandwidth: {peak:.1f} GB/s  "
          f"(target {TARGET_GBPS:.0f} -> {'REACHED' if peak >= TARGET_GBPS else 'NOT reached'}; "
          f"copy wall ~{COPY_WALL_GBPS:.0f})")
    assert all_pass, "accuracy check failed"


if __name__ == "__main__":
    main()
