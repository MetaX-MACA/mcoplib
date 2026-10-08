"""
Unit test + bandwidth benchmark for `rms_norm_dynamic_per_token_quant`
(mcoplib.op) on MetaX C600-U.

Full config coverage (per task JSON):
  dtype       = bfloat16 (input / after_res / after_norm)
  dst_dtype   = ["int8", "float8"]           (float8 == float8_e4m3fn)
  output_mode = "res"                         (residual add enabled)
  add_residual= true
  hidden_size = [4096, 5120]                  (Config A H=4096, Config B adds 5120)
  num_tokens  = 16..65536 (37 values)

Op semantics (matches the CUDA kernel — fast AND slow paths agree):
  xr         = x + residual                       # residual add (after_res)
  rms        = rsqrt(mean(xr^2, dim=-1) + eps)    # note: rsqrt value
  after_norm = (xr / rms) * weight                # kernel divides by rms
  val        = after_norm * smooth
  scale      = amax(|val|, dim=-1) / 127.0        # per-token, SAME for int8 & fp8
  out        = quant(val / scale)                 # int8: round+clamp127; fp8: clamp448
  scales_out = scale

Accuracy gates:
  quantized out (int8 or fp8) : cosine(dequant, val)   >= 0.999
  bf16 intermediates          : cosine(after_res, xr)  >= 0.99999
                                cosine(after_norm, ref) >= 0.99999
Bandwidth: back-to-back burst, one sync; effective bytes = every global array
touched once (in: input+residual+weight+smooth; out: after_res+after_norm+out+scales).

Run on a FREE GPU only (mx-smi): e.g. CUDA_VISIBLE_DEVICES=10,11.
"""
import os
import time
import torch
import mcoplib.op as ops

DEVICE = "cuda"
EPS = 1e-6
COS_Q = 0.999        # int8 / fp8 quantized-output gate
COS_F = 0.99999      # bf16 intermediate (after_res / after_norm) gate
TARGET_GBPS = 1300.0
FP8_DTYPE = torch.float8_e4m3fn

HIDDEN_SIZES = [4096, 5120]
NUM_TOKENS = [16, 64, 128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536,
              1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096,
              6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528,
              24576, 26624, 28672, 30720, 32768, 65536]

# dst_dtype label -> torch out dtype
DST_DTYPES = [("int8", torch.int8), ("float8", FP8_DTYPE)]


def make_inputs(T, H, out_dtype, w_dtype=torch.bfloat16, seed=1234):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(T, H, dtype=torch.bfloat16, device=DEVICE, generator=g)
    residual = torch.randn(T, H, dtype=torch.bfloat16, device=DEVICE, generator=g)
    weight = (torch.randn(H, dtype=w_dtype, device=DEVICE, generator=g) * 0.1 + 1.0)
    smooth = (torch.rand(H, dtype=w_dtype, device=DEVICE, generator=g) + 0.5)
    out = torch.zeros(T, H, dtype=out_dtype, device=DEVICE)
    scales = torch.zeros(T, dtype=torch.float32, device=DEVICE)
    after_res = torch.zeros(T, H, dtype=torch.bfloat16, device=DEVICE)
    after_norm = torch.zeros(T, H, dtype=torch.bfloat16, device=DEVICE)
    return dict(x=x, residual=residual, weight=weight, smooth=smooth, out=out,
                scales=scales, after_res=after_res, after_norm=after_norm)


def run(t):
    ops.rms_norm_dynamic_per_token_quant(
        t["out"], t["x"], t["weight"], t["smooth"], t["scales"],
        EPS, t["after_res"], t["after_norm"], t["residual"])


def ref(t):
    xf = t["x"].float() + t["residual"].float()      # xr = x + residual
    wf = t["weight"].float()
    sf = t["smooth"].float()
    meansq = (xf * xf).mean(dim=-1, keepdim=True)
    rms = torch.rsqrt(meansq + EPS)
    after_norm = xf / rms * wf                        # kernel divides by rms
    val = after_norm * sf
    scale = val.abs().amax(dim=-1) / 127.0
    return xf, after_norm, val, scale


def cosine(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def bench(t, warm_s=1.0, time_s=0.6):
    run(t); torch.cuda.synchronize()
    w0 = time.perf_counter()
    n = 0
    while True:
        for _ in range(32):
            run(t)
        n += 32
        torch.cuda.synchronize()
        if time.perf_counter() - w0 >= warm_s:
            break
    per = (time.perf_counter() - w0) / n
    rep = max(32, int(time_s / max(per, 1e-6)))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        run(t)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / rep


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','?')}  "
          f"device={torch.cuda.get_device_name(0)}")
    print(f"{'dst':>7} {'T':>7} {'H':>6} {'cos_q':>9} {'cos_res':>9} "
          f"{'cos_nrm':>9} {'ms':>9} {'GB/s':>8}  status")
    print("-" * 82)

    peak = 0.0
    peak_by_dst = {}
    worst = {"q": 1.0, "res": 1.0, "nrm": 1.0}
    fails = []
    rows = []
    for (dname, odt) in DST_DTYPES:
        for H in HIDDEN_SIZES:
            for T in NUM_TOKENS:
                t = make_inputs(T, H, odt)
                run(t); torch.cuda.synchronize()

                xr, an_ref, val, scale_ref = ref(t)
                deq = t["out"].float() * t["scales"][:, None]
                cq = cosine(deq, val)
                cres = cosine(t["after_res"].float(), xr)
                cnrm = cosine(t["after_norm"].float(), an_ref)
                worst["q"] = min(worst["q"], cq)
                worst["res"] = min(worst["res"], cres)
                worst["nrm"] = min(worst["nrm"], cnrm)

                ms = bench(t)
                sz_in = 2  # bf16
                wsz = t["weight"].element_size()
                bytes_moved = (T * H * sz_in            # input
                               + T * H * sz_in          # residual
                               + T * H * sz_in          # after_res write
                               + T * H * sz_in          # after_norm write
                               + T * H * 1              # out (int8/fp8, 1 byte)
                               + T * 4                  # scales
                               + H * wsz                # weight
                               + H * wsz)               # smooth
                gbps = bytes_moved / (ms * 1e-3) / 1e9
                peak = max(peak, gbps)
                peak_by_dst[dname] = max(peak_by_dst.get(dname, 0.0), gbps)

                ok = (cq >= COS_Q and cres >= COS_F and cnrm >= COS_F)
                if not ok:
                    fails.append((dname, T, H, cq, cres, cnrm))
                rows.append((dname, T, H, gbps))
                print(f"{dname:>7} {T:>7} {H:>6} {cq:>9.6f} {cres:>9.6f} "
                      f"{cnrm:>9.6f} {ms:>9.4f} {gbps:>8.1f}  {'OK' if ok else 'FAIL'}")

    print("-" * 82)
    for dname in peak_by_dst:
        print(f"peak {dname:>7} : {peak_by_dst[dname]:8.1f} GB/s")
    print(f"peak overall  : {peak:8.1f} GB/s  "
          f"vs target {TARGET_GBPS:.0f}  [{100*peak/TARGET_GBPS:.1f}%]  "
          f"({'PASS' if peak >= TARGET_GBPS else 'below'})")
    print(f"worst cos: quant={worst['q']:.6f} (>= {COS_Q})  "
          f"after_res={worst['res']:.6f} after_norm={worst['nrm']:.6f} (>= {COS_F})")
    assert not fails, f"accuracy FAILED for {len(fails)} cases: {fails[:8]}"
    print("ALL ACCURACY CHECKS PASSED")


if __name__ == "__main__":
    main()
