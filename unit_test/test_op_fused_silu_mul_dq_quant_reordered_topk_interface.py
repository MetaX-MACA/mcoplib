"""Unit test + bandwidth benchmark for
mcoplib.op.fused_silu_mul_dq_reorder_quant  (fused SiLU*mul + per-token dynamic
quant, reorder-topk) on MetaX C600-UL.

API (in-place):
  fused_silu_mul_dq_reorder_quant(out, scale, input, reorder_topk_ids,
                                  w2_scale, start_expert_id, end_expert_id)

  input            : [num_tokens, 2*inner_hidden]  bf16/fp16  (front half=gate, back=up)
  out              : [num_tokens, inner_hidden]     int8  OR  float8_e4m3fn
  scale            : [num_tokens, 1]                float32   (per-token dequant scale)
  reorder_topk_ids : [num_tokens]                   int32/int64 (expert id per row)
  w2_scale         : [] (empty -> HAS_SCALE=false)  or per-expert input_scale
  start/end_expert_id : rows with id outside [start,end] are skipped (left untouched)

Semantics (per valid row):
  g   = silu(gate) * up * value_scale          (value_scale = 1/w2_scale[expert] or 1)
  s   = max|g| / QMAX                           (QMAX = 127 int8 / 448 fp8)
  out = round/convert( g / s )   ,  scale = s

Accuracy: cosine similarity of dequant(out)*... vs reference. int8/fp8 >= 0.999.
Bandwidth printed per dtype+shape.  Output format mirrors
test_op_moe_scatter_dynamic_quant.py.

Run:
  export CUDA_VISIBLE_DEVICES=<idle>
  python unit_test/test_op_fused_silu_mul_dq_quant_reordered_topk_interface.py
"""

import os
import torch

from mcoplib.op import fused_silu_mul_dq_reorder_quant

DEVICE = "cuda"
FP8 = torch.float8_e4m3fn

TARGET_GBPS = 1360.0          # 85% of single-die 1.6T
SINGLE_DIE_GBPS = 1600.0

# dst name -> (torch dtype, QMAX, out_elem_bytes, cos_sim threshold)
DST = {
    "int8": (torch.int8, 127.0, 1, 0.999),
    "fp8":  (FP8,        448.0, 1, 0.999),
}

NUM_EXPERTS = 8               # experts covered by [start,end] = [0, NUM_EXPERTS-1]

# batch sizes required by the request
BATCHES = [1, 3, 7, 11, 32, 128, 256, 512, 1024, 2048, 3200]

# mainstream LLM MoE inner_hidden (down_proj input) sizes; kept multiple of 8
# so the vectorized (float4, N=8 for bf16) fast path is exercised.
INNER_HIDDENS = [1408, 1536, 1792, 2048, 2816, 3584, 4096, 5120, 6144, 8192]


def silu_ref_float(gate, up, value_scale):
    """Replicate the kernel's float math exactly:
       val0,val1 read as float; sigmoid = val0 * 1/(1+exp(-val0));
       g = up * sigmoid * value_scale."""
    val0 = gate.float()
    val1 = up.float()
    sigmoid = val0 * (1.0 / (1.0 + torch.exp(-val0)))
    return val1 * sigmoid * value_scale


def quant_ref(g, dst_name):
    """Per-token dynamic quant reference matching the kernel."""
    dst_dtype, qmax, _, _ = DST[dst_name]
    absmax = g.abs().amax(dim=-1, keepdim=True)            # [T,1]
    scale = absmax / qmax                                  # dequant scale
    tmp = qmax * torch.reciprocal(absmax.clamp_min(1e-20))
    q = g * tmp
    if dst_name == "int8":
        q = torch.round(q).clamp(-127, 127).to(torch.int8)
    else:
        q = q.clamp(-448.0, 448.0).to(FP8)
    return q, scale


def make_inputs(num_tokens, inner_hidden, in_dtype=torch.bfloat16, seed=1234):
    g = torch.Generator(device=DEVICE).manual_seed(seed + num_tokens + inner_hidden)
    inp = torch.randn(num_tokens, 2 * inner_hidden, dtype=in_dtype,
                      device=DEVICE, generator=g)
    ids = torch.randint(0, NUM_EXPERTS, (num_tokens,), device=DEVICE,
                        dtype=torch.int64, generator=g)
    return inp, ids


def cosine(a, b, eps=1e-8):
    a = a.double().reshape(-1)
    b = b.double().reshape(-1)
    return (torch.dot(a, b) /
            (a.norm().clamp_min(eps) * b.norm().clamp_min(eps))).item()


def run_case(num_tokens, inner_hidden, dst_name, warmup=10, rep=50):
    dst_dtype, qmax, out_bytes, thr = DST[dst_name]
    inp, ids = make_inputs(num_tokens, inner_hidden)

    out = torch.empty(num_tokens, inner_hidden, dtype=dst_dtype, device=DEVICE)
    scale = torch.empty(num_tokens, 1, dtype=torch.float32, device=DEVICE)
    w2_scale = torch.empty(0, device=DEVICE)

    fused_silu_mul_dq_reorder_quant(out, scale, inp, ids, w2_scale,
                                    0, NUM_EXPERTS - 1)
    torch.cuda.synchronize()

    # ---- reference (all rows valid: ids in [0,NUM_EXPERTS-1]) ----
    gate = inp[:, :inner_hidden]
    up = inp[:, inner_hidden:]
    g = silu_ref_float(gate, up, 1.0)
    q_ref, s_ref = quant_ref(g, dst_name)

    # dequantised comparison (robust to +-1 quant rounding vs kernel SFU)
    deq_k = out.float() * scale
    deq_r = q_ref.float() * s_ref
    sim = cosine(deq_k, deq_r)

    # trap / nan check
    bad = (not torch.isfinite(scale).all().item()) or \
          torch.isnan(out.float()).any().item()

    # ---- bandwidth: back-to-back burst in one sync ----
    for _ in range(warmup):
        fused_silu_mul_dq_reorder_quant(out, scale, inp, ids, w2_scale, 0, NUM_EXPERTS - 1)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        fused_silu_mul_dq_reorder_quant(out, scale, inp, ids, w2_scale, 0, NUM_EXPERTS - 1)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    ms = times[len(times) // 2]

    # bytes: read gate+up (2*inner bf16), write inner*out_bytes, +4B scale/row
    eff_bytes = num_tokens * (2 * inner_hidden * 2 + inner_hidden * out_bytes) + num_tokens * 4
    gbps = eff_bytes / (ms * 1e-3) / 1e9

    ok = (sim >= thr) and (not bad)
    tag = "OK" if ok else "FAIL"
    print(f"[{dst_name:>4}] T={num_tokens:5d} H={inner_hidden:5d}  "
          f"cos_sim={sim:.6f} {tag:4s}  {ms:8.4f} ms  {gbps:8.1f} GB/s"
          + ("  <TRAP/NAN>" if bad else ""))
    return ok, gbps


def main():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES={vis!r}  device_count={torch.cuda.device_count()}")
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"config: experts=[0,{NUM_EXPERTS-1}] in_dtype=bf16  "
          f"batches={BATCHES}  inner_hidden={INNER_HIDDENS}")
    print("=" * 104)

    overall = True
    peaks = {}
    for dst_name in DST:
        thr = DST[dst_name][3]
        print(f"---- dst_dtype = {dst_name} (cos_sim threshold {thr}) " + "-" * 40)
        peak, dpass = 0.0, True
        for H in INNER_HIDDENS:
            for T in BATCHES:
                ok, gbps = run_case(T, H, dst_name)
                peak = max(peak, gbps)
                dpass &= ok
        peaks[dst_name] = (peak, dpass)
        print(f"     [{dst_name}] peak = {peak:.1f} GB/s   "
              f"accuracy {'ALL PASS' if dpass else 'FAILURES PRESENT'}")
        overall &= dpass

    print("=" * 104)
    for dst_name, (peak, dpass) in peaks.items():
        print(f"{dst_name:>4}: peak {peak:8.1f} GB/s  "
              f"({'REACHED' if peak >= TARGET_GBPS else 'below'} 85% single-die "
              f"target {TARGET_GBPS:.0f})  accuracy {'PASS' if dpass else 'FAIL'}")
    print(f"single-die bandwidth reference: {SINGLE_DIE_GBPS:.0f} GB/s")
    assert overall, "accuracy check failed"


if __name__ == "__main__":
    main()
