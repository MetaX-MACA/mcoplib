"""Unit test + bandwidth benchmark for torch.ops._C.swiglu_step_and_mul_per_block_quant
on MetaX C600-U.

Op (9-arg):
  swiglu_step_and_mul_per_block_quant(
      out, input, limit, scales, group_size,
      scale_ub=None, is_scale_transposed=False, alpha=1.0, beta=0.0)

Semantics (per element, per token row):
  gate = input[:, :H], up = input[:, H:]           (input is [M, 2H])
  act   = gate * sigmoid(alpha * gate)             # SiLU_alpha(gate)
  g     = min(act, limit)
  u     = clamp(up, -limit, limit) + beta
  a     = g * u                                    # [M, H] result before quant
  # per-group absmax quant with group_size:
  amax  = max_g |a|;  scale = max(amax / qmax, min_scale);  scale_ub clamps (fp8)
  q     = round(a * (1/scale))  -> int8  (qmax 127, round-half-away, clamp [-128,127])
  q     = clamp(a * (1/scale), -qmax, qmax) -> fp8-e4m3fn (qmax 448)
  q     = clamp(a * (1/scale), -1, 1) -> bf16 (qmax 1, normalized; scale = amax)
  scales[M, ceil(H/gs)]  (or [ceil(H/gs), M] when is_scale_transposed)

Accuracy: cosine similarity (dequantized vs fp32 reference) >= 0.999.
Bandwidth: bytes = read M*2H*2 (bf16) + write M*H*elem + write M*G*4 (scales).

Run:
  export CUDA_VISIBLE_DEVICES=<free gpu>
  python unit_test/test_op_vllm_swiglu_step_and_mul_per_block_quant.py
"""

import torch

import torch.nn.functional as F

import mcoplib._C  # noqa: F401  (registers torch.ops._C)

DEVICE = "cuda"

TARGET_GBPS = 1360.0   # 85% of single-die ~1.6T
DUAL_DIE_GBPS = 3200.0

FLOAT8 = torch.float8_e4m3fn
DST_DTYPES = {
    "int8": (torch.int8, 127.0, 1),
    "fp8": (FLOAT8, 448.0, 1),
    # BF16 stores normalized values in [-1, 1]; scales contain the block amax.
    "bf16": (torch.bfloat16, 1.0, 2),
}
MIN_SCALE = {"int8": torch.finfo(torch.float32).eps,  # 1.192e-7
             "fp8": 1.0 / (448.0 * 512.0),            # ~4.36e-6
             "bf16": torch.finfo(torch.float32).eps}
SIM_THRESH = {"int8": 0.999, "fp8": 0.999, "bf16": 0.99999}

# (num_tokens, hidden_size) from the SWiGLU-step request.
SHAPES = [
    (128, 1024), (1024, 2048), (4096, 4096), (32, 1536), (1, 1536),
    (128, 1536), (4096, 1536),(8192, 1536), (16392, 1536), (32784, 1536), (65544, 1536), (3, 1537), (64, 257), (1, 4096), (8, 4096),
    (128, 4096), (1, 6144), (16, 6144), (128, 8192), (1, 11008),
    (32, 11008), (1024, 11008), (1, 14336), (64, 14336), (4096, 14336),
]


def _round_half_away(x):
    return torch.where(x >= 0, (x + 0.5).floor(), (x - 0.5).ceil())


def swiglu_step_reference(input, limit, group_size, dst_name, alpha=1.0,
                          beta=0.0, scale_ub=None):
    """FP32 reference matching the kernel exactly, incl. scale floor/clamp."""
    M, H2 = input.shape
    H = H2 // 2
    qmax, min_scale = DST_DTYPES[dst_name][1], MIN_SCALE[dst_name]
    gate = input[:, :H].float()
    up = input[:, H:].float()
    act = gate * torch.sigmoid(alpha * gate)
    g = torch.minimum(act, torch.tensor(limit, dtype=torch.float32,
                                        device=input.device))
    u = torch.clamp(up, -limit, limit) + beta
    a = g * u

    G = (H + group_size - 1) // group_size
    pad = G * group_size - H
    a_pad = F.pad(a, (0, pad)) if pad else a
    ag = a_pad.view(M, G, group_size)
    amax = ag.abs().amax(dim=2).clamp(min=min_scale)
    scale = (amax / qmax)
    if scale_ub is not None:
        scale = scale.clamp(max=float(scale_ub))
    scale = scale.clamp(min=min_scale)
    inv = 1.0 / scale                                    # [M, G]
    x = ag * inv[:, :, None]
    if dst_name == "int8":
        q = _round_half_away(x).clamp(-128, 127).to(torch.int8)
    elif dst_name == "bf16":
        # normalized per-block output; torch fp32->bf16 is RNE, same as the
        # kernel's static_cast<at::BFloat16>
        q = x.clamp(-1.0, 1.0).to(torch.bfloat16)
    else:
        q = x.clamp(-qmax, qmax).to(FLOAT8)
    out = q.view(M, -1)[:, :H].contiguous()
    return out, a, scale


def make_inputs(num_tokens, hidden_size, dtype, seed=1234):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    input = (torch.randn(num_tokens, hidden_size * 2, dtype=dtype,
                         device=DEVICE, generator=g) * 0.5)
    return input


def alloc_outputs(num_tokens, hidden_size, group_size, dst_dtype,
                  is_scale_transposed):
    G = (hidden_size + group_size - 1) // group_size
    out = torch.empty(num_tokens, hidden_size, dtype=dst_dtype, device=DEVICE)
    scales = (torch.empty(G, num_tokens, device=DEVICE)
              if is_scale_transposed
              else torch.empty(num_tokens, G, device=DEVICE))
    return out, scales


def run_op(input, out, scales, limit, group_size, dst_dtype,
           scale_ub=None, is_scale_transposed=False, alpha=1.0, beta=0.0):
    torch.ops._C.swiglu_step_and_mul_per_block_quant(
        out, input, limit, scales, group_size, scale_ub,
        is_scale_transposed, alpha, beta)


def cosine_sim(a, b):
    denom = (a.norm().item() * b.norm().item())
    return (a.dot(b).item() / max(denom, 1e-30))


def check_accuracy(input, out, scales, limit, group_size, dst_name,
                   scale_ub, is_scale_transposed, alpha, beta, M, H):
    # Keep the large-shape reference bounded: materializing all intermediate
    # tensors at once can exceed a 73 GiB device. Validate in row chunks.
    chunk = min(M, 1024)
    G = (H + group_size - 1) // group_size
    s_got = scales.t().float() if is_scale_transposed else scales.float()
    s_ref = torch.empty((M, G), device=input.device, dtype=torch.float32)
    dot = torch.zeros((), device=input.device, dtype=torch.float64)
    n_deq = torch.zeros((), device=input.device, dtype=torch.float64)
    n_ref = torch.zeros((), device=input.device, dtype=torch.float64)
    for start in range(0, M, chunk):
        stop = min(start + chunk, M)
        ref, a, ref_scale = swiglu_step_reference(
            input[start:stop], limit, group_size, dst_name, alpha, beta,
            scale_ub)
        s_ref[start:stop].copy_(ref_scale)
        scale_e = (ref_scale[:, :, None].expand(stop - start, G, group_size)
                   .reshape(stop - start, G * group_size)[:, :H])
        deq = out[start:stop].float() * scale_e
        dot += (deq.double() * a.double()).sum()
        n_deq += (deq.double() * deq.double()).sum()
        n_ref += (a.double() * a.double()).sum()
        del ref, a, ref_scale, scale_e, deq
    sim = (dot / torch.sqrt(n_deq * n_ref).clamp_min(1e-30)).item()
    scale_ok = bool(torch.allclose(s_got, s_ref, atol=0.0, rtol=1e-3))
    del s_ref, s_got
    return sim, scale_ok


def effective_bytes(M, H, group_size, out_bytes):
    G = (H + group_size - 1) // group_size
    return M * (H * 2 * 2 + H * out_bytes + G * 4)


def bench_case(M, H, dst_name, group_size=128, is_scale_transposed=False,
               scale_ub=None, alpha=1.0, beta=0.0, input_dtype=torch.bfloat16,
               warmup=10, rep=100):
    dst_dtype, qmax, out_bytes = DST_DTYPES[dst_name]
    input = make_inputs(M, H, input_dtype)
    out, scales = alloc_outputs(M, H, group_size, dst_dtype,
                                is_scale_transposed)
    limit = 8.0

    run_op(input, out, scales, limit, group_size, dst_dtype, scale_ub,
           is_scale_transposed, alpha, beta)
    torch.cuda.synchronize()
    sim, scale_ok = check_accuracy(
        input, out, scales, limit, group_size, dst_name, scale_ub,
        is_scale_transposed, alpha, beta, M, H)

    for _ in range(warmup):
        run_op(input, out, scales, limit, group_size, dst_dtype, scale_ub,
               is_scale_transposed, alpha, beta)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        run_op(input, out, scales, limit, group_size, dst_dtype, scale_ub,
               is_scale_transposed, alpha, beta)
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    median_ms = times_ms[len(times_ms) // 2]

    gbps = effective_bytes(M, H, group_size, out_bytes) / (median_ms * 1e-3) / 1e9
    thresh = SIM_THRESH[dst_name]
    ok = "OK" if (sim >= thresh and scale_ok) else "FAIL"
    tag = (f"gs={group_size}" + (",T" if is_scale_transposed else "") +
           (f",ub={scale_ub:.3g}" if scale_ub is not None else "") +
           (f",a={alpha},b={beta}" if (alpha, beta) != (1.0, 0.0) else ""))
    print(f"[{dst_name:>4}] M={M:5d} H={H:5d} {tag:24s} "
          f"cos_sim={sim:.6f} {ok:4s}  {median_ms:8.4f} ms  {gbps:8.1f} GB/s")
    return sim >= SIM_THRESH[dst_name] and scale_ok, gbps


def main():
    print(f"device_count={torch.cuda.device_count()}  "
          f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES','')}")
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"config: 20 shapes x {len(DST_DTYPES)} dst (int8/fp8/bf16) x bf16-in; "
          f"plus gs=64 / transposed / alpha-beta / fp16 / scale_ub / generic-tail")
    print("=" * 100)

    overall_pass = True
    peaks = {}
    for dst_name in DST_DTYPES:
        print(f"---- dst_dtype = {dst_name} (cos_sim threshold {SIM_THRESH[dst_name]}) "
              + "-" * 40)
        peak, dpass = 0.0, True
        for (M, H) in SHAPES:
            ok, gbps = bench_case(M, H, dst_name)
            peak = max(peak, gbps)
            dpass &= ok
        peaks[dst_name] = (peak, dpass)
        print(f"     [{dst_name}] peak = {peak:.1f} GB/s   "
              f"accuracy {'ALL PASS' if dpass else 'FAILURES PRESENT'}")
        overall_pass &= dpass

    print("+++++++++++++++ extra correctness cases " + "-" * 50)
    extras = [
        # group_size 64
        ("gs64", dict(group_size=64)),
        ("gs64-1024", dict(group_size=64, M=1024, H=2048)),
        ("gs64-generic", dict(group_size=64, M=3, H=1537)),
        ("gs64-generic257", dict(group_size=64, M=64, H=257)),
        # transposed scales
        ("transposed", dict(is_scale_transposed=True, M=1024, H=2048)),
        ("transposed-large", dict(is_scale_transposed=True, M=4096, H=4096)),
        ("transposed-generic", dict(is_scale_transposed=True, M=3, H=1537)),
        # alpha / beta
        ("alpha-beta", dict(alpha=1.7, beta=0.05)),
        ("alpha-beta2", dict(alpha=2.0, beta=-0.1, M=1024, H=2048)),
        # fp16 input
        ("fp16", dict(input_dtype=torch.float16, M=128, H=1024)),
        ("fp16-large", dict(input_dtype=torch.float16, M=4096, H=4096)),
    ]
    for name, kw in extras:
        M = kw.pop("M", 128)
        H = kw.pop("H", 1024)
        for dst_name in DST_DTYPES:
            ok, _ = bench_case(M, H, dst_name, **kw)
            overall_pass &= ok
    # scale_ub is fp8-only (host TORCH_CHECK rejects int8)
    for ub in (0.02, 0.1):
        for (M, H) in ((1024, 2048), (4096, 4096)):
            ok, _ = bench_case(M, H, "fp8",
                               scale_ub=torch.tensor(ub, device=DEVICE))
            overall_pass &= ok

    print("=" * 100)
    for dst_name, (peak, dpass) in peaks.items():
        print(f"{dst_name:>4}: peak {peak:8.1f} GB/s  "
              f"({'REACHED' if peak >= TARGET_GBPS else 'below'} single-die target "
              f"{TARGET_GBPS:.0f})  accuracy {'PASS' if dpass else 'FAIL'}")
    print(f"dual-die datasheet reference: {DUAL_DIE_GBPS:.0f} GB/s")
    assert overall_pass, "accuracy check failed"


if __name__ == "__main__":
    main()
