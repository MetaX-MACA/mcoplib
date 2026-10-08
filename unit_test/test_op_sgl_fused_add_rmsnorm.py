"""Unit test + bandwidth benchmark for sgl_fused_add_rmsnorm on MetaX C600-U.

Op: torch.ops.sgl_kernel.fused_add_rmsnorm(input, residual, weight, eps, enable_pdl)
  In-place fused Add + RMSNorm:
    residual <- input + residual            (residual updated in place)
    input    <- rmsnorm(residual) * weight  (input overwritten with the norm output)

Config (from task): dtype=bfloat16, sp_size=8, hidden_size=4096,
  num_tokens = [16, 1024, 2048, 4096, 8192, 10240].
Accuracy gate: cosine similarity >= 0.9999 on both outputs.
Bandwidth: effective HBM bytes = num_tokens * hidden * 2(bf16) * 4
  (read input + read residual + write input + write residual). Reported per shape,
  peak over the matrix; single-die goal 1.3 TB/s, datasheet dual-die 3480 GB/s.
"""
import os
import torch

# torch loads libtorch, then the op library registers torch.ops.sgl_kernel.*
import mcoplib.sgl_kernel as _K  # noqa: F401
import mcoplib._C  # noqa: F401

SP_SIZE = 8
HIDDEN = 4096
EPS = 1e-6
NUM_TOKENS = [16, 1024, 2048, 4096, 8192, 10240]
COS_SIM_THRESHOLD = 0.9999
TARGET_GBPS = 1300.0
DATASHEET_GBPS = 3480.0

OP = torch.ops.sgl_kernel.fused_add_rmsnorm


def make_inputs(num_tokens, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.float32, generator=g).to(torch.bfloat16)
    r = torch.randn(num_tokens, HIDDEN, dtype=torch.float32, generator=g).to(torch.bfloat16)
    w = (1.0 + 0.1 * torch.randn(HIDDEN, dtype=torch.float32, generator=g)).to(torch.bfloat16)
    return x.cuda(), r.cuda(), w.cuda()


def ref_forward(x, r, w, eps):
    res_f = x.float() + r.float()
    var = res_f.pow(2).mean(dim=-1, keepdim=True)
    out = res_f * torch.rsqrt(var + eps) * w.float()
    return out, res_f  # normalized output, updated residual (both fp32 reference)


def cos_sim(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def effective_bytes(num_tokens):
    # read input + read residual + write input + write residual, all bf16
    return num_tokens * HIDDEN * 2 * 4


def check_accuracy(num_tokens):
    x0, r0, w = make_inputs(num_tokens)
    ref_out, ref_res = ref_forward(x0, r0, w, EPS)
    x = x0.clone()
    r = r0.clone()
    OP(x, r, w, EPS, False)
    torch.cuda.synchronize()
    sim_out = cos_sim(x, ref_out)
    sim_res = cos_sim(r, ref_res)
    return min(sim_out, sim_res)


def bench_case(num_tokens, warm_s=1.0, time_s=0.8):
    """Continuous back-to-back burst timing to hold the DVFS boost clock."""
    x0, r0, w = make_inputs(num_tokens)
    x = x0.clone()
    r = r0.clone()

    # warmup burst (also lets clocks ramp)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    import time as _t
    t0 = _t.time()
    n = 0
    while _t.time() - t0 < warm_s:
        OP(x, r, w, EPS, False)
        n += 1
        if n % 64 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()

    # find a batch size that runs ~a few ms, then time many back-to-back launches
    reps = max(64, n)
    best_gbps = 0.0
    best_ms = 0.0
    rounds = 0
    t0 = _t.time()
    while _t.time() - t0 < time_s:
        start.record()
        for _ in range(reps):
            OP(x, r, w, EPS, False)
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end) / reps
        gbps = effective_bytes(num_tokens) / (ms * 1e-3) / 1e9
        if gbps > best_gbps:
            best_gbps = gbps
            best_ms = ms
        rounds += 1
    return best_ms, best_gbps


def main():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"CUDA_VISIBLE_DEVICES={vis!r}  device_count={torch.cuda.device_count()}")
    print(f"config: sp_size={SP_SIZE} hidden={HIDDEN} dtype=bf16 eps={EPS}")
    print(f"target: {TARGET_GBPS:.0f} GB/s single-die  (datasheet dual-die {DATASHEET_GBPS:.0f})")
    print("-" * 78)
    max_gbps = 0.0
    all_pass = True
    for T in NUM_TOKENS:
        sim = check_accuracy(T)
        ok = "OK  " if sim >= COS_SIM_THRESHOLD else "FAIL"
        if sim < COS_SIM_THRESHOLD:
            all_pass = False
        ms, gbps = bench_case(T)
        max_gbps = max(max_gbps, gbps)
        print(f"[sgl_fused_add_rmsnorm] T={T:6d}  cos_sim={sim:.6f} {ok}  "
              f"{ms:8.4f} ms  {gbps:8.1f} GB/s")
    print("-" * 78)
    print(f"Accuracy: {'ALL PASS' if all_pass else 'FAILURES PRESENT'} "
          f"(threshold cos_sim >= {COS_SIM_THRESHOLD})")
    print(f"Peak bandwidth: {max_gbps:.1f} GB/s  "
          f"({100.0 * max_gbps / TARGET_GBPS:.1f}% of {TARGET_GBPS:.0f} single-die, "
          f"{100.0 * max_gbps / DATASHEET_GBPS:.1f}% of {DATASHEET_GBPS:.0f} datasheet)")
    assert all_pass, "cosine similarity below threshold on at least one shape"


if __name__ == "__main__":
    main()
