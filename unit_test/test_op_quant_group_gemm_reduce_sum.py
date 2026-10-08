"""Unit test + perf benchmark for the triton quant_group_gemm_reduce_sum kernel
on MetaX C600-U.

Op semantics (per the base ComputeEngine reference `fake_quant_gemm` + reduce_sum):
  For each sp in [0, sp_size):
     G[sp] = (A[sp].bf16 @ W[sp].bf16).fp32          # A:[M,K] int8, W:[K,N] int8
     G[sp] *= per_token_scale[sp][:,None] * weight_scale[sp][None,:]
  output = sum_sp G[sp]                               # [M, N] bf16

Shapes come from the requested LLM config (note: num_tokens is the *global* token
count and is divided by sp_size inside the op, so M = num_tokens // sp_size):

  "quant_group_gemm_reduce_sum": [
    { "arg_type":"llm", "dtype":"int8", "dst_dtype":"bfloat16",
      "sp_size":8, "num_tokens":10240, "hidden_size":1024, "new_hidden_size":4096 }
  ]

  => sp_size=8, M = 10240//8 = 1280, K = 1024, N = 4096.

ROOFLINE: INT8 ops = 2*sp*M*N*K = 85.9 GOP; DRAM traffic ~54.5 MB.
  compute@480T = 0.179 ms ; mem@1.3TB/s = 0.042 ms  -> ratio 4.3x => COMPUTE BOUND.
  Target = 85% * 480 TOPS = 408 TOPS.  We therefore report INT8 TOPS as the
  headline metric (bandwidth is printed too, for reference).

Run:
  export CUDA_VISIBLE_DEVICES=<free gpu>
  python test_op_quant_group_gemm_reduce_sum.py
"""

import os
import pathlib
import torch
import triton

# # The kernel module registers a vendor impl at import time, which requires the
# # framework's base impls to be loaded first. Load them, then import the jit kernel.
# from xpu_perf.micro_perf.core.op import ProviderRegistry
# # discover_plugins reads PROVIDER_NAME from a package __init__.py; only op_defs/
# # has it ("base_ops") and it recursively imports llm_ops/*.py (registering the
# # base impl). Pointing at op_defs/llm_ops directly loads nothing (empty __init__).
# _OP_DEFS = pathlib.Path(__file__).resolve().parents[4] / "op_defs"
# try:
#     ProviderRegistry.load_all_base_impls(_OP_DEFS)
# except Exception as _e:
#     print(f"[warn] load_all_base_impls: {_e}")

from mcoplib.triton_quant_group_gemm_reduce_sum import _quant_group_gemm_reduce_sum_kernel

DEVICE = "cuda"
COS_SIM_THRESHOLD = 0.999          # int8 quantized GEMM -> 0.999 per requirement
INT8_PEAK_TOPS = 480.0             # C600-U single-die INT8 peak
TARGET_TOPS = INT8_PEAK_TOPS * 0.85    # 408 TOPS acceptance gate
DUAL_DIE_GBPS = 3480.0
SINGLE_DIE_GBPS = 1300.0

# ---- Requested config (JSON) ---------------------------------------------
DTYPE = torch.int8
DST_DTYPE = torch.bfloat16
CONFIGS = [
    # (sp_size, num_tokens_global, hidden_size, new_hidden_size, trans_w)
    (8, 10240, 1024, 4096, False),
]


def ref_quant_group_gemm_reduce_sum(A, tok_scale, W, w_scale, sp_size, trans_w):
    """torch reference matching the ComputeEngine base impl (fake_quant_gemm + sum)."""
    sp, M, K = A.shape
    N = w_scale.shape[-1]
    out = torch.zeros(M, N, dtype=torch.float32, device=A.device)
    for s in range(sp_size):
        w = W[s]
        if trans_w:
            w = w.transpose(0, 1)                       # [N,K]->[K,N]
        g = torch.matmul(A[s].to(torch.bfloat16), w.to(torch.bfloat16)).to(torch.float32)
        g = g * tok_scale[s].unsqueeze(-1) * w_scale[s].unsqueeze(0)
        out += g
    return out.to(torch.bfloat16)


def cos_sim(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-12).item()


def benchmark(fn, warmup=25, rep=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(st, en)])
    return times.median().item()  # ms


# ---- Kernel launch config (tunable; must stay in sync with the .py kernel) --
LAUNCH_CFG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
    "num_warps": 8,
    "num_stages": 3,
}


def run_kernel(A, tok_scale, W, w_scale, out, sp_size, M, K, N, trans_w, cfg):
    grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),)
    _quant_group_gemm_reduce_sum_kernel[grid](
        A, tok_scale, W, w_scale, out,
        sp_size, M, K, N, trans_w,
        BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )


def run():
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    print(f"CUDA_VISIBLE_DEVICES='{dev}'  device_count={torch.cuda.device_count()}")
    print(f"launch cfg: {LAUNCH_CFG}")
    print("=" * 100)

    all_pass = True
    peak_tops = 0.0
    peak_desc = ""

    for (sp_size, num_tokens_global, K, N, trans_w) in CONFIGS:
        M = num_tokens_global // sp_size
        g = torch.Generator(device=DEVICE).manual_seed(1234 + M + K + N)

        A = torch.randint(-127, 128, (sp_size, M, K), dtype=torch.int8, device=DEVICE, generator=g)
        if trans_w:
            W = torch.randint(-127, 128, (sp_size, N, K), dtype=torch.int8, device=DEVICE, generator=g)
        else:
            W = torch.randint(-127, 128, (sp_size, K, N), dtype=torch.int8, device=DEVICE, generator=g)
        tok_scale = (torch.rand(sp_size, M, dtype=torch.float32, device=DEVICE, generator=g) * 0.02 + 0.001)
        w_scale = (torch.rand(sp_size, N, dtype=torch.float32, device=DEVICE, generator=g) * 0.02 + 0.001)

        out = torch.zeros(M, N, dtype=DST_DTYPE, device=DEVICE)

        run_kernel(A, tok_scale, W, w_scale, out, sp_size, M, K, N, trans_w, LAUNCH_CFG)
        torch.cuda.synchronize()

        ref = ref_quant_group_gemm_reduce_sum(A, tok_scale, W, w_scale, sp_size, trans_w)
        cs = cos_sim(out, ref)
        ok = cs >= COS_SIM_THRESHOLD
        all_pass = all_pass and ok

        ms = benchmark(lambda: run_kernel(A, tok_scale, W, w_scale, out,
                                          sp_size, M, K, N, trans_w, LAUNCH_CFG))

        flops = 2.0 * sp_size * M * N * K
        tops = flops / (ms * 1e-3) / 1e12
        # DRAM traffic: read A + W, write out (fp32 accum output bf16)
        rbytes = sp_size * M * K + sp_size * K * N       # int8
        wbytes = M * N * 2                               # bf16
        gbps = (rbytes + wbytes) / (ms * 1e-3) / 1e9

        if tops > peak_tops:
            peak_tops = tops
            peak_desc = f"sp={sp_size} M={M} K={K} N={N}"

        tag = "OK " if ok else "BAD"
        print(f"[qggrs] sp={sp_size} M={M:<5} K={K:<5} N={N:<5} "
              f"cos_sim={cs:.6f} {tag} "
              f"{ms:8.4f} ms  {tops:8.1f} TOPS  {gbps:8.1f} GB/s")

    print("=" * 100)
    print(f"Accuracy: {'ALL PASS' if all_pass else 'SOME FAILED'} "
          f"(threshold cos_sim >= {COS_SIM_THRESHOLD})")
    pct = peak_tops / INT8_PEAK_TOPS * 100.0
    print(f"Peak INT8: {peak_tops:.1f} TOPS  [{peak_desc}]  "
          f"({pct:.1f}% of {INT8_PEAK_TOPS:.0f}T peak; target 85%={TARGET_TOPS:.0f}T -> "
          f"{'REACHED' if peak_tops >= TARGET_TOPS else 'NOT reached'})")


if __name__ == "__main__":
    run()
