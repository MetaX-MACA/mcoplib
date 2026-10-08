"""Unit test + bandwidth benchmark for mcoplib.op.moe_softmax_topk on MetaX C600-U.

Restricted to the requested LLM config:
  dtype=float32, num_experts=128, topk=8, pre-softmax, sp_size=[8, 1]
  num_tokens = [16, 64, 128, ..., 32768, 65536]

Op: moe_softmax_topk(topk_weights[T,k], topk_indices[T,k], gating_output[T,E], pre_softmax)
Semantics: softmax over E experts (fp32) -> top-k by argmax (ties -> smaller idx)
           -> renormalize the k weights to sum 1.

Bandwidth uses the minimal fused traffic (algorithmic lower bound):
  effective_bytes = T*E*4 (read gating) + T*k*4 (weights) + T*k*4 (indices)

Run:
  export CUDA_VISIBLE_DEVICES=<free gpu>
  python -m unit_test.test_op_moe_softmax_topk
  python -m unit_test.test_op_moe_softmax_topk --num-tokens 1024
  python -m unit_test.test_op_moe_softmax_topk --num-tokens 1024 2048 --sp-size 8
"""

import argparse
import os

import torch

import mcoplib.op as ops

DEVICE = "cuda"
COS_SIM_THRESHOLD = 0.9999
TARGET_GBPS = 1300.0          # single-die gate
DUAL_DIE_GBPS = 3480.0        # dual-die datasheet figure

# Requested config. sp_size is case metadata for this row-wise operator: it is
# intentionally not passed to moe_softmax_topk and does not change tensor shapes.
NUM_EXPERTS = 128
TOPK = 8
PRE_SOFTMAX = True
SP_SIZE_CASES = [8, 1]
NUM_TOKENS_CASES = [
    16, 48, 64, 128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536,
    1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096,
    6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528,
    24576, 26624, 28672, 30720, 32768, 65536,
]


def ref_softmax_topk(gating, k):
    """torch reference matching the kernel semantics."""
    probs = torch.softmax(gating.float(), dim=-1)
    vals, idx = torch.topk(probs, k, dim=-1, largest=True, sorted=True)
    weights = vals / vals.sum(dim=-1, keepdim=True)
    return weights.float(), idx.to(torch.int32)


def cos_sim(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-12).item()


def benchmark(fn, warmup=15, rep=60):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return times.median().item()  # ms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate and benchmark mcoplib.op.moe_softmax_topk.")
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        choices=NUM_TOKENS_CASES,
        default=None,
        help="one or more token counts; defaults to the complete configured list",
    )
    parser.add_argument(
        "--sp-size",
        type=int,
        nargs="+",
        choices=SP_SIZE_CASES,
        default=None,
        help="one or more sequence-parallel sizes; defaults to 8 and 1",
    )
    return parser.parse_args()


def run(num_tokens_cases=None, sp_size_cases=None):
    num_tokens_cases = num_tokens_cases or NUM_TOKENS_CASES
    sp_size_cases = sp_size_cases or SP_SIZE_CASES
    shape_cases = [
        (sp_size, num_tokens)
        for sp_size in sp_size_cases
        for num_tokens in num_tokens_cases
    ]

    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    print(f"CUDA_VISIBLE_DEVICES='{dev}'  device_count={torch.cuda.device_count()}")
    print(f"config: E={NUM_EXPERTS} topk={TOPK} pre_softmax={PRE_SOFTMAX} "
          f"sp_size={sp_size_cases} num_tokens={num_tokens_cases}")
    print("=" * 96)

    all_pass = True
    peak_gbps = 0.0
    peak_desc = ""

    for sp_size, T in shape_cases:
        g = torch.Generator(device=DEVICE).manual_seed(
            1234 + sp_size * 100_000 + T + NUM_EXPERTS)
        gating = torch.randn(T, NUM_EXPERTS, dtype=torch.float32, device=DEVICE, generator=g)

        topk_weights = torch.zeros(T, TOPK, dtype=torch.float32, device=DEVICE)
        topk_indices = torch.zeros(T, TOPK, dtype=torch.int32, device=DEVICE)

        ops.moe_softmax_topk(topk_weights, topk_indices, gating, PRE_SOFTMAX)
        torch.cuda.synchronize()

        ref_w, ref_i = ref_softmax_topk(gating, TOPK)
        cs = cos_sim(topk_weights, ref_w)

        # index-set agreement (order-independent).
        got_sets = topk_indices.sort(dim=-1).values
        ref_sets = ref_i.sort(dim=-1).values
        idx_match = (got_sets == ref_sets).all(dim=-1).float().mean().item()

        ok = (cs >= COS_SIM_THRESHOLD) and (idx_match >= 0.999)
        all_pass = all_pass and ok

        ms = benchmark(lambda: ops.moe_softmax_topk(topk_weights, topk_indices, gating, PRE_SOFTMAX))
        eff_bytes = T * NUM_EXPERTS * 4 + T * TOPK * 4 + T * TOPK * 4
        gbps = eff_bytes / (ms * 1e-3) / 1e9
        if gbps > peak_gbps:
            peak_gbps = gbps
            peak_desc = f"E={NUM_EXPERTS} k={TOPK} sp={sp_size} T={T}"

        tag = "OK " if ok else "BAD"
        print(f"[softmax_topk] E={NUM_EXPERTS:<4} k={TOPK:<2} "
              f"sp={sp_size:<2} T={T:<6} "
              f"cos_sim={cs:.6f} idx={idx_match:.4f} {tag} "
              f"{ms:8.4f} ms  {gbps:8.1f} GB/s")

    print("=" * 96)
    print(f"Accuracy: {'ALL PASS' if all_pass else 'SOME FAILED'} "
          f"(threshold cos_sim >= {COS_SIM_THRESHOLD})")
    print(f"Peak bandwidth: {peak_gbps:.1f} GB/s  [{peak_desc}]  "
          f"(single-die target {TARGET_GBPS:.0f} -> "
          f"{'REACHED' if peak_gbps >= TARGET_GBPS else 'NOT reached'}; "
          f"dual-die datasheet {DUAL_DIE_GBPS:.0f})")
    assert all_pass, "moe_softmax_topk accuracy check failed"


if __name__ == "__main__":
    args = parse_args()
    run(args.num_tokens, args.sp_size)
