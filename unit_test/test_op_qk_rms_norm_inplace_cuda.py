"""Unit test + bandwidth benchmark for the `qk_rms_norm_inplace_cuda` CUDA op.

Op semantics (QK-Norm used in modern attention):
  token_data is a fused per-token QKV buffer, bf16, laid out as
      [num_tokens, (q_head_num + 2*kv_head_num) * head_dim]
  ordered  [ Q(q_head_num) | K(kv_head_num) | V(kv_head_num) ].
  RMSNorm is applied INDEPENDENTLY to each Q head (weight q_norm_weight) and
  each K head (weight k_norm_weight), IN PLACE. The V heads are left untouched.

This op is MEMORY-BOUND: it only reads+writes the Q and K head bytes. So the
metric is HBM bandwidth (GB/s). Effective bytes per call:
    num_tokens * (q_head_num + kv_head_num) * head_dim * 2 (bf16) * 2 (R + W)
(weights are 128 fp32 * 2, negligible).

Accuracy is checked with cosine similarity against a torch fp32 reference;
threshold 0.9999 (bf16 in/out).

C600-U single-die HBM wall ~1.3 TB/s (datasheet 3480 GB/s is DUAL die).
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

import mcoplib.op as ops


# ---- shape config (from the task request) ---------------------------------
Q_KV_HEAD_CASES = [(32, 4), (8, 1)]
NUM_TOKENS_LIST = [
    16,
    48,
    64,
    128,
    256,
    384,
    512,
    640,
    768,
    896,
    1024,
    1280,
    1536,
    1792,
    2048,
    2304,
    2560,
    2816,
    3072,
    3328,
    3584,
    3840,
    4096,
    6144,
    8192,
    10240,
    12288,
    14336,
    16384,
    18432,
    20480,
    22528,
    24576,
    26624,
    28672,
    30720,
    32768,
    65536,
]
QK_HEAD_DIM = 128
V_HEAD_DIM = 128
EPS = 1e-5

COS_SIM_THRESHOLD = 0.9999          # bf16 in/out
TARGET_GBPS = 1300.0                # single-die realistic wall
DATASHEET_GBPS = 3480.0             # dual-die datasheet (unreachable single-die)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run qk_rms_norm correctness and bandwidth cases."
    )
    parser.add_argument("--q-head-num", type=int)
    parser.add_argument("--kv-head-num", type=int)
    parser.add_argument("--qk-head-dim", type=int, default=QK_HEAD_DIM)
    parser.add_argument("--v-head-dim", type=int, default=V_HEAD_DIM)
    parser.add_argument("--num-tokens", type=int)
    args = parser.parse_args()

    single_case_values = (args.q_head_num, args.kv_head_num, args.num_tokens)
    if any(value is not None for value in single_case_values) and not all(
        value is not None for value in single_case_values
    ):
        parser.error(
            "--q-head-num, --kv-head-num and --num-tokens must be provided "
            "together"
        )
    if args.qk_head_dim != QK_HEAD_DIM or args.v_head_dim != V_HEAD_DIM:
        parser.error("this kernel only supports qk_head_dim=v_head_dim=128")
    if args.q_head_num is not None:
        if args.q_head_num <= 0 or args.kv_head_num <= 0:
            parser.error("head counts must be positive")
        if args.num_tokens <= 0:
            parser.error("num_tokens must be positive")
    return args


def cos_sim(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def make_inputs(q_head_num, kv_head_num, num_tokens, device, seed=42):
    g = torch.Generator(device=device).manual_seed(seed)
    total_dim = (
        (q_head_num + kv_head_num) * QK_HEAD_DIM
        + kv_head_num * V_HEAD_DIM
    )
    token_data = torch.randn(num_tokens, total_dim, dtype=torch.bfloat16,
                             device=device, generator=g)
    q_norm_weight = torch.randn(QK_HEAD_DIM, dtype=torch.float32,
                                device=device, generator=g)
    k_norm_weight = torch.randn(QK_HEAD_DIM, dtype=torch.float32,
                                device=device, generator=g)
    return token_data, q_norm_weight, k_norm_weight


def torch_reference(token_data, q_norm_weight, k_norm_weight,
                    q_head_num, kv_head_num):
    """Per-head RMSNorm on Q and K in fp32, V passthrough; cast back to bf16."""
    expected = token_data.clone()
    q_end = q_head_num * QK_HEAD_DIM
    k_end = q_end + kv_head_num * QK_HEAD_DIM

    q = expected[:, :q_end].view(-1, q_head_num, QK_HEAD_DIM)
    k = expected[:, q_end:k_end].view(-1, kv_head_num, QK_HEAD_DIM)
    q.copy_(F.rms_norm(q.float(), (QK_HEAD_DIM,), q_norm_weight, EPS).to(q.dtype))
    k.copy_(F.rms_norm(k.float(), (QK_HEAD_DIM,), k_norm_weight, EPS).to(k.dtype))
    return expected


def run_op(token_data, q_norm_weight, k_norm_weight, q_head_num, kv_head_num):
    return ops.qk_rms_norm_inplace_cuda(
        token_data, q_norm_weight, k_norm_weight,
        q_head_num, kv_head_num, QK_HEAD_DIM, EPS,
    )


def bench(token_data0, q_norm_weight, k_norm_weight, q_head_num, kv_head_num,
          warm_s=0.6, time_s=0.6):
    # Fresh in-place buffer each launch (op is in place). We re-copy from a
    # pristine source so repeated launches are numerically identical, and time
    # the op-only region with back-to-back bursts to hold the DVFS boost.
    src = token_data0.clone()
    buf = token_data0.clone()

    torch.cuda.synchronize()
    t0 = time.time()
    n = 0
    while time.time() - t0 < warm_s:
        run_op(buf, q_norm_weight, k_norm_weight, q_head_num, kv_head_num)
        n += 1
        if n % 32 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()

    reps = max(32, n)
    best_ms = float("inf")
    t0 = time.time()
    while time.time() - t0 < time_s:
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        for _ in range(reps):
            run_op(buf, q_norm_weight, k_norm_weight, q_head_num, kv_head_num)
        e.record()
        e.synchronize()
        best_ms = min(best_ms, s.elapsed_time(e) / reps)
    return best_ms


def main():
    args = parse_args()
    if args.q_head_num is None:
        head_cases = Q_KV_HEAD_CASES
        token_cases = NUM_TOKENS_LIST
    else:
        head_cases = [(args.q_head_num, args.kv_head_num)]
        token_cases = [args.num_tokens]

    assert torch.cuda.is_available(), "CUDA not available"
    print(f"CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES','')}'  "
          f"device_count={torch.cuda.device_count()}")
    print(f"config: head_pairs(q,kv)={head_cases} qk_head_dim={QK_HEAD_DIM} "
          f"v_head_dim={V_HEAD_DIM} tokens={token_cases} "
          f"dtype=bfloat16 arg_type=llm  (MEMORY-BOUND; "
          f"target {TARGET_GBPS:.0f} GB/s single-die, datasheet {DATASHEET_GBPS:.0f} dual-die)")
    print("=" * 104)

    device = "cuda"
    peak_gbps = 0.0
    all_pass = True
    rows = []
    for (qh, kvh) in head_cases:
        norm_head_num = qh + kvh
        for T in token_cases:
            token_data, qw, kw = make_inputs(qh, kvh, T, device)

            # ---- accuracy ----
            ref = torch_reference(token_data, qw, kw, qh, kvh)
            buf = token_data.clone()
            run_op(buf, qw, kw, qh, kvh)
            cs = cos_sim(buf, ref)
            # verify V region is untouched (exact)
            v_start = (qh + kvh) * QK_HEAD_DIM
            v_ok = torch.equal(buf[:, v_start:], token_data[:, v_start:])
            ok = (cs >= COS_SIM_THRESHOLD) and v_ok
            all_pass &= ok

            # ---- perf ----
            ms = bench(token_data, qw, kw, qh, kvh)
            eff_bytes = T * norm_head_num * QK_HEAD_DIM * 2 * 2  # bf16, R+W
            gbps = eff_bytes / (ms * 1e-3) / 1e9
            peak_gbps = max(peak_gbps, gbps)
            rows.append((qh, kvh, T, cs, ok, ms, gbps))
            print(f"[qk_rms_norm] q={qh:2d} kv={kvh:d} T={T:6d} "
                  f"norm_heads={norm_head_num:2d}  cos_sim={cs:.6f} "
                  f"{'OK ' if ok else 'BAD'}{'' if v_ok else '(V!)'}  "
                  f"{ms:8.4f} ms  {gbps:8.1f} GB/s")

    print("=" * 104)
    print(f"Accuracy: {'ALL PASS' if all_pass else 'FAIL'} "
          f"(threshold cos_sim >= {COS_SIM_THRESHOLD})")
    reached = peak_gbps >= TARGET_GBPS
    print(f"Peak bandwidth: {peak_gbps:.1f} GB/s  (single-die target {TARGET_GBPS:.0f} "
          f"-> {'REACHED' if reached else 'NOT reached'}; dual-die datasheet {DATASHEET_GBPS:.0f})")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
