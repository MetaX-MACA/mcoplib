import os
import sys
import time

import torch

import mcoplib.sgl_kernel  # noqa: F401  (registers torch.ops.sgl_kernel.*)


# ---- shape config ---------------------------------------------------------
# Radix-select path (length > TopK): mirror the fused test's coverage plus large seqs.
BS_LIST = [ 132, 256, 1662, 4096]
SEQ_LEN_LIST = [4096, 18560, 18688, 32768, 65536, 66551, 107520]  # incl. single-window sample band [16385,65536] (real-model rows, trap regression guard)
# Real DeepSeek-v3.2 model shapes that hit the naive (length <= TopK) path.
REAL_MODEL_SHAPES = [(6, 2048), (1, 64)]     # (bs, seq_len)
K = 2048                      # TopK, deepseek v3.2
MAX_SEQ_LEN = 131072

TARGET_GBPS = 1300.0          # single-die gate
TARGET_85_GBPS = 0.85 * 1300.0
DUAL_DIE_GBPS = 3480.0        # dual-die datasheet (unreachable single-die)


def _prefill_inputs(bs, seq_len, device, seed=42):
    g = torch.Generator(device=device).manual_seed(seed)
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device=device, generator=g)
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device=device)
    src_page_table = torch.arange(0, seq_len, dtype=torch.int32, device=device)
    src_page_table = src_page_table.unsqueeze(0).expand(bs, -1)             # [bs, seq_len]
    cu_seqlens_q = torch.arange(0, bs + 1, dtype=torch.int32, device=device)
    row_starts = torch.zeros(bs, dtype=torch.int32, device=device)         # force PREFILL
    dst_page_table = score.new_empty((bs, K), dtype=torch.int32)
    return score, lengths, src_page_table, cu_seqlens_q, row_starts, dst_page_table


def run_op(score, lengths, dst, src, cu_seqlens_q, row_starts):
    torch.ops.sgl_kernel.fast_topk_transform_fused(
        score=score, lengths=lengths, dst_page_table=dst,
        src_page_table=src, cu_seqlens_q=cu_seqlens_q, row_starts=row_starts)


def check_setequal(score, our, seq_len, bs, max_permit_error=5):
    """Set-equality vs torch.topk with a tie tolerance on equal score values.
    (src_page_table is arange over the row, so a page index == its score index.)"""
    ref = torch.topk(score[:, :seq_len], K, dim=-1, sorted=False).indices
    our_cpu = our.cpu().tolist()
    ref_cpu = ref.cpu().tolist()
    wrong = 0
    for i in range(bs):
        our_set = set(our_cpu[i])
        ref_set = set(ref_cpu[i])
        more = our_set - ref_set
        less = ref_set - our_set
        if more or less:
            more_v = sorted(score[i, idx].item() for idx in more)
            less_v = sorted(score[i, idx].item() for idx in less)
            if more_v != less_v:            # not just a tie swap
                wrong += len(more)
    return wrong <= max_permit_error, wrong


def check_naive(our, seq_len, bs):
    """Naive path: dst[i]=src[i]=i for i<length, else -1."""
    ref = torch.full((bs, K), -1, dtype=torch.int32, device=our.device)
    ref[:, :seq_len] = torch.arange(0, seq_len, dtype=torch.int32, device=our.device)
    return torch.equal(our, ref)


def bench(fn, warm_s=0.5, time_s=0.6):
    torch.cuda.synchronize()
    t0 = time.time()
    n = 0
    while time.time() - t0 < warm_s:
        fn(); n += 1
        if n % 8 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    reps = max(10, n)
    best_ms = float("inf")
    t0 = time.time()
    while time.time() - t0 < time_s:
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(reps):
            fn()
        e.record(); e.synchronize()
        best_ms = min(best_ms, s.elapsed_time(e) / reps)
    return best_ms


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES','')}'  "
          f"device_count={torch.cuda.device_count()}")
    print(f"config: kernel=topk_transform_prefill  k={K} max_seq_len={MAX_SEQ_LEN} "
          f"dtype=fp32  (MEMORY-BOUND; target {TARGET_GBPS:.0f} GB/s single-die / "
          f"85%={TARGET_85_GBPS:.0f}, datasheet {DUAL_DIE_GBPS:.0f} dual-die)")
    print("=" * 104)

    device = "cuda"
    peak_gbps = 0.0
    all_pass = True

    # ---- naive-path real-model shapes (exact match) -----------------------
    for bs, seq_len in REAL_MODEL_SHAPES:
        score, lengths, src, cu, rs, dst = _prefill_inputs(bs, seq_len, device)
        run_op(score, lengths, dst, src, cu, rs)
        ok = check_naive(dst, seq_len, bs)
        all_pass &= ok
        ms = bench(lambda: run_op(score, lengths, dst, src, cu, rs))
        # naive path traffic: read <=length score + write TopK gather
        eff_bytes = bs * (seq_len + K) * 4
        gbps = eff_bytes / (ms * 1e-3) / 1e9
        print(f"[topk_prefill] bs={bs:5d} seq_len={seq_len:6d} k={K}  "
              f"{'EXACT OK' if ok else 'MISMATCH'}          {ms:9.4f} ms  {gbps:8.1f} GB/s")

    # ---- radix-select path (length > TopK) --------------------------------
    for bs in BS_LIST:
        for seq_len in SEQ_LEN_LIST:
            score, lengths, src, cu, rs, dst = _prefill_inputs(bs, seq_len, device)
            run_op(score, lengths, dst, src, cu, rs)
            ok, wrong = check_setequal(score, dst, seq_len, bs)
            all_pass &= ok
            ms = bench(lambda: run_op(score, lengths, dst, src, cu, rs))
            eff_bytes = bs * (2 * seq_len + K) * 4
            gbps = eff_bytes / (ms * 1e-3) / 1e9
            peak_gbps = max(peak_gbps, gbps)
            print(f"[topk_prefill] bs={bs:5d} seq_len={seq_len:6d} k={K}  "
                  f"{'SET OK ' if ok else 'SET BAD'}(wrong={wrong:4d})  "
                  f"{ms:9.4f} ms  {gbps:8.1f} GB/s")

    print("=" * 104)
    print(f"Accuracy: {'ALL PASS' if all_pass else 'FAIL'}  (set-equality, tie-tolerant)")
    reached = peak_gbps >= TARGET_GBPS
    print(f"Peak bandwidth (radix path): {peak_gbps:.1f} GB/s  (single-die target "
          f"{TARGET_GBPS:.0f} -> {'REACHED' if reached else 'NOT reached'}; "
          f"85%={TARGET_85_GBPS:.0f}; dual-die datasheet {DUAL_DIE_GBPS:.0f})")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
