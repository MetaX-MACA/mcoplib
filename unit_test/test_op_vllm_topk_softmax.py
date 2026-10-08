import sys
import torch

import mcoplib._moe_C  # noqa: F401  registers torch.ops._moe_C.topk_softmax

TARGET_GBPS = 1300.0        # single-die C500 target
DATASHEET_DUAL = 3480.0     # dual-die datasheet peak (context only)
COS_SIM_PASS = 0.99999

# (num_experts, topk) pairs derived from the required config tuples
#   (NUM_SHARED, NUM_EXPERTS, NUM_EXPERT_GROUP, TOPK_GROUP, TOPK)
# The GROUP / SHARED columns are model-config context and do not map to
# topk_softmax kernel args; only (num_experts, topk) reach the op.
CONFIGS = [
    (160, 8), (256, 8), (288, 8), (320, 8), (384, 8), (448, 8), (896, 16),
    (160, 9), (256, 9), (320, 9), (384, 9), (448, 9),
]

# token sweep: small -> latency bound, large -> bandwidth bound
TOKEN_SWEEP = [1, 16, 64, 256, 1024, 4096,5760, 8192, 16384, 32768]

WARMUP = 20
ITERS = 100


def topk_softmax(topk_weights, topk_indices, token_expert_indices,
                 gating_output, renormalize=False, bias=None):
    torch.ops._moe_C.topk_softmax(
        topk_weights, topk_indices, token_expert_indices,
        gating_output, renormalize, bias)


def reference(gating_output, topk, renormalize, bias=None):
    """torch softmax + topk reference (fp32)."""
    g = gating_output.float()
    probs = torch.softmax(g, dim=-1)
    if bias is not None:
        choice = probs + bias.float().unsqueeze(0)
    else:
        choice = probs
    # select by (biased) choice, but return unbiased probs as weights
    _, idx = torch.topk(choice, topk, dim=-1)
    w = torch.gather(probs, 1, idx)
    if renormalize:
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return w, idx


def cos_sim(a, b):
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    denom = (a.norm() * b.norm()).clamp_min(1e-30)
    return float((a @ b) / denom)


def values_match(gating_output, idx_ref, idx_test):
    """Tie-safe: the *values* at the selected experts must match as a set."""
    g = gating_output.float()
    vr = torch.gather(g, 1, idx_ref.long()).sort(dim=-1).values
    vt = torch.gather(g, 1, idx_test.long()).sort(dim=-1).values
    return torch.allclose(vr, vt, atol=1e-2, rtol=1e-2)


def bench_one(num_experts, topk, renormalize, dtype, verbose=True):
    peak = 0.0
    all_pass = True
    for T in TOKEN_SWEEP:
        gating = torch.randn((T, num_experts), dtype=dtype, device="cuda")
        tw = torch.empty((T, topk), dtype=torch.float32, device="cuda")
        ti = torch.empty((T, topk), dtype=torch.int32, device="cuda")
        tei = torch.empty((T, topk), dtype=torch.int32, device="cuda")

        # correctness
        topk_softmax(tw, ti, tei, gating, renormalize, None)
        torch.cuda.synchronize()
        w_ref, idx_ref = reference(gating, topk, renormalize, None)
        # sort both weight vectors descending for a stable cos_sim
        w_test_sorted = tw.sort(dim=-1, descending=True).values
        w_ref_sorted = w_ref.sort(dim=-1, descending=True).values
        cs = cos_sim(w_ref_sorted, w_test_sorted)
        vm = values_match(gating, idx_ref, ti)
        ok = (cs >= COS_SIM_PASS) and vm
        all_pass = all_pass and ok

        # timing
        for _ in range(WARMUP):
            topk_softmax(tw, ti, tei, gating, renormalize, None)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            topk_softmax(tw, ti, tei, gating, renormalize, None)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / ITERS

        read_bytes = T * num_experts * gating.element_size()
        write_bytes = T * topk * (4 + 4 + 4)
        gbps = (read_bytes + write_bytes) / (ms * 1e-3) / 1e9
        peak = max(peak, gbps)

        if verbose:
            status = "OK  " if ok else "FAIL"
            print(f"[topk_softmax] E={num_experts:4d} k={topk:2d} "
                  f"renorm={int(renormalize)} T={T:6d}  "
                  f"cos_sim={cs:.6f} {status} "
                  f"{ms:8.4f} ms  {gbps:8.1f} GB/s")
    return peak, all_pass


def main():
    dtype = torch.bfloat16
    if len(sys.argv) > 1 and sys.argv[1] == "fp32":
        dtype = torch.float32
    print(f"=== topk_softmax bandwidth/accuracy  dtype={dtype} ===")
    global_peak = 0.0
    global_pass = True
    peak_table = []
    for (E, k) in CONFIGS:
        for renorm in (False, True):
            peak, ok = bench_one(E, k, renorm, dtype)
            global_peak = max(global_peak, peak)
            global_pass = global_pass and ok
            peak_table.append((E, k, renorm, peak, ok))
            print()

    print("================ PEAK PER CONFIG ================")
    for (E, k, renorm, peak, ok) in peak_table:
        print(f"  E={E:4d} k={k:2d} renorm={int(renorm)}: "
              f"peak {peak:8.1f} GB/s  {'PASS' if ok else 'FAIL'}")
    reached = "reached" if global_peak >= TARGET_GBPS else "NOT reached"
    print(f"\nPeak bandwidth: {global_peak:.1f} GB/s  "
          f"(single-die target {int(TARGET_GBPS)} -> {reached}; "
          f"dual-die datasheet {int(DATASHEET_DUAL)})")
    print(f"Accuracy: {'ALL PASS' if global_pass else 'SOME FAILED'} "
          f"(cos_sim >= {COS_SIM_PASS})")
    sys.exit(0 if global_pass else 1)


if __name__ == "__main__":
    main()
