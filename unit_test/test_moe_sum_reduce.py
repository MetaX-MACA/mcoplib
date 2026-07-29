"""
Unit test for moe_sum_reduce kernel.

Tests:
1. Accuracy: cosine similarity >= 0.99999 vs torch reference
2. Performance: bandwidth (GB/s) measurement before/after optimization
3. Multiple token_num: 2048, 4096, 8192, 16384

Input:  [token_num, 8, 7168] bf16
Output: [token_num, 7168]    bf16
routed_scaling_factor: 1.0
"""

import torch
import torch.nn.functional as F
import time


def moe_sum_reduce_ref(input_tensor: torch.Tensor, routed_scaling_factor: float) -> torch.Tensor:
    """PyTorch reference: sum over topk dim, multiply by scale."""
    return torch.sum(input_tensor.float(), dim=1).to(input_tensor.dtype) * routed_scaling_factor


def benchmark_op(func, args, warmup=10, rep=100):
    """Benchmark a CUDA op, return avg latency (ms)."""
    for _ in range(warmup):
        func(*args)

    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        start_event[i].record()
        func(*args)
        end_event[i].record()
    torch.cuda.synchronize()

    dur = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float,
    )
    return dur.mean().item()


def run_kernel(input_tensor, output_tensor, routed_scaling_factor):
    """Call the moe_sum_reduce CUDA kernel."""
    torch.ops.sgl_kernel.moe_sum_reduce.default(input_tensor, output_tensor, routed_scaling_factor)


def test_moe_sum_reduce():
    torch.manual_seed(42)
    device = "cuda"
    topk = 8
    hidden_dim = 7168
    token_nums = [2048, 4096, 8192, 16384]
    routed_scaling_factor = 1.0
    dtype = torch.bfloat16

    # C500 theoretical bandwidth: 1.55 TB/s = 1550 GB/s
    theoretical_bw_gb = 1550.0

    print(f"\n{'='*80}")
    print(f"  MOE Sum Reduce Kernel Test")
    print(f"  Input: [token_num, {topk}, {hidden_dim}], dtype={dtype}")
    print(f"  Output: [token_num, {hidden_dim}], dtype={dtype}")
    print(f"  routed_scaling_factor: {routed_scaling_factor}")
    print(f"  Theoretical BW: {theoretical_bw_gb:.0f} GB/s")
    print(f"{'='*80}")

    all_pass = True
    perf_results = {}

    for token_num in token_nums:
        print(f"\n{'-'*80}")
        print(f"  token_num = {token_num}")
        print(f"{'-'*80}")

        # Create input
        x = torch.randn(token_num, topk, hidden_dim, dtype=dtype, device=device)
        output = torch.empty(token_num, hidden_dim, dtype=dtype, device=device)

        # Reference
        ref_out = moe_sum_reduce_ref(x, routed_scaling_factor)

        # Kernel
        run_kernel(x, output, routed_scaling_factor)

        # Accuracy check: cosine similarity
        cos_sim = F.cosine_similarity(
            output.flatten().float().unsqueeze(0),
            ref_out.flatten().float().unsqueeze(0),
        ).item()

        # Per-token cosine similarity
        per_token_cos = F.cosine_similarity(output.float(), ref_out.float(), dim=1)
        min_token_cos = per_token_cos.min().item()

        acc_pass = cos_sim >= 0.99999
        print(f"  Accuracy:")
        print(f"    cosine similarity:  {cos_sim:.8f}  (threshold: 0.99999)")
        print(f"    min per-token cos:  {min_token_cos:.8f}")
        print(f"    pass: {acc_pass}")

        if not acc_pass:
            print(f"    [FAIL] Accuracy below threshold!")
            all_pass = False

        # Bandwidth calculation
        # Read: token_num * topk * hidden_dim * 2 bytes (bf16 input)
        # Write: token_num * hidden_dim * 2 bytes (bf16 output)
        read_bytes = token_num * topk * hidden_dim * 2
        write_bytes = token_num * hidden_dim * 2
        total_bytes = read_bytes + write_bytes

        avg_ms = benchmark_op(run_kernel, (x, output, routed_scaling_factor), warmup=10, rep=100)
        bandwidth_gb = total_bytes / (avg_ms * 1e-3) / 1e9
        efficiency = (bandwidth_gb / theoretical_bw_gb) * 100.0

        print(f"  Performance:")
        print(f"    avg latency:    {avg_ms:.4f} ms")
        print(f"    read bytes:     {read_bytes / 1e9:.3f} GB")
        print(f"    write bytes:    {write_bytes / 1e9:.3f} GB")
        print(f"    total bytes:    {total_bytes / 1e9:.3f} GB")
        print(f"    bandwidth:      {bandwidth_gb:.2f} GB/s")
        print(f"    BW efficiency:  {efficiency:.2f}%")

        perf_results[token_num] = {
            'avg_ms': avg_ms,
            'bandwidth_gb': bandwidth_gb,
            'efficiency': efficiency,
            'cos_sim': cos_sim,
            'acc_pass': acc_pass,
        }

    # Summary
    print(f"\n{'='*80}")
    print(f"  Summary")
    print(f"{'='*80}")
    print(f"  {'token_num':>10}  {'latency(ms)':>12}  {'BW(GB/s)':>10}  {'eff%':>6}  {'cos_sim':>10}  {'pass':>5}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*6}  {'-'*10}  {'-'*5}")
    for tn in token_nums:
        r = perf_results[tn]
        print(f"  {tn:>10}  {r['avg_ms']:>12.4f}  {r['bandwidth_gb']:>10.2f}  {r['efficiency']:>6.2f}  {r['cos_sim']:>10.8f}  {'OK' if r['acc_pass'] else 'FAIL':>5}")

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*80}")

    return all_pass, perf_results


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is required")
        exit(1)

    import mcoplib.sgl_kernel

    all_pass, perf_results = test_moe_sum_reduce()
    if not all_pass:
        exit(1)
