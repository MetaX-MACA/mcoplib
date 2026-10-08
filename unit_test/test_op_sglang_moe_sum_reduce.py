"""Unit test + bandwidth benchmark for torch.ops.sgl_kernel.moe_sum_reduce on MetaX C600-U.

Op:
  moe_sum_reduce(Tensor input, Tensor output, float routed_scaling_factor) -> ()
Semantics:
  input  : [token_num, topk_num, hidden_dim]  bf16
  output : [token_num, hidden_dim]            bf16
  output[t, :] = ( sum_k input[t, k, :].float() ) * routed_scaling_factor  -> bf16

Roofline (bf16, topk=4):
  read  = topk * 2 B  = 8 B / out-elem
  write = 1    * 2 B  = 2 B / out-elem
  total = (topk+1) * 2 B = 10 B / out-elem   (4R:1W streaming, AI~0.4 op/B -> memory-bound)

Accuracy: cosine similarity vs torch fp32 reference, threshold >= 0.9999.
Bandwidth printed per shape. Free GPU auto-selected via mx-smi (or honor preset CUDA_VISIBLE_DEVICES).

Run:
  python unit_test/test_op_sglang_moe_sum_reduce.py
"""

import os
import subprocess
import torch


# ---------------------------------------------------------------------------
# Free-GPU auto-selection (respect preset CUDA_VISIBLE_DEVICES if given)
# ---------------------------------------------------------------------------
def _select_free_gpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    try:
        out = subprocess.check_output(["mx-smi"], stderr=subprocess.DEVNULL).decode()
    except Exception:
        return
    # Find physical GPU indices whose usage line shows 0% and Available.
    free = []
    cur = None
    for line in out.splitlines():
        # a GPU header line looks like: | 5  MetaX C600-UL | 10  On | ... | 0%  Disabled |
        parts = line.split("|")
        if len(parts) > 2 and "MetaX" in parts[1]:
            toks = parts[1].split()
            if toks and toks[0].isdigit():
                cur = int(toks[0])
        if cur is not None and "Available" in line and "0%" in line:
            if cur not in free:
                free.append(cur)
    if free:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(free[0])
        print(f"[gpu-select] mx-smi free GPUs {free} -> CUDA_VISIBLE_DEVICES={free[0]}")


_select_free_gpu()

import mcoplib.sgl_kernel  # noqa: E402  registers torch.ops.sgl_kernel.*

DEVICE = "cuda"
DTYPE = torch.bfloat16
HIDDEN = 6144
TOPK = 4
SCALE = 1.0
ELEM_BYTES = 2  # bf16
COS_THRESHOLD = 0.9999

TOKEN_CASES = [2048, 4096, 8192, 16384, 32768]

# HBM walls measured on this die (fp8-probe calibration, same die):
#   pure read ~1630, pure write ~1626, 1R1W copy ~1524, 2R1W ~1537 GB/s.
# 4R:1W streaming wall is measured separately; datasheet single-die = 1600.
SINGLE_DIE_DATASHEET = 1600.0
DUAL_DIE_DATASHEET = 3200.0


def moe_sum_reduce_ref(x, scale):
    """PyTorch reference: fp32 accumulate over topk, scale, cast back to bf16."""
    return (x.float().sum(dim=1) * scale).to(x.dtype)


def cosine_sim(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    dot = torch.dot(a, b).double()
    na = torch.dot(a, a).double().sqrt()
    nb = torch.dot(b, b).double().sqrt()
    return (dot / (na * nb + 1e-30)).item()


def bench(fn, warmup=15, rep=100):
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
    ms = torch.tensor([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return ms.min().item()  # best-of, matches roofline-probe methodology


def main():
    dev_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES='{dev_env}'  device_count={torch.cuda.device_count()}")
    print(f"config: topk={TOPK} hidden={HIDDEN} dtype=bf16->bf16 scale={SCALE}")
    print(f"traffic = (topk+1)*{ELEM_BYTES} = {(TOPK + 1) * ELEM_BYTES} B / out-elem (4R:1W)")
    print("=" * 92)

    torch.manual_seed(1234)
    all_pass = True
    peak = 0.0

    for T in TOKEN_CASES:
        x = torch.randn(T, TOPK, HIDDEN, dtype=DTYPE, device=DEVICE)
        y = torch.empty(T, HIDDEN, dtype=DTYPE, device=DEVICE)

        # correctness
        torch.ops.sgl_kernel.moe_sum_reduce(x, y, SCALE)
        torch.cuda.synchronize()
        ref = moe_sum_reduce_ref(x, SCALE)
        cs = cosine_sim(y, ref)
        ok = cs >= COS_THRESHOLD
        all_pass &= ok

        # bandwidth: total HBM bytes = read topk + write 1, all bf16
        total_bytes = T * HIDDEN * (TOPK + 1) * ELEM_BYTES
        ms = bench(lambda: torch.ops.sgl_kernel.moe_sum_reduce(x, y, SCALE))
        gbps = total_bytes / (ms * 1e-3) / 1e9
        peak = max(peak, gbps)

        print(f"[moe_sum_reduce] T={T:6d}  cos_sim={cs:.6f} {'OK  ' if ok else 'FAIL'}"
              f"  {ms:8.4f} ms  {gbps:8.1f} GB/s")

    print("=" * 92)
    print(f"Accuracy: {'ALL PASS' if all_pass else 'FAILURE'} (threshold cos_sim >= {COS_THRESHOLD})")
    tgt = SINGLE_DIE_DATASHEET
    reached = "reached" if peak >= 0.85 * tgt else "NOT reached"
    print(f"Peak bandwidth: {peak:.1f} GB/s  "
          f"(single-die datasheet {tgt:.0f}, 85% target {0.85*tgt:.0f} -> {reached}; "
          f"dual-die datasheet {DUAL_DIE_DATASHEET:.0f})")
    return all_pass


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
