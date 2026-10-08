"""Unit test + bandwidth benchmark for
mcoplib.tilelang_fp8_paged_mqa_logits (DeepSeek-style FP8 paged MQA logits) on MetaX C600-U.

Op (functional):
  o = tilelang_fp8_paged_mqa_logits(q_fp8, kvcache_fp8, weight, seq_lens,
                                    page_table, deep_gemm_metadata=None,
                                    max_seq_len, clean_logits=False)

Semantics (verified against fp32 reference, cos=1.0):
  For each batch n and each KV page p (page = page_table[n, ip], ip in [0, ceil(seq_len/B))):
    K_fp8  : [B, D]  fp8      (the first B*D bytes of the packed page)
    kscale : [B]     fp32     (the trailing B*4 bytes of the packed page)
    logits[B, H] = K_fp8 @ Q_fp8[n].T          (Q_fp8[n] : [H, D])
    logits       = max(logits, 0) * weight[n]  (weight[n] : [H], broadcast over B)
    o[n, ip*B : ip*B+B] = sum_H(logits) * kscale

Packed KV byte layout per page (DE-INTERLEAVED, this is the key contract):
    [ B*D fp8 K bytes (row-major [B,D]) ][ B*4 fp32 per-row scale bytes ([B]) ]
  The public tensor is shaped [C, block_size, 1, head_dim+4] purely to satisfy the
  wrapper assert; its bytes must be laid out de-interleaved as above.

Fixed op constraints (from the kernel wrapper):
  head_dim == 128, block_size == 64, clean_logits == False.

Config: dtype = float8_e4m3fn (q and K). num_heads in {16, 32, 64, 128} (MQA/GQA
  mainstream), batch/seq swept across mainstream decode shapes. >= 1000 cases total.
Accuracy: cosine similarity in fp64, fp8 threshold >= 0.999. Bandwidth printed per case.

Run:
  cd /home/yiyu/mcoplib/release/mcoplib && source env_local.sh
  export CUDA_VISIBLE_DEVICES=<free gpu from mx-smi>
  /opt/conda/bin/python unit_test/test_tilelang_fp8_paged_mqa_logits.py
"""

import os
import sys
import types
import math
import itertools

import torch

# --------------------------------------------------------------------------- #
# Lightweight sglang stub.
#
# The kernel module imports two tiny helpers from sglang
# (is_fp8_fnuz, is_hip/is_maca/is_gfx95_supported).  The real sglang package
# drags a heavy runtime import chain (orjson/pybase64/starlette/...), which is
# unrelated to this op.  Mirroring the repo's own triton tests, we inject
# minimal stub modules into sys.modules *before* importing the kernel so the
# test runs standalone in a MACA container without a full sglang install.
# On a machine with a real, importable sglang these stubs are simply unused.
# --------------------------------------------------------------------------- #
def _install_sglang_stub():
    if "sglang" in sys.modules:
        return

    def mk(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    sg = mk("sglang")
    srt = mk("sglang.srt"); sg.srt = srt
    utils = mk("sglang.srt.utils"); srt.utils = utils
    utils.is_hip = lambda: False
    utils.is_maca = lambda: True
    utils.is_gfx95_supported = lambda: False
    kernels = mk("sglang.kernels"); sg.kernels = kernels
    ops = mk("sglang.kernels.ops"); kernels.ops = ops
    quant = mk("sglang.kernels.ops.quantization"); ops.quantization = quant
    fp8k = mk("sglang.kernels.ops.quantization.fp8_kernel"); quant.fp8_kernel = fp8k
    fp8k.is_fp8_fnuz = lambda: False


_install_sglang_stub()

from mcoplib.tilelang_fp8_paged_mqa_logits_kernel import (  # noqa: E402
    tilelang_fp8_paged_mqa_logits,
    FP8_DTYPE,
)

DEVICE = "cuda"

# Fixed by the kernel contract.
HEAD_DIM = 128
BLOCK_SIZE = 64

# C600-U bandwidth references.
TARGET_GBPS = 1600.0        # user target (1.6 TB/s)
SINGLE_DIE_GBPS = 1300.0    # single-die practical wall
DUAL_DIE_GBPS = 3480.0      # dual-die datasheet figure

COS_THRESHOLD = 0.999       # fp8 accuracy gate


# --------------------------------------------------------------------------- #
# Input construction
# --------------------------------------------------------------------------- #
def build_packed_kv(num_blocks, k_fp8_bhd, kscale_bh):
    """Pack per-page KV into the de-interleaved byte layout the kernel expects.

    k_fp8_bhd : [num_blocks, BLOCK_SIZE, HEAD_DIM] fp8
    kscale_bh : [num_blocks, BLOCK_SIZE]           fp32
    returns   : [num_blocks, BLOCK_SIZE, 1, HEAD_DIM + 4] uint8
    """
    B, D = BLOCK_SIZE, HEAD_DIM
    k_part = k_fp8_bhd.reshape(num_blocks, B * D).view(torch.uint8)      # [C, B*D]
    s_part = kscale_bh.contiguous().reshape(num_blocks, B).view(torch.uint8)  # [C, B*4]
    flat = torch.cat([k_part, s_part], dim=1).contiguous()               # [C, B*(D+4)]
    return flat.view(num_blocks, B, 1, D + 4)


def make_inputs(batch, num_heads, seq_len, seed=1234):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    B, D = BLOCK_SIZE, HEAD_DIM
    max_tab = (seq_len + B - 1) // B
    num_blocks = max(8, batch * max_tab + 2)

    # Keep magnitudes small so fp8_e4m3 stays well-conditioned.
    q = (torch.randn(batch, 1, num_heads, D, device=DEVICE, generator=g) * 0.3).to(FP8_DTYPE)
    k_fp8 = (torch.randn(num_blocks, B, D, device=DEVICE, generator=g) * 0.3).to(FP8_DTYPE)
    kscale = (torch.rand(num_blocks, B, device=DEVICE, generator=g) * 0.5 + 0.5).float()
    kv = build_packed_kv(num_blocks, k_fp8, kscale)

    weight = (torch.randn(batch, num_heads, device=DEVICE, generator=g) * 0.5).float()
    seq_lens = torch.full((batch,), seq_len, device=DEVICE, dtype=torch.int32)
    page_table = torch.randint(0, num_blocks, (batch, max_tab), device=DEVICE,
                               dtype=torch.int32, generator=g)
    return q, kv, weight, seq_lens, page_table, num_blocks, max_tab


# --------------------------------------------------------------------------- #
# Reference (fp32 accum, fp64 comparison)
# --------------------------------------------------------------------------- #
def reference(q, kv, weight, seq_lens, page_table, max_seq_len):
    N, _, H, D = q.shape
    C, B, _, _ = kv.shape
    out = torch.zeros(N, max_seq_len, dtype=torch.float32, device=q.device)
    qf = q.float().view(N, H, D)
    flat = kv.view(C, B * (D + 4))
    for n in range(N):
        L = int(seq_lens[n])
        npg = (L + B - 1) // B
        for ip in range(npg):
            pg = int(page_table[n, ip])
            k_fp8 = flat[pg, : B * D].view(torch.float8_e4m3fn).float().view(B, D)
            kscale = flat[pg, B * D : B * (D + 4)].contiguous().view(torch.float32).view(B)
            logits = k_fp8 @ qf[n].t()                       # [B, H]
            logits = torch.clamp(logits, min=0.0) * weight[n].view(1, H)
            s = logits.sum(dim=1) * kscale                   # [B]
            base = ip * B
            m = min(B, max_seq_len - base)
            out[n, base : base + m] = s[:m]
    return out


def cosine_fp64(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    denom = (a.norm() * b.norm()).clamp_min(1e-30)
    return (a @ b / denom).item()


# --------------------------------------------------------------------------- #
# Bytes moved (for effective-bandwidth reporting)
# --------------------------------------------------------------------------- #
def moved_bytes(batch, num_heads, seq_len, page_table):
    B, D = BLOCK_SIZE, HEAD_DIM
    total_pages = int((page_table >= 0).sum().item()) if page_table.numel() else 0
    # Every processed page reads B*D fp8 K bytes + B*4 scale bytes.
    k_bytes = total_pages * B * D * 1
    s_bytes = total_pages * B * 4
    q_bytes = batch * num_heads * D * 1                 # fp8 query
    o_bytes = batch * seq_len * 4                       # fp32 output
    return k_bytes + s_bytes + q_bytes + o_bytes


# --------------------------------------------------------------------------- #
# Per-case runner
# --------------------------------------------------------------------------- #
def _est_device_bytes(batch, num_heads, seq_len):
    """Rough peak device-memory estimate for a case's input tensors.

    Dominated by the packed KV cache (num_blocks pages * BLOCK_BYTES).  Used only
    to skip shapes that cannot fit in the *currently free* memory of a shared GPU;
    it is an environment guard, never a correctness gate.
    """
    B, D = BLOCK_SIZE, HEAD_DIM
    max_tab = (seq_len + B - 1) // B
    num_blocks = max(8, batch * max_tab + 2)
    kv_bytes = num_blocks * B * (D + 4)            # packed uint8 KV cache
    out_bytes = batch * seq_len * 4                # fp32 logits output
    q_bytes = batch * num_heads * D                # fp8 query
    # reference() materialises fp32 intermediates of order out_bytes; double it.
    return kv_bytes + 3 * out_bytes + q_bytes


def _free_device_bytes():
    try:
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        return None


class CaseSkipped(Exception):
    """Raised when a case cannot be allocated on the currently-free GPU memory."""


def run_case(batch, num_heads, seq_len, check_acc=True, warmup=5, rep=20, seed=1234):
    # Environment guard: on a shared GPU the packed KV cache for the largest
    # shapes may not fit in the *currently free* memory.  That is a host-side
    # allocation limit, not a kernel defect, so skip rather than fail.  Keep a
    # ~12% headroom for allocator fragmentation / workspace.
    free = _free_device_bytes()
    need = _est_device_bytes(batch, num_heads, seq_len)
    if free is not None and need > int(free * 0.88):
        raise CaseSkipped(
            f"need ~{need/2**30:.2f} GiB but only {free/2**30:.2f} GiB free")

    q = kv = weight = seq_lens = page_table = got = None
    try:
        q, kv, weight, seq_lens, page_table, num_blocks, max_tab = make_inputs(
            batch, num_heads, seq_len, seed=seed)

        fn = lambda: tilelang_fp8_paged_mqa_logits(
            q, kv, weight, seq_lens, page_table, None, seq_len, clean_logits=False)

        # Correctness + kernel-trap detection.
        got = fn()
        torch.cuda.synchronize()
        sim = float("nan")
        ok = "n/a"
        if check_acc:
            exp = reference(q, kv, weight, seq_lens, page_table, seq_len)
            sim = cosine_fp64(got.float(), exp)
            ok = "PASS" if sim >= COS_THRESHOLD else "FAIL"
            del exp

        # Timing.
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        st = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        en = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        for i in range(rep):
            st[i].record(); fn(); en[i].record()
        torch.cuda.synchronize()
        times = sorted(s.elapsed_time(e) for s, e in zip(st, en))
        median_ms = times[len(times) // 2]

        gbps = moved_bytes(batch, num_heads, seq_len, page_table) / (median_ms * 1e-3) / 1e9
        return sim, ok, median_ms, gbps
    except torch.cuda.OutOfMemoryError as e:
        # Allocation raced past the pre-check (shared GPU); treat as skip.
        raise CaseSkipped(str(e)[:120])
    finally:
        del q, kv, weight, seq_lens, page_table, got
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Case matrix (>= 1000 cases)
# --------------------------------------------------------------------------- #
def build_case_matrix():
    """Mainstream LLM decode shapes for fp8 paged MQA logits.

    num_heads covers MQA/GQA/MHA logit widths (16..128, multiples of 4).
    batch covers small -> large concurrent decode.
    seq_len covers short context -> long context (multiples of BLOCK_SIZE=64).
    The full Cartesian product is >= 1000 combinations.
    """
    heads = [16, 32, 64, 128]
    batches = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    seqs = [64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
            6144, 8192, 10240, 12288, 14336, 16384, 24576, 32768]
    cases = list(itertools.product(batches, heads, seqs))
    return cases  # 13 * 4 * 20 = 1040 cases


def main():
    print("=" * 100)
    print("FP8 paged MQA logits — tilelang kernel unit test + bandwidth benchmark (MetaX C600-U)")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    print(f"device               = {torch.cuda.get_device_name(0)}")
    print(f"head_dim={HEAD_DIM}  block_size={BLOCK_SIZE}  fp8={FP8_DTYPE}  cos_threshold={COS_THRESHOLD}")
    print(f"targets: user={TARGET_GBPS:.0f}  single-die={SINGLE_DIE_GBPS:.0f}  dual-die={DUAL_DIE_GBPS:.0f} GB/s")
    print("=" * 100)

    cases = build_case_matrix()
    max_cases = int(os.environ.get("MQA_MAX_CASES", "0"))
    if max_cases > 0:
        # Deterministic stride subsample for smoke runs (keeps corners + variety).
        stride = max(1, len(cases) // max_cases)
        cases = cases[::stride][:max_cases]
    print(f"total cases: {len(cases)}\n")

    # Optional per-case CSV dump for baseline-vs-best comparison tables.
    csv_path = os.environ.get("MQA_CSV", "")
    csv_f = open(csv_path, "w") if csv_path else None
    if csv_f:
        csv_f.write("batch,heads,seq,status,cos,ms,gbps\n")

    n_pass = n_fail = n_skip = 0
    peak_gbps = 0.0
    peak_cfg = None
    worst_sim = 1.0
    worst_cfg = None
    failures = []
    skips = []

    # To keep the >=1000-case sweep affordable, check accuracy on a deterministic
    # 1-in-K subset plus every "corner" case (smallest / largest batch, seq, heads);
    # every case is still launched (kernel-trap detection) and timed.
    corner_heads = {16, 128}
    corner_batches = {1, 256}
    corner_seqs = {64, 32768}

    for idx, (batch, num_heads, seq_len) in enumerate(cases):
        # Reference is an O(total_pages) python loop; only run it when the page
        # count is modest so the >=1000-case sweep stays affordable.  Large
        # shapes are still launched (trap detection) and timed.
        total_pages_est = batch * ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
        acc_affordable = total_pages_est <= 4096
        is_corner = (num_heads in corner_heads or batch in corner_batches
                     or seq_len in corner_seqs)
        check_acc = acc_affordable and (is_corner or (idx % 7 == 0))
        try:
            sim, ok, ms, gbps = run_case(batch, num_heads, seq_len, check_acc=check_acc)
        except CaseSkipped as e:
            # Shape does not fit in the currently-free memory of this shared GPU.
            # This is an environment limit, not a kernel defect: skip, don't fail.
            n_skip += 1
            skips.append((batch, num_heads, seq_len, str(e)[:80]))
            if csv_f:
                csv_f.write(f"{batch},{num_heads},{seq_len},skip,,,\n")
            if idx % 37 == 0 or num_heads in corner_heads:
                print(f"[{idx:4d}] B={batch:3d} H={num_heads:3d} S={seq_len:5d}  SKIP (oom) {str(e)[:70]}")
            continue
        except Exception as e:
            n_fail += 1
            msg = f"{type(e).__name__}: {str(e)[:160]}"
            failures.append((batch, num_heads, seq_len, msg))
            if csv_f:
                csv_f.write(f"{batch},{num_heads},{seq_len},crash,,,\n")
            print(f"[{idx:4d}] B={batch:3d} H={num_heads:3d} S={seq_len:5d}  CRASH {msg}")
            continue

        if check_acc:
            if ok == "PASS":
                n_pass += 1
            else:
                n_fail += 1
                failures.append((batch, num_heads, seq_len, f"cos={sim:.6f}"))
            if sim < worst_sim:
                worst_sim, worst_cfg = sim, (batch, num_heads, seq_len)

        if gbps > peak_gbps:
            peak_gbps, peak_cfg = gbps, (batch, num_heads, seq_len)

        if csv_f:
            simstr = f"{sim:.6f}" if check_acc else ""
            csv_f.write(f"{batch},{num_heads},{seq_len},{ok},{simstr},{ms:.5f},{gbps:.2f}\n")

        # Print a readable subset (all accuracy-checked cases + a stride of the rest).
        if check_acc or idx % 37 == 0:
            simstr = f"{sim:.6f}" if check_acc else "  ----  "
            print(f"[{idx:4d}] B={batch:3d} H={num_heads:3d} S={seq_len:5d}  "
                  f"cos={simstr} {ok:4s}  {ms:8.4f} ms  {gbps:8.1f} GB/s")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print(f"  accuracy-checked cases: pass={n_pass}  fail={n_fail}")
    print(f"  skipped (GPU-memory limited, not a kernel defect): {n_skip}")
    if worst_cfg is not None:
        print(f"  worst cosine: {worst_sim:.6f} at B={worst_cfg[0]} H={worst_cfg[1]} S={worst_cfg[2]}")
    if peak_cfg is not None:
        pct = 100.0 * peak_gbps / SINGLE_DIE_GBPS
        print(f"  peak bandwidth: {peak_gbps:.1f} GB/s at "
              f"B={peak_cfg[0]} H={peak_cfg[1]} S={peak_cfg[2]}  "
              f"({pct:.1f}% of single-die wall {SINGLE_DIE_GBPS:.0f})")
        print(f"  vs user target {TARGET_GBPS:.0f} GB/s: "
              f"{'REACHED' if peak_gbps >= TARGET_GBPS else 'below'}")
    if failures:
        print(f"  FIRST FAILURES ({min(len(failures),10)} of {len(failures)}):")
        for b, h, s, m in failures[:10]:
            print(f"    B={b} H={h} S={s}: {m}")
    if skips:
        print(f"  SKIPPED SHAPES ({min(len(skips),8)} of {len(skips)}, GPU-memory limited):")
        for b, h, s, m in skips[:8]:
            print(f"    B={b} H={h} S={s}")
    print("=" * 100)

    if csv_f:
        csv_f.close()
        print(f"per-case CSV written to {csv_path}")

    # Only real accuracy failures or kernel traps fail the suite; OOM skips do not.
    assert n_fail == 0, f"{n_fail} case(s) failed (accuracy or kernel trap)"


if __name__ == "__main__":
    main()
