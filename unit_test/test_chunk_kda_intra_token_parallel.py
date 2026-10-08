# SPDX-License-Identifier: Apache-2.0
"""Accuracy + bandwidth unit test for chunk_kda_fwd_kernel_intra_token_parallel
(KDA prefill intra-chunk diagonal-block stage, MetaX C600-U / MACA).

Semantics per token i (head h), sub-chunk start i_ts = bos-relative BC block:
  for j in [i_ts, i]:
    Aqk[i, h, j % BT]   = scale * sum_K q_i * k_j * 2^(g_i - g_j)          (j <= i)
    Akk[i, h, j - i_ts] =         sum_K (beta_i*k_i) * k_j * 2^(g_i - g_j) (j <  i; 0 on diag)
Only the diagonal sub-chunk block is written; inter_solve fills the rest of Aqk.

Roofline (single die): memory-bound. Minimal HBM traffic = read q,k,g,beta once +
write the written Aqk/Akk cells once. The naive token-parallel kernel additionally
re-reads k_j/g_j ~8.5x per token (85% of its traffic), so effective GB/s (min bytes
/ time) exposes exactly that redundancy.  Read-dominated => ceiling ~ read-only wall.

Run (pick a free card first):
    cd /home/yiyu/mcoplib/mcoplib_dev/mcoplib
    source env_local.sh
    mx-smi                       # find GPU-Util 0% / ~540MiB Available
    export CUDA_VISIBLE_DEVICES=8
    /opt/conda/bin/python unit_test/test_op_chunk_kda_intra_token_parallel.py

Output format aligns with test_op_moe_scatter_dynamic_quant.py.
"""

import os
import pathlib
import sys
import time

import torch

# make `import mcoplib.<kernel>` resolve from the package root (…/mcoplib_dev/mcoplib)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mcoplib.triton_sglang_chunk_intra_token_parallel import (  # noqa: E402
    chunk_kda_fwd_intra_token_parallel,
)

# --------------------------------------------------------------------------- #
# Config (per requirement).
# --------------------------------------------------------------------------- #
H = 12                       # local KDA heads = num_heads/attn_tp = 96/8
K = 128                      # head_dim
BT = 64                      # chunk_size
BC = 16                      # sub_chunk_size
B = 1                        # varlen packed
SCALE = K ** -0.5
TOKEN_COUNTS = (2048, 4096, 8192, 16384)

COS_THRESHOLD = 0.99999
READONLY_WALL = 1474.0       # single-die read-only ceiling (op is read-dominated)
SINGLE_DIE_TARGET = 1600.0   # nominal target from requirement
DUAL_DIE_DATASHEET = 3480.0


# --------------------------------------------------------------------------- #
# Inputs (match production packing: uneven varlen + a short <BC tail).
# --------------------------------------------------------------------------- #
def _make_cu_seqlens(n):
    return torch.tensor(
        [0, int(n * 0.37), int(n * 0.66), n - 8, n],
        device="cuda", dtype=torch.int64,
    )


def _make_inputs(n, cu_seqlens, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, n, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, n, H, K, device="cuda", dtype=torch.bfloat16)
    beta = torch.rand(B, n, H, device="cuda", dtype=torch.bfloat16)
    # gk = chunk-local cumsum of negative gate increments (kda_gate_chunk_cumsum, fp32)
    step = -torch.rand(n, H, K, device="cuda", dtype=torch.float32) * 0.05
    gk = torch.empty_like(step)
    bounds = cu_seqlens.tolist()
    for bos, eos in zip(bounds[:-1], bounds[1:]):
        for c0 in range(bos, eos, BT):
            c1 = min(c0 + BT, eos)
            gk[c0:c1] = step[c0:c1].cumsum(0)
    return q, k, gk.unsqueeze(0), beta


def _launch(q, k, gk, beta, Aqk, Akk, cu_seqlens):
    chunk_kda_fwd_intra_token_parallel(
        q=q, k=k, gk=gk, beta=beta, Aqk=Aqk, Akk=Akk,
        scale=SCALE, cu_seqlens=cu_seqlens, chunk_size=BT, sub_chunk_size=BC,
    )


# --------------------------------------------------------------------------- #
# fp64 reference (factorized within each sub-chunk; mathematically identical to
# the pairwise 2^(g_i-g_j) form). Returns filled Aqk/Akk refs + written masks.
# --------------------------------------------------------------------------- #
def _reference(q, k, gk, beta, cu_seqlens, n):
    qf, kf, gf, bf = q.double(), k.double(), gk.double(), beta.double()
    Aqk_ref = torch.zeros(B, n, H, BT, device="cuda", dtype=torch.float64)
    Akk_ref = torch.zeros(B, n, H, BC, device="cuda", dtype=torch.float64)
    maskA = torch.zeros(B, n, H, BT, device="cuda", dtype=torch.bool)
    maskK = torch.zeros(B, n, H, BC, device="cuda", dtype=torch.bool)

    bounds = cu_seqlens.tolist()
    for bos, eos in zip(bounds[:-1], bounds[1:]):
        L = eos - bos
        for st in range(0, L, BC):
            m = min(BC, L - st)
            r0 = bos + st
            Q = qf[0, r0:r0 + m]          # [m,H,K]
            Kk = kf[0, r0:r0 + m]
            G = gf[0, r0:r0 + m]
            Bt = bf[0, r0:r0 + m]         # [m,H]
            gr = G - G[0:1]               # ref = sub-chunk first row (per channel)
            qg = Q * torch.exp2(gr)
            kg = Kk * torch.exp2(-gr)
            kbg = (Bt.unsqueeze(-1) * Kk) * torch.exp2(gr)
            Aqk_blk = SCALE * torch.einsum("ihk,jhk->ijh", qg, kg)   # [m,m,H]
            Akk_blk = torch.einsum("ihk,jhk->ijh", kbg, kg)          # [m,m,H]

            tri = torch.tril(torch.ones(m, m, device="cuda", dtype=torch.bool))
            strict = torch.tril(
                torch.ones(m, m, device="cuda", dtype=torch.float64), -1
            )
            col0 = st % BT
            Aqk_ref[0, r0:r0 + m, :, col0:col0 + m] = Aqk_blk.permute(0, 2, 1)
            maskA[0, r0:r0 + m, :, col0:col0 + m] = tri[:, None, :]
            Akk_ref[0, r0:r0 + m, :, 0:m] = (Akk_blk * strict[:, :, None]).permute(0, 2, 1)
            maskK[0, r0:r0 + m, :, 0:m] = tri[:, None, :]
    return Aqk_ref, Akk_ref, maskA, maskK


def _cos_sim(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def _min_bytes(cu_seqlens, n):
    """Minimal HBM traffic: read q,k,g,beta once + write the cells actually written."""
    bounds = cu_seqlens.tolist()
    P = 0  # total written (i,j) pairs per head = sum_subchunk m(m+1)/2
    for bos, eos in zip(bounds[:-1], bounds[1:]):
        L = eos - bos
        for st in range(0, L, BC):
            m = min(BC, L - st)
            P += m * (m + 1) // 2
    reads = n * H * K * (2 + 2 + 4) + n * H * 2          # q,k(bf16)+g(fp32)+beta
    writes = P * H * 2 + P * H * 4                        # Aqk(bf16)+Akk(fp32)
    return reads + writes


def _bench_burst(fn, warm=0.6, run=0.6):
    """back-to-back burst to hold DVFS; take best per-launch ms."""
    torch.cuda.synchronize()
    t0 = time.time(); n = 0
    while time.time() - t0 < warm:
        fn(); n += 1
        if n % 64 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    reps = max(64, n)
    best_ms = float("inf")
    t0 = time.time()
    while time.time() - t0 < run:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            fn()
        e.record(); e.synchronize()
        best_ms = min(best_ms, s.elapsed_time(e) / reps)
    return best_ms


def main():
    if not torch.cuda.is_available():
        print("no visible CUDA/MACA device: source env_local.sh + "
              "export CUDA_VISIBLE_DEVICES=<free card>", file=sys.stderr)
        sys.exit(1)

    dev = torch.cuda.current_device()
    print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')!r}  "
          f"device={torch.cuda.get_device_name(dev)}  count={torch.cuda.device_count()}")
    print(f"config: H={H} K={K} BT={BT} BC={BC} B={B} SCALE={SCALE:.6f} "
          f"dtype=bf16(q,k,Aqk)/fp32(g,Akk)")
    print(f"targets: read-only wall ~{READONLY_WALL:.0f} GB/s | nominal "
          f"{SINGLE_DIE_TARGET:.0f} (dual-die datasheet {DUAL_DIE_DATASHEET:.0f})")
    print("=" * 104)

    all_pass = True
    peak_gbps = 0.0
    peak_T = None

    for n in TOKEN_COUNTS:
        cu = _make_cu_seqlens(n)
        q, k, gk, beta = _make_inputs(n, cu)
        Aqk = torch.full((B, n, H, BT), float("nan"), device="cuda", dtype=torch.bfloat16)
        Akk = torch.full((B, n, H, BC), float("nan"), device="cuda", dtype=torch.float32)

        _launch(q, k, gk, beta, Aqk, Akk, cu)   # compile + autotune + result
        torch.cuda.synchronize()

        Aqk_ref, Akk_ref, maskA, maskK = _reference(q, k, gk, beta, cu, n)
        ka = Aqk[maskA].double(); ra = Aqk_ref[maskA]
        kk = Akk[maskK].double(); rk = Akk_ref[maskK]
        cos = _cos_sim(torch.cat([ka, kk]), torch.cat([ra, rk]))
        has_nan = bool(torch.isnan(ka).any() or torch.isnan(kk).any())
        # exact written-set check: no clobber of the unwritten off-diagonal cells
        n_written_ok = (int((~torch.isnan(Aqk)).sum()) == int(maskA.sum())
                        and int((~torch.isnan(Akk)).sum()) == int(maskK.sum()))
        ok = (cos >= COS_THRESHOLD) and not has_nan and n_written_ok
        all_pass = all_pass and ok

        ms = _bench_burst(lambda: _launch(q, k, gk, beta, Aqk, Akk, cu))
        gbps = _min_bytes(cu, n) / (ms * 1e6)
        if gbps > peak_gbps:
            peak_gbps, peak_T = gbps, n

        flag = "OK  " if ok else "BAD "
        extra = "" if n_written_ok else " [WRITE-SET MISMATCH]"
        print(f"[kda_intra] T={n:<6}  cos_sim={cos:.6f} {flag}  "
              f"{ms * 1e3:8.2f} us  {gbps:8.1f} GB/s  eff={100*gbps/READONLY_WALL:5.1f}%{extra}")

    print("=" * 104)
    print(f"Accuracy: {'ALL PASS' if all_pass else 'FAIL'} "
          f"(threshold cos_sim >= {COS_THRESHOLD})")
    reached = "reached" if peak_gbps >= 0.85 * READONLY_WALL else "NOT reached"
    print(f"Peak effective bandwidth: {peak_gbps:.1f} GB/s @ T={peak_T}  "
          f"({100*peak_gbps/READONLY_WALL:.1f}% of read-only wall {READONLY_WALL:.0f}; "
          f"85% gate -> {reached})")
    return all_pass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
