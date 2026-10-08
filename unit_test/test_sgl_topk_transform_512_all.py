# SPDX-License-Identifier: Apache-2.0
"""Unit test + bandwidth benchmark for topk_transform_v1 (C500-optimized port
of the JIT topk_transform_kernel in op/sglang/jit_kernels/topk_v1.cuh).

Test data and benchmark methodology aligned with unit_test/test_topk.py
(same bs/seq_len grid, same bytes_accessed formula, same warmup/iters).

================================================================================
Expanded accuracy suite (2481 pytest cases; 2439 new beyond the original 42).
================================================================================

Goal: prove topk_transform_v1 precision in EVERY situation reachable through
its public contract, including the data distributions of a real deployed LLM
(real serving scores are NOT Gaussian: heavy tails, dense ties from low-precision
compute, +/-inf attention masks, NaNs, denormals, 1 ULP ladders ...). The
distribution zoo is modeled after unit_test/test_sglang_topk_transformer_prefill_kernel.py
(clustered / separated / extreme builders) plus the adversarial + NaN suites
validated during the kernel's optimization.

Kernel contract under test (from op/sglang/jit_kernels/topk_v1.cu):
  * TopK is fixed at 512; k = 512 always.
  * Launch-level dispatch by max(seq_lens): <=16384 hist4096; <=2^20 sampled
    single-pass (512-thread wide launch when B > 64, else 1024 threads);
    > 2^20 exact two-pass radix. During CUDA-graph capture the allocated row
    width (scores.size(1)) plays the role of max(seq_lens).
  * Per row: seq_len <= 512 takes the naive transform (raw[i] = i for
    i < seq_len else -1; page[i] = page_to_indices(i) else -1).
  * page_size is any power of two; page_to_indices(i) =
    (page_table[i >> page_bits] << page_bits) | (i & (page_size - 1)).
  * seq_lens is a per-row device tensor -> ragged batches are in-contract and
    mixed-length launches (some rows naive, some hist-sized inside a sampled
    launch, ...) must stay exact.

Comparison contract (strictly stronger than the original tie-tolerant set
diff, which a duplicated-index bug could slip through):
  1. long rows (seq_len > 512): every raw index in [0, seq_len) and pairwise
     DISTINCT;
  2. the selected VALUE MULTISET equals torch.topk's exactly: NaN counts match,
     NaN payload bits match (<=512 NaNs are all selected by both sides, so the
     multiset is exactly defined; >512-NaN rows are generated with a single
     payload so any 512 of them are bitwise identical), finite parts compare
     with IEEE == (so +0.0/-0.0 are interchangeable, matching torch's own
     comparison semantics);
  3. whenever the top-512 cut is tie-free (512th largest value > 513rd), the
     sorted raw index set must equal torch.topk's sorted index set EXACTLY;
  4. page transform: torch.equal(page_to_indices(raw), page_indices) for every
     row, and the naive (-1) padding for short rows;
  5. rows are allocated wider than seq_len and the padding is filled with a
     1e30 canary: any read past seq_len either emits an out-of-range index or
     a canary value and fails the checks above.

Runtime discipline: accuracy cases launch the kernel once (no 100-iteration
benchmark); the perf battery below keeps the original timing loops.
"""

import math
import random
import zlib

import pytest
import torch

import mcoplib.sgl_kernel  # noqa: F401  (registers torch.ops.sgl_kernel)

TOPK = 512
MAX_SEQ_LEN = 66551
TABLE_LEN = 299062
PAGE_SIZE = 16
MAX_PERMIT_ERROR = 0
# torch.topk on this platform SELECTS with NaN-greatest semantics (every NaN
# makes the top-k, any payload) but its sorted OUTPUT places +NaN first and
# -NaN last, so a k=513-then-[:512] slice silently drops the trailing -NaN.
# The selection ground truth therefore comes from a direct k=512 call; the
# k=513 call below is only the cut-straddle probe for the index-set check.
K_REF = TOPK + 1
CANARY = 1e30  # fill for allocated columns >= seq_len (out-of-range read detector)

POS_NAN_BITS = 2143289344  # 0x7FC00000 (positive-payload quiet NaN)
NEG_NAN_BITS = -4194304    # 0xFFC00000 as int32 (negative-payload quiet NaN)

# Byte budget for one score tensor (mirrors the prefill kernel test).
SCORE_BUDGET_BYTES = 24 * (1024 ** 3)


def _ref_torch_impl(score: torch.Tensor, seq_len: int, topk: int) -> torch.Tensor:
    assert score.dim() == 2
    return torch.topk(score[:, :seq_len], topk, dim=-1, sorted=False).indices.to(torch.int32)


def _ref_page_to_indices(topk_indices: torch.Tensor, page_table: torch.Tensor, page_size: int) -> torch.Tensor:
    """page_to_indices(i) = (page_table[i >> page_bits] << page_bits) | (i & mask)"""
    page_bits = page_size.bit_length() - 1
    mask = page_size - 1
    page_idx = topk_indices >> page_bits
    rows = torch.arange(page_table.size(0), device=page_table.device).unsqueeze(-1)
    pages = page_table[rows, page_idx]
    return (pages << page_bits) | (topk_indices & mask)


def assert_equal(
    score: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_our: torch.Tensor,
    bs: int,
    k: int,
    seq_len: int,
):
    """Tie-tolerant comparison matching unit_test/test_topk.py::assert_equal."""
    indices_our_cpu = indices_our.cpu().tolist()
    indices_ref_cpu = indices_ref.cpu().tolist()
    for i in range(bs):
        ref_set_i = set(indices_ref_cpu[i])
        our_set_i = set(indices_our_cpu[i])
        more = our_set_i - ref_set_i
        less = ref_set_i - our_set_i
        if len(more) > MAX_PERMIT_ERROR or len(less) > MAX_PERMIT_ERROR:
            more_values = sorted(score[i, idx].item() for idx in more)
            less_values = sorted(score[i, idx].item() for idx in less)
            assert more_values == less_values, (
                f"{bs=}, {k=}, {seq_len=}, {i=}, {more=}, {less=} failed, "
                f"with {more_values=}, {less_values=}"
            )


def run_cuda_benchmark(kernel_name, run_func, bytes_accessed, warmup=5, iters=100):
    for _ in range(warmup):
        run_func()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        run_func()
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / iters
    avg_time_sec = avg_time_ms / 1000.0
    bandwidth_gb_s = (bytes_accessed / avg_time_sec) / 1e9 if avg_time_sec > 0 else 0.0
    print(f"  [PERF] {kernel_name:<30} | Avg Time: {avg_time_ms:8.4f} ms | Bandwidth: {bandwidth_gb_s:7.2f} GB/s")
    return bandwidth_gb_s


# ============================================================================
# Exact-comparison contract for the expanded accuracy suite (all vectorized;
# one kernel launch + one torch.topk per case).
# ============================================================================

def _nan_scalar(bits: int) -> torch.Tensor:
    """(1,) float32 tensor carrying an explicit NaN payload bit pattern."""
    return torch.tensor([bits], dtype=torch.int32, device="cuda").view(torch.float32)


def _nan_full(bits: int, bs: int, W: int) -> torch.Tensor:
    """(bs, W) tensor fully carrying one explicit NaN payload."""
    return (torch.full((bs * W,), bits, dtype=torch.int32, device="cuda")
            .view(torch.float32).view(bs, W))


def _canon_row(v: torch.Tensor):
    """Canonical (nan_bits, finite_values) form of one row's selected values.

    NaNs first sorted by payload bits (torch.topk ranks every NaN greatest, so
    all NaNs of a row with <= TOPK NaNs are selected by BOTH sides and the
    payload-bit multiset is exactly defined); finites sorted descending and
    compared with IEEE == (+0.0/-0.0 interchangeable, matching torch).
    """
    m = torch.isnan(v)
    nan_bits = v[m].view(torch.int32)
    nan_sorted = torch.sort(nan_bits, descending=True).values if nan_bits.numel() else nan_bits
    fin_sorted = torch.sort(v[~m], descending=True).values
    return nan_sorted, fin_sorted


def _check_topk_exact(tag, score, seq_lens, raw, page, page_table, page_size):
    """Full exactness check of one kernel run. See the module docstring."""
    device = score.device
    B, W = score.shape
    lens = seq_lens.long()
    ar = torch.arange(TOPK, device=device)
    pb = page_size.bit_length() - 1
    pmask = page_size - 1

    # ---- rows with seq_len <= TOPK: naive transform (identity + -1 pad) ----
    naive = lens <= TOPK
    if bool(naive.any()):
        nl = lens[naive].unsqueeze(1)
        arb = ar.unsqueeze(0)
        exp_raw = torch.where(arb < nl, arb, torch.full_like(arb, -1))
        assert torch.equal(raw[naive].long(), exp_raw), f"{tag}: naive raw_indices != identity/-1 pad"
        pt = page_table[naive]
        # torch.gather does NOT broadcast the index: with pcol shaped [1, 512]
        # it would silently gather from pt row 0 only (output [1, 512]) and
        # compare every naive row against row 0's page table. Advanced
        # indexing pt[:, pcol] gives the correct [n_naive, 512] per-row
        # gather. The clamp only keeps in-bounds for j >= len (masked to -1
        # by the where below); it never binds for j < len <= W.
        pcol = (ar >> pb).clamp(max=pt.size(1) - 1)
        xf = (pt[:, pcol].long() << pb) | (ar & pmask)
        exp_page = torch.where(arb < nl, xf, torch.full_like(xf, -1))
        assert torch.equal(page[naive].long(), exp_page), f"{tag}: naive page_indices mismatch"

    # ---- rows with seq_len > TOPK: exact top-512 contract ----
    longm = ~naive
    if not bool(longm.any()):
        return
    sL = score[longm]
    rawL = raw[longm]
    lensL = lens[longm].unsqueeze(1)
    nL = sL.size(0)

    # 1) index range + pairwise distinctness
    in_range = (rawL >= 0) & (rawL < lensL)
    if not bool(in_range.all()):
        gmap = longm.nonzero().flatten()
        bad_rows = (~in_range).any(dim=1).nonzero().flatten()
        detail = []
        for r in bad_rows[:3].tolist():
            bad = (~in_range[r]).nonzero().flatten()
            detail.append(
                f"row{int(gmap[r])}/len={int(lens[gmap[r]])} "
                f"bad@{bad[:8].tolist()}={rawL[r][bad[:8]].tolist()}")
        raise AssertionError(
            f"{tag}: {int((~in_range).sum())} raw indices out of [0, seq_len); "
            f"min={int(rawL.min())} max={int(rawL.max())} "
            f"lens_max={int(lens[longm].max())}; " + "; ".join(detail)
        )
    sraw = rawL.sort(dim=-1).values
    dup = sraw[:, 1:] == sraw[:, :-1]
    if bool(dup.any()):
        gmap = longm.nonzero().flatten()
        drows = dup.any(dim=1).nonzero().flatten()
        detail = []
        for r in drows[:3].tolist():
            vals, cnts = torch.unique(sraw[r], return_counts=True)
            detail.append(
                f"row{int(gmap[r])}/len={int(lens[gmap[r]])} "
                f"distinct={int(vals.numel())} topcnt={int(cnts.max())} "
                f"topval={int(vals[cnts.argmax()])}")
        raise AssertionError(
            f"{tag}: duplicate raw indices ({int(dup.sum())}); " + "; ".join(detail)
        )

    # 2) reference over the padded row: canary -> -inf so padding never wins.
    # ref_k (k=512) is the selection ground truth; ref_x (k=513) exists only
    # to detect a tie straddling the 512/513 cut (see K_REF note above: the
    # [:512] slice of a k=513 call is NOT the k=512 answer when -NaNs are
    # present, because torch sorts -NaNs to the end).
    col = torch.arange(W, device=device).unsqueeze(0)
    ref_in = torch.where(col < lensL, sL, torch.full_like(sL, float("-inf")))
    ref_k = torch.topk(ref_in, TOPK, dim=1, sorted=True)  # long rows: len >= 513
    ref_x = torch.topk(ref_in, K_REF, dim=1, sorted=True)
    ref_vals = ref_k.values
    ref_idx = ref_k.indices

    my_vals = sL.gather(1, rawL.long())
    a = my_vals.sort(dim=-1, descending=True).values
    b = ref_vals.sort(dim=-1, descending=True).values
    na, nb = torch.isnan(a), torch.isnan(b)
    nan_rows = (na.sum(-1) > 0) | (nb.sum(-1) > 0)

    cnt_bad = (na.sum(-1) != nb.sum(-1)).nonzero().flatten().tolist()
    assert not cnt_bad, f"{tag}: NaN count mismatch rows={cnt_bad[:8]}"

    plain = ~nan_rows
    if bool(plain.any()):
        bad = (~(a[plain] == b[plain])).any(-1).nonzero().flatten().tolist()
        if bad:
            r0 = bad[0]
            row_id = int(longm.nonzero().flatten()[plain.nonzero().flatten()[r0]])
            aa = a[plain][r0][:5].cpu().tolist()
            bb = b[plain][r0][:5].cpu().tolist()
            raise AssertionError(
                f"{tag}: value multiset mismatch in {len(bad)} rows; row {row_id} "
                f"top5 ours={aa} ref={bb}"
            )
    for r in nan_rows.nonzero().flatten().tolist():
        cn, cf = _canon_row(my_vals[r])
        rn, rf = _canon_row(ref_vals[r])
        assert cn.numel() == rn.numel() and cf.numel() == rf.numel() and \
            torch.equal(cn, rn) and torch.equal(cf, rf), (
            f"{tag}: NaN-aware value multiset mismatch row={r} "
            f"nan={int(cn.numel())}/{int(rn.numel())} "
            f"fin_top3={cf[:3].cpu().tolist()} vs {rf[:3].cpu().tolist()}"
        )

    # 3) exact index set whenever the top-512 cut is tie-free
    v511 = ref_x.values[:, TOPK - 1]
    v512 = ref_x.values[:, TOPK]
    uniq = v511 > v512  # NaN comparisons and equal-value ties both yield False
    if bool(uniq.any()):
        u = uniq.nonzero().flatten()
        mismatch = ~(sraw[u].long() == ref_idx[u].sort(dim=-1).values).all(-1)
        assert not bool(mismatch.any()), (
            f"{tag}: index set != torch.topk on tie-free rows "
            f"({int(mismatch.sum())} rows)"
        )

    # 4) page transform
    ptL = page_table[longm]
    phys = ptL.gather(1, (rawL >> pb).long())
    xf = (phys << pb) | (rawL & pmask)
    assert torch.equal(xf, page[longm]), f"{tag}: page transform mismatch"


# ============================================================================
# Distribution zoo: real-LLM-style and adversarial score patterns.
# Every builder writes the full (bs, W) rectangle; columns >= seq_len are then
# filled with the 1e30 canary, so ragged rows are handled uniformly.
# ============================================================================

_DIST_ALL = [
    "gauss", "gauss_x1000", "uniform", "uniform_pos", "cauchy", "student_t3",
    "lognormal", "exponential", "clustered511", "clustered512", "clustered513",
    "clustered600", "tie_across_chunk", "sample_low_head", "sample_high_head",
    "all_equal", "binary01", "quant15", "int8_levels", "bf16_round",
    "fp16_round", "separated_high", "extreme_mix", "inf_mix", "all_pos_inf",
    "all_neg_inf", "window_inf", "sparse_spikes", "negzero_poszero",
    "ulp_ladder", "big_small", "denormal_ladder", "nan_sparse_pos",
    "nan_sparse_neg", "nan600",
]


def _make_scores(dist, bs, W, lens, seed):
    """Build a (bs, W) float32 CUDA score tensor for `dist`.

    `lens` is the (bs,) int64 CUDA tensor of row lengths. Columns >= len are
    canaried with 1e30. Rows with > TOPK NaNs carry a single NaN payload by
    construction (torch's pick among distinct payloads is unspecified there).
    """
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    col = torch.arange(W, device=dev)
    inrow = col.unsqueeze(0) < lens.unsqueeze(1)

    def rnd(shape=None):
        shape = shape or (bs, W)
        return torch.randn(shape, generator=g, device=dev)

    def runif(shape=None):
        shape = shape or (bs, W)
        return torch.rand(shape, generator=g, device=dev)

    if dist == "gauss":
        s = rnd()
    elif dist == "gauss_x1000":
        s = rnd() * 1000.0
    elif dist == "uniform":
        s = runif() * 20.0 - 10.0
    elif dist == "uniform_pos":
        s = runif() * 100.0
    elif dist == "cauchy":
        # Heavy-tailed: real attention logits after QK with outlier channels.
        s = torch.tan((runif() - 0.5) * math.pi)
    elif dist == "student_t3":
        c1, c2, c3 = rnd(), rnd(), rnd()
        s = rnd() / ((c1 * c1 + c2 * c2 + c3 * c3) / 3.0).sqrt().clamp_min(1e-30)
    elif dist == "lognormal":
        s = torch.exp(rnd() * 3.0)
    elif dist == "exponential":
        s = -runif().clamp_min(1e-20).log()
    elif dist.startswith("clustered"):
        # Prefill-kernel style: n_equal identical highs in one radix bin; the
        # top-512 cut lands before / exactly at / inside the equal group.
        n = min(int(dist[len("clustered"):]), W)
        s = torch.full((bs, W), -100.0, device=dev)
        s[:, :n] = 100.0
    elif dist == "tie_across_chunk":
        # Equal highs straddling the sampled path's ~8192-element sample chunk
        # boundary: the sample histogram sees only part of the tie group.
        s = rnd()
        m1 = (runif() < 0.05) & (col < 8192).unsqueeze(0)
        m2 = (runif() < 0.05) & (col >= 8192).unsqueeze(0)
        s = torch.where(m1 | m2, torch.full_like(s, 50.0), s)
    elif dist == "sample_low_head":
        s = rnd()
        s[:, :8192] -= 20.0  # sample chunk all-tiny: threshold estimate too low
    elif dist == "sample_high_head":
        s = rnd()
        s[:, :8192] += 20.0  # sample chunk all-huge: staged count < 512
    elif dist == "all_equal":
        s = torch.ones(bs, W, device=dev)
    elif dist == "binary01":
        s = (runif() < 0.5).float()
    elif dist == "quant15":
        # ~15 levels -> ~4.4k elements/level at 66551: tie bin overflows the
        # 2048-entry staging buffer -> exact rescan round.
        s = torch.round(rnd() * 7 + 8)
    elif dist == "int8_levels":
        s = torch.round(runif() * 255 - 128)
    elif dist == "bf16_round":
        # Real serving computes scores in bf16: dense exact ties by rounding.
        s = rnd().to(torch.bfloat16).float()
    elif dist == "fp16_round":
        s = (rnd() * 100.0).to(torch.float16).float()
    elif dist == "separated_high":
        # Strictly increasing distinct highs across radix bins: tie-free top-k.
        n = min(2500, W)
        s = torch.full((bs, W), -1.0, device=dev)
        s[:, :n] = (torch.arange(n, device=dev, dtype=torch.float32) + 1.0) * 2.0
    elif dist == "extreme_mix":
        # Prefill-kernel style non-Gaussian extremes: huge finite, denormal, 0.
        s = rnd()
        s[:, 0::7] = 1e30
        s[:, 1::7] = -1e30
        s[:, 2::11] = 1.4e-45
        s[:, 3::13] = 0.0
    elif dist == "inf_mix":
        s = rnd() - 3.0
        m1 = runif() < 0.005
        m2 = runif() < 0.005
        s = torch.where(m1, torch.full_like(s, float("inf")),
                        torch.where(m2, torch.full_like(s, float("-inf")), s))
    elif dist == "all_pos_inf":
        s = torch.full((bs, W), float("inf"), device=dev)
    elif dist == "all_neg_inf":
        s = torch.full((bs, W), float("-inf"), device=dev)
    elif dist == "window_inf":
        # Sliding-window attention: only the last `window` positions of each
        # row are finite; everything before is -inf masked.
        window = 4096 if W > 8192 else max(1, W // 2)
        fin = inrow & (col.unsqueeze(0) >= (lens.unsqueeze(1) - window))
        s = torch.where(fin, rnd(), torch.full((bs, W), float("-inf"), device=dev))
    elif dist == "sparse_spikes":
        # 95% exact zeros + 5% spikes: activation-sparsity style topk.
        s = torch.where(runif() < 0.05, rnd(), torch.zeros(bs, W, device=dev))
    elif dist == "negzero_poszero":
        s = rnd() - 5.0
        m1 = runif() < 0.02
        m2 = runif() < 0.02
        s = torch.where(m1, torch.zeros_like(s),
                        torch.where(m2, torch.full_like(s, -0.0), s))
    elif dist == "ulp_ladder":
        # 1.0 .. 2.0 in ~equal ULP steps: all-distinct, densely packed floats.
        step = max(1, 0x00800000 // max(W, 1))
        bits = (0x3F800000 + col * step).to(torch.int32)
        s = bits.unsqueeze(0).expand(bs, W).view(torch.float32).contiguous()
    elif dist == "big_small":
        pattern = torch.tensor(
            [1e38, -1e38, 1e-38, -1e-38, 0.0, -0.0, 3.0e38, -3.0e38],
            device=dev, dtype=torch.float32)
        s = pattern[col % 8].unsqueeze(0).expand(bs, W).contiguous()
    elif dist == "denormal_ladder":
        bits = ((col % 1000) + 1).to(torch.int32)  # denormals 1.4e-45..1.4e-42
        s = bits.unsqueeze(0).expand(bs, W).view(torch.float32).contiguous()
    elif dist == "nan_sparse_pos":
        s = rnd()
        m = runif() < (3.0 / max(W, 1))
        s = torch.where(m, _nan_scalar(POS_NAN_BITS), s)
    elif dist == "nan_sparse_neg":
        s = rnd()
        m = runif() < (3.0 / max(W, 1))
        s = torch.where(m, _nan_scalar(NEG_NAN_BITS), s)
    elif dist == "nan600":
        # ~600 single-payload NaNs (> TOPK): any 512 of them are bitwise
        # identical, so the multiset stays exactly defined.
        s = rnd()
        m = runif() < min(1.0, 600.0 / max(W, 1))
        s = torch.where(m, _nan_scalar(POS_NAN_BITS), s)
    else:
        raise ValueError(f"unknown dist {dist}")

    return torch.where(inrow, s, torch.full_like(s, CANARY))


def _make_page_table(kind, bs, T, seed):
    dev = "cuda"
    if kind == "identity":
        return torch.arange(T, dtype=torch.int32, device=dev).unsqueeze(0).expand(bs, -1).contiguous()
    if kind == "randperm":
        # Real serving: each row's block table is a distinct physical page set.
        g = torch.Generator(device=dev).manual_seed(seed)
        return torch.argsort(torch.rand(bs, T, generator=g, device=dev), dim=1).to(torch.int32)
    if kind == "randint":
        g = torch.Generator(device=dev).manual_seed(seed)
        return torch.randint(0, 1 << 20, (bs, T), generator=g, device=dev, dtype=torch.int32)
    if kind == "reversed":
        return torch.arange(T - 1, -1, -1, dtype=torch.int32, device=dev).unsqueeze(0).expand(bs, -1).contiguous()
    raise ValueError(f"unknown page table kind {kind}")


def _run_accuracy_case(tag, dist, bs, lens_list, page_size=16, page_kind="identity",
                       score=None, seed_extra=""):
    """Build inputs for (dist, bs, lens), launch once, run the exact contract."""
    dev = "cuda"
    W = max(1, max(lens_list) if lens_list else 0)
    if bs * W * 4 > SCORE_BUDGET_BYTES:
        pytest.skip("score too large")
    lens = torch.tensor(lens_list, dtype=torch.int32, device=dev)
    seed = zlib.crc32(f"{tag}|{dist}|{bs}|{min(lens_list)}-{max(lens_list)}|{page_size}|{page_kind}|{seed_extra}".encode()) % (2**31 - 1)
    if score is None:
        score = _make_scores(dist, bs, W, lens.long(), seed)
    T = (W + page_size - 1) // page_size
    page_table = _make_page_table(page_kind, bs, T, seed)
    page_indices = torch.empty(bs, TOPK, dtype=torch.int32, device=dev)
    raw_indices = torch.empty(bs, TOPK, dtype=torch.int32, device=dev)
    torch.ops.sgl_kernel.topk_transform_v1.default(
        score, lens, page_indices, page_table, page_size, raw_indices)
    torch.cuda.synchronize()
    _check_topk_exact(tag, score, lens, raw_indices, page_indices, page_table, page_size)


# ============================================================================
# Original perf battery + CUDA-graph smoke tests (42 cases, unchanged).
# ============================================================================

@pytest.mark.parametrize("bs", [1, 132, 256, 4096, 1662])
@pytest.mark.parametrize("seq_len", [2048, 4096, 3096, 3500, 8024, 8096, 16384, 66551])
@torch.inference_mode()
def test_topk_transform_v1(bs: int, seq_len: int) -> None:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")

    page_table = torch.arange(0, TABLE_LEN, dtype=torch.int32, device="cuda")
    page_table = page_table.unsqueeze(0).expand(bs, -1).contiguous()

    # reference
    ref_topk = _ref_torch_impl(score, seq_len, TOPK)
    ref_page = _ref_page_to_indices(ref_topk, page_table, PAGE_SIZE)

    page_indices = score.new_empty((bs, TOPK), dtype=torch.int32)
    raw_indices = score.new_empty((bs, TOPK), dtype=torch.int32)

    def launch():
        torch.ops.sgl_kernel.topk_transform_v1.default(
            score, seq_lens, page_indices, page_table, PAGE_SIZE, raw_indices
        )

    # correctness: compare raw topk indices (tie-tolerant) and page transform
    launch()
    raw_ref_sorted = torch.sort(ref_topk, dim=-1).values
    raw_our_sorted = torch.sort(raw_indices, dim=-1).values
    assert_equal(score, raw_ref_sorted, raw_our_sorted, bs, TOPK, seq_len)
    # page transform must equal page_to_indices(raw_indices)
    recomputed_page = _ref_page_to_indices(raw_indices, page_table, PAGE_SIZE)
    assert torch.equal(recomputed_page, page_indices), f"page transform mismatch bs={bs} seq={seq_len}"

    # bytes_accessed aligned with unit_test/test_topk.py::test_topk_transform_kernel:
    # Read Score + Read SrcPageTable + Write DstPageTable = bs * (2 * seq_len + k) * 4
    bytes_accessed = bs * (2 * seq_len + TOPK) * 4
    print(f"\n[Case: BS={bs}, SeqLen={seq_len}, PageSize={PAGE_SIZE}, K={TOPK}]")
    bw = run_cuda_benchmark("topk_transform_v1", launch, bytes_accessed)

    # Target: average bandwidth >= 800 GB/s for seq_len >= 4096
    # if seq_len >= 4096:
    #     assert bw >= 800.0, f"bandwidth {bw:.1f} GB/s below target 800 GB/s (bs={bs}, seq={seq_len})"


@pytest.mark.parametrize("seq_len", [4096, 20000], ids=["histogram", "radix"])
@torch.inference_mode()
def test_topk_transform_v1_cuda_graph_replay(seq_len: int) -> None:
    """Captured execution must replay correctly with changed score contents."""
    bs = 2
    score = torch.randn(bs, seq_len, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    page_table = torch.arange(TABLE_LEN, dtype=torch.int32, device="cuda")
    page_table = page_table.unsqueeze(0).expand(bs, -1).contiguous()
    page_indices = torch.empty((bs, TOPK), dtype=torch.int32, device="cuda")
    raw_indices = torch.empty_like(page_indices)

    def launch() -> None:
        torch.ops.sgl_kernel.topk_transform_v1.default(
            score, seq_lens, page_indices, page_table, PAGE_SIZE, raw_indices
        )

    # Initialize one-time function attributes before capture.
    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            launch()
    except Exception as exc:
        pytest.fail(
            f"topk_transform_v1 graph capture failed for {seq_len=}: "
            f"{type(exc).__name__}: {exc}"
        )

    for seed in (101, 202):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        score.copy_(torch.randn(score.shape, generator=generator, device="cuda"))
        graph.replay()
        # Keep raw indices in their original (unsorted) order for the page
        # mapping check. Only sort copies for the unordered Top-K comparison.
        graph_raw = raw_indices.clone()
        graph_raw_sorted = torch.sort(graph_raw, dim=-1).values
        graph_page = page_indices.clone()

        launch()
        torch.cuda.synchronize()
        eager_raw = torch.sort(raw_indices, dim=-1).values
        assert_equal(score, eager_raw, graph_raw_sorted, bs, TOPK, seq_len)
        expected_page = _ref_page_to_indices(graph_raw, page_table, PAGE_SIZE)
        assert torch.equal(expected_page, graph_page), (
            f"graph page transform mismatch for {seq_len=}, {seed=}"
        )


# ============================================================================
# (A) Distribution zoo x shape grid: 35 distributions x 28 (bs, seq_len).
# Covers all three dispatch modes, the narrow/wide launch boundary (B=64/65),
# odd (non-float4-aligned) widths, and the DSV4 max decode length.
# ============================================================================

_MAIN_SHAPES = [
    (1, 513), (2, 513), (4, 1024), (1, 2048), (4, 2048), (132, 2048),
    (2, 4097), (4, 4097), (132, 4097), (1, 8199), (4, 8199), (132, 8199),
    (4, 16384), (132, 16384), (256, 16384),
    (2, 16385), (4, 16385), (132, 16385),
    (2, 20000), (4, 20000), (132, 20000), (256, 20000),
    (2, 66551), (4, 66551), (64, 66551), (65, 66551), (132, 66551), (256, 66551),
]


@pytest.mark.parametrize("shape", _MAIN_SHAPES, ids=[f"bs{b}-L{L}" for b, L in _MAIN_SHAPES])
@pytest.mark.parametrize("dist", _DIST_ALL)
@torch.inference_mode()
def test_topk_accuracy_distributions(shape, dist) -> None:
    bs, seq_len = shape
    _run_accuracy_case("main", dist, bs, [seq_len] * bs)


# ============================================================================
# (B) Ragged batches: per-row seq_lens inside one launch (real serving has
# variable request lengths). Mode is chosen by the max row length, so mixes
# exercise naive rows next to hist-sized rows inside a sampled launch etc.
# ============================================================================

def _ragged_lens(layout, bs, seed):
    rng = random.Random(zlib.crc32(f"ragged|{layout}|{bs}|{seed}".encode()))
    def cycle(vals):
        return [vals[i % len(vals)] for i in range(bs)]
    if layout == "naive_only":
        return cycle([0, 1, 100, 511, 512])
    if layout == "naive_span":
        return [min(512, (512 * i) // max(bs - 1, 1)) for i in range(bs)]
    if layout == "hist_small":
        return cycle([513, 600, 1024])
    if layout == "hist_mix":
        return cycle([513, 2048, 8199, 16384])
    if layout == "hist_boundary":
        return cycle([16383, 16384, 16385])  # max 16385 -> sampled launch
    if layout == "sampled_mix":
        return cycle([16385, 20000, 40000])
    if layout == "naive_hist":
        return cycle([7, 513, 16384])
    if layout == "naive_hist_sampled":
        return cycle([0, 5, 512, 513, 16384, 16385, 66551])
    if layout == "full_span":
        return [rng.randint(0, 66551) for _ in range(bs)]
    if layout == "near_512":
        return cycle([511, 512, 513, 514])
    if layout == "near_16384":
        return cycle([16383, 16384, 16385, 16386])
    if layout == "near_66551":
        return cycle([66550, 66551])
    raise ValueError(layout)


_RAGGED_LAYOUTS = [
    "naive_only", "naive_span", "hist_small", "hist_mix", "hist_boundary",
    "sampled_mix", "naive_hist", "naive_hist_sampled", "full_span",
    "near_512", "near_16384", "near_66551",
]
_RAGGED_DISTS = [
    "gauss", "quant15", "clustered600", "extreme_mix", "bf16_round",
    "window_inf", "sparse_spikes", "negzero_poszero", "inf_mix",
    "nan_sparse_pos", "all_equal", "separated_high",
]


@pytest.mark.parametrize("bs", [8, 64, 100])  # 64: narrow/wide launch boundary
@pytest.mark.parametrize("dist", _RAGGED_DISTS)
@pytest.mark.parametrize("layout", _RAGGED_LAYOUTS)
@torch.inference_mode()
def test_topk_ragged_batches(layout, dist, bs) -> None:
    lens = _ragged_lens(layout, bs, seed=1234)
    _run_accuracy_case(f"ragged-{layout}", dist, bs, lens, seed_extra="r")


# ============================================================================
# (C) Boundary seq_lens: every dispatch-mode boundary and the small-length
# end, including the exact two-pass radix mode (> 2^20) that no previous test
# reached. 30 lengths x 4 distributions x 2 batch sizes.
# ============================================================================

_BOUND_LENS = [
    513, 514, 515, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097,
    8191, 8192, 8193, 12000, 16383, 16384, 16385, 16386, 32768, 65535,
    65536, 66550, 66551, 69632, 70000, 1048576, 1048577, 1100000,
]


@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("dist", ["gauss", "quant15", "clustered600", "extreme_mix"])
@pytest.mark.parametrize("seq_len", _BOUND_LENS)
@torch.inference_mode()
def test_topk_boundary_lengths(seq_len, dist, bs) -> None:
    _run_accuracy_case("boundary", dist, bs, [seq_len] * bs)


# ============================================================================
# (D) page_size variants: the interface accepts any power of two; the page
# transform must stay exact for every page_bits from 0 to 9.
# ============================================================================

@pytest.mark.parametrize("dist", ["gauss", "clustered600"])
@pytest.mark.parametrize("seq_len", [2048, 20000, 66551])
@pytest.mark.parametrize("page_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
@torch.inference_mode()
def test_topk_page_size_variants(page_size, seq_len, dist) -> None:
    _run_accuracy_case("pagesize", dist, 4, [seq_len] * 4,
                       page_size=page_size, page_kind="randperm")


# ============================================================================
# (E) Page-table layouts: identity (debug), per-row random permutations (real
# block tables), duplicated random physical pages, reversed.
# ============================================================================

_PAGE_KIND_SHAPES = [(4, 2048), (132, 20000), (4, 66551)]


@pytest.mark.parametrize("dist", ["gauss", "quant15"])
@pytest.mark.parametrize("shape", _PAGE_KIND_SHAPES, ids=[f"bs{b}-L{L}" for b, L in _PAGE_KIND_SHAPES])
@pytest.mark.parametrize("kind", ["identity", "randperm", "randint", "reversed"])
@torch.inference_mode()
def test_topk_page_table_kinds(kind, shape, dist) -> None:
    bs, seq_len = shape
    _run_accuracy_case("pagetable", dist, bs, [seq_len] * bs, page_kind=kind)


# ============================================================================
# (F) NaN / inf zoo across all four dispatch paths (hist / sampled narrow /
# sampled wide / radix2pass). torch.topk ranks every NaN (any sign, any
# payload) as greatest; the kernel's key maps must agree. Rows with > TOPK
# NaNs use a single payload (torch's pick among distinct payloads is
# unspecified, so the multiset is only defined then).
# ============================================================================

_NAN_PATHS = {
    "hist": (2, 16384),
    "sampled_narrow": (2, 20000),
    "sampled_wide": (256, 66551),
    "radix2pass": (2, 1100000),
}
_NAN_PATTERNS = [
    "one_pos_nan", "one_neg_nan", "nan511", "nan512", "nan513",
    "nan600_inchunk", "all_nan_pos", "all_nan_neg", "nan_plus_ties",
    "inf_plus_nan", "all_pos_inf", "all_neg_inf", "nan_and_negzero",
    "nan_boundary_512_513",
]


def _make_nan_pattern_scores(pattern, bs, W, seed):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    col = torch.arange(W, device=dev)
    lens = torch.full((bs,), W, dtype=torch.int64, device=dev)
    inrow = col.unsqueeze(0) < lens.unsqueeze(1)
    pos = _nan_scalar(POS_NAN_BITS)
    neg = _nan_scalar(NEG_NAN_BITS)
    base = torch.randn(bs, W, generator=g, device=dev)

    def scattered(n, val):
        m = torch.rand(bs, W, generator=g, device=dev) < min(1.0, n / max(W, 1))
        return torch.where(m, val, base)  # (1,) NaN scalar broadcasts

    if pattern == "one_pos_nan":
        s = scattered(1, pos)
    elif pattern == "one_neg_nan":
        s = scattered(1, neg)
    elif pattern in ("nan511", "nan512", "nan513"):
        n = int(pattern[3:])
        s = base.clone()  # first-n columns NaN (single payload), rest gauss
        s[:, :n] = float("nan")
    elif pattern == "nan600_inchunk":
        m = (torch.rand(bs, W, generator=g, device=dev) < min(1.0, 600.0 / 8192)) & (col < 8192).unsqueeze(0)
        s = torch.where(m, pos, base)
    elif pattern == "all_nan_pos":
        s = _nan_full(POS_NAN_BITS, bs, W)
    elif pattern == "all_nan_neg":
        s = _nan_full(NEG_NAN_BITS, bs, W)
    elif pattern == "nan_plus_ties":
        s = base
        mt = torch.rand(bs, W, generator=g, device=dev) < min(1.0, 600.0 / max(W, 1))
        s = torch.where(mt, torch.full_like(s, 100.0), s)
        mn = torch.rand(bs, W, generator=g, device=dev) < (5.0 / max(W, 1))
        s = torch.where(mn, pos, s)
    elif pattern == "inf_plus_nan":
        s = base - 3.0
        mi = torch.rand(bs, W, generator=g, device=dev) < 0.005
        s = torch.where(mi, torch.full_like(s, float("inf")), s)
        mn = torch.rand(bs, W, generator=g, device=dev) < 0.005
        s = torch.where(mn, pos, s)
    elif pattern == "all_pos_inf":
        s = torch.full((bs, W), float("inf"), device=dev)
    elif pattern == "all_neg_inf":
        s = torch.full((bs, W), float("-inf"), device=dev)
    elif pattern == "nan_and_negzero":
        s = base - 5.0
        m1 = torch.rand(bs, W, generator=g, device=dev) < 0.02
        m2 = torch.rand(bs, W, generator=g, device=dev) < 0.02
        s = torch.where(m1, torch.zeros_like(s), torch.where(m2, torch.full_like(s, -0.0), s))
        mn = torch.rand(bs, W, generator=g, device=dev) < (3.0 / max(W, 1))
        s = torch.where(mn, pos, s)
    elif pattern == "nan_boundary_512_513":
        s = base.clone()
        if bs >= 1:
            s[0, :512] = float("nan")
        if bs >= 2:
            s[1, :513] = float("nan")
        if bs >= 3:
            s[2, :511] = float("nan")
    else:
        raise ValueError(pattern)

    # For the wide path only a few rows carry the pattern; the rest stay gauss
    # so one launch mixes pattern rows with ordinary rows.
    if bs > 4 and pattern not in ("nan_boundary_512_513", "all_nan_pos", "all_nan_neg"):
        keep = torch.zeros(bs, dtype=torch.bool, device=dev)
        keep[[0, 1, 100, 255]] = True
        s = torch.where(keep.unsqueeze(1), s, base)
    return torch.where(inrow, s, torch.full_like(s, CANARY))


@pytest.mark.parametrize("pattern", _NAN_PATTERNS)
@pytest.mark.parametrize("path", list(_NAN_PATHS))
@torch.inference_mode()
def test_topk_nan_inf_zoo(path, pattern) -> None:
    bs, W = _NAN_PATHS[path]
    if bs * W * 4 > SCORE_BUDGET_BYTES:
        pytest.skip("score too large")
    seed = zlib.crc32(f"nanzoo|{path}|{pattern}".encode()) % (2**31 - 1)
    score = _make_nan_pattern_scores(pattern, bs, W, seed)
    _run_accuracy_case(f"nanzoo-{path}", "custom", bs, [W] * bs, score=score)


# ============================================================================
# (G) Seeded fuzz: 600 random (dist, bs, lengths, page_size, page-table kind)
# combinations drawn from a deterministic per-seed RNG.
# ============================================================================

_FUZZ_BS = [1, 2, 3, 4, 5, 7, 8, 12, 16, 24, 32, 48, 64, 65, 96, 128]
_FUZZ_LEN = [513, 514, 600, 1000, 1024, 2048, 2049, 4097, 8199, 12000,
             16384, 16385, 20000, 33333, 50000, 66551]
_FUZZ_PS = [16] * 8 + [1, 2, 4, 8, 32, 64, 128, 256, 512]


@pytest.mark.parametrize("seed", range(600))
@torch.inference_mode()
def test_topk_seed_fuzz(seed) -> None:
    rng = random.Random(1000003 + seed)
    dist = rng.choice(_DIST_ALL)
    bs = rng.choice(_FUZZ_BS)
    if rng.random() < 0.20:
        hi = rng.choice([1024, 2048, 8199, 16384, 20000, 30000, 66551])
        lens = [rng.randint(0, hi) for _ in range(bs)]
    else:
        lens = [rng.choice(_FUZZ_LEN)] * bs
    page_size = rng.choice(_FUZZ_PS)
    kind = rng.choice(["identity", "randperm", "randint", "reversed"])
    _run_accuracy_case(f"fuzz{seed}", dist, bs, lens, page_size=page_size, page_kind=kind)


# ============================================================================
# (H) Row-stride / alignment: odd contiguous widths (rows start at non-float4
# addresses) and padded row strides (stride(0) > W, the over-read trap).
# ============================================================================

_STRIDE_SHAPES = [
    (1, 20003), (2, 20003), (5, 20003), (132, 20003),
    (1, 66551), (2, 66551), (5, 66551), (132, 66551),
    (1, 1048577), (2, 1048577), (5, 1048577), (132, 65537),
]


@pytest.mark.parametrize("shape", _STRIDE_SHAPES, ids=[f"bs{b}-L{L}" for b, L in _STRIDE_SHAPES])
@pytest.mark.parametrize("dist", ["gauss", "quant15"])
@torch.inference_mode()
def test_topk_row_alignment_odd_widths(shape, dist) -> None:
    bs, W = shape
    _run_accuracy_case("align", dist, bs, [W] * bs)


_PAD_SHAPES = [(4, 20000), (2, 66551), (7, 4097)]


@pytest.mark.parametrize("pad", [1, 3, 17])
@pytest.mark.parametrize("shape", _PAD_SHAPES, ids=[f"bs{b}-L{L}" for b, L in _PAD_SHAPES])
@pytest.mark.parametrize("dist", ["gauss", "clustered600"])
@torch.inference_mode()
def test_topk_row_stride_padded(shape, pad, dist) -> None:
    """score = base[:, :W] view with stride(0) = W + pad: the pad columns hold
    the canary and must never be read."""
    bs, W = shape
    dev = "cuda"
    lens_list = [W] * bs
    lens = torch.tensor(lens_list, dtype=torch.int32, device=dev)
    seed = zlib.crc32(f"stridepad|{dist}|{bs}|{W}|{pad}".encode()) % (2**31 - 1)
    base = _make_scores(dist, bs, W + pad, lens.long(), seed)
    score = base[:, :W]
    _run_accuracy_case(f"stridepad{pad}", dist, bs, lens_list, score=score)


# ============================================================================
# (I) CUDA-graph replay accuracy: captured launches replayed with fresh data
# must satisfy the exact contract (not just match a second eager launch), and
# mutating seq_lens between replays must be honored (static shapes, dynamic
# lengths -- the real serving pattern).
# ============================================================================

_GRAPH_SEQS = [2048, 16385, 20000, 66551, 1048577]


@pytest.mark.parametrize("dist", ["gauss", "quant15"])
@pytest.mark.parametrize("seq_len", _GRAPH_SEQS, ids=[f"L{L}" for L in _GRAPH_SEQS])
@torch.inference_mode()
def test_topk_cuda_graph_replay_accuracy(seq_len, dist) -> None:
    bs = 2
    dev = "cuda"
    lens = torch.full((bs,), seq_len, dtype=torch.int32, device=dev)
    seed = zlib.crc32(f"graph|{dist}|{seq_len}".encode()) % (2**31 - 1)
    score = _make_scores(dist, bs, seq_len, lens.long(), seed)
    T = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    page_table = _make_page_table("randperm", bs, T, seed)
    page_indices = torch.empty(bs, TOPK, dtype=torch.int32, device=dev)
    raw_indices = torch.empty(bs, TOPK, dtype=torch.int32, device=dev)

    def launch():
        torch.ops.sgl_kernel.topk_transform_v1.default(
            score, lens, page_indices, page_table, PAGE_SIZE, raw_indices)

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            launch()
    except Exception as exc:
        pytest.fail(f"graph capture failed {seq_len=} {dist=}: {exc}")

    for seed2 in (11, 22):
        score.copy_(_make_scores(dist, bs, seq_len, lens.long(), seed2 + 7919))
        graph.replay()
        torch.cuda.synchronize()
        graph_raw = raw_indices.clone()
        graph_page = page_indices.clone()
        launch()  # eager run on the same data
        torch.cuda.synchronize()
        # Output order is not deterministic (atomic-race scatter of the
        # above-threshold bulk), so the replay and the eager run are each
        # validated against the reference instead of compared bitwise.
        _check_topk_exact(f"graph-{dist}-L{seq_len}", score, lens,
                          graph_raw, graph_page, page_table, PAGE_SIZE)
        _check_topk_exact(f"grapheager-{dist}-L{seq_len}", score, lens,
                          raw_indices, page_indices, page_table, PAGE_SIZE)


_GRAPH_DYN_LENS = [[600, 600], [513, 20000], [16384, 20000], [0, 20000]]


@pytest.mark.parametrize("lens2", _GRAPH_DYN_LENS, ids=["L600", "mixed513", "mixed16384", "zero"])
@torch.inference_mode()
def test_topk_cuda_graph_replay_dynamic_lengths(lens2) -> None:
    """Capture at width 20000 (sampled mode), then rewrite seq_lens in the
    same device tensor and replay: rows shorter than the captured mode must
    stay exact (e.g. 600-length rows inside the sampled kernel)."""
    bs = 2
    W = 20000
    dev = "cuda"
    lens = torch.full((bs,), W, dtype=torch.int32, device=dev)
    seed = zlib.crc32(f"graphdyn|{lens2[0]}-{lens2[1]}".encode()) % (2**31 - 1)
    score = _make_scores("gauss", bs, W, lens.long(), seed)
    T = (W + PAGE_SIZE - 1) // PAGE_SIZE
    page_table = _make_page_table("randperm", bs, T, seed)
    page_indices = torch.empty(bs, TOPK, dtype=torch.int32, device=dev)
    raw_indices = torch.empty(bs, TOPK, dtype=torch.int32, device=dev)

    def launch():
        torch.ops.sgl_kernel.topk_transform_v1.default(
            score, lens, page_indices, page_table, PAGE_SIZE, raw_indices)

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    lens.copy_(torch.tensor(lens2, dtype=torch.int32, device=dev))
    score.copy_(_make_scores("gauss", bs, W, lens.long(), seed + 7919))
    graph.replay()
    torch.cuda.synchronize()
    graph_raw = raw_indices.clone()
    graph_page = page_indices.clone()
    launch()  # eager dispatch re-selects the mode from the new max length
    torch.cuda.synchronize()
    _check_topk_exact(f"graphdyn-{lens2}", score, lens,
                      graph_raw, graph_page, page_table, PAGE_SIZE)
    # Eager dispatch re-selects the mode from the new max length; both the
    # captured (sampled) kernel and the re-dispatched one must be exact.
    _check_topk_exact(f"graphdyneager-{lens2}", score, lens,
                      raw_indices, page_indices, page_table, PAGE_SIZE)


@pytest.mark.parametrize("dist", ["gauss", "quant15"])
@torch.inference_mode()
def test_topk_cuda_graph_replay_wide(dist) -> None:
    """Wide (512-thread) sampled launch under graph capture, bs=100 > 64."""
    bs = 100
    W = 66551
    dev = "cuda"
    lens = torch.full((bs,), W, dtype=torch.int32, device=dev)
    seed = zlib.crc32(f"graphwide|{dist}".encode()) % (2**31 - 1)
    score = _make_scores(dist, bs, W, lens.long(), seed)
    T = (W + PAGE_SIZE - 1) // PAGE_SIZE
    page_table = _make_page_table("randperm", bs, T, seed)
    page_indices = torch.empty(bs, TOPK, dtype=torch.int32, device=dev)
    raw_indices = torch.empty(bs, TOPK, dtype=torch.int32, device=dev)

    def launch():
        torch.ops.sgl_kernel.topk_transform_v1.default(
            score, lens, page_indices, page_table, PAGE_SIZE, raw_indices)

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    score.copy_(_make_scores(dist, bs, W, lens.long(), seed + 7919))
    graph.replay()
    torch.cuda.synchronize()
    graph_raw = raw_indices.clone()
    graph_page = page_indices.clone()
    launch()
    torch.cuda.synchronize()
    _check_topk_exact(f"graphwide-{dist}", score, lens,
                      graph_raw, graph_page, page_table, PAGE_SIZE)
    _check_topk_exact(f"graphwideeager-{dist}", score, lens,
                      raw_indices, page_indices, page_table, PAGE_SIZE)


# ============================================================================
# (J) raw_indices=None: the optional output may be omitted; the page transform
# must be identical to the run that does emit raw indices.
# ============================================================================

_RAW_NONE_SEQS = [600, 2048, 20000, 66551, 1100000]


@pytest.mark.parametrize("dist", ["gauss", "clustered512"])
@pytest.mark.parametrize("seq_len", _RAW_NONE_SEQS, ids=[f"L{L}" for L in _RAW_NONE_SEQS])
@torch.inference_mode()
def test_topk_raw_indices_optional(seq_len, dist) -> None:
    """raw_indices=None must still satisfy the full top-512 page contract.

    The kernel's output ORDER is not deterministic — the hist4096 scatter
    writes the above-threshold bulk in atomic-race order — so two calls are
    NOT guaranteed bitwise-equal. Each call is therefore validated against
    the torch reference independently. For the raw=None call the raw indices
    are decoded back from the page output through the row's inverse page
    permutation (the "randperm" table is injective), which keeps every
    contract check meaningful without a raw buffer. All lengths here are
    > TOPK, so no -1 padding can appear in the decode.
    """
    bs = 4
    dev = "cuda"
    lens = torch.tensor([seq_len] * bs, dtype=torch.int32, device=dev)
    seed = zlib.crc32(f"rawnone|{dist}|{seq_len}".encode()) % (2**31 - 1)
    score = _make_scores(dist, bs, seq_len, lens.long(), seed)
    T = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    page_table = _make_page_table("randperm", bs, T, seed)
    pb = PAGE_SIZE.bit_length() - 1
    pmask = PAGE_SIZE - 1

    page_a = torch.empty(bs, TOPK, dtype=torch.int32, device=dev)
    torch.ops.sgl_kernel.topk_transform_v1.default(
        score, lens, page_a, page_table, PAGE_SIZE, None)
    inv = torch.argsort(page_table, dim=1)  # inv[r][pt[r][c]] = c
    # Clamp keeps an unexpected page value from turning into an out-of-bounds
    # gather (which would poison the whole CUDA context); a bad slot instead
    # decodes to garbage that the contract check below rejects by name.
    pcol_a = (page_a >> pb).long().clamp(0, T - 1)
    raw_a = (inv.gather(1, pcol_a) << pb) | (page_a & pmask)
    _check_topk_exact(f"rawnone-A-{dist}-L{seq_len}", score, lens,
                      raw_a.to(torch.int32), page_a, page_table, PAGE_SIZE)

    page_b = torch.empty(bs, TOPK, dtype=torch.int32, device=dev)
    raw_b = torch.empty(bs, TOPK, dtype=torch.int32, device=dev)
    torch.ops.sgl_kernel.topk_transform_v1.default(
        score, lens, page_b, page_table, PAGE_SIZE, raw_b)
    torch.cuda.synchronize()
    _check_topk_exact(f"rawnone-B-{dist}-L{seq_len}", score, lens, raw_b,
                      page_b, page_table, PAGE_SIZE)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
