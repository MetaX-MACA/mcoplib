# SPDX-License-Identifier: Apache-2.0
"""Accuracy and bandwidth tests for the fast TopK decode transform."""

import gc
import os
from typing import Iterable, Tuple

import pytest
import torch

import mcoplib.sgl_kernel  # noqa: F401: registers torch.ops.sgl_kernel

TOPK = 2048
TABLE_LEN = 299062
MAX_SEQ_LEN = 107520
COS_THRESHOLD = 0.9999
BATCH_SIZES = (4,8,16,32,64,128,512,1024,2048, 4096)

SEQ_LENGTHS = (64, 3072, 68160, 66551, 107520)
WARMUP = 10
ITERS = 50


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _reference(
    score: torch.Tensor,
    seq_len: int,
    src_page_table: torch.Tensor,
    topk: int = TOPK,
) -> torch.Tensor:
    batch_size = score.size(0)
    if seq_len <= topk:
        out = torch.full(
            (batch_size, topk), -1, dtype=torch.int32, device=score.device
        )
        out[:, :seq_len] = src_page_table[:, :seq_len]
        return out

    indices = torch.topk(score[:, :seq_len], topk, dim=-1, sorted=False).indices
    return torch.gather(src_page_table, 1, indices).to(torch.int32)


def _effective_bytes(bs: int, seq_len: int, topk: int = TOPK) -> int:
    # Account only bytes the decode kernel must touch. For seq_len <= topk the
    # naive branch does not read score. The fast path reads score and gathers
    # exactly topk int32 page-table entries before writing topk int32 outputs.
    lengths_bytes = bs * 4
    page_read = bs * min(seq_len, topk) * 4
    dst_write = bs * topk * 4
    score_read = 0 if seq_len <= topk else bs * seq_len * 4
    return lengths_bytes + score_read + page_read + dst_write


def _benchmark(run, warmup: int = WARMUP, iters: int = ITERS) -> float:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def _make_inputs(bs: int, seq_len: int, table_len: int = TABLE_LEN):
    # Match the real decode contract: score retains the maximum sequence stride
    # while lengths selects the valid prefix. Build a real dense [B, table_len]
    # page table; a stride-0 expanded table unrealistically makes every row hit
    # one small L2-resident source and inflates measured bandwidth.
    score = torch.randn(bs, seq_len, dtype=torch.float32, device="cuda")
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    row_offset = torch.arange(bs, dtype=torch.int32, device="cuda")[:, None]
    col = torch.arange(table_len, dtype=torch.int32, device="cuda")[None, :]
    src_page_table = row_offset * table_len + col
    dst_page_table = torch.empty((bs, TOPK), dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(bs + 1, dtype=torch.int32, device="cuda")
    return score, lengths, src_page_table, dst_page_table, cu_seqlens_q


def run_case(bs: int, seq_len: int, *, check_accuracy: bool = True) -> Tuple[float, float, float]:
    torch.manual_seed(42 + bs + seq_len)
    torch.cuda.manual_seed_all(42 + bs + seq_len)
    score, lengths, src_page_table, dst, cu_seqlens_q = _make_inputs(bs, seq_len)

    def launch() -> None:
        torch.ops.sgl_kernel.fast_topk_transform_fused.default(
            score, lengths, dst, src_page_table, cu_seqlens_q, None
        )

    launch()
    torch.cuda.synchronize()
    cos = 1.0
    if check_accuracy:
        ref = _reference(score, seq_len, src_page_table)
        # The TopK order is intentionally unspecified. Sorting validates the
        # exact selected page-table set; equality is stricter than the requested
        # cosine threshold, which is printed as an extra metric.
        got_sorted = torch.sort(dst, dim=-1).values
        ref_sorted = torch.sort(ref, dim=-1).values
        if not torch.equal(got_sorted, ref_sorted):
            # Random normal input can contain exactly equal FP32 values at the
            # kth boundary. torch.topk and the radix kernel may legally choose
            # different indices with the same score. Verify every substitution
            # is value-equivalent, matching the operator's set semantics.
            for row in torch.where((got_sorted != ref_sorted).any(dim=1))[0]:
                got_set = set(dst[row].cpu().tolist())
                ref_set = set(ref[row].cpu().tolist())
                offset = row.item() * TABLE_LEN
                got_idx = [x - offset for x in got_set - ref_set]
                ref_idx = [x - offset for x in ref_set - got_set]
                got_only = sorted(score[row, got_idx].cpu().tolist())
                ref_only = sorted(score[row, ref_idx].cpu().tolist())
                assert got_only == ref_only, (bs, seq_len, row.item())
        cos = _cosine_similarity(got_sorted, ref_sorted)
        assert cos >= COS_THRESHOLD, (bs, seq_len, cos)

    ms = _benchmark(launch)
    gbps = _effective_bytes(bs, seq_len) / (ms * 1e-3) / 1e9
    status = "OK" if cos >= COS_THRESHOLD else "FAIL"
    print(
        f"[topk_decode] B={bs:5d} S={seq_len:6d} K={TOPK:4d}  "
        f"cos_sim={cos:.6f} {status:4s}  {ms:8.4f} ms  {gbps:8.1f} GB/s",
        flush=True,
    )
    del score, lengths, src_page_table, dst, cu_seqlens_q
    gc.collect()
    torch.cuda.empty_cache()
    return cos, ms, gbps


@pytest.mark.parametrize("bs", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENGTHS)
@torch.inference_mode()
def test_topk_transform_decode_kernel(bs: int, seq_len: int) -> None:
    run_case(bs, seq_len)


@pytest.mark.parametrize(
    "bs,seq_len,label",
    [
        (64, 3072, "GLM5.1 3K logical-BS16 mtp4"),
        (4, 64512, "GLM5.1 63K logical-BS1 mtp4"),
    ],
)
@torch.inference_mode()
def test_topk_transform_decode_kernel_glm51(bs: int, seq_len: int, label: str) -> None:
    # GLM5.1 multi-token-prediction decode shapes (added; existing shapes preserved).
    run_case(bs, seq_len)


MASS_BATCH_SIZES = (
    1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512
)
MASS_SEQ_LENGTHS = (
    1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65,
    66, 127, 128, 129, 130, 255, 256, 257, 260, 511, 512, 513, 1000,
    1023, 1024, 1025, 1536, 2000, 2047, 2048, 2049, 2160, 2176, 3000,
    3072, 4095, 4096, 6144, 8192,
)
MASS_DISTRIBUTIONS = (
    "randn", "small_randn", "small_uniform", "collapsed", "collapsed_neg",
    "allzero", "positive_heavy", "zero_boundary", "negative_boundary",
    "negative_only", "nan_inf", "denormal_signed_zero", "positive_only", "length_gradient",
)


def _mass_seed(bs: int, seq_len: int, distribution: str) -> int:
    h = 2166136261
    for char in distribution:
        h = ((h ^ ord(char)) * 16777619) & 0xFFFFFFFF
    return ((h ^ (bs * 1009) ^ (seq_len * 9176)) & 0x7FFFFFFF) + 1


def _mass_score(bs: int, seq_len: int, distribution: str, generator: torch.Generator):
    device = "cuda"
    if distribution == "randn":
        return torch.randn(bs, seq_len, generator=generator, device=device)
    if distribution == "small_randn":
        return torch.randn(bs, seq_len, generator=generator, device=device) * 1e-3
    if distribution == "small_uniform":
        return (torch.rand(bs, seq_len, generator=generator, device=device) - 0.5) * 1e-2
    if distribution in ("collapsed", "collapsed_neg"):
        score = torch.zeros(bs, seq_len, dtype=torch.float32, device=device)
        probability = max(0.05, min(0.9, 16.0 / max(1, seq_len)))
        mask = torch.rand(bs, seq_len, generator=generator, device=device) < probability
        mask[:, 0] = True
        values = torch.randn(bs, seq_len, generator=generator, device=device) * 5.0
        if distribution == "collapsed_neg":
            row_neg = torch.rand(bs, 1, generator=generator, device=device) < 0.5
            values = torch.where(row_neg, -values.abs(), values)
        score[mask] = values[mask]
        return score
    if distribution == "allzero":
        return torch.zeros(bs, seq_len, dtype=torch.float32, device=device)
    if distribution == "positive_heavy":
        score = torch.zeros(bs, seq_len, dtype=torch.float32, device=device)
        count = min(seq_len, TOPK)
        score[:, :count] = 0.25 + torch.arange(count, device=device) * 1e-6
        return score
    if distribution == "zero_boundary":
        score = torch.zeros(bs, seq_len, dtype=torch.float32, device=device)
        if seq_len > TOPK:
            score.fill_(-1.0)
            score[:, : TOPK - 1] = 0.5 + torch.arange(TOPK - 1, device=device) * 1e-6
            score[:, TOPK - 1] = 0.0
        return score
    if distribution == "negative_boundary":
        score = torch.full((bs, seq_len), -1.0, dtype=torch.float32, device=device)
        if seq_len > TOPK:
            score[:, : TOPK - 1] = 0.5 + torch.arange(TOPK - 1, device=device) * 1e-6
        return score
    if distribution == "negative_only":
        return -torch.rand(bs, seq_len, generator=generator, device=device) - 1e-3
    if distribution == "nan_inf":
        score = torch.zeros(bs, seq_len, dtype=torch.float32, device=device)
        special = (float("nan"), float("inf"), -float("inf"), 1e30, -1e30)
        for column, value in enumerate(special):
            if column < seq_len:
                score[:, column] = value
        if seq_len > 5:
            end = min(seq_len, 100)
            score[:, 5:end] = torch.arange(end - 5, device=device) * 0.5
        return score
    if distribution == "denormal_signed_zero":
        score = torch.zeros(bs, seq_len, dtype=torch.float32, device=device)
        values = (1e-45, -1e-45, -0.0, 7.5, -3.25)
        for column, value in enumerate(values):
            if column < seq_len:
                score[:, column] = value
        if seq_len > 5:
            end = min(seq_len, 60)
            score[:, 5:end] = torch.arange(end - 5, device=device) * 0.125
        return score
    if distribution == "positive_only":
        return torch.rand(bs, seq_len, generator=generator, device=device) * 3.0 + 1e-6
    if distribution == "length_gradient":
        return torch.randn(bs, seq_len, generator=generator, device=device)
    raise AssertionError(distribution)


def _mass_reference(score, lengths, src_page_table):
    batch_size = score.size(0)
    topk = torch.full((batch_size, TOPK), -1, dtype=torch.int32, device=score.device)
    if bool((lengths == lengths[0]).all()):
        length = int(lengths[0].item())
        if length <= TOPK:
            topk[:, :length] = src_page_table[:, :length]
            return topk
        indices = torch.topk(score[:, :length], TOPK, dim=-1, sorted=False).indices
        return torch.gather(src_page_table, 1, indices).to(torch.int32)
    for row in range(batch_size):
        length = int(lengths[row].item())
        if length <= TOPK:
            topk[row, :length] = src_page_table[row, :length]
        else:
            indices = torch.topk(score[row, :length], TOPK, sorted=False).indices
            topk[row] = src_page_table[row, indices]
    return topk


def _mass_check(score, lengths, src_page_table, dst, stride, context):
    reference = _mass_reference(score, lengths, src_page_table)
    # TopK order is unspecified on value ties: an all-zero row may select any
    # TopK positions, so comparing document ids is wrong. Validate the exact
    # selected score multiset instead (NaN-aware).
    row_stride = (torch.arange(dst.size(0), device=dst.device) * stride)[:, None]
    got_pos = dst - row_stride
    ref_pos = reference - row_stride

    def selected_values(pos):
        valid = (pos >= 0) & (pos < score.size(1)) & (pos < lengths[:, None])
        values = score.gather(1, pos.clamp_min(0))
        return torch.sort(values[valid].flatten()).values

    got_values = selected_values(got_pos)
    ref_values = selected_values(ref_pos)
    assert got_values.numel() == ref_values.numel(), (*context, got_values.numel(), ref_values.numel())
    equal = (got_values == ref_values) | (torch.isnan(got_values) & torch.isnan(ref_values))
    assert bool(equal.all()), context


def run_mass_case(bs: int, seq_len: int, distribution: str) -> None:
    seq_len = min(seq_len, MAX_SEQ_LEN)
    generator = torch.Generator(device="cuda").manual_seed(_mass_seed(bs, seq_len, distribution))
    score = _mass_score(bs, seq_len, distribution, generator)
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    if distribution == "length_gradient":
        step = max(1, seq_len // 8)
        lengths = torch.tensor(
            [max(1, seq_len - (row % 8) * step) for row in range(bs)],
            dtype=torch.int32,
            device="cuda",
        )
        for row, length in enumerate(lengths.tolist()):
            score[row, length:] = 0.0
    stride = seq_len
    row_offset = torch.arange(bs, dtype=torch.int32, device="cuda")[:, None]
    col = torch.arange(stride, dtype=torch.int32, device="cuda")[None, :]
    src_page_table = row_offset * stride + col
    dst = torch.empty((bs, TOPK), dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(bs + 1, dtype=torch.int32, device="cuda")
    torch.ops.sgl_kernel.fast_topk_transform_fused.default(
        score, lengths, dst, src_page_table, cu_seqlens_q, None
    )
    torch.cuda.synchronize()
    _mass_check(score, lengths, src_page_table, dst, stride, (bs, seq_len, distribution))


@pytest.mark.parametrize("bs", MASS_BATCH_SIZES)
@pytest.mark.parametrize("seq_len", MASS_SEQ_LENGTHS)
@pytest.mark.parametrize("distribution", MASS_DISTRIBUTIONS)
@torch.inference_mode()
def test_topk_transform_decode_mass_matrix(bs: int, seq_len: int, distribution: str) -> None:
    run_mass_case(bs, seq_len, distribution)


@pytest.mark.parametrize("distribution", ("randn", "collapsed", "allzero", "nan_inf"))
@pytest.mark.parametrize(
    "bs,seq_len",
    ((1, 202752), (6, 2048), (1, 64), (128, 2048), (128, 2176)),
)
@torch.inference_mode()
def test_topk_transform_decode_requested_shapes(bs: int, seq_len: int, distribution: str) -> None:
    run_mass_case(bs, min(seq_len, MAX_SEQ_LEN), distribution)


@pytest.mark.parametrize("distribution", ("randn", "collapsed"))
@pytest.mark.parametrize(
    "bs,seq_len",
    ((1024, 3072), (1024, 16384), (2048, 48128), (4096, 64512),
     (4096, 66551), (4096, 68160), (4096, 107520)),
)
@torch.inference_mode()
def test_topk_transform_decode_large_shapes(bs: int, seq_len: int, distribution: str) -> None:
    run_mass_case(bs, seq_len, distribution)


def _cases() -> Iterable[Tuple[int, int]]:
    # Run each sequence regime across all requested batch sizes. This ordering
    # keeps adjacent allocations similar and makes small/large-path trends clear.
    for seq_len in SEQ_LENGTHS:
        for bs in BATCH_SIZES:
            yield bs, seq_len
    # GLM5.1 production decode shapes (logical batch x mtp4 = actual batch):
    yield 64, 3072    # GLM5.1 3K context, logical BS16, mtp4 -> actual batch 64
    yield 4, 64512    # GLM5.1 63K context, logical BS1, mtp4 -> actual batch 4


@torch.inference_mode()
def main() -> None:
    assert torch.cuda.is_available()
    print(
        f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')!r}  "
        f"device_count={torch.cuda.device_count()}"
    )
    print(
        f"config: dtype=fp32 index=int32 topk={TOPK} table_len={TABLE_LEN} "
        f"max_seq_len={MAX_SEQ_LEN}"
    )
    print("=" * 104)
    peak = 0.0
    all_pass = True
    for bs, seq_len in _cases():
        cos, _, gbps = run_case(bs, seq_len)
        peak = max(peak, gbps)
        all_pass &= cos >= COS_THRESHOLD
    print("=" * 104)
    print(
        f"Accuracy: {'ALL PASS' if all_pass else 'FAILED'} "
        f"(threshold cos_sim >= {COS_THRESHOLD})"
    )
    print(f"Peak effective bandwidth: {peak:.1f} GB/s  (single-die target 1300)")


if __name__ == "__main__":
    main()
