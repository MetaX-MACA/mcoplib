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
BATCH_SIZES = (1,4,8, 16, 1024,2048, 4096)
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


def _cases() -> Iterable[Tuple[int, int]]:
    # Run each sequence regime across all requested batch sizes. This ordering
    # keeps adjacent allocations similar and makes small/large-path trends clear.
    for seq_len in SEQ_LENGTHS:
        for bs in BATCH_SIZES:
            yield bs, seq_len


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
