# SPDX-License-Identifier: Apache-2.0
"""tests for the mcoplib AOT kpool Top-K transform.

The production operator selects pool groups, expands each selected group to
``pool_size`` token indices, and optionally applies a page-table mapping or a
per-row ragged offset.  Radix selection does not guarantee score-sorted output,
so correctness compares expanded groups as unordered sets and checks tail and
padding exactly. This self-contained file includes interface generalization,
GLM-5 production-workload correctness, and CUDA-Event performance tests.

Usage (run from the mcoplib repository root)::

    # Fast/base/interface correctness. Expected: 37 passed.
    python -m pytest -o addopts="" -v -s -m "not full" \
      unit_test/test_kpool_topk_transform.py

    # GLM-5 correctness, including the >4096-candidate fallback. Expected: 5 passed.
    python -m pytest -o addopts="" -v -s -m full \
      unit_test/test_kpool_topk_transform.py

    # All correctness cases. Expected: 42 passed.
    python -m pytest -o addopts="" -v -s \
      unit_test/test_kpool_topk_transform.py

    # Representative interface-generalization performance matrix.
    python unit_test/test_kpool_topk_transform.py --benchmark \
      --suite general --warmup 5000 --iters 1000 --repeats 20 \
      --csv benchmark_results/kpool_submit_general_c600u.csv

    # Only the four GLM-5 production workloads.
    python unit_test/test_kpool_topk_transform.py --benchmark \
      --suite glm --warmup 5000 --iters 1000 --repeats 20 \
      --csv benchmark_results/kpool_submit_glm_c600u.csv

    # Isolate Prefill 63 Ki for mcTracer.
    mcTracer python unit_test/test_kpool_topk_transform.py --benchmark \
      --suite glm --case prefill_63ki --warmup 20 --iters 5 --repeats 3
"""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import pytest
import torch

import mcoplib.sgl_kernel  # noqa: F401  (registers torch.ops.sgl_kernel)


SUPPORTED_GROUP_TOPK = (128, 160, 192, 224, 256, 512)
POOL_SIZE = 4


def reference_kpool_topk_transform(
    score: torch.Tensor,
    lengths: torch.Tensor,
    pool_size: int,
    token_topk: int,
    *,
    page_table: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    page_table_row_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """PyTorch specification matching the attachment's public contract."""
    if token_topk % pool_size:
        raise ValueError("token_topk must be divisible by pool_size")
    if page_table is not None and topk_indices_offset is not None:
        raise ValueError("page_table and topk_indices_offset are mutually exclusive")

    batch = score.shape[0]
    group_topk = token_topk // pool_size
    tail_cols = pool_size - 1 if seq_lens is not None else 0
    output = torch.full(
        (batch, token_topk + tail_cols),
        -1,
        dtype=torch.int32,
        device=score.device,
    )

    for row in range(batch):
        length = int(lengths[row].item())
        row_start = 0 if row_starts is None else int(row_starts[row].item())

        if length <= group_topk:
            groups = torch.arange(length, device=score.device, dtype=torch.int64)
        else:
            groups = torch.topk(
                score[row, row_start : row_start + length],
                group_topk,
                sorted=False,
            ).indices.to(torch.int64)

        raw_history = (
            groups[:, None] * pool_size
            + torch.arange(pool_size, device=score.device, dtype=torch.int64)
        ).reshape(-1)
        raw = raw_history

        if seq_lens is not None:
            tail_count = int(seq_lens[row].item()) % pool_size
            tail = torch.arange(
                length * pool_size,
                length * pool_size + tail_count,
                device=score.device,
                dtype=torch.int64,
            )
            raw = torch.cat((raw, tail))

        if page_table is not None:
            table_row = (
                row
                if page_table_row_index is None
                else int(page_table_row_index[row].item())
            )
            mapped = page_table[table_row, raw]
        elif topk_indices_offset is not None:
            mapped = raw + int(topk_indices_offset[row].item())
        else:
            mapped = raw

        output[row, : mapped.numel()] = mapped.to(torch.int32)

    return output


def run_mcoplib_op(
    score: torch.Tensor,
    lengths: torch.Tensor,
    pool_size: int,
    group_topk: int,
    *,
    page_table: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    page_table_row_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Allocate the output and invoke the future mcoplib AOT operator."""
    token_topk = group_topk * pool_size
    tail_cols = pool_size - 1 if seq_lens is not None else 0
    output = torch.empty(
        (score.shape[0], token_topk + tail_cols),
        dtype=torch.int32,
        device=score.device,
    )
    torch.ops.sgl_kernel.kpool_topk_transform.default(
        score,
        lengths,
        output,
        pool_size,
        page_table,
        topk_indices_offset,
        row_starts,
        seq_lens,
        page_table_row_index,
    )
    return output


def assert_kpool_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    lengths: torch.Tensor,
    pool_size: int,
    group_topk: int,
) -> None:
    """Compare unordered selected groups, then deterministic tail/padding."""
    for row in range(actual.shape[0]):
        history_groups = min(int(lengths[row].item()), group_topk)
        history_tokens = history_groups * pool_size

        actual_groups = actual[row, :history_tokens].reshape(-1, pool_size)
        expected_groups = expected[row, :history_tokens].reshape(-1, pool_size)
        actual_canonical = sorted(map(tuple, actual_groups.cpu().tolist()))
        expected_canonical = sorted(map(tuple, expected_groups.cpu().tolist()))
        assert actual_canonical == expected_canonical
        assert torch.equal(
            actual[row, history_tokens:].cpu(),
            expected[row, history_tokens:].cpu(),
        )


def make_distinct_scores(
    lengths: torch.Tensor, row_starts: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Create per-row unique scores with optional valid-region offsets."""
    batch = lengths.numel()
    starts = torch.zeros_like(lengths) if row_starts is None else row_starts
    stride = int(torch.max(lengths + starts).item()) + 7
    score = torch.full((batch, stride), -1.0e6, dtype=torch.float32, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(20260901)
    for row in range(batch):
        length = int(lengths[row].item())
        start = int(starts[row].item())
        score[row, start : start + length] = torch.randperm(
            length, generator=generator, device="cuda", dtype=torch.int64
        ).float()
    return score


def make_fixed_width_distinct_scores(
    lengths: torch.Tensor,
    width: int,
    *,
    row_starts: Optional[torch.Tensor] = None,
    extra_row_stride: int = 0,
) -> torch.Tensor:
    """Create signed, distinct scores with an exact visible row width."""
    batch = lengths.numel()
    starts = torch.zeros_like(lengths) if row_starts is None else row_starts
    storage = torch.full(
        (batch, width + extra_row_stride),
        -1.0e6,
        dtype=torch.float32,
        device="cuda",
    )
    score = storage[:, :width]
    generator = torch.Generator(device="cuda").manual_seed(
        20260907 + batch + width
    )
    for row in range(batch):
        length = int(lengths[row].item())
        start = int(starts[row].item())
        assert 0 <= start and 0 <= length and start + length <= width
        values = torch.randperm(
            length, generator=generator, device="cuda", dtype=torch.int64
        ).float()
        score[row, start : start + length] = values - length * 0.5
    return score


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a MetaX CUDA-compatible GPU"
)


@pytest.mark.parametrize("group_topk", SUPPORTED_GROUP_TOPK)
@torch.inference_mode()
def test_all_group_topk_specializations_with_offset(group_topk: int) -> None:
    lengths = torch.tensor(
        [group_topk + 17, group_topk + 23], dtype=torch.int32, device="cuda"
    )
    score = make_distinct_scores(lengths)
    offsets = torch.tensor([10000, 20000], dtype=torch.int32, device="cuda")

    actual = run_mcoplib_op(
        score,
        lengths,
        POOL_SIZE,
        group_topk,
        topk_indices_offset=offsets,
    )
    expected = reference_kpool_topk_transform(
        score,
        lengths,
        POOL_SIZE,
        group_topk * POOL_SIZE,
        topk_indices_offset=offsets,
    )
    assert_kpool_equal(actual, expected, lengths, POOL_SIZE, group_topk)


@pytest.mark.parametrize("group_topk", SUPPORTED_GROUP_TOPK)
@torch.inference_mode()
def test_k_plus_one_fast_path(group_topk: int) -> None:
    lengths = torch.tensor(
        [group_topk + 1, group_topk + 1], dtype=torch.int32, device="cuda"
    )
    score = make_distinct_scores(lengths)
    offsets = torch.tensor([10000, 20000], dtype=torch.int32, device="cuda")

    actual = run_mcoplib_op(
        score,
        lengths,
        POOL_SIZE,
        group_topk,
        topk_indices_offset=offsets,
    )
    expected = reference_kpool_topk_transform(
        score,
        lengths,
        POOL_SIZE,
        group_topk * POOL_SIZE,
        topk_indices_offset=offsets,
    )
    assert_kpool_equal(actual, expected, lengths, POOL_SIZE, group_topk)


@pytest.mark.parametrize("row_lengths", ((129, 129), (145, 151)))
@torch.inference_mode()
def test_page_table_row_index_row_starts_and_tail(
    row_lengths: tuple[int, int],
) -> None:
    group_topk = 128
    lengths = torch.tensor(row_lengths, dtype=torch.int32, device="cuda")
    row_starts = torch.tensor([3, 5], dtype=torch.int32, device="cuda")
    score = make_distinct_scores(lengths, row_starts)
    seq_lens = lengths * POOL_SIZE + torch.tensor(
        [1, 3], dtype=torch.int32, device="cuda"
    )
    page_table_row_index = torch.tensor([2, 0], dtype=torch.int32, device="cuda")
    table_width = int(lengths.max().item()) * POOL_SIZE + POOL_SIZE
    page_table = torch.arange(
        table_width, dtype=torch.int32, device="cuda"
    ).repeat(3, 1)
    page_table += torch.arange(3, dtype=torch.int32, device="cuda")[:, None] * 100000

    actual = run_mcoplib_op(
        score,
        lengths,
        POOL_SIZE,
        group_topk,
        page_table=page_table,
        row_starts=row_starts,
        seq_lens=seq_lens,
        page_table_row_index=page_table_row_index,
    )
    expected = reference_kpool_topk_transform(
        score,
        lengths,
        POOL_SIZE,
        group_topk * POOL_SIZE,
        page_table=page_table,
        row_starts=row_starts,
        seq_lens=seq_lens,
        page_table_row_index=page_table_row_index,
    )
    assert_kpool_equal(actual, expected, lengths, POOL_SIZE, group_topk)


@torch.inference_mode()
def test_vectorized_scan_alignment_and_tail() -> None:
    """Cover every float4 alignment and scalar-tail size in the radix path."""
    group_topk = 128
    lengths = torch.tensor(
        [32768, 32764, 32764, 32764], dtype=torch.int32, device="cuda"
    )
    row_starts = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")

    # A width of 32768 selects the vec8 kernel and keeps every row base
    # 16-byte aligned. The four row starts then exercise every alignment;
    # these lengths also produce scalar tails of zero through three floats.
    score = torch.full((4, 32768), -1.0e6, dtype=torch.float32, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(20260902)
    for row in range(lengths.numel()):
        length = int(lengths[row].item())
        start = int(row_starts[row].item())
        score[row, start : start + length] = torch.randperm(
            length, generator=generator, device="cuda", dtype=torch.int64
        ).float()

    actual = run_mcoplib_op(
        score,
        lengths,
        POOL_SIZE,
        group_topk,
        row_starts=row_starts,
    )
    expected = reference_kpool_topk_transform(
        score,
        lengths,
        POOL_SIZE,
        group_topk * POOL_SIZE,
        row_starts=row_starts,
    )
    assert_kpool_equal(actual, expected, lengths, POOL_SIZE, group_topk)


@torch.inference_mode()
def test_long_rows_with_page_table() -> None:
    group_topk = 128
    lengths = torch.tensor([8191, 32768], dtype=torch.int32, device="cuda")
    score = make_distinct_scores(lengths)
    table_width = int(lengths.max().item()) * POOL_SIZE
    page_table = torch.arange(
        lengths.numel() * table_width, dtype=torch.int32, device="cuda"
    ).reshape(lengths.numel(), table_width)

    actual = run_mcoplib_op(
        score,
        lengths,
        POOL_SIZE,
        group_topk,
        page_table=page_table,
    )
    expected = reference_kpool_topk_transform(
        score,
        lengths,
        POOL_SIZE,
        group_topk * POOL_SIZE,
        page_table=page_table,
    )
    assert_kpool_equal(actual, expected, lengths, POOL_SIZE, group_topk)


@torch.inference_mode()
def test_short_row_tail_and_padding() -> None:
    group_topk = 128
    lengths = torch.tensor([3], dtype=torch.int32, device="cuda")
    score = torch.zeros((1, group_topk), dtype=torch.float32, device="cuda")
    seq_lens = torch.tensor([3 * POOL_SIZE + 2], dtype=torch.int32, device="cuda")

    actual = run_mcoplib_op(
        score, lengths, POOL_SIZE, group_topk, seq_lens=seq_lens
    )
    expected = reference_kpool_topk_transform(
        score,
        lengths,
        POOL_SIZE,
        group_topk * POOL_SIZE,
        seq_lens=seq_lens,
    )
    assert torch.equal(actual.cpu(), expected.cpu())


@torch.inference_mode()
def test_more_than_4096_candidates_in_one_coarse_radix_bucket() -> None:
    """Catch silent truncation of a large threshold bucket in the attachment."""
    group_topk = 128
    length = 5000
    base_bits = 0x3F800000  # float32 1.0
    score_bits = torch.arange(
        base_bits - 2500,
        base_bits + 2500,
        dtype=torch.int32,
        device="cuda",
    )
    score = score_bits.view(torch.float32).unsqueeze(0)
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")

    # All values round to the same FP16 value used by the coarse 8-bit radix
    # pass, while remaining distinct FP32 values for exact Top-K selection.
    assert torch.unique(score.half()).numel() == 1
    assert torch.unique(score).numel() == length

    actual = run_mcoplib_op(score, lengths, POOL_SIZE, group_topk)
    expected = reference_kpool_topk_transform(
        score, lengths, POOL_SIZE, group_topk * POOL_SIZE
    )
    assert_kpool_equal(actual, expected, lengths, POOL_SIZE, group_topk)


@pytest.mark.parametrize(
    "row_lengths",
    (
        pytest.param((0,), id="batch1_zero_length"),
        pytest.param((1, 128), id="batch2_short_and_k"),
        pytest.param(
            (0, 1, 127, 128, 129, 511, 513),
            id="batch7_length_boundaries",
        ),
        pytest.param(tuple((index * 37) % 514 for index in range(31)), id="batch31"),
        pytest.param(tuple((index * 37) % 514 for index in range(32)), id="batch32"),
        pytest.param(tuple((index * 37) % 514 for index in range(33)), id="batch33"),
    ),
)
@torch.inference_mode()
def test_interface_generalization_batch_and_seq_len(
    row_lengths: tuple[int, ...],
) -> None:
    """Cover batch sizes and valid lengths around short/radix boundaries."""
    group_topk = 128
    pool_size = 4
    lengths = torch.tensor(row_lengths, dtype=torch.int32, device="cuda")
    score = make_distinct_scores(lengths)
    remainders = (
        torch.arange(lengths.numel(), dtype=torch.int32, device="cuda") + 1
    ) % pool_size
    seq_lens = lengths * pool_size + remainders
    offsets = (
        torch.arange(lengths.numel(), dtype=torch.int32, device="cuda")
        * 10000
        - 50000
    )

    actual = run_mcoplib_op(
        score,
        lengths,
        pool_size,
        group_topk,
        topk_indices_offset=offsets,
        seq_lens=seq_lens,
    )
    expected = reference_kpool_topk_transform(
        score,
        lengths,
        pool_size,
        group_topk * pool_size,
        topk_indices_offset=offsets,
        seq_lens=seq_lens,
    )
    assert_kpool_equal(actual, expected, lengths, pool_size, group_topk)


@pytest.mark.parametrize(
    ("pool_size", "mode"),
    (
        pytest.param(2, "none", id="pool2_none"),
        pytest.param(3, "offset", id="pool3_offset"),
        pytest.param(4, "page_table", id="pool4_page_table"),
        pytest.param(5, "page_table_index", id="pool5_compact_page_table"),
        pytest.param(8, "offset", id="pool8_offset"),
    ),
)
@torch.inference_mode()
def test_interface_generalization_pool_size_mapping_and_tail(
    pool_size: int, mode: str
) -> None:
    """Exercise generic pool expansion, mapping, tail, and padding paths."""
    group_topk = 128
    lengths = torch.tensor(
        [0, 1, 127, 128, 129, 257], dtype=torch.int32, device="cuda"
    )
    score = make_distinct_scores(lengths)
    remainders = (
        torch.arange(lengths.numel(), dtype=torch.int32, device="cuda") + 1
    ) % pool_size
    seq_lens = lengths * pool_size + remainders

    offsets = None
    page_table = None
    page_table_row_index = None
    if mode == "offset":
        offsets = (
            torch.arange(lengths.numel(), dtype=torch.int32, device="cuda")
            * 10000
        )
    elif mode == "page_table":
        table_width = int(lengths.max().item()) * pool_size + pool_size
        storage_width = table_width + 7
        page_table_storage = torch.arange(
            lengths.numel() * storage_width,
            dtype=torch.int32,
            device="cuda",
        ).reshape(lengths.numel(), storage_width)
        page_table = page_table_storage[:, :table_width]
        assert not page_table.is_contiguous()
        assert page_table.stride(1) == 1
    elif mode == "page_table_index":
        table_width = int(lengths.max().item()) * pool_size + pool_size
        page_table = torch.arange(
            3 * table_width, dtype=torch.int32, device="cuda"
        ).reshape(3, table_width)
        page_table_row_index = torch.tensor(
            [2, 0, 2, 1, 0, 1], dtype=torch.int32, device="cuda"
        )

    actual = run_mcoplib_op(
        score,
        lengths,
        pool_size,
        group_topk,
        page_table=page_table,
        topk_indices_offset=offsets,
        seq_lens=seq_lens,
        page_table_row_index=page_table_row_index,
    )
    expected = reference_kpool_topk_transform(
        score,
        lengths,
        pool_size,
        group_topk * pool_size,
        page_table=page_table,
        topk_indices_offset=offsets,
        seq_lens=seq_lens,
        page_table_row_index=page_table_row_index,
    )
    assert_kpool_equal(actual, expected, lengths, pool_size, group_topk)


@torch.inference_mode()
def test_interface_generalization_noncontiguous_row_stride() -> None:
    """A score tensor may be non-contiguous when its inner stride is one."""
    group_topk = 128
    lengths = torch.tensor(
        [129, 257, 513, 1025], dtype=torch.int32, device="cuda"
    )
    width = 1032
    score = make_fixed_width_distinct_scores(
        lengths, width, extra_row_stride=7
    )
    assert not score.is_contiguous()
    assert score.stride(1) == 1
    offsets = torch.tensor(
        [0, 10000, 20000, 30000], dtype=torch.int32, device="cuda"
    )

    actual = run_mcoplib_op(
        score,
        lengths,
        POOL_SIZE,
        group_topk,
        topk_indices_offset=offsets,
    )
    expected = reference_kpool_topk_transform(
        score,
        lengths,
        POOL_SIZE,
        group_topk * POOL_SIZE,
        topk_indices_offset=offsets,
    )
    assert_kpool_equal(actual, expected, lengths, POOL_SIZE, group_topk)


@pytest.mark.parametrize(
    ("batch_size", "width"),
    (
        pytest.param(31, 16126, id="below_cache_batch_threshold"),
        pytest.param(32, 16125, id="below_vec8_width_threshold"),
        pytest.param(32, 16126, id="coarse_cache_lower_width"),
        pytest.param(32, 16384, id="coarse_cache_upper_width"),
        pytest.param(32, 16385, id="above_coarse_cache_width"),
    ),
)
@torch.inference_mode()
def test_interface_generalization_long_dispatch_boundaries(
    batch_size: int, width: int
) -> None:
    """Validate both sides of the Vec8 and coarse-cache dispatch boundaries."""
    group_topk = 512
    lengths = torch.tensor(
        [width - (row % 7) for row in range(batch_size)],
        dtype=torch.int32,
        device="cuda",
    )
    score = make_fixed_width_distinct_scores(lengths, width)
    offsets = (
        torch.arange(batch_size, dtype=torch.int32, device="cuda")
        * width
        * POOL_SIZE
    )

    actual = run_mcoplib_op(
        score,
        lengths,
        POOL_SIZE,
        group_topk,
        topk_indices_offset=offsets,
    )
    expected = reference_kpool_topk_transform(
        score,
        lengths,
        POOL_SIZE,
        group_topk * POOL_SIZE,
        topk_indices_offset=offsets,
    )
    assert_kpool_equal(actual, expected, lengths, POOL_SIZE, group_topk)


@torch.inference_mode()
def test_interface_generalization_coarse_cache_alignment() -> None:
    """Exercise all float4 alignments while the coarse cache is active."""
    batch_size = 32
    width = 16384
    group_topk = 512
    lengths = torch.full(
        (batch_size,), width - 4, dtype=torch.int32, device="cuda"
    )
    row_starts = torch.arange(
        batch_size, dtype=torch.int32, device="cuda"
    ) % 4
    score = make_fixed_width_distinct_scores(
        lengths, width, row_starts=row_starts
    )

    actual = run_mcoplib_op(
        score,
        lengths,
        POOL_SIZE,
        group_topk,
        row_starts=row_starts,
    )
    expected = reference_kpool_topk_transform(
        score,
        lengths,
        POOL_SIZE,
        group_topk * POOL_SIZE,
        row_starts=row_starts,
    )
    assert_kpool_equal(actual, expected, lengths, POOL_SIZE, group_topk)


@torch.inference_mode()
def test_interface_generalization_empty_batch() -> None:
    """The interface accepts an empty batch and must return without launch."""
    group_topk = 128
    score = torch.empty((0, 17), dtype=torch.float32, device="cuda")
    lengths = torch.empty((0,), dtype=torch.int32, device="cuda")
    actual = run_mcoplib_op(score, lengths, POOL_SIZE, group_topk)
    expected = reference_kpool_topk_transform(
        score, lengths, POOL_SIZE, group_topk * POOL_SIZE
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mapping", ["offset", "page_table"])
@torch.inference_mode()
def test_kpool_topk_transform_cuda_graph_replay(mapping: str) -> None:
    """Captured execution must replay correctly with changed score contents."""
    pool_size = 4
    group_topk = 128
    lengths = torch.tensor([257, 263], dtype=torch.int32, device="cuda")
    score = make_distinct_scores(lengths)
    output = torch.empty((2, group_topk * pool_size), dtype=torch.int32, device="cuda")
    offsets = None
    page_table = None
    if mapping == "offset":
        offsets = torch.tensor([10000, 20000], dtype=torch.int32, device="cuda")
    else:
        page_table = torch.arange(
            2 * int(lengths.max().item()) * pool_size,
            dtype=torch.int32,
            device="cuda",
        ).reshape(2, -1)

    def launch() -> None:
        torch.ops.sgl_kernel.kpool_topk_transform.default(
            score,
            lengths,
            output,
            pool_size,
            page_table,
            offsets,
            None,
            None,
            None,
        )

    # Initialize cached device properties and kernel attributes before capture.
    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            launch()
    except Exception as exc:
        pytest.fail(
            f"kpool_topk_transform graph capture failed for {mapping=}: "
            f"{type(exc).__name__}: {exc}"
        )

    for seed in (20260910, 20260911):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        score.copy_(torch.randn(score.shape, generator=generator, device="cuda"))
        graph.replay()
        graph_output = output.clone()

        launch()
        torch.cuda.synchronize()
        expected = reference_kpool_topk_transform(
            score,
            lengths,
            pool_size,
            group_topk * pool_size,
            page_table=page_table,
            topk_indices_offset=offsets,
        )
        assert_kpool_equal(graph_output, expected, lengths, pool_size, group_topk)
        assert_kpool_equal(output, expected, lengths, pool_size, group_topk)


# ---------------------------------------------------------------------------
# GLM-5 production-workload correctness
# ---------------------------------------------------------------------------

GLM_INDEX_TOPK = 2048
GLM_GROUP_TOPK = GLM_INDEX_TOPK // POOL_SIZE
GLM_OUT_COLS = GLM_INDEX_TOPK + POOL_SIZE - 1
GLM_PREFILL_CHUNK = 8192
GLM_CP_SIZE = 8
GLM_DECODE_OUTPUT_LEN = 1024


def glm_expected_monotonic(
    group_lengths: torch.Tensor,
    seq_lens: torch.Tensor,
    mapping_offsets: torch.Tensor,
) -> torch.Tensor:
    """Vectorized oracle for strictly increasing GLM workload scores."""
    device = group_lengths.device
    batch = group_lengths.numel()
    selected = torch.clamp(group_lengths, max=GLM_GROUP_TOPK)
    first_group = torch.clamp(group_lengths - GLM_GROUP_TOPK, min=0)

    ranks = torch.arange(GLM_GROUP_TOPK, dtype=torch.int32, device=device)
    slots = torch.arange(POOL_SIZE, dtype=torch.int32, device=device)
    group_ids = first_group[:, None] + ranks[None, :]
    raw_history = group_ids[:, :, None] * POOL_SIZE + slots[None, None, :]
    mapped_history = raw_history + mapping_offsets[:, None, None]
    valid_groups = ranks[None, :] < selected[:, None]
    history = torch.where(
        valid_groups[:, :, None],
        mapped_history,
        torch.full_like(mapped_history, -1),
    )

    output = torch.full(
        (batch, GLM_OUT_COLS), -1, dtype=torch.int32, device=device
    )
    output[:, :GLM_INDEX_TOPK] = history.reshape(batch, GLM_INDEX_TOPK)

    tail_count = torch.remainder(seq_lens, POOL_SIZE)
    history_tokens = selected * POOL_SIZE
    rows = torch.arange(batch, dtype=torch.int64, device=device)
    for slot in range(POOL_SIZE - 1):
        present = tail_count > slot
        present_rows = rows[present]
        columns = history_tokens[present].to(torch.int64) + slot
        values = (
            group_lengths[present] * POOL_SIZE
            + slot
            + mapping_offsets[present]
        )
        output[present_rows, columns] = values
    return output


def assert_glm_workload_equal(
    actual: torch.Tensor,
    expected: torch.Tensor,
    group_lengths: torch.Tensor,
) -> None:
    """Compare unordered history groups and exact tail/padding on device."""
    batch = actual.shape[0]
    selected = torch.clamp(group_lengths, max=GLM_GROUP_TOPK)
    valid_groups = (
        torch.arange(GLM_GROUP_TOPK, device=actual.device)[None, :]
        < selected[:, None]
    )
    sentinel = torch.iinfo(torch.int32).max

    actual_groups = actual[:, :GLM_INDEX_TOPK].reshape(
        batch, GLM_GROUP_TOPK, POOL_SIZE
    )
    expected_groups = expected[:, :GLM_INDEX_TOPK].reshape(
        batch, GLM_GROUP_TOPK, POOL_SIZE
    )
    actual_masked = torch.where(
        valid_groups[:, :, None], actual_groups, sentinel
    )
    expected_masked = torch.where(
        valid_groups[:, :, None], expected_groups, sentinel
    )
    actual_order = actual_masked[:, :, 0].argsort(dim=1)
    expected_order = expected_masked[:, :, 0].argsort(dim=1)
    actual_canonical = actual_masked.gather(
        1, actual_order[:, :, None].expand_as(actual_masked)
    )
    expected_canonical = expected_masked.gather(
        1, expected_order[:, :, None].expand_as(expected_masked)
    )
    torch.testing.assert_close(
        actual_canonical, expected_canonical, rtol=0, atol=0
    )

    columns = torch.arange(GLM_OUT_COLS, device=actual.device)[None, :]
    remainder = columns >= (selected * POOL_SIZE)[:, None]
    torch.testing.assert_close(
        actual[remainder], expected[remainder], rtol=0, atol=0
    )


def check_glm_prefill(input_len: int, *, clustered: bool = False) -> None:
    chunk_start = max(0, input_len - GLM_PREFILL_CHUNK)
    positions = torch.arange(
        chunk_start,
        input_len,
        GLM_CP_SIZE,
        dtype=torch.int32,
        device="cuda",
    )
    seq_lens = positions + 1
    group_lengths = torch.div(
        seq_lens, POOL_SIZE, rounding_mode="floor"
    )
    max_groups = int(group_lengths.max().item())

    score = torch.arange(max_groups, dtype=torch.float32, device="cuda")
    if clustered:
        score = 1.0 + score * 1.0e-6
    score = score.unsqueeze(0).expand(group_lengths.numel(), -1).contiguous()
    offsets = torch.full(
        (group_lengths.numel(),),
        100000,
        dtype=torch.int32,
        device="cuda",
    )

    actual = run_mcoplib_op(
        score,
        group_lengths,
        POOL_SIZE,
        GLM_GROUP_TOPK,
        topk_indices_offset=offsets,
        seq_lens=seq_lens,
    )
    expected = glm_expected_monotonic(group_lengths, seq_lens, offsets)
    assert_glm_workload_equal(actual, expected, group_lengths)


@pytest.mark.full
@pytest.mark.parametrize(
    "input_len", (63 * 1024, 3 * 1024), ids=("input_63ki", "input_3ki")
)
@torch.inference_mode()
def test_glm5_submit_prefill_cp8(input_len: int) -> None:
    check_glm_prefill(input_len)


@pytest.mark.full
@torch.inference_mode()
def test_glm5_submit_prefill_cp8_clustered_overflow() -> None:
    check_glm_prefill(63 * 1024, clustered=True)


def check_glm_decode(input_len: int) -> None:
    seq_lens = torch.tensor(
        [input_len, input_len + GLM_DECODE_OUTPUT_LEN - 1],
        dtype=torch.int32,
        device="cuda",
    )
    group_lengths = torch.div(
        seq_lens, POOL_SIZE, rounding_mode="floor"
    )
    max_groups = int(group_lengths.max().item())
    score = (
        torch.arange(max_groups, dtype=torch.float32, device="cuda")
        .unsqueeze(0)
        .expand(group_lengths.numel(), -1)
        .contiguous()
    )

    mapping_offsets = torch.tensor(
        [200000, 300000], dtype=torch.int32, device="cuda"
    )
    table_width = max_groups * POOL_SIZE + POOL_SIZE
    page_table = (
        torch.arange(table_width, dtype=torch.int32, device="cuda")[None, :]
        + mapping_offsets[:, None]
    )
    page_table_row_index = torch.arange(
        group_lengths.numel(), dtype=torch.int32, device="cuda"
    )

    actual = run_mcoplib_op(
        score,
        group_lengths,
        POOL_SIZE,
        GLM_GROUP_TOPK,
        page_table=page_table,
        seq_lens=seq_lens,
        page_table_row_index=page_table_row_index,
    )
    expected = glm_expected_monotonic(
        group_lengths, seq_lens, mapping_offsets
    )
    assert_glm_workload_equal(actual, expected, group_lengths)


@pytest.mark.full
@pytest.mark.parametrize(
    "input_len", (63 * 1024, 3 * 1024), ids=("input_63ki", "input_3ki")
)
@torch.inference_mode()
def test_glm5_submit_decode_dp16(input_len: int) -> None:
    check_glm_decode(input_len)


# ---------------------------------------------------------------------------
# CUDA-Event performance benchmark
# ---------------------------------------------------------------------------

BENCHMARK_SUITES = ("general", "glm", "all")
BENCHMARK_CASES = (
    "all",
    "prefill_63ki",
    "prefill_3ki",
    "decode_63ki",
    "decode_3ki",
)


@dataclass(frozen=True)
class BenchmarkCase:
    suite: str
    name: str
    batch_size: int
    width: int
    group_topk: int
    pool_size: int = POOL_SIZE
    mode: str = "offset"
    length_pattern: str = "full"

    @property
    def token_topk(self) -> int:
        return self.group_topk * self.pool_size

    @property
    def has_tail(self) -> bool:
        return self.length_pattern.startswith("glm_")

    @property
    def output_cols(self) -> int:
        return self.token_topk + (self.pool_size - 1 if self.has_tail else 0)


@dataclass
class BenchmarkResult:
    device: str
    torch_version: str
    suite: str
    case: str
    batch_size: int
    width: int
    valid_length_min: int
    valid_length_max: int
    pool_size: int
    group_topk: int
    token_topk: int
    mode: str
    length_pattern: str
    valid_score_count: int
    scanned_score_count: int
    warmup: int
    iterations: int
    repeats: int
    median_us: float
    best_us: float
    worst_us: float
    mean_us: float
    stdev_us: float
    pool_scores_per_second: float
    rows_per_second: float
    minimum_global_bytes: int
    effective_bandwidth_gbps: float


def build_general_benchmark_cases() -> list[BenchmarkCase]:
    """Representative performance matrix for every public input dimension."""
    return [
        # Batch scaling at a fixed shape.
        BenchmarkCase("batch", "B1", 1, 8192, 128),
        BenchmarkCase("batch", "B32", 32, 8192, 128),
        BenchmarkCase("batch", "B512", 512, 8192, 128),
        # Sequence-width scaling and the K/K+1 selection boundary.
        BenchmarkCase("seq_len", "L128", 32, 128, 128),
        BenchmarkCase("seq_len", "L129", 32, 129, 128),
        BenchmarkCase("seq_len", "L512", 32, 512, 128),
        BenchmarkCase("seq_len", "L65536", 32, 65536, 128),
        # All representative output sizes; correctness covers every specialization.
        BenchmarkCase("group_topk", "gK256", 32, 8192, 256),
        BenchmarkCase("group_topk", "gK512", 32, 8192, 512),
        # pool_size changes output expansion while preserving group selection.
        BenchmarkCase("pool_size", "pool2", 32, 8192, 128, pool_size=2),
        BenchmarkCase("pool_size", "pool3", 32, 8192, 128, pool_size=3),
        BenchmarkCase("pool_size", "pool8", 32, 8192, 128, pool_size=8),
        # Public index-mapping modes.
        BenchmarkCase("mapping", "none", 32, 8192, 128, mode="none"),
        BenchmarkCase("mapping", "page_table", 32, 8192, 128, mode="page_table"),
        # Optimized long-row dispatch boundaries.
        BenchmarkCase("dispatch", "B31_L16126", 31, 16126, 512),
        BenchmarkCase("dispatch", "L16125", 32, 16125, 512),
        BenchmarkCase("dispatch", "L16126", 32, 16126, 512),
        BenchmarkCase("dispatch", "L16384", 32, 16384, 512),
        BenchmarkCase("dispatch", "L16385", 32, 16385, 512),
    ]


def build_glm_benchmark_cases(selected: str) -> list[BenchmarkCase]:
    cases = [
        BenchmarkCase(
            "glm", "prefill_63ki", 1024, 16126, 512,
            mode="offset", length_pattern="glm_prefill_63ki",
        ),
        BenchmarkCase(
            "glm", "prefill_3ki", 384, 766, 512,
            mode="offset", length_pattern="glm_prefill_3ki",
        ),
        BenchmarkCase(
            "glm", "decode_63ki", 2, 16383, 512,
            mode="page_table", length_pattern="glm_decode_63ki",
        ),
        BenchmarkCase(
            "glm", "decode_3ki", 2, 1023, 512,
            mode="page_table", length_pattern="glm_decode_3ki",
        ),
    ]
    return cases if selected == "all" else [c for c in cases if c.name == selected]


def build_benchmark_cases(suite: str, selected: str) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    if suite in ("general", "all"):
        cases.extend(build_general_benchmark_cases())
    if suite in ("glm", "all"):
        cases.extend(build_glm_benchmark_cases(selected))
    return cases


def benchmark_row_lengths(case: BenchmarkCase) -> list[int]:
    if case.length_pattern == "full":
        return [case.width] * case.batch_size
    if case.length_pattern == "glm_prefill_63ki":
        positions = range(63 * 1024 - GLM_PREFILL_CHUNK, 63 * 1024, GLM_CP_SIZE)
        return [(position + 1) // case.pool_size for position in positions]
    if case.length_pattern == "glm_prefill_3ki":
        positions = range(0, 3 * 1024, GLM_CP_SIZE)
        return [(position + 1) // case.pool_size for position in positions]
    if case.length_pattern == "glm_decode_63ki":
        return [63 * 1024 // case.pool_size, (64 * 1024 - 1) // case.pool_size]
    if case.length_pattern == "glm_decode_3ki":
        return [3 * 1024 // case.pool_size, (4 * 1024 - 1) // case.pool_size]
    raise ValueError(f"unsupported length pattern: {case.length_pattern}")


def benchmark_seq_lens(case: BenchmarkCase) -> Optional[list[int]]:
    if case.length_pattern == "glm_prefill_63ki":
        return list(range(
            63 * 1024 - GLM_PREFILL_CHUNK + 1,
            63 * 1024 + 1,
            GLM_CP_SIZE,
        ))
    if case.length_pattern == "glm_prefill_3ki":
        return list(range(1, 3 * 1024 + 1, GLM_CP_SIZE))
    if case.length_pattern == "glm_decode_63ki":
        return [63 * 1024, 64 * 1024 - 1]
    if case.length_pattern == "glm_decode_3ki":
        return [3 * 1024, 4 * 1024 - 1]
    return None


def minimum_global_bytes(case: BenchmarkCase, row_lengths: Sequence[int]) -> int:
    """Return minimum logical traffic; this is not a profiler DRAM counter."""
    score_bytes = sum(
        length for length in row_lengths if length > case.group_topk
    ) * 4
    metadata_bytes = case.batch_size * 4
    output_bytes = case.batch_size * case.output_cols * 4
    seq_lens = benchmark_seq_lens(case)
    if seq_lens is not None:
        metadata_bytes += case.batch_size * 4
    if case.mode == "offset":
        metadata_bytes += case.batch_size * 4
    elif case.mode == "page_table":
        metadata_bytes += sum(
            min(length * case.pool_size, case.token_topk)
            for length in row_lengths
        ) * 4
        if seq_lens is not None:
            metadata_bytes += sum(length % case.pool_size for length in seq_lens) * 4
    return score_bytes + metadata_bytes + output_bytes


def make_benchmark_launcher(
    case: BenchmarkCase,
) -> tuple[Callable[[], None], list[torch.Tensor], list[int]]:
    generator = torch.Generator(device="cuda").manual_seed(
        20260901 + case.batch_size + case.width + case.group_topk + case.pool_size
    )
    score = torch.randn(
        case.batch_size,
        case.width,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    row_lengths = benchmark_row_lengths(case)
    lengths = torch.tensor(row_lengths, dtype=torch.int32, device="cuda")
    seq_lens_values = benchmark_seq_lens(case)
    seq_lens = (
        torch.tensor(seq_lens_values, dtype=torch.int32, device="cuda")
        if seq_lens_values is not None else None
    )
    output = torch.empty(
        (case.batch_size, case.output_cols), dtype=torch.int32, device="cuda"
    )

    offsets = None
    page_table = None
    page_table_row_index = None
    keepalive = [score, lengths, output]
    if seq_lens is not None:
        keepalive.append(seq_lens)
    if case.mode == "offset":
        offsets = (
            torch.arange(case.batch_size, dtype=torch.int32, device="cuda")
            * case.width * case.pool_size
        )
        keepalive.append(offsets)
    elif case.mode == "page_table":
        table_width = case.width * case.pool_size + (
            case.pool_size if case.has_tail else 0
        )
        columns = torch.arange(table_width, dtype=torch.int32, device="cuda")
        row_offsets = (
            torch.arange(case.batch_size, dtype=torch.int32, device="cuda")
            * table_width
        )
        page_table = row_offsets[:, None] + columns[None, :]
        keepalive.append(page_table)
        if case.length_pattern.startswith("glm_decode_"):
            page_table_row_index = torch.arange(
                case.batch_size, dtype=torch.int32, device="cuda"
            )
            keepalive.append(page_table_row_index)
    elif case.mode != "none":
        raise ValueError(f"unsupported mode: {case.mode}")

    def launch() -> None:
        torch.ops.sgl_kernel.kpool_topk_transform.default(
            score,
            lengths,
            output,
            case.pool_size,
            page_table,
            offsets,
            None,
            seq_lens,
            page_table_row_index,
        )

    return launch, keepalive, row_lengths


def benchmark_case(
    case: BenchmarkCase, warmup: int, iterations: int, repeats: int
) -> BenchmarkResult:
    launch, keepalive, row_lengths = make_benchmark_launcher(case)
    launch()
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    samples_us: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            launch()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0 / iterations)

    median_us = statistics.median(samples_us)
    elapsed_seconds = median_us * 1.0e-6
    valid_score_count = sum(row_lengths)
    scanned_score_count = sum(
        length for length in row_lengths if length > case.group_topk
    )
    logical_bytes = minimum_global_bytes(case, row_lengths)
    result = BenchmarkResult(
        device=torch.cuda.get_device_name(0),
        torch_version=str(torch.__version__),
        suite=case.suite,
        case=case.name,
        batch_size=case.batch_size,
        width=case.width,
        valid_length_min=min(row_lengths),
        valid_length_max=max(row_lengths),
        pool_size=case.pool_size,
        group_topk=case.group_topk,
        token_topk=case.token_topk,
        mode=case.mode,
        length_pattern=case.length_pattern,
        valid_score_count=valid_score_count,
        scanned_score_count=scanned_score_count,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
        median_us=median_us,
        best_us=min(samples_us),
        worst_us=max(samples_us),
        mean_us=statistics.mean(samples_us),
        stdev_us=statistics.stdev(samples_us) if len(samples_us) > 1 else 0.0,
        pool_scores_per_second=valid_score_count / elapsed_seconds,
        rows_per_second=case.batch_size / elapsed_seconds,
        minimum_global_bytes=logical_bytes,
        effective_bandwidth_gbps=logical_bytes / elapsed_seconds / 1.0e9,
    )

    torch.cuda.synchronize()
    del launch, keepalive
    gc.collect()
    torch.cuda.empty_cache()
    return result


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_benchmark_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="run the CUDA-Event benchmark instead of pytest",
    )
    parser.add_argument(
        "--suite", choices=BENCHMARK_SUITES, default="glm",
        help="general matrix, four GLM workloads, or both (default: glm)",
    )
    parser.add_argument(
        "--case", choices=BENCHMARK_CASES, default="all",
        help="isolate one GLM workload; requires --suite glm",
    )
    parser.add_argument("--warmup", type=positive_int, default=5000)
    parser.add_argument("--iters", type=positive_int, default=1000)
    parser.add_argument("--repeats", type=positive_int, default=20)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()
    if not args.benchmark:
        parser.error(
            "direct execution requires --benchmark; use pytest for correctness"
        )
    if args.case != "all" and args.suite != "glm":
        parser.error("--case selects a GLM workload and requires --suite glm")
    return args


def format_valid_lengths(result: BenchmarkResult) -> str:
    if result.valid_length_min == result.valid_length_max:
        return str(result.valid_length_max)
    return f"{result.valid_length_min}..{result.valid_length_max}"


def print_benchmark_header() -> None:
    print(
        f"{'suite':<10} {'case':<14} {'B':>5} {'width':>7} {'valid_len':>13} "
        f"{'gK':>4} {'pS':>3} {'mode':<10} {'median(us)':>11} {'best(us)':>10} "
        f"{'stdev':>8} {'Gpool/s':>10} {'minGB/s':>9}"
    )
    print("-" * 132)


def print_benchmark_result(result: BenchmarkResult) -> None:
    print(
        f"{result.suite:<10} {result.case:<14} {result.batch_size:>5d} "
        f"{result.width:>7d} {format_valid_lengths(result):>13} "
        f"{result.group_topk:>4d} {result.pool_size:>3d} {result.mode:<10} "
        f"{result.median_us:>11.3f} {result.best_us:>10.3f} "
        f"{result.stdev_us:>8.3f} "
        f"{result.pool_scores_per_second / 1.0e9:>10.3f} "
        f"{result.effective_bandwidth_gbps:>9.3f}"
    )


@torch.inference_mode()
def benchmark_main() -> None:
    args = parse_benchmark_args()
    if not torch.cuda.is_available():
        raise RuntimeError("requires a MetaX CUDA-compatible GPU")
    if not hasattr(torch.ops.sgl_kernel, "kpool_topk_transform"):
        raise RuntimeError("kpool_topk_transform is not registered")

    cases = build_benchmark_cases(args.suite, args.case)
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}")
    print(
        f"benchmark_suite={args.suite} glm_case={args.case} cases={len(cases)} "
        f"warmup={args.warmup} iters={args.iters} repeats={args.repeats}"
    )
    print("timing: CUDA Events; minGB/s: minimum logical traffic / median time")
    print_benchmark_header()

    started = time.perf_counter()
    results: list[BenchmarkResult] = []
    for case in cases:
        result = benchmark_case(case, args.warmup, args.iters, args.repeats)
        results.append(result)
        print_benchmark_result(result)
    print("-" * 132)
    print(f"completed {len(results)} cases in {time.perf_counter() - started:.1f}s")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))
        print(f"csv: {args.csv}")


if __name__ == "__main__":
    benchmark_main()
