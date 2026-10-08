"""Correctness and validation tests for torch.ops._C.region_topk_ids."""

import re

import pytest
import torch

import mcoplib._C  # noqa: F401 - registers torch.ops._C operators


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="region_topk_ids requires a CUDA device"
)


def _reference(logits: torch.Tensor, lengths: torch.Tensor, topk: int) -> torch.Tensor:
    rows, width = logits.shape
    result = torch.full((rows, topk), -1, device=logits.device, dtype=torch.int32)
    for row in range(rows):
        visible = max(0, min(int(lengths[row].item()), width))
        budget = min(topk, visible)
        if budget:
            # Stable sorting defines the cut-tie rule: lower region ids win.
            result[row, :budget] = torch.argsort(
                -logits[row, :visible], stable=True
            )[:budget].to(torch.int32)
    return result


def _assert_selection(logits: torch.Tensor, lengths: torch.Tensor, topk: int) -> None:
    actual = torch.ops._C.region_topk_ids(logits, lengths, topk)
    expected = _reference(logits, lengths, topk)
    assert actual.dtype == torch.int32
    assert actual.shape == (logits.shape[0], topk)
    # The operator emits selected ids in logical-id order, not score order.
    torch.testing.assert_close(
        torch.sort(actual, dim=1).values,
        torch.sort(expected, dim=1).values,
        rtol=0,
        atol=0,
    )
    # Integer bookkeeping must be deterministic across launches.
    assert torch.equal(actual, torch.ops._C.region_topk_ids(logits, lengths, topk))


@pytest.mark.parametrize(
    "rows,width,topk",
    [
        (256, 256, 16),       # short-row exact-rank path
        (33, 257, 128),       # radix boundary
        (4, 513, 257),        # few-row 1024-thread dispatch
        (512, 4096, 512),     # production shape
    ],
)
def test_region_topk_ids_random(rows: int, width: int, topk: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(rows + width + topk)
    logits = torch.randn((rows, width), device="cuda", generator=generator)
    lengths = torch.randint(
        -8, width + 9, (rows,), dtype=torch.int32, device="cuda", generator=generator
    )
    _assert_selection(logits, lengths, topk)


def test_region_topk_ids_ties_lengths_and_strided_rows() -> None:
    rows, width, topk = 37, 769, 257
    storage = torch.randn((rows, width + 7), device="cuda", dtype=torch.float32)
    logits = storage[:, :width]
    logits[:, :300] = 0.5
    logits[:, width // 2 :] = -2.0
    lengths = torch.tensor(
        [-10, 0, 1, 255, 256, 257, width, width + 20] * 5,
        device="cuda",
        dtype=torch.int32,
    )[:rows]
    assert logits.stride(0) == width + 7
    _assert_selection(logits, lengths, topk)


def test_region_topk_ids_topk_exceeds_visible_and_signed_zero() -> None:
    logits = torch.tensor(
        [[3.0, 0.0, -0.0, 2.0, -1.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
        device="cuda",
        dtype=torch.float32,
    )
    lengths = torch.tensor([4, 3], device="cuda", dtype=torch.int32)
    _assert_selection(logits, lengths, topk=7)


def test_region_topk_ids_hidden_columns_and_all_negative() -> None:
    rows, width, visible, topk = 4, 2048, 1709, 512
    logits = torch.full(
        (rows, width), 1.0e6, device="cuda", dtype=torch.float32
    )
    logits[:, :visible] = -torch.rand(
        (rows, visible), device="cuda", dtype=torch.float32
    )
    lengths = torch.full((rows,), visible, device="cuda", dtype=torch.int32)
    actual = torch.ops._C.region_topk_ids(logits, lengths, topk)
    # Values beyond `visible` dominate every live score and must still be ignored.
    assert bool((actual < visible).all())
    _assert_selection(logits, lengths, topk)


def test_region_topk_ids_zero_topk() -> None:
    logits = torch.randn((3, 8), device="cuda", dtype=torch.float32)
    lengths = torch.tensor([0, 4, 8], device="cuda", dtype=torch.int32)
    output = torch.ops._C.region_topk_ids(logits, lengths, 0)
    assert output.shape == (3, 0)
    assert output.dtype == torch.int32


@pytest.mark.parametrize(
    "logits,lengths,topk,message",
    [
        (
            lambda: torch.randn((2, 8), device="cuda", dtype=torch.float16),
            lambda: torch.ones(2, device="cuda", dtype=torch.int32),
            2,
            "logits must have dtype float32",
        ),
        (
            lambda: torch.randn((2, 8), device="cuda", dtype=torch.float32),
            lambda: torch.ones(2, device="cuda", dtype=torch.int64),
            2,
            "lengths must have dtype int32",
        ),
        (
            lambda: torch.randn((2, 8), device="cuda", dtype=torch.float32)[:, ::2],
            lambda: torch.ones(2, device="cuda", dtype=torch.int32),
            2,
            "logits stride(1) must be 1",
        ),
        (
            lambda: torch.randn((2, 8), device="cuda", dtype=torch.float32),
            lambda: torch.ones(3, device="cuda", dtype=torch.int32),
            2,
            "lengths size must match logits rows",
        ),
    ],
)
def test_region_topk_ids_validation(logits, lengths, topk: int, message: str) -> None:
    with pytest.raises(RuntimeError, match=re.escape(message)):
        torch.ops._C.region_topk_ids(logits(), lengths(), topk)


def test_region_topk_ids_rejects_negative_topk() -> None:
    logits = torch.randn((2, 8), device="cuda", dtype=torch.float32)
    lengths = torch.ones(2, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="topk must be non-negative"):
        torch.ops._C.region_topk_ids(logits, lengths, -1)
