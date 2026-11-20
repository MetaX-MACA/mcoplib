"""Correctness and CUDA Graph tests for torch.ops._C.prepare_topk_scores."""

import re

import pytest
import torch

import mcoplib._C  # noqa: F401 - registers torch.ops._C operators


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="prepare_topk_scores requires a CUDA device",
)

BLOCK_N = 256
SENTINEL = 7.0
REGIONS = 1024 * 1024 // 8

LIVE_TOKENS = (
    1,
    9,
    25,
    137,
    4089,
    8009,
    32761,
    32777,
    65529,
    262137,
    524281,
    786425,
    800009,
    1048569,
)


def _op():
    return torch.ops._C.prepare_topk_scores


def _golden(scores, valid, starts, ends, padded_regions):
    rows, num_regions = scores.shape
    cols = torch.arange(padded_regions, device=scores.device)
    visible = (
        (cols < num_regions)
        & (cols >= starts.to(torch.int64).unsqueeze(1))
        & (cols < ends.to(torch.int64).unsqueeze(1))
    )
    padded_valid = torch.zeros(
        (rows, padded_regions), dtype=torch.bool, device=scores.device
    )
    padded_values = torch.full(
        (rows, padded_regions), -float("inf"),
        dtype=torch.float32, device=scores.device
    )
    if num_regions:
        padded_valid[:, :num_regions] = valid
        padded_values[:, :num_regions] = scores
    return torch.where(visible & padded_valid, padded_values, -float("inf"))


def _active_columns(starts, ends, padded_regions):
    cols = torch.arange(padded_regions, device=starts.device)
    block_start = (cols // BLOCK_N) * BLOCK_N
    return (
        (block_start < ends.to(torch.int64).unsqueeze(1))
        & (block_start + BLOCK_N > starts.to(torch.int64).unsqueeze(1))
    )


def _make_inputs(rows, num_regions, starts, ends, holes=True):
    generator = torch.Generator(device="cpu").manual_seed(
        rows * 17 + num_regions + starts[0] * 31 + ends[0]
    )
    scores = torch.randn(
        (rows, num_regions), dtype=torch.float32, generator=generator
    ).cuda()
    valid = torch.ones((rows, num_regions), dtype=torch.bool)
    if holes and num_regions:
        valid[:, torch.arange(num_regions) % 17 == 0] = False
    return (
        scores,
        valid.cuda(),
        torch.tensor(starts, dtype=torch.int32, device="cuda"),
        torch.tensor(ends, dtype=torch.int32, device="cuda"),
    )


def _check_case(name, scores, valid, starts, ends, padded_regions):
    output = torch.full(
        (scores.shape[0], padded_regions),
        SENTINEL,
        dtype=torch.float32,
        device=scores.device,
    )
    _op()(
        scores, valid, starts, ends, output,
        scores.shape[1], padded_regions, True,
    )
    torch.cuda.synchronize()

    expected = _golden(scores, valid, starts, ends, padded_regions)
    active = _active_columns(starts, ends, padded_regions)
    assert torch.equal(output[active], expected[active]), name
    assert bool((output[~active] == SENTINEL).all()), name


@pytest.mark.parametrize("tokens", LIVE_TOKENS)
def test_prepare_topk_scores_live_windows(tokens):
    rows = 8
    end = min(REGIONS, tokens // 8)
    scores, valid, starts, ends = _make_inputs(
        rows, REGIONS, [0] * rows, [end] * rows, holes=False
    )
    _check_case(f"live={tokens}", scores, valid, starts, ends, REGIONS)


def test_prepare_topk_scores_mixed_rows_and_holes():
    starts = [0, 1, 10, 100, 0, 50, 131070, 0]
    ends = [1, 3, 4095, 100, 32768, 100001, 131071, 0]
    scores, valid, starts_t, ends_t = _make_inputs(
        8, REGIONS, starts, ends, holes=True
    )
    _check_case("mixed", scores, valid, starts_t, ends_t, REGIONS)


@pytest.mark.parametrize(
    "rows,num_regions,padded,starts,ends",
    [
        (3, 300, 400, [0, 100, 0], [100, 100, 0]),
        (4, 7, 7, [0, 1, 3, 7], [1, 3, 7, 7]),
        (2, 5, 9, [0, 2], [5, 2]),
        (2, 512, 512, [256, 0], [256, 0]),
    ],
)
def test_prepare_topk_scores_boundaries(
    rows, num_regions, padded, starts, ends
):
    scores, valid, starts_t, ends_t = _make_inputs(
        rows, num_regions, starts, ends, holes=True
    )
    _check_case(
        f"{num_regions}/{padded}",
        scores, valid, starts_t, ends_t, padded,
    )


@pytest.mark.parametrize(
    "num_regions,padded",
    [(5, 9), (300, 400), (512, 512)],
)
def test_prepare_topk_scores_without_summary_valid(num_regions, padded):
    rows = 3
    scores, valid, starts, ends = _make_inputs(
        rows, num_regions, [4, 2, 0], [4, 2, 0], holes=True
    )
    output = torch.full(
        (rows, padded), SENTINEL, dtype=torch.float32, device="cuda"
    )
    _op()(
        scores, valid, starts, ends, output,
        num_regions, padded, False,
    )
    expected = torch.full_like(output, -float("inf"))
    expected[:, :num_regions] = scores
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_prepare_topk_scores_cuda_graph_replay():
    rows, num_regions, padded = 8, 4096, 4096
    scores, valid, starts, ends = _make_inputs(
        rows, num_regions, [0] * rows, [num_regions] * rows, holes=True
    )
    output = torch.full(
        (rows, padded), SENTINEL, dtype=torch.float32, device="cuda"
    )

    def launch():
        _op()(
            scores, valid, starts, ends, output,
            num_regions, padded, True,
        )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            launch()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    for end in (0, 1, 257, num_regions):
        ends.fill_(end)
        output.fill_(SENTINEL)
        graph.replay()
        torch.cuda.synchronize()
        expected = _golden(scores, valid, starts, ends, padded)
        active = _active_columns(starts, ends, padded)
        assert torch.equal(output[active], expected[active])
        assert bool((output[~active] == SENTINEL).all())


@pytest.mark.parametrize(
    "make_args,message",
    [
        (
            lambda: (
                torch.randn((2, 8), device="cuda", dtype=torch.float16),
                torch.ones((2, 8), device="cuda", dtype=torch.bool),
                torch.zeros(2, device="cuda", dtype=torch.int32),
                torch.full((2,), 8, device="cuda", dtype=torch.int32),
                torch.empty((2, 8), device="cuda", dtype=torch.float32),
                8, 8, True,
            ),
            "scores_in must have dtype float32",
        ),
        (
            lambda: (
                torch.randn((2, 8), device="cuda", dtype=torch.float32),
                torch.ones((2, 8), device="cuda", dtype=torch.uint8),
                torch.zeros(2, device="cuda", dtype=torch.int32),
                torch.full((2,), 8, device="cuda", dtype=torch.int32),
                torch.empty((2, 8), device="cuda", dtype=torch.float32),
                8, 8, True,
            ),
            "summary_valid must have dtype bool",
        ),
        (
            lambda: (
                torch.randn((2, 8), device="cuda", dtype=torch.float32),
                torch.ones((2, 8), device="cuda", dtype=torch.bool),
                torch.zeros(2, device="cuda", dtype=torch.int64),
                torch.full((2,), 8, device="cuda", dtype=torch.int32),
                torch.empty((2, 8), device="cuda", dtype=torch.float32),
                8, 8, True,
            ),
            "row_starts must have dtype int32",
        ),
        (
            lambda: (
                torch.randn((2, 8), device="cuda", dtype=torch.float32),
                torch.ones((2, 8), device="cuda", dtype=torch.bool),
                torch.zeros(2, device="cuda", dtype=torch.int32),
                torch.full((2,), 8, device="cuda", dtype=torch.int32),
                torch.empty((2, 7), device="cuda", dtype=torch.float32),
                8, 7, True,
            ),
            "padded_regions must be at least num_regions",
        ),
    ],
)
def test_prepare_topk_scores_validation(make_args, message):
    with pytest.raises(RuntimeError, match=re.escape(message)):
        _op()(*make_args())
