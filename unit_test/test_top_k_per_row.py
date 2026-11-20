# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import mcoplib._C


OP_NAME = "top_k_per_row_decode"
SORTING_THRESHOLD = 12288
SPLIT_WORK_THRESHOLD = 200_000


def _get_op():
    if not torch.cuda.is_available():
        pytest.skip("This test requires CUDA")
    if not hasattr(torch.ops._C, OP_NAME):
        pytest.skip(f"mcoplib operator {OP_NAME} is not registered")
    return getattr(torch.ops._C, OP_NAME)


def _make_logits(num_rows, vocab_size, seed=1234, non_contiguous=False):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if non_contiguous:
        logits = torch.randn((num_rows, vocab_size * 2), generator=generator, dtype=torch.float32).to("cuda")[:, ::2]
    else:
        logits = torch.randn((num_rows, vocab_size), generator=generator, dtype=torch.float32).to("cuda")
    return logits


def _make_seq_lens(batch_size, next_n, vocab_size, top_k=1, two_dimensional=False, seed=5678):
    if vocab_size < top_k:
        raise ValueError(f"vocab_size ({vocab_size}) must be >= top_k ({top_k})")
    seq_len = min(vocab_size, next_n + top_k - 1)
    if two_dimensional:
        return torch.full((batch_size, next_n), seq_len, dtype=torch.int32, device="cuda")
    return torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")


def _run_decode(logits, seq_lens, next_n, top_k):
    num_rows = logits.size(0)
    indices = torch.full((num_rows, top_k), -1, device=logits.device, dtype=torch.int32)
    _get_op()(logits, next_n, seq_lens, indices, num_rows, logits.stride(0), logits.stride(1), top_k)
    torch.cuda.synchronize()
    return indices


def _check_indices(logits, indices, top_k):
    assert indices.dtype == torch.int32
    assert indices.shape == (logits.size(0), top_k)
    assert torch.all(indices >= 0).item(), f"Found invalid indices: {indices}"
    assert torch.all(indices < logits.size(1)).item(), f"Found out-of-range indices: {indices}"
    if top_k > 1:
        sorted_indices = indices.sort(dim=1).values
        assert torch.all(sorted_indices[:, 1:] != sorted_indices[:, :-1]).item(), f"Duplicate indices: {indices}"


def _check_exact_topk(logits, indices, seq_lens, next_n, top_k):
    num_rows = logits.size(0)
    batch_idx = torch.arange(num_rows, device=logits.device) // next_n
    next_n_idx = torch.arange(num_rows, device=logits.device) % next_n

    if seq_lens.dim() == 2:
        row_ends = seq_lens.reshape(-1).to(torch.int64)
    else:
        row_ends = seq_lens[batch_idx].to(torch.int64) - next_n + next_n_idx + 1

    for row in range(num_rows):
        row_end = max(0, min(int(row_ends[row].item()), logits.size(1)))
        valid_len = row_end
        assert valid_len >= top_k, f"Row {row}: valid_len={valid_len}, top_k={top_k}"

        expected = torch.topk(logits[row, :row_end], top_k, largest=True, sorted=True).values
        actual = logits[row].gather(0, indices[row].to(torch.int64))
        actual = torch.sort(actual, descending=True).values

        assert torch.allclose(actual, expected, rtol=0, atol=0), (
            f"Top-K mismatch at row {row}: row_end={row_end}, top_k={top_k}, "
            f"actual={actual.tolist()}, expected={expected.tolist()}"
        )


@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("next_n", [1, 2, 4, 8])
@pytest.mark.parametrize("two_dimensional_seq_lens", [False, True])
@pytest.mark.parametrize("vocab_size", [1024, 4096, SORTING_THRESHOLD - 1])
@pytest.mark.parametrize("top_k", [1, 8, 32])
def test_top_k_per_row_decode_basic(batch_size, next_n, two_dimensional_seq_lens, vocab_size, top_k):
    logits = _make_logits(batch_size * next_n, vocab_size)
    seq_lens = _make_seq_lens(batch_size, next_n, vocab_size, top_k=top_k, two_dimensional=two_dimensional_seq_lens)
    indices = _run_decode(logits, seq_lens, next_n, top_k)
    _check_indices(logits, indices, top_k)
    _check_exact_topk(logits, indices, seq_lens, next_n, top_k)


@pytest.mark.parametrize("vocab_size", [SORTING_THRESHOLD - 1, SORTING_THRESHOLD, SORTING_THRESHOLD + 1, SPLIT_WORK_THRESHOLD - 1, SPLIT_WORK_THRESHOLD, SPLIT_WORK_THRESHOLD + 1])
@pytest.mark.parametrize("top_k", [1, 8, 32])
@pytest.mark.parametrize("next_n", [1, 2])
def test_top_k_per_row_decode_algorithm_boundaries(vocab_size, top_k, next_n):
    batch_size = 2
    logits = _make_logits(batch_size * next_n, vocab_size, seed=42)
    seq_lens = _make_seq_lens(batch_size, next_n, vocab_size, top_k=top_k, two_dimensional=False)
    indices = _run_decode(logits, seq_lens, next_n, top_k)
    _check_indices(logits, indices, top_k)
    _check_exact_topk(logits, indices, seq_lens, next_n, top_k)


@pytest.mark.parametrize("two_dimensional_seq_lens", [False, True])
@pytest.mark.parametrize("next_n", [1, 2, 4, 8])
@pytest.mark.parametrize("top_k", [1, 8, 32])
def test_top_k_per_row_decode_non_contiguous_logits(two_dimensional_seq_lens, next_n, top_k):
    batch_size = 2
    vocab_size = 4096
    logits = _make_logits(batch_size * next_n, vocab_size, seed=9876, non_contiguous=True)
    seq_lens = _make_seq_lens(batch_size, next_n, vocab_size, top_k=top_k, two_dimensional=two_dimensional_seq_lens)
    indices = _run_decode(logits, seq_lens, next_n, top_k)
    _check_indices(logits, indices, top_k)
    _check_exact_topk(logits, indices, seq_lens, next_n, top_k)


@pytest.mark.parametrize("top_k", [8, 32])
@pytest.mark.parametrize("next_n", [1, 2, 4])
def test_top_k_per_row_decode_ties(top_k, next_n):
    batch_size = 2
    vocab_size = 4096
    logits = torch.zeros((batch_size * next_n, vocab_size), device="cuda", dtype=torch.float32)
    logits[:, :top_k * 2] = 1.0
    seq_lens = _make_seq_lens(batch_size, next_n, vocab_size, top_k=top_k, two_dimensional=False)
    indices = _run_decode(logits, seq_lens, next_n, top_k)
    _check_indices(logits, indices, top_k)
    selected_values = logits.gather(1, indices.to(torch.int64))
    assert torch.all(selected_values == 1.0).item(), f"Selected values contain unexpected entries: {selected_values}"


@pytest.mark.parametrize("next_n", [1, 2, 4])
def test_top_k_per_row_decode_repeatability(next_n):
    batch_size = 2
    vocab_size = 4096
    top_k = 32
    logits = _make_logits(batch_size * next_n, vocab_size, seed=2026)
    seq_lens = _make_seq_lens(batch_size, next_n, vocab_size, top_k=top_k, two_dimensional=False)
    first = _run_decode(logits, seq_lens, next_n, top_k)
    second = _run_decode(logits, seq_lens, next_n, top_k)
    assert torch.equal(first, second), f"Repeated calls produced different results.\nFirst:\n{first}\nSecond:\n{second}"