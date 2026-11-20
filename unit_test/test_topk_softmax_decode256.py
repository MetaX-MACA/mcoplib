# SPDX-License-Identifier: Apache-2.0
"""Correctness coverage for the E=256, K=8 decode topk-softmax path.

Run with:
    python unit_test/test_topk_softmax_decode256.py

The optimized path is selected by mcoplib for T < 1024.  T=1024 is also
covered as a dispatch-boundary regression case, but uses the existing fallback.
"""

import itertools

import torch

import mcoplib._moe_C  # noqa: F401: registers torch.ops._moe_C.topk_softmax


E = 256
K = 8
TOKEN_COUNTS = (1, 8, 32, 64, 87, 88, 89, 127, 128, 256, 512, 768, 1023, 1024)
DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def mcoplib_topk_softmax(logits, renormalize=False, bias=None, is_padding=None):
    num_tokens = logits.shape[0]
    weights = torch.empty((num_tokens, K), device=logits.device, dtype=torch.float32)
    indices = torch.empty((num_tokens, K), device=logits.device, dtype=torch.int32)
    source_rows = torch.empty((num_tokens, K), device=logits.device, dtype=torch.int32)
    torch.ops._moe_C.topk_softmax(
        weights, indices, source_rows, logits, renormalize, bias, is_padding
    )
    return weights, indices, source_rows


def reference(logits, renormalize=False, bias=None, is_padding=None):
    probs = torch.softmax(logits.float(), dim=-1)
    scores = probs if bias is None else probs + bias.float().unsqueeze(0)

    # Stable descending sort preserves ascending expert ID for equal scores.
    indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :K]
    weights = torch.gather(probs, dim=-1, index=indices)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)

    num_tokens = logits.shape[0]
    ranks = torch.arange(K, device=logits.device, dtype=torch.int32)
    rows = torch.arange(num_tokens, device=logits.device, dtype=torch.int32)
    source_rows = rows[:, None] + ranks[None, :] * num_tokens

    indices = indices.to(torch.int32)
    if is_padding is not None:
        weights = weights.masked_fill(is_padding[:, None], 0.0)
        indices = indices.masked_fill(is_padding[:, None], E)
    return weights, indices, source_rows


def assert_case(tokens, dtype, use_bias, renormalize, use_padding, seed=20261027):
    torch.manual_seed(seed + tokens)
    logits = torch.randn((tokens, E), device="cuda", dtype=dtype)
    bias = None
    if use_bias:
        bias = torch.randn((E,), device="cuda", dtype=torch.float32) * 0.02
    is_padding = None
    if use_padding:
        is_padding = torch.zeros((tokens,), device="cuda", dtype=torch.bool)
        is_padding[::7] = True

    actual = mcoplib_topk_softmax(logits, renormalize, bias, is_padding)
    expected = reference(logits, renormalize, bias, is_padding)
    torch.cuda.synchronize()

    atol = 5.0e-4 if dtype == torch.float32 else 3.0e-3
    rtol = 5.0e-4 if dtype == torch.float32 else 3.0e-3
    torch.testing.assert_close(actual[0], expected[0], atol=atol, rtol=rtol)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
    torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)


def test_random_matrix():
    modes = (
        (False, False, False),
        (True, False, False),
        (False, True, True),
        (True, True, True),
    )
    for tokens, dtype, (use_bias, renormalize, use_padding) in itertools.product(
        TOKEN_COUNTS, DTYPES, modes
    ):
        assert_case(tokens, dtype, use_bias, renormalize, use_padding)


def test_equal_scores_are_ordered_by_expert_id():
    tokens = 89  # exercises the 32-lane subwarp path just above its threshold
    logits = torch.zeros((tokens, E), device="cuda", dtype=torch.bfloat16)
    expected_indices = torch.arange(K, device="cuda", dtype=torch.int32).expand(tokens, K)
    expected_weights = torch.full(
        (tokens, K), 1.0 / E, device="cuda", dtype=torch.float32
    )

    first = None
    for _ in range(5):
        weights, indices, source_rows = mcoplib_topk_softmax(logits)
        torch.cuda.synchronize()
        torch.testing.assert_close(indices, expected_indices, atol=0, rtol=0)
        torch.testing.assert_close(weights, expected_weights, atol=1.0e-7, rtol=0)
        _, _, expected_source_rows = reference(logits)
        torch.testing.assert_close(source_rows, expected_source_rows, atol=0, rtol=0)
        if first is not None:
            torch.testing.assert_close(indices, first, atol=0, rtol=0)
        first = indices.clone()


def test_all_padding():
    tokens = 128
    logits = torch.randn((tokens, E), device="cuda", dtype=torch.bfloat16)
    is_padding = torch.ones((tokens,), device="cuda", dtype=torch.bool)
    weights, indices, source_rows = mcoplib_topk_softmax(
        logits, renormalize=True, is_padding=is_padding
    )
    _, _, expected_source_rows = reference(logits, True, None, is_padding)
    torch.cuda.synchronize()
    assert torch.count_nonzero(weights).item() == 0
    assert torch.all(indices == E).item()
    torch.testing.assert_close(source_rows, expected_source_rows, atol=0, rtol=0)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-compatible MetaX device is required")
    test_random_matrix()
    test_equal_scores_are_ordered_by_expert_id()
    test_all_padding()
    print("PASS: E=256 K=8 decode topk-softmax correctness")


if __name__ == "__main__":
    main()
