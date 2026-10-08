"""Accuracy, dispatch, and resource tests for TileLang sparse attention.

Run the default decode-focused suite on a GPU machine with::

    pytest -q -s unit_test/test_tilelang_sparse_attention.py

Run the larger prefill smoke case explicitly with::

    pytest -q -s -m full unit_test/test_tilelang_sparse_attention.py

The tests intentionally use a small KV cache and padded sparse indices.  They
exercise the same H64/D512/TopK2112 kernel contract without depending on large
external dumps or allocating the production-size KV cache.
"""

from __future__ import annotations

import os
import statistics

import pytest
import torch


pytest.importorskip("tilelang")

from mcoplib import tilelang_sparse_attention as sparse_attention


NUM_HEADS = 64
HEAD_DIM = 512
TOPK = 2112
KV_LEN = 4096
SM_SCALE = 0.0625
COSINE_THRESHOLD = 0.99999

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="TileLang sparse attention requires a GPU"
)


def _make_inputs(
    q_len: int,
    pattern: str,
    valid_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 1 <= valid_count <= TOPK:
        raise ValueError("valid_count must be in [1, TOPK]")

    device = torch.device("cuda")
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)

    q = torch.randn(
        (q_len, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    kv = torch.empty(
        (KV_LEN, 1, HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    kv.normal_(mean=0.0, std=0.02)

    base = (torch.arange(valid_count, dtype=torch.int64) * 631) % KV_LEN
    if pattern == "shared":
        valid_indices = base.unsqueeze(0).expand(q_len, valid_count)
    elif pattern == "disjoint":
        offsets = torch.arange(q_len, dtype=torch.int64).unsqueeze(1) * 1049
        valid_indices = (base.unsqueeze(0) + offsets) % KV_LEN
    else:
        raise ValueError(f"unsupported pattern: {pattern}")

    indices = torch.full((q_len, TOPK), -1, dtype=torch.int32)
    indices[:, :valid_count] = valid_indices.to(torch.int32)
    return q, kv, indices.unsqueeze(1).to(device)


def _reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """CPU FP64 reference that computes only valid sparse entries."""

    q_cpu = q.cpu().double()
    kv_cpu = kv.cpu().double()
    indices_cpu = indices[:, 0, :].cpu().long()
    output = torch.empty_like(q_cpu)

    for token in range(q_cpu.shape[0]):
        token_indices = indices_cpu[token]
        token_indices = token_indices[token_indices >= 0]
        selected_kv = kv_cpu[token_indices, 0, :]
        scores = torch.matmul(q_cpu[token], selected_kv.transpose(0, 1))
        probabilities = torch.softmax(scores * SM_SCALE, dim=-1)
        output[token] = torch.matmul(probabilities, selected_kv)
    return output


def _normalize_output(output: torch.Tensor) -> torch.Tensor:
    if output.ndim == 4 and output.shape[0] == 1:
        return output.squeeze(0)
    return output


def _cosine_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, float]:
    actual = actual.cpu().double()
    expected = expected.cpu().double()
    global_cosine = torch.nn.functional.cosine_similarity(
        actual.reshape(1, -1), expected.reshape(1, -1), dim=1
    ).item()
    row_cosines = torch.nn.functional.cosine_similarity(
        actual.reshape(-1, HEAD_DIM), expected.reshape(-1, HEAD_DIM), dim=1
    )
    return global_cosine, row_cosines.min().item()


def _benchmark_us(call, warmup: int, samples: int) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            call()
        torch.cuda.synchronize()

        timings = []
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)) * 1000.0)

    ordered = sorted(timings)
    p90_index = min(len(ordered) - 1, int(0.9 * len(ordered)))
    return {
        "mean_us": statistics.fmean(timings),
        "p50_us": statistics.median(timings),
        "p90_us": ordered[p90_index],
        "min_us": min(timings),
        "max_us": max(timings),
    }


def _assert_optional_latency_limit(stats: dict[str, float], env_name: str) -> None:
    configured = os.getenv(env_name)
    if configured is not None:
        assert stats["p50_us"] <= float(configured), (
            f"p50 {stats['p50_us']:.3f} us exceeds {env_name}={configured} us"
        )


@requires_gpu
@pytest.mark.parametrize(
    ("q_len", "pattern"),
    ((1, "shared"), (6, "shared"), (6, "disjoint")),
)
def test_public_decode_matches_fp64_reference(q_len: int, pattern: str) -> None:
    q, kv, indices = _make_inputs(q_len, pattern, valid_count=64)

    with torch.inference_mode():
        actual = sparse_attention.tilelang_sparse_fwd(
            q, kv, indices, SM_SCALE, d_v=HEAD_DIM, return_lse=False
        )
        torch.cuda.synchronize()
        expected = _reference(q, kv, indices)

    actual = _normalize_output(actual)
    global_cosine, row_min_cosine = _cosine_metrics(actual, expected)

    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert global_cosine >= COSINE_THRESHOLD
    assert row_min_cosine >= COSINE_THRESHOLD
    assert torch.allclose(actual.cpu().double(), expected, atol=0.05, rtol=0.05)


@requires_gpu
def test_pipeline_stage_matches_device_shared_memory() -> None:
    selected_stages, reported_limit = (
        sparse_attention._select_splitkv_num_stages()
    )
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    expected_limit = int(properties.shared_memory_per_block)
    expected_stages = (
        2
        if expected_limit
        >= sparse_attention._SPLITKV_STAGE2_SHARED_MEMORY_BYTES
        else 1
    )

    assert reported_limit == expected_limit
    assert selected_stages == expected_stages


def test_public_entry_dispatches_decode_and_prefill(monkeypatch) -> None:
    calls: list[str] = []

    def fake_factory(mode: str):
        def factory(num_heads, dim, tail_dim, topk, **kwargs):
            del num_heads, dim, tail_dim, topk, kwargs
            calls.append(mode)

            def kernel(q, kv, indices, lse):
                del kv, indices, lse
                return torch.zeros_like(q)

            return kernel

        return factory

    monkeypatch.setattr(sparse_attention, "_is_hip", False)
    monkeypatch.setattr(
        sparse_attention,
        "sparse_attention_fwd_kernel_v1_splitkv_decode",
        fake_factory("decode"),
    )
    monkeypatch.setattr(
        sparse_attention,
        "sparse_attention_fwd_kernel_v1_splitkv_prefill",
        fake_factory("prefill"),
    )

    for q_len, expected_mode in ((128, "decode"), (129, "prefill")):
        q = torch.empty((q_len, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16)
        kv = torch.empty((1, 1, HEAD_DIM), dtype=torch.bfloat16)
        indices = torch.zeros((q_len, 1, TOPK), dtype=torch.int32)
        output = sparse_attention.tilelang_sparse_fwd(
            q, kv, indices, SM_SCALE, d_v=HEAD_DIM, return_lse=False
        )
        assert output.shape == (1, q_len, NUM_HEADS, HEAD_DIM)
        assert calls[-1] == expected_mode


@requires_gpu
@pytest.mark.full
@pytest.mark.parametrize("pattern", ("shared", "disjoint"))
def test_public_decode_performance(pattern: str) -> None:
    q, kv, indices = _make_inputs(6, pattern, valid_count=TOPK)

    def call():
        return sparse_attention.tilelang_sparse_fwd(
            q, kv, indices, SM_SCALE, d_v=HEAD_DIM, return_lse=False
        )

    stats = _benchmark_us(call, warmup=20, samples=30)
    print(f"decode q_len=6 pattern={pattern}: {stats}")
    assert stats["min_us"] > 0.0
    _assert_optional_latency_limit(stats, "SPARSE_ATTN_DECODE_MAX_US")


@requires_gpu
@pytest.mark.full
def test_public_prefill_q1024_smoke() -> None:
    q, kv, indices = _make_inputs(1024, "disjoint", valid_count=8)

    with torch.inference_mode():
        output = sparse_attention.tilelang_sparse_fwd(
            q, kv, indices, SM_SCALE, d_v=HEAD_DIM, return_lse=False
        )
        torch.cuda.synchronize()

    output = _normalize_output(output)
    assert output.shape == q.shape
    assert torch.isfinite(output).all()


@requires_gpu
@pytest.mark.full
def test_public_prefill_q1024_performance() -> None:
    q, kv, indices = _make_inputs(1024, "disjoint", valid_count=TOPK)

    def call():
        return sparse_attention.tilelang_sparse_fwd(
            q, kv, indices, SM_SCALE, d_v=HEAD_DIM, return_lse=False
        )

    stats = _benchmark_us(call, warmup=5, samples=10)
    print(f"prefill q_len=1024 pattern=disjoint: {stats}")
    assert stats["min_us"] > 0.0
    _assert_optional_latency_limit(stats, "SPARSE_ATTN_PREFILL_MAX_US")
