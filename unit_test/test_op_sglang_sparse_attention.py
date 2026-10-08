"""Tests for the native CUDA sparse-attention operator."""

from __future__ import annotations

import pathlib
import sysconfig

import pytest
import torch

LIBRARY = (
    pathlib.Path(__file__).resolve().parents[1]
    / "mcoplib"
    / ("sgl_kernel" + sysconfig.get_config_var("EXT_SUFFIX"))
)
torch.ops.load_library(str(LIBRARY))

NUM_HEADS = 64
HEAD_DIM = 512
TOPK = 2112
SM_SCALE = 0.0625

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native sparse attention requires a GPU"
)


def _make_inputs(q_len: int, valid_count: int):
    torch.manual_seed(17)
    q = torch.randn(
        (q_len, NUM_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16
    )
    kv = torch.randn((256, 1, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    kv.mul_(0.02)
    indices = torch.full(
        (q_len, 1, TOPK), -1, device="cuda", dtype=torch.int32
    )
    for row in range(q_len):
        count = min(valid_count + row, 256)
        indices[row, 0, :count] = (
            torch.arange(count, device="cuda") * 37 + row * 11
        ).remainder(kv.shape[0]).to(torch.int32)
    return q, kv, indices


def _reference(q, kv, indices):
    outputs = []
    q_cpu = q.cpu().double()
    kv_cpu = kv[:, 0].cpu().double()
    indices_cpu = indices[:, 0].cpu()
    for row in range(q.shape[0]):
        row_indices = indices_cpu[row]
        row_indices = row_indices[row_indices >= 0].long()
        selected = kv_cpu[row_indices]
        scores = q_cpu[row] @ selected.t() * SM_SCALE
        outputs.append(torch.softmax(scores, dim=-1) @ selected)
    return torch.stack(outputs)


def _reference_lse(q, kv, indices):
    q_cpu = q.cpu().double()
    kv_cpu = kv[:, 0].cpu().double()
    raw = indices[:, 0].cpu()
    rows = []
    for token in range(q.shape[0]):
        row_indices = raw[token][raw[token] >= 0].long()
        scores = q_cpu[token] @ kv_cpu[row_indices].t() * SM_SCALE
        rows.append(torch.logsumexp(scores, dim=-1))
    return torch.stack(rows)

def _call(q, kv, indices):
    return torch.ops.sgl_kernel.sparse_attention_fwd(
        q, kv, indices, SM_SCALE, HEAD_DIM, False
    )[0]


@requires_gpu
@pytest.mark.parametrize("q_len", (1, 6, 16))
def test_sparse_attention_prefix16_matches_reference(q_len: int) -> None:
    q, kv, indices = _make_inputs(q_len, 1)
    expected = _reference(q, kv, indices)
    actual = _call(q, kv, indices)
    torch.cuda.synchronize()

    assert actual.shape == q.shape
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual.cpu().double(), expected, atol=0.05, rtol=0.05)


@requires_gpu
def test_sparse_attention_general_decode_matches_reference() -> None:
    q, kv, indices = _make_inputs(1, 64)
    expected = _reference(q, kv, indices)
    actual = _call(q, kv, indices)
    torch.cuda.synchronize()

    assert torch.isfinite(actual).all()
    assert torch.allclose(actual.cpu().double(), expected, atol=0.05, rtol=0.05)


@requires_gpu
def test_sparse_attention_prefill_matches_uniform_attention() -> None:
    q, kv, indices = _make_inputs(129, 64)
    q.zero_()
    indices[:, :, 64:] = -1
    expected = kv[:64, 0].float().mean(dim=0)
    actual = _call(q, kv, indices)
    torch.cuda.synchronize()

    assert torch.isfinite(actual).all()
    assert torch.allclose(
        actual.float(), expected[None, None, :].expand_as(actual), atol=0.05, rtol=0.05
    )


@requires_gpu
def test_sparse_attention_rejects_wrong_index_dtype() -> None:
    q, kv, indices = _make_inputs(1, 4)
    with pytest.raises(RuntimeError, match="indices must be int32"):
        _call(q, kv, indices.to(torch.int64))


@requires_gpu
def test_sparse_attention_returns_lse() -> None:
    q, kv, indices = _make_inputs(1, 4)
    expected_lse = _reference_lse(q, kv, indices)
    actual, actual_lse = torch.ops.sgl_kernel.sparse_attention_fwd(
        q, kv, indices, SM_SCALE, HEAD_DIM, True
    )
    torch.cuda.synchronize()

    assert torch.isfinite(actual).all()
    assert torch.allclose(
        actual_lse.cpu().double(), expected_lse, atol=0.01, rtol=0.01
    )
