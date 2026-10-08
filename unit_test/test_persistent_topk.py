# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for persistent topk."""

import time

import torch
import numpy as np
import mcoplib
import mcoplib._C
import pytest


def top_k_per_row_decode_numpy(logits, seq_lens, topk_tokens):
    if seq_lens.ndim > 1:
        seq_lens = seq_lens.ravel()

    num_rows = logits.shape[0]
    out = np.full((num_rows, topk_tokens), -1, dtype=np.int64)

    for i in range(num_rows):
        length = int(seq_lens[i])
        if length <= 0:
            continue

        k = min(topk_tokens, length)
        row = logits[i, :length].astype(np.float32)
        idx = np.argpartition(-row, k - 1)[:k]
        vals = row[idx]
        sort_order = np.argsort(-vals)
        idx = idx[sort_order]
        out[i, :k] = idx.astype(np.int64)

    return out


def make_logits_with_stride(num_rows, length, stride, seed=42):
    assert stride >= length

    np.random.seed(seed)

    logits_mod = np.random.randn(num_rows, length).astype(np.float32)
    storage_size = (num_rows - 1) * stride + length

    base = torch.empty(storage_size, device="cuda", dtype=torch.float32)

    logits = torch.as_strided(base, size=(num_rows, length), stride=(stride, 1))

    logits.copy_(torch.tensor(logits_mod, device="cuda", dtype=torch.float32))

    return logits_mod, logits


def check_case(length, topk, bs=1, spec=0, seed=42):
    np.random.seed(seed)

    next_n = 1 + spec
    num_rows = bs * next_n

    logits_mod = np.random.randn(num_rows, length).astype(np.float32)

    seq_lens_mod = np.full((bs, next_n), length, dtype=np.int32)

    ref = top_k_per_row_decode_numpy(logits_mod, seq_lens_mod, topk)

    logits = torch.tensor(logits_mod, device="cuda")
    seq_lens = torch.tensor(seq_lens_mod, device="cuda")

    workspace = torch.zeros(num_rows * 772 * 4, device="cuda", dtype=torch.uint8)
    output = torch.zeros(num_rows, topk, device="cuda", dtype=torch.int32)

    torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, length)

    torch.cuda.synchronize()

    gpu = np.sort(output.cpu().numpy(), axis=-1)
    cpu = np.sort(ref, axis=-1)

    assert np.array_equal(gpu, cpu), f"persistent_topk failed: {length=}, {topk=}, {bs=}, {spec=}"


@pytest.mark.parametrize("length", [512, 513, 8191, 8192, 8193, 16387, 32767, 32768, 32769])
@pytest.mark.parametrize("topk", [512])
def test_persistent_topk_k_boundary(length, topk):
    check_case(length, topk)


@pytest.mark.parametrize("length", [1024, 1025, 8191, 8192, 8193, 16387, 32768, 32769])
@pytest.mark.parametrize("topk", [1024])
def test_persistent_topk_k_boundary_1024(length, topk):
    check_case(length, topk)


@pytest.mark.parametrize("length", [2048, 2049, 8191, 8192, 8193, 16387, 32768, 32769])
@pytest.mark.parametrize("topk", [2048])
def test_persistent_topk_k_boundary_2048(length, topk):
    check_case(length, topk)


@pytest.mark.parametrize("bs", [1, 2, 16])
@pytest.mark.parametrize("spec", [0, 1, 3])
@pytest.mark.parametrize("length", [8193, 16387])
@pytest.mark.parametrize("topk", [512, 1024])
def test_persistent_topk_small_medium(bs, spec, length, topk):
    check_case(length, topk, bs=bs, spec=spec)


@pytest.mark.parametrize("num_rows", [1, 4, 5, 8, 9, 16, 32])
@pytest.mark.parametrize("topk", [512])
def test_persistent_topk_num_rows_smem_branches(num_rows, topk):
    length = 8193

    np.random.seed(1000 + num_rows)

    logits_mod = np.random.randn(num_rows, length).astype(np.float32)
    seq_lens_mod = np.random.randint(1, length + 1, size=(num_rows,)).astype(np.int32)

    logits = torch.tensor(logits_mod, device="cuda")
    seq_lens = torch.tensor(seq_lens_mod, device="cuda", dtype=torch.int32)

    ref = top_k_per_row_decode_numpy(logits_mod, seq_lens_mod, topk)

    workspace = torch.zeros(num_rows * 772 * 4, device="cuda", dtype=torch.uint8)
    output = torch.zeros(num_rows, topk, device="cuda", dtype=torch.int32)

    torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, length)

    torch.cuda.synchronize()

    gpu = np.sort(output.cpu().numpy(), axis=-1)
    cpu = np.sort(ref, axis=-1)

    assert np.array_equal(gpu, cpu), f"num_rows smem branch failed: {num_rows=}, {topk=}"


@pytest.mark.parametrize("num_rows", [33, 64, 128])
@pytest.mark.parametrize("length", [8193, 16387, 32769])
@pytest.mark.parametrize("topk", [512, 1024])
def test_persistent_topk_filtered_topk_rows(num_rows, length, topk):
    np.random.seed(2000 + num_rows + length + topk)

    logits_mod = np.random.randn(num_rows, length).astype(np.float32)
    seq_lens_mod = np.random.randint(length // 2, length + 1, size=(num_rows,)).astype(np.int32)

    logits = torch.tensor(logits_mod, device="cuda")
    seq_lens = torch.tensor(seq_lens_mod, device="cuda", dtype=torch.int32)

    ref = top_k_per_row_decode_numpy(logits_mod, seq_lens_mod, topk)

    workspace = torch.zeros(num_rows * 772 * 4, device="cuda", dtype=torch.uint8)
    output = torch.zeros(num_rows, topk, device="cuda", dtype=torch.int32)

    torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, length)

    torch.cuda.synchronize()

    gpu = np.sort(output.cpu().numpy(), axis=-1)
    cpu = np.sort(ref, axis=-1)

    assert np.array_equal(gpu, cpu), f"filtered_topk branch failed: {num_rows=}, {length=}, {topk=}"


@pytest.mark.parametrize("stride_kind", ["vec4", "vec2", "vec1"])
@pytest.mark.parametrize("num_rows", [1, 8, 32])
def test_persistent_topk_vectorization_paths(stride_kind, num_rows):
    length = 8193
    topk = 512

    if stride_kind == "vec4":
        stride = ((length + 3) // 4) * 4
        assert stride % 4 == 0
    elif stride_kind == "vec2":
        stride = length + 1
        if stride % 4 == 0:
            stride += 2
        if stride % 2 != 0:
            stride += 1
        assert stride % 4 != 0
        assert stride % 2 == 0
    else:
        stride = length
        if stride % 2 == 0:
            stride += 1
        assert stride % 2 != 0

    logits_mod, logits = make_logits_with_stride(num_rows, length, stride, seed=3000 + num_rows)

    assert logits.shape == (num_rows, length)
    assert logits.stride(1) == 1
    assert logits.stride(0) == stride

    seq_lens_mod = np.full(num_rows, length, dtype=np.int32)
    seq_lens = torch.tensor(seq_lens_mod, device="cuda", dtype=torch.int32)

    ref = top_k_per_row_decode_numpy(logits_mod, seq_lens_mod, topk)

    workspace = torch.zeros(num_rows * 772 * 4, device="cuda", dtype=torch.uint8)
    output = torch.zeros(num_rows, topk, device="cuda", dtype=torch.int32)

    torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, length)

    torch.cuda.synchronize()

    gpu = np.sort(output.cpu().numpy(), axis=-1)
    cpu = np.sort(ref, axis=-1)

    assert np.array_equal(gpu, cpu), f"vectorization path failed: {stride_kind=}, {num_rows=}, {stride=}"


@pytest.mark.parametrize("stride_kind", ["vec4", "vec2", "vec1"])
def test_persistent_topk_vectorization_with_ragged_lengths(stride_kind):
    length = 16387
    topk = 512
    num_rows = 8

    if stride_kind == "vec4":
        stride = ((length + 3) // 4) * 4
    elif stride_kind == "vec2":
        stride = length + 1
        if stride % 4 == 0:
            stride += 2
        if stride % 2 != 0:
            stride += 1
    else:
        stride = length
        if stride % 2 == 0:
            stride += 1

    logits_mod, logits = make_logits_with_stride(num_rows, length, stride, seed=4000)

    seq_lens_mod = np.array([0, 1, 511, 512, 513, 8191, 8192, length], dtype=np.int32)

    seq_lens = torch.tensor(seq_lens_mod, device="cuda", dtype=torch.int32)
    ref = top_k_per_row_decode_numpy(logits_mod, seq_lens_mod, topk)

    workspace = torch.zeros(num_rows * 772 * 4, device="cuda", dtype=torch.uint8)
    output = torch.zeros(num_rows, topk, device="cuda", dtype=torch.int32)

    torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, length)

    torch.cuda.synchronize()

    gpu = np.sort(output.cpu().numpy(), axis=-1)
    cpu = np.sort(ref, axis=-1)

    assert np.array_equal(gpu, cpu), f"vectorization ragged failed: {stride_kind=}, {stride=}"


def test_persistent_topk_lengths_1d():
    num_rows = 16
    length = 16387
    topk = 512

    np.random.seed(5000)

    logits_mod = np.random.randn(num_rows, length).astype(np.float32)
    seq_lens_mod = np.array([0, 1, 511, 512, 513, 1024, 2048, 4096, 8191, 8192, 8193, 12000, 14000, 16000, 16386, 16387], dtype=np.int32)

    logits = torch.tensor(logits_mod, device="cuda")
    seq_lens = torch.tensor(seq_lens_mod, device="cuda", dtype=torch.int32)

    ref = top_k_per_row_decode_numpy(logits_mod, seq_lens_mod, topk)

    workspace = torch.zeros(num_rows * 772 * 4, device="cuda", dtype=torch.uint8)
    output = torch.zeros(num_rows, topk, device="cuda", dtype=torch.int32)

    torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, length)

    torch.cuda.synchronize()

    gpu = np.sort(output.cpu().numpy(), axis=-1)
    cpu = np.sort(ref, axis=-1)

    assert np.array_equal(gpu, cpu), "1D lengths path failed"


def test_persistent_topk_lengths_2d_ragged():
    bs = 4
    next_n = 4
    num_rows = bs * next_n
    length = 16387
    topk = 512

    np.random.seed(5100)

    logits_mod = np.random.randn(num_rows, length).astype(np.float32)

    seq_lens_mod = np.array(
        [[0, 1, 512, 513], [8191, 8192, 8193, 1024], [4096, 12000, 16386, 16387], [2048, 2049, 10000, 16000]],
        dtype=np.int32,
    )

    logits = torch.tensor(logits_mod, device="cuda")
    seq_lens = torch.tensor(seq_lens_mod, device="cuda", dtype=torch.int32)

    ref = top_k_per_row_decode_numpy(logits_mod, seq_lens_mod, topk)

    workspace = torch.zeros(num_rows * 772 * 4, device="cuda", dtype=torch.uint8)
    output = torch.zeros(num_rows, topk, device="cuda", dtype=torch.int32)

    torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, length)

    torch.cuda.synchronize()

    gpu = np.sort(output.cpu().numpy(), axis=-1)
    cpu = np.sort(ref, axis=-1)

    assert np.array_equal(gpu, cpu), "2D ragged lengths path failed"


@pytest.mark.parametrize("max_seq_len", [32769, 65536])
@pytest.mark.parametrize("num_rows", [1, 4, 8, 16, 32])
def test_persistent_topk_cooperative_path(max_seq_len, num_rows):
    topk = 512

    np.random.seed(7000 + max_seq_len + num_rows)

    logits_mod = np.random.randn(num_rows, max_seq_len).astype(np.float32)
    seq_lens_mod = np.random.randint(max_seq_len // 2, max_seq_len + 1, size=(num_rows,)).astype(np.int32)

    logits = torch.tensor(logits_mod, device="cuda")
    seq_lens = torch.tensor(seq_lens_mod, device="cuda", dtype=torch.int32)

    ref = top_k_per_row_decode_numpy(logits_mod, seq_lens_mod, topk)

    workspace = torch.empty(num_rows * 772 * 4, device="cuda", dtype=torch.uint8)
    output = torch.empty(num_rows, topk, device="cuda", dtype=torch.int32)

    for iteration in range(3):
        output.fill_(-123)

        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, max_seq_len)

        torch.cuda.synchronize()

        gpu = np.sort(output.cpu().numpy(), axis=-1)
        cpu = np.sort(ref, axis=-1)

        assert np.array_equal(gpu, cpu), f"cooperative path failed: {max_seq_len=}, {num_rows=}, {iteration=}"


def test_persistent_topk_workspace_reuse_after_radix():
    num_rows = 8
    max_seq_len = 32769
    topk = 512

    workspace = torch.empty(num_rows * 772 * 4, device="cuda", dtype=torch.uint8)

    np.random.seed(8000)

    for iteration in range(5):
        logits_mod = np.random.randn(num_rows, max_seq_len).astype(np.float32)
        seq_lens_mod = np.random.randint(1, max_seq_len + 1, size=(num_rows,)).astype(np.int32)

        logits = torch.tensor(logits_mod, device="cuda")
        seq_lens = torch.tensor(seq_lens_mod, device="cuda", dtype=torch.int32)
        output = torch.empty(num_rows, topk, device="cuda", dtype=torch.int32)

        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, max_seq_len)

        torch.cuda.synchronize()

        ref = top_k_per_row_decode_numpy(logits_mod, seq_lens_mod, topk)

        gpu = np.sort(output.cpu().numpy(), axis=-1)
        cpu = np.sort(ref, axis=-1)

        assert np.array_equal(gpu, cpu), f"workspace reuse failed: {iteration=}"


def test_persistent_topk_workspace_too_small():
    num_rows = 1
    max_seq_len = 32769
    topk = 512

    logits_mod = np.random.randn(num_rows, max_seq_len).astype(np.float32)
    seq_lens_mod = np.full(num_rows, max_seq_len, dtype=np.int32)

    logits = torch.tensor(logits_mod, device="cuda")
    seq_lens = torch.tensor(seq_lens_mod, device="cuda", dtype=torch.int32)
    output = torch.zeros(num_rows, topk, device="cuda", dtype=torch.int32)
    workspace = torch.empty(0, device="cuda", dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="workspace too small"):
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, max_seq_len)


def test_persistent_topk_workspace_cuda_check():
    num_rows = 1
    max_seq_len = 8193
    topk = 512

    logits = torch.randn(num_rows, max_seq_len, device="cuda", dtype=torch.float32)
    seq_lens = torch.full((num_rows,), max_seq_len, device="cuda", dtype=torch.int32)
    output = torch.zeros(num_rows, topk, device="cuda", dtype=torch.int32)
    workspace = torch.zeros(num_rows * 772 * 4, dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="workspace must be CUDA tensor"):
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, max_seq_len)


def test_persistent_topk_workspace_dtype_check():
    num_rows = 1
    max_seq_len = 8193
    topk = 512

    logits = torch.randn(num_rows, max_seq_len, device="cuda", dtype=torch.float32)
    seq_lens = torch.full((num_rows,), max_seq_len, device="cuda", dtype=torch.int32)
    output = torch.zeros(num_rows, topk, device="cuda", dtype=torch.int32)
    workspace = torch.zeros(num_rows * 772 * 4, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="workspace must be uint8"):
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, max_seq_len)


def run_tests():
    bs_list = [1, 2, 16, 128, 256]
    spec_list = [0, 1, 3]
    maxlen_list = [16387, 65536]
    topk_list = [512, 1024]

    np.random.seed(42)

    for bs in bs_list:
        for spec in spec_list:
            for maxlen in maxlen_list:
                for topk in topk_list:
                    next_n = 1 + spec
                    num_rows = bs * next_n

                    logits_mod = np.random.randn(num_rows, maxlen).astype(np.float32)
                    seq_lens_mod = np.random.randint(maxlen // 2, maxlen + 1, size=(bs, next_n)).astype(np.int32)

                    out_mod = top_k_per_row_decode_numpy(logits_mod, seq_lens_mod, topk)

                    logits = torch.tensor(logits_mod, device="cuda")
                    seq_lens = torch.tensor(seq_lens_mod, device="cuda")

                    workspace = torch.zeros(num_rows * 772 * 4, device="cuda", dtype=torch.uint8)
                    output = torch.zeros(bs * next_n, topk, device="cuda", dtype=torch.int32)

                    torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

                    torch.cuda.synchronize()

                    torch_sorted = np.sort(output.cpu().numpy(), axis=-1)
                    np_sorted = np.sort(out_mod, axis=-1)

                    assert np.array_equal(torch_sorted, np_sorted), f"test persistent_topk failed. {bs=}, {spec=}, {maxlen=}, {topk=}"


def standard_generate(t=512):
    bs = 3
    spec = 0
    maxlen = 16387
    topk = t
    next_n = 1 + spec
    num_rows = bs * next_n

    logits_mod = np.random.randn(num_rows, maxlen).astype(np.float32)
    seq_lens_mod = np.random.randint(maxlen // 2, maxlen + 1, size=(bs, next_n)).astype(np.int32)

    logits = torch.tensor(logits_mod, device="cuda")
    seq_lens = torch.tensor(seq_lens_mod, device="cuda")
    workspace = torch.zeros(num_rows * 772 * 4, device="cuda", dtype=torch.uint8)
    output = torch.zeros(bs * next_n, topk, device="cuda", dtype=torch.int32)

    return bs, num_rows, maxlen, topk, logits, seq_lens, workspace, output


def run_error_tests():
    with pytest.raises(RuntimeError, match="logits must be 2D"):
        bs, num_rows, maxlen, topk, logits, seq_lens, workspace, output = standard_generate()
        logits_mod = np.random.randn(num_rows, num_rows, maxlen).astype(np.float32)
        logits = torch.tensor(logits_mod, device="cuda")
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

    with pytest.raises(RuntimeError, match="Only float32 supported"):
        bs, num_rows, maxlen, topk, logits, seq_lens, workspace, output = standard_generate()
        logits = logits.to(torch.int32)
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

    with pytest.raises(RuntimeError, match="lengths must be int32"):
        bs, num_rows, maxlen, topk, logits, seq_lens, workspace, output = standard_generate()
        seq_lens = seq_lens.to(torch.int64)
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

    with pytest.raises(RuntimeError, match="output must be int32"):
        bs, num_rows, maxlen, topk, logits, seq_lens, workspace, output = standard_generate()
        output = output.to(torch.float32)
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

    with pytest.raises(RuntimeError, match="lengths must be 1D or 2D"):
        bs, num_rows, maxlen, topk, logits, seq_lens, workspace, output = standard_generate()
        seq_lens_mod = np.random.randn(num_rows, num_rows, maxlen).astype(np.int32)
        seq_lens = torch.tensor(seq_lens_mod, device="cuda")
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

    with pytest.raises(RuntimeError, match="output must be 2D"):
        bs, num_rows, maxlen, topk, logits, seq_lens, workspace, output = standard_generate()
        output = torch.zeros(bs * num_rows, device="cuda", dtype=torch.int32)
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

    with pytest.raises(RuntimeError, match=r"logits strides\[1\] must be 1"):
        bs, num_rows, maxlen, topk, logits, seq_lens, workspace, output = standard_generate()

        logits_tmp = torch.empty((num_rows, maxlen * 2), device="cuda", dtype=torch.float32)
        logits = logits_tmp[:, ::2]

        assert logits.shape == (num_rows, maxlen)
        assert logits.stride(1) != 1

        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

    with pytest.raises(RuntimeError, match="output size mismatch"):
        bs, num_rows, maxlen, topk, logits, seq_lens, workspace, output = standard_generate()

        output = torch.zeros(topk, topk, device="cuda", dtype=torch.int32)

        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

    with pytest.raises(RuntimeError, match="persistent_topk supports k=512, k=1024, or k=2048, got k=128"):
        bs, num_rows, maxlen, topk, logits, seq_lens, workspace, output = standard_generate(128)

        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

    with pytest.raises(RuntimeError, match="lengths size mismatch"):
        bs, num_rows, maxlen, topk, logits, seq_lens, workspace, output = standard_generate()

        seq_lens = torch.zeros(num_rows + 1, device="cuda", dtype=torch.int32)

        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)


def run_test_histogram4096_short_path():
    cases = [(8193, 512), (8193, 1024), (16387, 512), (16387, 1024), (16387, 2048), (32768, 512)]

    for length, topk in cases:
        bs = 1
        num_rows = 1

        logits_mod = np.arange(length, dtype=np.float32).reshape(1, length)
        seq_lens_mod = np.array([[length]], dtype=np.int32)

        ref = top_k_per_row_decode_numpy(logits_mod, seq_lens_mod, topk)

        logits = torch.tensor(logits_mod, device="cuda")
        seq_lens = torch.tensor(seq_lens_mod, device="cuda")
        workspace = torch.zeros(num_rows * 772 * 4, device="cuda", dtype=torch.uint8)
        output = torch.zeros(num_rows, topk, device="cuda", dtype=torch.int32)

        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, length)

        torch.cuda.synchronize()

        gpu = np.sort(output.cpu().numpy(), axis=-1)
        cpu = np.sort(ref, axis=-1)

        assert np.array_equal(gpu, cpu), f"histogram4096 failed length={length}, topk={topk}"

        print(f"PASS histogram4096 length={length} topk={topk}")


if __name__ == "__main__":
    run_tests()
    run_test_histogram4096_short_path()
    run_error_tests()