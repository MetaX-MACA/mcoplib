# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import mcoplib._C


DEVICE = "cuda"
OP_NAME = "fused_kimi_k3_mla_kv_concat"

K_NOPE_DIM = 128
K_PE_DIM = 64
K_OUT_DIM = 192


def _require_op():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")
    if not hasattr(torch.ops._C, OP_NAME):
        pytest.skip(f"torch.ops._C.{OP_NAME} is not available")


def _expect_error(fn):
    with pytest.raises(RuntimeError):
        fn()


def _get_tolerance(dtype):
    if dtype == torch.float16:
        return 0.0, 0.0
    return 0.0, 0.0


def _reference(k_nope, k_pe):
    t, h, _ = k_nope.shape
    k_pe_expand = k_pe.unsqueeze(1).expand(t, h, K_PE_DIM)
    return torch.cat([k_nope, k_pe_expand], dim=-1)


def _run_op(k_nope, k_pe):
    t, h, _ = k_nope.shape
    k_out = torch.empty((t, h, K_OUT_DIM), dtype=k_nope.dtype, device=k_nope.device)
    torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out)
    torch.cuda.synchronize()
    return k_out


def _check_result(k_nope, k_pe, k_out):
    expected = _reference(k_nope, k_pe)
    assert k_out.shape == expected.shape
    assert k_out.dtype == expected.dtype
    assert k_out.device == expected.device
    atol, rtol = _get_tolerance(k_nope.dtype)
    torch.testing.assert_close(k_out, expected, atol=atol, rtol=rtol)


def _make_inputs(num_tokens, num_heads, dtype, seed=1234):
    torch.manual_seed(seed)
    k_nope = torch.randn((num_tokens, num_heads, K_NOPE_DIM), dtype=dtype, device=DEVICE)
    k_pe = torch.randn((num_tokens, K_PE_DIM), dtype=dtype, device=DEVICE)
    return k_nope, k_pe


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens,num_heads", [(1, 1), (1, 2), (1, 8), (2, 1), (2, 2), (3, 3), (7, 1), (8, 4), (16, 8), (31, 7), (32, 16), (63, 15), (64, 16), (127, 31), (128, 32), (255, 7), (256, 16), (511, 31), (512, 8), (1023, 16), (1024, 32)])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat(dtype, num_tokens, num_heads):
    _require_op()
    k_nope, k_pe = _make_inputs(num_tokens, num_heads, dtype, seed=1234 + num_tokens + num_heads)
    k_out = _run_op(k_nope, k_pe)
    _check_result(k_nope, k_pe, k_out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_zero_tokens(dtype):
    _require_op()

    num_heads = 8
    k_nope = torch.empty((0, num_heads, K_NOPE_DIM), dtype=dtype, device=DEVICE)
    k_pe = torch.empty((0, K_PE_DIM), dtype=dtype, device=DEVICE)
    k_out = torch.empty((0, num_heads, K_OUT_DIM), dtype=dtype, device=DEVICE)

    torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out)
    torch.cuda.synchronize()

    assert k_out.shape == (0, num_heads, K_OUT_DIM)
    assert k_out.numel() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_heads", [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 64])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_head_boundaries(dtype, num_heads):
    _require_op()

    k_nope, k_pe = _make_inputs(16, num_heads, dtype, seed=2000 + num_heads)
    k_out = _run_op(k_nope, k_pe)

    assert k_out.shape == (16, num_heads, K_OUT_DIM)
    _check_result(k_nope, k_pe, k_out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 7, 8, 15, 16, 31, 32, 33, 63, 64, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 2048])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_token_boundaries(dtype, num_tokens):
    _require_op()

    num_heads = 8
    k_nope, k_pe = _make_inputs(num_tokens, num_heads, dtype, seed=3000 + num_tokens)
    k_out = _run_op(k_nope, k_pe)

    assert k_out.shape == (num_tokens, num_heads, K_OUT_DIM)
    _check_result(k_nope, k_pe, k_out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_single_head(dtype):
    _require_op()

    k_nope, k_pe = _make_inputs(128, 1, dtype, seed=4001)
    k_out = _run_op(k_nope, k_pe)

    assert k_out.shape == (128, 1, K_OUT_DIM)
    _check_result(k_nope, k_pe, k_out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_large(dtype):
    _require_op()

    num_tokens = 4096
    num_heads = 32

    k_nope, k_pe = _make_inputs(num_tokens, num_heads, dtype, seed=5001)
    k_out = _run_op(k_nope, k_pe)

    assert k_out.shape == (num_tokens, num_heads, K_OUT_DIM)
    _check_result(k_nope, k_pe, k_out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_k_pe_broadcast(dtype):
    _require_op()

    num_tokens = 32
    num_heads = 16

    k_nope = torch.zeros((num_tokens, num_heads, K_NOPE_DIM), dtype=dtype, device=DEVICE)
    k_pe = torch.arange(num_tokens * K_PE_DIM, dtype=dtype, device=DEVICE).reshape(num_tokens, K_PE_DIM)

    k_out = _run_op(k_nope, k_pe)

    _check_result(k_nope, k_pe, k_out)

    pe_part = k_out[:, :, K_NOPE_DIM:]

    for head in range(1, num_heads):
        assert torch.equal(pe_part[:, 0, :], pe_part[:, head, :])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_k_nope_head_independence(dtype):
    _require_op()

    num_tokens = 8
    num_heads = 8

    k_nope = torch.empty((num_tokens, num_heads, K_NOPE_DIM), dtype=dtype, device=DEVICE)
    k_pe = torch.zeros((num_tokens, K_PE_DIM), dtype=dtype, device=DEVICE)

    for head in range(num_heads):
        k_nope[:, head, :] = float(head + 1)

    k_out = _run_op(k_nope, k_pe)

    for head in range(num_heads):
        expected = k_nope[:, head, :]
        actual = k_out[:, head, :K_NOPE_DIM]
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_token_independence(dtype):
    _require_op()

    num_tokens = 32
    num_heads = 4

    k_nope = torch.empty((num_tokens, num_heads, K_NOPE_DIM), dtype=dtype, device=DEVICE)
    k_pe = torch.empty((num_tokens, K_PE_DIM), dtype=dtype, device=DEVICE)

    for token in range(num_tokens):
        k_nope[token].fill_(float(token + 1))
        k_pe[token].fill_(float(-(token + 1)))

    k_out = _run_op(k_nope, k_pe)

    for token in range(num_tokens):
        expected = _reference(k_nope[token:token + 1], k_pe[token:token + 1])
        actual = k_out[token:token + 1]
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_zero_values(dtype):
    _require_op()

    num_tokens = 32
    num_heads = 8

    k_nope = torch.zeros((num_tokens, num_heads, K_NOPE_DIM), dtype=dtype, device=DEVICE)
    k_pe = torch.zeros((num_tokens, K_PE_DIM), dtype=dtype, device=DEVICE)

    k_out = _run_op(k_nope, k_pe)

    assert torch.count_nonzero(k_out).item() == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("value_nope,value_pe", [(-3.5, 7.25), (-1.0, -2.0), (1.5, -4.0), (10.0, 20.0), (-20.0, 30.0)])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_constant_values(dtype, value_nope, value_pe):
    _require_op()

    num_tokens = 16
    num_heads = 8

    k_nope = torch.full((num_tokens, num_heads, K_NOPE_DIM), value_nope, dtype=dtype, device=DEVICE)
    k_pe = torch.full((num_tokens, K_PE_DIM), value_pe, dtype=dtype, device=DEVICE)

    k_out = _run_op(k_nope, k_pe)

    expected_nope = torch.full((num_tokens, num_heads, K_NOPE_DIM), value_nope, dtype=dtype, device=DEVICE)
    expected_pe = torch.full((num_tokens, num_heads, K_PE_DIM), value_pe, dtype=dtype, device=DEVICE)
    expected = torch.cat([expected_nope, expected_pe], dim=-1)

    assert torch.equal(k_out, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_small_values(dtype):
    _require_op()

    num_tokens = 16
    num_heads = 8

    k_nope = torch.full((num_tokens, num_heads, K_NOPE_DIM), 1e-4, dtype=dtype, device=DEVICE)
    k_pe = torch.full((num_tokens, K_PE_DIM), -1e-4, dtype=dtype, device=DEVICE)

    k_out = _run_op(k_nope, k_pe)

    _check_result(k_nope, k_pe, k_out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_mixed_pattern(dtype):
    _require_op()

    num_tokens = 64
    num_heads = 8

    torch.manual_seed(6001)

    k_nope = torch.empty((num_tokens, num_heads, K_NOPE_DIM), dtype=dtype, device=DEVICE)
    k_pe = torch.empty((num_tokens, K_PE_DIM), dtype=dtype, device=DEVICE)

    k_nope.normal_(0.0, 2.0)
    k_pe.normal_(0.0, 2.0)

    k_nope[::2].abs_()
    k_pe[::2].abs_()
    k_nope[1::2].neg_()
    k_pe[1::2].neg_()

    k_out = _run_op(k_nope, k_pe)

    _check_result(k_nope, k_pe, k_out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_storage_offset(dtype):
    _require_op()

    num_tokens = 32
    num_heads = 8

    k_nope_base = torch.randn((num_tokens + 4, num_heads + 1, K_NOPE_DIM), dtype=dtype, device=DEVICE)
    k_nope = k_nope_base[4:, :num_heads, :]

    k_pe_base = torch.randn((num_tokens + 5, K_PE_DIM), dtype=dtype, device=DEVICE)
    k_pe = k_pe_base[5:, :]

    assert k_nope.shape == (num_tokens, num_heads, K_NOPE_DIM)
    assert k_pe.shape == (num_tokens, K_PE_DIM)
    assert k_nope.storage_offset() != 0
    assert k_pe.storage_offset() != 0
    assert k_nope.stride(2) == 1
    assert k_pe.stride(1) == 1

    k_out = _run_op(k_nope, k_pe)

    _check_result(k_nope, k_pe, k_out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_output_prefill(dtype):
    _require_op()

    num_tokens = 32
    num_heads = 8

    k_nope, k_pe = _make_inputs(num_tokens, num_heads, dtype, seed=7001)

    k_out = torch.full((num_tokens, num_heads, K_OUT_DIM), 17, dtype=dtype, device=DEVICE)

    torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out)
    torch.cuda.synchronize()

    _check_result(k_nope, k_pe, k_out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_input_unchanged(dtype):
    _require_op()

    k_nope, k_pe = _make_inputs(64, 8, dtype, seed=8001)

    k_nope_before = k_nope.clone()
    k_pe_before = k_pe.clone()

    k_out = _run_op(k_nope, k_pe)

    assert torch.equal(k_nope, k_nope_before)
    assert torch.equal(k_pe, k_pe_before)

    _check_result(k_nope, k_pe, k_out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_repeatability(dtype):
    _require_op()

    k_nope, k_pe = _make_inputs(128, 16, dtype, seed=9001)

    k_out0 = _run_op(k_nope, k_pe)
    k_out1 = _run_op(k_nope, k_pe)
    k_out2 = _run_op(k_nope, k_pe)

    assert torch.equal(k_out0, k_out1)
    assert torch.equal(k_out0, k_out2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_output_layout(dtype):
    _require_op()

    k_nope, k_pe = _make_inputs(8, 4, dtype, seed=10001)
    k_out = _run_op(k_nope, k_pe)

    assert k_out.shape == (8, 4, K_OUT_DIM)
    assert k_out.dtype == dtype
    assert k_out.device.type == "cuda"
    assert k_out.stride(2) == 1
    assert k_out.stride(1) == K_OUT_DIM
    assert k_out.stride(0) == 4 * K_OUT_DIM


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_nonzero_storage_offset_output(dtype):
    _require_op()

    num_tokens = 32
    num_heads = 8

    k_nope, k_pe = _make_inputs(num_tokens, num_heads, dtype, seed=11001)

    k_out_base = torch.empty((num_tokens + 1, num_heads, K_OUT_DIM), dtype=dtype, device=DEVICE)
    k_out = k_out_base[1:]

    assert k_out.shape == (num_tokens, num_heads, K_OUT_DIM)
    assert k_out.storage_offset() != 0
    assert k_out.stride(2) == 1

    torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out)
    torch.cuda.synchronize()

    _check_result(k_nope, k_pe, k_out)


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_invalid_k_nope_dtype():
    _require_op()

    k_nope = torch.randn((4, 4, K_NOPE_DIM), dtype=torch.float32, device=DEVICE)
    k_pe = torch.randn((4, K_PE_DIM), dtype=torch.float32, device=DEVICE)
    k_out = torch.empty((4, 4, K_OUT_DIM), dtype=torch.float32, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_invalid_k_pe_dtype():
    _require_op()

    k_nope = torch.randn((4, 4, K_NOPE_DIM), dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn((4, K_PE_DIM), dtype=torch.bfloat16, device=DEVICE)
    k_out = torch.empty((4, 4, K_OUT_DIM), dtype=torch.float16, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_invalid_output_dtype():
    _require_op()

    k_nope = torch.randn((4, 4, K_NOPE_DIM), dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn((4, K_PE_DIM), dtype=torch.float16, device=DEVICE)
    k_out = torch.empty((4, 4, K_OUT_DIM), dtype=torch.bfloat16, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out))


@pytest.mark.parametrize("bad_shape", [(4, 4, 127), (4, 4, 129), (4, 4), (4, 4, 128, 1)])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_invalid_k_nope_shape(bad_shape):
    _require_op()

    k_nope = torch.randn(bad_shape, dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn((4, K_PE_DIM), dtype=torch.float16, device=DEVICE)
    k_out = torch.empty((4, 4, K_OUT_DIM), dtype=torch.float16, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out))


@pytest.mark.parametrize("bad_shape", [(4, 63), (4, 65), (4,), (4, 64, 1)])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_invalid_k_pe_shape(bad_shape):
    _require_op()

    k_nope = torch.randn((4, 4, K_NOPE_DIM), dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn(bad_shape, dtype=torch.float16, device=DEVICE)
    k_out = torch.empty((4, 4, K_OUT_DIM), dtype=torch.float16, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out))


@pytest.mark.parametrize("bad_shape", [(4, 4, 191), (4, 4, 193), (4, 4), (4, 4, 192, 1)])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_invalid_output_shape(bad_shape):
    _require_op()

    k_nope = torch.randn((4, 4, K_NOPE_DIM), dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn((4, K_PE_DIM), dtype=torch.float16, device=DEVICE)
    k_out = torch.empty(bad_shape, dtype=torch.float16, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_mismatched_token_count_k_pe():
    _require_op()

    k_nope = torch.randn((4, 4, K_NOPE_DIM), dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn((5, K_PE_DIM), dtype=torch.float16, device=DEVICE)
    k_out = torch.empty((4, 4, K_OUT_DIM), dtype=torch.float16, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_mismatched_token_count_output():
    _require_op()

    k_nope = torch.randn((4, 4, K_NOPE_DIM), dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn((4, K_PE_DIM), dtype=torch.float16, device=DEVICE)
    k_out = torch.empty((5, 4, K_OUT_DIM), dtype=torch.float16, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_mismatched_head_count_output():
    _require_op()

    k_nope = torch.randn((4, 4, K_NOPE_DIM), dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn((4, K_PE_DIM), dtype=torch.float16, device=DEVICE)
    k_out = torch.empty((4, 5, K_OUT_DIM), dtype=torch.float16, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_k_nope_last_stride_not_one():
    _require_op()

    k_nope_base = torch.randn((4, 4, K_NOPE_DIM * 2), dtype=torch.float16, device=DEVICE)
    k_nope = k_nope_base[:, :, ::2]
    k_pe = torch.randn((4, K_PE_DIM), dtype=torch.float16, device=DEVICE)
    k_out = torch.empty((4, 4, K_OUT_DIM), dtype=torch.float16, device=DEVICE)

    assert k_nope.shape == (4, 4, K_NOPE_DIM)
    assert k_nope.stride(2) == 2

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_k_pe_last_stride_not_one():
    _require_op()

    k_nope = torch.randn((4, 4, K_NOPE_DIM), dtype=torch.float16, device=DEVICE)
    k_pe_base = torch.randn((4, K_PE_DIM * 2), dtype=torch.float16, device=DEVICE)
    k_pe = k_pe_base[:, ::2]
    k_out = torch.empty((4, 4, K_OUT_DIM), dtype=torch.float16, device=DEVICE)

    assert k_pe.shape == (4, K_PE_DIM)
    assert k_pe.stride(1) == 2

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_output_last_stride_not_one():
    _require_op()

    k_nope = torch.randn((4, 4, K_NOPE_DIM), dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn((4, K_PE_DIM), dtype=torch.float16, device=DEVICE)
    k_out_base = torch.empty((4, 4, K_OUT_DIM * 2), dtype=torch.float16, device=DEVICE)
    k_out = k_out_base[:, :, ::2]

    assert k_out.shape == (4, 4, K_OUT_DIM)
    assert k_out.stride(2) == 2

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_output_non_contiguous_head_view():
    _require_op()

    k_nope = torch.randn((4, 4, K_NOPE_DIM), dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn((4, K_PE_DIM), dtype=torch.float16, device=DEVICE)

    k_out_base = torch.empty((4, 5, K_OUT_DIM), dtype=torch.float16, device=DEVICE)
    k_out = k_out_base[:, :4, :]

    assert k_out.shape == (4, 4, K_OUT_DIM)
    assert k_out.stride(2) == 1
    assert not k_out.is_contiguous()

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_cpu_input():
    _require_op()

    k_nope = torch.randn((4, 4, K_NOPE_DIM), dtype=torch.float16, device="cpu")
    k_pe = torch.randn((4, K_PE_DIM), dtype=torch.float16, device="cpu")
    k_out = torch.empty((4, 4, K_OUT_DIM), dtype=torch.float16, device="cpu")

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_device_mismatch_k_pe():
    _require_op()

    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")

    k_nope = torch.randn((4, 4, K_NOPE_DIM), dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn((4, K_PE_DIM), dtype=torch.float16, device="cpu")
    k_out = torch.empty((4, 4, K_OUT_DIM), dtype=torch.float16, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_device_mismatch_output():
    _require_op()

    k_nope = torch.randn((4, 4, K_NOPE_DIM), dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn((4, K_PE_DIM), dtype=torch.float16, device=DEVICE)
    k_out = torch.empty((4, 4, K_OUT_DIM), dtype=torch.float16, device="cpu")

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_output_storage_offset(dtype):
    _require_op()

    num_tokens = 32
    num_heads = 8

    k_nope, k_pe = _make_inputs(num_tokens, num_heads, dtype, seed=12001)

    k_out_base = torch.empty((num_tokens + 1, num_heads, K_OUT_DIM), dtype=dtype, device=DEVICE)
    k_out = k_out_base[1:]

    assert k_out.storage_offset() != 0
    assert k_out.shape == (num_tokens, num_heads, K_OUT_DIM)
    assert k_out.stride(2) == 1

    torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out)
    torch.cuda.synchronize()

    _check_result(k_nope, k_pe, k_out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_output_prefill(dtype):
    _require_op()

    num_tokens = 16
    num_heads = 8

    k_nope, k_pe = _make_inputs(num_tokens, num_heads, dtype, seed=13001)

    k_out = torch.full((num_tokens, num_heads, K_OUT_DIM), -123, dtype=dtype, device=DEVICE)

    torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out)
    torch.cuda.synchronize()

    _check_result(k_nope, k_pe, k_out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_input_alias_protection(dtype):
    _require_op()

    num_tokens = 16
    num_heads = 8

    k_nope, k_pe = _make_inputs(num_tokens, num_heads, dtype, seed=14001)

    k_nope_before = k_nope.clone()
    k_pe_before = k_pe.clone()

    k_out = _run_op(k_nope, k_pe)

    assert torch.equal(k_nope, k_nope_before)
    assert torch.equal(k_pe, k_pe_before)
    assert k_out.data_ptr() != k_nope.data_ptr()
    assert k_out.data_ptr() != k_pe.data_ptr()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_manual_exact(dtype):
    _require_op()

    num_tokens = 4
    num_heads = 3

    k_nope = torch.zeros((num_tokens, num_heads, K_NOPE_DIM), dtype=dtype, device=DEVICE)
    k_pe = torch.zeros((num_tokens, K_PE_DIM), dtype=dtype, device=DEVICE)

    for t in range(num_tokens):
        for h in range(num_heads):
            k_nope[t, h].fill_(float((t + 1) * (h + 1)))

    for t in range(num_tokens):
        k_pe[t].fill_(float(-(t + 1)))

    k_out = _run_op(k_nope, k_pe)

    expected = _reference(k_nope, k_pe)

    assert torch.equal(k_out, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])