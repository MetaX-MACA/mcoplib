# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import mcoplib._C


DEVICE = "cuda"
OP_NAME = "fused_kimi_k3_mla_kv_concat_quant_fp8"
FP8_DTYPE = torch.float8_e4m3fn

K_NOPE = 128
K_PE = 64
K_TOTAL = 192
V_DIM = 128

DTYPE_KPE_CASES = [
    pytest.param(torch.float16, torch.float16, id="fp16-fp16"),
    pytest.param(torch.float16, FP8_DTYPE, id="fp16-fp8"),
    pytest.param(torch.bfloat16, torch.bfloat16, id="bf16-bf16"),
    pytest.param(torch.bfloat16, FP8_DTYPE, id="bf16-fp8"),
]


def _require_op():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")

    if not hasattr(torch.ops._C, OP_NAME):
        pytest.skip(f"torch.ops._C.{OP_NAME} is not available")


def _expect_error(fn, match=None):
    if match is None:
        with pytest.raises(RuntimeError):
            fn()
    else:
        with pytest.raises(RuntimeError, match=match):
            fn()


def _fp8_roundtrip(x):
    return x.to(FP8_DTYPE).to(x.dtype)


def _reference_k(k_nope, k_pe):
    num_tokens, num_heads, _ = k_nope.shape

    if k_pe.dtype == FP8_DTYPE:
        k_pe_ref = k_pe.to(k_nope.dtype)
    else:
        k_pe_ref = k_pe

    k_pe_ref = k_pe_ref.unsqueeze(1).expand(num_tokens, num_heads, K_PE)

    return torch.cat([k_nope, k_pe_ref], dim=-1)


def _reference_v(v):
    return v


def _make_inputs(num_tokens, num_heads, dtype=torch.float16, k_pe_dtype=None, seed=1234):
    if k_pe_dtype is None:
        k_pe_dtype = dtype

    assert k_pe_dtype == dtype or k_pe_dtype == FP8_DTYPE

    torch.manual_seed(seed)

    k_nope = torch.randn((num_tokens, num_heads, K_NOPE), dtype=dtype, device=DEVICE)
    k_pe = torch.randn((num_tokens, K_PE), dtype=dtype, device=DEVICE)

    if k_pe_dtype == FP8_DTYPE:
        k_pe = k_pe.to(FP8_DTYPE)

    v = torch.randn((num_tokens, num_heads, V_DIM), dtype=dtype, device=DEVICE)

    return k_nope, k_pe, v


def _run_op(k_nope, k_pe, v):
    num_tokens, num_heads, _ = k_nope.shape

    k_fp8 = torch.empty((num_tokens, num_heads, K_TOTAL), dtype=FP8_DTYPE, device=k_nope.device)
    v_fp8 = torch.empty((num_tokens, num_heads, V_DIM), dtype=FP8_DTYPE, device=k_nope.device)

    torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8)
    torch.cuda.synchronize()

    return k_fp8, v_fp8


def _check_result(k_nope, k_pe, v, k_fp8, v_fp8):
    k_expected = _reference_k(k_nope, k_pe)
    v_expected = _reference_v(v)

    k_actual = k_fp8.to(k_nope.dtype)
    v_actual = v_fp8.to(v.dtype)

    if k_nope.dtype == torch.float16:
        atol = 2e-1
        rtol = 2e-1
    else:
        atol = 3e-1
        rtol = 3e-1

    torch.testing.assert_close(k_actual, k_expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(v_actual, v_expected, atol=atol, rtol=rtol)


def _check_fp8_roundtrip_result(k_nope, k_pe, v, k_fp8, v_fp8):
    k_pe_dtype = k_nope.dtype
    k_expected_pe = _fp8_roundtrip(k_pe.to(k_pe_dtype))
    k_expected = torch.cat([_fp8_roundtrip(k_nope), k_expected_pe.unsqueeze(1).expand(k_nope.shape[0], k_nope.shape[1], K_PE)], dim=-1)
    v_expected = _fp8_roundtrip(v)

    k_actual = k_fp8.to(k_nope.dtype)
    v_actual = v_fp8.to(v.dtype)

    if k_nope.dtype == torch.float16:
        atol = 2e-1
        rtol = 2e-1
    else:
        atol = 3e-1
        rtol = 3e-1

    torch.testing.assert_close(k_actual, k_expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(v_actual, v_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@pytest.mark.parametrize("num_tokens,num_heads", [(1, 1), (1, 2), (1, 8), (2, 1), (8, 4), (16, 8), (32, 16), (128, 8), (512, 16)])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8(dtype, k_pe_dtype, num_tokens, num_heads):
    _require_op()

    k_nope, k_pe, v = _make_inputs(num_tokens, num_heads, dtype=dtype, k_pe_dtype=k_pe_dtype, seed=1234)

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    assert k_fp8.dtype == FP8_DTYPE
    assert v_fp8.dtype == FP8_DTYPE
    assert k_fp8.shape == (num_tokens, num_heads, K_TOTAL)
    assert v_fp8.shape == (num_tokens, num_heads, V_DIM)
    assert k_fp8.is_contiguous()
    assert v_fp8.is_contiguous()
    assert torch.isfinite(k_fp8.to(dtype)).all()
    assert torch.isfinite(v_fp8.to(dtype)).all()

    _check_result(k_nope, k_pe, v, k_fp8, v_fp8)


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_single_token(dtype, k_pe_dtype):
    _require_op()

    k_nope, k_pe, v = _make_inputs(1, 32, dtype=dtype, k_pe_dtype=k_pe_dtype, seed=2025)

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    _check_result(k_nope, k_pe, v, k_fp8, v_fp8)


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_k_pe_broadcast(dtype, k_pe_dtype):
    _require_op()

    k_nope, k_pe, v = _make_inputs(32, 16, dtype=dtype, k_pe_dtype=k_pe_dtype, seed=5678)

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    assert k_fp8.shape == (32, 16, K_TOTAL)
    assert v_fp8.shape == (32, 16, V_DIM)

    actual_pe = k_fp8[:, :, K_NOPE:]

    for head in range(1, 16):
        assert torch.equal(actual_pe[:, 0, :], actual_pe[:, head, :]), f"k_pe broadcast mismatch between head 0 and head {head}"


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_k_nope_and_v(dtype, k_pe_dtype):
    _require_op()

    k_nope, k_pe, v = _make_inputs(16, 8, dtype=dtype, k_pe_dtype=k_pe_dtype, seed=31415)

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    k_actual = k_fp8.to(dtype)
    v_actual = v_fp8.to(dtype)
    k_nope_actual = k_actual[:, :, :K_NOPE]

    if dtype == torch.float16:
        atol = 2e-1
        rtol = 2e-1
    else:
        atol = 3e-1
        rtol = 3e-1

    expected_k_nope = _fp8_roundtrip(k_nope)
    expected_v = _fp8_roundtrip(v)

    torch.testing.assert_close(k_nope_actual, expected_k_nope, atol=atol, rtol=rtol)
    torch.testing.assert_close(v_actual, expected_v, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_non_contiguous_input(dtype, k_pe_dtype):
    _require_op()

    torch.manual_seed(7777)

    num_tokens = 64
    num_heads = 8

    k_nope_base = torch.randn((num_tokens, num_heads + 1, K_NOPE), dtype=dtype, device=DEVICE)
    k_nope = k_nope_base[:, :num_heads, :]

    v_base = torch.randn((num_tokens, num_heads + 1, V_DIM), dtype=dtype, device=DEVICE)
    v = v_base[:, :num_heads, :]

    k_pe_base = torch.randn((num_tokens + 1, K_PE), dtype=dtype, device=DEVICE)
    k_pe = k_pe_base[1:, :]

    if k_pe_dtype == FP8_DTYPE:
        k_pe = k_pe.to(FP8_DTYPE)

    assert k_nope.shape == (num_tokens, num_heads, K_NOPE)
    assert v.shape == (num_tokens, num_heads, V_DIM)
    assert k_pe.shape == (num_tokens, K_PE)
    assert k_nope.stride(2) == 1
    assert v.stride(2) == 1
    assert k_pe.stride(1) == 1

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    _check_result(k_nope, k_pe, v, k_fp8, v_fp8)


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_zero_values(dtype, k_pe_dtype):
    _require_op()

    num_tokens = 32
    num_heads = 8

    k_nope = torch.zeros((num_tokens, num_heads, K_NOPE), dtype=dtype, device=DEVICE)
    k_pe = torch.zeros((num_tokens, K_PE), dtype=dtype, device=DEVICE)

    if k_pe_dtype == FP8_DTYPE:
        k_pe = k_pe.to(FP8_DTYPE)

    v = torch.zeros((num_tokens, num_heads, V_DIM), dtype=dtype, device=DEVICE)

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    assert torch.count_nonzero(k_fp8).item() == 0
    assert torch.count_nonzero(v_fp8).item() == 0


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_constant_values(dtype, k_pe_dtype):
    _require_op()

    num_tokens = 16
    num_heads = 4

    k_nope = torch.full((num_tokens, num_heads, K_NOPE), 1.5, dtype=dtype, device=DEVICE)
    k_pe = torch.full((num_tokens, K_PE), -2.25, dtype=dtype, device=DEVICE)

    if k_pe_dtype == FP8_DTYPE:
        k_pe = k_pe.to(FP8_DTYPE)

    v = torch.full((num_tokens, num_heads, V_DIM), 3.75, dtype=dtype, device=DEVICE)

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    k_expected = torch.cat([_fp8_roundtrip(k_nope), _fp8_roundtrip(k_pe.to(dtype)).unsqueeze(1).expand(num_tokens, num_heads, K_PE)], dim=-1)
    v_expected = _fp8_roundtrip(v)

    k_actual = k_fp8.to(dtype)
    v_actual = v_fp8.to(dtype)

    if dtype == torch.float16:
        atol = 2e-1
        rtol = 2e-1
    else:
        atol = 3e-1
        rtol = 3e-1

    torch.testing.assert_close(k_actual, k_expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(v_actual, v_expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_large(dtype, k_pe_dtype):
    _require_op()

    k_nope, k_pe, v = _make_inputs(4096, 32, dtype=dtype, k_pe_dtype=k_pe_dtype, seed=8888)

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    _check_result(k_nope, k_pe, v, k_fp8, v_fp8)


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_zero_tokens(dtype, k_pe_dtype):
    _require_op()

    num_tokens = 0
    num_heads = 8

    k_nope = torch.empty((num_tokens, num_heads, K_NOPE), dtype=dtype, device=DEVICE)
    k_pe = torch.empty((num_tokens, K_PE), dtype=dtype, device=DEVICE)

    if k_pe_dtype == FP8_DTYPE:
        k_pe = k_pe.to(FP8_DTYPE)

    v = torch.empty((num_tokens, num_heads, V_DIM), dtype=dtype, device=DEVICE)
    k_fp8 = torch.empty((num_tokens, num_heads, K_TOTAL), dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.empty((num_tokens, num_heads, V_DIM), dtype=FP8_DTYPE, device=DEVICE)

    torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8)
    torch.cuda.synchronize()

    assert k_fp8.shape == (0, num_heads, K_TOTAL)
    assert v_fp8.shape == (0, num_heads, V_DIM)
    assert k_fp8.numel() == 0
    assert v_fp8.numel() == 0


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_output_layout(dtype, k_pe_dtype):
    _require_op()

    k_nope, k_pe, v = _make_inputs(8, 4, dtype=dtype, k_pe_dtype=k_pe_dtype, seed=9999)

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    assert k_fp8.dtype == FP8_DTYPE
    assert v_fp8.dtype == FP8_DTYPE
    assert k_fp8.shape == (8, 4, K_TOTAL)
    assert v_fp8.shape == (8, 4, V_DIM)
    assert k_fp8.stride(2) == 1
    assert v_fp8.stride(2) == 1
    assert k_fp8.is_contiguous()
    assert v_fp8.is_contiguous()


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 7, 8, 15, 16, 31, 32, 33, 63, 64, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_token_boundaries(dtype, k_pe_dtype, num_tokens):
    _require_op()

    num_heads = 8

    k_nope, k_pe, v = _make_inputs(num_tokens, num_heads, dtype=dtype, k_pe_dtype=k_pe_dtype, seed=11000 + num_tokens)

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    assert k_fp8.shape == (num_tokens, num_heads, K_TOTAL)
    assert v_fp8.shape == (num_tokens, num_heads, V_DIM)
    assert k_fp8.is_contiguous()
    assert v_fp8.is_contiguous()

    _check_result(k_nope, k_pe, v, k_fp8, v_fp8)


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@pytest.mark.parametrize("num_heads", [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 64])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_head_boundaries(dtype, k_pe_dtype, num_heads):
    _require_op()

    num_tokens = 8

    k_nope, k_pe, v = _make_inputs(num_tokens, num_heads, dtype=dtype, k_pe_dtype=k_pe_dtype, seed=12000 + num_heads)

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    assert k_fp8.shape == (num_tokens, num_heads, K_TOTAL)
    assert v_fp8.shape == (num_tokens, num_heads, V_DIM)
    assert k_fp8.is_contiguous()
    assert v_fp8.is_contiguous()

    _check_result(k_nope, k_pe, v, k_fp8, v_fp8)


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_per_head_independence(dtype, k_pe_dtype):
    _require_op()

    num_tokens = 8
    num_heads = 4

    k_nope = torch.zeros((num_tokens, num_heads, K_NOPE), dtype=dtype, device=DEVICE)
    k_pe = torch.zeros((num_tokens, K_PE), dtype=dtype, device=DEVICE)
    v = torch.zeros((num_tokens, num_heads, V_DIM), dtype=dtype, device=DEVICE)

    if k_pe_dtype == FP8_DTYPE:
        k_pe = k_pe.to(FP8_DTYPE)

    for head in range(num_heads):
        k_nope[:, head, :] = float(head + 1)
        v[:, head, :] = float((head + 1) * 2)

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    k_actual = k_fp8.to(dtype)
    v_actual = v_fp8.to(dtype)

    for head in range(num_heads):
        expected_k = _fp8_roundtrip(k_nope[:, head, :])
        expected_v = _fp8_roundtrip(v[:, head, :])

        if dtype == torch.float16:
            atol = 2e-1
            rtol = 2e-1
        else:
            atol = 3e-1
            rtol = 3e-1

        torch.testing.assert_close(k_actual[:, head, :K_NOPE], expected_k, atol=atol, rtol=rtol)
        torch.testing.assert_close(v_actual[:, head, :], expected_v, atol=atol, rtol=rtol)

    for head in range(1, num_heads):
        assert torch.equal(k_fp8[:, 0, K_NOPE:], k_fp8[:, head, K_NOPE:])


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_token_independence(dtype, k_pe_dtype):
    _require_op()

    num_tokens = 16
    num_heads = 4

    k_nope = torch.zeros((num_tokens, num_heads, K_NOPE), dtype=dtype, device=DEVICE)
    k_pe = torch.zeros((num_tokens, K_PE), dtype=dtype, device=DEVICE)
    v = torch.zeros((num_tokens, num_heads, V_DIM), dtype=dtype, device=DEVICE)

    if k_pe_dtype == FP8_DTYPE:
        k_pe = k_pe.to(FP8_DTYPE)

    for token in range(num_tokens):
        k_nope[token] = float(token + 1)
        k_pe[token] = float(-(token + 1))
        v[token] = float((token + 1) * 2)

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    k_actual = k_fp8.to(dtype)
    v_actual = v_fp8.to(dtype)

    for token in range(num_tokens):
        expected_k_nope = _fp8_roundtrip(k_nope[token])
        expected_k_pe = _fp8_roundtrip(k_pe[token].to(dtype))
        expected_v = _fp8_roundtrip(v[token])

        if dtype == torch.float16:
            atol = 2e-1
            rtol = 2e-1
        else:
            atol = 3e-1
            rtol = 3e-1

        torch.testing.assert_close(k_actual[token, :, :K_NOPE], expected_k_nope, atol=atol, rtol=rtol)
        torch.testing.assert_close(k_actual[token, :, K_NOPE:], expected_k_pe.unsqueeze(0).expand(num_heads, K_PE), atol=atol, rtol=rtol)
        torch.testing.assert_close(v_actual[token], expected_v, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_fp8_input_roundtrip(dtype, k_pe_dtype):
    _require_op()

    k_nope, k_pe, v = _make_inputs(64, 8, dtype=dtype, k_pe_dtype=k_pe_dtype, seed=13000)

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    _check_fp8_roundtrip_result(k_nope, k_pe, v, k_fp8, v_fp8)


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_extreme_values(dtype, k_pe_dtype):
    _require_op()

    num_tokens = 16
    num_heads = 8

    values = torch.tensor([-448.0, -100.0, -20.0, -1.0, 0.0, 1.0, 20.0, 100.0, 448.0], dtype=dtype, device=DEVICE)

    k_nope = torch.empty((num_tokens, num_heads, K_NOPE), dtype=dtype, device=DEVICE)
    k_pe = torch.empty((num_tokens, K_PE), dtype=dtype, device=DEVICE)
    v = torch.empty((num_tokens, num_heads, V_DIM), dtype=dtype, device=DEVICE)

    if k_pe_dtype == FP8_DTYPE:
        k_pe = k_pe.to(FP8_DTYPE)

    for i in range(num_tokens):
        value = values[i % values.numel()]
        k_nope[i].fill_(value)
        k_pe[i].fill_(-value)
        v[i].fill_(value * 0.5)

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    assert torch.isfinite(k_fp8.to(dtype)).all()
    assert torch.isfinite(v_fp8.to(dtype)).all()

    _check_result(k_nope, k_pe, v, k_fp8, v_fp8)


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_small_values(dtype, k_pe_dtype):
    _require_op()

    num_tokens = 16
    num_heads = 8

    k_nope = torch.full((num_tokens, num_heads, K_NOPE), 1e-4, dtype=dtype, device=DEVICE)
    k_pe = torch.full((num_tokens, K_PE), -1e-4, dtype=dtype, device=DEVICE)
    v = torch.full((num_tokens, num_heads, V_DIM), 2e-4, dtype=dtype, device=DEVICE)

    if k_pe_dtype == FP8_DTYPE:
        k_pe = k_pe.to(FP8_DTYPE)

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    assert torch.isfinite(k_fp8.to(dtype)).all()
    assert torch.isfinite(v_fp8.to(dtype)).all()

    _check_result(k_nope, k_pe, v, k_fp8, v_fp8)


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_negative_positive_mix(dtype, k_pe_dtype):
    _require_op()

    num_tokens = 32
    num_heads = 8

    torch.manual_seed(14000)

    k_nope = torch.empty((num_tokens, num_heads, K_NOPE), dtype=dtype, device=DEVICE)
    k_pe = torch.empty((num_tokens, K_PE), dtype=dtype, device=DEVICE)
    v = torch.empty((num_tokens, num_heads, V_DIM), dtype=dtype, device=DEVICE)

    if k_pe_dtype == FP8_DTYPE:
        k_pe = torch.randn((num_tokens, K_PE), dtype=dtype, device=DEVICE).to(FP8_DTYPE)
    else:
        k_pe = torch.randn((num_tokens, K_PE), dtype=k_pe_dtype, device=DEVICE)

    k_nope.normal_(0.0, 2.0)
    v.normal_(0.0, 2.0)

    k_nope[::2].abs_()
    k_pe_float = k_pe.to(dtype)
    k_pe_float[::2].abs_()

    if k_pe_dtype == FP8_DTYPE:
        k_pe = k_pe_float.to(FP8_DTYPE)
    else:
        k_pe = k_pe_float.to(k_pe_dtype)

    v[::2].abs_()

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    _check_result(k_nope, k_pe, v, k_fp8, v_fp8)


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_repeatability(dtype, k_pe_dtype):
    _require_op()

    k_nope, k_pe, v = _make_inputs(128, 16, dtype=dtype, k_pe_dtype=k_pe_dtype, seed=15000)

    outputs = []

    for _ in range(3):
        k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)
        outputs.append((k_fp8.clone(), v_fp8.clone()))

    for i in range(1, len(outputs)):
        assert torch.equal(outputs[0][0], outputs[i][0])
        assert torch.equal(outputs[0][1], outputs[i][1])


@pytest.mark.parametrize("bad_dtype", [torch.float32, torch.int32])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_invalid_k_nope_dtype(bad_dtype):
    _require_op()

    num_tokens = 4
    num_heads = 4

    if bad_dtype == torch.int32:
        k_nope = torch.empty((num_tokens, num_heads, K_NOPE), dtype=bad_dtype, device=DEVICE)
    else:
        k_nope = torch.randn((num_tokens, num_heads, K_NOPE), dtype=bad_dtype, device=DEVICE)

    k_pe = torch.randn((num_tokens, K_PE), dtype=torch.float16, device=DEVICE)
    v = torch.randn((num_tokens, num_heads, V_DIM), dtype=torch.float16, device=DEVICE)
    k_fp8 = torch.empty((num_tokens, num_heads, K_TOTAL), dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.empty((num_tokens, num_heads, V_DIM), dtype=FP8_DTYPE, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8))


@pytest.mark.parametrize("bad_dtype", [torch.float32, torch.int32])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_invalid_v_dtype(bad_dtype):
    _require_op()

    num_tokens = 4
    num_heads = 4

    k_nope = torch.randn((num_tokens, num_heads, K_NOPE), dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn((num_tokens, K_PE), dtype=torch.float16, device=DEVICE)

    if bad_dtype == torch.int32:
        v = torch.empty((num_tokens, num_heads, V_DIM), dtype=bad_dtype, device=DEVICE)
    else:
        v = torch.randn((num_tokens, num_heads, V_DIM), dtype=bad_dtype, device=DEVICE)

    k_fp8 = torch.empty((num_tokens, num_heads, K_TOTAL), dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.empty((num_tokens, num_heads, V_DIM), dtype=FP8_DTYPE, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8))


@pytest.mark.parametrize("bad_k_pe_dtype", [torch.float32, torch.int32])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_invalid_k_pe_dtype(bad_k_pe_dtype):
    _require_op()

    num_tokens = 4
    num_heads = 4

    k_nope = torch.randn((num_tokens, num_heads, K_NOPE), dtype=torch.float16, device=DEVICE)

    if bad_k_pe_dtype == torch.int32:
        k_pe = torch.empty((num_tokens, K_PE), dtype=bad_k_pe_dtype, device=DEVICE)
    else:
        k_pe = torch.randn((num_tokens, K_PE), dtype=bad_k_pe_dtype, device=DEVICE)

    v = torch.randn((num_tokens, num_heads, V_DIM), dtype=torch.float16, device=DEVICE)
    k_fp8 = torch.empty((num_tokens, num_heads, K_TOTAL), dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.empty((num_tokens, num_heads, V_DIM), dtype=FP8_DTYPE, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8))


@pytest.mark.parametrize("bad_shape", [(4, 4, 127), (4, 4, 129), (4, 4), (4, 4, 128, 1)])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_invalid_k_nope_shape(bad_shape):
    _require_op()

    k_nope = torch.randn(bad_shape, dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn((4, K_PE), dtype=torch.float16, device=DEVICE)
    v = torch.randn((4, 4, V_DIM), dtype=torch.float16, device=DEVICE)
    k_fp8 = torch.empty((4, 4, K_TOTAL), dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.empty((4, 4, V_DIM), dtype=FP8_DTYPE, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8))


@pytest.mark.parametrize("bad_shape", [(4, 63), (4, 65), (4,), (4, 64, 1)])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_invalid_k_pe_shape(bad_shape):
    _require_op()

    k_nope = torch.randn((4, 4, K_NOPE), dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn(bad_shape, dtype=torch.float16, device=DEVICE)
    v = torch.randn((4, 4, V_DIM), dtype=torch.float16, device=DEVICE)
    k_fp8 = torch.empty((4, 4, K_TOTAL), dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.empty((4, 4, V_DIM), dtype=FP8_DTYPE, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8))


@pytest.mark.parametrize("bad_shape", [(4, 4, 127), (4, 4, 129), (4, 4), (4, 4, 128, 1)])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_invalid_v_shape(bad_shape):
    _require_op()

    k_nope = torch.randn((4, 4, K_NOPE), dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn((4, K_PE), dtype=torch.float16, device=DEVICE)
    v = torch.randn(bad_shape, dtype=torch.float16, device=DEVICE)
    k_fp8 = torch.empty((4, 4, K_TOTAL), dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.empty((4, 4, V_DIM), dtype=FP8_DTYPE, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8))


@pytest.mark.parametrize("bad_dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_invalid_k_fp8_dtype(bad_dtype):
    _require_op()

    k_nope, k_pe, v = _make_inputs(4, 4, dtype=torch.float16, k_pe_dtype=torch.float16, seed=16000)

    k_fp8 = torch.empty((4, 4, K_TOTAL), dtype=bad_dtype, device=DEVICE)
    v_fp8 = torch.empty((4, 4, V_DIM), dtype=FP8_DTYPE, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8))


@pytest.mark.parametrize("bad_dtype", [torch.float16, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_invalid_v_fp8_dtype(bad_dtype):
    _require_op()

    k_nope, k_pe, v = _make_inputs(4, 4, dtype=torch.float16, k_pe_dtype=torch.float16, seed=17000)

    k_fp8 = torch.empty((4, 4, K_TOTAL), dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.empty((4, 4, V_DIM), dtype=bad_dtype, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8))


@pytest.mark.parametrize("bad_shape", [(4, 4, 191), (4, 4, 193), (4, 4), (4, 4, 192, 1)])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_invalid_k_fp8_shape(bad_shape):
    _require_op()

    k_nope, k_pe, v = _make_inputs(4, 4, dtype=torch.float16, k_pe_dtype=torch.float16, seed=18000)

    k_fp8 = torch.empty(bad_shape, dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.empty((4, 4, V_DIM), dtype=FP8_DTYPE, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8))


@pytest.mark.parametrize("bad_shape", [(4, 4, 127), (4, 4, 129), (4, 4), (4, 4, 128, 1)])
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_invalid_v_fp8_shape(bad_shape):
    _require_op()

    k_nope, k_pe, v = _make_inputs(4, 4, dtype=torch.float16, k_pe_dtype=torch.float16, seed=19000)

    k_fp8 = torch.empty((4, 4, V_DIM), dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.empty(bad_shape, dtype=FP8_DTYPE, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_mismatched_token_count_k_pe():
    _require_op()

    k_nope = torch.randn((4, 4, K_NOPE), dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn((5, K_PE), dtype=torch.float16, device=DEVICE)
    v = torch.randn((4, 4, V_DIM), dtype=torch.float16, device=DEVICE)
    k_fp8 = torch.empty((4, 4, K_TOTAL), dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.empty((4, 4, V_DIM), dtype=FP8_DTYPE, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_mismatched_token_count_v():
    _require_op()

    k_nope = torch.randn((4, 4, K_NOPE), dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn((4, K_PE), dtype=torch.float16, device=DEVICE)
    v = torch.randn((5, 4, V_DIM), dtype=torch.float16, device=DEVICE)
    k_fp8 = torch.empty((4, 4, K_TOTAL), dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.empty((4, 4, V_DIM), dtype=FP8_DTYPE, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_mismatched_head_count_v():
    _require_op()

    k_nope = torch.randn((4, 4, K_NOPE), dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn((4, K_PE), dtype=torch.float16, device=DEVICE)
    v = torch.randn((4, 5, V_DIM), dtype=torch.float16, device=DEVICE)
    k_fp8 = torch.empty((4, 4, K_TOTAL), dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.empty((4, 4, V_DIM), dtype=FP8_DTYPE, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_mismatched_output_k_tokens():
    _require_op()

    k_nope, k_pe, v = _make_inputs(4, 4, dtype=torch.float16, k_pe_dtype=torch.float16, seed=20000)

    k_fp8 = torch.empty((5, 4, K_TOTAL), dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.empty((4, 4, V_DIM), dtype=FP8_DTYPE, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_mismatched_output_v_tokens():
    _require_op()

    k_nope, k_pe, v = _make_inputs(4, 4, dtype=torch.float16, k_pe_dtype=torch.float16, seed=21000)

    k_fp8 = torch.empty((4, 4, K_TOTAL), dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.empty((5, 4, V_DIM), dtype=FP8_DTYPE, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_mismatched_output_k_heads():
    _require_op()

    k_nope, k_pe, v = _make_inputs(4, 4, dtype=torch.float16, k_pe_dtype=torch.float16, seed=22000)

    k_fp8 = torch.empty((4, 5, K_TOTAL), dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.empty((4, 4, V_DIM), dtype=FP8_DTYPE, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_mismatched_output_v_heads():
    _require_op()

    k_nope, k_pe, v = _make_inputs(4, 4, dtype=torch.float16, k_pe_dtype=torch.float16, seed=23000)

    k_fp8 = torch.empty((4, 4, K_TOTAL), dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.empty((4, 5, V_DIM), dtype=FP8_DTYPE, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_v_last_stride_not_one():
    _require_op()

    num_tokens = 4
    num_heads = 4

    k_nope = torch.randn((num_tokens, num_heads, K_NOPE), dtype=torch.float16, device=DEVICE)
    k_pe = torch.randn((num_tokens, K_PE), dtype=torch.float16, device=DEVICE)

    v_base = torch.randn((num_tokens, num_heads, V_DIM * 2), dtype=torch.float16, device=DEVICE)
    v = v_base[:, :, ::2]

    assert v.shape == (num_tokens, num_heads, V_DIM)
    assert v.stride(2) == 2

    k_fp8 = torch.empty((num_tokens, num_heads, K_TOTAL), dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.empty((num_tokens, num_heads, V_DIM), dtype=FP8_DTYPE, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8), "v must match")


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_k_nope_last_stride_not_one():
    _require_op()

    num_tokens = 4
    num_heads = 4

    k_nope_base = torch.randn((num_tokens, num_heads, K_NOPE * 2), dtype=torch.float16, device=DEVICE)
    k_nope = k_nope_base[:, :, ::2]

    k_pe = torch.randn((num_tokens, K_PE), dtype=torch.float16, device=DEVICE)
    v = torch.randn((num_tokens, num_heads, V_DIM), dtype=torch.float16, device=DEVICE)

    assert k_nope.shape == (num_tokens, num_heads, K_NOPE)
    assert k_nope.stride(2) == 2

    k_fp8 = torch.empty((num_tokens, num_heads, K_TOTAL), dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.empty((num_tokens, num_heads, V_DIM), dtype=FP8_DTYPE, device=DEVICE)

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8))


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_v_fp8_non_contiguous():
    _require_op()

    k_nope, k_pe, v = _make_inputs(4, 4, dtype=torch.float16, k_pe_dtype=torch.float16, seed=24000)

    k_fp8 = torch.empty((4, 4, K_TOTAL), dtype=FP8_DTYPE, device=DEVICE)

    v_fp8_base = torch.empty((4, 4, V_DIM * 2), dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = v_fp8_base[:, :, ::2]

    assert v_fp8.shape == (4, 4, V_DIM)
    assert not v_fp8.is_contiguous()

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8), "v_fp8 must be")


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_output_prefill_values(dtype, k_pe_dtype):
    _require_op()

    num_tokens = 16
    num_heads = 8

    k_nope, k_pe, v = _make_inputs(num_tokens, num_heads, dtype=dtype, k_pe_dtype=k_pe_dtype, seed=26000)

    k_fp8 = torch.full((num_tokens, num_heads, K_TOTAL), -7, dtype=FP8_DTYPE, device=DEVICE)
    v_fp8 = torch.full((num_tokens, num_heads, V_DIM), 11, dtype=FP8_DTYPE, device=DEVICE)

    torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8)
    torch.cuda.synchronize()

    _check_result(k_nope, k_pe, v, k_fp8, v_fp8)


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_k_pe_storage_offset(dtype, k_pe_dtype):
    _require_op()

    num_tokens = 32
    num_heads = 8

    k_nope, _, v = _make_inputs(num_tokens, num_heads, dtype=dtype, k_pe_dtype=dtype, seed=28000)

    k_pe_base = torch.randn((num_tokens + 7, K_PE), dtype=dtype, device=DEVICE)

    if k_pe_dtype == FP8_DTYPE:
        k_pe_base = k_pe_base.to(FP8_DTYPE)

    k_pe = k_pe_base[7:]

    assert k_pe.shape == (num_tokens, K_PE)
    assert k_pe.stride(1) == 1
    assert k_pe.storage_offset() != 0

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    _check_result(k_nope, k_pe, v, k_fp8, v_fp8)


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_k_nope_storage_offset(dtype, k_pe_dtype):
    _require_op()

    num_tokens = 32
    num_heads = 8

    k_nope_base = torch.randn((num_tokens + 3, num_heads, K_NOPE), dtype=dtype, device=DEVICE)
    k_nope = k_nope_base[3:]

    k_pe = torch.randn((num_tokens, K_PE), dtype=dtype, device=DEVICE)

    if k_pe_dtype == FP8_DTYPE:
        k_pe = k_pe.to(FP8_DTYPE)

    v = torch.randn((num_tokens, num_heads, V_DIM), dtype=dtype, device=DEVICE)

    assert k_nope.shape == (num_tokens, num_heads, K_NOPE)
    assert k_nope.stride(2) == 1
    assert k_nope.storage_offset() != 0

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    _check_result(k_nope, k_pe, v, k_fp8, v_fp8)


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_v_storage_offset(dtype, k_pe_dtype):
    _require_op()

    num_tokens = 32
    num_heads = 8

    k_nope = torch.randn((num_tokens, num_heads, K_NOPE), dtype=dtype, device=DEVICE)
    k_pe = torch.randn((num_tokens, K_PE), dtype=dtype, device=DEVICE)

    if k_pe_dtype == FP8_DTYPE:
        k_pe = k_pe.to(FP8_DTYPE)

    v_base = torch.randn((num_tokens + 5, num_heads, V_DIM), dtype=dtype, device=DEVICE)
    v = v_base[5:]

    assert v.shape == (num_tokens, num_heads, V_DIM)
    assert v.stride(2) == 1
    assert v.storage_offset() != 0

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    _check_result(k_nope, k_pe, v, k_fp8, v_fp8)


@pytest.mark.parametrize("dtype,k_pe_dtype", DTYPE_KPE_CASES)
@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_output_is_not_input(dtype, k_pe_dtype):
    _require_op()

    k_nope, k_pe, v = _make_inputs(8, 4, dtype=dtype, k_pe_dtype=k_pe_dtype, seed=29000)

    k_nope_before = k_nope.clone()
    k_pe_before = k_pe.clone()
    v_before = v.clone()

    k_fp8, v_fp8 = _run_op(k_nope, k_pe, v)

    assert torch.equal(k_nope, k_nope_before)
    assert torch.equal(k_pe, k_pe_before)
    assert torch.equal(v, v_before)

    assert k_fp8.data_ptr() != k_nope.data_ptr()
    assert v_fp8.data_ptr() != v.data_ptr()


@torch.inference_mode()
def test_fused_kimi_k3_mla_kv_concat_quant_fp8_output_alias_independence():
    _require_op()

    k_nope, k_pe, v = _make_inputs(8, 4, dtype=torch.float16, k_pe_dtype=torch.float16, seed=30000)

    k_fp8 = torch.empty((8, 4, K_TOTAL), dtype=FP8_DTYPE, device=DEVICE)

    v_fp8 = k_fp8[:, :, :V_DIM]

    assert not v_fp8.is_contiguous()

    _expect_error(lambda: torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])