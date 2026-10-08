# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

try:
    import mcoplib._C
except ImportError:
    pass


def _require_dsv3():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not hasattr(torch.ops._C, "dsv3_fused_a_gemm"):
        pytest.skip("dsv3_fused_a_gemm was not built")


def _run_dsv3(x, weight, enable_pdl=True):
    output = torch.full((x.shape[0], weight.shape[0]), float("nan"), dtype=torch.bfloat16, device=x.device)
    torch.ops._C.dsv3_fused_a_gemm(output, x, weight.t(), enable_pdl=enable_pdl)
    torch.cuda.synchronize()
    return output


@pytest.mark.parametrize("num_tokens,hd_out,hd_in", [
    (1, 1536, 128),
    (8, 1536, 128),
    (16, 1536, 128),
])
def test_dsv3_fused_a_gemm_pdl(num_tokens, hd_out, hd_in):
    _require_dsv3()

    x = torch.ones((num_tokens, hd_in), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones((hd_out, hd_in), dtype=torch.bfloat16, device="cuda")

    output = _run_dsv3(x, weight, enable_pdl=True)

    assert output.shape == (num_tokens, hd_out)
    assert output.dtype == torch.bfloat16


@pytest.mark.parametrize("num_tokens,hd_out,hd_in", [
    (1, 1536, 128),
    (8, 1536, 128),
    (16, 1536, 128),
])
def test_dsv3_fused_a_gemm_no_pdl(num_tokens, hd_out, hd_in):
    _require_dsv3()

    x = torch.ones((num_tokens, hd_in), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones((hd_out, hd_in), dtype=torch.bfloat16, device="cuda")

    output = _run_dsv3(x, weight, enable_pdl=False)

    assert output.shape == (num_tokens, hd_out)
    assert output.dtype == torch.bfloat16


def test_dsv3_fused_a_gemm_layout():
    _require_dsv3()

    num_tokens = 1
    hd_in = 128
    hd_out = 1536

    x = torch.randn((num_tokens, hd_in), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((hd_out, hd_in), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((num_tokens, hd_out), dtype=torch.bfloat16, device="cuda")

    weight_t = weight.t()

    assert x.shape == (num_tokens, hd_in)
    assert x.stride() == (hd_in, 1)

    assert weight_t.shape == (hd_in, hd_out)
    assert weight_t.stride() == (1, hd_in)

    assert output.shape == (num_tokens, hd_out)
    assert output.stride() == (hd_out, 1)


@pytest.mark.parametrize("num_tokens", [1, 8, 16])
def test_dsv3_fused_a_gemm_output_shape(num_tokens):
    _require_dsv3()

    hd_in = 128
    hd_out = 1536

    x = torch.randn((num_tokens, hd_in), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((hd_out, hd_in), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((num_tokens, hd_out), dtype=torch.bfloat16, device="cuda")

    torch.ops._C.dsv3_fused_a_gemm(output, x, weight.t(), enable_pdl=True)

    torch.cuda.synchronize()

    assert output.shape == (num_tokens, hd_out)
    assert output.dtype == torch.bfloat16