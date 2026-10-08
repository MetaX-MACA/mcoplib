# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import mcoplib._C


FP8_DTYPE = torch.float8_e4m3fn
EPSILON = 1e-6

DTYPES = [torch.float16, torch.bfloat16]
TEST_CASES = [(7, 768), (83, 769), (4096, 8192)]
QUANT_SCALES = [0.01, 1.0, 10.0]
CUDA_DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def fp8_ulp_distance(a, b):
    a = a.view(torch.uint8).to(torch.int16)
    b = b.view(torch.uint8).to(torch.int16)

    def ordered(x):
        sign = (x & 0x80) != 0
        return torch.where(sign, 0x80 - (x & 0x7F), 0x80 + (x & 0x7F))

    return torch.abs(ordered(a) - ordered(b))


@pytest.mark.parametrize("num_tokens,hidden_size", TEST_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_scale", QUANT_SCALES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
@torch.inference_mode()
def test_fused_add_rms_norm_static_fp8_quant(num_tokens, hidden_size, dtype, quant_scale, device, strided_input):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(device)

    weight = torch.empty(hidden_size, dtype=dtype, device=device).normal_(mean=1.0, std=0.1)

    scale = 1.0 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size

    x_base = torch.randn(num_tokens, last_dim, dtype=dtype, device=device)
    x = x_base[..., :hidden_size]
    assert x.is_contiguous() != strided_input
    x *= scale

    residual = torch.randn_like(x) * scale
    residual_fused = residual.clone()

    out_quant = torch.empty(num_tokens, hidden_size, dtype=FP8_DTYPE, device=device)
    out_quant_fused = torch.empty_like(out_quant)

    quant_scale_t = torch.tensor(quant_scale, dtype=torch.float32, device=device)

    print("\n" + "=" * 80)
    print("Fused Add RMSNorm Static FP8 Quant Test")
    print("=" * 80)
    print(f"device         : {device}")
    print(f"dtype          : {dtype}")
    print(f"num_tokens     : {num_tokens}")
    print(f"hidden_size    : {hidden_size}")
    print(f"quant_scale    : {quant_scale}")
    print(f"strided        : {strided_input}")
    print(f"x.shape        : {tuple(x.shape)}")
    print(f"x.stride       : {x.stride()}")
    print(f"residual.shape : {tuple(residual.shape)}")

    # Fused path.
    torch.ops._C.fused_add_rms_norm_static_fp8_quant(
        out_quant_fused, x, residual_fused, weight, quant_scale_t, EPSILON
    )

    # Unfused path. fused_add_rms_norm is in-place.
    x_unfused_base = x_base.clone()
    x_unfused = x_unfused_base[..., :hidden_size]
    assert x_unfused.is_contiguous() != strided_input

    torch.ops._C.fused_add_rms_norm(
        x_unfused, residual, weight, EPSILON
    )

    torch.ops._C.static_scaled_fp8_quant(
        out_quant, x_unfused.contiguous(), quant_scale_t
    )

    torch.cuda.synchronize(device)

    # Residual verification.
    torch.testing.assert_close(
        residual_fused,
        residual,
        atol=1e-2,
        rtol=1e-2,
    )

    assert out_quant.shape == out_quant_fused.shape
    assert out_quant.dtype == FP8_DTYPE
    assert out_quant_fused.dtype == FP8_DTYPE
    assert out_quant.is_contiguous()
    assert out_quant_fused.is_contiguous()

    assert torch.isfinite(out_quant.float()).all()
    assert torch.isfinite(out_quant_fused.float()).all()

    # FP8 comparison.
    ulp = fp8_ulp_distance(out_quant, out_quant_fused)

    max_ulp = int(ulp.max().item())
    num_ulp1 = int((ulp > 1).sum().item())
    num_ulp2 = int((ulp > 2).sum().item())
    total = ulp.numel()

    diff = (out_quant.float() - out_quant_fused.float()).abs()

    print("\n[FP8 verification]")
    print(f"max ULP      : {max_ulp}")
    print(f"ULP > 1      : {num_ulp1}/{total}")
    print(f"ULP > 2      : {num_ulp2}/{total}")
    print(f"max diff     : {diff.max().item()}")
    print(f"mean diff    : {diff.mean().item()}")

    # Allow small FP8 conversion differences.
    assert max_ulp <= 1, f"FP8 max ULP too large: {max_ulp}"
    assert num_ulp1 == 0, f"FP8 values differ by more than 1 ULP: {num_ulp1}/{total}"

    print("\nPASS")