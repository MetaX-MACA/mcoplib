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
    a_bits = a.view(torch.uint8).to(torch.int16)
    b_bits = b.view(torch.uint8).to(torch.int16)

    def to_ordered(bits):
        sign = (bits & 0x80) != 0
        magnitude = bits & 0x7F
        return torch.where(sign, 0x80 - magnitude, 0x80 + magnitude)

    return torch.abs(to_ordered(a_bits) - to_ordered(b_bits))


def reference_rms_norm_static_fp8_quant(input, weight, scale, epsilon):
    x = input.float()
    w = weight.float()

    variance = (x * x).sum(dim=-1, keepdim=True)
    s_variance = torch.rsqrt(variance / input.shape[-1] + epsilon)

    # Exactly match the current kernel:
    # float out_norm = ((scalar_t)(x * s_variance)) * weight[idx];
    out_norm = (x * s_variance).to(input.dtype)
    out_norm = out_norm.float() * w

    # Current kernel:
    # scale_inv = 1 / scale
    # scaled_fp8_conversion<true>(out_norm, scale_inv)
    scaled = out_norm / scale

    # Match FP8 output.
    scaled = torch.clamp(scaled, -448.0, 448.0)

    return scaled.to(FP8_DTYPE)


def print_fp8_diagnostics(ref, out):
    ref_f = ref.float()
    out_f = out.float()

    ulp = fp8_ulp_distance(ref, out)
    diff = torch.abs(ref_f - out_f)

    total = ulp.numel()
    mismatch = int((ulp > 0).sum().item())
    ulp1 = int((ulp <= 1).sum().item())
    ulp2 = int((ulp <= 2).sum().item())
    gt1 = int((ulp > 1).sum().item())
    gt2 = int((ulp > 2).sum().item())
    max_ulp = int(ulp.max().item())

    print("\n[FP8 verification]")
    print(f"mismatch    : {mismatch}/{total} ({mismatch / total * 100:.6f}%)")
    print(f"ULP <= 1    : {ulp1}/{total} ({ulp1 / total * 100:.6f}%)")
    print(f"ULP <= 2    : {ulp2}/{total} ({ulp2 / total * 100:.6f}%)")
    print(f"ULP > 1     : {gt1}/{total} ({gt1 / total * 100:.6f}%)")
    print(f"ULP > 2     : {gt2}/{total} ({gt2 / total * 100:.6f}%)")
    print(f"max ULP     : {max_ulp}")
    print(f"max diff    : {diff.max().item()}")
    print(f"mean diff   : {diff.mean().item()}")

    if mismatch > 0:
        indices = torch.nonzero(ulp > 0, as_tuple=False)

        print("\n[FP8 mismatch samples]")

        for index in indices[:10]:
            idx = tuple(index.tolist())
            print(
                f"index={idx}, "
                f"ref={ref_f[idx].item()}, "
                f"out={out_f[idx].item()}, "
                f"diff={diff[idx].item()}, "
                f"ulp={ulp[idx].item()}"
            )

    return ulp


@pytest.mark.parametrize("num_tokens,hidden_size", TEST_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_scale", QUANT_SCALES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("strided_input", [False, True])
@torch.inference_mode()
def test_rms_norm_static_fp8_quant(num_tokens, hidden_size, dtype, quant_scale, device, strided_input):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(device)

    weight = torch.empty(hidden_size, dtype=dtype, device=device)
    weight.normal_(mean=1.0, std=0.1)

    input_scale = 1.0 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size

    x_base = torch.randn(num_tokens, last_dim, dtype=dtype, device=device)
    x = x_base[..., :hidden_size]

    assert x.is_contiguous() != strided_input

    x *= input_scale

    scale = torch.tensor(quant_scale, dtype=torch.float32, device=device)

    out = torch.empty(num_tokens, hidden_size, dtype=FP8_DTYPE, device=device)

    print("\n" + "=" * 80)
    print("RMSNorm Static FP8 Quant Test")
    print("=" * 80)
    print(f"device       : {device}")
    print(f"dtype        : {dtype}")
    print(f"num_tokens   : {num_tokens}")
    print(f"hidden_size  : {hidden_size}")
    print(f"quant_scale  : {quant_scale}")
    print(f"strided      : {strided_input}")
    print(f"x.shape      : {tuple(x.shape)}")
    print(f"x.stride     : {x.stride()}")
    print(f"weight.shape : {tuple(weight.shape)}")
    print(f"FP8 dtype    : {FP8_DTYPE}")

    ref = reference_rms_norm_static_fp8_quant(
        x,
        weight,
        scale,
        EPSILON,
    )

    torch.ops._C.rms_norm_static_fp8_quant(
        out,
        x,
        weight,
        scale,
        EPSILON,
    )

    torch.cuda.synchronize(device)

    assert out.shape == (num_tokens, hidden_size)
    assert out.dtype == FP8_DTYPE
    assert out.is_contiguous()

    assert torch.isfinite(ref.float()).all()
    assert torch.isfinite(out.float()).all()

    ulp = print_fp8_diagnostics(ref, out)

    max_ulp = int(ulp.max().item())
    num_gt1 = int((ulp > 1).sum().item())

    # Allow at most 1 FP8 ULP difference.
    assert max_ulp <= 1, f"FP8 max ULP too large: {max_ulp}"

    assert num_gt1 == 0, (
        f"FP8 values differ by more than 1 ULP: "
        f"{num_gt1}/{ulp.numel()}"
    )

    print("\nPASS")