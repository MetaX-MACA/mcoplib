# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import mcoplib._C


EPS = 1e-6
FP8_E4M3 = torch.float8_e4m3fn
FP8_E4M3FNUZ = torch.float8_e4m3fnuz
FP8_MAX = 448.0
INT8_MAX = 127.0
INT8_MIN = -128.0

DTYPES = [torch.bfloat16, torch.float32]
QUANT_DTYPES = [torch.int8, "fp8"]

TEST_CASES = [(1, 1024), (7, 5120), (83, 4096), (4096, 5120)]
GROUP_SIZES = [64, 128]
CUDA_DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def _op_available():
    return hasattr(torch.ops._C, "rms_norm_per_block_quant")


def _detect_fp8_dtype(device):
    x = torch.zeros(1, 64, dtype=torch.bfloat16, device=device)
    weight = torch.ones(64, dtype=torch.bfloat16, device=device)
    scales = torch.empty(1, 1, dtype=torch.float32, device=device)

    for fp8_dtype in [FP8_E4M3, FP8_E4M3FNUZ]:
        out = torch.empty(1, 64, dtype=fp8_dtype, device=device)
        try:
            torch.ops._C.rms_norm_per_block_quant(out, x, weight, scales, EPS, None, None, 64, False)
            torch.cuda.synchronize(device)
            return fp8_dtype
        except RuntimeError as e:
            if "Expected out.dtype" in str(e):
                continue
            raise

    raise RuntimeError("Unable to detect supported FP8 dtype")


def _get_quant_dtype(quant_dtype_name, device):
    if quant_dtype_name == "int8":
        return torch.int8
    return _detect_fp8_dtype(device)


def _get_qmax(quant_dtype):
    return INT8_MAX if quant_dtype == torch.int8 else FP8_MAX


def _get_min_scaling_factor(quant_dtype):
    if quant_dtype == torch.int8:
        return 1.0 / 127.0
    return 1.0 / (FP8_MAX * 512.0)


def _make_scale_tensor(num_tokens, num_groups, device, is_scale_transposed):
    if is_scale_transposed:
        base = torch.empty((num_groups, num_tokens), dtype=torch.float32, device=device)
        return base.transpose(0, 1)
    return torch.empty((num_tokens, num_groups), dtype=torch.float32, device=device)


def _make_scale_reference(num_tokens, num_groups, device, is_scale_transposed, values):
    if is_scale_transposed:
        out = torch.empty((num_groups, num_tokens), dtype=torch.float32, device=device).transpose(0, 1)
    else:
        out = torch.empty((num_tokens, num_groups), dtype=torch.float32, device=device)
    out.copy_(values)
    return out


def _reference(input, weight, scale_ub, residual, group_size, quant_dtype, epsilon):
    num_tokens, hidden_size = input.shape
    num_groups = hidden_size // group_size

    qmax = _get_qmax(quant_dtype)
    min_scaling_factor = _get_min_scaling_factor(quant_dtype)

    x = input.float()

    if residual is not None:
        x = x + residual.float()
        residual_out = x.to(residual.dtype)
    else:
        residual_out = None

    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    x_norm = (x * rms).to(input.dtype) * weight

    grouped = x_norm.float().reshape(num_tokens, num_groups, group_size)
    scales = grouped.abs().amax(dim=-1)

    if scale_ub is not None:
        scales = scales.clamp(max=scale_ub.float())

    scales = (scales / qmax).clamp(min=min_scaling_factor)

    quantized = grouped / scales.unsqueeze(-1)

    if quant_dtype == torch.int8:
        quantized = quantized.round().clamp(INT8_MIN, INT8_MAX)
    else:
        quantized = quantized.clamp(-FP8_MAX, FP8_MAX)

    out = quantized.reshape(num_tokens, hidden_size).to(quant_dtype)

    return out, scales, residual_out

def _compare_output(ref, out, quant_dtype):
    if quant_dtype == torch.int8:
        diff = (ref.to(torch.int16) - out.to(torch.int16)).abs()
        max_diff = int(diff.max().item())
        print(f"INT8 max diff : {max_diff}")
        assert max_diff <= 1, f"INT8 output differs by {max_diff}"
        return

    ref_u8 = ref.contiguous().view(torch.uint8).to(torch.int16)
    out_u8 = out.contiguous().view(torch.uint8).to(torch.int16)

    byte_diff = (ref_u8 - out_u8).abs()
    max_diff = int(byte_diff.max().item())
    mismatch = int((byte_diff > 0).sum().item())
    total = byte_diff.numel()

    print(f"FP8 byte mismatch : {mismatch}/{total} ({mismatch / total * 100:.6f}%)")
    print(f"FP8 max byte diff : {max_diff}")

    assert max_diff <= 1, f"FP8 output differs by more than 1 byte step: max_diff={max_diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _op_available(), reason="rms_norm_per_block_quant is not available")
@pytest.mark.parametrize("num_tokens,hidden_size", TEST_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_dtype_name", QUANT_DTYPES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("add_residual", [False, True])
@pytest.mark.parametrize("has_scale_ub", [False, True])
@pytest.mark.parametrize("is_scale_transposed", [False, True])
def test_rms_norm_per_block_quant(num_tokens, hidden_size, dtype, quant_dtype_name, group_size, add_residual, has_scale_ub, is_scale_transposed):
    if hidden_size % group_size != 0:
        pytest.skip("hidden_size must be divisible by group_size")

    device = CUDA_DEVICES[0]
    quant_dtype = _get_quant_dtype(quant_dtype_name, device)
    num_groups = hidden_size // group_size

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(device)

    input_scale = 1.0 / hidden_size

    input = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device) * input_scale
    weight = torch.normal(mean=1.0, std=1.0, size=(hidden_size,), dtype=dtype, device=device)

    residual = torch.randn_like(input) * input_scale if add_residual else None

    # Match official test: scale_ub = mean(input) when enabled.
    scale_ub = input.mean().to(torch.float32) if has_scale_ub else None

    ref_residual = residual.clone() if residual is not None else None
    ops_residual = residual.clone() if residual is not None else None

    # Reference computes logical [num_tokens, num_groups] scales.
    ref_out_calc, ref_scales_calc, ref_residual_calc = _reference(input, weight, scale_ub, ref_residual, group_size, quant_dtype, EPS)

    ref_out = ref_out_calc.clone()

    # Construct the output scale tensor exactly like the operator contract.
    ref_scales = _make_scale_reference(num_tokens, num_groups, device, is_scale_transposed, ref_scales_calc)

    ops_out = torch.empty_like(ref_out)
    ops_scales = _make_scale_tensor(num_tokens, num_groups, device, is_scale_transposed)

    print("\n" + "=" * 80)
    print("RMSNorm Per Block Quant Test")
    print("=" * 80)
    print(f"device              : {device}")
    print(f"dtype               : {dtype}")
    print(f"quant_dtype         : {quant_dtype}")
    print(f"num_tokens          : {num_tokens}")
    print(f"hidden_size         : {hidden_size}")
    print(f"group_size          : {group_size}")
    print(f"add_residual        : {add_residual}")
    print(f"has_scale_ub        : {has_scale_ub}")
    print(f"is_scale_transposed : {is_scale_transposed}")
    print(f"input.shape         : {tuple(input.shape)}")
    print(f"input.stride        : {input.stride()}")
    print(f"scales.shape        : {tuple(ops_scales.shape)}")
    print(f"scales.stride       : {ops_scales.stride()}")

    torch.ops._C.rms_norm_per_block_quant(ops_out, input, weight, ops_scales, EPS, scale_ub, ops_residual, group_size, is_scale_transposed)

    torch.cuda.synchronize(device)

    assert ops_out.shape == input.shape
    assert ops_out.dtype == quant_dtype
    assert ops_out.is_contiguous()

    assert ops_scales.dtype == torch.float32
    assert ops_scales.numel() >= num_tokens * num_groups

    # The scale values must agree. Compare contiguous logical values so that
    # the physical transposed stride does not affect the numerical comparison.
    torch.testing.assert_close(
        ref_scales.contiguous(),
        ops_scales.contiguous(),
        atol=1e-6,
        rtol=1e-5,
    )

    _compare_output(ref_out, ops_out, quant_dtype)

    if add_residual:
        assert ref_residual_calc is not None
        assert ops_residual is not None
        torch.testing.assert_close(ops_residual, ref_residual_calc, atol=1e-2, rtol=1e-2)

    print("PASS")