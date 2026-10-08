import torch

from mcoplib.op import fused_silu_mul_dq_mask_quant


GEMM1_ALPHA = 1.702
GEMM1_LIMIT = 7.0
INT8_SCALE_FACTOR = 0.0078740157  # 1 / 127, matching the CUDA kernel.


def _gpt_oss_swiglu_reference(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    hidden_size = input_tensor.shape[-1] // 2
    gate = input_tensor[..., :hidden_size].float()
    up = input_tensor[..., hidden_size:].float()

    gate = gate.clamp(max=GEMM1_LIMIT)
    up = up.clamp(min=-GEMM1_LIMIT, max=GEMM1_LIMIT)
    output = gate * torch.sigmoid(gate * GEMM1_ALPHA) * (up + 1.0)

    if weight is not None:
        output = output * weight.float()
    return output


def _dynamic_int8_quant_reference(
    input_tensor: torch.Tensor,
):
    absmax = input_tensor.abs().amax(dim=-1)
    scale = absmax * INT8_SCALE_FACTOR

    # Test inputs are deliberately nonzero. Keeping the guard makes failures
    # easier to understand if a future test case changes that assumption.
    if torch.any(absmax == 0):
        raise AssertionError("reference input contains an all-zero token")

    quant = torch.round(
        input_tensor * (127.0 / absmax).unsqueeze(-1)
    )
    quant = quant.clamp(-127, 127).to(torch.int8)
    return quant, scale


def _unpack_int8_output(
    packed_output: torch.Tensor,
    hidden_size: int,
):
    output_bytes = packed_output.view(torch.uint8)
    quant = output_bytes[..., :hidden_size].contiguous().view(torch.int8)
    scale = (
        output_bytes[..., hidden_size:hidden_size + 4]
        .contiguous()
        .view(torch.float32)
        .squeeze(-1)
    )
    return quant, scale


def _run_precision_case(
    case_name: str,
    num_experts: int,
    hidden_size: int,
    mask_values,
    input_dtype: torch.dtype,
    mask_dtype: torch.dtype,
    with_weight: bool,
):
    token_capacity = 12
    input_fp32 = torch.randn(
        num_experts,
        token_capacity,
        hidden_size * 2,
        device="cuda",
        dtype=torch.float32,
    ) * 3.0

    # Guarantee that both sides of the clamp are exercised.
    input_fp32[..., 0] = GEMM1_LIMIT + 3.0
    input_fp32[..., 1] = -GEMM1_LIMIT - 3.0
    input_fp32[..., hidden_size] = GEMM1_LIMIT + 4.0
    input_fp32[..., hidden_size + 1] = -GEMM1_LIMIT - 4.0
    input_tensor = input_fp32.to(input_dtype)

    mask = torch.tensor(mask_values, device="cuda", dtype=mask_dtype)
    weight = None
    if with_weight:
        weight = (
            torch.randn(hidden_size, device="cuda", dtype=torch.float32)
            * 0.25
            + 1.0
        ).to(input_dtype)

    out_stride = (input_tensor.shape[-1] // 4 + 257) // 256 * 256
    packed_output = torch.empty(
        num_experts,
        token_capacity,
        out_stride,
        device="cuda",
        dtype=input_dtype,
    )

    fused_silu_mul_dq_mask_quant(
        packed_output,
        input_tensor,
        mask,
        swiglu_limit=0.0,
        weight=weight,
        gemm1_alpha=GEMM1_ALPHA,
        gemm1_limit=GEMM1_LIMIT,
    )
    torch.cuda.synchronize()

    reference_activation = _gpt_oss_swiglu_reference(
        input_tensor,
        weight,
    )
    reference_quant, reference_scale = _dynamic_int8_quant_reference(
        reference_activation
    )
    actual_quant, actual_scale = _unpack_int8_output(
        packed_output,
        hidden_size,
    )

    max_quant_diff = 0
    max_scale_diff = 0.0
    max_dequant_error = 0.0

    for expert_id, valid_tokens in enumerate(mask_values):
        reference_quant_valid = reference_quant[expert_id, :valid_tokens]
        actual_quant_valid = actual_quant[expert_id, :valid_tokens]
        reference_scale_valid = reference_scale[expert_id, :valid_tokens]
        actual_scale_valid = actual_scale[expert_id, :valid_tokens]

        quant_diff = (
            actual_quant_valid.to(torch.int16)
            - reference_quant_valid.to(torch.int16)
        ).abs()
        max_quant_diff = max(max_quant_diff, int(quant_diff.max().item()))
        if max_quant_diff > 1:
            raise AssertionError(
                f"{case_name}: max INT8 difference is {max_quant_diff}, expected <= 1"
            )

        torch.testing.assert_close(
            actual_scale_valid,
            reference_scale_valid,
            rtol=2e-3,
            atol=1e-5,
            msg=lambda msg: f"{case_name}: scale mismatch\n{msg}",
        )
        max_scale_diff = max(
            max_scale_diff,
            float(
                (actual_scale_valid - reference_scale_valid)
                .abs()
                .max()
                .item()
            ),
        )

        actual_dequant = (
            actual_quant_valid.float()
            * actual_scale_valid.unsqueeze(-1)
        )
        reference_valid = reference_activation[expert_id, :valid_tokens]
        dequant_error = (actual_dequant - reference_valid).abs()
        allowed_error = (
            reference_scale_valid.unsqueeze(-1) * 1.5
            + reference_valid.abs() * 2e-3
            + 1e-5
        )
        if not torch.all(dequant_error <= allowed_error):
            raise AssertionError(
                f"{case_name}: dequantized output exceeds the expected "
                f"INT8 quantization error; max error={dequant_error.max().item()}"
            )
        max_dequant_error = max(
            max_dequant_error,
            float(dequant_error.max().item()),
        )

    print(
        f"{case_name}: passed; "
        f"max_quant_diff={max_quant_diff}, "
        f"max_scale_diff={max_scale_diff:.6e}, "
        f"max_dequant_error={max_dequant_error:.6e}"
    )


def test_fused_silu_mul_dq_mask_quant_gpt_oss_precision():
    if not torch.cuda.is_available():
        print("CUDA is unavailable; skip GPT-OSS SwiGLU precision test")
        return

    torch.manual_seed(20260805)
    test_cases = [
        # Covers the one-mask specialization with a 64-thread block.
        ("mask1_bf16_h512", 1, 512, [7], torch.bfloat16, torch.int32, False),
        # Primary mask2 path requested by the GPT-OSS integration.
        ("mask2_bf16_h1024", 2, 1024, [7, 5], torch.bfloat16, torch.int32, False),
        # Covers int64 masks, weight multiplication, and a 256-thread block.
        ("mask2_fp16_h2048_weight", 2, 2048, [5, 9], torch.float16, torch.int64, True),
        # Covers the generic mask kernel and a 512-thread block.
        ("mask3_bf16_h4096_weight", 3, 4096, [9, 6, 3], torch.bfloat16, torch.int32, True),
    ]

    for test_case in test_cases:
        _run_precision_case(*test_case)


if __name__ == "__main__":
    test_fused_silu_mul_dq_mask_quant_gpt_oss_precision()
