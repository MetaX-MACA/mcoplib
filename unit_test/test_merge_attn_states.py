# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import mcoplib._C

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_merge_attn_states import merge_attn_states as merge_attn_states_triton


# ============================================================
# MCOPLIB merge_attn_states
# ============================================================
merge_attn_states_mcoplib = torch.ops._C.merge_attn_states


# ============================================================
# Test cases
#
# (num_tokens, num_query_heads, head_size, input_dtype,
#  use_fp8, prefill_tokens_with_context)
# ============================================================
TEST_CASES = [
    # FP16
    (256, 8, 64, torch.float16, False, None),
    (1024, 32, 128, torch.float16, False, None),
    (4096, 64, 128, torch.float16, False, None),
    (4096, 32, 256, torch.float16, False, None),

    # BF16
    (1024, 32, 128, torch.bfloat16, False, None),
    (4096, 64, 128, torch.bfloat16, False, None),

    # prefill_tokens_with_context
    (1024, 32, 128, torch.float16, False, 128),
    (4096, 64, 128, torch.float16, False, 128),

    # FP8
    (1024, 32, 128, torch.float16, True, None),
    (4096, 64, 128, torch.float16, True, None),

    # FP8 + prefill_tokens_with_context
    (1024, 32, 128, torch.float16, True, 128),
    (4096, 64, 128, torch.float16, True, 128),
]


@pytest.mark.parametrize("num_tokens,num_query_heads,head_size,input_dtype,use_fp8,prefill_tokens_with_context", TEST_CASES)
@torch.inference_mode()
def test_merge_attn_states(num_tokens: int, num_query_heads: int, head_size: int, input_dtype: torch.dtype, use_fp8: bool, prefill_tokens_with_context: int | None):

    num_heads = num_query_heads

    # ========================================================
    # Output dtype
    # ========================================================
    output_dtype = input_dtype
    output_scale = None

    if use_fp8:
        output_dtype = current_platform.fp8_dtype()
        output_scale = torch.tensor([0.05], dtype=torch.float32, device="cuda")

    print("\n" + "=" * 100)
    print("merge_attn_states")
    print(f"  NUM_TOKENS                  : {num_tokens}")
    print(f"  NUM_HEADS                   : {num_heads}")
    print(f"  HEAD_SIZE                   : {head_size}")
    print(f"  INPUT_DTYPE                 : {input_dtype}")
    print(f"  OUTPUT_DTYPE                : {output_dtype}")
    print(f"  USE_FP8                     : {use_fp8}")
    print(f"  PREFILL_TOKENS_WITH_CONTEXT : {prefill_tokens_with_context}")
    print(f"  DEVICE                      : {current_platform.get_device_name()}")
    print("=" * 100)

    # ========================================================
    # Generate LSE
    # ========================================================
    prefix_lse = torch.randn(num_heads, num_tokens, dtype=torch.float32, device="cuda")
    suffix_lse = torch.randn(num_heads, num_tokens, dtype=torch.float32, device="cuda")

    # Generate inf values.
    # Make sure prefix and suffix are not both inf at the same position.
    mask_prefix = torch.rand(num_heads, num_tokens, device="cuda") < 0.1
    mask_suffix = torch.rand(num_heads, num_tokens, device="cuda") < 0.1

    combined_mask = torch.logical_and(mask_prefix, mask_suffix)
    mask_prefix = torch.logical_and(mask_prefix, ~combined_mask)
    mask_suffix = torch.logical_and(mask_suffix, ~combined_mask)

    prefix_lse[mask_prefix] = float("inf")
    suffix_lse[mask_suffix] = float("inf")

    # ========================================================
    # Allocate tensors
    # ========================================================
    output = torch.zeros((num_tokens, num_heads, head_size), dtype=output_dtype, device="cuda")
    output_lse = torch.zeros((num_heads, num_tokens), dtype=torch.float32, device="cuda")

    prefix_output = torch.randn((num_tokens, num_heads, head_size), dtype=input_dtype, device="cuda")
    suffix_output = torch.randn((num_tokens, num_heads, head_size), dtype=input_dtype, device="cuda")

    warmup_times = 5
    repeat_times = 20

    # ========================================================
    # 1. Triton reference
    # ========================================================
    output_triton = output.clone()
    output_lse_triton = output_lse.clone()

    for _ in range(warmup_times):
        merge_attn_states_triton(output_triton, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse_triton, prefill_tokens_with_context, output_scale)

    torch.accelerator.synchronize()

    total_time_triton = 0.0
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)

    for _ in range(repeat_times):
        start.record()
        merge_attn_states_triton(output_triton, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse_triton, prefill_tokens_with_context, output_scale)
        end.record()
        torch.accelerator.synchronize()
        total_time_triton += start.elapsed_time(end)

    avg_time_triton = total_time_triton / repeat_times

    # ========================================================
    # 2. MCOPLIB
    # ========================================================
    output_mcoplib = output.clone()
    output_lse_mcoplib = output_lse.clone()

    for _ in range(warmup_times):
        merge_attn_states_mcoplib(output_mcoplib, output_lse_mcoplib, prefix_output, prefix_lse, suffix_output, suffix_lse, prefill_tokens_with_context, output_scale)

    torch.accelerator.synchronize()

    total_time_mcoplib = 0.0
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)

    for _ in range(repeat_times):
        start.record()
        merge_attn_states_mcoplib(output_mcoplib, output_lse_mcoplib, prefix_output, prefix_lse, suffix_output, suffix_lse, prefill_tokens_with_context, output_scale)
        end.record()
        torch.accelerator.synchronize()
        total_time_mcoplib += start.elapsed_time(end)

    avg_time_mcoplib = total_time_mcoplib / repeat_times

    # ========================================================
    # 3. Performance
    # ========================================================
    performance_ratio = avg_time_triton / avg_time_mcoplib

    print("\nPerformance:")
    print(f"  Triton  : {avg_time_triton:.6f} ms")
    print(f"  MCOPLIB : {avg_time_mcoplib:.6f} ms")
    print(f"  Ratio   : {performance_ratio:.5f}x")

    if performance_ratio >= 1.0:
        print(f"  MCOPLIB is {performance_ratio:.5f}x faster than Triton")
    else:
        print(f"  MCOPLIB is {1.0 / performance_ratio:.5f}x slower than Triton")

    # ========================================================
    # 4. Correctness tolerance
    # ========================================================
    if use_fp8:
        atol, rtol = 1e-1, 1e-1
        assert output_scale is not None
        scale = output_scale.item()
    elif output_dtype == torch.bfloat16:
        atol, rtol = 1e-3, 1e-2
        scale = 1.0
    else:
        atol, rtol = 1e-3, 1e-3
        scale = 1.0

    # ========================================================
    # 5. Output correctness
    # ========================================================
    output_mcoplib_dequant = output_mcoplib.float() * scale
    output_triton_dequant = output_triton.float() * scale

    torch.testing.assert_close(output_mcoplib_dequant, output_triton_dequant, atol=atol, rtol=rtol)

    max_diff_output = torch.max(torch.abs(output_mcoplib_dequant - output_triton_dequant))

    print("\nOutput correctness:")
    print(f"  Max abs diff : {max_diff_output.item():.6e}")
    print("  MCOPLIB vs Triton: PASS")

    # ========================================================
    # 6. LSE correctness
    # ========================================================
    torch.testing.assert_close(output_lse_mcoplib.float(), output_lse_triton.float(), atol=atol, rtol=rtol)

    max_diff_lse = torch.max(torch.abs(output_lse_mcoplib.float() - output_lse_triton.float()))

    print("\nLSE correctness:")
    print(f"  Max abs diff : {max_diff_lse.item():.6e}")
    print("  MCOPLIB vs Triton: PASS")

    # ========================================================
    # 7. Final result
    # ========================================================
    print("\nResult:")
    print("  merge_attn_states: PASS")
    print("=" * 100)
