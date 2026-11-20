"""Explicit C500 lane ownership for the large-prefill FP8 cache kernel."""

import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl


@gluon.jit
def _bounded_fp8_e4m3_gluon(values):
    bits = values.to(gl.int32, bitcast=True)
    magnitude = bits & 0x7FFFFFFF
    sign = (bits >> 24) & 0x80
    # At exponent E+20, an FP32 ULP equals an E4M3 ULP at exponent E.
    # Clamp E to -6 so the same addition also rounds FP8 subnormals.
    magic_bits = gl.maximum(magnitude & 0x7F800000, 0x3C800000) + 0x0A000000
    rounded_bits = (gl.abs(values) + magic_bits.to(gl.float32, bitcast=True)).to(gl.int32, bitcast=True)
    rounded = (rounded_bits & 0x7FFFFF) + (magic_bits >> 20) - 1128
    encoded = sign | rounded
    return encoded.to(gl.uint8).to(gl.float8e4nv, bitcast=True)


@gluon.jit
def _quantize_k_cache_gluon_kernel(
    output_nope_q_ptr, output_nope_s_ptr, output_rope_ptr,
    k_nope_ptr, k_rope_ptr, num_tokens,
    output_nope_q_stride_0: int, output_nope_s_stride_0: int,
    output_rope_stride_0: int, k_nope_stride_0: int, k_rope_stride_0: int,
    BLOCK_M: gl.constexpr, NUM_GROUPS: gl.constexpr, GROUP_SIZE: gl.constexpr,
    DIM_NOPE: gl.constexpr, DIM_ROPE: gl.constexpr,
    FP8_MIN: gl.constexpr, FP8_MAX: gl.constexpr,
    BOUNDED_FP8: gl.constexpr,
    MAGIC_FP8: gl.constexpr = False,
):
    # Four rows x four groups x four reduction lanes = 64 physical lanes.
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 16], [4, 4, 4], [1, 1, 1], [2, 1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, layout))
    group_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, layout))
    column_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(0, layout))
    tokens = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, layout=row_layout)
    groups = gl.arange(0, NUM_GROUPS, layout=group_layout)
    columns = gl.arange(0, GROUP_SIZE, layout=column_layout)
    rows_3d = tokens.expand_dims(1).expand_dims(2)
    cols_3d = groups.expand_dims(0).expand_dims(2) * GROUP_SIZE + columns.expand_dims(0).expand_dims(0)
    mask = rows_3d < num_tokens
    values = gl.load(k_nope_ptr + rows_3d * k_nope_stride_0 + cols_3d, mask=mask, other=0.0).to(gl.float32)
    maximum = gl.max(gl.abs(values), axis=2)
    # Match the retained Triton lowering: hardware rcp(448), then multiply.
    denominator = gl.full((BLOCK_M, NUM_GROUPS), FP8_MAX, gl.float32, gl.SliceLayout(2, layout))
    reciprocal = gl.inline_asm_elementwise(
        "rcp_f32 $0, $1", constraints="=r,r", args=[denominator],
        dtype=gl.float32, is_pure=True, pack=1,
    )
    scales = gl.where(maximum > 0.0, maximum * reciprocal, 1.0)
    rope_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [1, 1], [1, 0])
    rope_rows = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, rope_layout))
    rope_cols = gl.arange(0, DIM_ROPE, layout=gl.SliceLayout(0, rope_layout))
    rope = gl.load(
        k_rope_ptr + rope_rows[:, None] * k_rope_stride_0 + rope_cols[None, :],
        mask=rope_rows[:, None] < num_tokens, other=0.0,
    )
    inverse = 1.0 / scales
    quantized = _bounded_fp8_e4m3_gluon(
        gl.minimum(gl.maximum(values * inverse.expand_dims(2), FP8_MIN), FP8_MAX)
    )
    gl.store(output_nope_q_ptr + rows_3d * output_nope_q_stride_0 + cols_3d, quantized, mask=mask)
    gl.store(
        output_nope_s_ptr + tokens.expand_dims(1) * output_nope_s_stride_0 + groups.expand_dims(0),
        scales, mask=tokens.expand_dims(1) < num_tokens,
    )
    gl.store(
        output_rope_ptr + rope_rows[:, None] * output_rope_stride_0 + rope_cols[None, :],
        rope, mask=rope_rows[:, None] < num_tokens,
    )
