"""Explicit warp64 ownership for C600-U FP8 K-cache quantization."""

import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl


@gluon.jit
def _quantize_k_cache_c600u_kernel(
    output_nope_q_ptr, output_nope_s_ptr, output_rope_ptr,
    k_nope_ptr, k_rope_ptr, num_tokens,
    output_nope_q_stride_0: int, output_nope_s_stride_0: int,
    output_rope_stride_0: int, k_nope_stride_0: int, k_rope_stride_0: int,
    BLOCK_M: gl.constexpr, NUM_GROUPS: gl.constexpr, GROUP_SIZE: gl.constexpr,
    DIM_ROPE: gl.constexpr, FP8_MIN: gl.constexpr, FP8_MAX: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 16], [4, 4, 4], [1, 1, 1], [2, 1, 0]
    )
    row_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, layout))
    group_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, layout))
    column_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(0, layout))
    tokens = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, layout=row_layout)
    groups = gl.arange(0, NUM_GROUPS, layout=group_layout)
    columns = gl.arange(0, GROUP_SIZE, layout=column_layout)
    rows = tokens.expand_dims(1).expand_dims(2)
    cols = groups.expand_dims(0).expand_dims(2) * GROUP_SIZE + columns.expand_dims(0).expand_dims(0)
    values = gl.load(
        k_nope_ptr + rows * k_nope_stride_0 + cols,
        mask=rows < num_tokens, other=0.0,
    ).to(gl.float32)
    maximum = gl.max(gl.abs(values), axis=2)
    # Match the retained C600-U TL lowering of max_abs / 448. Gluon's
    # division takes a different rounding path for this same expression.
    reciprocal = gl.full(
        (BLOCK_M, NUM_GROUPS), 0x3B124925, gl.int32, gl.SliceLayout(2, layout)
    ).to(gl.float32, bitcast=True)
    scales = gl.where(maximum > 0.0, maximum * reciprocal, 1.0)
    inverse = 1.0 / scales
    quantized = gl.minimum(
        gl.maximum(values * inverse.expand_dims(2), FP8_MIN), FP8_MAX
    ).to(output_nope_q_ptr.dtype.element_ty)
    gl.store(
        output_nope_q_ptr + rows * output_nope_q_stride_0 + cols,
        quantized, mask=rows < num_tokens,
    )
    gl.store(
        output_nope_s_ptr + tokens.expand_dims(1) * output_nope_s_stride_0 + groups.expand_dims(0),
        scales, mask=tokens.expand_dims(1) < num_tokens,
    )

    rope_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [1, 1], [1, 0])
    rope_rows = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, rope_layout))
    rope_cols = gl.arange(0, DIM_ROPE, layout=gl.SliceLayout(0, rope_layout))
    rope = gl.load(
        k_rope_ptr + rope_rows[:, None] * k_rope_stride_0 + rope_cols[None, :],
        mask=rope_rows[:, None] < num_tokens, other=0.0,
    )
    gl.store(
        output_rope_ptr + rope_rows[:, None] * output_rope_stride_0 + rope_cols[None, :],
        rope, mask=rope_rows[:, None] < num_tokens,
    )


@gluon.jit
def _quantize_k_cache_group_c600u_kernel(
    output_nope_q_ptr, output_nope_s_ptr, output_rope_ptr,
    k_nope_ptr, k_rope_ptr, num_tokens,
    output_nope_q_stride_0: int, output_nope_s_stride_0: int,
    output_rope_stride_0: int, k_nope_stride_0: int, k_rope_stride_0: int,
    BLOCK_M: gl.constexpr, GROUP_SIZE: gl.constexpr, DIM_ROPE: gl.constexpr,
    FP8_MIN: gl.constexpr, FP8_MAX: gl.constexpr,
    RAW_OUTPUT: gl.constexpr = False,
):
    if RAW_OUTPUT:
        output_nope_s_ptr = (output_nope_q_ptr + 4 * GROUP_SIZE).to(
            gl.pointer_type(gl.float32))
        output_nope_q_ptr = output_nope_q_ptr.to(gl.pointer_type(gl.float8e4nv))
        output_rope_ptr = output_rope_ptr.to(gl.pointer_type(gl.bfloat16))
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])
    tokens = gl.program_id(0) * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, layout))
    group = gl.program_id(1)
    columns = group * GROUP_SIZE + gl.arange(
        0, GROUP_SIZE, layout=gl.SliceLayout(0, layout))
    values = gl.load(
        k_nope_ptr + tokens[:, None] * k_nope_stride_0 + columns[None, :],
        mask=tokens[:, None] < num_tokens, other=0.0,
    ).to(gl.float32)
    maximum = gl.max(gl.abs(values), axis=1)
    # Match this device's retained TL group kernel: mul_f32 0x3b124925.
    reciprocal = gl.full((BLOCK_M,), 0x3B124925, gl.int32,
                         gl.SliceLayout(1, layout)).to(gl.float32, bitcast=True)
    scales = gl.where(maximum > 0.0, maximum * reciprocal, 1.0)
    quantized = gl.minimum(
        gl.maximum(values * (1.0 / scales)[:, None], FP8_MIN), FP8_MAX
    ).to(output_nope_q_ptr.dtype.element_ty)
    gl.store(
        output_nope_q_ptr + tokens[:, None] * output_nope_q_stride_0 + columns[None, :],
        quantized, mask=tokens[:, None] < num_tokens,
    )
    gl.store(output_nope_s_ptr + tokens * output_nope_s_stride_0 + group,
             scales, mask=tokens < num_tokens)
    if group == 0:
        rope_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [1, 1], [1, 0])
        rope_rows = gl.program_id(0) * BLOCK_M + gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(1, rope_layout))
        rope_cols = gl.arange(0, DIM_ROPE, layout=gl.SliceLayout(0, rope_layout))
        rope = gl.load(
            k_rope_ptr + rope_rows[:, None] * k_rope_stride_0 + rope_cols[None, :],
            mask=rope_rows[:, None] < num_tokens, other=0.0,
        )
        gl.store(
            output_rope_ptr + rope_rows[:, None] * output_rope_stride_0 + rope_cols[None, :],
            rope, mask=rope_rows[:, None] < num_tokens,
        )


@gluon.jit
def _quantize_k_cache_decode_inline_c600u_kernel(
    output_nope_ptr, output_rope_ptr, k_nope_ptr, k_rope_ptr,
    output_nope_q_stride_0: int, output_nope_s_stride_0: int,
    output_rope_stride_0: int, k_nope_stride_0: int, k_rope_stride_0: int,
    GROUP_SIZE: gl.constexpr, DIM_NOPE: gl.constexpr, DIM_ROPE: gl.constexpr,
    FP8_MIN: gl.constexpr, FP8_MAX: gl.constexpr,
):
    token = gl.program_id(0)
    block = gl.program_id(1)
    if block < 2:
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [2, 32], [1, 1], [1, 0])
        groups = block * 2 + gl.arange(0, 2, layout=gl.SliceLayout(1, layout))
        columns = gl.arange(0, GROUP_SIZE, layout=gl.SliceLayout(0, layout))
        offsets = groups[:, None] * GROUP_SIZE + columns[None, :]
        values = gl.load(k_nope_ptr + token * k_nope_stride_0 + offsets).to(gl.float32)
        maximum = gl.max(gl.abs(values), axis=1)
        # Decode retains scale=0 for a zero group, unlike the multi-token path.
        # Its retained TL kernel also uses correctly rounded division.
        scales = gl.fdiv(maximum, FP8_MAX, ieee_rounding=True)
        inverse = gl.fdiv(1.0, scales, ieee_rounding=True)
        quantized = gl.minimum(
            gl.maximum(values * inverse[:, None], FP8_MIN), FP8_MAX
        ).to(gl.float8e4nv)
        bits = quantized.to(gl.uint8, bitcast=True)
        sign = ((values.to(gl.uint32, bitcast=True) >> 24) & 0x80).to(gl.uint8)
        bits = gl.where((bits & 0x7F) == 0, sign, bits)
        output_q = output_nope_ptr.to(gl.pointer_type(gl.float8e4nv))
        output_s = (output_nope_ptr + DIM_NOPE).to(gl.pointer_type(gl.float32))
        gl.store(output_q + token * output_nope_q_stride_0 + offsets,
                 bits.to(gl.float8e4nv, bitcast=True))
        gl.store(output_s + token * output_nope_s_stride_0 + groups, scales)
    else:
        rope_layout: gl.constexpr = gl.BlockedLayout([2], [64], [1], [0])
        rope_columns = gl.arange(0, DIM_ROPE, layout=rope_layout)
        rope = gl.load(k_rope_ptr + token * k_rope_stride_0 + rope_columns)
        output = output_rope_ptr.to(gl.pointer_type(gl.bfloat16))
        gl.store(output + token * output_rope_stride_0 + rope_columns, rope)



@gluon.jit
def _quantize_k_cache_decode_four_group_c600u_kernel(
    output_nope_ptr, output_rope_ptr, k_nope_ptr, k_rope_ptr,
    output_nope_q_stride_0: int, output_nope_s_stride_0: int,
    output_rope_stride_0: int, k_nope_stride_0: int, k_rope_stride_0: int,
    GROUP_SIZE: gl.constexpr, DIM_NOPE: gl.constexpr, DIM_ROPE: gl.constexpr,
    FP8_MIN: gl.constexpr, FP8_MAX: gl.constexpr,
):
    token = gl.program_id(0)
    block = gl.program_id(1)
    if block == 0:
        layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [1, 1], [1, 0])
        groups = gl.arange(0, 4, layout=gl.SliceLayout(1, layout))
        columns = gl.arange(0, GROUP_SIZE, layout=gl.SliceLayout(0, layout))
        offsets = groups[:, None] * GROUP_SIZE + columns[None, :]
        values = gl.load(k_nope_ptr + token * k_nope_stride_0 + offsets).to(gl.float32)
        maximum = gl.max(gl.abs(values), axis=1)
        # Decode retains scale=0 for a zero group, unlike the multi-token path.
        # Its retained TL kernel also uses correctly rounded division.
        scales = gl.fdiv(maximum, FP8_MAX, ieee_rounding=True)
        inverse = gl.fdiv(1.0, scales, ieee_rounding=True)
        quantized = gl.minimum(
            gl.maximum(values * inverse[:, None], FP8_MIN), FP8_MAX
        ).to(gl.float8e4nv)
        bits = quantized.to(gl.uint8, bitcast=True)
        sign = ((values.to(gl.uint32, bitcast=True) >> 24) & 0x80).to(gl.uint8)
        bits = gl.where((bits & 0x7F) == 0, sign, bits)
        output_q = output_nope_ptr.to(gl.pointer_type(gl.float8e4nv))
        output_s = (output_nope_ptr + DIM_NOPE).to(gl.pointer_type(gl.float32))
        gl.store(output_q + token * output_nope_q_stride_0 + offsets,
                 bits.to(gl.float8e4nv, bitcast=True))
        gl.store(output_s + token * output_nope_s_stride_0 + groups, scales)
    else:
        rope_layout: gl.constexpr = gl.BlockedLayout([2], [64], [1], [0])
        rope_columns = gl.arange(0, DIM_ROPE, layout=rope_layout)
        rope = gl.load(k_rope_ptr + token * k_rope_stride_0 + rope_columns)
        output = output_rope_ptr.to(gl.pointer_type(gl.bfloat16))
        gl.store(output + token * output_rope_stride_0 + rope_columns, rope)
