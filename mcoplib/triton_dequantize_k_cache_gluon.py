"""Explicit C500 ownership for four-row GLM paged dequantization."""

import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl


@gluon.jit
def _e4m3_to_fp32_gluon(bits):
    code = bits.to(gl.uint32)
    magnitude = code & 0x7f
    sign = (code & 0x80) << 24
    normal = ((magnitude << 20) + (120 << 23)).to(gl.float32, bitcast=True)
    # For codes0..7: normal = 2^-7 + code*2^-10, so this FMA is exact.
    subnormal = gl.fma(normal, 2.0, -0.015625)
    value = gl.where(magnitude < 8, subnormal, normal)
    result = value.to(gl.uint32, bitcast=True) | sign
    result = gl.where(magnitude == 0x7f, 0x7fc00000 | sign, result)
    return result.to(gl.float32, bitcast=True)


@gluon.jit
def _dequantize_k_cache_gluon_kernel(
    output_ptr, input_nope_q_ptr, input_nope_s_ptr, input_rope_ptr, page_table_ptr,
    M, OUTPUT_STRIDE: gl.constexpr, INPUT_NOPE_Q_STRIDE: gl.constexpr,
    INPUT_NOPE_S_STRIDE: gl.constexpr, INPUT_ROPE_STRIDE: gl.constexpr,
    DIM_ROPE: gl.constexpr, USE_INT64_INPUT_INDEX: gl.constexpr,
    USE_INT64_OUTPUT_INDEX: gl.constexpr, BLOCK_M: gl.constexpr,
):
    # Four rows x two group lanes x eight column lanes per 64-lane warp.
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 8], [4, 2, 8], [1, 1, 1], [2, 1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, layout))
    group_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, layout))
    column_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(0, layout))
    rows = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, layout=row_layout)
    valid = rows < M
    ids = gl.load(page_table_ptr + rows, valid, 0)
    ids = ids.to(gl.int64) if USE_INT64_INPUT_INDEX else ids.to(gl.int32)
    output_rows = rows.to(gl.int64) if USE_INT64_OUTPUT_INDEX else rows
    groups = gl.arange(0, 2, layout=group_layout)
    elements = gl.arange(0, 128, layout=column_layout)
    byte_ptr = input_nope_q_ptr.to(gl.pointer_type(gl.uint8))
    # Decode and write two groups at a time to shorten FP32 live ranges.
    for chunk in gl.static_range(0, 2):
        chunk_groups = groups + chunk * 2
        offsets = chunk_groups.expand_dims(0).expand_dims(2) * 128 + elements.expand_dims(0).expand_dims(0)
        bits = gl.load(byte_ptr + ids.expand_dims(1).expand_dims(2) * INPUT_NOPE_Q_STRIDE
                       + offsets, valid.expand_dims(1).expand_dims(2), 0)
        values = _e4m3_to_fp32_gluon(bits)
        scales = gl.load(input_nope_s_ptr + ids[:, None] * INPUT_NOPE_S_STRIDE
                         + chunk_groups[None, :], valid[:, None], 0)
        gl.store(output_ptr + output_rows.expand_dims(1).expand_dims(2) * OUTPUT_STRIDE + offsets,
                 values * scales.expand_dims(2), valid.expand_dims(1).expand_dims(2))
    if DIM_ROPE > 0:
        rope_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [1, 1], [1, 0])
        rope_row_layout: gl.constexpr = gl.SliceLayout(1, rope_layout)
        rope_rows = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, layout=rope_row_layout)
        rope_ids = gl.convert_layout(ids, rope_row_layout)
        rope_output_rows = rope_rows.to(gl.int64) if USE_INT64_OUTPUT_INDEX else rope_rows
        rope_offsets = gl.arange(0, DIM_ROPE, layout=gl.SliceLayout(0, rope_layout))
        rope = gl.load(input_rope_ptr + rope_ids[:, None] * INPUT_ROPE_STRIDE
                       + rope_offsets[None, :], rope_rows[:, None] < M, 0)
        gl.store(output_ptr + rope_output_rows[:, None] * OUTPUT_STRIDE + 512
                 + rope_offsets[None, :], rope, rope_rows[:, None] < M)


@gluon.jit
def _dequantize_k_cache_gluon_masked_kernel(
    output_ptr, input_nope_q_ptr, input_nope_s_ptr, input_rope_ptr, page_table_ptr, row_mask_ptr,
    M, OUTPUT_STRIDE: gl.constexpr, INPUT_NOPE_Q_STRIDE: gl.constexpr,
    INPUT_NOPE_S_STRIDE: gl.constexpr, INPUT_ROPE_STRIDE: gl.constexpr,
    DIM_ROPE: gl.constexpr, USE_INT64_INPUT_INDEX: gl.constexpr,
    USE_INT64_OUTPUT_INDEX: gl.constexpr, BLOCK_M: gl.constexpr,
):
    # Two rows x four groups x eight lanes = one physical 64-lane warp.
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 16], [2, 4, 8], [1, 1, 1], [2, 1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, layout))
    group_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, layout))
    column_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(0, layout))
    rows = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, layout=row_layout)
    valid = rows < M
    active = valid & (gl.load(row_mask_ptr + rows, valid, 0) != 0)
    if gl.sum(active.to(gl.int32), axis=0) == 0:
        return
    ids = gl.load(page_table_ptr + rows, active, 0)
    ids = ids.to(gl.int64) if USE_INT64_INPUT_INDEX else ids.to(gl.int32)
    output_rows = rows.to(gl.int64) if USE_INT64_OUTPUT_INDEX else rows
    groups = gl.arange(0, 4, layout=group_layout)
    elements = gl.arange(0, 128, layout=column_layout)
    offsets = groups.expand_dims(0).expand_dims(2) * 128 + elements.expand_dims(0).expand_dims(0)
    byte_ptr = input_nope_q_ptr.to(gl.pointer_type(gl.uint8))
    bits = gl.load(byte_ptr + ids.expand_dims(1).expand_dims(2) * INPUT_NOPE_Q_STRIDE
                   + offsets, active.expand_dims(1).expand_dims(2), 0)
    values = _e4m3_to_fp32_gluon(bits)
    scales = gl.load(input_nope_s_ptr + ids[:, None] * INPUT_NOPE_S_STRIDE
                     + groups[None, :], active[:, None], 0)
    gl.store(output_ptr + output_rows.expand_dims(1).expand_dims(2) * OUTPUT_STRIDE + offsets,
             values * scales.expand_dims(2), active.expand_dims(1).expand_dims(2))
    if DIM_ROPE > 0:
        rope_layout: gl.constexpr = gl.BlockedLayout([1, 2], [2, 32], [1, 1], [1, 0])
        rope_row_layout: gl.constexpr = gl.SliceLayout(1, rope_layout)
        rope_rows = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, layout=rope_row_layout)
        rope_ids = gl.convert_layout(ids, rope_row_layout)
        rope_active = gl.convert_layout(active, rope_row_layout)
        rope_output_rows = rope_rows.to(gl.int64) if USE_INT64_OUTPUT_INDEX else rope_rows
        rope_offsets = gl.arange(0, DIM_ROPE, layout=gl.SliceLayout(0, rope_layout))
        rope = gl.load(input_rope_ptr + rope_ids[:, None] * INPUT_ROPE_STRIDE
                       + rope_offsets[None, :], rope_active[:, None], 0)
        gl.store(output_ptr + rope_output_rows[:, None] * OUTPUT_STRIDE + 512
                 + rope_offsets[None, :], rope, rope_active[:, None])


@gluon.jit
def _dequantize_k_cache_gluon_masked_adaptive_kernel(
    output_ptr, input_nope_q_ptr, input_nope_s_ptr, input_rope_ptr, page_table_ptr, row_mask_ptr,
    M, OUTPUT_STRIDE: gl.constexpr, INPUT_NOPE_Q_STRIDE: gl.constexpr,
    INPUT_NOPE_S_STRIDE: gl.constexpr, INPUT_ROPE_STRIDE: gl.constexpr,
    DIM_ROPE: gl.constexpr, USE_INT64_INPUT_INDEX: gl.constexpr,
    USE_INT64_OUTPUT_INDEX: gl.constexpr, BLOCK_M: gl.constexpr,
):
    # Two rows x four groups x eight lanes = one physical 64-lane warp.
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 16], [2, 4, 8], [1, 1, 1], [2, 1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, layout))
    group_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, layout))
    column_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(0, layout))
    rows = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, layout=row_layout)
    valid = rows < M
    active = valid & (gl.load(row_mask_ptr + rows, valid, 0) != 0)
    active_count = gl.sum(active.to(gl.int32), axis=0)
    if active_count == 0:
        return
    if active_count == 1:
        selected_row = gl.sum(gl.where(active, rows, 0), axis=0)
        selected_id = gl.load(page_table_ptr + selected_row)
        selected_id = selected_id.to(gl.int64) if USE_INT64_INPUT_INDEX else selected_id.to(gl.int32)
        selected_output_row = selected_row.to(gl.int64) if USE_INT64_OUTPUT_INDEX else selected_row
        if DIM_ROPE > 0:
            single_rope_layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
            single_rope_offsets = gl.arange(0, DIM_ROPE, layout=single_rope_layout)
            single_rope = gl.load(input_rope_ptr + selected_id * INPUT_ROPE_STRIDE + single_rope_offsets)
        single_layout: gl.constexpr = gl.BlockedLayout([8], [64], [1], [0])
        single_offsets = gl.arange(0, 512, layout=single_layout)
        single_byte_ptr = input_nope_q_ptr.to(gl.pointer_type(gl.uint8))
        single_bits = gl.load(single_byte_ptr + selected_id * INPUT_NOPE_Q_STRIDE + single_offsets)
        single_values = _e4m3_to_fp32_gluon(single_bits)
        single_scales = gl.load(input_nope_s_ptr + selected_id * INPUT_NOPE_S_STRIDE + single_offsets // 128)
        gl.store(output_ptr + selected_output_row * OUTPUT_STRIDE + single_offsets,
                 single_values * single_scales)
        if DIM_ROPE > 0:
            gl.store(output_ptr + selected_output_row * OUTPUT_STRIDE + 512 + single_rope_offsets,
                     single_rope)
        return
    ids = gl.load(page_table_ptr + rows, active, 0)
    ids = ids.to(gl.int64) if USE_INT64_INPUT_INDEX else ids.to(gl.int32)
    output_rows = rows.to(gl.int64) if USE_INT64_OUTPUT_INDEX else rows
    groups = gl.arange(0, 4, layout=group_layout)
    elements = gl.arange(0, 128, layout=column_layout)
    offsets = groups.expand_dims(0).expand_dims(2) * 128 + elements.expand_dims(0).expand_dims(0)
    byte_ptr = input_nope_q_ptr.to(gl.pointer_type(gl.uint8))
    bits = gl.load(byte_ptr + ids.expand_dims(1).expand_dims(2) * INPUT_NOPE_Q_STRIDE
                   + offsets, active.expand_dims(1).expand_dims(2), 0)
    values = _e4m3_to_fp32_gluon(bits)
    scales = gl.load(input_nope_s_ptr + ids[:, None] * INPUT_NOPE_S_STRIDE
                     + groups[None, :], active[:, None], 0)
    gl.store(output_ptr + output_rows.expand_dims(1).expand_dims(2) * OUTPUT_STRIDE + offsets,
             values * scales.expand_dims(2), active.expand_dims(1).expand_dims(2))
    if DIM_ROPE > 0:
        rope_layout: gl.constexpr = gl.BlockedLayout([1, 2], [2, 32], [1, 1], [1, 0])
        rope_row_layout: gl.constexpr = gl.SliceLayout(1, rope_layout)
        rope_rows = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, layout=rope_row_layout)
        rope_ids = gl.convert_layout(ids, rope_row_layout)
        rope_active = gl.convert_layout(active, rope_row_layout)
        rope_output_rows = rope_rows.to(gl.int64) if USE_INT64_OUTPUT_INDEX else rope_rows
        rope_offsets = gl.arange(0, DIM_ROPE, layout=gl.SliceLayout(0, rope_layout))
        rope = gl.load(input_rope_ptr + rope_ids[:, None] * INPUT_ROPE_STRIDE
                       + rope_offsets[None, :], rope_active[:, None], 0)
        gl.store(output_ptr + rope_output_rows[:, None] * OUTPUT_STRIDE + 512
                 + rope_offsets[None, :], rope, rope_active[:, None])


@gluon.jit
def _dequantize_k_cache_gluon_masked_prefetch_kernel(
    output_ptr, input_nope_q_ptr, input_nope_s_ptr, input_rope_ptr, page_table_ptr, row_mask_ptr,
    M, OUTPUT_STRIDE: gl.constexpr, INPUT_NOPE_Q_STRIDE: gl.constexpr,
    INPUT_NOPE_S_STRIDE: gl.constexpr, INPUT_ROPE_STRIDE: gl.constexpr,
    DIM_ROPE: gl.constexpr, USE_INT64_INPUT_INDEX: gl.constexpr,
    USE_INT64_OUTPUT_INDEX: gl.constexpr, BLOCK_M: gl.constexpr,
):
    # Two rows x four groups x eight lanes = one physical 64-lane warp.
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 8], [2, 4, 8], [1, 1, 1], [2, 1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, layout))
    group_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, layout))
    column_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(0, layout))
    rows = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, layout=row_layout)
    valid = rows < M
    active = valid & (gl.load(row_mask_ptr + rows, valid, 0) != 0)
    active_count = gl.sum(active.to(gl.int32), axis=0)
    if active_count == 0:
        return
    if active_count == 1:
        selected_row = gl.sum(gl.where(active, rows, 0), axis=0)
        selected_id = gl.load(page_table_ptr + selected_row)
        selected_id = selected_id.to(gl.int64) if USE_INT64_INPUT_INDEX else selected_id.to(gl.int32)
        selected_output_row = selected_row.to(gl.int64) if USE_INT64_OUTPUT_INDEX else selected_row
        if DIM_ROPE > 0:
            single_rope_layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
            single_rope_offsets = gl.arange(0, DIM_ROPE, layout=single_rope_layout)
            single_rope = gl.load(input_rope_ptr + selected_id * INPUT_ROPE_STRIDE + single_rope_offsets)
        single_layout: gl.constexpr = gl.BlockedLayout([8], [64], [1], [0])
        single_offsets = gl.arange(0, 512, layout=single_layout)
        single_byte_ptr = input_nope_q_ptr.to(gl.pointer_type(gl.uint8))
        single_bits = gl.load(single_byte_ptr + selected_id * INPUT_NOPE_Q_STRIDE + single_offsets)
        single_values = _e4m3_to_fp32_gluon(single_bits)
        single_scales = gl.load(input_nope_s_ptr + selected_id * INPUT_NOPE_S_STRIDE + single_offsets // 128)
        gl.store(output_ptr + selected_output_row * OUTPUT_STRIDE + single_offsets,
                 single_values * single_scales)
        if DIM_ROPE > 0:
            gl.store(output_ptr + selected_output_row * OUTPUT_STRIDE + 512 + single_rope_offsets,
                     single_rope)
        return
    ids = gl.load(page_table_ptr + rows, active, 0)
    ids = ids.to(gl.int64) if USE_INT64_INPUT_INDEX else ids.to(gl.int32)
    output_rows = rows.to(gl.int64) if USE_INT64_OUTPUT_INDEX else rows
    if DIM_ROPE > 0:
        rope_layout: gl.constexpr = gl.BlockedLayout([1, 2], [2, 32], [1, 1], [1, 0])
        rope_row_layout: gl.constexpr = gl.SliceLayout(1, rope_layout)
        rope_rows = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, layout=rope_row_layout)
        rope_ids = gl.convert_layout(ids, rope_row_layout)
        rope_active = gl.convert_layout(active, rope_row_layout)
        rope_output_rows = rope_rows.to(gl.int64) if USE_INT64_OUTPUT_INDEX else rope_rows
        rope_offsets = gl.arange(0, DIM_ROPE, layout=gl.SliceLayout(0, rope_layout))
        rope = gl.load(input_rope_ptr + rope_ids[:, None] * INPUT_ROPE_STRIDE
                       + rope_offsets[None, :], rope_active[:, None], 0)
    groups = gl.arange(0, 4, layout=group_layout)
    elements = gl.arange(0, 128, layout=column_layout)
    offsets = groups.expand_dims(0).expand_dims(2) * 128 + elements.expand_dims(0).expand_dims(0)
    byte_ptr = input_nope_q_ptr.to(gl.pointer_type(gl.uint8))
    bits = gl.load(byte_ptr + ids.expand_dims(1).expand_dims(2) * INPUT_NOPE_Q_STRIDE
                   + offsets, active.expand_dims(1).expand_dims(2), 0)
    values = _e4m3_to_fp32_gluon(bits)
    scales = gl.load(input_nope_s_ptr + ids[:, None] * INPUT_NOPE_S_STRIDE
                     + groups[None, :], active[:, None], 0)
    gl.store(output_ptr + output_rows.expand_dims(1).expand_dims(2) * OUTPUT_STRIDE + offsets,
             values * scales.expand_dims(2), active.expand_dims(1).expand_dims(2))
    if DIM_ROPE > 0:
        gl.store(output_ptr + rope_output_rows[:, None] * OUTPUT_STRIDE + 512
                 + rope_offsets[None, :], rope, rope_active[:, None])


@gluon.jit
def _dequantize_k_cache_gluon_rows2_kernel(
    output_ptr, input_nope_q_ptr, input_nope_s_ptr, input_rope_ptr, page_table_ptr,
    M, OUTPUT_STRIDE: gl.constexpr, INPUT_NOPE_Q_STRIDE: gl.constexpr,
    INPUT_NOPE_S_STRIDE: gl.constexpr, INPUT_ROPE_STRIDE: gl.constexpr,
    DIM_ROPE: gl.constexpr, USE_INT64_INPUT_INDEX: gl.constexpr,
    USE_INT64_OUTPUT_INDEX: gl.constexpr, BLOCK_M: gl.constexpr,
):
    # Two rows x four groups x eight lanes = one physical 64-lane warp.
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 8], [2, 4, 8], [1, 1, 1], [2, 1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, layout))
    group_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, layout))
    column_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(0, layout))
    rows = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, layout=row_layout)
    valid = rows < M
    ids = gl.load(page_table_ptr + rows, valid, 0)
    ids = ids.to(gl.int64) if USE_INT64_INPUT_INDEX else ids.to(gl.int32)
    output_rows = rows.to(gl.int64) if USE_INT64_OUTPUT_INDEX else rows
    groups = gl.arange(0, 4, layout=group_layout)
    elements = gl.arange(0, 128, layout=column_layout)
    offsets = groups.expand_dims(0).expand_dims(2) * 128 + elements.expand_dims(0).expand_dims(0)
    if DIM_ROPE > 0:
        rope_layout: gl.constexpr = gl.BlockedLayout([1, 2], [2, 32], [1, 1], [1, 0])
        rope_row_layout: gl.constexpr = gl.SliceLayout(1, rope_layout)
        rope_rows = gl.program_id(0) * BLOCK_M + gl.arange(0, BLOCK_M, layout=rope_row_layout)
        rope_ids = gl.convert_layout(ids, rope_row_layout)
        rope_output_rows = rope_rows.to(gl.int64) if USE_INT64_OUTPUT_INDEX else rope_rows
        rope_offsets = gl.arange(0, DIM_ROPE, layout=gl.SliceLayout(0, rope_layout))
        rope = gl.load(input_rope_ptr + rope_ids[:, None] * INPUT_ROPE_STRIDE
                       + rope_offsets[None, :], rope_rows[:, None] < M, 0)
    byte_ptr = input_nope_q_ptr.to(gl.pointer_type(gl.uint8))
    bits = gl.load(byte_ptr + ids.expand_dims(1).expand_dims(2) * INPUT_NOPE_Q_STRIDE
                   + offsets, valid.expand_dims(1).expand_dims(2), 0)
    values = _e4m3_to_fp32_gluon(bits)
    scales = gl.load(input_nope_s_ptr + ids[:, None] * INPUT_NOPE_S_STRIDE
                     + groups[None, :], valid[:, None], 0)
    gl.store(output_ptr + output_rows.expand_dims(1).expand_dims(2) * OUTPUT_STRIDE + offsets,
             values * scales.expand_dims(2), valid.expand_dims(1).expand_dims(2))
    if DIM_ROPE > 0:
        gl.store(output_ptr + rope_output_rows[:, None] * OUTPUT_STRIDE + 512
                 + rope_offsets[None, :], rope, rope_rows[:, None] < M)
