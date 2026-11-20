"""C600-U-only explicit ownership for two-row GLM paged dequantization."""

import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl


@gluon.jit
def _dequantize_k_cache_c600u_rows2_kernel(
    output_ptr, input_nope_q_ptr, input_nope_s_ptr, input_rope_ptr, page_table_ptr,
    M, OUTPUT_STRIDE: gl.constexpr, INPUT_NOPE_Q_STRIDE: gl.constexpr,
    INPUT_NOPE_S_STRIDE: gl.constexpr, INPUT_ROPE_STRIDE: gl.constexpr,
    DIM_ROPE: gl.constexpr, USE_INT64_INPUT_INDEX: gl.constexpr,
    USE_INT64_OUTPUT_INDEX: gl.constexpr,
):
    # Two rows x two groups x sixteen lanes; consume two group chunks.
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 8], [2, 2, 16], [1, 1, 1], [2, 1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, layout))
    group_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, layout))
    column_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(0, layout))
    rows = gl.program_id(0) * 2 + gl.arange(0, 2, layout=row_layout)
    valid = rows < M
    ids = gl.load(page_table_ptr + rows, valid, 0)
    ids = ids.to(gl.int64) if USE_INT64_INPUT_INDEX else ids.to(gl.int32)
    output_rows = rows.to(gl.int64) if USE_INT64_OUTPUT_INDEX else rows
    groups = gl.arange(0, 2, layout=group_layout)
    elements = gl.arange(0, 128, layout=column_layout)
    if DIM_ROPE > 0:
        rope_layout: gl.constexpr = gl.BlockedLayout([1, 2], [2, 32], [1, 1], [1, 0])
        rope_row_layout: gl.constexpr = gl.SliceLayout(1, rope_layout)
        rope_rows = gl.program_id(0) * 2 + gl.arange(0, 2, layout=rope_row_layout)
        rope_ids = gl.convert_layout(ids, rope_row_layout)
        rope_output_rows = rope_rows.to(gl.int64) if USE_INT64_OUTPUT_INDEX else rope_rows
        rope_offsets = gl.arange(0, DIM_ROPE, layout=gl.SliceLayout(0, rope_layout))
        rope = gl.load(input_rope_ptr + rope_ids[:, None] * INPUT_ROPE_STRIDE
                       + rope_offsets[None, :], rope_rows[:, None] < M, 0)
    for chunk in gl.static_range(0, 2):
        chunk_groups = groups + chunk * 2
        offsets = chunk_groups.expand_dims(0).expand_dims(2) * 128 + elements.expand_dims(0).expand_dims(0)
        quantized = gl.load(input_nope_q_ptr + ids.expand_dims(1).expand_dims(2) * INPUT_NOPE_Q_STRIDE
                            + offsets, valid.expand_dims(1).expand_dims(2), 0.0).to(gl.float32)
        scales = gl.load(input_nope_s_ptr + ids[:, None] * INPUT_NOPE_S_STRIDE
                         + chunk_groups[None, :], valid[:, None], 0)
        gl.store(output_ptr + output_rows.expand_dims(1).expand_dims(2) * OUTPUT_STRIDE + offsets,
                 quantized * scales.expand_dims(2), valid.expand_dims(1).expand_dims(2))
    if DIM_ROPE > 0:
        gl.store(output_ptr + rope_output_rows[:, None] * OUTPUT_STRIDE + 512
                 + rope_offsets[None, :], rope, rope_rows[:, None] < M)


@gluon.jit
def _dequantize_k_cache_c600u_pairtiles_kernel(
    output_ptr, input_nope_q_ptr, input_nope_s_ptr, input_rope_ptr, page_table_ptr,
    M, OUTPUT_STRIDE: gl.constexpr, INPUT_NOPE_Q_STRIDE: gl.constexpr,
    INPUT_NOPE_S_STRIDE: gl.constexpr, INPUT_ROPE_STRIDE: gl.constexpr,
    DIM_ROPE: gl.constexpr, USE_INT64_INPUT_INDEX: gl.constexpr,
    USE_INT64_OUTPUT_INDEX: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 8], [2, 2, 16], [1, 1, 1], [2, 1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, layout))
    group_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, layout))
    column_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(0, layout))
    if DIM_ROPE > 0:
        rope_layout: gl.constexpr = gl.BlockedLayout([1, 2], [2, 32], [1, 1], [1, 0])
        rope_row_layout: gl.constexpr = gl.SliceLayout(1, rope_layout)
    # Consume two independent DU8 row pairs without expanding four rows at once.
    for pair_tile in gl.static_range(0, 2):
        # Two rows x two groups x sixteen lanes; consume two group chunks.
        rows = gl.program_id(0) * 4 + pair_tile * 2 + gl.arange(0, 2, layout=row_layout)
        valid = rows < M
        ids = gl.load(page_table_ptr + rows, valid, 0)
        ids = ids.to(gl.int64) if USE_INT64_INPUT_INDEX else ids.to(gl.int32)
        output_rows = rows.to(gl.int64) if USE_INT64_OUTPUT_INDEX else rows
        groups = gl.arange(0, 2, layout=group_layout)
        elements = gl.arange(0, 128, layout=column_layout)
        if DIM_ROPE > 0:
            rope_rows = gl.program_id(0) * 4 + pair_tile * 2 + gl.arange(0, 2, layout=rope_row_layout)
            rope_ids = gl.convert_layout(ids, rope_row_layout)
            rope_output_rows = rope_rows.to(gl.int64) if USE_INT64_OUTPUT_INDEX else rope_rows
            rope_offsets = gl.arange(0, DIM_ROPE, layout=gl.SliceLayout(0, rope_layout))
            rope = gl.load(input_rope_ptr + rope_ids[:, None] * INPUT_ROPE_STRIDE
                           + rope_offsets[None, :], rope_rows[:, None] < M, 0)
        for chunk in gl.static_range(0, 2):
            chunk_groups = groups + chunk * 2
            offsets = chunk_groups.expand_dims(0).expand_dims(2) * 128 + elements.expand_dims(0).expand_dims(0)
            quantized = gl.load(input_nope_q_ptr + ids.expand_dims(1).expand_dims(2) * INPUT_NOPE_Q_STRIDE
                                + offsets, valid.expand_dims(1).expand_dims(2), 0.0).to(gl.float32)
            scales = gl.load(input_nope_s_ptr + ids[:, None] * INPUT_NOPE_S_STRIDE
                             + chunk_groups[None, :], valid[:, None], 0)
            gl.store(output_ptr + output_rows.expand_dims(1).expand_dims(2) * OUTPUT_STRIDE + offsets,
                     quantized * scales.expand_dims(2), valid.expand_dims(1).expand_dims(2))
        if DIM_ROPE > 0:
            gl.store(output_ptr + rope_output_rows[:, None] * OUTPUT_STRIDE + 512
                     + rope_offsets[None, :], rope, rope_rows[:, None] < M)
