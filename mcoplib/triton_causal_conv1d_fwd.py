from typing import List, Optional, Union

import torch
import triton
import triton.language as tl


PAD_SLOT_ID = -1


@triton.jit()
def _causal_conv1d_fwd_kernel(  # continuous batching
    # Pointers to matrices
    x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
    w_ptr,  # (dim, width)
    bias_ptr,
    initial_states_ptr,  # conv_states_ptr
    cache_indices_ptr,  # conv_state_indices_ptr
    has_initial_states_ptr,
    query_start_loc_ptr,
    o_ptr,  # (dim, seqlen) - actually pointing to x_ptr
    # Matrix dimensions
    dim: tl.constexpr,
    seqlen: tl.int32,  # cu_seqlen
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,  # stride to get to next sequence,
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
    stride_x_token: tl.constexpr,  # stride to get to next token (same feature-index, same sequence-index)
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = (
        KERNEL_WIDTH - 1
    )  # can be passed via argument if it's not the same as this value

    # one program handles one chunk in a single sequence
    # rather than mixing sequences - to make updating initial_states across sequences efficiently

    # single-sequence id
    idx_seq = tl.program_id(0)
    chunk_offset = tl.program_id(1)

    # BLOCK_N elements along the feature-dimension (channel)
    idx_feats = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    # find the actual sequence length
    seqlen = sequence_end_index - sequence_start_index

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    if segment_len <= 0:
        return

    # base of the sequence
    x_base = (
        x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim
    )  # [BLOCK_N,]

    if IS_CONTINUOUS_BATCHING:
        # cache_idx
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq).to(tl.int64)
    else:
        # cache_idx
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:  # noqa
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return
    conv_states_base = (
        conv_states_ptr
        + (conv_state_batch_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    # Does 2 things:
    # 1. READ prior-block init-state data - [done by every Triton programs]
    # 2. update conv_state with new data [only by the Triton program handles chunk_offset=0]
    if chunk_offset == 0:
        # read from conv_states
        load_init_state = False
        if HAS_INITIAL_STATES:  # the new HAS_INITIAL_STATES
            load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            # load from conv_states
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        else:
            # prior-tokens are zeros
            if KERNEL_WIDTH >= 2:  # STRATEGY1
                # first chunk and does not have prior-token, so just set to 0
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:  # STRATEGY1
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:  # STRATEGY1
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:  # STRATEGY1
                col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        # STEP 2:
        # here prepare data for updating conv_state
        if (
            state_len <= seqlen
        ):  # SMALL_CACHE=True (only move part of 'x' into conv_state cache)
            # just read from 'x'
            # copy 'x' data to conv_state
            # load only 'x' data (and set 0 before 'x' if seqlen < state_len)
            idx_tokens_last = (seqlen - state_len) + tl.arange(
                0, NP2_STATELEN
            )  # [BLOCK_M]
            x_ptrs = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )  # [BLOCK_M,BLOCK_N,]
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
            conv_states_ptrs_target = (
                conv_states_base[None, :]
                + (idx_tokens_conv * stride_conv_state_tok)[:, None]
            )

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
            tl.store(conv_states_ptrs_target, new_conv_state, mask)

        else:
            if load_init_state:
                # update conv_state by shifting left, i.e. take last few cols from conv_state + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_state_batch_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (
                    (conv_state_batch_coord < num_cache_lines)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                tl.debug_barrier()  # need this due to the bug in tl.where not enforcing this when data is the result of another tl.load
                new_conv_state = tl.where(
                    mask, conv_state, loaded_x
                )  # BUG in 'tl.where'  which requires a barrier before this
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:  # load_init_state == False
                # update conv_state by shifting left, BUT
                # set cols prior to 'x' as zeros + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        # read prior-token data from `x`
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 5:
            # ruff: noqa: F841
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 3 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token  # starting of chunk

    # PRE-LOAD WEIGHTS
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):

            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr
            + (sequence_start_index + token_offset + idx_token) * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)


@triton.jit()
def _causal_conv1d_fwd_kernel_optimized(
    x_ptr,
    w_ptr,
    bias_ptr,
    conv_states_ptr,
    cache_indices_ptr,
    has_initial_states_ptr,
    query_start_loc_ptr,
    o_ptr,
    dim: tl.constexpr,
    num_cache_lines: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_state_seq: tl.constexpr,
    stride_state_dim: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Width-4 streaming kernel tuned for MetaX C600U."""
    state_len: tl.constexpr = 3
    idx_seq = tl.program_id(0)
    idx_chunk = tl.program_id(1)
    idx_feats = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)
    feat_mask = idx_feats < dim

    sequence_start = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end = tl.load(query_start_loc_ptr + idx_seq + 1)
    sequence_len = sequence_end - sequence_start
    token_offset = idx_chunk * BLOCK_M
    segment_len = min(BLOCK_M, sequence_len - token_offset)
    if segment_len <= 0:
        return

    if IS_CONTINUOUS_BATCHING:
        state_batch = tl.load(cache_indices_ptr + idx_seq).to(tl.int64)
    else:
        state_batch = idx_seq
    if USE_PAD_SLOT:
        if state_batch == pad_slot_id:
            return

    x_base = x_ptr + sequence_start * stride_x_token + idx_feats * stride_x_dim
    state_base = (
        conv_states_ptr
        + state_batch * stride_state_seq
        + idx_feats * stride_state_dim
    )

    if idx_chunk == 0:
        load_initial_state = False
        if HAS_INITIAL_STATES:
            load_initial_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_initial_state:
            col0 = tl.load(
                state_base + 0 * stride_state_token, feat_mask, other=0.0
            )
            col1 = tl.load(
                state_base + 1 * stride_state_token, feat_mask, other=0.0
            )
            col2 = tl.load(
                state_base + 2 * stride_state_token, feat_mask, other=0.0
            )
        else:
            col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        # Update the cache once per sequence, including the short-sequence case.
        idx_state = tl.arange(0, 4)
        source_token = sequence_len - state_len + idx_state
        update_x = tl.load(
            x_base[None, :] + source_token[:, None] * stride_x_token,
            mask=(source_token >= 0)[:, None]
            & (idx_state < state_len)[:, None]
            & feat_mask[None, :],
            other=0.0,
        )
        update_old = tl.load(
            state_base[None, :]
            + (idx_state + sequence_len)[:, None] * stride_state_token,
            mask=(source_token < 0)[:, None]
            & (idx_state < state_len)[:, None]
            & feat_mask[None, :]
            & load_initial_state,
            other=0.0,
        )
        tl.debug_barrier()
        tl.store(
            state_base[None, :] + idx_state[:, None] * stride_state_token,
            update_x + update_old,
            mask=(idx_state < state_len)[:, None] & feat_mask[None, :],
        )
    else:
        prior = x_base + (token_offset - 1) * stride_x_token
        col2 = tl.load(prior, feat_mask, other=0.0, cache_modifier=".ca")
        col1 = tl.load(
            prior - stride_x_token,
            feat_mask,
            other=0.0,
            cache_modifier=".ca",
        )
        col0 = tl.load(
            prior - 2 * stride_x_token,
            feat_mask,
            other=0.0,
            cache_modifier=".ca",
        )

    w_base = w_ptr + idx_feats * stride_w_dim
    w0 = tl.load(w_base + 0 * stride_w_width, feat_mask, other=0.0)
    w1 = tl.load(w_base + 1 * stride_w_width, feat_mask, other=0.0)
    w2 = tl.load(w_base + 2 * stride_w_width, feat_mask, other=0.0)
    w3 = tl.load(w_base + 3 * stride_w_width, feat_mask, other=0.0)
    if HAS_BIAS:
        acc_preload = tl.load(
            bias_ptr + idx_feats, mask=feat_mask, other=0.0
        ).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    current_base = x_base + token_offset * stride_x_token
    if segment_len == BLOCK_M:
        # Production chunks use a fixed trip count.  This removes dynamic loop
        # control and the second-token tail predicate.
        for idx_token in tl.static_range(0, BLOCK_M, 2):
            current0 = tl.load(
                current_base + idx_token * stride_x_token,
                mask=feat_mask,
                other=0.0,
            )
            acc0 = acc_preload
            acc0 += col0 * w0
            acc0 += col1 * w1
            acc0 += col2 * w2
            acc0 += current0 * w3
            if SILU_ACTIVATION:
                acc0 = acc0 / (1.0 + tl.exp(-acc0))
            tl.store(
                o_ptr
                + (sequence_start + token_offset + idx_token) * stride_o_token
                + idx_feats * stride_o_dim,
                acc0,
                mask=feat_mask,
            )

            current1 = tl.load(
                current_base + (idx_token + 1) * stride_x_token,
                mask=feat_mask,
                other=0.0,
            )
            acc1 = acc_preload
            acc1 += col1 * w0
            acc1 += col2 * w1
            acc1 += current0 * w2
            acc1 += current1 * w3
            col0 = col2
            col1 = current0
            col2 = current1
            if SILU_ACTIVATION:
                acc1 = acc1 / (1.0 + tl.exp(-acc1))
            tl.store(
                o_ptr
                + (sequence_start + token_offset + idx_token + 1)
                * stride_o_token
                + idx_feats * stride_o_dim,
                acc1,
                mask=feat_mask,
            )
    else:
        # Only the final partial chunk needs dynamic loop and tail masks.
        for idx_token in range(0, segment_len, 2):
            current0 = tl.load(
                current_base + idx_token * stride_x_token,
                mask=feat_mask,
                other=0.0,
            )
            acc0 = acc_preload
            acc0 += col0 * w0
            acc0 += col1 * w1
            acc0 += col2 * w2
            acc0 += current0 * w3
            col0 = col1
            col1 = col2
            col2 = current0
            if SILU_ACTIVATION:
                acc0 = acc0 / (1.0 + tl.exp(-acc0))
            tl.store(
                o_ptr
                + (sequence_start + token_offset + idx_token) * stride_o_token
                + idx_feats * stride_o_dim,
                acc0,
                mask=feat_mask,
            )

            valid1 = idx_token + 1 < segment_len
            current1 = tl.load(
                current_base + (idx_token + 1) * stride_x_token,
                mask=valid1 & feat_mask,
                other=0.0,
            )
            acc1 = acc_preload
            acc1 += col0 * w0
            acc1 += col1 * w1
            acc1 += col2 * w2
            acc1 += current1 * w3
            col0 = col1
            col1 = col2
            col2 = current1
            if SILU_ACTIVATION:
                acc1 = acc1 / (1.0 + tl.exp(-acc1))
            tl.store(
                o_ptr
                + (sequence_start + token_offset + idx_token + 1)
                * stride_o_token
                + idx_feats * stride_o_dim,
                acc1,
                mask=valid1 & feat_mask,
            )


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: List[int],
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    validate_data=False,
    **kwargs,
):
    """support varlen + continuous batching when x is 2D tensor

    x: (dim,cu_seq_len)
        cu_seq_len = total tokens of all seqs in that batch
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
        [it use `cache_indices` to get the index to the cache of conv_state for that sequence

        conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True
             and after that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x'
        ]
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        if
        x = [5, 1, 1, 1] <- continuous batching (batch=4)
        then
        query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is
           the ending index of the last sequence
        [length(query_start_loc)-1 == batch]
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    seq_lens_cpu: (batch) int32
        The sequence lengths of the sequences in the batch
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
        [single boolean for each sequence in the batch: True or False]
    bias: (dim,)
    activation: either None or "silu" or "swish" or True
    pad_slot_id: int
        if cache_indices is passed, lets the kernel identify padded
        entries that will not be processed,
        for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at
        indices 0 and 3

    out: same shape as `x`
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"

    out = torch.empty_like(x)

    is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
    dim, cu_seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    stride_x_seq = 0
    stride_x_dim = x.stride(0)
    stride_x_token = x.stride(1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    stride_istate_seq = 0
    stride_istate_dim = 0
    stride_istate_token = 0
    num_cache_lines = 0
    if conv_states is not None:
        # extensions to support vLLM:
        # 1. conv_states is used to replaced initial_states
        # 2. conv_states serve as a cache with num cache lines can be larger than batch size
        # 3. mapping from sequence x[idx] to a cache line at index as specified via cache_indices[idx]
        # 4. computation can be skipped if cache_indices[idx] == pad_slot_id
        num_cache_lines = conv_states.size(0)
        assert (
            num_cache_lines == conv_states.shape[0]
            and dim == conv_states.shape[1]
            and width - 1 <= conv_states.shape[2]
        )
        stride_istate_seq = conv_states.stride(0)
        stride_istate_dim = conv_states.stride(1)
        stride_istate_token = conv_states.stride(2)
        # assert stride_istate_dim == 1
    if out.dim() == 2:
        stride_o_seq = 0
        stride_o_dim = out.stride(0)
        stride_o_token = out.stride(1)
    else:
        stride_o_seq = out.stride(0)
        stride_o_dim = out.stride(1)
        stride_o_token = out.stride(2)

    if validate_data:
        assert x.dim() == 2
        assert query_start_loc is not None
        assert query_start_loc.dim() == 1
        assert x.stride(0) == 1 or x.stride(1) == 1
        padded_batch = query_start_loc.size(0) - 1
        if bias is not None:
            assert bias.dim() == 1
            assert dim == bias.size(0)
        if cache_indices is not None:
            assert cache_indices.dim() == 1
            assert padded_batch == cache_indices.size(0)
        if has_initial_state is not None:
            assert has_initial_state.size() == (padded_batch,)
            assert (
                conv_states is not None
            ), "ERROR: `has_initial_state` is used, which needs also `conv_states`"
        assert weight.stride(1) == 1
        assert (dim, width) == weight.shape
        assert is_channel_last, "Need to run in channel-last layout"

    def grid(META):
        max_seq_len = max(seq_lens_cpu)
        return (
            len(seq_lens_cpu),  # batch_size
            (max_seq_len + META["BLOCK_M"] - 1) // META["BLOCK_M"],
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_fwd_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        out,
        # Matrix dimensions
        dim,
        cu_seqlen,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_INITIAL_STATES=has_initial_state is not None,
        HAS_CACHE=conv_states is not None,
        IS_CONTINUOUS_BATCHING=cache_indices is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        # launch_cooperative_grid=True
        BLOCK_M=8,
        BLOCK_N=256,
        num_stages=2,
    )
    return out


def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Union[torch.Tensor, None],
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens_cpu: List[int],
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    validate_data: bool = False,
    **kwargs,
):
    if weight.shape[1] != 4 or conv_states is None:
        return causal_conv1d_fn(
            x=x,
            weight=weight,
            bias=bias,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            seq_lens_cpu=seq_lens_cpu,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation=activation,
            pad_slot_id=pad_slot_id,
            validate_data=validate_data,
            **kwargs,
        )

    if isinstance(activation, bool) and activation:
        activation = "silu"

    dim, _ = x.shape
    out = torch.empty_like(x)
    num_cache_lines = conv_states.shape[0]

    if validate_data:
        padded_batch = query_start_loc.size(0) - 1
        assert x.dim() == 2
        assert query_start_loc.dim() == 1
        assert x.stride(0) == 1, "Need to run in channel-last layout"
        assert weight.shape == (dim, 4)
        assert weight.stride(1) == 1
        assert conv_states.shape[1] == dim
        assert conv_states.shape[2] >= 3
        if bias is not None:
            assert bias.shape == (dim,)
        if cache_indices is not None:
            assert cache_indices.shape == (padded_batch,)
        if has_initial_state is not None:
            assert has_initial_state.shape == (padded_batch,)

    
    default_num_warps = 2

    num_warps = kwargs.pop("_num_warps", default_num_warps)
    def grid(meta):
        return (
            len(seq_lens_cpu),
            triton.cdiv(max(seq_lens_cpu), meta["BLOCK_M"]),
            triton.cdiv(dim, meta["BLOCK_N"]),
        )

    _causal_conv1d_fwd_kernel_optimized[grid](
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        out,
        dim,
        num_cache_lines,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        out.stride(0),
        out.stride(1),
        pad_slot_id,
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_INITIAL_STATES=has_initial_state is not None,
        IS_CONTINUOUS_BATCHING=cache_indices is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_M=32,
        BLOCK_N=512,
        num_warps=num_warps,
        num_stages=2,
    )
    return out
