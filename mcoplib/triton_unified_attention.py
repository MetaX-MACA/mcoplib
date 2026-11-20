# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone paged Triton unified attention, including its shared helpers.

Extracted from vllm/v1/attention/ops/{triton_unified_attention,
triton_attention_helpers}.py. Requires only PyTorch and a compatible Triton
runtime; no project installation or plugin registration is needed.
INT4 dispatch to a separate upstream operator is outside this extraction.
Set UNIFIED_ATTENTION_BATCH_INVARIANT=1 to force the 2D path.
"""

# Original kernel authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import os
from enum import IntEnum
from typing import Any

import torch
import triton
import triton.language as tl


class KVQuantMode(IntEnum):
    """Quantization modes referenced by this operator, with upstream values."""

    NONE = 0
    FP8_PER_TENSOR = 1
    INT8_PER_TOKEN_HEAD = 2
    FP8_PER_TOKEN_HEAD = 3
    INT4_PER_TOKEN_HEAD = 4


def _fp8_dtype() -> torch.dtype:
    if torch.version.hip and torch.cuda.is_available():
        arch = torch.cuda.get_device_properties(0).gcnArchName
        if "gfx94" in arch:
            return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


FP8_DTYPE = _fp8_dtype()
float8_info = torch.finfo(FP8_DTYPE)
is_batch_invariant = bool(int(os.getenv("UNIFIED_ATTENTION_BATCH_INVARIANT", "0")))


@triton.jit
def cdiv_fn(x, y):
    """Ceiling division.  Kept as a helper to keep kernel bodies terse."""
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    """Softcap (aka tanh-style clamp) used to bound attention scores.

    ``x * tanh(S / x)`` computed without a direct ``tanh`` call.

    Use ``tanh(y) = 1 - 2 / (exp(2y) + 1)`` rather than
    ``(exp(y) - exp(-y)) / (exp(y) + exp(-y))``. The symmetric-exponential
    form overflows to ``inf`` for ``|y| > ~88`` and then evaluates to
    ``inf / inf = NaN``, poisoning the attention row. The single-exponential
    form saturates cleanly at +/-1 for either sign.
    """
    y = S / x
    return x * (1.0 - 2.0 / (tl.exp(2.0 * y) + 1.0))


# ===========================================================================
# Attention loop
# ===========================================================================


@triton.jit
def resolve_seq_and_query_len(
    query_start_len_ptr,
    seq_lens_ptr,
    q_block_global_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
):
    """Resolve the (sequence, q-block-within-sequence) pair and load the
    per-sequence lengths.

    Shared across every attention kernel — the ``q_block_global_idx``
    program id indexes into the flattened ``(seq, q_block_in_seq)``
    space, and a binary search over ``query_start_len_ptr`` recovers
    the (seq, local-q-block) pair.

    Returns ``(seq_idx, q_block_local_idx, cur_batch_in_all_start_index,
    cur_batch_query_len, seq_len)``.  Callers must still early-return
    when ``q_block_local_idx * BLOCK_Q >= cur_batch_query_len`` (Triton
    helpers cannot return from the caller).
    """
    # find_seq_idx is defined below; forward use is fine inside @triton.jit.
    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx
    cur_start = tl.load(query_start_len_ptr + seq_idx)
    cur_stop = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_stop - cur_start
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    return seq_idx, q_block_local_idx, cur_start, cur_batch_query_len, seq_len


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    """Binary search over the cumulative query-length prefix.

    When ``use_q_block_mode`` is True, the prefix values are reshaped
    into units of ``BLOCK_Q`` plus one entry per boundary — matching
    the q-block grid laid out by the attention kernels.  When False
    we search the plain cumulative-length prefix (used by
    ``reduce_segments`` which iterates over raw query tokens).
    """
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def init_softmax_M(
    sink_ptr,
    query_offset_1,
    query_mask_1,
    segm_idx_or_0,
    BLOCK_M: tl.constexpr,
    USE_SINKS: tl.constexpr,
    IS_3D: tl.constexpr,
):
    """Initial row-max ``M`` for the online softmax.

    Without sinks: ``-inf``.  With sinks: load the per-head sink bias
    once.  In 3D mode only segment 0 loads — ``reduce_segments`` adds
    the sink contribution exactly once across segments, so other
    segments must start from ``-inf``.

    ``segm_idx_or_0`` is the 3D segment index or 0 for 2D (caller
    passes ``0`` when ``IS_3D`` is False).
    """
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    if USE_SINKS:
        load_sinks = (not IS_3D) or (segm_idx_or_0 == 0)
        if load_sinks:
            M = tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(tl.float32)
    return M


@triton.jit
def compute_tile_loop_bounds(
    context_len,
    seq_len,
    cur_batch_query_len,
    q_block_local_idx,
    segm_idx_or_0,
    tiles_per_segment_or_0,
    TILE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    IS_3D: tl.constexpr,
    USE_CAUSAL: tl.constexpr = True,
    USE_PER_SEQ_CAUSAL: tl.constexpr = False,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
    USE_R_SWA: tl.constexpr = False,
    MM_PREFIX_CLAMP_SW: tl.constexpr = False,
    MAX_MM_RANGES: tl.constexpr = 0,
    mm_prefix_range_ptr=None,
    seq_idx=0,
):
    """Compute the tile-loop bounds ``(loop_lo, loop_hi)`` and the
    derived ``max_seq_prefix_len`` used for per-tile masking.

    Combines three concerns into one helper:

    1. Longest prefix spanned by any query token in this q-block.
       Clamped to ``seq_len`` (causal) or extended to it when
       mm_prefix is active or non-causal sequences need the full
       sequence.
    2. Sliding-window pruning: narrows ``[tile_start, tile_end)`` to
       only tiles that can contain an allowed key under SWA. For non-causal
       sequences, the window extends in both directions. For Gemma4's
       window-clamped multimodal prefix mask, the bounds include the union of
       the base mask and each image range intersecting the query block.
    3. 3D scoping: when ``IS_3D`` is True, further narrows to the
       segment's slice via ``(segm_idx * tiles_per_segment,
       (segm_idx + 1) * tiles_per_segment)``.
    """
    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    if USE_MM_PREFIX or USE_R_SWA or USE_PER_SEQ_CAUSAL or (not USE_CAUSAL):
        # Read the full sequence but never past seq_len: the causal-style
        # formula above can overshoot for non-causal sequences, and slots
        # >= seq_len are unwritten KV (last-block tail) that may hold NaN
        # (0 * NaN poisons the output). Per-element masking in
        # compute_kv_seq_mask handles the causal/non-causal boundary.
        max_seq_prefix_len = seq_len
    else:
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    # Prefix ranges normally override the sliding window, so they require the
    # complete sequence. Gemma4 instead clamps prefix attention to the left
    # edge of the sliding window. In that case, the union of possible keys is
    # bounded by the normal left window and the end of any multimodal range
    # intersecting this query block.
    can_prune_sliding = (not USE_R_SWA) and ((not USE_MM_PREFIX) or MM_PREFIX_CLAMP_SW)
    if SLIDING_WINDOW > 0 and can_prune_sliding:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        q_abs = context_len + qpos_lo
        if CHUNK_LOOKBACK > -1:
            first_allowed_key = ((q_abs // CHUNK_SIZE) - CHUNK_LOOKBACK) * CHUNK_SIZE
        else:
            first_allowed_key = q_abs - SLIDING_WINDOW + 1
        if USE_PER_SEQ_CAUSAL or (not USE_CAUSAL):
            # Non-causal: keys can be AHEAD of query within the window
            last_allowed_key = context_len + qpos_hi + SLIDING_WINDOW - 1
        else:
            last_allowed_key = context_len + qpos_hi
        if USE_MM_PREFIX and MM_PREFIX_CLAMP_SW:
            query_abs_lo = context_len + qpos_lo
            query_abs_hi = context_len + qpos_hi
            for i in range(MAX_MM_RANGES):
                range_start = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
                )
                range_end = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
                )
                intersects_query_block = (
                    (range_start < range_end)
                    & (range_start <= query_abs_hi)
                    & (range_end >= query_abs_lo)
                )
                mm_first_allowed_key = tl.maximum(
                    range_start, query_abs_lo - SLIDING_WINDOW + 1
                )
                first_allowed_key = tl.minimum(
                    first_allowed_key,
                    tl.where(
                        intersects_query_block,
                        mm_first_allowed_key,
                        first_allowed_key,
                    ),
                )
                last_allowed_key = tl.maximum(
                    last_allowed_key,
                    tl.where(intersects_query_block, range_end, last_allowed_key),
                )
        last_allowed_key = tl.minimum(last_allowed_key, seq_len - 1)
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    if IS_3D:
        loop_lo = max(segm_idx_or_0 * tiles_per_segment_or_0, tile_start)
        loop_hi = min((segm_idx_or_0 + 1) * tiles_per_segment_or_0, tile_end)
    else:
        loop_lo = tile_start
        loop_hi = tile_end

    return loop_lo, loop_hi, max_seq_prefix_len


@triton.jit
def store_segm_reduce_scalars(
    segm_max_ptr,
    segm_expsum_ptr,
    query_offset_0,
    query_offset_1,
    segm_idx,
    M,
    L,
    query_mask_0,
    query_mask_1,
    num_query_heads: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    """Store per-segment ``M`` and ``L`` for ``reduce_segments`` to
    combine into the final softmax.

    Shared across every 3D attention epilogue; the per-token output
    stripes are mode-specific (flat / 2-stream split / 4-stream split)
    and stay inlined.
    """
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def compute_kv_seq_mask(
    query_abs_pos,
    seq_offset,
    seq_idx,
    seq_len,
    mm_prefix_range_ptr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    USE_CAUSAL: tl.constexpr = True,
    USE_PER_SEQ_CAUSAL: tl.constexpr = False,
    per_seq_causal_ptr=None,
    rswa_prefix_lens_ptr=None,
    R_SWA_WINDOW: tl.constexpr = 0,
    USE_R_SWA: tl.constexpr = False,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
    MM_PREFIX_CLAMP_SW: tl.constexpr = False,
):
    """Build the KV mask for one tile.

    Causal (key <= query) by default; AND-ed with either chunked
    attention (``CHUNK_LOOKBACK >= 0``) or sliding window
    (``SLIDING_WINDOW > 0``); OR-ed with the bidirectional ranges from
    ``mm_prefix_range`` when PrefixLM / multimodal attention is active.
    Order matches FlexAttention: ``(causal AND window) OR mm_prefix``.
    Chunked attention takes precedence over sliding window when both
    are non-default — the launcher zeros ``CHUNK_LOOKBACK`` whenever
    sliding window is disabled.

    When ``USE_PER_SEQ_CAUSAL`` is set, each sequence carries its own
    causal flag via ``per_seq_causal_ptr``; non-causal sequences use a
    simple ``key < seq_len`` bound instead.  ``USE_CAUSAL=False``
    disables causal masking entirely.
    """
    if USE_PER_SEQ_CAUSAL:
        is_causal = tl.load(per_seq_causal_ptr + seq_idx)
        seq_mask = tl.where(
            is_causal,
            seq_offset[None, :] <= query_abs_pos,
            seq_offset[None, :] < seq_len,
        )
    elif USE_CAUSAL:
        seq_mask = seq_offset[None, :] <= query_abs_pos
    else:
        seq_mask = seq_offset[None, :] < seq_len

    # Apply sliding window / chunked attention to base mask
    # BEFORE mm_prefix OR.
    # Order must match FlexAttention:
    #   (causal AND sliding_window) OR mm_prefix
    if CHUNK_LOOKBACK > -1:
        seq_mask = seq_mask & (
            (query_abs_pos // CHUNK_SIZE - seq_offset[None, :] // CHUNK_SIZE)
            <= CHUNK_LOOKBACK
        )
    elif SLIDING_WINDOW > 0 and not USE_R_SWA:
        sw_left = (query_abs_pos - seq_offset) < SLIDING_WINDOW
        if USE_PER_SEQ_CAUSAL:
            sw_right = (seq_offset[None, :] - query_abs_pos) < SLIDING_WINDOW
            seq_mask = seq_mask & tl.where(is_causal, sw_left, sw_left & sw_right)
        elif not USE_CAUSAL:
            sw_right = (seq_offset[None, :] - query_abs_pos) < SLIDING_WINDOW
            seq_mask = seq_mask & sw_left & sw_right
        else:
            seq_mask = seq_mask & sw_left

    if USE_R_SWA:
        prefix_len = tl.load(rswa_prefix_lens_ptr + seq_idx)
        in_prefix = seq_offset[None, :] < prefix_len
        in_window = (query_abs_pos - seq_offset) < R_SWA_WINDOW
        seq_mask = seq_mask & (in_prefix | in_window)

    # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
    # Default (MM_PREFIX_CLAMP_SW=False): applied AFTER sliding window so
    # mm_prefix ranges override the SW restriction -> (causal AND SW) OR mm.
    # Gemma4 (MM_PREFIX_CLAMP_SW=True): the bidirectional image block must stay
    # within the sliding window, matching HF's (causal OR blockwise) AND
    # sliding_window. We AND each range with the SW past-bound
    # (query_abs_pos - seq_offset) < SLIDING_WINDOW (== HF sliding_window_overlay
    # kv > q - sw; future kv passes trivially). Inert for full-attention layers
    # (SLIDING_WINDOW <= 0).
    if USE_MM_PREFIX:
        for i in range(MAX_MM_RANGES):
            range_start = tl.load(
                mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
            )
            range_end = tl.load(
                mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
            )
            is_valid = range_start < range_end
            q_in_range = (
                (query_abs_pos >= range_start) & (query_abs_pos <= range_end) & is_valid
            )
            k_in_range = (
                (seq_offset[None, :] >= range_start)
                & (seq_offset[None, :] <= range_end)
                & is_valid
            )
            mm_mask = q_in_range & k_in_range
            if MM_PREFIX_CLAMP_SW and SLIDING_WINDOW > 0:
                mm_mask = mm_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)
            seq_mask |= mm_mask
    return seq_mask


@triton.jit
def apply_alibi_to_score(
    S,
    alibi_slope,
    seq_offset,
    context_len,
    query_pos,
    USE_ALIBI_SQRT: tl.constexpr,
):
    """Add the ALiBi positional bias (linear or sqrt variant) to S in-place."""
    if USE_ALIBI_SQRT:
        relative_pos = seq_offset - (context_len + query_pos[:, None])
        alibi_offset = tl.where(
            relative_pos <= 0,
            -tl.sqrt((-relative_pos).to(tl.float32)),
            0.0,
        )
    else:
        alibi_offset = seq_offset - context_len
    return S + alibi_slope[:, None] * alibi_offset


@triton.jit
def load_qq_bias_tile(
    qq_bias_row_ptrs,
    seq_offset,
    context_len,
    qq_bias_stride_0,
):
    """Load the qq-bias slice for keys that correspond to query rows."""
    key_rel_pos = seq_offset - context_len
    is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
    return tl.load(
        qq_bias_row_ptrs + key_rel_pos[None, :],
        mask=is_query_key[None, :],
        other=0.0,
    )


@triton.jit
def softmax_step(S, M, L):
    """Online softmax update for one tile.

    Returns ``(M_new, L_new, P, alpha)``.  Caller is responsible for
    rescaling its accumulator(s) by ``alpha[:, None]`` — done outside so
    kernels with a different number / shape of accumulators can reuse
    the same step.
    """
    # compute running maximum
    # m_j : (BLOCK_M,)
    m_j = tl.maximum(M, tl.max(S, axis=1))
    # For sliding window there's a chance the max is -inf due to masking of
    # the entire row. In this case we need to set m_j 0 to avoid NaN
    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    # P : (BLOCK_M, TILE_SIZE)
    P = tl.exp(S - m_j[:, None])
    # l_j : (BLOCK_M,)
    l_j = tl.sum(P, axis=1)
    # alpha : (BLOCK_M, )
    alpha = tl.exp(M - m_j)
    # update constants
    L_new = L * alpha + l_j
    return m_j, L_new, P, alpha


@triton.jit
def _cast_kv_tile(data, Q, tensor_scale, KV_QUANT_MODE: tl.constexpr):
    """Cast a loaded KV tile to Q's dtype, dequantizing if needed.

    Modes handled inside the core kernel:

    - ``KV_QUANT_MODE == 0`` (NONE) and ``2`` (INT8 per-token-head) and
      ``3`` (FP8 per-token-head): plain cast.  Per-token-head modes apply
      their scales separately on S/P inside the loop.
    - ``KV_QUANT_MODE == 1`` (FP8 per-tensor): dequantize using the
      tensor-wide scale, unless Q is also FP8 and the caller folds the scales
      into the attention score and output accumulator.
    """
    if KV_QUANT_MODE == 1:
        if Q.dtype.is_fp8():
            return data.to(Q.dtype)
        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype)
    return data.to(Q.dtype)


# ---------------------------------------------------------------------------
# Tensor-descriptor (TD) helpers
#
# Used when the caller enables ``USE_TD`` (Intel Xe2/Xe3 HW 2D block reads).
# When ``USE_TD`` is False the helpers are dead-code-eliminated at Triton
# compile time, leaving the pointer-arithmetic load/store path untouched.
# ---------------------------------------------------------------------------


@triton.jit
def _load_q_td(
    query_ptr,
    q_block_local_len,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    cur_batch_in_all_start_index,
    q_block_local_idx,
    kv_head_idx,
    num_queries_per_kv: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    """Load Q via a 2D tensor descriptor.

    Caller guarantees (via the wrapper's ``use_td_qo`` gate):
      * ``HEAD_SIZE == HEAD_SIZE_PADDED`` (head_size is a power of 2),
      * ``num_queries_per_kv`` is a power of 2,
      * the ``num_queries_per_kv`` heads of the current KV group are
        contiguous in memory (``query_stride_1 == HEAD_SIZE``, which is
        the default vLLM query layout).

    Under those preconditions the inner two axes are flattened into one
    row of size ``num_queries_per_kv * HEAD_SIZE`` with stride 1, which
    avoids the non-power-of-2 ``block_shape`` error from the Triton
    tensor-descriptor validator.  Returns (BLOCK_M, HEAD_SIZE_PADDED).
    """
    q_base = (
        query_ptr
        + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q) * query_stride_0
        + (kv_head_idx * num_queries_per_kv) * query_stride_1
    )
    q_desc = tl.make_tensor_descriptor(
        base=q_base,
        shape=(q_block_local_len, num_queries_per_kv * HEAD_SIZE),
        strides=(query_stride_0, 1),
        block_shape=(BLOCK_Q, num_queries_per_kv * HEAD_SIZE_PADDED),
    )
    return q_desc.load([0, 0]).reshape(BLOCK_M, HEAD_SIZE_PADDED)


@triton.jit
def _load_kv_tile_td(
    cache_ptr,
    physical_block_idx_scalar,
    kv_head_idx,
    offset_in_block,
    stride_cache_0: tl.int64,
    stride_cache_1: tl.int64,
    stride_cache_2: tl.int64,
    stride_cache_3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    """Load a KV cache tile via tensor descriptor.

    Returns shape (TILE_SIZE, HEAD_SIZE_PADDED). Caller transposes for K.
    Tensor descriptors zero-pad reads beyond the shape boundary, so
    ``HEAD_SIZE_PADDED > HEAD_SIZE`` is handled correctly.
    """
    base = (
        cache_ptr
        + physical_block_idx_scalar * stride_cache_0
        + kv_head_idx * stride_cache_2
    )
    desc = tl.make_tensor_descriptor(
        base=base,
        shape=(BLOCK_SIZE, HEAD_SIZE),
        strides=(stride_cache_1, stride_cache_3),
        block_shape=(TILE_SIZE, HEAD_SIZE_PADDED),
    )
    return desc.load([offset_in_block, 0])


@triton.jit
def _store_output_td(
    base_ptr,
    acc,
    q_block_local_len,
    stride_token: tl.int64,
    stride_head: tl.int64,
    num_queries_per_kv: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    """Store an output tile via a tensor descriptor.

    The 2D and 3D epilogues differ only in ``base_ptr`` and the
    ``(stride_token, stride_head)`` pair: 2D writes directly to the
    flat output buffer, 3D writes to a single per-segment slice of
    ``segm_output_ptr``.  Descriptor shape / block_shape / reshape
    are the same in both modes, so share one helper.
    """
    acc = acc.to(base_ptr.dtype.element_ty)
    output_desc = tl.make_tensor_descriptor(
        base=base_ptr,
        shape=(q_block_local_len, num_queries_per_kv, HEAD_SIZE),
        strides=(stride_token, stride_head, 1),
        block_shape=(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED),
    )
    output_desc.store(
        [0, 0, 0],
        acc.reshape(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED),
    )


@triton.jit
def kernel_unified_attention(
    # Output destination for the 2D path.  In 3D mode per-segment partials
    # go to the ``segm_*`` tensors (see bottom of signature) and
    # ``output_ptr`` is unused (callers may pass any non-null pointer).
    output_ptr,
    # Inputs
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    # Scalars
    scale,
    q_scale,
    k_scale,
    v_scale,
    out_scale,
    softcap,
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    USE_CAUSAL: tl.constexpr,  # bool
    USE_PER_SEQ_CAUSAL: tl.constexpr,  # bool
    per_seq_causal_ptr,  # [num_seqs] int32, or None
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,
    rswa_prefix_lens_ptr,
    R_SWA_WINDOW: tl.constexpr,  # int
    USE_R_SWA: tl.constexpr,  # bool
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    USE_FP8: tl.constexpr,
    # Toggles 2D vs 3D layout.  The 2D path runs the full sequence in one
    # tile loop and writes to ``output_ptr``.  The 3D path scopes the loop
    # to ``[segm_idx, segm_idx+1) × tiles_per_segment`` and writes
    # per-segment partials, finalized by ``reduce_segments``.
    IS_3D: tl.constexpr,
    # Parameters below default to None so Triton can skip materialising them
    # on call sites where the corresponding constexpr branch is dead.
    # Credit: @quinnlp identified this as a perf regression source in
    # intel/intel-xpu-backend-for-triton#6758 (review comment r3204641104).
    # Per-segment outputs: used in 3D mode; unused in 2D (IS_3D=False).
    segm_output_ptr=None,
    segm_max_ptr=None,
    segm_expsum_ptr=None,
    # Per-(token, head) scale caches: used iff KV_QUANT_MODE in {2, 3}.
    k_scale_cache_ptr=None,
    v_scale_cache_ptr=None,
    # ``tl.int64`` cannot be combined with a ``None`` default — Triton's JIT
    # rejects ``Optional[tl.int64]`` / ``tl.int64 | None`` at trace time, and
    # plain ``tl.int64 = None`` raises ``TypeError: 'NoneType' object cannot
    # be interpreted as an integer`` when callers omit these arguments.
    # ``int | None`` is the only annotation that lets the wrapper pass
    # ``None`` here so Triton can skip materialising the strides when the
    # ``USE_PER_TOKEN_HEAD_SCALES`` branch is dead.
    stride_ks_blk: int | None = None,
    stride_ks_slot: int | None = None,
    stride_ks_head: int | None = None,
    stride_vs_blk: int | None = None,
    stride_vs_slot: int | None = None,
    stride_vs_head: int | None = None,
    # KV cache quantization mode handled inside this kernel via constexpr
    # branches: NONE (0), FP8_PER_TENSOR (1), INT8_PER_TOKEN_HEAD (2),
    # FP8_PER_TOKEN_HEAD (3). Sub-byte INT4 (4) uses its own
    # int4_per_token_head kernel, not this one.
    KV_QUANT_MODE: tl.constexpr = 0,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    # Chunked / block-local attention.  ``CHUNK_LOOKBACK >= 0`` enables
    # chunked masking (used by Gemma3 block-local layers); takes precedence
    # over ``SLIDING_WINDOW`` inside the helpers.  ``-1`` disables.
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
    # Tensor-descriptor load/store for HW 2D block reads on Intel Xe2/Xe3.
    # ``USE_TD`` gates KV tile loads; ``USE_TD_QO`` separately gates Q/output
    # (see ``unified_attention`` wrapper for the gating rules).
    USE_TD: tl.constexpr = False,
    USE_TD_QO: tl.constexpr = False,
    Q_IS_FP8: tl.constexpr = False,
    # Gemma4: clamp mm_prefix bidirectional ranges by the sliding window
    # instead of letting them override it. Default False preserves the
    # original (causal AND SW) OR mm_prefix behavior for all other models.
    MM_PREFIX_CLAMP_SW: tl.constexpr = False,
):
    # Per-(token, head) scale caches: used iff KV_QUANT_MODE in {2, 3}.
    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = (KV_QUANT_MODE >= 2) and (
        KV_QUANT_MODE <= 3
    )
    USE_FP8_Q_DESCALE: tl.constexpr = KV_QUANT_MODE == 1 and Q_IS_FP8

    if USE_TD:
        tl.static_assert(
            BLOCK_SIZE % TILE_SIZE == 0,
            "USE_TD requires BLOCK_SIZE to be a multiple of TILE_SIZE",
        )

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    # Number of valid query rows in this block (used by TD descriptor
    # shapes, but always computed so the variable stays in scope).
    q_block_local_len = tl.minimum(
        BLOCK_Q, cur_batch_query_len - q_block_local_idx * BLOCK_Q
    )

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    if USE_TD_QO:
        Q = _load_q_td(
            query_ptr,
            q_block_local_len,
            query_stride_0,
            query_stride_1,
            cur_batch_in_all_start_index,
            q_block_local_idx,
            kv_head_idx,
            num_queries_per_kv,
            BLOCK_Q,
            BLOCK_M,
            HEAD_SIZE,
            HEAD_SIZE_PADDED,
        )
    else:
        Q = tl.load(
            query_ptr + query_offset,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            other=0.0,
        )

    block_table_offset = seq_idx * block_table_stride

    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    # acc : (BLOCK_M, HEAD_SIZE_PADDED)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    score_scale = scale
    value_scale = 1.0
    if USE_FP8_Q_DESCALE:
        score_scale = scale * tl.load(q_scale) * tl.load(k_scale)
        value_scale = tl.load(v_scale)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        USE_MM_PREFIX,
        IS_3D,
        USE_CAUSAL,
        USE_PER_SEQ_CAUSAL,
        CHUNK_LOOKBACK,
        CHUNK_SIZE,
        USE_R_SWA,
        MM_PREFIX_CLAMP_SW,
        MAX_MM_RANGES,
        mm_prefix_range_ptr,
        seq_idx,
    )

    # iterate through tiles (now limited to the sliding window range)
    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        if USE_TD:
            # All TILE_SIZE slots within a single KV tile map to one
            # physical block (guaranteed by ``BLOCK_SIZE % TILE_SIZE == 0``
            # from the static_assert above), so load the block index as
            # a scalar instead of a broadcast reduction.
            offset_in_block = (j * TILE_SIZE) % BLOCK_SIZE
            physical_block_scalar = tl.load(
                block_tables_ptr + block_table_offset + (j * TILE_SIZE) // BLOCK_SIZE
            ).to(tl.int64)
            # K : (HEAD_SIZE, TILE_SIZE)
            K_load = _load_kv_tile_td(
                key_cache_ptr,
                physical_block_scalar,
                kv_head_idx,
                offset_in_block,
                stride_k_cache_0,
                stride_k_cache_1,
                stride_k_cache_2,
                stride_k_cache_3,
                BLOCK_SIZE,
                TILE_SIZE,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            ).T
            # V : (TILE_SIZE, HEAD_SIZE)
            V_load = _load_kv_tile_td(
                value_cache_ptr,
                physical_block_scalar,
                kv_head_idx,
                offset_in_block,
                stride_v_cache_0,
                stride_v_cache_1,
                stride_v_cache_2,
                stride_v_cache_3,
                BLOCK_SIZE,
                TILE_SIZE,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_d[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )
            # K : (HEAD_SIZE, TILE_SIZE)
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0.0,
            )
            # V : (TILE_SIZE, HEAD_SIZE)
            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
            )
        K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)
        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)

        # Per-(token, head) scales for INT8 / FP8 per-token-head modes.
        if USE_PER_TOKEN_HEAD_SCALES:
            scale_idx = (
                physical_block_idx * stride_ks_blk
                + (seq_offset % BLOCK_SIZE) * stride_ks_slot
                + kv_head_idx * stride_ks_head
            )
            k_token_head_scales = tl.load(
                k_scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0
            )
            v_scale_idx = (
                physical_block_idx * stride_vs_blk
                + (seq_offset % BLOCK_SIZE) * stride_vs_slot
                + kv_head_idx * stride_vs_head
            )
            v_token_head_scales = tl.load(
                v_scale_cache_ptr + v_scale_idx, mask=tile_mask, other=1.0
            )

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            seq_len,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
            USE_CAUSAL,
            USE_PER_SEQ_CAUSAL,
            per_seq_causal_ptr,
            rswa_prefix_lens_ptr,
            R_SWA_WINDOW,
            USE_R_SWA,
            CHUNK_LOOKBACK,
            CHUNK_SIZE,
            MM_PREFIX_CLAMP_SW,
        )

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: fuse softmax_scale with per-head k_scale
            # to avoid a separate BLOCK_M × TILE_SIZE multiply on S.
            S += tl.dot(Q, K) * (score_scale * k_token_head_scales[None, :])
        else:
            S += score_scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        if USE_QQ_BIAS:
            S += load_qq_bias_tile(
                qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
            )

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            dist = context_len + qpos_lo - seq_offset[:, None]
            if USE_PER_SEQ_CAUSAL:
                is_causal_seq = tl.load(per_seq_causal_ptr + seq_idx)
                sw_mask_v = tl.where(
                    is_causal_seq,
                    dist < SLIDING_WINDOW,
                    (dist < SLIDING_WINDOW) & (dist > -SLIDING_WINDOW),
                )
            elif USE_CAUSAL:
                sw_mask_v = dist < SLIDING_WINDOW
            else:
                sw_mask_v = (dist < SLIDING_WINDOW) & (dist > -SLIDING_WINDOW)
            V = tl.where(sw_mask_v, V, 0.0)
        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: apply v_scale to P instead of V.
            P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
            acc += tl.dot(P_v, V)
        else:
            # Keep softmax weights representable when the P @ V dot uses FP8
            # tensor cores. Undo the power-of-two scale after the dot.
            if V.dtype == tl.float8e4nv:
                p_scale = 256.0
            elif V.dtype.is_fp8():
                p_scale = 128.0
            else:
                p_scale = 1.0
            acc += tl.dot((P * p_scale).to(V.dtype), V) / p_scale

    # ---- Epilogue ---------------------------------------------------------
    if IS_3D:
        if USE_FP8_Q_DESCALE:
            acc *= value_scale
        # Store per-segment partials; finalized by ``reduce_segments``.
        if USE_TD_QO:
            # 3D target: segm_output[token, head, segm_idx, :].  Advance
            # the base to the correct (token-start, head-start, segm)
            # slice; strides step between tokens / heads of the flattened
            # (T, H, SEGS, PAD) layout.
            segm_base = (
                segm_output_ptr
                + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q).to(
                    tl.int64
                )
                * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + (kv_head_idx * num_queries_per_kv)
                * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + segm_idx * HEAD_SIZE_PADDED
            )
            _store_output_td(
                segm_base,
                acc,
                q_block_local_len,
                num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED,
                NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED,
                num_queries_per_kv,
                BLOCK_Q,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
            segm_output_offset = (
                query_offset_0[:, None].to(tl.int64)
                * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
                + segm_idx * HEAD_SIZE_PADDED
                + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
            )
            tl.store(
                segm_output_ptr + segm_output_offset,
                acc,
                mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            )
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        acc = acc / L[:, None]
        if USE_FP8_Q_DESCALE:
            acc *= value_scale
        if USE_FP8:
            acc = acc * tl.load(out_scale)
            acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
        if USE_TD_QO:
            # 2D target: flat output[token, head, :].  Strides come
            # straight from the caller (``output_stride_0`` per token,
            # ``output_stride_1`` per head).
            output_base = (
                output_ptr
                + (cur_batch_in_all_start_index + q_block_local_idx * BLOCK_Q)
                * output_stride_0
                + (kv_head_idx * num_queries_per_kv) * output_stride_1
            )
            _store_output_td(
                output_base,
                acc,
                q_block_local_len,
                output_stride_0,
                output_stride_1,
                num_queries_per_kv,
                BLOCK_Q,
                HEAD_SIZE,
                HEAD_SIZE_PADDED,
            )
        else:
            output_offset = (
                query_offset_0[:, None] * output_stride_0
                + query_offset_1[:, None] * output_stride_1
                + offs_d[None, :]
            )
            tl.store(
                output_ptr + output_offset,
                acc,
                mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            )


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def _is_gemma3_attention(head_size: int, sliding_window: int) -> bool:
    """Detect Gemma3 models via unique (head_size, sliding_window) signature.

    Gemma3 models are the only ones using sliding_window=1024 with
    head_size 128 (27B) or 256 (1B, 4B, 12B). Other SWA models use
    different window sizes (Mistral=4096, Phi-3=2047).
    """
    return sliding_window == 1024 and head_size in (128, 256)


def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Select tile size with Gemma3-specific optimization."""
    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Default behavior
    if is_prefill:
        return 32
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    return 16 if element_size >= 2 else 32


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    seq_threshold_3D=None,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    # Optional tensor for prefix lengths (PrefixLM support)
    mm_prefix_range=None,
    # R-SWA support: prefix tokens stay globally visible, generated tokens use
    # a fixed sliding window.
    rswa_prefix_lens=None,
    rswa_window: int | None = None,
    use_alibi_sqrt=False,
    # KV cache quantization mode and per-token-head scale caches.
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE,
    k_scale_cache=None,  # [num_blocks, block_size, num_kv_heads] float32
    v_scale_cache=None,  # [num_blocks, block_size, num_kv_heads] float32
    # Chunked attention: restrict attention to aligned blocks with lookback.
    chunk_lookback=-1,
    # Tensor-descriptor mode: use ``tl.make_tensor_descriptor`` for Q/K/V
    # loads and output stores.  Enables HW 2D block reads on Intel Xe2/Xe3.
    # The non-TD branch is dead-code-eliminated at Triton compile time so
    # disabling this flag costs nothing. MACA does not support TMA and rejects
    # use_td=True; use the default pointer-based loads and stores instead.
    use_td: bool = False,
    # Gemma4: clamp mm_prefix bidirectional ranges by the sliding window.
    # Default False keeps the original behavior for every other model.
    mm_prefix_clamp_sliding_window: bool = False,
):
    if use_td :
        raise NotImplementedError(
            "make_tensor_descriptor no support , case not support tma on maca; use use_td=False "
            "for ordinary tl.load/tl.store"
        )

    # Resolve causal: bool or per-seq tensor.
    use_per_seq_causal = isinstance(causal, torch.Tensor)
    use_causal = bool(causal) if not use_per_seq_causal else True
    per_seq_causal_ptr = causal if use_per_seq_causal else None

    if kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
        raise NotImplementedError(
            "The standalone operator does not include INT4 kernels"
        )

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_per_token_head_scales = kv_quant_mode in (
        KVQuantMode.INT8_PER_TOKEN_HEAD,
        KVQuantMode.FP8_PER_TOKEN_HEAD,
    )
    if use_per_token_head_scales:
        assert k_scale_cache is not None and v_scale_cache is not None, (
            f"{kv_quant_mode.name} requires k_scale_cache / v_scale_cache"
        )

    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        if mm_prefix_range.ndim == 3:
            use_mm_prefix = True
            max_mm_ranges = mm_prefix_range.shape[1]
        else:
            raise ValueError(
                f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}"
            )

    use_rswa = rswa_window is not None and rswa_prefix_lens is not None

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Tuned launch parameters; ``None`` lets Triton pick its defaults.
    launch_num_warps: int | None = None
    launch_num_stages: int | None = None

    # Select the pipeline depth from the device shared-memory limit.  The
    # C500 exposes 64 KiB, while C600 devices expose more.  Use the opt-in
    # per-block limit when the backend provides it.
    launch_num_stages = 2
    if q.device.type == "cuda":
        device_properties = torch.cuda.get_device_properties(q.device)
        shared_memory_per_block = getattr(
            device_properties, "shared_memory_per_block_optin", 0
        )
        if not shared_memory_per_block:
            shared_memory_per_block = device_properties.shared_memory_per_block
        if shared_memory_per_block <= 64 * 1024:
            launch_num_stages = 1

    # head_size 256 with many query rows per sequence (e.g. diffusion-gemma
    # bidirectional canvas passes) is prefill-shaped, but the decode-oriented
    # defaults (BLOCK_Q=8, TILE=32, 4 warps) under-tile it. A wider KV tile +
    # more query rows per block + 8 warps is ~2x faster on B200.
    tuned_large_head = (
        head_size == 256
        and max_seqlen_q > 1
        and num_queries_per_kv <= 16
        and q.device.type == "cuda"
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability(q.device)[0] == 10
    )
    if tuned_large_head:
        BLOCK_M = 32
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        launch_num_warps = 8
        launch_num_stages = 2

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    # Compute chunked block size from sliding window if needed.
    chunk_size = -1
    if sliding_window_val > 0 and chunk_lookback > -1:
        chunk_size = sliding_window_val // (chunk_lookback + 1)
        assert chunk_size > 0, "sliding_window must be > chunk_lookback+1"
    elif sliding_window_val <= 0:
        chunk_lookback = -1

    TILE_SIZE_PREFILL = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=True
    )
    TILE_SIZE_DECODE = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=False
    )
    
    # NOTE(metax): On C500, a 512-wide FP16/BF16 head with BLOCK_M=16
    # and TILE_SIZE=32 needs 80 KiB even at num_stages=1. A 16-wide KV
    # tile needs 48 KiB. Apply this to both paths (2D also handles decode
    # above seq_threshold_3D). FP8 Q needs TILE_SIZE >= 32 for tl.dot.
    if launch_num_stages == 1 and head_size > 256 and q.element_size() == 2:
        TILE_SIZE_PREFILL = min(TILE_SIZE_PREFILL, 16)
        TILE_SIZE_DECODE = min(TILE_SIZE_DECODE, 16)

    # Wider KV tile for the tuned large-head path (see above). Only the 2D
    # path (used when max_seqlen_q > 1) reads TILE_SIZE_PREFILL.
    if tuned_large_head:
        TILE_SIZE_PREFILL = 128

    # USE_TD requires BLOCK_SIZE % TILE_SIZE == 0 (enforced by a
    # ``tl.static_assert`` in the kernel).  The default prefill tile
    # size (32) is larger than a common ``block_size=16``, so clamp it
    # down when TD is enabled.  Zero overhead when disabled.
    if use_td:
        TILE_SIZE_PREFILL = min(TILE_SIZE_PREFILL, block_size)
        TILE_SIZE_DECODE = min(TILE_SIZE_DECODE, block_size)

    # Tensor descriptors for Q load / output store require every element
    # of ``block_shape`` to be a power of 2.  ``num_queries_per_kv`` is
    # not always pow2 (e.g. Qwen2-7B: 28 / 4 = 7), so gate the Q/O paths
    # separately from the KV tile loads (whose ``block_shape`` does not
    # include ``num_queries_per_kv``).
    #
    # The Q/O descriptors also encode ``HEAD_SIZE_PADDED`` on the inner
    # axis while the backing buffers (both flat output and per-segment
    # output) are laid out with ``HEAD_SIZE``.  When they differ (e.g.
    # Phi-3's head_size=96 → HEAD_SIZE_PADDED=128) the store would spill
    # padded lanes into neighbouring heads because tensor-descriptor
    # stores don't mask the padded tail.  Fall back to the pointer path
    # for Q/O in that case — KV tile loads are unaffected because their
    # ``shape`` already matches ``block_shape`` on the inner axis.
    head_size_padded = triton.next_power_of_2(head_size)
    _is_pow2_nq = (num_queries_per_kv & (num_queries_per_kv - 1)) == 0
    _is_pow2_hs = head_size == head_size_padded
    use_td_qo = use_td and _is_pow2_nq and _is_pow2_hs

    # ``_load_q_td`` / ``_store_output_td`` flatten ``(num_queries_per_kv,
    # HEAD_SIZE)`` into a single contiguous inner axis.  That's only
    # equivalent to the pointer path when the ``num_queries_per_kv`` heads
    # for this KV group start at ``kv_head_idx * num_queries_per_kv`` and
    # lie exactly HEAD_SIZE apart — i.e. ``query_stride_1 == HEAD_SIZE``
    # and ``output_stride_1 == head_size``.  This is the default vLLM
    # query/output layout; assert it explicitly so we fail fast if a
    # future caller passes a non-contiguous query tensor.
    if use_td_qo:
        assert q.stride(1) == head_size, (
            f"USE_TD_QO requires contiguous query heads "
            f"(q.stride(1) = {q.stride(1)} != head_size = {head_size}); "
            f"set use_td=False or pad the query layout."
        )
        assert out.stride(1) == head_size, (
            f"USE_TD_QO requires contiguous output heads "
            f"(out.stride(1) = {out.stride(1)} != head_size = {head_size})."
        )

    # Launch the 2D kernel if
    # 1. No intermediate tiled softmax buffers for the 3D kernel have been allocated, or
    # 2. The batch includes at least one prefill request, or
    # 3. The number of sequences exceeds the configured threshold, or
    # 4. Batch invariance is enabled
    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    )

    # The kernel signature is the same for 2D and 3D — only the launch
    # grid + a handful of constexpr toggles differ.  Per-token-head scale
    # caches and their strides are passed as ``None`` when the
    # ``USE_PER_TOKEN_HEAD_SCALES`` branch is dead so Triton can skip
    # materialising those arguments and the associated registers.
    if use_per_token_head_scales:
        ks_strides = k_scale_cache.stride()
        vs_strides = v_scale_cache.stride()
        ks_blk, ks_slot, ks_head = ks_strides[0], ks_strides[1], ks_strides[2]
        vs_blk, vs_slot, vs_head = vs_strides[0], vs_strides[1], vs_strides[2]
        k_scale_ptr = k_scale_cache
        v_scale_ptr = v_scale_cache
    else:
        ks_blk = ks_slot = ks_head = None
        vs_blk = vs_slot = vs_head = None
        k_scale_ptr = None
        v_scale_ptr = None
    # 3D needs real segm tensors; 2D never touches them.  Pass ``None`` in
    # 2D mode so Triton can skip materialising these pointer arguments.
    segm_output_ptr = softmax_segm_output if use_3d else None
    segm_max_ptr = softmax_segm_max if use_3d else None
    segm_expsum_ptr = softmax_segm_expsum if use_3d else None
    num_segments = num_par_softmax_segments if use_3d else 1

    grid: tuple[Any, ...]
    if not use_3d:
        grid = (total_num_q_blocks, num_kv_heads)
        tile_size = TILE_SIZE_PREFILL
    else:
        grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        tile_size = TILE_SIZE_DECODE

    launch_kwargs: dict[str, int] = {}
    if launch_num_warps is not None:
        launch_kwargs["num_warps"] = launch_num_warps
    if launch_num_stages is not None:
        launch_kwargs["num_stages"] = launch_num_stages

    kernel_unified_attention[grid](
        output_ptr=out,
        segm_output_ptr=segm_output_ptr,
        segm_max_ptr=segm_max_ptr,
        segm_expsum_ptr=segm_expsum_ptr,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        qq_bias_ptr=qq_bias,
        k_scale_cache_ptr=k_scale_ptr,
        v_scale_cache_ptr=v_scale_ptr,
        scale=softmax_scale,
        q_scale=q_descale,
        k_scale=k_descale,
        v_scale=v_descale,
        out_scale=1 / output_scale if output_scale is not None else 1.0,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_QQ_BIAS=use_qq_bias,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        SLIDING_WINDOW=(1 + window_size[0]),
        USE_CAUSAL=use_causal,
        USE_PER_SEQ_CAUSAL=use_per_seq_causal,
        per_seq_causal_ptr=per_seq_causal_ptr,
        USE_MM_PREFIX=use_mm_prefix,
        MAX_MM_RANGES=max_mm_ranges,
        mm_prefix_range_ptr=mm_prefix_range,
        rswa_prefix_lens_ptr=rswa_prefix_lens if use_rswa else seqused_k,
        R_SWA_WINDOW=rswa_window or 0,
        USE_R_SWA=use_rswa,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        stride_ks_blk=ks_blk,
        stride_ks_slot=ks_slot,
        stride_ks_head=ks_head,
        stride_vs_blk=vs_blk,
        stride_vs_slot=vs_slot,
        stride_vs_head=vs_head,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        USE_FP8=output_scale is not None,
        IS_3D=use_3d,
        KV_QUANT_MODE=kv_quant_mode,
        Q_IS_FP8=(q.dtype == FP8_DTYPE),
        CHUNK_LOOKBACK=chunk_lookback,
        CHUNK_SIZE=chunk_size,
        USE_TD=use_td,
        USE_TD_QO=use_td_qo,
        MM_PREFIX_CLAMP_SW=mm_prefix_clamp_sliding_window,
        **launch_kwargs,
    )

    if use_3d:
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
        )
