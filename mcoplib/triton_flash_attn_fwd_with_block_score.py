# Copyright 2025 XunhaoLai. All rights reserved.
"""Standalone prefill sparse-attention kernels (bf16 / fp16 only).

Step 1 (lightning indexer): ``flash_prefill_with_topk_index`` — score every KV
block with the index head and select the top-k blocks (in-kernel bitonic top-k).
Optionally also produces the index-head attention output.

Step 3 (main sparse attention): ``flash_prefill_with_gqa_share_sparse`` —
attend only to the selected top-k blocks with the main head.

No fp8, no sglang, no project utils. Paged KV via ``req_to_token`` / ``slot_ids``.
"""

from typing import Any, Callable, Dict, Literal, Optional, Tuple
import functools
import torch
import triton
import triton.language as tl

def tensor_cache(maxsize: int = 8):
    """
    Cache function results using identity comparison.
    Zero-overhead cache hit: no hash, no DtoH, just pointer comparison.

    Args:
        maxsize: Maximum number of cached entries. Supports multi-GPU scenarios
                 where different devices have different tensor arguments.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        # LRU-style cache: list of (args, kwargs, result) tuples
        # Most recently used at the end
        _cache: list = []

        def _args_match(args: tuple, cached_args: tuple) -> bool:
            if len(args) != len(cached_args):
                return False
            for i in range(len(args)):
                if args[i] is not cached_args[i]:
                    return False
            return True

        def _kwargs_match(kwargs: dict, cached_kwargs: dict) -> bool:
            if not kwargs and not cached_kwargs:
                return True
            if kwargs.keys() != cached_kwargs.keys():
                return False
            for k, v in kwargs.items():
                if v is not cached_kwargs[k]:
                    return False
            return True

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Search cache (most recent first for better hit rate)
            for i in range(len(_cache) - 1, -1, -1):
                cached_args, cached_kwargs, cached_result = _cache[i]
                if _args_match(args, cached_args) and _kwargs_match(
                    kwargs, cached_kwargs
                ):
                    # Move to end (most recently used)
                    if i != len(_cache) - 1:
                        _cache.append(_cache.pop(i))
                    return cached_result

            # Cache miss
            result = fn(*args, **kwargs)

            # Add to cache
            if len(_cache) >= maxsize:
                _cache.pop(0)  # Remove oldest
            _cache.append((args, kwargs, result))
            return result

        # Expose cache for manual clearing if needed
        wrapper.cache_clear = lambda: _cache.clear()
        wrapper.cache_info = lambda: {"size": len(_cache), "maxsize": maxsize}

        return wrapper

    return decorator


@tensor_cache(maxsize=8)
def get_cu_seqblocks(
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    block_size_q: int,
    block_size_k: int,
) -> Tuple[torch.Tensor, int, int, torch.Tensor, int, int]:
    """Compute cumulative sequence block indices for blocked sparse attention.

    Converts token-level cumulative sequence lengths to block-level indices,
    which are needed for block-sparse attention kernels.

    Note:
        Results are cached (maxsize=8) based on input arguments. Repeated calls
        with the same cu_seqlens, max_seqlen, and block sizes will return cached
        results without recomputation.

    Args:
        cu_seqlens: Cumulative sequence lengths. Shape: [batch_size + 1], dtype: int32.
        max_seqlen: Maximum sequence length in the batch.
        block_size_q: Query block size.
        block_size_k: Key-value block size.

    Returns:
        A tuple of 6 values:
            - cu_seqblocks_q: Cumulative query block indices. Shape: [batch_size + 1]
            - max_seqblock_q: Maximum number of query blocks per sequence.
            - all_seqblock_q: Total number of query blocks across all sequences.
            - cu_seqblocks_k: Cumulative key block indices. Shape: [batch_size + 1]
            - max_seqblock_k: Maximum number of key blocks per sequence.
            - all_seqblock_k: Total number of key blocks across all sequences.
    """
    cu_seqblocks_q = torch.zeros_like(cu_seqlens)
    cu_seqblocks_k = torch.zeros_like(cu_seqlens)
    seq_lens = torch.diff(cu_seqlens)
    seqblocks_q = (seq_lens + block_size_q - 1) // block_size_q
    seqblocks_k = (seq_lens + block_size_k - 1) // block_size_k
    max_seqblock_q = (max_seqlen + block_size_q - 1) // block_size_q
    max_seqblock_k = (max_seqlen + block_size_k - 1) // block_size_k
    cu_seqblocks_q[1:] = seqblocks_q
    cu_seqblocks_k[1:] = seqblocks_k
    cu_seqblocks_q.cumsum_(0)
    cu_seqblocks_k.cumsum_(0)
    all_seqblock_q = seqblocks_q.sum().item()
    all_seqblock_k = seqblocks_k.sum().item()
    return (
        cu_seqblocks_q,
        max_seqblock_q,
        all_seqblock_q,
        cu_seqblocks_k,
        max_seqblock_k,
        all_seqblock_k,
    )

@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    # idx
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    # actual compare-and-swap
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids
    
@triton.jit
def _bitonic_merge(
    x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids

def _check_dtypes(q, k_cache, v_cache):
    assert q.dtype in (torch.bfloat16, torch.float16), f"q dtype must be bf16/fp16, got {q.dtype}"
    assert k_cache.dtype == q.dtype, f"k_cache dtype {k_cache.dtype} != q dtype {q.dtype}"
    assert v_cache.dtype == k_cache.dtype, f"v_cache dtype {v_cache.dtype} != k_cache dtype {k_cache.dtype}"

# --------------------------------------------------------------------------- #
# Step 1: score + (optional) index-head attention, with in-kernel top-k score  #
# --------------------------------------------------------------------------- #
@triton.heuristics(
    {
        "BLOCK_SIZE_KD": lambda args: triton.next_power_of_2(args["qk_head_dim"]),
        "BLOCK_SIZE_VD": lambda args: triton.next_power_of_2(args["v_head_dim"]),
        "HAS_SINK": lambda args: args["sink_ptr"] is not None,
    }
)
@triton.autotune(
    configs=[
        # BLOCK_SIZE_K is now the K/V sub-tile (block_size=bsk is processed in
        # SUB_TILES_PER_BLOCK = block_size//BLOCK_SIZE_K sub-tiles). Small
        # sub-tile -> small smem -> high occupancy on 64KB-smem/SM. 16 is the
        # bf16 mma K-dim floor; 32 is the larger alternative. num_stages=1
        # fits <=64KB; larger stages are pruned where they exceed the limit.
        # Note: BSQ=32 raises occupancy (18% vs 6%) but step1 is compute-bound
        # (~16 TFLOPS), so halving the Q tile costs more arithmetic intensity
        # than occupancy gains —实测 BSQ32 慢一倍, 故不纳入。BSQ64 BSK32 最优。
        triton.Config(
            {"BLOCK_SIZE_Q": 64, "BLOCK_SIZE_K": 32}, num_warps=2, num_stages=3
        ),
    ],
    key=[
        "qk_head_dim",
        "v_head_dim",
        "block_size",
        "use_gumbel_topk",
    ],
)
@triton.jit
def _flash_attn_fwd_with_block_score_kernel(
    q_ptr,  # Q: n x h x d
    k_cache_ptr,  # K paged: max_slots x kh x d
    v_cache_ptr,  # V paged: max_slots x kh x d
    sink_ptr,  # Sink: h x d
    o_ptr,  # O: n x h x d
    score_ptr,  # Score: h x n x max_seqblock
    req_to_token_ptr,  # req_to_token: max_reqs x max_kv_len
    # seqlens
    cu_seqlens,
    seq_lens,
    prefix_lens,
    slot_ids,
    # shape
    max_slots,
    num_heads,
    gqa_group_size,
    qk_head_dim,
    v_head_dim,
    block_size: tl.constexpr,
    # sm_scale
    sm_scale,
    # gumbel topk
    use_gumbel_topk: tl.constexpr,
    gumbel_seed,
    # stride
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_s,
    stride_k_h,
    stride_k_d,
    stride_v_s,
    stride_v_h,
    stride_v_d,
    stride_sink_h,
    stride_sink_d,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    stride_s_h,
    stride_s_q,
    stride_s_k,
    stride_r2t_b,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_VD: tl.constexpr,
    # has sink
    HAS_SINK: tl.constexpr,
    SCORE_TYPE: tl.constexpr,
    DISABLE_INDEX_VALUE: tl.constexpr,
):
    tl.static_assert(SCORE_TYPE == "max" or SCORE_TYPE == "lse")
    sm_scale_log2e = sm_scale * 1.4426950409
    # BLOCK_SIZE_K is the K/V sub-tile; block_size is the topk block size (bsk).
    # A selected block of `block_size` tokens is processed in sub-tiles of
    # BLOCK_SIZE_K (block_size must be a multiple of BLOCK_SIZE_K). Smaller
    # sub-tile -> smaller K/V shared-mem footprint -> higher occupancy on the
    # 64KB-smem/SM device (see step3 kernel for the same trick). Each bsk-block
    # writes exactly one score column, accumulated across its sub-tiles.
    tl.static_assert(block_size % BLOCK_SIZE_K == 0)
    SUB_TILES_PER_BLOCK: tl.constexpr = block_size // BLOCK_SIZE_K
    SINGLE_TILE: tl.constexpr = SUB_TILES_PER_BLOCK == 1
    # get batch id and head id
    pid_q = tl.num_programs(0) - 1 - tl.program_id(0)
    pid_bh = tl.program_id(1)
    # pid_q, pid_bh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads
    pid_kh = pid_h // gqa_group_size
    # get q k start and len after rmpad
    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    sid = tl.load(slot_ids + pid_b).to(tl.int64)
    sid = tl.where(sid < 0, sid + max_slots, sid)
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return
    block_num = (seq_len + block_size - 1) // block_size
    # init qkv pointer
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + seq_start * stride_q_n + pid_h * stride_q_h,
        shape=(q_len, qk_head_dim),
        strides=(stride_q_n, stride_q_d),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_KD),
        order=(1, 0),
    )
    s_ptrs = tl.make_block_ptr(
        base=score_ptr + seq_start * stride_s_q + pid_h * stride_s_h,
        shape=(block_num, q_len),
        strides=(stride_s_k, stride_s_q),
        offsets=(0, pid_q * BLOCK_SIZE_Q),
        block_shape=(1, BLOCK_SIZE_Q),
        order=(1, 0),
    )
    # load q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    if HAS_SINK and not DISABLE_INDEX_VALUE:
        off_d = tl.arange(0, BLOCK_SIZE_KD)
        sink = tl.load(
            sink_ptr + pid_h * stride_sink_h + off_d * stride_sink_d,
            mask=off_d < qk_head_dim,
            other=0,
        )
    # init statistics
    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q + prefix_len
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_kd = tl.arange(0, BLOCK_SIZE_KD)
    off_vd = tl.arange(0, BLOCK_SIZE_VD)
    kd_mask = off_kd < qk_head_dim
    vd_mask = off_vd < v_head_dim
    if not DISABLE_INDEX_VALUE:
        if HAS_SINK:
            m_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
            lse_i = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
            qsink = tl.sum(q * sink[None, :], axis=1) * sm_scale_log2e  # (BLOCK_SIZE_Q,)
            m_i += qsink
            lse_i += qsink
        else:
            m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
            lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
        acc_o = tl.full((BLOCK_SIZE_Q, BLOCK_SIZE_VD), 0, dtype=tl.float32)
    # attention. Outer loop steps by block_size (one bsk-block -> one score
    # column); inner loop splits that block into SUB_TILES_PER_BLOCK sub-tiles
    # of BLOCK_SIZE_K so the K/V shared-mem tile stays small (high occupancy).
    # Online softmax (m/lse/acc_o) and the per-block score (max/lse) both
    # accumulate across sub-tiles, so the split is exact.
    hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)
    pos_n = off_k
    slots_next = tl.load(req_to_token_ptr + sid * stride_r2t_b + pos_n,
                         mask=pos_n < seq_len, other=0)
    # for i in tl.range(0, hi, block_size):
    for ii in tl.range(0, (hi + block_size - 1) // block_size):
        i = ii * block_size
        # accumulate this bsk-block's score across its sub-tiles.
        # max: running max of per-token qk. lse: logsumexp via running (max, sum).
        if not SINGLE_TILE:
            blk_score_max = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
            blk_lse_m = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
            blk_lse_s = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
        for s in tl.static_range(0, SUB_TILES_PER_BLOCK):
            ks = i + s * BLOCK_SIZE_K
            # paged load K via req_to_token: pos -> slot -> k_cache
            pos = ks + off_k
            pos_mask = pos < seq_len
            slots = slots_next.to(tl.int64)
            pos_n = pos + BLOCK_SIZE_K
            slots_next = tl.load(req_to_token_ptr + sid * stride_r2t_b + pos_n,
                                 mask=pos_n < seq_len, other=0)
            slots = tl.where(slots < 0, slots + max_slots, slots)
            # k shape: [BLOCK_SIZE_KD, BLOCK_SIZE_K] (transposed for tl.dot)
            k = tl.load(
                k_cache_ptr
                + slots[None, :] * stride_k_s
                + pid_kh * stride_k_h
                + off_kd[:, None] * stride_k_d,
                mask=kd_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )
            # compute qk
            qk = tl.dot(q, k) * sm_scale_log2e
            # causal mask: a query at off_q attends to keys at abs pos ks+off_k
            valid = (off_q[:, None] >= (ks + off_k)[None, :]) & pos_mask[None, :]
            qk = tl.where(valid, qk, float("-inf"))
            # accumulate per-block score (max or lse) across sub-tiles
            sub_max = tl.max(qk, axis=1)  # (BLOCK_SIZE_Q,)
            if SINGLE_TILE:
                if SCORE_TYPE == "max":
                    score = sub_max
                else:  # "lse"
                    score = sub_max + tl.log2(
                        tl.sum(tl.exp2(qk - sub_max[:, None]), axis=1)
                    )
                    score = tl.where(sub_max == float("-inf"), float("-inf"), score)
            else:
                if SCORE_TYPE == "max":
                    blk_score_max = tl.maximum(blk_score_max, sub_max)
                else:  # "lse": logsumexp merge of this sub-tile into the block
                    new_m = tl.maximum(blk_lse_m, sub_max)
                    blk_lse_s = blk_lse_s * tl.exp2(blk_lse_m - new_m) + tl.sum(
                        tl.exp2(qk - new_m[:, None]), axis=1
                    )
                    blk_lse_m = new_m
            if not DISABLE_INDEX_VALUE:
                # compute m_ij and l_ij (online softmax over all sub-tiles)
                m_ij = tl.maximum(m_i, sub_max)
                p = tl.exp2(qk - m_ij[:, None])
                l_ij = tl.sum(p, axis=1)
                # scale acc_o
                acc_o_scale = tl.exp2(m_i - m_ij)
                acc_o = acc_o * acc_o_scale[:, None]
                # paged load V (sub-tile, same slots)
                v = tl.load(
                    v_cache_ptr
                    + slots[:, None] * stride_v_s
                    + pid_kh * stride_v_h
                    + off_vd[None, :] * stride_v_d,
                    mask=pos_mask[:, None] & vd_mask[None, :],
                    other=0.0,
                )
                p = p.to(v.dtype)
                acc_o += tl.dot(p, v)
                # update statistics
                m_i = m_ij
                lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        # finalize and save this bsk-block's score (one column)
        if not SINGLE_TILE:
            if SCORE_TYPE == "max":
                score = blk_score_max
            else:  # "lse"
                score = blk_lse_m + tl.log2(blk_lse_s)
                # fully-masked block (all -inf) -> NaN; clamp to -inf sentinel
                score = tl.where(score != score, float("-inf"), score)
        if use_gumbel_topk:
            local_seed = (pid_h | (pid_b << 7) | (gumbel_seed << 19)).to(tl.int32)
            noise_offset = (off_q << 13) | (i // block_size)
            noise = tl.rand(local_seed, offset=noise_offset)
            noise = tl.clamp(noise, min=1e-9, max=1 - 1e-9)  # avoid log(0)
            noise = -tl.log(-tl.log(noise)) * 1.4426950409
            score = score + noise
        tl.store(s_ptrs, tl.reshape(score, (1, BLOCK_SIZE_Q)).to(score_ptr.dtype.element_ty), boundary_check=(0, 1))
        # advance score ptr to next bsk-block column
        s_ptrs = tl.advance(s_ptrs, (1, 0))
    if not DISABLE_INDEX_VALUE:
        # final scale
        acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]
        # save output
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + seq_start * stride_o_n + pid_h * stride_o_h,
            shape=(q_len, v_head_dim),
            strides=(stride_o_n, stride_o_d),
            offsets=(pid_q * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_VD),
            order=(1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_K": 64}, num_warps=2, num_stages=2),
    ],
    key=[
        "topk"
    ],  # topk gates BLOCK_SIZE_K validity (assert BLOCK_SIZE_K > topk)
)
@triton.jit
def _topk_index_kernel(
    s_ptr,  # Score: h x n x max_seqblock
    ti_ptr,  # topk_idx: h x n x topk
    # size
    sample_interval: tl.constexpr,
    block_size: tl.constexpr,
    # seqlens
    cu_seqlens,
    cu_seqblocks_q,
    prefix_lens,
    # shape
    topk,  # not constexpr to avoid recompilation when topk changes
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    # stride
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_ti_h,
    stride_ti_n,
    stride_ti_t,
    # META parameters
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    MASK_INIT: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
):
    tl.static_assert(
        BLOCK_SIZE_K > BLOCK_SIZE_T
    )  # use BLOCK_SIZE_T instead of topk (stricter but safe)
    # get batch id and head id
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    # get q k start and len after rmpad
    seq_start = tl.load(cu_seqlens + pid_b)
    block_start = tl.load(cu_seqblocks_q + pid_b)
    block_num = tl.load(cu_seqblocks_q + pid_b + 1) - block_start
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= block_num:
        return
    # offsets
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    # init qkv pointer
    s_ptrs = (
        s_ptr
        + (seq_start + pid_q * sample_interval) * stride_s_n
        + pid_h * stride_s_h
        + off_k * stride_s_k
    )
    # init statistics
    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2
    # compute topk
    valid_blocks = (prefix_len + pid_q * sample_interval + block_size) // block_size
    for i in tl.range(0, valid_blocks, BLOCK_SIZE_K):
        # masks
        causal_mask = i + off_k < valid_blocks
        local_mask = i + off_k >= max(0, valid_blocks - local_blocks)
        init_mask = i + off_k < init_blocks
        # load score
        score = tl.load(s_ptrs, mask=causal_mask, other=-1e30).to(tl.float32)
        # handle NaN: NaN inputs cause bitonic sort to fail, resulting in invalid indices (-2)
        # appearing in the topk list. We replace NaN with -inf to maintain sort order.
        score = tl.where(score != score, -1e30, score)
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        # fill init and local part, make sure init part is always in topk
        # and at the first position. Note: must use causal_mask to protect
        # init_mask to avoid selecting blocks outside causal window
        if MASK_INIT:
            score = tl.where(causal_mask & init_mask, score - 1e29, score)
        else:
            score = tl.where(causal_mask & init_mask, 1e30, score)
        if MASK_LOCAL:
            score = tl.where(causal_mask & local_mask, score - 1e28, score)
        else:
            score = tl.where(causal_mask & local_mask, 1e29, score)
        # bitonic merge
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (tl.where(causal_mask, i + off_k + 1, 0), topk_idx)
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), j, 2, n_dims
            )
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims
            )
            topk_score_new = last_topk_score * left_half_mask + topk_score * (
                1 - left_half_mask
            )
            topk_idx_new = last_topk_idx * left_half_mask + topk_idx * (
                1 - left_half_mask
            )
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
            )
    # get topk, shape: [BLOCK_SIZE_T,]
    topk_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    topk_idx = tl.sum(
        topk_mask[:, None]
        * tl.reshape(topk_idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )
    # save topk
    ti_ptrs = (
        ti_ptr
        + (block_start + pid_q) * stride_ti_n
        + pid_h * stride_ti_h
        + off_t * stride_ti_t
    )
    topk_mask = tl.arange(0, BLOCK_SIZE_T) < min(topk, valid_blocks)
    tl.store(ti_ptrs, topk_idx.to(ti_ptrs.dtype.element_ty), mask=topk_mask)


@torch.no_grad()
def flash_prefill_with_topk_index(
    q: torch.Tensor,
    k_cache: torch.Tensor,  # paged
    v_cache: Optional[torch.Tensor],  # paged; ignored when disable_index_value=True
    sink: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_size_q: int,
    block_size_k: int,
    topk: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
    sm_scale: Optional[float] = None,
    score_type: str = "max",
    disable_index_value: bool = False,
    cu_seqblocks_q: Optional[torch.Tensor] = None,
    max_seqblock_q: Optional[int] = None,
    all_seqblock_q: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Step 1: score KV blocks with the index head and select top-k blocks.

    Returns ``(o, topk_idx)``. ``o`` is the index-head attention output
    ([total_q, num_heads, v_head_dim]) or ``None`` when ``disable_index_value``.
    ``topk_idx`` is [num_heads, all_seqblock_q, topk], 0-based block ids, -1 padded.
    """
    assert score_type in ("max", "lse"), f"score_type must be 'max' or 'lse', got {score_type!r}"
    # dtype check
    assert q.dtype in (torch.bfloat16, torch.float16)
    assert k_cache.dtype == q.dtype
    assert cu_seqlens.dtype == torch.int32
    # shape
    total_q, num_heads, qk_head_dim = q.shape
    max_slots, num_kv_heads, _ = k_cache.shape
    if disable_index_value:
        v_head_dim = qk_head_dim  # placeholder for BLOCK_SIZE_VD; V is never loaded
    else:
        assert v_cache is not None and v_cache.dtype == q.dtype
        assert v_cache.shape[1] == k_cache.shape[1]
        _check_dtypes(q, k_cache, v_cache)
        v_head_dim = v_cache.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    batch_size = cu_seqlens.shape[0] - 1
    assert qk_head_dim <= 256 and v_head_dim <= 256, "head_dim must be less than 256"
    if sink is not None:
        assert sink.shape[0] == num_heads and sink.shape[1] == qk_head_dim
    assert (
        init_blocks + local_blocks <= topk
    ), "init_blocks + local_blocks must be less than topk"
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5
    if cu_seqblocks_q is None or max_seqblock_q is None or all_seqblock_q is None:
        cu_seqblocks_q, max_seqblock_q, all_seqblock_q, _, _, _ = get_cu_seqblocks(
            cu_seqlens, max_seqlen_q, block_size_q, block_size_k
        )
    max_seqblock_k = triton.cdiv(max_seqlen_k, block_size_k)
    if disable_index_value:
        o = None
    else:
        o = torch.empty(total_q, num_heads, v_head_dim, dtype=q.dtype, device=q.device)
    score = torch.empty(
        (num_heads, max_seqblock_k, total_q),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )

    # launch kernel
    def grid(META):
        return (triton.cdiv(max_seqlen_q, META["BLOCK_SIZE_Q"]), batch_size * num_heads)

    _flash_attn_fwd_with_block_score_kernel[grid](
        q,
        k_cache,
        v_cache,
        sink,
        o,
        score,
        req_to_token,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        slot_ids,
        max_slots,
        num_heads,
        gqa_group_size,
        qk_head_dim,
        v_head_dim,
        block_size_k,
        sm_scale,
        False,
        1,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0) if v_cache is not None else 0,
        v_cache.stride(1) if v_cache is not None else 0,
        v_cache.stride(2) if v_cache is not None else 0,
        sink.stride(0) if sink is not None else 0,
        sink.stride(1) if sink is not None else 0,
        o.stride(0) if o is not None else 0,
        o.stride(1) if o is not None else 0,
        o.stride(2) if o is not None else 0,
        score.stride(0),
        score.stride(2),
        score.stride(1),
        req_to_token.stride(0),
        SCORE_TYPE=score_type,
        DISABLE_INDEX_VALUE=disable_index_value,
    )

    # topk extraction kernel
    topk_idx = torch.full(
        (num_heads, all_seqblock_q, topk),
        fill_value=-1,
        device=score.device,
        dtype=torch.int32,
    )
    # launch kernel
    grid = (max_seqblock_q, batch_size, num_heads)
    _topk_index_kernel[grid](
        score,
        topk_idx,
        block_size_q,
        block_size_k,
        cu_seqlens,
        cu_seqblocks_q,
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
        score.stride(0),
        score.stride(2),
        score.stride(1),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        MASK_INIT=False,
        MASK_LOCAL=False,
    )
    return o, topk_idx
