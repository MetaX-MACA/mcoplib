# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from dataclasses import dataclass

import torch

from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton

# Smallest positive value produced by Triton's fp32 `tl.rand`. Used to clamp
# zero draws before the flipped Gumbel transform below.
#
# Triton requires globals accessed from `@triton.jit` functions to be wrapped
# in `tl.constexpr(...)`. We can only do that when Triton is actually
# available — on the CPU worker path `tl` is a placeholder whose `constexpr`
# attribute is `None`, and `tl.constexpr(...)` would crash at import time.
_TL_RAND_MIN = tl.constexpr(4.6566127342e-10) if HAS_TRITON else 4.6566127342e-10

# Offset salt keeping the draft's Gumbel noise disjoint from the target's.
# Verification is a probability-ratio test, not a Gumbel coupling, so a proposal
# and the residual it is resampled from must not share a noise vector.
# Positions are int64 and never approach 2**30, so the streams cannot collide.
_DRAFT_NOISE_SALT = tl.constexpr(1 << 30) if HAS_TRITON else (1 << 30)

_INT32_MAX = tl.constexpr(2147483647) if HAS_TRITON else 2147483647

# CU count at/above which the "wide" profile is used.
_WIDE_CU_THRESHOLD = 64
# Chunk size of the original-like (compute-bound) grid.
_COMPUTE_BOUND_CHUNK = 1024

_SPLIT_BLOCK_SIZE = 1024

@dataclass(frozen=True)
class _Profile:
    num_cu: int
    num_warps: int
    chunk_align: int  # vocab split granularity of the split (looped) path
    programs_per_cu: int  # target programs per CU on the split path
    random_use_split: bool  # RNG sampling also uses the split path


@functools.lru_cache(maxsize=None)
def _get_profile(device_index: int) -> _Profile:
    try:
        num_cu = torch.cuda.get_device_properties(device_index).multi_processor_count
    except Exception:
        num_cu = 32
    wide = num_cu >= _WIDE_CU_THRESHOLD
    return _Profile(
        num_cu=num_cu,
        chunk_align=2048 if wide else 8192,
        programs_per_cu=4,
        random_use_split=not wide,
        num_warps = 2 if wide else 4
    )


@triton.jit
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _temperature_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def tl_rand64(seed, offset, includes_zero: tl.constexpr):
    lo, hi, _, _ = tl.randint4x(seed, offset)
    lo = lo.to(tl.uint32, bitcast=True).to(tl.uint64)
    hi = hi.to(tl.uint32, bitcast=True).to(tl.uint64)
    r = (hi << 32) | lo

    # 1 / 2**64
    scale = 5.421010862427522170037e-20
    u = r.to(tl.float64) * scale
    if not includes_zero:
        u = tl.maximum(u, 2.2250738585072014e-308)  # float64 tiny
    return u


@triton.jit
def tl_rand32(seed, offset, includes_zero: tl.constexpr):
    u = tl.rand(seed, offset)
    if not includes_zero:
        u = tl.maximum(u, _TL_RAND_MIN)
    return u


@triton.jit(do_not_specialize=["local_stride", "chunk_size"])
def _gumbel_sample_split_kernel(
    out_ptr,  # [num_tokens] int64, used when SINGLE_SPLIT
    local_argmax_ptr,  # [num_tokens, num_splits] int32
    local_max_ptr,  # [num_tokens, num_splits] fp32/fp64
    local_stride,
    # [max_num_reqs, num_cols, vocab_size]
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    logits_cache_col_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    chunk_size,
    SINGLE_SPLIT: tl.constexpr,
    IS_DRAFTING: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
    ALL_GREEDY: tl.constexpr,
    COMPUTE_BOUND: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    split_idx = tl.program_id(1)

    if ALL_GREEDY:
        temp = 0.0
        gumbel_seed = 0
    else:
        # Per-row scalars: loaded once per program.
        req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
        is_valid_req = req_state_idx >= 0
        temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
            tl.float32
        )
        seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
        pos = tl.load(pos_ptr + token_idx)
        if IS_DRAFTING:
            pos = pos + _DRAFT_NOISE_SALT
        gumbel_seed = tl.randint(seed, pos)

    if logits_cache_ptr is not None:
        cache_req_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
        cache_valid = cache_req_idx >= 0
        if PER_TOKEN_COL:
            col = tl.load(logits_cache_col_ptr + token_idx)
        else:
            col = tl.load(logits_cache_col_ptr)
        cache_row = (
            logits_cache_ptr
            + cache_req_idx * logits_cache_stride_0
            + col * logits_cache_stride_1
        )

    row_ptr = logits_ptr + token_idx * logits_stride
    start = tl.multiple_of(split_idx * chunk_size, BLOCK_SIZE)

    if COMPUTE_BOUND:
        # Single pass (original-equivalent).
        block = start + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        x = tl.load(row_ptr + block, mask=mask, other=float("-inf"))
        x = x.to(tl.float32)
        if logits_cache_ptr is not None:
            # Pre-temperature logits, identical to the original kernel.
            tl.store(cache_row + block, x, mask=mask & cache_valid)
        if not ALL_GREEDY:
            if APPLY_TEMPERATURE:
                if temp != 0.0:
                    x = x / temp  # fp32 divide, as in the original
        if USE_FP64:
            x = x.to(tl.float64)
        if not ALL_GREEDY:
            if temp != 0.0:
                if USE_FP64:
                    u = tl_rand64(gumbel_seed, block, includes_zero=False)
                    g = -tl.log(-tl.log(u))
                else:
                    u = tl_rand32(gumbel_seed, block, includes_zero=False)
                    g = -tl.log(-tldevice.log1p(-u))
                x = tl.where(mask, x + g, float("-inf"))
        best, local_idx = tl.max(x, axis=0, return_indices=True)
        best_idx = (start + local_idx).to(tl.int32)
    else:
        # Looped split path.
        end = tl.minimum(start + chunk_size, vocab_size)
        acc_val = tl.full([BLOCK_SIZE], float("-inf"), tl.float32)
        if USE_FP64:
            acc_val = acc_val.to(tl.float64)
        acc_idx = tl.zeros([BLOCK_SIZE], dtype=tl.int32) + start.to(tl.int32)

        for off in range(start, end, BLOCK_SIZE):
            off = tl.multiple_of(off, BLOCK_SIZE)
            block = off + tl.arange(0, BLOCK_SIZE)
            mask = block < end
            x = tl.load(row_ptr + block, mask=mask, other=float("-inf"))
            x = x.to(tl.float32)
            if logits_cache_ptr is not None:
                tl.store(cache_row + block, x, mask=mask & cache_valid)
            if not ALL_GREEDY:
                if APPLY_TEMPERATURE:
                    if temp != 0.0:
                        x = x / temp  # fp32 divide, as in the original
            if USE_FP64:
                x = x.to(tl.float64)
            if not ALL_GREEDY:
                if temp != 0.0:
                    if USE_FP64:
                        u = tl_rand64(gumbel_seed, block, includes_zero=False)
                        g = -tl.log(-tl.log(u))
                    else:
                        u = tl_rand32(gumbel_seed, block, includes_zero=False)
                        g = -tl.log(-tldevice.log1p(-u))
                    x = tl.where(mask, x + g, float("-inf"))
            # Strict '>' keeps the first index per lane on ties.
            upd = x > acc_val
            acc_val = tl.where(upd, x, acc_val)
            acc_idx = tl.where(upd, block.to(tl.int32), acc_idx)

        # Ties -> smallest token id (matches torch.argmax).
        best = tl.max(acc_val, axis=0)
        best_idx = tl.min(tl.where(acc_val == best, acc_idx, _INT32_MAX), axis=0)

    best_idx = tl.where(best_idx == _INT32_MAX, start.to(tl.int32), best_idx)
    best_idx = tl.minimum(tl.maximum(best_idx, 0), vocab_size - 1)

    if SINGLE_SPLIT:
        tl.store(out_ptr + token_idx, best_idx.to(tl.int64))
    else:
        tl.store(local_argmax_ptr + token_idx * local_stride + split_idx, best_idx)
        tl.store(local_max_ptr + token_idx * local_stride + split_idx, best)


@triton.jit(do_not_specialize=["local_stride", "num_splits"])
def _gumbel_reduce_kernel(
    out_ptr,
    local_argmax_ptr,
    local_max_ptr,
    local_stride,
    num_splits,
    vocab_size,
    BLOCK_SPLITS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_SPLITS)
    mask = offs < num_splits
    vals = tl.load(
        local_max_ptr + token_idx * local_stride + offs,
        mask=mask,
        other=float("-inf"),
    )
    idxs = tl.load(
        local_argmax_ptr + token_idx * local_stride + offs,
        mask=mask,
        other=_INT32_MAX,
    )
    best = tl.max(vals, axis=0)
    # Splits are ordered by token id, so min id == first split with the max.
    best_idx = tl.min(tl.where(mask & (vals == best), idxs, _INT32_MAX), axis=0)
    best_idx = tl.where(best_idx == _INT32_MAX, 0, best_idx)
    best_idx = tl.minimum(tl.maximum(best_idx, 0), vocab_size - 1)
    tl.store(out_ptr + token_idx, best_idx.to(tl.int64))

_PRECOMPILED_KEYS: set = set()


def _precompile_once(
    logits, expanded_idx_mapping, temperature, seed, pos,
    apply_temperature, is_drafting, logits_cache, logits_cache_col, use_fp64,
):
    if torch.cuda.is_current_stream_capturing():
        return
    vocab_size = logits.shape[1]
    key = (
        logits.device, logits.dtype, vocab_size, logits.stride(0),
        expanded_idx_mapping.dtype, pos.dtype, seed.dtype, temperature.dtype,
        apply_temperature, is_drafting, use_fp64,
        None if logits_cache is None
        else (logits_cache.dtype, logits_cache.stride(0), logits_cache.stride(1)),
        None if logits_cache_col is None
        else (logits_cache_col.dtype, logits_cache_col.dim()),
    )
    if key in _PRECOMPILED_KEYS:
        return
    _PRECOMPILED_KEYS.add(key)

    dev = logits.device
    dev_idx = dev.index if dev.index is not None else 0
    prof = _get_profile(dev_idx)
    max_k = prof.num_cu * prof.programs_per_cu + 1
    greedy_opts = (False, True) if logits_cache is not None else (False,)

    for greedy in greedy_opts:
        compute_bound = (not greedy) and (not prof.random_use_split)
        seen = set()
        for k in range(1, max_k + 1):
            if compute_bound:
                num_splits = triton.cdiv(vocab_size, _COMPUTE_BOUND_CHUNK)
            else:
                align = prof.chunk_align
                max_splits = triton.cdiv(vocab_size, align)
                n = max(1, min(triton.cdiv(prof.num_cu * prof.programs_per_cu, k),
                               max_splits))
                chunk_size = triton.cdiv(triton.cdiv(vocab_size, n), align) * align
                num_splits = triton.cdiv(vocab_size, chunk_size)
            single = num_splits == 1
            if single in seen:
                continue
            seen.add(single)

            d_logits = logits.new_zeros((k, logits.stride(0)))[:, :vocab_size]
            d_idx = torch.full((k,), -1, dtype=expanded_idx_mapping.dtype, device=dev)
            d_pos = torch.zeros((k,), dtype=pos.dtype, device=dev)
            d_col = None
            if logits_cache_col is not None:
                shape = (k,) if logits_cache_col.dim() > 0 else ()
                d_col = torch.zeros(shape, dtype=logits_cache_col.dtype, device=dev)
            gumbel_sample(
                d_logits, d_idx, temperature, seed, d_pos,
                apply_temperature, is_drafting,
                logits_cache=logits_cache,
                logits_cache_col=d_col,
                use_fp64=use_fp64,
                all_greedy=greedy,
            )

def gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    is_drafting: bool,
    logits_cache: torch.Tensor | None = None,  # [max_num_reqs, num_cols, vocab_size]
    logits_cache_col: torch.Tensor | None = None,  # scalar or [num_tokens]
    use_fp64: bool = False,
    all_greedy: bool = False,
) -> torch.Tensor:
    """
    all_greedy: caller guarantees every request in the batch has
        temperature == 0. Must be decided from CPU-side state (no GPU sync).
    """
    num_tokens, vocab_size = logits.shape
    if logits_cache is not None:
        assert logits_cache.size(-1) >= vocab_size, (
            f"draft logits cache vocab dim ({logits_cache.size(-1)}) is narrower "
            f"than the sampled logits ({vocab_size}). Cached logits would be "
            "truncated."
        )

    _precompile_once(
        logits, expanded_idx_mapping.contiguous(), temperature, seed,
        pos.contiguous(), apply_temperature, is_drafting, logits_cache,
        logits_cache_col.contiguous() if logits_cache_col is not None else None,
        use_fp64,
    )

    # Fast path: all greedy, nothing to cache -> plain argmax.
    if all_greedy and logits_cache is None:
        # NOTE(woosuk): Use int64 for later indexing.
        return logits.argmax(dim=-1).view(-1)

    # Enforce contiguity on non-strided input tensors
    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    pos = pos.contiguous()
    if logits_cache_col is not None:
        logits_cache_col = logits_cache_col.contiguous()

    sampled = logits.new_empty(num_tokens, dtype=torch.int64)
    if num_tokens == 0:
        return sampled

    dev_idx = logits.device.index if logits.device.index is not None else 0
    prof = _get_profile(dev_idx)

    compute_bound = (not all_greedy) and (not prof.random_use_split)
    if compute_bound:
        # Original-like grid: num_tokens x ceil(vocab / 1024), one pass each.
        chunk_size = _COMPUTE_BOUND_CHUNK
        num_splits = triton.cdiv(vocab_size, chunk_size)
    else:
        # Split grid: total programs ~= programs_per_cu * num_cu.
        align = prof.chunk_align
        target_programs = prof.num_cu * prof.programs_per_cu
        max_splits = triton.cdiv(vocab_size, align)
        num_splits = max(1, min(triton.cdiv(target_programs, num_tokens), max_splits))
        chunk_size = triton.cdiv(triton.cdiv(vocab_size, num_splits), align) * align
        num_splits = triton.cdiv(vocab_size, chunk_size)  # no empty splits
    single_split = num_splits == 1

    if single_split:
        local_argmax = sampled  # dummy, unused
        local_max = sampled  # dummy, unused
        local_stride = 0
    else:
        local_argmax = logits.new_empty(num_tokens, num_splits, dtype=torch.int32)
        local_max = logits.new_empty(
            num_tokens,
            num_splits,
            dtype=torch.float64 if use_fp64 else torch.float32,
        )
        local_stride = num_splits

    per_token_col = logits_cache_col is not None and logits_cache_col.dim() > 0
    _gumbel_sample_split_kernel[(num_tokens, num_splits)](
        sampled,
        local_argmax,
        local_max,
        local_stride,
        logits_cache,
        logits_cache.stride(0) if logits_cache is not None else 0,
        logits_cache.stride(1) if logits_cache is not None else 0,
        logits_cache_col,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        chunk_size,
        SINGLE_SPLIT=single_split,
        IS_DRAFTING=is_drafting,
        APPLY_TEMPERATURE=apply_temperature,
        USE_FP64=use_fp64,
        PER_TOKEN_COL=per_token_col,
        ALL_GREEDY=all_greedy,
        COMPUTE_BOUND=compute_bound,
        BLOCK_SIZE=_SPLIT_BLOCK_SIZE,
        num_warps=prof.num_warps,
        num_stages=1,
    )

    if not single_split:
        _gumbel_reduce_kernel[(num_tokens,)](
            sampled,
            local_argmax,
            local_max,
            local_stride,
            num_splits,
            vocab_size,
            BLOCK_SPLITS=triton.next_power_of_2(
                triton.cdiv(vocab_size, _SPLIT_BLOCK_SIZE)
            ),
            num_warps=1,
        )
    # NOTE(woosuk): Use int64 for later indexing.
    return sampled