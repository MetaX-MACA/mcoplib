# Adapted from flash-linear-attention project.
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
"""SGLang FLA chunk delta-state kernel, hosted as a standalone mcoplib op.

The public K=V=128 path uses the retained BV=64, four-warp specialization.
The original SGLang BV=32, four-warp configuration remains available as an
explicit baseline. Other K/V shapes retain the conservative baseline launch.

K=V=128 launches use these specializations:

1. On MetaX C500, ``_chunk_gated_delta_rule_fwd_kernel_h_k128_fused`` with
   BV=32 for the N=4, H=Hg=12, T=8192 packed production family;
2. ``_chunk_gated_delta_rule_fwd_kernel_h_k128_single_chunk`` for a single
   whole chunk at N=1, H=Hg=8;
3. ``_chunk_gated_delta_rule_fwd_kernel_h_k128_fused`` for N=1, H=Hg=8;
4. ``_chunk_gated_delta_rule_fwd_kernel_h_k128_tailfree`` when every sequence
   length is a whole number of chunks: it drops all ``boundary_check`` work and
   keeps the recurrent state transposed as ``[64, BV]`` so the loop needs no
   ``tl.trans``.

Other launches use ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64``.
"""

import functools
from typing import Any, Callable, Optional, Tuple

import torch
import triton
import triton.language as tl


CHUNK_SIZE = 64


@functools.lru_cache(maxsize=None)
def _is_metax_c500(device_index: int) -> bool:
    """Return whether a logical CUDA/MACA device is a MetaX C500.

    The packed H=12 specialization has opposite performance behavior on C500
    and C600-U. Cache the device-name lookup per logical device so launch-time
    dispatch supports heterogeneous multi-GPU processes without repeatedly
    querying the runtime.
    """

    device_name = torch.cuda.get_device_name(device_index).upper()
    return "METAX" in device_name and "C500" in device_name


def _tensor_cache(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Four-entry identity cache matching FLA's tensor metadata cache."""

    cache_entries = []

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries
        for i, (old_args, old_kwargs, result) in enumerate(cache_entries):
            if len(args) == len(old_args) and len(kwargs) == len(old_kwargs):
                if all(a is b for a, b in zip(args, old_args)) and all(
                    key in old_kwargs and value is old_kwargs[key]
                    for key, value in kwargs.items()
                ):
                    cache_entries.append(cache_entries.pop(i))
                    return result

        result = fn(*args, **kwargs)
        if len(cache_entries) >= 4:
            cache_entries.pop(0)
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


@_tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    indices = torch.cat(
        [torch.arange(int(n)) for n in triton.cdiv(lengths, chunk_size).tolist()]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@_tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(lengths, chunk_size)]
    ).cumsum(-1)


@_tensor_cache
def is_tail_free(cu_seqlens: torch.Tensor, chunk_size: int) -> bool:
    """True when every sequence length is a whole number of chunks.

    Cached on tensor identity: the check forces a device sync, and the callers
    that re-launch on the same ``cu_seqlens`` (benchmarks, repeated decode)
    must not pay it per launch.
    """
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    return bool(torch.all(lengths % chunk_size == 0).item())



@triton.jit
def _safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    initial_state,
    initial_state_indices,
    stride_init_state,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_UPDATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NT_BUCKET: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K

    # State-slot addressing. ``stride_init_state`` is supplied by the caller
    # (``initial_state.stride(0)``): envelope-strided state pools (page-major /
    # unified memory) have a per-slot pitch that is not H*V*K, and an int64
    # slot index keeps the pitch product from overflowing int32. A negative
    # slot id marks a padded row and is never read or written.
    index = tl.load(initial_state_indices + i_n).to(tl.int64)
    valid_state = index >= 0
    # Padded rows carry the -1 sentinel. At BV=64 the state tiles sit in
    # registers comfortably, so clamping the slot and scaling the loaded
    # state by zero is cheaper than a runtime branch around the load
    # (measured 1.045-1.062x). At BV=32 that extra live value tips an
    # already tight allocation over (n_regs 256, 24 spills) and regresses
    # ~12%, so the narrow configuration keeps the branch.
    UNCONDITIONAL_STATE_LOAD: tl.constexpr = BV == 64
    if UNCONDITIONAL_STATE_LOAD:
        b_keep = tl.where(valid_state, 1.0, 0.0)
        index_safe = tl.maximum(index, 0)
    else:
        b_keep = 1.0
        index_safe = index
    h0 = initial_state + index_safe * stride_init_state
    ht = initial_state + index_safe * stride_init_state
    if USE_INITIAL_STATE:
        h0 += i_h * V * K
    if INPLACE_UPDATE:
        ht += i_h * V * K

    if USE_INITIAL_STATE:
        if UNCONDITIONAL_STATE_LOAD or valid_state:
            p_h0_1 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
            )
            b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32) * b_keep
            if K > 64:
                p_h0_2 = tl.make_block_ptr(
                    h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
                )
                b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32) * b_keep
            if K > 128:
                p_h0_3 = tl.make_block_ptr(
                    h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
                )
                b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32) * b_keep
            if K > 192:
                p_h0_4 = tl.make_block_ptr(
                    h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
                )
                b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32) * b_keep

    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t * stride_h,
            (V, K),
            (K, 1),
            (i_v * BV, 0),
            (BV, 64),
            (1, 0),
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t * stride_h,
                (V, K),
                (K, 1),
                (i_v * BV, 64),
                (BV, 64),
                (1, 0),
            )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(
                h + i_t * stride_h,
                (V, K),
                (K, 1),
                (i_v * BV, 128),
                (BV, 64),
                (1, 0),
            )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t * stride_h,
                (V, K),
                (K, 1),
                (i_v * BV, 192),
                (BV, 64),
                (1, 0),
            )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))

        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v
        if SAVE_NEW_VALUE:
            p_v_new = tl.make_block_ptr(
                v_new,
                (T, V),
                (stride_v, 1),
                (i_t * BT, i_v * BV),
                (BT, BV),
                (1, 0),
            )
            tl.store(
                p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1)
            )

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v *= _safe_exp(b_g_last - b_g)[:, None]
            b_g_last = tl.exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=o_k1 < K,
                other=0.0,
            )
            if USE_EXP2:
                b_h1 *= tl.exp2(b_gk_last1)[None, :]
            else:
                b_h1 *= tl.exp(b_gk_last1)[None, :]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=o_k2 < K,
                    other=0.0,
                )
                if USE_EXP2:
                    b_h2 *= tl.exp2(b_gk_last2)[None, :]
                else:
                    b_h2 *= tl.exp(b_gk_last2)[None, :]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=o_k3 < K,
                    other=0.0,
                )
                if USE_EXP2:
                    b_h3 *= tl.exp2(b_gk_last3)[None, :]
                else:
                    b_h3 *= tl.exp(b_gk_last3)[None, :]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=o_k4 < K,
                    other=0.0,
                )
                if USE_EXP2:
                    b_h4 *= tl.exp2(b_gk_last4)[None, :]
                else:
                    b_h4 *= tl.exp(b_gk_last4)[None, :]

        b_v = b_v.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.trans(tl.dot(b_k, b_v))
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.trans(tl.dot(b_k, b_v))
        if K > 128:
            p_k = tl.make_block_ptr(
                k,
                (K, T),
                (1, stride_k),
                (128, i_t * BT),
                (64, BT),
                (0, 1),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.trans(tl.dot(b_k, b_v))
        if K > 192:
            p_k = tl.make_block_ptr(
                k,
                (K, T),
                (1, stride_k),
                (192, i_t * BT),
                (64, BT),
                (0, 1),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.trans(tl.dot(b_k, b_v))

    if INPLACE_UPDATE and valid_state:
        p_ht = tl.make_block_ptr(
            ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
        )
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
            )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def _chunk_gated_delta_rule_fwd_kernel_h_k128_fused(
    k,
    v,
    w,
    v_new,
    gk,
    h,
    initial_state,
    initial_state_indices,
    stride_init_state,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    BV: tl.constexpr,
    NT_BUCKET: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    """K=V=128 specialization using fused K=128 dot operands."""

    K: tl.constexpr = 128
    V: tl.constexpr = 128
    BT: tl.constexpr = 64
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)

    b_h = tl.zeros([BV, K], dtype=tl.float32)
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K
    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    v_new += ((bos * H + i_h) * V).to(tl.int64)

    index = tl.load(initial_state_indices + i_n).to(tl.int64)
    valid_state = index >= 0
    state = initial_state + index * stride_init_state + i_h * V * K

    if valid_state:
        p_h0 = tl.make_block_ptr(
            state, (V, K), (K, 1), (i_v * BV, 0), (BV, K), (1, 0)
        )
        b_h += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_h = tl.make_block_ptr(
            h + i_t * stride_h,
            (V, K),
            (K, 1),
            (i_v * BV, 0),
            (BV, K),
            (1, 0),
        )
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(
            w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, K), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, tl.trans(b_h).to(b_w.dtype))

        p_v = tl.make_block_ptr(
            v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v
        p_v_new = tl.make_block_ptr(
            v_new,
            (T, V),
            (stride_v, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        tl.store(
            p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1)
        )

        last_idx = min((i_t + 1) * BT, T) - 1
        o_k = tl.arange(0, K)
        b_gk_last = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k)
        if USE_EXP2:
            b_decay = tl.exp2(b_gk_last)
        else:
            b_decay = tl.exp(b_gk_last)
        b_h *= b_decay[None, :]

        p_k = tl.make_block_ptr(
            k, (K, T), (1, stride_k), (0, i_t * BT), (K, BT), (0, 1)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h += tl.trans(tl.dot(b_k, b_v.to(b_k.dtype)))

    if valid_state:
        p_ht = tl.make_block_ptr(
            state, (V, K), (K, 1), (i_v * BV, 0), (BV, K), (1, 0)
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def _chunk_gated_delta_rule_fwd_kernel_h_k128_single_chunk(
    k,
    v,
    w,
    v_new,
    gk,
    h,
    initial_state,
    initial_state_indices,
    stride_init_state,
    T,
    USE_EXP2: tl.constexpr,
):
    """Single-chunk specialization for N=1, H=Hg=8, K=V=128."""

    H: tl.constexpr = 8
    Hg: tl.constexpr = 8
    K: tl.constexpr = 128
    V: tl.constexpr = 128
    BT: tl.constexpr = 64
    BV: tl.constexpr = 64
    i_v, i_h = tl.program_id(0), tl.program_id(1)

    stride_v: tl.constexpr = H * V
    stride_k: tl.constexpr = Hg * K
    stride_w: tl.constexpr = H * K
    h += i_h * V * K
    v += i_h * V
    k += i_h * K
    w += i_h * K
    v_new += i_h * V

    index = tl.load(initial_state_indices).to(tl.int64)
    valid_state = index >= 0
    state = initial_state + index * stride_init_state + i_h * V * K

    b_h = tl.zeros([BV, K], dtype=tl.float32)
    if valid_state:
        p_h0 = tl.make_block_ptr(
            state, (V, K), (K, 1), (i_v * BV, 0), (BV, K), (1, 0)
        )
        b_h += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    p_h = tl.make_block_ptr(
        h, (V, K), (K, 1), (i_v * BV, 0), (BV, K), (1, 0)
    )
    tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

    p_w = tl.make_block_ptr(
        w, (T, K), (stride_w, 1), (0, 0), (BT, K), (1, 0)
    )
    b_w = tl.load(p_w, boundary_check=(0, 1))
    b_v = tl.dot(b_w, tl.trans(b_h).to(b_w.dtype))

    p_v = tl.make_block_ptr(
        v, (T, V), (stride_v, 1), (0, i_v * BV), (BT, BV), (1, 0)
    )
    b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v
    p_v_new = tl.make_block_ptr(
        v_new,
        (T, V),
        (stride_v, 1),
        (0, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    tl.store(p_v_new, b_v.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

    o_k = tl.arange(0, K)
    b_gk_last = tl.load(gk + (T - 1) * H * K + i_h * K + o_k)
    if USE_EXP2:
        b_decay = tl.exp2(b_gk_last)
    else:
        b_decay = tl.exp(b_gk_last)
    b_h *= b_decay[None, :]

    p_k = tl.make_block_ptr(
        k, (K, T), (1, stride_k), (0, 0), (K, BT), (0, 1)
    )
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_h += tl.trans(tl.dot(b_k, b_v.to(b_k.dtype)))

    if valid_state:
        p_ht = tl.make_block_ptr(
            state, (V, K), (K, 1), (i_v * BV, 0), (BV, K), (1, 0)
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def _chunk_gated_delta_rule_fwd_kernel_h_k128_tailfree(
    k,
    v,
    w,
    v_new,
    gk,
    h,
    initial_state,
    initial_state_indices,
    stride_init_state,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    BV: tl.constexpr,
    NT_BUCKET: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    """K=V=128 tail-free specialization.

    Only launched when every sequence length is a whole number of ``BT``
    chunks, so every block is full and no ``boundary_check`` is needed.

    The recurrent state is held transposed as ``[64, BV]``: dot1 consumes it
    directly as the right operand and dot2 produces it directly as the
    accumulator, so no ``tl.trans`` appears in the loop. The SGLang/FLA state
    contract is preserved (int64 slot index, caller-supplied envelope stride,
    negative slot id = padded row that is neither read nor written).
    """

    K: tl.constexpr = 128
    V: tl.constexpr = 128
    BT: tl.constexpr = 64
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)

    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K
    h += ((boh * H + i_h) * V * K).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    v_new += ((bos * H + i_h) * V).to(tl.int64)

    index = tl.load(initial_state_indices + i_n).to(tl.int64)
    valid_state = index >= 0
    index_safe = tl.maximum(index, 0)
    state = initial_state + index_safe * stride_init_state + i_h * V * K
    p_state1 = tl.make_block_ptr(
        state, (K, V), (1, K), (0, i_v * BV), (64, BV), (0, 1)
    )
    p_state2 = tl.make_block_ptr(
        state, (K, V), (1, K), (64, i_v * BV), (64, BV), (0, 1)
    )

    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    b_keep = tl.where(valid_state, 1.0, 0.0)
    b_h1 += tl.load(p_state1).to(tl.float32) * b_keep
    b_h2 += tl.load(p_state2).to(tl.float32) * b_keep

    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t * stride_h, (K, V), (1, K), (0, i_v * BV), (64, BV), (0, 1)
        )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty))
        p_h2 = tl.make_block_ptr(
            h + i_t * stride_h, (K, V), (1, K), (64, i_v * BV), (64, BV), (0, 1)
        )
        tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty))

        b_w1 = tl.load(
            tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        )
        b_w2 = tl.load(
            tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0))
        )
        b_v = tl.load(
            tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        )
        b_v = b_v - (
            tl.dot(b_w1, b_h1.to(b_w1.dtype)) + tl.dot(b_w2, b_h2.to(b_w2.dtype))
        )
        p_vn = tl.make_block_ptr(
            v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_vn, b_v.to(p_vn.dtype.element_ty))

        last_idx = min((i_t + 1) * BT, T) - 1
        o_k1 = tl.arange(0, 64)
        o_k2 = 64 + o_k1
        b_gk_last1 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k1)
        b_gk_last2 = tl.load(gk + (bos + last_idx) * H * K + i_h * K + o_k2)
        if USE_EXP2:
            b_decay1 = tl.exp2(b_gk_last1)
            b_decay2 = tl.exp2(b_gk_last2)
        else:
            b_decay1 = tl.exp(b_gk_last1)
            b_decay2 = tl.exp(b_gk_last2)
        b_h1 *= b_decay1[:, None]
        b_h2 *= b_decay2[:, None]

        b_k1 = tl.load(
            tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        )
        b_k2 = tl.load(
            tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
        )
        b_h1 += tl.dot(b_k1, b_v.to(b_k1.dtype))
        b_h2 += tl.dot(b_k2, b_v.to(b_k2.dtype))

    if valid_state:
        tl.store(p_state1, b_h1.to(p_state1.dtype.element_ty))
        tl.store(p_state2, b_h2.to(p_state2.dtype.element_ty))


def _launch_chunk_gated_delta_rule_fwd_h(
    *,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    h: torch.Tensor,
    v_new: Optional[torch.Tensor],
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_offsets: Optional[torch.Tensor] = None,
    logical_batch_size: Optional[int] = None,
    total_chunks: Optional[int] = None,
    block_v: int = 32,
    num_warps: int = 4,
    num_stages: int = 1,
    use_exp2: bool = False,
) -> None:
    """Launch into preallocated outputs; useful for kernel-only baselines."""

    B, T, Hg, K = k.shape
    H, V = u.shape[-2:]
    BT = CHUNK_SIZE
    if use_exp2 and g is not None:
        raise ValueError(
            "use_exp2 selects the base-2 gate path; it only applies to the "
            "per-channel gk argument and is incompatible with the scalar g "
            "argument"
        )
    # Envelope-strided pools have a per-slot pitch that is not H*V*K.
    stride_init_state = initial_state.stride(0)
    if cu_seqlens is None:
        N, NT = B, triton.cdiv(T, BT)
    else:
        N = logical_batch_size or len(cu_seqlens) - 1
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
        NT = total_chunks or int(chunk_offsets[-1].item())

    device_index = k.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    # On C500, the packed H=12 production family benefits from the fused K=128
    # operand path at BV=32. C600-U regresses on that path, so it and unmeasured
    # devices retain the generic BV=64 kernel.
    use_k128_packed_h12 = (
        _is_metax_c500(device_index)
        and K == 128
        and V == 128
        and T == 8192
        and N == 4
        and H == 12
        and Hg == 12
        and block_v == 64
        and num_warps == 4
        and num_stages == 1
        and g is None
        and gk is not None
        and v_new is not None
        and cu_seqlens is not None
    )
    if use_k128_packed_h12:
        _chunk_gated_delta_rule_fwd_kernel_h_k128_fused[(4, N * H)](
            k=k,
            v=u,
            w=w,
            v_new=v_new,
            gk=gk,
            h=h,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            stride_init_state=stride_init_state,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            T=T,
            H=H,
            Hg=Hg,
            BV=32,
            NT_BUCKET=0 if NT <= 32 else (1 if NT <= 128 else 2),
            USE_EXP2=use_exp2,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return

    use_k128_fused = (
        K == 128
        and V == 128
        # Boundary sweep: long sequences regress as soon as N > 1 or H > 8.
        # Keep this specialization on the verified production shape family.
        and N == 1
        and H == 8
        and Hg == 8
        and block_v == 64
        and num_warps == 4
        and num_stages == 1
        and g is None
        and gk is not None
        and v_new is not None
        and cu_seqlens is not None
    )
    if use_k128_fused and NT == 1 and T <= BT:
        _chunk_gated_delta_rule_fwd_kernel_h_k128_single_chunk[(2, H)](
            k=k,
            v=u,
            w=w,
            v_new=v_new,
            gk=gk,
            h=h,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            stride_init_state=stride_init_state,
            T=T,
            USE_EXP2=use_exp2,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return

    if use_k128_fused:
        # The long recurrent shape benefits from twice as many independent
        # V tiles. Short single-chunk shapes stay on the BV64 kernel above.
        k128_long_bv = 32
        _chunk_gated_delta_rule_fwd_kernel_h_k128_fused[
            (triton.cdiv(V, k128_long_bv), N * H)
        ](
            k=k,
            v=u,
            w=w,
            v_new=v_new,
            gk=gk,
            h=h,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            stride_init_state=stride_init_state,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            T=T,
            H=H,
            Hg=Hg,
            BV=k128_long_bv,
            NT_BUCKET=0 if NT <= 32 else (1 if NT <= 128 else 2),
            USE_EXP2=use_exp2,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return

    # Tail-free K=V=128 specialization. Restricted to the exact contract that
    # was measured (gk-only, BV=64, four warps, one stage) and to launches
    # where every sequence length is a whole number of chunks; every other
    # launch keeps the general kernel unchanged.
    use_k128_tailfree = (
        K == 128
        and V == 128
        and block_v == 64
        and num_warps == 4
        and num_stages == 1
        and g is None
        and gk is not None
        and v_new is not None
        and cu_seqlens is not None
        and not use_k128_fused
        and is_tail_free(cu_seqlens, BT)
    )
    if use_k128_tailfree:
        _chunk_gated_delta_rule_fwd_kernel_h_k128_tailfree[
            (triton.cdiv(V, block_v), N * H)
        ](
            k=k,
            v=u,
            w=w,
            v_new=v_new,
            gk=gk,
            h=h,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            stride_init_state=stride_init_state,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            T=T,
            H=H,
            Hg=Hg,
            BV=block_v,
            NT_BUCKET=0 if NT <= 32 else (1 if NT <= 128 else 2),
            USE_EXP2=use_exp2,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        stride_init_state=stride_init_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BV=block_v,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_INITIAL_STATE=True,
        INPLACE_UPDATE=True,
        USE_EXP2=use_exp2,
        SAVE_NEW_VALUE=v_new is not None,
        IS_VARLEN=cu_seqlens is not None,
        NT_BUCKET=0 if NT <= 32 else (1 if NT <= 128 else 2),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    use_exp2: bool = False,
    *,
    _block_v: Optional[int] = None,
    _num_warps: Optional[int] = None,
    _num_stages: int = 1,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the retained target specialization and update state in place.

    ``use_exp2`` mirrors SGLang/FLA: it selects the base-2 gate path for the
    per-channel ``gk`` argument, which requires the caller to have already
    scaled the gates into log2 space (``gate * log2(e)``). It is incompatible
    with the scalar ``g`` argument, which always keeps natural-exp.
    """

    B, T, Hg, K = k.shape
    H, V = u.shape[-2:]
    BT = CHUNK_SIZE
    if _block_v is None:
        _block_v = 64 if K == 128 and V == 128 else 32
    if _num_warps is None:
        _num_warps = 4
    if K > 256:
        raise ValueError("current kernel does not support K > 256")
    if H % Hg != 0:
        raise ValueError(f"H ({H}) must be divisible by Hg ({Hg})")
    if initial_state is None or initial_state_indices is None:
        raise ValueError(
            "initial_state and initial_state_indices are required because the "
            "production kernel updates the final state in place"
        )
    if use_exp2 and g is not None:
        raise ValueError(
            "use_exp2 selects the base-2 gate path; it only applies to the "
            "per-channel gk argument and is incompatible with the scalar g "
            "argument"
        )

    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        N = len(cu_seqlens) - 1
        NT = len(chunk_indices)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    h = k.new_empty(B, NT, H, V, K)
    v_new = torch.empty_like(u) if save_new_value else None
    _launch_chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        h=h,
        v_new=v_new,
        g=g,
        gk=gk,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        logical_batch_size=N,
        total_chunks=NT,
        block_v=_block_v,
        num_warps=_num_warps,
        num_stages=_num_stages,
        use_exp2=use_exp2,
    )
    return h, v_new


def chunk_gated_delta_rule_fwd_h_baseline(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the frozen BV=32, four-warp baseline."""

    return chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        save_new_value=save_new_value,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        _block_v=32,
        _num_warps=4,
        _num_stages=1,
    )


def chunk_gated_delta_rule_fwd_h_bv64_w4(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the retained BV=64, four-warp specialization for K=V=128."""

    if k.shape[-1] != 128 or u.shape[-1] != 128:
        raise ValueError("BV64/W4 specialization requires K=V=128")
    return chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        save_new_value=save_new_value,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        _block_v=64,
        _num_warps=4,
        _num_stages=1,
    )


__all__ = [
    "CHUNK_SIZE",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_gated_delta_rule_fwd_h_baseline",
    "chunk_gated_delta_rule_fwd_h_bv64_w4",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
]
