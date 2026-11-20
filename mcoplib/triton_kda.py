# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
"""Standalone Triton kernels used by the KDA/GLA forward path.

The implementations are adapted from SGLang's FLA KDA operator.  Keeping the
two entry points in mcoplib avoids importing the full SGLang kernel package at
runtime while preserving the original tensor layouts and varlen semantics.
"""

from __future__ import annotations

import functools
import inspect
import os
from typing import Any, Callable, Optional

import torch
import triton
import triton.language as tl


RCP_LN2 = 1.4426950216293335

_SUPPORTS_AUTOTUNE_CACHE = (
    "cache_results" in inspect.signature(triton.autotune).parameters
)
_AUTOTUNE_CACHE_KWARGS = (
    {"cache_results": os.getenv("FLA_CACHE_RESULTS", "1") == "1"}
    if _SUPPORTS_AUTOTUNE_CACHE
    else {}
)


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""

    return -(a // -b)


def _tensor_cache(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Small identity cache for device-side sequence metadata."""

    cache_entries = []

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries
        for index, (old_args, old_kwargs, result) in enumerate(cache_entries):
            if len(args) == len(old_args) and len(kwargs) == len(old_kwargs):
                if all(a is b for a, b in zip(args, old_args)) and all(
                    key in old_kwargs and value is old_kwargs[key]
                    for key, value in kwargs.items()
                ):
                    cache_entries.append(cache_entries.pop(index))
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
    """Map each packed chunk to ``(sequence_index, local_chunk_index)``."""

    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    indices = torch.cat(
        [torch.arange(int(n)) for n in triton.cdiv(lengths, chunk_size).tolist()]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


try:
    _CURRENT_TARGET = triton.runtime.driver.active.get_current_target()
    _IS_XCORE1000_OR_1008 = (
        _CURRENT_TARGET.backend == "maca" and _CURRENT_TARGET.arch == 80
    )
except BaseException:
    _IS_XCORE1000_OR_1008 = False


_CHUNK_GLA_CONFIGS = [
    triton.Config(
        {"BK": block_k, "BV": block_v},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for block_k in (32, 64)
    for block_v in (64, 128)
    for num_warps in (2, 4, 8)
    for num_stages in (2, 3, 4)
]


@triton.jit(do_not_specialize=["T"])
def _chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Produce the GLA output from recurrent state and intra-chunk terms."""

    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        nt = tl.cdiv(T, BT)
        i_tg = i_b * nt + i_t
        bos, eos = i_b * T, i_b * T + T

    causal_mask = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    output = tl.zeros([BT, BV], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        q_block_ptr = tl.make_block_ptr(
            q + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        g_block_ptr = tl.make_block_ptr(
            g + (bos * H + i_h) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        h_block_ptr = tl.make_block_ptr(
            h + (i_tg * H + i_h) * V * K,
            (V, K),
            (K, 1),
            (i_v * BV, i_k * BK),
            (BV, BK),
            (1, 0),
        )

        q_block = tl.load(q_block_ptr, boundary_check=(0, 1))
        q_block = (q_block * scale).to(q_block.dtype)
        gate_block = tl.load(g_block_ptr, boundary_check=(0, 1))
        gated_q = (q_block * tl.exp2(gate_block)).to(q_block.dtype)
        h_block = tl.load(h_block_ptr, boundary_check=(0, 1))
        # Keep the source kernel's condition: it avoids a backend issue while
        # remaining true for every compile-time loop iteration.
        if i_k >= 0:
            output += tl.dot(gated_q, tl.trans(h_block).to(gated_q.dtype))

    v_block_ptr = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        o + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_t * BT, i_v * BV),
        (BT, BV),
        (1, 0),
    )
    a_block_ptr = tl.make_block_ptr(
        A + (bos * H + i_h) * BT,
        (T, BT),
        (H * BT, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    v_block = tl.load(v_block_ptr, boundary_check=(0, 1))
    a_block = tl.load(a_block_ptr, boundary_check=(0, 1))
    a_block = tl.where(causal_mask, a_block, 0.0).to(v_block.dtype)
    output += tl.dot(a_block, v_block, allow_tf32=False)
    tl.store(
        o_block_ptr,
        output.to(o_block_ptr.dtype.element_ty),
        boundary_check=(0, 1),
    )


chunk_gla_fwd_kernel_o = triton.autotune(
    configs=_CHUNK_GLA_CONFIGS,
    key=["BT", "IS_VARLEN"],
    **_AUTOTUNE_CACHE_KWARGS,
)(_chunk_gla_fwd_kernel_o)


def _chunk_gla_c500_config(
    T: int, H: int, K: int, V: int, BT: int
) -> Optional[dict[str, int]]:
    """Return a zero-spill fixed launch for traced C500 shape families."""

    if not _IS_XCORE1000_OR_1008 or (K, V, BT) != (128, 128, 64):
        return None
    if H == 8 and T >= 1024:
        return {"BK": 64, "BV": 128, "num_warps": 4, "num_stages": 4}
    if H == 8 and T <= BT:
        return {"BK": 64, "BV": 32, "num_warps": 8, "num_stages": 4}
    if H == 8:
        return {"BK": 64, "BV": 64, "num_warps": 4, "num_stages": 3}
    if H == 64:
        return {"BK": 64, "BV": 64, "num_warps": 4, "num_stages": 3}
    return None


def _chunk_gla_c500_baseline_config(
    H: int, K: int, V: int, BT: int
) -> Optional[dict[str, int]]:
    """Reproduce the external C500 autotune winner used for baseline data."""

    if (
        not _IS_XCORE1000_OR_1008
        or (K, V, BT) != (128, 128, 64)
        or H not in (8, 64)
    ):
        return None
    return {"BK": 64, "BV": 64, "num_warps": 4, "num_stages": 3}


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
    chunk_indices: Optional[torch.LongTensor] = None,
    _use_c500_optimized: bool = True,
) -> torch.Tensor:
    """Launch :func:`chunk_gla_fwd_kernel_o` using FLA tensor layouts."""

    if q.ndim != 4 or v.ndim != 4 or g.ndim != 4:
        raise ValueError("q, v, and g must have shape [B, T, H, D]")
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    if cu_seqlens is not None and B != 1:
        raise ValueError("varlen mode requires packed tensors with batch size 1")
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    def grid(meta):
        return (cdiv(V, meta["BV"]), NT, B * H)

    static_config = (
        _chunk_gla_c500_config(T, H, K, V, BT)
        if _use_c500_optimized
        else _chunk_gla_c500_baseline_config(H, K, V, BT)
    )
    kernel = (
        _chunk_gla_fwd_kernel_o
        if static_config is not None
        else chunk_gla_fwd_kernel_o
    )
    kernel[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        IS_VARLEN=cu_seqlens is not None,
        **(static_config or {}),
    )
    return o


@triton.jit
def _softplus_fwd(x):
    return tl.where(x < 20.0, tl.log(1.0 + tl.exp(x)), x)


# C500 follows the same low-shared-memory search space observed in the source
# traces. Other targets retain the upstream high-shared-memory alternatives.
_GATE_BLOCK_SIZES = (16, 32) if _IS_XCORE1000_OR_1008 else (32, 64)
_GATE_CUMSUM_CONFIGS = [
    triton.Config({"BS": block_s}, num_warps=num_warps)
    for block_s in _GATE_BLOCK_SIZES
    for num_warps in (2, 4, 8)
]


@triton.jit(do_not_specialize=["T"])
def _kda_gate_chunk_cumsum_vector_kernel(
    s,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    lower_bound,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    """Fuse KDA gate activation with chunk-local cumulative sum."""

    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    s_block_ptr = tl.make_block_ptr(
        s + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        o + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )
    values = tl.load(s_block_ptr, boundary_check=(0, 1)).to(tl.float32)

    if HAS_BIAS:
        bias_block_ptr = tl.make_block_ptr(
            dt_bias + i_h * S,
            (S,),
            (1,),
            (i_s * BS,),
            (BS,),
            (0,),
        )
        bias = tl.load(bias_block_ptr, boundary_check=(0,)).to(tl.float32)
        values += bias[None, :]

    a_log = tl.load(A_log + i_h).to(tl.float32)
    if not USE_LOWER_BOUND:
        gate = -tl.exp(a_log) * _softplus_fwd(values)
    else:
        gate = lower_bound * tl.sigmoid(tl.exp(a_log) * values)

    output = tl.cumsum(gate, axis=0)
    if HAS_SCALE:
        output *= scale
    tl.store(
        o_block_ptr,
        output.to(o_block_ptr.dtype.element_ty),
        boundary_check=(0, 1),
    )


kda_gate_chunk_cumsum_vector_kernel = triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)(
    triton.autotune(
        configs=_GATE_CUMSUM_CONFIGS,
        key=["H", "S", "BT", "IS_VARLEN"],
        **_AUTOTUNE_CACHE_KWARGS,
    )(_kda_gate_chunk_cumsum_vector_kernel)
)


def _kda_gate_c500_config(
    T: int, H: int, S: int, BT: int
) -> Optional[dict[str, int]]:
    """Return the fixed zero-spill C500 launch selected by the shape sweep."""

    if not _IS_XCORE1000_OR_1008 or (S, BT) != (128, 64):
        return None
    if H == 8 and T >= 1024:
        return {"BS": 64, "num_warps": 2, "num_stages": 3}
    if H == 8:
        return {"BS": 16, "num_warps": 8, "num_stages": 3}
    if H == 64 and T == 1:
        return {"BS": 16, "num_warps": 1, "num_stages": 4}
    if H == 64:
        return {"BS": 16, "num_warps": 2, "num_stages": 3}
    return None


def _kda_gate_c500_baseline_config(
    H: int, S: int, BT: int
) -> Optional[dict[str, int]]:
    """Reproduce the external C500 autotune winners used for baseline data."""

    if not _IS_XCORE1000_OR_1008 or (S, BT) != (128, 64):
        return None
    if H == 8:
        return {"BS": 16, "num_warps": 8, "num_stages": 3}
    if H == 64:
        return {"BS": 16, "num_warps": 2, "num_stages": 3}
    return None


def kda_gate_chunk_cumsum(
    g: torch.Tensor,
    A_log: torch.Tensor,
    chunk_size: int,
    scale: Optional[float] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = torch.float,
    chunk_indices: Optional[torch.LongTensor] = None,
    lower_bound: Optional[float] = None,
    _use_c500_optimized: bool = True,
) -> torch.Tensor:
    """Apply the KDA gate and compute a cumulative sum within each chunk."""

    if g.ndim != 4:
        raise ValueError("g must have shape [B, T, H, S]")
    B, T, H, S = g.shape
    if cu_seqlens is not None and B != 1:
        raise ValueError("varlen mode requires packed tensors with batch size 1")
    if chunk_size < 1 or chunk_size & (chunk_size - 1):
        raise ValueError("chunk_size must be a power of two")
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = cdiv(T, chunk_size) if cu_seqlens is None else len(chunk_indices)
    output = torch.empty_like(g, dtype=output_dtype or g.dtype)

    def grid(meta):
        return (cdiv(meta["S"], meta["BS"]), NT, B * H)

    static_config = (
        _kda_gate_c500_config(T, H, S, chunk_size)
        if _use_c500_optimized
        else _kda_gate_c500_baseline_config(H, S, chunk_size)
    )
    kernel = (
        _kda_gate_chunk_cumsum_vector_kernel
        if static_config is not None
        else kda_gate_chunk_cumsum_vector_kernel
    )
    launch_meta = (
        {
            "HAS_BIAS": dt_bias is not None,
            "HAS_SCALE": scale is not None,
            "IS_VARLEN": cu_seqlens is not None,
            "USE_LOWER_BOUND": lower_bound is not None,
            **static_config,
        }
        if static_config is not None
        else {}
    )
    kernel[grid](
        s=g,
        A_log=A_log,
        dt_bias=dt_bias,
        o=output,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=lower_bound,
        T=T,
        H=H,
        S=S,
        BT=chunk_size,
        **launch_meta,
    )
    return output


__all__ = [
    "RCP_LN2",
    "chunk_gla_fwd_kernel_o",
    "chunk_gla_fwd_o_gk",
    "kda_gate_chunk_cumsum_vector_kernel",
    "kda_gate_chunk_cumsum",
    "prepare_chunk_indices",
]
