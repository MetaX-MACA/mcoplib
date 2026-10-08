"""Standalone extraction of SGLang's MHC post TileLang operator."""

import math

import tilelang
import tilelang.language as T
import torch


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
}


@tilelang.jit(pass_configs=PASS_CONFIGS)
def mhc_post_tilelang_baseline(
    a, b, c, d, x, hc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024
) -> tilelang.JITKernel:
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)
    a: T.Tensor((n, hc, hc), T.float32)
    b: T.Tensor((n, hc, h), T.bfloat16)
    c: T.Tensor((n, hc), T.float32)
    d: T.Tensor((n, h), T.bfloat16)
    x: T.Tensor((n, hc, h), T.bfloat16)

    with T.Kernel(n, threads=n_thr) as i_n:

        x_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        d_shared = T.alloc_shared(h_blk, T.bfloat16)

        x_local = T.alloc_fragment((hc, h_blk), T.float32)
        b_local = T.alloc_fragment((hc, h_blk), T.float32)
        d_local = T.alloc_fragment(h_blk, T.float32)

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        T.copy(a[i_n, 0, 0], a_local)
        T.copy(c[i_n, 0], c_local)

        for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
            T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
            T.copy(d[i_n, i0_h * h_blk], d_shared)

            T.copy(b_shared, b_local)
            T.copy(d_shared, d_local)
            for i_hco, i1_h in T.Parallel(hc, h_blk):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.serial(hc):
                    x_local[i_hco, i1_h] += (
                        a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
                    )
            T.copy(x_local, x_shared)
            T.copy(x_shared, x[i_n, 0, i0_h * h_blk])



@tilelang.jit(pass_configs=PASS_CONFIGS)
def mhc_post_tilelang_split_hidden(
    a, b, c, d, x, hc: int, hidden: int, n_thr: int = 128, h_blk: int = 512
) -> tilelang.JITKernel:
    """Split hidden tiles across thread blocks to increase grid parallelism."""
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)
    a: T.Tensor((n, hc, hc), T.float32)
    b: T.Tensor((n, hc, h), T.bfloat16)
    c: T.Tensor((n, hc), T.float32)
    d: T.Tensor((n, h), T.bfloat16)
    x: T.Tensor((n, hc, h), T.bfloat16)

    with T.Kernel(n, T.ceildiv(h, h_blk), threads=n_thr) as (i_n, i_h):

        h_start = i_h * h_blk
        x_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        d_shared = T.alloc_shared(h_blk, T.bfloat16)

        x_local = T.alloc_fragment((hc, h_blk), T.float32)
        b_local = T.alloc_fragment((hc, h_blk), T.float32)
        d_local = T.alloc_fragment(h_blk, T.float32)

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        T.copy(a[i_n, 0, 0], a_local)
        T.copy(c[i_n, 0], c_local)
        T.copy(b[i_n, 0, h_start], b_shared)
        T.copy(d[i_n, h_start], d_shared)

        T.copy(b_shared, b_local)
        T.copy(d_shared, d_local)
        for i_hco, i1_h in T.Parallel(hc, h_blk):
            x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
            for i_hci in T.serial(hc):
                x_local[i_hco, i1_h] += (
                    a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
                )
        T.copy(x_local, x_shared)
        T.copy(x_shared, x[i_n, 0, h_start])


def _select_mhc_post_config(hidden: int, num_tokens: int) -> tuple[int, int]:
    """Select (threads, hidden tile) for C600-U."""
    if num_tokens <= 56:
        return 128, 512
    if hidden in (5120, 7168):
        return 256, hidden // 4
    return 64, 512


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(residual)
    hc = residual.shape[-2]
    hidden = residual.shape[-1]
    threads, h_blk = _select_mhc_post_config(hidden, residual.shape[0])
    mhc_post_tilelang_split_hidden(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        out,
        hc,
        hidden,
        threads,
        h_blk,
    )
    return out


def mhc_post_baseline(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(residual)
    mhc_post_tilelang_baseline(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        out,
        residual.shape[-2],
        residual.shape[-1],
    )
    return out


__all__ = [
    "mhc_post_baseline",
    "mhc_post",
    "mhc_post_tilelang_baseline",
    "mhc_post_tilelang_split_hidden",
    "_select_mhc_post_config",
]
