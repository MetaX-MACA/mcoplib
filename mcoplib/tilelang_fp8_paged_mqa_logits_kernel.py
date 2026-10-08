import functools
from functools import lru_cache
from typing import Any, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.utils import is_gfx95_supported, is_hip, is_maca

tilelang.set_log_level("WARNING")

# Workaround a tilelang bug: BaseKernelAdapter._legalize_result_idx mutates the
# `out_idx` list in place when normalising negative indices to positive ones.
# That breaks any @tilelang.jit factory that compiles two prim_funcs with
# different param counts (e.g. our unified single/dual partial kernel) — the
# second compile sees indices already-converted for the first's len(params)
# and silently builds the wrong adapter, leading to IndexError at call time.
# Patch once on import to copy the list before mutation.
from tilelang.jit.adapter.base import (  # noqa: E402
    BaseKernelAdapter as _BaseKernelAdapter,
)

if not getattr(_BaseKernelAdapter, "_legalize_result_idx_patched", False):
    _orig_legalize = _BaseKernelAdapter._legalize_result_idx

    def _legalize_result_idx_safe(self, result_idx):
        if isinstance(result_idx, list):
            result_idx = list(result_idx)
        return _orig_legalize(self, result_idx)

    _BaseKernelAdapter._legalize_result_idx = _legalize_result_idx_safe
    _BaseKernelAdapter._legalize_result_idx_patched = True

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
# TL_DISABLE_FAST_MATH has deprecated in v0.1.7.post1 tilelang
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False

_is_hip = is_hip()
_is_maca = is_maca()
_is_gfx95_supported = is_gfx95_supported()
_is_fp8_fnuz = is_fp8_fnuz()

BF16 = "bfloat16"
FP8 = "float8_e4m3fnuz" if _is_fp8_fnuz else "float8_e4m3fn"
FP8_DTYPE = torch.float8_e4m3fnuz if _is_fp8_fnuz else torch.float8_e4m3fn
FP32 = "float32"
INT32 = "int32"
UINT8 = "uint8"


def fast_log2_ceil(x):
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)


def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


@lru_cache(maxsize=8)
def _pick_inner_iter(seq: int, ni: int, cu: int, block_per_cu: int) -> int:
    """
    Pick the largest valid inner_iter (power-of-two divisor of ni) that keeps
    enough work per CU (seq * ni / inner_iter / cu >= block_per_cu), so we avoid
    under-utilization while minimizing the number of partial groups.
    """

    max_it = int(seq * ni / (cu * block_per_cu))
    it = ni
    while it >= 2:
        if it <= max_it and ni % it == 0:
            return it
        it //= 2
    return 1





@functools.cache
def fp8_paged_mqa_logits_kernel(
    head_dim: int = 128,
    num_heads: int = 64,
    block_size: int = 64,
    clear_accum: bool = True,
    split_kv: int = 1,
) -> Any:
    N = T.symbolic("batch_size")
    L = T.symbolic("max_table_length")
    S = T.symbolic("max_seq_len")
    C = T.symbolic("num_blocks")
    B = block_size
    D = head_dim
    H = num_heads
    SK = int(split_kv)
    BLOCK_BYTES = B * (D + 4)
    SCALE_OFFSET = B * D

    assert D % 4 == 0
    assert H % 4 == 0
    assert D == 128
    assert SK >= 1

    @tilelang.jit(
        pass_configs={
            **pass_configs,
            tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        }
    )
    def fp8_paged_mqa_logits(
        q: T.Tensor[(N, H, D), FP8],
        kvcache_u8: T.Tensor[(C, BLOCK_BYTES), UINT8],
        weight: T.Tensor[(N, H), FP32],
        seq_lens: T.Tensor[(N,), INT32],
        page_table: T.Tensor[(N, L), INT32],
        o: T.Tensor[(N, S), FP32],
    ) -> None:
        _ = N, L, S, C, D, H, B
        with T.Kernel(N * SK) as bxs:
            bx = bxs % N
            pid_split = bxs // N
            seq_len = seq_lens[bx]
            np_total = T.ceildiv(seq_len, B)
            stride = T.ceildiv(np_total, SK)
            i_start = pid_split * stride
            n_iters = T.max(0, T.min(stride, np_total - i_start))

            q_smem = T.alloc_shared((H, D), FP8)
            q_s_frag = T.alloc_fragment((H,), FP32)
            T.copy(q[bx, 0, 0], q_smem)
            T.copy(weight[bx, 0], q_s_frag)

            for j in T.Pipelined(n_iters, num_stages=2):
                i = i_start + j
                page = page_table[bx, i]
                k_smem_u8 = T.alloc_shared((1, B * D), UINT8)
                T.copy(kvcache_u8[page : page + 1, 0:SCALE_OFFSET], k_smem_u8)
                k_smem = T.view(k_smem_u8, (B, D), FP8)
                k_s_smem_u8 = T.alloc_shared((1, B * 4), UINT8)
                T.copy(
                    kvcache_u8[page : page + 1, SCALE_OFFSET:BLOCK_BYTES],
                    k_s_smem_u8,
                )
                k_s_smem = T.view(k_s_smem_u8, (B,), FP32)
                k_s_frag = T.alloc_fragment((B,), FP32)
                T.copy(k_s_smem, k_s_frag)

                logits = T.alloc_fragment((B, H), FP32)
                if not clear_accum:
                    T.fill(logits, 0.0)
                T.gemm(
                    k_smem,
                    q_smem,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=clear_accum,
                )

                # post processing
                for h, j2 in T.Parallel(H, B):
                    logits[j2, h] = T.max(logits[j2, h], 0.0) * q_s_frag[h]
                logits_sum = T.alloc_fragment((B,), FP32)
                T.reduce_sum(logits, logits_sum, dim=1)
                for j2 in T.Parallel(B):
                    logits_sum[j2] *= k_s_frag[j2]
                T.copy(logits_sum, o[bx, i * B])

    return fp8_paged_mqa_logits


def tilelang_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    assert head_dim == 128, "TODO"
    assert block_size == 64, "TODO"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    logits = page_table.new_empty((batch_size, max_seq_len), dtype=torch.float32)

    NUM_CU = 256
    split_kv = split_kv = max(1, min(max_seq_len // block_size, NUM_CU // batch_size))
    kernel = fp8_paged_mqa_logits_kernel(
        head_dim=head_dim,
        num_heads=num_heads,
        block_size=block_size,
        clear_accum=clean_logits,
        split_kv=split_kv,
    )
    q_fp8 = q_fp8.view(batch_size, num_heads, head_dim)
    kvcache_u8 = kvcache_fp8.view(-1, block_size * (head_dim + 4))
    kernel(q_fp8, kvcache_u8, weight, seq_lens, page_table, logits)
    return logits