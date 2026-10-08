# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer K production for DeepSeek V4.1 kv-source layers.

In v4.1 the index key is derived from the *main* compressor's latent:
``k = k_norm(wk(latent))`` (reference model.py Indexer.forward), then RoPE'd
at the group's first-token position and MXFP4/FP8/INT8-quantized into the
paged indexer K cache. The ``wk(latent)`` GEMM runs in torch; this kernel
fuses the remaining k_norm → RoPE → quant → paged store.

Unlike the legacy (v4.0) indexer path there is no per-token pooling from a
compressor state cache: the latent already stands for a whole group, so only
group-boundary tokens ``(position + 1) % compress_ratio == 0`` produce a key.

Quant modes (``QUANT_MODE`` constexpr, fully decoupled code paths):

* ``0`` — per-token FP8 (e4m3) with a single fp32 power-of-2 scale.
* ``1`` — MXFP4 (2 E2M1 nibbles/byte + ue8m0 per 32 elements).
* ``2`` — per-token INT8: the whole 128-dim head is one quant block carrying
  a single fp32 scale ``amax / 127`` (dequant = ``code * scale``); its 4
  bytes live in the block's scale region after the value bytes — the same
  segregated per-token layout the FP8 mode uses.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    # NOTE: $1 is high nibble, $2 is low nibble
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _fp32_to_e2m1_nibble(x):
    # E2M1 round-to-nearest-even via the reference quantizer's boundaries:
    #   values    [0.00, 0.50, 1.00, 1.50, 2.00, 3.00, 4.00, 6.00]
    #   boundaries [0.25, 0.75, 1.25, 1.75, 2.50, 3.50, 5.00]
    # Bit-identical to cvt.rn.satfinite.e2m1x2.f32 on values in [-6, 6].
    ax = tl.abs(x)
    code = tl.where(ax > 0.25, 1, 0)
    code = tl.where(ax >= 0.75, 2, code)
    code = tl.where(ax > 1.25, 3, code)
    code = tl.where(ax >= 1.75, 4, code)
    code = tl.where(ax > 2.5, 5, code)
    code = tl.where(ax >= 3.5, 6, code)
    code = tl.where(ax > 5.0, 7, code)
    sign = tl.where(x < 0.0, 8, 0)
    return (code + sign).to(tl.uint8)


@triton.jit
def _fp32x2_to_fp4x2_sw(x_lo, x_hi):
    # Software MXFP4 pack: even index -> low nibble, odd index -> high nibble
    # (same convention as the PTX path above).
    return _fp32_to_e2m1_nibble(x_lo) | (_fp32_to_e2m1_nibble(x_hi) << 4)


# MXFP4 value layout: one quant block per 32 elements
# (16 packed e2m1 bytes + 1 ue8m0 scale byte).
MXFP4_BLOCK_SIZE = 32

# Quant modes for _indexer_k_norm_rope_quant_store_kernel.
_QUANT_FP8 = 0
_QUANT_FP4 = 1
_QUANT_INT8 = 2

# fp32 magic number for round-to-nearest-even integer conversion:
# adding 1.5 * 2**23 forces the fp32 adder to round to the nearest integer.
_RNE_MAGIC = 12582912.0  # 1.5 * 2**23
_RNE_MAGIC_BITS = 0x4B400000


def _fp4_asm_default() -> bool:
    """Pick the MXFP4 pack implementation.

    MACA (triton 3.6.0 / maca 3.8.x) cannot even load a kernel image that
    contains ``cvt.rn.satfinite.e2m1x2`` — the specialization fails at
    ``load_binary`` with "device kernel image is invalid". Default to the
    bit-identical software pack there; keep the PTX path everywhere else.
    ``MCOP_INDEXER_K_FP4_ASM=1|0`` overrides the detection for debugging.
    """
    if "MCOP_INDEXER_K_FP4_ASM" in os.environ:
        return os.environ["MCOP_INDEXER_K_FP4_ASM"] == "1"
    return "metax" not in torch.__version__.lower()


# Resolved once at import: the launcher sits on the critical path of every
# decode step, and per-call re-detection is measurable host overhead once
# kernel time drops to a few microseconds.
_FP4_ASM_DEFAULT = _fp4_asm_default()


def indexer_k_norm_rope_store(
    k_pre: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    compress_ratio: int,
    use_fp4_cache: bool,
    use_int8_cache: bool = False,
) -> None:
    """k_norm → RoPE → quant → paged store for indexer keys.

    Args:
        k_pre: [num_tokens, 128] bf16, the ``wk(latent)`` projection. Only
            group-boundary rows are read.
        positions: [num_tokens] int64 token positions. Positions of rows
            below ``num_tokens`` must be inside ``cos_sin_cache`` (the table
            spans the model's whole position range); rows past
            ``num_tokens`` may hold anything — they are never dereferenced.
        cos_sin_cache: [max_pos, rope_head_dim] GPT-J layout (cos half, then
            sin half), from the layer's compress-RoPE instance.
        rms_norm_weight: [128] k_norm weight.
        k_cache: uint8 paged indexer cache [num_blocks, block_size, row_bytes]
            with the values-then-scales layout: token ``s`` of block ``b``
            lives at flat offsets ``b * stride0 + (s // 1) * token_stride`` for
            its ``token_stride`` value bytes and
            ``b * stride0 + block_size * token_stride + s * scale_dim`` for its
            ``scale_dim`` scale bytes.
        kv_slot_mapping: [num_tokens] slots in the indexer cache (-1 = skip).
        compress_ratio: group size; keys are emitted at group boundaries.
        use_fp4_cache: MXFP4 (2 nibbles/byte + ue8m0 per 32) when True, else
            per-token FP8 with a single fp32 scale.
        use_int8_cache: per-token INT8 (one fp32 scale ``scale = amax / 127``
            for the whole 128-dim head, stored as 4 bytes in the scale
            region) when True. Mutually exclusive with ``use_fp4_cache``.
    """
    num_tokens = kv_slot_mapping.numel()
    assert k_pre.ndim == 2 and k_pre.shape[1] == 128
    assert k_pre.dtype == torch.bfloat16 and k_pre.stride(1) == 1
    assert num_tokens <= k_pre.shape[0] and num_tokens <= positions.numel()
    assert compress_ratio in (1, 2)
    assert cos_sin_cache.stride(-1) == 1, "cos_sin_cache rows must be contiguous"
    assert k_cache.stride(-1) == 1, "k_cache rows must be contiguous"
    assert not (use_int8_cache and use_fp4_cache), (
        "use_int8_cache and use_fp4_cache are mutually exclusive"
    )
    if num_tokens == 0:
        return

    head_dim = k_pre.shape[1]
    if use_int8_cache:
        token_stride = head_dim
        scale_dim = 4  # single fp32 scale per token
        quant_mode = _QUANT_INT8
    elif use_fp4_cache:
        token_stride = head_dim // 2
        scale_dim = head_dim // MXFP4_BLOCK_SIZE
        quant_mode = _QUANT_FP4
    else:
        token_stride = head_dim
        scale_dim = 4  # single float32 scale
        quant_mode = _QUANT_FP8

    # Value layout inside each cache block is linear row-major: the MACA
    # gather reader (cp_gather_indexer_k_quant_cache_triton) selects it for
    # every block size. The 16x16-tiled ("SHUFFLE") layout that ROCm's
    # deepgemm_fp8_paged_mqa_logits(Preshuffle=True) reader expects is not
    # provided by this build.

    # Tokens per program: keep >= 32 bytes of k_pre per thread on C600-U
    # (64 lanes/warp) once there is enough parallelism to fill the APs;
    # stay at one token per program for small batches so the grid stays wide.
    if num_tokens >= 2048:
        tp = 8
    elif num_tokens >= 256:
        tp = 4
    else:
        tp = 1

    # Keep the launch path lean: with a ~5 us kernel every host-side
    # microsecond is percent-level wall time, and triton's generated binder
    # pays for every parameter in the kernel signature on each call.
    _indexer_k_norm_rope_quant_store_kernel[((num_tokens + tp - 1) // tp,)](
        k_pre,
        k_pre.stride(0),
        positions,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        k_cache,
        kv_slot_mapping,
        num_tokens,
        k_cache.shape[1],
        HEAD_SIZE=head_dim,
        ROPE_HEAD_DIM=64,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=scale_dim,
        KV_BLOCK_STRIDE=k_cache.stride(0),
        FP8_MAX=448.0,
        QUANT_MODE=quant_mode,
        FP4_ASM=_FP4_ASM_DEFAULT,
        CS_ALIGNED=(cos_sin_cache.stride(0) % 64) == 0,
        TP=tp,
        num_warps=1,
    )


@triton.jit
def _indexer_k_norm_rope_quant_store_kernel(
    k_pre_ptr,
    k_pre_stride,
    positions_ptr,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    num_tokens,
    kv_cache_block_size,
    HEAD_SIZE: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_MODE: tl.constexpr,
    FP4_ASM: tl.constexpr,
    CS_ALIGNED: tl.constexpr,
    TP: tl.constexpr,
):
    # ── Token block: [TP] rows, masked instead of early returns ───────
    pid = tl.program_id(0)
    tok = pid * TP + tl.arange(0, TP)
    tok_valid = tok < num_tokens
    slots = tl.load(kv_slot_mapping_ptr + tok, mask=tok_valid, other=-1)
    positions = tl.load(positions_ptr + tok, mask=tok_valid, other=0)
    active = tok_valid & (slots >= 0)
    if COMPRESS_RATIO != 1:
        # Only the last token of a group publishes that group's index key.
        active = active & (((positions + 1) % COMPRESS_RATIO) == 0)

    NUM_PAIRS: tl.constexpr = HEAD_SIZE // 2
    NOPE_PAIRS: tl.constexpr = (HEAD_SIZE - ROPE_HEAD_DIM) // 2
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    tl.static_assert(NOPE_PAIRS == HALF_ROPE)

    # ── cos/sin table: two 32-wide affine loads, hoisted above k_pre ──
    # A latent stands for the first token of its group, so group j takes
    # position j * compress_ratio. Loads are unmasked: every in-range
    # token's position must be covered by the table (caller contract —
    # rows beyond num_tokens read position 0 via the masked load above).
    # Masking these loads costs a select per element and measured ~15%
    # slower on C600-U; skipped tokens' lanes produce garbage that is
    # filtered by the active mask on every store.
    compressed_pos = (positions // COMPRESS_RATIO) * COMPRESS_RATIO
    off = compressed_pos * cos_sin_stride
    if CS_ALIGNED:
        off = tl.multiple_of(off, 64)
    block = tl.arange(0, HEAD_SIZE)
    c_raw = tl.load(
        cos_sin_cache_ptr + off[:, None] + tl.arange(0, HALF_ROPE)[None, :]
    ).to(tl.float32)
    s_raw = tl.load(
        cos_sin_cache_ptr
        + off[:, None]
        + (HALF_ROPE + tl.arange(0, HALF_ROPE))[None, :]
    ).to(tl.float32)

    # ── k_norm (fp32 throughout, bf16 roundtrip like the reference) ────
    rms_w = tl.load(rms_norm_weight_ptr + block).to(tl.float32)
    k = tl.load(
        k_pre_ptr + tok[:, None] * k_pre_stride + block[None, :],
        mask=active[:, None],
        other=0.0,
    ).to(tl.float32)
    variance = tl.sum(k * k, axis=1) / HEAD_SIZE
    k = (k * tl.rsqrt(variance + rms_norm_eps)[:, None] * rms_w[None, :]).to(
        tl.bfloat16
    )
    k = k.to(tl.float32)

    # ── Register-based GPT-J forward RoPE in fp32 ─────────────────────
    # Pairs [0, NOPE_PAIRS) pass through (cos = 1, sin = 0); the rest rotate
    # with the table row. The constants are joined with the loaded halves so
    # the whole transform stays gather- and select-free.
    even, odd = tl.split(tl.reshape(k, (TP, NUM_PAIRS, 2)))  # each [TP, 64]
    ones_v = tl.full((TP, HALF_ROPE), 1.0, tl.float32)
    zeros_v = tl.zeros((TP, HALF_ROPE), tl.float32)
    cos_all = tl.reshape(
        tl.permute(tl.join(ones_v, c_raw), (0, 2, 1)), (TP, NUM_PAIRS)
    )
    sin_all = tl.reshape(
        tl.permute(tl.join(zeros_v, s_raw), (0, 2, 1)), (TP, NUM_PAIRS)
    )

    new_even = even * cos_all - odd * sin_all
    new_odd = odd * cos_all + even * sin_all

    # bf16 roundtrip for parity with the reference / Q-side kernel numerics.
    new_even = new_even.to(tl.bfloat16).to(tl.float32)
    new_odd = new_odd.to(tl.bfloat16).to(tl.float32)
    result = tl.interleave(new_even, new_odd)  # [TP, HEAD_SIZE] fp32

    # ── Paged cache pointers (segregated: values first, then scales) ──
    kv_block_idx = (slots // kv_cache_block_size).to(tl.int64)
    kv_pos_in_block = slots % kv_cache_block_size
    cache_block_ptr = k_cache_ptr + kv_block_idx * KV_BLOCK_STRIDE
    val_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    if QUANT_MODE == 1:
        # ── MXFP4: 32-element blocks = 16 consecutive even/odd pairs ──
        N_QUANT_BLOCKS: tl.constexpr = HEAD_SIZE // 32
        HALF_BLOCK: tl.constexpr = 16
        even_2d = tl.reshape(new_even, (TP, N_QUANT_BLOCKS, HALF_BLOCK))
        odd_2d = tl.reshape(new_odd, (TP, N_QUANT_BLOCKS, HALF_BLOCK))

        amax = tl.maximum(
            tl.max(tl.abs(even_2d), axis=2),
            tl.max(tl.abs(odd_2d), axis=2),
        )
        amax = tl.maximum(amax, 6.0 * (2**-126))
        # ue8m0 block scale: 2^ceil(log2(amax / 6.0)), stored (exp + 127).
        log2_ratio = tl.ceil(tl.log2(amax * (1.0 / 6.0)))
        log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
        inv_scale = tl.exp2(-log2_ratio)
        ue8m0 = (log2_ratio + 127.0).to(tl.uint8)  # [TP, N_QUANT_BLOCKS]

        inv_scale_col = inv_scale[:, :, None]
        if FP4_ASM:
            packed = _fp32x2_to_fp4x2(
                even_2d * inv_scale_col, odd_2d * inv_scale_col
            )  # (TP, N_BLOCKS, HALF_BLOCK) uint8
        else:
            # Bit-identical software pack for assemblers without
            # cvt.rn.satfinite.e2m1x2 (MACA).
            packed = _fp32x2_to_fp4x2_sw(
                even_2d * inv_scale_col, odd_2d * inv_scale_col
            )
        packed_flat = tl.reshape(packed, (TP, TOKEN_STRIDE))

        tl.store(
            val_ptr[:, None] + tl.arange(0, TOKEN_STRIDE)[None, :],
            packed_flat,
            mask=active[:, None],
        )
        tl.store(
            scale_ptr[:, None] + tl.arange(0, SCALE_DIM)[None, :],
            ue8m0,
            mask=active[:, None],
        )
    elif QUANT_MODE == 0:
        # ── Per-token FP8 (single 128-wide block), one fp32 scale ────
        INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX
        absmax = tl.maximum(tl.max(tl.abs(result), axis=1), 1e-4)
        exponent = tl.ceil(tl.log2(absmax * INV_FP8_MAX))
        inv_scale = tl.exp2(-exponent)
        # inv_scale is an exact power of two, so the product is exact and
        # |result * inv_scale| <= FP8_MAX by construction — no clamp needed.
        x_uint8 = (result * inv_scale[:, None]).to(tl.float8e4nv).to(
            tl.uint8, bitcast=True
        )
        scale_bits = tl.exp2(exponent).to(tl.uint32, bitcast=True)
        scale_bytes = (
            (scale_bits[:, None] >> (8 * tl.arange(0, 4)[None, :])) & 0xFF
        ).to(tl.uint8)
        tl.store(
            val_ptr[:, None] + block[None, :], x_uint8,
            mask=active[:, None],
        )
        tl.store(
            scale_ptr[:, None] + tl.arange(0, SCALE_DIM)[None, :],
            scale_bytes,
            mask=active[:, None],
        )
    else:
        # ── Per-token symmetric INT8: one 128-wide block, one fp32 scale ─
        amax = tl.maximum(tl.max(tl.abs(result), axis=1), 1e-4)  # [TP]
        inv_scale = 127.0 / amax
        qf = result * inv_scale[:, None]
        # Round-to-nearest-even integer via the fp32 magic-number add; the
        # bitcast/subtract pair recovers the two's-complement value for
        # |qf| < 2**22, which always holds here.
        qi = (qf + 12582912.0).to(tl.uint32, bitcast=True) - 0x4B400000
        q = qi.to(tl.int8)
        codes = q.to(tl.uint8, bitcast=True)
        scale_bits = (amax / 127.0).to(tl.uint32, bitcast=True)
        scale_bytes = (
            (scale_bits[:, None] >> (8 * tl.arange(0, 4)[None, :])) & 0xFF
        ).to(tl.uint8)

        tl.store(
            val_ptr[:, None] + block[None, :], codes,
            mask=active[:, None],
        )
        tl.store(
            scale_ptr[:, None] + tl.arange(0, SCALE_DIM)[None, :],
            scale_bytes,
            mask=active[:, None],
        )
