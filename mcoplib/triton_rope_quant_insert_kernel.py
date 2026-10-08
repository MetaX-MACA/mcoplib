import torch
import triton
import triton.language as tl

_ON_GFX950 = False

# QUANT_TYPE constexpr values shared by both kernels.
_QT_FP8 = 0
_QT_INT8 = 1
_QT_BF16 = 2


@triton.jit
def _rope_quant_insert_kernel(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    NUM_TOKENS,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    SANITIZE_CACHE_NANS: tl.constexpr,
    QUANT_TYPE: tl.constexpr = 0,
    BT: tl.constexpr = 1,
):
    """Paged writer, 2D-tiled: one program handles ``BT`` tokens as a ``[BT, 512]``
    tile so the scalar slot/position/cos-sin loads amortize across the tile and the
    latent load/cache store vectorize per row.

    ``QUANT_TYPE`` selects the layout (both segregated per page: values region then
    scales region):

    * fp8 (QUANT_TYPE=0) -- 584 bytes/token: 576 value bytes
      ([448 NoPE fp8(e4m3) | 128 RoPE bf16]) then 8 scale bytes (7 UE8M0 per-group
      exponents over the NoPE dims + 1 pad). The RoPE payload is stored as bf16.
    * int8 (QUANT_TYPE=1) -- 576 bytes/token: 512 value bytes (the FULL rotated row
      quantized to int8: 448 NoPE identity dims | 64 GPT-J-rotated RoPE dims) then
      64 scale bytes (16 fp32 per-group scales, one per 32 dims). RoPE is rotated
      first, then quantized alongside the NoPE dims (groups 14/15).
    """
    pid = tl.program_id(0)
    rows = pid * BT + tl.arange(0, BT)               # token ids [BT]
    m = rows < NUM_TOKENS
    d = tl.arange(0, 512)
    j = tl.arange(0, 32)
    s8 = tl.arange(0, 8)

    slot = tl.load(cache_slots + rows, m, other=-1)
    position = tl.load(positions + rows, m, other=0)
    valid = m & (slot >= 0) & ((position + 1) % COMPRESS_RATIO == 0)
    vmask = valid[:, None]

    base = rows[:, None].to(tl.int64) * 512
    normed = tl.load(latent + base + d[None, :], vmask).to(tl.float32)   # [BT,512]
    page = (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
    slot_in = (slot % CACHE_BLOCK)[:, None]

    if QUANT_TYPE == 1:

        values = page[:, None] + slot_in * 512                           # [BT,1]
        scales = page[:, None] + CACHE_BLOCK * 512 + slot_in * 64
        # quantize the whole row as 16 groups; only groups 0..13 (dims 0..447)
        # are stored/kept -- groups 14/15 here are provisional (unrotated) and are
        g16 = tl.reshape(normed, (BT, 16, 32))
        amaxN = tl.maximum(tl.max(tl.abs(g16), 2), 1e-4)                 # [BT,16]
        scaleN = amaxN * (1.0 / 127.0)
        qN = tl.extra.cuda.libdevice.round(g16 * tl.reshape(127.0 / amaxN, (BT, 16, 1)))
        qN = tl.clamp(qN, -127.0, 127.0).to(tl.int8)
        qN = tl.reshape(qN, (BT, 512)).to(tl.uint8, bitcast=True)
        tl.store(cache + values + d[None, :], qN, vmask & (d[None, :] < 448))
        # RoPE tail: load the 64 dims, GPT-J rotate (32 interleave pairs), quantize
        # as 2 groups of 32.
        tail = tl.load(latent + base + 448 + tl.arange(0, 64)[None, :], vmask).to(tl.float32)
        te, to = tl.split(tl.reshape(tail, (BT, 32, 2)))
        cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO)[:, None].to(tl.int64) * COS_STRIDE
        c = tl.load(cs + j[None, :], vmask, other=1.0).to(tl.float32)     # [BT,32]
        sn = tl.load(cs + 32 + j[None, :], vmask, other=0.0).to(tl.float32)
        rot = tl.interleave(te * c - to * sn, to * c + te * sn)           # [BT,64]
        if SANITIZE_CACHE_NANS:
            rot = tl.where(rot == rot, rot, 0.0)
        gR = tl.reshape(rot, (BT, 2, 32))
        amaxR = tl.maximum(tl.max(tl.abs(gR), 2), 1e-4)                  # [BT,2]
        scaleR = amaxR * (1.0 / 127.0)
        qR = tl.extra.cuda.libdevice.round(gR * tl.reshape(127.0 / amaxR, (BT, 2, 1)))
        qR = tl.clamp(qR, -127.0, 127.0).to(tl.int8)
        qR = tl.reshape(qR, (BT, 64)).to(tl.uint8, bitcast=True)
        tl.store(cache + values + 448 + tl.arange(0, 64)[None, :], qR, vmask)
        # 16 fp32 scales: groups 0..13 from the NoPE quant, 14..15 from the rotated
        # tail quant, contiguous in the scale region.
        sc_f32 = (cache + scales).to(tl.pointer_type(tl.float32))
        g16i = tl.arange(0, 16)[None, :]
        tl.store(sc_f32 + g16i, scaleN, vmask & (g16i < 14))
        tl.store(sc_f32 + 14 + tl.arange(0, 2)[None, :], scaleR, vmask)
        return

    # ---- fp8 (e4m3), 584B layout ---------------------------------------------
    normed = normed
    # GPT-J RoPE on the last 64 dims (32 interleave pairs); NoPE pairs are identity.
    tail = tl.load(latent + base + 448 + tl.arange(0, 64)[None, :], vmask).to(tl.float32)
    even, odd = tl.split(tl.reshape(tail, (BT, 32, 2)))
    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO)[:, None].to(tl.int64) * COS_STRIDE
    c = tl.load(cs + j[None, :], vmask, other=1.0).to(tl.float32)        # [BT,32]
    sn = tl.load(cs + 32 + j[None, :], vmask, other=0.0).to(tl.float32)
    rotated = tl.interleave(even * c - odd * sn, odd * c + even * sn)     # [BT,64]
    if SANITIZE_CACHE_NANS:
        rotated = tl.where(rotated == rotated, rotated, 0.0)

    # FP8 (e4m3): per-group(8x64) UE8M0 power-of-two scale over the NoPE dims;
    # RoPE payload stored as bf16.
    values = page[:, None] + slot_in * 576                               # [BT,1]
    scales = page[:, None] + CACHE_BLOCK * 576 + slot_in * 8
    quant = tl.reshape(normed, (BT, 8, 64))
    amax = tl.maximum(tl.max(tl.abs(quant), 2), 1e-4)                    # [BT,8]
    exponent = tl.ceil(tl.log2(amax * (1.0 / 448.0)))
    scaled = quant * tl.reshape(tl.exp2(-exponent), (BT, 8, 1))
    fp8 = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv)
    packed = tl.reshape(fp8, (BT, 512)).to(tl.uint8, bitcast=True)       # [BT,512]
    tl.store(cache + values + d[None, :], packed, vmask & (d[None, :] < 448))
    max_encoded: tl.constexpr = 254.0 if SANITIZE_CACHE_NANS else 255.0
    encoded = tl.minimum(tl.maximum(exponent + 127.0, 0.0), max_encoded)  # [BT,8]
    # Groups 0..6 cover the 448 stored NoPE dims -> 7 exponent bytes; the 8th
    # (padding) scale byte is zeroed.
    tl.store(cache + scales + s8[None, :], encoded.to(tl.uint8),
             vmask & (s8[None, :] < 7))
    tl.store(cache + scales + 7, tl.zeros((BT, 1), tl.uint8), vmask)
    # RoPE payload as bf16 in value bytes [448:576].
    rope_dst = (cache + values + 448).to(tl.pointer_type(tl.bfloat16))
    tl.store(rope_dst + tl.arange(0, 64)[None, :], rotated.to(tl.bfloat16), vmask)


@triton.jit
def _rope_plain_insert_kernel(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    fp8_scale,
    quant_scale,
    NUM_TOKENS,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
    BT: tl.constexpr = 1,
):
    """Plain ``[448 NoPE | 64 RoPE]`` row writer, 2D-tiled over ``BT`` tokens.

    The NoPE dims are passed through in their native precision (bf16 identity, or
    fp8 per-tensor / int8 per-token quant); RoPE is applied to the last 64 dims.
    """
    pid = tl.program_id(0)
    rows = pid * BT + tl.arange(0, BT)
    m = rows < NUM_TOKENS
    d = tl.arange(0, 512)
    j = tl.arange(0, 32)

    slot = tl.load(cache_slots + rows, m, other=-1)
    position = tl.load(positions + rows, m, other=0)
    valid = m & (slot >= 0) & ((position + 1) % COMPRESS_RATIO == 0)
    vmask = valid[:, None]

    base = rows[:, None].to(tl.int64) * 512
    cs = cos_sin + (position // COMPRESS_RATIO * COMPRESS_RATIO)[:, None].to(tl.int64) * COS_STRIDE

    dst = (
        (slot // CACHE_BLOCK).to(tl.int64)[:, None] * CACHE_STRIDE
        + (slot % CACHE_BLOCK)[:, None] * ROW_STRIDE
    )
    if QUANT_TYPE == 1:
        # int8 per-token symmetric round-to-nearest needs the whole 512 row under a
        # single amax, so build the full interleaved row (NoPE pairs are identity).
        normed = tl.load(latent + base + d[None, :], vmask).to(tl.float32)  # [BT,512]
        pair = tl.arange(0, 256) - 224
        even, odd = tl.split(tl.reshape(normed, (BT, 256, 2)))
        c = tl.load(cs + tl.maximum(pair, 0)[None, :], vmask & (pair >= 0)[None, :], other=1.0).to(tl.float32)
        sn = tl.load(cs + 32 + tl.maximum(pair, 0)[None, :], vmask & (pair >= 0)[None, :], other=0.0).to(tl.float32)
        row = tl.interleave(even * c - odd * sn, odd * c + even * sn)        # [BT,512]
        amax = tl.maximum(tl.max(tl.abs(row), 1), 1e-4)[:, None]            # [BT,1]
        scale = amax * (1.0 / 127.0)
        q = tl.extra.cuda.libdevice.round(row * (127.0 / amax))
        q = tl.clamp(q, -127.0, 127.0).to(tl.int8)
        tl.store(cache + dst + d[None, :], q, vmask)
        tl.store(quant_scale + rows, tl.reshape(scale, (BT,)), valid)       # per-token scale
        return

    # bf16 identity / fp8 per-tensor: NoPE passes through, RoPE on last 64 dims only.
    normed = tl.load(latent + base + d[None, :], vmask).to(tl.float32)      # [BT,512]
    tail = tl.load(latent + base + 448 + tl.arange(0, 64)[None, :], vmask).to(tl.float32)
    even, odd = tl.split(tl.reshape(tail, (BT, 32, 2)))
    c = tl.load(cs + j[None, :], vmask, other=1.0).to(tl.float32)
    sn = tl.load(cs + 32 + j[None, :], vmask, other=0.0).to(tl.float32)
    rope_tail = tl.interleave(even * c - odd * sn, odd * c + even * sn)     # [BT,64]

    if QUANT_TYPE == 0:            # fp8 per-tensor
        inv = 1.0 / tl.load(fp8_scale)
        nope = tl.clamp(normed * inv, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(cache + dst + d[None, :], nope, vmask & (d[None, :] < 448))
        rope_q = tl.clamp(rope_tail * inv, -448.0, 448.0).to(tl.float8e4nv)
        tl.store(cache + dst + 448 + tl.arange(0, 64)[None, :], rope_q, vmask)
    else:                          # bf16 identity
        tl.store(cache + dst + d[None, :], normed.to(tl.bfloat16),
                 vmask & (d[None, :] < 448))
        tl.store(cache + dst + 448 + tl.arange(0, 64)[None, :],
                 rope_tail.to(tl.bfloat16), vmask)


def rope_quant_insert(
    latent: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
    fp8_scale: torch.Tensor | None = None,
    quant_scale: torch.Tensor | None = None,
) -> None:
    """Apply GPT-J RoPE and publish a latent to the compressed KV cache.

    The BF16 latent supplies both NoPE quantization and RoPE input. It is read
    only for valid slots at group boundaries. The cache dtype selects the
    layout:

    * ``uint8`` (last dim 584) : the paged fp8_ds_mla layout -- 576 value bytes
      ([448 NoPE fp8(e4m3) | 128 RoPE bf16]) and 8 segregated scale bytes per token
      (7 UE8M0 per-group(8x64) exponents + 1 pad).
    * ``int8`` (last dim 576) : the paged int8 layout -- 512 value bytes (the full
      GPT-J-rotated row quantized to int8: 448 NoPE identity dims | 64 rotated RoPE
      dims) and 64 segregated scale bytes per token (16 fp32 per-group scales, one
      per 32 dims; symmetric round-to-nearest).
    * ``bfloat16`` / ``float8_e4m3fn`` / ``int8`` (last dim 512) : the plain
      ``[448 NoPE | 64 RoPE]`` rows read by FlashInfer. ``float8_e4m3fn`` is
      scaled by the per-tensor ``fp8_scale``; the 512-wide ``int8`` variant uses
      per-token symmetric round-to-nearest and returns its per-token scale in
      ``quant_scale`` (shape ``[num_tokens]``).

    Both kernels are 2D-tiled: a program processes ``BT`` tokens as a
    ``[BT, 512]`` tile, amortizing the scalar slot/position/cos-sin loads and
    vectorizing the latent load and cache store. ``num_warps`` is fixed at 1 --
    for this pure-streaming pattern extra warps only add store-unit contention
    (measured monotonic slowdown), so occupancy comes from the grid, not warps.
    """
    assert compress_ratio in (1, 2)
    assert latent.shape[1] == 512 and latent.dtype == torch.bfloat16
    assert latent.is_contiguous()
    num_tokens = slot_mapping.numel()
    assert num_tokens <= min(latent.shape[0], positions.numel())
    if num_tokens == 0:
        return

    # 2D tile width (tokens per program). A wider tile lets Triton coalesce the
    # [BT,512] load/store and amortize the per-token scalar loads, but every extra
    # row costs 512 live fp32 registers, so the quant math sets the ceiling: the
    # per-token int8 reduction + round spills past BT=4-8 (C600U has 255 regs/
    # thread), while the identity/per-tensor paths tolerate BT=8. ``is_int8`` picks
    # the register-pressure-limited schedule measured for each path. num_warps=1 is
    # optimal everywhere (extra warps only add store-unit contention).
    is_paged = kv_cache.dtype == torch.uint8 or (
        kv_cache.dtype == torch.int8 and kv_cache.shape[-1] in (576, 584)
    )
    is_int8 = kv_cache.dtype == torch.int8

    def _bt(n, int8, paged):
        if n < 16:
            return 1
        if int8:
            # per-token full-row reduction + libdevice.round dominate; small tiles
            # keep the reduction cheap and avoid register spilling.
            if paged:
                # tail-only rotate + two-store quant: register pressure peaks the
                # [BT,512] load/store at BT=4 for large T (measured 443 GB/s), then
                # spills past BT=4 (BT=8=343, BT=16=204). Small T stays narrow.
                if n >= 2048:
                    return 4
                if n >= 256:
                    return 2
                return 1
            return 4 if n >= 256 else 2
        # bf16 identity / fp8 per-group: coalesced streaming, tolerate wider tiles.
        if n >= 2048:
            return 8
        if n >= 256:
            return 4
        return 2
    bt = _bt(num_tokens, is_int8, is_paged)
    grid = ((num_tokens + bt - 1) // bt,)

    # ---- paged layouts, segregated value+scale region ----
    # int8 is shared with the plain path, so disambiguate paged vs plain by the last
    # dim: fp8 paged -> 584, int8 paged -> 576, plain row -> 512.
    if is_paged:
        quant_type = _QT_FP8 if kv_cache.dtype == torch.uint8 else _QT_INT8
        assert kv_cache.shape[-1] == (576 if quant_type == _QT_INT8 else 584)
        _rope_quant_insert_kernel[grid](
            latent,
            positions,
            cos_sin_cache,
            kv_cache,
            slot_mapping,
            num_tokens,
            COS_STRIDE=cos_sin_cache.stride(0),
            CACHE_STRIDE=kv_cache.stride(0),
            CACHE_BLOCK=kv_cache.shape[1],
            COMPRESS_RATIO=compress_ratio,
            SANITIZE_CACHE_NANS=_ON_GFX950,
            QUANT_TYPE=quant_type,
            BT=bt,
            num_warps=2,
        )
        return

    # ---- plain rows (bf16 / fp8 / int8), last dim 512 ----
    assert kv_cache.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.int8)
    assert kv_cache.shape[-1] == 512 and kv_cache.stride(-1) == 1
    if kv_cache.dtype == torch.float8_e4m3fn:
        quant_type = _QT_FP8
        assert fp8_scale is not None and fp8_scale.numel() == 1
        assert fp8_scale.dtype == torch.float32
    elif kv_cache.dtype == torch.int8:
        quant_type = _QT_INT8
        assert quant_scale is not None and quant_scale.numel() == num_tokens
        assert quant_scale.dtype == torch.float32
    else:
        quant_type = _QT_BF16
    _rope_plain_insert_kernel[grid](
        latent,
        positions,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        fp8_scale if quant_type == _QT_FP8 else None,
        quant_scale if quant_type == _QT_INT8 else None,
        num_tokens,
        COS_STRIDE=cos_sin_cache.stride(0),
        CACHE_STRIDE=kv_cache.stride(0),
        ROW_STRIDE=kv_cache.stride(1),
        CACHE_BLOCK=kv_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        QUANT_TYPE=quant_type,
        BT=bt,
        num_warps=1,
    )
