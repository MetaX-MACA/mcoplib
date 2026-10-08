from typing import Optional

import torch
import triton
import triton.language as tl

fp8_dtype = torch.float8_e4m3fn

# v4 KV cache layout (see dsv4.index_buf_accessor._set_k_and_s_triton_kernel):
#   per-token: 448 fp8 nope + 64 bf16 rope (= 576 contiguous bytes) +
#              7 ue8m0 scales padded to 8 bytes.
#   per-page:  [token 0..P-1 nope+rope (P*576 bytes)] [token 0..P-1 scale (P*8 bytes)]
#              padded up to a multiple of 576.
DIM_NOPE = 448
DIM_ROPE = 64
TILE_SIZE = 64  # one nope scale tile = 64 fp8 values
NUM_SCALE_TILES = DIM_NOPE // TILE_SIZE  # 7
NOPE_ROPE_BYTES = DIM_NOPE + DIM_ROPE * 2  # 576
PADDED_SCALE_PER_TOKEN = NUM_SCALE_TILES + 1  # 8

# Below this token count the 2D tile-split kernel wins (8x more programs fill
# the machine when a 1D token-block grid cannot; measured +10-16% at 1024,
# break-even at 2048, loses above as per-program setup starts to dominate).
SMALL_SPLIT_MAX_TOKENS = 1536


def _nt_bucket(num_tokens: int) -> int:
    """num_tokens size class used as the autotune key: nearest-1024 rounding
    (capped), so every token count is served by a config tuned on a size
    within ~1k tokens of it (DSV4 serving typically sees ~3500 and ~8024)
    without re-tuning for every exact value."""
    return min((num_tokens + 512) // 1024, 64)


def dequantize_k_cache_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize the DeepSeek v4 paged KV cache for a list of token IDs.

    Args:
        quant_k_cache: (num_pages, bytes_per_page_padded) uint8.
        page_table_1_flattened: (num_tokens,) int — token IDs into the cache.
        page_size: number of tokens per page.
        out: optional (num_tokens, 1, DIM_NOPE + DIM_ROPE) bf16 destination.
            May be a slice of a larger workspace; the kernel uses out.stride(0)
            so contiguous-along-dim-0 slices work.

    Returns:
        (num_tokens, 1, DIM_NOPE + DIM_ROPE) bfloat16.
    """
    assert quant_k_cache.is_contiguous()
    assert page_table_1_flattened.dtype in (torch.int32, torch.int64)

    # The buffer's dtype is whatever the pool exposes (often bf16); the
    # underlying storage is uint8. Reinterpret to byte-space first.
    quant_k_cache_u8 = quant_k_cache.view(torch.uint8)
    num_tokens = page_table_1_flattened.shape[0]
    bytes_per_page = quant_k_cache_u8.shape[-1]
    s_offset_bytes = page_size * NOPE_ROPE_BYTES

    # Three typed views over the same underlying bytes.
    buf_fp8 = quant_k_cache_u8.view(fp8_dtype).reshape(-1)
    buf_bf16 = quant_k_cache_u8.view(torch.bfloat16).reshape(-1)
    buf_uint8 = quant_k_cache_u8.reshape(-1)

    if out is None:
        out = torch.empty(
            (num_tokens, 1, DIM_NOPE + DIM_ROPE),
            dtype=torch.bfloat16,
            device=quant_k_cache.device,
        )
    else:
        assert out.shape == (num_tokens, 1, DIM_NOPE + DIM_ROPE)
        assert out.dtype == torch.bfloat16

    if num_tokens > 0:
        if num_tokens < SMALL_SPLIT_MAX_TOKENS:
            grid = lambda meta: (
                triton.cdiv(num_tokens, meta["T_TILE"]),
                PADDED_SCALE_PER_TOKEN,  # 8 columns: nope tiles 0..6 + rope
            )
            kernel = _dequantize_k_cache_paged_kernel_split
        else:
            grid = lambda meta: (triton.cdiv(num_tokens, meta["T_TILE"]),)
            kernel = _dequantize_k_cache_paged_kernel
        kernel[grid](
            out,
            buf_fp8,
            buf_bf16,
            buf_uint8,
            page_table_1_flattened,
            num_tokens,
            out.stride(0),
            _nt_bucket(num_tokens),
            BYTES_PER_PAGE=bytes_per_page,
            PAGE_SIZE=page_size,
            DIM_NOPE=DIM_NOPE,
            DIM_ROPE=DIM_ROPE,
            TILE_SIZE=TILE_SIZE,
            NUM_SCALE_TILES=NUM_SCALE_TILES,
            NOPE_ROPE_BYTES=NOPE_ROPE_BYTES,
            PADDED_SCALE_PER_TOKEN=PADDED_SCALE_PER_TOKEN,
            S_OFFSET_BYTES=s_offset_bytes,
        )
    return out


@triton.autotune(
    configs=[
        triton.Config({"T_TILE": 1}, num_warps=1),
        triton.Config({"T_TILE": 2}, num_warps=1),
        triton.Config({"T_TILE": 4}, num_warps=1),
        triton.Config({"T_TILE": 8}, num_warps=1),
        triton.Config({"T_TILE": 2}, num_warps=2),
        triton.Config({"T_TILE": 4}, num_warps=2),
        triton.Config({"T_TILE": 8}, num_warps=2),
        triton.Config({"T_TILE": 16}, num_warps=2),
        triton.Config({"T_TILE": 32}, num_warps=2),
        triton.Config({"T_TILE": 8}, num_warps=4),
        triton.Config({"T_TILE": 16}, num_warps=4),
        triton.Config({"T_TILE": 32}, num_warps=4),
        triton.Config({"T_TILE": 16}, num_warps=8),
        triton.Config({"T_TILE": 32}, num_warps=8),
        triton.Config({"T_TILE": 64}, num_warps=8),
    ],
    key=["NT_BUCKET"],
)
@triton.jit
def _dequantize_k_cache_paged_kernel(
    output_ptr,
    buf_fp8_ptr,
    buf_bf16_ptr,
    buf_uint8_ptr,
    page_table_ptr,
    num_tokens,
    output_stride_0,
    NT_BUCKET,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
    T_TILE: tl.constexpr,
):
    # One program dequantizes T_TILE consecutive tokens end to end.  The 7
    # ue8m0 scales arrive via one contiguous 8-byte load per token, the 448
    # fp8 nope values via a single [T_TILE, 8, 64] tile load (tile 7 masked
    # out; token data is 64B-aligned so every lane vectorizes), and the 64
    # bf16 rope values via one vector load; both output halves use fully
    # coalesced stores.  exp2 runs once per scale slot instead of once per
    # output lane, keeping per-thread traffic >= 32B for every config.
    pid = tl.program_id(0)
    token_ids = pid * T_TILE + tl.arange(0, T_TILE)
    tok_mask = token_ids < num_tokens
    locs = tl.load(page_table_ptr + token_ids, mask=tok_mask, other=0).to(tl.int64)
    page_byte_base = (locs // PAGE_SIZE) * BYTES_PER_PAGE
    data_base = page_byte_base + (locs % PAGE_SIZE) * NOPE_ROPE_BYTES
    scale_base = (
        page_byte_base + S_OFFSET_BYTES + (locs % PAGE_SIZE) * PADDED_SCALE_PER_TOKEN
    )
    out_row = token_ids.to(tl.int64) * output_stride_0

    # scales: one contiguous 8-byte load per token (7 ue8m0 + 1 pad byte).
    # The pad lane may hold garbage (its exp2 can even hit inf), but tile 7
    # is masked out everywhere below, so it is never stored.
    s_offs = tl.arange(0, PADDED_SCALE_PER_TOKEN)
    scale_u8 = tl.load(
        buf_uint8_ptr + scale_base[:, None] + s_offs[None, :],
        mask=tok_mask[:, None],
        other=0,
    )
    scale_pow2 = tl.exp2(scale_u8.to(tl.float32) - 127.0)

    # nope tiles: [T_TILE, 8, 64] fp8 -> fp32 -> * 2^(scale-127) -> bf16
    tiles = tl.arange(0, PADDED_SCALE_PER_TOKEN)
    cols = tl.arange(0, TILE_SIZE)
    tile_ok = (tiles < NUM_SCALE_TILES)[None, :, None]
    fp8_vals = tl.load(
        buf_fp8_ptr
        + data_base[:, None, None]
        + tiles[None, :, None] * TILE_SIZE
        + cols[None, None, :],
        mask=tok_mask[:, None, None] & tile_ok,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output_ptr
        + out_row[:, None, None]
        + tiles[None, :, None] * TILE_SIZE
        + cols[None, None, :],
        (fp8_vals * scale_pow2[:, :, None]).to(output_ptr.dtype.element_ty),
        mask=tok_mask[:, None, None] & tile_ok,
    )

    # rope tail: 64 bf16 copied verbatim
    rope_offs = tl.arange(0, DIM_ROPE)
    rope_vals = tl.load(
        buf_bf16_ptr + ((data_base + DIM_NOPE) // 2)[:, None] + rope_offs[None, :],
        mask=tok_mask[:, None],
        other=0.0,
    )
    tl.store(
        output_ptr + out_row[:, None] + DIM_NOPE + rope_offs[None, :],
        rope_vals,
        mask=tok_mask[:, None],
    )


@triton.autotune(
    configs=[
        triton.Config({"T_TILE": 1}, num_warps=1),
        triton.Config({"T_TILE": 2}, num_warps=1),
        triton.Config({"T_TILE": 4}, num_warps=1),
        triton.Config({"T_TILE": 8}, num_warps=1),
        triton.Config({"T_TILE": 16}, num_warps=1),
        triton.Config({"T_TILE": 32}, num_warps=1),
        triton.Config({"T_TILE": 64}, num_warps=1),
        triton.Config({"T_TILE": 4}, num_warps=2),
        triton.Config({"T_TILE": 8}, num_warps=2),
        triton.Config({"T_TILE": 16}, num_warps=2),
        triton.Config({"T_TILE": 32}, num_warps=2),
        triton.Config({"T_TILE": 16}, num_warps=4),
        triton.Config({"T_TILE": 32}, num_warps=4),
        triton.Config({"T_TILE": 64}, num_warps=4),
    ],
    key=["NT_BUCKET"],
)
@triton.jit
def _dequantize_k_cache_paged_kernel_split(
    output_ptr,
    buf_fp8_ptr,
    buf_bf16_ptr,
    buf_uint8_ptr,
    page_table_ptr,
    num_tokens,
    output_stride_0,
    NT_BUCKET,
    BYTES_PER_PAGE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
    T_TILE: tl.constexpr,
):
    # Small-batch variant on a 2D grid: axis 0 walks token blocks, axis 1 =
    # 0..6 selects one 64-wide nope tile and 7 handles the rope tail.  Every
    # load and store is a dense [T_TILE, 64] tile, and the 8x program count
    # fills thread slots a token-block-only grid would leave idle when
    # num_tokens is small.  Each nope program reads only its own ue8m0 scale
    # byte; the scale rows stay hot in L2 across the eight column programs.
    pid = tl.program_id(0)
    part = tl.program_id(1)
    token_ids = pid * T_TILE + tl.arange(0, T_TILE)
    tok_mask = token_ids < num_tokens
    locs = tl.load(page_table_ptr + token_ids, mask=tok_mask, other=0).to(tl.int64)
    page_byte_base = (locs // PAGE_SIZE) * BYTES_PER_PAGE
    in_page = locs % PAGE_SIZE
    data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    out_row = token_ids.to(tl.int64) * output_stride_0
    cols = tl.arange(0, TILE_SIZE)

    if part < NUM_SCALE_TILES:
        # one 64-wide nope tile: fp8 -> fp32 -> * 2^(scale-127) -> bf16
        scale_base = (
            page_byte_base + S_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN
        )
        scale_u8 = tl.load(buf_uint8_ptr + scale_base + part, mask=tok_mask, other=0)
        scale_pow2 = tl.exp2(scale_u8.to(tl.float32) - 127.0)
        fp8_vals = tl.load(
            buf_fp8_ptr + data_base[:, None] + part * TILE_SIZE + cols[None, :],
            mask=tok_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        tl.store(
            output_ptr + out_row[:, None] + part * TILE_SIZE + cols[None, :],
            (fp8_vals * scale_pow2[:, None]).to(output_ptr.dtype.element_ty),
            mask=tok_mask[:, None],
        )
    else:
        # rope tail: 64 bf16 copied verbatim
        rope_offs = tl.arange(0, DIM_ROPE)
        rope_vals = tl.load(
            buf_bf16_ptr + ((data_base + DIM_NOPE) // 2)[:, None] + rope_offs[None, :],
            mask=tok_mask[:, None],
            other=0.0,
        )
        tl.store(
            output_ptr + out_row[:, None] + DIM_NOPE + rope_offs[None, :],
            rope_vals,
            mask=tok_mask[:, None],
        )
