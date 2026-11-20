# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone tests for the adjacent triton_unified_attention.py operator.

Dependencies: torch, triton, pytest (Python >= 3.10). Use a matching GPU runtime.
Run from this directory, independent of either repository's pytest settings:

    python -m pytest -c /dev/null --confcutdir=. test_triton_unified_attention.py -v

The original test cases, parameter matrix and tolerances are preserved. Large
NUM_BLOCKS and long-prefill cases require substantial GPU memory. GPU tests skip
when no GPU is available; a skipped run does not validate kernel correctness.
"""

import math
import os

import pytest
import torch
import triton
import triton.language as tl
from mcoplib.triton_unified_attention import (
    FP8_DTYPE,
    KVQuantMode,
    apply_softcap,
    compute_tile_loop_bounds,
    unified_attention,
)

if torch.cuda.is_available():
    DEVICE_TYPE = "cuda"  # Includes MetaX and ROCm PyTorch builds.
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE_TYPE = "xpu"
else:
    DEVICE_TYPE = "cpu"

pytestmark = pytest.mark.skipif(
    DEVICE_TYPE == "cpu",
    reason="Unified attention requires a GPU and compatible Triton",
)


@pytest.fixture(autouse=True)
def _restore_default_device():
    previous_device = torch.get_default_device()
    try:
        yield
    finally:
        torch.set_default_device(previous_device)


# Effective bandwidth counts logical input/output bytes, not allocated KV capacity
# or hardware-measured HBM transactions. Timing includes the complete operator.
_METRICS = []


def _cosine_similarity(actual, expected):
    actual = actual.reshape(-1)
    expected = expected.reshape(-1)
    assert actual.numel() == expected.numel()
    sums = torch.zeros(3, dtype=torch.float64, device=actual.device)
    for offset in range(0, actual.numel(), 1 << 20):
        a = actual[offset : offset + (1 << 20)].double()
        b = expected[offset : offset + (1 << 20)].double()
        sums += torch.stack(((a * b).sum(), (a * a).sum(), (b * b).sum()))
    dot, norm_a, norm_b = sums.cpu().tolist()
    if norm_a == 0 or norm_b == 0:
        return 1.0 if norm_a == norm_b == 0 else 0.0
    return dot / math.sqrt(norm_a * norm_b)


def _graph_time(fn):
    gpu = torch.cuda if DEVICE_TYPE == "cuda" else torch.xpu
    for _ in range(10):
        fn()
    gpu.synchronize()
    graph = gpu.CUDAGraph()
    with gpu.graph(graph):
        for _ in range(10):
            fn()
    graph.replay()
    gpu.synchronize()
    start = gpu.Event(enable_timing=True)
    end = gpu.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 100


def _skip_c500_fp8() -> None:
    # C500 cannot compile or execute Triton FP8 dot products.
    if DEVICE_TYPE == "cuda" and ("C500" in torch.cuda.get_device_name().upper() or 
                                  "C550" in torch.cuda.get_device_name().upper() or 
                                  "C588" in torch.cuda.get_device_name().upper()):
        pytest.skip("MetaX C500 Triton backend does not support FP8 dot")


def _attention_bytes(args):
    q, k, v, out = (args[name] for name in ("q", "k", "v", "out"))
    block_size = k.shape[1]
    lengths = args["seqused_k"].cpu().tolist()
    pages = sum(triton.cdiv(length, block_size) for length in lengths)
    kv_tokens = sum(lengths)
    total = q.numel() * q.element_size() + out.numel() * out.element_size()
    total += kv_tokens * (math.prod(k.shape[2:]) * k.element_size()
                          + math.prod(v.shape[2:]) * v.element_size())
    total += pages * args["block_table"].element_size()
    for name in ("cu_seqlens_q", "seqused_k", "q_descale", "k_descale",
                 "v_descale", "output_scale", "alibi_slopes", "qq_bias",
                 "sinks", "mm_prefix_range", "rswa_prefix_lens", "causal"):
        tensor = args.get(name)
        if isinstance(tensor, torch.Tensor):
            total += tensor.numel() * tensor.element_size()
    for name in ("k_scale_cache", "v_scale_cache"):
        tensor = args.get(name)
        if tensor is not None:
            total += kv_tokens * k.shape[2] * tensor.element_size()
    return total


def _run_attention(**args):
    unified_attention(**args)
    byte_count = _attention_bytes(args)
    milliseconds = _graph_time(lambda: unified_attention(**args))
    tensors = [args[name] for name in ("q", "k", "v", "out")]
    quantized = any(t.dtype == torch.int8 or str(t.dtype).startswith("torch.float8")
                    for t in tensors)
    return {
        "ms": milliseconds,
        "gbps": byte_count / (milliseconds * 1e6),
        "threshold": 0.999 if quantized else 0.99999,
        "config": (f"T={args['q'].shape[0]:7d} H={args['q'].shape[1]}/{args['k'].shape[2]} "
                   f"D={args['q'].shape[2]} dtype={args['q'].dtype}/{args['k'].dtype}->{args['out'].dtype} "
                   f"td={args.get('use_td', False)}"),
    }


def _report_metrics(actual, expected, metrics, name="unified_attention"):
    cosine = _cosine_similarity(actual, expected)
    passed = math.isfinite(cosine) and cosine >= metrics["threshold"]
    print(f"[{name}] {metrics['config']} cos_sim={cosine:.6f} "
          f"{'OK' if passed else 'FAIL'} {metrics['ms']:.4f} ms "
          f"{metrics['gbps']:.1f} GB/s (effective, graph)", flush=True)
    _METRICS.append((cosine, metrics["gbps"], passed))
    assert passed, f"cos_sim={cosine:.9f} < {metrics['threshold']}"


@pytest.fixture(scope="module", autouse=True)
def _metrics_summary():
    gpu = torch.cuda if DEVICE_TYPE == "cuda" else torch.xpu
    print(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES', '<unset>')!r} "
          f"device_count={gpu.device_count()}", flush=True)
    yield
    if _METRICS:
        passed = all(row[2] for row in _METRICS)
        print("=" * 92, flush=True)
        print(f"Operator cosine checks: {'ALL PASS' if passed else 'FAIL'} "
              "(cos_sim >= 0.99999 unquantized; >= 0.999 fp8/int8)", flush=True)
        print(f"Peak effective bandwidth: {max(row[1] for row in _METRICS):.1f} GB/s "
              "(logical bytes; not measured HBM bandwidth)", flush=True)


NUM_HEADS = [(4, 4), (8, 2), (5, 1)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16]

DTYPES = [torch.bfloat16]
QDTYPES = [None, FP8_DTYPE]

# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]

# 0: use 2D kernel for decode
# 8: use 3D kernel for decode
SEQ_THRESHOLD_3D_VALUES = [0, 8]


@triton.jit
def _compute_clamped_mm_tile_bounds(
    output_ptr,
    mm_prefix_range_ptr,
    USE_CAUSAL: tl.constexpr,
    USE_PER_SEQ_CAUSAL: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        0,  # context_len
        4096,  # seq_len
        4096,  # cur_batch_query_len
        140,  # q_block_local_idx: query positions [1120, 1127]
        0,  # segm_idx_or_0
        0,  # tiles_per_segment_or_0
        32,  # TILE_SIZE
        16,  # BLOCK_M
        8,  # BLOCK_Q
        2,  # num_queries_per_kv
        1024,  # SLIDING_WINDOW
        True,  # USE_MM_PREFIX
        False,  # IS_3D
        USE_CAUSAL,
        USE_PER_SEQ_CAUSAL,
        CHUNK_LOOKBACK,
        CHUNK_SIZE,
        False,  # USE_R_SWA
        True,  # MM_PREFIX_CLAMP_SW
        2,  # MAX_MM_RANGES
        mm_prefix_range_ptr,
        0,  # seq_idx
    )
    tl.store(output_ptr, loop_lo)
    tl.store(output_ptr + 1, loop_hi)
    tl.store(output_ptr + 2, max_seq_prefix_len)


def test_clamped_mm_prefix_prunes_sliding_window_tiles() -> None:
    """Retain an intersecting image range without scanning the full sequence."""
    mm_prefix_ranges = torch.tensor(
        [[[1024, 2303], [0, 0]]], dtype=torch.int32, device=DEVICE_TYPE
    )
    bounds = torch.empty(3, dtype=torch.int32, device=DEVICE_TYPE)

    _compute_clamped_mm_tile_bounds[(1,)](
        bounds,
        mm_prefix_ranges,
        USE_CAUSAL=True,
        USE_PER_SEQ_CAUSAL=False,
        CHUNK_LOOKBACK=-1,
        CHUNK_SIZE=-1,
    )

    # The range is wider than the sliding window, so the upper bound must use
    # its inclusive endpoint rather than query_pos + sliding_window.
    milliseconds = _graph_time(lambda: _compute_clamped_mm_tile_bounds[(1,)](
        bounds, mm_prefix_ranges, USE_CAUSAL=True, USE_PER_SEQ_CAUSAL=False,
        CHUNK_LOOKBACK=-1, CHUNK_SIZE=-1))
    _report_metrics(bounds, torch.tensor([3, 72, 4096], device=bounds.device),
                    {"config": "clamped_mm_prefix", "ms": milliseconds,
                     "gbps": (mm_prefix_ranges.numel() * mm_prefix_ranges.element_size()
                              + bounds.numel() * bounds.element_size()) / (milliseconds * 1e6),
                     "threshold": 0.99999}, name="tile_bounds")
    assert bounds.tolist() == [3, 72, 4096]


@pytest.mark.parametrize(
    ("use_causal", "use_per_seq_causal"), [(False, False), (True, True)]
)
def test_clamped_mm_prefix_preserves_noncausal_right_window(
    use_causal: bool, use_per_seq_causal: bool
) -> None:
    """Union the non-causal right window with intersecting image ranges."""
    mm_prefix_ranges = torch.tensor(
        [[[1024, 1500], [0, 0]]], dtype=torch.int32, device=DEVICE_TYPE
    )
    bounds = torch.empty(3, dtype=torch.int32, device=DEVICE_TYPE)

    _compute_clamped_mm_tile_bounds[(1,)](
        bounds,
        mm_prefix_ranges,
        USE_CAUSAL=use_causal,
        USE_PER_SEQ_CAUSAL=use_per_seq_causal,
        CHUNK_LOOKBACK=-1,
        CHUNK_SIZE=-1,
    )

    # q_hi=1127 and window=1024 admit keys through 2150, beyond the image
    # range endpoint at 1500. Tile 67 must therefore remain in the loop.
    milliseconds = _graph_time(lambda: _compute_clamped_mm_tile_bounds[(1,)](
        bounds, mm_prefix_ranges, USE_CAUSAL=use_causal,
        USE_PER_SEQ_CAUSAL=use_per_seq_causal, CHUNK_LOOKBACK=-1, CHUNK_SIZE=-1))
    _report_metrics(bounds, torch.tensor([3, 68, 4096], device=bounds.device),
                    {"config": "noncausal_mm_prefix", "ms": milliseconds,
                     "gbps": (mm_prefix_ranges.numel() * mm_prefix_ranges.element_size()
                              + bounds.numel() * bounds.element_size()) / (milliseconds * 1e6),
                     "threshold": 0.99999}, name="tile_bounds")
    assert bounds.tolist() == [3, 68, 4096]


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().tolist()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    head_size_v = value_cache.shape[-1]

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len].float() * scale
        if q_descale is not None:
            q = q * q_descale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i][:num_kv_blocks]

        # Gather bytes for FP8 caches: advanced indexing may not support FP8.
        if str(key_cache.dtype).startswith("torch.float8"):
            k = key_cache.view(torch.uint8)[block_indices].view(key_cache.dtype)
        else:
            k = key_cache[block_indices]
        if str(value_cache.dtype).startswith("torch.float8"):
            v = value_cache.view(torch.uint8)[block_indices].view(value_cache.dtype)
        else:
            v = value_cache[block_indices]
        k = k.view(-1, num_kv_heads, head_size)[:kv_len].float()
        v = v.view(-1, num_kv_heads, head_size_v)[:kv_len].float()
        if k_descale is not None:
            k = k * k_descale[i].view(1, -1, 1)
        if v_descale is not None:
            v = v * v_descale[i].view(1, -1, 1)

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k.float())
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("hqk,khd->qhd", attn, v.float())
        if not str(query.dtype).startswith("torch.float8"):
            out = out.to(query.dtype)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@torch.inference_mode()
def test_fp8_softmax_preserves_small_probabilities() -> None:
    _skip_c500_fp8()
    """Keep exp(-8) contributions that underflow when cast directly to E4M3."""
    device = torch.device(DEVICE_TYPE)
    num_tokens = block_size = 32
    head_size = 128

    query = torch.zeros(1, 1, head_size, dtype=FP8_DTYPE, device=device)
    query[..., 0] = 1
    key_cache = torch.zeros(1, block_size, 1, head_size, dtype=FP8_DTYPE, device=device)
    key_cache[:, 1:, :, 0] = -8
    value_cache = torch.ones_like(key_cache)
    value_cache[:, 0] = 0
    output = torch.empty(1, 1, head_size, dtype=torch.bfloat16, device=device)

    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    seqused_k = torch.tensor([num_tokens], dtype=torch.int32, device=device)
    block_table = torch.tensor([[0]], dtype=torch.int32, device=device)
    scale = torch.ones(1, dtype=torch.float32, device=device)

    metrics = _run_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=num_tokens,
        softmax_scale=1.0,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0,
        q_descale=scale,
        k_descale=scale,
        v_descale=scale,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )

    scores = torch.einsum("qhd,thd->qht", query.float(), key_cache[0].float())
    probabilities = torch.softmax(scores, dim=-1)
    expected = torch.einsum("qht,thd->qhd", probabilities, value_cache[0].float())

    _report_metrics(output, expected, metrics)
    torch.testing.assert_close(output.float(), expected, atol=5e-5, rtol=1e-2)


def ref_paged_clamped_mm_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    mm_ranges: list[list[tuple[int, int]]],
    scale: float,
    sliding_window: int,
    chunk_lookback: int,
) -> torch.Tensor:
    block_tables_cpu = block_tables.cpu().tolist()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    chunk_size = sliding_window // (chunk_lookback + 1)

    outputs: list[torch.Tensor] = []
    query_start = 0
    for req_idx, (query_len, kv_len) in enumerate(zip(query_lens, kv_lens)):
        query_i = query[query_start : query_start + query_len].float()
        context_len = kv_len - query_len
        query_pos = torch.arange(query_len, device=query.device) + context_len
        key_pos = torch.arange(kv_len, device=query.device)

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_cpu[req_idx][:num_kv_blocks]
        key_i = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        value_i = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        if query_i.shape[1] != key_i.shape[1]:
            repeats = query_i.shape[1] // key_i.shape[1]
            key_i = torch.repeat_interleave(key_i, repeats, dim=1)
            value_i = torch.repeat_interleave(value_i, repeats, dim=1)

        delta = query_pos[:, None] - key_pos[None, :]
        keep = (delta >= 0) & (
            query_pos[:, None] // chunk_size - key_pos[None, :] // chunk_size
            <= chunk_lookback
        )
        for range_start, range_end in mm_ranges[req_idx]:
            q_in_range = (query_pos >= range_start) & (query_pos <= range_end)
            k_in_range = (key_pos >= range_start) & (key_pos <= range_end)
            mm_mask = q_in_range[:, None] & k_in_range[None, :]
            keep |= mm_mask & (delta < sliding_window)

        scores = torch.einsum("qhd,khd->hqk", query_i, key_i.float()) * scale
        scores.masked_fill_(~keep[None], float("-inf"))
        probs = scores.softmax(-1).to(value_i.dtype)
        outputs.append(torch.einsum("hqk,khd->qhd", probs, value_i))
        query_start += query_len

    return torch.cat(outputs)


@torch.inference_mode()
def test_triton_unified_attn_clamped_mm_matches_dense_reference() -> None:
    torch.manual_seed(0)
    query_lens = [384, 96]
    kv_lens_list = [384, 320]
    mm_ranges = [[(32, 351)], [(160, 287)]]
    sliding_window = 128
    chunk_lookback = 0
    block_size = 16
    num_query_heads = 4
    num_kv_heads = 2
    head_size = 128
    scale = head_size**-0.5

    query = torch.randn(
        sum(query_lens),
        num_query_heads,
        head_size,
        dtype=torch.bfloat16,
        device=DEVICE_TYPE,
    )
    num_blocks_per_req = [
        (kv_len + block_size - 1) // block_size for kv_len in kv_lens_list
    ]
    total_blocks = sum(num_blocks_per_req)
    key_cache = torch.randn(
        total_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device=DEVICE_TYPE,
    )
    value_cache = torch.randn_like(key_cache)

    block_tables = torch.zeros(
        len(query_lens),
        max(num_blocks_per_req),
        dtype=torch.int32,
        device=DEVICE_TYPE,
    )
    block_start = 0
    for req_idx, num_blocks in enumerate(num_blocks_per_req):
        block_tables[req_idx, :num_blocks] = torch.arange(
            block_start,
            block_start + num_blocks,
            dtype=torch.int32,
            device=DEVICE_TYPE,
        )
        block_start += num_blocks

    cu_seqlens_q = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device=DEVICE_TYPE
    ).cumsum(0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32, device=DEVICE_TYPE)
    mm_prefix_range = torch.tensor(
        [[[32, 351]], [[160, 287]]], dtype=torch.int32, device=DEVICE_TYPE
    )
    actual = torch.empty_like(query)

    metrics = _run_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=actual,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(query_lens),
        seqused_k=kv_lens,
        max_seqlen_k=max(kv_lens_list),
        softmax_scale=scale,
        causal=True,
        window_size=(sliding_window - 1, 0),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        mm_prefix_range=mm_prefix_range,
        chunk_lookback=chunk_lookback,
        mm_prefix_clamp_sliding_window=True,
    )

    expected = ref_paged_clamped_mm_attn(
        query,
        key_cache,
        value_cache,
        query_lens,
        kv_lens_list,
        block_tables,
        mm_ranges,
        scale,
        sliding_window,
        chunk_lookback,
    )
    chunk_only = ref_paged_clamped_mm_attn(
        query,
        key_cache,
        value_cache,
        query_lens,
        kv_lens_list,
        block_tables,
        [[], []],
        scale,
        sliding_window,
        chunk_lookback,
    )

    _report_metrics(actual, expected, metrics)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
    assert not torch.allclose(expected, chunk_only, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 64, 128, 256])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("q_dtype", QDTYPES)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    dtype: torch.dtype,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    q_dtype: torch.dtype | None,
    seq_threshold_3D: int,
) -> None:
    torch.set_default_device(DEVICE_TYPE)

    if q_dtype is not None:
        _skip_c500_fp8()


    torch.manual_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = torch.empty_like(query)

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None
    k_descale = None
    v_descale = None
    kv_quant_mode = KVQuantMode.NONE
    if q_dtype is not None:
        # Use non-1 scales so FP8 Q/K/V descale handling is tested explicitly.
        q_scale = torch.tensor(0.75, dtype=torch.float32)
        k_scale = torch.tensor(0.5, dtype=torch.float32)
        v_scale = torch.tensor(0.25, dtype=torch.float32)
        q_descale = q_scale
        scale_shape = (num_seqs, num_kv_heads)
        k_descale = torch.full(scale_shape, k_scale.item(), dtype=torch.float32)
        v_descale = torch.full(scale_shape, v_scale.item(), dtype=torch.float32)
        maybe_quantized_query = (query / q_scale).to(q_dtype)
        maybe_quantized_key_cache = (key_cache / k_scale).to(q_dtype)
        maybe_quantized_value_cache = (value_cache / v_scale).to(q_dtype)
        kv_quant_mode = KVQuantMode.FP8_PER_TENSOR

    num_par_softmax_segments = 16
    head_size_padded = triton.next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    metrics = _run_attention(
        q=maybe_quantized_query,
        k=maybe_quantized_key_cache,
        v=maybe_quantized_value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        kv_quant_mode=kv_quant_mode,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )
    if q_dtype is not None:
        # Operator accuracy uses its actual quantized inputs and descales.
        # Keep the original end-to-end assert_close below as a separate check.
        quantized_ref = ref_paged_attn(
            query=maybe_quantized_query,
            key_cache=maybe_quantized_key_cache,
            value_cache=maybe_quantized_value_cache,
            query_lens=query_lens,
            kv_lens=kv_lens,
            block_tables=block_tables,
            scale=scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        print(f"[quantization] reference=actual_inputs "
              f"end_to_end_cos={_cosine_similarity(output, ref_output):.9f} "
              f"input_quantization_cos={_cosine_similarity(quantized_ref, ref_output):.9f}",
              flush=True)
        _report_metrics(output, quantized_ref, metrics)
        torch.testing.assert_close(output.float(), quantized_ref, atol=1.5e-1, rtol=1.5e-1)
    else:
        _report_metrics(output, ref_output, metrics)
    atol, rtol = 1.5e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    (
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn_bf16_query_fp8_kv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    num_blocks: int,
    seq_threshold_3D: int,
) -> None:
    """Test bf16 Q with FP8 per-tensor KV cache (dequant via _cast_kv_tile)."""
    _skip_c500_fp8()
    torch.set_default_device(DEVICE_TYPE)
    torch.manual_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (-1, -1)
    scale = head_size**-0.5

    dtype = torch.bfloat16
    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)

    k_scale = torch.tensor(0.5, dtype=torch.float32)
    v_scale = torch.tensor(0.25, dtype=torch.float32)
    fp8_key_cache = (key_cache / k_scale).to(FP8_DTYPE)
    fp8_value_cache = (value_cache / v_scale).to(FP8_DTYPE)

    scale_shape = (num_seqs, num_kv_heads)
    k_descale = torch.full(scale_shape, k_scale.item(), dtype=torch.float32)
    v_descale = torch.full(scale_shape, v_scale.item(), dtype=torch.float32)

    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = torch.empty_like(query)

    num_par_softmax_segments = 16
    head_size_padded = triton.next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    metrics = _run_attention(
        q=query,
        k=fp8_key_cache,
        v=fp8_value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_t,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=k_descale,
        v_descale=v_descale,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
    )

    quantized_ref = ref_paged_attn(
        query=query,
        key_cache=fp8_key_cache,
        value_cache=fp8_value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    print(f"[quantization] reference=actual_inputs "
          f"end_to_end_cos={_cosine_similarity(output, ref_output):.9f} "
          f"input_quantization_cos={_cosine_similarity(quantized_ref, ref_output):.9f}",
          flush=True)
    _report_metrics(output, quantized_ref, metrics)
    torch.testing.assert_close(output, quantized_ref, atol=1.5e-1, rtol=1.5e-1)
    atol, rtol = 1.5e-1, 1.5e-1
    (
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output - ref_output))}",
    )


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 1328), (5, 18), (129, 463)],
        [(1, 523), (1, 37), (1, 2011)],
        [(1, 1)] * 533,
        [(533, 533)] * 533,
    ],
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 64, 128, 256])
@pytest.mark.parametrize("soft_cap", [None, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("seq_threshold_3D", SEQ_THRESHOLD_3D_VALUES)
@torch.inference_mode()
def test_triton_unified_attn_fp16_input_fp8_output(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    seq_threshold_3D: int,
) -> None:
    """Test with fp16 input and fp8 output using output_scale."""
    _skip_c500_fp8()
    torch.set_default_device(DEVICE_TYPE)

    torch.manual_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    dtype = torch.float16
    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = torch.empty(sum(query_lens), num_query_heads, head_size, dtype=FP8_DTYPE)

    output_scale = torch.tensor(0.5, dtype=torch.float32)

    num_par_softmax_segments = 16
    head_size_padded = triton.next_power_of_2(head_size)
    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_par_softmax_segments),
        dtype=torch.float32,
    )

    metrics = _run_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_tensor,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        output_scale=output_scale,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )

    output_fp16 = output.to(torch.float32) * output_scale.item()
    output_fp16 = output_fp16.to(torch.float16)

    _report_metrics(output_fp16, ref_output, metrics)
    atol, rtol = 2e-1, 2e-1
    (
        torch.testing.assert_close(output_fp16, ref_output, atol=atol, rtol=rtol),
        f"{torch.max(torch.abs(output_fp16 - ref_output))}",
    )


@triton.jit
def _softcap_probe_kernel(
    scores_ptr,
    out_ptr,
    numel,
    soft_cap,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    scores = tl.load(scores_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, apply_softcap(scores, soft_cap), mask=mask)


@torch.inference_mode()
def test_softcap_does_not_overflow_on_large_scores() -> None:
    """Scores above ~88 * soft_cap must not overflow to inf/NaN.

    The exp-based softcap computes ``(exp(y) - exp(-y)) / (exp(y) + exp(-y))``
    with ``y = S / soft_cap``. For ``|y| > ~88`` the exponentials overflow to
    ``inf`` and the ratio becomes ``inf / inf = NaN``, poisoning the whole
    attention row. Gemma-2 style models use ``attn_logit_softcapping = 50``,
    so scores above 4400 are in range for their large attention logits.
    """
    soft_cap = 50.0
    scores = torch.tensor(
        [1.0, 100.0, 1000.0, 3000.0, 5000.0, 10000.0, -1.0, -10000.0, 4400.0],
        device=DEVICE_TYPE,
        dtype=torch.float32,
    )
    out = torch.empty_like(scores)
    _softcap_probe_kernel[(1,)](scores, out, scores.numel(), soft_cap, BLOCK=16)
    milliseconds = _graph_time(lambda: _softcap_probe_kernel[(1,)](
        scores, out, scores.numel(), soft_cap, BLOCK=16))
    ref = soft_cap * torch.tanh(scores / soft_cap)
    _report_metrics(out, ref, {"config": "large_scores", "ms": milliseconds,
                    "gbps": (scores.numel() * scores.element_size()
                             + out.numel() * out.element_size()) / (milliseconds * 1e6),
                    "threshold": 0.99999}, name="softcap")
    assert torch.isfinite(out).all(), out
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
