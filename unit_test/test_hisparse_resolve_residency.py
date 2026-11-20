import pytest
import torch
import mcoplib._C
import re


OP = torch.ops._C.hisparse_resolve_residency
DEVICE = "cuda"
ROW_BYTES = 16
BLOCK_SIZE = 4


def _base_inputs(rows=2, top_k=2, hot_size=4, block_size=BLOCK_SIZE):
    host_rows = 16
    num_hot_blocks = 2
    hot_table_cols = max(1, (hot_size + block_size - 1) // block_size)

    return {
        "host_cache": torch.zeros((host_rows, ROW_BYTES), dtype=torch.uint8, pin_memory=True),
        "hot_cache": torch.zeros((num_hot_blocks, block_size, ROW_BYTES), dtype=torch.uint8, device=DEVICE),
        "hot_block_table": torch.zeros((max(rows, 1), hot_table_cols), dtype=torch.int32, device=DEVICE),
        "global_indices": torch.zeros((rows, top_k), dtype=torch.int32, device=DEVICE),
        "hot_indices": torch.zeros((rows, top_k), dtype=torch.int32, device=DEVICE),
        "device_global_indices": torch.zeros((max(rows, 1), hot_size), dtype=torch.int32, device=DEVICE)[:rows],
        "lru_slots": torch.zeros((max(rows, 1), hot_size), dtype=torch.int16, device=DEVICE)[:rows],
        "request_state_indices": None,
        "region_stride": hot_size,
        "max_union_rows": min(top_k, hot_size),
        "miss_mask": None,
        "stats": None,
        "attention_indices": None,
        "attention_block_stride": 0,
        "request_ids": None,
        "source_block_table": None,
        "source_block_size": block_size,
        "resolved_global_indices": None,
        "valid_counts": None,
        "swap_host_physical_rows": None,
        "swap_device_physical_rows": None,
        "swap_counts": None,
        "resident_block_table": None,
        "resident_block_size": block_size,
        "resident_null_block": 0,
    }


def _invoke(args):
    OP(
        args["host_cache"],
        args["hot_cache"],
        args["hot_block_table"],
        args["global_indices"],
        args["hot_indices"],
        args["device_global_indices"],
        args["lru_slots"],
        args["request_state_indices"],
        args["region_stride"],
        args["max_union_rows"],
        args["miss_mask"],
        args["stats"],
        args["attention_indices"],
        args["attention_block_stride"],
        args["request_ids"],
        args["source_block_table"],
        args["source_block_size"],
        args["resolved_global_indices"],
        args["valid_counts"],
        args["swap_host_physical_rows"],
        args["swap_device_physical_rows"],
        args["swap_counts"],
        args["resident_block_table"],
        args["resident_block_size"],
        args["resident_null_block"],
    )


def _assert_raises(args, message):
    with pytest.raises(RuntimeError, match=re.escape(message)):
        _invoke(args)


# ---------------------------------------------------------------------------
# Basic invocation
# ---------------------------------------------------------------------------

def test_hisparse_resolve_residency_basic():
    args = _base_inputs()
    _invoke(args)
    torch.cuda.synchronize()


def test_hisparse_resolve_residency_empty_rows():
    args = _base_inputs(rows=0, top_k=2, hot_size=4)
    _invoke(args)
    torch.cuda.synchronize()


def test_hisparse_resolve_residency_empty_top_k():
    args = _base_inputs(rows=2, top_k=0, hot_size=4)
    _invoke(args)
    torch.cuda.synchronize()


def test_hisparse_resolve_residency_multiple_rows_and_top_k():
    args = _base_inputs(rows=4, top_k=3, hot_size=8, block_size=4)
    args["max_union_rows"] = 5
    args["hot_block_table"] = torch.zeros((4, 2), dtype=torch.int32, device=DEVICE)
    _invoke(args)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# max_union_rows validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_union_rows", [1, 2, 3, 4])
def test_hisparse_resolve_residency_accepts_valid_max_union_rows(max_union_rows):
    args = _base_inputs(rows=2, top_k=2, hot_size=4)
    args["max_union_rows"] = max_union_rows
    _invoke(args)
    torch.cuda.synchronize()


@pytest.mark.parametrize("max_union_rows", [0, -1, -4])
def test_hisparse_resolve_residency_rejects_non_positive_max_union_rows(max_union_rows):
    args = _base_inputs(rows=2, top_k=2, hot_size=4)
    args["max_union_rows"] = max_union_rows
    _assert_raises(args, r"max_union_rows must be in")


@pytest.mark.parametrize("max_union_rows", [5, 8, 32769])
def test_hisparse_resolve_residency_rejects_max_union_rows_above_hot_size(max_union_rows):
    args = _base_inputs(rows=2, top_k=2, hot_size=4)
    args["max_union_rows"] = max_union_rows
    _assert_raises(args, r"max_union_rows must be in")


def test_hisparse_resolve_residency_max_union_rows_one():
    args = _base_inputs(rows=1, top_k=1, hot_size=4)
    args["max_union_rows"] = 1
    _invoke(args)
    torch.cuda.synchronize()


def test_hisparse_resolve_residency_max_union_rows_equals_hot_size():
    args = _base_inputs(rows=2, top_k=2, hot_size=4)
    args["max_union_rows"] = 4
    _invoke(args)
    torch.cuda.synchronize()


def test_hisparse_resolve_residency_max_union_rows_greater_than_top_k():
    args = _base_inputs(rows=2, top_k=2, hot_size=8, block_size=4)
    args["max_union_rows"] = 6
    args["hot_block_table"] = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    assert args["max_union_rows"] > args["global_indices"].size(1)
    assert args["max_union_rows"] <= args["lru_slots"].size(1)
    _invoke(args)
    torch.cuda.synchronize()


def test_hisparse_resolve_residency_rejects_max_union_rows_above_hot_size_with_small_top_k():
    args = _base_inputs(rows=1, top_k=1, hot_size=4)
    args["max_union_rows"] = 5
    _assert_raises(args, r"max_union_rows must be in")


def test_hisparse_resolve_residency_empty_rows_skips_max_union_rows_validation():
    args = _base_inputs(rows=0, top_k=2, hot_size=4)
    args["max_union_rows"] = 0
    _invoke(args)


def test_hisparse_resolve_residency_empty_top_k_skips_max_union_rows_validation():
    args = _base_inputs(rows=2, top_k=0, hot_size=4)
    args["max_union_rows"] = 0
    _invoke(args)


# ---------------------------------------------------------------------------
# Device and dtype validation
# ---------------------------------------------------------------------------

def test_hisparse_resolve_residency_rejects_non_pinned_host_cache():
    args = _base_inputs()
    args["host_cache"] = torch.zeros((16, ROW_BYTES), dtype=torch.uint8)
    _assert_raises(args, "host_cache must be pinned CPU memory")


def test_hisparse_resolve_residency_rejects_hot_cache_on_cpu():
    args = _base_inputs()
    args["hot_cache"] = torch.zeros((2, BLOCK_SIZE, ROW_BYTES), dtype=torch.uint8)
    _assert_raises(args, "hot_cache must be on CUDA")


@pytest.mark.parametrize("field", ["global_indices", "hot_indices", "device_global_indices"])
def test_hisparse_resolve_residency_rejects_wrong_int32_dtype(field):
    args = _base_inputs()
    args[field] = args[field].to(torch.int64)
    _assert_raises(
        args,
        {
            "global_indices": "global_indices/hot_indices must be int32",
            "hot_indices": "global_indices/hot_indices must be int32",
            "device_global_indices": "device_global_indices must be int32",
        }[field],
    )


def test_hisparse_resolve_residency_rejects_wrong_lru_dtype():
    args = _base_inputs()
    args["lru_slots"] = args["lru_slots"].to(torch.int32)
    _assert_raises(args, "lru_slots must be int16")


def test_hisparse_resolve_residency_rejects_wrong_hot_block_table_dtype():
    args = _base_inputs()
    args["hot_block_table"] = args["hot_block_table"].to(torch.int64)
    _assert_raises(args, "hot_block_table must be a 2D int32 CUDA tensor")


def test_hisparse_resolve_residency_rejects_one_dimensional_hot_block_table():
    args = _base_inputs()
    args["hot_block_table"] = torch.zeros((2,), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "hot_block_table must be a 2D int32 CUDA tensor")


# ---------------------------------------------------------------------------
# Shape and layout validation
# ---------------------------------------------------------------------------

def test_hisparse_resolve_residency_rejects_non_row_major_global_indices():
    args = _base_inputs(rows=2, top_k=3)
    args["global_indices"] = torch.zeros((3, 2), dtype=torch.int32, device=DEVICE).transpose(0, 1)
    args["hot_indices"] = torch.zeros((2, 3), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "global_indices must be row-major 2D")


def test_hisparse_resolve_residency_rejects_wrong_hot_indices_shape():
    args = _base_inputs()
    args["hot_indices"] = torch.zeros((2, 3), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "hot_indices must have one contiguous row per launch row")


def test_hisparse_resolve_residency_rejects_noncontiguous_hot_indices():
    args = _base_inputs()
    args["hot_indices"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)[:, ::2]
    _assert_raises(args, "hot_indices must have one contiguous row per launch row")


def test_hisparse_resolve_residency_rejects_noncontiguous_device_global_indices():
    args = _base_inputs()
    args["device_global_indices"] = torch.zeros((2, 8), dtype=torch.int32, device=DEVICE)[:, ::2]
    _assert_raises(args, "device_global_indices must be contiguous 2D")


def test_hisparse_resolve_residency_rejects_wrong_device_global_indices_dim():
    args = _base_inputs()
    args["device_global_indices"] = torch.zeros((8,), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "device_global_indices must be contiguous 2D")


def test_hisparse_resolve_residency_rejects_wrong_lru_rows():
    args = _base_inputs()
    args["lru_slots"] = torch.zeros((3, 4), dtype=torch.int16, device=DEVICE)
    _assert_raises(args, "lru_slots must be contiguous with one row per request state")


def test_hisparse_resolve_residency_rejects_noncontiguous_lru_slots():
    args = _base_inputs()
    args["lru_slots"] = torch.zeros((2, 8), dtype=torch.int16, device=DEVICE)[:, ::2]
    _assert_raises(args, "lru_slots must be contiguous with one row per request state")


def test_hisparse_resolve_residency_rejects_non_16_byte_aligned_rows():
    args = _base_inputs()
    args["hot_cache"] = torch.zeros((2, BLOCK_SIZE, 17), dtype=torch.uint8, device=DEVICE)
    args["host_cache"] = torch.zeros((16, 17), dtype=torch.uint8, pin_memory=True)
    _assert_raises(args, "KV rows must be 16-byte aligned")


def test_hisparse_resolve_residency_rejects_noncontiguous_hot_cache_rows():
    args = _base_inputs()
    args["hot_cache"] = torch.empty_strided(
        (2, BLOCK_SIZE, ROW_BYTES),
        (BLOCK_SIZE * ROW_BYTES * 2, ROW_BYTES * 2, 1),
        dtype=torch.uint8,
        device=DEVICE,
    )
    _assert_raises(args, "hot-cache rows must be contiguous")


# ---------------------------------------------------------------------------
# Hot buffer and region validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("top_k", [2, 3])
def test_hisparse_resolve_residency_rejects_hot_size_smaller_than_top_k(top_k):
    hot_size = top_k - 1
    args = _base_inputs(rows=2, top_k=top_k, hot_size=hot_size)

    assert args["global_indices"].size(1) == top_k
    assert args["lru_slots"].size(1) == hot_size
    assert args["lru_slots"].size(1) < args["global_indices"].size(1)

    _assert_raises(args, "hot buffer size must be >= top_k")


def test_hisparse_resolve_residency_rejects_hot_size_above_int16_limit():
    hot_size = 32769
    args = _base_inputs(rows=1, top_k=1, hot_size=hot_size)
    _assert_raises(args, "hot buffer size must fit int16 slots")


def test_hisparse_resolve_residency_rejects_region_stride_mismatch():
    args = _base_inputs()
    args["region_stride"] = args["region_stride"] + 1
    _assert_raises(args, "region_stride must match the LRU size")


def test_hisparse_resolve_residency_rejects_device_global_indices_short_width():
    args = _base_inputs()
    args["device_global_indices"] = torch.zeros((2, 3), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "device_global_indices must cover the full hot region")


def test_hisparse_resolve_residency_rejects_hot_block_table_too_few_columns():
    args = _base_inputs(rows=2, top_k=2, hot_size=8, block_size=4)
    args["hot_block_table"] = torch.zeros((2, 1), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "hot_block_table has too few columns")


def test_hisparse_resolve_residency_rejects_hot_block_table_too_few_rows():
    args = _base_inputs(rows=2)
    args["hot_block_table"] = torch.zeros((1, 1), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "hot_block_table has too few rows")


# ---------------------------------------------------------------------------
# Request/source block table validation
# ---------------------------------------------------------------------------

def test_hisparse_resolve_residency_rejects_request_ids_without_source_table():
    args = _base_inputs()
    args["request_ids"] = torch.zeros((2,), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "request_ids and source_block_table must be provided together")


def test_hisparse_resolve_residency_rejects_source_table_without_request_ids():
    args = _base_inputs()
    args["source_block_table"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "request_ids and source_block_table must be provided together")


def test_hisparse_resolve_residency_rejects_request_ids_wrong_dtype():
    args = _base_inputs()
    args["request_ids"] = torch.zeros((2,), dtype=torch.int64, device=DEVICE)
    args["source_block_table"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "request_ids must be contiguous int32 on CUDA with one entry per row")


def test_hisparse_resolve_residency_rejects_request_ids_too_short():
    args = _base_inputs(rows=2)
    args["request_ids"] = torch.zeros((1,), dtype=torch.int32, device=DEVICE)
    args["source_block_table"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "request_ids must be contiguous int32 on CUDA with one entry per row")


def test_hisparse_resolve_residency_rejects_source_table_wrong_dtype():
    args = _base_inputs()
    args["request_ids"] = torch.zeros((2,), dtype=torch.int32, device=DEVICE)
    args["source_block_table"] = torch.zeros((2, 4), dtype=torch.int64, device=DEVICE)
    _assert_raises(args, "source_block_table must be a row-major 2D int32 CUDA tensor")


def test_hisparse_resolve_residency_rejects_source_block_size_zero():
    args = _base_inputs()
    args["request_ids"] = torch.zeros((2,), dtype=torch.int32, device=DEVICE)
    args["source_block_table"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)
    args["source_block_size"] = 0
    _assert_raises(args, "source_block_size must be positive")


def test_hisparse_resolve_residency_rejects_too_few_hot_rows_for_source_requests():
    args = _base_inputs(rows=2)
    args["request_ids"] = torch.zeros((3,), dtype=torch.int32, device=DEVICE)
    args["source_block_table"] = torch.zeros((3, 4), dtype=torch.int32, device=DEVICE)
    args["hot_block_table"] = torch.zeros((2, 1), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "hot_block_table has too few rows")


# ---------------------------------------------------------------------------
# Resident table validation
# ---------------------------------------------------------------------------

def test_hisparse_resolve_residency_rejects_resident_table_without_source_table():
    args = _base_inputs()
    args["resident_block_table"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "resident lookup requires request_ids and source_block_table")


def test_hisparse_resolve_residency_rejects_resident_table_wrong_dtype():
    args = _base_inputs()
    args["request_ids"] = torch.zeros((2,), dtype=torch.int32, device=DEVICE)
    args["source_block_table"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)
    args["resident_block_table"] = torch.zeros((2, 4), dtype=torch.int64, device=DEVICE)
    _assert_raises(args, "resident_block_table must be a row-major 2D int32 CUDA tensor")


def test_hisparse_resolve_residency_rejects_resident_block_size_mismatch():
    args = _base_inputs()
    args["request_ids"] = torch.zeros((2,), dtype=torch.int32, device=DEVICE)
    args["source_block_table"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)
    args["resident_block_table"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)
    args["resident_block_size"] = BLOCK_SIZE + 1
    _assert_raises(args, "resident and hot block sizes must match")


def test_hisparse_resolve_residency_rejects_negative_resident_null_block():
    args = _base_inputs()
    args["request_ids"] = torch.zeros((2,), dtype=torch.int32, device=DEVICE)
    args["source_block_table"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)
    args["resident_block_table"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)
    args["resident_null_block"] = -1
    _assert_raises(args, "resident null block must be non-negative")


# ---------------------------------------------------------------------------
# Optional outputs validation
# ---------------------------------------------------------------------------

def test_hisparse_resolve_residency_rejects_resolved_indices_wrong_shape():
    args = _base_inputs()
    args["resolved_global_indices"] = torch.zeros((2, 3), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "resolved_global_indices must be contiguous int32 matching indices")


def test_hisparse_resolve_residency_rejects_resolved_indices_wrong_dtype():
    args = _base_inputs()
    args["resolved_global_indices"] = torch.zeros((2, 2), dtype=torch.int64, device=DEVICE)
    _assert_raises(args, "resolved_global_indices must be contiguous int32 matching indices")


def test_hisparse_resolve_residency_rejects_valid_counts_wrong_dtype():
    args = _base_inputs()
    args["valid_counts"] = torch.zeros((2,), dtype=torch.int64, device=DEVICE)
    _assert_raises(args, "valid_counts must be int32 on CUDA with one entry per row")


def test_hisparse_resolve_residency_rejects_valid_counts_wrong_length():
    args = _base_inputs()
    args["valid_counts"] = torch.zeros((1,), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "valid_counts must be int32 on CUDA with one entry per row")


def test_hisparse_resolve_residency_rejects_partial_compact_swaps():
    args = _base_inputs()
    args["swap_host_physical_rows"] = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "compact swap tensors must be provided together")


def test_hisparse_resolve_residency_rejects_compact_swaps_wrong_shape():
    args = _base_inputs()
    args["swap_host_physical_rows"] = torch.zeros((2, 3), dtype=torch.int32, device=DEVICE)
    args["swap_device_physical_rows"] = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    args["swap_counts"] = torch.zeros((2,), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "compact swaps must be contiguous int32 matching indices")


def test_hisparse_resolve_residency_rejects_compact_swaps_wrong_dtype():
    args = _base_inputs()
    args["swap_host_physical_rows"] = torch.zeros((2, 2), dtype=torch.int64, device=DEVICE)
    args["swap_device_physical_rows"] = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    args["swap_counts"] = torch.zeros((2,), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "compact swaps must be contiguous int32 matching indices")


# ---------------------------------------------------------------------------
# Request state validation
# ---------------------------------------------------------------------------

def test_hisparse_resolve_residency_rejects_short_device_global_indices_without_request_state():
    args = _base_inputs(rows=3, top_k=2, hot_size=4)
    args["device_global_indices"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)
    args["lru_slots"] = torch.zeros((2, 4), dtype=torch.int16, device=DEVICE)
    args["request_state_indices"] = None

    assert args["device_global_indices"].size(0) < args["global_indices"].size(0)
    assert args["request_state_indices"] is None

    _assert_raises(args, "device_global_indices has too few rows")


def test_hisparse_resolve_residency_rejects_request_state_wrong_dtype():
    args = _base_inputs()
    args["request_state_indices"] = torch.zeros((2,), dtype=torch.int64, device=DEVICE)
    _assert_raises(args, "request_state_indices must be contiguous 1D int32 on CUDA")


def test_hisparse_resolve_residency_rejects_request_state_wrong_dim():
    args = _base_inputs()
    args["request_state_indices"] = torch.zeros((2, 1), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "request_state_indices must be contiguous 1D int32 on CUDA")


def test_hisparse_resolve_residency_with_request_state_indices():
    args = _base_inputs(rows=2, top_k=2, hot_size=4)
    args["request_state_indices"] = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    _invoke(args)
    torch.cuda.synchronize()


def test_hisparse_resolve_residency_with_non_identity_request_state_indices():
    args = _base_inputs(rows=2, top_k=2, hot_size=4)
    args["request_state_indices"] = torch.tensor([1, 0], dtype=torch.int32, device=DEVICE)
    _invoke(args)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# miss_mask, stats, attention_indices validation
# ---------------------------------------------------------------------------

def test_hisparse_resolve_residency_rejects_miss_mask_wrong_dtype():
    args = _base_inputs()
    args["miss_mask"] = torch.zeros((2, 2), dtype=torch.int64, device=DEVICE)
    _assert_raises(args, "miss_mask must be a contiguous int32 CUDA tensor matching")


def test_hisparse_resolve_residency_rejects_miss_mask_wrong_shape():
    args = _base_inputs()
    args["miss_mask"] = torch.zeros((2, 3), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "miss_mask must be a contiguous int32 CUDA tensor matching")


def test_hisparse_resolve_residency_rejects_stats_wrong_dtype():
    args = _base_inputs()
    args["stats"] = torch.zeros((2,), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "stats must be a contiguous two-element uint64 CUDA tensor")


def test_hisparse_resolve_residency_rejects_stats_wrong_length():
    args = _base_inputs()
    args["stats"] = torch.zeros((3,), dtype=torch.uint64, device=DEVICE)
    _assert_raises(args, "stats must be a contiguous two-element uint64 CUDA tensor")


def test_hisparse_resolve_residency_rejects_attention_indices_wrong_dtype():
    args = _base_inputs()
    args["attention_indices"] = torch.zeros((2, 2), dtype=torch.int64, device=DEVICE)
    args["attention_block_stride"] = BLOCK_SIZE
    _assert_raises(args, "attention_indices must be row-major int32 matching global_indices")


def test_hisparse_resolve_residency_rejects_attention_indices_wrong_shape():
    args = _base_inputs()
    args["attention_indices"] = torch.zeros((2, 3), dtype=torch.int32, device=DEVICE)
    args["attention_block_stride"] = BLOCK_SIZE
    _assert_raises(args, "attention_indices must be row-major int32 matching global_indices")


def test_hisparse_resolve_residency_rejects_attention_block_stride_too_small():
    args = _base_inputs()
    args["attention_indices"] = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    args["attention_block_stride"] = BLOCK_SIZE - 1
    _assert_raises(args, "attention block stride must cover one hot block")


# ---------------------------------------------------------------------------
# Optional arguments: all enabled
# ---------------------------------------------------------------------------

def test_hisparse_resolve_residency_all_optional_outputs_enabled():
    args = _base_inputs(rows=2, top_k=2, hot_size=4)
    args["miss_mask"] = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    args["stats"] = torch.zeros((2,), dtype=torch.uint64, device=DEVICE)
    args["attention_indices"] = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    args["attention_block_stride"] = BLOCK_SIZE
    args["resolved_global_indices"] = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    args["valid_counts"] = torch.zeros((2,), dtype=torch.int32, device=DEVICE)
    args["swap_host_physical_rows"] = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    args["swap_device_physical_rows"] = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    args["swap_counts"] = torch.zeros((2,), dtype=torch.int32, device=DEVICE)

    _invoke(args)
    torch.cuda.synchronize()


def test_hisparse_resolve_residency_with_request_source_and_resident_tables():
    args = _base_inputs(rows=2, top_k=2, hot_size=4)
    args["request_ids"] = torch.zeros((2,), dtype=torch.int32, device=DEVICE)
    args["source_block_table"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)
    args["resident_block_table"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)
    args["resident_block_size"] = BLOCK_SIZE
    args["resident_null_block"] = 0

    _invoke(args)
    torch.cuda.synchronize()


def test_hisparse_resolve_residency_with_request_source_table_without_resident_table():
    args = _base_inputs(rows=2, top_k=2, hot_size=4)
    args["request_ids"] = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    args["source_block_table"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)

    _invoke(args)
    torch.cuda.synchronize()


def test_hisparse_resolve_residency_with_all_optional_outputs_and_tables():
    args = _base_inputs(rows=2, top_k=2, hot_size=4)
    args["request_ids"] = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    args["source_block_table"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)
    args["resident_block_table"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)
    args["miss_mask"] = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    args["stats"] = torch.zeros((2,), dtype=torch.uint64, device=DEVICE)
    args["attention_indices"] = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    args["attention_block_stride"] = BLOCK_SIZE
    args["resolved_global_indices"] = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    args["valid_counts"] = torch.zeros((2,), dtype=torch.int32, device=DEVICE)
    args["swap_host_physical_rows"] = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    args["swap_device_physical_rows"] = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    args["swap_counts"] = torch.zeros((2,), dtype=torch.int32, device=DEVICE)

    _invoke(args)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Non-contiguous optional inputs
# ---------------------------------------------------------------------------

def test_hisparse_resolve_residency_rejects_noncontiguous_request_ids():
    args = _base_inputs()
    args["request_ids"] = torch.zeros((4,), dtype=torch.int32, device=DEVICE)[::2]
    args["source_block_table"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "request_ids must be contiguous int32 on CUDA with one entry per row")


def test_hisparse_resolve_residency_rejects_non_row_major_source_table():
    args = _base_inputs()
    args["request_ids"] = torch.zeros((2,), dtype=torch.int32, device=DEVICE)
    args["source_block_table"] = torch.zeros((4, 2), dtype=torch.int32, device=DEVICE).transpose(0, 1)
    _assert_raises(args, "source_block_table must be a row-major 2D int32 CUDA tensor")


def test_hisparse_resolve_residency_rejects_non_row_major_resident_table():
    args = _base_inputs()
    args["request_ids"] = torch.zeros((2,), dtype=torch.int32, device=DEVICE)
    args["source_block_table"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)
    args["resident_block_table"] = torch.zeros((4, 2), dtype=torch.int32, device=DEVICE).transpose(0, 1)
    _assert_raises(args, "resident_block_table must be a row-major 2D int32 CUDA tensor")


def test_hisparse_resolve_residency_rejects_noncontiguous_resolved_indices():
    args = _base_inputs()
    args["resolved_global_indices"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)[:, ::2]
    _assert_raises(args, "resolved_global_indices must be contiguous int32 matching indices")


def test_hisparse_resolve_residency_rejects_noncontiguous_miss_mask():
    args = _base_inputs()
    args["miss_mask"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)[:, ::2]
    _assert_raises(args, "miss_mask must be a contiguous int32 CUDA tensor matching")


def test_hisparse_resolve_residency_rejects_noncontiguous_stats():
    args = _base_inputs()
    args["stats"] = torch.zeros((4,), dtype=torch.uint64, device=DEVICE)[::2]
    _assert_raises(args, "stats must be a contiguous two-element uint64 CUDA tensor")


def test_hisparse_resolve_residency_rejects_noncontiguous_compact_swaps():
    args = _base_inputs()
    args["swap_host_physical_rows"] = torch.zeros((2, 4), dtype=torch.int32, device=DEVICE)[:, ::2]
    args["swap_device_physical_rows"] = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    args["swap_counts"] = torch.zeros((2,), dtype=torch.int32, device=DEVICE)
    _assert_raises(args, "compact swaps must be contiguous int32 matching indices")


# ---------------------------------------------------------------------------
# Additional shape and validation boundaries
# ---------------------------------------------------------------------------

def test_hisparse_resolve_residency_supports_hot_block_table_noncontiguous_columns():
    args = _base_inputs(rows=2, top_k=2, hot_size=8, block_size=BLOCK_SIZE)
    required_cols = (args["region_stride"] + BLOCK_SIZE - 1) // BLOCK_SIZE
    base = torch.zeros((2, required_cols * 2), dtype=torch.int32, device=DEVICE)
    args["hot_block_table"] = base[:, ::2]
    assert args["hot_block_table"].shape[1] >= required_cols
    assert args["hot_block_table"].stride(1) != 1
    _invoke(args)


def test_hisparse_resolve_residency_rejects_global_indices_wrong_dtype():
    args = _base_inputs()
    args["global_indices"] = args["global_indices"].to(torch.int64)
    _assert_raises(args, "global_indices/hot_indices must be int32")


def test_hisparse_resolve_residency_rejects_hot_indices_wrong_dtype():
    args = _base_inputs()
    args["hot_indices"] = args["hot_indices"].to(torch.int64)
    _assert_raises(args, "global_indices/hot_indices must be int32")


def test_hisparse_resolve_residency_rejects_lru_slots_wrong_dim():
    args = _base_inputs()
    args["lru_slots"] = torch.zeros((8,), dtype=torch.int16, device=DEVICE)
    _assert_raises(args, "lru_slots must be contiguous with one row per request state")


def test_hisparse_resolve_residency_rejects_hot_cache_wrong_dim():
    args = _base_inputs()
    args["hot_cache"] = torch.zeros((2, BLOCK_SIZE * ROW_BYTES), dtype=torch.uint8, device=DEVICE)
    _assert_raises(args, "hot_cache must be [num_blocks, block_size, row_width]")


def test_hisparse_resolve_residency_rejects_attention_indices_non_row_major():
    args = _base_inputs()
    args["attention_indices"] = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE).transpose(0, 1)
    args["attention_block_stride"] = BLOCK_SIZE
    _assert_raises(args, "attention_indices must be row-major int32 matching global_indices")


def test_hisparse_resolve_residency_accepts_attention_block_stride_equal_to_block_size():
    args = _base_inputs()
    args["attention_indices"] = torch.zeros((2, 2), dtype=torch.int32, device=DEVICE)
    args["attention_block_stride"] = BLOCK_SIZE
    _invoke(args)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Output initialization and repeated invocation
# ---------------------------------------------------------------------------

def test_hisparse_resolve_residency_optional_outputs_have_expected_shapes():
    args = _base_inputs(rows=2, top_k=2, hot_size=4)
    args["miss_mask"] = torch.full((2, 2), -1, dtype=torch.int32, device=DEVICE)
    args["stats"] = torch.zeros((2,), dtype=torch.uint64, device=DEVICE)
    args["resolved_global_indices"] = torch.full((2, 2), -1, dtype=torch.int32, device=DEVICE)
    args["valid_counts"] = torch.full((2,), -1, dtype=torch.int32, device=DEVICE)
    args["swap_host_physical_rows"] = torch.full((2, 2), -1, dtype=torch.int32, device=DEVICE)
    args["swap_device_physical_rows"] = torch.full((2, 2), -1, dtype=torch.int32, device=DEVICE)
    args["swap_counts"] = torch.full((2,), -1, dtype=torch.int32, device=DEVICE)

    _invoke(args)
    torch.cuda.synchronize()

    assert args["miss_mask"].shape == (2, 2)
    assert args["stats"].shape == (2,)
    assert args["resolved_global_indices"].shape == (2, 2)
    assert args["valid_counts"].shape == (2,)
    assert args["swap_host_physical_rows"].shape == (2, 2)
    assert args["swap_device_physical_rows"].shape == (2, 2)
    assert args["swap_counts"].shape == (2,)


def test_hisparse_resolve_residency_repeated_invocation():
    args = _base_inputs(rows=2, top_k=2, hot_size=4)
    for _ in range(3):
        _invoke(args)
    torch.cuda.synchronize()