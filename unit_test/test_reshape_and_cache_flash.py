# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import mcoplib._C


RESHAPE_AND_CACHE_SCENARIOS = [
    (1, 1, 16, 16),
    (1, 1, 32, 16),
    (1, 1, 64, 16),
    (1, 1, 128, 16),
    (1, 4, 64, 16),
    (1, 8, 128, 16),
    (2, 8, 128, 16),
    (4, 8, 128, 16),
    (8, 8, 128, 16),
    (16, 8, 128, 16),
    (32, 8, 128, 16),
    (64, 8, 128, 16),
    (128, 8, 128, 16),
    (256, 8, 128, 16),
    (512, 8, 128, 16),
    (1024, 8, 128, 16),
    (128, 16, 128, 16),
    (128, 32, 128, 16),
    (128, 32, 256, 16),
]


DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]


BLOCK_SIZE_SCENARIOS = [
    (8, 8),
    (16, 8),
    (32, 8),
    (64, 8),
    (16, 16),
    (32, 16),
    (64, 16),
    (32, 32),
]


SLOT_MAPPING_SCENARIOS = [
    "sequential",
    "random",
    "same_block",
    "cross_block",
]


def _reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype="auto", k_scale=None, v_scale=None):
    return torch.ops._C_cache_ops.reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale)


def _make_inputs(num_tokens, num_heads, head_size, block_size, num_blocks, dtype, slot_mapping_type="random", seed=1234):
    torch.manual_seed(seed)

    device = "cuda"

    key = torch.randn((num_tokens, num_heads, head_size), dtype=dtype, device=device)
    value = torch.randn((num_tokens, num_heads, head_size), dtype=dtype, device=device)

    cache_shape = (num_blocks, block_size, num_heads, head_size)

    key_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
    value_cache = torch.zeros(cache_shape, dtype=dtype, device=device)

    total_slots = num_blocks * block_size

    if total_slots < num_tokens:
        raise ValueError(f"Cache size ({total_slots}) is smaller than num_tokens ({num_tokens})")

    if slot_mapping_type == "sequential":
        slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=device)

    elif slot_mapping_type == "random":
        slot_mapping = torch.randperm(total_slots, dtype=torch.long, device=device)[:num_tokens]

    elif slot_mapping_type == "same_block":
        if num_tokens > block_size:
            raise ValueError(f"same_block requires num_tokens <= block_size, got num_tokens={num_tokens}, block_size={block_size}")
        block_id = num_blocks // 2
        offset = torch.arange(num_tokens, dtype=torch.long, device=device)
        slot_mapping = block_id * block_size + offset

    elif slot_mapping_type == "cross_block":
        slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=device)

        if num_tokens > block_size:
            slot_mapping = slot_mapping + block_size

    else:
        raise ValueError(f"Unsupported slot_mapping_type: {slot_mapping_type}")

    k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    return key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale


def _reference_reconstruct(key_cache, value_cache, slot_mapping, block_size):
    slot_mapping_long = slot_mapping.long()

    block_indices = slot_mapping_long // block_size
    block_offsets = slot_mapping_long % block_size

    key_recon = key_cache[block_indices, block_offsets]
    value_recon = value_cache[block_indices, block_offsets]

    return key_recon, value_recon


def _check_tensor_close(actual, expected, name, rtol=8e-3, atol=8e-3):
    assert actual.shape == expected.shape, f"{name} shape mismatch: actual={actual.shape}, expected={expected.shape}"

    assert actual.dtype == expected.dtype, f"{name} dtype mismatch: actual={actual.dtype}, expected={expected.dtype}"

    assert torch.isfinite(actual).all(), f"{name} contains non-finite values"

    assert torch.isfinite(expected).all(), f"{name} reference contains non-finite values"

    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError:
        diff = (actual.float() - expected.float()).abs()

        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        max_index = torch.argmax(diff.reshape(-1)).item()
        max_index = torch.unravel_index(max_index, diff.shape)

        print()
        print(f"❌ {name} mismatch")
        print(f"max_abs_diff={max_diff}")
        print(f"mean_abs_diff={mean_diff}")
        print(f"max_diff_index={tuple(x.item() for x in max_index)}")
        print(f"actual={actual[max_index].item()}")
        print(f"expected={expected[max_index].item()}")

        raise


def _check_cache_untouched(actual_cache, original_cache, name):
    torch.testing.assert_close(actual_cache, original_cache, rtol=0.0, atol=0.0)


def _run_case(num_tokens, num_heads, head_size, block_size, num_blocks=128, dtype=torch.float16, kv_cache_dtype="auto", slot_mapping_type="random", rtol=8e-3, atol=8e-3):
    key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale = _make_inputs(num_tokens, num_heads, head_size, block_size, num_blocks, dtype, slot_mapping_type)

    original_key_cache = key_cache.clone()
    original_value_cache = value_cache.clone()

    _reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale)

    torch.cuda.synchronize()

    key_recon, value_recon = _reference_reconstruct(key_cache, value_cache, slot_mapping, block_size)

    _check_tensor_close(key_recon.to(key.dtype), key, "key", rtol, atol)
    _check_tensor_close(value_recon.to(value.dtype), value, "value", rtol, atol)

    return key, value, key_cache, value_cache, slot_mapping


# ============================================================
# Basic scenarios
# ============================================================

@pytest.mark.parametrize("num_tokens,num_heads,head_size,block_size", RESHAPE_AND_CACHE_SCENARIOS)
def test_reshape_and_cache_flash_basic(num_tokens, num_heads, head_size, block_size):
    _run_case(num_tokens, num_heads, head_size, block_size, num_blocks=128, dtype=torch.float16)


# ============================================================
# Dtype
# ============================================================

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_tokens,num_heads,head_size,block_size", [
    (1, 8, 128, 16),
    (8, 8, 128, 16),
    (32, 8, 128, 16),
    (128, 8, 128, 16),
])
def test_reshape_and_cache_flash_dtype(num_tokens, num_heads, head_size, block_size, dtype):
    _run_case(num_tokens, num_heads, head_size, block_size, num_blocks=128, dtype=dtype)


# ============================================================
# Block size
# ============================================================

@pytest.mark.parametrize("block_size,num_tokens", BLOCK_SIZE_SCENARIOS)
def test_reshape_and_cache_flash_block_size(block_size, num_tokens):
    _run_case(num_tokens, 8, 128, block_size, num_blocks=128, dtype=torch.float16)


# ============================================================
# Slot mapping
# ============================================================

@pytest.mark.parametrize("slot_mapping_type", SLOT_MAPPING_SCENARIOS)
def test_reshape_and_cache_flash_slot_mapping(slot_mapping_type):
    if slot_mapping_type == "same_block":
        num_tokens = 8
        block_size = 16
    else:
        num_tokens = 128
        block_size = 16

    _run_case(num_tokens, 8, 128, block_size, num_blocks=128, dtype=torch.float16, slot_mapping_type=slot_mapping_type)


# ============================================================
# Small token boundaries
# ============================================================

@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33])
def test_reshape_and_cache_flash_small_token_boundaries(num_tokens):
    _run_case(num_tokens, 8, 128, 16, num_blocks=128, dtype=torch.float16, slot_mapping_type="sequential")


# ============================================================
# Block boundaries
# ============================================================

@pytest.mark.parametrize("num_tokens", [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129])
def test_reshape_and_cache_flash_block_boundaries(num_tokens):
    _run_case(num_tokens, 8, 128, 16, num_blocks=256, dtype=torch.float16, slot_mapping_type="sequential")


# ============================================================
# Head size
# ============================================================

@pytest.mark.parametrize("head_size", [16, 32, 64, 80, 96, 128, 160, 192, 256])
def test_reshape_and_cache_flash_head_size(head_size):
    _run_case(32, 8, head_size, 16, num_blocks=128, dtype=torch.float16)


# ============================================================
# Number of heads
# ============================================================

@pytest.mark.parametrize("num_heads", [1, 2, 4, 8, 16, 32, 64])
def test_reshape_and_cache_flash_num_heads(num_heads):
    _run_case(32, num_heads, 128, 16, num_blocks=128, dtype=torch.float16)


# ============================================================
# Large tokens
# ============================================================

@pytest.mark.parametrize("num_tokens", [256, 512, 1024, 2048])
def test_reshape_and_cache_flash_large_tokens(num_tokens):
    _run_case(num_tokens, 8, 128, 16, num_blocks=256, dtype=torch.float16, slot_mapping_type="sequential")


# ============================================================
# Large head dimension
# ============================================================

@pytest.mark.parametrize("num_tokens,num_heads,head_size", [
    (32, 8, 256),
    (32, 16, 256),
    (64, 8, 256),
    (128, 8, 256),
    (128, 16, 256),
])
def test_reshape_and_cache_flash_large_head_size(num_tokens, num_heads, head_size):
    _run_case(num_tokens, num_heads, head_size, 16, num_blocks=256, dtype=torch.float16)


# ============================================================
# Random slot mapping
# ============================================================

@pytest.mark.parametrize("seed", [1, 2, 3, 1234, 5678, 9999])
def test_reshape_and_cache_flash_random_slot_mapping(seed):
    num_tokens = 128
    num_heads = 8
    head_size = 128
    block_size = 16
    num_blocks = 256

    key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale = _make_inputs(num_tokens, num_heads, head_size, block_size, num_blocks, torch.float16, "random", seed)

    _reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, "auto", k_scale, v_scale)

    torch.cuda.synchronize()

    key_recon, value_recon = _reference_reconstruct(key_cache, value_cache, slot_mapping, block_size)

    _check_tensor_close(key_recon, key, "key", 8e-3, 8e-3)
    _check_tensor_close(value_recon, value, "value", 8e-3, 8e-3)


# ============================================================
# Sequential slot mapping
# ============================================================

@pytest.mark.parametrize("num_tokens", [1, 8, 16, 32, 64, 128, 256])
def test_reshape_and_cache_flash_sequential_slot_mapping(num_tokens):
    _run_case(num_tokens, 8, 128, 16, num_blocks=256, dtype=torch.float16, slot_mapping_type="sequential")


# ============================================================
# Cross block slot mapping
# ============================================================

@pytest.mark.parametrize("num_tokens", [16, 17, 32, 33, 64, 65, 128])
def test_reshape_and_cache_flash_cross_block_slot_mapping(num_tokens):
    _run_case(num_tokens, 8, 128, 16, num_blocks=256, dtype=torch.float16, slot_mapping_type="cross_block")


# ============================================================
# Cache larger than input
# ============================================================

@pytest.mark.parametrize("num_blocks", [16, 32, 64, 128, 256, 512])
def test_reshape_and_cache_flash_cache_capacity(num_blocks):
    num_tokens = 128
    block_size = 16

    if num_blocks * block_size < num_tokens:
        pytest.skip("cache capacity is smaller than num_tokens")

    _run_case(num_tokens, 8, 128, block_size, num_blocks=num_blocks, dtype=torch.float16)


# ============================================================
# KV cache dtype auto
# ============================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_reshape_and_cache_flash_kv_cache_dtype_auto(dtype):
    _run_case(32, 8, 128, 16, num_blocks=128, dtype=dtype, kv_cache_dtype="auto")


# ============================================================
# Key / value independently validated
# ============================================================

def test_reshape_and_cache_flash_key_value_independent():
    num_tokens = 64
    num_heads = 8
    head_size = 128
    block_size = 16
    num_blocks = 128

    key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale = _make_inputs(num_tokens, num_heads, head_size, block_size, num_blocks, torch.float16, "random")

    _reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, "auto", k_scale, v_scale)

    torch.cuda.synchronize()

    key_recon, value_recon = _reference_reconstruct(key_cache, value_cache, slot_mapping, block_size)

    _check_tensor_close(key_recon, key, "key", 8e-3, 8e-3)
    _check_tensor_close(value_recon, value, "value", 8e-3, 8e-3)


# ============================================================
# API
# ============================================================

def test_reshape_and_cache_flash_api():
    num_tokens = 8
    num_heads = 8
    head_size = 128
    block_size = 16
    num_blocks = 16

    key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale = _make_inputs(num_tokens, num_heads, head_size, block_size, num_blocks, torch.float16, "sequential")

    _reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, "auto", k_scale, v_scale)

    torch.cuda.synchronize()

    assert key.shape == (num_tokens, num_heads, head_size)
    assert value.shape == (num_tokens, num_heads, head_size)
    assert key_cache.shape == (num_blocks, block_size, num_heads, head_size)
    assert value_cache.shape == (num_blocks, block_size, num_heads, head_size)

    assert key.dtype == torch.float16
    assert value.dtype == torch.float16
    assert key_cache.dtype == torch.float16
    assert value_cache.dtype == torch.float16

    assert slot_mapping.dtype == torch.int64

    assert torch.isfinite(key_cache).all()
    assert torch.isfinite(value_cache).all()