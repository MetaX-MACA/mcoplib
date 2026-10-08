# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import mcoplib._C


DEVICE = "cuda:0"
BATCH_SIZE = 64
BLOCK_SIZE = 16
NUM_KV_HEADS = 1
HEAD_SIZE = 576
SEQ_LEN = 1024
NUM_BLOCKS = BATCH_SIZE * ((SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE)
TOTAL_TOKENS = BATCH_SIZE * SEQ_LEN
DTYPE = torch.float16
KV_CACHE_DTYPE = "auto"


def _op_available():
    return hasattr(torch.ops._C_cache_ops, "gather_and_maybe_dequant_cache")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _op_available(), reason="gather_and_maybe_dequant_cache is not available")
@torch.inference_mode()
def test_gather_and_maybe_dequant_cache_config():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(DEVICE)

    blocks_per_seq = (SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE

    src_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
    dst = torch.zeros(TOTAL_TOKENS, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)

    block_table = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=DEVICE).view(BATCH_SIZE, blocks_per_seq)
    cu_seq_lens = torch.arange(0, (BATCH_SIZE + 1) * SEQ_LEN, step=SEQ_LEN, dtype=torch.int32, device=DEVICE)
    token_to_seq = torch.arange(BATCH_SIZE, dtype=torch.int32, device=DEVICE).repeat_interleave(SEQ_LEN)
    scale = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)

    expected = torch.empty_like(dst)
    src_ref = src_cache.float()

    for b in range(BATCH_SIZE):
        seq_start = b * SEQ_LEN

        for t in range(SEQ_LEN):
            block_idx = t // BLOCK_SIZE
            block_off = t % BLOCK_SIZE
            physical_block_id = int(block_table[b, block_idx].item())
            expected[seq_start + t, 0] = src_ref[physical_block_id, block_off].to(DTYPE)

    torch.ops._C_cache_ops.gather_and_maybe_dequant_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        TOTAL_TOKENS,
        KV_CACHE_DTYPE,
        scale,
        None,
    )

    torch.cuda.synchronize(DEVICE)

    print("\n" + "=" * 80)
    print("gather_and_maybe_dequant_cache")
    print("=" * 80)
    print(f"device         : {DEVICE}")
    print(f"batch_size     : {BATCH_SIZE}")
    print(f"block_size     : {BLOCK_SIZE}")
    print(f"num_kv_heads   : {NUM_KV_HEADS}")
    print(f"head_size      : {HEAD_SIZE}")
    print(f"seq_len        : {SEQ_LEN}")
    print(f"num_blocks     : {NUM_BLOCKS}")
    print(f"total_tokens   : {TOTAL_TOKENS}")
    print(f"dtype          : {DTYPE}")
    print(f"kv_cache_dtype : {KV_CACHE_DTYPE}")
    print(f"src_cache.shape: {tuple(src_cache.shape)}")
    print(f"dst.shape      : {tuple(dst.shape)}")

    assert src_cache.shape == (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    assert dst.shape == (TOTAL_TOKENS, NUM_KV_HEADS, HEAD_SIZE)
    assert src_cache.dtype == DTYPE
    assert dst.dtype == DTYPE
    assert block_table.dtype == torch.int32
    assert cu_seq_lens.dtype == torch.int32
    assert token_to_seq.dtype == torch.int32
    assert torch.isfinite(src_cache).all()
    assert torch.isfinite(dst).all()

    diff = (dst.float() - expected.float()).abs()

    print("\n[Verification]")
    print(f"max diff       : {diff.max().item()}")
    print(f"mean diff      : {diff.mean().item()}")

    torch.testing.assert_close(dst, expected, atol=0, rtol=0)

    print("PASS")