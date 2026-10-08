# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

try:
    import mcoplib._C
except ImportError:
    pass


OP = torch.ops._C_cache_ops.cp_gather_and_upconvert_fp8_kv_cache


def _make_src_cache(num_blocks, block_size, device, fp8_value=0x00):
    src = torch.empty((num_blocks, block_size, 656), dtype=torch.uint8, device=device)

    # 512 bytes FP8 raw storage.
    src[:, :, :512].fill_(fp8_value)

    # 4 x float32 scale = 16 bytes.
    scales = torch.ones((num_blocks, block_size, 4), dtype=torch.float32, device=device)
    src[:, :, 512:528].copy_(scales.view(torch.uint8))

    # 128 bytes rope = 32 x int32.
    rope = torch.arange(num_blocks * block_size * 32, dtype=torch.int32, device=device).view(num_blocks, block_size, 32)
    src[:, :, 528:656].copy_(rope.view(torch.uint8))

    return src


def _make_block_table(batch_size, blocks_per_req, device):
    num_blocks = batch_size * blocks_per_req
    physical_blocks = torch.arange(num_blocks, dtype=torch.int32, device=device).view(batch_size, blocks_per_req)
    block_table = torch.empty_like(physical_blocks)

    for req_id in range(batch_size):
        block_table[req_id].copy_(physical_blocks[req_id])

    return block_table


def _make_reversed_block_table(batch_size, blocks_per_req, device):
    num_blocks = batch_size * blocks_per_req
    physical_blocks = torch.arange(num_blocks, dtype=torch.int32, device=device).view(batch_size, blocks_per_req)
    block_table = torch.empty_like(physical_blocks)

    for req_id in range(batch_size):
        block_table[req_id].copy_(physical_blocks[req_id].flip(0))

    return block_table


def _build_rope_reference(src_cache, block_table, workspace_starts, batch_size, block_size, total_tokens, seq_starts=None):
    device = src_cache.device
    dst_ref = torch.zeros((total_tokens, 576), dtype=torch.bfloat16, device=device)

    for req_id in range(batch_size):
        output_begin = min(int(workspace_starts[req_id].item()), total_tokens)

        if req_id + 1 < batch_size:
            output_end = min(int(workspace_starts[req_id + 1].item()), total_tokens)
        else:
            output_end = total_tokens

        seq_len = max(output_end - output_begin, 0)

        if seq_starts is None:
            source_begin = 0
        else:
            source_begin = int(seq_starts[req_id].item())

        for token in range(seq_len):
            source_pos = source_begin + token
            logical_block = source_pos // block_size
            block_offset = source_pos % block_size
            physical_block = int(block_table[req_id, logical_block].item())

            src_token = src_cache[physical_block, block_offset]

            rope_src = src_token[528:656].view(torch.int32)
            rope_dst = dst_ref[output_begin + token, 512:576].view(torch.int32)

            rope_dst.copy_(rope_src)

    return dst_ref


def _run_case(batch_size, block_size, seq_lens, source_starts, use_seq_starts=False, reverse_block_table=False):
    assert torch.cuda.is_available()

    device = torch.device("cuda:0")

    assert len(seq_lens) == batch_size
    assert len(source_starts) == batch_size

    workspace_starts = [0]

    for seq_len in seq_lens[:-1]:
        workspace_starts.append(workspace_starts[-1] + seq_len)

    total_tokens = sum(seq_lens)
    assert total_tokens > 0

    max_source_end = 0

    for seq_len, source_start in zip(seq_lens, source_starts):
        max_source_end = max(max_source_end, source_start + seq_len)

    blocks_per_req = (max_source_end + block_size - 1) // block_size
    assert blocks_per_req > 0

    num_blocks = batch_size * blocks_per_req

    src_cache = _make_src_cache(num_blocks, block_size, device)

    if reverse_block_table:
        block_table = _make_reversed_block_table(batch_size, blocks_per_req, device)
    else:
        block_table = _make_block_table(batch_size, blocks_per_req, device)

    workspace_starts_tensor = torch.tensor(workspace_starts, dtype=torch.int32, device=device)

    seq_starts_tensor = None

    if use_seq_starts:
        seq_starts_tensor = torch.tensor(source_starts, dtype=torch.int32, device=device)

    dst = torch.zeros((total_tokens, 576), dtype=torch.bfloat16, device=device)

    OP(src_cache, dst, block_table, workspace_starts_tensor, batch_size, seq_starts_tensor)

    torch.cuda.synchronize()

    fp8_output = dst[:, :512]

    assert torch.isfinite(fp8_output.float()).all().item(), "kernel produced NaN/Inf in FP8 output"

    assert torch.equal(fp8_output, torch.zeros_like(fp8_output))

    dst_ref = _build_rope_reference(src_cache, block_table, workspace_starts_tensor, batch_size, block_size, total_tokens, seq_starts_tensor)

    assert torch.equal(dst[:, 512:], dst_ref[:, 512:])


# ================================================================
# Basic
# ================================================================


def test_cp_gather_and_upconvert_fp8_kv_cache_zero():
    _run_case(batch_size=1, block_size=16, seq_lens=[16], source_starts=[0])


def test_cp_gather_and_upconvert_fp8_kv_cache_single_partial_block():
    _run_case(batch_size=1, block_size=16, seq_lens=[7], source_starts=[0])


def test_cp_gather_and_upconvert_fp8_kv_cache_single_cross_block():
    _run_case(batch_size=1, block_size=16, seq_lens=[17], source_starts=[0])


def test_cp_gather_and_upconvert_fp8_kv_cache_multiple_blocks():
    _run_case(batch_size=1, block_size=16, seq_lens=[33], source_starts=[0])


# ================================================================
# Multi request
# ================================================================


def test_cp_gather_and_upconvert_fp8_kv_cache_multi_request():
    _run_case(batch_size=4, block_size=16, seq_lens=[16, 16, 16, 16], source_starts=[0, 0, 0, 0])


def test_cp_gather_and_upconvert_fp8_kv_cache_multi_request_different_lengths():
    _run_case(batch_size=4, block_size=16, seq_lens=[7, 16, 17, 33], source_starts=[0, 0, 0, 0])


def test_cp_gather_and_upconvert_fp8_kv_cache_multi_request_all_partial():
    _run_case(batch_size=4, block_size=16, seq_lens=[3, 7, 11, 15], source_starts=[0, 0, 0, 0])


# ================================================================
# Cross block
# ================================================================


def test_cp_gather_and_upconvert_fp8_kv_cache_cross_block_17():
    _run_case(batch_size=1, block_size=16, seq_lens=[17], source_starts=[0])


def test_cp_gather_and_upconvert_fp8_kv_cache_cross_block_31():
    _run_case(batch_size=1, block_size=16, seq_lens=[31], source_starts=[0])


def test_cp_gather_and_upconvert_fp8_kv_cache_cross_block_32():
    _run_case(batch_size=1, block_size=16, seq_lens=[32], source_starts=[0])


def test_cp_gather_and_upconvert_fp8_kv_cache_cross_block_33():
    _run_case(batch_size=1, block_size=16, seq_lens=[33], source_starts=[0])


# ================================================================
# seq_starts
# ================================================================


def test_cp_gather_and_upconvert_fp8_kv_cache_seq_start_aligned():
    _run_case(batch_size=1, block_size=16, seq_lens=[16], source_starts=[16], use_seq_starts=True)


def test_cp_gather_and_upconvert_fp8_kv_cache_seq_start_two_blocks():
    _run_case(batch_size=1, block_size=16, seq_lens=[17], source_starts=[32], use_seq_starts=True)


def test_cp_gather_and_upconvert_fp8_kv_cache_seq_start_unaligned():
    _run_case(batch_size=1, block_size=16, seq_lens=[16], source_starts=[3], use_seq_starts=True)


def test_cp_gather_and_upconvert_fp8_kv_cache_seq_start_unaligned_cross_block():
    _run_case(batch_size=1, block_size=16, seq_lens=[20], source_starts=[7], use_seq_starts=True)


def test_cp_gather_and_upconvert_fp8_kv_cache_multi_request_seq_starts():
    _run_case(batch_size=4, block_size=16, seq_lens=[7, 16, 17, 23], source_starts=[3, 16, 31, 47], use_seq_starts=True)


# ================================================================
# Non-trivial block table
# ================================================================


def test_cp_gather_and_upconvert_fp8_kv_cache_reversed_block_table():
    _run_case(batch_size=1, block_size=16, seq_lens=[33], source_starts=[0], reverse_block_table=True)


def test_cp_gather_and_upconvert_fp8_kv_cache_multi_request_reversed_block_table():
    _run_case(batch_size=4, block_size=16, seq_lens=[17, 31, 16, 33], source_starts=[0, 0, 0, 0], reverse_block_table=True)


def test_cp_gather_and_upconvert_fp8_kv_cache_seq_start_reversed_block_table():
    _run_case(batch_size=3, block_size=16, seq_lens=[17, 20, 33], source_starts=[3, 17, 35], use_seq_starts=True, reverse_block_table=True)


# ================================================================
# Different block sizes
# ================================================================


def test_cp_gather_and_upconvert_fp8_kv_cache_block_size_8():
    _run_case(batch_size=2, block_size=8, seq_lens=[9, 17], source_starts=[0, 0])


def test_cp_gather_and_upconvert_fp8_kv_cache_block_size_16():
    _run_case(batch_size=2, block_size=16, seq_lens=[17, 33], source_starts=[0, 0])


def test_cp_gather_and_upconvert_fp8_kv_cache_block_size_32():
    _run_case(batch_size=2, block_size=32, seq_lens=[33, 65], source_starts=[0, 0])


# ================================================================
# Large mapping
# ================================================================


def test_cp_gather_and_upconvert_fp8_kv_cache_large_request():
    _run_case(
        batch_size=8,
        block_size=16,
        seq_lens=[17, 31, 33, 7, 16, 24, 48, 65],
        source_starts=[0, 0, 0, 0, 0, 0, 0, 0],
    )


def test_cp_gather_and_upconvert_fp8_kv_cache_large_request_seq_starts():
    _run_case(
        batch_size=8,
        block_size=16,
        seq_lens=[7, 17, 23, 31, 16, 33, 41, 48],
        source_starts=[3, 16, 7, 32, 48, 5, 64, 17],
        use_seq_starts=True,
        reverse_block_table=True,
    )