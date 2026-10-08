import pytest
import torch

try:
    import mcoplib._C
except ImportError:
    pass


def _cp_gather_cache_available():
    return hasattr(torch.ops._C_cache_ops, "cp_gather_cache")


def _run_cp_gather_cache(batch_size, block_size, num_heads, head_size, seq_len, dtype, device):
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = batch_size * blocks_per_seq
    total_tokens = batch_size * seq_len

    src_shape = (num_blocks, block_size, num_heads, head_size)
    dst_shape = (total_tokens, num_heads, head_size)

    src_cache = torch.randn(src_shape, dtype=dtype, device=device)
    dst = torch.zeros(dst_shape, dtype=dtype, device=device)

    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(batch_size, blocks_per_seq)
    cu_seq_lens = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=device)

    torch.ops._C_cache_ops.cp_gather_cache(src_cache, dst, block_table, cu_seq_lens, batch_size, None)

    return src_cache, dst, block_table, cu_seq_lens


def _build_reference(src_cache, block_table, cu_seq_lens, batch_size, block_size, device):
    src_cpu = src_cache.cpu()
    table_cpu = block_table.cpu()
    cu_lens_cpu = cu_seq_lens.cpu()

    total_tokens = cu_lens_cpu[-1].item()
    dst_ref = torch.zeros((total_tokens, src_cache.shape[2], src_cache.shape[3]), dtype=src_cache.dtype)

    for b in range(batch_size):
        seq_start = cu_lens_cpu[b].item()
        seq_end = cu_lens_cpu[b + 1].item()
        cur_len = seq_end - seq_start

        for t in range(cur_len):
            block_idx = t // block_size
            block_off = t % block_size
            phys_block = table_cpu[b, block_idx].item()
            dst_ref[seq_start + t] = src_cpu[phys_block, block_off]

    return dst_ref.to(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not _cp_gather_cache_available(), reason="cp_gather_cache is not available")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size,seq_len", [(1, 1), (1, 16), (2, 32), (4, 128), (8, 256)])
def test_cp_gather_cache(batch_size, seq_len, dtype):
    device = "cuda:0"
    block_size = 16
    num_heads = 32
    head_size = 128

    src_cache, dst, block_table, cu_seq_lens = _run_cp_gather_cache(
        batch_size,
        block_size,
        num_heads,
        head_size,
        seq_len,
        dtype,
        device,
    )

    dst_ref = _build_reference(
        src_cache,
        block_table,
        cu_seq_lens,
        batch_size,
        block_size,
        device,
    )

    assert dst.shape == dst_ref.shape
    assert dst.dtype == dst_ref.dtype
    assert torch.equal(dst, dst_ref), (
        f"cp_gather_cache result mismatch: "
        f"max_abs_diff={(dst.float() - dst_ref.float()).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not _cp_gather_cache_available(), reason="cp_gather_cache is not available")
@pytest.mark.parametrize("batch_size,seq_len", [(2, 17), (4, 33), (8, 65)])
def test_cp_gather_cache_non_aligned_sequence(batch_size, seq_len):
    device = "cuda:0"
    block_size = 16
    num_heads = 8
    head_size = 64
    dtype = torch.float16

    src_cache, dst, block_table, cu_seq_lens = _run_cp_gather_cache(
        batch_size,
        block_size,
        num_heads,
        head_size,
        seq_len,
        dtype,
        device,
    )

    dst_ref = _build_reference(
        src_cache,
        block_table,
        cu_seq_lens,
        batch_size,
        block_size,
        device,
    )

    assert dst.shape == dst_ref.shape
    assert torch.equal(dst, dst_ref), (
        f"non-aligned cp_gather_cache mismatch: "
        f"max_abs_diff={(dst.float() - dst_ref.float()).abs().max().item()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not _cp_gather_cache_available(), reason="cp_gather_cache is not available")
def test_cp_gather_cache_block_mapping():
    device = "cuda:0"
    batch_size = 2
    block_size = 16
    num_heads = 4
    head_size = 32
    seq_len = 32
    dtype = torch.float16

    blocks_per_seq = 2
    num_blocks = 4
    total_tokens = batch_size * seq_len

    src_cache = torch.empty(
        (num_blocks, block_size, num_heads, head_size),
        dtype=dtype,
        device=device,
    )

    for block_idx in range(num_blocks):
        src_cache[block_idx].fill_(float(block_idx + 1))

    dst = torch.zeros(
        (total_tokens, num_heads, head_size),
        dtype=dtype,
        device=device,
    )

    block_table = torch.tensor(
        [
            [2, 0],
            [3, 1],
        ],
        dtype=torch.int32,
        device=device,
    )

    cu_seq_lens = torch.tensor(
        [0, 32, 64],
        dtype=torch.int32,
        device=device,
    )

    torch.ops._C_cache_ops.cp_gather_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        batch_size,
        None,
    )

    expected = torch.empty_like(dst)
    expected[:16].fill_(3)
    expected[16:32].fill_(1)
    expected[32:48].fill_(4)
    expected[48:64].fill_(2)

    assert torch.equal(dst, expected), (
        f"block mapping mismatch: "
        f"max_abs_diff={(dst.float() - expected.float()).abs().max().item()}"
    )