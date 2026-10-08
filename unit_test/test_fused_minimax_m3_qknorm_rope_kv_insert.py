# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit test for the horizontally-fused MiniMax-M3 attention pre-processing
kernel.

fused_minimax_m3_qknorm_rope_kv_insert
    - q / k / index_q / index_k: Gemma RMSNorm + partial NeoX RoPE
    - sparse mode: scatter k/v into paged KV cache and index key into index cache
    - skip_index_branch=True: only q/k/v path is executed, index branch is skipped
"""

import pytest
import torch
import vllm._custom_ops as ops
import mcoplib._C


HEAD_DIM = 128
ROTARY_DIM = 64


def _op_available() -> bool:
    return hasattr(torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert")


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _op_available(),
    reason="CUDA not available or fused MiniMax-M3 op not built in",
)


def make_cos_sin_cache(max_pos, rotary_dim, base, dtype, device):
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache.to(dtype)


def gemma_rmsnorm(x, weight, eps):
    """x: [..., 128]; weight: [128]. Returns original dtype."""
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    out = out * (1.0 + weight.float())
    return out.to(x.dtype)


def apply_rope_neox_partial(x, positions, cos_sin_cache, rotary_dim):
    """NeoX-style RoPE on the leading rotary_dim dims."""
    half = rotary_dim // 2
    cs = cos_sin_cache[positions].float()
    cos = cs[..., :half].unsqueeze(1)
    sin = cs[..., half:].unsqueeze(1)

    rot = x[..., :rotary_dim].float()
    x1 = rot[..., :half]
    x2 = rot[..., half:]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    out = x.clone()
    out[..., :half] = o1
    out[..., half:rotary_dim] = o2
    return out.to(x.dtype)


def norm_rope_ref(x, weight, positions, cos_sin_cache, eps):
    normed = gemma_rmsnorm(x, weight, eps)
    return apply_rope_neox_partial(normed, positions, cos_sin_cache, ROTARY_DIM)


# ---------------------------------------------------------------------------
# Test 1: dense mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_tokens", [1, 7, 64, 513])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 2), (16, 4), (64, 4)])
def test_dense_norm_rope(num_tokens, num_heads, num_kv_heads):
    torch.manual_seed(0)
    device, dtype, eps = "cuda", torch.bfloat16, 1e-6
    base, max_pos = 5_000_000.0, 4096

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, dtype, device)
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64, device=device)

    qsz, kvsz = num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    qkv = torch.randn(num_tokens, qsz + 2 * kvsz, dtype=dtype, device=device)
    qkv_orig = qkv.clone()

    # Dense mode: skip_index_branch=False
    torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv, q_w, k_w, cos_sin, positions, num_heads, num_kv_heads, ROTARY_DIM, eps,
        None, None, 0, None, None, None, None, 0, None, None, "auto", False
    )

    q_out, k_out, v_out = qkv.split([qsz, kvsz, kvsz], dim=-1)
    q_in, k_in, v_in = qkv_orig.split([qsz, kvsz, kvsz], dim=-1)

    q_ref = norm_rope_ref(q_in.view(num_tokens, num_heads, HEAD_DIM), q_w, positions, cos_sin, eps).view(num_tokens, qsz)
    k_ref = norm_rope_ref(k_in.view(num_tokens, num_kv_heads, HEAD_DIM), k_w, positions, cos_sin, eps).view(num_tokens, kvsz)

    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_out, v_in, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Test 2: sparse mode - full index branch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_tokens", [1, 7, 64, 513])
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_sparse_full(num_tokens, block_size, kv_cache_dtype):
    torch.manual_seed(1)
    device, dtype, eps = "cuda", torch.bfloat16, 1e-6
    base, max_pos = 5_000_000.0, 4096
    num_heads, num_kv_heads, num_idx_heads = 16, 4, 4

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    iq_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    ik_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, dtype, device)
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64, device=device)

    qsz, kvsz = num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    iqsz, iksz = num_idx_heads * HEAD_DIM, HEAD_DIM

    qkv = torch.randn(num_tokens, qsz + 2 * kvsz + iqsz + iksz, dtype=dtype, device=device)
    qkv_orig = qkv.clone()
    splits = [qsz, kvsz, kvsz, iqsz, iksz]

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    kv_cache_storage_dtype = torch.uint8 if kv_cache_dtype == "fp8" else dtype

    kv_cache = torch.zeros(num_blocks, num_kv_heads, block_size, 2 * HEAD_DIM, dtype=kv_cache_storage_dtype, device=device)
    index_cache = torch.zeros(num_blocks, block_size, HEAD_DIM, dtype=dtype, device=device)

    slot_mapping = torch.randperm(num_blocks * block_size, dtype=torch.int64, device=device)[:num_tokens]
    index_slot_mapping = torch.roll(slot_mapping, shifts=1)

    q_out = torch.empty(num_tokens, qsz, dtype=dtype, device=device)
    index_q = torch.empty(num_tokens, iqsz, dtype=dtype, device=device)

    torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv, q_w, k_w, cos_sin, positions, num_heads, num_kv_heads, ROTARY_DIM, eps,
        iq_w, ik_w, num_idx_heads, slot_mapping, index_slot_mapping, kv_cache,
        index_cache, block_size, q_out, index_q, kv_cache_dtype, False
    )

    _, k_out, v_out, _, index_k = qkv.split(splits, dim=-1)
    q_in, k_in, v_in, iq_orig, ik_orig = qkv_orig.split(splits, dim=-1)

    q_ref = norm_rope_ref(q_in.view(num_tokens, num_heads, HEAD_DIM), q_w, positions, cos_sin, eps).view(num_tokens, qsz)
    k_ref = norm_rope_ref(k_in.view(num_tokens, num_kv_heads, HEAD_DIM), k_w, positions, cos_sin, eps).view(num_tokens, kvsz)
    iq_ref = norm_rope_ref(iq_orig.view(num_tokens, num_idx_heads, HEAD_DIM), iq_w, positions, cos_sin, eps).view(num_tokens, iqsz)
    ik_ref = norm_rope_ref(ik_orig.view(num_tokens, 1, HEAD_DIM), ik_w, positions, cos_sin, eps).view(num_tokens, HEAD_DIM)

    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(index_q, iq_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(index_k, ik_ref, rtol=1e-2, atol=1e-2)

    k_ref_h = k_ref.view(num_tokens, num_kv_heads, HEAD_DIM)
    v_ref_h = v_in.view(num_tokens, num_kv_heads, HEAD_DIM)

    if kv_cache_dtype == "fp8":
        expected_kv_cache = torch.zeros_like(kv_cache)
        expected_k_cache, expected_v_cache = expected_kv_cache.transpose(1, 2).split(HEAD_DIM, dim=-1)
        scale = torch.ones((), device=device)

        ops.reshape_and_cache_flash(
            k_out.view(num_tokens, num_kv_heads, HEAD_DIM), v_out.view(num_tokens, num_kv_heads, HEAD_DIM),
            expected_k_cache, expected_v_cache, slot_mapping, kv_cache_dtype, scale, scale
        )

        torch.testing.assert_close(kv_cache, expected_kv_cache, rtol=0, atol=0)
    else:
        for t in range(num_tokens):
            s = slot_mapping[t].item()
            b, pos = s // block_size, s % block_size

            torch.testing.assert_close(kv_cache[b, :, pos, :HEAD_DIM], k_ref_h[t], rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(kv_cache[b, :, pos, HEAD_DIM:], v_ref_h[t], rtol=0, atol=0)

    expected_index_cache = torch.zeros_like(index_cache).view(-1, HEAD_DIM)
    expected_index_cache[index_slot_mapping] = index_k

    torch.testing.assert_close(index_cache.view(-1, HEAD_DIM), expected_index_cache, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Test 3: sparse mode - skip index branch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_tokens", [1, 7, 64, 513])
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_sparse_skip_index_branch(num_tokens, block_size, kv_cache_dtype):
    torch.manual_seed(2)
    device, dtype, eps = "cuda", torch.bfloat16, 1e-6
    base, max_pos = 5_000_000.0, 4096
    num_heads, num_kv_heads, num_idx_heads = 16, 4, 4

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, dtype, device)
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64, device=device)

    qsz, kvsz = num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    iqsz, iksz = num_idx_heads * HEAD_DIM, HEAD_DIM

    # skip_index_branch=True 仍然要求 sparse qkv layout。
    qkv = torch.randn(num_tokens, qsz + 2 * kvsz + iqsz + iksz, dtype=dtype, device=device)
    qkv_orig = qkv.clone()
    splits = [qsz, kvsz, kvsz, iqsz, iksz]

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    kv_cache_storage_dtype = torch.uint8 if kv_cache_dtype == "fp8" else dtype

    kv_cache = torch.zeros(num_blocks, num_kv_heads, block_size, 2 * HEAD_DIM, dtype=kv_cache_storage_dtype, device=device)

    index_cache = torch.randn(num_blocks, block_size, HEAD_DIM, dtype=dtype, device=device)
    index_cache_orig = index_cache.clone()

    slot_mapping = torch.randperm(num_blocks * block_size, dtype=torch.int64, device=device)[:num_tokens]

    q_out = torch.empty(num_tokens, qsz, dtype=dtype, device=device)

    # 注意：不能省略 index_q_norm_weight。
    # schema 中它没有默认值，所以必须显式传 None。
    torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(
        qkv, q_w, k_w, cos_sin, positions, num_heads, num_kv_heads, ROTARY_DIM, eps,
        None, None, num_idx_heads, slot_mapping, None, kv_cache, index_cache,
        block_size, q_out, None, kv_cache_dtype, True
    )

    _, k_out, v_out, index_q_out, index_k_out = qkv.split(splits, dim=-1)
    q_in, k_in, v_in, index_q_in, index_k_in = qkv_orig.split(splits, dim=-1)

    q_ref = norm_rope_ref(q_in.view(num_tokens, num_heads, HEAD_DIM), q_w, positions, cos_sin, eps).view(num_tokens, qsz)
    k_ref = norm_rope_ref(k_in.view(num_tokens, num_kv_heads, HEAD_DIM), k_w, positions, cos_sin, eps).view(num_tokens, kvsz)

    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_out, v_in, rtol=0, atol=0)

    # skip_index_branch=True 后 index branch 不应修改 qkv 中 index 部分。
    torch.testing.assert_close(index_q_out, index_q_in, rtol=0, atol=0)
    torch.testing.assert_close(index_k_out, index_k_in, rtol=0, atol=0)

    # index cache 也不应该被写入。
    torch.testing.assert_close(index_cache, index_cache_orig, rtol=0, atol=0)

    if kv_cache_dtype == "fp8":
        expected_kv_cache = torch.zeros_like(kv_cache)
        expected_k_cache, expected_v_cache = expected_kv_cache.transpose(1, 2).split(HEAD_DIM, dim=-1)
        scale = torch.ones((), device=device)

        ops.reshape_and_cache_flash(
            k_out.view(num_tokens, num_kv_heads, HEAD_DIM), v_out.view(num_tokens, num_kv_heads, HEAD_DIM),
            expected_k_cache, expected_v_cache, slot_mapping, kv_cache_dtype, scale, scale
        )

        torch.testing.assert_close(kv_cache, expected_kv_cache, rtol=0, atol=0)
    else:
        k_ref_h = k_ref.view(num_tokens, num_kv_heads, HEAD_DIM)
        v_ref_h = v_in.view(num_tokens, num_kv_heads, HEAD_DIM)

        for t in range(num_tokens):
            s = slot_mapping[t].item()
            b, pos = s // block_size, s % block_size

            torch.testing.assert_close(kv_cache[b, :, pos, :HEAD_DIM], k_ref_h[t], rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(kv_cache[b, :, pos, HEAD_DIM:], v_ref_h[t], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Test 4: fp8 index outputs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="e4m3 conversion requires CUDA SM89+.",
)
@pytest.mark.parametrize("num_tokens", [1, 7, 64, 513])
@pytest.mark.parametrize("block_size", [16, 64])
def test_sparse_full_fp8_index(num_tokens, block_size):
    torch.manual_seed(1)
    device, dtype, eps = "cuda", torch.bfloat16, 1e-6
    base, max_pos = 5_000_000.0, 4096
    num_heads, num_kv_heads, num_idx_heads = 16, 4, 4

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    iq_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    ik_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1

    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, dtype, device)
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64, device=device)

    qsz, kvsz = num_heads * HEAD_DIM, num_kv_heads * HEAD_DIM
    iqsz = num_idx_heads * HEAD_DIM
    iksz = HEAD_DIM

    qkv0 = torch.randn(num_tokens, qsz + 2 * kvsz + iqsz + iksz, dtype=dtype, device=device)

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    slot_mapping = torch.randperm(num_blocks * block_size, dtype=torch.int64, device=device)[:num_tokens]
    index_slot_mapping = torch.roll(slot_mapping, shifts=1)

    def run(index_dtype):
        qkv = qkv0.clone()

        kv_cache = torch.zeros(num_blocks, num_kv_heads, block_size, 2 * HEAD_DIM, dtype=dtype, device=device)
        index_cache = torch.zeros(num_blocks, block_size, HEAD_DIM, dtype=index_dtype, device=device)
        q_out = torch.empty(num_tokens, qsz, dtype=dtype, device=device)
        index_q = torch.empty(num_tokens, iqsz, dtype=index_dtype, device=device)

        torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv, q_w, k_w, cos_sin, positions, num_heads, num_kv_heads, ROTARY_DIM, eps,
            iq_w, ik_w, num_idx_heads, slot_mapping, index_slot_mapping,
            kv_cache, index_cache, block_size, q_out, index_q, "auto", False
        )

        return qkv, kv_cache, index_cache, q_out, index_q

    qkv_bf, kvc_bf, idxc_bf, qo_bf, iq_bf = run(torch.bfloat16)
    qkv_fp, kvc_fp, idxc_fp, qo_fp, iq_fp = run(torch.float8_e4m3fn)

    assert iq_fp.dtype == torch.float8_e4m3fn
    assert idxc_fp.dtype == torch.float8_e4m3fn

    torch.testing.assert_close(qo_fp, qo_bf, rtol=0, atol=0)
    torch.testing.assert_close(qkv_fp, qkv_bf, rtol=0, atol=0)
    torch.testing.assert_close(kvc_fp, kvc_bf, rtol=0, atol=0)
    torch.testing.assert_close(iq_fp.float(), iq_bf.float(), rtol=0.13, atol=0.05)
    torch.testing.assert_close(idxc_fp.float(), idxc_bf.float(), rtol=0.13, atol=0.05)