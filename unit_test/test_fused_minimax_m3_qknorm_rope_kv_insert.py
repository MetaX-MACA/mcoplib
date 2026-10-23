# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for MCOPLIB fused_minimax_m3_qknorm_rope_kv_insert."""

import pytest
import torch
import vllm._custom_ops as ops
import mcoplib._C

from vllm.v1.attention.ops.triton_merge_attn_states import (
    merge_attn_states as merge_attn_states_triton,
)


HEAD_DIM = 128
ROTARY_DIM = 64


def _op_available() -> bool:
    return hasattr(torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert")


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _op_available(),
    reason="CUDA not available or fused MiniMax-M3 op not built in",
)


def make_cos_sin_cache(max_pos, rotary_dim, base, dtype, device):
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache.to(dtype)


def gemma_rmsnorm(x, weight, eps):
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    out = out * (1.0 + weight.float())
    return out.to(x.dtype)


def apply_rope_neox_partial(x, positions, cos_sin_cache, rotary_dim):
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


# ============================================================
# Test 1: dense
#
# Dense layout:
# q + k + v
#
# q_out is used as explicit output.
# index branch is disabled by num_index_heads=0.
# ============================================================

@pytest.mark.parametrize("num_tokens,num_heads,num_kv_heads", [(1, 8, 2), (64, 16, 4), (513, 64, 4)])
def test_dense_norm_rope(num_tokens, num_heads, num_kv_heads):
    torch.manual_seed(0)

    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    base = 5_000_000.0
    max_pos = 4096

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1

    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, dtype, device)
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64, device=device)

    qsz = num_heads * HEAD_DIM
    kvsz = num_kv_heads * HEAD_DIM

    qkv = torch.randn(num_tokens, qsz + 2 * kvsz, dtype=dtype, device=device)
    qkv_orig = qkv.clone()

    q_out = torch.empty(num_tokens, qsz, dtype=dtype, device=device)

    torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(qkv, q_w, k_w, cos_sin, positions, num_heads, num_kv_heads, ROTARY_DIM, eps, None, None, 0, None, None, None, None, 0, q_out, None, "auto", False, None, 1.0)

    q_out, k_out, v_out = qkv.split([qsz, kvsz, kvsz], dim=-1)
    q_in, k_in, v_in = qkv_orig.split([qsz, kvsz, kvsz], dim=-1)
    q_out_ref = norm_rope_ref(q_in.view(num_tokens, num_heads, HEAD_DIM), q_w, positions, cos_sin, eps).view(num_tokens, qsz)
    k_ref = norm_rope_ref(k_in.view(num_tokens, num_kv_heads, HEAD_DIM), k_w, positions, cos_sin, eps).view(num_tokens, kvsz)

    q_result = q_out
    k_result = qkv[..., qsz:qsz + kvsz]
    v_result = qkv[..., qsz + kvsz:]

    torch.testing.assert_close(q_result, q_out_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_result, k_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_result, v_in, rtol=0, atol=0)

    print(f"\n[DENSE] tokens={num_tokens}, heads={num_heads}, kv_heads={num_kv_heads} PASS")


# ============================================================
# Test 2: sparse full index branch
#
# Sparse layout:
# q + k + v + index_q + index_k
# ============================================================

@pytest.mark.parametrize("num_tokens,block_size", [(1, 16), (64, 16), (513, 64)])
def test_sparse_full(num_tokens, block_size):
    torch.manual_seed(1)

    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    base = 5_000_000.0
    max_pos = 4096

    num_heads = 16
    num_kv_heads = 4
    num_index_heads = 4

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    iq_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    ik_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1

    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, dtype, device)
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64, device=device)

    qsz = num_heads * HEAD_DIM
    kvsz = num_kv_heads * HEAD_DIM
    iqsz = num_index_heads * HEAD_DIM
    iksz = HEAD_DIM

    total_dim = qsz + kvsz + kvsz + iqsz + iksz

    qkv = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    qkv_orig = qkv.clone()

    splits = [qsz, kvsz, kvsz, iqsz, iksz]

    num_blocks = (num_tokens + block_size - 1) // block_size + 1

    kv_cache = torch.zeros(num_blocks, num_kv_heads, block_size, 2 * HEAD_DIM, dtype=dtype, device=device)
    index_cache = torch.zeros(num_blocks, block_size, HEAD_DIM, dtype=dtype, device=device)

    slot_mapping = torch.randperm(num_blocks * block_size, dtype=torch.int64, device=device)[:num_tokens]
    index_slot_mapping = torch.roll(slot_mapping, shifts=1)

    q_out = torch.empty(num_tokens, qsz, dtype=dtype, device=device)
    index_q_out = torch.empty(num_tokens, iqsz, dtype=dtype, device=device)

    torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(qkv, q_w, k_w, cos_sin, positions, num_heads, num_kv_heads, ROTARY_DIM, eps, iq_w, ik_w, num_index_heads, slot_mapping, index_slot_mapping, kv_cache, index_cache, block_size, q_out, index_q_out, "auto", False, None, 1.0)

    q_in, k_in, v_in, iq_in, ik_in = qkv_orig.split(splits, dim=-1)
    _, k_result, v_result, _, index_k_result = qkv.split(splits, dim=-1)

    q_ref = norm_rope_ref(q_in.view(num_tokens, num_heads, HEAD_DIM), q_w, positions, cos_sin, eps).view(num_tokens, qsz)
    k_ref = norm_rope_ref(k_in.view(num_tokens, num_kv_heads, HEAD_DIM), k_w, positions, cos_sin, eps).view(num_tokens, kvsz)
    iq_ref = norm_rope_ref(iq_in.view(num_tokens, num_index_heads, HEAD_DIM), iq_w, positions, cos_sin, eps).view(num_tokens, iqsz)
    ik_ref = norm_rope_ref(ik_in.view(num_tokens, 1, HEAD_DIM), ik_w, positions, cos_sin, eps).view(num_tokens, iksz)

    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_result, k_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_result, v_in, rtol=0, atol=0)
    torch.testing.assert_close(index_q_out, iq_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(index_k_result, ik_ref, rtol=1e-2, atol=1e-2)

    print(f"\n[SPARSE FULL] tokens={num_tokens}, block={block_size} PASS")

    _, k_out, v_out, index_q_out, index_k_out = qkv.split(splits, dim=-1)
    q_in, k_in, v_in, index_q_in, index_k_in = qkv_orig.split(splits, dim=-1)

# ============================================================
# Test 3: sparse + skip index branch
#
# 注意：
# skip_index_branch=True 时仍然必须使用 sparse qkv layout。
# ============================================================

@pytest.mark.parametrize("num_tokens,block_size", [(1, 16), (64, 16), (513, 64)])
def test_sparse_skip_index_branch(num_tokens, block_size):
    torch.manual_seed(2)

    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    base = 5_000_000.0
    max_pos = 4096

    num_heads = 16
    num_kv_heads = 4
    num_index_heads = 4

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1

    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, dtype, device)
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64, device=device)

    qsz = num_heads * HEAD_DIM
    kvsz = num_kv_heads * HEAD_DIM
    iqsz = num_index_heads * HEAD_DIM
    iksz = HEAD_DIM

    total_dim = qsz + kvsz + kvsz + iqsz + iksz

    qkv = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    qkv_orig = qkv.clone()

    splits = [qsz, kvsz, kvsz, iqsz, iksz]

    num_blocks = (num_tokens + block_size - 1) // block_size + 1

    kv_cache = torch.zeros(num_blocks, num_kv_heads, block_size, 2 * HEAD_DIM, dtype=dtype, device=device)

    index_cache = torch.randn(num_blocks, block_size, HEAD_DIM, dtype=dtype, device=device)
    index_cache_orig = index_cache.clone()

    slot_mapping = torch.randperm(num_blocks * block_size, dtype=torch.int64, device=device)[:num_tokens]

    q_out = torch.empty(num_tokens, qsz, dtype=dtype, device=device)

    torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(qkv, q_w, k_w, cos_sin, positions, num_heads, num_kv_heads, ROTARY_DIM, eps, None, None, num_index_heads, slot_mapping, None, kv_cache, index_cache, block_size, q_out, None, "auto", True, None, 1.0)

    q_in, k_in, v_in, index_q_in, index_k_in = qkv_orig.split(splits, dim=-1)
    _, k_result, v_result, index_q_result, index_k_result = qkv.split(splits, dim=-1)

    q_ref = norm_rope_ref(q_in.view(num_tokens, num_heads, HEAD_DIM), q_w, positions, cos_sin, eps).view(num_tokens, qsz)
    k_ref = norm_rope_ref(k_in.view(num_tokens, num_kv_heads, HEAD_DIM), k_w, positions, cos_sin, eps).view(num_tokens, kvsz)

    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_result, k_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_result, v_in, rtol=0, atol=0)

    # skip_index_branch=True 后 index 分支不能修改 qkv。
    torch.testing.assert_close(index_q_result, index_q_in, rtol=0, atol=0)
    torch.testing.assert_close(index_k_result, index_k_in, rtol=0, atol=0)

    # index cache 不能被修改。
    torch.testing.assert_close(index_cache, index_cache_orig, rtol=0, atol=0)

    print(f"\n[SPARSE SKIP INDEX] tokens={num_tokens}, block={block_size} PASS")


# ============================================================
# Test 4: both empty / small sanity case
# ============================================================

def test_small_sanity():
    torch.manual_seed(3)

    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    base = 5_000_000.0
    max_pos = 128

    num_tokens = 8
    num_heads = 8
    num_kv_heads = 2

    q_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1
    k_w = torch.randn(HEAD_DIM, dtype=dtype, device=device) * 0.1

    cos_sin = make_cos_sin_cache(max_pos, ROTARY_DIM, base, dtype, device)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)

    qsz = num_heads * HEAD_DIM
    kvsz = num_kv_heads * HEAD_DIM

    qkv = torch.randn(num_tokens, qsz + 2 * kvsz, dtype=dtype, device=device)
    qkv_orig = qkv.clone()

    q_out = torch.empty(num_tokens, qsz, dtype=dtype, device=device)

    torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(qkv, q_w, k_w, cos_sin, positions, num_heads, num_kv_heads, ROTARY_DIM, eps, None, None, 0, None, None, None, None, 0, q_out, None, "auto", False, None, 1.0)

    q_in, k_in, v_in = qkv_orig.split([qsz, kvsz, kvsz], dim=-1)

    q_ref = norm_rope_ref(q_in.view(num_tokens, num_heads, HEAD_DIM), q_w, positions, cos_sin, eps).view(num_tokens, qsz)
    k_ref = norm_rope_ref(k_in.view(num_tokens, num_kv_heads, HEAD_DIM), k_w, positions, cos_sin, eps).view(num_tokens, kvsz)

    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(qkv[..., qsz:qsz + kvsz], k_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(qkv[..., qsz + kvsz:], v_in, rtol=0, atol=0)

    print("\n[SMALL SANITY] PASS")
