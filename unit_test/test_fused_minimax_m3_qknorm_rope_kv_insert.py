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


# ============================================================
# Test 5: sparse + NVFP4 main KV cache
#
# kv_cache_dtype="nvfp4" takes the branch at
# fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu:1149: the main K/V cache is
# then HND-packed E2M1 pages ([nb, 2*nkv, block_size, 72] uint8) dequantized by
# per-layer scales, so the cache is compared byte-for-byte against a reference
# implementation of storeNvfp4CacheElems rather than against k/v themselves.
# ============================================================

NVFP4_DATA_BYTES = HEAD_DIM // 2                          # 64: two E2M1 per byte
NVFP4_SCALE_BYTES = HEAD_DIM // 16                        # 8: one E4M3 per 16 dims
NVFP4_PAGE_DIM = NVFP4_DATA_BYTES + NVFP4_SCALE_BYTES     # 72: one head's page

# (midpoint, code above it, does the midpoint itself take the code above?)
# Exact ties round to the even code, matching cvt.rn.satfinite.e2m1x2.
E2M1_THRESHOLDS = (
    (0.25, 1, False),
    (0.75, 2, True),
    (1.25, 3, False),
    (1.75, 4, True),
    (2.50, 5, False),
    (3.50, 6, True),
    (5.00, 7, False),
)
# fl(1 / 6), the reciprocal multiply nvfp4GroupScale corrects against
RCP6 = float.fromhex("0x1.555556p-3")
FP8_MIN = 1.0 / 512      # smallest E4M3 block scale


def _fmul_rz(a, b):
    """float32 multiply rounded toward zero (the kernel's __fmul_rz)."""
    rn = a * b
    exact = a.double() * b.double()  # exact: 24x24 significand bits fit in fp64
    toward_zero = torch.nextafter(rn, torch.zeros_like(rn))
    return torch.where(rn.abs() > exact.abs(), toward_zero, rn)


def _e2m1_codes(q):
    """E2M1 code (sign | magnitude) of q: RNE, saturating at +-6."""
    ax = q.abs().clamp(max=6.0)
    code = torch.zeros_like(ax, dtype=torch.int64)
    for thresh, above, inclusive in E2M1_THRESHOLDS:
        hit = (ax >= thresh) if inclusive else (ax > thresh)
        code = torch.where(hit, torch.full_like(code, above), code)
    # Zero is stored as +0 (clearNegZeroNibbles).
    sign = torch.where((q < 0) & (code > 0), torch.full_like(code, 8),
                       torch.zeros_like(code))
    return code | sign


def _nvfp4_quantize(x, scale):
    """Reference of the quantization in storeNvfp4CacheElems.

    x     : [M, 128] float32 model-dtype values, i.e. exactly what the kernel
            reads back out of qkv (bf16-rounded) before quantizing it.
    scale : float, the layer's dequantization (global) scale.
    Returns (data [M, 64] uint8, sf [M, 8] uint8).
    """
    m = x.shape[0]
    groups = x.view(m, NVFP4_SCALE_BYTES, 16)
    amax = groups.abs().amax(dim=-1)                                   # [M, 8]

    if scale == 1.0:
        # nvfp4UnitGroupScale: no global scale, uncorrected amax * RN(1/6).
        sf = (amax * RCP6).clamp(min=FP8_MIN).to(torch.float8_e4m3fn)
        sf_value = sf.float().unsqueeze(-1).expand_as(groups).reshape(m, HEAD_DIM)
        # Truncated multiply, not a division: see the comment in the kernel.
        q = _fmul_rz(x, 1.0 / sf_value)
    else:
        # nvfp4GroupScale: E4M3(sf_scale * amax / 6) floored at 1/512.
        sf_scale = 1.0 / torch.tensor(scale, dtype=torch.float32, device=x.device)
        # divRnByRcp is a correctly rounded a/6; do it in fp64 so torch cannot
        # turn the scalar division into a reciprocal multiply.
        amax = amax.clamp(min=1e-12)  # fmaxf(group_max, 1e-12f)
        amax_6 = torch.minimum((amax.double() / 6.0).float(), amax * RCP6)
        sf = (sf_scale * amax_6).clamp(min=FP8_MIN).to(torch.float8_e4m3fn)
        sf_value = sf.float().unsqueeze(-1).expand_as(groups).reshape(m, HEAD_DIM)
        # nvfp4Quotient: correctly rounded division of the clamped quotient.
        q = (x * sf_scale).clamp(-4096.0, 4096.0) / sf_value

    # One 2-byte word per lane: element 4L+i sits in nibble i (low nibble first).
    nib = _e2m1_codes(q).view(m, NVFP4_DATA_BYTES // 2, 4)
    words = nib[..., 0] | (nib[..., 1] << 4) | (nib[..., 2] << 8) | (nib[..., 3] << 12)
    data = torch.stack(((words & 0xFF).to(torch.uint8),
                        ((words >> 8) & 0xFF).to(torch.uint8)), dim=-1)
    return data.reshape(m, NVFP4_DATA_BYTES), sf.view(torch.uint8)


def _nvfp4_ref_cache(base_cache, k_vals, v_vals, slot_mapping, block_size,
                     k_scale, v_scale):
    """Reference of the kernel's NVFP4 page writes on top of ``base_cache``.

    ``base_cache`` must be the cache *before* the op, so that a write the
    reference does not expect (wrong slot, negative slot, stray head) shows up
    as a mismatch instead of being pre-absorbed. k_vals / v_vals are
    [T, nkv, 128] in the qkv dtype and must hold exactly what the kernel
    quantizes: K as written back by the norm+RoPE step, V verbatim (V is
    neither normed nor roped).
    """
    ref = base_cache.clone().view(-1)
    nkv = k_vals.shape[1]
    device = ref.device
    page_stride = 2 * nkv * block_size * NVFP4_PAGE_DIM
    slot_stride = block_size * NVFP4_PAGE_DIM

    # Negative slots are padded/unscheduled tokens: nothing is written.
    valid = slot_mapping >= 0
    slot = slot_mapping[valid]
    page = torch.div(slot, block_size, rounding_mode="floor")
    token = slot % block_size
    # A page holds 2*nkv slot blocks of [block_size, 72] bytes, each one head's
    # packed [block_size, 64] E2M1 data followed by its [block_size, 8] scales
    # (values first, then scales -- not interleaved per token).
    base = ((page * page_stride).unsqueeze(1)
            + (2 * torch.arange(nkv, device=device)).unsqueeze(0) * slot_stride)
    base = base.reshape(-1)                                   # [T' * nkv]
    tok = token.repeat_interleave(nkv)

    dims = torch.arange(NVFP4_DATA_BYTES, device=device)
    grp = torch.arange(NVFP4_SCALE_BYTES, device=device)
    # K scales are token-major, V uses the SM100 trtllm-gen 4x4 token-quad swizzle.
    k_sf = tok.unsqueeze(1) * NVFP4_SCALE_BYTES + grp
    v_sf = (((tok.unsqueeze(1) // 4) * 4 + grp // 2) * NVFP4_SCALE_BYTES
            + (grp % 2) * 4 + tok.unsqueeze(1) % 4)

    sides = ((0, k_vals, k_scale, k_sf), (1, v_vals, v_scale, v_sf))
    for side, vals, scale, sf_off in sides:
        data, sf = _nvfp4_quantize(
            vals[valid].reshape(-1, HEAD_DIM).float(), scale)
        slot_base = (base + side * slot_stride).unsqueeze(1)   # [M, 1]
        ref[slot_base + tok.unsqueeze(1) * NVFP4_DATA_BYTES + dims] = data
        ref[slot_base + block_size * NVFP4_DATA_BYTES + sf_off] = sf
    return ref.view(base_cache.shape)


def _assert_nvfp4_cache_equal(kv_cache, ref_cache):
    if torch.equal(kv_cache, ref_cache):
        return
    diff = (kv_cache != ref_cache).view(-1).nonzero()
    _, num_slots, block_size, width = kv_cache.shape
    off = int(diff[0])
    page, rest = divmod(off, num_slots * block_size * width)
    slot, rest = divmod(rest, block_size * width)
    if rest < block_size * NVFP4_DATA_BYTES:
        region = f"data[{rest % NVFP4_DATA_BYTES}] of token {rest // NVFP4_DATA_BYTES}"
    else:
        region = f"scales[{rest - block_size * NVFP4_DATA_BYTES}]"
    raise AssertionError(
        f"NVFP4 pages differ in {diff.numel()} bytes; first at byte {off}: "
        f"page={page} head={slot // 2} side={'KV'[slot % 2]} {region}")


@pytest.mark.parametrize("k_scale,v_scale", [(1.0, 1.0), (0.07, 0.03125)])
@pytest.mark.parametrize("num_tokens,block_size", [(1, 16), (64, 16), (257, 64)])
@pytest.mark.parametrize("skip_index_branch", [False, True])
def test_sparse_nvfp4(num_tokens, block_size, k_scale, v_scale,
                      skip_index_branch):
    torch.manual_seed(4)

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
    splits = [qsz, kvsz, kvsz, iqsz, iksz]

    qkv = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
    qkv_orig = qkv.clone()

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    num_slots = num_blocks * block_size

    # NVFP4 main cache: HND pages of 72 bytes per head slot, uint8 storage.
    kv_cache = torch.zeros(num_blocks, 2 * num_kv_heads, block_size,
                           NVFP4_PAGE_DIM, dtype=torch.uint8, device=device)
    kv_cache_orig = kv_cache.clone()   # pre-call snapshot for the reference
    # Random content so an un-inserted row is distinguishable from a written one.
    index_cache = torch.randn(num_slots, HEAD_DIM, dtype=dtype, device=device)
    index_cache_orig = index_cache.clone()

    slot_mapping = torch.randperm(num_slots, dtype=torch.int64, device=device)[:num_tokens]
    index_slot_mapping = torch.roll(slot_mapping, shifts=1)
    if num_tokens > 1:
        slot_mapping[1] = -1            # padded / unscheduled token
    if num_tokens > 2:
        index_slot_mapping[2] = -1

    kv_k_scale = torch.tensor([k_scale], dtype=torch.float32, device=device)
    kv_v_scale = torch.tensor([v_scale], dtype=torch.float32, device=device)

    q_out = torch.empty(num_tokens, qsz, dtype=dtype, device=device)
    index_q_out = (None if skip_index_branch
                   else torch.empty(num_tokens, iqsz, dtype=dtype, device=device))

    torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(qkv, q_w, k_w, cos_sin, positions, num_heads, num_kv_heads, ROTARY_DIM, eps, iq_w, ik_w, num_index_heads, slot_mapping, index_slot_mapping, kv_cache, index_cache, block_size, q_out, index_q_out, "nvfp4", skip_index_branch, None, 1.0, kv_k_scale, kv_v_scale)

    q_in, k_in, v_in, iq_in, ik_in = qkv_orig.split(splits, dim=-1)
    _, k_result, v_result, iq_result, ik_result = qkv.split(splits, dim=-1)

    q_ref = norm_rope_ref(q_in.view(num_tokens, num_heads, HEAD_DIM), q_w, positions, cos_sin, eps).view(num_tokens, qsz)
    k_ref = norm_rope_ref(k_in.view(num_tokens, num_kv_heads, HEAD_DIM), k_w, positions, cos_sin, eps).view(num_tokens, kvsz)

    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_result, k_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_result, v_in, rtol=0, atol=0)

    if skip_index_branch:
        # Index branch compiled out: qkv index block and index cache untouched.
        torch.testing.assert_close(iq_result, iq_in, rtol=0, atol=0)
        torch.testing.assert_close(ik_result, ik_in, rtol=0, atol=0)
        torch.testing.assert_close(index_cache, index_cache_orig, rtol=0, atol=0)
    else:
        iq_ref = norm_rope_ref(iq_in.view(num_tokens, num_index_heads, HEAD_DIM), iq_w, positions, cos_sin, eps).view(num_tokens, iqsz)
        torch.testing.assert_close(index_q_out, iq_ref, rtol=1e-2, atol=1e-2)
        # The index-K insert is a plain copy of the normed/roped index_k.
        ref_ic = index_cache_orig.clone()
        ic_valid = index_slot_mapping >= 0
        ref_ic[index_slot_mapping[ic_valid]] = ik_result[ic_valid]
        torch.testing.assert_close(index_cache, ref_ic, rtol=0, atol=0)

    # Main K/V cache: byte-exact NVFP4 pages. K is taken *after* the in-place
    # norm+RoPE so the reference quantizes the very values the kernel does.
    ref_cache = _nvfp4_ref_cache(
        kv_cache_orig,
        k_result.view(num_tokens, num_kv_heads, HEAD_DIM),
        v_in.view(num_tokens, num_kv_heads, HEAD_DIM),
        slot_mapping, block_size, k_scale, v_scale)
    # Verify the complete packed NVFP4 page. The per-head page stores the
    # [block_size, 64] data plane first, followed by the [block_size, 8]
    # scale plane; it is not indexed as token * 72 for this packed format.
    assert ref_cache.ne(0).any().item(), "reference wrote no page: vacuous test"
    _assert_nvfp4_cache_equal(kv_cache, ref_cache)

    print(f"\n[NVFP4] tokens={num_tokens}, block={block_size}, "
          f"k_scale={k_scale}, v_scale={v_scale}, "
          f"skip_index_branch={skip_index_branch} PASS")
