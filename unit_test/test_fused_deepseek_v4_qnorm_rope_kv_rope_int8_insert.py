# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


try:
    import mcoplib._C
except ImportError:
    pass


OP_NAME = "fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert"

DEVICE = "cuda"

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM

DEFAULT_BLOCK_SIZE = 16
CACHE_TOKEN_BYTES = 576

EPS = 1e-6

SUPPORTED_Q_HEAD_PADDED = (8, 16, 32, 64, 128)


def _require_op():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")
    if not hasattr(torch.ops._C, OP_NAME):
        pytest.skip(f"torch.ops._C.{OP_NAME} is not available")


def _make_cos_sin_cache(max_pos, rope_dim=ROPE_DIM):
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=DEVICE) / rope_dim))
    positions = torch.arange(max_pos, dtype=torch.float32, device=DEVICE)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).contiguous()


def _rmsnorm_no_weight(x, eps):
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    return x_float * torch.rsqrt(variance + eps)


def _apply_rope_gptj_last_k(x, position_ids, cos_sin_cache):
    rope_dim = cos_sin_cache.shape[-1]
    half = rope_dim // 2
    positions = position_ids.detach().cpu().to(torch.float32).reshape(-1, 1)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)).reshape(1, -1)
    freqs = positions * inv_freq
    cs = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(device=x.device, dtype=torch.float32)
    cos = cs[..., :half]
    sin = cs[..., half:]
    rope = x[..., -rope_dim:].float()
    original_shape = rope.shape
    rope = rope.reshape(*original_shape[:-1], half, 2)
    even = rope[..., 0]
    odd = rope[..., 1]
    while cos.dim() < even.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    new_even = even * cos - odd * sin
    new_odd = even * sin + odd * cos
    rotated = torch.stack((new_even, new_odd), dim=-1).reshape(original_shape)
    output = x.float().clone()
    output[..., -rope_dim:] = rotated
    return output.to(x.dtype)


def _reference_q(q_in, position_ids, cos_sin_cache, eps):
    q_norm = _rmsnorm_no_weight(q_in, eps)
    return _apply_rope_gptj_last_k(q_norm, position_ids, cos_sin_cache).to(q_in.dtype)


def _make_cache(num_blocks, block_size, fill_value=0x7F):
    block_bytes = block_size * CACHE_TOKEN_BYTES
    k_cache = torch.full((num_blocks, block_bytes), fill_value, dtype=torch.uint8, device=DEVICE)
    return k_cache.contiguous()


def _slot_to_block_offset(slot, block_size):
    block_id = slot // block_size
    block_offset = slot % block_size
    return block_id, block_offset


def _run_op(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, q_head_padded, eps, block_size):
    result = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, q_head_padded, eps, block_size)
    assert isinstance(result, torch.Tensor)
    torch.cuda.synchronize()
    return result, k_cache


def _make_inputs(num_tokens, num_heads, q_head_padded=None, block_size=DEFAULT_BLOCK_SIZE, num_blocks=None, dtype=torch.bfloat16, seed=1234):
    if q_head_padded is None:
        q_head_padded = num_heads
    if num_blocks is None:
        num_blocks = max(1, (num_tokens + block_size - 1) // block_size)
    assert q_head_padded >= num_heads
    assert q_head_padded in SUPPORTED_Q_HEAD_PADDED
    torch.manual_seed(seed)
    q_in = torch.randn((num_tokens, num_heads, HEAD_DIM), dtype=dtype, device=DEVICE).contiguous()
    kv = torch.randn((num_tokens, HEAD_DIM), dtype=dtype, device=DEVICE).contiguous()
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    position_ids = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    cos_sin_cache = _make_cos_sin_cache(max(num_tokens + 16, 64), ROPE_DIM)
    k_cache = _make_cache(num_blocks, block_size, fill_value=0x7F)
    return q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens,num_heads,q_head_padded", [(1, 1, 8), (2, 2, 8), (8, 4, 8), (16, 8, 8), (32, 16, 16)])
@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert(dtype, num_tokens, num_heads, q_head_padded):
    _require_op()
    block_size = DEFAULT_BLOCK_SIZE
    num_blocks = max(1, (num_tokens + block_size - 1) // block_size)
    q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache = _make_inputs(num_tokens, num_heads, q_head_padded, block_size, num_blocks, dtype=dtype, seed=1000 + num_tokens + num_heads)
    q_cache_before = k_cache.clone()
    q_out, k_cache = _run_op(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, q_head_padded, EPS, block_size)
    expected = _reference_q(q_in, position_ids, cos_sin_cache, EPS)
    assert q_out.shape == (num_tokens, q_head_padded, HEAD_DIM)
    assert q_out.dtype == dtype
    assert q_out.is_contiguous()
    torch.testing.assert_close(q_out[:, :num_heads], expected, atol=2e-2 if dtype == torch.float16 else 4e-2, rtol=2e-2 if dtype == torch.float16 else 4e-2)
    if q_head_padded > num_heads:
        assert torch.equal(q_out[:, num_heads:], torch.zeros_like(q_out[:, num_heads:]))
    assert torch.isfinite(q_out.float()).all()
    assert torch.isfinite(k_cache.float()).all()
    assert torch.any(k_cache != q_cache_before)


@pytest.mark.parametrize("num_tokens,num_heads,q_head_padded", [(1, 1, 32), (8, 8, 16), (16, 16, 32), (32, 16, 32)])
@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_q_head_padding(num_tokens, num_heads, q_head_padded):
    _require_op()
    block_size = DEFAULT_BLOCK_SIZE
    num_blocks = max(1, (num_tokens + block_size - 1) // block_size)
    q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache = _make_inputs(num_tokens, num_heads, q_head_padded, block_size, num_blocks, dtype=torch.bfloat16, seed=2000 + num_tokens + num_heads)
    q_out, _ = _run_op(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, q_head_padded, EPS, block_size)
    expected = _reference_q(q_in, position_ids, cos_sin_cache, EPS)
    assert q_out.shape == (num_tokens, q_head_padded, HEAD_DIM)
    assert q_out.dtype == torch.bfloat16
    assert q_out.is_contiguous()
    torch.testing.assert_close(q_out[:, :num_heads], expected, atol=4e-2, rtol=4e-2)
    assert torch.equal(q_out[:, num_heads:], torch.zeros_like(q_out[:, num_heads:]))


@pytest.mark.parametrize("num_tokens,num_insert", [(8, 0), (8, 1), (8, 4), (8, 8)])
@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_partial_slot_mapping(num_tokens, num_insert):
    _require_op()
    num_heads = 8
    q_head_padded = 16
    block_size = DEFAULT_BLOCK_SIZE
    num_blocks = max(1, (num_tokens + block_size - 1) // block_size)
    q_in, kv, k_cache, _, position_ids, cos_sin_cache = _make_inputs(num_tokens, num_heads, q_head_padded, block_size, num_blocks, dtype=torch.bfloat16, seed=3000 + num_insert)
    slot_mapping = torch.full((num_insert,), -1, dtype=torch.int64, device=DEVICE)
    if num_insert > 0:
        slot_mapping.copy_(torch.arange(num_insert, dtype=torch.int64, device=DEVICE))
    cache_before = k_cache.clone()
    q_out, k_cache = _run_op(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, q_head_padded, EPS, block_size)
    expected = _reference_q(q_in, position_ids, cos_sin_cache, EPS)
    assert q_out.dtype == torch.bfloat16
    torch.testing.assert_close(q_out[:, :num_heads], expected, atol=4e-2, rtol=4e-2)
    assert torch.equal(q_out[:, num_heads:], torch.zeros_like(q_out[:, num_heads:]))
    if num_insert == 0:
        assert torch.equal(k_cache, cache_before)


@pytest.mark.parametrize("num_tokens", [1, 7, 16, 17, 31, 32, 33])
@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_position_boundaries(num_tokens):
    _require_op()
    num_heads = 8
    q_head_padded = 16
    block_size = 16
    num_blocks = max(1, (num_tokens + block_size - 1) // block_size)
    q_in, kv, k_cache, slot_mapping, _, cos_sin_cache = _make_inputs(num_tokens, num_heads, q_head_padded, block_size, num_blocks, dtype=torch.bfloat16, seed=4000 + num_tokens)
    position_ids = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    q_out, _ = _run_op(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, q_head_padded, EPS, block_size)

    assert q_out.shape == (num_tokens, q_head_padded, HEAD_DIM)
    assert q_out.dtype == torch.bfloat16
    assert q_out.is_contiguous()
    assert torch.isfinite(q_out.float()).all()
    assert torch.equal(q_out[:, num_heads:], torch.zeros_like(q_out[:, num_heads:]))


@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_reused_positions():
    _require_op()
    num_tokens = 16
    num_heads = 8
    q_head_padded = 16
    block_size = 16
    num_blocks = 1
    q_in, kv, k_cache, slot_mapping, _, cos_sin_cache = _make_inputs(num_tokens, num_heads, q_head_padded, block_size, num_blocks, dtype=torch.bfloat16, seed=5000)
    position_ids = torch.zeros((num_tokens,), dtype=torch.int64, device=DEVICE)

    q_out, _ = _run_op(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, q_head_padded, EPS, block_size)

    assert q_out.shape == (num_tokens, q_head_padded, HEAD_DIM)
    assert q_out.dtype == torch.bfloat16
    assert q_out.is_contiguous()
    assert torch.isfinite(q_out.float()).all()
    assert torch.equal(q_out[:, num_heads:], torch.zeros_like(q_out[:, num_heads:]))


@pytest.mark.parametrize("num_tokens,num_heads,q_head_padded", [(4, 1, 32), (8, 8, 32), (16, 16, 32), (32, 32, 32)])
@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_head_boundaries(num_tokens, num_heads, q_head_padded):
    _require_op()
    block_size = 16
    num_blocks = max(1, (num_tokens + block_size - 1) // block_size)
    q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache = _make_inputs(num_tokens, num_heads, q_head_padded, block_size, num_blocks, dtype=torch.bfloat16, seed=6000 + num_heads)
    q_out, _ = _run_op(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, q_head_padded, EPS, block_size)
    expected = _reference_q(q_in, position_ids, cos_sin_cache, EPS)
    assert q_out.dtype == torch.bfloat16
    torch.testing.assert_close(q_out[:, :num_heads], expected, atol=4e-2, rtol=4e-2)
    assert torch.equal(q_out[:, num_heads:], torch.zeros_like(q_out[:, num_heads:]))


@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_cache_mapping():
    _require_op()
    num_tokens = 8
    num_heads = 8
    q_head_padded = 16
    block_size = 4
    num_blocks = 2
    q_in, kv, _, _, position_ids, cos_sin_cache = _make_inputs(num_tokens, num_heads, q_head_padded, block_size, num_blocks, dtype=torch.bfloat16, seed=7000)
    slot_mapping = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int64, device=DEVICE)
    k_cache = _make_cache(num_blocks, block_size, fill_value=0x55)
    cache_before = k_cache.clone()
    q_out, k_cache = _run_op(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, q_head_padded, EPS, block_size)
    expected = _reference_q(q_in, position_ids, cos_sin_cache, EPS)
    assert q_out.dtype == torch.bfloat16
    torch.testing.assert_close(q_out[:, :num_heads], expected, atol=4e-2, rtol=4e-2)
    for slot in range(num_tokens):
        block_id, block_offset = _slot_to_block_offset(slot, block_size)
        before_token = cache_before[block_id, block_offset]
        after_token = k_cache[block_id, block_offset]
        assert torch.any(after_token != before_token), f"cache slot {slot} was not written"
    assert torch.equal(q_out[:, num_heads:], torch.zeros_like(q_out[:, num_heads:]))


@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_negative_slot_mapping():
    _require_op()
    num_tokens = 8
    num_heads = 8
    q_head_padded = 16
    block_size = 8
    num_blocks = 1
    q_in, kv, k_cache, _, position_ids, cos_sin_cache = _make_inputs(num_tokens, num_heads, q_head_padded, block_size, num_blocks, dtype=torch.bfloat16, seed=8000)
    k_cache.fill_(0x5A)
    slot_mapping = torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3], dtype=torch.int64, device=DEVICE)
    q_out, k_cache = _run_op(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, q_head_padded, EPS, block_size)
    expected = _reference_q(q_in, position_ids, cos_sin_cache, EPS)
    assert q_out.dtype == torch.bfloat16
    torch.testing.assert_close(q_out[:, :num_heads], expected, atol=4e-2, rtol=4e-2)
    assert torch.equal(q_out[:, num_heads:], torch.zeros_like(q_out[:, num_heads:]))
    for slot in range(4):
        block_id, block_offset = _slot_to_block_offset(slot, block_size)
        assert torch.any(k_cache[block_id, block_offset] != 0x5A), f"valid cache slot {slot} was not written"


@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_input_contract():
    _require_op()
    num_tokens = 16
    num_heads = 8
    q_head_padded = 16
    block_size = 16
    num_blocks = 1
    q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache = _make_inputs(num_tokens, num_heads, q_head_padded, block_size, num_blocks, dtype=torch.bfloat16, seed=9000)

    q_out, _ = _run_op(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, q_head_padded, EPS, block_size)

    assert q_in.shape == (num_tokens, num_heads, HEAD_DIM)
    assert q_in.dtype == torch.bfloat16
    assert q_in.is_contiguous()
    assert kv.shape == (num_tokens, HEAD_DIM)
    assert kv.dtype == torch.bfloat16
    assert kv.is_contiguous()
    assert slot_mapping.shape == (num_tokens,)
    assert slot_mapping.dtype == torch.int64
    assert slot_mapping.is_contiguous()
    assert position_ids.shape == (num_tokens,)
    assert position_ids.dtype == torch.int64
    assert position_ids.is_contiguous()
    assert cos_sin_cache.shape == (max(num_tokens + 16, 64), ROPE_DIM)
    assert cos_sin_cache.dtype == torch.float32
    assert cos_sin_cache.is_contiguous()
    assert q_out.shape == (num_tokens, q_head_padded, HEAD_DIM)
    assert q_out.dtype == torch.bfloat16
    assert q_out.is_contiguous()


@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_q_output_contiguous():
    _require_op()
    num_tokens = 8
    num_heads = 16
    q_head_padded = 32
    block_size = 16
    num_blocks = 1
    q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache = _make_inputs(num_tokens, num_heads, q_head_padded, block_size, num_blocks, dtype=torch.bfloat16, seed=10000)
    q_out, _ = _run_op(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, q_head_padded, EPS, block_size)
    assert q_out.shape == (num_tokens, q_head_padded, HEAD_DIM)
    assert q_out.dtype == torch.bfloat16
    assert q_out.is_contiguous()
    assert q_out.stride(2) == 1
    assert q_out.stride(1) == HEAD_DIM
    assert q_out.stride(0) == q_head_padded * HEAD_DIM


@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_eps():
    _require_op()
    num_tokens = 8
    num_heads = 8
    q_head_padded = 16
    block_size = 16
    num_blocks = 1

    q_in, kv, _, slot_mapping, position_ids, cos_sin_cache = _make_inputs(num_tokens, num_heads, q_head_padded, block_size, num_blocks, dtype=torch.bfloat16, seed=11000)

    q_outputs = []

    for eps in [1e-5, 1e-6, 1e-4]:
        q_input = q_in.clone()
        kv_input = kv.clone()
        slot_input = slot_mapping.clone()
        position_input = position_ids.clone()
        cos_input = cos_sin_cache.clone()
        k_cache = _make_cache(num_blocks, block_size, fill_value=0x7F)

        q_out, _ = _run_op(q_input, kv_input, k_cache, slot_input, position_input, cos_input, q_head_padded, eps, block_size)

        assert isinstance(q_out, torch.Tensor)
        assert q_out.shape == (num_tokens, q_head_padded, HEAD_DIM)
        assert q_out.dtype == torch.bfloat16
        assert q_out.is_contiguous()

        assert torch.isfinite(q_out.float()).all()

        assert torch.equal(
            q_out[:, num_heads:],
            torch.zeros_like(q_out[:, num_heads:]),
        )

        q_outputs.append(q_out)

    assert not torch.equal(q_outputs[0], q_outputs[1]) or not torch.equal(q_outputs[1], q_outputs[2])


@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_large():
    _require_op()
    num_tokens = 2048
    num_heads = 16
    q_head_padded = 32
    block_size = 16
    num_blocks = (num_tokens + block_size - 1) // block_size
    q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache = _make_inputs(num_tokens, num_heads, q_head_padded, block_size, num_blocks, dtype=torch.bfloat16, seed=12000)
    q_out, k_cache = _run_op(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, q_head_padded, EPS, block_size)
    expected = _reference_q(q_in, position_ids, cos_sin_cache, EPS)
    assert q_out.shape == (num_tokens, q_head_padded, HEAD_DIM)
    assert q_out.dtype == torch.bfloat16
    torch.testing.assert_close(q_out[:, :num_heads], expected, atol=4e-2, rtol=4e-2)
    assert torch.equal(q_out[:, num_heads:], torch.zeros_like(q_out[:, num_heads:]))
    assert torch.isfinite(q_out.float()).all()
    assert torch.isfinite(k_cache.float()).all()


@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_invalid_q_shape():
    _require_op()
    q_in = torch.randn((4, 8, 511), dtype=torch.bfloat16, device=DEVICE)
    kv = torch.randn((4, 512), dtype=torch.bfloat16, device=DEVICE)
    k_cache = _make_cache(1, 16)
    slot_mapping = torch.arange(4, dtype=torch.int64, device=DEVICE)
    position_ids = torch.arange(4, dtype=torch.int64, device=DEVICE)
    cos_sin_cache = _make_cos_sin_cache(16)
    with pytest.raises(RuntimeError, match="q_in shape"):
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, 8, EPS, 16)


@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_invalid_kv_shape():
    _require_op()
    q_in = torch.randn((4, 8, 512), dtype=torch.bfloat16, device=DEVICE)
    kv = torch.randn((4, 511), dtype=torch.bfloat16, device=DEVICE)
    k_cache = _make_cache(1, 16)
    slot_mapping = torch.arange(4, dtype=torch.int64, device=DEVICE)
    position_ids = torch.arange(4, dtype=torch.int64, device=DEVICE)
    cos_sin_cache = _make_cos_sin_cache(16)
    with pytest.raises(RuntimeError, match="kv shape"):
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, 8, EPS, 16)


@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_invalid_q_head_padded():
    _require_op()
    q_in = torch.randn((4, 8, 512), dtype=torch.bfloat16, device=DEVICE)
    kv = torch.randn((4, 512), dtype=torch.bfloat16, device=DEVICE)
    k_cache = _make_cache(1, 16)
    slot_mapping = torch.arange(4, dtype=torch.int64, device=DEVICE)
    position_ids = torch.arange(4, dtype=torch.int64, device=DEVICE)
    cos_sin_cache = _make_cos_sin_cache(16)
    with pytest.raises(RuntimeError, match="q_head_padded"):
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, 4, EPS, 16)


@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_invalid_slot_mapping_length():
    _require_op()
    q_in = torch.randn((4, 8, 512), dtype=torch.bfloat16, device=DEVICE)
    kv = torch.randn((4, 512), dtype=torch.bfloat16, device=DEVICE)
    k_cache = _make_cache(1, 16)
    slot_mapping = torch.arange(5, dtype=torch.int64, device=DEVICE)
    position_ids = torch.arange(4, dtype=torch.int64, device=DEVICE)
    cos_sin_cache = _make_cos_sin_cache(16)
    with pytest.raises(RuntimeError, match="slot_mapping"):
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, 8, EPS, 16)


@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_invalid_cos_sin_dtype():
    _require_op()
    q_in = torch.randn((4, 8, 512), dtype=torch.bfloat16, device=DEVICE)
    kv = torch.randn((4, 512), dtype=torch.bfloat16, device=DEVICE)
    k_cache = _make_cache(1, 16)
    slot_mapping = torch.arange(4, dtype=torch.int64, device=DEVICE)
    position_ids = torch.arange(4, dtype=torch.int64, device=DEVICE)
    cos_sin_cache = torch.randn((16, 64), dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(RuntimeError, match="cos_sin_cache must be float32"):
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, 8, EPS, 16)


@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_invalid_cache_dtype():
    _require_op()
    q_in = torch.randn((4, 8, 512), dtype=torch.bfloat16, device=DEVICE)
    kv = torch.randn((4, 512), dtype=torch.bfloat16, device=DEVICE)
    k_cache = torch.empty((1, 16 * CACHE_TOKEN_BYTES), dtype=torch.float16, device=DEVICE)
    slot_mapping = torch.arange(4, dtype=torch.int64, device=DEVICE)
    position_ids = torch.arange(4, dtype=torch.int64, device=DEVICE)
    cos_sin_cache = _make_cos_sin_cache(16)
    with pytest.raises(RuntimeError, match="k_cache must be uint8"):
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, 8, EPS, 16)


@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_mismatched_q_kv_rows():
    _require_op()
    q_in = torch.randn((4, 8, 512), dtype=torch.bfloat16, device=DEVICE)
    kv = torch.randn((5, 512), dtype=torch.bfloat16, device=DEVICE)
    k_cache = _make_cache(1, 16)
    slot_mapping = torch.arange(4, dtype=torch.int64, device=DEVICE)
    position_ids = torch.arange(4, dtype=torch.int64, device=DEVICE)
    cos_sin_cache = _make_cos_sin_cache(16)
    with pytest.raises(RuntimeError, match="row counts must match"):
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, 8, EPS, 16)


@torch.inference_mode()
def test_fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_mismatched_q_position_rows():
    _require_op()
    q_in = torch.randn((4, 8, 512), dtype=torch.bfloat16, device=DEVICE)
    kv = torch.randn((4, 512), dtype=torch.bfloat16, device=DEVICE)
    k_cache = _make_cache(1, 16)
    slot_mapping = torch.arange(4, dtype=torch.int64, device=DEVICE)
    position_ids = torch.arange(5, dtype=torch.int64, device=DEVICE)
    cos_sin_cache = _make_cos_sin_cache(16)
    with pytest.raises(RuntimeError, match="row counts must match"):
        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert(q_in, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, 8, EPS, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])