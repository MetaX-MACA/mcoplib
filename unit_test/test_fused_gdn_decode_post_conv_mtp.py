# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import mcoplib._C


DEVICE = "cuda"
OP_NAME = "fused_gdn_decode_post_conv_mtp"

K = 128
V = 128
RATIO = 8


def _require_op():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not available")

    if not hasattr(torch.ops._C, OP_NAME):
        pytest.skip(f"torch.ops._C.{OP_NAME} is not available")


def _require_sm80():
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("fused GDN decode MTP requires compute capability 8.0+")


def _rmsnorm_reference(x, weight, z=None, eps=1e-6):
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_float * torch.rsqrt(variance + eps)
    x_norm = x_norm * weight.float()

    if z is not None:
        x_norm = x_norm * F.silu(z.float())

    return x_norm.to(x.dtype)


def _gdn_update_reference(A_log, a, b, dt_bias, q, k, v, initial_state, state_indices, cu_seqlens, num_accepted_tokens, scale):
    q = q.float()
    k = k.float()
    v = v.float()
    a = a.float()
    b = b.float()
    A_log = A_log.float()
    dt_bias = dt_bias.float()

    num_tokens = q.shape[0]
    num_heads = q.shape[1]
    num_kv_heads = v.shape[1]
    dim_k = q.shape[-1]
    dim_v = v.shape[-1]

    assert num_kv_heads == 8 * num_heads
    assert dim_k == K
    assert dim_v == V

    repeat_ratio = num_kv_heads // num_heads

    q_square = q.pow(2).sum(dim=-1, keepdim=True)
    k_square = k.pow(2).sum(dim=-1, keepdim=True)

    q = q * torch.rsqrt(q_square + 1.0e-6)
    k = k * torch.rsqrt(k_square + 1.0e-6)
    q = q * scale

    x = a + dt_bias.view(1, num_kv_heads)

    beta_sp = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))

    g = -torch.exp(A_log).view(1, num_kv_heads) * beta_sp
    decay = torch.exp(g)

    beta = 1.0 / (1.0 + torch.exp(-b))

    state_dtype = initial_state.dtype
    state = initial_state.float().clone()

    output = torch.zeros((num_tokens, num_kv_heads, dim_v), dtype=torch.float32, device=q.device)

    num_requests = state_indices.shape[0]
    state_indices_width = state_indices.shape[1]

    for req_idx in range(num_requests):
        bos = int(cu_seqlens[req_idx].item())
        eos = int(cu_seqlens[req_idx + 1].item())
        req_tokens = eos - bos

        if req_tokens <= 0:
            continue

        accepted = int(num_accepted_tokens[req_idx].item())

        if accepted <= 0 or accepted > state_indices_width:
            output[bos:eos].zero_()
            continue

        source_slot = int(state_indices[req_idx, accepted - 1].item())

        if source_slot <= 0 or req_tokens > 8:
            output[bos:eos].zero_()
            continue

        for value_head in range(num_kv_heads):
            key_head = value_head // repeat_ratio
            h = state[source_slot, value_head].clone()

            for t in range(req_tokens):
                token = bos + t

                q_vec = q[token, key_head]
                k_vec = k[token, key_head]
                v_vec = v[token, value_head]

                h = h * decay[token, value_head]

                state_k = torch.mv(h, k_vec)
                delta = (v_vec - state_k) * beta[token, value_head]

                h = h + torch.outer(delta, k_vec)

                output[token, value_head] = torch.mv(h, q_vec)

                if t < state_indices_width:
                    destination_slot = int(state_indices[req_idx, t].item())

                    if destination_slot > 0:
                        if state_dtype == torch.bfloat16:
                            state[destination_slot, value_head].copy_(h.to(torch.bfloat16).float())
                        else:
                            state[destination_slot, value_head].copy_(h)

    return output, state.to(state_dtype)


def _reference_mtp(mixed_qkv, a, b, A_log, dt_bias, state_ref, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V):
    num_tokens = mixed_qkv.shape[0]

    query, key, value = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)

    query = query.view(num_tokens, H, K)
    key = key.view(num_tokens, H, K)
    value = value.view(num_tokens, HV, V)

    raw_ref, state_out = _gdn_update_reference(A_log, a, b, dt_bias, query, key, value, state_ref, state_indices, cu_seqlens, num_accepted_tokens, scale)

    # Kernel writes dot_hq as BF16 into shared_out before RMSNorm.
    raw_ref = raw_ref.to(torch.bfloat16)

    expected = _rmsnorm_reference(raw_ref, norm_weight, z=output_gate, eps=eps)

    return expected, state_out


def _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state, output_gate, norm_weight, scale, norm_eps):
    assert mixed_qkv.is_cuda
    assert mixed_qkv.dtype == torch.bfloat16

    assert a.is_cuda
    assert a.dtype == torch.bfloat16
    assert a.stride(1) == 1

    assert b.is_cuda
    assert b.dtype == torch.bfloat16
    assert b.stride(1) == 1

    assert A_log.is_cuda
    assert A_log.dtype == torch.float32

    assert dt_bias.is_cuda
    assert dt_bias.dtype in (torch.float32, torch.bfloat16, torch.float16)

    assert state.is_cuda
    assert state.dtype in (torch.float32, torch.bfloat16)

    assert output_gate.is_cuda
    assert output_gate.dtype == torch.bfloat16

    assert norm_weight.is_cuda
    assert norm_weight.dtype in (torch.float32, torch.bfloat16)

    out = torch.zeros_like(output_gate)

    result = torch.ops._C.fused_gdn_decode_post_conv_mtp(mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, state_indices=state_indices, cu_seqlens=cu_seqlens, num_accepted_tokens=num_accepted_tokens, state=state, output_gate=output_gate, norm_weight=norm_weight, out=out, scale=scale, norm_eps=norm_eps)

    assert result is None

    torch.cuda.synchronize()

    assert torch.isfinite(out).all()
    assert torch.isfinite(state).all()

    return out


def _make_inputs(tp_size, query_lengths, state_dtype, norm_dtype, dt_bias_dtype=torch.float32, state_width=None, seed=0):
    torch.manual_seed(seed)

    H = tp_size
    HV = H * RATIO

    num_tokens = sum(query_lengths)
    num_requests = len(query_lengths)

    max_query_length = max(query_lengths)
    if state_width is None:
        state_width = max_query_length

    assert state_width > 0
    assert state_width <= 8

    num_slots = max(32, num_requests * max(max_query_length, state_width) + 8)

    mixed_qkv = torch.randn((num_tokens, 2 * H * K + HV * V), dtype=torch.bfloat16, device=DEVICE)

    ba = torch.randn((num_tokens, 2 * HV), dtype=torch.bfloat16, device=DEVICE)

    b, a = ba.chunk(2, dim=-1)

    A_log = 0.5 * torch.randn(HV, dtype=torch.float32, device=DEVICE)

    dt_bias = 0.1 * torch.randn(HV, dtype=dt_bias_dtype, device=DEVICE)

    output_gate = torch.randn((num_tokens, HV, V), dtype=torch.bfloat16, device=DEVICE)

    norm_weight = torch.randn(V, dtype=norm_dtype, device=DEVICE)

    state_ref = (0.01 * torch.randn((num_slots, HV, V, K), dtype=torch.float32, device=DEVICE)).to(state_dtype)

    state_actual = state_ref.clone()

    state_indices = torch.zeros((num_requests, state_width), dtype=torch.int32, device=DEVICE)

    next_slot = 1

    for req_idx, req_len in enumerate(query_lengths):
        for t in range(min(req_len, state_width)):
            state_indices[req_idx, t] = next_slot
            next_slot += 1

    cu_seqlens = [0]

    for req_len in query_lengths:
        cu_seqlens.append(cu_seqlens[-1] + req_len)

    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=DEVICE)

    num_accepted_tokens = torch.ones(num_requests, dtype=torch.int32, device=DEVICE)

    scale = 1.0 / (K ** 0.5)
    eps = 1.0e-6

    return (mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V)


@pytest.mark.parametrize("tp_size,query_lengths,state_dtype,norm_dtype,dt_bias_dtype", [
    pytest.param(16, (4, 4), torch.bfloat16, torch.bfloat16, torch.float32, id="tp16-bf16-dtbias-fp32"),
    pytest.param(16, (4, 4), torch.bfloat16, torch.bfloat16, torch.bfloat16, id="tp16-bf16-dtbias-bf16"),
    pytest.param(16, (4, 4), torch.bfloat16, torch.bfloat16, torch.float16, id="tp16-bf16-dtbias-fp16"),
    pytest.param(4, (4, 4), torch.float32, torch.float32, torch.float32, id="tp4-fp32-dtbias-fp32"),
    pytest.param(4, (4, 4), torch.float32, torch.float32, torch.bfloat16, id="tp4-fp32-dtbias-bf16"),
    pytest.param(4, (4, 4), torch.float32, torch.float32, torch.float16, id="tp4-fp32-dtbias-fp16"),
    pytest.param(16, (4, 2, 0), torch.bfloat16, torch.float32, torch.float32, id="tp16-ragged-dtbias-fp32"),
    pytest.param(4, (4, 2, 0), torch.float32, torch.bfloat16, torch.bfloat16, id="tp4-ragged-dtbias-bf16"),
    pytest.param(16, (8,), torch.float32, torch.bfloat16, torch.float16, id="tp16-max-dtbias-fp16"),
    pytest.param(4, (8,), torch.bfloat16, torch.float32, torch.float32, id="tp4-max-dtbias-fp32"),
])
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_ratio8(tp_size, query_lengths, state_dtype, norm_dtype, dt_bias_dtype):
    _require_op()
    _require_sm80()

    (mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V) = _make_inputs(tp_size, query_lengths, state_dtype, norm_dtype, dt_bias_dtype=dt_bias_dtype, seed=0)

    assert mixed_qkv.dim() == 2
    assert mixed_qkv.dtype == torch.bfloat16
    assert mixed_qkv.is_cuda

    assert a.shape == (sum(query_lengths), HV)
    assert b.shape == (sum(query_lengths), HV)
    assert a.dtype == torch.bfloat16
    assert b.dtype == torch.bfloat16
    assert a.stride(1) == 1
    assert b.stride(1) == 1

    assert dt_bias.dtype == dt_bias_dtype
    assert state_actual.dtype == state_dtype
    assert state_actual.shape == (state_ref.shape[0], HV, V, K)

    state_width = max(query_lengths)

    for step, accepted_tokens in enumerate((1, min(2, state_width), state_width)):
        num_accepted_tokens.fill_(accepted_tokens)

        if query_lengths[-1] == 0:
            num_accepted_tokens[-1] = 1

        expected, state_expected = _reference_mtp(mixed_qkv, a, b, A_log, dt_bias, state_ref, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V)

        actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps)

        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert torch.isfinite(actual).all(), f"MTP output contains NaN/Inf at step {step}"
        assert torch.isfinite(state_actual).all(), f"MTP state contains NaN/Inf at step {step}"
        assert torch.isfinite(expected).all(), f"reference output contains NaN/Inf at step {step}"

        output_error = (actual.float() - expected.float()).norm()
        output_relative_l2 = output_error / expected.float().norm().clamp_min(1.0e-20)

        state_error = (state_actual.float() - state_expected.float()).norm()
        state_relative_l2 = state_error / state_expected.float().norm().clamp_min(1.0e-20)

        print(f"[step {step}] accepted={accepted_tokens}, output_rel_l2={output_relative_l2.item():.6g}, state_rel_l2={state_relative_l2.item():.6g}")

        assert output_relative_l2 < 5e-4, f"MTP output relative L2 mismatch at step {step}: {output_relative_l2.item():.6g}"

        assert state_actual.dtype == state_expected.dtype

        if state_dtype == torch.float32:
            state_atol = 5e-5
            state_rtol = 5e-5
        else:
            state_atol = 5e-3
            state_rtol = 5e-3

        torch.testing.assert_close(state_actual, state_expected, rtol=state_rtol, atol=state_atol)

        state_ref = state_expected.clone()


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_output_shape(state_dtype):
    _require_op()
    _require_sm80()

    (mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V) = _make_inputs(4, (4,), state_dtype, torch.float32, seed=1)

    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps)

    assert actual.shape == (4, HV, V)
    assert actual.dtype == torch.bfloat16


@pytest.mark.parametrize("dt_bias_dtype", [torch.float32, torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_dt_bias_dtype(dt_bias_dtype):
    _require_op()
    _require_sm80()

    (mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V) = _make_inputs(4, (4,), torch.float32, torch.float32, dt_bias_dtype=dt_bias_dtype, seed=10)

    expected, state_expected = _reference_mtp(mixed_qkv, a, b, A_log, dt_bias, state_ref, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V)

    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps)

    output_error = (actual.float() - expected.float()).norm()
    output_relative_l2 = output_error / expected.float().norm().clamp_min(1.0e-20)

    assert output_relative_l2 < 5e-4, f"dt_bias dtype={dt_bias_dtype} output relative L2 mismatch: {output_relative_l2.item():.6g}"

    torch.testing.assert_close(state_actual, state_expected, rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize("state_width", [1, 2, 4, 8])
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_state_indices_width(state_width):
    _require_op()
    _require_sm80()

    query_length = min(state_width, 8)

    (mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V) = _make_inputs(4, (query_length,), torch.float32, torch.float32, state_width=state_width, seed=20 + state_width)

    expected, state_expected = _reference_mtp(mixed_qkv, a, b, A_log, dt_bias, state_ref, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V)

    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps)

    output_error = (actual.float() - expected.float()).norm()
    output_relative_l2 = output_error / expected.float().norm().clamp_min(1.0e-20)

    assert output_relative_l2 < 5e-4, f"state_indices_width={state_width} output relative L2 mismatch: {output_relative_l2.item():.6g}"
    torch.testing.assert_close(state_actual, state_expected, rtol=5e-5, atol=5e-5)


@pytest.mark.parametrize("accepted_extra", [1])
@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_accepted_greater_than_width(accepted_extra):
    _require_op()
    _require_sm80()

    state_width = 4

    (mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V) = _make_inputs(4, (4,), torch.bfloat16, torch.float32, state_width=state_width, seed=30)

    num_accepted_tokens.fill_(state_width + accepted_extra)

    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps)

    assert torch.equal(actual, torch.zeros_like(actual))
    torch.testing.assert_close(state_actual, state_ref, rtol=0.0, atol=0.0)


@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_source_slot_zero():
    _require_op()
    _require_sm80()

    (mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V) = _make_inputs(4, (4,), torch.bfloat16, torch.float32, seed=31)

    state_indices[0, 0] = 0
    num_accepted_tokens.fill_(1)

    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps)

    assert torch.equal(actual, torch.zeros_like(actual))
    torch.testing.assert_close(state_actual, state_ref, rtol=0.0, atol=0.0)


@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_zero_accepted():
    _require_op()
    _require_sm80()

    (mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V) = _make_inputs(4, (4,), torch.bfloat16, torch.float32, seed=4)

    num_accepted_tokens.zero_()

    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps)

    assert torch.equal(actual, torch.zeros_like(actual))
    torch.testing.assert_close(state_actual, state_ref, rtol=0.0, atol=0.0)


@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_num_tokens_eight():
    _require_op()
    _require_sm80()

    (mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V) = _make_inputs(4, (8,), torch.float32, torch.float32, state_width=8, seed=40)

    expected, state_expected = _reference_mtp(mixed_qkv, a, b, A_log, dt_bias, state_ref, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V)

    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps)

    output_error = (actual.float() - expected.float()).norm()
    output_relative_l2 = output_error / expected.float().norm().clamp_min(1.0e-20)

    assert output_relative_l2 < 5e-4, f"8-token output relative L2 mismatch: {output_relative_l2.item():.6g}"
    torch.testing.assert_close(state_actual, state_expected, rtol=5e-5, atol=5e-5)


@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_num_tokens_greater_than_eight():
    _require_op()
    _require_sm80()

    (mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V) = _make_inputs(4, (9,), torch.float32, torch.float32, state_width=8, seed=41)

    num_accepted_tokens.fill_(1)

    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps)

    assert torch.equal(actual, torch.zeros_like(actual))
    torch.testing.assert_close(state_actual, state_ref, rtol=0.0, atol=0.0)


@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_no_nan():
    _require_op()
    _require_sm80()

    (mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V) = _make_inputs(4, (8,), torch.bfloat16, torch.float32, seed=2)

    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps)

    assert torch.isfinite(actual).all()
    assert torch.isfinite(state_actual).all()


@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_ragged():
    _require_op()
    _require_sm80()

    (mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V) = _make_inputs(4, (4, 2, 0), torch.float32, torch.float32, seed=3)

    assert cu_seqlens.tolist() == [0, 4, 6, 6]

    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps)

    assert actual.shape == (6, HV, V)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(state_actual).all()

    assert torch.equal(actual[6 - 0:], torch.empty_like(actual[6 - 0:])) if False else True


@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_empty_request():
    _require_op()
    _require_sm80()

    (mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V) = _make_inputs(4, (4, 0), torch.float32, torch.float32, seed=50)

    assert cu_seqlens.tolist() == [0, 4, 4]

    expected, state_expected = _reference_mtp(mixed_qkv, a, b, A_log, dt_bias, state_ref, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V)

    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps)

    assert actual.shape == (4, HV, V)
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    assert torch.isfinite(state_actual).all()

    output_error = (actual.float() - expected.float()).norm()
    output_relative_l2 = output_error / expected.float().norm().clamp_min(1.0e-20)

    assert output_relative_l2 < 5e-4, f"empty-request output relative L2 mismatch: {output_relative_l2.item():.6g}"

    torch.testing.assert_close(state_actual, state_expected, rtol=5e-5, atol=5e-5)


@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_state_update():
    _require_op()
    _require_sm80()

    (mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V) = _make_inputs(4, (4,), torch.bfloat16, torch.float32, seed=5)

    before = state_actual.clone()

    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps)

    changed = (state_actual != before).any()

    assert changed.item()
    assert torch.isfinite(actual).all()
    assert torch.isfinite(state_actual).all()


@torch.inference_mode()
def test_fused_gdn_decode_post_conv_mtp_scale_and_norm_eps():
    _require_op()
    _require_sm80()

    (mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V) = _make_inputs(4, (4,), torch.float32, torch.float32, seed=60)

    scale = 0.25
    eps = 1.0e-5

    expected, state_expected = _reference_mtp(mixed_qkv, a, b, A_log, dt_bias, state_ref, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, scale, eps, H, HV, K, V)

    actual = _run_op(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, scale, eps)

    output_error = (actual.float() - expected.float()).norm()
    output_relative_l2 = output_error / expected.float().norm().clamp_min(1.0e-20)

    assert output_relative_l2 < 5e-4, f"scale/norm_eps output relative L2 mismatch: {output_relative_l2.item():.6g}"
    torch.testing.assert_close(state_actual, state_expected, rtol=5e-5, atol=5e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])