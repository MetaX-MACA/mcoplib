"""Qwen3.5 GDN recurrent 算子的精度与性能测试。

测试范围：
1. packed Decode：验证原始及优化后的 load order、不同 num_warps、Q/K norm，
   以及 grouped value-head 实验路径；同时检查输出、state 原地更新、无效 state
   index 和非连续 slot stride。
2. 通用接口：验证 prefill、verify、定长和变长序列路径。
3. 泛化性能：覆盖 Qwen3.5 4B/9B（H=16, HV=32）与 27B TP=2 本地 shard
   （H=8, HV=24），以及需求中的
   BS=1～128。每个性能 case 先用 PyTorch reference 检查输出和 state，再测试
   mcoplib 自动分派路径的耗时。

快速精度测试：
    pytest -q unit_test/test_fused_recurrent_gated_delta_rule.py

完整 packed Decode 精度与性能测试：
    pytest -q -s -m full \
        unit_test/test_fused_recurrent_gated_delta_rule.py::test_packed_decode_accuracy_and_performance

性能测试使用随机输入和互相独立的 state slot，避免重复 state index 引起并发写
冲突。每组预热 100 次、测试 1000 次、重复 5 轮，打印 median/min/max；测试只报告
当前优化实现的 kernel 耗时，不与 baseline 比较，也不设置易受机器负载影响的性能阈值。
"""
import statistics

import pytest
import torch
import torch.nn.functional as F

from mcoplib.triton_fused_recurrent_gated_delta_rule import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_recurrent_gated_delta_rule_update,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

GENERAL_BATCH_SIZES = (
    1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104,
    112, 120, 128,
)
PACKED_DECODE_SHAPES = (
    pytest.param(16, 32, 128, id="qwen35_4b_9b"),
    pytest.param(8, 24, 128, id="qwen27b_tp2"),
)


def reference(q, k, v, g, beta, state, normalize=False):
    q, k, v = q.float(), k.float(), v.float()
    if normalize:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    group = v.shape[2] // q.shape[2]
    q = q.repeat_interleave(group, dim=2) * q.shape[-1] ** -0.5
    k = k.repeat_interleave(group, dim=2)
    state = state.float().clone()
    outputs, history = [], []
    for t in range(q.shape[1]):
        state = state * g[:, t].float().exp()[..., None, None]
        delta = (v[:, t] - (state * k[:, t, :, None]).sum(-1)) * beta[:, t].float()[..., None]
        state = state + delta[..., None] * k[:, t, :, None]
        outputs.append((state * q[:, t, :, None]).sum(-1))
        history.append(state.clone())
    return torch.stack(outputs, 1), state, torch.stack(history, 1)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("warps", [1, 4])
@pytest.mark.parametrize(
    "load_order,group_value_heads,reuse_q",
    [
        pytest.param(0, False, False, id="order0"),
        pytest.param(1, False, False, id="order1"),
        pytest.param(2, False, False, id="order2"),
        pytest.param(3, False, False, id="order3"),
        pytest.param(None, True, False, id="group_hv_reuse_k"),
        pytest.param(None, True, True, id="group_hv_reuse_qk"),
    ],
)
def test_packed_decode(
    normalize, warps, load_order, group_value_heads, reuse_q
):
    torch.manual_seed(17)
    B, H, HV, D = 3, 2, 4, 128
    mixed = torch.randn(B, (2 * H + HV) * D, device="cuda", dtype=torch.bfloat16) * 0.1
    a = torch.randn(B, HV, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    alog = torch.randn(HV, device="cuda") * 0.1
    bias = torch.randn(HV, device="cuda", dtype=torch.bfloat16) * 0.1
    # A padded slot stride and permuted indices catch address/aliasing mistakes.
    backing = torch.randn(4, HV * D * D + 64, device="cuda") * 0.01
    state = backing[:, :HV * D * D].view(4, HV, D, D)
    before = state.clone()
    indices = torch.tensor([2, -1, 0], device="cuda", dtype=torch.int32)
    q, k, v = mixed.split([H * D, H * D, HV * D], dim=-1)
    g = -alog.exp() * F.softplus(a.float() + bias.float())
    # Match the source's explicit rounding of sigmoid to b.dtype.
    beta = b.float().sigmoid().to(b.dtype).float()
    expected, updated, _ = reference(
        q.reshape(B, 1, H, D), k.reshape(B, 1, H, D),
        v.reshape(B, 1, HV, D), g[:, None], beta[:, None],
        before[indices.clamp_min(0).long()], normalize,
    )
    expected[1] = 0
    out = torch.empty(B, 1, HV, D, device="cuda", dtype=mixed.dtype)
    actual, returned = fused_recurrent_gated_delta_rule_packed_decode(
        mixed, a, b, alog, bias, D ** -0.5, state, out, indices,
        normalize, num_warps=warps, load_order=load_order,
        group_value_heads=group_value_heads, reuse_q=reuse_q,
    )
    assert returned.data_ptr() == state.data_ptr()
    torch.testing.assert_close(actual.float(), expected, atol=3e-3, rtol=3e-2)
    torch.testing.assert_close(state[2], updated[0], atol=3e-4, rtol=3e-3)
    torch.testing.assert_close(state[0], updated[2], atol=3e-4, rtol=3e-3)
    torch.testing.assert_close(state[1], before[1], atol=0, rtol=0)
    torch.testing.assert_close(state[3], before[3], atol=0, rtol=0)


@pytest.mark.parametrize("B", [1, 2, 4, 8, 16, 48, 128])
def test_packed_decode_27b_tp2_auto_dispatch(B):
    """Exercise every branch of the production 27B TP=2 grouped-HV dispatch."""
    torch.manual_seed(27000 + B)
    H, HV, D = 8, 24, 128
    mixed = torch.randn(
        B, (2 * H + HV) * D, device="cuda", dtype=torch.bfloat16
    ) * 0.1
    a = torch.randn(B, HV, device="cuda", dtype=torch.bfloat16) * 0.5 - 1
    b = torch.randn(B, HV, device="cuda", dtype=torch.bfloat16) * 0.5
    alog = torch.randn(HV, device="cuda") * 0.1
    bias = torch.randn(HV, device="cuda", dtype=torch.bfloat16) * 0.1
    state = torch.randn(B, HV, D, D, device="cuda") * 0.01
    before = state.clone()
    indices = torch.arange(B, device="cuda", dtype=torch.int32)
    out = torch.empty(B, 1, HV, D, device="cuda", dtype=torch.bfloat16)

    q, k, v = mixed.split([H * D, H * D, HV * D], dim=-1)
    g = -alog.exp() * F.softplus(a.float() + bias.float())
    beta = b.float().sigmoid().to(b.dtype).float()
    expected_out, expected_state, _ = reference(
        q.reshape(B, 1, H, D), k.reshape(B, 1, H, D),
        v.reshape(B, 1, HV, D), g[:, None], beta[:, None], before,
        normalize=True,
    )
    actual_out, returned = fused_recurrent_gated_delta_rule_packed_decode(
        mixed, a, b, alog, bias, D ** -0.5, state, out, indices, True,
        bv=32, num_warps=1, num_stages=3,
    )
    assert returned.data_ptr() == state.data_ptr()
    torch.testing.assert_close(actual_out.float(), expected_out, atol=3e-3, rtol=3e-2)
    torch.testing.assert_close(state, expected_state, atol=3e-4, rtol=3e-3)


@pytest.mark.full
@pytest.mark.parametrize("B", GENERAL_BATCH_SIZES)
@pytest.mark.parametrize("H,HV,D", PACKED_DECODE_SHAPES)
def test_packed_decode_accuracy_and_performance(B, H, HV, D):
    """Check the optimized auto path, then report its latency without a baseline."""
    torch.manual_seed(20260910 + B + HV * 1009)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", dtype=dtype)

    mixed = rand(B, (2 * H + HV) * D) * 0.1
    a = rand(B, HV) * 0.5 - 1
    b = rand(B, HV) * 0.5
    alog = rand(HV, dtype=torch.float32) * 0.1
    bias = rand(HV) * 0.1
    initial_state = rand(B, HV, D, D, dtype=torch.float32) * 0.01
    state = initial_state.clone()
    indices = torch.arange(B, device="cuda", dtype=torch.int32)
    out = torch.empty(B, 1, HV, D, device="cuda", dtype=torch.bfloat16)

    q, k, v = mixed.float().split([H * D, H * D, HV * D], dim=-1)
    g = -alog.exp() * F.softplus(a.float() + bias.float())
    beta = b.float().sigmoid().to(b.dtype).float()
    expected_out, expected_state, _ = reference(
        q.reshape(B, 1, H, D),
        k.reshape(B, 1, H, D),
        v.reshape(B, 1, HV, D),
        g[:, None],
        beta[:, None],
        initial_state,
        normalize=True,
    )

    def launch():
        return fused_recurrent_gated_delta_rule_packed_decode(
            mixed, a, b, alog, bias, D ** -0.5, state, out, indices,
            True, bv=32, num_warps=1, num_stages=3, load_order=None,
        )

    actual_out, returned_state = launch()
    torch.cuda.synchronize()
    assert returned_state.data_ptr() == state.data_ptr()
    torch.testing.assert_close(actual_out.float(), expected_out, atol=3e-3, rtol=3e-2)
    torch.testing.assert_close(state, expected_state, atol=3e-4, rtol=3e-3)

    warmup, repeat, rounds = 100, 1000, 5
    samples_us = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(rounds):
        state.copy_(initial_state)
        for _ in range(warmup):
            launch()
        torch.cuda.synchronize()
        start.record()
        for _ in range(repeat):
            launch()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000 / repeat)

    print(
        f"\nPERF packed_decode B={B} H={H} HV={HV} K={D} V={D} "
        f"BV=32 warps=1 stages=3 load_order=auto "
        f"median_us={statistics.median(samples_us):.3f} "
        f"min_us={min(samples_us):.3f} max_us={max(samples_us):.3f} "
        f"warmup={warmup} repeat={repeat} rounds={rounds}"
    )


@pytest.mark.parametrize("varlen", [False, True])
def test_forward_and_verify(varlen):
    torch.manual_seed(23)
    B, T, H, HV, D = 2, 3, 2, 4, 32
    q = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn(B, T, HV, D, device="cuda", dtype=q.dtype) * 0.1
    g = -torch.rand(B, T, HV, device="cuda")
    beta = torch.ones_like(v[..., 0]).contiguous()
    state = torch.randn(B, HV, D, D, device="cuda") * 0.01
    expected, final, history = reference(q, k, v, g, beta, state)
    cu = torch.tensor([0, T, 2 * T], device="cuda", dtype=torch.int32) if varlen else None
    if varlen:
        q, k, v, g, beta = [x.reshape(1, B * T, *x.shape[2:]) for x in (q, k, v, g, beta)]
    out, ht = fused_recurrent_gated_delta_rule(q, k, v, g, initial_state=state,
        output_final_state=True, cu_seqlens=cu)
    torch.testing.assert_close(out.float().reshape_as(expected), expected, atol=3e-3, rtol=3e-2)
    torch.testing.assert_close(ht, final, atol=3e-4, rtol=3e-3)
    original = state.clone()
    cache = torch.empty(B, T, HV, D, D, device="cuda")
    idx = torch.arange(B, device="cuda", dtype=torch.int32)
    out = fused_recurrent_gated_delta_rule_update(q, k, v, g, beta,
        initial_state_source=state, initial_state_indices=idx, cu_seqlens=cu,
        disable_state_update=True, intermediate_states_buffer=cache,
        intermediate_state_indices=idx)
    torch.testing.assert_close(out.float().reshape_as(expected), expected, atol=3e-3, rtol=3e-2)
    torch.testing.assert_close(cache, history, atol=3e-4, rtol=3e-3)
    torch.testing.assert_close(state, original, atol=0, rtol=0)
