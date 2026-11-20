"""Correctness tests for the independent MoeInitRoutingV4 extension.

CUDA_VISIBLE_DEVICES=<idle GPU> python -m pytest -q this_file.py
Set MCOP_ROUTING_FULL=1 to include the 48 BF16 target shapes.
MCOP_ROUTING_EXTENSION selects an explicit op.so; otherwise use the local build.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import os
from pathlib import Path
import sysconfig

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_extension():
    explicit = os.environ.get("MCOP_ROUTING_EXTENSION")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    tag = sysconfig.get_python_version().replace(".", "")
    path = Path(explicit) if explicit else ROOT / f"build/temp.linux-x86_64-cpython-{tag}/op{suffix}"
    if not path.is_file():
        raise FileNotFoundError(f"Build the default op module or set MCOP_ROUTING_EXTENSION: {path}")
    spec = importlib.util.spec_from_file_location("op", path.resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "moe_init_routing_v4_out"), f"Stale extension: {path}"
    return module


@pytest.fixture(scope="session")
def ops():
    if not torch.cuda.is_available():
        pytest.skip("GPU unavailable")
    module = load_extension()
    torch.manual_seed(20261021)
    print(f"extension={module.__file__}; device={torch.cuda.get_device_name(0)}")
    return module


def allocate_workspace(ops, x, ids, experts, mode=-1):
    return torch.empty(ops.moe_init_routing_v4_workspace_size(x, ids, experts, mode),
                       dtype=torch.uint8, device=x.device)


def permutation(ids, experts, active_range=None):
    flat = ids.cpu().reshape(-1).long()
    lo, hi = active_range or (0, experts)
    valid = (flat >= lo) & (flat < hi)
    keys = torch.where(valid, flat, experts)
    order = torch.argsort(keys, stable=True)
    order = order[:int(valid.sum())]
    counts = torch.bincount(flat[valid], minlength=experts)[lo:hi]
    return order, counts


def check_routing(x, ids, result, experts, *, row_idx_type=0,
                  expert_tokens_num_type=1, expert_tokens_num_flag=True,
                  active_num=0, active_expert_range=None, scale=None,
                  topk_weight=None, quant_mode=-1):
    data, rows, counts, scales, weights = result
    order, expected_counts = permutation(ids, experts, active_expert_range)
    capacity = min(active_num, ids.numel()) if active_num else ids.numel()
    effective = min(capacity, len(order))
    expected_rows = torch.full((ids.numel(),), -1, dtype=torch.int32)
    if row_idx_type == 1:
        expected_rows[:len(order)] = order.int()
    else:
        expected_rows[order] = torch.arange(len(order), dtype=torch.int32)
    torch.testing.assert_close(rows.cpu(), expected_rows, rtol=0, atol=0)
    if expert_tokens_num_flag:
        if expert_tokens_num_type == 0:
            expected_counts = expected_counts.cumsum(0)
        torch.testing.assert_close(counts.cpu(), expected_counts, rtol=0, atol=0)
    else:
        assert counts.numel() == 0
    assert data.shape[0] == capacity
    if quant_mode == -1:
        assert data.dtype == x.dtype and data.shape[1] == x.shape[1]
        if effective and x.shape[1]:
            source = x.view(torch.uint8)
            actual = data.view(torch.uint8)
            for begin in range(0, effective, 1024):
                end = min(effective, begin + 1024)
                tokens = (order[begin:end] // ids.shape[1]).to(x.device)
                torch.testing.assert_close(actual[begin:end], source[tokens], rtol=0, atol=0)
        if scale is not None and effective:
            tokens = (order[:effective] // ids.shape[1]).to(x.device)
            actual_bytes = scales.view(torch.uint8).reshape(scales.shape[0], -1)
            source_bytes = scale.view(torch.uint8).reshape(scale.shape[0], -1)
            torch.testing.assert_close(actual_bytes[:effective], source_bytes[tokens], rtol=0, atol=0)
        elif scale is None:
            assert scales.numel() == 0
    if topk_weight is not None:
        torch.testing.assert_close(weights[:effective, 0].cpu(),
                                   topk_weight.cpu().reshape(-1)[order[:effective]], rtol=0, atol=0)
    else:
        assert weights.numel() == 0
    return order[:effective]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32,
                                   torch.int8, torch.float8_e4m3fn, torch.float8_e5m2,
                                   torch.float4_e2m1fn_x2])
@pytest.mark.parametrize("row_type,count_type", itertools.product((0, 1), (0, 1)))
def test_passthrough(ops, dtype, row_type, count_type):
    # Raw bytes test preservation of low precision payloads and floating NaNs.
    x = torch.empty((17, 66), dtype=dtype, device="cuda")
    x.view(torch.uint8).copy_(torch.randint(0, 256, x.view(torch.uint8).shape,
                                           dtype=torch.uint8, device="cuda"))
    ids = torch.randint(0, 11, (17, 5), dtype=torch.int32, device="cuda")
    weights = torch.randn(17, 5, device="cuda")
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2):
        hidden = x.shape[1] * (2 if dtype == torch.float4_e2m1fn_x2 else 1)
        scale = torch.randint(0, 255, (17, (hidden + 63) // 64, 2),
                              dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
    else:
        scale = torch.randn(17, device="cuda")
    kw = dict(row_idx_type=row_type, expert_tokens_num_type=count_type,
              scale=scale, topk_weight=weights)
    result = ops.moe_init_routing_v4(x, ids, 11, **kw)
    check_routing(x, ids, result, 11, **kw)
    workspace = allocate_workspace(ops, x, ids, 11)
    out = [torch.empty_like(t) for t in result]
    ops.moe_init_routing_v4_out(x, ids, *out, workspace, 11, **kw)
    check_routing(x, ids, out, 11, **kw)


def check_bf16_token_scatter_layout(ops, hidden, capacity, row_type=0,
                                    input_offset=0, output_offset=0,
                                    empty_range=False, topk=20):
    n, experts = 17, 11
    byte_count = n * hidden * 2
    source_storage = torch.full((byte_count + input_offset + 256,), 0xa5,
                                dtype=torch.uint8, device="cuda")
    source = source_storage[input_offset:input_offset + byte_count].reshape(n, hidden * 2)
    # H4096 covers all65536 BF16 source bit patterns, including NaNs and zeros.
    bits = torch.arange(n * hidden, dtype=torch.int32, device="cuda").remainder(65536).to(torch.int16)
    source.copy_(bits.view(torch.uint8).reshape_as(source))
    source_before = source_storage.clone()
    x = source.view(torch.bfloat16)
    ids = torch.arange(n * topk, device="cuda").reshape(n, topk).remainder(experts).int()
    if topk > 1:
        ids[:, 1] = ids[:, 0]  # Duplicate experts still have distinct output ranks.
    ids[0, 0], ids[-1, -1] = -1, experts + 3
    kw = dict(active_num=capacity, active_expert_range=[2, 2] if empty_range else [2, 9],
              row_idx_type=row_type, expert_tokens_num_type=row_type,
              scale=torch.arange(n, device="cuda", dtype=torch.float32),
              topk_weight=torch.arange(n * topk, device="cuda", dtype=torch.float32).reshape(n, topk))
    result = ops.moe_init_routing_v4(x, ids, experts, **kw)
    check_routing(x, ids, result, experts, **kw)
    out = [torch.empty_like(t) for t in result]
    size = result[0].nbytes
    storage = torch.full((size + output_offset + 256,), 0xa5,
                         dtype=torch.uint8, device="cuda")
    out[0] = storage[output_offset:output_offset + size].view(x.dtype).reshape(result[0].shape)
    assert x.data_ptr() % 128 == input_offset % 128
    assert out[0].data_ptr() % 128 == output_offset % 128
    for tensor in out:
        tensor.view(torch.uint8).fill_(0xa5)
    workspace = allocate_workspace(ops, x, ids, experts)
    ops.moe_init_routing_v4_out(x, ids, *out, workspace, experts, **kw)
    order = check_routing(x, ids, out, experts, **kw)
    for index in (0, 3, 4):
        assert bool((out[index][len(order):].view(torch.uint8) == 0xa5).all())
    assert bool((storage[:output_offset] == 0xa5).all())
    assert bool((storage[output_offset + size:] == 0xa5).all())
    torch.testing.assert_close(source_storage, source_before, rtol=0, atol=0)


@pytest.mark.parametrize("hidden,capacity", [(h, 17) for h in
    (1984, 2047, 2048, 2049, 2112, 3008, 3072, 4032, 4095, 4096, 4097, 4160)]
    + [(4096, c) for c in (0, 1, 2, 3, 339)])
@pytest.mark.parametrize("row_type", [0, 1])
def test_bf16_token_scatter_boundaries(ops, hidden, capacity, row_type):
    check_bf16_token_scatter_layout(ops, hidden, capacity, row_type)


@pytest.mark.parametrize("input_offset", [0, 16])
@pytest.mark.parametrize("output_offset", [0, 2, 16, 128])
def test_bf16_token_scatter_alignment(ops, input_offset, output_offset):
    check_bf16_token_scatter_layout(ops, 4096, 339,
                                    input_offset=input_offset, output_offset=output_offset)


@pytest.mark.parametrize("row_type", [0, 1])
def test_bf16_token_scatter_empty_range(ops, row_type):
    check_bf16_token_scatter_layout(ops, 4096, 339, row_type, empty_range=True)


@pytest.mark.parametrize("topk", [1, 2, 7, 15, 25, 30, 33])
def test_bf16_token_scatter_topk(ops, topk):
    check_bf16_token_scatter_layout(ops, 4096, 3, topk=topk)


def check_fp4_passthrough_layout(ops, hidden, capacity, with_scale,
                                 input_offset=0, output_offset=0, empty_range=False):
    n, packed = 12, hidden // 2
    source_storage = torch.full((n * packed + input_offset + 256,), 0xa5,
                                dtype=torch.uint8, device="cuda")
    source = source_storage[input_offset:input_offset + n * packed].reshape(n, packed)
    source.copy_(torch.randint(0, 256, source.shape, dtype=torch.uint8, device="cuda"))
    source_before = source_storage.clone()
    x = source.view(torch.float4_e2m1fn_x2)
    ids = torch.tensor([[7, -1, 3], [2, 3, 5], [1, 13, 3]] * 4,
                       dtype=torch.int32, device="cuda")
    scale = (torch.randint(0, 256, (n, (hidden + 63) // 64, 2),
                           dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
             if with_scale else None)
    kw = dict(active_num=capacity, active_expert_range=[2, 2] if empty_range else [2, 6],
              row_idx_type=int(with_scale), expert_tokens_num_type=int(with_scale),
              scale=scale, topk_weight=torch.arange(n * 3, device="cuda", dtype=torch.float32).reshape(n, 3))
    expected = ops.moe_init_routing_v4(x, ids, 8, **kw)
    check_routing(x, ids, expected, 8, **kw)
    outputs = [torch.empty_like(t) for t in expected]
    size = expected[0].nbytes
    storage = torch.full((size + output_offset + 256,), 0xa5,
                         dtype=torch.uint8, device="cuda")
    outputs[0] = storage[output_offset:output_offset + size].view(x.dtype).reshape(expected[0].shape)
    assert outputs[0].is_contiguous() and outputs[0].data_ptr() % 128 == output_offset % 128
    assert x.is_contiguous() and x.data_ptr() % 128 == input_offset % 128
    for tensor in outputs:
        tensor.view(torch.uint8).fill_(0xa5)
    workspace = allocate_workspace(ops, x, ids, 8)
    ops.moe_init_routing_v4_out(x, ids, *outputs, workspace, 8, **kw)
    order = check_routing(x, ids, outputs, 8, **kw)
    for index in (0, 3, 4):
        assert bool((outputs[index][len(order):].view(torch.uint8) == 0xa5).all())
    assert bool((storage[:output_offset] == 0xa5).all())
    assert bool((storage[output_offset + size:] == 0xa5).all())
    torch.testing.assert_close(source_storage, source_before, rtol=0, atol=0)


@pytest.mark.parametrize("hidden", [4094, 4096, 4098, 4352, 5888, 6144, 7936, 8160, 8192, 8194])
@pytest.mark.parametrize("capacity", [1, 2, 3, 33])
@pytest.mark.parametrize("with_scale", [False, True])
def test_fp4_passthrough_rows2_boundaries(ops, hidden, capacity, with_scale):
    check_fp4_passthrough_layout(ops, hidden, capacity, with_scale)


@pytest.mark.parametrize("input_offset", [0, 16])
@pytest.mark.parametrize("output_offset", [0, 1, 16, 128])
@pytest.mark.parametrize("with_scale", [False, True])
def test_fp4_passthrough_rows2_alignment(ops, input_offset, output_offset, with_scale):
    check_fp4_passthrough_layout(ops, 4096, 33, with_scale, input_offset, output_offset)


@pytest.mark.parametrize("with_scale", [False, True])
def test_fp4_passthrough_rows2_empty_range(ops, with_scale):
    check_fp4_passthrough_layout(ops, 4096, 33, with_scale, empty_range=True)


@pytest.mark.parametrize("topk", [1, 2, 7, 15, 20, 25, 30, 31, 32, 33])
@pytest.mark.parametrize("mode", [-1, 3, 5, 9])
def test_data_gather_topk_routing(ops, topk, mode):
    n, hidden, experts = 17, 4096, 11
    if mode == -1:
        values = torch.arange(n * (hidden // 2), dtype=torch.int32).reshape(n, -1)
        values += torch.arange(n, dtype=torch.int32)[:, None] * 17
        x = values.remainder(256).to(torch.uint8).cuda().view(torch.float4_e2m1fn_x2)
    else:
        values = torch.arange(n * hidden, dtype=torch.float32).reshape(n, hidden)
        x = ((values.remainder(97) - 48) / 8).to(torch.bfloat16).cuda()
    ids = torch.arange(n * topk, device="cuda").reshape(n, topk).remainder(experts).int()
    ids[0, 0], ids[-1, -1] = -1, experts + 2
    kw = dict(quant_mode=mode, active_num=max(1, ids.numel() // 2),
              active_expert_range=[1, 10], row_idx_type=topk & 1)
    result = ops.moe_init_routing_v4(x, ids, experts, **kw)
    expected_data, expected_scale = quantization_reference(x, mode) if mode != -1 else (None, None)
    out = [torch.empty_like(t) for t in result]
    for tensor in out:
        tensor.view(torch.uint8).fill_(0xa5)
    workspace = allocate_workspace(ops, x, ids, experts, mode)
    ops.moe_init_routing_v4_out(x, ids, *out, workspace, experts, **kw)
    for outputs in (result, out):
        order = check_routing(x, ids, outputs, experts, **kw)
        if mode != -1:
            tokens = order // topk
            torch.testing.assert_close(outputs[0][:len(order)].view(torch.uint8).cpu(),
                                       expected_data[tokens], rtol=0, atol=0)
            torch.testing.assert_close(outputs[3][:len(order)].view(torch.uint8).cpu(),
                                       expected_scale[tokens].view(torch.uint8), rtol=0, atol=0)
    for index in (0, 3):
        assert bool((out[index][len(order):].view(torch.uint8) == 0xa5).all())


@pytest.mark.parametrize("shape", [(0, 16, 3), (5, 16, 0), (3, 0, 2), (1, 1, 1),
                                    (33, 257, 15)])
def test_empty_and_tails(ops, shape):
    n, h, k = shape
    x = torch.randn(n, h, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, 19, (n, k), dtype=torch.int32, device="cuda")
    result = ops.moe_init_routing_v4(x, ids, 19)
    check_routing(x, ids, result, 19)


@pytest.mark.parametrize("row_type", [0, 1])
@pytest.mark.parametrize("active_range", [[2, 6], [0, 0], [0, 8]])
def test_active_range_and_cap(ops, row_type, active_range):
    x = torch.randn(9, 129, dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor([[7, 1, 3], [2, 3, 5], [1, 7, 3]] * 3, dtype=torch.int32, device="cuda")
    kw = dict(active_num=7, active_expert_range=active_range, row_idx_type=row_type)
    result = ops.moe_init_routing_v4(x, ids, 8, **kw)
    check_routing(x, ids, result, 8, **kw)


@pytest.mark.parametrize("experts", [1, 2, 3, 255, 256, 257, 511, 512, 513,
                                      800, 1000, 1023, 1024, 1025, 1200,
                                      8191, 8192, 8193, 10240])
@pytest.mark.parametrize("row_type", [0, 1])
def test_routing_radix_boundaries(ops, experts, row_type):
    # Include the sentinel's extra bit at powers of two, interleaved duplicate
    # keys for stability, and invalid IDs that must be replaced before sorting.
    # 85 routes use a single tile; 5125 routes exercise the multi-tile path.
    tokens, topk = (17, 5) if row_type == 0 else (1025, 5)
    x = torch.arange(tokens * 4, dtype=torch.float32).reshape(tokens, 4).to(torch.bfloat16).cuda()
    choices = torch.tensor([experts - 1, 0, experts // 2, experts - 1, 0,
                            -1, experts, -(2**31), 2**31 - 1,
                            experts // 2, experts - 1], dtype=torch.int32, device="cuda")
    ids = choices[torch.arange(tokens * topk, device="cuda") % len(choices)].reshape(tokens, topk)
    scale = torch.arange(tokens, dtype=torch.float32, device="cuda")
    weights = torch.arange(tokens * topk, dtype=torch.float32, device="cuda").reshape(tokens, topk)
    kw = dict(row_idx_type=row_type, expert_tokens_num_type=row_type,
              active_num=19 if row_type else 0,
              active_expert_range=[experts // 3, experts] if row_type else [],
              scale=scale, topk_weight=weights)
    result = ops.moe_init_routing_v4(x, ids, experts, **kw)
    order = check_routing(x, ids, result, experts, **kw)
    # Pass exactly the queried workspace. A canary detects a query/execute
    # bit-range mismatch that writes beyond the required workspace allocation.
    size = ops.moe_init_routing_v4_workspace_size(x, ids, experts)
    storage = torch.full((size + 256,), 0xa5, dtype=torch.uint8, device="cuda")
    workspace = storage[:size]
    out = [torch.empty_like(t) for t in result]
    for tensor in out:
        tensor.view(torch.uint8).fill_(0xa5)
    ops.moe_init_routing_v4_out(x, ids, *out, workspace, experts, **kw)
    check_routing(x, ids, out, experts, **kw)
    assert bool((storage[size:] == 0xa5).all())
    # Invalid-route/capacity tails must remain unwritten. Metadata still covers
    # all valid routes; topk weights identify original flattened route order.
    for index in (0, 3, 4):
        assert bool((out[index][len(order):].view(torch.uint8) == 0xa5).all())


@pytest.mark.parametrize("routes,experts", [
    (n, e) for n in [4095, 4096, 4097, 8191, 8192, 8193]
    for e in [1, 800, 1200, 2047, 2048]
] + [(8193, 2), (8193, 2046), (12347, 1537)])
@pytest.mark.parametrize("row_type", [0, 1])
@pytest.mark.parametrize("pattern", ["mixed", "hotspot", "invalid"])
def test_counting_sort_boundaries(ops, routes, experts, row_type, pattern):
    # Check the production contract against the independent stable CPU order.
    # Cover dispatch neighbors, complete/partial tiles, and repeated keys that
    # span waves/tiles, rather than asserting the implementation's layout.
    r = torch.arange(routes, device="cuda")
    x = torch.arange(routes * 4, device="cuda", dtype=torch.float32).reshape(routes, 4).bfloat16()
    if pattern == "mixed":
        choices = torch.tensor([experts - 1, 0, experts // 2, experts - 1,
                                -1, experts, -(2**31), 2**31 - 1, 0],
                               dtype=torch.int32, device="cuda")
        ids = choices[r % choices.numel()]
    elif pattern == "hotspot":
        ids = torch.where(r % 17 == 0, experts // 2, experts - 1).int()
    else:
        ids = torch.full((routes,), -1, dtype=torch.int32, device="cuda")
    ids = ids.reshape(routes, 1)
    weights = r.float().reshape(routes, 1)
    kw = dict(row_idx_type=row_type, expert_tokens_num_type=row_type,
              active_num=routes // 3 if row_type else 0,
              active_expert_range=[experts // 3, experts] if row_type else [],
              topk_weight=weights)
    result = ops.moe_init_routing_v4(x, ids, experts, **kw)
    order = check_routing(x, ids, result, experts, **kw)
    size = ops.moe_init_routing_v4_workspace_size(x, ids, experts)
    storage = torch.full((size + 256,), 0xa5, dtype=torch.uint8, device="cuda")
    out = [torch.empty_like(t) for t in result]
    for tensor in out:
        tensor.view(torch.uint8).fill_(0xa5)
    ops.moe_init_routing_v4_out(x, ids, *out, storage[:size], experts, **kw)
    check_routing(x, ids, out, experts, **kw)
    assert bool((storage[size:] == 0xa5).all())
    for index in (0, 3, 4):
        assert bool((out[index][len(order):].view(torch.uint8) == 0xa5).all())


@pytest.mark.parametrize("routes,experts", [
    (n, 1000) for n in [524287, 524288, 524289, 2457599, 2457600, 2457601]
] + [(524289, 10240), (786433, 1200), (1048577, 800), (2457599, 1200)])
@pytest.mark.parametrize("row_type", [0, 1])
@pytest.mark.parametrize("pattern", ["full", "mixed", "empty"])
@pytest.mark.parametrize("with_weights", [False, True])
def test_inverse_sort_boundaries(ops, routes, experts, row_type, pattern, with_weights):
    # A cyclic expert sequence has a closed-form stable inverse permutation.
    # Use H=0 to exercise millions of routes without a large feature allocation.
    original = torch.arange(routes, dtype=torch.int64)
    expert = original % experts
    ids_cpu = expert.int()
    lo, hi = (0, experts) if pattern == "full" else (experts // 3, experts - 2)
    if pattern == "mixed":
        ids_cpu[expert == experts - 1] = -1
        ids_cpu[expert == experts - 2] = experts
    if pattern == "empty":
        lo, hi = 0, 0
    quotient, remainder = divmod(routes, experts)
    valid = (expert >= lo) & (expert < hi)
    prefix = expert * quotient + expert.clamp_max(remainder)
    inverse = torch.where(valid, prefix - (lo * quotient + min(lo, remainder)) +
                          original // experts, -1).int()
    expected_rows = inverse
    if row_type:
        expected_rows = torch.full((routes,), -1, dtype=torch.int32)
        expected_rows[inverse[valid].long()] = original[valid].int()
    expected_counts = quotient + (torch.arange(experts) < remainder).long()
    expected_counts = expected_counts[lo:hi]
    if row_type == 0:
        expected_counts = expected_counts.cumsum(0)
    x = torch.empty((routes, 0), dtype=torch.bfloat16, device="cuda")
    ids = ids_cpu.reshape(routes, 1).cuda()
    weight = original.float().reshape(routes, 1).cuda() if with_weights else None
    kw = dict(active_num=19, active_expert_range=[lo, hi], row_idx_type=row_type,
              expert_tokens_num_type=row_type, topk_weight=weight)
    result = ops.moe_init_routing_v4(x, ids, experts, **kw)
    size = ops.moe_init_routing_v4_workspace_size(x, ids, experts)
    storage = torch.full((size + 256,), 0xa5, dtype=torch.uint8, device="cuda")
    outputs = [torch.empty_like(t) for t in result]
    for tensor in outputs:
        tensor.view(torch.uint8).fill_(0xa5)
    ops.moe_init_routing_v4_out(x, ids, *outputs, storage[:size], experts, **kw)
    effective = min(19, int(valid.sum()))
    # Every present expert has at least19 routes in these cases.
    expected_weights = lo + torch.arange(effective, dtype=torch.float32) * experts
    for actual in (result, outputs):
        torch.testing.assert_close(actual[1].cpu(), expected_rows, rtol=0, atol=0)
        torch.testing.assert_close(actual[2].cpu(), expected_counts, rtol=0, atol=0)
        if with_weights:
            torch.testing.assert_close(actual[4][:effective, 0].cpu(), expected_weights, rtol=0, atol=0)
        else:
            assert actual[4].numel() == 0
    assert bool((storage[size:] == 0xa5).all())
    assert bool((outputs[4][effective:].view(torch.uint8) == 0xa5).all())


def test_validation_and_overlap(ops):
    x = torch.randn(7, 16, dtype=torch.bfloat16, device="cuda")
    ids = torch.zeros(7, 1, dtype=torch.int32, device="cuda")
    for kw in (dict(expert_num=0), dict(expert_num=8, row_idx_type=2),
               dict(expert_num=8, active_num=-1), dict(expert_num=8, quant_mode=123),
               dict(expert_num=8, active_expert_range=[0, 9]),
               dict(expert_num=8, expert_tokens_num_type=2)):
        with pytest.raises((RuntimeError, ValueError)):
            ops.moe_init_routing_v4(x, ids, **kw)
    for bad_x, bad_ids in ((x.t(), ids), (x, ids.long()), (x.cpu(), ids),
                           (x[:, ::2], ids), (x.detach().requires_grad_(), ids)):
        with pytest.raises((RuntimeError, ValueError)):
            ops.moe_init_routing_v4(bad_x, bad_ids, 8)
    result = ops.moe_init_routing_v4(x, ids, 8)
    ws = allocate_workspace(ops, x, ids, 8)
    with pytest.raises(RuntimeError):
        ops.moe_init_routing_v4_out(x, ids, *result, ws[:-1], 8)
    with pytest.raises(RuntimeError):
        ops.moe_init_routing_v4_out(x, ids, x, *result[1:], ws, 8)


def quantization_reference(x, mode):
    """CPU reference from the fixed upstream scale and rounding contracts.

    Scale powers are derived mathematically; E2M1 uses nearest-level distances.
    Input and output here describe one row per original token, before routing.
    """
    source = x.cpu()
    values = source.float()
    n, hidden = values.shape
    group_size = 128 if mode in (4, 5) else 32
    groups = (hidden + group_size - 1) // group_size
    scale_columns = groups if mode in (4, 5) else (hidden + 63) // 64 * 2
    scales = torch.zeros((n, scale_columns), dtype=torch.float32 if mode in (4, 5) else torch.uint8)
    encoded = torch.empty((n, hidden // 2 if mode == 9 else hidden), dtype=torch.uint8)
    output_dtype = torch.float8_e5m2 if mode in (2, 4) else torch.float8_e4m3fn
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)
    for token in range(n):
        for group in range(groups):
            lo, hi = group * group_size, min(hidden, (group + 1) * group_size)
            v = values[token, lo:hi]
            scale_values = v
            if mode not in (4, 5) and source.dtype == torch.float16:
                # Upstream FP16 scale selection truncates to BF16.
                scale_values = (v.view(torch.int32) & -65536).view(torch.float32)
            amax = float(scale_values.abs().max())
            emax = 2 if mode == 9 else (15 if mode in (2, 4) else 8)
            if amax == 0:
                code = 1 if mode in (4, 5) else 0
            else:
                mantissa, exponent = math.frexp(amax)
                code = exponent - 1 - emax + 127
                if mode != 9:
                    code += int(mantissa > .875)
                    code = code if 1 <= code <= 254 else 1
                else:
                    code = max(code, 0)
            factor = math.ldexp(1., code - 127)
            if mode in (4, 5):
                scales[token, group] = factor
            else:
                scales[token, group] = code
            reciprocal = math.ldexp(1., 127 - code)
            if amax == 0 and mode in (2, 3):
                reciprocal = 0.
            scaled = v * reciprocal
            if mode == 9 or (mode in (2, 3) and source.dtype == torch.bfloat16):
                scaled = scaled.to(torch.bfloat16).float()
            if mode == 9:
                distances = (scaled.abs().double()[:, None] - levels).abs()
                # Resolve equidistant levels toward the even encoding.
                tied = distances == distances.min(dim=1, keepdim=True).values
                ranks = torch.arange(8) + (torch.arange(8) % 2) * 8
                codes = torch.where(tied, ranks, 100).argmin(dim=1).to(torch.uint8)
                codes |= torch.signbit(scaled).to(torch.uint8) * 8
                encoded[token, lo // 2:hi // 2] = codes[::2] | (codes[1::2] << 4)
            else:
                maximum = torch.finfo(output_dtype).max
                encoded[token, lo:hi] = scaled.clamp(-maximum, maximum).to(output_dtype).view(torch.uint8)
    if mode == 9:
        scales = scales.reshape(n, (hidden + 63) // 64, 2)
    return encoded, scales


@pytest.mark.parametrize("mode", [2, 3, 4, 5, 9])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden", [0, 2, 30, 32, 34, 128, 130, 258])
def test_quantization(ops, mode, dtype, hidden):
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(7, hidden, generator=generator).to(dtype).cuda()
    if hidden:
        x[0].zero_()
        # Exact powers, midpoint ties, saturation, and signed zero.
        edge = torch.tensor([-.0, .25, -.75, 1.25, -1.75, 2.5, -3.5, 5., -6., 7., 8., 448.],
                            dtype=dtype, device="cuda")
        x[1, :min(hidden, len(edge))] = edge[:hidden]
    ids = torch.randint(0, 13, (7, 3), dtype=torch.int32, device="cuda")
    kw = dict(quant_mode=mode, topk_weight=torch.randn(7, 3, device="cuda"))
    expected_data, expected_scale = quantization_reference(x, mode)
    result = ops.moe_init_routing_v4(x, ids, 13, **kw)
    order = check_routing(x, ids, result, 13, **kw)
    tokens = order // ids.shape[1]
    torch.testing.assert_close(result[0].view(torch.uint8).cpu(), expected_data[tokens], rtol=0, atol=0)
    actual_scale = result[3].cpu()
    if mode not in (4, 5):
        actual_scale = actual_scale.view(torch.uint8)
    torch.testing.assert_close(actual_scale, expected_scale[tokens], rtol=0, atol=0)
    ws = allocate_workspace(ops, x, ids, 13, mode)
    ws.fill_(0xa5)
    out = [torch.empty_like(t) for t in result]
    ops.moe_init_routing_v4_out(x, ids, *out, ws, 13, **kw)
    for actual, expected in zip(out, result):
        torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode,hidden", [
    (3, h) for h in (1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 4095, 4097)
] + [
    (9, h) for h in (2, 30, 32, 34, 62, 64, 66, 126, 128, 130, 4094, 4098)
])
def test_mx_subgroup_boundaries(ops, dtype, mode, hidden):
    # Three source rows put the last four-group CTA tile at every padding
    # position. Odd FP8 widths and even FP4 tails exercise partial groups;
    # distinct group maxima detect exchange across adjacent 32-lane groups.
    x = torch.arange(3 * hidden, dtype=torch.float32).reshape(3, hidden)
    x = ((x.remainder(19) - 9) / 2).to(dtype).cuda()
    for group in range((hidden + 31) // 32):
        x[:, group * 32:min(hidden, (group + 1) * 32)] *= 2.0 ** (group % 7 - 3)
    x[0].zero_()
    x[0, 0] = -0.0
    ids = torch.tensor([[4, 1, 4], [0, 4, 1], [1, 0, 4]], dtype=torch.int32, device="cuda")
    expected_data, expected_scale = quantization_reference(x, mode)
    result = ops.moe_init_routing_v4(x, ids, 5, quant_mode=mode)
    order = check_routing(x, ids, result, 5, quant_mode=mode)
    tokens = order // ids.shape[1]
    torch.testing.assert_close(result[0].view(torch.uint8).cpu(), expected_data[tokens], rtol=0, atol=0)
    torch.testing.assert_close(result[3].view(torch.uint8).cpu(), expected_scale[tokens], rtol=0, atol=0)
    workspace = allocate_workspace(ops, x, ids, 5, mode)
    out = [torch.empty_like(t) for t in result]
    ops.moe_init_routing_v4_out(x, ids, *out, workspace, 5, quant_mode=mode)
    for actual, expected in zip(out, result):
        torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)


def _check_mx_vector_case(ops, dtype, mode, hidden, tokens, *, unaligned=False,
                          input_offset=0):
    source = torch.arange(tokens * hidden, dtype=torch.float32).reshape(tokens, hidden)
    source = ((source.remainder(31) - 15) / 8).to(dtype)
    threshold = torch.tensor(3.5, dtype=dtype)
    above = torch.nextafter(threshold, torch.full_like(threshold, float("inf"))).item()
    for group in range(hidden // 32):
        factor = 2.0 ** (group % 7 - 3)
        source[:, group * 32:(group + 1) * 32] *= factor
        # Put the maximum in each of the eight lanes and four local components.
        source[-1, group * 32 + (group % 8) * 4 + (group // 8) % 4] = above * factor
    if tokens > 1:
        source[0].zero_()
    source[0, 0] = -0.0
    x = source.cuda()
    if unaligned:
        input_offset = 1
    if input_offset:
        storage = torch.full((x.numel() + input_offset + 1,), 7,
                             dtype=dtype, device="cuda")
        x = storage[input_offset:-1].reshape(source.shape).copy_(x)
        assert x.is_contiguous() and x.data_ptr() % 16 == (input_offset * 2) % 16
    ids = (torch.arange(tokens * 3, device="cuda").reshape(tokens, 3) % 5).int()
    kw = dict(quant_mode=mode, row_idx_type=int(dtype == torch.float16),
              topk_weight=torch.arange(tokens * 3, device="cuda", dtype=torch.float32).reshape(tokens, 3))
    expected_data, expected_scale = quantization_reference(x, mode)
    result = ops.moe_init_routing_v4(x, ids, 5, **kw)
    order = check_routing(x, ids, result, 5, **kw)
    torch.testing.assert_close(result[0].view(torch.uint8).cpu(),
                               expected_data[order // 3], rtol=0, atol=0)
    torch.testing.assert_close(result[3].view(torch.uint8).cpu(),
                               expected_scale[order // 3], rtol=0, atol=0)
    workspace = allocate_workspace(ops, x, ids, 5, mode)
    out = [torch.empty_like(t) for t in result]
    for tensor in out:
        tensor.view(torch.uint8).fill_(0xa5)
    ops.moe_init_routing_v4_out(x, ids, *out, workspace, 5, **kw)
    for actual, expected in zip(out, result):
        torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(x.cpu().view(torch.uint8), source.view(torch.uint8), rtol=0, atol=0)
    if input_offset:
        assert bool((storage[:input_offset] == 7).all()) and storage[-1].item() == 7


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", [3, 9])
@pytest.mark.parametrize("hidden,tokens", [(32, 1), (32, 15), (32, 16), (32, 17),
                                           (64, 3), (96, 3), (480, 1), (512, 1),
                                           (544, 1), (4096, 3)])
def test_mx_vector_tiles(ops, dtype, mode, hidden, tokens):
    # Total groups1/15/16/17 and partial CTA tiles, including odd scale padding.
    _check_mx_vector_case(ops, dtype, mode, hidden, tokens)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", [3, 9])
def test_mx_vector_unaligned_input(ops, dtype, mode):
    _check_mx_vector_case(ops, dtype, mode, 4096, 3, unaligned=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden,tokens", [(32, 31), (32, 32), (32, 33),
                                           (96, 11), (1056, 1), (2048, 1)])
@pytest.mark.parametrize("mode", [3, 9])
@pytest.mark.parametrize("input_offset", [0, 4])
def test_mx_vectors8_tiles(ops, dtype, hidden, tokens, mode, input_offset):
    # 31/32/33 groups cross the new CTA boundary; odd groups exercise scale
    # padding. An 8B input offset keeps the four-value fallback active.
    _check_mx_vector_case(ops, dtype, mode, hidden, tokens,
                          input_offset=input_offset)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode,hidden", [(3, 2048), (3, 4096), (9, 4096), (9, 8192)])
@pytest.mark.parametrize("shuffled", [False, True])
def test_mx_all_input_patterns(ops, dtype, mode, shuffled, hidden):
    path = os.environ.get("MCOP_ROUTING_REFERENCE_EXTENSION")
    if not path:
        pytest.skip("set MCOP_ROUTING_REFERENCE_EXTENSION to frozen Q5")
    spec = importlib.util.spec_from_file_location("routing_reference.op", Path(path).resolve())
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    assert reference.moe_init_routing_v4_out is not ops.moe_init_routing_v4_out
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    if shuffled:
        bits = bits[torch.randperm(len(bits), generator=torch.Generator().manual_seed(20261023))]
    x = bits.view(dtype).reshape(-1, hidden).cuda()
    ids = (torch.arange(x.shape[0] * 3, device="cuda") % 5).reshape(-1, 3).int()
    expected = reference.moe_init_routing_v4(x, ids, 5, quant_mode=mode)
    result = ops.moe_init_routing_v4(x, ids, 5, quant_mode=mode)
    workspace = allocate_workspace(ops, x, ids, 5, mode)
    out = [torch.empty_like(t) for t in result]
    ops.moe_init_routing_v4_out(x, ids, *out, workspace, 5, quant_mode=mode)
    for actual_outputs in (result, out):
        for actual, wanted in zip(actual_outputs, expected):
            torch.testing.assert_close(actual.view(torch.uint8), wanted.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden", [1, 63, 64, 65, 127, 128, 129, 255, 256, 257,
                                   2048, 3072, 4095, 4096, 4097])
def test_fp8_warp_boundaries(ops, dtype, hidden):
    # Odd total group counts exercise a padding wave. A maximum in the second
    # half detects a reduction that misses one of each lane's two input values.
    x = torch.arange(3 * hidden, dtype=torch.float32).reshape(3, hidden)
    x = ((x.remainder(19) - 9) / 8).to(dtype).cuda()
    threshold = torch.tensor(3.5, dtype=dtype)
    above = torch.nextafter(threshold, torch.full_like(threshold, float("inf"))).item()
    for group in range((hidden + 127) // 128):
        lo, hi = group * 128, min(hidden, (group + 1) * 128)
        x[2, lo:hi] *= 2.0 ** (group % 7 - 3)
        x[1, min(lo + 64, hi - 1)] = above
    x[0].zero_()
    x[0, 0] = -0.0
    ids = torch.tensor([[4, 1, 4], [0, 4, 1], [1, 0, 4]], dtype=torch.int32, device="cuda")
    expected_data, expected_scale = quantization_reference(x, 5)
    result = ops.moe_init_routing_v4(x, ids, 5, quant_mode=5)
    order = check_routing(x, ids, result, 5, quant_mode=5)
    tokens = order // ids.shape[1]
    torch.testing.assert_close(result[0].view(torch.uint8).cpu(), expected_data[tokens], rtol=0, atol=0)
    torch.testing.assert_close(result[3].view(torch.uint8).cpu(),
                               expected_scale[tokens].view(torch.uint8), rtol=0, atol=0)
    workspace = allocate_workspace(ops, x, ids, 5, 5)
    out = [torch.empty_like(t) for t in result]
    ops.moe_init_routing_v4_out(x, ids, *out, workspace, 5, quant_mode=5)
    for actual, expected in zip(out, result):
        torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden,tokens", [(128, 1), (128, 2), (128, 3), (128, 4), (128, 5), (256, 3),
                                           (384, 3), (3968, 3), (4096, 3), (4224, 3)])
@pytest.mark.parametrize("input_offset", [0, 1, 2])
def test_fp8_vector_pairs(ops, dtype, hidden, tokens, input_offset):
    # Full groups, four-group tile boundaries, and0/2/4-byte input offsets
    # exercise vector4/vector2/scalar dispatch. Maxima cover local components.
    source = torch.arange(tokens * hidden, dtype=torch.float32).reshape(tokens, hidden)
    source = ((source.remainder(31) - 15) / 8).to(dtype)
    threshold = torch.tensor(3.5, dtype=dtype)
    above = torch.nextafter(threshold, torch.full_like(threshold, float("inf"))).item()
    for token in range(tokens):
        for group in range(hidden // 128):
            factor = 2.0 ** (group % 7 - 3)
            source[token, group * 128:(group + 1) * 128] *= factor
            source[token, group * 128 + (group * 17 + token * 63) % 128] = above * factor
    source[0].zero_()
    source[0, 0] = -0.0
    if tokens == 1:
        source[0, -1] = above
    x = source.cuda()
    if input_offset:
        storage = torch.full((x.numel() + input_offset + 1,), 7, dtype=dtype, device="cuda")
        x = storage[input_offset:-1].reshape(source.shape).copy_(x)
        assert x.is_contiguous() and x.data_ptr() % 16 == (input_offset * 2) % 16
    else:
        assert x.data_ptr() % 8 == 0
    ids = (torch.arange(tokens * 3, device="cuda").reshape(tokens, 3) % 5).int()
    kw = dict(quant_mode=5, row_idx_type=int(dtype == torch.float16),
              topk_weight=torch.arange(tokens * 3, device="cuda", dtype=torch.float32).reshape(tokens, 3))
    expected_data, expected_scale = quantization_reference(x, 5)
    result = ops.moe_init_routing_v4(x, ids, 5, **kw)
    order = check_routing(x, ids, result, 5, **kw)
    torch.testing.assert_close(result[0].view(torch.uint8).cpu(),
                               expected_data[order // 3], rtol=0, atol=0)
    torch.testing.assert_close(result[3].view(torch.uint8).cpu(),
                               expected_scale[order // 3].view(torch.uint8), rtol=0, atol=0)
    workspace = allocate_workspace(ops, x, ids, 5, 5)
    out = [torch.empty_like(t) for t in result]
    for tensor in out:
        tensor.view(torch.uint8).fill_(0xa5)
    ops.moe_init_routing_v4_out(x, ids, *out, workspace, 5, **kw)
    for actual, expected in zip(out, result):
        torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(x.cpu().view(torch.uint8), source.view(torch.uint8), rtol=0, atol=0)
    if input_offset:
        assert bool((storage[:input_offset] == 7).all()) and storage[-1].item() == 7


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden,tokens", [(128, 6), (128, 7), (128, 8), (128, 9),
                                           (128, 15), (128, 16), (128, 17), (384, 3)])
@pytest.mark.parametrize("input_offset", [0, 4])
def test_fp8_vectors8_tiles(ops, dtype, hidden, tokens, input_offset):
    # Input offset4 means8B: retain vectors4 instead of entering vectors8.
    # Compare all data/scale bytes and input canaries with the independent oracle.
    test_fp8_vector_pairs(ops, dtype, hidden, tokens, input_offset)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", [3, 5])
def test_fp8_encoding_midpoints(ops, dtype, mode):
    # All finite E4M3 codepoints, every midpoint and its adjacent source-dtype
    # values. A448 anchor fixes each group scale at1 for the bounded payload.
    levels = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    mid = ((levels[:-1] + levels[1:]) * 0.5).to(dtype)
    values = torch.cat([levels.to(dtype), mid,
                        torch.nextafter(mid, torch.full_like(mid, float("inf"))),
                        torch.nextafter(mid, torch.full_like(mid, -float("inf")))])
    values = torch.cat([values, -values])
    values = values[values.abs() <= 448]
    group = 32 if mode == 3 else 128
    payload_columns = group - 1
    x = torch.zeros(((len(values) + payload_columns - 1) // payload_columns, group), dtype=dtype)
    x[:, 0] = 448
    payload = torch.zeros(x.shape[0] * payload_columns, dtype=dtype)
    payload[:len(values)] = values
    x[:, 1:] = payload.reshape(-1, payload_columns)
    x = x.cuda()
    ids = (torch.arange(x.shape[0] * 3, device="cuda") % 5).reshape(-1, 3).int()
    result = ops.moe_init_routing_v4(x, ids, 5, quant_mode=mode)
    order = check_routing(x, ids, result, 5, quant_mode=mode)
    expected = x.cpu().float().to(torch.float8_e4m3fn).view(torch.uint8)
    torch.testing.assert_close(result[0].view(torch.uint8).cpu(),
                               expected[order // 3], rtol=0, atol=0)
    if mode == 3:
        # One E8M0 scale byte plus a zero padding byte for H=32.
        expected_scale = torch.zeros(result[3].shape, dtype=torch.uint8)
        expected_scale[:, 0] = 127
        torch.testing.assert_close(result[3].view(torch.uint8).cpu(), expected_scale, rtol=0, atol=0)
    else:
        torch.testing.assert_close(result[3].cpu(), torch.ones_like(result[3].cpu()), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mxfp4_encoding_midpoints(ops, dtype):
    # A 6 anchor fixes each Group32 scale at 1. Check all E2M1 midpoints
    # and both adjacent source values, including Half -> BF16 product rounding.
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=dtype)
    mid = (levels[:-1] + levels[1:]) * .5
    values = torch.cat([levels, mid,
                        torch.nextafter(mid, torch.full_like(mid, float("inf"))),
                        torch.nextafter(mid, torch.full_like(mid, -float("inf")))])
    values = torch.cat([values, -values])
    x = torch.zeros(((len(values) + 30) // 31, 32), dtype=dtype)
    x[:, 0] = 6
    payload = torch.zeros(x.shape[0] * 31, dtype=dtype)
    payload[:len(values)] = values
    x[:, 1:] = payload.reshape(-1, 31)
    x = x.cuda()
    ids = (torch.arange(x.shape[0] * 3, device="cuda") % 5).reshape(-1, 3).int()
    result = ops.moe_init_routing_v4(x, ids, 5, quant_mode=9)
    order = check_routing(x, ids, result, 5, quant_mode=9)
    expected_data, expected_scale = quantization_reference(x, 9)
    torch.testing.assert_close(result[0].view(torch.uint8).cpu(),
                               expected_data[order // 3], rtol=0, atol=0)
    torch.testing.assert_close(result[3].view(torch.uint8).cpu(),
                               expected_scale[order // 3], rtol=0, atol=0)
    workspace = allocate_workspace(ops, x, ids, 5, 9)
    workspace.fill_(0xa5)
    out = [torch.empty_like(t) for t in result]
    ops.moe_init_routing_v4_out(x, ids, *out, workspace, 5, quant_mode=9)
    for actual, expected in zip(out, result):
        torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8),
                                   rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shuffled", [False, True])
@pytest.mark.parametrize("hidden", [2048, 4096])
def test_fp8_all_input_patterns(ops, dtype, shuffled, hidden):
    # A frozen prior production extension is the oracle for nonfinite group
    # scale/inverse semantics. Distinct module names avoid pybind cache aliasing.
    path = os.environ.get("MCOP_ROUTING_REFERENCE_EXTENSION")
    if not path:
        pytest.skip("set MCOP_ROUTING_REFERENCE_EXTENSION to frozen Q3")
    spec = importlib.util.spec_from_file_location("routing_reference.op", Path(path).resolve())
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    assert reference.moe_init_routing_v4_out is not ops.moe_init_routing_v4_out
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    if shuffled:
        bits = bits[torch.randperm(len(bits), generator=torch.Generator().manual_seed(20261023))]
    x = bits.view(dtype).reshape(-1, hidden).cuda()
    ids = (torch.arange(x.shape[0] * 3, device="cuda") % 5).reshape(-1, 3).int()
    expected = reference.moe_init_routing_v4(x, ids, 5, quant_mode=5)
    actual = ops.moe_init_routing_v4(x, ids, 5, quant_mode=5)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left.view(torch.uint8), right.view(torch.uint8), rtol=0, atol=0)
    workspace = allocate_workspace(ops, x, ids, 5, 5)
    out = [torch.empty_like(t) for t in actual]
    ops.moe_init_routing_v4_out(x, ids, *out, workspace, 5, quant_mode=5)
    for left, right in zip(out, expected):
        torch.testing.assert_close(left.view(torch.uint8), right.view(torch.uint8), rtol=0, atol=0)


def test_stream_and_graph(ops):
    x = torch.randn(37, 130, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, 23, (37, 7), dtype=torch.int32, device="cuda")
    kw = dict(expert_tokens_num_flag=False)
    result = ops.moe_init_routing_v4(x, ids, 23, **kw)
    ws = allocate_workspace(ops, x, ids, 23)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            ops.moe_init_routing_v4_out(x, ids, *result, ws, 23, **kw)
    torch.cuda.current_stream().wait_stream(stream)
    check_routing(x, ids, result, 23, **kw)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.moe_init_routing_v4_out(x, ids, *result, ws, 23, **kw)
    graph.replay()
    check_routing(x, ids, result, 23, **kw)


def _check_quantized_stream_and_graph(ops, mode, hidden, tokens=29, row_type=0, topk=4):
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    x[7].zero_()
    x[7, 0] = -0.0
    ids = torch.randint(0, 23, (tokens, topk), dtype=torch.int32, device="cuda")
    kw = dict(quant_mode=mode, expert_tokens_num_flag=False, row_idx_type=row_type)
    expected = ops.moe_init_routing_v4(x, ids, 23, **kw)
    out = [torch.empty_like(t) for t in expected]
    workspace = allocate_workspace(ops, x, ids, 23, mode)

    def poison():
        for tensor in out:
            if tensor.numel():
                tensor.view(torch.uint8).fill_(0xa5)

    def exact():
        for actual, reference in zip(out, expected):
            torch.testing.assert_close(actual.view(torch.uint8), reference.view(torch.uint8), rtol=0, atol=0)

    poison()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ops.moe_init_routing_v4_out(x, ids, *out, workspace, 23, **kw)
    torch.cuda.current_stream().wait_stream(stream)
    exact()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.moe_init_routing_v4_out(x, ids, *out, workspace, 23, **kw)
    poison()
    graph.replay()
    exact()


@pytest.mark.parametrize("tokens", [1025, 131073])
@pytest.mark.parametrize("row_type", [0, 1])
def test_counting_sort_stream_and_graph(ops, row_type, tokens):
    _check_quantized_stream_and_graph(ops, 5, 128, tokens=tokens, row_type=row_type)


@pytest.mark.parametrize("mode", [3, 9])
@pytest.mark.parametrize("hidden", [130, 4096])
def test_mx_stream_and_graph(ops, mode, hidden):
    _check_quantized_stream_and_graph(ops, mode, hidden)


@pytest.mark.parametrize("hidden", [2048, 4096])
@pytest.mark.parametrize("row_type", [0, 1])
def test_bf16_token_scatter_stream_and_graph(ops, hidden, row_type):
    _check_quantized_stream_and_graph(ops, -1, hidden, row_type=row_type)


@pytest.mark.parametrize("hidden", [129, 4096])
def test_fp8_stream_and_graph(ops, hidden):
    _check_quantized_stream_and_graph(ops, 5, hidden)


@pytest.mark.parametrize("mode", [3, 9])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden,capacity,empty_range", [
    (4096, cap, False) for cap in (1, 15, 16, 17, 33)
] + [(h, 17, False) for h in (4032, 4034, 4098)] + [(4096, 17, True)])
def test_mx_scale_tiles(ops, mode, dtype, hidden, capacity, empty_range):
    x = torch.randn(12, hidden, dtype=dtype, device="cuda")
    x[0].zero_()
    x[0, 0] = -0.0
    ids = torch.tensor([[7, -1, 3], [2, 3, 5], [1, 13, 3]] * 4,
                       dtype=torch.int32, device="cuda")
    kw = dict(quant_mode=mode, active_num=capacity,
              active_expert_range=[0, 0] if empty_range else [2, 6],
              row_idx_type=int(dtype == torch.float16))
    encoded, scales = quantization_reference(x, mode)
    result = ops.moe_init_routing_v4(x, ids, 8, **kw)
    order = check_routing(x, ids, result, 8, **kw)
    count = len(order)
    tokens = order // ids.shape[1]
    torch.testing.assert_close(result[0][:count].view(torch.uint8).cpu(),
                               encoded[tokens], rtol=0, atol=0)
    torch.testing.assert_close(result[3][:count].view(torch.uint8).cpu(),
                               scales[tokens].view(torch.uint8), rtol=0, atol=0)
    workspace = allocate_workspace(ops, x, ids, 8, mode)
    out = [torch.empty_like(t) for t in result]
    workspace.fill_(0xa5)
    if mode in (3, 9) and dtype == torch.float16 and hidden == 4096 and capacity == 17 and not empty_range:
        # Contiguous user outputs may have an unaligned storage offset.
        storage = torch.empty(result[3].numel() + 1, dtype=torch.uint8, device="cuda")
        out[3] = storage[1:].view(result[3].dtype).reshape(result[3].shape)
    for tensor in out:
        tensor.view(torch.uint8).fill_(0xa5)
    ops.moe_init_routing_v4_out(x, ids, *out, workspace, 8, **kw)
    check_routing(x, ids, out, 8, **kw)
    for index in (0, 3):
        torch.testing.assert_close(out[index][:count].view(torch.uint8),
                                   result[index][:count].view(torch.uint8), rtol=0, atol=0)
        assert bool((out[index][count:].view(torch.uint8) == 0xa5).all())


@pytest.mark.parametrize("mode,hidden", [
    (3, h) for h in [2032, 2048, 2064, 2176, 2944, 3072, 3200, 3968, 4080, 4096, 4112]
] + [(9, h) for h in [4064, 4096, 4128, 4352, 5888, 6144, 6400, 7936, 8160, 8192, 8224]]
  + [(5, h) for h in [2047, 2048, 2049, 2064, 2176, 2944, 3072, 3968, 4080, 4095, 4096, 4097]])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("capacity", [1, 2, 3, 17])
def test_mx_data_rows2_boundaries(ops, mode, hidden, dtype, capacity):
    test_mx_scale_tiles(ops, mode, dtype, hidden, capacity, False)


@pytest.mark.parametrize("mode,hidden", [(3, 4096), (9, 4096), (5, 4096),
                                        (3, 2048), (5, 2048), (9, 8192)])
@pytest.mark.parametrize("row_type", [0, 1])
@pytest.mark.parametrize("offset", [0, 1, 16])
def test_mx_data_rows2_output_alignment(ops, mode, hidden, row_type, offset):
    x = torch.randn(12, hidden, dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor([[7, -1, 3], [2, 3, 5], [1, 13, 3]] * 4,
                       dtype=torch.int32, device="cuda")
    kw = dict(quant_mode=mode, active_num=17, active_expert_range=[2, 6],
              row_idx_type=row_type)
    expected = ops.moe_init_routing_v4(x, ids, 8, **kw)
    outputs = [torch.empty_like(t) for t in expected]
    data = outputs[0]
    storage = torch.full((data.nbytes + offset + 256,), 0xa5,
                         dtype=torch.uint8, device="cuda")
    outputs[0] = storage[offset:offset + data.nbytes].view(data.dtype).reshape(data.shape)
    for tensor in outputs:
        tensor.view(torch.uint8).fill_(0xa5)
    workspace = allocate_workspace(ops, x, ids, 8, mode)
    ops.moe_init_routing_v4_out(x, ids, *outputs, workspace, 8, **kw)
    order = check_routing(x, ids, outputs, 8, **kw)
    effective = min(17, len(order))
    for index in (0, 3):
        torch.testing.assert_close(outputs[index][:effective].view(torch.uint8),
                                   expected[index][:effective].view(torch.uint8), rtol=0, atol=0)
        assert bool((outputs[index][effective:].view(torch.uint8) == 0xa5).all())
    assert bool((storage[:offset] == 0xa5).all())
    assert bool((storage[offset + data.nbytes:] == 0xa5).all())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("empty_range", [False, True])
def test_fp8_data_rows2_unwritten_tails(ops, dtype, empty_range):
    # Capacity33 exceeds the20 valid routes, or every route is invalid for an
    # empty expert range. Unwritten data/FP32-scale tails must retain canaries.
    test_mx_scale_tiles(ops, 5, dtype, 4096, 33, empty_range)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden", [4033, 4095, 4097])
def test_mx_scale_fp8_odd_boundaries(ops, dtype, hidden):
    # MXFP8 permits odd widths: exercise the first 128-B scale row, the
    # last partial input group in that range, and the next fallback width.
    test_mx_scale_tiles(ops, 3, dtype, hidden, 17, False)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden", [3968, 3969, 3970, 4000, 4096, 4097])
@pytest.mark.parametrize("capacity", [3, 33])
def test_fp8_scale_grouped_boundaries(ops, dtype, hidden, capacity):
    # Group128 FP32 scales have exactly128 bytes per row for3969<=H<=4096.
    test_mx_scale_tiles(ops, 5, dtype, hidden, capacity, False)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden", [3969, 4096, 4097])
@pytest.mark.parametrize("float_offset", [0, 1, 4])
def test_fp8_scale_grouped_output_alignment(ops, dtype, hidden, float_offset):
    x = torch.randn(12, hidden, dtype=dtype, device="cuda")
    ids = torch.tensor([[7, -1, 3], [2, 3, 5], [1, 13, 3]] * 4,
                       dtype=torch.int32, device="cuda")
    kw = dict(quant_mode=5, active_num=33, active_expert_range=[2, 6],
              row_idx_type=int(dtype == torch.float16))
    expected = ops.moe_init_routing_v4(x, ids, 8, **kw)
    outputs = [torch.empty_like(t) for t in expected]
    scale_bytes = expected[3].nbytes
    offset = float_offset * 4
    storage = torch.full((scale_bytes + offset + 256,), 0xa5,
                         dtype=torch.uint8, device="cuda")
    outputs[3] = storage[offset:offset + scale_bytes].view(torch.float32).reshape(expected[3].shape)
    assert outputs[3].is_contiguous() and outputs[3].data_ptr() % 16 == offset % 16
    for tensor in outputs:
        tensor.view(torch.uint8).fill_(0xa5)
    workspace = allocate_workspace(ops, x, ids, 8, 5)
    workspace.fill_(0xa5)
    ops.moe_init_routing_v4_out(x, ids, *outputs, workspace, 8, **kw)
    order = check_routing(x, ids, outputs, 8, **kw)
    count = len(order)
    _, scales = quantization_reference(x, 5)
    torch.testing.assert_close(outputs[3][:count].view(torch.uint8).cpu(),
                               scales[order // 3].view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(outputs[0][:count].view(torch.uint8),
                               expected[0][:count].view(torch.uint8), rtol=0, atol=0)
    for index in (0, 3):
        assert bool((outputs[index][count:].view(torch.uint8) == 0xa5).all())
    assert bool((storage[:offset] == 0xa5).all())
    assert bool((storage[offset + scale_bytes:] == 0xa5).all())


def check_scale_token_tiles(ops, mode, hidden, tokens, dtype, topk=20, scale_offset=0,
                            row_type=0, patterned_groups=False):
    x = torch.randn(tokens, hidden, dtype=dtype, device="cuda")
    if patterned_groups:
        exponents = torch.arange(hidden // 32, device="cuda").remainder(17) - 8
        x.reshape(tokens, -1, 32).mul_(torch.exp2(exponents.float()).reshape(1, -1, 1))
    x[0].zero_()
    ids = torch.arange(tokens * topk, device="cuda").reshape(tokens, topk).remainder(11).int()
    if topk > 1:
        ids[:, 1] = ids[:, 0]
    ids[0, 0], ids[-1, -1] = -1, 13
    kw = dict(quant_mode=mode, row_idx_type=row_type, active_num=max(1, ids.numel() // 2),
              active_expert_range=[2, 9])
    encoded, scales = quantization_reference(x, mode)
    result = ops.moe_init_routing_v4(x, ids, 11, **kw)
    out = [torch.empty_like(t) for t in result]
    scale_size = result[3].nbytes
    guard = torch.full((scale_size + scale_offset + 256,), 0xa5,
                       dtype=torch.uint8, device="cuda")
    out[3] = guard[scale_offset:scale_offset + scale_size].view(result[3].dtype).reshape(result[3].shape)
    assert out[3].data_ptr() % 16 == scale_offset % 16
    for tensor in out:
        tensor.view(torch.uint8).fill_(0xa5)
    workspace = allocate_workspace(ops, x, ids, 11, mode)
    workspace.fill_(0xa5)
    ops.moe_init_routing_v4_out(x, ids, *out, workspace, 11, **kw)
    for outputs in (result, out):
        order = check_routing(x, ids, outputs, 11, **kw)
        source_tokens = order // topk
        torch.testing.assert_close(outputs[0][:len(order)].view(torch.uint8).cpu(),
                                   encoded[source_tokens], rtol=0, atol=0)
        torch.testing.assert_close(outputs[3][:len(order)].view(torch.uint8).cpu(),
                                   scales[source_tokens].view(torch.uint8), rtol=0, atol=0)
    for index in (0, 3):
        assert bool((out[index][len(order):].view(torch.uint8) == 0xa5).all())
    assert bool((guard[:scale_offset] == 0xa5).all())
    assert bool((guard[scale_offset + scale_size:] == 0xa5).all())


@pytest.mark.parametrize("tokens", [1, 15, 16, 17, 33])
@pytest.mark.parametrize("mode", [3, 5, 9])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scale_token_tiles(ops, tokens, mode, dtype):
    check_scale_token_tiles(ops, mode, 4096, tokens, dtype)


@pytest.mark.parametrize("mode,hidden", [(3, 4033), (5, 3969), (9, 4034)])
def test_scale_token_partial_group_half(ops, mode, hidden):
    check_scale_token_tiles(ops, mode, hidden, 17, torch.float16)


@pytest.mark.parametrize("mode", [3, 5, 9])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scale_token_topk_one(ops, mode, dtype):
    check_scale_token_tiles(ops, mode, 4096, 17, dtype, topk=1)


@pytest.mark.parametrize("mode,hidden", [(3, 2048), (5, 2048), (9, 8192)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens", [1, 17, 33])
@pytest.mark.parametrize("topk", [1, 20])
def test_fused_scale_extended_token_tails(ops, mode, hidden, dtype, tokens, topk):
    # Odd token tails, invalid/duplicate routes, capacity, poisoned workspace
    # and byte-exact independent quantization reference on the new scale widths.
    check_scale_token_tiles(ops, mode, hidden, tokens, dtype, topk=topk)


@pytest.mark.parametrize("mode,hidden", [(3, 2048), (5, 2048), (9, 8192)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("vector_aligned", [False, True])
def test_fused_scale_extended_output_alignment(ops, mode, hidden, dtype, vector_aligned):
    # A legal scalar-aligned offset may force the generic fallback; a16B offset
    # retains fusion. Protect both ends of the user-provided scale allocation.
    offset = 16 if vector_aligned else (4 if mode == 5 else 1)
    check_scale_token_tiles(ops, mode, hidden, 17, dtype, scale_offset=offset)


@pytest.mark.parametrize("mode,hidden", [(3, 2048), (5, 2048), (9, 8192),
                                        (3, 4096), (5, 4096), (9, 4096)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens", [1, 17, 33])
@pytest.mark.parametrize("topk", [1, 20])
def test_forward_fused_token_tails(ops, mode, hidden, dtype, tokens, topk):
    # Independent quantization and forward permutation references, odd row tails,
    # invalid/duplicate routes, capacity and poisoned workspace on all six layouts.
    check_scale_token_tiles(ops, mode, hidden, tokens, dtype, topk=topk, row_type=1)


@pytest.mark.parametrize("mode,hidden", [(3, 2048), (5, 2048), (9, 8192),
                                        (3, 4096), (5, 4096), (9, 4096)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("vector_aligned", [False, True])
def test_forward_fused_scale_alignment(ops, mode, hidden, dtype, vector_aligned):
    offset = 16 if vector_aligned else (4 if mode == 5 else 1)
    check_scale_token_tiles(ops, mode, hidden, 17, dtype,
                            scale_offset=offset, row_type=1)


@pytest.mark.parametrize("mode,hidden", [(3, 2048), (5, 2048), (9, 8192),
                                        (3, 4096), (5, 4096), (9, 4096)])
def test_forward_fused_stream_and_graph(ops, mode, hidden):
    _check_quantized_stream_and_graph(ops, mode, hidden, row_type=1)


@pytest.mark.parametrize("mode,hidden", [(3, 2048), (3, 4096), (5, 2048),
                                        (5, 4096), (9, 4096), (9, 8192)])
@pytest.mark.parametrize("tokens", [1, 3, 17])
@pytest.mark.parametrize("topk", [1, 3, 30])
def test_direct_quant_scatter_tiles(ops, mode, hidden, tokens, topk):
    # Independent quantization/permutation, odd token tails, invalid/duplicate
    # routes, capacity truncation, poisoned workspace and output canaries.
    check_scale_token_tiles(ops, mode, hidden, tokens, torch.bfloat16, topk=topk)


@pytest.mark.parametrize("mode,hidden", [(3, 2048), (3, 4096), (5, 2048),
                                        (5, 4096), (9, 4096), (9, 8192)])
def test_direct_quant_scatter_stream_and_graph(ops, mode, hidden):
    _check_quantized_stream_and_graph(ops, mode, hidden, row_type=0)


@pytest.mark.parametrize("hidden", [2047, 2049, 3072, 4095])
def test_direct_mx8_layout_fallback(ops, hidden):
    # Complete-tile neighbors and interior fallback: independent reference,
    # padded MX scales, invalid routes, capacity and poisoned output tails.
    check_scale_token_tiles(ops, 3, hidden, 3, torch.bfloat16, topk=3)


@pytest.mark.parametrize("hidden", [4094, 4098, 6144, 8190, 8194])
def test_direct_mx4_layout_fallback(ops, hidden):
    # MXFP4 requires even hidden; cover valid neighbors of both complete tiles
    # and an interior partial tile, with independent quantization/permutation.
    check_scale_token_tiles(ops, 9, hidden, 3, torch.bfloat16, topk=3)


@pytest.mark.parametrize("hidden", [4096, 8192])
@pytest.mark.parametrize("offset", [1, 17])
def test_direct_mx4_scale_byte_offset(ops, hidden, offset):
    # Direct scales are byte-addressed; legal unaligned views retain canaries.
    check_scale_token_tiles(ops, 9, hidden, 17, torch.bfloat16,
                            topk=3, scale_offset=offset)


@pytest.mark.parametrize("hidden", [4096, 8192])
@pytest.mark.parametrize("offset", [0, 1, 2, 17, 18])
@pytest.mark.parametrize("topk", [1, 3, 7])
def test_direct_mx4_paired_scales(ops, hidden, offset, topk):
    # Neighboring groups differ across lane32 and tile4096 boundaries. The
    # independent reference checks data and scales together; canaries cover
    # aligned16bit stores and the legal odd-byte fallback on an odd token tail.
    check_scale_token_tiles(ops, 9, hidden, 3, torch.bfloat16, topk=topk,
                            scale_offset=offset, patterned_groups=True)


@pytest.mark.parametrize("hidden", [4096, 8192])
@pytest.mark.parametrize("tokens", [1, 3, 17])
@pytest.mark.parametrize("topk", [63, 64, 65])
def test_direct_mx4_rankwave_boundary(ops, hidden, tokens, topk):
    # Full-wave rank storage boundary and original-kernel fallback, with odd
    # tokens, invalid/duplicate routes, capacity truncation and independent bytes.
    check_scale_token_tiles(ops, 9, hidden, tokens, torch.bfloat16, topk=topk,
                            patterned_groups=True)


@pytest.mark.parametrize("hidden", [4096, 8192])
@pytest.mark.parametrize("topk", [63, 64, 65])
def test_direct_mx4_rankwave_stream_and_graph(ops, hidden, topk):
    _check_quantized_stream_and_graph(ops, 9, hidden, tokens=9, topk=topk)


@pytest.mark.parametrize("hidden", [2048, 4096])
@pytest.mark.parametrize("tokens", [1, 3, 17])
@pytest.mark.parametrize("topk", [63, 64, 65])
def test_direct_mx8_rankwave_boundary(ops, hidden, tokens, topk):
    # Rank storage limit, unchanged H2048 and K>64 fallbacks, odd tokens,
    # invalid/duplicate routes and capacity against independent data/scale bytes.
    check_scale_token_tiles(ops, 3, hidden, tokens, torch.bfloat16, topk=topk,
                            patterned_groups=True)


@pytest.mark.parametrize("hidden", [2048, 4096])
@pytest.mark.parametrize("topk", [63, 64, 65])
def test_direct_mx8_rankwave_stream_and_graph(ops, hidden, topk):
    _check_quantized_stream_and_graph(ops, 3, hidden, tokens=9, topk=topk)


TARGETS = list(itertools.product((10240, 20480, 40960, 81920),
                                 (15, 20, 25, 30), (800, 1000, 1200)))


@pytest.mark.skipif(os.environ.get("MCOP_ROUTING_FULL") != "1", reason="set MCOP_ROUTING_FULL=1")
@pytest.mark.parametrize("seq,topk,experts", TARGETS)
def test_full_matrix(ops, seq, topk, experts):
    x = torch.randn(seq, 4096, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, experts, (seq, topk), dtype=torch.int32, device="cuda")
    result = ops.moe_init_routing_v4(x, ids, experts)
    check_routing(x, ids, result, experts)


def quantization_reference_batched(x, mode):
    """Bounded-memory Torch reference, cross-checked against the CPU reference.

    Calculate scales with frexp/ldexp, use Torch's FP8 conversion, and choose
    FP4 codes by nearest-level distance. This does not call the tested extension.
    """
    n, hidden = x.shape
    size = 128 if mode in (4, 5) else 32
    groups = (hidden + size - 1) // size
    scale_columns = groups if mode in (4, 5) else (hidden + 63) // 64 * 2
    encoded = torch.empty((n, hidden // 2 if mode == 9 else hidden), dtype=torch.uint8, device=x.device)
    scales = torch.zeros((n, scale_columns), dtype=torch.float32 if mode in (4, 5) else torch.uint8, device=x.device)
    dtype = torch.float8_e5m2 if mode in (2, 4) else torch.float8_e4m3fn
    emax = 2 if mode == 9 else (15 if mode in (2, 4) else 8)
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    ranks = torch.arange(8, device=x.device)
    ranks = ranks + (ranks % 2) * 8
    if not hidden:
        return encoded, scales.reshape(n, (hidden + 63) // 64, 2) if mode == 9 else scales
    for begin in range(0, n, 256):
        end = min(n, begin + 256)
        values = torch.nn.functional.pad(x[begin:end].float(), (0, groups * size - hidden))
        values = values.reshape(end - begin, groups, size)
        scale_values = values
        if mode not in (4, 5) and x.dtype == torch.float16:
            scale_values = (values.view(torch.int32) & -65536).view(torch.float32)
        amax = scale_values.abs().amax(dim=-1)
        mantissa, exponent = torch.frexp(amax)
        code = exponent - 1 - emax + 127
        if mode == 9:
            code = code.clamp_min(0)
        else:
            code = code + (mantissa > .875).int()
            code = torch.where((code >= 1) & (code <= 254), code, 1)
        code = torch.where(amax == 0, 1 if mode in (4, 5) else 0, code)
        reciprocal = torch.ldexp(torch.ones_like(amax), 127 - code)
        if mode in (2, 3):
            reciprocal = torch.where(amax == 0, 0., reciprocal)
        if mode in (4, 5):
            scales[begin:end, :groups] = torch.ldexp(torch.ones_like(amax), code - 127)
        else:
            scales[begin:end, :groups] = code.to(torch.uint8)
        scaled = values * reciprocal[..., None]
        if mode == 9 or (mode in (2, 3) and x.dtype == torch.bfloat16):
            scaled = scaled.to(torch.bfloat16).float()
        scaled = scaled.reshape(end - begin, -1)[:, :hidden]
        if mode == 9:
            distances = (scaled.abs()[..., None] - levels).abs()
            tied = distances == distances.amin(dim=-1, keepdim=True)
            codes = torch.where(tied, ranks, 100).argmin(dim=-1).to(torch.uint8)
            codes |= torch.signbit(scaled).to(torch.uint8) * 8
            encoded[begin:end] = codes[:, ::2] | (codes[:, 1::2] << 4)
        else:
            maximum = torch.finfo(dtype).max
            encoded[begin:end] = scaled.clamp(-maximum, maximum).to(dtype).view(torch.uint8)
    if mode == 9:
        scales = scales.reshape(n, (hidden + 63) // 64, 2)
    return encoded, scales


@pytest.mark.parametrize("mode", [2, 3, 4, 5, 9])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden", [34, 130])
def test_batched_quantization_reference(ops, mode, dtype, hidden):
    source = torch.randn(9, hidden, dtype=dtype, device="cuda")
    source[0].zero_()
    cpu_data, cpu_scales = quantization_reference(source, mode)
    data, scales = quantization_reference_batched(source, mode)
    torch.testing.assert_close(data.cpu(), cpu_data, rtol=0, atol=0)
    torch.testing.assert_close(scales.cpu(), cpu_scales, rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("MCOP_ROUTING_FULL_LOW_PRECISION") != "1",
                    reason="set MCOP_ROUTING_FULL_LOW_PRECISION=1")
@pytest.mark.parametrize("kind", ["fp8", "fp4", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("seq,topk,experts", TARGETS)
def test_full_low_precision(ops, seq, topk, experts, kind):
    ids = torch.randint(0, experts, (seq, topk), dtype=torch.int32, device="cuda")
    if kind == "fp4":
        source = torch.randint(0, 256, (seq, 4096 // 2), dtype=torch.uint8, device="cuda")
        x = source.view(torch.float4_e2m1fn_x2)
        result = ops.moe_init_routing_v4(x, ids, experts)
        check_routing(x, ids, result, experts)
        return
    mode = {"fp8": 5, "mxfp8": 3, "mxfp4": 9}[kind]
    x = torch.randn(seq, 4096, dtype=torch.bfloat16, device="cuda")
    expected_data, expected_scales = quantization_reference_batched(x, mode)
    result = ops.moe_init_routing_v4(x, ids, experts, quant_mode=mode)
    order = check_routing(x, ids, result, experts, quant_mode=mode)
    actual_data = result[0].view(torch.uint8)
    actual_scales = result[3] if mode == 5 else result[3].view(torch.uint8)
    assert actual_data.shape[1:] == expected_data.shape[1:]
    assert actual_scales.shape[1:] == expected_scales.shape[1:]
    for begin in range(0, len(order), 1024):
        end = min(len(order), begin + 1024)
        tokens = (order[begin:end] // topk).to(x.device)
        torch.testing.assert_close(actual_data[begin:end], expected_data[tokens], rtol=0, atol=0)
        torch.testing.assert_close(actual_scales[begin:end], expected_scales[tokens], rtol=0, atol=0)


# Device Tensor activeNum contract: same graph, input address and output storage;
# only the activeNum contents change. Legacy scalar tests above remain controls.
def _active_guarded_like(tensor):
    size = tensor.numel() * tensor.element_size()
    storage = torch.full((size + 128,), 0xa5, dtype=torch.uint8, device=tensor.device)
    view = storage[64:64 + size].view(tensor.dtype).reshape(tensor.shape)
    return view, storage


def _check_active_result(out, reference, info, value, routes, available, capacity):
    logical = -1 if value < 0 else (routes if value == 0 else min(value, routes))
    status = 1 if value < 0 else (2 if logical > capacity else 0)
    effective = 0 if status else min(logical, available)
    assert info.cpu().tolist() == [logical, effective, available, status]
    for index, (actual, expected) in enumerate(zip(out, reference)):
        if index in (1, 2):
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        elif actual.dim() > 1 or actual.numel():
            # Includes scalar-per-row passthrough scales, byte scales, packed
            # data and weights. Invalid tails must survive every replay.
            torch.testing.assert_close(actual[:effective].view(torch.uint8).cpu(),
                                       expected[:effective].view(torch.uint8).cpu(), rtol=0, atol=0)
            assert torch.all(actual[effective:].view(torch.uint8).cpu() == 0xa5)


@pytest.mark.parametrize("mode", [-1, 2, 3, 4, 5, 9])
@pytest.mark.parametrize("row_type", [0, 1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden", [0, 130, 4096])
def test_active_tensor_graph_replay(ops, mode, row_type, dtype, hidden):
    x = torch.randn(7, hidden, dtype=dtype, device="cuda")
    ids = torch.tensor([[1, 1, 4], [3, 0, 2], [2, -1, 8], [3, 1, 2],
                        [4, 0, 1], [2, 3, 3], [1, 2, 4]], dtype=torch.int32, device="cuda")
    weights = torch.randn(ids.shape, device="cuda")
    kw = dict(quant_mode=mode, row_idx_type=row_type, expert_tokens_num_type=row_type,
              active_expert_range=[1, 4], topk_weight=weights,
              scale=torch.randn(7, device="cuda") if mode == -1 else None)
    reference = ops.moe_init_routing_v4(x, ids, 5, **kw)
    check_routing(x, ids, reference, 5, **kw)
    order, _ = permutation(ids, 5, [1, 4])
    if mode != -1 and hidden == 130:
        data, scales = quantization_reference(x, mode)
        tokens = order // ids.shape[1]
        torch.testing.assert_close(reference[0][:len(order)].view(torch.uint8).cpu(),
                                   data[tokens], rtol=0, atol=0)
        actual = reference[3][:len(order)].cpu()
        if mode not in (4, 5):
            actual = actual.view(torch.uint8)
        torch.testing.assert_close(actual, scales[tokens], rtol=0, atol=0)
    guarded = [_active_guarded_like(t) for t in reference]
    out = [pair[0] for pair in guarded]
    info, info_storage = _active_guarded_like(torch.empty(4, dtype=torch.int64, device="cuda"))
    active = torch.tensor(3, dtype=torch.int64, device="cuda")
    # Exercise both allowed metadata forms without changing them during replay.
    if row_type:
        active = active.reshape(1)
    workspace = allocate_workspace(ops, x, ids, 5, mode)
    call_kw = dict(kw, active_num_optional=active, active_info=info)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            ops.moe_init_routing_v4_out(x, ids, *out, workspace, 5, **call_kw)
    torch.cuda.current_stream().wait_stream(stream)
    _check_active_result(out, reference, info, 3, ids.numel(), len(order), ids.numel())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        ops.moe_init_routing_v4_out(x, ids, *out, workspace, 5, **call_kw)
    torch.cuda.current_stream().wait_stream(stream)
    for value in (0, 1, 3, ids.numel() - 1, ids.numel(), ids.numel() + 1,
                  1, -1, 3, 2**63 - 1):
        for tensor in out:
            tensor.view(torch.uint8).fill_(0xa5)
        workspace.fill_(0xa5)
        active.fill_(value)
        graph.replay()
        _check_active_result(out, reference, info, value, ids.numel(), len(order), ids.numel())
        for _, storage in guarded + [(info, info_storage)]:
            assert torch.all(storage[:64].cpu() == 0xa5)
            assert torch.all(storage[-64:].cpu() == 0xa5)


@pytest.mark.parametrize("mode", [-1, 2, 3, 4, 5, 9])
@pytest.mark.parametrize("capacity", [0, 3])
def test_active_tensor_small_storage_graph(ops, mode, capacity):
    x = torch.randn(5, 130, dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor([[1, 1, 1]] * 5, dtype=torch.int32, device="cuda")
    kw = dict(quant_mode=mode, topk_weight=torch.randn(ids.shape, device="cuda"))
    reference = ops.moe_init_routing_v4(x, ids, 5, **kw)
    out = [torch.empty_like(t[:capacity]) if i in (0, 3, 4) and t.dim() > 1
           else torch.empty_like(t) for i, t in enumerate(reference)]
    active = torch.tensor([1], dtype=torch.int64, device="cuda")
    info = torch.empty(4, dtype=torch.int64, device="cuda")
    workspace = allocate_workspace(ops, x, ids, 5, mode)
    call_kw = dict(kw, active_num_optional=active, active_info=info)
    ops.moe_init_routing_v4_out(x, ids, *out, workspace, 5, **call_kw)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.moe_init_routing_v4_out(x, ids, *out, workspace, 5, **call_kw)
    for value in (0, 1, 3, 4, -1, 2):
        for tensor in out:
            tensor.view(torch.uint8).fill_(0xa5)
        active.fill_(value)
        graph.replay()
        _check_active_result(out, reference, info, value, ids.numel(), ids.numel(), capacity)


@pytest.mark.parametrize("tokens,topk", [(0, 3), (3, 0)])
@pytest.mark.parametrize("mode", [-1, 9])
def test_active_tensor_empty(ops, tokens, topk, mode):
    x = torch.empty(tokens, 130, dtype=torch.bfloat16, device="cuda")
    ids = torch.empty(tokens, topk, dtype=torch.int32, device="cuda")
    out = ops.moe_init_routing_v4(x, ids, 5, quant_mode=mode)
    workspace = allocate_workspace(ops, x, ids, 5, mode)
    active = torch.tensor(0, dtype=torch.int64, device="cuda")
    info = torch.empty(4, dtype=torch.int64, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.moe_init_routing_v4_out(x, ids, *out, workspace, 5, quant_mode=mode,
                                   active_num_optional=active, active_info=info)
    for value in (0, 3, -1, 1):
        active.fill_(value)
        out[2].fill_(-1)
        graph.replay()
        assert info.cpu().tolist() == ([-1, 0, 0, 1] if value < 0 else [0, 0, 0, 0])
        assert out[2].cpu().tolist() == [0] * 5


@pytest.mark.parametrize("mode", [-1, 2, 3, 4, 5, 9])
@pytest.mark.parametrize("value", [0, 1, 99])
def test_active_tensor_eager_exact_shape(ops, mode, value):
    x = torch.randn(5, 130, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, 5, (5, 3), dtype=torch.int32, device="cuda")
    active = torch.tensor(value, dtype=torch.int64, device="cuda")
    info = torch.empty(4, dtype=torch.int64, device="cuda")
    expected = ops.moe_init_routing_v4(x, ids, 5, quant_mode=mode, active_num=value)
    actual = ops.moe_init_routing_v4(x, ids, 5, quant_mode=mode,
                                   active_num_optional=active, active_info=info)
    for a, b in zip(actual, expected):
        assert a.shape == b.shape
        torch.testing.assert_close(a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0)
    capacity = ids.numel() if value == 0 else min(value, ids.numel())
    assert info.cpu().tolist() == [capacity, capacity, ids.numel(), 0]
    # Omitting active_info is allowed on the allocating eager API.
    assert len(ops.moe_init_routing_v4(x, ids, 5, quant_mode=mode, active_num_optional=active)) == 5


def test_active_tensor_metadata_and_pairing(ops):
    x = torch.randn(5, 130, dtype=torch.bfloat16, device="cuda")
    ids = torch.ones(5, 3, dtype=torch.int32, device="cuda")
    out = ops.moe_init_routing_v4(x, ids, 5)
    workspace = allocate_workspace(ops, x, ids, 5)
    active = torch.tensor(1, dtype=torch.int64, device="cuda")
    info = torch.empty(4, dtype=torch.int64, device="cuda")

    def call(**kw):
        ops.moe_init_routing_v4_out(x, ids, *out, workspace, 5, **kw)

    for bad in (torch.tensor(1, dtype=torch.int64), active.int(), active.reshape(1, 1),
                torch.ones(2, dtype=torch.int64, device="cuda")):
        with pytest.raises(RuntimeError, match="active_num_optional"):
            call(active_num_optional=bad, active_info=info)
    for bad in (torch.empty(4, dtype=torch.int64), info.int(), info.reshape(2, 2), info[:3],
                torch.empty(8, dtype=torch.int64, device="cuda")[::2]):
        with pytest.raises(RuntimeError, match="active_info"):
            call(active_num_optional=active, active_info=bad)
    with pytest.raises(RuntimeError, match="require active_info"):
        call(active_num_optional=active)
    with pytest.raises(RuntimeError, match="cannot both"):
        call(active_num=1, active_num_optional=active, active_info=info)
    with pytest.raises(RuntimeError, match="requires active_num_optional"):
        call(active_info=info)
    with pytest.raises(RuntimeError):
        call(active_num_optional=info[:1], active_info=info)
    with pytest.raises(RuntimeError):
        call(active_num_optional=active, active_info=out[2][:4])
    active.fill_(-1)
    with pytest.raises(RuntimeError, match="nonnegative"):
        ops.moe_init_routing_v4(x, ids, 5, active_num_optional=active)


def test_active_tensor_allocating_capture_rejected(ops):
    x = torch.randn(5, 130, dtype=torch.bfloat16, device="cuda")
    ids = torch.ones(5, 3, dtype=torch.int32, device="cuda")
    active = torch.tensor(3, dtype=torch.int64, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="fixed storage"):
        with torch.cuda.graph(graph):
            ops.moe_init_routing_v4(x, ids, 5, active_num_optional=active)


@pytest.mark.parametrize("mode", [-1, 2, 3, 4, 5, 9])
def test_active_tensor_no_available_routes(ops, mode):
    x = torch.randn(5, 130, dtype=torch.bfloat16, device="cuda")
    ids = torch.ones(5, 3, dtype=torch.int32, device="cuda")
    kw = dict(quant_mode=mode, active_expert_range=[2, 2], expert_tokens_num_flag=False)
    reference = ops.moe_init_routing_v4(x, ids, 5, **kw)
    out = [torch.empty_like(t) for t in reference]
    workspace = allocate_workspace(ops, x, ids, 5, mode)
    active = torch.tensor(0, dtype=torch.int64, device="cuda")
    info = torch.empty(4, dtype=torch.int64, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.moe_init_routing_v4_out(x, ids, *out, workspace, 5, **kw,
                                   active_num_optional=active, active_info=info)
    for value in (0, 3, 1):
        for tensor in out:
            tensor.view(torch.uint8).fill_(0xa5)
        active.fill_(value)
        graph.replay()
        _check_active_result(out, reference, info, value, ids.numel(), 0, ids.numel())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32,
                                   torch.int8, torch.float8_e4m3fn, torch.float8_e5m2,
                                   torch.float4_e2m1fn_x2])
def test_active_tensor_passthrough_payload(ops, dtype):
    x = torch.empty(5, 65, dtype=dtype, device="cuda")
    x.view(torch.uint8).random_(0, 256)
    ids = torch.tensor([[1, 0, 1]] * 5, dtype=torch.int32, device="cuda")
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2):
        hidden = x.size(1) * (2 if dtype == torch.float4_e2m1fn_x2 else 1)
        scale = torch.randint(0, 255, (5, (hidden + 63) // 64, 2),
                              dtype=torch.uint8, device="cuda").view(torch.float8_e8m0fnu)
    else:
        scale = torch.randn(5, device="cuda")
    active = torch.tensor([3], dtype=torch.int64, device="cuda")
    actual = ops.moe_init_routing_v4(x, ids, 3, scale=scale, active_num_optional=active)
    check_routing(x, ids, actual, 3, active_num=3, scale=scale)


@pytest.mark.parametrize("row_type", [0, 1])
def test_active_tensor_large_sort(ops, row_type):
    x = torch.randn(131073, 2, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(-1, 13, (x.size(0), 4), dtype=torch.int32, device="cuda")
    active = torch.tensor(3, dtype=torch.int64, device="cuda")
    kw = dict(row_idx_type=row_type, active_expert_range=[1, 11])
    actual = ops.moe_init_routing_v4(x, ids, 13, **kw, active_num_optional=active)
    check_routing(x, ids, actual, 13, active_num=3, **kw)
