# SPDX-License-Identifier: Apache-2.0
"""Self-contained MoE router correctness, integration and performance tests.

Run from the repository after sourcing env_local.sh and checking mx-smi::

    export CUDA_VISIBLE_DEVICES=<idle GPU>
    python unit_test/test_triton_moe_router_kernel.py -q -s
    MOE_ROUTER_CASE_SET=ci python unit_test/test_triton_moe_router_kernel.py -q -s
    python -m pytest -q -s unit_test/test_triton_moe_router_kernel.py

Full/CI definition matrices, adversarial inputs, general/extended routing,
selected-sum reductions and all eight legacy benchmarks are collected by pytest.
Real SGLang integration cases remain separate from the standalone definition
cases; missing external dependencies are reported as skips, NEVER as passes.
Weights and ordered IDs in standalone cases use zero-tolerance checks. Undefined
nonfinite cases compare NaN/Inf masks and exact finite portions. Timing uses ten
warmups followed by CUDA-graph replay; bandwidth is logical, not physical HBM.
MOE_ROUTER_VALIDATION_REPORT writes a JSON report when invoking this file directly;
MOE_ROUTER_RESULTS appends precision/performance records in either entry mode.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import pytest
import torch
import triton
import triton.language as tl

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
TARGET_MODULE = os.environ.get('MOE_ROUTER_MODULE', 'mcoplib.triton_sglang_moe_fused_gate')
try:
    _target = importlib.import_module(TARGET_MODULE)
except Exception as exc:
    _target = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None
_gate = _target.moe_fused_gate if _target is not None else None
RESULTS = []


@pytest.fixture(scope='module', autouse=True)
def _require_device_and_target():
    if not torch.cuda.is_available():
        pytest.skip('router requires CUDA/MACA hardware')
    if _target is None:
        pytest.fail(f'could not import target {TARGET_MODULE!r}: {_IMPORT_ERROR}')


def get_ci_test_range(full, ci):
    return ci if os.environ.get('MOE_ROUTER_CASE_SET') == 'ci' else full

DEVICE = "cuda"


# Independent FP32 mathematical definitions
def stable_topk(values, k, eligible=None):
    work = values.clone()
    remaining = torch.ones_like(values, dtype=torch.bool) if eligible is None else eligible.clone()
    ids = torch.arange(values.shape[1], device=values.device)[None, :]
    result = []
    for _ in range(k):
        vmax = torch.where(remaining, work, -float('inf')).amax(1, keepdim=True)
        selected = torch.where(remaining & (work == vmax), ids, values.shape[1]).amin(1)
        result.append(selected)
        remaining.scatter_(1, selected[:, None], False)
    return torch.stack(result, 1)

def reference_gate(scores, bias, topk, scoring_func='sigmoid', num_fused_shared_experts=0, renormalize=True, routed_scaling_factor=1.0, apply_routed_scaling_factor_on_output=False, moe_softcapping=0.0, num_expert_group=1, topk_group=1, *, bias_alt=None, input_ids=None, bias_alt_token_id=None, num_token_non_padded=None, renormalize_epsilon=0.0, packed_out=None, sqrtsoftplus_log1p=False):
    """FP32 activation reference with explicit stable ties and shared slots."""
    x = scores.float()
    b = torch.zeros(scores.shape[1], device=scores.device) if bias is None else bias.float()
    b = b[None, :].expand_as(x)
    if input_ids is not None:
        b = torch.where((input_ids == bias_alt_token_id)[:, None], bias_alt.float()[None, :], b)
    if scoring_func == 'sigmoid':
        activated = torch.sigmoid(x)
        ranked = activated + b
    elif scoring_func == 'sqrtsoftplus':
        activated = torch.nn.functional.softplus(x).sqrt()
        ranked = activated + b
    elif scoring_func == 'softmax':
        if moe_softcapping:
            x = moe_softcapping * torch.tanh(x / moe_softcapping)
        ranked = x + b
        activated = torch.softmax(ranked, dim=-1)
    else:
        raise ValueError(scoring_func)
    ranked = torch.where(torch.isnan(ranked), float('inf') if scoring_func == 'sqrtsoftplus' and sqrtsoftplus_log1p else -1e+30, ranked)
    eligible = None
    if num_expert_group > 1:
        grouped = ranked.reshape(scores.shape[0], num_expert_group, -1)
        group_scores = grouped.topk(2, dim=-1).values.sum(-1)
        group_scores = torch.where(torch.isnan(group_scores), -1e+30, group_scores)
        group_ids = stable_topk(group_scores, topk_group)
        keep = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(1, group_ids, True)
        eligible = keep.repeat_interleave(grouped.shape[-1], -1)
        ranked = ranked.masked_fill(~eligible, -float('inf'))
    routed = topk - num_fused_shared_experts
    idx = stable_topk(ranked, routed, eligible)
    w = activated.gather(1, idx)
    total = w.sum(1, keepdim=True)
    if num_fused_shared_experts:
        shared = total.expand(-1, num_fused_shared_experts) / routed_scaling_factor
        w = torch.cat((w, shared), 1)
        si = torch.arange(scores.shape[1], scores.shape[1] + num_fused_shared_experts, device=scores.device)[None, :].expand(scores.shape[0], -1)
        idx = torch.cat((idx, si), 1)
    if renormalize:
        divisor = total + renormalize_epsilon if renormalize_epsilon else torch.where(total > 0, total, 1.0)
        w = w / divisor
    if apply_routed_scaling_factor_on_output:
        w = w * routed_scaling_factor
    idx = idx.to(torch.int32)
    if num_token_non_padded is not None:
        live = torch.arange(scores.shape[0], device=scores.device) < num_token_non_padded
        w = torch.where(live[:, None], w, 0.0)
        idx = torch.where(live[:, None], idx, -1)
    if packed_out is not None:
        bits = w.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 65535
        packed_out.copy_(idx << 16 | bits)
    return (w, idx)

def reference_production(hidden, gating, bias, topk, renormalize, num_expert_group=1, topk_group=1, num_fused_shared_experts=0, routed_scaling_factor=1.0, apply_routed_scaling_factor_on_output=False):
    return reference_gate(gating, bias, topk, 'sigmoid', num_fused_shared_experts, renormalize, routed_scaling_factor, apply_routed_scaling_factor_on_output, num_expert_group=num_expert_group, topk_group=topk_group)


# Exactness, CUDA graph replay and performance reporting
def check_and_report(label, fn, scores, bias, topk, reference, *, kwargs=None, exact_weights=True, compare_reference=True):
    kwargs = {} if kwargs is None else kwargs
    ref_kwargs = dict(kwargs)
    if kwargs.get('packed_out') is not None:
        ref_kwargs['packed_out'] = torch.empty_like(kwargs['packed_out'])
    weights, ids = fn(scores, bias, topk=topk, **kwargs)
    ref_w, ref_i = reference(scores, bias, topk=topk, **ref_kwargs)
    torch.cuda.synchronize()
    assert weights.shape == ref_w.shape == (scores.shape[0], topk)
    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    assert torch.isfinite(weights).all(), 'nonfinite output weights'
    id_ok = torch.equal(ids, ref_i)
    max_abs = float((weights - ref_w).abs().max()) if weights.numel() else 0.0
    relative = (weights - ref_w).abs() / ref_w.abs().clamp_min(1e-20)
    max_rel = float(relative.max()) if weights.numel() else 0.0
    flat, ref_flat = (weights.double().flatten(), ref_w.double().flatten())
    norm = flat.norm() * ref_flat.norm()
    cos = float(torch.dot(flat, ref_flat) / norm) if float(norm) > 0 else float(torch.equal(weights, ref_w))
    weights_exact = torch.equal(weights, ref_w)
    if kwargs.get('packed_out') is not None:
        bits = weights.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 65535
        assert torch.equal(kwargs['packed_out'], ids << 16 | bits), 'incorrect packed encoding'
    strict = exact_weights or os.environ.get('MOE_ROUTER_REQUIRE_EXACT_WEIGHTS') == '1'
    close = torch.allclose(weights, ref_w, rtol=0 if strict else 0.0001, atol=0 if strict else 1e-06)
    if compare_reference and (not id_ok or not close):
        failed = {'case': label, 'M': scores.shape[0], 'N': scores.shape[1], 'K': topk, 'dtype': str(scores.dtype), 'status': 'accuracy_failed', 'scoring_func': kwargs.get('scoring_func', 'sigmoid'), 'ordered_ids_exact': bool(id_ok), 'weights_bit_exact': bool(weights_exact), 'weight_mismatches': int((weights != ref_w).sum()), 'max_abs_error': max_abs, 'max_rel_error': max_rel, 'latency_ms': None, 'graph_replay_checked': False}
        RESULTS.append(failed)
        dest = os.environ.get('MOE_ROUTER_RESULTS')
        if dest:
            with Path(dest).open('a', encoding='utf-8') as f:
                f.write(json.dumps(failed) + '\n')
        print(f'[router:{label}] ACCURACY FAILED {json.dumps(failed)}', flush=True)
    if compare_reference:
        assert id_ok, f'{label}: ordered topk IDs differ in {int((ids != ref_i).sum())} slots'
        torch.testing.assert_close(weights, ref_w, rtol=0 if strict else 0.0001, atol=0 if strict else 1e-06)
        assert cos >= 0.99999
    routed = topk - kwargs.get('num_fused_shared_experts', 0)
    active = ids[:, :routed] >= 0
    assert (ids[:, :routed][active] < scores.shape[1]).all()
    for k in range(1, routed):
        assert not ((ids[:, k:k + 1] == ids[:, :k]) & (ids[:, k:k + 1] >= 0)).any(), 'duplicate routed expert'
    for _ in range(10):
        fn(scores, bias, topk=topk, **kwargs)
    torch.cuda.synchronize()
    from triton.testing import do_bench_cudagraph
    ms = float(do_bench_cudagraph(lambda: fn(scores, bias, topk=topk, **kwargs), rep=50))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gw, gi = fn(scores, bias, topk=topk, **kwargs)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(gw, weights, rtol=0, atol=0)
    assert torch.equal(gi, ids)
    logical_bytes = scores.numel() * scores.element_size() + weights.numel() * weights.element_size() + ids.numel() * ids.element_size()
    if bias is not None:
        logical_bytes += bias.numel() * bias.element_size()
    for key in ('bias_alt', 'input_ids', 'num_token_non_padded'):
        t = kwargs.get(key)
        if t is not None:
            logical_bytes += t.numel() * t.element_size()
    if kwargs.get('packed_out') is not None:
        logical_bytes += kwargs['packed_out'].numel() * 4
    gbps = logical_bytes / (ms * 1000000.0)
    row = {'case': label, 'M': scores.shape[0], 'N': scores.shape[1], 'K': topk, 'dtype': str(scores.dtype), 'scoring_func': kwargs.get('scoring_func', 'sigmoid'), 'groups': kwargs.get('num_expert_group', 1), 'cos_sim': cos, 'ordered_ids_exact': bool(id_ok), 'weights_bit_exact': bool(weights_exact), 'max_abs_error': max_abs, 'max_rel_error': max_rel, 'latency_ms': ms, 'effective_GB_s': gbps, 'byte_definition': 'logical input+output bytes, not measured physical HBM traffic', 'graph_replay_checked': True}
    RESULTS.append(row)
    dest = os.environ.get('MOE_ROUTER_RESULTS')
    if dest:
        with Path(dest).open('a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')
    print(f'[router:{label}] T={scores.shape[0]:6d} N={scores.shape[1]:4d} topk={topk:2d} dtype={str(scores.dtype):14s} cos_sim={cos:.8f} ids_exact={int(id_ok)} w_exact={int(weights_exact)} max_abs={max_abs:.3e} OK {ms:9.5f} ms {gbps:9.1f} GB/s (effective)', flush=True)
    return (weights, ids)


# Standalone full/CI matrices (46 full / 39 CI)
def matrix_moe_fused_gate(scores, bias, topk, **kwargs):
    return check_and_report('triton', _gate, scores, bias, topk, reference_gate, kwargs=kwargs)

class matrix_DefinitionAOT:
    """Independent torch mathematics, NOT an sgl_kernel binary integration."""

    @staticmethod
    def topk_softmax(w, i, gating, renormalize):
        rw, ri = reference_gate(gating, None, w.shape[1], 'softmax', renormalize=renormalize)
        w.copy_(rw)
        i.copy_(ri)

    @staticmethod
    def topk_sigmoid(w, i, gating, renormalize, bias):
        if bias is None:
            bias = torch.zeros(gating.shape[1], device=gating.device)
        rw, ri = reference_gate(gating, bias, w.shape[1], 'sigmoid', renormalize=renormalize)
        w.copy_(rw)
        i.copy_(ri)
DEVICE = 'cuda'

def matrix__scatter_by_expert(weights: torch.Tensor, indices: torch.Tensor, num_columns: int) -> torch.Tensor:
    """Scatter (weight, id) pairs into a dense ``[M, num_columns]`` tensor.

    Makes the comparison independent of the per-row slot order, so the test does
    not depend on how ties between equal scores are broken.
    """
    dense = torch.zeros((weights.shape[0], num_columns), dtype=torch.float32, device=weights.device)
    dense.scatter_(1, indices.long(), weights.float())
    return dense

def matrix__reference_gate(scores: torch.Tensor, bias: torch.Tensor, topk: int, scoring_func: str, num_fused_shared_experts: int, renormalize: bool, routed_scaling_factor: float, apply_routed_scaling_factor_on_output: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Definition-based eager reference matching the CUDA fused-gate semantics."""
    if scoring_func == 'sigmoid':
        activated = scores.sigmoid()
    else:
        activated = torch.nn.functional.softplus(scores).sqrt()
    biased = activated + bias.unsqueeze(0)
    num_experts = scores.size(1)
    num_routed = topk - num_fused_shared_experts
    bs = biased.size(0)
    work = biased.clone()
    arange = torch.arange(num_experts, device=scores.device).unsqueeze(0)
    routed_idx = torch.empty(bs, num_routed, dtype=torch.int32, device=scores.device)
    routed_wgt = torch.empty(bs, num_routed, dtype=torch.float32, device=scores.device)
    for k in range(num_routed):
        vals, _ = work.max(dim=1, keepdim=True)
        lane = torch.where(work == vals, arange, num_experts + 1)
        winner = lane.min(dim=1).values.to(torch.int32)
        routed_idx[:, k] = winner
        routed_wgt[:, k] = activated.gather(1, winner.long().unsqueeze(1)).squeeze(1)
        work.scatter_(1, winner.long().unsqueeze(1), float('-inf'))
    routed_sum = routed_wgt.sum(dim=1, keepdim=True)
    weights = torch.empty(bs, topk, dtype=torch.float32, device=scores.device)
    indices = torch.empty(bs, topk, dtype=torch.int32, device=scores.device)
    weights[:, :num_routed] = routed_wgt
    indices[:, :num_routed] = routed_idx
    if num_fused_shared_experts > 0:
        weights[:, num_routed:] = routed_sum / routed_scaling_factor
        for j in range(num_fused_shared_experts):
            indices[:, num_routed + j] = num_experts + j
    if renormalize:
        norm = torch.where(routed_sum > 0.0, routed_sum, torch.ones_like(routed_sum))
        weights = weights / norm
    if apply_routed_scaling_factor_on_output:
        weights = weights * routed_scaling_factor
    return (weights, indices)

def matrix__make_inputs(M: int, num_experts: int, seed: int):
    torch.manual_seed(seed)
    scores = torch.randn(M, num_experts, dtype=torch.float32, device=DEVICE) * 2.0
    bias = torch.randn(num_experts, dtype=torch.float32, device=DEVICE) * 0.5
    return (scores, bias)
matrix__REFERENCE_CASES = get_ci_test_range([(1, 128, 4, 'sigmoid', 0, True, True), (7, 256, 6, 'sqrtsoftplus', 1, False, False), (64, 384, 8, 'sigmoid', 0, False, True), (256, 384, 6, 'sqrtsoftplus', 0, True, False), (1024, 128, 4, 'sigmoid', 1, False, False), (1024, 512, 8, 'sqrtsoftplus', 1, True, True), (7, 512, 4, 'sigmoid', 1, True, False), (256, 256, 8, 'sqrtsoftplus', 0, False, True)], [(1, 128, 4, 'sigmoid', 0, True, True), (7, 256, 6, 'sqrtsoftplus', 1, False, False), (64, 384, 8, 'sigmoid', 0, False, True), (256, 384, 6, 'sqrtsoftplus', 0, True, False), (1024, 128, 4, 'sigmoid', 1, False, False), (1024, 512, 8, 'sqrtsoftplus', 1, True, True)])

@pytest.mark.parametrize('M,num_experts,topk,scoring_func,num_shared,renormalize,apply_scale', matrix__REFERENCE_CASES)
def test_moe_fused_gate_matches_reference(M: int, num_experts: int, topk: int, scoring_func: str, num_shared: int, renormalize: bool, apply_scale: bool) -> None:
    scores, bias = matrix__make_inputs(M, num_experts, seed=num_experts * 100 + topk)
    scale = 2.5
    kwargs = dict(topk=topk, scoring_func=scoring_func, num_fused_shared_experts=num_shared, renormalize=renormalize, routed_scaling_factor=scale, apply_routed_scaling_factor_on_output=apply_scale)
    triton_w, triton_i = matrix_moe_fused_gate(scores, bias, **kwargs)
    ref_w, ref_i = matrix__reference_gate(scores, bias, **kwargs)
    torch.cuda.synchronize()
    num_columns = num_experts + num_shared
    torch.testing.assert_close(matrix__scatter_by_expert(triton_w, triton_i, num_columns), matrix__scatter_by_expert(ref_w, ref_i, num_columns), rtol=0.0001, atol=1e-05)

@pytest.mark.parametrize('M,num_experts,topk,num_shared,scoring_func', [(8192, 256, 6, 0, 'sqrtsoftplus'), (8192, 384, 6, 0, 'sqrtsoftplus'), (8192, 384, 8, 0, 'sigmoid'), (8192, 512, 8, 1, 'sigmoid')])
def test_moe_fused_gate_large_model_shapes(M: int, num_experts: int, topk: int, num_shared: int, scoring_func: str) -> None:
    scores, bias = matrix__make_inputs(M, num_experts, seed=123)
    kwargs = dict(scoring_func=scoring_func, num_fused_shared_experts=num_shared, renormalize=True, routed_scaling_factor=2.5, apply_routed_scaling_factor_on_output=True)
    _assert_exact_route(scores, bias, topk, **kwargs)

@pytest.mark.parametrize('num_experts,topk', [(256, 6), (384, 8), (512, 8)])
@pytest.mark.parametrize('apply_scale', [True, False])
def test_moe_fused_gate_matches_production_definition(num_experts: int, topk: int, apply_scale: bool) -> None:
    """Check ungrouped production-shaped parameters against the standalone definition."""
    biased_grouped_topk_impl = reference_production
    M = 128
    scores, bias = matrix__make_inputs(M, num_experts, seed=7)
    scale = 2.5
    triton_w, triton_i = matrix_moe_fused_gate(scores, bias, topk=topk, scoring_func='sigmoid', renormalize=True, routed_scaling_factor=scale, apply_routed_scaling_factor_on_output=apply_scale)
    hidden_states = torch.empty((M, 1), dtype=torch.float32, device=DEVICE)
    ref_w, ref_i = biased_grouped_topk_impl(hidden_states, scores, bias, topk, True, num_expert_group=1, topk_group=1, routed_scaling_factor=scale, apply_routed_scaling_factor_on_output=apply_scale)
    torch.cuda.synchronize()
    torch.testing.assert_close(matrix__scatter_by_expert(triton_w, triton_i, num_experts), matrix__scatter_by_expert(ref_w, ref_i, num_experts), rtol=0.0001, atol=1e-05)

def test_moe_fused_gate_shapes_and_dtypes() -> None:
    """Output shapes/dtypes and renormalized weights for a DeepSeek-V4 config."""
    M, N, K = (64, 256, 8)
    scores, bias = matrix__make_inputs(M, N, seed=0)
    w, i = matrix_moe_fused_gate(scores, bias, topk=K, scoring_func='sqrtsoftplus')
    assert w.shape == (M, K)
    assert i.shape == (M, K)
    assert w.dtype == torch.float32
    assert i.dtype == torch.int32
    assert (i >= 0).all() and (i < N).all()
    torch.testing.assert_close(w.sum(dim=1), torch.ones(M, device=DEVICE), rtol=0.0001, atol=1e-05)

def matrix__reference_softmax(gating: torch.Tensor, topk: int, renormalize: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Plain-softmax topk reference (the AOT ``topk_softmax`` semantics)."""
    num_experts = gating.size(1)
    probs = torch.softmax(gating.float(), dim=-1)
    work = gating.float().clone()
    arange = torch.arange(num_experts, device=gating.device).unsqueeze(0)
    M = gating.size(0)
    idx = torch.empty(M, topk, dtype=torch.int32, device=gating.device)
    wgt = torch.empty(M, topk, dtype=torch.float32, device=gating.device)
    for k in range(topk):
        vals, _ = work.max(dim=1, keepdim=True)
        lane = torch.where(work == vals, arange, num_experts + 1)
        winner = lane.min(dim=1).values.to(torch.int32)
        idx[:, k] = winner
        wgt[:, k] = probs.gather(1, winner.long().unsqueeze(1)).squeeze(1)
        work.scatter_(1, winner.long().unsqueeze(1), float('-inf'))
    if renormalize:
        wgt = wgt / wgt.sum(dim=1, keepdim=True)
    return (wgt, idx)
matrix__SOFTMAX_AOT_CASES = get_ci_test_range([(1, 128, 4, True, torch.float32), (1, 256, 8, False, torch.float32), (200, 128, 4, False, torch.bfloat16), (200, 256, 8, True, torch.bfloat16), (1024, 512, 6, False, torch.float32), (1024, 512, 6, True, torch.bfloat16)], [(1, 128, 4, True, torch.float32), (200, 256, 8, False, torch.bfloat16), (1024, 512, 6, False, torch.float32)])

@pytest.mark.parametrize('M,num_experts,topk,renormalize,dtype', matrix__SOFTMAX_AOT_CASES)
def test_moe_fused_gate_softmax_matches_aot_definition(M: int, num_experts: int, topk: int, renormalize: bool, dtype: torch.dtype) -> None:
    """Check the AOT parameter matrix against independent softmax mathematics."""
    sgl_kernel = matrix_DefinitionAOT
    torch.manual_seed(num_experts * 13 + topk)
    gating = torch.randn(M, num_experts, dtype=dtype, device=DEVICE) * 2.0
    zero_bias = torch.zeros(num_experts, dtype=torch.float32, device=DEVICE)
    tri_w, tri_i = matrix_moe_fused_gate(gating, zero_bias, topk=topk, scoring_func='softmax', renormalize=renormalize)
    ref_w, ref_i = matrix__reference_softmax(gating, topk, renormalize)
    aot_w = torch.empty(M, topk, dtype=torch.float32, device=DEVICE)
    aot_i = torch.empty(M, topk, dtype=torch.int32, device=DEVICE)
    sgl_kernel.topk_softmax(aot_w, aot_i, gating, renormalize)
    torch.cuda.synchronize()
    dense_tri = matrix__scatter_by_expert(tri_w, tri_i, num_experts)
    torch.testing.assert_close(dense_tri, matrix__scatter_by_expert(ref_w, ref_i, num_experts), rtol=0.001, atol=0.001)
    torch.testing.assert_close(dense_tri, matrix__scatter_by_expert(aot_w, aot_i, num_experts), rtol=0.001, atol=0.001)

@pytest.mark.parametrize('M', [1, 8, 32])
def test_moe_fused_gate_softmax_none_bias_matches_zero_bias(M: int) -> None:
    torch.manual_seed(M)
    scores = torch.randn(M, 256, dtype=torch.float32, device=DEVICE)
    zero_bias = torch.zeros(256, dtype=torch.float32, device=DEVICE)
    none_w, none_i = matrix_moe_fused_gate(scores, None, topk=8, scoring_func='softmax', renormalize=True)
    zero_w, zero_i = matrix_moe_fused_gate(scores, zero_bias, topk=8, scoring_func='softmax', renormalize=True)
    torch.testing.assert_close(none_w, zero_w, rtol=0, atol=0)
    torch.testing.assert_close(none_i, zero_i, rtol=0, atol=0)
matrix__SIGMOID_AOT_CASES = get_ci_test_range([(1, 128, 4, True, True, torch.float32), (1, 256, 8, False, True, torch.float32), (200, 128, 4, False, False, torch.bfloat16), (200, 256, 8, True, False, torch.bfloat16), (1024, 128, 4, True, False, torch.float32), (1024, 256, 8, False, True, torch.bfloat16)], [(1, 128, 4, True, True, torch.float32), (200, 128, 4, False, False, torch.bfloat16), (200, 256, 8, True, False, torch.bfloat16), (1024, 256, 8, False, True, torch.float32)])

@pytest.mark.parametrize('M,num_experts,topk,renormalize,with_bias,dtype', matrix__SIGMOID_AOT_CASES)
def test_moe_fused_gate_sigmoid_matches_aot_definition(M: int, num_experts: int, topk: int, renormalize: bool, with_bias: bool, dtype: torch.dtype) -> None:
    """Check the AOT parameter matrix against independent sigmoid mathematics."""
    sgl_kernel = matrix_DefinitionAOT
    torch.manual_seed(num_experts * 17 + topk)
    gating = torch.randn(M, num_experts, dtype=dtype, device=DEVICE) * 2.0
    bias = torch.randn(num_experts, dtype=torch.float32, device=DEVICE) * 0.5 if with_bias else torch.zeros(num_experts, dtype=torch.float32, device=DEVICE)
    tri_w, tri_i = matrix_moe_fused_gate(gating, bias, topk=topk, scoring_func='sigmoid', renormalize=renormalize)
    aot_w = torch.empty(M, topk, dtype=torch.float32, device=DEVICE)
    aot_i = torch.empty(M, topk, dtype=torch.int32, device=DEVICE)
    sgl_kernel.topk_sigmoid(aot_w, aot_i, gating, renormalize, bias if with_bias else None)
    torch.cuda.synchronize()
    torch.testing.assert_close(matrix__scatter_by_expert(tri_w, tri_i, num_experts), matrix__scatter_by_expert(aot_w, aot_i, num_experts), rtol=0.001, atol=0.001)

@pytest.mark.parametrize('num_experts,num_expert_group,topk_group,topk', [(256, 8, 4, 8), (128, 8, 4, 6), (256, 4, 2, 8)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_moe_fused_gate_grouped_matches_production_definition(num_experts: int, num_expert_group: int, topk_group: int, topk: int, dtype: torch.dtype) -> None:
    """Check grouped parameters with FP32 activation and top-two group scores."""
    biased_grouped_topk_impl = reference_production
    M = 256
    torch.manual_seed(num_experts * 7 + num_expert_group * 13 + topk)
    gating = torch.randn(M, num_experts, dtype=dtype, device=DEVICE) * 2.0
    bias = torch.randn(num_experts, dtype=torch.float32, device=DEVICE) * 0.5
    hidden = torch.randn(M, 16, dtype=dtype, device=DEVICE)
    tri_w, tri_i = matrix_moe_fused_gate(gating, bias, topk=topk, scoring_func='sigmoid', renormalize=True, num_expert_group=num_expert_group, topk_group=topk_group)
    ref_w, ref_i = biased_grouped_topk_impl(hidden, gating, bias, topk, True, num_expert_group=num_expert_group, topk_group=topk_group, num_fused_shared_experts=0, routed_scaling_factor=1.0, apply_routed_scaling_factor_on_output=False)
    torch.cuda.synchronize()
    torch.testing.assert_close(matrix__scatter_by_expert(tri_w, tri_i, num_experts), matrix__scatter_by_expert(ref_w, ref_i, num_experts), rtol=0.001, atol=0.001)

@pytest.mark.parametrize('num_experts,num_expert_group,topk_group,topk,num_fused_shared_experts', [(256, 8, 4, 8, 0), (256, 8, 4, 9, 1)])
def test_grouped_standalone_api_matches_definition(num_experts: int, num_expert_group: int, topk_group: int, topk: int, num_fused_shared_experts: int) -> None:
    """Check the grouped API and shared slots against independent mathematics."""
    M = 256
    torch.manual_seed(num_experts * 3 + num_expert_group * 5 + topk)
    gating = torch.randn(M, num_experts, dtype=torch.float32, device=DEVICE) * 2.0
    bias = torch.randn(num_experts, dtype=torch.float32, device=DEVICE) * 0.5
    hidden = torch.randn(M, 16, dtype=torch.float32, device=DEVICE)
    kwargs = dict(num_expert_group=num_expert_group, topk_group=topk_group, num_fused_shared_experts=num_fused_shared_experts, routed_scaling_factor=2.5, apply_routed_scaling_factor_on_output=False)
    def_w, def_i = reference_production(hidden, gating, bias, topk, True, **kwargs)
    jit_w, jit_i = matrix_moe_fused_gate(gating, bias, topk, renormalize=True, **kwargs)
    torch.cuda.synchronize()
    topk_routed = topk - num_fused_shared_experts
    torch.testing.assert_close(matrix__scatter_by_expert(def_w[:, :topk_routed], def_i[:, :topk_routed], num_experts), matrix__scatter_by_expert(jit_w[:, :topk_routed], jit_i[:, :topk_routed], num_experts), rtol=0.001, atol=0.001)

@pytest.mark.parametrize('bias_dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('scoring_func', ['sigmoid', 'sqrtsoftplus'])
def test_moe_fused_gate_accepts_non_fp32_bias(bias_dtype: torch.dtype, scoring_func: str) -> None:
    """A non-fp32 correction bias must route identically to its fp32 copy.

    DeepSeek-V4 stores ``e_score_correction_bias`` in bf16 and topk.py now passes
    it straight through (no host-side fp32 copy per MoE invocation). The kernel
    upcasts the bias to fp32 on load, so a bf16/fp16 bias must be bit-identical to
    casting it to fp32 first -- this guards both the relaxed dtype assertion and
    the removed per-call cast. (top-6 exercises the DSV4 non-pow2 routing width.)
    """
    M, num_experts, topk = (256, 128, 6)
    torch.manual_seed(num_experts * 11 + topk)
    gating = torch.randn(M, num_experts, dtype=torch.float32, device=DEVICE) * 2.0
    bias_lowp = torch.randn(num_experts, dtype=bias_dtype, device=DEVICE) * 0.5
    bias_fp32 = bias_lowp.to(torch.float32)
    lowp_w, lowp_i = matrix_moe_fused_gate(gating, bias_lowp, topk=topk, scoring_func=scoring_func, renormalize=True)
    fp32_w, fp32_i = matrix_moe_fused_gate(gating, bias_fp32, topk=topk, scoring_func=scoring_func, renormalize=True)
    torch.cuda.synchronize()
    torch.testing.assert_close(matrix__scatter_by_expert(lowp_w, lowp_i, num_experts), matrix__scatter_by_expert(fp32_w, fp32_i, num_experts), rtol=0, atol=0)


# Adversarial and invalid-argument cases (43)
def _run(scores, bias, topk, **kwargs):
    ref_kwargs = dict(kwargs)
    if kwargs.get('packed_out') is not None:
        ref_kwargs['packed_out'] = torch.empty_like(kwargs['packed_out'])
    weights, ids = _gate(scores, bias, topk=topk, **kwargs)
    ref_w, ref_i = reference_gate(scores, bias, topk=topk, **ref_kwargs)
    torch.cuda.synchronize()
    assert weights.shape == (scores.shape[0], topk)
    assert ids.shape == weights.shape
    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    assert torch.equal(ids, ref_i), f'ordered IDs differ: got={ids.tolist()} expected={ref_i.tolist()}'
    if torch.isfinite(ref_w).all():
        assert torch.isfinite(weights).all(), 'finite reference produced nonfinite weights'
        err = (weights - ref_w).abs()
        exact = torch.equal(weights, ref_w)
        max_abs = float(err.max()) if err.numel() else 0.0
        print(f'[router-edge] weights_exact={exact} max_abs={max_abs:.3e}')
        strict = True
        torch.testing.assert_close(weights, ref_w, rtol=0 if strict else 0.0001, atol=0 if strict else 1e-06)
    else:
        assert torch.equal(torch.isnan(weights), torch.isnan(ref_w))
        assert torch.equal(torch.isinf(weights), torch.isinf(ref_w))
        finite = torch.isfinite(ref_w)
        torch.testing.assert_close(weights[finite], ref_w[finite], rtol=0, atol=0)
    if kwargs.get('packed_out') is not None:
        assert torch.equal(kwargs['packed_out'], ref_kwargs['packed_out']), 'packed output differs from reference'
    RESULTS.append({'target': TARGET_MODULE, 'M': scores.shape[0], 'N': scores.shape[1], 'K': topk, 'weights_bit_exact': bool(torch.equal(weights, ref_w))})
    return (weights, ids)

def _inputs(m, n, dtype=torch.float32, *, seed=123):
    gen = torch.Generator(device='cpu').manual_seed(seed)
    s = torch.randn((m, n), generator=gen, dtype=torch.float32).to(device='cuda', dtype=dtype)
    b = torch.randn((n,), generator=gen, dtype=torch.float32).to(device='cuda')
    return (s, b)

@pytest.mark.parametrize('m,n,k', [(1, 3, 1), (3, 5, 3), (2, 7, 5), (0, 5, 2)])
def test_odd_shapes_and_empty_rows(m, n, k):
    s, b = _inputs(m, n)
    _run(s, b, k)

@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_supported_activation_dtypes(dtype):
    s, b = _inputs(3, 7, dtype)
    _run(s, b, 3)

def test_exact_ties_choose_lowest_ids_and_duplicate_group_maxima_count_twice():
    s = torch.zeros((2, 8), device='cuda', dtype=torch.float32)
    b = torch.zeros(8, device='cuda', dtype=torch.float32)
    _run(s, b, 4, num_expert_group=4, topk_group=2)

@pytest.mark.parametrize('bias_values,expected', [([float('inf'), -float('inf'), 0.0, 0.0], 2), ([float('inf'), -float('inf'), -float('inf'), -float('inf')], 0)])
def test_nonfinite_group_sums_keep_valid_expert_ids(bias_values, expected):
    s = torch.zeros((1, 4), device='cuda')
    b = torch.tensor(bias_values, device='cuda')
    w, ids = _run(s, b, 1, num_expert_group=2, topk_group=1)
    assert ids.item() == expected
    assert w.item() == 1.0

@pytest.mark.parametrize('cap', [0.7, 1.3, 2.5, 7.0, 1e-12, 1000000000000.0, 2e+38])
def test_softcap_scalar_rounding(cap):
    s, b = _inputs(17, 33, seed=3057)
    _run(s * 2, b * 0.5, 6, scoring_func='softmax', moe_softcapping=cap, routed_scaling_factor=2.5, apply_routed_scaling_factor_on_output=True)

def test_near_ties_remain_ordered_by_value_not_rounded_tie():
    s = torch.tensor([[0.0, 1e-06, 2e-06, -1e-06, 0.5]], device='cuda')
    b = torch.zeros(5, device='cuda')
    _run(s, b, 4)

def test_all_negative_infinity_ranking_bias_has_no_duplicate_ids():
    s = torch.tensor([[0.0, 1.0, -1.0, 2.0, -2.0]], device='cuda')
    b = torch.full((5,), -float('inf'), device='cuda')
    _, ids = _run(s, b, 5, renormalize=False)
    assert ids[0].tolist() == [0, 1, 2, 3, 4]

@pytest.mark.parametrize('shared', [1, 2])
def test_shared_experts_and_scaling(shared):
    s, b = _inputs(3, 7)
    _run(s, b, 4 + shared, num_fused_shared_experts=shared, routed_scaling_factor=2.5, apply_routed_scaling_factor_on_output=True)

def test_alternate_token_bias_and_strided_token_ids():
    s, b = _inputs(4, 7)
    alt = torch.linspace(-0.4, 0.4, 14, device='cuda')[::2]
    ids = torch.tensor([11, 7, 11, 7], device='cuda', dtype=torch.int64)[::1]
    _run(s, b, 3, bias_alt=alt, input_ids=ids, bias_alt_token_id=11)

@pytest.mark.parametrize('live', [0, 1, 3])
def test_padding_including_zero_and_noncontiguous_singleton(live):
    s, b = _inputs(4, 7)
    count = torch.tensor([live, 99], device='cuda', dtype=torch.int32)[::2]
    assert count.numel() == 1 and count.stride(0) == 2
    _run(s, b, 3, num_token_non_padded=count)

def test_packed_output_matches_bfloat16_bit_encoding():
    s, b = _inputs(3, 7)
    packed = torch.empty((3, 4), device='cuda', dtype=torch.int32)
    w, i = _run(s, b, 4, packed_out=packed)
    bits = w.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 65535
    assert torch.equal(packed, i << 16 | bits)

def test_group_scores_that_overflow_to_negative_infinity():
    s = torch.zeros((2, 8), device='cuda')
    b = torch.full((8,), -3e+38, device='cuda')
    _, ids = _run(s, b, 4, num_expert_group=4, topk_group=2)
    assert ids[0].tolist() == [0, 1, 2, 3]

def test_softcap_reconstruction_preserves_operation_order():
    s = torch.tensor([[2e+38, 0.0]], device='cuda')
    _run(s, None, 1, scoring_func='softmax', moe_softcapping=2e+38)

def test_overlapping_packed_destination_is_rejected():
    s, b = _inputs(3, 7)
    packed = torch.empty((3, 1), device='cuda', dtype=torch.int32).expand(3, 4)
    with pytest.raises((AssertionError, ValueError)):
        _gate(s, b, 4, packed_out=packed)

def test_softmax_softcap_and_no_bias():
    s, _ = _inputs(3, 7)
    _run(s, None, 4, scoring_func='softmax', moe_softcapping=1.3)

def test_renormalization_epsilon():
    s, b = _inputs(3, 7)
    _run(s, b, 4, renormalize_epsilon=0.125)

@pytest.mark.parametrize('kwargs', [{'topk': 0}, {'topk': 8}, {'topk': 2, 'num_fused_shared_experts': 2}, {'topk': 2, 'num_expert_group': 3}, {'topk': 3, 'num_expert_group': 4, 'topk_group': 1}, {'topk': 2, 'routed_scaling_factor': 0.0, 'num_fused_shared_experts': 1}, {'topk': 2, 'renormalize_epsilon': -1.0}, {'topk': 2, 'moe_softcapping': float('nan')}])
def test_invalid_parameters_fail(kwargs):
    s, b = _inputs(2, 7)
    topk = kwargs['topk']
    options = {key: value for key, value in kwargs.items() if key != 'topk'}
    with pytest.raises((AssertionError, ValueError, RuntimeError)):
        _gate(s, b, topk=topk, **options)

def test_packed_expert_id_overflow_is_rejected():
    s = torch.zeros((1, 2), device='cuda')
    b = torch.zeros(2, device='cuda')
    packed = torch.empty((1, 65537), device='cuda', dtype=torch.int32)
    with pytest.raises((AssertionError, ValueError)):
        _gate(s, b, 65537, num_fused_shared_experts=65536, packed_out=packed)

def test_shared_tail_larger_than_expert_axis():
    s, b = _inputs(2, 3)
    _run(s, b, 10, num_fused_shared_experts=8, routed_scaling_factor=2.5)

def test_softmax_alternate_bias_padding_and_packing():
    s, b = _inputs(4, 7)
    alt = b.flip(0).contiguous()
    token_ids = torch.tensor([11, 7, 11, 7], device='cuda', dtype=torch.int64)
    live = torch.tensor([2], device='cuda', dtype=torch.int32)
    packed = torch.empty((4, 4), device='cuda', dtype=torch.int32)
    _run(s, b, 4, scoring_func='softmax', bias_alt=alt, input_ids=token_ids, bias_alt_token_id=11, num_token_non_padded=live, packed_out=packed)

def test_noncontiguous_score_and_bias_strides():
    base, _ = _inputs(4, 14)
    s = base[:, ::2]
    b = torch.arange(14, device='cuda', dtype=torch.float32)[::2]
    assert not s.is_contiguous() and (not b.is_contiguous())
    _run(s, b, 3)


# Legacy independent sigmoid reference and graph timer
def _launch(scores, bias, renormalize=True, apply_scale=False):
    return _gate(scores, bias, K, scoring_func='sigmoid', num_fused_shared_experts=0, renormalize=renormalize, routed_scaling_factor=ROUTED_SCALING_FACTOR, apply_routed_scaling_factor_on_output=apply_scale, num_expert_group=1, topk_group=1)

def _torch_reference(scores, bias, renormalize=True, apply_scale=False):
    """fp32 参考: 迭代 argmax, 平局取最小 expert id (torch.argmax 取首个 max, 一致)。"""
    activated = torch.sigmoid(scores.float())
    biased = activated + bias.float()[None, :]
    biased = torch.nan_to_num(biased, nan=-1e+30)
    m = scores.shape[0]
    weights = torch.empty(m, K, dtype=torch.float32, device=scores.device)
    ids = torch.empty(m, K, dtype=torch.int32, device=scores.device)
    cur = biased.clone()
    rows = torch.arange(m, device=scores.device)
    for j in range(K):
        win = cur.argmax(dim=1)
        ids[:, j] = win.to(torch.int32)
        weights[:, j] = activated[rows, win]
        cur[rows, win] = -float('inf')
    if renormalize:
        s = weights.sum(dim=1, keepdim=True)
        weights = weights / torch.where(s > 0, s, torch.ones_like(s))
    if apply_scale:
        weights = weights * ROUTED_SCALING_FACTOR
    return (weights, ids)

def _cos_sim(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()

def _bench_burst(fn, warm=0.6, run=0.6):
    """Prewarm then measure graph replay; logical bandwidth, not physical HBM."""
    from triton.testing import do_bench_cudagraph
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    return do_bench_cudagraph(fn, rep=100)

N = 896
K = 16
TOKEN_COUNTS = (2048, 4096, 8192, 16384)
ROUTED_SCALING_FACTOR = 2.5
COS_THRESHOLD = 0.99999
SINGLE_DIE_TARGET = 1300.0


@pytest.mark.parametrize('num_tokens', TOKEN_COUNTS)
@pytest.mark.parametrize('apply_scale', [False, True])
def test_legacy_accuracy_and_bandwidth(num_tokens, apply_scale):
    # Match the original sequence, including RNG consumption before each case.
    torch.manual_seed(0)
    for m in TOKEN_COUNTS:
        for scale in (False, True):
            scores = torch.randn(m, N, device='cuda', dtype=torch.float32)
            bias = torch.randn(N, device='cuda', dtype=torch.float32)
            if m == num_tokens and scale == apply_scale:
                break
        else:
            continue
        break
    weights, ids = _launch(scores, bias, apply_scale=apply_scale)
    ref_w, ref_i = _torch_reference(scores, bias, apply_scale=apply_scale)
    torch.cuda.synchronize()
    assert torch.equal(ids, ref_i), 'legacy ordered IDs differ'
    assert torch.equal(weights, ref_w), 'legacy FP32 weights differ'
    assert torch.isfinite(weights).all()
    cosine = _cos_sim(weights, ref_w)
    assert cosine >= COS_THRESHOLD
    ms = float(_bench_burst(lambda: _launch(scores, bias, apply_scale=apply_scale)))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gw, gi = _launch(scores, bias, apply_scale=apply_scale)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(gw, ref_w) and torch.equal(gi, ref_i)
    logical_bytes = num_tokens * (N * 4 + K * 4 * 2) + N * 4
    row = dict(case='legacy', M=num_tokens, N=N, K=K, apply_scale=apply_scale,
               latency_ms=ms, effective_GB_s=logical_bytes/(ms*1e6),
               ordered_ids_exact=True, weights_exact=True, max_abs_error=0.0,
               cos_sim=cosine, graph_replay_checked=True)
    RESULTS.append(row)
    destination = os.environ.get('MOE_ROUTER_RESULTS')
    if destination:
        with Path(destination).open('a') as stream:
            stream.write(json.dumps(row) + '\n')
    print(f"[router] T={num_tokens} topk={K} scale={int(apply_scale)} "
          f"cos_sim={cosine:.8f} ids_exact=1 w_exact=1 max_abs=0 "
          f"{ms:.4f} ms {row['effective_GB_s']:.1f} GB/s (logical)", flush=True)

@triton.jit
def _debug_softmax_prob_kernel(
    S,
    B,
    P_OUT,
    SUM_OUT,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_CUSTOM_SUM: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)

    # 与 moe_fused_gate 的 softmax 路径保持一致
    x = tl.load(
        S + row * N + offs,
        offs < N,
        other=0.0,
    ).to(tl.float32)

    bias = tl.load(
        B + offs,
        offs < N,
        other=0.0,
    ).to(tl.float32)

    logits = tl.where(
        offs < N,
        x + bias,
        -float("inf"),
    )

    max_logit = tl.max(logits, 0)
    exp_row = tl.exp(logits - max_logit)

    if USE_CUSTOM_SUM:
        row_sum = _target._softmax_vendor_sum(
            exp_row,
            BLOCK_N,
            False,
        )
    else:
        row_sum = tl.sum(exp_row, 0)

    probs = exp_row / row_sum

    tl.store(SUM_OUT + row, row_sum)
    tl.store(
        P_OUT + row * N + offs,
        probs,
        offs < N,
    )
    
def compare_softmax_intermediate(scores, bias):
    scores = scores.float().contiguous()
    bias = bias.float().contiguous()

    M, N = scores.shape
    BLOCK_N = triton.next_power_of_2(N)

    ref_probs = torch.softmax(
        scores + bias.unsqueeze(0),
        dim=-1,
    )

    for use_custom_sum in (True, False):
        probs = torch.empty_like(scores)
        row_sums = torch.empty(
            (M,),
            device=scores.device,
            dtype=torch.float32,
        )

        _debug_softmax_prob_kernel[(M,)](
            scores,
            bias,
            probs,
            row_sums,
            N,
            BLOCK_N,
            use_custom_sum,
            num_warps=1,
            enable_fp_fusion=False,
        )

        torch.cuda.synchronize()

        diff = (probs - ref_probs).abs()
        mismatch_count = int((probs != ref_probs).sum().item())

        print(
            f"[softmax-debug] N={N}, "
            f"use_custom_sum={use_custom_sum}, "
            f"mismatches={mismatch_count}/{probs.numel()}, "
            f"max_abs={diff.max().item():.12e}, "
            f"max_row_sum={row_sums.max().item():.12g}, "
            f"min_row_sum={row_sums.min().item():.12g}",
            flush=True,
        )

def _assert_exact_route(scores, bias, topk, **kwargs):
    weights, ids = _gate(scores, bias, topk, **kwargs)
    ref_w, ref_i = reference_gate(scores, bias, topk, **kwargs)
    torch.cuda.synchronize()
    mismatches = int((weights != ref_w).sum())
    print(f'[router-exact] M={scores.shape[0]} N={scores.shape[1]} topk={topk} '
          f'dtype={scores.dtype} ids_exact={torch.equal(ids, ref_i)} '
          f'weight_mismatches={mismatches}', flush=True)
    if not torch.equal(weights, ref_w):
        compare_softmax_intermediate(scores, bias)
    assert torch.equal(ids, ref_i)
    assert torch.equal(weights, ref_w)


_EXTENDED_CASES = [
    (m, 128, k, 'sigmoid', {})
    for m in (1, 5, 8, 16, 32, 65) for k in (31, 32, 63, 64, 65, 128)
] + [
    (m, n, 7, 'softmax', dict(num_fused_shared_experts=1,
         routed_scaling_factor=2.5, apply_routed_scaling_factor_on_output=True))
    for n in (32, 64, 127, 128, 129, 255, 256, 257, 512, 513, 768, 896, 1000, 1024)
    for m in (1, 17)
] + [
    (17, 33, 6, 'softmax', dict(moe_softcapping=cap,
         routed_scaling_factor=2.5, apply_routed_scaling_factor_on_output=True))
    for cap in (0.7, 1.3, 2.5, 7.0, 1e-12, 1e12)
]


@pytest.mark.parametrize('m,n,k,scoring,kwargs', _EXTENDED_CASES)
def test_extended_routing(m, n, k, scoring, kwargs):
    torch.manual_seed(3001 + m + n + k)
    scores = torch.randn(m, n, device='cuda', dtype=torch.float32) * 2
    bias = torch.randn(n, device='cuda', dtype=torch.float32) * 0.5
    _assert_exact_route(scores, bias, k, scoring_func=scoring, **kwargs)


@pytest.mark.parametrize('scoring', ['sigmoid', 'sqrtsoftplus', 'softmax'])
@pytest.mark.parametrize('n,k', [(3,1), (7,3), (33,5), (128,7), (256,16), (512,6), (896,16)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_general_scoring_and_dtype(scoring, n, k, dtype):
    torch.manual_seed(71 + n + k)
    scores = torch.randn(17, n, device='cuda', dtype=dtype) * 2
    bias = torch.randn(n, device='cuda') * 0.5
    _assert_exact_route(scores, bias, k + 1, scoring_func=scoring,
        num_fused_shared_experts=1, routed_scaling_factor=2.5,
        renormalize=True, apply_routed_scaling_factor_on_output=True)


@pytest.mark.parametrize('k', [1,2,3,5,7,9,15,17,31,32,33,63,64,65])
def test_general_selected_width(k):
    torch.manual_seed(k + 97)
    scores = torch.randn(5, 128, device='cuda')
    bias = torch.randn(128, device='cuda')
    _assert_exact_route(scores, bias, k)


@triton.jit
def _check_selected_sum(X, Y, R: tl.constexpr, B: tl.constexpr,
                        V: tl.constexpr, W: tl.constexpr):
    row = tl.program_id(0)
    ix = tl.arange(0, B)
    x = tl.load(X + row * R + ix, ix < R, 0.0)
    total = _reduce_selected(tl.reshape(x, (1, B)), R, B, 1, V, W)
    tl.store(Y + row, tl.reshape(total, ()))


_reduce_selected = _target._torch_selected_sum if _target is not None else None


@pytest.mark.parametrize('r', [1,2,3,4,5,7,9,15,16,17,31,32,33,63,64,65,127,128,129,130,132,255,256,257,512,513,1024,4096])
@pytest.mark.parametrize('m', [1,5,8,16,17,32,65,257])
def test_selected_sum(m, r):
    v, w, h = _target._selected_sum_config(m, r)
    torch.manual_seed(971 + m + r)
    x = torch.exp(torch.randn(m, r, device='cuda') * 4)
    reference = x.sum(1)
    output = torch.empty_like(reference)
    _check_selected_sum[(m,)](x, output, r, triton.next_power_of_2(r), v, w,
                              num_warps=1, enable_fp_fusion=False)
    torch.cuda.synchronize()
    mismatches = int((output != reference).sum())
    print(f'selected_sum {m} {r} VWH {v} {w} {h} mismatches {mismatches}', flush=True)
    assert torch.equal(output, reference)

_EXTERNAL_REQUIREMENTS = {'test_sglang_integration_moe_fused_gate_matches_reference': [], 'test_sglang_integration_moe_fused_gate_matches_production_impl': ['sglang.srt.layers.moe.topk'], 'test_sglang_integration_moe_fused_gate_shapes_and_dtypes': [], 'test_sglang_integration_moe_fused_gate_softmax_matches_aot': [], 'test_sglang_integration_moe_fused_gate_softmax_none_bias_matches_zero_bias': [], 'test_sglang_integration_moe_fused_gate_sigmoid_matches_aot': [], 'test_sglang_integration_moe_fused_gate_grouped_matches_production_impl': ['sglang.srt.layers.moe.topk'], 'test_sglang_integration_grouped_dispatch_flag_matches_default': ['sglang.srt.environ', 'sglang.srt.layers.moe.topk'], 'test_sglang_integration_moe_fused_gate_accepts_non_fp32_bias': []}


@pytest.fixture
def _require_sglang_integration(request):
    name = request.node.originalname
    required = ['sglang.kernels.ops.moe.moe_fused_gate'] + _EXTERNAL_REQUIREMENTS[name]
    for dependency in required:
        try:
            importlib.import_module(dependency)
        except ModuleNotFoundError as exc:
            missing = exc.name
            if missing and (dependency == missing or dependency.startswith(missing + '.')):
                pytest.skip(f'real SGLang integration dependency unavailable: {missing}')
            raise
    module = importlib.import_module('sglang.kernels.ops.moe.moe_fused_gate')
    global external_gate
    external_gate = module.moe_fused_gate


# Optional REAL SGLang integration matrices (not standalone substitutes)
external_DEVICE = 'cuda'

def external__scatter_by_expert(weights: torch.Tensor, indices: torch.Tensor, num_columns: int) -> torch.Tensor:
    """Scatter (weight, id) pairs into a dense ``[M, num_columns]`` tensor.

    Makes the comparison independent of the per-row slot order, so the test does
    not depend on how ties between equal scores are broken.
    """
    dense = torch.zeros((weights.shape[0], num_columns), dtype=torch.float32, device=weights.device)
    dense.scatter_(1, indices.long(), weights.float())
    return dense

def external__reference_gate(scores: torch.Tensor, bias: torch.Tensor, topk: int, scoring_func: str, num_fused_shared_experts: int, renormalize: bool, routed_scaling_factor: float, apply_routed_scaling_factor_on_output: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Definition-based eager reference matching the CUDA fused-gate semantics."""
    if scoring_func == 'sigmoid':
        activated = scores.sigmoid()
    else:
        activated = torch.nn.functional.softplus(scores).sqrt()
    biased = activated + bias.unsqueeze(0)
    num_experts = scores.size(1)
    num_routed = topk - num_fused_shared_experts
    bs = biased.size(0)
    work = biased.clone()
    arange = torch.arange(num_experts, device=scores.device).unsqueeze(0)
    routed_idx = torch.empty(bs, num_routed, dtype=torch.int32, device=scores.device)
    routed_wgt = torch.empty(bs, num_routed, dtype=torch.float32, device=scores.device)
    for k in range(num_routed):
        vals, _ = work.max(dim=1, keepdim=True)
        lane = torch.where(work == vals, arange, num_experts + 1)
        winner = lane.min(dim=1).values.to(torch.int32)
        routed_idx[:, k] = winner
        routed_wgt[:, k] = activated.gather(1, winner.long().unsqueeze(1)).squeeze(1)
        work.scatter_(1, winner.long().unsqueeze(1), float('-inf'))
    routed_sum = routed_wgt.sum(dim=1, keepdim=True)
    weights = torch.empty(bs, topk, dtype=torch.float32, device=scores.device)
    indices = torch.empty(bs, topk, dtype=torch.int32, device=scores.device)
    weights[:, :num_routed] = routed_wgt
    indices[:, :num_routed] = routed_idx
    if num_fused_shared_experts > 0:
        weights[:, num_routed:] = routed_sum / routed_scaling_factor
        for j in range(num_fused_shared_experts):
            indices[:, num_routed + j] = num_experts + j
    if renormalize:
        norm = torch.where(routed_sum > 0.0, routed_sum, torch.ones_like(routed_sum))
        weights = weights / norm
    if apply_routed_scaling_factor_on_output:
        weights = weights * routed_scaling_factor
    return (weights, indices)

def external__make_inputs(M: int, num_experts: int, seed: int):
    torch.manual_seed(seed)
    scores = torch.randn(M, num_experts, dtype=torch.float32, device=external_DEVICE) * 2.0
    bias = torch.randn(num_experts, dtype=torch.float32, device=external_DEVICE) * 0.5
    return (scores, bias)
external__REFERENCE_CASES = get_ci_test_range([(1, 128, 4, 'sigmoid', 0, True, True), (7, 256, 6, 'sqrtsoftplus', 1, False, False), (64, 384, 8, 'sigmoid', 0, False, True), (256, 384, 6, 'sqrtsoftplus', 0, True, False), (1024, 128, 4, 'sigmoid', 1, False, False), (1024, 512, 8, 'sqrtsoftplus', 1, True, True), (7, 512, 4, 'sigmoid', 1, True, False), (256, 256, 8, 'sqrtsoftplus', 0, False, True)], [(1, 128, 4, 'sigmoid', 0, True, True), (7, 256, 6, 'sqrtsoftplus', 1, False, False), (64, 384, 8, 'sigmoid', 0, False, True), (256, 384, 6, 'sqrtsoftplus', 0, True, False), (1024, 128, 4, 'sigmoid', 1, False, False), (1024, 512, 8, 'sqrtsoftplus', 1, True, True)])

@pytest.mark.parametrize('M,num_experts,topk,scoring_func,num_shared,renormalize,apply_scale', external__REFERENCE_CASES)
@pytest.mark.usefixtures('_require_sglang_integration')
def test_sglang_integration_moe_fused_gate_matches_reference(M: int, num_experts: int, topk: int, scoring_func: str, num_shared: int, renormalize: bool, apply_scale: bool) -> None:
    scores, bias = external__make_inputs(M, num_experts, seed=num_experts * 100 + topk)
    scale = 2.5
    kwargs = dict(topk=topk, scoring_func=scoring_func, num_fused_shared_experts=num_shared, renormalize=renormalize, routed_scaling_factor=scale, apply_routed_scaling_factor_on_output=apply_scale)
    triton_w, triton_i = external_gate(scores, bias, **kwargs)
    ref_w, ref_i = external__reference_gate(scores, bias, **kwargs)
    torch.cuda.synchronize()
    num_columns = num_experts + num_shared
    torch.testing.assert_close(external__scatter_by_expert(triton_w, triton_i, num_columns), external__scatter_by_expert(ref_w, ref_i, num_columns), rtol=0.0001, atol=1e-05)

@pytest.mark.parametrize('num_experts,topk', [(256, 6), (384, 8), (512, 8)])
@pytest.mark.parametrize('apply_scale', [True, False])
@pytest.mark.usefixtures('_require_sglang_integration')
def test_sglang_integration_moe_fused_gate_matches_production_impl(num_experts: int, topk: int, apply_scale: bool) -> None:
    """Match production ``biased_grouped_topk_impl`` on the path it replaces.

    The kernel supersedes the ungrouped sigmoid CUDA path in ``topk.py``; that
    reference hardcodes sigmoid and no fused shared expert, and the production
    path always renormalizes (the impl only applies the scaling factor when
    ``renormalize`` is set), so we compare on the renormalized path.
    """
    from sglang.srt.layers.moe.topk import biased_grouped_topk_impl
    M = 128
    scores, bias = external__make_inputs(M, num_experts, seed=7)
    scale = 2.5
    triton_w, triton_i = external_gate(scores, bias, topk=topk, scoring_func='sigmoid', renormalize=True, routed_scaling_factor=scale, apply_routed_scaling_factor_on_output=apply_scale)
    hidden_states = torch.empty((M, 1), dtype=torch.float32, device=external_DEVICE)
    ref_w, ref_i = biased_grouped_topk_impl(hidden_states, scores, bias, topk, True, num_expert_group=1, topk_group=1, routed_scaling_factor=scale, apply_routed_scaling_factor_on_output=apply_scale)
    torch.cuda.synchronize()
    torch.testing.assert_close(external__scatter_by_expert(triton_w, triton_i, num_experts), external__scatter_by_expert(ref_w, ref_i, num_experts), rtol=0.0001, atol=1e-05)

@pytest.mark.usefixtures('_require_sglang_integration')
def test_sglang_integration_moe_fused_gate_shapes_and_dtypes() -> None:
    """Output shapes/dtypes and renormalized weights for a DeepSeek-V4 config."""
    M, N, K = (64, 256, 8)
    scores, bias = external__make_inputs(M, N, seed=0)
    w, i = external_gate(scores, bias, topk=K, scoring_func='sqrtsoftplus')
    assert w.shape == (M, K)
    assert i.shape == (M, K)
    assert w.dtype == torch.float32
    assert i.dtype == torch.int32
    assert (i >= 0).all() and (i < N).all()
    torch.testing.assert_close(w.sum(dim=1), torch.ones(M, device=external_DEVICE), rtol=0.0001, atol=1e-05)

def external__reference_softmax(gating: torch.Tensor, topk: int, renormalize: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Plain-softmax topk reference (the AOT ``topk_softmax`` semantics)."""
    num_experts = gating.size(1)
    probs = torch.softmax(gating.float(), dim=-1)
    work = gating.float().clone()
    arange = torch.arange(num_experts, device=gating.device).unsqueeze(0)
    M = gating.size(0)
    idx = torch.empty(M, topk, dtype=torch.int32, device=gating.device)
    wgt = torch.empty(M, topk, dtype=torch.float32, device=gating.device)
    for k in range(topk):
        vals, _ = work.max(dim=1, keepdim=True)
        lane = torch.where(work == vals, arange, num_experts + 1)
        winner = lane.min(dim=1).values.to(torch.int32)
        idx[:, k] = winner
        wgt[:, k] = probs.gather(1, winner.long().unsqueeze(1)).squeeze(1)
        work.scatter_(1, winner.long().unsqueeze(1), float('-inf'))
    if renormalize:
        wgt = wgt / wgt.sum(dim=1, keepdim=True)
    return (wgt, idx)
external__SOFTMAX_AOT_CASES = get_ci_test_range([(1, 128, 4, True, torch.float32), (1, 256, 8, False, torch.float32), (200, 128, 4, False, torch.bfloat16), (200, 256, 8, True, torch.bfloat16), (1024, 512, 6, False, torch.float32), (1024, 512, 6, True, torch.bfloat16)], [(1, 128, 4, True, torch.float32), (200, 256, 8, False, torch.bfloat16), (1024, 512, 6, False, torch.float32)])

@pytest.mark.parametrize('M,num_experts,topk,renormalize,dtype', external__SOFTMAX_AOT_CASES)
@pytest.mark.usefixtures('_require_sglang_integration')
def test_sglang_integration_moe_fused_gate_softmax_matches_aot(M: int, num_experts: int, topk: int, renormalize: bool, dtype: torch.dtype) -> None:
    """Triton softmax path matches the AOT ``topk_softmax`` it replaces in fused_topk."""
    sgl_kernel = pytest.importorskip('sgl_kernel')
    torch.manual_seed(num_experts * 13 + topk)
    gating = torch.randn(M, num_experts, dtype=dtype, device=external_DEVICE) * 2.0
    zero_bias = torch.zeros(num_experts, dtype=torch.float32, device=external_DEVICE)
    tri_w, tri_i = external_gate(gating, zero_bias, topk=topk, scoring_func='softmax', renormalize=renormalize)
    ref_w, ref_i = external__reference_softmax(gating, topk, renormalize)
    aot_w = torch.empty(M, topk, dtype=torch.float32, device=external_DEVICE)
    aot_i = torch.empty(M, topk, dtype=torch.int32, device=external_DEVICE)
    sgl_kernel.topk_softmax(aot_w, aot_i, gating, renormalize)
    torch.cuda.synchronize()
    dense_tri = external__scatter_by_expert(tri_w, tri_i, num_experts)
    torch.testing.assert_close(dense_tri, external__scatter_by_expert(ref_w, ref_i, num_experts), rtol=0.001, atol=0.001)
    torch.testing.assert_close(dense_tri, external__scatter_by_expert(aot_w, aot_i, num_experts), rtol=0.001, atol=0.001)

@pytest.mark.parametrize('M', [1, 8, 32])
@pytest.mark.usefixtures('_require_sglang_integration')
def test_sglang_integration_moe_fused_gate_softmax_none_bias_matches_zero_bias(M: int) -> None:
    torch.manual_seed(M)
    scores = torch.randn(M, 256, dtype=torch.float32, device=external_DEVICE)
    zero_bias = torch.zeros(256, dtype=torch.float32, device=external_DEVICE)
    none_w, none_i = external_gate(scores, None, topk=8, scoring_func='softmax', renormalize=True)
    zero_w, zero_i = external_gate(scores, zero_bias, topk=8, scoring_func='softmax', renormalize=True)
    torch.testing.assert_close(none_w, zero_w, rtol=0, atol=0)
    torch.testing.assert_close(none_i, zero_i, rtol=0, atol=0)
external__SIGMOID_AOT_CASES = get_ci_test_range([(1, 128, 4, True, True, torch.float32), (1, 256, 8, False, True, torch.float32), (200, 128, 4, False, False, torch.bfloat16), (200, 256, 8, True, False, torch.bfloat16), (1024, 128, 4, True, False, torch.float32), (1024, 256, 8, False, True, torch.bfloat16)], [(1, 128, 4, True, True, torch.float32), (200, 128, 4, False, False, torch.bfloat16), (200, 256, 8, True, False, torch.bfloat16), (1024, 256, 8, False, True, torch.float32)])

@pytest.mark.parametrize('M,num_experts,topk,renormalize,with_bias,dtype', external__SIGMOID_AOT_CASES)
@pytest.mark.usefixtures('_require_sglang_integration')
def test_sglang_integration_moe_fused_gate_sigmoid_matches_aot(M: int, num_experts: int, topk: int, renormalize: bool, with_bias: bool, dtype: torch.dtype) -> None:
    """Triton sigmoid path matches the AOT ``topk_sigmoid`` it replaces in fused_topk."""
    sgl_kernel = pytest.importorskip('sgl_kernel')
    torch.manual_seed(num_experts * 17 + topk)
    gating = torch.randn(M, num_experts, dtype=dtype, device=external_DEVICE) * 2.0
    bias = torch.randn(num_experts, dtype=torch.float32, device=external_DEVICE) * 0.5 if with_bias else torch.zeros(num_experts, dtype=torch.float32, device=external_DEVICE)
    tri_w, tri_i = external_gate(gating, bias, topk=topk, scoring_func='sigmoid', renormalize=renormalize)
    aot_w = torch.empty(M, topk, dtype=torch.float32, device=external_DEVICE)
    aot_i = torch.empty(M, topk, dtype=torch.int32, device=external_DEVICE)
    sgl_kernel.topk_sigmoid(aot_w, aot_i, gating, renormalize, bias if with_bias else None)
    torch.cuda.synchronize()
    torch.testing.assert_close(external__scatter_by_expert(tri_w, tri_i, num_experts), external__scatter_by_expert(aot_w, aot_i, num_experts), rtol=0.001, atol=0.001)

@pytest.mark.parametrize('num_experts,num_expert_group,topk_group,topk', [(256, 8, 4, 8), (128, 8, 4, 6), (256, 4, 2, 8)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.usefixtures('_require_sglang_integration')
def test_sglang_integration_moe_fused_gate_grouped_matches_production_impl(num_experts: int, num_expert_group: int, topk_group: int, topk: int, dtype: torch.dtype) -> None:
    """Grouped Triton routing must match the definitional biased_grouped_topk_impl.

    The kernel adds DeepSeek-V3 grouped routing (per-group top-2-sum group scores,
    keep topk_group groups, then top-k within). biased_grouped_topk_impl is the
    eager reference the production grouped path is defined against.
    """
    from sglang.srt.layers.moe.topk import biased_grouped_topk_impl
    M = 256
    torch.manual_seed(num_experts * 7 + num_expert_group * 13 + topk)
    gating = torch.randn(M, num_experts, dtype=dtype, device=external_DEVICE) * 2.0
    bias = torch.randn(num_experts, dtype=torch.float32, device=external_DEVICE) * 0.5
    hidden = torch.randn(M, 16, dtype=dtype, device=external_DEVICE)
    tri_w, tri_i = external_gate(gating, bias, topk=topk, scoring_func='sigmoid', renormalize=True, num_expert_group=num_expert_group, topk_group=topk_group)
    ref_w, ref_i = biased_grouped_topk_impl(hidden, gating, bias, topk, True, num_expert_group=num_expert_group, topk_group=topk_group, num_fused_shared_experts=0, routed_scaling_factor=1.0, apply_routed_scaling_factor_on_output=False)
    torch.cuda.synchronize()
    torch.testing.assert_close(external__scatter_by_expert(tri_w, tri_i, num_experts), external__scatter_by_expert(ref_w, ref_i, num_experts), rtol=0.001, atol=0.001)

@pytest.mark.parametrize('num_experts,num_expert_group,topk_group,topk,num_fused_shared_experts', [(256, 8, 4, 8, 0), (256, 8, 4, 9, 1)])
@pytest.mark.usefixtures('_require_sglang_integration')
def test_sglang_integration_grouped_dispatch_flag_matches_default(num_experts: int, num_expert_group: int, topk_group: int, topk: int, num_fused_shared_experts: int) -> None:
    """The opt-in SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK dispatch must match the
    default grouped path (flashinfer/AOT) that biased_grouped_topk_gpu selects when
    the flag is off. This covers the wiring, not just the raw kernel — validated
    bit-exact on DeepSeek-V3.2 e2e; here we assert parity against the default path.
    """
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.topk import biased_grouped_topk_gpu
    M = 256
    torch.manual_seed(num_experts * 3 + num_expert_group * 5 + topk)
    gating = torch.randn(M, num_experts, dtype=torch.float32, device=external_DEVICE) * 2.0
    bias = torch.randn(num_experts, dtype=torch.float32, device=external_DEVICE) * 0.5
    hidden = torch.randn(M, 16, dtype=torch.float32, device=external_DEVICE)
    kwargs = dict(num_expert_group=num_expert_group, topk_group=topk_group, num_fused_shared_experts=num_fused_shared_experts, routed_scaling_factor=2.5, apply_routed_scaling_factor_on_output=False)
    with envs.SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK.override(False):
        def_w, def_i = biased_grouped_topk_gpu(hidden, gating, bias, topk, True, **kwargs)
    with envs.SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK.override(True):
        jit_w, jit_i = biased_grouped_topk_gpu(hidden, gating, bias, topk, True, **kwargs)
    torch.cuda.synchronize()
    topk_routed = topk - num_fused_shared_experts
    torch.testing.assert_close(external__scatter_by_expert(def_w[:, :topk_routed], def_i[:, :topk_routed], num_experts), external__scatter_by_expert(jit_w[:, :topk_routed], jit_i[:, :topk_routed], num_experts), rtol=0.001, atol=0.001)

@pytest.mark.parametrize('bias_dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('scoring_func', ['sigmoid', 'sqrtsoftplus'])
@pytest.mark.usefixtures('_require_sglang_integration')
def test_sglang_integration_moe_fused_gate_accepts_non_fp32_bias(bias_dtype: torch.dtype, scoring_func: str) -> None:
    """A non-fp32 correction bias must route identically to its fp32 copy.

    DeepSeek-V4 stores ``e_score_correction_bias`` in bf16 and topk.py now passes
    it straight through (no host-side fp32 copy per MoE invocation). The kernel
    upcasts the bias to fp32 on load, so a bf16/fp16 bias must be bit-identical to
    casting it to fp32 first -- this guards both the relaxed dtype assertion and
    the removed per-call cast. (top-6 exercises the DSV4 non-pow2 routing width.)
    """
    M, num_experts, topk = (256, 128, 6)
    torch.manual_seed(num_experts * 11 + topk)
    gating = torch.randn(M, num_experts, dtype=torch.float32, device=external_DEVICE) * 2.0
    bias_lowp = torch.randn(num_experts, dtype=bias_dtype, device=external_DEVICE) * 0.5
    bias_fp32 = bias_lowp.to(torch.float32)
    lowp_w, lowp_i = external_gate(gating, bias_lowp, topk=topk, scoring_func=scoring_func, renormalize=True)
    fp32_w, fp32_i = external_gate(gating, bias_fp32, topk=topk, scoring_func=scoring_func, renormalize=True)
    torch.cuda.synchronize()
    torch.testing.assert_close(external__scatter_by_expert(lowp_w, lowp_i, num_experts), external__scatter_by_expert(fp32_w, fp32_i, num_experts), rtol=0, atol=0)


class _ValidationRecorder:
    def __init__(self):
        self.cases = []

    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or report.failed or report.skipped:
            self.cases.append(dict(nodeid=report.nodeid, phase=report.when,
                outcome=report.outcome, duration_s=report.duration,
                details=str(report.longrepr) if report.failed or report.skipped else ''))


def main():
    recorder = _ValidationRecorder()
    started = time.time()
    arguments = [str(Path(__file__).resolve())] + sys.argv[1:]
    status = pytest.main(arguments, plugins=[recorder])
    destination = os.environ.get('MOE_ROUTER_VALIDATION_REPORT')
    if destination:
        report = dict(module=TARGET_MODULE,
            module_path=str(Path(_target.__file__).resolve()) if _target else None,
            case_set=os.environ.get('MOE_ROUTER_CASE_SET', 'full'),
            exit_code=int(status), elapsed_s=time.time()-started, cases=recorder.cases)
        with Path(destination).open('x') as stream:
            json.dump(report, stream, indent=2)
    return int(status)


if __name__ == '__main__':
    raise SystemExit(main())
