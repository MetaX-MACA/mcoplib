"""Correctness and performance coverage for the production KDA verify shapes.

Pytest runs correctness only.  Run the same-process strategy benchmark with::

    CUDA_VISIBLE_DEVICES=8 python \
      unit_test/test_fused_sigmoid_gating_delta_rule_update_target_shapes.py \
      --benchmark

The benchmark intentionally times the allocating public-wrapper contract: the
output, and the optimized strategy's private gate buffers, are allocated inside
the timed region.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import pytest
import torch

from mcoplib.triton_fused_sigmoid_gating_delta_rule_update import (
    _fused_sigmoid_gating_delta_rule_update_with_strategy,
    fused_sigmoid_gating_delta_rule_update,
)


HEADS = 64
K_DIM = 128
V_DIM = 128
CACHE_SLOTS = 7
CACHE_STEPS = 6
LOWER_BOUND = -5.0
CACHE_SENTINEL = 1234.5


@dataclass(frozen=True)
class ShapeCase:
    name: str
    total_tokens: int
    cu_seqlens: tuple[int, ...]
    h0_indices: tuple[int, ...]
    intermediate_state_indices: tuple[int, ...]
    qk_heads: int = HEADS
    value_heads: int = HEADS
    key_dim: int = K_DIM
    value_dim: int = V_DIM
    cache_slots: int = CACHE_SLOTS


CASES = (
    ShapeCase(
        name="t12_n2",
        total_tokens=12,
        cu_seqlens=(0, 6, 12),
        h0_indices=(0, 1),
        intermediate_state_indices=(2, 3),
    ),
    ShapeCase(
        name="t6_n1",
        total_tokens=6,
        cu_seqlens=(0, 6),
        h0_indices=(0,),
        # The captured tensor has shape [2], although N == 1.  Only element 0
        # is consumed; retaining the second value reproduces that contract.
        intermediate_state_indices=(2, -1),
    ),
)

MULTI_SEQUENCE_CASES = (
    ShapeCase(
        name="t24_n4",
        total_tokens=24,
        cu_seqlens=(0, 6, 12, 18, 24),
        h0_indices=(0, 1, 2, 3),
        intermediate_state_indices=(3, 4, 5, 6),
    ),
)

DYNAMIC_CASES = (
    ShapeCase(
        name="t10_n2_l4_l6",
        total_tokens=10,
        cu_seqlens=(0, 4, 10),
        h0_indices=(0, 1),
        intermediate_state_indices=(2, 3),
    ),
    ShapeCase(
        name="t8_n2_l3_l5",
        total_tokens=8,
        cu_seqlens=(0, 3, 8),
        h0_indices=(0, 1),
        intermediate_state_indices=(2, 3),
    ),
    ShapeCase(
        name="t5_n1",
        total_tokens=5,
        cu_seqlens=(0, 5),
        h0_indices=(0,),
        intermediate_state_indices=(2, -1),
    ),
)

GQA_CASES = (
    ShapeCase(
        name="t6_n1_h1_hv64",
        total_tokens=6,
        cu_seqlens=(0, 6),
        h0_indices=(0,),
        intermediate_state_indices=(2, -1),
        qk_heads=1,
        value_heads=64,
    ),
    ShapeCase(
        name="t6_n1_h8_hv64",
        total_tokens=6,
        cu_seqlens=(0, 6),
        h0_indices=(0,),
        intermediate_state_indices=(2, -1),
        qk_heads=8,
        value_heads=64,
    ),
    ShapeCase(
        name="t10_n2_l4_l6_h8_hv64",
        total_tokens=10,
        cu_seqlens=(0, 4, 10),
        h0_indices=(0, 1),
        intermediate_state_indices=(2, 3),
        qk_heads=8,
        value_heads=64,
    ),
    ShapeCase(
        name="t12_n2_h8_hv64",
        total_tokens=12,
        cu_seqlens=(0, 6, 12),
        h0_indices=(0, 1),
        intermediate_state_indices=(2, 3),
        qk_heads=8,
        value_heads=64,
    ),
    ShapeCase(
        name="t6_n1_h32_hv64",
        total_tokens=6,
        cu_seqlens=(0, 6),
        h0_indices=(0,),
        intermediate_state_indices=(2, -1),
        qk_heads=32,
        value_heads=64,
    ),
)

DIMENSION_CASES = tuple(
    ShapeCase(
        name=f"t3_n1_h8_hv8_k{key_dim}_v{value_dim}",
        total_tokens=3,
        cu_seqlens=(0, 3),
        h0_indices=(0,),
        intermediate_state_indices=(2, -1),
        qk_heads=8,
        value_heads=8,
        key_dim=key_dim,
        value_dim=value_dim,
    )
    for key_dim in (64, 128, 256)
    for value_dim in (64, 128, 256)
)

AUTO_DIMENSION_CASES = (
    ShapeCase(
        "auto_k64_v64_t6_n1",
        6,
        (0, 6),
        (0,),
        (2, -1),
        key_dim=64,
        value_dim=64,
        cache_slots=4,
    ),
    ShapeCase(
        "fallback_k64_v64_t5_n1",
        5,
        (0, 5),
        (0,),
        (2, -1),
        key_dim=64,
        value_dim=64,
        cache_slots=4,
    ),
    ShapeCase(
        "auto_k64_v128_t5_n2",
        5,
        (0, 2, 5),
        (0, 1),
        (2, 3),
        key_dim=64,
        value_dim=128,
        cache_slots=4,
    ),
    ShapeCase(
        "fallback_k64_v128_t4_n2",
        4,
        (0, 2, 4),
        (0, 1),
        (2, 3),
        key_dim=64,
        value_dim=128,
        cache_slots=4,
    ),
    ShapeCase(
        "auto_k64_v256_t3_n1",
        3,
        (0, 3),
        (0,),
        (2, -1),
        key_dim=64,
        value_dim=256,
        cache_slots=4,
    ),
    ShapeCase(
        "auto_k128_v64_t3_n2",
        3,
        (0, 1, 3),
        (0, 1),
        (2, 3),
        value_dim=64,
        cache_slots=4,
    ),
    ShapeCase(
        "fallback_k128_v64_t2_n1",
        2,
        (0, 2),
        (0,),
        (2, -1),
        value_dim=64,
        cache_slots=4,
    ),
    ShapeCase(
        "auto_k128_v256_t5_n1",
        5,
        (0, 5),
        (0,),
        (2, -1),
        value_dim=256,
        cache_slots=4,
    ),
    ShapeCase(
        "auto_k256_v256_t1_n1",
        1,
        (0, 1),
        (0,),
        (2, -1),
        key_dim=256,
        value_dim=256,
        cache_slots=4,
    ),
)


@dataclass
class Inputs:
    spec: ShapeCase
    A_log: torch.Tensor
    a: torch.Tensor
    dt_bias: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    b: torch.Tensor
    h0_source: torch.Tensor
    h0_indices: torch.Tensor
    cu_seqlens: torch.Tensor
    intermediate_states_buffer: torch.Tensor
    intermediate_state_indices: torch.Tensor


def make_inputs(
    spec: ShapeCase,
    device: torch.device,
    seed: int = 20260914,
    *,
    packed_qkv: bool = False,
) -> Inputs:
    generator = torch.Generator(device=device).manual_seed(seed + spec.total_tokens)

    def randn(*shape: int, dtype: torch.dtype, scale: float = 1.0) -> torch.Tensor:
        value = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
        return (value * scale).to(dtype)

    total_tokens = spec.total_tokens
    H, HV = spec.qk_heads, spec.value_heads
    K, V = spec.key_dim, spec.value_dim
    assert HV % H == 0
    if packed_qkv:
        assert H == HV and K == V
        qkv = randn(1, total_tokens, 3, H, K, dtype=torch.bfloat16)
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]
        v.mul_(0.25)
    else:
        q = randn(1, total_tokens, H, K, dtype=torch.bfloat16)
        k = randn(1, total_tokens, H, K, dtype=torch.bfloat16)
        v = randn(
            1, total_tokens, HV, V, dtype=torch.bfloat16, scale=0.25
        )
    return Inputs(
        spec=spec,
        A_log=randn(1, 1, HV, 1, dtype=torch.float32, scale=0.1),
        a=randn(1, total_tokens, HV * K, dtype=torch.bfloat16, scale=0.5),
        dt_bias=randn(HV * K, dtype=torch.float32, scale=0.1),
        q=q,
        k=k,
        v=v,
        b=randn(1, total_tokens, HV, dtype=torch.bfloat16, scale=0.5),
        h0_source=randn(
            spec.cache_slots, HV, V, K, dtype=torch.float32, scale=0.01
        ),
        h0_indices=torch.tensor(spec.h0_indices, device=device, dtype=torch.int32),
        cu_seqlens=torch.tensor(spec.cu_seqlens, device=device, dtype=torch.int32),
        intermediate_states_buffer=torch.full(
            (spec.cache_slots, CACHE_STEPS, HV, V, K),
            CACHE_SENTINEL,
            device=device,
            dtype=torch.float32,
        ),
        intermediate_state_indices=torch.tensor(
            spec.intermediate_state_indices, device=device, dtype=torch.int32
        ),
    )


def call(inputs: Inputs, strategy: str) -> torch.Tensor:
    kwargs = dict(
        A_log=inputs.A_log,
        a=inputs.a,
        dt_bias=inputs.dt_bias,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        q=inputs.q,
        k=inputs.k,
        v=inputs.v,
        b=inputs.b,
        initial_state_source=inputs.h0_source,
        initial_state_indices=inputs.h0_indices,
        scale=inputs.spec.key_dim**-0.5,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=inputs.cu_seqlens,
        is_kda=True,
        lower_bound=LOWER_BOUND,
        disable_state_update=True,
        intermediate_states_buffer=inputs.intermediate_states_buffer,
        intermediate_state_indices=inputs.intermediate_state_indices,
        cache_steps=CACHE_STEPS,
        retrieve_parent_token=None,
        cache_ring=False,
    )
    if strategy == "auto":
        return fused_sigmoid_gating_delta_rule_update(**kwargs)
    return _fused_sigmoid_gating_delta_rule_update_with_strategy(
        **kwargs, strategy=strategy
    )


def reference(inputs: Inputs) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    H, HV = inputs.spec.qk_heads, inputs.spec.value_heads
    K, V = inputs.spec.key_dim, inputs.spec.value_dim
    q = inputs.q[0].float()
    k = inputs.k[0].float()
    v = inputs.v[0].float()
    a = inputs.a.reshape(inputs.spec.total_tokens, HV, K).float()
    beta_raw = inputs.b[0].float()
    A = torch.exp(inputs.A_log.reshape(HV).float())
    dt_bias = inputs.dt_bias.reshape(HV, K).float()
    q = q * torch.rsqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)
    qk_head_indices = torch.arange(HV, device=q.device) // (HV // H)
    q = q[:, qk_head_indices]
    k = k[:, qk_head_indices]

    output = torch.empty(
        (1, inputs.spec.total_tokens, HV, V),
        device=q.device,
        dtype=torch.float32,
    )
    expected_cache: dict[int, torch.Tensor] = {}
    for sequence, (bos, eos) in enumerate(
        zip(inputs.spec.cu_seqlens[:-1], inputs.spec.cu_seqlens[1:])
    ):
        state_slot = inputs.spec.h0_indices[sequence]
        cache_slot = inputs.spec.intermediate_state_indices[sequence]
        state = inputs.h0_source[state_slot].clone()
        states = []
        for token in range(bos, eos):
            gate = LOWER_BOUND * torch.sigmoid(
                A[:, None] * (a[token] + dt_bias)
            )
            state = state * torch.exp(gate[:, None, :])
            delta = v[token] - torch.sum(
                state * k[token, :, None, :], dim=-1
            )
            delta = delta * torch.sigmoid(beta_raw[token])[:, None]
            state = state + delta[:, :, None] * k[token, :, None, :]
            output[0, token] = torch.sum(
                state * (q[token] * (K**-0.5))[:, None, :], dim=-1
            )
            states.append(state.clone())
        expected_cache[cache_slot] = torch.stack(states)
    return output, expected_cache


def assert_correct(inputs: Inputs, strategy: str) -> None:
    h0_before = inputs.h0_source.clone()
    cache_before = inputs.intermediate_states_buffer.clone()
    expected_output, expected_cache = reference(inputs)

    actual_output = call(inputs, strategy)
    torch.cuda.synchronize(inputs.q.device)

    assert actual_output.shape == (
        1,
        inputs.spec.total_tokens,
        inputs.spec.value_heads,
        inputs.spec.value_dim,
    )
    torch.testing.assert_close(
        actual_output.float(), expected_output, atol=2e-2, rtol=1e-2
    )
    # DISABLE_STATE_UPDATE=True is an exact mutation contract.
    torch.testing.assert_close(inputs.h0_source, h0_before, atol=0.0, rtol=0.0)

    selected_slots = set(expected_cache)
    for slot, expected in expected_cache.items():
        written_steps = expected.shape[0]
        torch.testing.assert_close(
            inputs.intermediate_states_buffer[slot, :written_steps],
            expected,
            atol=3e-3,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            inputs.intermediate_states_buffer[slot, written_steps:],
            cache_before[slot, written_steps:],
            atol=0.0,
            rtol=0.0,
        )
    untouched = [
        slot for slot in range(inputs.spec.cache_slots) if slot not in selected_slots
    ]
    torch.testing.assert_close(
        inputs.intermediate_states_buffer[untouched],
        cache_before[untouched],
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a MACA/CUDA GPU")
@pytest.mark.parametrize("spec", CASES, ids=lambda spec: spec.name)
@pytest.mark.parametrize("strategy", ("original", "auto"))
def test_target_verify_shapes(spec: ShapeCase, strategy: str) -> None:
    inputs = make_inputs(spec, torch.device("cuda:0"))
    assert_correct(inputs, strategy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a MACA/CUDA GPU")
@pytest.mark.parametrize("spec", CASES, ids=lambda spec: spec.name)
def test_target_verify_packed_qkv_strides(spec: ShapeCase) -> None:
    inputs = make_inputs(spec, torch.device("cuda:0"), packed_qkv=True)
    expected_token_stride = 3 * spec.qk_heads * spec.key_dim
    assert inputs.q.stride() == (
        spec.total_tokens * expected_token_stride,
        expected_token_stride,
        spec.key_dim,
        1,
    )
    assert inputs.k.stride() == inputs.q.stride()
    assert inputs.v.stride() == inputs.q.stride()
    assert not inputs.q.is_contiguous()
    assert_correct(inputs, "auto")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a MACA/CUDA GPU")
@pytest.mark.parametrize("spec", MULTI_SEQUENCE_CASES, ids=lambda spec: spec.name)
def test_multi_sequence_auto_dispatch(spec: ShapeCase) -> None:
    inputs = make_inputs(spec, torch.device("cuda:0"))
    assert_correct(inputs, "auto")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a MACA/CUDA GPU")
@pytest.mark.parametrize("spec", DYNAMIC_CASES, ids=lambda spec: spec.name)
def test_dynamic_norm_generalization(spec: ShapeCase) -> None:
    inputs = make_inputs(spec, torch.device("cuda:0"))
    assert_correct(inputs, "auto")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a MACA/CUDA GPU")
@pytest.mark.parametrize(
    "spec",
    (
        ShapeCase("t1_n1_fallback", 1, (0, 1), (0,), (2, -1)),
        ShapeCase("t2_n2_fallback", 2, (0, 1, 2), (0, 1), (2, 3)),
    ),
    ids=lambda spec: spec.name,
)
def test_dynamic_norm_short_fallback(spec: ShapeCase) -> None:
    inputs = make_inputs(spec, torch.device("cuda:0"))
    assert_correct(inputs, "auto")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a MACA/CUDA GPU")
@pytest.mark.parametrize("spec", GQA_CASES, ids=lambda spec: spec.name)
@pytest.mark.parametrize("strategy", ("original", "auto"))
def test_gqa_norm_generalization(spec: ShapeCase, strategy: str) -> None:
    inputs = make_inputs(spec, torch.device("cuda:0"))
    assert_correct(inputs, strategy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a MACA/CUDA GPU")
@pytest.mark.parametrize("spec", AUTO_DIMENSION_CASES, ids=lambda spec: spec.name)
def test_auto_dimension_dispatch(spec: ShapeCase) -> None:
    inputs = make_inputs(spec, torch.device("cuda:0"))
    assert_correct(inputs, "auto")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a MACA/CUDA GPU")
@pytest.mark.parametrize("spec", DIMENSION_CASES, ids=lambda spec: spec.name)
def test_dimension_norm_generalization(spec: ShapeCase) -> None:
    inputs = make_inputs(spec, torch.device("cuda:0"))
    assert_correct(inputs, "dual_opt_norms_dims")


def benchmark_strategy(
    inputs: Inputs,
    strategies: tuple[str, ...],
    *,
    warmup: int,
    repeats: int,
    batches: int,
) -> dict[str, list[float]]:
    for strategy in strategies:
        for _ in range(warmup):
            output = call(inputs, strategy)
        assert output is not None
    torch.cuda.synchronize(inputs.q.device)

    samples = {strategy: [] for strategy in strategies}
    for batch in range(batches):
        order = strategies if batch % 2 == 0 else tuple(reversed(strategies))
        for strategy in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                output = call(inputs, strategy)
            end.record()
            end.synchronize()
            assert output is not None
            samples[strategy].append(start.elapsed_time(end) * 1000.0 / repeats)
    return samples


def run_benchmark(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    strategies = tuple(item.strip() for item in args.strategies.split(",") if item.strip())
    weighted_medians = {strategy: 0.0 for strategy in strategies}
    for spec in CASES:
        # Reproduce views from the captured [B, T, 3, H, D] QKV payload.
        inputs = make_inputs(spec, device, packed_qkv=True)
        # Correctness is a hard gate before timing every strategy.
        for strategy in strategies:
            inputs.intermediate_states_buffer.fill_(CACHE_SENTINEL)
            assert_correct(inputs, strategy)
        samples = benchmark_strategy(
            inputs,
            strategies,
            warmup=args.warmup,
            repeats=args.repeats,
            batches=args.batches,
        )
        baseline = statistics.median(samples["original"])
        print(f"\n{spec.name}: T={spec.total_tokens}, N={len(spec.cu_seqlens) - 1}")
        for strategy in strategies:
            values = samples[strategy]
            median = statistics.median(values)
            weighted_medians[strategy] += 0.5 * median
            print(
                f"  {strategy:>12}: samples_us={[round(v, 3) for v in values]} "
                f"median={median:.3f} range={min(values):.3f}-{max(values):.3f} "
                f"speedup={baseline / median:.3f}x"
            )

    baseline = weighted_medians["original"]
    print("\n50/50 weighted median latency:")
    for strategy in strategies:
        latency = weighted_medians[strategy]
        print(f"  {strategy:>12}: {latency:.3f} us, speedup={baseline / latency:.3f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--strategies", default="original,single_opt,dual_opt,auto"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--batches", type=int, default=11)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if not arguments.benchmark:
        raise SystemExit("pass --benchmark to run the target-shape A/B benchmark")
    run_benchmark(arguments)
