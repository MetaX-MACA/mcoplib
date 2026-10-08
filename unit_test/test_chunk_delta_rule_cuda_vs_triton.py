# SPDX-License-Identifier: Apache-2.0
"""CUDA twin vs Triton on the shapes the Triton unit test already uses.

Reuses ``test_triton_sglang_chunk_delta_h`` for its case list, its input
builder and its cosine criterion, so the shapes, the data distribution and the
acceptance threshold are the production ones rather than a convenient subset.

    source env_local.sh
    CUDA_VISIBLE_DEVICES=<physical-id> /opt/conda/bin/python \
        unit_test/test_chunk_delta_rule_cuda_vs_triton.py [--full] [--json-out FILE]

``--full`` adds the extended case list (H=12/24, packed logical batch 1..256,
T up to 16384), which exercises partial BT tails on every shape.
"""

import argparse
import json
import math
import os
import pathlib
import sys
from typing import Dict

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import test_triton_sglang_chunk_delta_h as tri  # noqa: E402
import mcoplib.op as cuda_ext  # noqa: E402


CUDA_INTERFACE_AVAILABLE = torch.cuda.is_available() and hasattr(
    cuda_ext, "chunk_gated_delta_rule_fwd_h"
)


@torch.no_grad()
def compare_case(case: tri.Case, seed: int = 20260908) -> Dict[str, object]:
    inputs = tri._make_inputs(case, seed=seed)
    state_t = inputs["initial_state"].clone()
    state_c = inputs["initial_state"].clone()

    common = dict(
        k=inputs["k"],
        w=inputs["w"],
        u=inputs["u"],
        gk=inputs["gk"],
        initial_state_indices=inputs["initial_state_indices"],
        cu_seqlens=inputs["cu_seqlens"],
    )
    h_t, vn_t = tri.chunk_gated_delta_rule_fwd_h(initial_state=state_t, **common)
    h_c, vn_c = cuda_ext.chunk_gated_delta_rule_fwd_h(
        initial_state=state_c, **common
    )
    torch.cuda.synchronize()

    if h_t.shape != h_c.shape:
        raise AssertionError(f"h shape {tuple(h_t.shape)} vs {tuple(h_c.shape)}")
    if vn_t.shape != vn_c.shape:
        raise AssertionError(f"v_new shape {tuple(vn_t.shape)} vs {tuple(vn_c.shape)}")

    metrics: Dict[str, object] = {"label": case.label, "cu_values": list(tri._cu_values(case))}
    passed = True
    for name, a, b in (("h", h_t, h_c), ("v_new", vn_t, vn_c),
                       ("state", state_t, state_c)):
        cosine = tri._cosine(a, b)
        max_abs = float((a.float() - b.float()).abs().max())
        bitwise = bool(torch.equal(a, b))
        metrics[f"{name}_cosine"] = cosine
        metrics[f"{name}_max_abs"] = max_abs
        metrics[f"{name}_bitwise"] = bitwise
        if not (cosine >= tri.COSINE_THRESHOLD and torch.isfinite(b).all().item()):
            passed = False
    metrics["passed"] = passed
    del inputs, state_t, state_c, h_t, vn_t, h_c, vn_c
    torch.cuda.empty_cache()
    return metrics


def test_cuda_extension_exports_chunk_gated_delta_rule() -> None:
    """The CUDA implementation is a direct mcoplib.op binding."""

    assert hasattr(cuda_ext, "chunk_gated_delta_rule_fwd_h")
    assert hasattr(cuda_ext, "chunk_gated_delta_rule_fwd_h_native_out")


@pytest.mark.skipif(
    not CUDA_INTERFACE_AVAILABLE,
    reason="rebuilt mcoplib CUDA extension and a CUDA/MACA device are required",
)
@pytest.mark.parametrize(
    "case",
    [
        tri.Case(34, 1, 8, (0, 34), torch.float32),
        tri.Case(135, 2, 8, (0, 71, 135), torch.float32),
        tri.Case(135, 2, 8, (0, 71, 135), torch.bfloat16),
    ],
    ids=(
        "single-ragged-tail",
        "unequal-ragged-sequences-fp32-state",
        "unequal-ragged-sequences-bf16-state",
    ),
)
def test_cuda_interface_matches_triton(case: tri.Case) -> None:
    metrics = compare_case(case)
    assert metrics["passed"], metrics


@pytest.mark.skipif(
    not CUDA_INTERFACE_AVAILABLE,
    reason="rebuilt mcoplib CUDA extension and a CUDA/MACA device are required",
)
def test_cuda_interface_save_new_value_false() -> None:
    case = tri.Case(34, 1, 8, (0, 34), torch.float32)
    inputs = tri._make_inputs(case)
    state_t = inputs["initial_state"].clone()
    state_c = inputs["initial_state"].clone()
    common = dict(
        k=inputs["k"],
        w=inputs["w"],
        u=inputs["u"],
        gk=inputs["gk"],
        initial_state_indices=inputs["initial_state_indices"],
        cu_seqlens=inputs["cu_seqlens"],
        save_new_value=False,
    )
    h_t, v_t = tri.chunk_gated_delta_rule_fwd_h(
        initial_state=state_t, **common
    )
    h_c, v_c = cuda_ext.chunk_gated_delta_rule_fwd_h(
        initial_state=state_c, **common
    )
    assert v_t is None and v_c is None
    assert tri._cosine(h_t, h_c) >= tri.COSINE_THRESHOLD
    assert tri._cosine(state_t, state_c) >= tri.COSINE_THRESHOLD

    # Exercise the actual no-output native instance. Passing None means there
    # is no v_new allocation for the kernel to address or write.
    total_chunks = math.ceil(case.total_tokens / tri.CHUNK_SIZE)
    offsets = torch.tensor(
        [0, total_chunks], device="cuda", dtype=torch.int32
    )
    h_native = torch.empty(
        1, total_chunks, case.heads, tri.V, tri.K,
        device="cuda", dtype=tri.DTYPE,
    )
    state_native = inputs["initial_state"].clone()
    cuda_ext.chunk_gated_delta_rule_fwd_h_native_out(
        inputs["k"], inputs["w"], inputs["u"], inputs["gk"],
        state_native, inputs["initial_state_indices"],
        inputs["cu_seqlens"], offsets, h_native, None,
        case.logical_batch, total_chunks, False, True, False,
    )
    torch.cuda.synchronize()
    assert tri._cosine(h_t, h_native) >= tri.COSINE_THRESHOLD
    assert tri._cosine(state_t, state_native) >= tri.COSINE_THRESHOLD


def _chunk_local_g(
    total_tokens: int, heads: int, cu_values: tuple[int, ...]
) -> torch.Tensor:
    """Match the chunk-local cumulative scalar-g contract used by Triton."""

    increments = -0.0125 * torch.rand(
        total_tokens, heads, device="cuda", dtype=torch.float32
    )
    g = torch.empty_like(increments)
    for bos, eos in zip(cu_values[:-1], cu_values[1:]):
        for start in range(bos, eos, tri.CHUNK_SIZE):
            end = min(start + tri.CHUNK_SIZE, eos)
            g[start:end] = increments[start:end].cumsum(0)
    return g.unsqueeze(0)


@pytest.mark.skipif(
    not CUDA_INTERFACE_AVAILABLE,
    reason="rebuilt mcoplib CUDA extension and a CUDA/MACA device are required",
)
@pytest.mark.parametrize(
    "gate_mode,save_new_value",
    [
        pytest.param("g-only", True, id="g-only"),
        pytest.param("gk-only", True, id="gk-only"),
        pytest.param("g-and-gk", True, id="g-and-gk"),
        pytest.param("g-only", False, id="g-only-no-v-new"),
    ],
)
def test_cuda_interface_scalar_gate_modes(
    gate_mode: str, save_new_value: bool
) -> None:
    """CUDA matches Triton for every supported scalar/channel gate pairing."""

    case = tri.Case(135, 2, 8, (0, 71, 135), torch.float32)
    inputs = tri._make_inputs(case, seed=20260924)
    scalar_g = _chunk_local_g(
        case.total_tokens, case.heads, tri._cu_values(case)
    )
    state_t = inputs["initial_state"].clone()
    state_c = inputs["initial_state"].clone()
    common = dict(
        k=inputs["k"],
        w=inputs["w"],
        u=inputs["u"],
        g=scalar_g if gate_mode != "gk-only" else None,
        gk=inputs["gk"] if gate_mode != "g-only" else None,
        initial_state_indices=inputs["initial_state_indices"],
        cu_seqlens=inputs["cu_seqlens"],
        save_new_value=save_new_value,
    )
    h_t, v_t = tri.chunk_gated_delta_rule_fwd_h(
        initial_state=state_t, **common
    )
    h_c, v_c = cuda_ext.chunk_gated_delta_rule_fwd_h(
        initial_state=state_c, **common
    )
    torch.cuda.synchronize()
    assert tri._cosine(h_t, h_c) >= tri.COSINE_THRESHOLD
    if save_new_value:
        assert tri._cosine(v_t, v_c) >= tri.COSINE_THRESHOLD
    else:
        assert v_t is None and v_c is None
    assert tri._cosine(state_t, state_c) >= tri.COSINE_THRESHOLD


@pytest.mark.skipif(
    not CUDA_INTERFACE_AVAILABLE,
    reason="rebuilt mcoplib CUDA extension and a CUDA/MACA device are required",
)
@pytest.mark.parametrize("save_new_value", (True, False), ids=("save", "no-v-new"))
def test_cuda_interface_without_gate(save_new_value: bool) -> None:
    """USE_G=false and USE_GK=false follows Triton's ungated recurrence."""

    case = tri.Case(34, 1, 8, (0, 34), torch.float32)
    inputs = tri._make_inputs(case, seed=20260924)
    state_t = inputs["initial_state"].clone()
    state_c = inputs["initial_state"].clone()
    common = dict(
        k=inputs["k"], w=inputs["w"], u=inputs["u"], g=None, gk=None,
        initial_state_indices=inputs["initial_state_indices"],
        cu_seqlens=inputs["cu_seqlens"], save_new_value=save_new_value,
    )
    h_t, v_t = tri.chunk_gated_delta_rule_fwd_h(
        initial_state=state_t, **common
    )
    h_c, v_c = cuda_ext.chunk_gated_delta_rule_fwd_h(
        initial_state=state_c, **common
    )
    torch.cuda.synchronize()
    assert tri._cosine(h_t, h_c) >= tri.COSINE_THRESHOLD
    if save_new_value:
        assert tri._cosine(v_t, v_c) >= tri.COSINE_THRESHOLD
    else:
        assert v_t is None and v_c is None
    assert tri._cosine(state_t, state_c) >= tri.COSINE_THRESHOLD


@pytest.mark.skipif(
    not CUDA_INTERFACE_AVAILABLE,
    reason="rebuilt mcoplib CUDA extension and a CUDA/MACA device are required",
)
def test_cuda_interface_rejects_float64() -> None:
    """float64 is deliberately outside the CUDA interface contract."""

    case = tri.Case(34, 1, 8, (0, 34), torch.float32)
    inputs = tri._make_inputs(case, seed=20260924)
    with pytest.raises(RuntimeError, match="float32, float16, or bfloat16"):
        cuda_ext.chunk_gated_delta_rule_fwd_h(
            k=inputs["k"], w=inputs["w"], u=inputs["u"],
            gk=inputs["gk"].to(torch.float64),
            initial_state=inputs["initial_state"].clone(),
            initial_state_indices=inputs["initial_state_indices"],
            cu_seqlens=inputs["cu_seqlens"],
        )


@pytest.mark.skipif(
    not CUDA_INTERFACE_AVAILABLE,
    reason="rebuilt mcoplib CUDA extension and a CUDA/MACA device are required",
)
@pytest.mark.parametrize(
    "gate_mode,state_dtype",
    [
        pytest.param("gk-only", torch.float32,
                     id="fp16-input-fp32-state"),
        pytest.param("g-and-gk", torch.float16,
                     id="fp16-input-state-with-scalar-g"),
        pytest.param("gk-only", torch.bfloat16,
                     id="fp16-input-bf16-state"),
    ],
)
def test_cuda_interface_fp16_dtype_matrix(
    gate_mode: str, state_dtype: torch.dtype
) -> None:
    """The generic CUDA path accepts Triton's practical fp16 dtype family.

    MetaX Triton's tl.exp/tl.exp2 accepts fp32/fp64 only, so its effective
    contract keeps g/gk in fp32 even when compute tensors and state use fp16.
    """

    case = tri.Case(34, 1, 8, (0, 34), torch.float32)
    inputs = tri._make_inputs(case, seed=20260924)
    scalar_g = _chunk_local_g(
        case.total_tokens, case.heads, tri._cu_values(case)
    )
    gk = inputs["gk"]
    state_seed = inputs["initial_state"].to(state_dtype)
    state_t = state_seed.clone()
    state_c = state_seed.clone()
    common = dict(
        k=inputs["k"].to(torch.float16),
        w=inputs["w"].to(torch.float16),
        u=inputs["u"].to(torch.float16),
        g=scalar_g if gate_mode == "g-and-gk" else None,
        gk=gk,
        initial_state_indices=inputs["initial_state_indices"],
        cu_seqlens=inputs["cu_seqlens"],
    )
    h_t, v_t = tri.chunk_gated_delta_rule_fwd_h(
        initial_state=state_t, **common
    )
    h_c, v_c = cuda_ext.chunk_gated_delta_rule_fwd_h(
        initial_state=state_c, **common
    )
    torch.cuda.synchronize()
    assert tri._cosine(h_t, h_c) >= tri.COSINE_THRESHOLD
    assert tri._cosine(v_t, v_c) >= tri.COSINE_THRESHOLD
    assert tri._cosine(state_t, state_c) >= tri.COSINE_THRESHOLD


def _make_dense_general_inputs(
    *, batch: int, tokens: int, heads: int, k_heads: int, K: int, V: int
) -> Dict[str, torch.Tensor]:
    torch.manual_seed(20260924)
    torch.cuda.manual_seed_all(20260924)
    scale = 1.0 / math.sqrt(K)
    k = torch.randn(
        batch, tokens, k_heads, K, device="cuda", dtype=tri.DTYPE
    ) * scale
    w = torch.randn(
        batch, tokens, heads, K, device="cuda", dtype=tri.DTYPE
    ) * scale
    u = torch.randn(
        batch, tokens, heads, V, device="cuda", dtype=tri.DTYPE
    )
    increments = -0.0125 * torch.rand(
        batch, tokens, heads, K, device="cuda", dtype=torch.float32
    )
    gk = torch.empty_like(increments)
    for b in range(batch):
        for start in range(0, tokens, tri.CHUNK_SIZE):
            end = min(start + tri.CHUNK_SIZE, tokens)
            gk[b, start:end] = increments[b, start:end].cumsum(0)
    return {
        "k": k,
        "w": w,
        "u": u,
        "gk": gk,
        "initial_state": torch.randn(
            batch, heads, V, K, device="cuda", dtype=torch.float32
        ) * 0.01,
        "initial_state_indices": torch.arange(
            batch, device="cuda", dtype=torch.int32
        ),
    }


@pytest.mark.skipif(
    not CUDA_INTERFACE_AVAILABLE,
    reason="rebuilt mcoplib CUDA extension and a CUDA/MACA device are required",
)
@pytest.mark.parametrize(
    "batch,tokens,heads,k_heads,K,V",
    [
        pytest.param(2, 17, 4, 2, 64, 96, id="dense-b2-k64-v96-gqa"),
        pytest.param(1, 19, 4, 4, 192, 80, id="k192-v80"),
        pytest.param(1, 9, 2, 1, 256, 33, id="k256-v33-gqa"),
    ],
)
def test_cuda_interface_general_shape_matrix(
    batch: int, tokens: int, heads: int, k_heads: int, K: int, V: int
) -> None:
    """Generic CUDA covers Triton's K<=256, arbitrary-V and dense-B shapes."""

    inputs = _make_dense_general_inputs(
        batch=batch, tokens=tokens, heads=heads, k_heads=k_heads, K=K, V=V
    )
    state_t = inputs["initial_state"].clone()
    state_c = inputs["initial_state"].clone()
    common = dict(
        k=inputs["k"], w=inputs["w"], u=inputs["u"], gk=inputs["gk"],
        initial_state_indices=inputs["initial_state_indices"],
    )
    h_t, v_t = tri.chunk_gated_delta_rule_fwd_h(
        initial_state=state_t, **common
    )
    h_c, v_c = cuda_ext.chunk_gated_delta_rule_fwd_h(
        initial_state=state_c, **common
    )
    torch.cuda.synchronize()
    assert h_c.shape == h_t.shape
    assert v_c.shape == v_t.shape
    assert tri._cosine(h_t, h_c) >= tri.COSINE_THRESHOLD
    assert tri._cosine(v_t, v_c) >= tri.COSINE_THRESHOLD
    assert tri._cosine(state_t, state_c) >= tri.COSINE_THRESHOLD


@pytest.mark.skipif(
    not CUDA_INTERFACE_AVAILABLE,
    reason="rebuilt mcoplib CUDA extension and a CUDA/MACA device are required",
)
@pytest.mark.parametrize("slot", (0, -(1 << 32)), ids=("valid-slot", "wide-negative-sentinel"))
def test_cuda_interface_int64_state_indices(slot: int) -> None:
    """int64 indices remain int64; in particular, -2^32 must not become slot 0."""

    case = tri.Case(34, 1, 8, (0, 34), torch.float32)
    inputs = tri._make_inputs(case)
    indices = torch.tensor([slot], device="cuda", dtype=torch.int64)
    state_t = inputs["initial_state"].clone()
    state_c = inputs["initial_state"].clone()
    common = dict(
        k=inputs["k"], w=inputs["w"], u=inputs["u"], gk=inputs["gk"],
        initial_state_indices=indices, cu_seqlens=inputs["cu_seqlens"],
    )
    h_t, v_t = tri.chunk_gated_delta_rule_fwd_h(initial_state=state_t, **common)
    h_c, v_c = cuda_ext.chunk_gated_delta_rule_fwd_h(
        initial_state=state_c, **common
    )
    torch.cuda.synchronize()
    # A negative sentinel starts from the zero state, so h is exactly zero and
    # cosine similarity is undefined (the helper returns 0 for two zero norms).
    assert torch.equal(h_t, h_c) or (
        tri._cosine(h_t, h_c) >= tri.COSINE_THRESHOLD
    )
    assert tri._cosine(v_t, v_c) >= tri.COSINE_THRESHOLD
    assert tri._cosine(state_t, state_c) >= tri.COSINE_THRESHOLD


def _make_feature_contract_inputs(
    *, gqa: bool, pool_sentinel: bool, use_exp2: bool,
    state_dtype: torch.dtype,
) -> Dict[str, object]:
    """Small ragged case spanning the retained production feature switches."""

    torch.manual_seed(20260924)
    torch.cuda.manual_seed_all(20260924)
    total_tokens, sequences, heads = 135, 2, 8
    k_heads = 4 if gqa else heads
    cu_values = (0, 71, total_tokens)
    scale = 1.0 / math.sqrt(tri.K)
    k = torch.randn(
        1, total_tokens, k_heads, tri.K,
        device="cuda", dtype=tri.DTYPE,
    ) * scale
    w = torch.randn(
        1, total_tokens, heads, tri.K,
        device="cuda", dtype=tri.DTYPE,
    ) * scale
    u = torch.randn(
        1, total_tokens, heads, tri.V,
        device="cuda", dtype=tri.DTYPE,
    )
    gk = tri._chunk_local_gk(total_tokens, heads, cu_values)
    if use_exp2:
        gk.mul_(math.log2(math.e))
    cu = torch.tensor(cu_values, device="cuda", dtype=torch.int32)

    slots = 3 if pool_sentinel else sequences
    padding = 64 if pool_sentinel else 0
    pitch = heads * tri.V * tri.K + padding
    flat_elements = (slots - 1) * pitch + heads * tri.V * tri.K
    flat = (
        torch.randn(flat_elements, device="cuda", dtype=state_dtype) * 0.01
    )
    indices = torch.tensor(
        [2, -1] if pool_sentinel else [0, 1],
        device="cuda", dtype=torch.int32,
    )

    def state_from(base: torch.Tensor) -> torch.Tensor:
        return base.as_strided(
            (slots, heads, tri.V, tri.K),
            (pitch, tri.V * tri.K, tri.K, 1),
        )

    return {
        "common": dict(
            k=k, w=w, u=u, gk=gk, initial_state_indices=indices,
            cu_seqlens=cu, use_exp2=use_exp2,
        ),
        "flat": flat,
        "state_from": state_from,
    }


@pytest.mark.skipif(
    not CUDA_INTERFACE_AVAILABLE,
    reason="rebuilt mcoplib CUDA extension and a CUDA/MACA device are required",
)
@pytest.mark.parametrize(
    "gqa,pool_sentinel,use_exp2,state_dtype",
    [
        pytest.param(True, False, False, torch.float32, id="gqa"),
        pytest.param(False, True, False, torch.float32, id="pool-sentinel"),
        pytest.param(False, False, True, torch.float32, id="exp2"),
        pytest.param(True, True, True, torch.bfloat16, id="combined-bf16"),
    ],
)
def test_cuda_interface_production_feature_matrix(
    gqa: bool, pool_sentinel: bool, use_exp2: bool,
    state_dtype: torch.dtype,
) -> None:
    inputs = _make_feature_contract_inputs(
        gqa=gqa, pool_sentinel=pool_sentinel, use_exp2=use_exp2,
        state_dtype=state_dtype,
    )
    flat_t = inputs["flat"].clone()
    flat_c = inputs["flat"].clone()
    state_t = inputs["state_from"](flat_t)
    state_c = inputs["state_from"](flat_c)
    h_t, v_t = tri.chunk_gated_delta_rule_fwd_h(
        initial_state=state_t, **inputs["common"]
    )
    h_c, v_c = cuda_ext.chunk_gated_delta_rule_fwd_h(
        initial_state=state_c, **inputs["common"]
    )
    torch.cuda.synchronize()
    assert tri._cosine(h_t, h_c) >= tri.COSINE_THRESHOLD
    assert tri._cosine(v_t, v_c) >= tri.COSINE_THRESHOLD
    assert tri._cosine(flat_t, flat_c) >= tri.COSINE_THRESHOLD


@pytest.mark.skipif(
    not CUDA_INTERFACE_AVAILABLE,
    reason="rebuilt mcoplib CUDA extension and a CUDA/MACA device are required",
)
def test_cuda_interface_chunk_plan_cache_reuse_and_invalidation() -> None:
    """A hit reuses metadata; an in-place cu_seqlens edit rebuilds it."""

    case = tri.Case(128, 2, 8, (0, 64, 128), torch.float32)
    inputs = tri._make_inputs(case)
    common = dict(
        k=inputs["k"],
        w=inputs["w"],
        u=inputs["u"],
        gk=inputs["gk"],
        initial_state_indices=inputs["initial_state_indices"],
        cu_seqlens=inputs["cu_seqlens"],
    )

    first_state = inputs["initial_state"].clone()
    second_state = inputs["initial_state"].clone()
    first_h, first_v = cuda_ext.chunk_gated_delta_rule_fwd_h(
        initial_state=first_state, **common
    )
    second_h, second_v = cuda_ext.chunk_gated_delta_rule_fwd_h(
        initial_state=second_state, **common
    )
    torch.cuda.synchronize()
    assert torch.equal(first_h, second_h)
    assert torch.equal(first_v, second_v)
    assert torch.equal(first_state, second_state)

    # This changes the plan from two whole chunks to two partial sequences and
    # three chunks in total.  A stale identity-only cache would retain the old
    # output shape and dense dispatch; the C++ cache keys on tensor version too.
    inputs["cu_seqlens"].copy_(
        torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32)
    )
    fresh_cu = inputs["cu_seqlens"].clone()
    state_t = inputs["initial_state"].clone()
    state_c = inputs["initial_state"].clone()
    h_t, v_t = tri.chunk_gated_delta_rule_fwd_h(
        initial_state=state_t, **{**common, "cu_seqlens": fresh_cu}
    )
    h_c, v_c = cuda_ext.chunk_gated_delta_rule_fwd_h(
        initial_state=state_c, **common
    )
    torch.cuda.synchronize()
    assert h_c.shape == h_t.shape
    assert tri._cosine(h_t, h_c) >= tri.COSINE_THRESHOLD
    assert tri._cosine(v_t, v_c) >= tri.COSINE_THRESHOLD
    assert tri._cosine(state_t, state_c) >= tri.COSINE_THRESHOLD


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--json-out", type=pathlib.Path)
    args = parser.parse_args()

    torch.manual_seed(0)
    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}"
          f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"cosine threshold: {tri.COSINE_THRESHOLD}\n")
    print(f"{'case':<34} {'h':>9} {'v_new':>9} {'state':>9}  {'bitwise':>8}  {'result':>6}")
    results = []
    for case in tri._cases(args.full):
        metrics = compare_case(case)
        results.append(metrics)
        worst = min(metrics["h_cosine"], metrics["v_new_cosine"], metrics["state_cosine"])
        all_bitwise = all(metrics[f"{n}_bitwise"] for n in ("h", "v_new", "state"))
        print(f"{metrics['label']:<34} {metrics['h_cosine']:9.7f} "
              f"{metrics['v_new_cosine']:9.7f} {metrics['state_cosine']:9.7f}  "
              f"{str(all_bitwise):>8}  {'PASS' if metrics['passed'] else 'FAIL':>6}")
    ok = all(m["passed"] for m in results)
    print(f"\ncases={len(results)}  pass={sum(1 for m in results if m['passed'])}  "
          f"overall={'PASS' if ok else 'FAIL'}")
    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"threshold": tri.COSINE_THRESHOLD, "cases": results}, indent=2))
        print(f"wrote {args.json_out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
