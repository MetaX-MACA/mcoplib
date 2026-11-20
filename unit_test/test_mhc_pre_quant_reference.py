"""Compare standalone and fused MHC pre RMSNorm + INT8 quant paths.

Baselines:

* ``rms_quant``: ``mhc_pre(combine only)`` followed by the SGLang
  ``rms_norm_dynamic_per_token_quant_custom`` implementation, which writes
  INT8, BF16 RMSNorm and per-token scale outputs in one kernel.
* ``pack_quant``: ``mhc_pre(combine only)`` followed by
  ``fused_add_rms_norm_dynamic_per_token_quant_padding_output``.

The baselines are compared with one ``mhc_pre`` call using respectively
``quant_mode="int8"`` and ``quant_mode="pack_int8"``.  The script reports
accuracy and end-to-end GPU latency. Both reference operators write a BF16
RMSNorm output; the packed reference also writes an unpacked INT8 output. The
measured latency includes those output writes.

Run on C500 after building/installing mcoplib and DeepGEMM::

    pytest -q -s unit_test/test_mhc_pre_quant_reference.py

Environment variables::

    MHC_PRE_QUANT_SHAPES=1,8,193,2048
    MHC_PRE_QUANT_WARMUP=5
    MHC_PRE_QUANT_ROUNDS=5
    MHC_PRE_QUANT_DECODE_REPEAT=50
    MHC_PRE_QUANT_PREFILL_REPEAT=10
    MHC_PRE_QUANT_MIN_COSINE=0.999
    MHC_PRE_QUANT_MAX_SCALE_REL=0.02

Bitwise equality is not required: the standalone operators and fused epilogue
round BF16 values at different points, so scales and quantization bins can
differ slightly.
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass
from typing import Callable, NamedTuple

import pytest
import torch


pytest.importorskip("deep_gemm")

import mcoplib.sgl_kernel  # noqa: E402,F401: registers torch.ops.sgl_kernel
from mcoplib.custom_ops import (  # noqa: E402
    fused_add_rms_norm_dynamic_per_token_quant_padding_output,
)
from mcoplib.mhc import mhc_pre  # noqa: E402


HC = 4
HIDDEN = 5120
HC_HIDDEN = HC * HIDDEN
MIX = HC * (2 + HC)
PACKED_ROW_BYTES = ((HIDDEN + 4 + 511) // 512) * 512
PACKED_ROW_ELEMENTS = PACKED_ROW_BYTES // 2

RMS_EPS = 1e-6
HC_PRE_EPS = 1e-6
HC_SINKHORN_EPS = 1e-6
HC_POST_MULT = 2.0
SINKHORN_REPEAT = 20
NORM_EPS = 1e-6

# Complete set of decode and prefill token counts observed in the production
# MHC log.
DEFAULT_NUM_TOKENS = (
    1,
    4,
    8,
    193,
    2048,
    4096,
    6125,
    12249,
    12255,
    14301,
    16345,
)
REFERENCE_MODES = ("rms_quant", "pack_quant")
_BENCHMARK_RESULTS: list[dict[str, object]] = []


@dataclass(frozen=True)
class Inputs:
    residual: torch.Tensor
    fn: torch.Tensor
    hc_scale: torch.Tensor
    hc_base: torch.Tensor
    norm_weight: torch.Tensor


class PathOutput(NamedTuple):
    post_mix: torch.Tensor
    comb_mix: torch.Tensor
    quantized: torch.Tensor
    scales: torch.Tensor
    packed: torch.Tensor | None
    output_rms: torch.Tensor | None


def _selected_shapes() -> tuple[int, ...]:
    value = os.getenv("MHC_PRE_QUANT_SHAPES")
    if value is None:
        return DEFAULT_NUM_TOKENS
    shapes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not shapes or any(shape <= 0 for shape in shapes):
        raise ValueError("MHC_PRE_QUANT_SHAPES must contain positive integers")
    return shapes


def _make_inputs(num_tokens: int) -> Inputs:
    torch.manual_seed(20261028 + num_tokens)
    device = torch.device("cuda")
    residual = torch.randn(
        (num_tokens, HC, HIDDEN), device=device, dtype=torch.bfloat16
    )
    fn = torch.randn((MIX, HC_HIDDEN), device=device, dtype=torch.float32)
    fn.mul_(HC_HIDDEN**-0.5)
    hc_scale = torch.tensor((0.8, 1.1, 0.9), device=device, dtype=torch.float32)
    hc_base = torch.randn((MIX,), device=device, dtype=torch.float32) * 0.05
    norm_weight = (
        torch.randn((HIDDEN,), device=device, dtype=torch.bfloat16) * 0.1 + 1.0
    )
    return Inputs(
        residual=residual,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        norm_weight=norm_weight,
    )


def _run_combine_only(inputs: Inputs):
    """Run the old mhc_pre contract and return its BF16 collapsed output."""
    post_mix, comb_mix, collapsed = mhc_pre(
        inputs.residual,
        inputs.fn,
        inputs.hc_scale,
        inputs.hc_base,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT,
        SINKHORN_REPEAT,
    )
    return post_mix, comb_mix, collapsed


def _run_reference(inputs: Inputs, reference_mode: str) -> PathOutput:
    """Run old mhc_pre followed by a standalone RMSNorm+quant operator."""
    post_mix, comb_mix, collapsed = _run_combine_only(inputs)

    if reference_mode == "rms_quant":
        quantized = torch.empty_like(collapsed, dtype=torch.int8)
        output_rms = torch.empty_like(collapsed, dtype=torch.bfloat16)
        scales = torch.empty(
            (collapsed.shape[0],), device=collapsed.device, dtype=torch.float32
        )
        torch.ops.sgl_kernel.rms_norm_dynamic_per_token_quant_custom.default(
            quantized,
            output_rms,
            collapsed,
            inputs.norm_weight,
            scales,
            NORM_EPS,
            None,
            None,
        )
        return PathOutput(
            post_mix,
            comb_mix,
            quantized,
            scales.flatten(),
            None,
            output_rms,
        )

    if reference_mode == "pack_quant":
        packed, residual_out, output_rms, quantized, scales = (
            fused_add_rms_norm_dynamic_per_token_quant_padding_output(
                input=collapsed,
                residual=None,
                weight=inputs.norm_weight,
                pad_size=PACKED_ROW_ELEMENTS,
                epsilon=NORM_EPS,
            )
        )
        assert residual_out is None
        return PathOutput(
            post_mix,
            comb_mix,
            quantized,
            scales.flatten(),
            packed,
            output_rms,
        )

    raise ValueError(f"unknown reference mode: {reference_mode}")


def _run_fused(inputs: Inputs, reference_mode: str) -> PathOutput:
    """Run the new mhc_pre combine + RMSNorm + quant/pack-quant path."""
    quant_mode = "int8" if reference_mode == "rms_quant" else "pack_int8"
    result = mhc_pre(
        inputs.residual,
        inputs.fn,
        inputs.hc_scale,
        inputs.hc_base,
        RMS_EPS,
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        HC_POST_MULT,
        SINKHORN_REPEAT,
        norm_weight=inputs.norm_weight,
        norm_eps=NORM_EPS,
        quant_mode=quant_mode,
    )

    if quant_mode == "int8":
        post_mix, comb_mix, layer_input, scales, output_rms = result
        return PathOutput(
            post_mix,
            comb_mix,
            layer_input,
            scales.flatten(),
            None,
            output_rms,
        )

    post_mix, comb_mix, layer_input, scales, output_rms, quantized = result
    return PathOutput(
        post_mix,
        comb_mix,
        quantized,
        scales.flatten(),
        layer_input,
        output_rms,
    )


def _cosine(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        lhs.flatten().float(), rhs.flatten().float(), dim=0
    ).item()


def _max_abs(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return (lhs.float() - rhs.float()).abs().max().item()


def _max_relative(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    denominator = rhs.float().abs().clamp_min(torch.finfo(torch.float32).eps)
    return ((lhs.float() - rhs.float()).abs() / denominator).max().item()


def _accuracy(reference: PathOutput, fused: PathOutput) -> dict[str, float]:
    reference_dequant = reference.quantized.float() * reference.scales[:, None]
    fused_dequant = fused.quantized.float() * fused.scales[:, None]
    int_diff = (
        fused.quantized.to(torch.int16) - reference.quantized.to(torch.int16)
    ).abs()
    return {
        "cosine": _cosine(fused_dequant, reference_dequant),
        "dequant_max_abs": _max_abs(fused_dequant, reference_dequant),
        "scale_max_abs": _max_abs(fused.scales, reference.scales),
        "scale_max_rel": _max_relative(fused.scales, reference.scales),
        "int8_max_diff": int_diff.max().item(),
        "int8_mismatch_ratio": (int_diff != 0).float().mean().item(),
    }


def _benchmark_ms(fn: Callable[[], object], repeat: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeat


def _benchmark_pair_ms(
    reference_fn: Callable[[], object],
    fused_fn: Callable[[], object],
    *,
    warmup: int,
    repeat: int,
    rounds: int,
) -> tuple[float, float]:
    """Alternate order and return median end-to-end GPU time for each path."""
    for _ in range(warmup):
        reference_fn()
        fused_fn()
    torch.cuda.synchronize()

    reference_times = []
    fused_times = []
    for round_index in range(rounds):
        if round_index % 2 == 0:
            reference_times.append(_benchmark_ms(reference_fn, repeat))
            fused_times.append(_benchmark_ms(fused_fn, repeat))
        else:
            fused_times.append(_benchmark_ms(fused_fn, repeat))
            reference_times.append(_benchmark_ms(reference_fn, repeat))
    return statistics.median(reference_times), statistics.median(fused_times)


def _print_table(headers, rows) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(separator)
    print(
        "| "
        + " | ".join(
            str(value).ljust(widths[index])
            for index, value in enumerate(headers)
        )
        + " |"
    )
    print(separator)
    for row in rows:
        print(
            "| "
            + " | ".join(
                str(value).rjust(widths[index])
                for index, value in enumerate(row)
            )
            + " |"
        )
    print(separator)


@pytest.fixture(scope="module", autouse=True)
def _benchmark_summary():
    _BENCHMARK_RESULTS.clear()
    yield
    if not _BENCHMARK_RESULTS:
        return

    rows = []
    for result in sorted(
        _BENCHMARK_RESULTS,
        key=lambda item: (
            str(item["reference"]),
            int(item["tokens"]),
        ),
    ):
        rows.append(
            (
                result["reference"],
                result["tokens"],
                f'{result["reference_ms"]:.4f}',
                f'{result["fused_ms"]:.4f}',
                f'{result["speedup"]:.2f}x',
                f'{result["cosine"]:.7f}',
                f'{result["dequant_max_abs"]:.3g}',
                f'{result["scale_max_rel"]:.3g}',
                f'{result["int8_max_diff"]:.0f}',
                f'{100.0 * result["int8_mismatch_ratio"]:.3f}%',
            )
        )

    print("\nold mhc_pre + standalone RMSNorm/quant vs fused mhc_pre")
    _print_table(
        (
            "reference",
            "N",
            "old+quant(ms)",
            "fused(ms)",
            "speedup",
            "deq_cos",
            "deq_max",
            "scale_rel",
            "q_maxdiff",
            "q_mismatch",
        ),
        rows,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA/MACA device"
)
@pytest.mark.parametrize("reference_mode", REFERENCE_MODES)
@pytest.mark.parametrize("num_tokens", _selected_shapes())
def test_mhc_pre_fused_quant_against_standalone(
    num_tokens: int,
    reference_mode: str,
):
    inputs = _make_inputs(num_tokens)

    # First calls force lazy loading/JIT and also provide accuracy outputs.
    reference = _run_reference(inputs, reference_mode)
    fused = _run_fused(inputs, reference_mode)
    torch.cuda.synchronize()

    # MHC coefficient generation must be unaffected by the output epilogue.
    torch.testing.assert_close(fused.post_mix, reference.post_mix)
    torch.testing.assert_close(fused.comb_mix, reference.comb_mix)

    assert fused.quantized.shape == reference.quantized.shape == (
        num_tokens,
        HIDDEN,
    )
    assert fused.quantized.dtype == reference.quantized.dtype == torch.int8
    assert fused.scales.shape == reference.scales.shape == (num_tokens,)
    assert torch.isfinite(fused.scales).all()
    assert torch.all(fused.scales > 0)

    assert fused.output_rms is not None and reference.output_rms is not None
    assert fused.output_rms.shape == reference.output_rms.shape == (
        num_tokens,
        HIDDEN,
    )
    assert (
        fused.output_rms.dtype
        == reference.output_rms.dtype
        == torch.bfloat16
    )
    torch.testing.assert_close(
        fused.output_rms,
        reference.output_rms,
        rtol=2e-2,
        atol=2e-2,
    )

    if reference_mode == "pack_quant":
        assert fused.packed is not None and reference.packed is not None
        assert fused.packed.shape == reference.packed.shape == (
            num_tokens,
            PACKED_ROW_ELEMENTS,
        )
        fused_bytes = fused.packed.view(torch.uint8)
        reference_bytes = reference.packed.view(torch.uint8)

        # Padding is not compared because the old optimized reference kernel
        # does not promise to initialize it.
        torch.testing.assert_close(
            reference_bytes[:, :HIDDEN], reference.quantized.view(torch.uint8)
        )
        torch.testing.assert_close(
            fused_bytes[:, :HIDDEN], fused.quantized.view(torch.uint8)
        )
        fused_embedded_scales = (
            fused_bytes[:, HIDDEN : HIDDEN + 4]
            .contiguous()
            .view(torch.float32)
            .flatten()
        )
        torch.testing.assert_close(fused_embedded_scales, fused.scales)
        assert torch.count_nonzero(fused_bytes[:, HIDDEN + 4 :]) == 0

    metrics = _accuracy(reference, fused)
    min_cosine = float(os.getenv("MHC_PRE_QUANT_MIN_COSINE", "0.999"))
    max_scale_rel = float(os.getenv("MHC_PRE_QUANT_MAX_SCALE_REL", "0.02"))
    assert metrics["cosine"] >= min_cosine
    assert metrics["scale_max_rel"] <= max_scale_rel

    phase = "DECODE" if num_tokens <= 193 else "PREFILL"
    repeat_default = "50" if phase == "DECODE" else "10"
    repeat = int(os.getenv(f"MHC_PRE_QUANT_{phase}_REPEAT", repeat_default))
    warmup = int(os.getenv("MHC_PRE_QUANT_WARMUP", "5"))
    rounds = int(os.getenv("MHC_PRE_QUANT_ROUNDS", "5"))
    reference_ms, fused_ms = _benchmark_pair_ms(
        lambda: _run_reference(inputs, reference_mode),
        lambda: _run_fused(inputs, reference_mode),
        warmup=warmup,
        repeat=repeat,
        rounds=rounds,
    )

    _BENCHMARK_RESULTS.append(
        {
            "reference": reference_mode,
            "tokens": num_tokens,
            "reference_ms": reference_ms,
            "fused_ms": fused_ms,
            "speedup": reference_ms / fused_ms,
            **metrics,
        }
    )
    print(
        f"N={num_tokens} reference={reference_mode} "
        f"cos={metrics['cosine']:.7f} "
        f"deq_max={metrics['dequant_max_abs']:.3g} "
        f"scale_rel={metrics['scale_max_rel']:.3g} "
        f"q_maxdiff={metrics['int8_max_diff']:.0f} "
        f"mismatch={100.0 * metrics['int8_mismatch_ratio']:.3f}% "
        f"old+quant={reference_ms:.4f}ms fused={fused_ms:.4f}ms "
        f"speedup={reference_ms / fused_ms:.2f}x"
    )
