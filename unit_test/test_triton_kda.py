"""Correctness coverage for the standalone KDA/GLA Triton kernels."""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass

import pytest
import torch
import triton

from mcoplib import triton_kda as kda_implementation
from mcoplib.triton_kda import (
    RCP_LN2,
    chunk_gla_fwd_kernel_o,
    chunk_gla_fwd_o_gk,
    kda_gate_chunk_cumsum_vector_kernel,
    kda_gate_chunk_cumsum,
    prepare_chunk_indices,
)


BT = 64
K = 128
V = 128
S = 128
SCALE = 1.0 / math.sqrt(K)

# Unique tensor-shape families recovered from the GLM5.3-flash C500 trace.
GLM53_FLASH_SHAPES = (
    (64, 64),
    (19, 8),
    (8192, 8),
    (4, 8),
    (17, 64),
    (1, 64),
    (19, 64),
    (17, 8),
    (1, 8),
    (3072, 8),
    (8164, 8),
    (8130, 8),
    (8170, 8),
    (5056, 8),
    (7168, 8),
    (8177, 8),
    (8141, 8),
    (8133, 8),
    (8148, 8),
    (8149, 8),
    (4672, 8),
    (2985, 8),
    (5092, 8),
    (41, 64),
    (36, 64),
    (13, 64),
    (49, 64),
)


@dataclass(frozen=True)
class BaselineCase:
    name: str
    tokens: int
    heads: int
    gla_launches: int
    gla_pct: float
    gate_launches: int
    gate_pct: float


# These six tensor shapes cover 97.38% of chunk_gla and 97.79% of gate-cumsum
# launches in the external GLM5.3-flash trace. Counts combine variants that
# differ only in the autotuner-selected launch configuration.
GLM53_FLASH_BASELINE_CASES = (
    BaselineCase("decode_t64_h64", 64, 64, 378896, 71.12, 331722, 84.44),
    BaselineCase("prefill_t8192_h8", 8192, 8, 33728, 6.33, 33728, 8.59),
    BaselineCase("short_t19_h8", 19, 8, 54035, 10.14, 9566, 2.43),
    BaselineCase("short_t4_h8", 4, 8, 26580, 4.99, 4885, 1.24),
    BaselineCase("short_t17_h64", 17, 64, 12820, 2.41, 2098, 0.53),
    BaselineCase("short_t1_h64", 1, 64, 12774, 2.40, 2170, 0.55),
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA/MACA device"
)


def _chunk_gla_reference(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h: torch.Tensor,
    A: torch.Tensor,
) -> torch.Tensor:
    _, tokens, _, _ = q.shape
    result = torch.empty_like(v)
    for chunk, start in enumerate(range(0, tokens, BT)):
        end = min(start + BT, tokens)
        length = end - start
        gated_q = (
            q[0, start:end].float()
            * SCALE
            * torch.exp2(g[0, start:end].float())
        )
        state_term = torch.einsum(
            "thk,hvk->thv", gated_q, h[0, chunk].float()
        )
        a_block = A[0, start:end, :, :length].float().permute(1, 0, 2)
        a_block = torch.tril(a_block)
        v_block = v[0, start:end].float().permute(1, 0, 2)
        local_term = torch.matmul(a_block, v_block).permute(1, 0, 2)
        result[0, start:end] = (state_term + local_term).to(result.dtype)
    return result


def _gate_reference(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
) -> torch.Tensor:
    _, tokens, heads, state_size = raw_gate.shape
    values = raw_gate.float() + dt_bias.view(heads, state_size)[None, None]
    rate = torch.exp(A_log.reshape(heads).float())[None, None, :, None]
    gate = lower_bound * torch.sigmoid(rate * values)
    output = torch.empty_like(gate)
    for start in range(0, tokens, BT):
        end = min(start + BT, tokens)
        output[:, start:end] = torch.cumsum(gate[:, start:end], dim=1)
    return output * RCP_LN2


@pytest.mark.parametrize("tokens,heads", [(64, 64), (19, 8)])
def test_chunk_gla_fwd_o_gk_glm53_flash(tokens: int, heads: int) -> None:
    torch.manual_seed(53 + tokens + heads)
    chunks = (tokens + BT - 1) // BT
    q = torch.randn((1, tokens, heads, K), device="cuda", dtype=torch.bfloat16) * 0.1
    v = torch.randn((1, tokens, heads, V), device="cuda", dtype=torch.bfloat16) * 0.1
    g = -torch.rand((1, tokens, heads, K), device="cuda", dtype=torch.float32)
    h = torch.randn(
        (1, chunks, heads, V, K), device="cuda", dtype=torch.bfloat16
    ) * 0.1
    A = torch.randn(
        (1, tokens, heads, BT), device="cuda", dtype=torch.bfloat16
    ) * 0.05
    output = torch.empty_like(v)
    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

    actual = chunk_gla_fwd_o_gk(
        q=q,
        v=v,
        g=g,
        A=A,
        h=h,
        o=output,
        scale=SCALE,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        chunk_indices=chunk_indices,
    )
    expected = _chunk_gla_reference(q, v, g, h, A)

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("tokens,heads", [(64, 64), (19, 8)])
def test_kda_gate_chunk_cumsum_glm53_flash(tokens: int, heads: int) -> None:
    torch.manual_seed(97 + tokens + heads)
    raw_gate = torch.randn(
        (1, tokens, heads, S), device="cuda", dtype=torch.bfloat16
    ) * 0.25
    A_log = torch.randn((1, 1, heads, 1), device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn((heads * S,), device="cuda", dtype=torch.float32) * 0.1
    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    lower_bound = -5.0

    actual = kda_gate_chunk_cumsum(
        raw_gate,
        A_log=A_log,
        chunk_size=BT,
        scale=RCP_LN2,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
        chunk_indices=chunk_indices,
        lower_bound=lower_bound,
    )
    expected = _gate_reference(raw_gate, A_log, dt_bias, lower_bound)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.full
@pytest.mark.parametrize("tokens,heads", [(8192, 8)])
def test_glm53_flash_prefill_shape(tokens: int, heads: int) -> None:
    """Run the dominant long-prefill gate shape without burdening default CI."""

    raw_gate = torch.zeros(
        (1, tokens, heads, S), device="cuda", dtype=torch.bfloat16
    )
    A_log = torch.zeros((1, 1, heads, 1), device="cuda", dtype=torch.float32)
    dt_bias = torch.zeros((heads * S,), device="cuda", dtype=torch.float32)
    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)

    output = kda_gate_chunk_cumsum(
        raw_gate,
        A_log=A_log,
        chunk_size=BT,
        scale=RCP_LN2,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        lower_bound=-5.0,
    )

    assert output.shape == raw_gate.shape
    assert torch.isfinite(output).all()


def _benchmark_ms(call) -> tuple[float, list[float]]:
    call()
    torch.cuda.synchronize()
    samples = [
        float(triton.testing.do_bench(call, warmup=25, rep=100))
        for _ in range(5)
    ]
    return statistics.median(samples), samples


def _format_autotune_config(kernel) -> str:
    config = None
    for _ in range(4):
        config = getattr(kernel, "best_config", None)
        if config is not None or not hasattr(kernel, "fn"):
            break
        kernel = kernel.fn
    if config is None:
        return "unknown"
    kwargs = "/".join(f"{key}{value}" for key, value in config.kwargs.items())
    stages = getattr(config, "num_stages", None)
    suffix = f"/W{config.num_warps}"
    if stages is not None:
        suffix += f"/S{stages}"
    return kwargs + suffix


def _format_static_config(config: dict[str, int]) -> str:
    meta_keys = {"num_warps": "W", "num_stages": "S"}
    return "/".join(
        f"{meta_keys.get(key, key)}{value}" for key, value in config.items()
    )


@torch.no_grad()
def run_glm53_flash_baseline() -> None:
    """Measure the imported external implementation before optimization."""

    if not torch.cuda.is_available():
        raise RuntimeError("baseline performance measurement requires CUDA/MACA")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("method: 25 ms warmup + 100 ms measurement, 5 rounds, median")
    print("implementation: external kda.py copy in mcoplib (unoptimized)\n")

    for operator in ("chunk_gla_fwd_kernel_o", "kda_gate_chunk_cumsum_vector_kernel"):
        print(operator)
        print(
            f"{'shape':>22} {'launches':>10} {'trace%':>8} "
            f"{'median(us)':>12} {'config':>24}"
        )
        for case in GLM53_FLASH_BASELINE_CASES:
            torch.manual_seed(2026 + case.tokens + case.heads)
            tokens, heads = case.tokens, case.heads
            cu_seqlens = torch.tensor(
                [0, tokens], device="cuda", dtype=torch.int32
            )
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

            if operator == "chunk_gla_fwd_kernel_o":
                chunks = (tokens + BT - 1) // BT
                q = torch.randn(
                    (1, tokens, heads, K), device="cuda", dtype=torch.bfloat16
                )
                v = torch.randn(
                    (1, tokens, heads, V), device="cuda", dtype=torch.bfloat16
                )
                g = -torch.rand(
                    (1, tokens, heads, K), device="cuda", dtype=torch.float32
                )
                h = torch.randn(
                    (1, chunks, heads, V, K),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                A = torch.randn(
                    (1, tokens, heads, BT),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                output = torch.empty_like(v)

                def run():
                    return chunk_gla_fwd_o_gk(
                        q=q,
                        v=v,
                        g=g,
                        A=A,
                        h=h,
                        o=output,
                        scale=SCALE,
                        cu_seqlens=cu_seqlens,
                        chunk_size=BT,
                        chunk_indices=chunk_indices,
                        _use_c500_optimized=False,
                    )

                launches, trace_pct = case.gla_launches, case.gla_pct
                kernel = chunk_gla_fwd_kernel_o
                static_config = (
                    kda_implementation._chunk_gla_c500_baseline_config(
                        heads, K, V, BT
                    )
                )
            else:
                raw_gate = torch.randn(
                    (1, tokens, heads, S),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                A_log = torch.randn(
                    (1, 1, heads, 1), device="cuda", dtype=torch.float32
                )
                dt_bias = torch.randn(
                    (heads * S,), device="cuda", dtype=torch.float32
                )

                def run():
                    return kda_gate_chunk_cumsum(
                        raw_gate,
                        A_log=A_log,
                        chunk_size=BT,
                        scale=RCP_LN2,
                        dt_bias=dt_bias,
                        cu_seqlens=cu_seqlens,
                        output_dtype=torch.float32,
                        chunk_indices=chunk_indices,
                        lower_bound=-5.0,
                        _use_c500_optimized=False,
                    )

                launches, trace_pct = case.gate_launches, case.gate_pct
                kernel = kda_gate_chunk_cumsum_vector_kernel
                static_config = (
                    kda_implementation._kda_gate_c500_baseline_config(
                        heads, S, BT
                    )
                )

            median_ms, samples = _benchmark_ms(run)
            config_text = (
                _format_static_config(static_config)
                if static_config is not None
                else _format_autotune_config(kernel)
            )
            print(
                f"{case.name:>22} {launches:>10} {trace_pct:>7.2f}% "
                f"{median_ms * 1e3:>12.3f} {config_text:>24}"
            )
            print(f"  rounds_ms={samples}")

            del run
            torch.cuda.empty_cache()
        print()


def test_c500_static_dispatch_policy() -> None:
    if not kda_implementation._IS_XCORE1000_OR_1008:
        pytest.skip("C500-specific dispatch policy")

    assert kda_implementation._chunk_gla_c500_config(64, 64, K, V, BT) == {
        "BK": 64,
        "BV": 64,
        "num_warps": 4,
        "num_stages": 3,
    }
    assert kda_implementation._chunk_gla_c500_config(8192, 8, K, V, BT) == {
        "BK": 64,
        "BV": 128,
        "num_warps": 4,
        "num_stages": 4,
    }
    assert kda_implementation._chunk_gla_c500_config(19, 8, K, V, BT) == {
        "BK": 64,
        "BV": 32,
        "num_warps": 8,
        "num_stages": 4,
    }
    assert kda_implementation._chunk_gla_c500_config(512, 8, K, V, BT) == {
        "BK": 64,
        "BV": 64,
        "num_warps": 4,
        "num_stages": 3,
    }
    assert kda_implementation._kda_gate_c500_config(64, 64, S, BT) == {
        "BS": 16,
        "num_warps": 2,
        "num_stages": 3,
    }
    assert kda_implementation._kda_gate_c500_config(8192, 8, S, BT) == {
        "BS": 64,
        "num_warps": 2,
        "num_stages": 3,
    }
    assert kda_implementation._kda_gate_c500_config(19, 8, S, BT) == {
        "BS": 16,
        "num_warps": 8,
        "num_stages": 3,
    }
    assert kda_implementation._kda_gate_c500_config(512, 8, S, BT) == {
        "BS": 16,
        "num_warps": 8,
        "num_stages": 3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-perf",
        action="store_true",
        help="measure the unoptimized external implementation copy",
    )
    args = parser.parse_args()
    if not args.baseline_perf:
        parser.error("pass --baseline-perf to run the performance table")
    run_glm53_flash_baseline()


if __name__ == "__main__":
    main()
