import gc
import random
from typing import Tuple

import pytest
import torch

from mcoplib.triton_silu_mul_fp8_quant_deep_gemm import (
    persistent_masked_m_swiglu_mul_quant,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROUP_SIZE = 128
LIMIT = 7.0  # SwiGLU-step clamp limit, matches SwiGLU-OAI / DeepSeek-V3 default
SEED = 42
COS_THRESHOLD = 0.999

# DeepSeek-V3 step3.5 / step3.7 representative shapes + small edge cases.
# (E, T, H, scale_fmt_label)
CASES = [
    # Edge cases
    (1, 1, 128, "fp32"),
    (1, 4, 128, "fp32"),
    (2, 4, 256, "fp32"),
    (1, 4, 384, "fp32"),
    (8, 16, 512, "fp32"),
    (8, 16, 640, "fp32"),
    (8, 16, 768, "fp32"),
    (8, 16, 896, "fp32"),
    (8, 16, 1024, "fp32"),
    (8, 16, 1152, "fp32"),
    # step3.5 / step3.7 representative shapes.
    (8, 64, 7168, "fp32"),
    (8, 128, 7168, "fp32"),
    (8, 512, 7168, "fp32"),
    (8, 1024, 7168, "fp32"),
    (1, 4, 1280, "fp32"),
    (17, 31, 768, "fp32"),
    (32, 64, 256, "fp32"),
    (256, 8, 7168, "fp32"),
    (256, 32, 7168, "fp32"),
    (256, 64, 7168, "fp32"),
    # UE8M0 ceil'd scales path (fp32 storage but ceil'd to power-of-2).
    (8, 16, 1024, "ue8m0"),
    (8, 64, 7168, "ue8m0"),
    (8, 512, 7168, "ue8m0"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def _token_random(E, T, H2, tokens_per_expert, device="cuda"):
    """Random bf16 input with a wide range of scales per token (matches the
    CUDA test's _token_random -- exposes the clamp limit realistically)."""
    y = torch.empty((E, T, H2), dtype=torch.bfloat16, device=device)
    for e in range(E):
        nt = tokens_per_expert[e].item()
        for t in range(nt):
            exp = random.choice(range(1, 20))
            y[e, t].uniform_(-(2 ** exp), 2 ** exp)
    return y


def _fp8_max_bf16() -> torch.Tensor:
    fp8_min, fp8_max = get_fp8_min_max()
    return torch.tensor([fp8_max], device="cuda", dtype=torch.bfloat16)


def _fp8_min_bf16() -> torch.Tensor:
    fp8_min, fp8_max = get_fp8_min_max()
    return torch.tensor([fp8_min], device="cuda", dtype=torch.bfloat16)


def silu_bf16(x_bf16: torch.Tensor) -> torch.Tensor:
    # kernel computes in fp32 then casts back: x / (1 + exp(-x)).
    x_f32 = x_bf16.to(torch.float32)
    act_f32 = x_f32 / (1.0 + torch.exp(-x_f32))
    return act_f32.to(torch.bfloat16)


def ref_swiglustep_mul_quant(
    gate: torch.Tensor,   # bf16 [..., H]
    up: torch.Tensor,     # bf16 [..., H]
    fp8_dtype: torch.dtype,
    group_size: int,
    ceil_ue8m0: bool,
    limit: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference that matches the kernel arithmetic in bf16:
        Gate_act = min(SiLU(gate), limit)        (one-sided, bf16)
        Up_act   = clip(up, -limit, limit)        (symmetric, bf16)
        Result   = Gate_act * Up_act
    then per-group absmax FP8 quantization in bfloat16 precision."""
    assert gate.dtype == torch.bfloat16
    assert up.dtype == torch.bfloat16
    assert gate.size(-1) % group_size == 0

    eps_bf16 = torch.tensor([1e-10], device=gate.device, dtype=torch.bfloat16)
    one_bf16 = torch.tensor([1.0], device=gate.device, dtype=torch.bfloat16)
    fp8_max_bf16 = _fp8_max_bf16()
    fp8_min_bf16 = _fp8_min_bf16()
    fp8_max_inv = one_bf16 / fp8_max_bf16
    limit_bf16 = torch.tensor([limit], device=gate.device, dtype=torch.bfloat16)
    neg_limit_bf16 = torch.tensor(
        [-limit], device=gate.device, dtype=torch.bfloat16
    )

    # SwiGLU-step activation in bf16.
    gate_act = torch.min(silu_bf16(gate), limit_bf16)
    up_act = torch.clamp(up, min=neg_limit_bf16, max=limit_bf16)
    a_m = gate_act * up_act

    x_og_shape = a_m.shape
    num_groups = a_m.numel() // group_size

    a_m = a_m.to(torch.bfloat16).view((-1, group_size))
    amax = a_m.abs().amax(dim=1).clamp(min=eps_bf16)
    s = amax * fp8_max_inv
    if ceil_ue8m0:
        s = torch.exp2(
            torch.ceil(torch.log2(s.to(torch.bfloat16))).to(torch.bfloat16)
        ).to(torch.bfloat16)

    inv_s = one_bf16 / s
    inv_s = inv_s.view((num_groups, 1))
    xq = torch.clamp(
        a_m * inv_s,
        min=fp8_min_bf16.item(),
        max=fp8_max_bf16.item(),
    ).to(fp8_dtype)

    xq = xq.view(x_og_shape)
    s = s.view((-1, xq.size(-1) // group_size))
    return xq, s.to(torch.float32)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float32).reshape(-1)
    b = b.to(torch.float32).reshape(-1)
    if a.numel() == 0:
        return 1.0
    num = float((a * b).sum().item())
    den = float(torch.norm(a).item() * torch.norm(b).item()) + 1e-30
    return num / den


def _bandwidth_gb(E, T, H, scale_fmt):
    # bytes read: input bf16 (E*T*2H*2)
    # bytes written: y_q fp8 (E*T*H*1) + scales (fp32 or packed UE8M0)
    bytes_read = E * T * 2 * H * 2
    bytes_written_q = E * T * H * 1
    if scale_fmt == "ue8m0":
        # UE8M0 packs 4 scales per int32 -> 1 byte per scale effectively,
        # stored as int32 (4 bytes per 4 scales).
        bytes_written_s = E * T * (H // GROUP_SIZE)  # uint8-equivalent
    else:
        bytes_written_s = E * T * (H // GROUP_SIZE) * 4
    return (bytes_read + bytes_written_q + bytes_written_s) / 1.0e9


def _bandwidth_efficiency(gbps: float) -> Tuple[float, float]:
    # Metax C500 (C280-class) peak HBM2e bandwidth ~1.55 TB/s.
    peak = 1.55e3  # GB/s
    eff = (gbps / peak) * 100.0
    return peak, eff


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("E,T,H,scale_fmt", CASES)
@torch.inference_mode()
def test_persistent_masked_m_swiglustep_mul_quant(E, T, H, scale_fmt):
    _set_seed(SEED)

    tokens_per_expert = torch.randint(
        low=0, high=max(1, T), size=(E,), dtype=torch.int32, device="cuda"
    )
    # Guarantee at least one expert has work; otherwise the kernel returns
    # early for every block and there is nothing to validate.
    if tokens_per_expert.sum().item() == 0:
        tokens_per_expert[0] = min(T, 4)

    y = _token_random(E, T, 2 * H, tokens_per_expert)
    gate = y[..., :H].contiguous()
    up = y[..., H:].contiguous()
    input_two = torch.cat([gate, up], dim=-1)  # [E, T, 2H]

    ceil_ue8m0 = scale_fmt == "ue8m0"
    quant_scale_fmt = (
        DeepGemmQuantScaleFMT.FLOAT32_CEIL_UE8M0
        if ceil_ue8m0
        else DeepGemmQuantScaleFMT.FLOAT32
    )

    # 1-element bf16 limit tensor (matches the kernel's bf16 arithmetic).
    limit_tensor = torch.tensor([LIMIT], dtype=torch.bfloat16, device="cuda")

    y_q = torch.empty((E, T, H), dtype=torch.float8_e4m3fn, device="cuda")
    if scale_fmt == "ue8m0":
        G = H // GROUP_SIZE
        n_packed = (G + 3) // 4
        y_s_alloc = torch.zeros(
            (E, n_packed, T), dtype=torch.int32, device="cuda"
        )
        y_s = y_s_alloc.permute(0, 2, 1)  # [E, T, n_packed] (non-contiguous)
    else:
        y_s_alloc = None
        y_s = torch.empty(
            (E, T, H // GROUP_SIZE), dtype=torch.float32, device="cuda"
        )

    # Warm-up
    for _ in range(10):
        persistent_masked_m_swiglu_mul_quant(
            input_two, tokens_per_expert, limit_tensor,
            group_size=GROUP_SIZE, quant_scale_fmt=quant_scale_fmt,
        )
    torch.cuda.synchronize()

    # Timing
    n_iters = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        persistent_masked_m_swiglu_mul_quant(
            input_two, tokens_per_expert, limit_tensor,
            group_size=GROUP_SIZE, quant_scale_fmt=quant_scale_fmt,
        )
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / n_iters

    # Final fused kernel run (capture y_q, y_s)
    y_q, y_s = persistent_masked_m_swiglu_mul_quant(
        input_two, tokens_per_expert, limit_tensor,
        group_size=GROUP_SIZE, quant_scale_fmt=quant_scale_fmt,
    )
    torch.cuda.synchronize()

    # ---- reference ----
    ref_q = torch.empty((E, T, H), dtype=torch.float8_e4m3fn, device="cuda")
    ref_s_f32 = torch.empty(
        (E, T, H // GROUP_SIZE), dtype=torch.float32, device="cuda"
    )
    for e in range(E):
        nt = tokens_per_expert[e].item()
        if nt == 0:
            continue
        qe, se = ref_swiglustep_mul_quant(
            gate[e, :nt], up[e, :nt], torch.float8_e4m3fn,
            GROUP_SIZE, ceil_ue8m0, LIMIT,
        )
        ref_q[e, :nt] = qe
        ref_s_f32[e, :nt] = se

    # ---- precision checks ----
    total_tokens = tokens_per_expert.sum().item()
    # 1. Quantized output cosine similarity (per-expert, then overall).
    sim_all = []
    for e in range(E):
        nt = tokens_per_expert[e].item()
        if nt == 0:
            continue
        sim = _cosine_sim(y_q[e, :nt], ref_q[e, :nt])
        sim_all.append(sim)
    overall_sim = sum(sim_all) / max(1, len(sim_all))

    # 2. Scales comparison (fp32 path only -- UE8M0 layout differs).
    if not ceil_ue8m0:
        for e in range(E):
            nt = tokens_per_expert[e].item()
            if nt == 0:
                continue
            torch.testing.assert_close(
                y_s[e, :nt], ref_s_f32[e, :nt], rtol=1e-3, atol=1e-3
            )

    # ---- bandwidth / latency printout ----
    bw_gb = _bandwidth_gb(E, T, H, scale_fmt)
    avg_s = avg_ms / 1.0e3
    bw_gbps = bw_gb / avg_s if avg_s > 0 else 0.0
    peak, eff = _bandwidth_efficiency(bw_gbps)

    print(
        f"\n[E={E:3d} T={T:5d} H={H:5d} fp8_e4m3 {scale_fmt:5s}] "
        f"tokens={total_tokens:6d} limit={LIMIT:.1f} "
        f"avg={avg_ms:.4f} ms | bw={bw_gbps:7.2f} GB/s "
        f"(peak {peak:.0f} GB/s, eff {eff:5.1f}%) "
        f"| cosine_sim={overall_sim:.6f}"
    )

    assert overall_sim >= COS_THRESHOLD, (
        f"Precision failure: cosine similarity {overall_sim:.6f} < "
        f"{COS_THRESHOLD} "
        f"(E={E} T={T} H={H} {scale_fmt} limit={LIMIT})"
    )


if __name__ == "__main__":
    for case in CASES:
        test_persistent_masked_m_swiglustep_mul_quant(*case)
