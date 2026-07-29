import random
from typing import Tuple

import pytest
import torch
import mcoplib._C


FP8_DTYPES = (torch.float8_e4m3fn,)

# Group size is hard-coded by the kernel.
GROUP_SIZE = 128

# SwiGLU-step clamp limit.  Same magnitude as the upstream SwiGLU-OAI
# / swigluoai_and_mul default (limit=7.0) used by DeepSeek-V3 step3.5/3.7.
LIMIT = 7.0

# fp8 e4m3 min/max (matching the C++ get_fp8_min/get_fp8_max helpers).
FP8_BOUNDS = {
    torch.float8_e4m3fn: (-448.0, 448.0),
    torch.float8_e4m3fnuz: (-240.0, 240.0),
}


# ---- shapes ---------------------------------------------------------------
# Same shape set as test_peristent_masked_m_silu_mul_quant.py: small/edge
# cases plus step3.5 / step3.7-class DeepSeek shapes, fp32 and ue8m0 scale
# formats.
CASES = [
    # (E, T, H, fp8_dtype, scale_fmt_label)
    (1, 1, 128, torch.float8_e4m3fn, "fp32"),
    (1, 4, 128, torch.float8_e4m3fn, "fp32"),
    (2, 4, 256, torch.float8_e4m3fn, "fp32"),
    (1, 4, 384, torch.float8_e4m3fn, "fp32"),
    (8, 16, 512, torch.float8_e4m3fn, "fp32"),
    (8, 16, 640, torch.float8_e4m3fn, "fp32"),
    (8, 16, 768, torch.float8_e4m3fn, "fp32"),
    (8, 16, 896, torch.float8_e4m3fn, "fp32"),
    (8, 16, 1024, torch.float8_e4m3fn, "fp32"),
    (8, 16, 1152, torch.float8_e4m3fn, "fp32"),
    # step3.5 / step3.7 representative shapes.
    (8, 64, 7168, torch.float8_e4m3fn, "fp32"),
    (8, 128, 7168, torch.float8_e4m3fn, "fp32"),
    (8, 512, 7168, torch.float8_e4m3fn, "fp32"),
    (8, 1024, 7168, torch.float8_e4m3fn, "fp32"),
    (1, 4, 1280, torch.float8_e4m3fn, "fp32"),
    (17, 31, 768, torch.float8_e4m3fn, "fp32"),
    (32, 64, 256, torch.float8_e4m3fn, "fp32"),
    (256, 8, 7168, torch.float8_e4m3fn, "fp32"),
    (256, 32, 7168, torch.float8_e4m3fn, "fp32"),
    (256, 64, 7168, torch.float8_e4m3fn, "fp32"),
    # UE8M0 packed scales (int32) -- same shapes, ceil-ue8m0 path.
    (8, 16, 1024, torch.float8_e4m3fn, "ue8m0"),
    (8, 64, 7168, torch.float8_e4m3fn, "ue8m0"),
    (8, 512, 7168, torch.float8_e4m3fn, "ue8m0"),
]


def _set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def _token_random(E, T, H2, tokens_per_expert, device="cuda"):
    """Random bf16 input with a wide range of scales per token."""
    y = torch.empty((E, T, H2), dtype=torch.bfloat16, device=device)
    for e in range(E):
        nt = tokens_per_expert[e].item()
        for t in range(nt):
            exp = random.choice(range(1, 20))
            y[e, t].uniform_(-(2 ** exp), 2 ** exp)
    return y


def _fp8_max_bf16(fp8_dtype: torch.dtype) -> torch.Tensor:
    _, mx = FP8_BOUNDS[fp8_dtype]
    return torch.tensor([mx], device="cuda", dtype=torch.bfloat16)


def _fp8_min_bf16(fp8_dtype: torch.dtype) -> torch.Tensor:
    mn, _ = FP8_BOUNDS[fp8_dtype]
    return torch.tensor([mn], device="cuda", dtype=torch.bfloat16)


def silu_bf16(x_bf16: torch.Tensor) -> torch.Tensor:
    # kernel computes in bf16: x / (1 + exp(-x)) in float then back to bf16.
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
    """Reference that matches the kernel arithmetic:
        Gate_act = min(SiLU(gate), limit)        (one-sided, bf16)
        Up_act   = clip(up, -limit, limit)        (symmetric, bf16)
        Result   = Gate_act * Up_act
    then per-group absmax FP8 quantization in bfloat16 precision."""
    assert gate.dtype == torch.bfloat16
    assert up.dtype == torch.bfloat16
    assert gate.size(-1) % group_size == 0

    eps_bf16 = torch.tensor([1e-10], device=gate.device, dtype=torch.bfloat16)
    one_bf16 = torch.tensor([1.0], device=gate.device, dtype=torch.bfloat16)
    fp8_max_bf16 = _fp8_max_bf16(fp8_dtype)
    fp8_min_bf16 = _fp8_min_bf16(fp8_dtype)
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


def _pack_ue8m0_scales(scales_f32: torch.Tensor,
                       tokens_per_expert: torch.Tensor) -> torch.Tensor:
    """Pack float32 UE8M0 scales into an int32 tensor using the deep-gemm
    layout the kernel writes: T is the fast-changing dimension within each
    int32 pack, packs are placed along the slow axis.

    The kernel writes the exponent byte of group g for token t at byte
    offset:
        e * (n_packed * T * 4) + (g // 4) * (T * 4) + t * 4 + (g % 4)
    This corresponds to a contiguous int32 allocation of shape
    [E, n_packed, T] (i.e. T-contiguous), permuted to [E, T, n_packed] for
    the kernel call.

    Returns the contiguous int32 tensor of shape [E, n_packed, T] holding
    the packed UE8M0 bytes.
    """
    E, T, G = scales_f32.size()
    i32_padding = (4 - (G % 4)) % 4
    n_packed = (G + i32_padding) // 4
    # Contiguous allocation in deep-gemm byte order: [E, n_packed, T] int32.
    ref_s = torch.zeros(
        (E, n_packed, T), dtype=torch.int32, device=scales_f32.device
    )
    ref_s_u8 = ref_s.view(torch.uint8).view(E, n_packed, T, 4)
    for e in range(E):
        nt = tokens_per_expert[e].item()
        if nt == 0:
            continue
        # UE8M0 packs the (ceil'd) exponent byte of the float32 scale.
        # byte_view shape: [nt, G]
        byte_view = (scales_f32[e, :nt].view(torch.int32) >> 23).to(torch.uint8)
        for g in range(G):
            pack = g // 4
            slot = g % 4
            # ref_s_u8[e, pack, t, slot] = byte_view[t, g]
            ref_s_u8[e, pack, :nt, slot] = byte_view[:, g]
    return ref_s


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float32).reshape(-1)
    b = b.to(torch.float32).reshape(-1)
    if a.numel() == 0:
        return 1.0
    num = float((a * b).sum().item())
    den = float(torch.norm(a).item() * torch.norm(b).item()) + 1e-30
    return num / den


def _bandwidth_gb(E, T, H, fp8_dtype, scale_fmt):
    # bytes read: input bf16 (E*T*2H*2), bytes written: y_q fp8 (E*T*H*1)
    # plus scales (fp32: E*T*(H/128)*4 or uint8 packed: E*T*(H/128) bytes).
    bytes_read = E * T * 2 * H * 2
    bytes_written_q = E * T * H * 1
    if scale_fmt == "ue8m0":
        bytes_written_s = E * T * (H // GROUP_SIZE)  # uint8 / packed
    else:
        bytes_written_s = E * T * (H // GROUP_SIZE) * 4
    return (bytes_read + bytes_written_q + bytes_written_s) / 1.0e9


def _bandwidth_efficiency(gbps: float) -> Tuple[float, float]:
    # Metax C500 (C280-class) peak HBM2e bandwidth ~1.55 TB/s.
    peak = 1.55e3  # GB/s
    eff = (gbps / peak) * 100.0
    return peak, eff


@pytest.mark.parametrize("E,T,H,fp8_dtype,scale_fmt", CASES)
@torch.inference_mode()
def test_persistent_masked_m_swiglustep_mul_quant(E, T, H, fp8_dtype, scale_fmt):
    _set_seed(42)

    tokens_per_expert = torch.randint(
        low=0, high=max(1, T), size=(E,), dtype=torch.int32, device="cuda"
    )
    # Make sure at least one expert has work, else the kernel returns early
    # for every block and there's nothing to validate.
    if tokens_per_expert.sum().item() == 0:
        tokens_per_expert[0] = min(T, 4)

    y = _token_random(E, T, 2 * H, tokens_per_expert)
    gate = y[..., :H].contiguous()
    up = y[..., H:].contiguous()
    input_two = torch.cat([gate, up], dim=-1)  # [E, T, 2H]

    ceil_ue8m0 = scale_fmt == "ue8m0"
    use_ue8m0 = scale_fmt == "ue8m0"

    # Scalar limit tensor (bf16) -- matches the kernel's bf16 arithmetic.
    limit_tensor = torch.tensor([LIMIT], dtype=torch.bfloat16, device="cuda")

    if use_ue8m0:
        # packed int32 scales tensor: the kernel uses a deep-gemm byte layout
        # where T is the fast-changing dim within each int32 pack and packs
        # are along the slow axis.  We allocate contiguous [E, n_packed, T]
        # (T-contiguous) and pass a permuted [E, T, n_packed] view to the
        # kernel.  After the kernel runs we extract bytes from the original
        # contiguous allocation to compare against the reference.
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
    y_q = torch.empty((E, T, H), dtype=fp8_dtype, device="cuda")

    # warm-up
    for _ in range(10):
        torch.ops._C.persistent_masked_m_swiglu_mul_quant(
            input_two, tokens_per_expert, y_q, y_s, limit_tensor, use_ue8m0
        )
    torch.cuda.synchronize()

    # timing
    n_iters = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        torch.ops._C.persistent_masked_m_swiglu_mul_quant(
            input_two, tokens_per_expert, y_q, y_s, limit_tensor, use_ue8m0
        )
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / n_iters

    # ---- reference ----
    ref_q = torch.empty((E, T, H), dtype=fp8_dtype, device="cuda")
    ref_s_f32 = torch.empty(
        (E, T, H // GROUP_SIZE), dtype=torch.float32, device="cuda"
    )
    for e in range(E):
        nt = tokens_per_expert[e].item()
        if nt == 0:
            continue
        qe, se = ref_swiglustep_mul_quant(
            gate[e, :nt], up[e, :nt], fp8_dtype, GROUP_SIZE, ceil_ue8m0, LIMIT
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

    # 2. Scales comparison.
    if use_ue8m0:
        G = H // GROUP_SIZE
        # Reference packed tensor in the kernel's deep-gemm byte layout.
        ref_s_alloc = _pack_ue8m0_scales(ref_s_f32, tokens_per_expert)
        # Both y_s_alloc and ref_s_alloc are [E, n_packed, T] contiguous int32;
        # view them as uint8 [E, n_packed, T, 4] and compare only valid tokens.
        y_s_u8 = y_s_alloc.view(torch.uint8).view(E, -1, T, 4)
        ref_s_u8 = ref_s_alloc.view(torch.uint8).view(E, -1, T, 4)
        n_packed_actual = y_s_u8.size(1)
        for e in range(E):
            nt = tokens_per_expert[e].item()
            if nt == 0:
                continue
            for pack in range(n_packed_actual):
                for slot in range(4):
                    g = pack * 4 + slot
                    if g >= G:
                        continue
                    got = y_s_u8[e, pack, :nt, slot]
                    exp = ref_s_u8[e, pack, :nt, slot]
                    assert torch.equal(got, exp), (
                        f"UE8M0 scale mismatch E={e} G={g} "
                        f"got={got.tolist()} exp={exp.tolist()}"
                    )
    else:
        for e in range(E):
            nt = tokens_per_expert[e].item()
            if nt == 0:
                continue
            torch.testing.assert_close(
                y_s[e, :nt], ref_s_f32[e, :nt], rtol=1e-3, atol=1e-3
            )

    # ---- bandwidth / latency printout ----
    bw_gb = _bandwidth_gb(E, T, H, fp8_dtype, scale_fmt)
    avg_s = avg_ms / 1.0e3
    bw_gbps = bw_gb / avg_s if avg_s > 0 else 0.0
    peak, eff = _bandwidth_efficiency(bw_gbps)

    print(
        f"\n[E={E:3d} T={T:5d} H={H:5d} {fp8_dtype} {scale_fmt:5s}] "
        f"tokens={total_tokens:6d} limit={LIMIT:.1f} "
        f"avg={avg_ms:.4f} ms | bw={bw_gbps:7.2f} GB/s "
        f"(peak {peak:.0f} GB/s, eff {eff:5.1f}%) "
        f"| cosine_sim={overall_sim:.6f}"
    )

    assert overall_sim >= 0.999, (
        f"Precision failure: cosine similarity {overall_sim:.6f} < 0.999 "
        f"(E={E} T={T} H={H} {fp8_dtype} {scale_fmt} limit={LIMIT})"
    )
