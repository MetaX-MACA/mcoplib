# SPDX-License-Identifier: Apache-2.0
"""Reference checks for Step5 fused_qknorm_rope.

`fused_qknorm_rope` is the current MetaX Triton kernel. `formula_fused_qknorm_rope`
is the formula in README.md. The two must match element for element.

Run on a CUDA machine with Triton:

    python test_fused_qknorm_rope.py

When the mcoplib op is ready, point `IMPL` at it and run the same checks.
"""

from __future__ import annotations

import math
import sys

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


if triton is not None:

    @triton.jit
    def _qknorm_rope_kernel(
        qkv_ptr,
        q_out_ptr,
        k_out_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        positions_ptr,
        stride_qkv_token,
        stride_q_out_token,
        stride_k_out_token,
        stride_cos,
        stride_sin,
        NUM_Q_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        EPS: tl.constexpr,
        NORM_WEIGHT_BIAS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ) -> None:
        token = tl.program_id(0)
        packed_head = tl.program_id(1)
        is_q = packed_head < NUM_Q_HEADS
        head_base = token * stride_qkv_token + packed_head * HEAD_DIM

        dims = tl.arange(0, BLOCK_D)
        dim_ok = dims < HEAD_DIM
        values = tl.load(qkv_ptr + head_base + dims, mask=dim_ok, other=0.0).to(
            tl.float32
        )
        sum_sq = tl.sum(values * values)
        inverse_rms = tl.rsqrt(sum_sq / HEAD_DIM + EPS)

        q_weight = tl.load(q_weight_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
        k_weight = tl.load(k_weight_ptr + dims, mask=dim_ok, other=0.0).to(tl.float32)
        weight = tl.where(is_q, q_weight, k_weight) + NORM_WEIGHT_BIAS
        # Round to the activation dtype before RoPE.
        normalized = (values * inverse_rms * weight).to(q_out_ptr.dtype.element_ty)

        q_base = token * stride_q_out_token + packed_head * HEAD_DIM
        k_base = token * stride_k_out_token + (packed_head - NUM_Q_HEADS) * HEAD_DIM
        tl.store(
            q_out_ptr + q_base + dims,
            normalized,
            mask=dim_ok & is_q & (dims >= 2 * ROTARY_DIM),
        )
        tl.store(
            k_out_ptr + k_base + dims,
            normalized,
            mask=dim_ok & ~is_q & (dims >= 2 * ROTARY_DIM),
        )

        if ROTARY_DIM > 0:
            pairs = tl.arange(0, BLOCK_R)
            pair_ok = pairs < ROTARY_DIM
            raw0 = tl.load(qkv_ptr + head_base + pairs, mask=pair_ok, other=0.0).to(
                tl.float32
            )
            raw1 = tl.load(
                qkv_ptr + head_base + ROTARY_DIM + pairs, mask=pair_ok, other=0.0
            ).to(tl.float32)
            qw0 = tl.load(q_weight_ptr + pairs, mask=pair_ok, other=0.0).to(tl.float32)
            qw1 = tl.load(
                q_weight_ptr + ROTARY_DIM + pairs, mask=pair_ok, other=0.0
            ).to(tl.float32)
            kw0 = tl.load(k_weight_ptr + pairs, mask=pair_ok, other=0.0).to(tl.float32)
            kw1 = tl.load(
                k_weight_ptr + ROTARY_DIM + pairs, mask=pair_ok, other=0.0
            ).to(tl.float32)
            w0 = tl.where(is_q, qw0, kw0) + NORM_WEIGHT_BIAS
            w1 = tl.where(is_q, qw1, kw1) + NORM_WEIGHT_BIAS
            value0 = (raw0 * inverse_rms * w0).to(q_out_ptr.dtype.element_ty).to(
                tl.float32
            )
            value1 = (raw1 * inverse_rms * w1).to(q_out_ptr.dtype.element_ty).to(
                tl.float32
            )

            position = tl.load(positions_ptr + token).to(tl.int64)
            cos_value = tl.load(
                cos_ptr + position * stride_cos + pairs, mask=pair_ok, other=0.0
            ).to(tl.float32)
            sin_value = tl.load(
                sin_ptr + position * stride_sin + pairs, mask=pair_ok, other=0.0
            ).to(tl.float32)
            rotated0 = (value0 * cos_value - value1 * sin_value).to(
                q_out_ptr.dtype.element_ty
            )
            rotated1 = (value0 * sin_value + value1 * cos_value).to(
                q_out_ptr.dtype.element_ty
            )
            tl.store(q_out_ptr + q_base + pairs, rotated0, mask=pair_ok & is_q)
            tl.store(
                q_out_ptr + q_base + ROTARY_DIM + pairs,
                rotated1,
                mask=pair_ok & is_q,
            )
            tl.store(k_out_ptr + k_base + pairs, rotated0, mask=pair_ok & ~is_q)
            tl.store(
                k_out_ptr + k_base + ROTARY_DIM + pairs,
                rotated1,
                mask=pair_ok & ~is_q,
            )


def _prepare_outs(qkv, q_width, kv_width, q_out, k_out, v_out):
    outs = (q_out, k_out, v_out)
    if any(t is None for t in outs) and any(t is not None for t in outs):
        raise ValueError("q_out, k_out, v_out must be all given or all omitted")
    tokens = qkv.shape[0]
    if q_out is None:
        q_out = torch.empty((tokens, q_width), device=qkv.device, dtype=qkv.dtype)
        k_out = torch.empty((tokens, kv_width), device=qkv.device, dtype=qkv.dtype)
        v_out = torch.empty((tokens, kv_width), device=qkv.device, dtype=qkv.dtype)
        return q_out, k_out, v_out, False
    for name, tensor, width in (
        ("q_out", q_out, q_width),
        ("k_out", k_out, kv_width),
        ("v_out", v_out, kv_width),
    ):
        if tensor.shape != (tokens, width):
            raise ValueError(f"{name} must have shape {(tokens, width)}, got {tuple(tensor.shape)}")
        if tensor.dtype != qkv.dtype or tensor.device != qkv.device:
            raise ValueError(f"{name} dtype and device must match qkv")
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous in the last dimension")
    return q_out, k_out, v_out, True


def _rms_then_rope(x, weight, cos, sin, positions, rotary_pairs, eps, bias, dtype):
    """x: [tokens, heads, head_dim] float32. Returns bf16/fp16 [tokens, heads, head_dim]."""
    scale = weight.reshape(-1).to(torch.float32) + bias
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = (x * inv_rms * scale).to(dtype)
    if rotary_pairs == 0:
        return y
    pos = positions.reshape(-1).to(torch.int64)
    cos_sel = cos.index_select(0, pos).to(torch.float32).unsqueeze(1)
    sin_sel = sin.index_select(0, pos).to(torch.float32).unsqueeze(1)
    y32 = y.to(torch.float32)
    y0 = y32[..., :rotary_pairs]
    y1 = y32[..., rotary_pairs : 2 * rotary_pairs]
    tail = y[..., 2 * rotary_pairs :]
    rot0 = (y0 * cos_sel - y1 * sin_sel).to(dtype)
    rot1 = (y0 * sin_sel + y1 * cos_sel).to(dtype)
    return torch.cat((rot0, rot1, tail), dim=-1)


def formula_fused_qknorm_rope(
    qkv,
    q_weight,
    k_weight,
    cos,
    sin,
    positions,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_pairs,
    eps=1e-5,
    norm_weight_bias=1.0,
    q_out=None,
    k_out=None,
    v_out=None,
):
    """README formula. Norm is rounded to the activation dtype before RoPE."""
    tokens = qkv.shape[0]
    q_width = num_q_heads * head_dim
    kv_width = num_kv_heads * head_dim
    q_out, k_out, v_out, _ = _prepare_outs(qkv, q_width, kv_width, q_out, k_out, v_out)
    if tokens == 0:
        return q_out, k_out, v_out

    q = qkv[:, :q_width].reshape(tokens, num_q_heads, head_dim).to(torch.float32)
    k = qkv[:, q_width : q_width + kv_width].reshape(tokens, num_kv_heads, head_dim).to(
        torch.float32
    )
    v = qkv[:, q_width + kv_width : q_width + 2 * kv_width]
    q_out.copy_(
        _rms_then_rope(
            q, q_weight, cos, sin, positions, rotary_pairs, eps, norm_weight_bias, qkv.dtype
        ).reshape(tokens, q_width)
    )
    k_out.copy_(
        _rms_then_rope(
            k, k_weight, cos, sin, positions, rotary_pairs, eps, norm_weight_bias, qkv.dtype
        ).reshape(tokens, kv_width)
    )
    v_out.copy_(v)
    return q_out, k_out, v_out


def fused_qknorm_rope(
    qkv,
    q_weight,
    k_weight,
    cos,
    sin,
    positions,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_pairs,
    eps=1e-5,
    norm_weight_bias=1.0,
    q_out=None,
    k_out=None,
    v_out=None,
):
    """Current MetaX Triton. Writes q_out/k_out/v_out when they are provided."""
    if triton is None or not qkv.is_cuda:
        raise RuntimeError("fused_qknorm_rope reference requires Triton on CUDA")
    if rotary_pairs < 0 or 2 * rotary_pairs > head_dim:
        raise ValueError(
            f"rotary_pairs must satisfy 0 <= 2*rotary_pairs <= head_dim, got "
            f"rotary_pairs={rotary_pairs}, head_dim={head_dim}"
        )
    packed_width = (num_q_heads + 2 * num_kv_heads) * head_dim
    if qkv.ndim != 2 or qkv.shape[-1] < packed_width or qkv.stride(-1) != 1:
        raise ValueError(
            f"qkv must be [tokens, >= {packed_width}] and contiguous in the last dim, "
            f"got {tuple(qkv.shape)} stride={qkv.stride()}"
        )

    tokens = qkv.shape[0]
    q_width = num_q_heads * head_dim
    kv_width = num_kv_heads * head_dim
    q_out, k_out, v_out, _ = _prepare_outs(qkv, q_width, kv_width, q_out, k_out, v_out)
    if tokens == 0:
        return q_out, k_out, v_out

    v_out.copy_(qkv.narrow(-1, q_width + kv_width, kv_width))
    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()
    positions = positions.reshape(-1).contiguous()
    _qknorm_rope_kernel[(tokens, num_q_heads + num_kv_heads)](
        qkv,
        q_out,
        k_out,
        q_weight,
        k_weight,
        cos,
        sin,
        positions,
        qkv.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        cos.stride(0),
        sin.stride(0),
        NUM_Q_HEADS=num_q_heads,
        HEAD_DIM=head_dim,
        ROTARY_DIM=rotary_pairs,
        EPS=float(eps),
        NORM_WEIGHT_BIAS=float(norm_weight_bias),
        BLOCK_D=_next_power_of_2(head_dim),
        BLOCK_R=_next_power_of_2(max(rotary_pairs, 1)),
        num_warps=4,
    )
    return q_out, k_out, v_out


# Swap this for the mcoplib op when it lands.
IMPL = fused_qknorm_rope

# 64 Q heads, 4 KV groups. TP=8 replicates the KV head (4 < 8).
# TP=4 shards it evenly (4 / 4 = 1). Sliding rotates 96 pairs; DSA rotates 32.
# bias 1.0 is zero_centered. vLLM positions are int64.
_DECODE_TOKENS = [1, 2, 4, 8, 16, 32, 64]  # concurrent sequences, up to max_num_seqs
_MTP_SPEC_TOKENS = 3  # each decode seq is scheduled as 1 + 3 tokens

# tp, local q heads, local kv heads, chunked-prefill tokens
_SPLITS = (
    (8, 8, 1, 8192),
    (4, 16, 1, 4096),
)
# scenario, rotary_pairs, preallocate outputs
_PREFILL_SCENARIOS = (
    ("prefill", 96, False),
    ("prefill", 32, True),
)
_DECODE_SCENARIOS = (
    ("decode", 96, False),
    ("decode", 32, True),
    ("decode_mtp3", 96, False),
    ("decode_mtp3", 32, True),
)


def _scenario_positions(scenario, prefill_tokens, num_seqs):
    if scenario == "prefill":
        # One chunk: a long prompt already past the first chunk, then three
        # new prompts packed into the rest of the chunk.
        half = prefill_tokens // 2
        quarter = prefill_tokens // 4
        eighth = prefill_tokens // 8
        parts = (
            torch.arange(prefill_tokens, prefill_tokens + half),
            torch.arange(0, quarter),
            torch.arange(0, eighth),
            torch.arange(0, eighth),
        )
        return torch.cat(parts)
    if scenario == "decode":
        # One new token per sequence, at different context lengths.
        pos = torch.arange(num_seqs) * (65536 // num_seqs)
        pos[-1] = 65535
        return pos
    if scenario == "decode_mtp3":
        # MTP3 pads every decode sequence to 1 + 3 consecutive positions.
        bases = torch.arange(num_seqs) * (65536 // num_seqs)
        bases[-1] = 65535 - _MTP_SPEC_TOKENS
        offsets = torch.arange(1 + _MTP_SPEC_TOKENS)
        return (bases[:, None] + offsets).reshape(-1)
    raise ValueError(f"unknown scenario {scenario}")


def _scenario_tokens(scenario, prefill_tokens, num_seqs):
    if scenario == "prefill":
        return prefill_tokens
    if scenario == "decode":
        return num_seqs
    if scenario == "decode_mtp3":
        return num_seqs * (1 + _MTP_SPEC_TOKENS)
    raise ValueError(f"unknown scenario {scenario}")


def _make_inputs(
    scenario, rotary_pairs, device, num_q_heads, num_kv_heads, prefill_tokens, num_seqs
):
    tokens = _scenario_tokens(scenario, prefill_tokens, num_seqs)
    head_dim = 192
    packed = (num_q_heads + 2 * num_kv_heads) * head_dim
    g = torch.Generator(device="cpu")
    g.manual_seed(tokens * 1000 + num_q_heads * 10 + rotary_pairs)
    qkv = torch.randn(tokens, packed, dtype=torch.float32, generator=g).to(torch.bfloat16).to(device)
    q_weight = torch.randn(head_dim, dtype=torch.float32, generator=g).to(device)
    k_weight = torch.randn(head_dim, dtype=torch.float32, generator=g).to(device)
    positions = _scenario_positions(scenario, prefill_tokens, num_seqs).to(torch.int64)
    max_pos = int(positions.max().item()) + 1
    angles = torch.rand(max_pos, rotary_pairs, dtype=torch.float32, generator=g) * (
        2 * math.pi
    )
    cos = torch.cos(angles).to(torch.bfloat16).to(device)
    sin = torch.sin(angles).to(torch.bfloat16).to(device)
    return qkv, q_weight, k_weight, cos, sin, positions.to(device)


def _run_case(
    tp, num_q_heads, num_kv_heads, prefill_tokens, scenario, rotary_pairs, prealloc, num_seqs
):
    device = "cuda"
    head_dim = 192
    qkv, q_weight, k_weight, cos, sin, positions = _make_inputs(
        scenario,
        rotary_pairs,
        device,
        num_q_heads,
        num_kv_heads,
        prefill_tokens,
        num_seqs,
    )
    tokens = qkv.shape[0]
    qkv_before = qkv.clone()
    kwargs = dict(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_pairs=rotary_pairs,
        eps=1e-5,
        norm_weight_bias=1.0,
    )
    outs = {}
    if prealloc:
        q_width = num_q_heads * head_dim
        kv_width = num_kv_heads * head_dim
        outs = dict(
            q_out=torch.full((tokens, q_width), torch.nan, device=device, dtype=torch.bfloat16),
            k_out=torch.full((tokens, kv_width), torch.nan, device=device, dtype=torch.bfloat16),
            v_out=torch.full((tokens, kv_width), torch.nan, device=device, dtype=torch.bfloat16),
        )
    got = IMPL(qkv, q_weight, k_weight, cos, sin, positions, **kwargs, **outs)
    ref = formula_fused_qknorm_rope(
        qkv, q_weight, k_weight, cos, sin, positions, **kwargs
    )
    q, k, v = got
    if prealloc:
        assert q is outs["q_out"] and k is outs["k_out"] and v is outs["v_out"]
    assert torch.equal(qkv, qkv_before), "qkv was mutated"
    label = f"tp{tp} {scenario} seqs={num_seqs} q"
    _assert_within_bf16_ulp(q, ref[0], label, tokens, rotary_pairs)
    _assert_within_bf16_ulp(
        k, ref[1], f"tp{tp} {scenario} seqs={num_seqs} k", tokens, rotary_pairs
    )
    v_src = qkv[:, num_q_heads * head_dim + num_kv_heads * head_dim :].contiguous()
    assert torch.equal(v, v_src), "V is not a bit-exact copy"
    assert torch.equal(v, ref[2])


# 4 bf16 ulps at max(|actual|, |expected|, 1). A 1-ulp difference in the
# rounded norm is multiplied by cos/sin; on a cancelled component the
# relative error is large while the absolute error stays about one ulp of
# the pre-rope value.
_BF16_ULP = 2**-7


def _assert_within_bf16_ulp(actual, expected, name, tokens, rotary_pairs):
    if torch.equal(actual, expected):
        return
    a = actual.float()
    b = expected.float()
    diff = (a - b).abs()
    tol = torch.maximum(a.abs(), b.abs()).clamp_min(1.0) * (4 * _BF16_ULP)
    if bool(torch.all(diff <= tol).item()):
        return
    bad = diff > tol
    nbad = int(bad.sum().item())
    max_abs = diff.max().item()
    shown = bad.view(-1).nonzero().flatten()[:6].tolist()
    flat_a = a.view(-1)
    flat_b = b.view(-1)
    samples = ", ".join(
        f"{i}:{flat_a[i].item():.4g} vs {flat_b[i].item():.4g}" for i in shown
    )
    raise AssertionError(
        f"{name} mismatch tokens={tokens} rotary_pairs={rotary_pairs} "
        f"beyond_tol={nbad}/{actual.numel()} max_abs={max_abs:.3e} samples=[{samples}]"
    )


def test_matches_formula():
    for tp, num_q_heads, num_kv_heads, prefill_tokens in _SPLITS:
        for scenario, rotary_pairs, prealloc in _PREFILL_SCENARIOS:
            _run_case(
                tp,
                num_q_heads,
                num_kv_heads,
                prefill_tokens,
                scenario,
                rotary_pairs,
                prealloc,
                num_seqs=0,
            )
        for num_seqs in _DECODE_TOKENS:
            for scenario, rotary_pairs, prealloc in _DECODE_SCENARIOS:
                _run_case(
                    tp,
                    num_q_heads,
                    num_kv_heads,
                    prefill_tokens,
                    scenario,
                    rotary_pairs,
                    prealloc,
                    num_seqs,
                )


def test_empty_tokens():
    device = "cuda"
    head_dim, num_q_heads, num_kv_heads, rotary_pairs = 192, 8, 1, 96
    packed = (num_q_heads + 2 * num_kv_heads) * head_dim
    qkv = torch.empty(0, packed, device=device, dtype=torch.bfloat16)
    q_weight = torch.empty(head_dim, device=device, dtype=torch.float32)
    k_weight = torch.empty(head_dim, device=device, dtype=torch.float32)
    cos = torch.empty(4, rotary_pairs, device=device, dtype=torch.bfloat16)
    sin = torch.empty(4, rotary_pairs, device=device, dtype=torch.bfloat16)
    positions = torch.empty(0, device=device, dtype=torch.int64)
    q, k, v = IMPL(
        qkv, q_weight, k_weight, cos, sin, positions,
        num_q_heads, num_kv_heads, head_dim, rotary_pairs, 1e-5, 1.0,
    )
    assert q.shape == (0, num_q_heads * head_dim)
    assert k.shape == (0, num_kv_heads * head_dim)
    assert v.shape == (0, num_kv_heads * head_dim)


def test_bf16_round_trip_changes_result():
    """RoPE must see the bf16-rounded norm, not the fp32 norm."""
    device = "cuda"
    rotary_pairs = 32
    num_q_heads = 8
    num_kv_heads = 1
    qkv, q_weight, k_weight, cos, sin, positions = _make_inputs(
        "decode", rotary_pairs, device, num_q_heads, num_kv_heads, 8192, num_seqs=4
    )
    tokens = qkv.shape[0]
    rounded = formula_fused_qknorm_rope(
        qkv, q_weight, k_weight, cos, sin, positions,
        num_q_heads, num_kv_heads, 192, rotary_pairs, 1e-5, 1.0,
    )[0]
    q_width = num_q_heads * 192
    q = qkv[:, :q_width].reshape(tokens, num_q_heads, 192).float()
    scale = q_weight.float() + 1.0
    inv_rms = torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-5)
    y32 = q * inv_rms * scale
    pos = positions.to(torch.int64)
    cos_sel = cos.index_select(0, pos).float().unsqueeze(1)
    sin_sel = sin.index_select(0, pos).float().unsqueeze(1)
    y0 = y32[..., :rotary_pairs]
    y1 = y32[..., rotary_pairs : 2 * rotary_pairs]
    no_round = torch.cat(
        (
            (y0 * cos_sel - y1 * sin_sel).to(torch.bfloat16),
            (y0 * sin_sel + y1 * cos_sel).to(torch.bfloat16),
            y32[..., 2 * rotary_pairs :].to(torch.bfloat16),
        ),
        dim=-1,
    ).reshape(tokens, q_width)
    assert not torch.equal(rounded, no_round)


def _cuda_ready() -> bool:
    return triton is not None and torch.cuda.is_available()


def main() -> int:
    if not _cuda_ready():
        missing = []
        if triton is None:
            missing.append("Triton")
        if not torch.cuda.is_available():
            missing.append("CUDA")
        print("SKIP: reference test needs " + " and ".join(missing))
        return 0
    test_empty_tokens()
    test_matches_formula()
    test_bf16_round_trip_changes_result()
    n = len(_SPLITS) * (
        len(_PREFILL_SCENARIOS) + len(_DECODE_TOKENS) * len(_DECODE_SCENARIOS)
    )
    print(f"ok: {n} shapes matched the formula")
    return 0


if __name__ == "__main__":
    sys.exit(main())
