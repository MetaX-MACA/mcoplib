from __future__ import annotations

import math
import os
import sys
import zlib

from mcoplib.unit_test.test_op_moe_quant_group_gemm_combine import HIDDEN
import torch
import triton
import triton.language as tl

import mcoplib._C  # noqa: F401

COS_TH = float(os.environ.get("COS_TH", "0.99999"))

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
DTYPES = [d.strip() for d in os.environ.get("DTYPES", "bf16,fp16").split(",")
          if d.strip() in _DTYPE_MAP]
if not DTYPES:
    raise SystemExit(f"DTYPES env has no valid entry; valid: {','.join(_DTYPE_MAP)}")

TOKENS = [1, 2, 4, 8, 16, 32, 64, 85, 128, 256, 512, 1024, 2048, 4096, 8192,
          16384, 32768, 65536]
# d set: alignment corners + MoE/dense widths + TP shards; model provenance in
# docs/silu_and_mul_clamp_analysis.md §6.
HIDDEN = [96, 257, 512, 608, 688, 768, 864, 896, 1184, 1408, 1536, 1537, 1792,
          2048, 2368, 2880, 3072, 3584, 4096, 5120, 5632, 6144, 7168, 8192,
          9216, 9728]
SHAPES = [(t, d) for d in HIDDEN for t in TOKENS]

PROD_LIMIT, PROD_ALPHA, PROD_BETA = 7.0, 1.0, 0.0
LIMITS = (0.0, 1.0, 7.0, 10.0)
ALPHAS = (0.5, 1.0, 1.702)
BETAS = (0.0, 1.0)

WARMUP = 3
REP = 100


@triton.jit
def _step4_kernel(gate_ptr, up_ptr, out_ptr, n, limit, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0).to(tl.float32)
    gate = tl.minimum(gate * tl.sigmoid(gate), limit)
    up = tl.maximum(tl.minimum(up, limit), -limit)
    tl.store(out_ptr + offsets, (gate * up).to(out_ptr.dtype.element_ty), mask=mask)


def _validate(gate, up, limit):
    if gate.shape != up.shape or up.dtype != gate.dtype or gate.dtype not in _DTYPE_MAP.values():
        raise ValueError(f"gate/up must be equal-shape bf16/fp16 tensors, got {gate.dtype}")
    if not gate.is_cuda or not up.is_cuda or not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError("gate and up must be contiguous CUDA tensors")
    value = float(limit)
    if not math.isfinite(value) or value < 0:
        raise ValueError("limit must be finite and non-negative")
    return value


def _torch_ref(gate, up, limit, alpha=1.0, beta=0.0, step4=False):
    limit = _validate(gate, up, limit)
    gate_f, up_f = gate.float(), up.float()
    if step4:
        gate_f = (gate_f / (1.0 + torch.exp(-gate_f * alpha))).clamp(max=limit)
        up_f = up_f.clamp(min=-limit, max=limit)
    else:
        gate_f = gate_f.clamp(max=limit)
        up_f = up_f.clamp(min=-limit, max=limit)
        gate_f = gate_f / (1.0 + torch.exp(-gate_f * alpha))
    # fp32 reference, not quantized back to input dtype (same as bench _ref)
    return gate_f * (up_f + beta)


def _torch_impl(gate, up, limit, alpha, beta, step4):
    g, u = gate, up
    if step4:
        g = (g / (1.0 + torch.exp(-g * alpha))).clamp(max=limit)
        u = u.clamp(min=-limit, max=limit)
    else:
        g = g.clamp(max=limit)
        u = u.clamp(min=-limit, max=limit)
        g = g / (1.0 + torch.exp(-g * alpha))
    return g * (u + beta)


def _triton_ref(gate, up, limit):
    limit = _validate(gate, up, limit)
    out = torch.empty_like(gate)
    if gate.numel():
        _step4_kernel[(triton.cdiv(gate.numel(), 1024),)](
            gate, up, out, gate.numel(), limit,
            BLOCK=1024, num_warps=4
        )
    return out


def _mcop(gate, up, limit, alpha, beta, step4):
    _validate(gate, up, limit)
    packed = torch.cat((gate, up), dim=-1)
    out = torch.empty_like(gate)
    torch.ops._C.silu_and_mul_with_clamp(out, packed, limit, alpha, beta, step4)
    return out


def _cosine(actual, expected):
    # fp32 dot / (|a||b|), no eps clamp; all-zero pair = 1.0 (matches bench _cos)
    a, b = actual.float().flatten(), expected.float().flatten()
    denom = float(a.norm() * b.norm())
    return 1.0 if denom == 0.0 else float((a @ b) / denom)


def _max_abs(actual, expected):
    if not actual.numel():
        return 0.0
    return float((actual.float() - expected.float()).abs().max())


def _seed(shape, limit, alpha, beta, step4, dtype_name):
    key = repr((shape, limit, alpha, beta, bool(step4), dtype_name)).encode()
    return 20260903 + zlib.crc32(key)


def _mean_us(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(REP)]
    for s, e in ev:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in ev) / len(ev) * 1e3


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("tests require a CUDA/MACA device")

    failures, tri_failures = [], []
    tri_checks, tri_min = 0, 1.0
    param_agg = {}   # (limit, alpha, beta) -> [leg_min_cos, s4_min_cos, leg_max_abs, s4_max_abs]
    cases = 0

    def _fmt(c):
        return f"{c:.6f}{' !' if not (c >= COS_TH) else ''}"

    width = 150
    for dtype_name in DTYPES:
        dtype = _DTYPE_MAP[dtype_name]

        for (T, d) in SHAPES:
            for limit in LIMITS:
                for alpha in ALPHAS:
                    for beta in BETAS:
                        for step4 in (False, True):
                            torch.manual_seed(
                                _seed((T, d), limit, alpha, beta, step4, dtype_name))
                            gate = torch.randn((T, d), device="cuda", dtype=dtype)
                            up = torch.randn_like(gate)
                            expected = _torch_ref(gate, up, limit, alpha, beta, step4)
                            actual = _mcop(gate, up, limit, alpha, beta, step4)
                            cos = _cosine(actual, expected)
                            mabs = _max_abs(actual, expected)
                            desc = (f"{dtype_name} ({T},{d}) limit={limit} "
                                    f"alpha={alpha} beta={beta} step4={step4}")
                            if not (cos >= COS_TH):  # NaN-sensitive
                                failures.append((f"mcop {desc}", cos))
                            if step4 and alpha == 1.0 and beta == 0.0:
                                tsim = _cosine(_triton_ref(gate, up, limit), expected)
                                tri_checks += 1
                                if tsim == tsim:
                                    tri_min = min(tri_min, tsim)
                                if not (tsim >= COS_TH):
                                    tri_failures.append((f"triton {desc}", tsim))
                            i = 0 if not step4 else 1
                            pagg = param_agg.setdefault((limit, alpha, beta),
                                                        [2.0, 2.0, 0.0, 0.0])
                            pagg[i] = min(pagg[i], cos)
                            pagg[2 + i] = max(pagg[2 + i], mabs)
                            cases += 1

        rows = []
        for (T, d) in SHAPES:
            torch.manual_seed(
                _seed((T, d), PROD_LIMIT, PROD_ALPHA, PROD_BETA, False, dtype_name))
            gate = torch.randn((T, d), device="cuda", dtype=dtype)
            up = torch.randn_like(gate)
            exp_leg = _torch_ref(gate, up, PROD_LIMIT, PROD_ALPHA, PROD_BETA, False)
            exp_s4 = _torch_ref(gate, up, PROD_LIMIT, PROD_ALPHA, PROD_BETA, True)
            legacy = _mcop(gate, up, PROD_LIMIT, PROD_ALPHA, PROD_BETA, False)
            s4 = _mcop(gate, up, PROD_LIMIT, PROD_ALPHA, PROD_BETA, True)
            tri = _triton_ref(gate, up, PROD_LIMIT)
            cos_leg, cos_s4 = _cosine(legacy, exp_leg), _cosine(s4, exp_s4)
            cos_tri = _cosine(tri, exp_s4)
            mabs = max(_max_abs(legacy, exp_leg), _max_abs(s4, exp_s4))

            packed = torch.cat((gate, up), dim=-1)
            out = torch.empty_like(gate)
            tri_out = torch.empty_like(gate)
            t_leg = _mean_us(lambda: torch.ops._C.silu_and_mul_with_clamp(
                out, packed, PROD_LIMIT, PROD_ALPHA, PROD_BETA, False))
            t_s4 = _mean_us(lambda: torch.ops._C.silu_and_mul_with_clamp(
                out, packed, PROD_LIMIT, PROD_ALPHA, PROD_BETA, True))

            def _tri():
                _step4_kernel[(triton.cdiv(gate.numel(), 1024),)](
                    gate, up, tri_out, gate.numel(), PROD_LIMIT,
                    BLOCK=1024, num_warps=4)

            t_tri = _mean_us(_tri)
            t_torch = _mean_us(lambda: _torch_impl(
                gate, up, PROD_LIMIT, PROD_ALPHA, PROD_BETA, True))
            vs = t_tri / t_s4 if t_s4 > 0 else float("inf")
            gbps = 3 * T * d * gate.element_size() / t_s4 / 1e3
            rows.append(f"{f'd{d}':<24}{T:>9}{t_leg:>12.3f}{t_s4:>12.3f}{t_tri:>12.3f}"
                        f"{t_torch:>12.3f}{vs:>11.3f}{gbps:>11.1f}"
                        f"{_fmt(cos_leg):>11}{_fmt(cos_s4):>11}{_fmt(cos_tri):>11}"
                        f"{mabs:>10.1e}")

        print("=" * width)
        print(f"silu_and_mul_with_clamp  dtype={dtype_name}  cos_th={COS_TH}  "
              f"vs_triton=triton_us/step4_us (single cfg, not raced-best)")
        print("=" * width)
        print(f"{'case':<24}{'n':>9}{'legacy_us':>12}{'step4_us':>12}"
              f"{'triton_us':>12}{'torch_us':>12}{'vs_triton':>11}"
              f"{'step4_GBps':>11}{'cos_leg':>11}{'cos_s4':>11}{'cos_tri':>11}"
              f"{'max_abs':>10}")
        print("-" * width)
        for r in rows:
            print(r)
        print("-" * width)
        print()

    print(f"param matrix over all (T,d)  (min/max over dtypes, both modes)")
    print(f"{'limit':>6}{'alpha':>7}{'beta':>6}"
          f"{'cos_leg_min':>13}{'cos_s4_min':>13}{'abs_leg_max':>13}{'abs_s4_max':>13}")
    print("-" * 70)
    for (limit, alpha, beta), a in sorted(param_agg.items()):
        print(f"{limit:>6}{alpha:>7}{beta:>6}"
              f"{_fmt(a[0]):>13}{_fmt(a[1]):>13}{a[2]:>13.1e}{a[3]:>13.1e}")
    print("-" * 70)
    print(f"' !' flags < COS_TH or NaN; triton cross-checks: {tri_checks} "
          f"(min cos {tri_min:.6f})")

    all_failures = failures + tri_failures
    if all_failures:
        print(f"\nFAIL: {len(failures)} / {cases} mcop cases below {COS_TH} "
              f"(+{len(tri_failures)} triton cross-checks):")
        for desc, cos in all_failures[:40]:
            print(f"  {desc}: cos={cos:.6f}")
        if len(all_failures) > 40:
            print(f"  ... and {len(all_failures) - 40} more")
        sys.exit(1)
    if cases == 0:
        raise SystemExit("no cases executed")
    print(f"\nPASS: {cases} cases; cos >= {COS_TH} on all dtypes/params/shapes")


if __name__ == "__main__":
    main()
