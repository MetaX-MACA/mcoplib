"""Validate vLLM SiLU+mul clamp in legacy and Step4 modes."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

import mcoplib._C  # noqa: F401


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
    if gate.shape != up.shape or gate.dtype != torch.bfloat16 or up.dtype != torch.bfloat16:
        raise ValueError("gate and up must have equal BF16 shapes")
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
    return (gate_f * (up_f + beta)).to(gate.dtype)


def _triton_ref(gate, up, limit):
    limit = _validate(gate, up, limit)
    out = torch.empty_like(gate)
    if gate.numel():
        _step4_kernel[(triton.cdiv(gate.numel(), 1024),)](
            gate, up, out, gate.numel(), limit, BLOCK=1024, num_warps=4
        )
    return out


def _mcop(gate, up, limit, alpha, beta, step4):
    _validate(gate, up, limit)
    packed = torch.cat((gate, up), dim=-1)
    out = torch.empty_like(gate)
    torch.ops._C.silu_and_mul_with_clamp(out, packed, limit, alpha, beta, step4)
    return out


def _cosine(actual, expected):
    if not actual.numel() or torch.equal(actual, expected):
        return 1.0
    return float(torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0))


def _run():
    if not torch.cuda.is_available():
        raise RuntimeError("tests require a CUDA/MACA device")
    shapes = [(0, 1536), (1, 1536), (128, 1536), (3, 1537), (2, 3, 257),
              (1, 4096), (8, 4096), (128, 4096), (1, 6144), (16, 6144),
              (128, 8192), (1, 11008), (32, 11008), (1, 14336), (64, 14336)]
    limits = (0.0, 1.0, 7.0, 10.0)
    alphas = (0.5, 1.0, 1.7)
    cases = 0
    for shape in shapes:
        for limit in limits:
            for alpha in alphas:
                for step4 in (False, True):
                    torch.manual_seed(20260903 + sum(shape) + int(limit * 10) + int(alpha * 10) + step4)
                    gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                    up = torch.randn_like(gate)
                    expected = _torch_ref(gate, up, limit, alpha=alpha, step4=step4)
                    actual = _mcop(gate, up, limit, alpha, 0.0, step4)
                    sim = _cosine(actual, expected)
                    if sim < 0.99999:
                        raise AssertionError(f"mcop cosine {sim} for {shape}, {limit}, {alpha}, {step4}")
                    if step4 and alpha == 1.0:
                        triton_out = _triton_ref(gate, up, limit)
                        tsim = _cosine(triton_out, expected)
                        if tsim < 0.99999:
                            raise AssertionError(f"triton cosine {tsim} for {shape}, {limit}")
                    cases += 1
    print(f"PASS: {cases} cases; BF16 cosine >= 0.99999")


if __name__ == "__main__":
    _run()
