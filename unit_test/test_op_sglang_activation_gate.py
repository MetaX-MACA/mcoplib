"""Correctness (cosine similarity) + bandwidth unit test for the sglang activation
gate kernels in op/sglang/csrc/elementwise/activation.cu:

  torch.ops.sgl_kernel.silu_and_mul(out, input)
  torch.ops.sgl_kernel.gelu_and_mul(out, input)          # erf ("none") GELU
  torch.ops.sgl_kernel.gelu_tanh_and_mul(out, input)     # tanh GELU

Schema: input[..., :2d], out[..., :d]. Semantics (verified from kernel source):
  gate = input[..., :d]; up = input[..., d:2d]; out = act(gate) * up
    silu:      act(x) = x * sigmoid(x)
    gelu:      act(x) = 0.5*x*(1+erf(x/sqrt(2)))            == F.gelu(x, 'none')
    gelu_tanh: act(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715 x^3)))
                                                            == F.gelu(x, 'tanh')

These ops emit the same float dtype as the input (no int8/fp8 output path), so the
accuracy gate is cosine similarity >= 0.9999 for every dtype. Only needs
`import mcoplib.sgl_kernel`.

Bandwidth: effective HBM bytes = (read 2*d + write d) * num_tokens * elem_size,
reported per dtype/shape.
"""
import math
import torch
import pytest
import mcoplib.sgl_kernel  # noqa: F401

OPS = {
    "silu_and_mul": (torch.ops.sgl_kernel.silu_and_mul,
                     lambda g, u: torch.nn.functional.silu(g) * u),
    "gelu_and_mul": (torch.ops.sgl_kernel.gelu_and_mul,
                     lambda g, u: torch.nn.functional.gelu(g, approximate="none") * u),
    "gelu_tanh_and_mul": (torch.ops.sgl_kernel.gelu_tanh_and_mul,
                          lambda g, u: torch.nn.functional.gelu(g, approximate="tanh") * u),
}

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
# (num_tokens, d)  d = hidden intermediate; input last dim = 2*d
SHAPES = [
    (16, 2048),
    (1024, 4096),
    (2048, 5120),
    (4096, 8192),
    (8192, 14336),   # llama-style MoE/mlp intermediate
    (1, 1537),       # exercises the d==1537 special (LAUNCH_ACTIVATION_GATE) path
    (333, 1000),     # non-vectorizable d (scalar fallback path)
]
COS_THRESH = 0.9999          # float dtypes
COS_THRESH_LOWP = 0.999      # int8/fp8 (none of these ops produce those, kept for clarity)
TARGET_GBPS = 1300.0


def cos_sim(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def make_input(n, d, dtype, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, 2 * d, generator=g).to(dtype)
    return x.to(dev)


def ref(inp, ref_act):
    d = inp.shape[-1] // 2
    gate = inp[..., :d].float()
    up = inp[..., d:].float()
    return ref_act(gate, up)


@pytest.mark.parametrize("name", list(OPS.keys()))
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_activation_gate(name, dtype, shape):
    dev = torch.device("cuda")
    op, ref_act = OPS[name]
    n, d = shape
    inp = make_input(n, d, dtype, dev)
    out = torch.empty(n, d, dtype=dtype, device=dev)
    op(out, inp)
    torch.cuda.synchronize()  # trap check
    ref_out = ref(inp, ref_act)
    c = cos_sim(out, ref_out)
    maxerr = (out.float() - ref_out).abs().max().item()
    # >=8 elementwise checks on row 0
    atol = 3e-2 if dtype != torch.float32 else 1e-4
    got = out[0, :8].float().cpu(); exp = ref_out[0, :8].float().cpu()
    print(f"[{name}] dt={str(dtype).split('.')[-1]:>9} shape=({n},{d}) "
          f"cos={c:.6f} maxerr={maxerr:.3e}")
    assert c >= COS_THRESH, f"{name} {dtype} {shape} cos {c} < {COS_THRESH}"
    assert torch.allclose(got, exp, atol=atol, rtol=1e-2), \
        f"{name} value check failed\n got={got}\n exp={exp}"


def _bench(op, n, d, dtype, dev, iters=100, warmup=10):
    inp = make_input(n, d, dtype, dev, seed=1)
    out = torch.empty(n, d, dtype=dtype, device=dev)
    for _ in range(warmup):
        op(out, inp)
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        op(out, inp)
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    esz = inp.element_size()
    io = n * (2 * d + d) * esz     # read 2d + write d
    gbps = io / (ms * 1e-3) / 1e9
    return ms, gbps


@pytest.mark.parametrize("name", list(OPS.keys()))
def test_activation_bandwidth(name):
    dev = torch.device("cuda")
    op, _ = OPS[name]
    print(f"\n[{name} bandwidth] iters=100 warmup=10  bytes=n*3d*esz")
    print(f"{'dtype':>9} {'n':>6} {'d':>7} {'ms':>9} {'GB/s':>9} {'%wall':>7}")
    peak = 0.0
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for (n, d) in SHAPES:
            ms, gbps = _bench(op, n, d, dtype, dev)
            peak = max(peak, gbps)
            print(f"{str(dtype).split('.')[-1]:>9} {n:6d} {d:7d} "
                  f"{ms:9.4f} {gbps:9.1f} {100*gbps/TARGET_GBPS:6.1f}%")
    print(f"[{name} bandwidth] peak = {peak:.1f} GB/s "
          f"({100*peak/TARGET_GBPS:.1f}% of {TARGET_GBPS})")
    assert peak > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
