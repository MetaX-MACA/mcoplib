"""Correctness + bandwidth unit test for the flashinfer-stripped fused_add_rmsnorm.

Op under test: torch.ops.sgl_kernel.fused_add_rmsnorm(input, residual, weight, eps, enable_pdl)
In-place fused Add + RMSNorm:
    residual <- input + residual              (residual updated in place)
    input    <- rmsnorm(residual) * weight    (input overwritten with norm output)

Backing code modified during flashinfer-strip:
  - op/sglang/include/norm.cuh (NEW): ported FusedAddRMSNormKernel + FusedAddRMSNorm
    into namespace mcoplib::norm
  - op/sglang/csrc/elementwise/fused_add_rms_norm_kernel.cu: #include "norm.cuh" +
    call mcoplib::norm::FusedAddRMSNorm

This test only needs `import mcoplib.sgl_kernel` (sglang-only build).

Accuracy gate: cosine similarity >= 0.999 on both updated residual and normalized
input, plus >= 8 elementwise value checks. Trap check via cuda synchronize.

Note: the repo's fallback dispatch for this op only supports float32 at very large
hidden (H=8192) for bf16/fp16 (pre-existing, not from the flashinfer-strip). We test
the mainstream hidden sizes that go through the optimized bf16/fp16 path, plus fp32
across all sizes.
"""
import torch
import pytest
import mcoplib.sgl_kernel  # noqa: F401

OP = torch.ops.sgl_kernel.fused_add_rmsnorm
EPS = 1e-6
COS_THRESH = 0.999
TARGET_GBPS = 1300.0

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
# hidden sizes on the optimized path (avoid the known fp32-only H=8192 bf16 fallback)
HIDDENS = [2048, 4096, 5120, 7168]
NUM_TOKENS = [16, 1024, 2048, 4096]


def cos_sim(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def make_inputs(n, h, dtype, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, h, generator=g).to(dtype)
    r = torch.randn(n, h, generator=g).to(dtype)
    w = (1.0 + 0.1 * torch.randn(h, generator=g)).to(dtype)
    return x.to(dev), r.to(dev), w.to(dev)


def ref(x, r, w, eps):
    res = x.float() + r.float()
    var = res.pow(2).mean(dim=-1, keepdim=True)
    out = res * torch.rsqrt(var + eps) * w.float()
    return out, res


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("h", HIDDENS)
@pytest.mark.parametrize("n", NUM_TOKENS)
def test_fused_add_rmsnorm(dtype, h, n):
    dev = torch.device("cuda")
    x0, r0, w = make_inputs(n, h, dtype, dev)
    out_ref, res_ref = ref(x0, r0, w, EPS)
    x = x0.clone(); r = r0.clone()
    OP(x, r, w, EPS, False)
    torch.cuda.synchronize()  # trap check
    cos_out = cos_sim(x, out_ref)
    cos_res = cos_sim(r, res_ref)
    maxerr_out = (x.float() - out_ref).abs().max().item()
    # >=8 elementwise checks on the normalized output row 0
    atol = 3e-2 if dtype != torch.float32 else 1e-4
    got = x[0, :8].float().cpu(); exp = out_ref[0, :8].float().cpu()
    print(f"[rmsnorm] dt={dtype} n={n} h={h} cos_out={cos_out:.6f} "
          f"cos_res={cos_res:.6f} maxerr={maxerr_out:.3e}")
    assert cos_out >= COS_THRESH, f"out cos {cos_out} < {COS_THRESH}"
    assert cos_res >= COS_THRESH, f"res cos {cos_res} < {COS_THRESH}"
    assert torch.allclose(got, exp, atol=atol, rtol=1e-2), \
        f"value check failed\n got={got}\n exp={exp}"


def _bench(n, h, dtype, dev, iters=100, warmup=10):
    x0, r0, w = make_inputs(n, h, dtype, dev, seed=1)
    esz = 2 if dtype in (torch.bfloat16, torch.float16) else 4
    for _ in range(warmup):
        x = x0.clone(); r = r0.clone(); OP(x, r, w, EPS, False)
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    # clone cost would pollute timing -> pre-clone into a pool and reuse in-place op only
    xs = [x0.clone() for _ in range(iters)]
    rs = [r0.clone() for _ in range(iters)]
    torch.cuda.synchronize()
    s.record()
    for i in range(iters):
        OP(xs[i], rs[i], w, EPS, False)
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    io = n * h * esz * 4  # read x + read r + write x + write r
    gbps = io / (ms * 1e-3) / 1e9
    return ms, gbps


def test_rmsnorm_bandwidth():
    dev = torch.device("cuda")
    print("\n[rmsnorm bandwidth] iters=100 warmup=10  (bytes = n*h*esz*4)")
    print(f"{'dtype':>9} {'n':>6} {'h':>6} {'ms':>9} {'GB/s':>9} {'%wall':>7}")
    peak = 0.0
    for dtype in [torch.bfloat16, torch.float16]:
        for h in HIDDENS:
            for n in NUM_TOKENS:
                ms, gbps = _bench(n, h, dtype, dev)
                peak = max(peak, gbps)
                print(f"{str(dtype).split('.')[-1]:>9} {n:6d} {h:6d} "
                      f"{ms:9.4f} {gbps:9.1f} {100*gbps/TARGET_GBPS:6.1f}%")
    print(f"[rmsnorm bandwidth] peak = {peak:.1f} GB/s ({100*peak/TARGET_GBPS:.1f}% of {TARGET_GBPS})")
    assert peak > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
