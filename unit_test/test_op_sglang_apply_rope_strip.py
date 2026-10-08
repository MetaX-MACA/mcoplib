"""Correctness + bandwidth unit test for the flashinfer-stripped RoPE kernel.

Op under test: torch.ops.sgl_kernel.apply_rope_pos_ids_cos_sin_cache
Backing code modified during flashinfer-strip (op/sglang/csrc/elementwise/pos_enc.cuh):
  - device helpers vec_apply_llama_rope_cos_sin / _interleave / _interleave_reuse_half
  - non-Enhanced BatchQKApplyRotaryPosIdsCosSinCacheKernel / ...HeadParallelismKernel
  - launcher BatchQKApplyRotaryPosIdsCosSinCache
This test only needs `import mcoplib.sgl_kernel` (sglang-only build); it does NOT
import mcoplib._C or mcoplib.op.

cos_sin_cache layout (verified from kernel source): row = [cos(0:half) | sin(0:half)],
rotary_dim = 2*half. Non-interleave path = standard rotate-half RoPE. Interleave path
uses the reuse_half variant: pairs (2i, 2i+1) rotated with cos[i], sin[i].

Accuracy gate: cosine similarity >= 0.999 (bf16/fp16) / 0.9999 (fp32) on both q_rope
and k_rope, plus >= 8 elementwise value checks. Trap check via cuda synchronize.
"""
import os
import math
import torch
import pytest
import mcoplib.sgl_kernel  # noqa: F401  registers torch.ops.sgl_kernel.*

OP = torch.ops.sgl_kernel.apply_rope_pos_ids_cos_sin_cache

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
# (nnz, num_q_heads, num_kv_heads, head_dim, rotary_dim)
SHAPES = [
    (16, 4, 2, 64, 64),        # small
    (128, 8, 8, 128, 128),     # full-dim
    (1024, 32, 8, 128, 128),   # GQA, mid
    (4096, 32, 32, 128, 64),   # partial rotary (rotary_dim < head_dim)
    (8192, 32, 8, 128, 128),   # large
]
INTERLEAVE = [False, True]
COS_THRESH = {torch.bfloat16: 0.999, torch.float16: 0.999, torch.float32: 0.9999}
MAX_SEQ = 16384
TARGET_GBPS = 1300.0  # C500 single-die practical wall


def cos_sim(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def make_cos_sin_cache(max_seq, rotary_dim, device):
    # Build a physically meaningful RoPE table: theta_j = 10000^{-2j/rotary_dim}
    half = rotary_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)              # [max_seq, half]
    cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)  # [max_seq, rotary_dim]
    return cache.to(device)


def ref_rope(x, cos_sin_cache, pos_ids, rotary_dim, interleave):
    """Reference in fp32. x: [nnz, H, head_dim]."""
    nnz, H, head_dim = x.shape
    half = rotary_dim // 2
    xf = x.float()
    cos = cos_sin_cache[pos_ids, :half].unsqueeze(1)   # [nnz,1,half]
    sin = cos_sin_cache[pos_ids, half:rotary_dim].unsqueeze(1)
    rot = xf[..., :rotary_dim]
    passthru = xf[..., rotary_dim:]
    if not interleave:
        x1 = rot[..., :half]
        x2 = rot[..., half:]
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin
        out_rot = torch.cat([r1, r2], dim=-1)
    else:
        # pairs (2i, 2i+1) share cos[i]/sin[i]
        xe = rot[..., 0::2]   # even indices -> [.., half]
        xo = rot[..., 1::2]   # odd indices
        re = xe * cos - xo * sin
        ro = xo * cos + xe * sin
        out_rot = torch.empty_like(rot)
        out_rot[..., 0::2] = re
        out_rot[..., 1::2] = ro
    return torch.cat([out_rot, passthru], dim=-1)


def run_case(nnz, hq, hk, hd, rd, dtype, interleave, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(nnz, hq, hd, generator=g).to(dtype).to(dev)
    k = torch.randn(nnz, hk, hd, generator=g).to(dtype).to(dev)
    cache = make_cos_sin_cache(MAX_SEQ, rd, dev)
    pos = torch.randint(0, MAX_SEQ, (nnz,), generator=g).to(torch.int64).to(dev)
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    OP(q, k, q_rope, k_rope, cache, pos, interleave, False, None, None, None, None)
    torch.cuda.synchronize()  # trap check
    q_ref = ref_rope(q, cache, pos, rd, interleave)
    k_ref = ref_rope(k, cache, pos, rd, interleave)
    return q_rope, k_rope, q_ref, k_ref


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("interleave", INTERLEAVE)
def test_apply_rope(dtype, shape, interleave):
    dev = torch.device("cuda")
    nnz, hq, hk, hd, rd = shape
    q_rope, k_rope, q_ref, k_ref = run_case(nnz, hq, hk, hd, rd, dtype, interleave, dev)

    cq = cos_sim(q_rope, q_ref)
    ck = cos_sim(k_rope, k_ref)
    thr = COS_THRESH[dtype]
    # >=8 elementwise checks on q_rope (rotary region of head 0, token 0)
    atol = 3e-2 if dtype != torch.float32 else 1e-4
    got = q_rope[0, 0, :8].float().cpu()
    exp = q_ref[0, 0, :8].float().cpu()
    maxerr = (q_rope.float() - q_ref.float()).abs().max().item()
    print(f"[rope] dt={dtype} shape={shape} il={interleave} "
          f"cos_q={cq:.6f} cos_k={ck:.6f} maxerr={maxerr:.3e}")
    assert cq >= thr, f"q cos {cq} < {thr}"
    assert ck >= thr, f"k cos {ck} < {thr}"
    assert torch.allclose(got, exp, atol=atol, rtol=1e-2), \
        f"value check failed\n got={got}\n exp={exp}"


def _bench_case(nnz, hq, hk, hd, rd, dtype, interleave, dev, iters=100, warmup=10):
    g = torch.Generator(device="cpu").manual_seed(1)
    q = torch.randn(nnz, hq, hd, generator=g).to(dtype).to(dev)
    k = torch.randn(nnz, hk, hd, generator=g).to(dtype).to(dev)
    cache = make_cos_sin_cache(MAX_SEQ, rd, dev)
    pos = torch.randint(0, MAX_SEQ, (nnz,), generator=g).to(torch.int64).to(dev)
    q_rope = torch.empty_like(q)
    k_rope = torch.empty_like(k)
    for _ in range(warmup):
        OP(q, k, q_rope, k_rope, cache, pos, interleave, False, None, None, None, None)
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        OP(q, k, q_rope, k_rope, cache, pos, interleave, False, None, None, None, None)
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    esz = 2 if dtype in (torch.bfloat16, torch.float16) else 4
    # read q,k + write q_rope,k_rope + read cos/sin + pos
    io = (nnz * hq * hd + nnz * hk * hd) * esz * 2
    io += nnz * rd * 4 + nnz * 8
    gbps = io / (ms * 1e-3) / 1e9
    return ms, gbps


def test_rope_bandwidth():
    dev = torch.device("cuda")
    print("\n[rope bandwidth] iters=100 warmup=10")
    print(f"{'dtype':>9} {'shape':>28} {'il':>5} {'ms':>9} {'GB/s':>9} {'%wall':>7}")
    peak = 0.0
    for dtype in [torch.bfloat16, torch.float16]:
        for shape in SHAPES:
            for il in (False,):
                nnz, hq, hk, hd, rd = shape
                ms, gbps = _bench_case(nnz, hq, hk, hd, rd, dtype, il, dev)
                peak = max(peak, gbps)
                print(f"{str(dtype).split('.')[-1]:>9} {str(shape):>28} {str(il):>5} "
                      f"{ms:9.4f} {gbps:9.1f} {100*gbps/TARGET_GBPS:6.1f}%")
    print(f"[rope bandwidth] peak = {peak:.1f} GB/s ({100*peak/TARGET_GBPS:.1f}% of {TARGET_GBPS})")
    assert peak > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
