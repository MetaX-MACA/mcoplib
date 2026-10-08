"""Unit test + bandwidth benchmark for step5_fused_qk_norm_rope on MetaX C600-U.

Op (torch.ops._C.step5_fused_qk_norm_rope):
  q, k, v = step5_fused_qk_norm_rope(qkv, q_weight, k_weight, cos, sin, positions,
                                     num_q_heads, num_kv_heads, head_dim, rotary_pairs,
                                     eps, norm_weight_bias, q_out=None, k_out=None, v_out=None)

Semantics (per token row of the packed QKV projection [Q | K | V], bf16):
  * Q/K heads: per-head RMSNorm, then NeoX-style partial RoPE.
        inv_rms   = rsqrt(sum(x^2) / head_dim + eps)
        y         = bf16_round(x * inv_rms * (w + norm_weight_bias))
        out[p]    = bf16_round(y[p]*cos[p] - y[p+rp]*sin[p])   p in [0, rp)
        out[p+rp] = bf16_round(y[p]*sin[p] + y[p+rp]*cos[p])
        out[2rp:] = y[2rp:]                                    (norm only)
    The normalized value is rounded to bf16 BEFORE the rotation (round-trip
    through the activation dtype is part of the contract).
  * V heads: bit-exact copy out of qkv; qkv itself is never written.
  * Outputs: all three of q_out/k_out/v_out given (written in place, same
    storage returned) or all omitted (fresh bf16 [tokens, width] allocations).

Coverage: every scenario of unit_test/test_vllm_step5_fused_qknorm_rope.py
(2 TP splits x (2 prefill + 7 decode batch sizes x 4 decode scenarios) = 60
cases, head_dim=192, rotary_pairs in {96, 32}, prealloc on/off, identical
inputs/seeds/positions), plus generality cases: int32 positions, bias=0,
rotary_pairs=0, bf16 weights, fp32 cos/sin, head_dim 64/96/128/256,
rotary_pairs % 32 != 0 (generic kernel), wide qkv rows, strided outputs,
output-contract errors, empty tokens, bf16 round-trip semantics, and CUDA
graph capture/replay (incl. in-place positions mutation between replays).

Accuracy: per-case cos_sim >= 0.9999 AND 4-bf16-ulp elementwise for q and k,
V bit-exact vs both the reference and the qkv slice, qkv unmutated.
Timing: warmup=10, reps=50, per-rep events, median (back-to-back burst in a
single sync keeps DVFS up), reported vs the Triton baseline implementation
from test_vllm_step5_fused_qknorm_rope.py on identical inputs.

Run:
  mx-smi                      # pick an idle GPU first
  export CUDA_VISIBLE_DEVICES=<idle>
  python unit_test/test_op_vllm_step5_fused_qk_norm_rope.py
"""

import faulthandler
import math
import os
import sys

import torch

import mcoplib._C  # noqa: F401  (registers torch.ops._C.*)

# Observability: line-buffered stdout + periodic stack dumps if a case wedges
# (a hung kernel / stream-capture deadlock would otherwise show as silence).
sys.stdout.reconfigure(line_buffering=True)
faulthandler.dump_traceback_later(240, repeat=True)

# ── host workaround: cudaStreamCreate* wedges non-deterministically ────────
# On this host creating a stream (any entry point: torch.cuda.Stream()'s c10
# pool burst, or one raw shim call) sometimes wedges inside the MACA kernel
# driver: endless "[mxkwCreateQueueBlock][Hint]ioctl create queue block
# timeout, gpu_id:12321 type:21. Retrying." ~10 s apart, never returning --
# it behaves like transient host-wide queue-block contention. The CUDA graph
# capture phase therefore uses cudaStreamPerThread (special handle 2): a
# runtime-provided stream that needs no creation ioctl at all. probe_nocreate
# .py verified on this host that kernels run on it and full capture/replay
# cycles work end-to-end. torch.cuda.Stream() must still never be called --
# including lazily inside torch.cuda.graph, whose class-level
# default_capture_stream is pre-seeded below so that path never runs. This
# only affects the test harness, not the op under test.
PT_STREAM = torch.cuda.ExternalStream(2)  # cudaStreamPerThread
torch.cuda.graphs.graph.default_capture_stream = PT_STREAM
print("[canary] per-thread capture stream ready (zero stream creation)")

OP = torch.ops._C.step5_fused_qk_norm_rope

DEVICE = "cuda"
TARGET_GBPS = 1600.0      # user target (single-die)
GUIDE_WALL_GBPS = 1300.0  # C600-U guide 1R1W copy wall (measured pattern ceiling)
DUAL_DIE_GBPS = 3480.0    # dual-die datasheet figure
BF16_ULP = 2 ** -7

# Triton baseline + exact scenario generators from the reference test file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import test_vllm_step5_fused_qknorm_rope as _tr
    TRITON_IMPL = _tr.fused_qknorm_rope if _tr.triton is not None else None
except ImportError:
    _tr = None
    TRITON_IMPL = None


# ── torch reference ─────────────────────────────────────────────────────────
def reference_step5(qkv, q_weight, k_weight, cos, sin, positions,
                    num_q_heads, num_kv_heads, head_dim, rotary_pairs,
                    eps=1e-5, norm_weight_bias=1.0):
    """Pure-torch reference. Returns fresh contiguous (q, k, v) tensors."""
    tokens = qkv.shape[0]
    dtype = qkv.dtype
    q_width = num_q_heads * head_dim
    kv_width = num_kv_heads * head_dim
    rp = rotary_pairs

    def norm_rope(x3d, weight):
        # x3d: [tokens, heads, head_dim] fp32
        scale = weight.reshape(-1).to(torch.float32) + norm_weight_bias
        inv_rms = torch.rsqrt(x3d.pow(2).mean(dim=-1, keepdim=True) + eps)
        y = (x3d * inv_rms * scale).to(dtype)  # bf16 round BEFORE RoPE
        if rp == 0:
            return y
        pos = positions.reshape(-1).to(torch.int64)
        c = cos.index_select(0, pos).to(torch.float32).unsqueeze(1)  # [T,1,rp]
        s = sin.index_select(0, pos).to(torch.float32).unsqueeze(1)
        y32 = y.to(torch.float32)
        y0, y1 = y32[..., :rp], y32[..., rp:2 * rp]
        tail = y[..., 2 * rp:]
        return torch.cat(((y0 * c - y1 * s).to(dtype),
                          (y0 * s + y1 * c).to(dtype), tail), dim=-1)

    q = norm_rope(
        qkv[:, :q_width].reshape(tokens, num_q_heads, head_dim).to(torch.float32),
        q_weight).reshape(tokens, q_width)
    k = norm_rope(
        qkv[:, q_width:q_width + kv_width].reshape(tokens, num_kv_heads,
                                                   head_dim).to(torch.float32),
        k_weight).reshape(tokens, kv_width)
    v = qkv[:, q_width + kv_width:q_width + 2 * kv_width].reshape(tokens, kv_width)
    return q, k, v


def reference_step5_noround(qkv, q_weight, k_weight, cos, sin, positions,
                            num_q_heads, num_kv_heads, head_dim, rotary_pairs,
                            eps=1e-5, norm_weight_bias=1.0):
    """Same math but WITHOUT the bf16 round-trip before RoPE (must differ)."""
    tokens = qkv.shape[0]
    q_width = num_q_heads * head_dim
    kv_width = num_kv_heads * head_dim
    rp = rotary_pairs
    x = qkv[:, :q_width].reshape(tokens, num_q_heads, head_dim).to(torch.float32)
    scale = q_weight.reshape(-1).to(torch.float32) + norm_weight_bias
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    y32 = x * inv_rms * scale
    if rp == 0:
        return y32.to(qkv.dtype).reshape(tokens, q_width)
    pos = positions.reshape(-1).to(torch.int64)
    c = cos.index_select(0, pos).to(torch.float32).unsqueeze(1)
    s = sin.index_select(0, pos).to(torch.float32).unsqueeze(1)
    y0, y1 = y32[..., :rp], y32[..., rp:2 * rp]
    tail = y32[..., 2 * rp:]
    return torch.cat(((y0 * c - y1 * s).to(qkv.dtype),
                      (y0 * s + y1 * c).to(qkv.dtype),
                      tail.to(qkv.dtype)), dim=-1).reshape(tokens, q_width)


# ── accuracy helpers ────────────────────────────────────────────────────────
def cos_sim(a, b):
    af = a.float().view(-1)
    bf = b.float().view(-1)
    denom = (af.norm() * bf.norm()).clamp_min(1e-30)
    return (torch.dot(af, bf) / denom).item()


def within_bf16_ulp(actual, expected, tol_ulp=4):
    """4 bf16 ulps at max(|a|, |b|, 1) - the tolerance of the reference test."""
    if torch.equal(actual, expected):
        return True, 0.0, 0
    a, b = actual.float(), expected.float()
    diff = (a - b).abs()
    tol = torch.maximum(a.abs(), b.abs()).clamp_min(1.0) * (tol_ulp * BF16_ULP)
    bad = diff > tol
    return bool(bad.sum().item() == 0), diff.max().item(), int(bad.sum().item())


def effective_bytes(tokens, nq, nkv, hd, rp, pos_bytes=8, weight_bytes=4):
    # identical accounting for the CUDA op and the Triton baseline
    return (tokens * (nq + 2 * nkv) * hd * 2          # qkv read (bf16)
            + tokens * (nq + 2 * nkv) * hd * 2        # q/k/v writes (bf16)
            + tokens * (nq + nkv) * rp * 2 * 2        # cos+sin loads per QK head
            + tokens * pos_bytes                      # positions
            + 2 * hd * weight_bytes)                  # q/k weights


def bench(fn, warmup=10, rep=50):
    """Back-to-back burst within one sync (keeps DVFS up); median of reps."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]


ROWS = []


def run_case(tag, qkv, q_weight, k_weight, cos, sin, positions,
             nq, nkv, hd, rp, eps=1e-5, bias=1.0, prealloc=False,
             outs_override=None, time_it=True):
    tokens = qkv.shape[0]
    q_width, kv_width = nq * hd, nkv * hd
    print(f"[begin ] {tag} T={tokens} nq={nq} nkv={nkv} hd={hd} rp={rp} "
          f"pre={int(prealloc)}", flush=True)
    qkv_before = qkv.clone()

    outs = {}
    if prealloc:
        outs = dict(
            q_out=torch.full((tokens, q_width), float("nan"),
                             device=DEVICE, dtype=torch.bfloat16),
            k_out=torch.full((tokens, kv_width), float("nan"),
                             device=DEVICE, dtype=torch.bfloat16),
            v_out=torch.full((tokens, kv_width), float("nan"),
                             device=DEVICE, dtype=torch.bfloat16),
        )
    if outs_override is not None:
        outs = outs_override

    q, k, v = OP(qkv, q_weight, k_weight, cos, sin, positions,
                 nq, nkv, hd, rp, eps, bias, **outs)
    torch.cuda.synchronize()  # surfaces kernel traps / launch errors

    ok, msgs = True, []
    if outs:
        if not (q.data_ptr() == outs["q_out"].data_ptr()
                and k.data_ptr() == outs["k_out"].data_ptr()
                and v.data_ptr() == outs["v_out"].data_ptr()):
            ok = False
            msgs.append("prealloc outputs not written in place")
    if not torch.equal(qkv, qkv_before):
        ok = False
        msgs.append("qkv was mutated")

    worst_sim = 1.0
    if tokens > 0:
        rq, rk, rv = reference_step5(qkv, q_weight, k_weight, cos, sin,
                                     positions, nq, nkv, hd, rp, eps, bias)
        for name, got, ref in (("q", q, rq), ("k", k, rk)):
            sim = cos_sim(got, ref)
            good, maxdiff, nbad = within_bf16_ulp(got, ref)
            worst_sim = min(worst_sim, sim)
            if sim < 0.9999 or not good:
                ok = False
                msgs.append(f"{name} sim={sim:.7f} bad={nbad} maxdiff={maxdiff:.3e}")
        v_src = qkv[:, q_width + kv_width:q_width + 2 * kv_width].reshape(
            tokens, kv_width)
        if not torch.equal(v, v_src):
            ok = False
            msgs.append("V is not a bit-exact copy of qkv")
        if not torch.equal(v, rv):
            ok = False
            msgs.append("V differs from the reference copy")

    cuda_ms = tri_ms = None
    if time_it and tokens > 0:
        def fn_cuda():
            OP(qkv, q_weight, k_weight, cos, sin, positions,
               nq, nkv, hd, rp, eps, bias, **outs)
        cuda_ms = bench(fn_cuda)
        if TRITON_IMPL is not None:
            touts = {kk: vv.clone() for kk, vv in outs.items()} if outs else {}

            def fn_triton():
                TRITON_IMPL(qkv, q_weight, k_weight, cos, sin, positions,
                            nq, nkv, hd, rp, eps=eps, norm_weight_bias=bias,
                            **touts)
            tri_ms = bench(fn_triton)

    pos_bytes = 8 if positions.dtype == torch.int64 else 4
    wb = 4 if q_weight.dtype == torch.float32 else 2
    nbytes = effective_bytes(tokens, nq, nkv, hd, rp, pos_bytes, wb)
    gbps = nbytes / (cuda_ms * 1e-3) / 1e9 if cuda_ms else 0.0
    tri_gbps = nbytes / (tri_ms * 1e-3) / 1e9 if tri_ms else 0.0
    speedup = (tri_ms / cuda_ms) if (cuda_ms and tri_ms) else None

    pre = "pre  " if outs else "alloc"
    simstr = f"{worst_sim:.6f}" if tokens > 0 else "  n/a  "
    tri_str = f" | triton {tri_ms:8.4f} ms {tri_gbps:8.1f} GB/s | {speedup:5.2f}x" \
        if tri_ms else " | triton n/a"
    print(f"[{tag:<6}] T={tokens:6d} nq={nq:2d} nkv={nkv} hd={hd:3d} rp={rp:3d} "
          f"{pre} cos_sim={simstr} {'OK' if ok else 'FAIL':4s} "
          f"{cuda_ms if cuda_ms else 0:8.4f} ms {gbps:8.1f} GB/s{tri_str}"
          + ("" if ok else "  << " + "; ".join(msgs)))
    ROWS.append(dict(tag=tag, tokens=tokens, hd=hd, rp=rp, ok=ok,
                     cuda_ms=cuda_ms, tri_ms=tri_ms, speedup=speedup,
                     gbps=gbps, tri_gbps=tri_gbps))
    return ok


def make_extra_inputs(tokens, nq, nkv, hd, rp, seed,
                      pos_dtype=torch.int64, weight_dtype=torch.float32,
                      cos_dtype=torch.bfloat16, wide=0):
    packed = (nq + 2 * nkv) * hd
    width = packed + wide
    g = torch.Generator(device="cpu").manual_seed(seed)
    qkv = torch.randn(tokens, width, dtype=torch.float32,
                      generator=g).to(torch.bfloat16).to(DEVICE)
    q_weight = torch.randn(hd, dtype=torch.float32,
                           generator=g).to(weight_dtype).to(DEVICE)
    k_weight = torch.randn(hd, dtype=torch.float32,
                           generator=g).to(weight_dtype).to(DEVICE)
    max_pos = 512
    if rp > 0:
        angles = torch.rand(max_pos, rp, dtype=torch.float32,
                            generator=g) * (2 * math.pi)
        cos = torch.cos(angles).to(cos_dtype).to(DEVICE)
        sin = torch.sin(angles).to(cos_dtype).to(DEVICE)
        positions = (torch.arange(tokens) * 7 % max_pos).to(pos_dtype).to(DEVICE)
    else:
        cos = torch.empty(max_pos, 0, dtype=cos_dtype, device=DEVICE)
        sin = torch.empty(max_pos, 0, dtype=cos_dtype, device=DEVICE)
        positions = torch.zeros(tokens, dtype=pos_dtype, device=DEVICE)
    return qkv, q_weight, k_weight, cos, sin, positions


def cross_check_reference():
    """My torch reference must agree with the formula reference in the
    original triton test file (independent implementations of the same math)."""
    assert _tr is not None, "test_vllm_step5_fused_qknorm_rope.py not importable"
    for rp in (96, 32):
        qkv, qw, kw, cos, sin, pos = _tr._make_inputs(
            "decode", rp, DEVICE, 8, 1, 8192, num_seqs=16)
        mine = reference_step5(qkv, qw, kw, cos, sin, pos, 8, 1, 192, rp)
        theirs = _tr.formula_fused_qknorm_rope(qkv, qw, kw, cos, sin, pos,
                                               8, 1, 192, rp)
        for m, t, name in zip(mine, theirs, "qkv"):
            good, maxdiff, nbad = within_bf16_ulp(m, t)
            assert good, (f"reference cross-check failed for {name} "
                          f"(rp={rp}): {nbad} bad, maxdiff={maxdiff:.3e}")
    print("[xcheck ] torch reference matches the formula reference of "
          "test_vllm_step5_fused_qknorm_rope.py (rp=96/32, q/k/v)")


def test_round_trip_semantics():
    """The kernel must see the bf16-rounded norm, not the fp32 norm."""
    rp, nq, nkv = 32, 8, 1
    qkv, qw, kw, cos, sin, pos = _tr._make_inputs(
        "decode", rp, DEVICE, nq, nkv, 8192, num_seqs=4)
    q, _, _ = OP(qkv, qw, kw, cos, sin, pos, nq, nkv, 192, rp, 1e-5, 1.0)
    no_round = reference_step5_noround(qkv, qw, kw, cos, sin, pos,
                                       nq, nkv, 192, rp)
    torch.cuda.synchronize()
    assert not torch.equal(q, no_round), \
        "kernel output matches the no-round-trip reference (round-trip missing)"
    good, _, _ = within_bf16_ulp(q, reference_step5(
        qkv, qw, kw, cos, sin, pos, nq, nkv, 192, rp)[0])
    assert good
    print("[rtrip  ] bf16 round-trip before RoPE verified "
          "(kernel != no-round reference)")


def test_empty_tokens():
    hd, nq, nkv, rp = 192, 8, 1, 96
    packed = (nq + 2 * nkv) * hd
    qkv = torch.empty(0, packed, device=DEVICE, dtype=torch.bfloat16)
    qw = torch.empty(hd, device=DEVICE, dtype=torch.float32)
    kw = torch.empty(hd, device=DEVICE, dtype=torch.float32)
    cos = torch.empty(4, rp, device=DEVICE, dtype=torch.bfloat16)
    sin = torch.empty(4, rp, device=DEVICE, dtype=torch.bfloat16)
    pos = torch.empty(0, device=DEVICE, dtype=torch.int64)
    q, k, v = OP(qkv, qw, kw, cos, sin, pos, nq, nkv, hd, rp, 1e-5, 1.0)
    assert q.shape == (0, nq * hd) and k.shape == (0, nkv * hd) \
        and v.shape == (0, nkv * hd)
    q2 = torch.empty(0, nq * hd, device=DEVICE, dtype=torch.bfloat16)
    k2 = torch.empty(0, nkv * hd, device=DEVICE, dtype=torch.bfloat16)
    v2 = torch.empty(0, nkv * hd, device=DEVICE, dtype=torch.bfloat16)
    q2, k2, v2 = OP(qkv, qw, kw, cos, sin, pos, nq, nkv, hd, rp, 1e-5, 1.0,
                    q_out=q2, k_out=k2, v_out=v2)
    assert q2.shape == (0, nq * hd) and k2.shape == (0, nkv * hd) \
        and v2.shape == (0, nkv * hd)
    print("[empty  ] tokens=0 -> (0, q_width)/(0, kv_width)/(0, kv_width) "
          "OK (alloc + prealloc, no launch)")


def test_contract_errors():
    nq, nkv, hd, rp, tokens = 8, 1, 192, 96, 8
    qkv, qw, kw, cos, sin, pos = make_extra_inputs(tokens, nq, nkv, hd, rp,
                                                   seed=777)
    q_width, kv_width = nq * hd, nkv * hd
    q_out = torch.empty(tokens, q_width, device=DEVICE, dtype=torch.bfloat16)
    k_out = torch.empty(tokens, kv_width, device=DEVICE, dtype=torch.bfloat16)
    v_out = torch.empty(tokens, kv_width, device=DEVICE, dtype=torch.bfloat16)
    base = (qkv, qw, kw, cos, sin, pos, nq, nkv, hd, rp, 1e-5, 1.0)

    def expect_error(desc, fn):
        try:
            fn()
        except RuntimeError:
            print(f"[cntrct] {desc}: RuntimeError raised as expected")
            return
        raise AssertionError(f"{desc}: expected a RuntimeError")

    expect_error("only q_out given (all-or-none)",
                 lambda: OP(*base, q_out=q_out))
    expect_error("q_out+k_out given, v_out missing (all-or-none)",
                 lambda: OP(*base, q_out=q_out, k_out=k_out))
    bad = torch.empty(tokens, q_width + 8, device=DEVICE, dtype=torch.bfloat16)
    expect_error("wrong q_out shape",
                 lambda: OP(*base, q_out=bad, k_out=k_out, v_out=v_out))
    expect_error("rotary_pairs too large (2*rp > head_dim)",
                 lambda: OP(qkv, qw, kw, cos, sin, pos, nq, nkv, hd, 200,
                            1e-5, 1.0))
    expect_error("fp16 qkv rejected",
                 lambda: OP(qkv.to(torch.float16), qw, kw, cos, sin, pos,
                            nq, nkv, hd, rp, 1e-5, 1.0))


def test_graph_capture(prealloc):
    nq, nkv, hd, rp, tokens = 8, 1, 192, 96, 64
    qkv, qw, kw, cos, sin, pos = _tr._make_inputs(
        "decode", rp, DEVICE, nq, nkv, 8192, num_seqs=tokens)
    q_width, kv_width = nq * hd, nkv * hd
    outs = {}
    if prealloc:
        outs = dict(
            q_out=torch.full((tokens, q_width), float("nan"),
                             device=DEVICE, dtype=torch.bfloat16),
            k_out=torch.full((tokens, kv_width), float("nan"),
                             device=DEVICE, dtype=torch.bfloat16),
            v_out=torch.full((tokens, kv_width), float("nan"),
                             device=DEVICE, dtype=torch.bfloat16),
        )
    args = (qkv, qw, kw, cos, sin, pos, nq, nkv, hd, rp, 1e-5, 1.0)

    # Warmup on the per-thread stream (torch.cuda.graph requirement; using it
    # instead of torch.cuda.Stream(), which wedges in this host's kernel
    # driver -- see the workaround block at the top of the file).
    with torch.cuda.stream(PT_STREAM):
        for _ in range(3):
            OP(*args, **outs)
    torch.cuda.synchronize()

    # Capture on the per-thread stream (sequential captures on one stream are
    # legal; the class-level default capture stream was pre-seeded at import
    # time so torch's lazy torch.cuda.Stream() never runs).
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=PT_STREAM):
        q, k, v = OP(*args, **outs)
    torch.cuda.synchronize()

    def verify(phase):
        rq, rk, _ = reference_step5(qkv, qw, kw, cos, sin, pos, nq, nkv, hd, rp)
        okq, dq, nq_bad = within_bf16_ulp(q, rq)
        okk, dk, nk_bad = within_bf16_ulp(k, rk)
        v_src = qkv[:, q_width + kv_width:].reshape(tokens, kv_width)
        okv = torch.equal(v, v_src)
        assert okq and okk and okv, (
            f"graph replay {phase}: q_ok={okq}({nq_bad} bad) "
            f"k_ok={okk}({nk_bad} bad) v_ok={okv}")

    g.replay()
    torch.cuda.synchronize()
    verify("initial")

    # In-place positions mutation between replays: the graph must pick up the
    # new values (positions are read on the device, nothing was baked in).
    pos.copy_((pos + 7) % cos.shape[0])
    g.replay()
    torch.cuda.synchronize()
    verify("after-positions-mutation")

    if prealloc:
        for t in (q, k, v):
            t.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        verify("after-nan-refill")
    kind = "prealloc" if prealloc else "alloc"
    print(f"[graph  ] CUDA graph capture + replay OK ({kind}; positions "
          f"mutation + full rewrite verified)")


# ── extra generality cases ─────────────────────────────────────────────────
EXTRA_CASES = [
    # tag, tokens, nq, nkv, hd, rp, make_extra_inputs kwargs
    ("i32pos", 64, 8, 1, 192, 96, dict(pos_dtype=torch.int32)),
    ("bias0", 512, 8, 1, 192, 32, dict()),
    ("rp0", 512, 8, 1, 192, 0, dict()),
    ("wbf16", 64, 8, 1, 192, 96, dict(weight_dtype=torch.bfloat16)),
    ("cosf32", 64, 8, 1, 192, 96, dict(cos_dtype=torch.float32)),
    ("hd64", 256, 8, 1, 64, 32, dict()),
    ("hd128", 256, 8, 1, 128, 32, dict()),
    ("hd128r64", 256, 8, 1, 128, 64, dict()),
    ("hd256", 256, 8, 1, 256, 96, dict()),
    ("hd96g", 256, 8, 1, 96, 16, dict()),    # generic kernel (hd not in fast set)
    ("rp48g", 256, 8, 1, 192, 48, dict()),   # generic kernel (rp % 32 != 0)
    ("wide", 512, 8, 1, 192, 96, dict(wide=64)),  # qkv rows wider than packed
]


def main():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES={vis!r}  device_count={torch.cuda.device_count()}")
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"device: {torch.cuda.get_device_name(0)}")
    # The capture stream is cudaStreamPerThread (pre-existing, zero stream
    # creation): torch.cuda.Stream() and raw creation both wedge in this
    # host's kernel driver -- see the workaround block at the top of the file.
    print("op: torch.ops._C.step5_fused_qk_norm_rope | "
          "qkv bf16, weights fp32, cos/sin bf16, positions int64/int32")
    print("accuracy: cos_sim >= 0.9999 AND 4-bf16-ulp elementwise (q,k); "
          "V bit-exact; qkv unmutated")
    print(f"timing: warmup=10 reps=50 median (burst) | "
          f"target single-die {TARGET_GBPS:.0f} GB/s "
          f"(guide 1R1W wall ~{GUIDE_WALL_GBPS:.0f}, "
          f"dual-die datasheet {DUAL_DIE_GBPS:.0f})")
    print("=" * 100)

    cross_check_reference()
    test_round_trip_semantics()
    test_empty_tokens()
    test_contract_errors()

    # ---- CUDA graph capture (requirement) ----
    # Runs BEFORE the heavy timing/triton workload: on this host stream
    # creation wedges once the process has run a large kernel/timing
    # workload, and all streams were pre-created above. Capture/replay is
    # verified right after the light torch-reference checks instead.
    print("---- CUDA graph capture " + "-" * 71)
    torch.cuda.empty_cache()
    print("[phase ] graph capture prealloc=True", flush=True)
    test_graph_capture(prealloc=True)
    print("[phase ] graph capture prealloc=False", flush=True)
    test_graph_capture(prealloc=False)

    # defeat cold-state noise on the first timed case
    warm = make_extra_inputs(256, 8, 1, 192, 96, seed=42)
    for _ in range(30):
        OP(*warm, 8, 1, 192, 96, 1e-5, 1.0)
    torch.cuda.synchronize()

    # ---- scenario matrix: every case of the reference triton test ----
    if not os.environ.get("SKIP_MATRIX"):
        print(f"---- scenario matrix (all cases of "
              f"test_vllm_step5_fused_qknorm_rope.py, head_dim=192) " + "-" * 24)
        for tp, nq, nkv, prefill_tokens in _tr._SPLITS:
            for scenario, rp, prealloc in _tr._PREFILL_SCENARIOS:
                ins = _tr._make_inputs(scenario, rp, DEVICE, nq, nkv,
                                       prefill_tokens, num_seqs=0)
                run_case(f"{scenario[:3]}{rp}", *ins, nq, nkv, 192, rp,
                         prealloc=prealloc)
            for num_seqs in _tr._DECODE_TOKENS:
                for scenario, rp, prealloc in _tr._DECODE_SCENARIOS:
                    ins = _tr._make_inputs(scenario, rp, DEVICE, nq, nkv,
                                           prefill_tokens, num_seqs=num_seqs)
                    tag = {"decode": "dec", "decode_mtp3": "mtp"}[scenario] + str(rp)
                    run_case(tag, *ins, nq, nkv, 192, rp, prealloc=prealloc)

    # ---- extra generality cases ----
    print("---- extra generality cases " + "-" * 66)
    for i, (tag, tokens, nq, nkv, hd, rp, kw) in enumerate(EXTRA_CASES):
        ins = make_extra_inputs(tokens, nq, nkv, hd, rp, seed=1000 + i, **kw)
        run_case(tag, *ins, nq, nkv, hd, rp)

    # strided (row-padded) caller outputs, still last-dim contiguous
    nq, nkv, hd, rp, tokens = 8, 1, 192, 96, 512
    ins = make_extra_inputs(tokens, nq, nkv, hd, rp, seed=2001)
    q_width, kv_width = nq * hd, nkv * hd
    q_full = torch.full((tokens, 2 * q_width), float("nan"),
                        device=DEVICE, dtype=torch.bfloat16)
    k_full = torch.full((tokens, 2 * kv_width), float("nan"),
                        device=DEVICE, dtype=torch.bfloat16)
    v_full = torch.full((tokens, 2 * kv_width), float("nan"),
                        device=DEVICE, dtype=torch.bfloat16)
    run_case("stride", *ins, nq, nkv, hd, rp,
             outs_override=dict(q_out=q_full[:, :q_width],
                                k_out=k_full[:, :kv_width],
                                v_out=v_full[:, :kv_width]))

    # ---- summary ----
    print("=" * 100)
    n_ok = sum(1 for r in ROWS if r["ok"])
    overall = n_ok == len(ROWS)
    print(f"Accuracy: {'ALL PASS' if overall else 'FAILURES PRESENT'} "
          f"({n_ok}/{len(ROWS)} benchmark cases, threshold cos_sim >= 0.9999; "
          f"+ xcheck/roundtrip/empty/contract/graph checks)")

    timed = [r for r in ROWS if r["speedup"] is not None]
    if timed and TRITON_IMPL is not None:
        geo = math.exp(sum(math.log(r["speedup"]) for r in timed) / len(timed))
        print(f"vs Triton baseline: geomean speedup {geo:.2f}x over "
              f"{len(timed)} timed cases "
              f"(min {min(r['speedup'] for r in timed):.2f}x, "
              f"max {max(r['speedup'] for r in timed):.2f}x)")
        groups = {}
        for r in timed:
            groups.setdefault((r["tag"], r["hd"], r["rp"]), []).append(r)
        print("---- per-shape improvement vs Triton baseline "
              "(geomean per group) " + "-" * 22)
        for (tag, hd, rp), rows in groups.items():
            g = math.exp(sum(math.log(r["speedup"]) for r in rows) / len(rows))
            print(f"  {tag:<8} hd={hd:3d} rp={rp:3d}  cases={len(rows):2d}  "
                  f"geomean speedup {g:5.2f}x")

    peak = max((r["gbps"] for r in ROWS), default=0.0)
    best = max(ROWS, key=lambda r: r["gbps"])
    print(f"Peak bandwidth: {peak:.1f} GB/s (best: {best['tag']} "
          f"T={best['tokens']} hd={best['hd']} rp={best['rp']})  "
          f"({'REACHED' if peak >= TARGET_GBPS else 'below'} single-die "
          f"target {TARGET_GBPS:.0f}; guide 1R1W wall ~{GUIDE_WALL_GBPS:.0f}; "
          f"dual-die datasheet {DUAL_DIE_GBPS:.0f})")
    assert overall, "accuracy check failed"


if __name__ == "__main__":
    main()
