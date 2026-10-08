"""Unit test + bandwidth benchmark for mcoplib.op.moe_swiglu_dynamic_quantize
on MetaX C600-U.

Full requested matrix:
  dtype = bfloat16
  dst_dtype in {int8, float8_e4m3fn}
  ep_size in {1, 8}
  experts_per_rank = num_experts / ep_size
  num_experts = 128, topk = 8, hidden_size = 1536
  num_tokens = the full sweep below

Parallel shape convention (matching xpu-perf):
  routed rows = num_tokens * topk / ep_size

Op semantics (per routed token t belonging to local expert e):
  gate, up = scatter_tokens[t, :H], scatter_tokens[t, H:2H]      (bf16)
  g        = silu(gate) * up * smooth_scale[e]                   (fp32)
  qmax     = 127 (int8)  |  448 (fp8 e4m3)
  scale[t] = max(|g|) / qmax
  y[t]     = quant(g / scale[t])   -> int8 (round) | fp8_e4m3 (cast)

Accuracy is checked against a torch reference that runs the SAME dynamic-quant
algorithm to the SAME dst dtype, then compares dequantized values (this tests
kernel correctness, not the inherent quant loss):
  int8  cos_sim >= 0.999
  fp8   cos_sim >= 0.99999
and the per-token fp32 scale tensor cos_sim >= 0.99999 for both.

Core bandwidth uses the main scatter/output logical bytes per routed token:
  eff_bytes = num_routed * (2*H*2  [bf16 gate+up read]
                            + H*1   [1-byte int8/fp8 out write]
                            + 4)    [fp32 scale write]
The xpu-perf-compatible logical bandwidth additionally counts the local smooth
table and the two expert metadata tensors once per invocation. These are
logical byte-count metrics, not hardware-counter measurements of HBM traffic.

Run:
  export CUDA_VISIBLE_DEVICES=<free gpu>
  python unit_test/test_op_moe_swiglu_dynamic_quantize.py

The default sweep extends to 131072 tokens to confirm the steady-state tail.
Use MOE_SWIGLU_MIN_TOKENS and MOE_SWIGLU_MAX_TOKENS to select a subset.
"""

import os
import torch

import mcoplib.op as ops

DEVICE = "cuda"
TARGET_GBPS = 1300.0               # single-die gate

COS_INT8 = 0.999
COS_FP8 = 0.99999
COS_SCALE = 0.99999

DTYPE = torch.bfloat16
NUM_EXPERTS = 128
EP_SIZES = [1, 8]
TOPK = 8
HIDDEN = 1536                      # output hidden; input is 2*HIDDEN (gate||up)

DST_DTYPES = [("int8", torch.int8, COS_INT8),
              ("fp8", torch.float8_e4m3fn, COS_FP8)]

ALL_NUM_TOKENS_CASES = [
    16, 64, 128, 256, 384, 512, 640, 768, 896, 1024,
    1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328,
    3584, 3840, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
    18432, 20480, 22528, 24576, 26624, 28672, 30720, 32768,
    65536, 98304, 131072,
]
MIN_TOKENS = int(os.environ.get("MOE_SWIGLU_MIN_TOKENS", "16"))
MAX_TOKENS = int(os.environ.get("MOE_SWIGLU_MAX_TOKENS", "131072"))
NUM_TOKENS_CASES = [
    t for t in ALL_NUM_TOKENS_CASES if MIN_TOKENS <= t <= MAX_TOKENS
]
if not NUM_TOKENS_CASES:
    raise ValueError(
        f"no token cases selected: min_tokens={MIN_TOKENS}, "
        f"max_tokens={MAX_TOKENS}"
    )
STEADY_STATE_MIN_TOKENS = 65536
STEADY_STATE_TOL = 0.02


def build_expert_counts(num_routed, num_experts, seed):
    """Roughly-uniform contiguous routing: split num_routed rows over experts."""
    g = torch.Generator().manual_seed(seed)
    base = num_routed // num_experts
    counts = torch.full((num_experts,), base, dtype=torch.int64)
    rem = num_routed - base * num_experts
    if rem > 0:
        idx = torch.randperm(num_experts, generator=g)[:rem]
        counts[idx] += 1
    if base > 4:
        noise = torch.randint(-base // 4, base // 4 + 1, (num_experts,), generator=g)
        counts = (counts + noise).clamp(min=0)
        diff = int(num_routed - counts.sum().item())
        i = 0
        while diff != 0:
            j = i % num_experts
            if diff > 0:
                counts[j] += 1
                diff -= 1
            elif counts[j] > 0:
                counts[j] -= 1
                diff += 1
            i += 1
    assert int(counts.sum().item()) == num_routed
    return counts.to(torch.int32)


def _cos_from_dots(dot_ab, na, nb):
    denom = (na ** 0.5) * (nb ** 0.5)
    return (dot_ab / denom) if denom > 0 else 1.0


def benchmark(fn, warmup=15, rep=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        st[i].record(); fn(); en[i].record()
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(st, en)])
    return times.median().item()  # ms


def run():
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    print(f"CUDA_VISIBLE_DEVICES='{dev}'  device_count={torch.cuda.device_count()}")
    print(f"config: num_experts={NUM_EXPERTS} topk={TOPK} "
          f"ep_sizes={EP_SIZES} hidden={HIDDEN} dtype=bf16 "
          f"tokens=[{MIN_TOKENS},{MAX_TOKENS}]")
    print(f"gates: int8 cos>={COS_INT8}  fp8 cos>={COS_FP8}  scale cos>={COS_SCALE}")

    all_pass = True
    peak = {"int8": (0.0, ""), "fp8": (0.0, "")}
    # Per-case results used by the steady-state summary.
    results = []

    for ep_size in EP_SIZES:
        assert NUM_EXPERTS % ep_size == 0
        experts_per_rank = NUM_EXPERTS // ep_size
        for dname, dtdtype, cos_gate in DST_DTYPES:
            qmax = 127.0 if dname == "int8" else 448.0
            print("=" * 100)
            print(f"### ep_size={ep_size}  "
                  f"experts_per_rank={experts_per_rank}  "
                  f"dst_dtype={dname}")
            print("=" * 100)
            for T in NUM_TOKENS_CASES:
                total_routes = T * TOPK
                assert total_routes % ep_size == 0
                num_routed = total_routes // ep_size
                seed = 42 + T + ep_size * 7 + (0 if dname == "int8" else 3)
                counts = build_expert_counts(num_routed, experts_per_rank, seed)
                starts = torch.zeros(experts_per_rank, dtype=torch.int32)
                starts[1:] = torch.cumsum(counts, 0)[:-1].to(torch.int32)

                g = torch.Generator(device=DEVICE).manual_seed(seed)
                try:
                    scatter = torch.randn(num_routed, 2 * HIDDEN, dtype=DTYPE,
                                          device=DEVICE, generator=g)
                    smooth = torch.randn(experts_per_rank, HIDDEN,
                                         dtype=torch.float32, device=DEVICE,
                                         generator=g) * 0.5
                    counts_d = counts.to(DEVICE)
                    starts_d = starts.to(DEVICE)
                    y = torch.zeros(num_routed, HIDDEN, dtype=dtdtype, device=DEVICE)
                    pts = torch.zeros(num_routed, dtype=torch.float32, device=DEVICE)
                except RuntimeError as ex:
                    print(f"[skip] ep={ep_size} T={T} routes={num_routed} "
                          f"alloc OOM: {ex}")
                    torch.cuda.empty_cache()
                    continue

                ops.moe_swiglu_dynamic_quantize(
                    scatter, smooth, starts_d, counts_d,
                    y, pts, experts_per_rank)
                torch.cuda.synchronize()

                # Streaming accuracy: accumulate per-expert dot products so
                # the large-shape reference never materializes a full [R,H]
                # temporary tensor.
                H = HIDDEN
                x1_all = scatter[:, :H]
                x2_all = scatter[:, H:2 * H]
                dot_y = 0.0; ny_c = 0.0; ny_r = 0.0
                dot_s = 0.0; ns_c = 0.0; ns_r = 0.0
                starts_l = starts.tolist(); counts_l = counts.tolist()
                for e in range(experts_per_rank):
                    s = starts_l[e]; c = counts_l[e]
                    if c == 0:
                        continue
                    x1 = x1_all[s:s + c].float()
                    x2 = x2_all[s:s + c].float()
                    blk = (torch.nn.functional.silu(x1) * x2
                           * smooth[e].unsqueeze(0))
                    rs = blk.abs().amax(dim=-1) / qmax
                    rs_safe = torch.where(rs > 0, rs, torch.ones_like(rs))
                    r = blk / rs_safe.unsqueeze(-1)
                    if dtdtype == torch.int8:
                        q = torch.round(r).clamp(-qmax, qmax)
                    else:
                        q = (r.clamp(-qmax, qmax)
                             .to(torch.float8_e4m3fn).float())
                    deq_ref = q * rs.unsqueeze(-1)
                    deq_cuda = (y[s:s + c].float()
                                * pts[s:s + c].unsqueeze(-1))
                    dot_y += (deq_cuda * deq_ref).sum().item()
                    ny_c += (deq_cuda * deq_cuda).sum().item()
                    ny_r += (deq_ref * deq_ref).sum().item()
                    sc_cuda = pts[s:s + c]
                    dot_s += (sc_cuda * rs).sum().item()
                    ns_c += (sc_cuda * sc_cuda).sum().item()
                    ns_r += (rs * rs).sum().item()
                    del x1, x2, blk, r, q, deq_ref, deq_cuda
                cs_y = _cos_from_dots(dot_y, ny_c, ny_r)
                cs_s = _cos_from_dots(dot_s, ns_c, ns_r)
                ok = (cs_y >= cos_gate) and (cs_s >= COS_SCALE)
                all_pass = all_pass and ok
                torch.cuda.empty_cache()

                ms = benchmark(lambda: ops.moe_swiglu_dynamic_quantize(
                    scatter, smooth, starts_d, counts_d, y, pts, experts_per_rank))

                core_bytes = num_routed * (2 * HIDDEN * 2 + HIDDEN * 1 + 4)
                smooth_bytes = experts_per_rank * HIDDEN * 4
                metadata_bytes = experts_per_rank * 2 * 4  # starts + counts
                logical_bytes = core_bytes + smooth_bytes + metadata_bytes
                gbps = core_bytes / (ms * 1e-3) / 1e9
                logical_gbps = logical_bytes / (ms * 1e-3) / 1e9
                if gbps > peak[dname][0]:
                    peak[dname] = (
                        gbps,
                        f"ep={ep_size} T={T} routes={num_routed}",
                    )
                results.append((ep_size, dname, T, num_routed,
                                cs_y, cs_s, ok, ms, gbps))

                accuracy = f"cos_y={cs_y:.6f} cos_s={cs_s:.6f}"
                tag = "OK " if ok else "BAD"
                print(f"[{dname:4}] ep={ep_size} T={T:<6} "
                      f"routes={num_routed:<7} "
                      f"{accuracy} {tag} "
                      f"{ms:8.4f} ms  core={gbps:8.1f} GB/s  "
                      f"xpu-logical={logical_gbps:8.1f} GB/s")

                del scatter, smooth, y, pts, counts_d, starts_d
                torch.cuda.empty_cache()

    print("=" * 100)
    print(f"Accuracy: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    for dn, _, _ in DST_DTYPES:
        gv, desc = peak[dn]
        print(f"Peak {dn}: {gv:.1f} GB/s  [{desc}]  "
              f"(target {TARGET_GBPS:.0f} -> "
              f"{'REACHED' if gv >= TARGET_GBPS else 'NOT reached'})")

    print("=" * 100)
    print(f"Steady-state tail (tokens >= {STEADY_STATE_MIN_TOKENS}, "
          f"plateau tolerance={STEADY_STATE_TOL:.0%})")
    for ep_size in EP_SIZES:
        for dname, _, _ in DST_DTYPES:
            tail = sorted(
                (r for r in results
                 if r[0] == ep_size and r[1] == dname
                 and r[2] >= STEADY_STATE_MIN_TOKENS),
                key=lambda r: r[2],
            )
            if not tail:
                print(f"[{dname:4}] ep={ep_size}: no completed large-shape case")
                continue
            points = ", ".join(
                f"T={r[2]} routes={r[3]} {r[8]:.1f}GB/s" for r in tail
            )
            if len(tail) >= 2:
                bws = [r[8] for r in tail]
                spread = (max(bws) - min(bws)) / max(bws)
                state = "PLATEAU" if spread <= STEADY_STATE_TOL else "STILL RISING"
                print(f"[{dname:4}] ep={ep_size}: {points}; "
                      f"tail_spread={spread:.2%} {state}")
            else:
                print(f"[{dname:4}] ep={ep_size}: {points}; need >=2 points")
    if not all_pass:
        raise AssertionError("moe_swiglu_dynamic_quantize accuracy check failed")
    return results


if __name__ == "__main__":
    run()
