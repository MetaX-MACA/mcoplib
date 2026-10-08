"""Standalone checks + benchmark for dequantize_and_gather_k_cache.

Run directly:  python test_dequantize_and_gather_k_cache.py

Layouts under test
------------------
FP8  (use_int8=False):
    token data : 576 B  = [0:448] fp8 (448 dims) + [448:576] bf16 (64 dims)
    scale      :   8 B  = 7 x UE8M0 (uint8), quant_block = 64
    output dim : 512

INT8 (use_int8=True):
    token data : 512 B  = 512 x int8
    scale      :  64 B  = 16 x fp32, quant_block = 32
    output dim : 512
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import torch

# Adjust this import path to wherever the op lives in your tree.
from mcoplib.triton_dequantize_and_gather_k_cache import (
    dequantize_and_gather_k_cache,
)

# --------------------------------------------------------------------------- #
# Layout descriptors
# --------------------------------------------------------------------------- #

OUTPUT_DIM = 512

WARMUP_ITERS = 5
BENCH_ITERS = 20


@dataclass(frozen=True)
class Layout:
    use_int8: bool
    quant_dim: int        # number of quantized elements per token
    quant_block: int      # elements per quantization group
    n_quant_blocks: int
    raw_bf16_dim: int     # un-quantized trailing dims (fp8 layout only)
    token_data_size: int  # bytes
    scale_dim: int        # bytes of scale per token


FP8_LAYOUT = Layout(
    use_int8=False,
    quant_dim=448,
    quant_block=64,
    n_quant_blocks=7,
    raw_bf16_dim=64,
    token_data_size=576,   # 448 fp8 + 64 * 2 B bf16
    scale_dim=8,           # 7 used + 1 pad
)

INT8_LAYOUT = Layout(
    use_int8=True,
    quant_dim=512,
    quant_block=32,
    n_quant_blocks=16,
    raw_bf16_dim=0,
    token_data_size=512,
    scale_dim=64,          # 16 * 4 B
)


def layout_for(use_int8: bool) -> Layout:
    return INT8_LAYOUT if use_int8 else FP8_LAYOUT


def is_fp8_fnuz() -> bool:
    """FNUZ (e4m3fnuz) is the ROCm/gfx94x encoding; CUDA uses e4m3fn."""
    if getattr(torch.version, "hip", None) is None:
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return False
    return any(tag in arch for tag in ("gfx940", "gfx941", "gfx942", "gfx90a"))


def fnuz_variants(use_int8: bool) -> list[bool]:
    """INT8 has no encoding variants; FP8 must be tested in the platform's mode."""
    return [False] if use_int8 else [is_fp8_fnuz()]


def dtype_tag(use_int8: bool, use_fnuz: bool) -> str:
    if use_int8:
        return "int8"
    return "fp8-fnuz" if use_fnuz else "fp8-fn"


def compute_block_stride(layout: Layout, cache_block_size: int) -> int:
    """Mirror the host-side `dispatch` padding logic."""
    unpadded = cache_block_size * (layout.token_data_size + layout.scale_dim)
    tok = layout.token_data_size
    return ((unpadded + tok - 1) // tok) * tok


def fp8_dtype_and_max(use_fnuz: bool) -> tuple[torch.dtype, float]:
    if use_fnuz:
        return torch.float8_e4m3fnuz, 240.0
    return torch.float8_e4m3fn, 448.0


# --------------------------------------------------------------------------- #
# Reference quantizer / dequantizer (pure PyTorch)
# --------------------------------------------------------------------------- #


def quantize_fp8_ue8m0(x, quant_block, use_fnuz):
    """Block-wise FP8 quantization with power-of-two (UE8M0) scales."""
    fp8_dt, fp8_max = fp8_dtype_and_max(use_fnuz)
    n, d = x.shape
    g = d // quant_block
    xg = x.float().reshape(n, g, quant_block)

    amax = xg.abs().amax(dim=-1).clamp(min=1e-30)
    exp = torch.ceil(torch.log2(amax / fp8_max))
    encoded = (exp + 127.0).clamp(0, 255).to(torch.uint8)
    # Decode exactly the way the kernel does, so ref and kernel agree bit-for-bit.
    scale = torch.exp2(encoded.float() - 127.0)

    q = (xg / scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(fp8_dt)
    deq = (q.float() * scale.unsqueeze(-1)).reshape(n, d).to(torch.bfloat16)

    packed = q.reshape(n, d).view(torch.uint8)
    return packed, encoded, deq


def quantize_int8(x, quant_block):
    """Block-wise symmetric INT8 quantization with fp32 scales."""
    n, d = x.shape
    g = d // quant_block
    xg = x.float().reshape(n, g, quant_block)

    amax = xg.abs().amax(dim=-1).clamp(min=1e-30)
    scale = (amax / 127.0).float()

    q = torch.round(xg / scale.unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
    deq = (q.float() * scale.unsqueeze(-1)).reshape(n, d).to(torch.bfloat16)

    packed = q.reshape(n, d).view(torch.uint8)
    return packed, scale, deq


# --------------------------------------------------------------------------- #
# Cache builder
# --------------------------------------------------------------------------- #


def build_cache(
    layout, seq_lens, block_table, cache_block_size, num_blocks,
    use_fnuz, device, seed=0,
):
    """Create a paged, quantized K cache plus the exact reference dequantization."""
    gen = torch.Generator(device=device).manual_seed(seed)
    block_stride = compute_block_stride(layout, cache_block_size)

    # Fill with garbage so any out-of-range read is obviously wrong.
    k_cache = torch.randint(
        0, 256, (num_blocks, block_stride), dtype=torch.uint8, device=device
    )

    refs: list[torch.Tensor] = []

    for b, seq_len in enumerate(seq_lens):
        # Wide dynamic range across dims exercises per-group scaling.
        base = torch.randn(
            seq_len, OUTPUT_DIM, generator=gen, device=device, dtype=torch.float32
        )
        dim_scale = torch.exp2(
            torch.randint(-8, 8, (1, OUTPUT_DIM), generator=gen, device=device).float()
        )
        orig = (base * dim_scale).to(torch.bfloat16)

        ref = torch.empty_like(orig)

        qpart = orig[:, : layout.quant_dim]
        if layout.use_int8:
            packed, scales, deq = quantize_int8(qpart, layout.quant_block)
            scale_bytes = scales.contiguous().view(torch.uint8)  # [N, G*4]
        else:
            packed, scales, deq = quantize_fp8_ue8m0(qpart, layout.quant_block, use_fnuz)
            scale_bytes = scales  # already uint8 [N, G]

        ref[:, : layout.quant_dim] = deq

        if layout.raw_bf16_dim:
            raw = orig[:, layout.quant_dim :].contiguous()
            ref[:, layout.quant_dim :] = raw
            raw_bytes = raw.view(torch.uint8)  # [N, raw_bf16_dim*2]
        else:
            raw_bytes = None

        refs.append(ref)

        # Scatter token-by-token into the paged cache.
        for pos in range(seq_len):
            blk = block_table[b, pos // cache_block_size].item()
            slot = pos % cache_block_size

            data_off = slot * layout.token_data_size
            k_cache[blk, data_off : data_off + packed.shape[1]] = packed[pos]
            if raw_bytes is not None:
                off = data_off + layout.quant_dim
                k_cache[blk, off : off + raw_bytes.shape[1]] = raw_bytes[pos]

            scale_off = (
                cache_block_size * layout.token_data_size + slot * layout.scale_dim
            )
            used = scale_bytes.shape[1]
            k_cache[blk, scale_off : scale_off + used] = scale_bytes[pos]

    return k_cache, refs


def make_block_table(seq_lens, cache_block_size, num_blocks, device, seed=1):
    """Non-identity, shuffled mapping so logical != physical block index."""
    max_blocks = max(math.ceil(s / cache_block_size) for s in seq_lens)
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(num_blocks, generator=g)

    table = torch.full((len(seq_lens), max_blocks), -1, dtype=torch.int32, device=device)
    cursor = 0
    for b, s in enumerate(seq_lens):
        n = math.ceil(s / cache_block_size)
        assert cursor + n <= num_blocks, "not enough physical blocks"
        table[b, :n] = perm[cursor : cursor + n].to(device=device, dtype=torch.int32)
        cursor += n
    return table


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #

PASS = "[PASS]"
FAIL = "[ERROR]"


class Report:
    """Collects per-case check results and prints them in the [ACC] block style."""

    def __init__(self, title: str):
        self.title = title
        self.lines: list[str] = []
        self.ok = True

    def check(self, name: str, ok: bool, detail: str):
        self.ok &= bool(ok)
        self.lines.append(f"    {name:<18} {detail}  {PASS if ok else FAIL}")

    def emit(self):
        print(f"[ACC] {self.title}")
        for line in self.lines:
            print(line)
        print()


@dataclass
class PerfRow:
    case: str
    dtype: str
    cache_block: int
    num_reqs: int
    tokens: int          # tokens actually emitted
    offset: int
    kernel_ms: float
    tflops: float
    gbps: float
    passed: bool


# --------------------------------------------------------------------------- #
# Benchmark helper
# --------------------------------------------------------------------------- #


def benchmark_case(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS) -> float:
    """Median-free mean over `iters` CUDA-event-timed launches (ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def perf_metrics(layout: Layout, tokens: int, ms: float) -> tuple[float, float]:
    """Return (TFLOPS, GB/s) for a dequant+gather of `tokens` tokens.

    FLOPs  : one multiply + one convert per quantized element.
    Bytes  : payload + scales read, bf16 output written.
    """
    if ms <= 0.0 or tokens == 0:
        return 0.0, 0.0
    flops = tokens * layout.quant_dim * 2.0
    read_b = tokens * (layout.token_data_size + layout.scale_dim)
    write_b = tokens * OUTPUT_DIM * 2
    secs = ms * 1e-3
    return flops / secs / 1e12, (read_b + write_b) / secs / 1e9


# --------------------------------------------------------------------------- #
# Core case driver
# --------------------------------------------------------------------------- #


def run_case(
    *,
    name: str,
    use_int8: bool,
    seq_lens_list: list[int],
    gather_lens_list: list[int] | None,
    cache_block_size: int,
    offset: int,
    use_fnuz: bool,
    device: torch.device,
    seed: int = 0,
    extra_checks=None,
    perf_rows: list[PerfRow] | None = None,
):
    layout = layout_for(use_int8)
    num_reqs = len(seq_lens_list)

    # Over-allocate blocks so the shuffled table is genuinely non-contiguous.
    needed = sum(math.ceil(s / cache_block_size) for s in seq_lens_list)
    num_blocks = needed + 8

    block_table = make_block_table(
        seq_lens_list, cache_block_size, num_blocks, device, seed=seed + 1
    )
    k_cache, refs = build_cache(
        layout, seq_lens_list, block_table, cache_block_size,
        num_blocks, use_fnuz, device, seed=seed,
    )

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
    if gather_lens_list is None:
        gather_lens = None
        eff = list(seq_lens_list)
    else:
        gather_lens = torch.tensor(gather_lens_list, dtype=torch.int32, device=device)
        eff = list(gather_lens_list)

    max_tokens = offset + max(eff)

    # Poison the output: untouched slots must stay poisoned.
    POISON = 1234.0
    out = torch.full(
        (num_reqs, max_tokens, OUTPUT_DIM), POISON, dtype=torch.bfloat16, device=device
    )

    def launch():
        dequantize_and_gather_k_cache(
            out, k_cache, seq_lens, gather_lens, block_table,
            cache_block_size, offset, use_int8=use_int8, use_fnuz=use_fnuz,
        )

    launch()
    torch.cuda.synchronize()

    dtype = dtype_tag(use_int8, use_fnuz)
    rep = Report(
        f"{name}  dtype={dtype}  cache_block={cache_block_size}  offset={offset}  "
        f"seq_lens={seq_lens_list}  gather={gather_lens_list}"
    )

    # ---- 1. exact match against the reference dequantization ---------------
    max_abs = 0.0
    bad_rows = 0
    total_rows = 0
    for b in range(num_reqs):
        n = eff[b]
        if n == 0:
            continue
        start = seq_lens_list[b] - n
        expected = refs[b][start : start + n].float()
        got = out[b, offset : offset + n].float()
        diff = (got - expected).abs()
        max_abs = max(max_abs, float(diff.max()))
        bad_rows += int((diff.amax(dim=-1) > 0).sum())
        total_rows += n

    rep.check(
        "exact dequant",
        max_abs == 0.0,
        f"max_abs={max_abs:.4e}  mismatch_rows={bad_rows}/{total_rows} "
        f"({(bad_rows / max(total_rows, 1) * 100):.2f}%)",
    )

    # ---- 2. untouched region must be untouched ----------------------------
    pre_bad = int((out[:, :offset] != POISON).sum()) if offset > 0 else 0
    rep.check("pre-offset poison", pre_bad == 0, f"mismatch={pre_bad}")

    tail_bad = 0
    for b in range(num_reqs):
        tail = out[b, offset + eff[b] :]
        if tail.numel():
            tail_bad += int((tail != POISON).sum())
    rep.check("tail poison", tail_bad == 0, f"mismatch={tail_bad}")

    if extra_checks is not None:
        extra_checks(rep, out, refs, eff, layout)

    # ---- 3. timing ---------------------------------------------------------
    kernel_ms = benchmark_case(launch)
    tflops, gbps = perf_metrics(layout, total_rows, kernel_ms)
    rep.check(
        "perf",
        True,
        f"kernel={kernel_ms:.4f} ms  {tflops:.2f} TFLOPS  {gbps:.1f} GB/s",
    )

    rep.emit()

    if perf_rows is not None:
        perf_rows.append(
            PerfRow(
                case=name,
                dtype=dtype,
                cache_block=cache_block_size,
                num_reqs=num_reqs,
                tokens=total_rows,
                offset=offset,
                kernel_ms=kernel_ms,
                tflops=tflops,
                gbps=gbps,
                passed=rep.ok,
            )
        )

    return rep.ok


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #


def gather_all_tokens(device, results, perf):
    """gather_lens=None -> every token of every sequence is emitted."""
    for use_int8 in (False, True):
        for cache_block_size in (16, 64, 128):
            for offset in (0, 1, 2, 16):
                for use_fnuz in fnuz_variants(use_int8):
                    results.append(
                        run_case(
                            name="gather_all",
                            use_int8=use_int8,
                            # Deliberately not multiples of cache_block_size.
                            seq_lens_list=[1, 7, 63, 65, 200],
                            gather_lens_list=None,
                            cache_block_size=cache_block_size,
                            offset=offset,
                            use_fnuz=use_fnuz,
                            device=device,
                            perf_rows=perf,
                        )
                    )


def partial_gather(device, results, perf):
    """gather_lens selects only the trailing window of each sequence."""
    seq_lens = [10, 130, 257, 512]
    gather_lens = [1, 7, 130, 512]  # includes full-length and single-token cases
    for use_int8 in (False, True):
        for cache_block_size in (64, 128):
            for use_fnuz in fnuz_variants(use_int8):
                results.append(
                    run_case(
                        name="partial_gather",
                        use_int8=use_int8,
                        seq_lens_list=seq_lens,
                        gather_lens_list=gather_lens,
                        cache_block_size=cache_block_size,
                        offset=2,
                        use_fnuz=use_fnuz,
                        device=device,
                        perf_rows=perf,
                    )
                )


def more_tokens_than_workers(device, results, perf):
    """Exercise the grid-stride loop past NUM_WORKERS=128."""
    for use_int8 in (False, True):
        for use_fnuz in fnuz_variants(use_int8):
            results.append(
                run_case(
                    name="many_tokens",
                    use_int8=use_int8,
                    seq_lens_list=[1000],
                    gather_lens_list=None,
                    cache_block_size=64,
                    offset=0,
                    use_fnuz=use_fnuz,
                    device=device,
                    perf_rows=perf,
                )
            )


def zero_gather_len(device, results, perf):
    """gather_len == 0 must be a no-op for that request."""
    for use_int8 in (False, True):
        for use_fnuz in fnuz_variants(use_int8):
            results.append(
                run_case(
                    name="zero_gather",
                    use_int8=use_int8,
                    seq_lens_list=[64, 64],
                    gather_lens_list=[0, 64],
                    cache_block_size=64,
                    offset=0,
                    use_fnuz=use_fnuz,
                    device=device,
                    perf_rows=perf,
                )
            )


def throughput_sweep(device, results, perf):
    """Large single-request sweep: the shape where TFLOPS/GB/s actually saturate."""
    for use_int8 in (False, True):
        for use_fnuz in fnuz_variants(use_int8):
            for seq_len in (4096, 8192, 16384, 32768, 65536):
                results.append(
                    run_case(
                        name="throughput",
                        use_int8=use_int8,
                        seq_lens_list=[seq_len],
                        gather_lens_list=None,
                        cache_block_size=128,
                        offset=0,
                        use_fnuz=use_fnuz,
                        device=device,
                        perf_rows=perf,
                    )
                )


def quantization_accuracy(device, results):
    """Round-trip error vs. the pre-quantization tensor stays within budget.

    This is a property of the *format*, not the kernel; it guards against a
    silently wrong scale decode (e.g. UE8M0 bias off by one) that would still
    be self-consistent between kernel and reference.
    """
    for use_int8 in (False, True):
        layout = layout_for(use_int8)
        use_fnuz = fnuz_variants(use_int8)[0]

        torch.manual_seed(0)
        x = torch.randn(4096, layout.quant_dim, device=device, dtype=torch.bfloat16)

        if use_int8:
            _, _, deq = quantize_int8(x, layout.quant_block)
            # symmetric int8, 32-wide groups -> ~0.4% relative RMS
            budget = 0.01
        else:
            _, _, deq = quantize_fp8_ue8m0(x, layout.quant_block, use_fnuz)
            # e4m3 has 3 mantissa bits; power-of-two scale costs up to 2x headroom
            budget = 0.08

        err = float((deq.float() - x.float()).norm() / x.float().norm())

        rep = Report(
            f"quantization_accuracy  dtype={dtype_tag(use_int8, use_fnuz)}  "
            f"shape={tuple(x.shape)}"
        )
        rep.check("rel RMS error", err < budget, f"err={err:.4e}  budget={budget:.4e}")
        rep.emit()
        results.append(rep.ok)


# --------------------------------------------------------------------------- #
# Perf table
# --------------------------------------------------------------------------- #


def print_perf_table(rows: list[PerfRow]):
    header = (
        f"{'case':<16}{'dtype':>10}{'cache_blk':>11}{'reqs':>6}"
        f"{'tokens':>9}{'offset':>8}{'kernel ms':>12}{'TFLOPS':>10}{'GB/s':>10}{'':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.case:<16}{r.dtype:>10}{r.cache_block:>11}{r.num_reqs:>6}"
            f"{r.tokens:>9}{r.offset:>8}{r.kernel_ms:>12.4f}"
            f"{r.tflops:>10.2f}{r.gbps:>10.1f}"
            f"{(PASS if r.passed else FAIL):>8}"
        )
    print()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required; skipping.")
        return 0

    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"fp8 encoding: {'e4m3fnuz' if is_fp8_fnuz() else 'e4m3fn'}")
    print(f"warmup={WARMUP_ITERS}  iters={BENCH_ITERS}\n")

    results: list[bool] = []
    perf: list[PerfRow] = []

    gather_all_tokens(device, results, perf)
    partial_gather(device, results, perf)
    more_tokens_than_workers(device, results, perf)
    zero_gather_len(device, results, perf)
    throughput_sweep(device, results, perf)
    quantization_accuracy(device, results)

    n_pass = sum(1 for ok in results if ok)
    n_total = len(results)

    if n_pass == n_total:
        print("=== ACCURACY: ALL PASS ===")
    else:
        print(f"=== ACCURACY: {n_total - n_pass}/{n_total} CASES FAILED ===")
    print()

    print_perf_table(perf)
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())