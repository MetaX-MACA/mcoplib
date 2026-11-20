# SPDX-License-Identifier: Apache-2.0
"""Unit test + bandwidth benchmark for topk_transform_v1 (C500-optimized port
of the JIT topk_transform_kernel in op/sglang/jit_kernels/topk_v1.cuh).

Test data and benchmark methodology aligned with unit_test/test_topk.py
(same bs/seq_len grid, same bytes_accessed formula, same warmup/iters).
"""

import pytest
import torch

import mcoplib.sgl_kernel  # noqa: F401  (registers torch.ops.sgl_kernel)

TOPK_LIST = [512,1024]
TOPK = 512
MAX_SEQ_LEN = 66551
TABLE_LEN = 299062
PAGE_SIZE = 16
MAX_PERMIT_ERROR = 0


def _ref_torch_impl(score: torch.Tensor, seq_len: int, topk: int) -> torch.Tensor:
    assert score.dim() == 2
    return torch.topk(score[:, :seq_len], topk, dim=-1, sorted=False).indices.to(torch.int32)


def _ref_page_to_indices(topk_indices: torch.Tensor, page_table: torch.Tensor, page_size: int) -> torch.Tensor:
    """page_to_indices(i) = (page_table[i >> page_bits] << page_bits) | (i & mask)"""
    page_bits = page_size.bit_length() - 1
    mask = page_size - 1
    page_idx = topk_indices >> page_bits
    rows = torch.arange(page_table.size(0), device=page_table.device).unsqueeze(-1)
    pages = page_table[rows, page_idx]
    return (pages << page_bits) | (topk_indices & mask)


def assert_equal(
    score: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_our: torch.Tensor,
    bs: int,
    k: int,
    seq_len: int,
):
    """Tie-tolerant comparison matching unit_test/test_topk.py::assert_equal."""
    indices_our_cpu = indices_our.cpu().tolist()
    indices_ref_cpu = indices_ref.cpu().tolist()
    for i in range(bs):
        ref_set_i = set(indices_ref_cpu[i])
        our_set_i = set(indices_our_cpu[i])
        more = our_set_i - ref_set_i
        less = ref_set_i - our_set_i
        if len(more) > MAX_PERMIT_ERROR or len(less) > MAX_PERMIT_ERROR:
            more_values = sorted(score[i, idx].item() for idx in more)
            less_values = sorted(score[i, idx].item() for idx in less)
            assert more_values == less_values, (
                f"{bs=}, {k=}, {seq_len=}, {i=}, {more=}, {less=} failed, "
                f"with {more_values=}, {less_values=}"
            )


def run_cuda_benchmark(kernel_name, run_func, bytes_accessed, warmup=5, iters=100):
    for _ in range(warmup):
        run_func()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        run_func()
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / iters
    avg_time_sec = avg_time_ms / 1000.0
    bandwidth_gb_s = (bytes_accessed / avg_time_sec) / 1e9 if avg_time_sec > 0 else 0.0
    print(f"  [PERF] {kernel_name:<30} | Avg Time: {avg_time_ms:8.4f} ms | Bandwidth: {bandwidth_gb_s:7.2f} GB/s")
    return bandwidth_gb_s


@pytest.mark.parametrize("bs", [1, 132, 256, 4096, 1662])
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 66551])
@torch.inference_mode()
def test_topk_transform_v1(bs: int, seq_len: int) -> None:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")

    page_table = torch.arange(0, TABLE_LEN, dtype=torch.int32, device="cuda")
    page_table = page_table.unsqueeze(0).expand(bs, -1).contiguous()
    
    for topk in TOPK_LIST:
        print(f"============topk={topk}==========")
        # reference
        ref_topk = _ref_torch_impl(score, seq_len, topk)
        ref_page = _ref_page_to_indices(ref_topk, page_table, PAGE_SIZE)

        page_indices = score.new_empty((bs, topk), dtype=torch.int32)
        raw_indices = score.new_empty((bs, topk), dtype=torch.int32)

        def launch():
            torch.ops.sgl_kernel.topk_transform_v1.default(
                score, seq_lens, page_indices, page_table, PAGE_SIZE, raw_indices
            )

        # correctness: compare raw topk indices (tie-tolerant) and page transform
        launch()
        raw_ref_sorted = torch.sort(ref_topk, dim=-1).values
        raw_our_sorted = torch.sort(raw_indices, dim=-1).values
        assert_equal(score, raw_ref_sorted, raw_our_sorted, bs, topk, seq_len)
        # page transform must equal page_to_indices(raw_indices)
        recomputed_page = _ref_page_to_indices(raw_indices, page_table, PAGE_SIZE)
        assert torch.equal(recomputed_page, page_indices), f"page transform mismatch bs={bs} seq={seq_len}"

        # bytes_accessed aligned with unit_test/test_topk.py::test_topk_transform_kernel:
        # Read Score + Read SrcPageTable + Write DstPageTable = bs * (2 * seq_len + k) * 4
        bytes_accessed = bs * (2 * seq_len + topk) * 4
        print(f"\n[Case: BS={bs}, SeqLen={seq_len}, PageSize={PAGE_SIZE}, K={topk}]")
        bw = run_cuda_benchmark("topk_transform_v1", launch, bytes_accessed)

        # Target: average bandwidth >= 800 GB/s for seq_len >= 4096
        # if seq_len >= 4096:
        #     assert bw >= 800.0, f"bandwidth {bw:.1f} GB/s below target 800 GB/s (bs={bs}, seq={seq_len})"


def _fp32_keys_cpu(score: torch.Tensor) -> torch.Tensor:
    """Integer order: NaNs greatest, signed zeros equal, subnormals preserved."""
    bits = score.cpu().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    magnitude = bits & 0x7FFFFFFF
    bits = torch.where(magnitude == 0, 0, bits)
    keys = torch.where((bits & 0x80000000) != 0, (~bits) & 0xFFFFFFFF,
                       bits | 0x80000000)
    return torch.where(magnitude > 0x7F800000, 0xFFFFFFFF, keys)


@pytest.mark.parametrize("seq_len", [4096, 20000], ids=["histogram", "radix"])
@pytest.mark.parametrize("topk", TOPK_LIST)
@pytest.mark.parametrize("with_raw", [True, False], ids=["raw", "page_only"])
@torch.inference_mode()
def test_topk_transform_v1_cuda_graph_replay(seq_len: int, topk: int, with_raw: bool) -> None:
    """Verify eager and replay independently, including data-dependent paths."""
    bs = 2
    score = torch.randn(bs, seq_len, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    n_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    page_table = torch.arange(n_pages, dtype=torch.int32, device="cuda")
    page_table = page_table.unsqueeze(0).expand(bs, -1).contiguous()
    page_indices = torch.empty((bs, topk), dtype=torch.int32, device="cuda")
    raw_indices = torch.empty_like(page_indices) if with_raw else None

    def launch() -> None:
        torch.ops.sgl_kernel.topk_transform_v1.default(
            score, seq_lens, page_indices, page_table, PAGE_SIZE, raw_indices
        )

    def check(tag: str) -> None:
        pages = page_indices.cpu().long()
        table = page_table.cpu().long()
        if raw_indices is None:
            # The page table is a permutation, so page-only output is invertible.
            inverse = torch.argsort(table, dim=1)
            physical = (pages.clamp_min(0) >> 4)
            assert bool((physical < n_pages).all()), f"{tag}: invalid physical page"
            raw = (inverse.gather(1, physical) << 4) | (pages.clamp_min(0) & 15)
            raw[pages == -1] = -1
        else:
            raw = raw_indices.cpu().long()
        keys = _fp32_keys_cpu(score)
        for row, length in enumerate(seq_lens.cpu().tolist()):
            selected = raw[row]
            if length <= topk:
                expected = torch.arange(topk)
                expected[length:] = -1
                assert torch.equal(selected, expected), f"{tag}: naive output"
            else:
                assert bool(((selected >= 0) & (selected < length)).all()), tag
                assert selected.unique().numel() == topk, f"{tag}: duplicate index"
                expected = torch.topk(keys[row, :length], topk).values.sort().values
                actual = keys[row, selected].sort().values
                assert torch.equal(actual, expected), f"{tag}: wrong top-k values"
            valid = selected >= 0
            expected_page = torch.full_like(selected, -1)
            expected_page[valid] = ((table[row, selected[valid] >> 4] << 4) |
                                    (selected[valid] & 15))
            assert torch.equal(pages[row], expected_page), f"{tag}: page mapping"

    launch()  # Initialize function attributes outside capture.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    generator = torch.Generator().manual_seed(101)
    for phase in ("normal", "subnormal", "negative_nan", "dense", "normal_again"):
        host = torch.randn((bs, seq_len), generator=generator)
        if phase == "subnormal":
            host = torch.randint(1, 0x800000, (bs, seq_len), generator=generator,
                                 dtype=torch.int32).view(torch.float32)
            host.view(torch.int32)[1] |= -2147483648
        elif phase == "negative_nan":
            host.view(torch.int32)[:, :topk + 17] = -4194304
        elif phase == "dense":
            host = (torch.arange(bs * seq_len, dtype=torch.int32).reshape(bs, seq_len)
                    % 4096 + 0x3F800000).view(torch.float32)
        score.copy_(host)
        seq_lens.copy_(torch.tensor([seq_len, topk - 3 if phase == "dense" else seq_len - 7],
                                    dtype=torch.int32, device="cuda"))
        # Change page contents without changing captured addresses.
        page_table.copy_(torch.arange(n_pages - 1, -1, -1, dtype=torch.int32,
                                      device="cuda").expand(bs, -1))
        graph.replay()
        check(f"graph/{phase}/K{topk}/W{seq_len}")
        launch()
        check(f"eager/{phase}/K{topk}/W{seq_len}")


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-s", "-v", __file__]))
