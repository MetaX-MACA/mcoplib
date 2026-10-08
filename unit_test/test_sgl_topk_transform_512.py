# SPDX-License-Identifier: Apache-2.0
"""Unit test + bandwidth benchmark for topk_transform_v1 (C500-optimized port
of the JIT topk_transform_kernel in op/sglang/jit_kernels/topk_v1.cuh).

Test data and benchmark methodology aligned with unit_test/test_topk.py
(same bs/seq_len grid, same bytes_accessed formula, same warmup/iters).
"""

import pytest
import torch

import mcoplib.sgl_kernel  # noqa: F401  (registers torch.ops.sgl_kernel)

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

    # reference
    ref_topk = _ref_torch_impl(score, seq_len, TOPK)
    ref_page = _ref_page_to_indices(ref_topk, page_table, PAGE_SIZE)

    page_indices = score.new_empty((bs, TOPK), dtype=torch.int32)
    raw_indices = score.new_empty((bs, TOPK), dtype=torch.int32)

    def launch():
        torch.ops.sgl_kernel.topk_transform_v1.default(
            score, seq_lens, page_indices, page_table, PAGE_SIZE, raw_indices
        )

    # correctness: compare raw topk indices (tie-tolerant) and page transform
    launch()
    raw_ref_sorted = torch.sort(ref_topk, dim=-1).values
    raw_our_sorted = torch.sort(raw_indices, dim=-1).values
    assert_equal(score, raw_ref_sorted, raw_our_sorted, bs, TOPK, seq_len)
    # page transform must equal page_to_indices(raw_indices)
    recomputed_page = _ref_page_to_indices(raw_indices, page_table, PAGE_SIZE)
    assert torch.equal(recomputed_page, page_indices), f"page transform mismatch bs={bs} seq={seq_len}"

    # bytes_accessed aligned with unit_test/test_topk.py::test_topk_transform_kernel:
    # Read Score + Read SrcPageTable + Write DstPageTable = bs * (2 * seq_len + k) * 4
    bytes_accessed = bs * (2 * seq_len + TOPK) * 4
    print(f"\n[Case: BS={bs}, SeqLen={seq_len}, PageSize={PAGE_SIZE}, K={TOPK}]")
    bw = run_cuda_benchmark("topk_transform_v1", launch, bytes_accessed)

    # Target: average bandwidth >= 800 GB/s for seq_len >= 4096
    # if seq_len >= 4096:
    #     assert bw >= 800.0, f"bandwidth {bw:.1f} GB/s below target 800 GB/s (bs={bs}, seq={seq_len})"


@pytest.mark.parametrize("seq_len", [4096, 20000], ids=["histogram", "radix"])
@torch.inference_mode()
def test_topk_transform_v1_cuda_graph_replay(seq_len: int) -> None:
    """Captured execution must replay correctly with changed score contents."""
    bs = 2
    score = torch.randn(bs, seq_len, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    page_table = torch.arange(TABLE_LEN, dtype=torch.int32, device="cuda")
    page_table = page_table.unsqueeze(0).expand(bs, -1).contiguous()
    page_indices = torch.empty((bs, TOPK), dtype=torch.int32, device="cuda")
    raw_indices = torch.empty_like(page_indices)

    def launch() -> None:
        torch.ops.sgl_kernel.topk_transform_v1.default(
            score, seq_lens, page_indices, page_table, PAGE_SIZE, raw_indices
        )

    # Initialize one-time function attributes before capture.
    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            launch()
    except Exception as exc:
        pytest.fail(
            f"topk_transform_v1 graph capture failed for {seq_len=}: "
            f"{type(exc).__name__}: {exc}"
        )

    for seed in (101, 202):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        score.copy_(torch.randn(score.shape, generator=generator, device="cuda"))
        graph.replay()
        # Keep raw indices in their original (unsorted) order for the page
        # mapping check. Only sort copies for the unordered Top-K comparison.
        graph_raw = raw_indices.clone()
        graph_raw_sorted = torch.sort(graph_raw, dim=-1).values
        graph_page = page_indices.clone()

        launch()
        torch.cuda.synchronize()
        eager_raw = torch.sort(raw_indices, dim=-1).values
        assert_equal(score, eager_raw, graph_raw_sorted, bs, TOPK, seq_len)
        expected_page = _ref_page_to_indices(graph_raw, page_table, PAGE_SIZE)
        assert torch.equal(expected_page, graph_page), (
            f"graph page transform mismatch for {seq_len=}, {seed=}"
        )


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
