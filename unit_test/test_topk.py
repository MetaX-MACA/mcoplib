# # SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MiniMax QK RMS-norm and TopK: NCCL reference vs Lamport/SGLang fused kernel."""

import time
import pytest
import torch
import torch.nn as nn
import mcoplib.sgl_kernel

def _ref_torch_impl(score: torch.Tensor, seq_len: int, topk: int) -> torch.Tensor:
    assert score.dim() == 2
    return torch.topk(score[:, :seq_len], topk, dim=-1, sorted=False).indices

def _ref_torch_transform_decode_impl(
    score: torch.Tensor,
    seq_len: int,
    src_page_table: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    batch_size, _ = score.shape
    assert score.shape[0] == src_page_table.shape[0]
    assert seq_len >= topk
    indices = _ref_torch_impl(score, seq_len, topk)
    topk_indices = torch.empty(
        (batch_size, topk), dtype=torch.int32, device=score.device
    )
    for i in range(batch_size):
        topk_indices[i] = src_page_table[i, indices[i]]
    return topk_indices

MAX_SEQ_LEN = 66551
MAX_PERMIT_ERROR = 0

def assert_equal(
    score: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_our: torch.Tensor,
    bs: int,
    k: int,
    seq_len: int,
):
    indices_our_cpu = indices_our.cpu().tolist()
    indices_ref_cpu = indices_ref.cpu().tolist()
    for i in range(bs):
        indices_ref_set_i = set(indices_ref_cpu[i])
        indices_our_set_i = set(indices_our_cpu[i])
        more = indices_our_set_i - indices_ref_set_i
        less = indices_ref_set_i - indices_our_set_i
        if len(more) > MAX_PERMIT_ERROR or len(less) > MAX_PERMIT_ERROR:
            more_values = sorted(score[i, idx].item() for idx in more)
            less_values = sorted(score[i, idx].item() for idx in less)
            assert (
                more_values == less_values
            ), f"{bs=}, {k=}, {seq_len=}, {i=}, {more=}, {less=} failed, with {more_values=}, {less_values=}"

# =========================================================================
# 【核心新增】：性能基准测试核心逻辑
# =========================================================================
def run_cuda_benchmark(kernel_name, run_func, bytes_accessed, warmup=5, iters=100):
    """高精度 CUDA Kernel 计时与带宽计算函数"""
    # 1. Warmup 热身阶段
    for _ in range(warmup):
        run_func()
    torch.cuda.synchronize()

    # 2. 建立 CUDA Events 高精度计时器
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        run_func()
    end_event.record()
    
    # 强制同步等待执行流结束
    torch.cuda.synchronize()
    
    # 3. 计算耗时与有效带宽
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / iters
    avg_time_sec = avg_time_ms / 1000.0
    
    # 带宽计算公式 (使用标准十进制 GB/s)
    bandwidth_gb_s = (bytes_accessed / avg_time_sec) / 1e9

    print(f"  [PERF] {kernel_name:<30} | Avg Time: {avg_time_ms:8.4f} ms | Bandwidth: {bandwidth_gb_s:7.2f} GB/s")


@pytest.mark.parametrize("bs", [1, 132, 256, 4096])
@pytest.mark.parametrize("k", [2048])  # optimized for deepseek v3.2
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 65536])
@torch.inference_mode()
def test_topk_kernel(bs: int, k: int, seq_len: int) -> None:
    # 1. 固定随机种子确保可复现性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")

    indices_ref = _ref_torch_impl(score, seq_len, k)

    assert (
        k == 2048
    ), "fast_topk_v2 is only optimized for deepseek v3.2 model, where topk=2048"
    assert score.dim() == 2
    topk_indices = score.new_empty((score.size(0), k), dtype=torch.int32)
    row_starts_opt = None

    # 定义执行闭包用于 Benchmark
    def launch_kernel():
        torch.ops.sgl_kernel.fast_topk.default(score, topk_indices, lengths, row_starts_opt)

    # 运行功能验证
    launch_kernel()
    indices_our = topk_indices

    # 排序比对精度
    indices_ref_sorted = torch.sort(indices_ref, dim=-1).values
    indices_our_sorted = torch.sort(indices_our, dim=-1).values
    assert_equal(score, indices_ref_sorted, indices_our_sorted, bs, k, seq_len)

    # 计算总访存字节数: Read Score + Write Indices
    bytes_accessed = bs * (seq_len + k) * 4
    print(f"\n[Case: BS={bs}, SeqLen={seq_len}, K={k}]")
    run_cuda_benchmark("fast_topk", launch_kernel, bytes_accessed)


@pytest.mark.parametrize("bs", [1, 16,32, 64, 131, 256, 1024,2048, 4096])
@pytest.mark.parametrize("k", [2048])
@pytest.mark.parametrize("seq_len", [66551])
@pytest.mark.parametrize("table_len", [299062, 107520])
@torch.inference_mode()
def test_topk_transform_kernel(bs: int, k: int, seq_len: int, table_len: int ) -> None:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    src_page_table = torch.arange(0, table_len, dtype=torch.int32, device="cuda")
    src_page_table = src_page_table.unsqueeze(0).expand(bs, -1)
    
    cu_seqlens_q = torch.arange(0, bs + 1, dtype=torch.int32, device="cuda")
    dst_page_table_ref = _ref_torch_transform_decode_impl(
        score=score,
        seq_len=seq_len,
        src_page_table=src_page_table,
        topk=k,
    )

    assert (
        k == 2048
    ), "fast_topk_transform_fused is only optimized for deepseek v3.2 model, where topk=2048"
    assert score.dim() == 2
    dst_page_table = score.new_empty((score.size(0), k), dtype=torch.int32)
    row_starts_opt = None

    def launch_fused_kernel():
        torch.ops.sgl_kernel.fast_topk_transform_fused.default(
            score, lengths, dst_page_table, src_page_table, cu_seqlens_q, row_starts_opt
        )

    # 功能验证
    launch_fused_kernel()
    dst_page_table_our = dst_page_table

    # 排序比对精度
    dst_page_table_our_sorted = torch.sort(dst_page_table_our, dim=-1).values
    dst_page_table_ref_sorted = torch.sort(dst_page_table_ref, dim=-1).values
    assert_equal(score, dst_page_table_ref_sorted, dst_page_table_our_sorted, bs, k, seq_len)

    # 计算总访存字节数: Read Score + Read SrcPageTable + Write DstPageTable
    bytes_accessed = bs * (2 * seq_len + k) * 4
    print(f"\n[Case: BS={bs}, SeqLen={seq_len}, K={k}]")
    run_cuda_benchmark("fast_topk_transform_fused", launch_fused_kernel, bytes_accessed)


if __name__ == "__main__":
    # 执行 pytest
    pytest.main(["-s", "-v", __file__])