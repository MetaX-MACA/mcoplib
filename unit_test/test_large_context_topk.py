import torch
import time
import math

# 尝试导入自定义的 CUDA op
try:
    import mcoplib._C
except ImportError:
    print("Warning: 无法导入 mcoplib._C，请确保算子已正确编译安装在环境中。")
    print("测试脚本将继续，但在调用 torch.ops._C.large_context_topk 时可能会报错。")


def cpu_large_context_topk_reference(logits, seq_lens, row_starts=None):
    """
    完全对齐 CUDA 源码逻辑的 CPU 参考实现

    CUDA kernel 行为说明（来自 topk.cu）：
      - logits 基址 = input + batch_idx * input_stride
      - logits_offset = row_starts[batch_idx]（row_starts 为 None 时为 0）
      - 实际读取范围：logits[logits_offset + 0 .. logits_offset + seq_len - 1]
      - 输出 idx 为局部下标（0-based，相对于 logits_offset），范围 [0, seq_len)
      - naive 路径 (seq_len <= TOP_K)：直接输出 0..seq_len-1（不排序，不筛选，全部在 top-k 内）
      - fast radix 路径 (seq_len > TOP_K)：输出 top-2048 的局部下标集合（无序）
    """
    TOP_K = 2048
    batch_size = logits.size(0)
    ref_indices = torch.full((batch_size, TOP_K), -1, dtype=torch.int32)

    for b in range(batch_size):
        offset = int(row_starts[b]) if row_starts is not None else 0
        length = int(seq_lens[b])
        row = logits[b, offset: offset + length].float().cpu()
        k = min(TOP_K, length)

        if length <= TOP_K:
            # naive 路径：kernel 直接输出 0..seq_len-1，不做任何排序/筛选
            ref_indices[b, :k] = torch.arange(k, dtype=torch.int32)
        else:
            # fast radix 路径：取 top-k 的局部下标（0-based）
            _, idx = torch.topk(row, k)
            ref_indices[b, :k] = idx.int()

    return ref_indices


def check_topk_valid(logits, cuda_indices, seq_lens, row_starts=None, label=""):
    """
    正确的 Top-K 验证方式：
    
    不做严格的集合相等比较（因为 tie 情况下 kernel 与 torch.topk 的 tie-breaking 不同），
    而是验证 kernel 输出的每个索引"是否合法"：
    
      合法条件：cuda 选出的所有元素，其 logit 值 >= 第 TOP_K 小的 logit 值（即边界值）。
      等价于：cuda 输出的最小 logit 值 >= ref top-k 中的最小 logit 值（允许相等，即 tie）。

    同时验证：
      1. 输出 index 数量正确（无多余 -1）
      2. 所有 index 在合法范围 [0, seq_len) 内
      3. 没有重复 index
      4. 所有选中元素的 logit 值 >= top-k 边界值（核心正确性）
    """
    TOP_K = cuda_indices.size(1)
    batch_size = cuda_indices.size(0)
    all_ok = True

    for b in range(batch_size):
        offset = int(row_starts[b]) if row_starts is not None else 0
        length = int(seq_lens[b])
        k = min(TOP_K, length)

        row_logits = logits[b, offset: offset + length].float().cpu()
        cuda_row   = cuda_indices[b].cpu().tolist()
        valid_idxs = [x for x in cuda_row if x != -1]

        # 检查 1：数量
        if len(valid_idxs) != k:
            print(f"  ✗ [{label}] batch={b}: 期望 {k} 个有效索引，实际得到 {len(valid_idxs)} 个")
            all_ok = False
            continue

        # 检查 2：范围合法性
        out_of_range = [x for x in valid_idxs if x < 0 or x >= length]
        if out_of_range:
            print(f"  ✗ [{label}] batch={b}: 索引越界 {out_of_range[:5]}")
            all_ok = False
            continue

        # 检查 3：无重复
        if len(set(valid_idxs)) != len(valid_idxs):
            print(f"  ✗ [{label}] batch={b}: 存在重复索引")
            all_ok = False
            continue

        if length <= TOP_K:
            # naive 路径：期望输出恰好是 {0, 1, ..., seq_len-1}
            if set(valid_idxs) != set(range(length)):
                print(f"  ✗ [{label}] batch={b}: naive 路径输出不等于全集")
                all_ok = False
            continue

        # 检查 4（radix 路径）：所有选中 logit 值 >= top-k 边界值
        # 边界值 = 第 k 大的值（即 top-k 中最小的那个）
        topk_boundary = torch.topk(row_logits, k).values[-1].item()
        cuda_logits   = row_logits[valid_idxs]
        cuda_min_val  = cuda_logits.min().item()

        if cuda_min_val < topk_boundary - 1e-6:
            # 有元素的 logit 值明显低于边界，说明选错了（不是 tie 问题）
            wrong_idxs = [x for x in valid_idxs if row_logits[x].item() < topk_boundary - 1e-6]
            print(f"  ✗ [{label}] batch={b}: 存在 logit 低于 top-k 边界的错误索引")
            print(f"    top-k 边界值={topk_boundary:.6f}, cuda 选出的最小值={cuda_min_val:.6f}")
            print(f"    错误索引示例(前5): {wrong_idxs[:5]}")
            all_ok = False
            continue

        # tie 情况：允许 kernel 与 torch.topk 选择不同的边界元素
        ref_set  = set(torch.topk(row_logits, k).indices.tolist())
        cuda_set = set(valid_idxs)
        diff     = cuda_set - ref_set
        if diff:
            # 确认差异元素是否都是 tie（logit 值等于边界值）
            non_tie = [x for x in diff if abs(row_logits[x].item() - topk_boundary) > 1e-6]
            if non_tie:
                print(f"  ✗ [{label}] batch={b}: 存在非 tie 的索引差异 {non_tie[:5]}")
                all_ok = False
            # else: 纯 tie 差异，属于合法行为，不报错

    return all_ok


def run_test():
    TOP_K = 2048

    # --- 1. 模型配置参数 ---
    NUM_TOKENS    = 4096   # 模拟一次典型的 prefill token 数量
    SEQ_LEN_LONG  = 32768  # > TOP_K，触发 fast radix 路径
    SEQ_LEN_SHORT = 512    # <= TOP_K，触发 naive 路径

    print(f"=== 开始测试 large_context_topk ===")
    print(f"配置: Tokens={NUM_TOKENS}, TOP_K={TOP_K}, "
          f"LongSeqLen={SEQ_LEN_LONG}, ShortSeqLen={SEQ_LEN_SHORT}")

    torch.manual_seed(42)  # 固定随机数种子保证可复现

    # =========================================================================
    # 场景一：seq_len > TOP_K，触发 fast radix top-k 路径
    # =========================================================================
    print("\n[1/4] 精度校验 —— 长序列 (seq_len > TOP_K, fast radix 路径)...")

    logits_long = torch.randn(
        (NUM_TOKENS, SEQ_LEN_LONG), dtype=torch.float32, device='cuda'
    )
    seq_lens_long = torch.full(
        (NUM_TOKENS,), SEQ_LEN_LONG, dtype=torch.int32, device='cuda'
    )
    cuda_indices_long = torch.full(
        (NUM_TOKENS, TOP_K), -1, dtype=torch.int32, device='cuda'
    )

    # 获取 CUDA 结果（row_starts_opt 必须显式传 None）
    torch.ops._C.large_context_topk(logits_long, cuda_indices_long, seq_lens_long, None)
    torch.cuda.synchronize()  # 强制同步以确保 CUDA Kernel 执行完毕

    ok = check_topk_valid(logits_long, cuda_indices_long, seq_lens_long, label="长序列 radix")
    assert ok, "✗ 精度测试失败：长序列 radix 路径输出不合法！"
    print("  ✅ Top-K 索引 (长序列) 校验通过（允许 tie-breaking 差异）。")

    # =========================================================================
    # 场景二：seq_len <= TOP_K，触发 naive 全返路径
    # =========================================================================
    print("\n[2/4] 精度校验 —— 短序列 (seq_len <= TOP_K, naive 路径)...")

    logits_short = torch.randn(
        (NUM_TOKENS, SEQ_LEN_SHORT), dtype=torch.float32, device='cuda'
    )
    seq_lens_short = torch.full(
        (NUM_TOKENS,), SEQ_LEN_SHORT, dtype=torch.int32, device='cuda'
    )
    cuda_indices_short = torch.full(
        (NUM_TOKENS, TOP_K), -1, dtype=torch.int32, device='cuda'
    )

    torch.ops._C.large_context_topk(logits_short, cuda_indices_short, seq_lens_short, None)
    torch.cuda.synchronize()

    ok = check_topk_valid(logits_short, cuda_indices_short, seq_lens_short, label="短序列 naive")
    assert ok, "✗ 精度测试失败：短序列 naive 路径输出不合法！"
    print("  ✅ Top-K 索引 (短序列) 校验通过。")

    # =========================================================================
    # 场景三：带 row_starts 偏移的精度校验
    # =========================================================================
    print("\n[3/4] 精度校验 —— 带 row_starts 偏移...")

    BATCH_RS    = 8
    SEQ_RS      = 8192
    OFFSET_STEP = 64
    logits_rs = torch.randn(
        (BATCH_RS, SEQ_RS + BATCH_RS * OFFSET_STEP),
        dtype=torch.float32, device='cuda'
    )
    seq_lens_rs = torch.full(
        (BATCH_RS,), SEQ_RS, dtype=torch.int32, device='cuda'
    )
    row_starts_rs = torch.arange(
        0, BATCH_RS * OFFSET_STEP, OFFSET_STEP, dtype=torch.int32, device='cuda'
    )
    cuda_indices_rs = torch.full(
        (BATCH_RS, TOP_K), -1, dtype=torch.int32, device='cuda'
    )

    torch.ops._C.large_context_topk(
        logits_rs, cuda_indices_rs, seq_lens_rs, row_starts_rs
    )
    torch.cuda.synchronize()

    ok = check_topk_valid(logits_rs, cuda_indices_rs, seq_lens_rs,
                          row_starts=row_starts_rs, label="row_starts")
    assert ok, "✗ 精度测试失败：带 row_starts 时输出不合法！"
    print("  ✅ Top-K 索引 (带 row_starts) 校验通过。")

    # =========================================================================
    # 场景四：耗时统计 (Performance Test)
    # =========================================================================
    print("\n[4/4] 正在进行耗时统计分析 (Profile CPU vs CUDA)...")

    WARMUP = 10
    RUNS   = 100

    logits_perf = torch.randn(
        (NUM_TOKENS, SEQ_LEN_LONG), dtype=torch.float32, device='cuda'
    )
    seq_lens_perf = torch.full(
        (NUM_TOKENS,), SEQ_LEN_LONG, dtype=torch.int32, device='cuda'
    )
    cuda_indices_perf = torch.full(
        (NUM_TOKENS, TOP_K), -1, dtype=torch.int32, device='cuda'
    )

    # 4.1 Profile CPU（使用 torch.topk 作为参照基准）
    def cpu_ref_topk(logits, seq_lens):
        for b in range(logits.size(0)):
            length = int(seq_lens[b])
            torch.topk(logits[b, :length].float().cpu(), min(TOP_K, length))

    for _ in range(WARMUP):
        cpu_ref_topk(logits_perf, seq_lens_perf)

    start_time = time.time()
    for _ in range(RUNS):
        cpu_ref_topk(logits_perf, seq_lens_perf)
    cpu_avg_time = (time.time() - start_time) / RUNS * 1000  # 转换为毫秒(ms)

    # 4.2 Profile CUDA
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    for _ in range(WARMUP):
        torch.ops._C.large_context_topk(
            logits_perf, cuda_indices_perf, seq_lens_perf, None
        )
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(RUNS):
        torch.ops._C.large_context_topk(
            logits_perf, cuda_indices_perf, seq_lens_perf, None
        )
    end_event.record()
    torch.cuda.synchronize()
    cuda_avg_time = start_event.elapsed_time(end_event) / RUNS  # elapsed_time 默认单位为毫秒

    print(f"  🕐 CPU 平均耗时 ({RUNS}次跑均): {cpu_avg_time:.4f} ms")
    print(f"  🕐 GPU CUDA 算子平均耗时:       {cuda_avg_time:.4f} ms")
    print(f"  🚀 CUDA 加速比: {cpu_avg_time / cuda_avg_time:.2f}x")

    assert not math.isnan(cuda_avg_time), "✗ 性能测试失败：CUDA 耗时为 NaN"
    assert cuda_avg_time > 0,             "✗ 性能测试失败：CUDA 耗时应大于 0"

    print("\n=== 测试圆满结束 ===")


if __name__ == "__main__":
    run_test()