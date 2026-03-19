import torch
import torch.nn.functional as F
import time
import math

# 尝试导入自定义的 CUDA op
try:
    import mcoplib._moe_C
except ImportError:
    print("Warning: 无法导入 mcoplib._moe_C，请确保算子已正确编译安装在环境中。")
    print("测试脚本将继续，但在调用 torch.ops._moe_C.router_gemm_bf16_fp32 时可能会报错。")


def cpu_reference(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    CPU fp32 参考实现：output = input @ weight.T
    与 cuBLAS CUBLAS_COMPUTE_32F 的累加行为对齐：先升到 fp32 再做 GEMM。
    """
    return torch.mm(input.float().cpu(), weight.float().cpu().t())


def check_accuracy(cuda_out: torch.Tensor, ref_out: torch.Tensor, label: str,
                   cos_thr: float = 0.9999, max_diff_thr: float = 0.1):
    """
    精度校验（router 输出是 fp32，精度要求比 bf16 输出更严）：
      1. 输出 dtype 必须是 fp32
      2. 无 NaN / Inf
      3. 余弦相似度 >= cos_thr
      4. 最大绝对误差 <= max_diff_thr
    """
    assert cuda_out.dtype == torch.float32, \
        f"✗ [{label}] 输出 dtype 应为 float32，实际为 {cuda_out.dtype}"

    has_nan = torch.isnan(cuda_out).any().item()
    has_inf = torch.isinf(cuda_out).any().item()
    assert not has_nan, f"✗ [{label}] 输出含 NaN！"
    assert not has_inf, f"✗ [{label}] 输出含 Inf！"

    cuda_f = cuda_out.float().flatten()
    ref_f  = ref_out.float().flatten()

    cos_sim       = F.cosine_similarity(cuda_f.unsqueeze(0), ref_f.unsqueeze(0)).item()
    precision_err = 1.0 - cos_sim
    max_diff      = torch.max(torch.abs(cuda_f - ref_f)).item()

    print(f"  📐 [{label}] 余弦相似度: {cos_sim:.8f}  (误差 1-cos: {precision_err:.2e})")
    print(f"  📐 [{label}] 最大绝对误差 (Max Diff): {max_diff:.8f}")

    assert not math.isnan(cos_sim), f"✗ [{label}] 余弦相似度为 NaN"
    assert cos_sim >= cos_thr, \
        f"✗ [{label}] 精度不足：余弦相似度 {cos_sim:.8f} < {cos_thr}"
    assert max_diff <= max_diff_thr, \
        f"✗ [{label}] 最大误差 {max_diff:.8f} 超过阈值 {max_diff_thr}"
    print(f"  ✅ [{label}] 精度校验通过。")


def run_test():
    # --- 1. 典型 MoE 路由配置参数 ---
    # DeepSeek V3: num_experts=256, hidden=7168; 用小尺寸加速 CPU 参考
    CONFIGS = [
        # (M=num_tokens, N=num_experts, K=hidden_dim, 描述)
        (1,    256, 7168, "单token,  DeepSeek V3 规格"),
        (8,    256, 7168, "小batch,  DeepSeek V3 规格"),
        (128,  256, 7168, "中batch,  DeepSeek V3 规格"),
        (4096, 256, 7168, "大batch,  prefill 场景"),
        (16,    64,  512, "小规格,   通用 MoE"),
        (32,   128, 1024, "中规格,   通用 MoE"),
    ]

    print("=== 开始测试 router_gemm_bf16_fp32 ===")
    print(f"功能: input[M,K] @ weight[N,K].T -> output[M,N] (bf16 x bf16 -> fp32)")

    torch.manual_seed(42)

    # =========================================================================
    # 场景一：多种 MoE 配置精度校验
    # =========================================================================
    print("\n[1/3] 精度校验 —— 多种 MoE 路由配置...")

    for M, N, K, desc in CONFIGS:
        input_  = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        weight_ = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')

        ref_out  = cpu_reference(input_, weight_)
        cuda_out = torch.ops._moe_C.router_gemm_bf16_fp32(input_, weight_)
        torch.cuda.synchronize()

        assert cuda_out.shape == (M, N), \
            f"✗ 输出形状错误: 期望({M},{N}), 实际{cuda_out.shape}"
        check_accuracy(cuda_out.cpu(), ref_out,
                       label=f"M={M},N={N},K={K} ({desc})")

    # =========================================================================
    # 场景二：边界值与特殊输入
    # =========================================================================
    print("\n[2/3] 边界值校验...")

    # 2.1 全零 input → 输出全零
    print("  2.1 全零 input...")
    input_zero  = torch.zeros(16, 256, dtype=torch.bfloat16, device='cuda')
    weight_rand = torch.randn(64, 256, dtype=torch.bfloat16, device='cuda')
    out_zero = torch.ops._moe_C.router_gemm_bf16_fp32(input_zero, weight_rand)
    torch.cuda.synchronize()
    assert out_zero.abs().max().item() == 0.0, "✗ 全零 input 时输出不为零"
    print("  ✅ 全零 input 校验通过。")

    # 2.2 全零 weight → 输出全零
    print("  2.2 全零 weight...")
    input_rand  = torch.randn(16, 256, dtype=torch.bfloat16, device='cuda')
    weight_zero = torch.zeros(64, 256, dtype=torch.bfloat16, device='cuda')
    out_zero2 = torch.ops._moe_C.router_gemm_bf16_fp32(input_rand, weight_zero)
    torch.cuda.synchronize()
    assert out_zero2.abs().max().item() == 0.0, "✗ 全零 weight 时输出不为零"
    print("  ✅ 全零 weight 校验通过。")

    # 2.3 M=1（单 token decode 场景）
    print("  2.3 M=1 单 token 场景...")
    input_1  = torch.randn(1, 7168, dtype=torch.bfloat16, device='cuda')
    weight_1 = torch.randn(256, 7168, dtype=torch.bfloat16, device='cuda')
    ref_1    = cpu_reference(input_1, weight_1)
    out_1    = torch.ops._moe_C.router_gemm_bf16_fp32(input_1, weight_1)
    torch.cuda.synchronize()
    check_accuracy(out_1.cpu(), ref_1, label="M=1 单token")

    # 2.4 N=1（单专家）
    print("  2.4 N=1 单专家场景...")
    input_n1  = torch.randn(32, 512, dtype=torch.bfloat16, device='cuda')
    weight_n1 = torch.randn(1, 512, dtype=torch.bfloat16, device='cuda')
    ref_n1    = cpu_reference(input_n1, weight_n1)
    out_n1    = torch.ops._moe_C.router_gemm_bf16_fp32(input_n1, weight_n1)
    torch.cuda.synchronize()
    check_accuracy(out_n1.cpu(), ref_n1, label="N=1 单专家")

    # 2.5 输出 dtype 严格验证
    print("  2.5 输出 dtype 验证...")
    input_t  = torch.randn(8, 128, dtype=torch.bfloat16, device='cuda')
    weight_t = torch.randn(32, 128, dtype=torch.bfloat16, device='cuda')
    out_t    = torch.ops._moe_C.router_gemm_bf16_fp32(input_t, weight_t)
    assert out_t.dtype  == torch.float32,   f"✗ 输出 dtype 应为 float32，实际 {out_t.dtype}"
    assert out_t.shape  == (8, 32),         f"✗ 输出形状错误：{out_t.shape}"
    assert out_t.device.type == 'cuda',     "✗ 输出应在 CUDA 设备上"
    print("  ✅ 输出 dtype/shape/device 验证通过。")


    # =========================================================================
    # 场景三：耗时统计 (Performance Test)
    # =========================================================================
    print("\n[3/3] 正在进行耗时统计分析 (Profile CPU vs CUDA)...")

    WARMUP = 10
    RUNS   = 100

    PERF_CASES = [
        (1,    256, 7168, "decode  M=1"),
        (128,  256, 7168, "prefill M=128"),
        (4096, 256, 7168, "prefill M=4096"),
    ]

    for M, N, K, desc in PERF_CASES:
        input_p  = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        weight_p = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
        print(f"\n  --- {desc} (M={M}, N={N}, K={K}) ---")

        # 4.1 Profile CPU
        for _ in range(WARMUP):
            cpu_reference(input_p, weight_p)
        start_time = time.time()
        for _ in range(RUNS):
            cpu_reference(input_p, weight_p)
        cpu_avg_time = (time.time() - start_time) / RUNS * 1000  # ms

        # 4.2 Profile CUDA
        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)

        for _ in range(WARMUP):
            torch.ops._moe_C.router_gemm_bf16_fp32(input_p, weight_p)
        torch.cuda.synchronize()

        start_event.record()
        for _ in range(RUNS):
            torch.ops._moe_C.router_gemm_bf16_fp32(input_p, weight_p)
        end_event.record()
        torch.cuda.synchronize()
        cuda_avg_time = start_event.elapsed_time(end_event) / RUNS  # ms

        print(f"  🕐 CPU 平均耗时 ({RUNS}次跑均): {cpu_avg_time:.4f} ms")
        print(f"  🕐 GPU CUDA 算子平均耗时:       {cuda_avg_time:.4f} ms")
        print(f"  🚀 CUDA 加速比: {cpu_avg_time / cuda_avg_time:.2f}x")

        assert not math.isnan(cuda_avg_time), "✗ 性能测试失败：CUDA 耗时为 NaN"
        assert cuda_avg_time > 0,             "✗ 性能测试失败：CUDA 耗时应大于 0"

    print("\n=== 测试圆满结束 ===")


if __name__ == "__main__":
    run_test()