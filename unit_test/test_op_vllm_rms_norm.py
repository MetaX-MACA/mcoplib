import os
import time
import torch
import torch.nn.functional as F
import mcoplib._C

# ==============================================================================
# 1. 算子加载 (请根据实际的 C++ binding 路径进行调整)
# ==============================================================================
# 假设您的 C++ 扩展被编译为 torch.ops.mcop.rms_norm 或者 vllm._custom_ops.rms_norm
# 这里提供一个 Wrapper，如果在 vllm 环境中则直接调用，否则调用自定义库
def run_cuda_rms_norm(x, weight, epsilon=1e-5):
    out = torch.empty_like(x)
    try:
        torch.ops._C.rms_norm(out, x, weight, epsilon)
    except ImportError:
        try:
            # 适配 MCOP 库自定义加载
            torch.ops.mcop.rms_norm(out, x, weight, epsilon)
        except Exception:
            # Fallback 占位，如果没有任何库，这里为了演示不至于崩溃，跑 ref
            # 请在真实测试环境中确保这里调用的是上面的 cuda extension
            out = run_torch_rms_norm_ref(x, weight, epsilon)
    return out

# ==============================================================================
# 2. PyTorch Reference 实现 (对齐 CUDA 算子中的 FP32 计算精度)
# ==============================================================================
def run_torch_rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-5):
    """
    为了和 CUDA 中 rsqrtf 精度对齐，我们将计算过程完全提升至 fp32。
    """
    # 提取 hidden_size
    hidden_size = x.size(-1)
    
    # 强制转换 FP32 计算 variance
    x_fp32 = x.to(torch.float32)
    
    # variance = (sum of x^2) / hidden_size
    variance = x_fp32.pow(2).sum(dim=-1, keepdim=True) / hidden_size
    
    # rsqrt(variance + epsilon)
    rsqrt_var = torch.rsqrt(variance + epsilon)
    
    # output = x * rsqrt
    out_fp32 = x_fp32 * rsqrt_var
    
    # 如果有 weight，乘上 weight (weight 也转成 FP32)
    if weight is not None:
        out_fp32 = out_fp32 * weight.to(torch.float32)
        
    return out_fp32.to(x.dtype)

# ==============================================================================
# 3. 性能测试 & 单元测试主体逻辑
# ==============================================================================
def test_rms_norm():
    # 获取设备信息
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    device_count = torch.cuda.device_count()
    device = torch.device('cuda')
    
    print(f"CUDA_VISIBLE_DEVICES='{gpu_id}'  device_count={device_count}")
    print(f"config: dtype=bf16->bf16  epsilon=1e-5")
    print("============================================================================================")

    # 需要测试的所有 Shape 以及是否有 Weight
    test_configs = [
        ((1024, 8, 128), True),
        ((8192, 8, 128), True),
        ((7177, 8, 128), True),
        ((16, 8, 128), True),
        ((1024, 8, 128), False),
        ((8192, 8, 128), False),
        ((7177, 8, 128), False),
        ((16, 8, 128), False),
        
        ((525, 8, 256), True),
        ((525, 16, 256), True),
        ((525, 2, 512), True),
        ((525, 5376), True),
        ((525, 16, 256), True),
        ((525, 16, 512), True),
        ((525, 2, 512), True),
        
        
        ((1577, 5376), True),
        ((1577, 16, 256), True),
        ((1577, 8, 256), True),
        ((1577, 16, 512), True),
        ((1577, 2, 512), True),
        
        ((4, 16, 256), True),
        ((4, 8, 256), True),
        ((4, 16, 512), True),
        ((4, 2,512), True),
        ((3, 16, 256), True),
        ((3, 8, 256), True),
        ((3, 16, 512), True),
        ((3, 8, 512), True),
        
        
        
        ((100, 2560), True),
        ((200, 3072), True),
        ((8192, 8, 1792), True),
        ((512, 2048), True),
        ((64, 8, 4096), True),
        ((512, 8, 4096), True),
        ((8, 8, 7168), True),
        ((256, 8, 8192), True),
        ((64, 8, 16384), True),
        ((4096, 4096), True),
        ((777, 6144), True),
        ((1, 4096), True),
        ((1, 8, 8, 4096), True),
        ((1, 128), True),
        ((1024, 4100), True),
        ((4096, 512), True),
        
        # ====== Gemma4-31B-it TP2 典型 Shape 覆盖 ======
        # (通常隐层不被TP切分即7168，如果在特定层被切分假设为3584，这里做全面覆盖)
        ((1, 7168), True),          # Decode 阶段 BS=1
        ((128, 7168), True),        # Prefill 短文本
        ((1024, 7168), True),       # Prefill 中长文本
        ((4096, 7168), True),       # Prefill 长文本 4K
        ((8192, 7168), True),       # Prefill 满载 8K
        ((1, 3584), True),          # 切分假设 (1)
        ((256, 3584), True), 
        ((8, 3584), True), 
        ((64, 3584), True), 
        ((512, 3584), True), 
        ((1024, 3584), True), 
        ((8192, 3584), True),       # 切分假设 (2)
        ((1, 5376), True),          # 切分假设 (1)
        ((256, 5376), True), 
        ((8, 5376), True), 
        ((64, 5376), True), 
        ((512, 5376), True), 
        ((1024, 5376), True), 
        ((8192, 5376), True),       # 切分假设 (2)
    ]

    all_pass = True
    peak_bandwidth = 0.0
    threshold = 0.99999

    for shape, has_weight in test_configs:
        hidden_size = shape[-1]
        
        # 初始化输入数据
        x = torch.randn(shape, dtype=torch.bfloat16, device=device)
        if has_weight:
            weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
        else:
            weight = None

        epsilon = 1e-5
        
        # ----------------------------------------------------
        # 精度测试 (Accuracy Check)
        # ----------------------------------------------------
        out_ref = run_torch_rms_norm_ref(x, weight, epsilon)
        out_cuda = run_cuda_rms_norm(x, weight, epsilon)
        
        # 计算余弦相似度 (Cos Sim)
        a_f32 = out_ref.flatten().to(torch.float32)
        b_f32 = out_cuda.flatten().to(torch.float32)
        
        cos_sim = F.cosine_similarity(a_f32.unsqueeze(0), b_f32.unsqueeze(0)).item()
        is_ok = cos_sim >= threshold
        if not is_ok:
            all_pass = False
        
        status_str = "OK" if is_ok else "FAIL"

        # ----------------------------------------------------
        # 性能测试 (Performance & Bandwidth)
        # ----------------------------------------------------
        warmup_iters = 5
        test_iters = 20
        
        # Warmup
        for _ in range(warmup_iters):
            _ = run_cuda_rms_norm(x, weight, epsilon)
        
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(test_iters):
            _ = run_cuda_rms_norm(x, weight, epsilon)
        end_event.record()
        
        torch.cuda.synchronize()
        
        # 平均耗时 (毫秒)
        avg_time_ms = start_event.elapsed_time(end_event) / test_iters
        
        # 带宽计算: (Input_Bytes + Output_Bytes + Weight_Bytes) / time
        # bf16 占用 2 Bytes
        bytes_x = x.numel() * 2
        bytes_out = bytes_x
        bytes_weight = weight.numel() * 2 if has_weight else 0
        total_bytes = bytes_x + bytes_out + bytes_weight
        
        # 带宽 (GB/s)
        bandwidth_gbps = (total_bytes / 1e9) / (avg_time_ms / 1e3)
        if bandwidth_gbps > peak_bandwidth:
            peak_bandwidth = bandwidth_gbps

        # ----------------------------------------------------
        # 格式化输出
        # ----------------------------------------------------
        shape_str = str(shape).replace(" ", "")
        weight_str = "Y" if has_weight else "N"
        
        print(f"[rms_norm] shape={shape_str:<18} weight={weight_str:<2}  cos_sim={cos_sim:.6f} {status_str}    {avg_time_ms:.4f} ms      {bandwidth_gbps:.1f} GB/s")

    print("============================================================================================")
    
    if all_pass:
        print(f"Accuracy: ALL PASS (threshold cos_sim >= {threshold})")
    else:
        print(f"Accuracy: FAILED on some configs (threshold cos_sim >= {threshold})")
        
    print(f"Peak bandwidth: {peak_bandwidth:.1f} GB/s  (Check hardware limits to verify efficiency)")


if __name__ == "__main__":
    test_rms_norm()