import torch
import torch.nn.functional as F
import json
import os
import time
import statistics 

# ================= 真实计算逻辑 =================

THEORETICAL_PEAK_GBPS = 1843.0 

current_dir = os.path.dirname(os.path.abspath(__file__))
# 对应你要求的同名 json
config_path = os.path.join(current_dir, "../config", "fused_bias_dropout.json")

def load_config():
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f: return json.load(f)

def run_real_benchmark():
    try:
        cfg = load_config()
        
        # --- 1. 严格读取 JSON，没有 seq_len，没有默认值，缺 Key 报错 ---
        try:
            # 你的 JSON key: batch_size, hidden_size, dtype, prob, samples, device_id
            device_id = cfg["device_id"]
            B = cfg["batch_size"]     # 对应 batch_size (这里视为第一维 N)
            H = cfg["hidden_size"]    # 对应 hidden_size
            PROB = cfg["prob"]        # 对应 prob
            dtype_str = cfg["dtype"]
            SAMPLES = cfg["samples"]
        except KeyError as e:
            raise KeyError(f"JSON 配置缺少必要的键: {e}")

        DEVICE = f"cuda:{device_id}"
        
        dtype_map = {
            "float16": torch.float16, 
            "float32": torch.float32, 
            "bfloat16": torch.bfloat16
        }
        if dtype_str not in dtype_map:
            raise ValueError(f"不支持的 dtype: {dtype_str}")
        DTYPE = dtype_map[dtype_str]

        torch.manual_seed(0)
        
        # 尝试导入算子
        try:
            import mcoplib.op as op
            op_func = op.fused_bias_dropout
        except ImportError:
            print("Warning: 'mcoplib.op' not found. Running in Reference-Only mode.")
            op_func = None

        with torch.cuda.device(DEVICE):
            # --- 2. 数据准备 (按照无 seq_len 处理，生成 2D 张量) ---
            # 形状: [batch_size, hidden_size]
            input_tensor = torch.randn(B, H, dtype=DTYPE, device=DEVICE, requires_grad=False)
            residual_tensor = torch.randn_like(input_tensor)
            out = torch.empty_like(input_tensor) # 占位，算子通常返回新 tensor，但也可能原地

            # --- 3. 真实精度验证 ---
            # 逻辑: (Input + Residual) -> Dropout
            ref_add = input_tensor + residual_tensor
            
            if PROB > 0.0:
                # Prob > 0 时有随机性，仅做逻辑参考
                ref = F.dropout(ref_add, p=PROB, training=True)
            else:
                ref = ref_add

            if op_func:
                # 调用算子
                out = op_func(input_tensor, residual_tensor, PROB)
            else:
                out = ref.clone()

            # 验证结果
            if PROB == 0.0:
                is_pass = torch.allclose(out, ref, rtol=1e-3, atol=1e-3)
                acc_pass_str = "Yes" if is_pass else "No"
                
                cos_sim = F.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0, eps=1e-8)
                cos_dist_val = 1.0 - cos_sim.item()
                if cos_dist_val < 0: cos_dist_val = 0.0
            else:
                acc_pass_str = "Skip(Prob>0)"
                cos_dist_val = 0.0

            # --- 4. 性能测试 ---
            # 预热
            for _ in range(50):
                if op_func: _ = op_func(input_tensor, residual_tensor, PROB)
            torch.cuda.synchronize()

            # === A. 逐次测量 (Latency) ===
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(SAMPLES)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(SAMPLES)]
            cpu_deltas = []

            global_start = time.perf_counter()
            
            for i in range(SAMPLES):
                t0 = time.perf_counter()
                start_events[i].record()
                if op_func:
                    # 必须赋值防止被优化
                    _ = op_func(input_tensor, residual_tensor, PROB)
                end_events[i].record()
                t1 = time.perf_counter()
                cpu_deltas.append(t1 - t0)

            torch.cuda.synchronize()
            global_end = time.perf_counter()

            # === B. 批量连续测量 (Batch Throughput) === 
            batch_start = torch.cuda.Event(enable_timing=True)
            batch_end = torch.cuda.Event(enable_timing=True)
            
            batch_start.record()
            for _ in range(SAMPLES):
                if op_func:
                    _ = op_func(input_tensor, residual_tensor, PROB)
            batch_end.record()
            torch.cuda.synchronize()
            
            batch_total_ms = batch_start.elapsed_time(batch_end)

            # --- 5. 指标计算 ---
            
            # GPU Time
            gpu_times = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]
            if len(gpu_times) > 0:
                avg_gpu_us = statistics.mean(gpu_times)
                stdev_gpu = statistics.stdev(gpu_times) if len(gpu_times) > 1 else 0.0
                gpu_noise_val = (stdev_gpu / avg_gpu_us * 100) if avg_gpu_us > 0 else 0.0
                gpu_noise_str = f"{gpu_noise_val:.2f}%"
            else:
                avg_gpu_us = 0.0
                gpu_noise_str = "--"

            # Batch GPU
            if SAMPLES > 0 and op_func:
                batch_gpu_us_val = (batch_total_ms * 1000) / SAMPLES
                batch_gpu_str = f"{batch_gpu_us_val:.3f} us"
                batch_cnt_str = f"{SAMPLES}x"
            else:
                batch_gpu_str = "--"
                batch_cnt_str = "--"

            # CPU Time
            avg_cpu_us = ((global_end - global_start) / SAMPLES) * 1e6
            if len(cpu_deltas) > 1:
                cpu_mean_internal = statistics.mean(cpu_deltas)
                cpu_stdev = statistics.stdev(cpu_deltas)
                if cpu_mean_internal > 1e-9:
                    cpu_noise_val = (cpu_stdev / cpu_mean_internal) * 100
                    cpu_noise_str = f"{cpu_noise_val:.2f}%"
                else:
                    cpu_noise_str = "--"
            else:
                cpu_noise_str = "--"

            # BW & Elem/s
            calc_base_us = avg_gpu_us 
            element_size = 2 if DTYPE in [torch.float16, torch.bfloat16] else 4
            
            # 既然是 2D [B, H]，元素总数即为 B * H
            total_elements = B * H
            # 读写量：读Input + 读Residual + 写Output = 3倍数据量
            total_bytes = (total_elements * 3) * element_size
            
            if calc_base_us > 0:
                gb_per_sec = (total_bytes / 1e9) / (calc_base_us / 1e6)
                bw_util_val = (gb_per_sec / THEORETICAL_PEAK_GBPS) * 100
                elems_per_sec = (total_elements / 1e9) / (calc_base_us / 1e6)
                
                bw_str = f"{gb_per_sec:.3f} GB/s"
                bw_util_str = f"{bw_util_val:.2f}%"
                elems_str = f"{elems_per_sec:.3f}G"
            else:
                bw_str = "--"
                bw_util_str = "--"
                elems_str = "--"

            # ================= 6. 输出 =================
            headers = [
                "Acc_Pass", "Cos_Dist", "Op", "dtype", 
                "Shape(B H)", "Samples", "CPU Time", "Noise",
                "GPU Time", "Noise", "Elem/s", 
                "GlbMem BW", "BWUtil", 
                 "Samples","Batch GPU"
            ]
            
            vals = [
                acc_pass_str,
                f"{cos_dist_val:.2e}" if PROB == 0 else "--", 
                "bias_dropout",
                dtype_str,
                f"{B}x{H}",         # 只显示 B 和 H
                f"{SAMPLES}x",
                f"{avg_cpu_us:.3f} us",
                cpu_noise_str,
                f"{avg_gpu_us:.3f} us",
                gpu_noise_str,
                elems_str,
                bw_str,
                bw_util_str,
                batch_cnt_str,
                batch_gpu_str
            ]

            padding = 1
            widths = [max(len(h), len(v)) + padding for h, v in zip(headers, vals)]
            
            header_line = "| " + " | ".join([f"{h:^{w}}" for h, w in zip(headers, widths)]) + " |"
            sep_line    = "|-" + "-|-".join([f"{'-'*w}" for w in widths]) + "-|"
            row_line    = "| " + " | ".join([f"{v:^{w}}" for v, w in zip(vals, widths)]) + " |"

            print("-" * len(header_line))
            print(header_line)
            print(sep_line)
            print(row_line)
            print("-" * len(header_line))

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        exit(1)

if __name__ == "__main__":
    run_real_benchmark()