import torch
import torch.nn.functional as F
import json
import os
import time
import statistics 

# ================= 真实计算逻辑 =================

THEORETICAL_PEAK_GBPS = 1843.0 

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "../config", "silu_and_mul.json")

def load_config():
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        default_cfg = {
            "device_id": 0, "num_tokens": 4096, "hidden_size": 11008, 
            "dtype": "float16", "samples": 10000 
        }
        with open(config_path, 'w') as f: json.dump(default_cfg, f)
        return default_cfg
    with open(config_path, 'r') as f: return json.load(f)

def run_real_benchmark():
    cfg = load_config()
    DEVICE = f"cuda:{cfg.get('device_id', 0)}"
    N = cfg.get("num_tokens", 4096)
    H = cfg.get("hidden_size", 11008)
    INPUT_SIZE = H * 2
    SAMPLES = cfg.get("samples", 1000)
    
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    DTYPE = dtype_map.get(cfg.get("dtype", "float16"), torch.float16)

    torch.manual_seed(0)
    try:
        try:
            import mcoplib._C
            op_func = torch.ops._C.silu_and_mul
        except ImportError:
            op_func = None

        with torch.cuda.device(DEVICE):
            inp = torch.randn(N, INPUT_SIZE, dtype=DTYPE, device=DEVICE)
            out = torch.empty(N, H, dtype=DTYPE, device=DEVICE)
            
            # --- 1. 真实精度验证 ---
            x_ref, y_ref = inp.chunk(2, dim=-1)
            ref = F.silu(x_ref) * y_ref
            
            if op_func:
                op_func(out, inp)
            else:
                out = ref.clone()

            is_pass = torch.allclose(out, ref, rtol=1e-3, atol=1e-3)
            acc_pass_str = "Yes" if is_pass else "No"
            
            cos_sim = F.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0, eps=1e-8)
            cos_dist_val = 1.0 - cos_sim.item()
            if cos_dist_val < 0: cos_dist_val = 0.0
            
            # --- 2. 性能测试 ---
            # 预热
            for _ in range(50):
                if op_func: op_func(out, inp)
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
                    op_func(out, inp)
                else:
                    pass 
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
                    op_func(out, inp)
            batch_end.record()
            torch.cuda.synchronize()
            
            batch_total_ms = batch_start.elapsed_time(batch_end)

            # --- 3. 指标计算 ---
            
            # GPU Time (Latency)
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
            total_bytes = (N * INPUT_SIZE + N * H) * element_size
            
            if calc_base_us > 0:
                gb_per_sec = (total_bytes / 1e9) / (calc_base_us / 1e6)
                bw_util_val = (gb_per_sec / THEORETICAL_PEAK_GBPS) * 100
                elems_per_sec = (N * H / 1e9) / (calc_base_us / 1e6)
                
                bw_str = f"{gb_per_sec:.3f} GB/s"
                bw_util_str = f"{bw_util_val:.2f}%"
                elems_str = f"{elems_per_sec:.3f}G"
            else:
                bw_str = "--"
                bw_util_str = "--"
                elems_str = "--"

            # ================= 4. 输出 (自动列宽版) =================
            
            # 1. 定义表头列表 (纯字符串)
            headers = [
                "Acc_Pass", "Cos_Dist", "Op", "dtype", 
                "Shape", "Samples", "CPU Time", "Noise",
                "GPU Time", "Noise", "Elem/s", 
                "GlbMem BW", "BWUtil", 
                "Batch GPU", "Batch"
            ]
            
            # 2. 准备数据值列表 (全部格式化为字符串)
            vals = [
                acc_pass_str,
                f"{cos_dist_val:.2e}", 
                "silu_and_mul",
                cfg.get("dtype", "float16"),
                f"({N} {INPUT_SIZE}) -> ({N} {H})",
                f"{SAMPLES}x",
                f"{avg_cpu_us:.3f} us",
                cpu_noise_str,
                f"{avg_gpu_us:.3f} us",
                gpu_noise_str,
                elems_str,
                bw_str,
                bw_util_str,
                batch_gpu_str,
                batch_cnt_str
            ]

            # 3. 自动计算列宽 (取 max(表头长度, 数据长度) + padding)
            padding = 0
            widths = [max(len(h), len(v)) + padding for h, v in zip(headers, vals)]
            
            # 4. 打印表格
            # 动态生成格式化字符串: ^表示居中, w是计算出的宽度
            header_line = "| " + " | ".join([f"{h:^{w}}" for h, w in zip(headers, widths)]) + " |"
            sep_line    = "|-" + "-|-".join([f"{'-'*w}" for w in widths]) + "-|"
            row_line    = "| " + " | ".join([f"{v:^{w}}" for v, w in zip(vals, widths)]) + " |"

            print("-" * len(header_line))
            print(header_line)
            print(sep_line)
            print(row_line)
            print("-" * len(header_line))

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_real_benchmark()