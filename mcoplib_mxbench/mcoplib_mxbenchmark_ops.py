import sys
import os
import json
import csv
import re
import importlib
import argparse
import tempfile
import torch
import difflib  # 用于相似度匹配

try:
    import cuda.bench._nvbench as bench
except ImportError:
    print("错误: 运行环境缺少 nvbench，请检查环境配置。")
    sys.exit(1)

# 引入基类以进行类型检查
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase
import mcoplib_mxbenchmark_op_wrapper

# =============================================================================
#  全局配置：支持的算子列表
# =============================================================================
SUPPORTED_OPERATORS = [
    "fused_rope_fwd",
    "fused_bias_dropout",
    "rms_norm",
    "paged_attention_v2",
    "topk_softmax",
    "rotary_embeding",
    "concat_and_cache_mla",
    "moe_sum",
    "silu_and_mul",
    "gelu_new",
    "gelu_fast",
    "gelu_quick",
    "fused_add_rms_norm",
    "apply_repetition_penalties",
    "rms_norm_static_fp8_quant"
    # 在这里继续添加新算子...
]

# =============================================================================
#  加载基类
# =============================================================================
def get_base_dir():
    return os.path.dirname(os.path.abspath(__file__))

def list_supported_operators():
    print("\n" + "="*40 + f"\n{' Supported Operators ':=^40}\n" + "="*40)
    if not SUPPORTED_OPERATORS:
        print("  (No operators defined in SUPPORTED_OPERATORS list)")
    else:
        for op in sorted(SUPPORTED_OPERATORS):
            print(f"  * {op}")
    print("="*40 + "\n")

def load_operator_runner(op_name):
    current_dir = get_base_dir()
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    # -----------------------------------------------------------
    # 1. 寻找配置文件 (.json) - 支持模糊搜索
    # -----------------------------------------------------------
    config_dir = os.path.join(current_dir, "config")
    if not os.path.exists(config_dir):
        print(f"Error: 配置目录不存在: {config_dir}")
        sys.exit(1)

    target_json_path = None
    exact_json = os.path.join(config_dir, f"{op_name}.json")
    
    if os.path.exists(exact_json):
        target_json_path = exact_json
    else:
        print(f"提示: 未找到精确匹配的配置文件 '{op_name}.json'，尝试模糊搜索 Config...")
        try:
            pattern = re.compile(op_name, re.IGNORECASE)
            json_files = [f for f in os.listdir(config_dir) if f.endswith(".json")]
            matched_jsons = [f for f in json_files if pattern.search(f)]

            if len(matched_jsons) == 0:
                print(f"Error: 找不到配置文件，且模糊搜索 '{op_name}' 无匹配项。")
                sys.exit(1)
            elif len(matched_jsons) == 1:
                target_json_path = os.path.join(config_dir, matched_jsons[0])
                print(f">> [Config Match] 锁定配置文件: {matched_jsons[0]}")
            else:
                best_matches = difflib.get_close_matches(op_name, matched_jsons, n=1, cutoff=0)
                best_match = best_matches[0] if best_matches else matched_jsons[0]
                print(f"Warning: Config 模糊匹配到多个，自动选择最相似项: {best_match}")
                target_json_path = os.path.join(config_dir, best_match)
        except re.error as e:
            print(f"Error: 正则表达式错误: {e}")
            sys.exit(1)
    # 无论用户输入什么，只要匹配到了 fused_rope_fwd.json，标准名就是 fused_rope_fwd
    canonical_name = os.path.splitext(os.path.basename(target_json_path))[0]
    
    with open(target_json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

# -----------------------------------------------------------
    # 2. 寻找 Runner 文件 (.py)
    # -----------------------------------------------------------
    runners_dir = os.path.join(current_dir, "runners")
    if not os.path.exists(runners_dir):
        print(f"Error: Runners 目录不存在: {runners_dir}")
        sys.exit(1)

    # 优先尝试用【标准名】去匹配 Runner，命中率更高
    exact_runner_name = f"mcoplib_mxbenchmark_{canonical_name}_runners"
    found_module_name = None
    
    try:
        module_path = f"runners.{exact_runner_name}"
        if importlib.util.find_spec(module_path) is not None:
            found_module_name = module_path
    except:
        pass

    if not found_module_name:
        # 如果标准名没找到，进入模糊搜索流程
        print(f"提示: 未找到标准 Runner '{exact_runner_name}.py'，尝试搜索...")
        py_files = [f for f in os.listdir(runners_dir) if f.endswith(".py") and f != "__init__.py"]
        
        try:
            matched_runners = []
            
            # 【策略 A】：先试着用 Config 文件名 (canonical_name) 搜
            # 例如：用 "fused_rope_fwd_m" 去搜
            if canonical_name:
                pattern_canon = re.compile(canonical_name, re.IGNORECASE)
                matched_runners = [f for f in py_files if pattern_canon.search(f)]

            # 【策略 B】：如果策略 A 没搜到，或者搜到了 0 个，
            # 这里的逻辑关键：回退使用用户输入的 op_name 再次搜索！
            # 例如：用 "fused_rope_fwd" 去搜
            if not matched_runners and op_name and op_name != canonical_name:
                print(f"提示: 标准名 '{canonical_name}' 匹配失败，尝试回退使用输入名 '{op_name}' 搜索...")
                pattern_op = re.compile(op_name, re.IGNORECASE)
                matched_runners = [f for f in py_files if pattern_op.search(f)]
                
        except re.error as e:
            print(f"Error: 正则匹配错误: {e}")
            matched_runners = []
        
        target_runner_file = None
        if len(matched_runners) == 0:
            print(f"Error: 在 runners 目录下找不到相关 Python 文件。")
            print(f"       已尝试关键词: '{canonical_name}' 和 '{op_name}'")
            sys.exit(1)
        elif len(matched_runners) == 1:
            target_runner_file = matched_runners[0]
        else:
            # 如果匹配到多个，优先选和 canonical_name 最像的，其次选和 op_name 最像的
            search_key = canonical_name if canonical_name else op_name
            best_matches = difflib.get_close_matches(search_key, matched_runners, n=1, cutoff=0)
            target_runner_file = best_matches[0] if best_matches else matched_runners[0]
            print(f"Warning: Runner 模糊匹配到多个，自动选择最相似项: {target_runner_file}")

        if target_runner_file:
            module_base = target_runner_file.replace(".py", "")
            found_module_name = f"runners.{module_base}"
            print(f">> [Runner Match] 锁定 Runner 模块: {module_base}")
    # -----------------------------------------------------------
    # 3. 加载模块与类
    # -----------------------------------------------------------
    try:
        module = importlib.import_module(found_module_name)
    except ImportError as e:
        print(f"\nError: 无法加载模块 '{found_module_name}'\n详情: {e}")
        sys.exit(1)

    target_cls = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, OpBenchmarkBase) and attr is not OpBenchmarkBase:
            target_cls = attr
            break
            
    if target_cls is None:
        print(f"Error: 在 {found_module_name} 中找不到 OpBenchmarkBase 的子类。")
        sys.exit(1)

    print(f">> 加载类: {target_cls.__name__} (Benchmark Name: {canonical_name})")
    
    return target_cls(canonical_name, config)

# =============================================================================
#  Benchmark Wrapper
# =============================================================================
def create_benchmark_wrapper(op_instance):
    def benchmark_func(state):
        dev_id = state.get_device()
        
        # 1. 验证 (Verification)
        try:
            passed, diff_val = op_instance.run_verification(dev_id)
            state.add_summary("Acc_Pass", "Yes" if passed else "No")
            state.add_summary("Cos_Dist", f"{diff_val:.2e}") 
            print(f"\n[ACC Verify] {op_instance.name} -> {'PASS' if passed else 'FAIL'} (1-CosSim: {diff_val:.2e})")
        except Exception as e:
            print(f"\n[ACC Verify] Error: {e}")
            state.add_summary("Acc_Pass", "Error")

        # ===========================================================
        # [新增] 手动物理预热 (Manual Warmup)
        # 不依赖 nvbench 参数，直接在 Python 层跑 10 次，把 GPU 唤醒
        # ===========================================================
        print(f"  >> [Warmup] Running 10 iterations to heat up GPU...", end="", flush=True)
        try:
            for _ in range(10):
                op_instance.run_verification(dev_id)
            print(" Done.")
        except Exception as e:
            print(f" (Warmup failed: {e})", end="")

        # 恢复正常的采样数逻辑
        # 采样数自动补偿 -16 (为了抵消 nvbench 内部可能的预热计数，保持最终报告的 Sample 数与 Config 一致)
        raw_samples = op_instance.config.get("samples", 100)
        target_samples = max(1, raw_samples - 16) 
        state.set_min_samples(target_samples)
        op_instance.define_metrics(state)
        
        tc_s = torch.cuda.ExternalStream(
            stream_ptr=state.get_stream().addressof(), 
            device=torch.cuda.device(dev_id)
        )
        launcher = op_instance.prepare_and_get_launcher(dev_id, tc_s)
        state.exec(launcher, sync=True)
    return benchmark_func



# =============================================================================
#  CSV 处理工具
# =============================================================================
def load_csv_data(filepath):
    if not filepath or not os.path.exists(filepath): return [], []
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        try: header = [h.strip() for h in next(reader)]
        except StopIteration: return [], []
        data = [dict(zip(header, row)) for row in reader if len(row) >= len(header)]
        return header, data

def preprocess_data(header, rows):
    if not rows: return header, rows
    new_header = []
    for h in header:
        if h == "Benchmark": new_header.append("op_name")
        elif h != "Skipped": new_header.append(h)

    valid_rows = [r for r in rows if r.get("Skipped", "No").strip().lower() != "yes"]
    cleaned_rows = []
    for r in valid_rows:
        new_r = {}
        for k, v in r.items():
            if k == "Skipped": continue
            target_key = "op_name" if k == "Benchmark" else k
            new_r[target_key] = v
        cleaned_rows.append(new_r)
    return new_header, cleaned_rows

def parse_time_val(val_str):
    if not val_str: return None
    val_str = str(val_str).strip().lower()
    try: return float(val_str)
    except: pass
    units = {'ns': 1e-9, 'us': 1e-6, 'ms': 1e-3, 's': 1.0}
    match = re.match(r"([0-9\.]+)\s*([a-z]+)", val_str)
    if match:
        num, unit = match.groups()
        if unit in units: return float(num) * units[unit]
    return None

def format_duration(seconds):
    if seconds is None: return "N/A"
    if seconds < 1e-7: return f"{seconds*1e9:.3f} ns"
    elif seconds < 1e-4: return f"{seconds*1e6:.3f} us"
    elif seconds < 1.0: return f"{seconds*1e3:.3f} ms"
    else: return f"{seconds:.3f} s"

def get_row_key(row_dict, header):
    # 排除了常见数值列，dtype 不在其中，因此会自动成为 key 的一部分
    exclude = ["Samples", "CPU Time", "GPU Time", "Noise", "Elem/s", "GlobalMem", "BWUtil", "Acc_Pass", "Max_Diff", "Cos_Dist"]
    key_cols = [h for h in header if not any(x in h for x in exclude) and h != "Skipped"]
    return tuple(row_dict.get(k, "") for k in key_cols)

def get_op_display_name(row):
    return row.get("op_name", row.get("Op", row.get("Benchmark", "?")))

def _write_csv(path, header, rows):
    try:
        target_dir = os.path.dirname(path)
        if target_dir and not os.path.exists(target_dir): os.makedirs(target_dir)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        print(f"写入CSV失败: {e}")

# =============================================================================
#  Update / Generate / Compare 逻辑
# =============================================================================
def perform_smart_update(temp_csv_path, target_csv_path):
    print("\n" + "="*80 +f"\n{' Smart Update Mode (>5% Gain) ':^80}\n" + "="*80)
    raw_h, raw_r = load_csv_data(temp_csv_path)
    new_h, new_r = preprocess_data(raw_h, raw_r)
    if not new_r: return

    old_raw_h, old_raw_r = load_csv_data(target_csv_path)
    old_h, old_r = preprocess_data(old_raw_h, old_raw_r)

    # 1. 构建全量表头
    combined_header = list(old_h)
    for col in new_h:
        if col not in combined_header: combined_header.append(col)
    
    if "op_name" in combined_header:
        combined_header.remove("op_name")
        combined_header.insert(0, "op_name")

    # 使用 combined_header 生成索引，包含 dtype
    history_map = {get_row_key(r, combined_header): idx for idx, r in enumerate(old_r)}
    
    updated_cnt = 0
    kept_cnt = 0
    final_rows = list(old_r)

    for new_row in new_r:
        for col in combined_header:
            if col not in new_row: new_row[col] = ""
            
        key = get_row_key(new_row, combined_header)
        match_idx = history_map.get(key, -1)
        
        new_time = parse_time_val(new_row.get("GPU Time (sec)", ""))
        
        d_type = new_row.get("dtype", "")
        op_str = get_op_display_name(new_row)
        if d_type: op_str += f" [{d_type}]"

        if match_idx >= 0:
            old_row = final_rows[match_idx]
            old_time = parse_time_val(old_row.get("GPU Time (sec)", ""))
            
            if new_time is not None and old_time is not None and old_time > 0:
                ratio = (old_time - new_time) / old_time
                if ratio > 0.05:
                    final_rows[match_idx].update(new_row)
                    updated_cnt += 1
                    print(f"[UPDATE] {op_str:<35} | 提升: {ratio*100:>6.2f}%")
                else:
                    kept_cnt += 1
                    if ratio >= 0:
                        print(f"[KEEP]   {op_str:<35} | 提升: {ratio*100:>6.2f}%")
                    else:
                        print(f"[KEEP]   {op_str:<35} | 下降: {ratio*100:>6.2f}%")
            else:
                kept_cnt += 1
                print(f"[KEEP]   {op_str:<35} | N/A (时间数据无效)")
        
    _write_csv(target_csv_path, combined_header, final_rows)
    print("-" * 80 + f"\n完成: 更新 {updated_cnt}, 保持 {kept_cnt}\n" + "=" * 80 + "\n")

def perform_generate(temp_csv_path, target_csv_path):
    print("\n" + "="*80 +f"\n{' Generate Mode (Fill Missing) ':^80}\n" + "="*80)
    raw_h, raw_r = load_csv_data(temp_csv_path)
    new_h, new_r = preprocess_data(raw_h, raw_r)
    if not new_r: return

    if os.path.exists(target_csv_path):
        old_raw_h, old_raw_r = load_csv_data(target_csv_path)
        old_h, old_r = preprocess_data(old_raw_h, old_raw_r)
    else:
        old_h, old_r = [], []

    # 1. 构建全量表头
    combined_header = list(old_h)
    if not combined_header: combined_header = list(new_h)
    else:
        for col in new_h:
            if col not in combined_header: combined_header.append(col)
            
    if "op_name" in combined_header:
        combined_header.remove("op_name")
        combined_header.insert(0, "op_name")

    # 使用 combined_header 生成索引
    history_map = {get_row_key(r, combined_header): idx for idx, r in enumerate(old_r)}
    
    final_rows = list(old_r)
    rows_to_append = []
    append_cnt = 0
    skip_cnt = 0

    for new_row in new_r:
        for col in combined_header:
            if col not in new_row: new_row[col] = ""
            
        key = get_row_key(new_row, combined_header)
        match_idx = history_map.get(key, -1)
        
        d_type = new_row.get("dtype", "")
        op_str = get_op_display_name(new_row)
        if d_type: op_str += f" [{d_type}]"

        if match_idx >= 0:
            print(f"[SKIP]   {op_str:<35} (已存在)")
            skip_cnt += 1
        else:
            rows_to_append.append(new_row)
            print(f"[APPEND] {op_str:<35}")
            append_cnt += 1

    final_rows.extend(rows_to_append)
    if append_cnt > 0:
        _write_csv(target_csv_path, combined_header, final_rows)
    else:
        print("提示: 没有新数据需要写入。")
    print("-" * 80 + f"\n完成: 新增 {append_cnt}, 跳过 {skip_cnt}\n" + "=" * 80 + "\n")

def perform_comparison(cur_raw, hist_raw):
    _, cur_rows = preprocess_data([], cur_raw)
    _, hist_rows = preprocess_data([], hist_raw)
    if not cur_rows: return
    print("\n" + "="*95 + "\n" + f"{' Performance Comparison ':^95}" + "\n" + "="*95)
    
    row_fmt = "{:<15} | {:<15} | {:<10} | {:<15} | {:<15} | {:<20}"
    dummy_header = list(cur_rows[0].keys())
    
    for row in cur_rows:
        key = get_row_key(row, dummy_header)
        gpu = parse_time_val(row.get("GPU Time (sec)", ""))
        cpu = parse_time_val(row.get("CPU Time (sec)", ""))
        op_name = get_op_display_name(row)
        d_type = row.get("dtype", "-")
        
        # 1. 获取精度验证状态
        # 在 create_benchmark_wrapper 中定义的 key 是 "Acc_Pass"，值为 "Yes"/"No"
        raw_acc = row.get("Acc_Pass", "N/A")
        if raw_acc == "Yes":
            acc_status = "Pass"
        elif raw_acc == "No":
            acc_status = "Fail"
        else:
            acc_status = raw_acc  # 如果是 Error 或其他状态直接显示

        print(f" {op_name} | {row.get('Shape', '')} | {d_type}")
        print("-" * 95)
        print(row_fmt.format("Type", "GPU Time", "Ratio", "CPU Time", "Ratio", ""))
        print("-" * 95)
        print(row_fmt.format("Current", format_duration(gpu), "-", format_duration(cpu), "-", ""))
        
        matches = []
        for idx, h_row in enumerate(hist_rows):
            if get_row_key(h_row, list(h_row.keys())) == key: matches.append((idx, h_row))
        
        # 2. 初始化性能比值变量
        perf_ratio_str = "null"

        if matches:
            _, h_row = matches[-1] 
            h_gpu = parse_time_val(h_row.get("GPU Time (sec)", ""))
            h_cpu = parse_time_val(h_row.get("CPU Time (sec)", ""))
            
            # 计算比值 (Base / Current)
            if gpu and h_gpu:
                # 记录比值供 Result 使用
                perf_ratio_str = f"{h_gpu/gpu*100:.2f}%"
                gr = perf_ratio_str
            else:
                gr = "N/A"

            cr = f"{h_cpu/cpu*100:.2f}%" if (cpu and h_cpu) else "N/A"
            print(row_fmt.format("Base", format_duration(h_gpu), gr, format_duration(h_cpu), cr, ""))
        else:
            print(f"{'Base':<15} | {'N/A':<15} | {'N/A':<10} | {'N/A':<15} | {'N/A':<15} |")
        
        # 3. 输出 Result 模块
        print("Result:")
        print(f"Acc verify:{acc_status}")
        print(f"Performance verify:{perf_ratio_str}")
        print("\n")

# =============================================================================
#  Main Execution Block
# =============================================================================
# =============================================================================
#  Main Execution Block (修复缩进版)
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCOPLIB 算子性能测试框架")
    parser.add_argument("--op", type=str, default=None, help="算子名称 (必填，除非使用 --list)")
    parser.add_argument("--list", action="store_true", help="列出所有支持的算子并退出")
    parser.add_argument("--csv", type=str, default=None, help="结果 CSV 路径")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--update", action="store_true", help="更新模式：仅当性能提升 >5%% 时更新 CSV")
    group.add_argument("--generate", action="store_true", help="生成模式：仅补充缺失的测试项 (已存在则跳过)")
    group.add_argument("--compare", action="store_true", help="对比模式：仅显示对比，不写入 CSV")
    
    args, unknown = parser.parse_known_args()

    # 1. 处理 --list
    if args.list:
        list_supported_operators()
        sys.exit(0)

    # 2. 校验 --op
    if not args.op:
        parser.error("参数 --op 是必需的 (除非使用了 --list)")
    op_name = args.op
    op_instance = load_operator_runner(op_name)  
    
    # 3. 注册 Benchmark
    bench.register(create_benchmark_wrapper(op_instance)).set_name(op_instance.name)

    # 4. 确定是否需要临时文件
    active_mode = args.update or args.compare or args.generate
    need_temp_csv = active_mode and args.csv
    temp_csv = None
    
    if need_temp_csv:
        try:
            fd, temp_csv = tempfile.mkstemp(suffix=".csv")
            os.close(fd)
        except Exception as e:
            print(f"Error creating temp file: {e}"); sys.exit(1)

    try:
        run_args = [sys.argv[0]]
        
        has_cli_device = any(x.startswith("--device") or x == "-d" for x in unknown)
        
        if has_cli_device:
            print(">> [Device] 来源: 命令行参数")
        else:
            config_device = op_instance.config.get("device_id")
            if config_device is not None:
                run_args.extend(["--device", str(config_device)])
                print(f">> [Device] 来源: 配置文件 (ID: {config_device})")
            else:
                run_args.extend(["--device", "0"])
                print(">> [Device] 来源: 默认值 (ID: 0)")

        if temp_csv: 
            run_args.extend(["--csv", temp_csv])

        # =======================================================
        # [配置优化]
        # 去掉了可能不支持的 --warmup，改用 --min-time
        # 并配合上面的 create_benchmark_wrapper 中的手动预热
        # =======================================================
        
        # 设置最小运行时间为 1 秒，确保采集足够的样本以获得稳定平均值
        run_args.extend(["--min-time", "1.5"])
        run_args.extend(["--timeout", "60"])
            
        # 把剩余未知参数加进去
        run_args.extend(unknown)
        
        # 运行测试
        bench.run_all_benchmarks(run_args)
        
    except Exception as e:
        if temp_csv and os.path.exists(temp_csv): os.remove(temp_csv)
        raise e

    # 5. 后处理逻辑
    if temp_csv and os.path.exists(temp_csv) and args.csv and active_mode:
        try:
            if args.compare:
                _, c_d = load_csv_data(temp_csv)
                _, h_d = load_csv_data(args.csv)
                perform_comparison(c_d, h_d)
            elif args.update:
                perform_smart_update(temp_csv, args.csv)
            elif args.generate:
                perform_generate(temp_csv, args.csv)
        finally:
            if os.path.exists(temp_csv): os.remove(temp_csv)