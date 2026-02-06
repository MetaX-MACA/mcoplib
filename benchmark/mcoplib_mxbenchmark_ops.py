import sys
import os
import json
import csv
import re
import importlib
import argparse
import tempfile
import torch
import difflib  # Used for similarity matching

try:
    import cuda.bench._nvbench as bench
except ImportError:
    print("[ERROR] Runtime environment missing 'nvbench'. Please check configuration.")
    sys.exit(1)

# Import base class for type checking
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase
import mcoplib_mxbenchmark_op_wrapper

# =============================================================================
#  Global Config: Supported Operators
# =============================================================================
SUPPORTED_OPERATORS = [
    "apply_repetition_penalties",
    "awq_dequantize",
    "awq_gemm",
    "awq_to_gptq_4bit",
    "batched_moe_align_block_size",  
    "batched_rotary_embedding",
    "concat_and_cache_mla",
    "concat_mla_absorb_q",           
    "concat_mla_k",                  
    "convert_fp8",
    "convert_vertical_slash_indexes",
    "copy_to_gpu_no_ce",             
    "cp_gather_cache",
    "cp_gather_indexer_k_quant_cache", 
    "cutlass_group_gemm_supported",
    "cutlass_scaled_mm_azp",
    "cutlass_scaled_mm_supports_block_fp8",
    "cutlass_scaled_mm_supports_fp4",
    "cutlass_scaled_mm_supports_fp8",
    "cutlass_scaled_mm",
    "dynamic_per_token_scaled_fp8_quant",
    "dynamic_scaled_int8_quant",
    "fast_topk_transform_fused",        
    "fast_topk_transform_ragged_fused", 
    "fast_topk",                        
    "fatrelu_and_mul",
    "fused_add_rms_norm_static_fp8_quant",
    "fused_add_rms_norm",
    "fused_add_rmsnorm",            
    "fused_bias_dropout",
    "fused_mla_absorb_rotary_emb",   
    "fused_rope_fwd",
    "gather_and_maybe_dequant_cache",
    "gelu_and_mul",
    "gelu_fast",
    "gelu_new",
    "gelu_quick",
    "gelu_tanh_and_mul",
    "get_cuda_view_from_cpu_tensor",
    "gptq_gemm",
    "mctlass_moe_w4a16_gemm_kernel_mnk",   
    "merge_attn_states",        
    "merge_state_v2",             
    "merge_state",               
    "moe_align_block_size",          
    "moe_lora_align_block_size",     
    "moe_sum",
    "mul_and_silu",
    "paged_attention_v1",
    "paged_attention_v2",
    "reshape_and_cache_flash",
    "reshape_and_cache",
    "rms_norm_static_fp8_quant",
    "rms_norm",
    "selective_scan_fwd",            
    "silu_and_mul_quant",
    "silu_and_mul",
    "static_scaled_fp8_quant",
    "static_scaled_int8_quant",      
    "swap_blocks",              
    "swigluoai_and_mul",
    "top_k_per_row_decode",
    "top_k_per_row",
    "topk_softmax",
    "transfer_kv_all_layer_direct_lf_pf",  
    "transfer_kv_direct",              
    "transfer_kv_per_layer_direct_pf_lf"  
]

# =============================================================================
#  Loader Logic
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
    # 1. Find Config File (.json) - Supports Fuzzy Search
    # -----------------------------------------------------------
    config_dir = os.path.join(current_dir, "config")
    if not os.path.exists(config_dir):
        print(f"[ERROR] Config directory not found: {config_dir}")
        sys.exit(1)

    target_json_path = None
    exact_json = os.path.join(config_dir, f"{op_name}.json")
    
    if os.path.exists(exact_json):
        target_json_path = exact_json
    else:
        print(f"[INFO] Exact config match not found for '{op_name}', trying fuzzy search...")
        try:
            pattern = re.compile(op_name, re.IGNORECASE)
            json_files = [f for f in os.listdir(config_dir) if f.endswith(".json")]
            matched_jsons = [f for f in json_files if pattern.search(f)]

            if len(matched_jsons) == 0:
                print(f"[ERROR] Config file not found, and fuzzy search for '{op_name}' yielded no results.")
                sys.exit(1)
            elif len(matched_jsons) == 1:
                target_json_path = os.path.join(config_dir, matched_jsons[0])
                print(f"[CONFIG] Selected: {matched_jsons[0]}")
            else:
                best_matches = difflib.get_close_matches(op_name, matched_jsons, n=1, cutoff=0)
                best_match = best_matches[0] if best_matches else matched_jsons[0]
                print(f"[WARN] Multiple config matches found. Auto-selecting best match: {best_match}")
                target_json_path = os.path.join(config_dir, best_match)
        except re.error as e:
            print(f"[ERROR] Regex error: {e}")
            sys.exit(1)
            
    canonical_name = os.path.splitext(os.path.basename(target_json_path))[0]
    
    with open(target_json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # -----------------------------------------------------------
    # 2. Find Runner File (.py)
    # -----------------------------------------------------------
    runners_dir = os.path.join(current_dir, "runners")
    if not os.path.exists(runners_dir):
        print(f"[ERROR] Runners directory not found: {runners_dir}")
        sys.exit(1)

    exact_runner_name = f"mcoplib_mxbenchmark_{canonical_name}_runners"
    found_module_name = None
    
    try:
        module_path = f"runners.{exact_runner_name}"
        if importlib.util.find_spec(module_path) is not None:
            found_module_name = module_path
    except:
        pass

    if not found_module_name:
        print(f"[INFO] Standard runner '{exact_runner_name}.py' not found, searching...")
        py_files = [f for f in os.listdir(runners_dir) if f.endswith(".py") and f != "__init__.py"]
        
        try:
            matched_runners = []
            
            # Strategy A: Config Name
            if canonical_name:
                pattern_canon = re.compile(canonical_name, re.IGNORECASE)
                matched_runners = [f for f in py_files if pattern_canon.search(f)]

            # Strategy B: Input Name Fallback
            if not matched_runners and op_name and op_name != canonical_name:
                print(f"[INFO] Canonical name match failed, falling back to input name '{op_name}'...")
                pattern_op = re.compile(op_name, re.IGNORECASE)
                matched_runners = [f for f in py_files if pattern_op.search(f)]
                
        except re.error as e:
            print(f"[ERROR] Regex error: {e}")
            matched_runners = []
        
        target_runner_file = None
        if len(matched_runners) == 0:
            print(f"[ERROR] No relevant Python files found in runners directory.")
            print(f"        Keywords tried: '{canonical_name}' and '{op_name}'")
            sys.exit(1)
        elif len(matched_runners) == 1:
            target_runner_file = matched_runners[0]
        else:
            search_key = canonical_name if canonical_name else op_name
            best_matches = difflib.get_close_matches(search_key, matched_runners, n=1, cutoff=0)
            target_runner_file = best_matches[0] if best_matches else matched_runners[0]
            print(f"[WARN] Multiple runner matches found. Auto-selecting best match: {target_runner_file}")

        if target_runner_file:
            module_base = target_runner_file.replace(".py", "")
            found_module_name = f"runners.{module_base}"
            print(f"[RUNNER] Selected Module: {module_base}")
    
    # -----------------------------------------------------------
    # 3. Load Module & Class
    # -----------------------------------------------------------
    try:
        module = importlib.import_module(found_module_name)
    except ImportError as e:
        print(f"\n[ERROR] Failed to load module '{found_module_name}'\nDetails: {e}")
        sys.exit(1)

    target_cls = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, OpBenchmarkBase) and attr is not OpBenchmarkBase:
            target_cls = attr
            break
            
    if target_cls is None:
        print(f"[ERROR] No subclass of OpBenchmarkBase found in {found_module_name}.")
        sys.exit(1)

    print(f"[LOADER] Loaded Class: {target_cls.__name__} (Benchmark Name: {canonical_name})")
    
    return target_cls(canonical_name, config)

# =============================================================================
#  Benchmark Wrapper (Integrated Sync Fix)
# =============================================================================
def create_benchmark_wrapper(op_instance):
    def benchmark_func(state):
        dev_id = state.get_device()
        is_verified = False
        
        # 1. Verification
        try:
            passed, diff_val = op_instance.run_verification(dev_id)
            state.add_summary("Acc_Pass", "Yes" if passed else "No")
            state.add_summary("Cos_Dist", f"{diff_val:.2e}") 
            print(f"\n[VERIFY] {op_instance.name} -> {'PASS' if passed else 'FAIL'} (1-CosSim: {diff_val:.2e})")
            is_verified = passed
        except Exception as e:
            is_verified = False
            print(f"\n[VERIFY] Error: {e}")
            state.add_summary("Acc_Pass", "Error")

        if not is_verified:
            print(f"\n[FATAL] {op_instance.name} verification failed. Terminating all processes...")
            import os
            os._exit(0)

        # 2. Manual Warmup
        print(f"  >> [WARMUP] Running 10 iterations...", end="", flush=True)
        try:
            for _ in range(10):
                op_instance.run_verification(dev_id)
            print(" Done.")
        except Exception as e:
            print(f" (Failed: {e})", end="")

        # 3. Sample Configuration
        raw_samples = op_instance.config.get("samples", 100)
        target_samples = max(1, raw_samples - 16) 
        state.set_min_samples(target_samples)
        op_instance.define_metrics(state)
        
        tc_s = torch.cuda.ExternalStream(
            stream_ptr=state.get_stream().addressof(), 
            device=torch.cuda.device(dev_id)
        )
        launcher = op_instance.prepare_and_get_launcher(dev_id, tc_s)
        
        # 4. Execution (Sync Fix)
        force_sync = getattr(op_instance, '_force_sync', False)
        
        if force_sync:
            try:
                state.exec(launcher, sync=True)
            except TypeError as e:
                print(f"\n[WARN] 'state.exec(..., sync=True)' failed: {e}")
                print("       Falling back to default async mode.")
                state.exec(launcher)
        else:
            state.exec(launcher)

    return benchmark_func


# =============================================================================
#  CSV Utils
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
    return f"{seconds * 1e6:.3f} us"

def get_row_key(row_dict, header):
    exclude = ["Samples", "CPU Time", "GPU Time", "Noise", "Elem/s", "GlobalMem", "BWUtil", "Acc_Pass", "Max_Diff", "Cos_Dist","Batch GPU"]
    key_cols = [h for h in header if not any(x in h for x in exclude) and h != "Skipped"]
    return tuple(row_dict.get(k, "") for k in key_cols)

def get_op_display_name(row):
    return row.get("op_name", row.get("Op", row.get("Benchmark", "?")))

def get_effective_gpu_time(row):
    t_batch = parse_time_val(row.get("Batch GPU (sec)", ""))
    if t_batch is not None: return t_batch
    
    t_batch_raw = parse_time_val(row.get("Batch GPU", ""))
    if t_batch_raw is not None: return t_batch_raw

    t_gpu = parse_time_val(row.get("GPU Time (sec)", ""))
    if t_gpu is not None: return t_gpu
        
    t_gpu_raw = parse_time_val(row.get("GPU Time", ""))
    if t_gpu_raw is not None: return t_gpu_raw
        
    return None

def _write_csv(path, header, rows):
    try:
        target_dir = os.path.dirname(path)
        if target_dir and not os.path.exists(target_dir): os.makedirs(target_dir)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        print(f"[ERROR] Failed to write CSV: {e}")

# =============================================================================
#  Update / Generate / Compare Logic
# =============================================================================
def perform_smart_update(temp_csv_path, target_csv_path):
    print("\n" + "="*80 +f"\n{' Smart Update Mode (>5% Gain) ':^80}\n" + "="*80)
    raw_h, raw_r = load_csv_data(temp_csv_path)
    new_h, new_r = preprocess_data(raw_h, raw_r)
    if not new_r: return

    old_raw_h, old_raw_r = load_csv_data(target_csv_path)
    old_h, old_r = preprocess_data(old_raw_h, old_raw_r)

    combined_header = list(old_h)
    for col in new_h:
        if col not in combined_header: combined_header.append(col)
    
    if "op_name" in combined_header:
        combined_header.remove("op_name")
        combined_header.insert(0, "op_name")

    history_map = {get_row_key(r, combined_header): idx for idx, r in enumerate(old_r)}
    
    updated_cnt = 0
    kept_cnt = 0
    final_rows = list(old_r)

    for new_row in new_r:
        for col in combined_header:
            if col not in new_row: new_row[col] = ""
            
        key = get_row_key(new_row, combined_header)
        match_idx = history_map.get(key, -1)
        
        new_time = get_effective_gpu_time(new_row)
        d_type = new_row.get("dtype", "")
        op_str = get_op_display_name(new_row)
        if d_type: op_str += f" [{d_type}]"

        if match_idx >= 0:
            old_row = final_rows[match_idx]
            old_time = get_effective_gpu_time(old_row)
            
            if new_time is not None and old_time is not None and old_time > 0:
                ratio = (old_time - new_time) / old_time
                if ratio > 0.05:
                    final_rows[match_idx].update(new_row)
                    updated_cnt += 1
                    print(f"[UPDATE] {op_str:<35} | Gain: {ratio*100:>6.2f}%")
                else:
                    kept_cnt += 1
                    if ratio >= 0:
                        print(f"[KEEP]   {op_str:<35} | Gain: {ratio*100:>6.2f}%")
                    else:
                        print(f"[KEEP]   {op_str:<35} | Loss: {ratio*100:>6.2f}%")
            else:
                kept_cnt += 1
                print(f"[KEEP]   {op_str:<35} | N/A (Invalid Time)")
        
    _write_csv(target_csv_path, combined_header, final_rows)
    print("-" * 80 + f"\n[SUMMARY] Updated: {updated_cnt}, Kept: {kept_cnt}\n" + "=" * 80 + "\n")

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

    combined_header = list(old_h)
    if not combined_header: combined_header = list(new_h)
    else:
        for col in new_h:
            if col not in combined_header: combined_header.append(col)
            
    if "op_name" in combined_header:
        combined_header.remove("op_name")
        combined_header.insert(0, "op_name")

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
            print(f"[SKIP]   {op_str:<35} (Exists)")
            skip_cnt += 1
        else:
            rows_to_append.append(new_row)
            print(f"[APPEND] {op_str:<35}")
            append_cnt += 1

    final_rows.extend(rows_to_append)
    if append_cnt > 0:
        _write_csv(target_csv_path, combined_header, final_rows)
    else:
        print("[INFO] No new data to write.")
    print("-" * 80 + f"\n[SUMMARY] Appended: {append_cnt}, Skipped: {skip_cnt}\n" + "=" * 80 + "\n")

def perform_comparison(cur_raw, hist_raw):
    _, cur_rows = preprocess_data([], cur_raw)
    _, hist_rows = preprocess_data([], hist_raw)
    if not cur_rows: return
    print("\n" + "="*95 + "\n" + f"{' Performance Comparison ':^95}" + "\n" + "="*95)
    
    row_fmt = "{:<15} | {:<15} | {:<10} | {:<15} | {:<15} | {:<20}"
    dummy_header = list(cur_rows[0].keys())
    
    for row in cur_rows:
        key = get_row_key(row, dummy_header)
        
        gpu = get_effective_gpu_time(row)
        cpu = parse_time_val(row.get("CPU Time (sec)", ""))
        op_name = get_op_display_name(row)
        d_type = row.get("dtype", "-")
        
        time_label = "Batch GPU"
        if parse_time_val(row.get("Batch GPU (sec)", "")) is None and \
           parse_time_val(row.get("GPU Time (sec)", "")) is not None:
             time_label = "GPU Time (Sync)"

        raw_acc = row.get("Acc_Pass", "N/A")
        if raw_acc == "Yes": acc_status = "Pass"
        elif raw_acc == "No": acc_status = "Fail"
        else: acc_status = raw_acc

        print(f" {op_name} | {row.get('Shape', '')} | {d_type}")
        print("-" * 95)
        print(row_fmt.format("Type", time_label, "Ratio", "CPU Time", "Ratio", ""))
        print("-" * 95)
        print(row_fmt.format("Current", format_duration(gpu), "-", format_duration(cpu), "-", ""))
        
        matches = []
        for idx, h_row in enumerate(hist_rows):
            if get_row_key(h_row, dummy_header) == key: 
                matches.append((idx, h_row))
        
        perf_ratio_str = "None"

        if matches:
            _, h_row = matches[-1] 
            h_gpu = get_effective_gpu_time(h_row)
            h_cpu = parse_time_val(h_row.get("CPU Time (sec)", ""))
            
            if gpu and h_gpu:
                perf_ratio_str = f"{h_gpu/gpu*100:.2f}%"
                gr = perf_ratio_str
            else:
                gr = "N/A"

            cr = f"{h_cpu/cpu*100:.2f}%" if (cpu and h_cpu) else "N/A"
            print(row_fmt.format("Base", format_duration(h_gpu), gr, format_duration(h_cpu), cr, ""))
        else:
            print(f"{'Base':<15} | {'N/A':<15} | {'N/A':<10} | {'N/A':<15} | {'N/A':<15} |")
        
        print("Result:")
        print(f"Acc verify:{acc_status}")
        print(f"Performance verify:{perf_ratio_str}")
        print("\n")

# =============================================================================
#  Main Execution Block
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCOPLIB Operator Performance Benchmark")
    parser.add_argument("--op", type=str, default=None, help="Operator name (Required, unless --list is used)")
    parser.add_argument("--list", action="store_true", help="List all supported operators and exit")
    parser.add_argument("--csv", type=str, default=None, help="Path to result CSV")
    
    group = parser.add_mutually_exclusive_group()
    # 修复：这里的 > 5% 必须写成 > 5%%，否则 argparse 报错 incomplete format
    group.add_argument("--update", action="store_true", help="Update Mode: Update CSV only if performance gain > 5%%")
    group.add_argument("--generate", action="store_true", help="Generate Mode: Only fill missing test items (Skip existing)")
    group.add_argument("--compare", action="store_true", help="Compare Mode: Display comparison only, no CSV write")
    
    args, unknown = parser.parse_known_args()

    # 1. Handle --list
    if args.list:
        list_supported_operators()
        sys.exit(0)

    # 2. Validate Core Argument --op
    if not args.op:
        # Print full help first
        parser.print_help()
        print("\n" + "-"*80)
        print(" [ERROR] Argument --op is required (unless --list is used)")
        print(" [TIP]   Please use --help to view the standard usage.")
        print("-"*80 + "\n")
        sys.exit(1)
    
    # 3. Load Operator
    op_name = args.op
    op_instance = load_operator_runner(op_name)
    
    # 4. Handle CSV Default Logic
    default_csv_path = "statistics/mcoplib_ops_performance_C500.csv"
    if args.csv is None:
        print(f"[WARN] --csv not specified, using default path: {default_csv_path}")
        args.csv = default_csv_path

    # 5. Pre-flight Check (File/Directory Existence)
    active_mode = args.update or args.compare or args.generate
    
    if active_mode:
        abs_csv_path = os.path.abspath(args.csv)
        csv_dir = os.path.dirname(abs_csv_path)

        # Check Directory
        if not os.path.exists(csv_dir) and csv_dir != "":
            print(f"[ERROR] Target directory does not exist: {csv_dir}")
            print("        Please check path or create directory.")
            if args.compare:
                print("Result:")
                print("Acc verify:None")
                print("Performance verify:None")
                print("\n")
            sys.exit(1)

        # Check File (Required for compare/update)
        if (args.compare or args.update) and not os.path.exists(abs_csv_path):
            print(f"[ERROR] Target CSV file does not exist: {abs_csv_path}")
            print(f"        Mode '--{'compare' if args.compare else 'update'}' requires this file for baseline data.")
            if args.compare:
                print("Result:")
                print("Acc verify:None")
                print("Performance verify:None")
                print("\n")
            sys.exit(1)

        # =========================================================================
        # Check if Operator exists in CSV (Only for compare/update)
        # =========================================================================
        if args.compare or args.update:
            check_header, check_rows = load_csv_data(abs_csv_path)
            _, check_clean_rows = preprocess_data(check_header, check_rows)
            
            target_name = op_instance.name
            op_found = False
            
            for row in check_clean_rows:
                if row.get("op_name") == target_name:
                    op_found = True
                    break
            
            if not op_found:
                print("\n" + "-"*80)
                print(f" [WARN] Operator '{target_name}' not found in baseline CSV.")
                print(f"        CSV Path: {abs_csv_path}")
                print(f"        Mode: --{'compare' if args.compare else 'update'}")
                print("        >> Skipping execution due to missing baseline data.")
                print("-"*80 + "\n")
                
                if args.compare:
                    print("Result:")
                    print("Acc verify:None")
                    print("Performance verify:None")
                    print("\n")
                
                sys.exit(0) 
        # =========================================================================

    # 6. Register Benchmark
    bench.register(create_benchmark_wrapper(op_instance)).set_name(op_instance.name)

    # 7. Prepare Temp File
    need_temp_csv = active_mode
    temp_csv = None
    
    if need_temp_csv:
        try:
            fd, temp_csv = tempfile.mkstemp(suffix=".csv")
            os.close(fd)
        except Exception as e:
            print(f"[ERROR] Error creating temp file: {e}"); sys.exit(1)

    try:
        run_args = [sys.argv[0]]
        
        has_cli_device = any(x.startswith("--device") or x == "-d" for x in unknown)
        
        if has_cli_device:
            print(">> [DEVICE] Source: CLI Argument")
        else:
            config_device = op_instance.config.get("device_id")
            if config_device is not None:
                run_args.extend(["--device", str(config_device)])
                print(f">> [DEVICE] Source: Config File (ID: {config_device})")
            else:
                run_args.extend(["--device", "0"])
                print(">> [DEVICE] Source: Default (ID: 0)")

        if temp_csv: 
            run_args.extend(["--csv", temp_csv])

        run_args.extend(["--min-time", "1.5"])
        run_args.extend(["--timeout", "600"])
        run_args.extend(["--throttle-threshold", "0"])
        run_args.extend(unknown)
        
        bench.run_all_benchmarks(run_args)
        
    except Exception as e:
        if temp_csv and os.path.exists(temp_csv): os.remove(temp_csv)
        raise e

    # 8. Post-processing
    if temp_csv and os.path.exists(temp_csv) and active_mode:
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