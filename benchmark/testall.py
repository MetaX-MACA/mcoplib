import os
import sys
import subprocess
import time

# ================= 配置路径与规则 =================
CONFIG_DIR = "config"
RUNNERS_DIR = "runners" 
TARGET_SCRIPT = "mcoplib_mxbenchmark_ops.py"
OUTPUT_FILE = "teslalloutput.txt"

STATISTICS_DIR = "statistics"
CSV_FILENAME = "mcoplib_ops_performance_C500.csv"
CSV_SAVE_PATH = os.path.join(STATISTICS_DIR, CSV_FILENAME)

RUNNER_PREFIX = "mcoplib_mxbenchmark_"
RUNNER_SUFFIX = "_runners.py"

STD_WIDTHS = [10, 10, 42, 40, 32, 9, 12, 8, 12, 8, 10, 14, 8, 9, 12]
DEFAULT_EXTRA_WIDTH = 12 

IGNORE_OPS = [
    "gptq_shuffle",
    "rotary_embeding",
    "fused_moe_gate_deepseek",
    "copy_blocks",
    "copy_blocks_mla",
    "indexer_k_cache",
    "mctlass_fused_moe_kernel_w4a16",
    "cp_gather_indexer_k_cache"
]
# =================================================

def validate_directories():
    """
    预检函数：检查 config 和 runners 目录下的算子是否一一对应。
    并在返回前过滤掉需要忽略的算子。
    """
    if not os.path.exists(CONFIG_DIR) or not os.path.exists(RUNNERS_DIR):
        print(f"错误: 找不到 '{CONFIG_DIR}' 或 '{RUNNERS_DIR}' 目录。")
        return False, []

    config_ops = {os.path.splitext(f)[0] for f in os.listdir(CONFIG_DIR) if f.endswith(".json")}
    
    runner_ops = set()
    for f in os.listdir(RUNNERS_DIR):
        if f.startswith(RUNNER_PREFIX) and f.endswith(RUNNER_SUFFIX):
            op_name = f[len(RUNNER_PREFIX):-len(RUNNER_SUFFIX)]
            runner_ops.add(op_name)

    missing_in_runners = config_ops - runner_ops
    missing_in_config = runner_ops - config_ops

    if not missing_in_runners and not missing_in_config:
        print(f"[Pass] 校验通过：两个文件夹中的算子数量一致 (共 {len(config_ops)} 个)，且名称完全对应。")
        
        all_ops = sorted(list(config_ops))
        if IGNORE_OPS:
            ignored_actual = [op for op in all_ops if op in IGNORE_OPS]
            final_ops = [op for op in all_ops if op not in IGNORE_OPS]
            if ignored_actual:
                print(f"[Warn]  已根据配置忽略以下 {len(ignored_actual)} 个算子: {', '.join(ignored_actual)}")
            print()
            return True, final_ops
        
        print()
        return True, all_ops
    else:
        print("[Fail] 校验失败：config 目录与 runners 目录内的算子不匹配！")
        print(f"  - Config 数量: {len(config_ops)}")
        print(f"  - Runners 数量: {len(runner_ops)}\n")
        
        if missing_in_runners:
            print("[!] 以下算子在 config 中存在，但缺少对应的 runner 文件:")
            for op in missing_in_runners:
                print(f"    - {op}.json -> 缺少 {RUNNER_PREFIX}{op}{RUNNER_SUFFIX}")
        
        if missing_in_config:
            print("\n[!] 以下算子在 runners 中存在，但缺少对应的 config JSON 文件:")
            for op in missing_in_config:
                print(f"    - {RUNNER_PREFIX}{op}{RUNNER_SUFFIX} -> 缺少 {op}.json")
        
        return False, []

def align_metax_table(lines):
    """
    对 MetaX 的输出表格进行固定列宽的强制对齐。
    """
    aligned_lines = []
    for line in lines:
        if not line.strip().startswith('|'):
            aligned_lines.append(line)
            continue

        raw_parts = line.split('|')
        cells = raw_parts[1:-1] 
        is_separator = all(c in '- ' for c in "".join(cells))

        formatted_cells = []
        for i, cell in enumerate(cells):
            width = STD_WIDTHS[i] if i < len(STD_WIDTHS) else DEFAULT_EXTRA_WIDTH
            if is_separator:
                formatted_cells.append('-' * width)
            else:
                formatted_cells.append(cell.strip().center(width))

        aligned_line = ' | ' + ' | '.join(formatted_cells) + ' | '
        aligned_lines.append(aligned_line)

    return aligned_lines

def run_op_benchmark(op_name):
    command = [
        sys.executable, 
        TARGET_SCRIPT, 
        "--op", op_name,
        "--generate", 
        "--csv", CSV_SAVE_PATH
    ]
    
    acc_verify_line = None
    metax_lines = []

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output_lines = result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        print(f"[Error] 算子 {op_name} 运行异常！")
        output_lines = e.stdout.splitlines() 

    metax_found_idx = -1
    for idx, line in enumerate(output_lines):
        if "[ACC Verify]" in line:
            acc_verify_line = line.strip()
        
        if "### [" in line and "MetaX" in line:
            metax_found_idx = idx

    if metax_found_idx != -1:
        start_idx = metax_found_idx + 2
        end_idx = min(metax_found_idx + 8, len(output_lines))
        raw_metax_lines = [output_lines[i].strip() for i in range(start_idx, end_idx)]
        metax_lines = align_metax_table(raw_metax_lines)

    return acc_verify_line, metax_lines

def ensure_output_dir():
    """确保 statistics 文件夹存在"""
    if not os.path.exists(STATISTICS_DIR):
        try:
            os.makedirs(STATISTICS_DIR)
            print(f"已创建目录: {STATISTICS_DIR}")
        except OSError as e:
            print(f"错误: 无法创建目录 {STATISTICS_DIR}: {e}")
            sys.exit(1)

def main():
    is_valid, op_names = validate_directories()
    if not is_valid:
        print("\n请修正文件对应关系后再运行脚本。程序已终止。")
        return

    ensure_output_dir()

    total_ops_count = len(op_names)

    print("=" * 60)
    print(f"开始执行 Benchmark (Generate 模式)")
    print(f"结果将保存至: {CSV_SAVE_PATH}")
    print(f"共 {total_ops_count} 个算子...")
    print("=" * 60)

    total_start_time = time.time()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        ignored_info = f" (配置中已忽略 {len(IGNORE_OPS)} 个)" if IGNORE_OPS else ""
        f.write(f"=== Benchmark 提取结果 (实际执行算子数量: {total_ops_count} 个{ignored_info}) ===\n")
        f.write(f"模式: Generate | CSV输出: {CSV_SAVE_PATH}\n\n")
        f.flush()

        for op_name in op_names:
            print(f"正在运行算子: {op_name} ... ", end="", flush=True)

            op_start_time = time.time()
            acc_result, metax_result = run_op_benchmark(op_name)
            op_end_time = time.time()
            elapsed_time = op_end_time - op_start_time

            print(f"[完成] (耗时: {elapsed_time:.2f} 秒)") 

            acc_display = acc_result if acc_result else "[ACC Verify]"
            
            print(f" -> {acc_display}")
            if metax_result:
                for line in metax_result:
                    print(line)
            else:
                print("  -> (未捕获到性能表格，数据可能已写入 CSV 或已跳过)")
            print()

            f.write(f"[{op_name}] -> {acc_display}  (耗时: {elapsed_time:.2f} 秒)\n")
            if metax_result:
                for line in metax_result:
                    f.write(line + "\n")
            else:
                f.write("  (未捕获到性能表格)\n")
            f.write("\n")
            
            f.flush() 

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n全部执行完毕！总执行算子: {total_ops_count} 个，总耗时: {total_elapsed_time:.2f} 秒\n")

    print(f"\n全部执行完毕！总耗时: {total_elapsed_time:.2f} 秒")
    print(f"详细日志已保存至 '{OUTPUT_FILE}'")
    print(f"性能数据已保存至 '{CSV_SAVE_PATH}'")

if __name__ == "__main__":
    main()