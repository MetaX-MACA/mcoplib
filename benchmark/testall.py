"""
testall.py — Batch driver for mcoplib operator benchmarks.

Usage:
  python testall.py                      # default: --generate
  python testall.py --generate           # only fill missing CSV rows (skip existing)
  python testall.py --compare            # compare each op vs CSV baseline; fail if >5%% slower
  python testall.py --update             # overwrite CSV rows for all ops
  python testall.py --csv PATH           # override CSV path (default: statistics/mcoplib_ops_performance_C500.csv)
  python testall.py --output PATH        # specify output Excel file path (.xlsx) for compare report (supports absolute and relative paths)
  python testall.py --ops a,b,c          # only run named ops
  python testall.py --dry-run            # list ops without running

The script:
  1. Calls `python mcoplib_mxbenchmark_ops.py --list` to enumerate supported ops.
  2. For each op, runs `python mcoplib_mxbenchmark_ops.py --op <name> --<mode> --csv <csv>`.
  3. Parses stdout to extract accuracy status and performance status.
  4. Prints a per-op summary table and a final roll-up (success/failure counts + failing op names).
  5. In --compare mode, generates an Excel report (.xlsx) with summary and per-operator sheets.
"""

import os
import re
import sys
import csv
import ast
import time
import argparse
import subprocess
from datetime import datetime
try:
    import openpyxl
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    from openpyxl.utils import get_column_letter
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("[WARN] openpyxl not installed. Excel report generation will be skipped.")
    print("[INFO] Install openpyxl to enable Excel reports: pip install openpyxl")

TARGET_SCRIPT = "mcoplib_mxbenchmark_ops.py"
STATISTICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "statistics")
def _resolve_default_csv_filename():
    """Device-aware default result CSV. Test 'C600-UL' before 'C600-U'
    (substring). Falls back to C500 if torch/GPU is unavailable."""
    try:
        import torch
        name = torch.cuda.get_device_name(0)
    except Exception:
        name = ""
    upper = (name or "").upper()
    if "C600-UL" in upper:
        return "mcoplib_ops_performance_C600ul.csv"
    if "C600-U" in upper:
        return "mcoplib_ops_performance_C600u.csv"
    return "mcoplib_ops_performance_C500.csv"


CSV_FILENAME = _resolve_default_csv_filename()
DEFAULT_CSV = os.path.join(STATISTICS_DIR, CSV_FILENAME)
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testall_output.txt")

LIST_OP_PREFIX = "  * "
VERIFY_RE = re.compile(r"\[VERIFY\]\s+(\S+)\s*->\s*(PASS|FAIL)", re.IGNORECASE)
ACC_VERIFY_RE = re.compile(r"Acc verify:\s*(\w+)", re.IGNORECASE)
PERF_VERIFY_RE = re.compile(r"Performance verify:\s*(\S+)", re.IGNORECASE)
APPEND_RE = re.compile(r"\[APPEND\]\s+(\S+)", re.IGNORECASE)
SKIP_RE = re.compile(r"\[SKIP\]\s+(\S+)", re.IGNORECASE)
UPDATE_RE = re.compile(r"\[UPDATE\]\s+(\S+)", re.IGNORECASE)
IGNORED_RE = re.compile(r"\[IGNORED\]\s+(\S+)", re.IGNORECASE)
SUMMARY_RE = re.compile(
    r"\[SUMMARY\]\s+(Appended|Skipped|Updated|Kept):\s*(\d+)", re.IGNORECASE
)

# Threshold for performance regression in --compare mode (must match the
# single-op script's own 5%% rule).
PERF_REGRESSION_THRESHOLD_PCT = 5.0


def _load_ignored_operators():
    """Read IGNORED_OPERATORS list from mcoplib_mxbenchmark_ops.py source.

    Parses the module AST instead of importing it, so we don't trigger
    nvbench import side effects (which sys.exit on failure).
    """
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TARGET_SCRIPT)
    if not os.path.exists(script_path):
        return set()
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception:
        return set()

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "IGNORED_OPERATORS":
                    if isinstance(node.value, ast.List):
                        result = set()
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                result.add(elt.value)
                        return result
    return set()


def get_supported_operators():
    """Call the single-op script's --list and parse the operator names."""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TARGET_SCRIPT)
    cmd = [sys.executable, script_path, "--list"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=60
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] `--list` failed: {e}")
        print(e.stdout)
        print(e.stderr)
        return []
    except subprocess.TimeoutExpired:
        print("[ERROR] `--list` timed out")
        return []

    ops = []
    for line in result.stdout.splitlines():
        if line.startswith(LIST_OP_PREFIX):
            ops.append(line[len(LIST_OP_PREFIX):].strip())
    return ops


def parse_run_output(stdout):
    """Extract accuracy and performance status from a single op run."""
    info = {
        "verify_pass": None,       # True/False/None
        "acc_verify": None,        # "Pass"/"Fail"/"None"/None
        "perf_verify": None,       # "NN.NN%"/"None"/None
        "csv_action": None,        # "APPEND"/"SKIP"/"UPDATE"/"IGNORED"/None
        "fatal": False,
        "error_msg": None,
    }

    for line in stdout.splitlines():
        m = VERIFY_RE.search(line)
        if m:
            info["verify_pass"] = (m.group(2).upper() == "PASS")
            continue
        m = ACC_VERIFY_RE.search(line)
        if m:
            info["acc_verify"] = m.group(1).strip()
            continue
        m = PERF_VERIFY_RE.search(line)
        if m:
            info["perf_verify"] = m.group(1).strip()
            continue
        m = APPEND_RE.search(line)
        if m:
            info["csv_action"] = "APPEND"
            continue
        m = SKIP_RE.search(line)
        if m:
            info["csv_action"] = "SKIP"
            continue
        m = UPDATE_RE.search(line)
        if m:
            info["csv_action"] = "UPDATE"
            continue
        m = IGNORED_RE.search(line)
        if m:
            info["csv_action"] = "IGNORED"
            continue
        if "[FATAL]" in line:
            info["fatal"] = True
            info["error_msg"] = line.strip()
            continue

    return info


def _op_in_csv(op_name, csv_path):
    """Return True if op_name appears as op_name column in the CSV."""
    if not os.path.exists(csv_path):
        return False
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("op_name") == op_name:
                    return True
    except Exception:
        return False
    return False


def classify_result(op_name, mode, info, returncode, csv_path, csv_before):
    """Decide whether an op's run was a success or failure for the given mode.

    Returns (status, detail) where status is one of:
      SUCCESS, FAILED, SKIPPED.
    """
    # Ignored operators are always SKIPPED, regardless of mode.
    if info.get("csv_action") == "IGNORED":
        return "SKIPPED", "ignored"

    if returncode != 0:
        return "SKIPPED", f"exit={returncode}"

    if info["fatal"]:
        return "SKIPPED", "verification fatal"

    # compare mode: needs Acc verify:Pass and Performance verify within 5%.
    # 没跑成功的情况归类为 SKIPPED，性能下降超过5%归类为 FAILED
    if mode == "compare":
        acc = info["acc_verify"]
        perf = info["perf_verify"]
        if acc is None or acc.lower() == "none":
            # Could be that the op isn't in CSV (single-op script exits early).
            if not _op_in_csv(op_name, csv_path):
                return "SKIPPED", "no baseline"
            # Op is in CSV but no Acc verify line captured — likely the single-op
            # script's verification failed silently (os._exit before flush).
            return "SKIPPED", "no acc result"
        if acc.lower() != "pass":
            return "SKIPPED", f"acc={acc}"
        if perf is None or perf.lower() == "none":
            return "SKIPPED", "no perf ratio"
        try:
            ratio_pct = float(perf.rstrip("%"))
        except ValueError:
            return "SKIPPED", f"bad perf ratio: {perf}"
        slowdown_pct = (1.0 - ratio_pct / 100.0) * 100.0
        if slowdown_pct > PERF_REGRESSION_THRESHOLD_PCT:
            return "FAILED", f"perf={perf} (slow {slowdown_pct:.2f}%)"
        return "SUCCESS", f"acc=Pass perf={perf}"

    # generate mode: APPEND = success, SKIP = skipped (already exists).
    if mode == "generate":
        if info["csv_action"] == "APPEND":
            return "SUCCESS", "appended"
        if info["csv_action"] == "SKIP":
            return "SKIPPED", "exists (use --update to refresh)"
        if info["verify_pass"] is False:
            return "FAILED", "acc fail"
        # No csv_action line captured — could be silent verification failure
        # (the single-op script calls os._exit(0) on acc failure, dropping
        # buffered output). Use CSV existence as the source of truth.
        if not csv_before and _op_in_csv(op_name, csv_path):
            return "SUCCESS", "appended(csv)"
        if csv_before and _op_in_csv(op_name, csv_path):
            return "SKIPPED", "exists(csv)"
        return "FAILED", "no result"

    # update mode: UPDATE = success.
    if mode == "update":
        if info["csv_action"] == "UPDATE":
            return "SUCCESS", "updated"
        if info["verify_pass"] is False:
            return "FAILED", "acc fail"
        # Fall back to CSV existence check.
        if _op_in_csv(op_name, csv_path):
            return "SUCCESS", "updated(csv)"
        return "FAILED", "no result"

    return "SUCCESS", "ok"


def run_one_op(op_name, mode, csv_path, output_csv=None):
    """Run the single-op script for one op. Returns (info, returncode, elapsed)."""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), TARGET_SCRIPT)
    # Ensure csv_path is absolute to avoid working directory issues in subprocess
    csv_path = os.path.abspath(csv_path)
    cmd = [
        sys.executable, script_path,
        "--op", op_name,
        f"--{mode}",
        "--csv", csv_path,
    ]
    if output_csv and mode == "compare":
        # Ensure output_csv is absolute
        output_csv = os.path.abspath(output_csv)
        cmd.extend(["--output", output_csv])
    
    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800
        )
        rc = result.returncode
        stdout = result.stdout
    except subprocess.TimeoutExpired:
        return None, 124, time.time() - start
    elapsed = time.time() - start
    return parse_run_output(stdout), rc, elapsed


def print_summary_table(rows):
    """Print a per-op table. rows: list of dict with keys op, acc, perf, status, detail, elapsed."""
    cols = [
        ("Op", 36), ("Acc", 8), ("Perf", 14),
        ("Status", 10), ("Detail", 28), ("Time", 9),
    ]
    header = " | ".join(c.ljust(w) for c, w in cols)
    sep = "-+-".join("-" * w for _, w in cols)
    print(header)
    print(sep)
    for r in rows:
        line = " | ".join([
            r["op"][:36].ljust(36),
            str(r["acc"] or "-").ljust(8),
            str(r["perf"] or "-").ljust(14),
            r["status"].ljust(10),
            str(r["detail"] or "")[:28].ljust(28),
            f"{r['elapsed']:.1f}s".ljust(9),
        ])
        print(line)


def ensure_csv_exists(csv_path):
    """For --compare/--update the single-op script requires the CSV to exist."""
    if not os.path.exists(csv_path):
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)
        # Write an empty CSV with the canonical header so --compare won't hard-fail
        # at the file-existence check; instead each op will hit the "not found in
        # baseline" path and report SKIPPED.
        header = (
            "op_name,Device,Device Name,Acc_Pass,Cos_Dist,Op,dtype,Shape,Samples,"
            "CPU Time (sec),Noise,GPU Time (sec),Noise,Elem/s (elem/sec),"
            "GlobalMem BW (bytes/sec),BWUtil,Samples,Batch GPU (sec)"
        )
        with open(csv_path, "w", newline="") as f:
            f.write(header + "\n")


def generate_excel_report(mode, csv_path, output_csv, total_ops, success, failed, skipped, 
                          total_elapsed, failing_ops, skipped_ops, rows, output_path=None):
    """Generate Excel report with summary and per-operator sheets.
    
    Args:
        output_path: Optional path for the Excel file. If specified, the report
                     will be saved to this path (supports absolute and relative paths).
                     If None, a default timestamped filename in STATISTICS_DIR is used.
    """
    if not EXCEL_AVAILABLE:
        print("[WARN] openpyxl not available, skipping Excel report generation")
        return
    
    # Determine output file path
    if output_path:
        # Use user-specified output path (convert to absolute if relative)
        filepath = os.path.abspath(output_path)
        # Ensure the directory exists
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    else:
        # Generate default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_compare.xlsx"
        filepath = os.path.join(STATISTICS_DIR, filename)
        # Ensure statistics directory exists
        if not os.path.exists(STATISTICS_DIR):
            os.makedirs(STATISTICS_DIR, exist_ok=True)
    
    # Create workbook
    wb = openpyxl.Workbook()
    
    # Use the default sheet as Summary sheet (rename it)
    if "Sheet" in wb.sheetnames:
        ws_summary = wb["Sheet"]
        ws_summary.title = "Summary"
    else:
        ws_summary = wb.create_sheet("Summary", 0)
    
    # Summary sheet styling
    title_font = Font(name='Arial', size=14, bold=True)
    header_font = Font(name='Arial', size=12, bold=True)
    normal_font = Font(name='Arial', size=11)
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_alignment = Alignment(horizontal='center', vertical='center')
    normal_alignment = Alignment(horizontal='left', vertical='center')
    
    # Title
    ws_summary['A1'] = f"Testall.py Report - {mode.upper()} Mode"
    ws_summary['A1'].font = title_font
    ws_summary.merge_cells('A1:B1')
    
    ws_summary['A2'] = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ws_summary['A2'].font = normal_font
    ws_summary.merge_cells('A2:B2')
    
    ws_summary['A3'] = f"CSV File: {os.path.basename(csv_path)}"
    ws_summary['A3'].font = normal_font
    ws_summary.merge_cells('A3:B3')
    
    # Empty row
    ws_summary['A5'] = ""
    
    # Statistics table
    ws_summary['A6'] = "Statistics"
    ws_summary['A6'].font = header_font
    ws_summary.merge_cells('A6:B6')
    
    # Statistics data
    stats_data = [
        ("Total Operators", total_ops),
        ("Success", success),
        ("Failed", failed),
        ("Skipped", skipped),
        ("Total Time", f"{total_elapsed:.2f}s"),
    ]
    
    for i, (label, value) in enumerate(stats_data, start=7):
        ws_summary[f'A{i}'] = label
        ws_summary[f'A{i}'].font = normal_font
        ws_summary[f'B{i}'] = value
        ws_summary[f'B{i}'].font = normal_font
    
    # Empty row
    next_row = len(stats_data) + 8
    ws_summary[f'A{next_row}'] = ""
    
    # Failed operators
    if failing_ops:
        next_row += 1
        ws_summary[f'A{next_row}'] = "Failed Operators"
        ws_summary[f'A{next_row}'].font = header_font
        ws_summary[f'A{next_row}'].fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
        ws_summary.merge_cells(f'A{next_row}:B{next_row}')
        
        for i, op in enumerate(failing_ops, start=1):
            ws_summary[f'A{next_row + i}'] = f"{i}."
            ws_summary[f'A{next_row + i}'].font = normal_font
            ws_summary[f'B{next_row + i}'] = op
            ws_summary[f'B{next_row + i}'].font = normal_font
        next_row += len(failing_ops) + 1
    
    # Skipped operators
    if skipped_ops:
        next_row += 1
        ws_summary[f'A{next_row}'] = "Skipped Operators"
        ws_summary[f'A{next_row}'].font = header_font
        ws_summary[f'A{next_row}'].fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
        ws_summary.merge_cells(f'A{next_row}:B{next_row}')
        
        for i, op in enumerate(skipped_ops, start=1):
            ws_summary[f'A{next_row + i}'] = f"{i}."
            ws_summary[f'A{next_row + i}'].font = normal_font
            ws_summary[f'B{next_row + i}'] = op
            ws_summary[f'B{next_row + i}'].font = normal_font
    
    # Adjust column widths
    ws_summary.column_dimensions['A'].width = 25
    ws_summary.column_dimensions['B'].width = 40
    
    # Create per-operator sheets
    used_sheet_names = set()
    for row in rows:
        op_name = row['op']
            
        # Create sheet for this operator
        # Excel sheet names: max 31 chars, cannot contain \ / ? * [ ] :
        # Also cannot contain control characters (newline, tab, etc.)
        sheet_name = op_name
        # Replace all illegal characters
        for ch in ['\\', '/', '?', '*', '[', ']', ':']:
            sheet_name = sheet_name.replace(ch, '_')
        # Remove control characters (newline, tab, carriage return, etc.)
        sheet_name = ''.join(ch if ch.isprintable() and ch not in '\r\n\t' else '_' for ch in sheet_name)
        sheet_name = sheet_name.strip("'")
        if not sheet_name or sheet_name.isspace():
            sheet_name = "unnamed"
        sheet_name = sheet_name.strip()[:31]
        
        # Handle duplicate sheet names
        base_name = sheet_name
        counter = 1
        while sheet_name in used_sheet_names:
            suffix = f"_{counter}"
            sheet_name = f"{base_name[:31-len(suffix)]}{suffix}"
            counter += 1
        used_sheet_names.add(sheet_name)
        
        ws_op = wb.create_sheet(sheet_name)
        
        # Load operator data from CSV
        op_data = _get_op_data_from_csv(op_name, csv_path, output_csv)
        
        if op_data:
            # Write header
            headers = list(op_data[0].keys())
            for col_idx, header in enumerate(headers, start=1):
                cell = ws_op.cell(row=1, column=col_idx, value=header)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = header_alignment
            
            # Write data
            for row_idx, data_row in enumerate(op_data, start=2):
                for col_idx, header in enumerate(headers, start=1):
                    value = data_row.get(header, "")
                    ws_op.cell(row=row_idx, column=col_idx, value=value)
            
            # Adjust column widths
            for col_idx, header in enumerate(headers, start=1):
                max_length = len(header)
                for data_row in op_data:
                    value = str(data_row.get(header, ""))
                    if len(value) > max_length:
                        max_length = len(value)
                ws_op.column_dimensions[get_column_letter(col_idx)].width = min(max_length + 2, 50)
        else:
            # No data found in CSV (e.g., SKIPPED ops), create a sheet with Failed status
            headers = ["Op_Name", "shape", "datatype", "Current Batch GPU",
                       "Base Batch GPU", "ACC verify", "Performance verify"]
            for col_idx, header in enumerate(headers, start=1):
                cell = ws_op.cell(row=1, column=col_idx, value=header)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = header_alignment
            
            # Write a single row with Failed status
            data_row = [op_name, "", "", "", "", "Failed", ""]
            for col_idx, value in enumerate(data_row, start=1):
                ws_op.cell(row=2, column=col_idx, value=value)
            
            # Adjust column widths
            for col_idx, header in enumerate(headers, start=1):
                ws_op.column_dimensions[get_column_letter(col_idx)].width = min(len(header) + 2, 50)
    
    # Save workbook
    try:
        # Ensure all sheet names are valid before saving
        for ws in wb.worksheets:
            # Validate sheet name
            name = ws.title
            if len(name) > 31:
                ws.title = name[:31]
            if not name or name.isspace():
                ws.title = "Sheet"
        
        wb.save(filepath)
        print(f"[INFO] Excel report generated: {filepath}")
    except Exception as e:
        print(f"[WARN] Failed to save Excel report: {e}")


def _get_op_data_from_csv(op_name, csv_path, output_csv=None):
    """Extract data for a specific operator from CSV file.
    
    In compare mode,优先从 output_csv 读取对比数据，如果不存在则从 csv_path 读取基准数据。
    """
    # In compare mode, try to read from output_csv first (contains comparison data)
    if output_csv and os.path.exists(output_csv):
        try:
            with open(output_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                op_data = [row for row in reader if row.get('Op_Name') == op_name]
            if op_data:
                return op_data
        except Exception as e:
            print(f"[WARN] Failed to read output CSV for {op_name}: {e}")
    
    # Fallback to baseline CSV
    if not os.path.exists(csv_path):
        return []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            op_data = [row for row in reader if row.get('Op') == op_name]
        return op_data
    except Exception as e:
        print(f"[WARN] Failed to read CSV for {op_name}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Batch driver for mcoplib operator benchmarks"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--generate", action="store_true",
        help="Only fill missing CSV rows (skip existing) [default]"
    )
    mode_group.add_argument(
        "--compare", action="store_true",
        help="Compare each op vs CSV baseline; fail if >5%% slower"
    )
    mode_group.add_argument(
        "--update", action="store_true",
        help="Overwrite CSV rows for all ops"
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="CSV path")
    parser.add_argument("--output", default=None, help="Output Excel file path (.xlsx) for the final report (only in --compare mode). Supports absolute and relative paths.")
    parser.add_argument("--ops", default=None, help="Comma-separated op subset")
    parser.add_argument("--dry-run", action="store_true", help="List ops without running")
    args = parser.parse_args()

    if args.compare:
        mode = "compare"
    elif args.update:
        mode = "update"
    else:
        mode = "generate"

    # Requirement: the --output report path must end in '.xlsx'. Validate up
    # front (before running any op) so a bad path fails fast rather than after
    # a 1-2 hour benchmark run.
    if args.output is not None:
        if not args.output.lower().endswith(".xlsx"):
            print(f"[ERROR] --output must end with '.xlsx' (got: {args.output})")
            return 1
        if mode != "compare":
            print("[ERROR] --output is only valid in --compare mode.")
            return 1

    print("=" * 80)
    print(f"testall.py | mode={mode} | csv={args.csv}")
    if mode == "generate":
        print("[INFO] --generate: only fills missing CSV rows; ops already in CSV are SKIPPED.")
        print("       Use --update to overwrite existing rows, or --compare to regression-test.")
    elif mode == "update":
        print("[INFO] --update: overwrites CSV rows for all ops (verification must pass first).")
    elif mode == "compare":
        print("[INFO] --compare: compares each op vs CSV baseline; fails if >5% slower. No CSV writes.")
    print("=" * 80)

    ops = get_supported_operators()
    if args.ops:
        wanted = {s.strip() for s in args.ops.split(",") if s.strip()}
        ops = [o for o in ops if o in wanted]
    if not ops:
        print("[ERROR] No operators to run.")
        return 1

    # Load the ignored-operators list from the single-op script source.
    ignored_set = _load_ignored_operators()
    if ignored_set:
        # Split ops into active vs ignored, but preserve order: ignored ops
        # appear in the summary table as SKIPPED so the user sees them.
        ignored_in_list = [o for o in ops if o in ignored_set]
        ops = [o for o in ops if o not in ignored_set]
        if ignored_in_list:
            print(f"[INFO] {len(ignored_in_list)} operator(s) ignored: "
                  f"{', '.join(ignored_in_list)}")
        print(f"[INFO] {len(ops)} operators to run in --{mode} mode.\n")
    else:
        print(f"[INFO] {len(ops)} operators to run in --{mode} mode.\n")

    if args.dry_run:
        for o in ops:
            print(f"  * {o}")
        return 0

    # Initialize temporary output CSV for compare mode
    temp_output_csv = None
    
    if mode in ("compare", "update"):
        ensure_csv_exists(args.csv)
        # In compare mode, always create a temporary CSV for collecting comparison data
        if mode == "compare":
            temp_output_csv = os.path.join("/tmp", f"temp_compare_{int(time.time())}.csv")
            print(f"[INFO] Temporary compare CSV: {temp_output_csv}")
            # If --output is specified, ensure its directory exists
            if args.output:
                output_dir = os.path.dirname(os.path.abspath(args.output))
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
    elif mode == "generate":
        csv_dir = os.path.dirname(os.path.abspath(args.csv))
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)

    rows = []
    success = failed = skipped = 0
    failing_ops = []
    skipped_ops = []
    total_start = time.time()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as logf:
        logf.write(f"=== testall.py | mode={mode} | csv={args.csv} ===\n")
        logf.write(f"=== {len(ops)} operators ===\n\n")
        logf.flush()

        for idx, op in enumerate(ops, 1):
            print(f"[{idx}/{len(ops)}] {op} ... ", end="", flush=True)
            csv_before = _op_in_csv(op, args.csv)

            # --generate fast path: if the op is already in the CSV, skip the
            # expensive subprocess + nvbench run entirely. The single-op script
            # would also SKIP, but only after running the full benchmark.
            if mode == "generate" and csv_before:
                status, detail, info, rc, elapsed = (
                    "SKIPPED", "exists (use --update to refresh)", None, 0, 0.0
                )
            else:
                # Always use temp_output_csv for collecting comparison data
                output_csv = temp_output_csv if mode == "compare" else None
                info, rc, elapsed = run_one_op(op, mode, args.csv, output_csv)
                if info is None:
                    info = {"verify_pass": None, "acc_verify": None,
                            "perf_verify": None, "csv_action": None,
                            "fatal": False, "error_msg": "timeout"}
                    rc = 124
                status, detail = classify_result(
                    op, mode, info, rc, args.csv, csv_before
                )
            if status == "SUCCESS":
                success += 1
            elif status == "FAILED":
                failed += 1
                failing_ops.append(op)
            elif status == "SKIPPED":
                skipped += 1
                skipped_ops.append(op)
            print(f"{status} ({detail}) [{elapsed:.1f}s]")

            logf.write(f"[{op}] -> {status} | {detail} | rc={rc} | {elapsed:.2f}s\n")
            if info:
                logf.write(
                    f"  verify_pass={info['verify_pass']} "
                    f"acc={info['acc_verify']} perf={info['perf_verify']} "
                    f"action={info['csv_action']} fatal={info['fatal']}\n"
                )
            logf.flush()

            rows.append({
                "op": op,
                "acc": info.get("acc_verify") if info else None,
                "perf": info.get("perf_verify") if info else None,
                "status": status,
                "detail": detail,
                "elapsed": elapsed,
            })

        total_elapsed = time.time() - total_start
        logf.write(
            f"\n=== SUMMARY | mode={mode} | "
            f"success={success} failed={failed} skipped={skipped} | "
            f"total={len(ops)} | {total_elapsed:.1f}s ===\n"
        )
        if failing_ops:
            logf.write(f"FAILED OPS: {', '.join(failing_ops)}\n")

    print("\n" + "=" * 80)
    print(f"Per-op results (mode={mode}):")
    print("=" * 80)
    print_summary_table(rows)

    print("\n" + "=" * 80)
    print(
        f"SUMMARY | mode={mode} | total={len(ops)} "
        f"success={success} failed={failed} skipped={skipped} "
        f"| {total_elapsed:.1f}s"
    )
    if failing_ops:
        print(f"FAILED OPS ({failed}):")
        for o in failing_ops:
            print(f"  - {o}")
    else:
        print("No failures.")
    print(f"\nDetailed log: {OUTPUT_FILE}")
    print("=" * 80)
    
    # Generate Excel report for compare mode
    if mode == "compare":
        print("\\n[INFO] Generating Excel report (.xlsx file)...")
        generate_excel_report(
            mode=mode,
            csv_path=args.csv,
            output_csv=temp_output_csv,
            total_ops=len(ops),
            success=success,
            failed=failed,
            skipped=skipped,
            total_elapsed=total_elapsed,
            failing_ops=failing_ops,
            skipped_ops=skipped_ops,
            rows=rows,
            output_path=args.output
        )
        # Clean up temporary output CSV if it was created
        if temp_output_csv and os.path.exists(temp_output_csv):
            try:
                os.remove(temp_output_csv)
                print(f"[INFO] Cleaned up temporary CSV: {temp_output_csv}")
            except Exception as e:
                print(f"[WARN] Failed to remove temporary CSV: {e}")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())