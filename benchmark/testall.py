"""
testall.py — Batch driver for mcoplib operator benchmarks.

Usage:
  python testall.py                      # default: pure benchmark mode (see below)
  python testall.py --generate           # only fill missing CSV rows (skip existing)
  python testall.py --compare            # compare each op vs CSV baseline; fail if >5%% slower
  python testall.py --update             # overwrite CSV rows for all ops
  python testall.py --csv PATH           # override CSV path (default: statistics/mcoplib_ops_performance_C500.csv)
  python testall.py --output PATH        # specify output Excel file path (.xlsx) for compare report (supports absolute and relative paths)
  python testall.py --ops a,b,c          # only run named ops

Default (no mode flag): pure benchmark mode. Every op in scope is benchmarked
for real (equivalent to `python mcoplib_mxbenchmark_ops.py --op <name>` per
op) with NO generate/update/compare actions — the CSV is neither read for
baselines nor written. Per-result stats tables are printed and, with
--report, aggregated into the 7-column Excel report.
  python testall.py --infer_framework FW # only run ops of one framework: op | vllm | sglang
                                         #   vllm   -> op/vllm kernels (mcoplib._C / mcoplib._moe_C,
                                         #              torch.ops._C / torch.ops._moe_C)
                                         #   sglang -> op/sglang kernels (mcoplib.sgl_kernel,
                                         #              torch.ops.sgl_kernel)
                                         #   op     -> mcoplib's own ops (mcoplib.op module plus
                                         #              the package's triton_/tilelang_ python kernels)
                                         # unset  -> run all ops
  python testall.py --report PATH        # aggregate per-result performance stats into one Excel
                                         # table (op_name, dtype, shape, acc_pass, Gpu Time,
                                         # GlobalMem BW, BWUtil; one row per benchmark result).
                                         # Works in every mode. Default: benchmark dir,
                                         # mirroring the CSV name.
  python testall.py --dry-run            # list ops without running

The script:
  1. Calls `python mcoplib_mxbenchmark_ops.py --list` to enumerate supported ops.
  2. For each op, runs `python mcoplib_mxbenchmark_ops.py --op <name> [--<mode>] --csv <csv>`
     (no mode flag in the default pure benchmark mode).
  3. Parses stdout to extract accuracy status and performance status.
  4. Prints a per-op summary table and a final roll-up (success/failure counts + failing op names).
  5. In --compare mode, generates an Excel report (.xlsx) with summary and per-operator sheets.
  6. Always aggregates every benchmark result into a 7-column Excel table (see --report).
  7. Writes the overall summary (total/success/failed/skipped counts, elapsed,
     failed/skipped op lists, overall result) to testall_output.json (replaces
     the old testall_output.txt per-op log).
"""

import os
import re
import sys
import csv
import ast
import time
import json
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
# Overall summary JSON written at the end of every run (replaces the old
# testall_output.txt per-op log).
SUMMARY_JSON_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testall_output.json")
SUMMARY_HTML_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testall_output.html")

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
AVG_PERF_RE = re.compile(r"Average Performance Ratio:\s*([^\s,]+)", re.IGNORECASE)

# Threshold for performance regression in --compare mode (must match the
# single-op script's own 5%% rule).
PERF_REGRESSION_THRESHOLD_PCT = 5.0

# ---------------------------------------------------------------------------
# Framework classification (--infer_framework)
#
# mcoplib exposes its kernels through several extension modules:
#   - mcoplib._C / mcoplib._moe_C, built from op/vllm/ (the _C extension also
#     registers the torch.ops._C_cache_ops namespace)   -> "vllm" framework ops
#   - mcoplib.sgl_kernel, built from op/sglang/         -> "sglang" framework ops
#   - everything else: the mcoplib.op module (built from op/ direct sources)
#     plus the package's own triton_/tilelang_ python kernels -> "op"
# An op is classified by the invocation pattern found in its runner file.
# ---------------------------------------------------------------------------
FRAMEWORK_CHOICES = ("op", "vllm", "sglang")
FRAMEWORK_PATTERNS = {
    "vllm": re.compile(
        r"torch\.ops\._C\.|torch\.ops\._C_cache_ops\.|torch\.ops\._moe_C\."
        r"|mcoplib\._C\b|mcoplib\._moe_C\b"
    ),
    "sglang": re.compile(r"torch\.ops\.sgl_kernel\.|mcoplib\.sgl_kernel\b"),
}
RUNNERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runners")


def detect_op_framework(op_name):
    """Classify an op as 'vllm', 'sglang' or 'op' from its runner file.

    Scans runners/mcoplib_mxbenchmark_<op>_runners.py for the invocation
    pattern of the kernel under test. Returns None when the runner file
    cannot be read (such an op matches no framework filter).
    """
    runner_path = os.path.join(
        RUNNERS_DIR, f"mcoplib_mxbenchmark_{op_name}_runners.py"
    )
    try:
        with open(runner_path, "r", encoding="utf-8") as f:
            src = f.read()
    except OSError:
        return None
    if FRAMEWORK_PATTERNS["vllm"].search(src):
        return "vllm"
    if FRAMEWORK_PATTERNS["sglang"].search(src):
        return "sglang"
    return "op"


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
        m = AVG_PERF_RE.search(line)
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


# ---------------------------------------------------------------------------
# Parsing of the per-result pipe tables the single-op script prints at the end
# of every run ("# Benchmark Results" markdown section). Used both to echo the
# per-op stats into this run's output and to feed the --report table in
# --compare mode (where the current run's full rows are not persisted anywhere
# else: the subprocess writes them to a temp CSV it deletes afterwards).
# ---------------------------------------------------------------------------
BENCH_SECTION_TITLE = "# Benchmark Results"


def extract_benchmark_section(stdout):
    """Return the '# Benchmark Results' markdown section of a single-op run's
    stdout (the per-result pipe tables), or '' when absent."""
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == BENCH_SECTION_TITLE:
            section = "\n".join(lines[i:]).rstrip()
            # Drop a trailing markdown code fence, if present.
            if section.endswith("```"):
                section = section[: section.rfind("```")].rstrip()
            return section
    return ""


def _is_separator(line):
    """True if line is a pipe-table separator (|-----|-----|...)."""
    s = line.strip()
    return s.startswith("|") and "-" in s and all(c in "|-: " for c in s)


def _cells(line):
    """Split a pipe-table line into stripped cells."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _rejoin_done_split(head_line, cont_line):
    """Rejoin a table line split by the child's unflushed "Done." print.

    When the single-op script's stdout is a pipe, its exit flush of the
    unflushed warmup progress print ("Done.") can land mid-line, splitting a
    table line into "<line prefix> Done." + "<line suffix>". Strip the marker
    and glue the halves back together. Returns None when cont_line is not a
    plausible continuation (no pipe-table fragment, or it carries its own
    marker), leaving the caller's original line untouched.
    """
    cont = cont_line.strip()
    if "|" not in cont:
        return None
    joined = head_line.strip()[: -len("Done.")].rstrip() + cont
    if "Done." in joined:  # continuation itself carries the marker
        return None
    return joined


def parse_benchmark_tables(stdout, op_name):
    """Parse the per-result pipe tables from a single-op run's stdout.

    Returns report-row dicts (op_name/dtype/shape/acc_pass/Gpu Time/
    GlobalMem BW/BWUtil), one per benchmark result row. Column positions
    are taken from each table's own header line, so extra axis columns
    (Bits, NE_N, ...) don't break the parse. Some ops define neither a
    dtype axis nor a bandwidth metric, so their tables have no dtype /
    GlobalMem BW / BWUtil columns at all — those fields default to ""
    (matching how such ops appear in the results CSV).

    Lines split by the "Done." flush (see _rejoin_done_split) are recovered:
    a split header or data row is rejoined with its continuation half and
    accepted only when the rejoined column count matches the table's own
    structure (the separator line for headers, the header for rows), so a
    well-formed line is never altered.
    """
    rows = []
    lines = stdout.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        # A results-table header: pipe table carrying the per-result columns.
        # (Acc_Pass + Shape is the stable signature across table variants.)
        if (stripped.startswith("|") and "Acc_Pass" in stripped
                and "Shape" not in stripped and stripped.endswith("Done.")
                and i + 2 < len(lines) and _is_separator(lines[i + 2])):
            # Header split before its Shape column: the signature test fails
            # even though this is a header; rejoin and re-test before moving on.
            cand = _rejoin_done_split(stripped, lines[i + 1])
            if cand and "Shape" in cand \
                    and len(_cells(cand)) == len(_cells(lines[i + 2])):
                stripped = cand
                i += 1  # consume the continuation half of the split header
        if not (stripped.startswith("|") and "Acc_Pass" in stripped
                and "Shape" in stripped):
            i += 1
            continue
        header = _cells(stripped)
        # Header split after its Shape column still passes the signature test
        # but keeps a wrong column count; the separator line below the
        # continuation validates the rejoined count.
        if (stripped.endswith("Done.") and i + 2 < len(lines)
                and _is_separator(lines[i + 2])
                and len(header) != len(_cells(lines[i + 2]))):
            joined = _rejoin_done_split(stripped, lines[i + 1])
            if joined is not None \
                    and len(_cells(joined)) == len(_cells(lines[i + 2])):
                header = _cells(joined)
                i += 1  # consume the continuation half of the split header
        i += 1
        # Skip the header separator line (|-----|-----|...).
        if i < len(lines) and _is_separator(lines[i]):
            i += 1
        while i < len(lines):
            s = lines[i].strip()
            if not s.startswith("|"):
                # A standalone "Done." line (injection landed exactly at a
                # line boundary): skip it and keep reading this table's rows.
                if s == "Done." and i + 1 < len(lines):
                    nxt = lines[i + 1].strip()
                    if (nxt.startswith("|") and "Acc_Pass" not in nxt
                            and not _is_separator(nxt)):
                        i += 1
                        continue
                break
            raw = s
            cells = [c.strip() for c in raw.strip("|").split("|")]
            # Data row split by the "Done." flush ("<row prefix> Done." +
            # "<row suffix>"): rejoin and accept only on an exact column
            # count match, so a well-formed row is never altered. ("Done."
            # can never occur in a real row cell here.) If the rejoin is not
            # possible, the following line is the orphaned suffix fragment:
            # consume it so the rest of this table's rows are still read.
            if raw.endswith("Done.") and i + 1 < len(lines):
                joined = _rejoin_done_split(raw, lines[i + 1])
                if joined is not None and len(_cells(joined)) == len(header):
                    cells = _cells(joined)
                    i += 1  # consume the continuation half of the split row
                elif ("|" in lines[i + 1]
                        and not lines[i + 1].strip().startswith("|")):
                    i += 1  # skip the orphaned continuation fragment
            if len(cells) == len(header):
                d = dict(zip(header, cells))
                rows.append({
                    "op_name": d.get("Op") or op_name,
                    "dtype": d.get("dtype", ""),
                    "shape": d.get("Shape", ""),
                    "acc_pass": d.get("Acc_Pass", ""),
                    # The printed column is "GPU Time" (e.g. '29.339 us');
                    # the report column is spelled "Gpu Time" (REPORT_COLUMNS).
                    "Gpu Time": d.get("GPU Time", ""),
                    "GlobalMem BW": d.get("GlobalMem BW", ""),
                    "BWUtil": d.get("BWUtil", ""),
                })
            i += 1
    return rows


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

    # bench mode (no-flag default): pure benchmark run, no CSV actions.
    # Success = result tables were printed and no result row failed its
    # accuracy check. (Acc_Pass lives in the per-result tables; empty cells
    # carry no verdict and don't fail the op.)
    if mode == "bench":
        stdout = info.get("stdout") or ""
        bench_rows = parse_benchmark_tables(stdout, op_name)
        if not bench_rows:
            return "SKIPPED", "no benchmark results"
        n_fail = sum(1 for r in bench_rows
                     if r["acc_pass"].strip().lower() not in ("yes", "pass", ""))
        if n_fail:
            return "FAILED", f"acc fail ({n_fail}/{len(bench_rows)} results)"
        return "SUCCESS", f"acc=Pass ({len(bench_rows)} results)"

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
        "--csv", csv_path,
    ]
    # Pure benchmark mode ("bench", the no-flag default): invoke the single-op
    # script WITHOUT any mode flag so it only benchmarks the op and prints the
    # per-result tables — no generate/update/compare CSV actions. (--csv is
    # inert without a mode flag; it only silences the script's default-path
    # warning.)
    if mode != "bench":
        cmd.append(f"--{mode}")
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
    info = parse_run_output(stdout)
    # Keep the raw stdout so the caller can echo the per-result benchmark
    # tables and (in --compare/--bench modes) feed the --report aggregation.
    info["stdout"] = stdout
    return info, rc, elapsed


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


# ---------------------------------------------------------------------------
# Aggregated performance report (--report)
#
# One Excel row per benchmark result (an op with multiple shape/dtype
# combinations gets multiple rows), with exactly these columns:
#   op_name | dtype | shape | acc_pass | Gpu Time | GlobalMem BW | BWUtil
# Data source depends on the mode:
#   - generate/update: the results CSV (freshly appended rows plus previously
#     recorded ones for the ops in scope)
#   - compare/bench: the per-result tables parsed from each op's run stdout
#     (in compare mode the current run's rows are otherwise discarded by the
#     single-op script; in bench mode no CSV is written at all)
# ---------------------------------------------------------------------------
REPORT_COLUMNS = ["op_name", "dtype", "shape", "acc_pass", "Gpu Time",
                  "GlobalMem BW", "BWUtil"]
DEFAULT_REPORT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    CSV_FILENAME.replace(".csv", ".xlsx"),
)


def _fmt_bw_bytes(val):
    """Format a bytes/sec number string like the benchmark tables do
    (e.g. 1446396330269.95 -> '1.446 TB/s')."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val or "")
    for unit, factor in (("TB/s", 1e12), ("GB/s", 1e9), ("MB/s", 1e6), ("KB/s", 1e3)):
        if v >= factor:
            return f"{v / factor:.3f} {unit}"
    return f"{v:.1f} B/s"


def _fmt_bwutil(val):
    """Format a BWUtil ratio like the benchmark tables do (0.7847 -> '78.47%')."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val or "")
    return f"{v * 100:.2f}%"


def _fmt_time_sec(val):
    """Format a seconds number string like the benchmark tables' GPU Time
    cells do (e.g. 2.9339e-05 -> '29.339 us'). Empty/invalid stays as-is."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val or "")
    for unit, factor in (("s", 1.0), ("ms", 1e-3)):
        if v >= factor:
            return f"{v / factor:.3f} {unit}"
    return f"{v * 1e6:.3f} us"


def collect_report_rows_from_csv(op_scope, csv_path):
    """Collect report rows for ops in op_scope from a benchmark results CSV."""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("op_name") not in op_scope:
                    continue
                rows.append({
                    "op_name": r.get("op_name", ""),
                    "dtype": r.get("dtype", ""),
                    "shape": r.get("Shape", ""),
                    "acc_pass": r.get("Acc_Pass", ""),
                    "Gpu Time": _fmt_time_sec(
                        r.get("GPU Time (sec)") or r.get("GPU Time", "")
                    ),
                    "GlobalMem BW": _fmt_bw_bytes(r.get("GlobalMem BW (bytes/sec)", "")),
                    "BWUtil": _fmt_bwutil(r.get("BWUtil", "")),
                })
    except Exception as e:
        print(f"[WARN] Failed to read '{csv_path}' for --report: {e}")
    return rows


def generate_report_excel(report_rows, output_path, mode, infer_framework):
    """Write the aggregated 7-column performance table to an Excel file."""
    if not EXCEL_AVAILABLE:
        print("[WARN] openpyxl not available, skipping --report Excel generation")
        return
    filepath = os.path.abspath(output_path)
    out_dir = os.path.dirname(filepath)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Performance"

    header_font = Font(name='Arial', size=12, bold=True)
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_alignment = Alignment(horizontal='center', vertical='center')

    for col_idx, name in enumerate(REPORT_COLUMNS, start=1):
        cell = ws.cell(row=1, column=col_idx, value=name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment

    for row_idx, r in enumerate(report_rows, start=2):
        for col_idx, name in enumerate(REPORT_COLUMNS, start=1):
            ws.cell(row=row_idx, column=col_idx, value=r.get(name, ""))

    for col_idx, name in enumerate(REPORT_COLUMNS, start=1):
        width = len(name)
        for r in report_rows:
            width = max(width, len(str(r.get(name, ""))))
        ws.column_dimensions[get_column_letter(col_idx)].width = min(width + 2, 60)

    try:
        wb.save(filepath)
        print(f"[INFO] Performance report ({len(report_rows)} result rows) "
              f"saved to: {filepath}")
    except Exception as e:
        print(f"[WARN] Failed to save performance report: {e}")


def write_summary_json(path, total, success, failed, skipped,
                       total_elapsed, failing_ops, skipped_ops):
    """Write the run's overall summary as a JSON file.

    Content is the whole-run roll-up only (no per-op details): total/success/
    failed/skipped counts, elapsed time (string with unit, e.g. "21.32s"),
    the failed and skipped op lists, and
    the top-level "result" verdict. "result" is "success" only when every op
    in scope succeeded; any FAILED or SKIPPED op makes the whole run "failed".
    """
    summary = {
        "result": "success" if (failed == 0 and skipped == 0) else "failed",
        "total": total,
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "elapsed_time": f"{total_elapsed:.2f}s",
        "failed_ops": list(failing_ops),
        "skipped_ops": list(skipped_ops),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

def write_summary_html(path, mode, total, success, failed, skipped,
                       total_elapsed, failing_ops, skipped_ops, rows):
    """Generate a standalone HTML report for testall."""

    from html import escape

    result = "SUCCESS" if (failed == 0 and skipped == 0) else "FAILED"
    generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    status_rows = []

    for row in rows:
        status = str(row.get("status") or "-")
        status_class = status.lower()

        status_rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('op') or ''))}</td>"
            f"<td>{escape(str(row.get('acc') or '-'))}</td>"
            f"<td>{escape(str(row.get('perf') or '-'))}</td>"
            f'<td><span class="status {escape(status_class)}">'
            f"{escape(status)}</span></td>"
            f"<td>{escape(str(row.get('detail') or ''))}</td>"
            f"<td>{float(row.get('elapsed') or 0):.1f}s</td>"
            "</tr>"
        )

    if not status_rows:
        status_rows.append(
            '<tr><td colspan="6" class="empty">No operator results</td></tr>'
        )

    failed_html = "".join(
        f"<li>{escape(str(op))}</li>" for op in failing_ops
    ) or "<li>None</li>"

    skipped_html = "".join(
        f"<li>{escape(str(op))}</li>" for op in skipped_ops
    ) or "<li>None</li>"

    result_class = "success" if result == "SUCCESS" else "failed"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Testall Benchmark Report</title>

<style>
* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    padding: 30px;
    background: #f4f6fa;
    color: #263238;
    font-family: Arial, Helvetica, sans-serif;
}}

.container {{
    max-width: 1400px;
    margin: 0 auto;
}}

h1 {{
    margin-bottom: 8px;
    color: #1f2937;
}}

.subtitle {{
    color: #6b7280;
    margin-bottom: 25px;
}}

.summary {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 16px;
    margin-bottom: 25px;
}}

.card {{
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}}

.card .label {{
    color: #6b7280;
    font-size: 13px;
}}

.card .value {{
    margin-top: 10px;
    font-size: 27px;
    font-weight: bold;
}}

.success {{
    color: #16803c;
}}

.failed {{
    color: #dc2626;
}}

.skipped {{
    color: #d97706;
}}

.result {{
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 20px;
}}

.section {{
    background: white;
    padding: 22px;
    margin-bottom: 22px;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}}

.section h2 {{
    margin-top: 0;
    font-size: 19px;
}}

.table-wrapper {{
    overflow-x: auto;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}

th {{
    background: #e9eef6;
    color: #374151;
    text-align: left;
    padding: 12px;
    white-space: nowrap;
}}

td {{
    padding: 11px 12px;
    border-bottom: 1px solid #e5e7eb;
}}

tr:hover {{
    background: #f9fafb;
}}

.status {{
    font-weight: bold;
}}

.empty {{
    text-align: center;
    color: #9ca3af;
    padding: 25px;
}}

ul {{
    line-height: 1.9;
    overflow-wrap: anywhere;
}}

.footer {{
    text-align: center;
    color: #9ca3af;
    font-size: 12px;
    margin-top: 25px;
}}

@media (max-width: 600px) {{
    body {{
        padding: 12px;
    }}
}}
</style>
</head>

<body>
<div class="container">

<h1>Testall Benchmark Report</h1>

<div class="subtitle">
    Mode: {escape(mode.upper())}
    &nbsp;|&nbsp; Generated: {generated_time}
    &nbsp;|&nbsp; Total Time: {total_elapsed:.2f}s
</div>

<div class="result {result_class}">
    Overall Result: {result}
</div>

<div class="summary">

    <div class="card">
        <div class="label">Total Operators</div>
        <div class="value">{total}</div>
    </div>

    <div class="card">
        <div class="label">Success</div>
        <div class="value success">{success}</div>
    </div>

    <div class="card">
        <div class="label">Failed</div>
        <div class="value failed">{failed}</div>
    </div>

    <div class="card">
        <div class="label">Skipped</div>
        <div class="value skipped">{skipped}</div>
    </div>

    <div class="card">
        <div class="label">Elapsed Time</div>
        <div class="value">{total_elapsed:.2f}s</div>
    </div>

</div>

<div class="section">
    <h2>Operator Results</h2>

    <div class="table-wrapper">
    <table>
        <thead>
            <tr>
                <th>Operator</th>
                <th>Acc</th>
                <th>Perf</th>
                <th>Status</th>
                <th>Detail</th>
                <th>Time</th>
            </tr>
        </thead>
        <tbody>
            {''.join(status_rows)}
        </tbody>
    </table>
    </div>
</div>

<div class="section">
    <h2>Failed Operators ({failed})</h2>
    <ul>{failed_html}</ul>
</div>

<div class="section">
    <h2>Skipped Operators ({skipped})</h2>
    <ul>{skipped_html}</ul>
</div>

<div class="footer">
    Generated by testall.py
</div>

</div>
</body>
</html>
"""

    try:
        output_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(output_dir, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"[INFO] HTML report generated: {os.path.abspath(path)}")

    except Exception as e:
        print(f"[WARN] Failed to generate HTML report: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch driver for mcoplib operator benchmarks"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--generate", action="store_true",
        help="Only fill missing CSV rows (skip existing)"
    )
    mode_group.add_argument(
        "--compare", action="store_true",
        help="Compare each op vs CSV baseline; fail if >5%% slower"
    )
    mode_group.add_argument(
        "--update", action="store_true",
        help="Overwrite CSV rows for all ops"
    )
    # No mode flag = pure benchmark mode ("bench"): every op in scope is
    # benchmarked for real with no CSV generate/update/compare actions.
    parser.add_argument("--csv", default=DEFAULT_CSV, help="CSV path")
    parser.add_argument("--output", default=None, help="Output Excel file path (.xlsx) for the final report (only in --compare mode). Supports absolute and relative paths.")
    parser.add_argument(
        "--infer_framework", choices=list(FRAMEWORK_CHOICES), default=None,
        help="Only run ops whose kernels belong to the given framework: "
             "'vllm' (op/vllm kernels, invoked via mcoplib._C / mcoplib._moe_C "
             "and torch.ops._C / torch.ops._moe_C), 'sglang' (op/sglang kernels, "
             "invoked via mcoplib.sgl_kernel and torch.ops.sgl_kernel), 'op' "
             "(mcoplib's own ops: the mcoplib.op module plus the package's "
             "triton_/tilelang_ python kernels). Unset: run all ops."
    )
    parser.add_argument(
        "--report", default=None,
        help=("Aggregate every benchmark result into one Excel table with columns "
              "op_name, dtype, shape, acc_pass, Gpu Time, GlobalMem BW, BWUtil "
              "(one row per result; ops with multiple results get multiple rows). "
              f"Works in every mode. Default: {DEFAULT_REPORT}")
    )
    parser.add_argument("--ops", default=None, help="Comma-separated op subset")
    parser.add_argument("--dry-run", action="store_true", help="List ops without running")
    args = parser.parse_args()

    if args.compare:
        mode = "compare"
    elif args.update:
        mode = "update"
    elif args.generate:
        mode = "generate"
    else:
        # No mode flag: pure benchmark mode. Every op in scope is benchmarked
        # for real with no generate/update/compare CSV actions; per-result
        # stats are printed and aggregated into the --report table.
        mode = "bench"

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

    # --report works in every mode; validate up front like --output so a bad
    # path fails fast rather than after a long benchmark run.
    if args.report is not None:
        if not args.report.lower().endswith(".xlsx"):
            print(f"[ERROR] --report must end with '.xlsx' (got: {args.report})")
            return 1
        if args.output and os.path.abspath(args.report) == os.path.abspath(args.output):
            print("[ERROR] --report and --output must not be the same file.")
            return 1

    print("=" * 80)
    print(f"testall.py | mode={mode} | csv={args.csv}"
          f" | infer_framework={args.infer_framework or 'all'}")
    if mode == "generate":
        print("[INFO] --generate: only fills missing CSV rows; ops already in CSV are SKIPPED.")
        print("       Use --update to overwrite existing rows, or --compare to regression-test.")
    elif mode == "update":
        print("[INFO] --update: overwrites CSV rows for all ops (verification must pass first).")
    elif mode == "compare":
        print("[INFO] --compare: compares each op vs CSV baseline; fails if >5% slower. No CSV writes.")
    elif mode == "bench":
        print("[INFO] Pure benchmark mode (no mode flag): every op is benchmarked for real")
        print("       (equivalent to `mcoplib_mxbenchmark_ops.py --op <name>` per op) with no")
        print("       generate/update/compare CSV actions. Use --generate/--update/--compare")
        print("       for CSV-managing modes.")
    print("=" * 80)

    ops = get_supported_operators()
    if args.ops:
        wanted = {s.strip() for s in args.ops.split(",") if s.strip()}
        ops = [o for o in ops if o in wanted]
    if args.infer_framework:
        total_before = len(ops)
        selected, unknown = [], []
        for o in ops:
            fw = detect_op_framework(o)
            if fw is None:
                unknown.append(o)
            elif fw == args.infer_framework:
                selected.append(o)
        if unknown:
            print(f"[WARN] Could not classify the framework of {len(unknown)} op(s) "
                  f"(runner file missing or unreadable); excluded: {', '.join(unknown)}")
        ops = selected
        print(f"[INFO] --infer_framework {args.infer_framework}: "
              f"{len(ops)} of {total_before} operator(s) selected.")
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
    report_rows_from_stdout = []  # --compare mode: per-result rows parsed from stdout
    total_start = time.time()

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

        # Echo the per-result benchmark tables (Acc_Pass | Cos_Dist | Op |
        # dtype | Shape | ... | GlobalMem BW | BWUtil | ...) printed by the
        # single-op run so every op's performance stats are visible in this
        # run's output.
        op_stdout = (info or {}).get("stdout") or ""
        if op_stdout:
            bench_section = extract_benchmark_section(op_stdout)
            if bench_section:
                print(bench_section)
        if mode in ("compare", "bench"):
            # The single-op script discards the current run's full result
            # rows (temp CSV deleted after comparison; in bench mode no
            # CSV is written at all), so collect them from the stdout
            # tables for the --report aggregation.
            report_rows_from_stdout.extend(parse_benchmark_tables(op_stdout, op))

        rows.append({
            "op": op,
            "acc": info.get("acc_verify") if info else None,
            "perf": info.get("perf_verify") if info else None,
            "status": status,
            "detail": detail,
            "elapsed": elapsed,
        })

    total_elapsed = time.time() - total_start

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
    print(f"\nSummary JSON: {SUMMARY_JSON_FILE}")
    print("=" * 80)

    # Overall summary JSON (replaces the old testall_output.txt log). Written
    # once the op run and its roll-up are complete — before the auxiliary
    # --report/--output Excel generation, so a failure there cannot lose the
    # test summary.
    write_summary_json(
        SUMMARY_JSON_FILE, len(ops), success, failed, skipped,
        total_elapsed, failing_ops, skipped_ops,
    )

    # Generate HTML report in compare mode.
    if mode == "compare":
        write_summary_html(
            SUMMARY_HTML_FILE,
            mode,
            len(ops),
            success,
            failed,
            skipped,
            total_elapsed,
            failing_ops,
            skipped_ops,
            rows,
        )

    # --- Aggregated performance report (--report) ---
    # One row per benchmark result; ops in scope with several shape/dtype
    # combinations contribute several rows.
    report_path = args.report or DEFAULT_REPORT
    if mode in ("compare", "bench"):
        report_rows = report_rows_from_stdout
    else:
        report_rows = collect_report_rows_from_csv(set(ops), args.csv)
    if not report_rows:
        print("[WARN] No benchmark result rows collected for the --report table.")
    generate_report_excel(report_rows, report_path, mode, args.infer_framework)

    # Generate Excel report for compare mode
    if mode == "compare":
        print("\n[INFO] Generating Excel report (.xlsx file)...")
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
    main()