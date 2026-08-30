#!/usr/bin/env python3
"""Run an mcoplib benchmark command and capture reproducible logs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def build_command(root: Path, op: str, extra_args: list[str]) -> list[str]:
    return [
        sys.executable,
        str(root / "benchmark" / "mcoplib_mxbenchmark_ops.py"),
        "--op",
        op,
        *extra_args,
    ]


def run_with_logs(root: Path, op: str, log_root: Path, extra_args: list[str]) -> int:
    if not op or not all(ch.isalnum() or ch == "_" for ch in op):
        print(
            f"Error: Invalid operator name '{op}'. Only alphanumeric characters and underscores are allowed.",
            file=sys.stderr,
        )
        return 1

    benchmark_dir = root / "benchmark"
    script_path = benchmark_dir / "mcoplib_mxbenchmark_ops.py"
    if not script_path.is_file():
        print(f"Error: Benchmark script not found at '{script_path}'", file=sys.stderr)
        return 1

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{op}"
    run_dir = log_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(root, op, extra_args)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "op": op,
        "command": command,
        "environment": {
            "MACA_PATH": os.environ.get("MACA_PATH"),
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "MACA_VISIBLE_DEVICES": os.environ.get("MACA_VISIBLE_DEVICES"),
        },
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (run_dir / "stdout.log").open("w", encoding="utf-8") as out, (
        run_dir / "stderr.log"
    ).open("w", encoding="utf-8") as err:
        proc = subprocess.run(command, cwd=benchmark_dir, stdout=out, stderr=err, text=True)

    (run_dir / "exit_code.txt").write_text(str(proc.returncode) + "\n", encoding="utf-8")
    print(f"Benchmark logs written to: {run_dir}")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run mcoplib benchmark with structured logs.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--op", required=True)
    parser.add_argument("--log-root", type=Path, default=Path("benchmark_logs"))
    args, extra = parser.parse_known_args()

    return run_with_logs(args.root.resolve(), args.op, args.log_root.resolve(), extra)


if __name__ == "__main__":
    raise SystemExit(main())
