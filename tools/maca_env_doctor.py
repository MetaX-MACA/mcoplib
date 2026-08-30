#!/usr/bin/env python3
"""Collect a lightweight MACA runtime environment report."""

from __future__ import annotations


import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


DEFAULT_PATHS = ["/opt/maca", "/opt/maca-3.1.0"]


def command_output(cmd: list[str]) -> dict[str, object]:
    executable = shutil.which(cmd[0])
    if executable is None:
        return {
            "command": cmd,
            "available": False,
            "stdout": "",
            "stderr": "",
            "returncode": None,
        }
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
    except OSError as exc:
        return {
            "command": cmd,
            "available": True,
            "stdout": "",
            "stderr": str(exc),
            "returncode": None,
        }
    return {
        "command": cmd,
        "available": True,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "returncode": proc.returncode,
    }


def collect(extra_paths: list[str]) -> dict[str, object]:
    search_paths = [p for p in [*DEFAULT_PATHS, *extra_paths] if p]
    existing = [str(Path(p).resolve()) for p in search_paths if Path(p).exists()]
    env_names = ["MACA_HOME", "MATRIX_HOME", "LD_LIBRARY_PATH", "PATH", "PYTHONPATH"]
    return {
        "environment": {name: os.environ.get(name, "") for name in env_names},
        "existing_paths": existing,
        "tools": {
            "mx-smi": command_output(["mx-smi"]),
            "mxcc": command_output(["mxcc", "--version"]),
            "python": command_output(["python3", "--version"]),
        },
    }


def self_test() -> None:
    data = collect([])
    if "tools" not in data or "environment" not in data:
        raise RuntimeError(f"self-test failed: {data}")
    print(json.dumps({"ok": True, "tool_count": len(data["tools"])}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", action="append", default=[], help="Additional path to include in the report.")
    parser.add_argument("--self-test", action="store_true", help="Run a no-GPU sanity check.")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    print(json.dumps(collect(args.path), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
