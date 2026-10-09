#!/usr/bin/env python3
"""Collect a reproducible MACA validation environment snapshot."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _command_version(command: str, args: list[str]) -> dict[str, object]:
    path = shutil.which(command)
    if not path:
        return {"path": "", "available": False, "version": ""}
    try:
        completed = subprocess.run([path, *args], text=True, capture_output=True, timeout=5)
        version = (completed.stdout or completed.stderr).strip().splitlines()[:5]
        return {
            "path": path,
            "available": True,
            "returncode": completed.returncode,
            "version": version,
        }
    except (subprocess.SubprocessError, OSError) as exc:
        return {
            "path": path,
            "available": True,
            "returncode": -1,
            "version": [f"Error: {exc}"],
        }


def snapshot() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "environment": {
            "MACA_PATH": os.environ.get("MACA_PATH", ""),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        },
        "tools": {
            "mxcc": _command_version("mxcc", ["--version"]),
            "cmake_maca": _command_version("cmake_maca", ["--version"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = json.dumps(snapshot(), indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
