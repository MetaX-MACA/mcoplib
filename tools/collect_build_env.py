#!/usr/bin/env python3
"""Collect build-time diagnostics for mcoplib MACA environments."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "path": None}

    try:
        proc = subprocess.run(
            [executable, *command[1:]],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - defensive diagnostics
        return {"available": True, "path": executable, "error": str(exc)}

    return {
        "available": True,
        "path": executable,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _python_package(name: str) -> dict[str, Any]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python 3.9+ is required
        return {"installed": False, "error": "importlib.metadata unavailable"}

    try:
        return {"installed": True, "version": version(name)}
    except PackageNotFoundError:
        return {"installed": False}


def _maca_version(maca_path: str | None) -> str | None:
    if not maca_path:
        return None

    version_file = Path(maca_path) / "Version.txt"
    if not version_file.is_file():
        return None

    try:
        first_line = version_file.read_text(encoding="utf-8").splitlines()[0]
    except Exception:
        return None

    return first_line.split(":")[-1].strip()


def collect_env() -> dict[str, Any]:
    maca_path = os.environ.get("MACA_PATH")
    cuda_home = os.environ.get("CUDA_HOME")

    report: dict[str, Any] = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "environment": {
            "MACA_PATH": maca_path,
            "CUDA_HOME": cuda_home,
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "PATH": os.environ.get("PATH"),
        },
        "maca": {
            "version": _maca_version(maca_path),
            "version_file": str(Path(maca_path) / "Version.txt") if maca_path else None,
            "exists": bool(maca_path and Path(maca_path).exists()),
        },
        "tools": {
            "cmake_maca": _run(["cmake_maca", "--version"]),
            "make_maca": _run(["make_maca", "--version"]),
            "mxcc": _run(["mxcc", "--version"]),
            "ninja": _run(["ninja", "--version"]),
            "gcc": _run(["gcc", "--version"]),
            "g++": _run(["g++", "--version"]),
        },
        "python_packages": {
            "torch": _python_package("torch"),
            "pybind11": _python_package("pybind11"),
            "packaging": _python_package("packaging"),
            "ninja": _python_package("ninja"),
            "setuptools": _python_package("setuptools"),
        },
    }

    try:
        import torch

        report["torch_runtime"] = {
            "cuda": getattr(torch.version, "cuda", None),
            "hip": getattr(torch.version, "hip", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except Exception as exc:
        report["torch_runtime"] = {"error": str(exc)}

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect mcoplib build environment diagnostics.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    args = parser.parse_args()

    report = collect_env()
    text = json.dumps(report, indent=2 if args.pretty else None, sort_keys=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
