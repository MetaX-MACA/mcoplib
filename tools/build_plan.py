#!/usr/bin/env python3
"""Print a non-invasive mcoplib build plan as JSON."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


SUBMODULE_FLAGS = [
    "BUILD_VLLM_SUBMODULE",
    "BUILD_SGLANG_SUBMODULE",
    "BUILD_LMDEPLOY_SUBMODULE",
    "BUILD_DEFAULT_OP_SUBMODULE",
    "BUILD_MOE_SUBMODULE",
    "BUILD_SGL_KERNEL_SUBMODULE",
    "BUILD_SGL_GROUPED_GEMM_SUBMODULE",
    "BUILD_SGL_MOE_W4A16_SUBMODULE",
]


def read_maca_version(maca_path: str | None) -> str | None:
    if not maca_path:
        return None
    version_file = Path(maca_path) / "Version.txt"
    if not version_file.is_file():
        return None
    lines = version_file.read_text(encoding="utf-8").splitlines()
    return lines[0].split(":")[-1].strip() if lines else None


def build_plan(root: Path) -> dict[str, object]:
    maca_path = os.environ.get("MACA_PATH")
    return {
        "root": str(root),
        "maca_path": maca_path,
        "maca_version": read_maca_version(maca_path),
        "tools": {
            "cmake_maca": shutil.which("cmake_maca"),
            "make_maca": shutil.which("make_maca"),
            "mxcc": shutil.which("mxcc"),
            "ninja": shutil.which("ninja"),
        },
        "submodules": {flag: os.environ.get(flag, "ON") for flag in SUBMODULE_FLAGS},
        "cache": {
            "FETCHCONTENT_BASE_DIR": os.environ.get(
                "FETCHCONTENT_BASE_DIR", str(root / ".deps")
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Print mcoplib build plan JSON.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = json.dumps(build_plan(args.root.resolve()), indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
