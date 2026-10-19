#!/usr/bin/env python3
"""Summarize CUDA/MACA operator source complexity for validation planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SOURCE_SUFFIXES = {".cu", ".cuh", ".cpp", ".h"}


def analyze_file(path: Path, root: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "path": path.relative_to(root).as_posix(),
        "lines": len(text.splitlines()),
        "kernel_launches": text.count("<<<"),
        "templates": text.count("template"),
        "torch_bindings": text.count("PYBIND11_MODULE"),
    }


def build_report(root: Path) -> dict[str, object]:
    op_dir = root / "op"
    if not op_dir.is_dir():
        return {"file_count": 0, "total_lines": 0, "top_by_lines": []}

    files = [
        analyze_file(path, root)
        for path in sorted(op_dir.rglob("*"))
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    ]
    return {
        "file_count": len(files),
        "total_lines": sum(item["lines"] for item in files),
        "top_by_lines": sorted(files, key=lambda item: item["lines"], reverse=True)[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--output", type=Path, help="write JSON report to this path")
    args = parser.parse_args()

    text = json.dumps(build_report(args.root), indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
