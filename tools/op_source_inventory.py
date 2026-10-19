#!/usr/bin/env python3
"""Build a JSON inventory of mcoplib operator source groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GROUPS = {
    "vllm": "op/vllm",
    "sglang": "op/sglang",
    "lmdeploy": "op/lmdeploy",
    "cv": "op/cv",
    "native": "op",
}

SOURCE_SUFFIXES = {".cu", ".cuh", ".cpp", ".cc", ".h", ".hpp", ".py"}


def _sources(root: Path, relative_dir: str) -> list[str]:
    base = root / relative_dir
    if not base.exists():
        return []
    return sorted(
        path.relative_to(root).as_posix()
        for path in base.rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    )


def build_inventory(root: Path) -> dict[str, object]:
    raw_files: dict[str, list[str]] = {}
    for name, relative_dir in GROUPS.items():
        raw_files[name] = _sources(root, relative_dir)

    other_files: set[str] = set()
    for name in GROUPS:
        if name != "native":
            other_files.update(raw_files[name])
    if "native" in raw_files:
        raw_files["native"] = [path for path in raw_files["native"] if path not in other_files]

    groups: dict[str, object] = {}
    for name, relative_dir in GROUPS.items():
        files = raw_files[name]
        groups[name] = {"root": relative_dir, "count": len(files), "files": files}
    return {"root": str(root), "groups": groups}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create mcoplib operator source inventory.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = build_inventory(args.root.resolve())
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
