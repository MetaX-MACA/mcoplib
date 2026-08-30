#!/usr/bin/env python3
"""Export mcoplib unit test metadata as JSON."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def _test_functions(path: Path) -> tuple[list[str], str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as exc:
        return [], f"{exc.__class__.__name__}: {exc}"
    tests = sorted(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    )
    return tests, ""


def inventory(root: Path) -> dict[str, object]:
    tests = []
    for path in sorted((root / "unit_test").glob("test_*.py")):
        funcs, parse_error = _test_functions(path)
        tests.append(
            {
                "path": path.relative_to(root).as_posix(),
                "test_functions": funcs,
                "count": len(funcs),
                "parse_error": parse_error,
            }
        )
    return {
        "file_count": len(tests),
        "test_function_count": sum(item["count"] for item in tests),
        "parse_error_count": sum(1 for item in tests if item["parse_error"]),
        "tests": tests,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = json.dumps(inventory(args.root), indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
