#!/usr/bin/env python3
"""Lint launch scripts for common MACA deployment footguns."""

from __future__ import annotations


import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory


GLOBS = ["*.sh", "**/*.sh"]
REQUIRED_TOKENS = ["MACA_HOME", "mxcc"]
FAIL_FAST_RE = re.compile(r"^\s*set\s+(?:-[A-Za-z]*e[A-Za-z]*|-o\s+errexit)\b", re.M)


def lint(root: Path) -> dict[str, object]:
    if not root.is_dir():
        raise NotADirectoryError(f"lint root is not a directory: {root}")

    findings: list[dict[str, str]] = []
    scripts = {
        path
        for pattern in GLOBS
        for path in root.glob(pattern)
        if path.is_file()
    }
    for path in sorted(scripts):
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(root).as_posix()
        if "\r\n" in text:
            findings.append(
                {
                    "path": rel,
                    "severity": "warning",
                    "message": "script uses CRLF line endings",
                }
            )
        if not FAIL_FAST_RE.search(text) and path.suffix in (".sh", ""):
            findings.append(
                {
                    "path": rel,
                    "severity": "warning",
                    "message": "shell script does not enable fail-fast mode",
                }
            )
        for token in REQUIRED_TOKENS:
            if token not in text:
                findings.append(
                    {
                        "path": rel,
                        "severity": "info",
                        "message": f"missing optional token: {token}",
                    }
                )
    return {"finding_count": len(findings), "findings": findings}


def self_test() -> None:
    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        script = root / "build.sh"
        script.write_text("#!/usr/bin/env bash\nset -ex\nmxcc --version\n", encoding="utf-8")
        data = lint(root)
    if "findings" not in data:
        raise RuntimeError(f"self-test failed: {data}")
    print(json.dumps({"ok": True, "finding_count": data["finding_count"]}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    root = Path(args.root)
    if not root.is_dir():
        parser.error(f"--root is not a directory: {root}")
    print(json.dumps(lint(root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
