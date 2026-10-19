#!/usr/bin/env python3
"""Validate mcoplib mxbenchmark config/runner pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RUNNER_PREFIX = "mcoplib_mxbenchmark_"
RUNNER_SUFFIX = "_runners.py"


def collect_errors(root: Path) -> list[str]:
    benchmark_dir = root / "benchmark"
    config_dir = benchmark_dir / "config"
    runners_dir = benchmark_dir / "runners"
    errors: list[str] = []

    if not config_dir.is_dir():
        return [f"missing config directory: {config_dir}"]
    if not runners_dir.is_dir():
        return [f"missing runners directory: {runners_dir}"]

    configs = {path.stem: path for path in config_dir.glob("*.json")}
    runners = {
        path.name[len(RUNNER_PREFIX) : -len(RUNNER_SUFFIX)]: path
        for path in runners_dir.glob(f"{RUNNER_PREFIX}*{RUNNER_SUFFIX}")
    }

    for name in sorted(set(configs) - set(runners)):
        errors.append(f"missing runner for config {configs[name].relative_to(root).as_posix()}")
    for name in sorted(set(runners) - set(configs)):
        errors.append(f"missing config for runner {runners[name].relative_to(root).as_posix()}")

    for name, path in sorted(configs.items()):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"invalid JSON or unreadable file in {path.relative_to(root).as_posix()}: {exc}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Lint mxbenchmark config and runner pairs.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    root = args.root.resolve()
    errors = collect_errors(root)
    if errors:
        print("mxbenchmark config lint failed:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    print("mxbenchmark config lint passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
