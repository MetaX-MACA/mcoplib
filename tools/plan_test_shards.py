#!/usr/bin/env python3
"""Create deterministic shards for mcoplib unit tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def discover_tests(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in (root / "unit_test").glob("test_*.py"))


def plan_shards(tests: list[str], shard_count: int) -> list[list[str]]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    shards = [[] for _ in range(shard_count)]
    for index, test in enumerate(tests):
        shards[index % shard_count].append(test)
    return shards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--shards", type=int, default=4, help="number of shards to create")
    parser.add_argument("--output", type=Path, help="write JSON plan to this path")
    args = parser.parse_args()

    tests = discover_tests(args.root)
    shards = plan_shards(tests, args.shards)
    payload = {
        "test_count": len(tests),
        "shard_count": args.shards,
        "shards": [{"index": index, "tests": shard} for index, shard in enumerate(shards)],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
