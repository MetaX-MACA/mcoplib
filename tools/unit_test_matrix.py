#!/usr/bin/env python3
"""Generate a reproducible validation matrix for MACA GPU testing."""

from __future__ import annotations


import argparse
import itertools
import json


DEFAULT_TESTS = ["unit_test/test_rms_norm.py", "unit_test/test_fused_moe_gate.py"]
DIMENSIONS = {"dtype": ["fp16", "bf16"], "scope": ["smoke", "full"]}


def build_matrix(tests: list[str]) -> list[dict[str, str]]:
    keys = list(DIMENSIONS)
    rows = []
    for test in tests:
        for values in itertools.product(*(DIMENSIONS[k] for k in keys)):
            row = {"test": test}
            row.update(dict(zip(keys, values)))
            rows.append(row)
    return rows


def self_test() -> None:
    rows = build_matrix(DEFAULT_TESTS[:1])
    if not rows or "test" not in rows[0]:
        raise RuntimeError(f"self-test failed: {rows}")
    print(json.dumps({"ok": True, "rows": len(rows)}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="append", default=[], help="Override or extend test names.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    tests = args.test or DEFAULT_TESTS
    print(json.dumps(build_matrix(tests), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
