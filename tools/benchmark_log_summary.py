#!/usr/bin/env python3
"""Parse benchmark or distributed-test logs into a compact JSON summary."""

from __future__ import annotations


import argparse
import json
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


METRIC_RE = re.compile(
    r"(?P<name>[A-Za-z0-9_./-]+).*?(?P<latency>[0-9.]+)\s*(?:us|ms)",
    re.IGNORECASE,
)
ERROR_RE = re.compile(
    r"(error|failed|timeout|traceback|segmentation fault|core dumped)",
    re.IGNORECASE,
)


def parse_log(path: Path) -> dict[str, object]:
    metrics: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, start=1):
            match = METRIC_RE.search(line)
            if match:
                item = {"line": lineno, "text": line.strip()}
                item.update(
                    {k: v for k, v in match.groupdict().items() if v is not None}
                )
                metrics.append(item)
            if ERROR_RE.search(line):
                errors.append({"line": lineno, "text": line.strip()})
    return {
        "path": str(path),
        "metrics": metrics,
        "errors": errors,
        "metric_count": len(metrics),
        "error_count": len(errors),
    }


def self_test() -> None:
    with TemporaryDirectory() as tmp_dir:
        sample = Path(tmp_dir) / "sample_log_for_parser.txt"
        sample.write_text(
            "all_reduce size 1024 algbw 11.5 busbw 10.1 latency 2.5 ms "
            "throughput 33.0 tokens/s 12.0 GB/s\nERROR timeout\n",
            encoding="utf-8",
        )
        data = parse_log(sample)
    if data["metric_count"] < 1 or data["error_count"] != 1:
        raise RuntimeError(f"self-test failed: {data}")
    print(
        json.dumps(
            {"ok": True, "metric_count": data["metric_count"]}, ensure_ascii=False
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", help="Log files to parse.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    results = []
    for p in args.logs:
        path = Path(p)
        if not path.is_file():
            print(f"Error: log file does not exist or is not a file: {p}", file=sys.stderr)
            return 2
        results.append(parse_log(path))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
