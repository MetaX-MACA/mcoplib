#!/usr/bin/env python3
"""Fail a validation run when JSON metrics cross configured thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("results", data.get("records", []))
    if not isinstance(data, list):
        raise ValueError("metrics JSON must be a list or contain a results list")
    return [item for item in data if isinstance(item, dict)]


def evaluate(path: Path, metric: str, minimum: float | None, maximum: float | None) -> dict[str, Any]:
    failures = []
    for index, record in enumerate(_records(path)):
        if metric not in record:
            continue
        value = float(record[metric])
        if minimum is not None and value < minimum:
            failures.append({"index": index, "metric": metric, "value": value, "reason": "below_minimum"})
        if maximum is not None and value > maximum:
            failures.append({"index": index, "metric": metric, "value": value, "reason": "above_maximum"})
    return {"metric": metric, "failure_count": len(failures), "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--min", type=float, dest="minimum")
    parser.add_argument("--max", type=float, dest="maximum")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = evaluate(args.metrics, args.metric, args.minimum, args.maximum)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 1 if report["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
