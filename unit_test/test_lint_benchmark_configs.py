import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.lint_benchmark_configs import collect_errors


class LintBenchmarkConfigsTest(unittest.TestCase):
    def test_detects_missing_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "benchmark" / "config").mkdir(parents=True)
            (root / "benchmark" / "runners").mkdir(parents=True)
            (root / "benchmark" / "config" / "foo.json").write_text("{}", encoding="utf-8")

            errors = collect_errors(root)

        self.assertEqual(errors, ["missing runner for config benchmark/config/foo.json"])

    def test_valid_pair_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "benchmark" / "config").mkdir(parents=True)
            (root / "benchmark" / "runners").mkdir(parents=True)
            (root / "benchmark" / "config" / "foo.json").write_text(
                json.dumps({"samples": 1}), encoding="utf-8"
            )
            (root / "benchmark" / "runners" / "mcoplib_mxbenchmark_foo_runners.py").write_text(
                "", encoding="utf-8"
            )

            errors = collect_errors(root)

        self.assertEqual(errors, [])

    def test_reports_unreadable_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "benchmark" / "config").mkdir(parents=True)
            (root / "benchmark" / "runners").mkdir(parents=True)
            (root / "benchmark" / "config" / "foo.json").write_bytes(b"\xff\xfe")
            (root / "benchmark" / "runners" / "mcoplib_mxbenchmark_foo_runners.py").write_text(
                "", encoding="utf-8"
            )

            errors = collect_errors(root)

        self.assertEqual(len(errors), 1)
        self.assertIn("invalid JSON or unreadable file", errors[0])


if __name__ == "__main__":
    unittest.main()
