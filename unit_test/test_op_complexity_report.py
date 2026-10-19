import tempfile
import unittest
from pathlib import Path

from tools.op_complexity_report import build_report


class OpComplexityReportTest(unittest.TestCase):
    def test_counts_operator_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            op = root / "op"
            op.mkdir()
            (op / "kernel.cu").write_text("template <typename T>\nvoid f(){ k<<<1,1>>>(); }\n", encoding="utf-8")

            report = build_report(root)

        self.assertEqual(report["file_count"], 1)
        self.assertEqual(report["top_by_lines"][0]["kernel_launches"], 1)

    def test_returns_empty_report_when_op_directory_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = build_report(Path(tmpdir))

        self.assertEqual(report, {"file_count": 0, "total_lines": 0, "top_by_lines": []})


if __name__ == "__main__":
    unittest.main()
