import json
import tempfile
import unittest
from pathlib import Path

from tools.metric_threshold_gate import evaluate


class MetricThresholdGateTest(unittest.TestCase):
    def test_reports_values_below_minimum(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            path.write_text(json.dumps([{"latency_ms": 1.0}, {"latency_ms": 5.0}]), encoding="utf-8")

            report = evaluate(path, "latency_ms", minimum=2.0, maximum=None)

        self.assertEqual(report["failure_count"], 1)
        self.assertEqual(report["failures"][0]["reason"], "below_minimum")


if __name__ == "__main__":
    unittest.main()
