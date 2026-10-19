import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcoplib.profiler import _trace_file_path


class ProfilerTracePathTest(unittest.TestCase):
    @patch("mcoplib.profiler._timestamp")
    def test_trace_file_path_is_unique(self, mock_timestamp):
        mock_timestamp.side_effect = ["20260610T120000000001", "20260610T120000000002"]
        with tempfile.TemporaryDirectory() as tmp_path:
            first = _trace_file_path(tmp_path, "fused_mla", 0)
            second = _trace_file_path(tmp_path, "fused_mla", 0)

        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith(".json"))
        self.assertEqual(os.path.dirname(first), tmp_path)

    def test_trace_file_path_sanitizes_function_name(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            path = _trace_file_path(tmp_path, "op/name with spaces", 1)

        filename = os.path.basename(path)
        self.assertTrue(filename.startswith("op_name_with_spaces_trace_rank_1_"))
        self.assertNotIn("/", filename)
        self.assertNotIn(" ", filename)

    def test_trace_file_path_truncates_long_function_name(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            path = _trace_file_path(tmp_path, "x" * 300, 0)

        filename = os.path.basename(path)
        prefix = filename.split("_trace_rank_", 1)[0]
        self.assertEqual(len(prefix), 128)


if __name__ == "__main__":
    unittest.main()
