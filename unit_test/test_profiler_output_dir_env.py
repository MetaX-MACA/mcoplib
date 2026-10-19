import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcoplib.profiler import _resolve_output_dir


class ProfilerOutputDirEnvTest(unittest.TestCase):
    def test_uses_default_without_env(self):
        old = os.environ.pop("PROFILER_OUTPUT_DIR", None)
        try:
            self.assertEqual(_resolve_output_dir("./profiles"), "./profiles")
        finally:
            if old is not None:
                os.environ["PROFILER_OUTPUT_DIR"] = old

    def test_env_overrides_default(self):
        old = os.environ.get("PROFILER_OUTPUT_DIR")
        os.environ["PROFILER_OUTPUT_DIR"] = "/tmp/mcoplib-profiles"
        try:
            self.assertEqual(_resolve_output_dir("./profiles"), "/tmp/mcoplib-profiles")
        finally:
            if old is None:
                os.environ.pop("PROFILER_OUTPUT_DIR", None)
            else:
                os.environ["PROFILER_OUTPUT_DIR"] = old


if __name__ == "__main__":
    unittest.main()
