import os
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_plan import build_plan, read_maca_version


class BuildPlanTest(unittest.TestCase):
    def test_read_maca_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "Version.txt").write_text("Version:3.5.3.20\n", encoding="utf-8")
            self.assertEqual(read_maca_version(tmp), "3.5.3.20")

            Path(tmp, "Version.txt").write_text("", encoding="utf-8")
            self.assertIsNone(read_maca_version(tmp))

            Path(tmp, "Version.txt").write_text("Version3.5.3.20\n", encoding="utf-8")
            self.assertEqual(read_maca_version(tmp), "Version3.5.3.20")

    def test_build_plan_uses_default_submodule_flags(self):
        root = Path.cwd()
        old = os.environ.pop("BUILD_DEFAULT_OP_SUBMODULE", None)
        try:
            plan = build_plan(root)
        finally:
            if old is not None:
                os.environ["BUILD_DEFAULT_OP_SUBMODULE"] = old

        self.assertEqual(plan["submodules"]["BUILD_DEFAULT_OP_SUBMODULE"], "ON")
        self.assertEqual(plan["submodules"]["BUILD_VLLM_SUBMODULE"], "ON")
        self.assertEqual(plan["submodules"]["BUILD_SGLANG_SUBMODULE"], "ON")


if __name__ == "__main__":
    unittest.main()
