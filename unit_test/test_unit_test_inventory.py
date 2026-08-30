import tempfile
import unittest
from pathlib import Path

from tools.unit_test_inventory import inventory


class UnitTestInventoryTest(unittest.TestCase):
    def test_counts_test_functions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            unit = root / "unit_test"
            unit.mkdir()
            (unit / "test_demo.py").write_text(
                "def test_a(): pass\n"
                "async def test_async_case(): pass\n"
                "def helper(): pass\n",
                encoding="utf-8",
            )

            report = inventory(root)

        self.assertEqual(report["file_count"], 1)
        self.assertEqual(report["test_function_count"], 2)


if __name__ == "__main__":
    unittest.main()
