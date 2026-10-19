import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.op_source_inventory import build_inventory


class OpSourceInventoryTest(unittest.TestCase):
    def test_inventory_counts_group_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "op" / "vllm").mkdir(parents=True)
            (root / "op" / "vllm" / "kernel.cu").write_text("", encoding="utf-8")
            (root / "op" / "vllm" / "README.md").write_text("", encoding="utf-8")
            (root / "op" / "native_kernel.cu").write_text("", encoding="utf-8")

            inventory = build_inventory(root)

        self.assertEqual(inventory["groups"]["vllm"]["count"], 1)
        self.assertEqual(inventory["groups"]["vllm"]["files"], ["op/vllm/kernel.cu"])
        self.assertEqual(inventory["groups"]["native"]["count"], 1)
        self.assertEqual(inventory["groups"]["native"]["files"], ["op/native_kernel.cu"])


if __name__ == "__main__":
    unittest.main()
