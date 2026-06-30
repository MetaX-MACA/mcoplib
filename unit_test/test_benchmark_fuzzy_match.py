import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

from mcoplib_mxbenchmark_ops import find_literal_matches


class BenchmarkFuzzyMatchTest(unittest.TestCase):
    def test_matches_literal_special_characters(self):
        candidates = ["op+a.json", "opxa.json", "other.json"]
        self.assertEqual(find_literal_matches("op+a", candidates), ["op+a.json"])

    def test_unbalanced_regex_character_is_safe(self):
        candidates = ["scaled_mm[fp8].json", "scaled_mm_fp8.json"]
        self.assertEqual(find_literal_matches("mm[fp8]", candidates), ["scaled_mm[fp8].json"])


if __name__ == "__main__":
    unittest.main()
