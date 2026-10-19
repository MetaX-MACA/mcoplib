import unittest

from tools.plan_test_shards import plan_shards


class PlanTestShardsTest(unittest.TestCase):
    def test_distributes_tests_deterministically(self):
        shards = plan_shards(["a.py", "b.py", "c.py", "d.py", "e.py"], 2)

        self.assertEqual(shards, [["a.py", "c.py", "e.py"], ["b.py", "d.py"]])


if __name__ == "__main__":
    unittest.main()
