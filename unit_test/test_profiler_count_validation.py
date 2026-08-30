import types
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcoplib.profiler import _normalize_profile_count, profiler


class ProfilerCountValidationTest(unittest.TestCase):
    def test_normalize_profile_count_accepts_numeric_strings(self):
        self.assertEqual(_normalize_profile_count("repeat", "3", 1), 3)

    def test_normalize_profile_count_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "warmup must be an integer"):
            _normalize_profile_count("warmup", "bad", 0)

    def test_normalize_profile_count_rejects_values_below_minimum(self):
        with self.assertRaisesRegex(ValueError, "repeat must be >= 1"):
            _normalize_profile_count("repeat", 0, 1)

    def test_profiler_validates_counts_when_decorator_is_created(self):
        with self.assertRaisesRegex(ValueError, "repeat must be >= 1"):
            profiler(repeat=0)

    def test_profiler_schedule_uses_zero_internal_warmup(self):
        schedule_kwargs = {}
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

        profiler_module = types.ModuleType("torch.profiler")

        def fake_schedule(**kwargs):
            schedule_kwargs.update(kwargs)
            return kwargs

        class FakeProfile:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def step(self):
                return None

        profiler_module.profile = lambda **kwargs: FakeProfile(**kwargs)
        profiler_module.schedule = fake_schedule
        profiler_module.ProfilerActivity = types.SimpleNamespace(CPU="cpu", CUDA="cuda")
        fake_torch.profiler = profiler_module

        previous_torch = sys.modules.get("torch")
        previous_profiler = sys.modules.get("torch.profiler")
        sys.modules["torch"] = fake_torch
        sys.modules["torch.profiler"] = profiler_module
        try:
            calls = []

            @profiler(warmup=2, repeat=3)
            def sample():
                calls.append("run")
                return "ok"

            self.assertEqual(sample(), "ok")
        finally:
            if previous_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous_torch
            if previous_profiler is None:
                sys.modules.pop("torch.profiler", None)
            else:
                sys.modules["torch.profiler"] = previous_profiler

        self.assertEqual(schedule_kwargs["wait"], 0)
        self.assertEqual(schedule_kwargs["warmup"], 0)
        self.assertEqual(schedule_kwargs["active"], 1)
        self.assertEqual(schedule_kwargs["repeat"], 3)
        self.assertEqual(len(calls), 5)


if __name__ == "__main__":
    unittest.main()
