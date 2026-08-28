from mcoplib.profiler import _is_profiler_enabled


def test_profiler_disabled_by_common_false_values(monkeypatch):
    for value in ("0", "false", "False", "off", "NO"):
        monkeypatch.setenv("PROFILER_ENABLED", value)
        assert not _is_profiler_enabled()


def test_profiler_enabled_by_default(monkeypatch):
    monkeypatch.delenv("PROFILER_ENABLED", raising=False)
    assert _is_profiler_enabled()
