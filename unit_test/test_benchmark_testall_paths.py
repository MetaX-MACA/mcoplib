import os
from pathlib import Path

from benchmark import testall


def test_testall_paths_are_script_relative():
    script_dir = Path(testall.__file__).resolve().parent

    for path in (
        testall.CONFIG_DIR,
        testall.RUNNERS_DIR,
        testall.TARGET_SCRIPT,
        testall.OUTPUT_FILE,
        testall.STATISTICS_DIR,
        testall.CSV_SAVE_PATH,
    ):
        assert os.path.isabs(path)
        resolved = Path(path).resolve()
        assert resolved == script_dir or script_dir in resolved.parents
