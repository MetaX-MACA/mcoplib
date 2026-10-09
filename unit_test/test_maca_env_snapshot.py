import subprocess
from unittest.mock import patch
import unittest

from tools.maca_env_snapshot import _command_version, snapshot


class MacaEnvSnapshotTest(unittest.TestCase):
    def test_snapshot_has_expected_sections(self):
        report = snapshot()

        self.assertIn("python", report)
        self.assertIn("environment", report)
        self.assertIn("tools", report)

    @patch("tools.maca_env_snapshot.shutil.which", return_value="/usr/bin/mxcc")
    @patch(
        "tools.maca_env_snapshot.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["mxcc", "--version"], timeout=5),
    )
    def test_command_version_handles_timeout(self, _run, _which):
        report = _command_version("mxcc", ["--version"])

        self.assertTrue(report["available"])
        self.assertEqual(report["returncode"], -1)
        self.assertIn("Error:", report["version"][0])


if __name__ == "__main__":
    unittest.main()
