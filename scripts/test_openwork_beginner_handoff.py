"""Model-free automatic handoff checks. Uses injected replies, never a real profile."""

import platform
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@unittest.skipUnless(platform.system() == "Darwin", "macOS handoff")
class OpenWorkBeginnerHandoffTests(unittest.TestCase):
    def test_verified_task_binding_and_failure_boundaries(self):
        with tempfile.TemporaryDirectory(prefix="stark-openwork-handoff-") as directory:
            executable = Path(directory) / "check"
            built = subprocess.run(
                ["xcrun", "swiftc", "-parse-as-library",
                 str(ROOT / "apps/menu-bar/Sources/OpenWorkBeginnerHandoff.swift"),
                 str(ROOT / "apps/menu-bar/Tests/OpenWorkBeginnerHandoffTests.swift"),
                 "-o", str(executable)],
                capture_output=True, timeout=120, check=False,
            )
            self.assertEqual(built.returncode, 0, "handoff compilation failed")
            result = subprocess.run([str(executable)], capture_output=True, timeout=30, check=False)
            self.assertEqual(result.returncode, 0, "handoff invariant failed")


if __name__ == "__main__":
    unittest.main()
