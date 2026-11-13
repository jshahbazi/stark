"""Compile the real Pi controller; exercise only injected helpers and clipboard."""

from __future__ import annotations

import platform
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@unittest.skipUnless(platform.system() == "Darwin", "macOS Swift checks")
class PiConnectionTests(unittest.TestCase):
    def test_manual_setup_environment_state_and_explicit_launch(self):
        with tempfile.TemporaryDirectory(prefix="stark-pi-controller-") as temporary:
            executable = Path(temporary) / "pi-controller-checks"
            sources = ROOT / "apps/menu-bar"
            compiled = subprocess.run(
                ["xcrun", "swiftc", "-parse-as-library", "-target", "arm64-apple-macosx26.4",
                 str(sources / "Sources/PiConnectionController.swift"),
                 str(sources / "Sources/ConnectionClipboard.swift"),
                 str(sources / "Tests/PiConnectionTests.swift"), "-o", str(executable)],
                stdin=subprocess.DEVNULL, capture_output=True,
                timeout=120, check=False,
            )
            self.assertEqual(compiled.returncode, 0, "pi_controller_compilation_failed")
            checked = subprocess.run([str(executable)], stdin=subprocess.DEVNULL,
                                     capture_output=True, timeout=20, check=False)
            diagnostic = checked.stderr.decode("utf-8", errors="replace").strip()
            self.assertEqual(checked.returncode, 0, diagnostic if diagnostic.startswith("pi_connection_check_failed_") else "pi_controller_check_failed")


if __name__ == "__main__":
    unittest.main()
