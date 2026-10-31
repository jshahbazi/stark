"""Native credential checks use injected storage and a disposable Keychain service.

Run with uv run --project tools/converter --locked --python 3.13.13
python scripts/test_api_credentials.py. No production credential is accessed.
"""

from __future__ import annotations

import platform
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@unittest.skipUnless(platform.system() == "Darwin", "macOS Keychain required")
class APICredentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="stark-credential-tests-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.executable = Path(cls.temporary.name) / "credential-checks"
        compiled = subprocess.run(
            [
                "xcrun",
                "swiftc",
                "-parse-as-library",
                str(ROOT / "apps/menu-bar/Sources/APICredentialStore.swift"),
                str(ROOT / "apps/menu-bar/Tests/APICredentialStoreTests.swift"),
                "-o",
                str(cls.executable),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
            check=False,
        )
        if compiled.returncode:
            raise RuntimeError("credential_test_compilation_failed")

    def run_check(self, *arguments: str):
        result = subprocess.run(
            [str(self.executable), *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        if result.returncode == 77:
            self.skipTest("isolated_keychain_unavailable")
        self.assertEqual(result.returncode, 0, "credential_check_failed")

    def test_persistence_failures_creation_race_and_port_preferences(self):
        self.run_check()

    def test_real_isolated_keychain_cross_process_reuse_and_regeneration(self):
        self.run_check("isolated-keychain")


if __name__ == "__main__":
    unittest.main()
