"""Native public-config installer checks; only disposable files are used."""

from __future__ import annotations

import platform
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = (
    "create",
    "comments",
    "empty-array-comment",
    "malformed",
    "duplicate-keys",
    "bounds",
    "symlink-file",
    "symlink-directory",
    "symlink-ancestor",
    "hardlink-file",
    "permissions",
    "read-only",
    "connector-safety",
    "conflict",
    "duplicate-owned",
    "remove-ranges",
    "concurrent-edit",
    "concurrent-create",
    "concurrent-symlink",
    "concurrent-directory",
    "abort",
    "installer-lock",
    "invalid-socket",
    "port-reconnect",
    "shared-client-entry",
    "namespace-reconnect",
    "app-location-migration",
    "combined-migration",
    "migration-conflicts",
    "remove-stale",
    "invalid-base-url",
)


@unittest.skipUnless(platform.system() == "Darwin", "macOS native filesystem checks")
class OpenWorkConnectorInstallerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="stark-connector-installer-tests-"
        )
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.executable = Path(cls.temporary.name) / "connector-checks"
        result = subprocess.run(
            [
                "xcrun",
                "swiftc",
                "-parse-as-library",
                str(ROOT / "apps/menu-bar/Sources/OpenWorkConnectorInstaller.swift"),
                str(ROOT / "apps/menu-bar/Tests/OpenWorkConnectorInstallerTests.swift"),
                "-o",
                str(cls.executable),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
            check=False,
        )
        if result.returncode:
            raise RuntimeError("connector_installer_test_compilation_failed")

    def test_public_configuration_transaction_scenarios(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                result = subprocess.run(
                    [str(self.executable), scenario],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                    check=False,
                )
                self.assertEqual(
                    result.returncode, 0, "connector_installer_check_failed"
                )


if __name__ == "__main__":
    unittest.main()
