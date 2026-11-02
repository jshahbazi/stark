"""Native coordinator/broker checks use injected approval storage and private sockets.

Run through the locked converter environment. No production Keychain item or
OpenWork or OpenCode profile is read or modified; the socket namespaces are disposable.
"""

from __future__ import annotations

import platform
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = (
    "approval",
    "approval-errors",
    "approval-ui",
    "controller-ack",
    "controller-refresh",
    "controller-session",
    "controller-revoke",
    "controller-store-failure",
    "controller-revoke-timeout",
    "controller-send-failure",
    "controller-shutdown",
    "broker-auth",
    "broker-paths",
    "broker-framing",
    "broker-deadline",
    "broker-write",
    "broker-cleanup",
    "opencode-approval",
    "opencode-pending",
    "opencode-deny",
    "opencode-disabled",
    "opencode-revoke",
    "opencode-prune",
    "opencode-engine-path",
    "opencode-config-directory",
    "launch-environment",
    "coordinator",
)


@unittest.skipUnless(
    platform.system() == "Darwin", "macOS native desktop connection checks"
)
class DesktopConnectionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="stark-desktop-connection-tests-"
        )
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.executable = Path(cls.temporary.name) / "desktop-connection-checks"
        found = subprocess.run(
            ["/usr/bin/xcrun", "--find", "swiftc"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        compiler = Path(found.stdout.decode().strip())
        if found.returncode or not compiler.is_absolute() or not compiler.is_file():
            raise RuntimeError("desktop_connection_test_compiler_unavailable")
        sdk_result = subprocess.run(
            ["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
        )
        sdk = Path(sdk_result.stdout.decode().strip())
        if sdk_result.returncode or not sdk.is_absolute() or not sdk.is_dir():
            raise RuntimeError("desktop_connection_test_sdk_unavailable")
        sources = ROOT / "apps/menu-bar/Sources"
        compiled = subprocess.run(
            [
                str(compiler),
                "-parse-as-library",
                "-sdk",
                str(sdk),
                str(sources / "APICredentialStore.swift"),
                str(sources / "DesktopConnectionApproval.swift"),
                str(sources / "DesktopClientPeer.swift"),
                str(sources / "DesktopConnectionBroker.swift"),
                str(sources / "OpenWorkConnectorInstaller.swift"),
                str(sources / "OpenWorkConnectionController.swift"),
                str(sources / "OpenCodeConnectionController.swift"),
                str(sources / "DesktopConnections.swift"),
                str(ROOT / "apps/menu-bar/Tests/DesktopConnectionsTests.swift"),
                "-o",
                str(cls.executable),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
            check=False,
        )
        if compiled.returncode:
            raise RuntimeError("desktop_connection_test_compilation_failed")

    def test_private_approval_coordinator_and_broker_scenarios(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                result = subprocess.run(
                    [str(self.executable), scenario],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20,
                    check=False,
                )
                self.assertEqual(
                    result.returncode, 0, "desktop_connection_check_failed"
                )


if __name__ == "__main__":
    unittest.main()
