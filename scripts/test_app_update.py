"""Model-free tests of published update selection and transactional app replacement."""

from __future__ import annotations

import importlib.util
import json
import os
import plistlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "update_app", ROOT / "scripts/update_app.py"
)
assert SPEC is not None and SPEC.loader is not None
updater = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(updater)
installer = updater.installer


def fixture(path: Path, version: str) -> Path:
    executable = path / "Contents/MacOS/Stark"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    (path / "Contents/Info.plist").write_bytes(
        plistlib.dumps(
            {
                "CFBundleIdentifier": installer.APP_ID,
                "CFBundleExecutable": "Stark",
                "CFBundleShortVersionString": version,
            }
        )
    )
    return path


@unittest.skipUnless(sys.platform == "darwin", "Darwin atomic exchange required")
class UpdateTransactionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="stark updater test ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.target = fixture(self.root / "Applications/Stark.app", "0.3.0")
        self.candidate = fixture(self.root / "download/Stark.app", "0.4.0")
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.old_digest = installer.validate(self.target)
        self.addCleanup(patch.stopall)
        patch.object(
            updater,
            "verify_release",
            side_effect=lambda app, _version: installer.validate(app),
        ).start()
        self.wait_for_exit = updater.wait_for_exit
        patch.object(updater, "wait_for_exit").start()
        self.command = patch.object(installer, "command", return_value=b"").start()

    def managed(self, schema=2, default=False):
        prefix = self.root / (".local" if default else "custom CLI")
        state = prefix / "lib/stark/managed-app.json"
        state.parent.mkdir(parents=True)
        record = {
            "schema_version": schema,
            "installation_id": "a" * 32,
            "path": str(self.target),
            "app_id": installer.APP_ID,
            "sha256": self.old_digest,
        }
        if schema == 2:
            record["state_path"] = str(state)
        owner = self.target.parent / installer.OWNER_NAME
        state.write_text(json.dumps(record))
        owner.write_bytes(state.read_bytes())
        return state, owner

    def apply(self):
        updater.apply_update(
            self.candidate, self.target, "0.4.0", os.getpid(), self.workspace
        )

    def test_unmanaged_update_keeps_candidate_and_replaces_complete_app(self):
        self.apply()
        self.assertEqual(
            installer.validate(self.target), installer.validate(self.candidate)
        )
        self.assertEqual(
            self.command.call_args.args[0], ["/usr/bin/open", "-n", self.target]
        )
        self.assertTrue((self.workspace / "ready").is_file())
        self.assertEqual(list(self.target.parent.glob(".stark-update-*")), [])

    def test_custom_prefix_ownership_is_updated_and_cli_uninstall_still_works(self):
        state, owner = self.managed()
        current = state.parent / "current"
        current.symlink_to("releases/0.3.0")
        self.apply()
        self.assertEqual(installer.read_record(state), installer.read_record(owner))
        self.assertEqual(
            installer.read_record(state)["sha256"], installer.validate(self.target)
        )
        self.assertEqual(current.readlink(), Path("releases/0.3.0"))
        with patch.object(installer, "running", return_value=False):
            installer.uninstall(state.parents[2])
        self.assertFalse(self.target.exists())
        self.assertFalse(state.exists())

    def test_legacy_default_install_is_identified_by_both_matching_records(self):
        state, owner = self.managed(schema=1, default=True)
        with patch.object(Path, "home", return_value=self.root):
            self.apply()
        self.assertEqual(installer.read_record(state), installer.read_record(owner))
        self.assertEqual(
            installer.read_record(state)["sha256"], installer.validate(self.target)
        )

    def test_legacy_custom_prefix_is_preserved_before_app_quits(self):
        state, owner = self.managed(schema=1)
        previous = state.read_bytes()
        with self.assertRaisesRegex(installer.AppError, "cannot locate"):
            self.apply()
        self.assertEqual(installer.validate(self.target), self.old_digest)
        self.assertEqual(state.read_bytes(), previous)
        self.assertEqual(owner.read_bytes(), previous)
        self.assertFalse((self.workspace / "ready").exists())

    def test_relaunch_failure_rolls_back_app_and_both_records(self):
        state, owner = self.managed()
        previous = state.read_bytes()
        self.command.side_effect = [installer.AppError("launch failed"), b""]
        with self.assertRaisesRegex(installer.AppError, "launch failed"):
            self.apply()
        self.assertEqual(installer.validate(self.target), self.old_digest)
        self.assertEqual(state.read_bytes(), previous)
        self.assertEqual(owner.read_bytes(), previous)
        self.assertEqual(self.command.call_count, 2)

    def test_record_failure_rolls_back_app_and_both_records(self):
        state, owner = self.managed()
        previous = state.read_bytes()
        write = installer.write_bytes
        failed = False

        def interrupted(path, content):
            nonlocal failed
            if path == owner and not failed:
                failed = True
                raise OSError("synthetic record failure")
            write(path, content)

        with (
            patch.object(installer, "write_bytes", side_effect=interrupted),
            self.assertRaises(OSError),
        ):
            self.apply()
        self.assertEqual(installer.validate(self.target), self.old_digest)
        self.assertEqual(state.read_bytes(), previous)
        self.assertEqual(owner.read_bytes(), previous)

    def test_changed_target_and_concurrent_installer_are_preserved(self):
        def change(*_):
            (self.target / "Contents/change").write_text("concurrent mutation")

        with (
            patch.object(updater, "wait_for_exit", side_effect=change),
            self.assertRaisesRegex(installer.AppError, "changed during update"),
        ):
            self.apply()
        self.assertTrue((self.target / "Contents/change").exists())
        lock = self.target.parent / ".Stark.app.install-lock"
        lock.mkdir()
        with self.assertRaisesRegex(installer.AppError, "in progress"):
            self.apply()
        self.assertTrue(lock.exists())

    def test_failed_candidate_verification_never_quits_or_replaces_app(self):
        with (
            patch.object(
                updater,
                "verify_release",
                side_effect=installer.AppError("invalid signature"),
            ),
            self.assertRaises(installer.AppError),
        ):
            self.apply()
        self.assertFalse((self.workspace / "ready").exists())
        self.assertEqual(installer.validate(self.target), self.old_digest)

    def test_wait_for_exit_times_out_while_parent_is_alive(self):
        with (
            patch.object(updater.time, "monotonic", side_effect=[0, 2]),
            self.assertRaisesRegex(installer.AppError, "did not exit"),
        ):
            self.wait_for_exit(os.getpid(), self.target, 1)


SWIFT_DRIVER = r"""
import Foundation
precondition(AppUpdateFiles.supportsArm64(Data([0xcf,0xfa,0xed,0xfe,0x0c,0,0,1])))
precondition(!AppUpdateFiles.supportsArm64(Data([0xcf,0xfa,0xed,0xfe,7,0,0,1])))
precondition(!AppUpdateFiles.supportsArm64(Data([0xca,0xfe,0xba,0xbe,0xff,0xff,0xff,0xff])))
var fat = Data([0xca,0xfe,0xba,0xbe,0,0,0,1,1,0,0,0x0c])
fat.append(Data(repeating: 0, count: 16))
precondition(AppUpdateFiles.supportsArm64(fat))
let valid = ["0.3.0", "0.10.0", "1.0.0", "12.34.567"]
let invalid = ["0.3", "01.2.3", "1.2.3-beta", "v1.2.3", "-1.2.3", "1.2.3.4", "1.2.999999999999999", "1.2.٣"]
precondition(valid.allSatisfy { ReleaseVersion($0) != nil })
precondition(invalid.allSatisfy { ReleaseVersion($0) == nil })
precondition(ReleaseVersion("0.10.0")! > ReleaseVersion("0.9.9")!)
let asset: [String: Any] = [
    "name": "stark-0.4.0-aarch64-apple-darwin.dmg", "size": 100,
    "browser_download_url": "https://github.com/jshahbazi/stark/releases/download/v0.4.0/stark-0.4.0-aarch64-apple-darwin.dmg",
    "digest": "sha256:" + String(repeating: "a", count: 64)
]
func decode(_ assets: [[String: Any]], tag: String = "v0.4.0", prerelease: Bool = false) throws -> AppRelease {
    let data = try JSONSerialization.data(withJSONObject: ["tag_name": tag, "draft": false, "prerelease": prerelease, "assets": assets])
    return try JSONDecoder().decode(AppRelease.self, from: data)
}
let newer = try decode([asset]).update(after: "0.3.0")
let same = try decode([asset]).update(after: "0.4.0")
let older = try decode([asset]).update(after: "0.5.0")
precondition(newer?.version == "0.4.0" && same == nil && older == nil)
for key in ["name", "digest", "browser_download_url", "size"] {
    var broken = asset
    broken[key] = key == "size" ? 2_147_483_648 : "invalid"
    do { _ = try decode([broken]).update(after: "0.3.0"); fatalError("unsafe release accepted") } catch {}
}
do { _ = try decode([asset], prerelease: true).update(after: "0.3.0"); fatalError("prerelease accepted") } catch {}
do { _ = try decode([]).update(after: "0.3.0"); fatalError("CLI-only release accepted") } catch {}
if CommandLine.arguments.count == 5 {
    let image = URL(fileURLWithPath: CommandLine.arguments[1])
    let workspace = URL(fileURLWithPath: CommandLine.arguments[2])
    let release = AppUpdateDownload(version: "0.3.0", url: image, sha256: CommandLine.arguments[3], size: Int64(CommandLine.arguments[4])!)
    let candidate = try AppUpdateFiles.prepare(image, release: release, workspace: workspace)
    precondition(FileManager.default.fileExists(atPath: candidate.appendingPathComponent("Contents/MacOS/Stark").path))
}
print("Swift updater checks passed")
"""


@unittest.skipUnless(sys.platform == "darwin", "AppKit required")
class SwiftUpdateTests(unittest.TestCase):
    def test_production_release_selection_and_full_app_compilation(self):
        with tempfile.TemporaryDirectory(prefix="stark swift updater ") as temporary:
            root = Path(temporary)
            driver = root / "main.swift"
            driver.write_text(SWIFT_DRIVER)
            result = subprocess.run(
                [
                    "xcrun",
                    "swiftc",
                    "-target",
                    "arm64-apple-macos26.4",
                    str(ROOT / "apps/menu-bar/Sources/AppUpdate.swift"),
                    str(driver),
                    "-o",
                    str(root / "checks"),
                ],
                capture_output=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            subprocess.run([str(root / "checks")], check=True, timeout=30)
            sources = sorted((ROOT / "apps/menu-bar/Sources").glob("*.swift"))
            subprocess.run(
                [
                    "xcrun",
                    "swiftc",
                    "-typecheck",
                    "-target",
                    "arm64-apple-macos26.4",
                    *map(str, sources),
                ],
                check=True,
                timeout=120,
                capture_output=True,
            )


if __name__ == "__main__":
    unittest.main()
