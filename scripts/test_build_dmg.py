"""Model-free disk-image integrity, signing order, and mount cleanup checks."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dmg as dmg
import update_app as updater

SOURCE = "a" * 40
IDENTITY = {"sha1": "B" * 40, "team_id": "ABCDEFGHIJ"}
SUBMISSION = "12345678-1234-1234-1234-123456789abc"


class DMGTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "source/stark"
        self.app = self.bundle / dmg.signing.APPLICATION
        executable = self.app / "Contents/MacOS/Stark"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b"synthetic executable")
        (self.app / "Contents/Info.plist").write_bytes(
            plistlib.dumps(
                {
                    "CFBundleIdentifier": dmg.signing.APPLICATION_ID,
                    "CFBundleExecutable": "Stark",
                    "CFBundleShortVersionString": "0.3.0",
                }
            )
        )
        runtime = self.app / "Contents/Resources/runtime/share/stark"
        runtime.mkdir(parents=True)
        (runtime / "manifest.json").write_text(
            json.dumps({"source": {"revision": SOURCE}, "version": "0.3.0"})
        )
        share = self.bundle / "share/stark"
        share.mkdir(parents=True)
        (share / "VERSION").write_text("0.3.0\n")
        self.manifest = {
            "schema_version": 1,
            "version": "0.3.0",
            "target": dmg.signing.distribution.TARGET,
            "minimum_macos": dmg.signing.distribution.MIN_MACOS,
            "source": {"revision": SOURCE, "dirty": False},
            "files_sha256": dmg.signing.distribution.file_manifest(self.bundle),
        }
        (self.bundle / dmg.signing.MANIFEST).write_text(json.dumps(self.manifest))
        self.archive = self.root / "input.tar.gz"
        self.digest = dmg.signing.distribution.create_archive(self.bundle, self.archive)
        self.options = argparse.Namespace(
            archive=self.archive,
            input_sha256=self.digest,
            expected_source=SOURCE,
            version="0.3.0",
            sign_identity=IDENTITY["sha1"],
            keychain=Path("/private/synthetic.keychain"),
            notary_profile="synthetic-profile",
            output_dir=self.root / "output",
        )
        self.commands = []
        self.fail_stage = None
        self.tamper = False
        self.tamper_installer = False
        self.notarization_status = "Accepted"
        self.image_app = self.app
        self.image_installer = None
        self.installer_snapshot = self.root / "installer-snapshot.app"
        self.metadata = (
            f"CodeDirectory v=20500 flags=0x10000(runtime)\nTeamIdentifier={IDENTITY['team_id']}\nTimestamp=Oct 23, 2026\n"
        ).encode()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            patch.object(dmg.platform, "system", return_value="Darwin")
        )
        self.stack.enter_context(
            patch.object(dmg.platform, "machine", return_value="arm64")
        )
        self.stack.enter_context(
            patch.object(dmg.signing, "resolve_identity", return_value=IDENTITY)
        )
        self.stack.enter_context(
            patch.object(dmg.signing, "run", side_effect=self.run_command)
        )

    def run_command(self, arguments, stage, env, **kwargs):
        arguments = list(map(str, arguments))
        self.commands.append((stage, arguments))
        if stage == self.fail_stage or (
            isinstance(self.fail_stage, set) and stage in self.fail_stage
        ):
            raise dmg.signing.SigningError(stage)
        stdout, stderr = b"", b""
        if stage == "dmg_recipe_revision":
            stdout = (SOURCE + "\n").encode()
        elif stage == "dmg_installer_compilation_failed":
            Path(arguments[arguments.index("-o") + 1]).write_bytes(
                b"synthetic installer"
            )
        elif stage == "dmg_creation_failed":
            self.image_app = (
                Path(arguments[arguments.index("-srcfolder") + 1]) / "Stark.app"
            )
            self.image_installer = self.image_app.parent / dmg.INSTALLER_NAME
            shutil.copytree(self.image_installer, self.installer_snapshot)
            Path(arguments[-1]).write_bytes(b"synthetic compressed image")
        elif stage in {"code_signature_metadata", "dmg_signature_metadata"}:
            stderr = self.metadata
        elif stage == "notarization_failed":
            stdout = json.dumps(
                {"status": self.notarization_status, "id": SUBMISSION}
            ).encode()
        elif stage == "dmg_mount_failed":
            self.assertIn("-readonly", arguments)
            self.assertIn("-nobrowse", arguments)
            self.assertIn("-noautoopen", arguments)
            destination = Path(arguments[arguments.index("-mountpoint") + 1])
            shutil.copytree(self.image_app, destination / "Stark.app")
            if self.image_installer is not None:
                shutil.copytree(self.image_installer, destination / dmg.INSTALLER_NAME)
            (destination / "Applications").symlink_to("/Applications")
            if self.tamper:
                (destination / "Stark.app/Contents/MacOS/Stark").write_bytes(b"changed")
            if self.tamper_installer:
                (
                    destination / dmg.INSTALLER_NAME / "Contents/MacOS/InstallStark"
                ).write_bytes(b"changed")
        return subprocess.CompletedProcess(arguments, 0, stdout=stdout, stderr=stderr)

    def build(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return dmg.build(self.options)

    def stages(self):
        return [stage for stage, _ in self.commands]

    def test_success_preserves_application_and_verifies_before_publication(self):
        before = self.archive.read_bytes()
        receipt = self.build()
        self.assertEqual(self.archive.read_bytes(), before)
        self.assertEqual(receipt["input_archive_sha256"], self.digest)
        self.assertEqual(
            receipt["application_inventory_sha256"], dmg.inventory_digest(self.app)
        )
        self.assertTrue(receipt["application_bytes_preserved"])
        self.assertEqual(receipt["installer_bundle_id"], dmg.INSTALLER_ID)
        self.assertEqual(
            receipt["installer_inventory_sha256"],
            dmg.inventory_digest(self.installer_snapshot),
        )
        self.assertEqual(
            dmg.inventory_digest(
                self.installer_snapshot / "Contents/Helpers/Stark.app"
            ),
            dmg.inventory_digest(self.app),
        )
        metadata = plistlib.loads(
            (self.installer_snapshot / "Contents/Info.plist").read_bytes()
        )
        self.assertNotEqual(metadata["CFBundleIdentifier"], dmg.signing.APPLICATION_ID)
        for name in ("install_app.py", "update_app.py"):
            self.assertEqual(
                (self.installer_snapshot / "Contents/Resources" / name).read_bytes(),
                (dmg.ROOT / "scripts" / name).read_bytes(),
            )
        self.assertEqual(receipt["notarization"]["status"], "accepted")
        self.assertEqual(receipt["distribution_source"]["revision"], SOURCE)
        self.assertIn(
            "scripts/build_dmg.py", receipt["distribution_source"]["files_sha256"]
        )
        stages = self.stages()
        self.assertLess(
            stages.index("application_ticket_invalid"),
            stages.index("dmg_creation_failed"),
        )
        self.assertLess(
            stages.index("dmg_installer_signing_failed"),
            stages.index("dmg_creation_failed"),
        )
        self.assertIn("dmg_installer_gatekeeper_rejected", stages)
        self.assertLess(
            stages.index("dmg_signing_failed"), stages.index("notarization_failed")
        )
        self.assertLess(
            stages.index("notarization_failed"), stages.index("dmg_stapling_failed")
        )
        self.assertLess(
            stages.index("dmg_stapling_failed"), stages.index("dmg_mount_failed")
        )
        self.assertEqual(stages[-1], "dmg_detach_failed")
        names = sorted(path.name for path in self.options.output_dir.iterdir())
        self.assertEqual(len(names), 3)
        self.assertTrue(
            all(
                name.startswith("stark-0.3.0-aarch64-apple-darwin.dmg")
                for name in names
            )
        )

    def test_hash_mismatch_does_not_access_signing_identity_or_publish(self):
        self.options.input_sha256 = "0" * 64
        with self.assertRaisesRegex(
            dmg.signing.SigningError, "input_archive_hash_mismatch"
        ):
            self.build()
        self.assertNotIn("dmg_signing_failed", self.stages())
        self.assertEqual(list(self.options.output_dir.iterdir()), [])

    def test_wrong_source_or_version_cannot_be_repackaged(self):
        for field, value, message in (
            ("expected_source", "c" * 40, "input_source_identity"),
            ("version", "0.4.0", "dmg_input_version_mismatch"),
        ):
            with self.subTest(field=field):
                previous = getattr(self.options, field)
                setattr(self.options, field, value)
                with self.assertRaisesRegex(dmg.signing.SigningError, message):
                    self.build()
                setattr(self.options, field, previous)

    def test_existing_output_and_dangling_symlinks_are_never_replaced(self):
        self.options.output_dir.mkdir()
        path = self.options.output_dir / "stark-0.3.0-aarch64-apple-darwin.dmg"
        path.symlink_to("/nonexistent/synthetic")
        with self.assertRaisesRegex(
            dmg.signing.SigningError, "dmg_output_already_exists"
        ):
            self.build()
        self.assertTrue(path.is_symlink())
        self.assertNotIn("dmg_creation_failed", self.stages())

    def test_unsigned_input_is_rejected_before_image_creation(self):
        self.fail_stage = "code_signature_invalid"
        with self.assertRaisesRegex(dmg.signing.SigningError, "code_signature_invalid"):
            self.build()
        self.assertNotIn("dmg_creation_failed", self.stages())

    def test_notarization_rejection_publishes_nothing(self):
        self.notarization_status = "Invalid"
        with self.assertRaisesRegex(
            dmg.signing.SigningError, "notarization_not_accepted"
        ):
            self.build()
        self.assertNotIn("dmg_stapling_failed", self.stages())
        self.assertEqual(list(self.options.output_dir.iterdir()), [])

    def test_mount_verification_failure_detaches_and_publishes_nothing(self):
        self.tamper = True
        with self.assertRaisesRegex(
            dmg.signing.SigningError, "dmg_application_bytes_changed"
        ):
            self.build()
        self.assertEqual(self.stages()[-1], "dmg_detach_failed")
        self.assertEqual(list(self.options.output_dir.iterdir()), [])

    def test_app_signature_failure_after_mount_always_detaches(self):
        self.fail_stage = "dmg_application_seal_invalid"
        with self.assertRaisesRegex(dmg.signing.SigningError, self.fail_stage):
            self.build()
        self.assertEqual(self.stages()[-1], "dmg_detach_failed")

    def test_installer_tampering_detaches_and_publishes_nothing(self):
        self.tamper_installer = True
        with self.assertRaisesRegex(
            dmg.signing.SigningError, "dmg_installer_bytes_changed"
        ):
            self.build()
        self.assertEqual(self.stages()[-1], "dmg_detach_failed")
        self.assertEqual(list(self.options.output_dir.iterdir()), [])

    def test_installer_compilation_and_signing_failure_publish_nothing(self):
        for stage in (
            "dmg_installer_compilation_failed",
            "dmg_installer_signing_failed",
        ):
            with self.subTest(stage=stage):
                self.fail_stage = stage
                with self.assertRaisesRegex(dmg.signing.SigningError, stage):
                    self.build()
                self.assertNotIn("dmg_creation_failed", self.stages())
                self.assertEqual(list(self.options.output_dir.iterdir()), [])

    def test_disk_image_requires_secure_timestamp(self):
        self.metadata = f"TeamIdentifier={IDENTITY['team_id']}\n".encode()
        with self.assertRaisesRegex(
            dmg.signing.SigningError, "dmg_secure_timestamp_missing"
        ):
            dmg.verify_image_signature(self.archive, IDENTITY, {})

    def test_busy_private_mount_is_force_detached(self):
        self.fail_stage = "dmg_detach_failed"
        self.build()
        self.assertEqual(self.stages()[-1], "dmg_force_detach_failed")
        self.assertIn("-force", self.commands[-1][1])

    def test_failed_detach_never_publishes_an_image(self):
        self.fail_stage = {"dmg_detach_failed", "dmg_force_detach_failed"}
        with self.assertRaisesRegex(
            dmg.signing.SigningError, "dmg_force_detach_failed"
        ):
            self.build()
        self.assertEqual(list(self.options.output_dir.iterdir()), [])

    def test_receiving_verification_never_imports_signing_credentials(self):
        receipt = self.build()
        self.image_app = self.app
        self.image_installer = self.installer_snapshot
        image = self.options.output_dir / "stark-0.3.0-aarch64-apple-darwin.dmg"
        receipt_path = image.with_suffix(".dmg.notarization.json")
        options = argparse.Namespace(
            verify_dmg=image,
            receipt=receipt_path,
            expected_source=SOURCE,
            expected_signer=IDENTITY["sha1"],
            version="0.3.0",
            output=self.root / "receiving.json",
        )
        with (
            patch.object(
                dmg.signing,
                "resolve_identity",
                side_effect=AssertionError("must not read credentials"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = dmg.verify_download(options)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["dmg_sha256"], receipt["dmg_sha256"])
        self.assertIn("installer_gatekeeper", result["checks"])
        image.write_bytes(b"different downloaded bytes")
        with self.assertRaisesRegex(
            dmg.signing.SigningError, "dmg_download_hash_mismatch"
        ):
            dmg.verify_download(options)


@unittest.skipUnless(sys.platform == "darwin", "Xcode Swift compiler required")
class NativeInstallerTests(unittest.TestCase):
    def test_native_workspace_passes_helper_validation_and_reports_early_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "Harness.swift"
            source.write_text(
                """import Foundation
@main struct Harness {
    static func main() throws {
        let workspace = try InstallerFiles.createWorkspace()
        let updateWorkspace = try AppUpdateFiles.createWorkspace(prefix: "stark-update-")
        let early = Data(#"{"error":"installation workspace is invalid"}"#.utf8)
        do {
            _ = try InstallerFiles.helperResult(status: 1, output: early, workspace: workspace)
            preconditionFailure("early helper failure was hidden")
        } catch let failure as InstallerFailure {
            precondition(failure.message == "installation workspace is invalid")
        }
        try Data(#"{"error":"recorded failure"}"#.utf8).write(to: workspace.appendingPathComponent("result.json"))
        do {
            _ = try InstallerFiles.helperResult(status: 1, output: early, workspace: workspace)
            preconditionFailure("recorded helper failure was hidden")
        } catch let failure as InstallerFailure {
            precondition(failure.message == "recorded failure")
        }
        try FileManager.default.removeItem(at: workspace.appendingPathComponent("result.json"))
        for output in [Data("invalid".utf8), Data(repeating: 65, count: 16_385)] {
            do {
                _ = try InstallerFiles.helperResult(status: 1, output: output, workspace: workspace)
                preconditionFailure("invalid helper reply was accepted")
            } catch InstallerError.install {}
        }
        let installed = try InstallerFiles.helperResult(status: 0,
            output: Data(#"{"path":"/Applications/Stark.app","version":"0.6.1"}"#.utf8), workspace: workspace)
        precondition(installed.version == "0.6.1")
        let paths = ["installer": workspace, "updater": updateWorkspace].mapValues { directory in
            ["workspace": directory.path,
             "candidate": directory.appendingPathComponent("Stark.app").path,
             "helper": directory.appendingPathComponent("update_app.py").path]
        }
        let data = try JSONSerialization.data(withJSONObject: paths)
        FileHandle.standardOutput.write(data)
    }
}
"""
            )
            executable = root / "harness"
            compiled = subprocess.run(
                [
                    "xcrun",
                    "swiftc",
                    "-O",
                    "-D",
                    "STARK_INSTALLER_LIBRARY",
                    "-target",
                    "arm64-apple-macos26.4",
                    "-framework",
                    "AppKit",
                    "-framework",
                    "SwiftUI",
                    str(dmg.ROOT / "apps/menu-bar/Sources/AppUpdate.swift"),
                    str(dmg.ROOT / "apps/installer/Sources/InstallerApp.swift"),
                    str(source),
                    "-o",
                    str(executable),
                ],
                capture_output=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(
                compiled.returncode, 0, "native regression fixture compilation failed"
            )
            result = subprocess.run(
                [str(executable)], capture_output=True, timeout=30, check=False
            )
            self.assertEqual(
                result.returncode, 0, "native installer regression fixture failed"
            )
            paths = json.loads(result.stdout)
            for record in paths.values():
                self.addCleanup(
                    shutil.rmtree, Path(record["workspace"]), ignore_errors=True
                )
            self.assertEqual(set(paths), {"installer", "updater"})
            for name, record in paths.items():
                with self.subTest(workspace=name):
                    workspace = Path(record["workspace"])
                    self.assertTrue(workspace.is_dir())
                    self.assertEqual(
                        updater.physical_path(workspace, "invalid workspace"),
                        workspace,
                    )
                    self.assertEqual(workspace.stat().st_mode & 0o777, 0o700)
                    self.assertEqual(workspace.stat().st_uid, os.getuid())
                    for field, filename in (
                        ("candidate", "Stark.app"),
                        ("helper", "update_app.py"),
                    ):
                        child = Path(record[field])
                        self.assertEqual(child, workspace / filename)
                        self.assertEqual(
                            updater.physical_path(child, "invalid child path"), child
                        )

    def test_installer_compiles_with_shared_release_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    "xcrun",
                    "swiftc",
                    "-O",
                    "-target",
                    "arm64-apple-macos26.4",
                    "-framework",
                    "AppKit",
                    "-framework",
                    "SwiftUI",
                    str(dmg.ROOT / "apps/menu-bar/Sources/AppUpdate.swift"),
                    str(dmg.ROOT / "apps/installer/Sources/InstallerApp.swift"),
                    "-o",
                    str(Path(temporary) / "InstallStark"),
                ],
                capture_output=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(
                result.returncode, 0, "native installer compilation failed"
            )


if __name__ == "__main__":
    unittest.main()
