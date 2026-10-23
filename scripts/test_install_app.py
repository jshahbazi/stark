"""Combined installation ownership and rollback checks, without a model."""

from __future__ import annotations

import importlib.util
import json
import os
import plistlib
import shutil
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "install_app", Path(__file__).with_name("install_app.py")
)
assert SPEC is not None and SPEC.loader is not None
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


def app_fixture(app: Path, text: str = "one") -> Path:
    executable = app / "Contents/MacOS/Stark"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    (app / "Contents/Info.plist").write_bytes(
        plistlib.dumps(
            {
                "CFBundleIdentifier": installer.APP_ID,
                "CFBundleExecutable": "Stark",
            }
        )
    )
    (app / "Contents/version").write_text(text)
    return app


class AppValidationTests(unittest.TestCase):
    def test_rejects_symlink_payload_entries_and_wrong_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app = app_fixture(root / "Stark.app")
            (app / "Contents/alias").symlink_to(root)
            with self.assertRaisesRegex(installer.AppError, "unsupported filesystem"):
                installer.validate(app)
            (app / "Contents/alias").unlink()
            (app / "Contents/Info.plist").write_bytes(
                plistlib.dumps(
                    {
                        "CFBundleIdentifier": "example.unrelated",
                        "CFBundleExecutable": "Stark",
                    }
                )
            )
            with self.assertRaisesRegex(installer.AppError, "unrelated application"):
                installer.validate(app)

    def test_rejects_overlap_root_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for prefix, apps in (
                (Path("/"), root),
                (root, Path("/")),
                (Path("relative"), root),
                (root / "prefix", root / "prefix/apps"),
                (root / "Stark.app/data", root),
            ):
                with (
                    self.subTest(prefix=prefix, apps=apps),
                    self.assertRaises(installer.AppError),
                ):
                    installer.paths(prefix, apps)

    def test_uninstall_without_managed_app_is_compatible_with_cli_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            installer.uninstall(Path(temporary))

    def test_invalid_state_does_not_authorize_deletion(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state"
            state.write_text(json.dumps({"path": "/Applications/Other.app"}))
            with self.assertRaisesRegex(installer.AppError, "record is invalid"):
                installer.read_record(state)


@unittest.skipUnless(sys.platform == "darwin", "exercises Darwin atomic replacement")
class AppTransactionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="stark app transaction ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.prefix = self.root / "CLI prefix"
        (self.prefix / "bin").mkdir(parents=True)
        self.base = self.prefix / "lib/stark"
        (self.base / "releases").mkdir(parents=True)
        self.apps = self.root / "App parent"
        self.target = self.apps / "Stark.app"
        self.state = self.base / installer.STATE_NAME
        self.owner = self.apps / installer.OWNER_NAME
        self.addCleanup(patch.stopall)
        patch.object(installer, "verify_signature").start()
        patch.object(installer, "running", return_value=False).start()

    def bundle(self, version: str) -> Path:
        bundle = self.base / "releases" / version
        app_fixture(bundle / "app/Stark.app", version)
        return bundle

    def install(self, version: str) -> Path:
        bundle = self.bundle(version)
        installer.install(bundle, self.prefix, self.apps)
        return bundle

    def test_install_reinstall_upgrade_relocation_and_uninstall(self):
        first = self.install("0.1.2")
        identifier = installer.read_record(self.state)["installation_id"]
        installer.install(first, self.prefix, self.apps)
        self.install("0.1.3")
        self.assertEqual((self.base / "current").readlink(), Path("releases/0.1.3"))
        self.assertEqual(
            installer.read_record(self.state)["installation_id"], identifier
        )
        self.assertEqual(
            installer.read_record(self.state), installer.read_record(self.owner)
        )
        self.assertTrue(self.target.is_dir())
        self.assertFalse(self.target.is_symlink())
        moved = self.root / "moved CLI"
        self.prefix.rename(moved)
        data = self.root / "Library/Application Support/Stark/models/keep"
        data.parent.mkdir(parents=True)
        data.write_text("private model marker")
        installer.uninstall(moved)
        self.assertFalse(self.target.exists())
        self.assertFalse(self.owner.exists())
        self.assertFalse((moved / "lib/stark" / installer.STATE_NAME).exists())
        self.assertEqual(data.read_text(), "private model marker")

    def test_state_write_failure_restores_app_current_links_and_both_records(self):
        self.install("0.1.2")
        previous_digest = installer.validate(self.target)
        previous_state = self.state.read_bytes()
        previous_owner = self.owner.read_bytes()
        bundle = self.bundle("0.1.3")
        original_write = installer.write_bytes
        failed = False

        def write(path, content):
            nonlocal failed
            if path == self.owner and not failed:
                failed = True
                raise OSError("synthetic state failure")
            original_write(path, content)

        with (
            patch.object(installer, "write_bytes", side_effect=write),
            self.assertRaises(OSError),
        ):
            installer.install(bundle, self.prefix, self.apps)
        self.assertEqual(installer.validate(self.target), previous_digest)
        self.assertEqual(self.state.read_bytes(), previous_state)
        self.assertEqual(self.owner.read_bytes(), previous_owner)
        self.assertEqual((self.base / "current").readlink(), Path("releases/0.1.2"))
        self.assertEqual(list(self.apps.glob(".stark-install-*")), [])

    def test_failed_first_cli_activation_removes_new_app_and_links(self):
        original_switch = installer.switch_current

        def switch(base, destination):
            if destination is not None:
                raise OSError("synthetic link failure")
            original_switch(base, destination)

        with (
            patch.object(installer, "switch_current", side_effect=switch),
            self.assertRaises(OSError),
        ):
            self.install("0.1.2")
        self.assertFalse(self.target.exists())
        self.assertFalse(self.state.exists())
        self.assertFalse(self.owner.exists())
        self.assertFalse((self.base / "current").is_symlink())
        self.assertEqual(list((self.prefix / "bin").iterdir()), [])

    def test_interrupt_after_link_syscall_removes_the_new_command_link(self):
        original_link = Path.symlink_to
        interrupted = False

        def link(path, destination, *args, **kwargs):
            nonlocal interrupted
            original_link(path, destination, *args, **kwargs)
            if path.parent == self.prefix / "bin" and not interrupted:
                interrupted = True
                os.kill(os.getpid(), signal.SIGINT)

        with (
            patch.object(Path, "symlink_to", link),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.install("0.1.2")
        self.assertTrue(interrupted)
        self.assertEqual(list((self.prefix / "bin").iterdir()), [])
        self.assertFalse(self.target.exists())
        self.assertFalse(self.state.exists())

    def test_uninstall_rollback_rename_failure_preserves_the_only_app_copy(self):
        self.install("0.1.2")
        expected = installer.validate(self.target)
        original_unlink = Path.unlink
        original_rename = installer.rename

        def unlink(path, *args, **kwargs):
            if path == self.state:
                raise OSError("synthetic state removal failure")
            original_unlink(path, *args, **kwargs)

        def rename(source, destination, **kwargs):
            if destination == self.target:
                raise OSError("synthetic app restore failure")
            original_rename(source, destination, **kwargs)

        with (
            patch.object(Path, "unlink", unlink),
            patch.object(installer, "rename", side_effect=rename),
            self.assertRaisesRegex(
                installer.AppError, "private app backup was preserved"
            ),
        ):
            installer.uninstall(self.prefix)
        backups = list(self.apps.glob(".stark-uninstall-*/Stark.app"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(installer.validate(backups[0]), expected)
        self.assertFalse(self.target.exists())
        self.assertTrue(self.state.exists())
        self.assertTrue((self.base / "current").is_symlink())

    def test_uninstall_rollback_record_failure_preserves_restored_app(self):
        self.install("0.1.2")
        expected = installer.validate(self.target)
        original_unlink = Path.unlink

        def unlink(path, *args, **kwargs):
            if path == self.state:
                raise OSError("synthetic state removal failure")
            original_unlink(path, *args, **kwargs)

        with (
            patch.object(Path, "unlink", unlink),
            patch.object(
                installer, "write_bytes", side_effect=OSError("record restore")
            ),
            self.assertRaisesRegex(
                installer.AppError, "private app backup was preserved"
            ),
        ):
            installer.uninstall(self.prefix)
        self.assertEqual(installer.validate(self.target), expected)
        self.assertEqual(len(list(self.apps.glob(".stark-uninstall-*"))), 1)
        self.assertTrue(self.state.exists())

    def test_symlink_unrelated_running_and_modified_apps_are_preserved(self):
        self.apps.mkdir()
        self.target.symlink_to(self.root)
        with self.assertRaisesRegex(installer.AppError, "symlink"):
            self.install("0.1.2")
        self.target.unlink()
        self.target.mkdir()
        (self.target / "keep").write_text("unrelated")
        with self.assertRaisesRegex(installer.AppError, "metadata"):
            installer.install(self.base / "releases/0.1.2", self.prefix, self.apps)
        self.assertEqual((self.target / "keep").read_text(), "unrelated")
        shutil.rmtree(self.target)
        installer.install(self.base / "releases/0.1.2", self.prefix, self.apps)
        with (
            patch.object(installer, "running", return_value=True),
            self.assertRaisesRegex(installer.AppError, "running"),
        ):
            installer.uninstall(self.prefix)
        (self.target / "Contents/version").write_text("user modification")
        with self.assertRaisesRegex(installer.AppError, "changed"):
            installer.uninstall(self.prefix)
        self.assertTrue(self.state.exists())
        self.assertTrue(self.target.exists())

    def test_second_prefix_cannot_claim_the_same_app(self):
        self.install("0.1.2")
        second = self.root / "second prefix"
        (second / "bin").mkdir(parents=True)
        bundle = second / "lib/stark/releases/0.1.2"
        shutil.copytree(self.base / "releases/0.1.2", bundle)
        with self.assertRaisesRegex(installer.AppError, "another Stark installation"):
            installer.install(bundle, second, self.apps)
        self.assertTrue(self.state.exists())
        self.assertFalse((second / "lib/stark/current").exists())

    def test_failed_signature_and_copy_leave_current_installation_untouched(self):
        self.install("0.1.2")
        before = installer.validate(self.target)
        bundle = self.bundle("0.1.3")
        with (
            patch.object(
                installer,
                "verify_signature",
                side_effect=installer.AppError("signature"),
            ),
            self.assertRaisesRegex(installer.AppError, "signature"),
        ):
            installer.install(bundle, self.prefix, self.apps)
        self.assertEqual(installer.validate(self.target), before)
        with (
            patch.object(installer.shutil, "copytree", side_effect=OSError("copy")),
            self.assertRaises(OSError),
        ):
            installer.install(bundle, self.prefix, self.apps)
        self.assertEqual(installer.validate(self.target), before)
        self.assertEqual((self.base / "current").readlink(), Path("releases/0.1.2"))


if __name__ == "__main__":
    unittest.main()
