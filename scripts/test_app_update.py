"""Model-free tests of published update selection and transactional app replacement."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
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
        self.assertEqual(self.command.call_count, 1)

    def test_failure_is_recorded_before_reopening_previous_app(self):
        launches = []

        def command(arguments, _message):
            launches.append(arguments)
            if len(launches) == 1:
                raise installer.AppError("updated app could not launch")
            result = json.loads((self.workspace / "result.json").read_text())
            self.assertEqual(result["error"], "updated app could not launch")
            self.assertEqual(installer.validate(self.target), self.old_digest)
            return b""

        self.command.side_effect = command
        with (
            patch.object(installer, "running", return_value=False),
            patch.object(
                sys,
                "argv",
                [
                    "update_app.py",
                    "--candidate",
                    str(self.candidate),
                    "--target",
                    str(self.target),
                    "--version",
                    "0.4.0",
                    "--pid",
                    str(os.getpid()),
                    "--workspace",
                    str(self.workspace),
                ],
            ),
        ):
            self.assertEqual(updater.main(), 1)
        self.assertEqual(len(launches), 2)

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


@unittest.skipUnless(sys.platform == "darwin", "Darwin atomic exchange required")
class DMGInstallTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="stark dmg install ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.system = self.root / "Applications"
        self.system.mkdir()
        self.home = self.root / "home"
        self.home.mkdir()
        self.candidate = fixture(self.root / "disk image/Stark.app", "0.4.0")
        self.workspace = self.root / "workspace"
        self.workspace.mkdir(mode=0o700)
        self.processes = b""
        self.addCleanup(patch.stopall)
        patch.object(updater, "SYSTEM_APPLICATIONS", self.system).start()
        patch.object(Path, "home", return_value=self.home).start()
        self.verify = patch.object(
            updater,
            "verify_release",
            side_effect=lambda app, _version: installer.validate(app),
        ).start()
        self.signature = patch.object(installer, "verify_signature").start()
        self.stop = patch.object(installer, "stop_app").start()
        self.wait_for_launch = updater.wait_for_launch
        self.launch = patch.object(updater, "wait_for_launch").start()
        self.command = patch.object(
            installer,
            "command",
            side_effect=lambda arguments, _message: (
                self.processes if arguments[0] == "/bin/ps" else b""
            ),
        ).start()

    def running(self, *applications):
        self.processes = "".join(
            f"{os.getuid()} {app}/Contents/MacOS/Stark\n" for app in applications
        ).encode()

    def invoke(self):
        output = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                [
                    "update_app.py",
                    "--candidate",
                    str(self.candidate),
                    "--workspace",
                    str(self.workspace),
                    "--install-from-dmg",
                ],
            ),
            patch.object(sys, "stdout", output),
        ):
            result = updater.main()
        return result, json.loads(output.getvalue())

    def test_running_custom_copy_is_replaced_and_other_copy_is_preserved(self):
        target = fixture(self.root / "custom apps/Stark.app", "0.3.0")
        other = fixture(self.system / "Stark.app", "0.2.0")
        previous_other = installer.validate(other)
        self.running(target)
        result, output = self.invoke()
        self.assertEqual(result, 0)
        self.assertEqual(output, {"path": str(target), "version": "0.4.0"})
        self.stop.assert_called_once_with(target)
        self.assertEqual(installer.validate(other), previous_other)
        self.assertEqual(installer.validate(target), installer.validate(self.candidate))
        self.assertEqual(
            self.command.call_args.args[0], ["/usr/bin/open", "-n", target]
        )
        self.assertFalse(self.workspace.exists())

    def test_first_install_uses_system_directory_and_launches_exact_target(self):
        target = self.system / "Stark.app"
        result, output = self.invoke()
        self.assertEqual(result, 0)
        self.assertEqual(output["path"], str(target))
        self.assertEqual(installer.validate(target), installer.validate(self.candidate))
        self.assertEqual(
            self.command.call_args.args[0], ["/usr/bin/open", "-n", target]
        )
        self.assertTrue(self.candidate.is_dir())

    def test_first_install_creates_user_applications_when_system_is_unwritable(self):
        access = os.access
        with patch.object(
            os,
            "access",
            side_effect=lambda path, mode: (
                False if Path(path) == self.system else access(path, mode)
            ),
        ):
            result, output = self.invoke()
        self.assertEqual(result, 0)
        self.assertEqual(output["path"], str(self.home / "Applications/Stark.app"))

    def test_existing_user_copy_is_selected_before_empty_system_directory(self):
        target = fixture(self.home / "Applications/Stark.app", "0.3.0")
        result, output = self.invoke()
        self.assertEqual(result, 0)
        self.assertEqual(output["path"], str(target))

    def test_staging_and_signature_verification_finish_before_quitting(self):
        target = fixture(self.system / "Stark.app", "0.3.0")
        old_digest = installer.validate(target)

        def stop(app):
            self.assertEqual(app, target)
            self.assertEqual(installer.validate(target), old_digest)
            staged = list(self.system.glob(".stark-update-*/Stark.app"))
            self.assertEqual(len(staged), 1)
            self.assertGreaterEqual(
                sum(call.args[0] == staged[0] for call in self.verify.call_args_list), 1
            )
            self.assertEqual(
                installer.validate(staged[0]), installer.validate(self.candidate)
            )

        self.stop.side_effect = stop
        self.assertEqual(self.invoke()[0], 0)

    def test_invalid_staged_signature_never_quits_the_old_app(self):
        target = fixture(self.system / "Stark.app", "0.3.0")
        previous = installer.validate(target)

        def verify(app, _version):
            if app != self.candidate:
                raise installer.AppError("update signature verification failed")
            return installer.validate(app)

        self.verify.side_effect = verify
        result, output = self.invoke()
        self.assertEqual(result, 1)
        self.assertEqual(output, {"error": "update signature verification failed"})
        self.stop.assert_not_called()
        self.assertEqual(installer.validate(target), previous)
        self.assertTrue((self.workspace / "result.json").is_file())

    def test_managed_cli_records_are_updated_without_changing_the_cli_release(self):
        target = fixture(self.system / "Stark.app", "0.3.0")
        state = self.root / "CLI/lib/stark/managed-app.json"
        state.parent.mkdir(parents=True)
        current = state.parent / "current"
        current.symlink_to("releases/0.3.0")
        record = {
            "schema_version": 2,
            "installation_id": "a" * 32,
            "path": str(target),
            "state_path": str(state),
            "app_id": installer.APP_ID,
            "sha256": installer.validate(target),
        }
        owner = self.system / installer.OWNER_NAME
        state.write_bytes(installer.record_bytes(record))
        owner.write_bytes(state.read_bytes())
        self.assertEqual(self.invoke()[0], 0)
        self.assertEqual(installer.read_record(state), installer.read_record(owner))
        self.assertEqual(
            installer.read_record(state)["sha256"], installer.validate(target)
        )
        self.assertEqual(current.readlink(), Path("releases/0.3.0"))

    def test_launch_failure_restores_old_app_and_retains_fixed_error(self):
        target = fixture(self.system / "Stark.app", "0.3.0")
        previous = installer.validate(target)
        launches = []

        def command(arguments, _message):
            if arguments[0] == "/bin/ps":
                return self.processes
            launches.append(arguments)
            if len(launches) == 1:
                raise installer.AppError("updated app could not launch")
            self.assertEqual(installer.validate(target), previous)
            self.assertEqual(
                json.loads((self.workspace / "result.json").read_text()),
                {"error": "updated app could not launch"},
            )
            return b""

        self.command.side_effect = command
        with patch.object(installer, "running", return_value=False):
            result, output = self.invoke()
        self.assertEqual(result, 1)
        self.assertEqual(output, {"error": "updated app could not launch"})
        self.assertEqual(installer.validate(target), previous)
        self.assertEqual(len(launches), 2)

    def test_first_install_launch_failure_removes_new_app_without_reopening_it(self):
        def command(arguments, _message):
            if arguments[0] != "/bin/ps":
                raise installer.AppError("updated app could not launch")
            return b""

        self.command.side_effect = command
        self.assertEqual(self.invoke()[0], 1)
        self.assertFalse((self.system / "Stark.app").exists())
        self.assertEqual(
            sum(
                call.args[0][0] == "/usr/bin/open"
                for call in self.command.call_args_list
            ),
            1,
        )

    def test_process_start_failure_restores_previous_app(self):
        target = fixture(self.system / "Stark.app", "0.3.0")
        previous = installer.validate(target)
        self.launch.side_effect = installer.AppError(
            "installed app did not start; previous installation was kept"
        )
        with patch.object(installer, "running", return_value=False):
            result, output = self.invoke()
        self.assertEqual(result, 1)
        self.assertEqual(
            output,
            {"error": "installed app did not start; previous installation was kept"},
        )
        self.assertEqual(installer.validate(target), previous)

    def test_launch_wait_requires_exact_same_user_target_process(self):
        target = self.system / "Stark.app"
        executable = str(target / "Contents/MacOS/Stark")
        with (
            patch.object(
                updater,
                "application_processes",
                side_effect=[
                    [[str(os.getuid() + 1), executable]],
                    [[str(os.getuid()), executable + " other"]],
                    [[str(os.getuid()), executable]],
                ],
            ),
            patch.object(updater.time, "sleep") as sleep,
        ):
            self.wait_for_launch(target)
        self.assertEqual(sleep.call_count, 2)

    def test_launch_wait_times_out_with_fixed_failure(self):
        with (
            patch.object(updater, "application_processes", return_value=[]),
            patch.object(updater.time, "monotonic", side_effect=[0, 11]),
            self.assertRaisesRegex(installer.AppError, "installed app did not start"),
        ):
            self.wait_for_launch(self.system / "Stark.app")

    def test_multiple_running_copies_are_preserved(self):
        first = fixture(self.system / "Stark.app", "0.3.0")
        second = fixture(self.home / "Applications/Stark.app", "0.3.0")
        self.running(first, second)
        result, output = self.invoke()
        self.assertEqual(result, 1)
        self.assertEqual(
            output,
            {
                "error": "multiple Stark copies are running; quit them and retry installation"
            },
        )
        self.stop.assert_not_called()
        self.assertEqual(updater.release_version(first), "0.3.0")
        self.assertEqual(updater.release_version(second), "0.3.0")

    def test_wrong_bundle_identifier_is_preserved(self):
        target = fixture(self.system / "Stark.app", "0.3.0")
        plist = target / "Contents/Info.plist"
        info = plistlib.loads(plist.read_bytes())
        info["CFBundleIdentifier"] = "example.unrelated"
        plist.write_bytes(plistlib.dumps(info))
        result, output = self.invoke()
        self.assertEqual(result, 1)
        self.assertEqual(
            output, {"error": "refusing to replace an unrelated application"}
        )
        self.stop.assert_not_called()
        self.assertEqual(
            plistlib.loads(plist.read_bytes())["CFBundleIdentifier"],
            "example.unrelated",
        )

    def test_unsigned_running_copy_is_preserved(self):
        target = fixture(self.root / "custom/Stark.app", "0.3.0")
        self.running(target)
        self.signature.side_effect = installer.AppError(
            "app signature verification failed"
        )
        self.assertEqual(self.invoke()[0], 1)
        self.stop.assert_not_called()
        self.assertEqual(updater.release_version(target), "0.3.0")

    def test_other_users_and_non_bundle_executables_are_ignored(self):
        self.processes = (
            f"{os.getuid() + 1} /does/not/exist/Stark.app/Contents/MacOS/Stark\n"
            f"{os.getuid()} /does/not/exist/not-Stark.app/Contents/MacOS/Stark\n"
            f"{os.getuid()} /does/not/exist/Stark.app/Contents/MacOS/Stark extra\n"
        ).encode()
        self.assertEqual(updater.running_installations(), [])
        self.signature.assert_not_called()

    def test_target_running_in_another_user_session_is_not_stopped(self):
        target = fixture(self.system / "Stark.app", "0.3.0")
        self.processes = (f"{os.getuid() + 1} {target}/Contents/MacOS/Stark\n").encode()
        result, output = self.invoke()
        self.assertEqual(result, 1)
        self.assertEqual(
            output,
            {
                "error": "Stark is running in another user session; quit it before installation"
            },
        )
        self.stop.assert_not_called()
        self.launch.assert_not_called()

    def test_symlink_target_is_rejected(self):
        actual = fixture(self.root / "elsewhere/Stark.app", "0.3.0")
        (self.system / "Stark.app").symlink_to(actual)
        self.assertEqual(self.invoke()[0], 1)
        self.stop.assert_not_called()
        self.assertTrue((self.system / "Stark.app").is_symlink())

    def test_unwritable_running_target_does_not_fall_back_to_another_copy(self):
        target = fixture(self.root / "read only/Stark.app", "0.3.0")
        self.running(target)
        access = os.access
        with patch.object(
            os,
            "access",
            side_effect=lambda path, mode: (
                False if Path(path) == target.parent else access(path, mode)
            ),
        ):
            result, output = self.invoke()
        self.assertEqual(result, 1)
        self.assertEqual(output, {"error": "installation destination is not writable"})
        self.stop.assert_not_called()
        self.assertFalse((self.system / "Stark.app").exists())

    def test_source_destination_overlap_is_rejected_before_quitting(self):
        self.running(self.candidate)
        result, output = self.invoke()
        self.assertEqual(result, 1)
        self.assertEqual(
            output, {"error": "installation source and destination must not overlap"}
        )
        self.stop.assert_not_called()

    def test_version_format_matches_published_release_format(self):
        plist = self.candidate / "Contents/Info.plist"
        for invalid in (
            "0.3",
            "01.2.3",
            "1.2.3-beta",
            "v1.2.3",
            "1.2.3.4",
            "1.2.9999999999",
            "1.2.٣",
        ):
            info = plistlib.loads(plist.read_bytes())
            info["CFBundleShortVersionString"] = invalid
            plist.write_bytes(plistlib.dumps(info))
            with (
                self.subTest(version=invalid),
                self.assertRaisesRegex(installer.AppError, "version is invalid"),
            ):
                updater.release_version(self.candidate)

    def test_public_workspace_is_rejected_without_writing_into_it(self):
        self.workspace.chmod(0o755)
        result, output = self.invoke()
        self.assertEqual(result, 1)
        self.assertEqual(output, {"error": "installation workspace must be private"})
        self.assertEqual(list(self.workspace.iterdir()), [])
        self.stop.assert_not_called()

    def test_target_changed_during_staging_is_preserved_before_quitting(self):
        target = fixture(self.system / "Stark.app", "0.3.0")

        def verify(app, _version):
            if app != self.candidate:
                (target / "Contents/changed").write_text("concurrent installation")
            return installer.validate(app)

        self.verify.side_effect = verify
        result, output = self.invoke()
        self.assertEqual(result, 1)
        self.assertEqual(output, {"error": "installed app changed during update"})
        self.stop.assert_not_called()
        self.assertTrue((target / "Contents/changed").is_file())

    def test_unexpected_exception_is_content_free(self):
        self.verify.side_effect = RuntimeError("private exception contents")
        result, output = self.invoke()
        self.assertEqual(result, 1)
        self.assertEqual(
            output, {"error": "app update failed; previous installation was kept"}
        )
        self.assertEqual(
            json.loads((self.workspace / "result.json").read_text()), output
        )
        self.stop.assert_not_called()


@unittest.skipUnless(sys.platform == "darwin", "AppKit required")
class NativeDMGHandoffTests(unittest.TestCase):
    def test_real_old_app_quits_and_replaced_app_starts_at_exact_target(self):
        with tempfile.TemporaryDirectory(prefix="stark native handoff ") as temporary:
            root = Path(temporary).resolve()
            target = fixture(root / "installed/Stark.app", "0.3.0")
            candidate = fixture(root / "candidate/Stark.app", "0.4.0")
            workspace = root / "workspace"
            workspace.mkdir(mode=0o700)
            source = root / "main.swift"
            source.write_text(
                "import AppKit\n"
                "class Delegate: NSObject, NSApplicationDelegate {\n"
                "    func applicationDidFinishLaunching(_ note: Notification) {\n"
                "        if CommandLine.arguments.count > 1 {\n"
                '            try! "ready".write(toFile: CommandLine.arguments[1], '
                "atomically: true, encoding: .utf8)\n"
                "        }\n"
                "    }\n"
                "}\n"
                "let app = NSApplication.shared\n"
                "let delegate = Delegate()\n"
                "app.delegate = delegate\n"
                "app.setActivationPolicy(.accessory)\n"
                "app.run()\n"
            )
            binary = root / "fixture-app"
            compiled = subprocess.run(
                ["xcrun", "swiftc", str(source), "-o", str(binary)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(
                compiled.returncode, 0, "native handoff fixture did not compile"
            )
            identifier = "com.starkengine.install-test." + uuid.uuid4().hex
            for app in (target, candidate):
                shutil.copy2(binary, app / "Contents/MacOS/Stark")
                plist = app / "Contents/Info.plist"
                info = plistlib.loads(plist.read_bytes())
                info.update(
                    CFBundleIdentifier=identifier,
                    CFBundlePackageType="APPL",
                    CFBundleName="Stark Installer Test",
                    CFBundleVersion=info["CFBundleShortVersionString"],
                    LSUIElement=True,
                )
                plist.write_bytes(plistlib.dumps(info))
            ready = root / "fixture-ready"
            process = subprocess.Popen(
                [str(target / "Contents/MacOS/Stark"), str(ready)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                updater.wait_for_launch(target)
                deadline = updater.time.monotonic() + 10
                while not ready.exists():
                    self.assertLess(
                        updater.time.monotonic(), deadline, "fixture did not initialize"
                    )
                    updater.time.sleep(0.05)
                with (
                    patch.object(installer, "APP_ID", identifier),
                    patch.object(
                        updater,
                        "verify_release",
                        side_effect=lambda app, _version: installer.validate(app),
                    ),
                    patch.object(
                        updater, "running_installations", return_value=[target]
                    ),
                ):
                    updater.apply_update(candidate, target, "0.4.0", None, workspace)
                    self.assertEqual(process.wait(timeout=10), 0)
                    self.assertEqual(updater.release_version(target), "0.4.0")
                    self.assertTrue(installer.running(target))
                    self.assertEqual(
                        installer.validate(target), installer.validate(candidate)
                    )
            finally:
                installer.stop_app(target)
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)


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
