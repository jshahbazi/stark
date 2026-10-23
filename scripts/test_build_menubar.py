"""Model-free provenance and safe application installation checks."""

from __future__ import annotations

import importlib.util
import json
import plistlib
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "build_menubar", Path(__file__).with_name("build_menubar.py")
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def make_app(path: Path, identifier: str = builder.APP_ID) -> None:
    (path / "Contents").mkdir(parents=True)
    with (path / "Contents/Info.plist").open("wb") as stream:
        plistlib.dump({"CFBundleIdentifier": identifier}, stream)
    (path / "Contents/payload").write_text("complete staged app")


def make_runtime(path: Path) -> None:
    (path / builder.PYTHON).parent.mkdir(parents=True)
    (path / builder.PYTHON).write_text("synthetic interpreter")
    (path / builder.CONVERTER).mkdir(parents=True)
    manifest = path / builder.RUNTIME_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "version": "fixture",
                "files_sha256": {
                    str(builder.PYTHON): builder.sha256(path / builder.PYTHON)
                },
            }
        )
    )


def make_converter_wheel(
    wheel: Path, snapshot: Path, source: dict, omit: str | None = None
) -> None:
    wheel.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in source["source_files_sha256"]:
            if name.startswith("stark_convert/") and name != omit:
                archive.write(snapshot / name, name)
        for name in source["source_inputs"]:
            member = "stark_convert/_build_inputs/" + name
            if member != omit:
                archive.write(snapshot / name, member)
        metadata = f"stark_convert-{source['version']}.dist-info"
        archive.writestr(
            metadata + "/METADATA",
            f"Name: {source['project_name']}\nVersion: {source['version']}\nLicense-File: LICENSE.lalamo\n",
        )
        archive.writestr(metadata + "/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
        archive.writestr(metadata + "/RECORD", "fixture record\n")
        license_path = metadata + "/licenses/LICENSE.lalamo"
        if license_path != omit:
            archive.write(snapshot / "LICENSE.lalamo", license_path)


class RuntimeVerificationTests(unittest.TestCase):
    def test_worker_probe_accepts_only_fixed_model_free_protocol_reply(self):
        valid = subprocess.CompletedProcess(
            [], 1, b'{"ready":false,"error":"invalid_arguments"}\n'
        )
        with patch.object(builder.subprocess, "run", return_value=valid):
            builder.verify_worker(Path("worker"))
        invalid = subprocess.CompletedProcess([], 0, b'{"ready":true}\n')
        with (
            patch.object(builder.subprocess, "run", return_value=invalid),
            self.assertRaisesRegex(
                builder.BuildError, "generation_worker_self_check_failed"
            ),
        ):
            builder.verify_worker(Path("worker"))

    def test_preserves_manifest_and_rejects_modified_base_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            make_runtime(runtime)
            original = (runtime / builder.RUNTIME_MANIFEST).read_bytes()
            self.assertEqual(builder.verify_runtime(runtime)["version"], "fixture")
            self.assertEqual(
                (runtime / builder.RUNTIME_MANIFEST).read_bytes(), original
            )
            (runtime / ".archive-sha256").write_text("a" * 64 + "\n")
            self.assertEqual(builder.verify_runtime(runtime)["version"], "fixture")
            (runtime / "unexpected-private-artifact").write_text("fixture")
            with self.assertRaises(builder.BuildError):
                builder.verify_runtime(runtime)
            (runtime / "unexpected-private-artifact").unlink()
            (runtime / builder.PYTHON).write_text("different runtime")
            with self.assertRaisesRegex(
                builder.BuildError, "runtime_manifest_verification_failed"
            ):
                builder.verify_runtime(runtime)

    def test_rejects_manifest_path_escape_and_runtime_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            make_runtime(runtime)
            manifest = runtime / builder.RUNTIME_MANIFEST
            manifest.write_text(json.dumps({"files_sha256": {"../outside": "x"}}))
            with self.assertRaises(builder.BuildError):
                builder.verify_runtime(runtime)
            make_runtime_path = runtime / builder.PYTHON
            manifest.write_text(
                json.dumps(
                    {
                        "files_sha256": {
                            str(builder.PYTHON): builder.sha256(make_runtime_path)
                        }
                    }
                )
            )
            (runtime / "unexpected-link").symlink_to(make_runtime_path)
            with self.assertRaises(builder.BuildError):
                builder.verify_runtime(runtime)


class DesktopDependencyTests(unittest.TestCase):
    def test_requirements_include_only_locked_renderer_packages_and_wheel_hashes(self):
        requirements, versions = builder.desktop_requirements(
            builder.ROOT / "tools/converter/uv.lock"
        )
        self.assertEqual(set(versions), {"jinja2", "markupsafe"})
        self.assertEqual(versions, {"jinja2": "3.1.6", "markupsafe": "3.0.4"})
        self.assertEqual(len(requirements.splitlines()), 2)
        self.assertIn(
            "jinja2==3.1.6 --hash=sha256:85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67",
            requirements,
        )
        self.assertIn(
            "--hash=sha256:73e77980c7207854f00fc4e71fb1626868d5740ab4012623d55c7a99ad122a72",
            requirements,
        )
        self.assertNotIn(
            "0137fb05990d35f1275a587e9aee6d56da821fc83491a0fb838183be43f66d6d",
            requirements,
        )
        self.assertNotIn("numpy", requirements)

    def test_changed_dependency_closure_fails_instead_of_installing_unlocked_packages(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            lockfile = Path(temporary) / "uv.lock"
            source = (builder.ROOT / "tools/converter/uv.lock").read_text()
            source = source.replace(
                '    { name = "markupsafe" },',
                '    { name = "unreviewed-dependency" },',
                1,
            )
            lockfile.write_text(source)
            with self.assertRaisesRegex(
                builder.BuildError, "desktop_dependency_lock_invalid"
            ):
                builder.desktop_requirements(lockfile)

    def test_installer_requires_hashes_and_records_license_and_package_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            embedded = temporary_path / "resources/runtime"
            work = temporary_path / "work"
            work.mkdir()

            def install(command, stage):
                self.assertEqual(stage, "locked_desktop_dependencies")
                self.assertIn("--no-deps", command)
                self.assertIn("--require-hashes", command)
                self.assertEqual(command[command.index("--only-binary") + 1], ":all:")
                overlay = Path(command[command.index("--target") + 1])
                overlay.mkdir()
                (overlay / ".lock").touch()
                for name, version in {"jinja2": "3.1.6", "markupsafe": "3.0.4"}.items():
                    (overlay / name).mkdir(parents=True)
                    (overlay / name / "__init__.py").write_text("fixture")
                    metadata = overlay / f"{name}-{version}.dist-info"
                    metadata.mkdir()
                    (metadata / "LICENSE.txt").write_text("fixture license")
                    (metadata / "METADATA").write_text("fixture metadata")
                    (metadata / "uv_cache.json").write_text("private build path")
                return ""

            with (
                patch.object(builder.shutil, "which", return_value="/uv"),
                patch.object(builder, "run", side_effect=install),
            ):
                result = builder.install_desktop_dependencies(embedded, work)
            self.assertEqual(len(result["files_sha256"]), 6)
            self.assertTrue(
                any(path.endswith("LICENSE.txt") for path in result["files_sha256"])
            )
            self.assertFalse(
                any(path.endswith("uv_cache.json") for path in result["files_sha256"])
            )
            self.assertFalse(
                any(path.endswith("/.lock") for path in result["files_sha256"])
            )
            self.assertTrue(
                (embedded.parent / "menu-bar-python-requirements.txt").is_file()
            )


class ConverterPackageTests(unittest.TestCase):
    def test_snapshot_uses_only_the_standard_library(self):
        with tempfile.TemporaryDirectory() as temporary:
            code = (
                "import json, runpy, sys; from pathlib import Path; "
                "builder = runpy.run_path(sys.argv[1]); "
                "source = builder['converter_snapshot'](Path(sys.argv[2]), Path(sys.argv[3])); "
                "assert not any(name.split('.')[0] in "
                "{'numpy', 'tokenizers', 'huggingface_hub', 'stark_convert'} for name in sys.modules); "
                "print(json.dumps(source['toolchain_provenance']))"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-c",
                    code,
                    str(builder.ROOT / "scripts/build_menubar.py"),
                    str(builder.ROOT / "tools/converter"),
                    str(Path(temporary) / "source"),
                ],
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                result.returncode, 0, "dependency_free_converter_snapshot_failed"
            )
            self.assertEqual(
                json.loads(result.stdout)["converter_lock_sha256"],
                builder.sha256(builder.ROOT / "tools/converter/uv.lock"),
            )

    def test_python_target_lock_accepts_only_empty_regular_uv_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / ".lock"
            lock.touch()
            builder.remove_python_target_lock(root)
            self.assertFalse(lock.exists())
            lock.write_text("unexpected contents")
            with self.assertRaisesRegex(
                builder.BuildError, "python_target_lock_invalid"
            ):
                builder.remove_python_target_lock(root)
            lock.unlink()
            lock.symlink_to(root / "missing-target")
            with self.assertRaises(builder.BuildError):
                builder.remove_python_target_lock(root)

    def test_snapshot_and_wheel_preserve_current_modules_capsule_and_license(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "source"
            source = builder.converter_snapshot(
                builder.ROOT / "tools/converter", snapshot
            )
            for name in (
                "cli.py",
                "sources.py",
                "streaming.py",
                "progress.py",
                "desktop.py",
                "chat.py",
            ):
                relative = "stark_convert/" + name
                self.assertEqual(
                    source["source_files_sha256"][relative],
                    builder.sha256(builder.ROOT / "tools/converter" / relative),
                )
            self.assertEqual(source["build_system"]["requires"], ["setuptools==83.0.0"])
            self.assertEqual(
                source["toolchain_provenance"]["converter_lock_sha256"],
                builder.sha256(builder.ROOT / "tools/converter/uv.lock"),
            )
            wheel = root / f"stark_convert-{source['version']}-py3-none-any.whl"
            make_converter_wheel(wheel, snapshot, source)
            result = builder.validate_converter_wheel(wheel, source)
            self.assertEqual(result["wheel_sha256"], builder.sha256(wheel))
            self.assertEqual(
                result["toolchain_provenance"], source["toolchain_provenance"]
            )

    def test_wheel_rejects_missing_progress_capsule_license_and_invalid_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "source"
            source = builder.converter_snapshot(
                builder.ROOT / "tools/converter", snapshot
            )
            wheel = root / f"stark_convert-{source['version']}-py3-none-any.whl"
            for missing in (
                "stark_convert/progress.py",
                "stark_convert/_build_inputs/uv.lock",
                f"stark_convert-{source['version']}.dist-info/licenses/LICENSE.lalamo",
            ):
                with self.subTest(missing=missing):
                    make_converter_wheel(wheel, snapshot, source, omit=missing)
                    with self.assertRaisesRegex(
                        builder.BuildError, "converter_wheel_validation_failed"
                    ):
                        builder.validate_converter_wheel(wheel, source)
            wheel.write_bytes(b"not a wheel")
            with self.assertRaises(builder.BuildError):
                builder.validate_converter_wheel(wheel, source)

    def test_install_replaces_stale_package_and_records_full_overlay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            embedded = root / "resources/runtime"
            work = root / "work"
            work.mkdir()
            stale_package = embedded / builder.CONVERTER
            stale_package.mkdir(parents=True)
            (stale_package / "stale.py").write_text("old module")
            stale_metadata = stale_package.parent / "stark_convert-0.0.0.dist-info"
            stale_metadata.mkdir()

            def command_result(command, stage):
                if stage == "current_converter_wheel":
                    self.assertIn("--wheel", command)
                    self.assertEqual(
                        command[command.index("--python") + 1],
                        str(embedded / builder.PYTHON),
                    )
                    snapshot = Path(command[command.index("build") + 1])
                    source = builder.converter_snapshot(
                        builder.ROOT / "tools/converter", root / "expected-source"
                    )
                    wheels = Path(command[command.index("--out-dir") + 1])
                    make_converter_wheel(
                        wheels / f"stark_convert-{source['version']}-py3-none-any.whl",
                        snapshot,
                        source,
                    )
                elif stage == "current_converter_install":
                    self.assertIn("--no-deps", command)
                    overlay = Path(command[command.index("--target") + 1])
                    with zipfile.ZipFile(command[-1]) as archive:
                        archive.extractall(overlay)
                    (overlay / ".lock").touch()
                    (overlay / "bin").mkdir()
                    (overlay / "bin/stark-convert").write_text("staging path")
                    (overlay / "stark_convert" / "__pycache__").mkdir()
                    (
                        overlay / "stark_convert" / "__pycache__/progress.pyc"
                    ).write_bytes(b"bytecode")
                    (next(overlay.glob("*.dist-info")) / "direct_url.json").write_text(
                        "staging path"
                    )
                else:
                    self.fail("unexpected subprocess stage")
                return ""

            with (
                patch.object(builder.shutil, "which", return_value="/uv"),
                patch.object(builder, "run", side_effect=command_result),
            ):
                result = builder.install_converter(embedded, work)
            self.assertFalse((stale_package / "stale.py").exists())
            self.assertFalse(stale_metadata.exists())
            self.assertEqual(
                builder.sha256(stale_package / "progress.py"),
                result["source_files_sha256"]["stark_convert/progress.py"],
            )
            self.assertEqual(
                builder.sha256(stale_package / "_build_inputs/uv.lock"),
                builder.sha256(builder.ROOT / "tools/converter/uv.lock"),
            )
            self.assertTrue(
                any(
                    name.endswith("/licenses/LICENSE.lalamo")
                    for name in result["files_sha256"]
                )
            )
            self.assertFalse(
                any(
                    name.endswith((".pyc", "direct_url.json")) or "/bin/" in name
                    for name in result["files_sha256"]
                )
            )
            self.assertFalse(
                any(name.endswith("/.lock") for name in result["files_sha256"])
            )
            self.assertEqual(
                len(list(stale_package.parent.glob("stark_convert-*.dist-info"))), 1
            )


@unittest.skipUnless(
    sys.platform == "darwin" and shutil.which("swiftc"),
    "native controller regression requires macOS and swiftc",
)
class NativeControllerTests(unittest.TestCase):
    def test_failed_install_requests_finish_progress_and_retry_resets_counts(self):
        source = (builder.ROOT / "apps/menu-bar/Sources/StarkApp.swift").read_text()
        controller, separator, _views = source.partition("struct StatusLine: View {")
        self.assertTrue(separator, "cannot locate the native controller source")
        harness = r"""
extension EngineController {
    func receiveTestEvent(_ event: [String: Any]) {
        let bytes = try! JSONSerialization.data(withJSONObject: event)
        apply(try! JSONSerialization.jsonObject(with: bytes) as! [String: Any])
    }
}

func snapshot(_ engine: EngineController) -> [String: Any] {
    func stage(_ value: InstallProgress) -> [String: Any] {
        ["state": value.state, "completed_bytes": value.completedBytes,
         "total_bytes": value.totalBytes.map { $0 as Any } ?? NSNull(), "fraction": value.fraction]
    }
    return ["status": engine.state, "download": stage(engine.downloadProgress),
            "conversion": stage(engine.conversionProgress)]
}

let remote = EngineController()
remote.install(source: "Qwen/Qwen3-0.6B", kind: "huggingface")
let local = EngineController()
local.install(source: "/unavailable/model", kind: "local")
let retry = EngineController()
retry.receiveTestEvent(["event": "status", "state": "error", "message": "Operation failed"])
for phase in ["download", "conversion"] {
    retry.receiveTestEvent(["event": "progress", "phase": phase, "state": "failed",
                            "completed_bytes": 75, "total_bytes": 100])
}
let beforeReset = snapshot(retry)
retry.resetInstallProgress(kind: "huggingface")
let afterReset = snapshot(retry)
retry.install(source: "Qwen/Qwen3-0.6B", kind: "huggingface")
let result: [String: Any] = ["remote": snapshot(remote), "local": snapshot(local),
                           "before_reset": beforeReset, "after_reset": afterReset,
                           "retry": snapshot(retry)]
print(String(data: try! JSONSerialization.data(withJSONObject: result), encoding: .utf8)!)
"""
        with tempfile.TemporaryDirectory(prefix="stark-controller-test-") as temporary:
            root = Path(temporary)
            swift = root / "main.swift"
            executable = root / "controller-check"
            swift.write_text(controller + harness)
            compiled = subprocess.run(
                ["swiftc", str(swift), "-o", str(executable)],
                capture_output=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(
                compiled.returncode, 0, "native controller harness failed to compile"
            )
            checked = subprocess.run(
                [str(executable)], capture_output=True, timeout=10, check=False
            )
            self.assertEqual(
                checked.returncode, 0, "native controller harness failed to run"
            )
            result = json.loads(checked.stdout)

        for name in ("remote", "local", "retry"):
            with self.subTest(install=name):
                self.assertEqual(result[name]["status"], "error")
                self.assertEqual(result[name]["conversion"]["state"], "failed")
                self.assertEqual(
                    result[name]["download"]["state"],
                    "skipped" if name == "local" else "failed",
                )
        for phase in ("download", "conversion"):
            self.assertEqual(result["before_reset"][phase]["completed_bytes"], 75)
            self.assertEqual(result["before_reset"][phase]["total_bytes"], 100)
            self.assertEqual(
                result["after_reset"][phase],
                {
                    "state": "waiting",
                    "completed_bytes": 0,
                    "total_bytes": None,
                    "fraction": 0,
                },
            )
            self.assertEqual(result["retry"][phase]["completed_bytes"], 0)
            self.assertIsNone(result["retry"][phase]["total_bytes"])


class InstallTests(unittest.TestCase):
    def test_fresh_install_publishes_complete_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app, destination = root / "built/Stark.app", root / "Applications/Stark.app"
            make_app(app)
            with patch.object(builder, "run", return_value=""):
                builder.install_app(app, destination)
            self.assertEqual(
                (destination / "Contents/payload").read_text(), "complete staged app"
            )
            self.assertTrue(app.is_dir())
            self.assertEqual(list(destination.parent.glob(".stark-install-*")), [])

    def test_refuses_unrelated_bundle_and_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app, destination = root / "built/Stark.app", root / "Applications/Stark.app"
            make_app(app)
            make_app(destination, "org.other.app")
            with self.assertRaisesRegex(
                builder.BuildError, "install_destination_is_unrelated"
            ):
                builder.install_app(app, destination)
            link = root / "Stark.app"
            link.symlink_to(destination, target_is_directory=True)
            with self.assertRaisesRegex(
                builder.BuildError, "install_destination_is_symlink"
            ):
                builder.install_app(app, link)

    def test_running_app_is_left_intact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app, destination = root / "built/Stark.app", root / "Applications/Stark.app"
            make_app(app)
            make_app(destination)
            with (
                patch.object(builder, "app_running", return_value=True),
                self.assertRaisesRegex(builder.BuildError, "installed_app_is_running"),
            ):
                builder.install_app(app, destination)
            self.assertEqual(
                (destination / "Contents/payload").read_text(), "complete staged app"
            )

    def test_failed_staging_or_atomic_exchange_keeps_previous_app(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app, destination = root / "built/Stark.app", root / "Applications/Stark.app"
            make_app(app)
            make_app(destination)
            (destination / "Contents/payload").write_text("previous app")
            with (
                patch.object(builder, "app_running", return_value=False),
                patch.object(
                    builder,
                    "run",
                    side_effect=builder.BuildError("staged_signature_failed"),
                ),
                self.assertRaises(builder.BuildError),
            ):
                builder.install_app(app, destination)
            self.assertEqual(
                (destination / "Contents/payload").read_text(), "previous app"
            )
            with (
                patch.object(builder, "app_running", return_value=False),
                patch.object(builder, "run", return_value=""),
                patch.object(
                    builder,
                    "exchange_paths",
                    side_effect=builder.BuildError("app_atomic_exchange_failed"),
                ),
                self.assertRaises(builder.BuildError),
            ):
                builder.install_app(app, destination)
            self.assertEqual(
                (destination / "Contents/payload").read_text(), "previous app"
            )
            self.assertEqual(list(destination.parent.glob(".stark-install-*")), [])


if __name__ == "__main__":
    unittest.main()
