"""Model-free provenance and safe application installation checks."""

from __future__ import annotations

import importlib.util
import json
import plistlib
import subprocess
import tempfile
import unittest
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
            self.assertTrue(
                (embedded.parent / "menu-bar-python-requirements.txt").is_file()
            )


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
