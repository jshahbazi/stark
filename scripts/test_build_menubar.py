"""Model-free provenance and safe application installation checks."""

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
    def test_local_builds_never_claim_a_release_patch_version(self):
        for version in (None, "0.11.14", "0.11.14-preview.1", "0.0.0-dev.abc.12"):
            with self.subTest(version=version):
                info = {"CFBundleShortVersionString": "0.11.14"}
                builder.set_application_version(info, version, "20261031234500123456")
                self.assertEqual(info["CFBundleShortVersionString"], "0.0.0")
                self.assertEqual(info["CFBundleVersion"], "20261031234500123456")
                self.assertEqual(info["StarkBuildChannel"], "development")

    def test_local_build_number_defaults_to_microsecond_utc_timestamp(self):
        info = {}
        builder.set_application_version(info, None)
        self.assertRegex(info["CFBundleVersion"], r"^[1-9][0-9]{19}$")

    def test_invalid_local_build_metadata_is_rejected(self):
        with self.assertRaisesRegex(builder.BuildError, "application_version_invalid"):
            builder.set_application_version({}, "../../0.1.2")
        for number in ("", "0", "01", "0.11.14", "-1", "build-1", "1" * 21):
            with (
                self.subTest(build_number=number),
                self.assertRaisesRegex(
                    builder.BuildError, "application_build_number_invalid"
                ),
            ):
                builder.set_application_version({}, None, number)

    def test_combined_runtime_is_pruned_without_recursive_application_embedding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            make_runtime(runtime)
            (runtime / "app/Stark.app").mkdir(parents=True)
            (runtime / "app/Stark.app/payload").write_bytes(b"old application")
            manifest_path = runtime / builder.RUNTIME_MANIFEST
            manifest = json.loads(manifest_path.read_text())
            manifest["files_sha256"]["app/Stark.app/payload"] = builder.sha256(
                runtime / "app/Stark.app/payload"
            )
            manifest_path.write_text(json.dumps(manifest))
            parent_manifest = manifest_path.read_bytes()
            builder.verify_runtime(runtime)
            embedded = root / "new-app/Contents/Resources/runtime"
            excluded = builder.copy_runtime_for_application(runtime, embedded, manifest)
            self.assertEqual(excluded, ["app"])
            self.assertFalse((embedded / "app").exists())
            self.assertEqual(manifest_path.read_bytes(), parent_manifest)
            derived = builder.verify_runtime(embedded)
            self.assertEqual(
                derived["application_embedding"]["parent_manifest_sha256"],
                builder.sha256(manifest_path),
            )
            self.assertNotIn("app/Stark.app/payload", derived["files_sha256"])

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

    def test_score_worker_probe_uses_distinct_failure_code(self):
        invalid = subprocess.CompletedProcess([], 0, b'{"ready":true}\n')
        with (
            patch.object(builder.subprocess, "run", return_value=invalid),
            self.assertRaisesRegex(
                builder.BuildError, "score_worker_self_check_failed"
            ),
        ):
            builder.verify_worker(Path("score-worker"), "score")

    def test_template_worker_probe_checks_native_renderer_protocol(self):
        valid = subprocess.CompletedProcess(
            [], 0, b'{"status":"ok","protocol":"stark-template-v1"}\n'
        )
        with patch.object(builder.subprocess, "run", return_value=valid) as probe:
            builder.verify_worker(Path("template-worker"), "template")
        self.assertEqual(probe.call_args.args[0], ["template-worker", "--self-test"])
        for reply in (
            subprocess.CompletedProcess([], 1, b"E"),
            subprocess.CompletedProcess([], 0, b'{"ready":true}\n'),
        ):
            with (
                self.subTest(reply=reply),
                patch.object(builder.subprocess, "run", return_value=reply),
                self.assertRaisesRegex(
                    builder.BuildError, "template_worker_self_check_failed"
                ),
            ):
                builder.verify_worker(Path("template-worker"), "template")

    def test_missing_score_worker_is_rejected_before_packaging(self):
        with self.assertRaisesRegex(
            builder.BuildError, "score_worker_self_check_failed"
        ):
            builder.verify_worker(Path("/definitely/missing/stark-score"), "score")

    def test_score_build_record_must_match_score_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "stark-score"
            record = root / "score-worker-build.json"
            binary.write_bytes(b"score worker")
            record.write_text(json.dumps({"worker_sha256": "0" * 64}))
            with self.assertRaisesRegex(
                builder.BuildError, "score_worker_build_record_hash_mismatch"
            ):
                builder.verify_worker_build_record(
                    record, binary, "score_worker_build_record_hash_mismatch"
                )

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


class OpenWorkConnectorBundleTests(unittest.TestCase):
    def make_source(self, root: Path) -> Path:
        source = root / "connector"
        source.mkdir()
        package = {
            "name": "@stark/openwork-provider",
            "version": "0.1.0",
            "type": "module",
            "main": "./server.js",
            "exports": {".": "./index.js", "./server": "./server.js"},
        }
        for name in builder.OPENWORK_CONNECTOR_FILES:
            data = (
                json.dumps(package).encode()
                if name == "package.json"
                else f"public fixture: {name}\n".encode()
            )
            (source / name).write_bytes(data)
        return source

    def make_lock(self, root: Path, source: Path, *, pinned: bool = True) -> Path:
        path = root / "connector-lock.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "protocol_version": 1,
                    "repository": builder.OPENWORK_CONNECTOR_REPOSITORY,
                    "revision": "a" * 40 if pinned else None,
                    "files_sha256": {
                        name: builder.sha256(source / name)
                        for name in builder.OPENWORK_CONNECTOR_FILES
                    }
                    if pinned
                    else {},
                }
            )
        )
        return path

    def resources(self, root: Path) -> Path:
        resources = root / "Resources"
        resources.mkdir()
        return resources

    def test_local_pinned_source_embeds_only_allowlist_and_records_resource_hashes(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_source(root)
            lock = self.make_lock(root, source)
            resources = self.resources(root)
            (source / "unrelated-executable").write_bytes(b"not included")
            (source / "unrelated-executable").chmod(0o755)
            metadata = builder.install_openwork_connector(
                resources, source=source, lockfile=lock
            )
            target = resources / "OpenWorkConnector"
            self.assertEqual(
                {path.name for path in target.iterdir()},
                set(builder.OPENWORK_CONNECTOR_FILES),
            )
            self.assertFalse(metadata["development_override"])
            self.assertEqual(metadata["revision"], "a" * 40)
            self.assertEqual(metadata["source_mode"], "local")
            self.assertEqual(metadata["lock_sha256"], builder.sha256(lock))
            for name in builder.OPENWORK_CONNECTOR_FILES:
                self.assertEqual(
                    (target / name).read_bytes(), (source / name).read_bytes()
                )
                self.assertEqual((target / name).stat().st_mode & 0o777, 0o644)
                self.assertEqual(
                    metadata["files_sha256"][name], builder.sha256(target / name)
                )
                self.assertEqual(
                    metadata["resources_sha256"][f"OpenWorkConnector/{name}"],
                    builder.sha256(target / name),
                )

    def test_unpinned_local_source_requires_explicit_development_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_source(root)
            lock = self.make_lock(root, source, pinned=False)
            resources = self.resources(root)
            with self.assertRaisesRegex(
                builder.BuildError, "openwork_connector_lock_unpinned"
            ):
                builder.install_openwork_connector(
                    resources, source=source, lockfile=lock
                )
            with self.assertRaisesRegex(
                builder.BuildError, "openwork_connector_development_source_required"
            ):
                builder.install_openwork_connector(
                    resources, allow_development=True, lockfile=lock
                )
            metadata = builder.install_openwork_connector(
                resources, source=source, allow_development=True, lockfile=lock
            )
            self.assertTrue(metadata["development_override"])
            self.assertIsNone(metadata["revision"])
            self.assertEqual(
                set(metadata["files_sha256"]), set(builder.OPENWORK_CONNECTOR_FILES)
            )

    def test_pinned_hash_mismatch_cannot_be_bypassed_by_development_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_source(root)
            lock = self.make_lock(root, source)
            resources = self.resources(root)
            (source / "server.js").write_text("changed source")
            for development in (False, True):
                with self.assertRaisesRegex(
                    builder.BuildError, "openwork_connector_hash_mismatch"
                ):
                    builder.install_openwork_connector(
                        resources,
                        source=source,
                        allow_development=development,
                        lockfile=lock,
                    )
                self.assertFalse((resources / "OpenWorkConnector").exists())

    def test_rejects_unsafe_local_files_and_source_directory(self):
        for failure in (
            "symlink",
            "hardlink",
            "missing",
            "executable",
            "writable",
            "directory",
            "root-symlink",
        ):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                source = self.make_source(root)
                lock = self.make_lock(root, source)
                resources = self.resources(root)
                selected = source / "server.js"
                if failure == "symlink":
                    selected.unlink()
                    selected.symlink_to(source / "connector.js")
                elif failure == "hardlink":
                    os.link(selected, root / "linked.js")
                elif failure == "missing":
                    selected.unlink()
                elif failure == "executable":
                    selected.chmod(0o755)
                elif failure == "writable":
                    selected.chmod(0o666)
                elif failure == "directory":
                    selected.unlink()
                    selected.mkdir()
                else:
                    alias = root / "alias"
                    alias.symlink_to(source, target_is_directory=True)
                    source = alias
                with self.assertRaisesRegex(
                    builder.BuildError, "openwork_connector_source_invalid"
                ):
                    builder.install_openwork_connector(
                        resources, source=source, lockfile=lock
                    )
                self.assertFalse((resources / "OpenWorkConnector").exists())

    def test_lock_rejects_mutable_revisions_other_hosts_path_escape_and_duplicate_keys(
        self,
    ):
        for failure in (
            "repository",
            "revision",
            "file",
            "protocol",
            "duplicate",
            "symlink",
        ):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                source = self.make_source(root)
                lock = self.make_lock(root, source)
                value = json.loads(lock.read_text())
                if failure == "repository":
                    value["repository"] = "https://example.invalid/connector"
                elif failure == "revision":
                    value["revision"] = "main"
                elif failure == "file":
                    value["files_sha256"]["../outside"] = "0" * 64
                elif failure == "protocol":
                    value["protocol_version"] = True
                elif failure == "duplicate":
                    lock.write_text(
                        lock.read_text().replace(
                            '"schema_version": 1',
                            '"schema_version": 1, "schema_version": 1',
                        )
                    )
                elif failure == "symlink":
                    alias = root / "lock-alias"
                    alias.symlink_to(lock)
                    lock = alias
                if failure not in {"duplicate", "symlink"}:
                    lock.write_text(json.dumps(value))
                with self.assertRaisesRegex(
                    builder.BuildError, "openwork_connector_lock_invalid"
                ):
                    builder.read_openwork_connector_lock(lock)

    def test_file_and_aggregate_size_limits(self):
        for failure in ("file", "total"):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                source = self.make_source(root)
                if failure == "file":
                    (source / "server.js").write_bytes(
                        b"x" * (builder.MAX_CONNECTOR_FILE_BYTES + 1)
                    )
                else:
                    for name in builder.OPENWORK_CONNECTOR_FILES:
                        if name != "package.json":
                            (source / name).write_bytes(
                                b"x" * builder.MAX_CONNECTOR_FILE_BYTES
                            )
                lock = self.make_lock(root, source, pinned=False)
                with self.assertRaisesRegex(
                    builder.BuildError, "openwork_connector_source_invalid"
                ):
                    builder.install_openwork_connector(
                        self.resources(root),
                        source=source,
                        allow_development=True,
                        lockfile=lock,
                    )

    def test_default_download_uses_only_pinned_revision_and_verifies_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_source(root)
            lock = self.make_lock(root, source)
            payload = {
                name: (source / name).read_bytes()
                for name in builder.OPENWORK_CONNECTOR_FILES
            }
            with patch.object(
                builder,
                "_download_openwork_connector_file",
                side_effect=lambda revision, name: payload[name],
            ) as download:
                metadata = builder.install_openwork_connector(
                    self.resources(root), lockfile=lock
                )
            self.assertEqual(download.call_count, len(builder.OPENWORK_CONNECTOR_FILES))
            self.assertTrue(
                all(call.args[0] == "a" * 40 for call in download.call_args_list)
            )
            self.assertEqual(metadata["source_mode"], "download")
            self.assertFalse(metadata["development_override"])

    def test_download_transport_rejects_redirects_wrong_status_and_large_payloads(self):
        class Response(io.BytesIO):
            def __init__(self, payload, *, status=200, url=None, size=None):
                super().__init__(payload)
                self.status = status
                self.url = url
                self.headers = {
                    "Content-Length": str(len(payload) if size is None else size)
                }

            def geturl(self):
                return self.url

        expected = (
            "https://raw.githubusercontent.com/jshahbazi/stark-openwork/"
            + "a" * 40
            + "/server.js"
        )
        for failure in ("ok", "redirect", "status", "size", "body"):
            with self.subTest(failure=failure):
                data = (
                    b"public fixture"
                    if failure != "body"
                    else b"x" * (builder.MAX_CONNECTOR_FILE_BYTES + 1)
                )
                response = Response(
                    data,
                    status=302 if failure == "status" else 200,
                    url="https://example.invalid/redirect"
                    if failure == "redirect"
                    else expected,
                    size=builder.MAX_CONNECTOR_FILE_BYTES + 1
                    if failure == "size"
                    else 1
                    if failure == "body"
                    else None,
                )
                opener = unittest.mock.Mock()
                opener.open.return_value = response
                with patch.object(
                    builder.urllib.request, "build_opener", return_value=opener
                ) as factory:
                    if failure == "ok":
                        self.assertEqual(
                            builder._download_openwork_connector_file(
                                "a" * 40, "server.js"
                            ),
                            data,
                        )
                        self.assertEqual(
                            opener.open.call_args.args[0].full_url, expected
                        )
                        self.assertIsNone(
                            factory.call_args.args[0].redirect_request(
                                None, None, None, None, None, None
                            )
                        )
                    else:
                        with self.assertRaisesRegex(
                            builder.BuildError, "openwork_connector_download_failed"
                        ):
                            builder._download_openwork_connector_file(
                                "a" * 40, "server.js"
                            )
        for revision, name in (("main", "server.js"), ("a" * 40, "../outside")):
            with self.assertRaisesRegex(
                builder.BuildError, "openwork_connector_download_failed"
            ):
                builder._download_openwork_connector_file(revision, name)

    def test_copy_failure_removes_only_its_partial_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_source(root)
            lock = self.make_lock(root, source)
            resources = self.resources(root)
            unrelated = resources / "preserve"
            unrelated.write_text("unchanged")
            with (
                patch.object(Path, "write_bytes", side_effect=OSError),
                self.assertRaisesRegex(
                    builder.BuildError, "openwork_connector_copy_failed"
                ),
            ):
                builder.install_openwork_connector(
                    resources, source=source, lockfile=lock
                )
            self.assertFalse((resources / "OpenWorkConnector").exists())
            self.assertEqual(unrelated.read_text(), "unchanged")

    def test_destination_and_dependency_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_source(root)
            resources = self.resources(root)
            target = resources / "OpenWorkConnector"
            target.mkdir()
            (target / "preserve").write_text("unchanged")
            lock = self.make_lock(root, source)
            with self.assertRaisesRegex(
                builder.BuildError, "openwork_connector_destination_invalid"
            ):
                builder.install_openwork_connector(
                    resources, source=source, lockfile=lock
                )
            self.assertEqual((target / "preserve").read_text(), "unchanged")
            shutil.rmtree(target)
            package_path = source / "package.json"
            package = json.loads(package_path.read_text())
            package["dependencies"] = {"unexpected": "1.0.0"}
            package_path.write_text(json.dumps(package))
            lock = self.make_lock(root, source)
            with self.assertRaisesRegex(
                builder.BuildError, "openwork_connector_package_invalid"
            ):
                builder.install_openwork_connector(
                    resources, source=source, lockfile=lock
                )

    def test_destination_created_by_another_writer_is_not_removed_on_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_source(root)
            lock = self.make_lock(root, source)
            resources = self.resources(root)
            target = resources / "OpenWorkConnector"
            original_mkdir = Path.mkdir

            def raced_mkdir(path, *arguments, **keywords):
                original_mkdir(path, *arguments, **keywords)
                (path / "other-writer").write_text("preserve")
                raise FileExistsError

            with (
                patch.object(Path, "mkdir", raced_mkdir),
                self.assertRaisesRegex(
                    builder.BuildError, "openwork_connector_copy_failed"
                ),
            ):
                builder.install_openwork_connector(
                    resources, source=source, lockfile=lock
                )
            self.assertEqual((target / "other-writer").read_text(), "preserve")

    def test_package_exports_cannot_reference_unbundled_or_parent_paths(self):
        for exports in ("./index.js", {".": "./index.js", "./server": "../outside.js"}):
            with (
                self.subTest(exports=exports),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                source = self.make_source(root)
                package_path = source / "package.json"
                package = json.loads(package_path.read_text())
                package["exports"] = exports
                package_path.write_text(json.dumps(package))
                lock = self.make_lock(root, source)
                with self.assertRaisesRegex(
                    builder.BuildError, "openwork_connector_package_invalid"
                ):
                    builder.install_openwork_connector(
                        self.resources(root), source=source, lockfile=lock
                    )


class NativeNoticeTests(unittest.TestCase):
    def test_current_notices_overlay_and_hashes_preserve_base_notices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            embedded, work = root / "runtime", root / "work"
            work.mkdir()
            notices = embedded / "share/stark/licenses"
            notices.mkdir(parents=True)
            (notices / "base-runtime-notice").write_text("preserve")
            (notices / "pytorch-LICENSE").write_text("stale")
            cargo = root / "cargo"
            cargo.touch()

            def collect(overlay, selected_cargo, environment):
                self.assertEqual(selected_cargo, str(cargo))
                self.assertTrue(environment["PATH"].startswith(str(cargo.parent)))
                target = overlay / "share/stark/licenses"
                target.mkdir()
                (target / "pytorch-LICENSE").write_text("current pinned notice")
                (target / "cargo-inventory.json").write_text('{"packages": []}')

            distribution = {
                "clean_environment": lambda: {"PATH": "/usr/bin"},
                "dependency_licenses": collect,
                "RUST_VERSION": "1.98.1",
                "BuildError": RuntimeError,
            }
            with (
                patch.object(builder.shutil, "which", return_value=str(cargo)),
                patch.object(builder.runpy, "run_path", return_value=distribution),
            ):
                result = builder.install_native_notices(embedded, work)
            self.assertEqual((notices / "base-runtime-notice").read_text(), "preserve")
            self.assertEqual(
                (notices / "pytorch-LICENSE").read_text(), "current pinned notice"
            )
            self.assertEqual(result["rust_version"], "1.98.1")
            self.assertEqual(
                result["cargo_lock_sha256"], builder.sha256(builder.ROOT / "Cargo.lock")
            )
            self.assertEqual(
                result["files_sha256"]["share/stark/licenses/pytorch-LICENSE"],
                builder.sha256(notices / "pytorch-LICENSE"),
            )

    def test_collector_failure_does_not_publish_partial_notices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cargo = root / "cargo"
            cargo.touch()
            distribution = {
                "clean_environment": lambda: {"PATH": "/usr/bin"},
                "dependency_licenses": lambda *_args: (_ for _ in ()).throw(
                    RuntimeError("private failure")
                ),
                "BuildError": RuntimeError,
            }
            with (
                patch.object(builder.shutil, "which", return_value=str(cargo)),
                patch.object(builder.runpy, "run_path", return_value=distribution),
                self.assertRaisesRegex(
                    builder.BuildError, "native_dependency_license_collection_failed"
                ),
            ):
                builder.install_native_notices(root / "runtime", root)
            self.assertFalse((root / "runtime").exists())


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
                "desktop_media.py",
                "chat.py",
                "media.py",
                "vision_runtime.py",
                "image_checks.py",
                "vision_family.py",
                "model_support.py",
                "model_inventory.py",
                "model_catalog.py",
                "model_removal.py",
                "vision_runtime_qualifications.json",
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
                "stark_convert/desktop_media.py",
                "stark_convert/image_checks.py",
                "stark_convert/vision_family.py",
                "stark_convert/model_support.py",
                "stark_convert/model_inventory.py",
                "stark_convert/model_catalog.py",
                "stark_convert/model_removal.py",
                "stark_convert/media.py",
                "stark_convert/vision_runtime_qualifications.json",
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
                [
                    "swiftc",
                    str(swift),
                    str(
                        builder.ROOT / "apps/menu-bar/Sources/APICredentialStore.swift"
                    ),
                    *[
                        str(path)
                        for pattern in (
                            "Desktop*.swift",
                            "OpenWork*.swift",
                            "OpenCode*.swift",
                        )
                        for path in sorted(
                            (builder.ROOT / "apps/menu-bar/Sources").glob(pattern)
                        )
                    ],
                    "-o",
                    str(executable),
                ],
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

    def test_running_app_quits_after_signature_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            app, destination = root / "built/Stark.app", root / "Applications/Stark.app"
            make_app(app)
            make_app(destination)
            events = []
            with (
                patch.object(
                    builder, "run", side_effect=lambda *_: events.append("verified")
                ),
                patch.object(
                    builder, "stop_app", side_effect=lambda *_: events.append("stopped")
                ),
                patch.object(
                    builder,
                    "exchange_paths",
                    side_effect=lambda *_: events.append("exchanged"),
                ),
            ):
                builder.install_app(app, destination)
            self.assertEqual(events, ["verified", "stopped", "exchanged"])
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
