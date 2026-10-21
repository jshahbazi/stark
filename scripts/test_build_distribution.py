"""Model-free distribution construction and provenance regression checks."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "build_distribution", Path(__file__).with_name("build_distribution.py")
)
assert SPEC is not None and SPEC.loader is not None
distribution = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(distribution)


class BundleConstructionTests(unittest.TestCase):
    def test_relocation_checks_user_facing_converter_identity_and_restores_tree(self):
        for version, help_text, expected_error in (
            (b"stark-convert 7.8.9\n", b"usage: stark-convert [-h]\n", None),
            (b"cli.py 7.8.9\n", b"usage: stark-convert [-h]\n", "version_identity"),
            (b"stark-convert 7.8.9\n", b"usage: cli.py [-h]\n", "help_identity"),
        ):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                bundle = root / "stark"
                bundle.mkdir()
                project = root / "tools" / "converter"
                project.mkdir(parents=True)
                (project / "pyproject.toml").write_text(
                    '[project]\nversion = "7.8.9"\n'
                )

                def response(
                    command,
                    stage,
                    *_args,
                    version_output=version,
                    help_output=help_text,
                    **_kwargs,
                ):
                    output = b""
                    if stage == "relocated_launcher_converter_version":
                        output = version_output
                    if stage == "relocated_launcher_converter_help":
                        output = help_output
                    return subprocess.CompletedProcess(
                        command, 0, stdout=output, stderr=b""
                    )

                with (
                    self.subTest(version=version, help_text=help_text),
                    patch.object(distribution, "ROOT", root),
                    patch.object(distribution, "run", side_effect=response),
                    patch.object(distribution, "verify_native"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    if expected_error is None:
                        distribution.verify_relocation(bundle, {})
                    else:
                        with self.assertRaisesRegex(
                            distribution.BuildError, expected_error
                        ):
                            distribution.verify_relocation(bundle, {})
                self.assertTrue(bundle.is_dir())
                self.assertFalse((root / "relocated bundle" / "stark").exists())

    def test_supplemental_notices_require_matching_crate_source_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "scripts" / "licenses" / "distribution"
            directory.mkdir(parents=True)
            notice = directory / "LICENSE-MIT"
            notice.write_text("synthetic test notice\n")
            source = root / "crate"
            source.mkdir()
            (source / ".cargo_vcs_info.json").write_text(
                json.dumps({"git": {"sha1": "a" * 40}, "path_in_vcs": "crates/test"})
            )
            package = {
                "name": "test",
                "version": "1.0.0",
                "repository": "https://example.test/repo",
            }
            supplement = {
                "repository": package["repository"],
                "source_revision": "a" * 40,
                "path_in_vcs": "crates/test",
                "files": {"LICENSE-MIT": distribution.sha256(notice)},
                "reference_revision": "b" * 40,
            }
            manifest = directory / "manifest.json"

            def write_manifest():
                manifest.write_text(
                    json.dumps(
                        {"schema_version": 1, "packages": {"test@1.0.0": supplement}}
                    )
                )

            write_manifest()
            with patch.object(distribution, "ROOT", root):
                paths, recorded = distribution.supplemental_notices(package, source)
                self.assertEqual(paths, [notice])
                self.assertEqual(recorded["reference_revision"], "b" * 40)
                notice.write_text("changed notice\n")
                with self.assertRaisesRegex(distribution.BuildError, "hash_mismatch"):
                    distribution.supplemental_notices(package, source)
                supplement["source_revision"] = "c" * 40
                write_manifest()
                with self.assertRaisesRegex(distribution.BuildError, "source_mismatch"):
                    distribution.supplemental_notices(package, source)
                supplement["source_revision"] = "a" * 40
                supplement["files"] = {"../../../private": "0" * 64}
                write_manifest()
                with self.assertRaisesRegex(distribution.BuildError, "path_invalid"):
                    distribution.supplemental_notices(package, source)

    def test_owned_source_manifest_includes_untracked_code_and_excludes_root_extras(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in distribution.RECIPE_INPUTS:
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text("tracked recipe\n")
            native = root / "crates" / "engine" / "src.rs"
            native.parent.mkdir(parents=True)
            native.write_text("tracked source\n")
            environment = os.environ.copy()
            environment.update(
                GIT_AUTHOR_NAME="Codex",
                GIT_AUTHOR_EMAIL="noreply@github.com",
                GIT_COMMITTER_NAME="Codex",
                GIT_COMMITTER_EMAIL="noreply@github.com",
            )
            for command in (
                ["git", "init", "-q"],
                ["git", "add", "."],
                ["git", "commit", "-q", "-m", "initial"],
            ):
                subprocess.run(
                    command, cwd=root, env=environment, capture_output=True, check=True
                )
            extra = root / "crates" / "engine" / "new.rs"
            extra.write_text("new untracked source\n")
            (root / "unrelated.js").write_text("private unrelated file\n")
            with patch.object(distribution, "ROOT", root):
                # run's default cwd was bound to the real ROOT at definition.
                real_run = distribution.run

                def temporary_run(*args, **kwargs):
                    kwargs["cwd"] = root
                    return real_run(*args, **kwargs)

                with patch.object(distribution, "run", side_effect=temporary_run):
                    receipt = distribution.source_provenance(
                        environment, allow_dirty=True
                    )
                    self.assertTrue(receipt["dirty"])
                    self.assertIn("crates/engine/new.rs", receipt["files_sha256"])
                    self.assertNotIn("unrelated.js", receipt["files_sha256"])
                    with self.assertRaisesRegex(
                        distribution.BuildError, "source_tree_has_changes"
                    ):
                        distribution.source_provenance(environment, allow_dirty=False)
                    extra.unlink()
                    receipt = distribution.source_provenance(
                        environment, allow_dirty=False
                    )
                    self.assertFalse(receipt["dirty"])

    def test_native_builds_use_separate_feature_sets_and_copy_each_executable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = root / "target" / "release"
            release.mkdir(parents=True)
            for name in ("stark-score", "stark-generate"):
                (release / name).write_bytes(name.encode())
            bundle = root / "bundle"
            with (
                patch.object(distribution, "ROOT", root),
                patch.object(distribution, "run") as command,
                patch.object(distribution, "verify_native") as verify,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                distribution.build_native(bundle, "cargo", {})
            calls = [call.args[0] for call in command.call_args_list]
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][-2:], ["--features", "optimized"])
            self.assertNotIn("--features", calls[1])
            self.assertTrue(
                all("--locked" in call and "+1.98.1" in call for call in calls)
            )
            self.assertEqual(verify.call_count, 2)
            self.assertEqual(
                (bundle / "libexec" / "stark-score").read_bytes(), b"stark-score"
            )
            self.assertEqual(
                (bundle / "libexec" / "stark-generate").read_bytes(), b"stark-generate"
            )

    def test_version_cannot_escape_archive_or_install_directory(self):
        for version in (
            "0.1.0",
            "0.1.0-preview.1",
            "1.2.3-build.7",
            "0.1.0-dev.abcdef",
        ):
            self.assertEqual(distribution.validate_version(version), version)
        for version in (
            "",
            "/tmp/x",
            "../1",
            "1/../../x",
            "1 x",
            "-1",
            "1\n",
            "1+build.7",
            "1.2.3+build.7",
            "1.2.3\n",
        ):
            with (
                self.subTest(version=version),
                self.assertRaises(argparse.ArgumentTypeError),
            ):
                distribution.validate_version(version)

    def test_native_requires_arm64_and_only_system_linkage(self):
        def successful(command, *_args, **_kwargs):
            if command[:2] == ["/usr/bin/otool", "-l"]:
                return subprocess.CompletedProcess(
                    command, 0, stdout=b"      minos 26.4\n", stderr=b""
                )
            output = {
                "/usr/bin/lipo": b"arm64\n",
                "/usr/bin/otool": b"binary:\n\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0)\n",
                "/binary": b'{"ready":false,"error":"invalid_arguments"}\n',
            }[command[0]]
            return subprocess.CompletedProcess(command, 0, stdout=output, stderr=b"")

        with patch.object(distribution, "run", side_effect=successful):
            distribution.verify_native(Path("/binary"), {})
        with (
            patch.object(
                distribution,
                "run",
                return_value=subprocess.CompletedProcess([], 0, stdout=b"x86_64\n"),
            ),
            self.assertRaisesRegex(distribution.BuildError, "architecture"),
        ):
            distribution.verify_native(Path("/binary"), {})

        def homebrew_linkage(command, *args, **kwargs):
            if command[0] == "/usr/bin/otool":
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=b"binary:\n\t/opt/homebrew/lib/libx.dylib (compatibility version 1)\n",
                )
            return successful(command, *args, **kwargs)

        with (
            patch.object(distribution, "run", side_effect=homebrew_linkage),
            self.assertRaisesRegex(distribution.BuildError, "non_system_dependency"),
        ):
            distribution.verify_native(Path("/binary"), {})

    def test_native_smoke_rejects_diagnostics_or_unexpected_reply(self):
        for stdout, stderr in (
            (b'{"ready":true}', b""),
            (b"not-json", b""),
            (b'{"ready":false,"error":"invalid_arguments"}', b"native diagnostic"),
        ):
            responses = [
                subprocess.CompletedProcess([], 0, stdout=b"arm64\n", stderr=b""),
                subprocess.CompletedProcess([], 0, stdout=b"binary:\n", stderr=b""),
                subprocess.CompletedProcess(
                    [], 0, stdout=b"      minos 26.4\n", stderr=b""
                ),
                subprocess.CompletedProcess([], 1, stdout=stdout, stderr=stderr),
            ]
            with (
                self.subTest(stdout=stdout, stderr=stderr),
                patch.object(distribution, "run", side_effect=responses),
                self.assertRaisesRegex(
                    distribution.BuildError, "invalid_arguments_reply"
                ),
            ):
                distribution.verify_native(Path("/binary"), {})

    def test_runtime_dereferences_internal_aliases_and_refuses_external_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "python3.13").write_bytes(b"runtime")
            (source / "python").symlink_to("python3.13")
            distribution.copy_runtime(source, root / "copied")
            self.assertEqual((root / "copied" / "python").read_bytes(), b"runtime")
            self.assertFalse((root / "copied" / "python").is_symlink())
            (root / "private-file").write_bytes(b"private")
            (source / "outside").symlink_to(root / "private-file")
            with self.assertRaisesRegex(distribution.BuildError, "external_symlink"):
                distribution.copy_runtime(source, root / "refused")
            self.assertFalse((root / "refused").exists())

    def test_archives_are_deterministic_regular_files_with_neutral_ownership(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "stark"
            (bundle / "bin").mkdir(parents=True)
            program = bundle / "bin" / "stark"
            program.write_text("#!/bin/sh\nexit 0\n")
            program.chmod(0o755)
            first = root / "one.tar.gz"
            second = root / "two.tar.gz"
            first_hash = distribution.create_archive(bundle, first)
            os.utime(program, (1740000000, 1740000000))
            self.assertEqual(distribution.create_archive(bundle, second), first_hash)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                first.with_name("one.tar.gz.sha256").read_text(),
                f"{first_hash}  one.tar.gz\n",
            )
            with tarfile.open(first) as archive:
                for entry in archive:
                    self.assertTrue(entry.isdir() or entry.isfile())
                    self.assertTrue(
                        entry.name == "stark" or entry.name.startswith("stark/")
                    )
                    self.assertEqual((entry.uid, entry.gid, entry.mtime), (0, 0, 0))
                    self.assertFalse(entry.mode & 0o022)

    def test_manifest_hashes_regular_files_and_refuses_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "file").write_bytes(b"value")
            self.assertEqual(
                distribution.file_manifest(root),
                {"file": distribution.sha256(root / "file")},
            )
            (root / "alias").symlink_to("file")
            with self.assertRaisesRegex(distribution.BuildError, "symlink"):
                distribution.file_manifest(root)

    def test_subprocess_failure_does_not_publish_exception_or_content(self):
        result = subprocess.CompletedProcess(
            ["test"], 1, stdout=b"private stdout", stderr=b"private stderr"
        )
        with (
            patch.object(subprocess, "run", return_value=result),
            self.assertRaises(distribution.BuildError) as caught,
        ):
            distribution.run(["test"], "test_stage", {})
        self.assertEqual(str(caught.exception), "test_stage: command_failed (exit 1)")

    def test_environment_removes_python_injection_native_capture_and_hub_token(self):
        injected = {
            "PYTHONPATH": "/bad",
            "PYTHONHOME": "/bad",
            "HF_TOKEN": "secret",
            "STARK_CAPTURE_PATH": "/private",
            "RUSTFLAGS": "-C target-cpu=native",
            "DYLD_INSERT_LIBRARIES": "/bad.dylib",
            "DYLD_LIBRARY_PATH": "/bad",
            "LD_PRELOAD": "/bad.so",
            "PATH": "/bin",
        }
        with patch.dict(os.environ, injected, clear=True):
            environment = distribution.clean_environment()
        for name in injected.keys() - {"PATH"}:
            self.assertNotIn(name, environment)
        self.assertEqual(environment["PATH"], "/bin")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")


if __name__ == "__main__":
    unittest.main()
