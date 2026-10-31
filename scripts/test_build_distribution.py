"""Model-free distribution construction and provenance regression checks."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import plistlib
import subprocess
import sys
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
    def test_menu_app_uses_new_runtime_worker_and_external_staging_directory(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            bundle = temporary / "stark"
            (bundle / "libexec").mkdir(parents=True)
            worker = bundle / "libexec/stark-generate"
            worker.write_bytes(b"fresh release worker")
            score_worker = bundle / "libexec/stark-score"
            score_worker.write_bytes(b"fresh release score worker")
            calls = []

            def command_result(command, stage, *_args, **_kwargs):
                self.assertEqual(stage, "menu_bar_build")
                calls.append(command)
                self.assertEqual(command[command.index("--runtime") + 1], str(bundle))
                self.assertEqual(command[command.index("--worker") + 1], str(worker))
                self.assertEqual(
                    command[command.index("--score-worker") + 1], str(score_worker)
                )
                self.assertNotEqual(
                    command[command.index("--score-worker") + 1],
                    str(distribution.ROOT / "target/release/stark-score"),
                )
                self.assertEqual(command[command.index("--version") + 1], "0.1.2")
                output = Path(command[command.index("--output") + 1])
                self.assertFalse(output.is_relative_to(bundle))
                app = output / "Stark.app/Contents"
                app.mkdir(parents=True)
                (app / "payload").write_bytes(b"complete application")
                return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

            with (
                patch.object(distribution, "run", side_effect=command_result),
                patch.object(distribution, "verify_application") as verification,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                distribution.build_application(bundle, temporary, "0.1.2", {})
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                (bundle / "app/Stark.app/Contents/payload").read_bytes(),
                b"complete application",
            )
            verification.assert_called_once_with(bundle / "app/Stark.app", {})

    def test_menu_app_verification_rejects_wrong_identity_before_executing(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            app = Path(temporary_name) / "Stark.app"
            contents = app / "Contents"
            (contents / "MacOS").mkdir(parents=True)
            (contents / "MacOS/Stark").write_bytes(b"synthetic app")
            with (contents / "Info.plist").open("wb") as stream:
                plistlib.dump(
                    {
                        "CFBundleIdentifier": "unrelated.app",
                        "LSMinimumSystemVersion": "26.4",
                    },
                    stream,
                )
            with (
                patch.object(distribution, "run") as command,
                self.assertRaisesRegex(
                    distribution.BuildError, "menu_bar_bundle_metadata_invalid"
                ),
            ):
                distribution.verify_application(app, {})
            command.assert_not_called()

    def test_sysconfig_normalization_is_deterministic_and_uses_relocated_prefix(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            outputs = []
            for stage in ("one", "different-staging-name"):
                managed_prefix = root / stage
                runtime = managed_prefix / "copied"
                configuration = (
                    runtime / "lib" / "python3.13" / "_sysconfigdata__darwin_darwin.py"
                )
                configuration.parent.mkdir(parents=True)
                variables = {
                    "prefix": str(managed_prefix),
                    "LIBDIR": str(managed_prefix / "lib"),
                    "DESTDIRS": str(managed_prefix / "bin")
                    + " "
                    + str(managed_prefix / "lib"),
                    "STATIC": 7,
                }
                configuration.write_text("build_time_vars = " + repr(variables) + "\n")
                distribution.normalize_python_configuration(runtime, managed_prefix)
                outputs.append(configuration.read_bytes())
                spec = importlib.util.spec_from_file_location(
                    "normalized_sysconfig", configuration
                )
                assert spec is not None and spec.loader is not None
                module = importlib.util.module_from_spec(spec)
                with patch.object(sys, "base_prefix", "/relocated bundle/python"):
                    spec.loader.exec_module(module)
                normalized = module.build_time_vars
                self.assertEqual(normalized["prefix"], "/relocated bundle/python")
                self.assertEqual(normalized["LIBDIR"], "/relocated bundle/python/lib")
                self.assertEqual(normalized["STATIC"], 7)
                self.assertNotIn(str(managed_prefix).encode(), outputs[-1])
            self.assertEqual(outputs[0], outputs[1])

    def test_converter_export_excludes_staging_command_header(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            bundle = temporary / "bundle"
            (bundle / "share" / "stark").mkdir(parents=True)
            work = temporary / "work"
            work.mkdir()
            export_commands = []

            def command_result(command, stage, *_args, **_kwargs):
                output = b""
                if stage == "python_download":
                    runtime = work / "managed-python" / "cpython-3.13.13-test" / "bin"
                    runtime.mkdir(parents=True)
                    (runtime / "python3.13").write_bytes(b"synthetic interpreter")
                    configuration = (
                        runtime.parent
                        / "lib"
                        / "python3.13"
                        / "_sysconfigdata__darwin_darwin.py"
                    )
                    configuration.parent.mkdir(parents=True)
                    configuration.write_text(
                        "build_time_vars = "
                        + repr({"prefix": str(runtime.parent)})
                        + "\n"
                    )
                    for metadata_name in ("direct_url.json", "uv_cache.json"):
                        metadata = (
                            configuration.parent
                            / "site-packages"
                            / "pip-1.0.dist-info"
                            / metadata_name
                        )
                        metadata.parent.mkdir(parents=True, exist_ok=True)
                        metadata.write_text("synthetic staging metadata\n")
                if stage == "python_version":
                    output = b"Python 3.13.13\n"
                if stage == "python_download":
                    site_packages = (
                        work
                        / "managed-python"
                        / "cpython-3.13.13-test"
                        / "lib"
                        / "python3.13"
                        / "site-packages"
                    )
                    (site_packages / "pip").mkdir(parents=True, exist_ok=True)
                    (site_packages / "pip" / "__init__.py").write_text("bootstrap\n")
                    (site_packages / "pip-26.1.2.dist-info").mkdir()
                    (site_packages / "pip-26.1.2.dist-info" / "METADATA").write_text(
                        "bootstrap\n"
                    )
                    ensurepip = (
                        work
                        / "managed-python"
                        / "cpython-3.13.13-test"
                        / "lib"
                        / "python3.13"
                        / "ensurepip"
                    )
                    ensurepip.mkdir(parents=True, exist_ok=True)
                    (ensurepip / "__init__.py").write_text("bootstrap\n")
                    converter = site_packages / "stark_convert"
                    converter.mkdir()
                    (converter / "__init__.py").write_text("converter\n")
                if stage == "converter_locked_requirements":
                    export_commands.append(command)
                    destination = Path(command[command.index("--output-file") + 1])
                    destination.write_text(
                        "synthetic-package==1.0 --hash=sha256:test\n"
                    )
                if stage == "converter_wheel":
                    wheels = work / "wheels"
                    wheels.mkdir()
                    (wheels / "stark_convert-0.9.1-py3-none-any.whl").write_bytes(
                        b"synthetic wheel"
                    )
                return subprocess.CompletedProcess(
                    command, 0, stdout=output, stderr=b""
                )

            with (
                patch.object(distribution, "run", side_effect=command_result),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                distribution.build_python(bundle, work, "uv", {})
            self.assertEqual(len(export_commands), 1)
            self.assertIn("--no-header", export_commands[0])
            self.assertIn("--locked", export_commands[0])
            self.assertFalse(
                any((bundle / "libexec" / "python").rglob("direct_url.json"))
            )
            self.assertFalse(
                any((bundle / "libexec" / "python").rglob("uv_cache.json"))
            )
            self.assertFalse(
                (
                    bundle
                    / "libexec"
                    / "python"
                    / "lib"
                    / "python3.13"
                    / "site-packages"
                    / "pip"
                ).exists()
            )
            self.assertFalse(
                any(
                    (
                        bundle
                        / "libexec"
                        / "python"
                        / "lib"
                        / "python3.13"
                        / "site-packages"
                    ).glob("pip-*.dist-info")
                )
            )
            self.assertFalse(
                (
                    bundle / "libexec" / "python" / "lib" / "python3.13" / "ensurepip"
                ).exists()
            )
            self.assertTrue(
                (
                    bundle
                    / "libexec"
                    / "python"
                    / "lib"
                    / "python3.13"
                    / "site-packages"
                    / "stark_convert"
                    / "__init__.py"
                ).exists()
            )
            self.assertEqual(
                (bundle / "share" / "stark" / "converter-requirements.txt").read_text(),
                "synthetic-package==1.0 --hash=sha256:test\n",
            )

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

    def test_relocation_can_repeat_after_adding_the_application(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "stark"
            bundle.mkdir()
            project = root / "tools/converter"
            project.mkdir(parents=True)
            (project / "pyproject.toml").write_text('[project]\nversion = "7.8.9"\n')

            def response(command, stage, *_args, **_kwargs):
                output = {
                    "relocated_launcher_converter_version": b"stark-convert 7.8.9\n",
                    "relocated_launcher_converter_help": b"usage: stark-convert [-h]\n",
                }.get(stage, b"")
                return subprocess.CompletedProcess(
                    command, 0, stdout=output, stderr=b""
                )

            with (
                patch.object(distribution, "ROOT", root),
                patch.object(distribution, "run", side_effect=response),
                patch.object(distribution, "verify_native"),
                patch.object(distribution, "verify_application") as application,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                distribution.verify_relocation(bundle, {})
                (bundle / "app/Stark.app").mkdir(parents=True)
                distribution.verify_relocation(bundle, {})
            self.assertTrue((bundle / "app/Stark.app").is_dir())
            self.assertFalse((root / "relocated bundle/stark").exists())
            application.assert_called_once_with(
                root / "relocated bundle/stark/app/Stark.app", {}
            )

    def test_relocation_preserves_conflicting_or_symlink_staging_paths(self):
        for kind in ("directory", "broken-link", "parent-link"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                bundle = root / "stark"
                bundle.mkdir()
                (bundle / "payload").write_bytes(b"original bundle")
                destination = root / "relocated bundle/stark"
                if kind == "parent-link":
                    outside = root / "outside"
                    outside.mkdir()
                    destination.parent.symlink_to(outside, target_is_directory=True)
                else:
                    destination.parent.mkdir()
                    if kind == "directory":
                        destination.mkdir()
                        (destination / "payload").write_bytes(b"existing staging data")
                    else:
                        destination.symlink_to(root / "missing")
                with (
                    patch.object(distribution, "run") as command,
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaisesRegex(
                        distribution.BuildError,
                        "relocation_(destination_exists|parent_is_symlink)",
                    ),
                ):
                    distribution.verify_relocation(bundle, {})
                command.assert_not_called()
                self.assertEqual((bundle / "payload").read_bytes(), b"original bundle")
                if kind == "directory":
                    self.assertEqual(
                        (destination / "payload").read_bytes(), b"existing staging data"
                    )

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

    def test_locked_objc2_metal_supplemental_notices(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / ".cargo_vcs_info.json").write_text(
                json.dumps(
                    {
                        "git": {"sha1": "7b1abfd750a2cacaea71d6a56ecfb83cb7de560b"},
                        "path_in_vcs": "framework-crates/objc2-metal",
                    }
                )
            )
            paths, provenance = distribution.supplemental_notices(
                {
                    "name": "objc2-metal",
                    "version": "0.3.2",
                    "repository": "https://github.com/madsmtm/objc2",
                },
                source,
            )
            self.assertEqual(
                {path.name for path in paths},
                {"objc2-LICENSE.md", "objc2-LICENSE-MIT.txt", "objc2-LICENSE-ZLIB.txt"},
            )
            self.assertEqual(provenance["license"], "Zlib OR Apache-2.0 OR MIT")

    def test_native_and_vendored_licenses_are_copied_into_distribution_notices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in (
                "LICENSE",
                "LICENSE.ggml-hybrid",
                "LICENSE.qwen38-reference",
                "LICENSE.flint-reference",
                "provenance/UPSTREAM.md",
                "tools/converter/LICENSE.lalamo",
                "vendor/cxx/LICENSE-MIT",
                "vendor/cxx/LICENSE-APACHE",
                "vendor/lily-core/LICENSE",
                "vendor/lily-core/LICENSE.mlx-serve",
                "vendor/lily-core/LICENSE.mlx",
                "vendor/lily-core/NOTICE",
                "vendor/lily-core/UPSTREAM.md",
                "scripts/licenses/distribution/manifest.json",
                "scripts/licenses/distribution/README.md",
                "scripts/licenses/distribution/pytorch-LICENSE",
                "scripts/licenses/distribution/ggml-LICENSE",
                "scripts/licenses/distribution/sleef-LICENSE.txt",
                "scripts/licenses/distribution/sleef-ATTRIBUTION.txt",
                "scripts/licenses/distribution/arm-optimized-routines-LICENSE.txt",
                "scripts/licenses/distribution/arm-optimized-routines-ATTRIBUTION.txt",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{name}\n")
            # Exercise the complete notices shipped with the path dependency;
            # Cargo's registry-license inventory deliberately skips local crates.
            lily_notices = {
                name: (distribution.ROOT / "vendor/lily-core" / name).read_bytes()
                for name in (
                    "LICENSE",
                    "LICENSE.mlx-serve",
                    "LICENSE.mlx",
                    "NOTICE",
                    "UPSTREAM.md",
                )
            }
            for name, contents in lily_notices.items():
                (root / "vendor/lily-core" / name).write_bytes(contents)
            license_path = root / "scripts/licenses/distribution/pytorch-LICENSE"
            ggml_license_path = root / "scripts/licenses/distribution/ggml-LICENSE"
            sleef_license_path = (
                root / "scripts/licenses/distribution/sleef-LICENSE.txt"
            )
            sleef_attribution_path = (
                root / "scripts/licenses/distribution/sleef-ATTRIBUTION.txt"
            )
            arm_license_path = (
                root
                / "scripts/licenses/distribution/arm-optimized-routines-LICENSE.txt"
            )
            arm_attribution_path = (
                root
                / "scripts/licenses/distribution/arm-optimized-routines-ATTRIBUTION.txt"
            )
            manifest_path = root / "scripts/licenses/distribution/manifest.json"
            manifest = {
                "source_adaptations": {
                    "pytorch-bf16-vision-layernorm-moments": {
                        "license_file": "pytorch-LICENSE",
                        "license_sha256": distribution.sha256(license_path),
                        "additional_notices": [
                            {
                                "license_file": "ggml-LICENSE",
                                "license_sha256": distribution.sha256(
                                    ggml_license_path
                                ),
                            },
                            {
                                "license_file": "sleef-LICENSE.txt",
                                "license_sha256": distribution.sha256(
                                    sleef_license_path
                                ),
                            },
                            {
                                "license_file": "sleef-ATTRIBUTION.txt",
                                "license_sha256": distribution.sha256(
                                    sleef_attribution_path
                                ),
                            },
                            {
                                "license_file": "arm-optimized-routines-LICENSE.txt",
                                "license_sha256": distribution.sha256(arm_license_path),
                            },
                            {
                                "license_file": "arm-optimized-routines-ATTRIBUTION.txt",
                                "license_sha256": distribution.sha256(
                                    arm_attribution_path
                                ),
                            },
                        ],
                    }
                }
            }
            manifest_path.write_text(json.dumps(manifest))
            sysroot = root / "rust"
            rust_docs = sysroot / "share/doc/rust"
            (rust_docs / "licenses").mkdir(parents=True)
            (rust_docs / "COPYRIGHT-library.html").write_text("rust library notice")
            score_id = "path+file:///workspace#stark-score@0.1.0"
            generate_id = "path+file:///workspace#stark-generate@0.1.0"
            metadata = {
                "packages": [
                    {"id": score_id, "name": "stark-score", "source": None},
                    {"id": generate_id, "name": "stark-generate", "source": None},
                ],
                "resolve": {
                    "nodes": [
                        {"id": score_id, "deps": []},
                        {"id": generate_id, "deps": []},
                    ]
                },
            }
            bundle = root / "bundle"
            (bundle / "share/stark").mkdir(parents=True)

            def command_result(command, *_args, **_kwargs):
                if command[0] == "rustc":
                    stdout = str(sysroot).encode()
                else:
                    stdout = json.dumps(metadata).encode()
                return subprocess.CompletedProcess(command, 0, stdout=stdout)

            with (
                patch.object(distribution, "ROOT", root),
                patch.object(distribution, "run", side_effect=command_result),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                distribution.dependency_licenses(bundle, "cargo", {})

            self.assertEqual(
                (bundle / "share/stark/licenses/pytorch-LICENSE").read_text(),
                "scripts/licenses/distribution/pytorch-LICENSE\n",
            )
            for name, contents in lily_notices.items():
                self.assertEqual(
                    (
                        bundle / "share/stark/licenses/vendor/lily-core" / name
                    ).read_bytes(),
                    contents,
                )
            lily_notice = (
                bundle / "share/stark/licenses/vendor/lily-core/NOTICE"
            ).read_text()
            for retained_notice in (
                "Copyright 2026 Perplexity AI, Inc.",
                "Copyright 2026 Fabio Greter",
                "MLX and MLX-LM are licensed under the MIT License",
                "Copyright © 2023 Apple Inc.",
                'THE SOFTWARE IS PROVIDED "AS IS"',
            ):
                self.assertIn(retained_notice, lily_notice)
            self.assertIn(
                "e22d58fa7cc488493fb1b5d810a198bd29730aeb",
                (
                    bundle / "share/stark/licenses/vendor/lily-core/UPSTREAM.md"
                ).read_text(),
            )
            for name in (
                "LICENSE.ggml-hybrid",
                "LICENSE.qwen38-reference",
                "LICENSE.flint-reference",
            ):
                self.assertEqual(
                    (bundle / "share/stark/licenses" / name).read_text(), f"{name}\n"
                )
            self.assertEqual(
                (bundle / "share/stark/licenses/ggml-LICENSE").read_text(),
                "scripts/licenses/distribution/ggml-LICENSE\n",
            )
            self.assertEqual(
                (bundle / "share/stark/licenses/sleef-LICENSE.txt").read_text(),
                "scripts/licenses/distribution/sleef-LICENSE.txt\n",
            )
            self.assertEqual(
                (bundle / "share/stark/licenses/sleef-ATTRIBUTION.txt").read_text(),
                "scripts/licenses/distribution/sleef-ATTRIBUTION.txt\n",
            )
            self.assertEqual(
                (
                    bundle / "share/stark/licenses/arm-optimized-routines-LICENSE.txt"
                ).read_text(),
                "scripts/licenses/distribution/arm-optimized-routines-LICENSE.txt\n",
            )
            self.assertEqual(
                (
                    bundle
                    / "share/stark/licenses/arm-optimized-routines-ATTRIBUTION.txt"
                ).read_text(),
                "scripts/licenses/distribution/arm-optimized-routines-ATTRIBUTION.txt\n",
            )
            manifest["source_adaptations"]["pytorch-bf16-vision-layernorm-moments"][
                "license_sha256"
            ] = "0" * 64
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(
                distribution.BuildError, "source_adaptation_license_hash_mismatch"
            ):
                distribution.source_adaptation_notices(
                    manifest_path.parent, "pytorch-bf16-vision-layernorm-moments"
                )

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
            lily_sources = []
            for name in ("LICENSE", "NOTICE", "UPSTREAM.md", "src/lib.rs"):
                source = root / "vendor/lily-core" / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(f"vendored source {name}\n")
                lily_sources.append(source)
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
                    for source in lily_sources:
                        self.assertEqual(
                            receipt["files_sha256"][
                                source.relative_to(root).as_posix()
                            ],
                            distribution.sha256(source),
                        )
                    self.assertNotIn("unrelated.js", receipt["files_sha256"])
                    with self.assertRaisesRegex(
                        distribution.BuildError, "source_tree_has_changes"
                    ):
                        distribution.source_provenance(environment, allow_dirty=False)
                    extra.unlink()
                    for source in lily_sources:
                        source.unlink()
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
