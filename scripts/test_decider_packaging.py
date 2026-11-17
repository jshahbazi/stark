"""Model-free application staging checks for the separate native Decider worker."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_build_menubar as fixtures

builder = fixtures.builder
WORKERS = ("stark-generate", "stark-score", "stark-template", "stark-decide")
DESKTOP_HELP = "--worker --score-worker --template-worker --decide-worker --data-dir --attachment-root"


class DeciderApplicationPackagingTests(unittest.TestCase):
    @contextlib.contextmanager
    def staging_fixture(
        self, *, desktop_help=DESKTOP_HELP, decide_architecture="arm64"
    ):
        with tempfile.TemporaryDirectory(prefix="stark-decider-package-") as temporary:
            root = Path(temporary)
            runtime = root / "older-cli"
            fixtures.make_runtime(runtime)
            manifest_path = runtime / builder.RUNTIME_MANIFEST
            base = json.loads(manifest_path.read_text())
            # An older immutable CLI input has its original three workers.
            for name in WORKERS[:-1]:
                path = runtime / "libexec" / name
                path.write_bytes(("old " + name).encode())
                base["files_sha256"]["libexec/" + name] = builder.sha256(path)
            manifest_path.write_text(json.dumps(base))
            workers = root / "current-workers"
            workers.mkdir()
            for name in WORKERS:
                (workers / name).write_bytes(("current " + name).encode())
            options = argparse.Namespace(
                runtime=runtime,
                worker=workers / "stark-generate",
                score_worker=workers / "stark-score",
                template_worker=workers / "stark-template",
                decide_worker=workers / "stark-decide",
                output=root / "output",
                version=None,
                build_number="202611160000000001",
                install=False,
                open=False,
                ad_hoc=True,
            )
            tooling = {"fixture": "source-bound converter"}
            source = {
                "revision": "a" * 40,
                "dirty": True,
                "files_sha256": {"fixture": "b" * 64},
            }
            calls = []

            def run(command, stage):
                calls.append((command, stage))
                if stage.endswith("worker_architecture"):
                    return (
                        decide_architecture
                        if stage == "decide_worker_architecture"
                        else "arm64"
                    )
                if stage == "swift_build":
                    Path(command[-1]).write_bytes(b"synthetic compiled application")
                    return ""
                if stage == "embedded_python_imports":
                    self.assertIn("stark_convert.desktop_decider", command[-1])
                    self.assertIn("DesktopDeciderWorker", command[-1])
                    return json.dumps(tooling)
                if stage == "desktop_entrypoint":
                    return desktop_help
                self.fail("unexpected_staging_command")

            def notices(embedded, _temporary):
                notice = embedded / "share/stark/licenses/current-workers.txt"
                notice.parent.mkdir(parents=True)
                notice.write_bytes(b"source-bound native notices")
                return {
                    "files_sha256": {
                        "share/stark/licenses/current-workers.txt": builder.sha256(
                            notice
                        )
                    }
                }

            def sign(app, _identity, _keychain):
                return json.loads(
                    (app / "Contents/Resources/menu-bar-manifest.json").read_text()
                )

            with (
                patch.object(builder.platform, "system", return_value="Darwin"),
                patch.object(builder.platform, "machine", return_value="arm64"),
                patch.object(builder, "resolve_signing_identity", return_value=None),
                patch.object(builder, "verify_worker") as probes,
                patch.object(builder, "run", side_effect=run),
                patch.object(builder, "source_identity", return_value=source),
                patch.object(
                    builder,
                    "install_openwork_connector",
                    return_value={
                        "resources_sha256": {},
                        "development_override": False,
                    },
                ),
                patch.object(
                    builder, "install_pi_client", return_value={"resources_sha256": {}}
                ),
                patch.object(builder, "install_native_notices", side_effect=notices),
                patch.object(
                    builder,
                    "install_converter",
                    return_value={"files_sha256": {}, "toolchain_provenance": tooling},
                ),
                patch.object(
                    builder,
                    "install_desktop_dependencies",
                    return_value={
                        "files_sha256": {},
                        "versions": {},
                        "requirements_sha256": "c" * 64,
                    },
                ),
                patch.object(
                    builder,
                    "make_icon",
                    side_effect=lambda _art, resources, _temp: (
                        resources / "Stark.icns"
                    ).write_bytes(b"icon"),
                ),
                patch.object(
                    builder, "sign_application", side_effect=sign
                ) as signatures,
                patch.object(builder, "install_app") as installs,
            ):
                yield (
                    options,
                    runtime,
                    workers,
                    source,
                    calls,
                    probes,
                    signatures,
                    installs,
                )

    def test_older_cli_is_immutable_while_all_four_current_workers_are_overlaid(self):
        with self.staging_fixture() as (
            options,
            runtime,
            workers,
            source,
            _calls,
            probes,
            _signatures,
            installs,
        ):
            manifest_before = (runtime / builder.RUNTIME_MANIFEST).read_bytes()
            app = builder.build(options)
            embedded = app / "Contents/Resources/runtime"
            manifest = json.loads(
                (app / "Contents/Resources/menu-bar-manifest.json").read_text()
            )
            self.assertEqual(manifest["source"], source)
            self.assertEqual(
                manifest["base_runtime"]["manifest_sha256"],
                builder.sha256(runtime / builder.RUNTIME_MANIFEST),
            )
            self.assertEqual(
                (runtime / builder.RUNTIME_MANIFEST).read_bytes(), manifest_before
            )
            self.assertFalse((runtime / "libexec/stark-decide").exists())
            self.assertEqual(
                [
                    (
                        call.args[0].name,
                        call.args[1] if len(call.args) > 1 else "generation",
                    )
                    for call in probes.call_args_list
                ],
                [
                    ("stark-generate", "generation"),
                    ("stark-score", "score"),
                    ("stark-decide", "decide"),
                    ("stark-template", "template"),
                ],
            )
            for name in WORKERS:
                self.assertEqual(
                    (embedded / "libexec" / name).read_bytes(),
                    (workers / name).read_bytes(),
                )
                self.assertEqual(
                    manifest["overlays_sha256"]["libexec/" + name],
                    builder.sha256(workers / name),
                )
            for name in WORKERS[:-1]:
                self.assertEqual(
                    (runtime / "libexec" / name).read_bytes(), ("old " + name).encode()
                )
            self.assertEqual(
                manifest["validation"]["decide_worker_invalid_arguments"], "passed"
            )
            installs.assert_not_called()

    def test_old_desktop_help_without_decide_worker_cannot_publish_application(self):
        with self.staging_fixture(
            desktop_help=DESKTOP_HELP.replace("--decide-worker", "")
        ) as (
            options,
            _runtime,
            _workers,
            _source,
            _calls,
            _probes,
            signatures,
            installs,
        ):
            with self.assertRaisesRegex(
                builder.BuildError, "^desktop_entrypoint_contract_failed$"
            ):
                builder.build(options)
            signatures.assert_not_called()
            installs.assert_not_called()
            self.assertFalse((options.output / "Stark.app").exists())

    def test_missing_or_non_arm64_decide_worker_stops_before_staging_or_compilation(
        self,
    ):
        for missing in (True, False):
            with (
                self.subTest(missing=missing),
                self.staging_fixture(decide_architecture="x86_64") as (
                    options,
                    _runtime,
                    _workers,
                    _source,
                    calls,
                    _probes,
                    signatures,
                    installs,
                ),
            ):
                if missing:
                    options.decide_worker.unlink()
                with self.assertRaisesRegex(
                    builder.BuildError, "^arm64_decide_worker_missing$"
                ):
                    builder.build(options)
                self.assertNotIn("swift_build", [stage for _command, stage in calls])
                signatures.assert_not_called()
                installs.assert_not_called()
                self.assertFalse(options.output.exists())

    def test_decide_worker_argument_defaults_and_explicit_override_are_separate(self):
        for arguments, expected in (
            ([], builder.ROOT / "target/release/stark-decide"),
            (
                ["--decide-worker", "/private/fixture-decide"],
                Path("/private/fixture-decide"),
            ),
        ):
            with (
                self.subTest(arguments=arguments),
                patch.object(
                    builder, "build", return_value=Path("/private/staged/Stark.app")
                ) as build,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(builder.main(arguments), 0)
                self.assertEqual(build.call_args.args[0].decide_worker, expected)


if __name__ == "__main__":
    unittest.main()
