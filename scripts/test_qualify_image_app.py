"""Model-free checks for the opt-in packaged image validation harness."""

from __future__ import annotations

import importlib.util
import json
import os
import plistlib
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "qualify_image_app", Path(__file__).with_name("qualify_image_app.py")
)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


class QualificationTests(unittest.TestCase):
    def test_metadata_reads_are_bounded_and_refuse_special_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            regular = root / "metadata"
            regular.write_bytes(b"123456789")
            self.assertEqual(
                qualification.metadata_bytes(regular, 9, "invalid"), b"123456789"
            )
            with self.assertRaises(qualification.QualificationError):
                qualification.metadata_bytes(regular, 8, "invalid")
            alias = root / "alias"
            alias.symlink_to(regular)
            with self.assertRaises(OSError):
                qualification.metadata_bytes(alias, 9, "invalid")
            fifo = root / "fifo"
            os.mkfifo(fifo)
            with self.assertRaises(qualification.QualificationError):
                qualification.metadata_bytes(fifo, 9, "invalid")

    def test_cli_defaults_to_reference_and_local_mode_is_explicit(self):
        for level in qualification.EVIDENCE_LEVELS:
            with self.subTest(level=level), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                arguments = [
                    "--app",
                    str(root),
                    "--model",
                    str(root),
                    "--output",
                    str(root / "result.json"),
                ]
                if level == "locally_tested":
                    arguments.extend(["--evidence-level", level])
                with (
                    patch.object(
                        qualification, "qualify", return_value={"checks": {}}
                    ) as run,
                    patch("builtins.print"),
                ):
                    self.assertEqual(qualification.main(arguments), 0)
                run.assert_called_once_with(root, root, level)

    def test_readiness_requires_the_requested_evidence_level(self):
        for level in qualification.EVIDENCE_LEVELS:
            ready = {
                "images": True,
                "image_attachment_transport": "private-file-v1",
                "image_verification": level,
            }
            qualification.verify_readiness(ready, level)
            other = next(
                item for item in qualification.EVIDENCE_LEVELS if item != level
            )
            with self.assertRaisesRegex(
                qualification.QualificationError, "image_evidence_level_mismatch"
            ):
                qualification.verify_readiness(ready, other)
        with self.assertRaisesRegex(
            qualification.QualificationError, "packaged_worker_unqualified"
        ):
            qualification.verify_readiness({"images": False}, "locally_tested")

    def test_local_generation_is_bounded_without_a_quality_claim(self):
        valid = {
            "text": "positive",
            "input_tokens": 100,
            "output_tokens": 2,
            "finish_reason": "stop_token",
        }
        qualification.verify_reply(valid, "locally_tested", 2048, 32)
        with self.assertRaisesRegex(
            qualification.QualificationError, "synthetic_shape_mismatch"
        ):
            qualification.verify_reply(valid, "reference_verified", 2048, 32)
        qualification.verify_reply(
            {**valid, "text": "Circle"}, "reference_verified", 2048, 32
        )
        for change in (
            {"text": ""},
            {"text": "   "},
            {"text": "a" * (qualification.MAX_REPLY_BYTES + 1)},
            {"input_tokens": True},
            {"input_tokens": 2048},
            {"output_tokens": 0},
            {"output_tokens": True},
            {"output_tokens": 33},
            {"finish_reason": "cancelled"},
        ):
            with (
                self.subTest(fields=list(change)),
                self.assertRaises(qualification.QualificationError),
            ):
                qualification.verify_reply(
                    {**valid, **change}, "locally_tested", 2048, 32
                )

    def test_schema_three_requires_checking_before_model_publication(self):
        observation = qualification.InstallObservation(3)
        observation.observe({"event": "image_check", "state": "checking"})
        observation.observe({"event": "models", "models": [{"id": "model"}]})
        observation.observe({"event": "installed"})
        self.assertTrue(
            observation.checking and observation.published and observation.installed
        )
        for event in (
            {"event": "models", "models": [{"id": "model"}]},
            {"event": "installed"},
        ):
            with self.assertRaisesRegex(
                qualification.QualificationError, "install_image_check_missing"
            ):
                qualification.InstallObservation(3).observe(event)
        with self.assertRaisesRegex(
            qualification.QualificationError, "image_check_after_model_publication"
        ):
            observation.observe({"event": "image_check", "state": "checking"})
        qualification.InstallObservation(2).observe(
            {"event": "models", "models": [{"id": "model"}]}
        )

    def test_local_receipt_snapshot_detects_replacement_and_rejects_weak_permissions(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = root / ("a" * 64 + ".json")
            payload = json.dumps(
                {
                    "schema_version": 1,
                    "evidence_level": "locally_tested",
                    "identity": {"worker_sha256": "b" * 64},
                    "checks": {key: True for key in qualification.LOCAL_CHECKS},
                }
            )
            receipt.write_text(payload)
            receipt.chmod(0o600)
            before = qualification.receipt_snapshot(root)
            self.assertEqual(before, qualification.receipt_snapshot(root))
            replacement = root / ".replacement"
            replacement.write_text(payload)
            replacement.chmod(0o600)
            replacement.replace(receipt)
            self.assertNotEqual(before, qualification.receipt_snapshot(root))
            receipt.chmod(0o644)
            with self.assertRaisesRegex(
                qualification.QualificationError, "local_receipt_permissions_invalid"
            ):
                qualification.receipt_snapshot(root)

    def test_packaged_qualification_distinguishes_reference_and_family_execution(self):
        for level, schema in (("reference_verified", 2), ("locally_tested", 3)):
            with self.subTest(level=level), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                app, model = root / "Stark.app", root / "model"
                model.mkdir()
                (model / "config.json").write_text(
                    json.dumps(
                        {
                            "type": "Qwen35VisionLanguageModelConfig",
                            "artifact_contract": {"schema_version": schema},
                        }
                    )
                )
                resources = app / "Contents/Resources"
                resources.mkdir(parents=True)
                (app / "Contents/Info.plist").write_bytes(
                    plistlib.dumps({"CFBundleIdentifier": "com.starkengine.menubar"})
                )
                (resources / "menu-bar-manifest.json").write_text("{}")
                runtime = resources / "runtime"
                for path in (
                    runtime / "libexec/stark-generate",
                    app / "Contents/MacOS/Stark",
                    runtime
                    / qualification.PACKAGE
                    / "vision_runtime_qualifications.json",
                ):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"delivered fixture")

                class FakeBackend:
                    def __init__(
                        self,
                        arguments,
                        environment,
                        cwd,
                        evidence=level,
                        version=schema,
                    ):
                        self.level, self.schema = evidence, version
                        self.library = Path(environment["HOME"]) / "model-library"
                        self.attachments = Path(
                            arguments[arguments.index("--attachment-root") + 1]
                        )
                        self.events = [{"event": "models", "models": []}]
                        self.installs = 0
                        self.settings = {
                            "images": True,
                            "image_verification": self.level,
                            "image_attachment_transport": "private-file-v1",
                            "context_tokens": 8192,
                            "max_new_tokens": 512,
                            "max_context_tokens": 16384,
                            "max_output_tokens": 4096,
                        }

                    def send(self, command):
                        kind = command["command"]
                        if kind == "install":
                            self.installs += 1
                            self.library.mkdir(exist_ok=True)
                            if self.schema == 3:
                                self.events.append(
                                    {"event": "image_check", "state": "checking"}
                                )
                            if self.installs == 1:
                                (self.library / "models.json").write_text(
                                    json.dumps(
                                        [
                                            {
                                                "id": "model",
                                                "image_verification": self.level,
                                            }
                                        ]
                                    )
                                )
                                if self.level == "locally_tested":
                                    receipts = self.library / "image-checks"
                                    receipts.mkdir(mode=0o700)
                                    path = receipts / ("a" * 64 + ".json")
                                    path.write_text(
                                        json.dumps(
                                            {
                                                "schema_version": 1,
                                                "evidence_level": self.level,
                                                "identity": {"worker_sha256": "b" * 64},
                                                "checks": {
                                                    key: True
                                                    for key in qualification.LOCAL_CHECKS
                                                },
                                            }
                                        )
                                    )
                                    path.chmod(0o600)
                                self.events.append(
                                    {"event": "models", "models": [{"id": "model"}]}
                                )
                            self.events.extend(
                                [
                                    {"event": "installed", "model": {"id": "model"}},
                                    {
                                        "event": "status",
                                        "message": "Model installed"
                                        if self.installs == 1
                                        else "Model already installed",
                                    },
                                ]
                            )
                        elif kind == "run":
                            self.events.extend(
                                [
                                    {"event": "ready", **self.settings},
                                    {"event": "status", "state": "ready"},
                                ]
                            )
                        elif kind == "configure":
                            self.settings.update(
                                {
                                    key: command[key]
                                    for key in ("context_tokens", "max_new_tokens")
                                }
                            )
                            self.events.append({"event": "configured", **self.settings})
                        elif kind == "generate":
                            (self.attachments / command["image"]["id"]).unlink()
                            self.events.extend(
                                [
                                    {
                                        "event": "reply",
                                        "text": "Circle"
                                        if self.level == "reference_verified"
                                        else "positive",
                                        "input_tokens": 100,
                                        "output_tokens": 2,
                                        "finish_reason": "stop_token",
                                    },
                                    {"event": "status", "state": "ready"},
                                ]
                            )

                    def until(self, predicate, *, observe=None):
                        while self.events:
                            event = self.events.pop(0)
                            if observe is not None:
                                observe(event)
                            if predicate(event):
                                return event
                        raise AssertionError("missing fixture event")

                    def close(self):
                        pass

                def run(arguments, environment, cwd):
                    if "-c" in arguments:
                        inspection = arguments[arguments.index("-c") + 1]
                        self.assertIn("'image_checks'", inspection)
                        self.assertIn("'vision_family'", inspection)
                    return b'{"isolated_package_imports":true}'

                with (
                    patch.object(qualification, "Backend", FakeBackend),
                    patch.object(qualification, "run", run),
                    patch.object(qualification, "verify_native_notices"),
                ):
                    result = qualification.qualify(app, model, level)
                self.assertEqual(result["evidence_level"], level)
                self.assertEqual(result["artifact_schema_version"], schema)
                self.assertEqual(
                    result["checks"]["ordinary_catalog_qualification"],
                    level == "reference_verified",
                )
                self.assertEqual(
                    result["checks"]["synthetic_circle_matches"],
                    2 if level == "reference_verified" else None,
                )
                self.assertEqual(
                    result["checks"]["checking_before_model_publication"],
                    True if schema == 3 else None,
                )
                self.assertEqual(
                    result["checks"]["local_receipt_unchanged_on_reinstall_and_run"],
                    True if schema == 3 else None,
                )
                self.assertNotIn("positive", json.dumps(result))

    def test_packaged_native_notices_are_source_bound_and_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            notices = runtime / "share/stark/licenses"
            notices.mkdir(parents=True)
            hashes = {}
            for name in qualification.ADAPTATION_NOTICES:
                path = notices / name
                path.write_bytes(b"pinned source notice")
                hashes["share/stark/licenses/" + name] = qualification.digest(path)
            path = notices / "cargo-inventory.json"
            path.write_text(
                json.dumps(
                    {
                        "packages": [
                            {"name": name}
                            for name in ("image", "jpeg-decoder", "png", "fdeflate")
                        ]
                    }
                )
            )
            hashes["share/stark/licenses/cargo-inventory.json"] = qualification.digest(
                path
            )
            manifest = {
                "source": {"files_sha256": {"Cargo.lock": "a" * 64}},
                "native_license_notices": {
                    "rust_version": "1.98.1",
                    "cargo_lock_sha256": "a" * 64,
                },
                "signed_overlays_sha256": hashes,
            }
            qualification.verify_native_notices(runtime, manifest)
            (notices / "transformers-LICENSE").write_bytes(b"stale or changed notice")
            with self.assertRaisesRegex(
                qualification.QualificationError, "native_notice_missing_or_changed"
            ):
                qualification.verify_native_notices(runtime, manifest)
            manifest["native_license_notices"]["cargo_lock_sha256"] = "b" * 64
            with self.assertRaisesRegex(
                qualification.QualificationError, "native_notice_source_mismatch"
            ):
                qualification.verify_native_notices(runtime, manifest)

    def test_public_fixture_is_static_rgb_png_with_circle(self):
        data = qualification.circle_png()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        chunks = []
        position = 8
        while position < len(data):
            length = struct.unpack(">I", data[position : position + 4])[0]
            kind = data[position + 4 : position + 8]
            payload = data[position + 8 : position + 8 + length]
            crc = struct.unpack(
                ">I", data[position + 8 + length : position + 12 + length]
            )[0]
            self.assertEqual(crc, zlib.crc32(kind + payload))
            chunks.append((kind, payload))
            position += 12 + length
        self.assertEqual([kind for kind, _ in chunks], [b"IHDR", b"IDAT", b"IEND"])
        self.assertEqual(
            struct.unpack(">IIBBBBB", chunks[0][1]), (256, 256, 8, 2, 0, 0, 0)
        )
        pixels = zlib.decompress(chunks[1][1])
        self.assertEqual(len(pixels), (256 * 3 + 1) * 256)
        self.assertEqual(pixels[1:4], b"\xff\xff\xff")
        center = 128 * (256 * 3 + 1) + 1 + 128 * 3
        self.assertEqual(pixels[center : center + 3], b"\x00\x00\x00")

    def test_bundle_inventory_rejects_symlink_and_detects_resource_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = Path(temporary)
            resource = app / "resource"
            resource.write_bytes(b"before")
            before = qualification.inventory(app)
            resource.write_bytes(b"after")
            self.assertNotEqual(before, qualification.inventory(app))
            (app / "alias").symlink_to(resource)
            with self.assertRaisesRegex(
                qualification.QualificationError, "bundle_entry_invalid"
            ):
                qualification.inventory(app)

    def test_failure_never_prints_exception_dump_or_private_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(
                    qualification,
                    "qualify",
                    side_effect=ValueError("private image path or API key"),
                ),
                patch("builtins.print") as output,
            ):
                result = qualification.main(
                    [
                        "--app",
                        str(root),
                        "--model",
                        str(root),
                        "--output",
                        str(root / "out"),
                    ]
                )
            self.assertEqual(result, 1)
            self.assertEqual(
                json.loads(output.call_args.args[0]),
                {"passed": False, "stage": "qualification_failed"},
            )
            self.assertFalse((root / "out").exists())

    def test_existing_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "result.json"
            output.write_text("source-bound existing evidence")
            with (
                patch.object(qualification, "qualify", return_value={"checks": {}}),
                patch("builtins.print"),
            ):
                result = qualification.main(
                    ["--app", str(root), "--model", str(root), "--output", str(output)]
                )
            self.assertEqual(result, 1)
            self.assertEqual(output.read_text(), "source-bound existing evidence")


if __name__ == "__main__":
    unittest.main()
