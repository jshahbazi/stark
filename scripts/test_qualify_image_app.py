"""Model-free checks for the opt-in packaged image validation harness."""

from __future__ import annotations

import importlib.util
import json
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
