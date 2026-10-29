"""Small, model-free checks for lossless container alignment and publication."""

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import align_native_safetensors as alignment


def write_container(
    path, tensors, *, start_modulo=7, header_override=None, payload_extra=b""
):
    payload = bytearray()
    header = {}
    for name, dtype, shape, data in tensors:
        begin = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [begin, len(payload)],
        }
    encoded = (
        json.dumps(header, separators=(",", ":")).encode()
        if header_override is None
        else header_override
    )
    while (8 + len(encoded)) % 16 != start_modulo:
        encoded += b" "
    path.write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + payload + payload_extra
    )
    return header, bytes(payload)


def model_fixture(directory):
    directory.mkdir()
    first, _ = write_container(
        directory / "model-00001.safetensors",
        [
            ("layer.weight", "U32", [2], b"12345678"),
        ],
    )
    second, _ = write_container(
        directory / "model-00002.safetensors",
        [
            ("layer.scales", "F16", [2], b"abcd"),
            ("layer.biases", "F16", [2], b"efgh"),
        ],
        start_modulo=11,
    )
    mapping = {name: "model-00001.safetensors" for name in first}
    mapping.update({name: "model-00002.safetensors" for name in second})
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    (directory / "config.json").write_text('{"test":true}\n')
    (directory / "tokenizer.json").write_text('{"test_tokenizer":true}\n')
    receipt = {
        "schema_version": 1,
        "weight_bytes": sum(
            path.stat().st_size for path in directory.glob("*.safetensors")
        ),
        "native_loading": "unverified",
        "execution": "unverified",
    }
    (directory / "native-source.json").write_text(json.dumps(receipt) + "\n")
    return receipt


class AlignmentTests(unittest.TestCase):
    def test_misaligned_cross_shard_tensors_preserve_payload_metadata_and_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "source"
            original_receipt = model_fixture(model)
            before = {path.name: path.read_bytes() for path in model.iterdir()}
            events = []
            output = base / "derived"
            with patch.object(alignment, "BUFFER_BYTES", 3):
                receipt = alignment.align_model(model, output, events.append)
            self.assertEqual(
                before, {path.name: path.read_bytes() for path in model.iterdir()}
            )
            self.assertTrue(receipt["tensor_payload_bytes_unchanged"])
            self.assertEqual(len(receipt["shards"]), 2)
            for shard in receipt["shards"]:
                source = alignment.inspect_container(model / shard["file"])
                derived = alignment.inspect_container(output / shard["file"])
                self.assertEqual((8 + len(derived.header)) % 16, 0)
                self.assertEqual(json.loads(source.header), json.loads(derived.header))
                original_payload = before[shard["file"]][8 + len(source.header) :]
                derived_payload = (output / shard["file"]).read_bytes()[
                    8 + len(derived.header) :
                ]
                self.assertEqual(original_payload, derived_payload)
                self.assertEqual(
                    hashlib.sha256(original_payload).hexdigest(),
                    shard["payload_sha256"],
                )
            for name in [
                "config.json",
                "tokenizer.json",
                "model.safetensors.index.json",
            ]:
                self.assertEqual(before[name], (output / name).read_bytes())
            self.assertEqual(
                before["native-source.json"],
                (output / "native-source.original.json").read_bytes(),
            )
            derived_receipt = json.loads((output / "native-source.json").read_bytes())
            self.assertEqual(derived_receipt["native_loading"], "unverified")
            self.assertEqual(derived_receipt["execution"], "unverified")
            self.assertEqual(
                derived_receipt["weight_bytes"], original_receipt["weight_bytes"] + 14
            )
            self.assertEqual(
                derived_receipt["storage_alignment"]["receipt_sha256"],
                alignment.hash_file(output / alignment.RECEIPT),
            )
            self.assertEqual(events[-1]["event"], "alignment_complete")
            self.assertFalse(
                any(
                    path.name.startswith(".derived.alignment-")
                    for path in base.iterdir()
                )
            )

    def test_existing_destination_and_atomic_race_are_never_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model_fixture(base / "source")
            destination = base / "existing"
            destination.mkdir()
            inode = destination.stat().st_ino
            with self.assertRaisesRegex(alignment.AlignmentError, "destination_exists"):
                alignment.align_model(base / "source", destination, lambda _: None)
            staging = base / "staging"
            staging.mkdir()
            with self.assertRaises(OSError):
                alignment.publish_new_directory(staging, destination)
            self.assertTrue(staging.is_dir())
            self.assertEqual(destination.stat().st_ino, inode)

    def test_hidden_weight_shards_are_rejected_while_hidden_cache_is_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for index, filename in enumerate([".hidden.safetensors", ".hidden.gguf"]):
                model = base / f"source-{index}"
                model_fixture(model)
                (model / "model.safetensors.index.json").unlink()
                (model / filename).write_bytes(b"hidden weight")
                with self.assertRaisesRegex(
                    alignment.AlignmentError, "hidden_weight_shard"
                ):
                    alignment.align_model(
                        model, base / f"derived-{index}", lambda _: None
                    )
            model = base / "cached-source"
            model_fixture(model)
            (model / ".cache").mkdir()
            (model / ".cache" / "download-state").write_bytes(b"ignored cache")
            alignment.align_model(model, base / "cached-derived", lambda _: None)
            self.assertFalse((base / "cached-derived" / ".cache").exists())

    def test_metadata_copy_and_case_collisions_never_overwrite_a_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            incoming = base / "incoming"
            destination = base / "destination"
            incoming.write_bytes(b"new metadata")
            destination.write_bytes(b"existing metadata")
            with self.assertRaisesRegex(
                alignment.AlignmentError, "metadata_destination_collision"
            ):
                alignment.copy_metadata_file(incoming, destination)
            self.assertEqual(destination.read_bytes(), b"existing metadata")
            model = base / "source"
            model_fixture(model)
            original_iterdir = Path.iterdir
            alternate = model / "CONFIG.JSON"
            if not alternate.exists():
                alternate.write_text("{}")
            entries = list(model.iterdir())
            if not any(path.name == alternate.name for path in entries):
                entries.append(alternate)

            def collision_entries(path):
                return (
                    iter(entries) if path == model.resolve() else original_iterdir(path)
                )

            with (
                patch.object(Path, "iterdir", collision_entries),
                self.assertRaisesRegex(
                    alignment.AlignmentError, "source_destination_name_collision"
                ),
            ):
                alignment.align_model(model, base / "derived", lambda _: None)
            self.assertFalse((base / "derived").exists())
            (model / "Alignment-Receipt.JSON").write_text("{}")
            with self.assertRaisesRegex(
                alignment.AlignmentError, "source_already_has_alignment_receipt"
            ):
                alignment.align_model(model, base / "derived", lambda _: None)

    def test_derived_layout_resets_validation_and_preserves_verified_source_receipt(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "source"
            receipt = model_fixture(model)
            receipt.update(native_loading="verified", execution="verified")
            path = model / "native-source.json"
            path.write_text(json.dumps(receipt) + "\n")
            original_bytes = path.read_bytes()
            output = base / "derived"
            alignment.align_model(model, output, lambda _: None)
            self.assertEqual(path.read_bytes(), original_bytes)
            self.assertEqual(
                (output / "native-source.original.json").read_bytes(), original_bytes
            )
            derived = json.loads((output / "native-source.json").read_bytes())
            self.assertEqual(derived["native_loading"], "unverified")
            self.assertEqual(derived["execution"], "unverified")

    def test_malformed_containers_fail_before_publication(self):
        malformed = [
            b"bad",
            struct.pack("<Q", alignment.MAX_HEADER_BYTES + 1),
        ]
        headers = [
            b'{"x":{"dtype":"U8","shape":[true],"data_offsets":[0,1]}}',
            b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,2]}}',
            b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}',
            b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}',
            b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}',
            b'{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"y":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}',
        ]
        malformed.extend(
            struct.pack("<Q", len(header)) + header + b"ab" for header in headers
        )
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for index, data in enumerate(malformed):
                model = base / str(index)
                model.mkdir()
                (model / "bad.safetensors").write_bytes(data)
                output = base / f"output-{index}"
                with self.assertRaises(alignment.AlignmentError):
                    alignment.align_model(model, output, lambda _: None)
                self.assertFalse(output.exists())

    def test_index_mismatch_duplicate_shards_and_nested_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base / "source"
            model_fixture(model)
            with self.assertRaisesRegex(
                alignment.AlignmentError, "destination_inside_source"
            ):
                alignment.align_model(model, model / "derived", lambda _: None)
            index = model / "model.safetensors.index.json"
            index.write_text('{"weight_map":{}}')
            with self.assertRaisesRegex(
                alignment.AlignmentError, "index_inventory_mismatch"
            ):
                alignment.align_model(model, base / "derived", lambda _: None)
            (model / "model-00003.safetensors").write_bytes(
                (model / "model-00001.safetensors").read_bytes()
            )
            with self.assertRaisesRegex(
                alignment.AlignmentError, "duplicate_tensor_across_shards"
            ):
                alignment.align_model(model, base / "derived", lambda _: None)
            self.assertFalse((base / "derived").exists())

    def test_corrupt_copy_is_detected_and_its_staging_directory_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model_fixture(base / "source")
            verify = alignment.verify_copy

            def corrupt(container, destination, digest, progress):
                with destination.open("r+b") as stream:
                    stream.seek(-1, 2)
                    stream.write(b"!")
                return verify(container, destination, digest, progress)

            with (
                patch.object(alignment, "verify_copy", side_effect=corrupt),
                self.assertRaisesRegex(
                    alignment.AlignmentError, "payload_hash_mismatch"
                ),
            ):
                alignment.align_model(base / "source", base / "derived", lambda _: None)
            self.assertFalse((base / "derived").exists())
            self.assertFalse(
                any(
                    path.name.startswith(".derived.alignment-")
                    for path in base.iterdir()
                )
            )


if __name__ == "__main__":
    unittest.main()
