"""Model-free checks for preserved reference evidence and source verification."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from gemma3_image_reference import REVISION, validate_resume


class ResumeReferenceTests(unittest.TestCase):
    def setup_fixture(self, root):
        source, previous, output = (root / name for name in ("source", "previous", "output"))
        for directory in (source, previous, output):
            directory.mkdir()
        records = []
        for name in ("config.json", "model.safetensors"):
            payload = ("bounded source fixture " + name).encode()
            (source / name).write_bytes(payload)
            records.append({"repo": "google/gemma-3-4b-it", "revision": REVISION, "filename": name,
                "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
        manifest = root / "source-manifest.json"
        manifest.write_text(json.dumps({"primary_repo": "google/gemma-3-4b-it", "primary_revision": REVISION,
                                       "files": records}))
        for name in ("red", "blue", "pattern"):
            suffixes = (".png", "-patches.f32", "-features.f32")
            if name != "pattern":
                suffixes += ("-logits.f32", "-tokens.json")
            for suffix in suffixes:
                (previous / (name + suffix)).write_bytes((name + suffix).encode())
        return source, previous, manifest, output

    def test_resume_freezes_all_previous_inputs_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, previous, manifest, output = self.setup_fixture(Path(temporary))
            original = {path.name: path.read_bytes() for path in previous.iterdir()}
            selected = validate_resume(source, previous, manifest, output)
            self.assertEqual(set(selected), set(original))
            self.assertEqual(original, {path.name: path.read_bytes() for path in previous.iterdir()})
            receipt = json.loads((output / "resume-inputs.json").read_text())
            self.assertEqual(receipt["completed_cases"], ["red", "blue"])
            self.assertEqual(receipt["missing_case"], "pattern")
            self.assertEqual(len(receipt["preserved_fixture_hashes"]), 13)
            self.assertEqual((output / "resume-inputs.json").stat().st_mode & 0o777, 0o600)

    def test_source_payload_change_is_rejected_before_resuming(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, previous, manifest, output = self.setup_fixture(Path(temporary))
            (source / "model.safetensors").write_bytes(b"changed source")
            with self.assertRaisesRegex(ValueError, "resume_source_payload_changed"):
                validate_resume(source, previous, manifest, output)
            self.assertFalse((output / "resume-inputs.json").exists())

    def test_incomplete_or_unknown_prior_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, previous, manifest, output = self.setup_fixture(Path(temporary))
            (previous / "pattern-logits.f32").write_bytes(b"partial interrupted output")
            with self.assertRaisesRegex(ValueError, "resume_fixture_inventory_changed"):
                validate_resume(source, previous, manifest, output)


if __name__ == "__main__":
    unittest.main()
