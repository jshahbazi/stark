"""Model-free evidence/fixture checks for matched dense profiling."""

import json
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dense_profile_mlx import ROOT, prepare, validate_fixture, weights_digest


class DenseProfileTests(unittest.TestCase):
    def test_private_fixture_reuses_exact_input_and_does_not_publish_tokens(self):
        runtime = ROOT / "runtime"
        runtime.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="dense-profile-test-", dir=runtime
        ) as temporary:
            folder = Path(temporary)
            revision = "a" * 40
            source = folder / revision
            source.mkdir()
            (source / "config.json").write_text(
                json.dumps({"vocab_size": 16, "dtype": "bfloat16"})
            )
            (source / "model.safetensors").write_bytes(
                b"public synthetic weight identity"
            )
            cases = folder / "cases.json"
            cases.write_text(
                json.dumps({"cases": [{"tokens": [1, 3, 7], "context_tokens": 1024}]})
            )
            output = folder / "fixture.json"
            report = prepare(
                SimpleNamespace(
                    source=source,
                    revision=revision,
                    repo="test/Fixture",
                    tokens_from=cases,
                    case=0,
                    output=output,
                )
            )
            fixture = json.loads(output.read_text())
            validate_fixture(fixture)
            self.assertEqual(fixture["input_token_ids"], ([1, 3, 7] * 171)[:512])
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertEqual(report["input_tokens"], 512)
            self.assertEqual(report["decode_forward_steps"], 128)
            self.assertNotIn("input_token_ids", report)
            self.assertNotIn("tokens", report)
            self.assertEqual(
                report["source_weights_digest"],
                weights_digest(fixture["source_weights"]),
            )
            bad = {**fixture, "decode_tokens": 127}
            with self.assertRaises(ValueError):
                validate_fixture(bad)
            with self.assertRaises(ValueError):
                validate_fixture(
                    {**fixture, "unreviewed_output": "must not pass through"}
                )

    def test_source_weight_digest_is_order_independent_and_binds_sizes_and_hashes(self):
        records = [
            {"filename": "model-b.safetensors", "bytes": 32, "sha256": "b" * 64},
            {"filename": "model-a.safetensors", "bytes": 16, "sha256": "a" * 64},
        ]
        identity = weights_digest(records)
        self.assertEqual(identity, "6e71b27b56cfdeda6a0c9537533e859386a7d3e7a1bc4a2ade7d95a945c37b0e")
        self.assertEqual(identity, weights_digest(list(reversed(records))))
        changed_size = [records[0], {**records[1], "bytes": 17}]
        changed_hash = [records[0], {**records[1], "sha256": "c" * 64}]
        self.assertNotEqual(identity, weights_digest(changed_size))
        self.assertNotEqual(identity, weights_digest(changed_hash))


if __name__ == "__main__":
    unittest.main()
