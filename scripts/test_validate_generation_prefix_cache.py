"""Model-free regression checks for the model onboarding qualification harness."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

_spec = importlib.util.spec_from_file_location(
    "prefix_qualification",
    Path(__file__).with_name("validate_generation_prefix_cache.py"),
)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


class PrefixQualificationTests(unittest.TestCase):
    def test_native_weight_digest_binds_gguf_and_external_ngram_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "weights-00001.gguf").write_bytes(b"synthetic-weight-one")
            first = harness.weights_sha256(model)
            (model / "weights-00002.gguf").write_bytes(b"synthetic-weight-two")
            second = harness.weights_sha256(model)
            self.assertNotEqual(first, second)
            (model / "ngram_table.bin").write_bytes(b"synthetic-ngram-one")
            third = harness.weights_sha256(model)
            self.assertNotEqual(second, third)
            (model / "ngram_table.bin").write_bytes(b"synthetic-ngram-two")
            self.assertNotEqual(third, harness.weights_sha256(model))

    def test_missing_weights_cannot_produce_an_empty_manifest_receipt(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(SystemExit, "no supported weight files"),
        ):
            harness.weights_sha256(Path(directory))

    def test_identical_errors_cannot_qualify_reuse(self):
        exactness = {
            "checks": [
                {
                    "successful": False,
                    "identical_to_fresh": True,
                    "identical_to_baseline_worker": True,
                    "reuse_matches_rule": True,
                }
            ],
            "invalidations": [{"as_documented": True}],
            "follow_ups_reuse": True,
        }
        self.assertFalse(harness.exactness_passed(exactness))
        exactness["checks"][0]["successful"] = True
        self.assertTrue(harness.exactness_passed(exactness))
        exactness["checks"] = []
        self.assertFalse(harness.exactness_passed(exactness))

    def test_boundary_snapshot_requires_two_remaining_tokens_and_matching_key(self):
        expected = harness.Expectation(1024, boundary_only=True)
        ids = list(range(1100))
        key = ("synthetic-scope", 4096, False)
        expected.record(ids, key, {})
        self.assertEqual(expected.predict(ids[:1025], key), 0)
        self.assertEqual(expected.predict(ids[:1026], key), 1024)
        self.assertEqual(expected.predict(ids, (key[0], 4095, False)), 0)
        self.assertEqual(expected.predict(ids, (key[0], 4096, True)), 0)
        ids[512] += 1
        self.assertEqual(expected.predict(ids, key), 0)

    def test_timing_mismatch_or_missing_warm_reuse_fails_qualification(self):
        def turn(reused):
            return {
                "outputs_identical": True,
                "baseline": {"reused_input_tokens": [0]},
                "prefix_cache": {"reused_input_tokens": [reused]},
            }

        timing = {"turns": [turn(0), turn(1024)]}
        self.assertTrue(harness.timing_passed(timing))
        timing["turns"][1]["outputs_identical"] = False
        self.assertFalse(harness.timing_passed(timing))
        timing["turns"][1] = turn(0)
        self.assertFalse(harness.timing_passed(timing))

    def test_divergence_changes_tokens_even_when_neighbors_are_equal(self):
        ids = [1, 1, 1, 2]
        changed = harness.diverge(ids, 1)
        self.assertEqual(changed, [1, 2, 1, 2])
        self.assertEqual(ids, [1, 1, 1, 2])

    def test_template_preflight_rejects_unsupported_completed_tool_cycles(self):
        class RejectCompletedToolCycle:
            def render(self, **arguments):
                if (
                    sum(message["role"] == "tool" for message in arguments["messages"])
                    > 1
                ):
                    raise ValueError("synthetic rejection")
                return "synthetic rendering"

        args = SimpleNamespace(
            seed=11,
            mode="exactness",
            tools=2,
            instruction_paragraphs=2,
            follow_up_sentences=(4, 3, 8),
            progress=False,
        )
        with self.assertRaisesRegex(ValueError, "synthetic rejection"):
            harness.preflight_sessions(args, RejectCompletedToolCycle(), {})


if __name__ == "__main__":
    unittest.main()
