"""Model-free checks for the candidate/source correctness validator."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gguf_experiment as common
import qwen3_prefill_correctness as correctness
import qwen3_prefill_experiment as prefill


class CorrectnessTests(unittest.TestCase):
    def response(self, logits, generation_hash="a" * 64):
        return {
            "logits": logits,
            "logits_sha256": prefill.logits_hash(logits),
            "generation_sha256": generation_hash,
        }

    def test_exact_and_rounded_candidates_are_distinguished_without_raw_values(self):
        logits = [float(index) / 8 for index in range(40)]
        outputs = {
            "original": [self.response(logits)],
            "preserved": [self.response(logits)],
            "rounded": [self.response([value + 0.125 for value in logits], "b" * 64)],
        }
        aggregate = correctness.comparisons(outputs, "original")
        self.assertEqual(
            aggregate["preserved_versus_original"]["exact_logit_hash_agreements"], 1
        )
        self.assertEqual(
            aggregate["preserved_versus_original"]["exact_generation_hash_agreements"],
            1,
        )
        changed = aggregate["rounded_versus_original"]
        self.assertEqual(changed["exact_logit_hash_agreements"], 0)
        self.assertEqual(changed["exact_generation_hash_agreements"], 0)
        self.assertAlmostEqual(
            changed["logits"]["root_mean_square_error"]["median"], 0.125
        )
        self.assertAlmostEqual(
            changed["logits"]["reference_to_candidate_kl"]["median"], 0, places=12
        )
        self.assertNotIn(json.dumps(logits), json.dumps(aggregate))
        self.assertNotIn("generation_sha256", json.dumps(aggregate))

    def test_generation_checks_input_binding_and_first_token(self):
        response = {
            "generated_tokens": 32,
            "generated_tokens_sha256": "a" * 64,
            "input_tokens_sha256": common.digest_ids([1, 2]),
            "first_token_id": 2,
        }
        self.assertEqual(
            correctness.validate_generation(response, [1, 2], 32, [0, 1, 2]), 1
        )
        self.assertEqual(
            correctness.validate_generation(response, [1, 2], 32, [2, 1, 0]), 0
        )
        with self.assertRaisesRegex(
            common.ExperimentFailure, "invalid_generation_response"
        ):
            correctness.validate_generation(response, [2, 1], 32, [0, 1, 2])
        response["first_token_id"] = True
        with self.assertRaisesRegex(
            common.ExperimentFailure, "invalid_generation_response"
        ):
            correctness.validate_generation(response, [1, 2], 32, [0, 1, 2])

    def test_cli_requires_complete_source_reference_and_known_baseline(self):
        base = [
            "--model",
            "models/test",
            "--binary",
            "original=runtime/original-worker",
            "--binary",
            "candidate=runtime/candidate-worker",
            "--output",
            str(common.ROOT / "runtime/nonexistent-correctness-test.json"),
        ]
        args, binaries, sources = correctness.parse_arguments(base)
        self.assertEqual(args.baseline, "original")
        self.assertEqual(len(binaries), 2)
        self.assertEqual(sources, {})
        with self.assertRaisesRegex(
            common.ExperimentFailure, "incomplete_reference_arguments"
        ):
            correctness.parse_arguments(base + ["--llama-server", "/tmp/llama-server"])
        with self.assertRaisesRegex(common.ExperimentFailure, "unknown_baseline"):
            correctness.parse_arguments(base + ["--baseline", "unknown"])

    def test_private_output_and_duplicate_binary_labels_are_enforced(self):
        base = [
            "--model",
            "models/test",
            "--binary",
            "original=runtime/original-worker",
        ]
        with self.assertRaisesRegex(
            common.ExperimentFailure, "invalid_binary_argument"
        ):
            correctness.parse_arguments(
                base
                + [
                    "--binary",
                    "original=runtime/second-worker",
                    "--output",
                    "/tmp/test.json",
                ]
            )
        with self.assertRaisesRegex(
            common.ExperimentFailure, "output_must_be_private_runtime"
        ):
            correctness.parse_arguments(
                base
                + [
                    "--binary",
                    "candidate=runtime/candidate-worker",
                    "--output",
                    "/tmp/nonexistent-correctness-output.json",
                ]
            )

    def test_workers_close_serially_and_private_values_stay_out_of_report(self):
        events = []
        logits = [0.0] * 151936
        logits[2] = 1.0
        score = {
            "logits": logits,
            "logits_sha256": prefill.logits_hash(logits),
            "input_tokens_sha256": common.digest_ids([1, 2]),
        }

        class FakeWorker:
            def __init__(self, binary, model, timeout):
                self.name = binary
                self.ready = {"vocab_size": 151936}
                events.append((self.name, "open"))

            def request(self, value):
                if value["operation"] == "tokenize":
                    return {
                        "token_ids": [1, 2],
                        "token_ids_sha256": common.digest_ids([1, 2]),
                    }
                if value["operation"] == "score":
                    return score
                return {
                    "generated_tokens": 32,
                    "generated_tokens_sha256": "a" * 64,
                    "input_tokens_sha256": common.digest_ids([1, 2]),
                    "first_token_id": 2,
                }

            def close(self):
                events.append((self.name, "close"))

        args = SimpleNamespace(
            model="model", timeout=1, output_tokens=32, context_tokens=1024, repeats=2
        )
        report = {"variants": {}}
        with patch.object(common, "Worker", FakeWorker), patch("builtins.print"):
            tokens, outputs = correctness.observe_variants(
                args,
                {"original": "old-worker", "candidate": "new-worker"},
                [{"text": "PRIVATE_SENTINEL", "limit": None}],
                report,
            )
        self.assertEqual(
            events,
            [
                ("old-worker", "open"),
                ("old-worker", "close"),
                ("new-worker", "open"),
                ("new-worker", "close"),
            ],
        )
        self.assertTrue(report["stark_workers_closed_before_reference"])
        self.assertEqual(tokens, [[1, 2]])
        self.assertIn("logits", outputs["original"][0])
        serialized = json.dumps(report)
        self.assertNotIn("PRIVATE_SENTINEL", serialized)
        self.assertNotIn("first_token_id", serialized)
        self.assertNotIn('"logits":', serialized)
        self.assertTrue(report["variants"]["candidate"]["same_model_score_repeatable"])


if __name__ == "__main__":
    unittest.main()
