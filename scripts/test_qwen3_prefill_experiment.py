"""Model-free checks for ordering, numerical hashes and private aggregate outputs."""

import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

import gguf_experiment as common
import qwen3_prefill_experiment as experiment


class PrefillExperimentTests(unittest.TestCase):
    def test_complete_order_cycle_balances_positions(self):
        names = ["baseline", "mlx8", "candidate"]
        orders = experiment.balanced_orders(names, 9, 1729)
        self.assertEqual(orders, experiment.balanced_orders(names, 9, 1729))
        for start in range(0, 9, 3):
            cycle = orders[start : start + 3]
            for position in range(3):
                self.assertEqual({order[position] for order in cycle}, set(names))

    def test_float_hash_matches_native_f32_bit_contract(self):
        values = [0.0, -0.0, 1.25, -3.5]
        self.assertEqual(
            experiment.logits_hash(values),
            hashlib.sha256(struct.pack("<4f", *values)).hexdigest(),
        )
        with self.assertRaises(common.ExperimentFailure):
            experiment.logits_hash([1e100])

    def test_mismatched_score_hash_fails_before_comparison(self):
        response = {
            "logits": [0.0, 1.0],
            "logits_sha256": "0" * 64,
            "input_tokens_sha256": common.digest_ids([2]),
        }
        with self.assertRaisesRegex(common.ExperimentFailure, "score_hash_mismatch"):
            experiment.validate_score(response, [2], 2)

    def test_unknown_metric_fields_never_persist(self):
        response = {
            "metrics": {
                "prefill_seconds": 0.1,
                "state_creation_seconds": 0.01,
                "readout_seconds": 0.02,
                "total_seconds": 0.13,
                "private_unknown_measurement": 0.75,
            }
        }
        result = experiment.row(response, 32, "score")
        self.assertNotIn("private_unknown_measurement", result)
        self.assertEqual(result["prefill_tokens_per_second"], 320)

    def test_binary_variants_and_comparison_arguments(self):
        args, variants = experiment.parse_arguments(
            [
                "--model",
                "baseline=models/baseline",
                "--model",
                "candidate=models/candidate",
                "--binary",
                "candidate=runtime/new-worker",
                "--compare",
                "candidate=baseline",
                "--output",
                str(common.ROOT / "runtime/nonexistent-prefill-test-aggregate.json"),
            ]
        )
        self.assertEqual(args.comparisons, [("candidate", "baseline")])
        self.assertEqual(
            variants["candidate"]["binary"], common.ROOT / "runtime/new-worker"
        )
        with self.assertRaisesRegex(
            common.ExperimentFailure, "invalid_comparison_argument"
        ):
            experiment.parse_arguments(
                [
                    "--model",
                    "baseline=models/baseline",
                    "--model",
                    "candidate=models/candidate",
                    "--compare",
                    "candidate=unknown",
                    "--output",
                    str(
                        common.ROOT / "runtime/nonexistent-prefill-test-aggregate.json"
                    ),
                ]
            )

    def test_context_overflow_and_external_output_are_rejected(self):
        base = [
            "--model",
            "baseline=models/baseline",
            "--model",
            "candidate=models/candidate",
        ]
        with self.assertRaisesRegex(
            common.ExperimentFailure, "invalid_experiment_arguments"
        ):
            experiment.parse_arguments(
                base
                + [
                    "--lengths",
                    "1024",
                    "--context-tokens",
                    "1024",
                    "--output",
                    str(
                        common.ROOT / "runtime/nonexistent-prefill-test-aggregate.json"
                    ),
                ]
            )
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(
                common.ExperimentFailure, "output_must_be_private_runtime"
            ),
        ):
            experiment.parse_arguments(
                base + ["--output", str(Path(directory) / "aggregate.json")]
            )

    def test_candidate_can_compare_against_old_worker_and_mlx(self):
        self.assertEqual(
            experiment.comparison_pairs(
                ["candidate=baseline", "candidate=mlx8"],
                ["baseline", "mlx8", "candidate"],
            ),
            [("candidate", "baseline"), ("candidate", "mlx8")],
        )
        with self.assertRaisesRegex(
            common.ExperimentFailure, "invalid_comparison_argument"
        ):
            experiment.comparison_pairs(
                ["candidate=baseline", "candidate=baseline"], ["baseline", "candidate"]
            )


if __name__ == "__main__":
    unittest.main()
