"""Model-free checks for numerical comparisons and content-free protocol errors."""

import hashlib
import math
import queue
import unittest

import gguf_experiment as experiment


class LogitComparisonTests(unittest.TestCase):
    def test_identity_and_shift_invariant_distribution(self):
        reference = [float(index) / 8 for index in range(40)]
        identity = experiment.compare_logits(reference, reference)
        self.assertEqual(identity["root_mean_square_error"], 0)
        self.assertEqual(identity["reference_to_candidate_kl"], 0)
        self.assertAlmostEqual(identity["cosine_similarity"], 1)
        self.assertEqual(identity["top_k_overlap"], 1)
        shifted = experiment.compare_logits(
            reference, [value + 3 for value in reference]
        )
        self.assertAlmostEqual(shifted["root_mean_square_error"], 3)
        self.assertAlmostEqual(shifted["reference_to_candidate_kl"], 0, places=12)

    def test_kl_matches_known_two_class_distribution(self):
        reference = [math.log(0.75), math.log(0.25)]
        candidate = [math.log(0.5), math.log(0.5)]
        comparison = experiment.compare_logits(reference, candidate, top_k=2)
        expected = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
        self.assertAlmostEqual(comparison["reference_to_candidate_kl"], expected)

    def test_nonfinite_and_shape_errors_are_sanitized(self):
        for candidate in ([1], [float("nan")] * 32, [float("inf")] * 32):
            with self.assertRaises(experiment.ExperimentFailure) as result:
                experiment.compare_logits([1.0] * 32, candidate)
            self.assertIn(
                str(result.exception), {"logit_shape_mismatch", "invalid_logits"}
            )

    def test_reference_top_logprob_agreement(self):
        logits = [float(index) / 8 for index in range(40)]
        partition = experiment.log_partition(logits)
        top = [
            {"id": index, "logprob": logits[index] - partition}
            for index in reversed(range(8, 40))
        ]
        comparison = experiment.compare_reference_top(top, logits)
        self.assertEqual(comparison["top1_agreement"], 1)
        self.assertEqual(comparison["top32_overlap"], 1)
        self.assertEqual(comparison["reference_top32_logprob_rmse"], 0)
        changed = [dict(row) for row in top]
        changed[0]["logprob"] += 2
        self.assertAlmostEqual(
            experiment.compare_reference_top(changed, logits)[
                "reference_top32_logprob_rmse"
            ],
            math.sqrt(4 / 32),
        )


class ProtocolTests(unittest.TestCase):
    def test_source_server_response_variants(self):
        rows = [
            {"top_logprobs": [{"id": index, "logprob": -1.0} for index in range(32)]}
        ]
        for field in ("completion_probabilities", "probs"):
            self.assertEqual(
                experiment.reference_top_rows({field: rows}), rows[0]["top_logprobs"]
            )
        with self.assertRaises(experiment.ExperimentFailure):
            experiment.reference_top_rows({"probs": []})

    def test_worker_error_never_includes_native_content(self):
        worker = experiment.Worker.__new__(experiment.Worker)
        worker.timeout, worker.lines = 1, queue.Queue()
        worker.lines.put(b'{"error":"sensitive native details"}\n')
        with self.assertRaises(experiment.ExperimentFailure) as result:
            worker.receive()
        self.assertEqual(str(result.exception), "worker_operation_failed")
        worker.lines.put(b"invalid sensitive native details\n")
        with self.assertRaises(experiment.ExperimentFailure) as result:
            worker.receive()
        self.assertEqual(str(result.exception), "invalid_worker_response")

    def test_token_hash_is_u32_little_endian(self):
        self.assertEqual(
            experiment.digest_ids([1, 256]),
            hashlib.sha256(bytes([1, 0, 0, 0, 0, 1, 0, 0])).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
