"""Estimator and fail-closed acceptance checks using scalar-only synthetic data."""
import copy
import math
import unittest

from summarize import ENGINES, summarize


def fixture_receipt():
    fixtures = [{"name": f"p{i}", "population": "performance", "input_tokens": 42, "output_limit": 128}
                for i in range(6)]
    fixtures += [{"name": f"q{i}", "population": "sanity", "input_tokens": 42, "output_limit": 16}
                 for i in range(4)]
    blocks = []
    for block in range(2):
        for engine in ENGINES:
            rate = (100, 400)[block] * (1 if engine == "stark_optimized" else 2)
            samples = [{"fixture": f["name"], "input_tokens": 42, "output_tokens": f["output_limit"],
                        "wall_seconds": f["output_limit"] / rate, "fixed_work_eligible": True,
                        "finish_reason": "length", "token_sha256": "0" * 64, "text_sha256": "1" * 64,
                        "strict_sanity_correct": False} for f in fixtures]
            blocks.append({"engine": engine, "block": block, "status": "complete", "samples": samples})
    return {"fixtures": fixtures, "blocks": blocks, "protocol": {"blocks": 2, "repeats": 1}, "timing_accepted": True}


class SummaryTests(unittest.TestCase):
    def test_load_estimator_and_paired_ratio(self):
        result = summarize(fixture_receipt())
        self.assertTrue(result["allow_public_table"])
        self.assertTrue(math.isclose(result["rates"]["stark_optimized"]["p0"]["geomean_tokens_per_second"], 200))
        self.assertTrue(math.isclose(result["overall_engine_over_stark_geomean"]["mlx_lm"], 2))
        self.assertIsNone(result["cross_engine_hash_agreement_performance_fixture_count"]["token_hash_agreement_counts"]["tensorfold_default"])

    def test_missing_baseline_duplicate_and_interference_reject_publication(self):
        original = fixture_receipt()
        for change in ("missing", "duplicate", "interference"):
            receipt = copy.deepcopy(original)
            if change == "missing":
                receipt["blocks"].pop(0)
            elif change == "duplicate":
                receipt["blocks"].append(copy.deepcopy(receipt["blocks"][0]))
            else:
                receipt["blocks"][0]["status"] = "excluded_interference"
                receipt["timing_accepted"] = False
            result = summarize(receipt)
            self.assertFalse(result["allow_public_table"])
            self.assertEqual(result["rates"], {})

    def test_input_mismatch_fails_closed(self):
        receipt = fixture_receipt()
        receipt["blocks"][0]["samples"][0]["input_tokens"] = 43
        with self.assertRaisesRegex(ValueError, "fixed_work_count_mismatch"):
            summarize(receipt)


if __name__ == "__main__":
    unittest.main()
