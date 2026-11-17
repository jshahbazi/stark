"""Model-free tests for Decider HTTP profile aggregation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from profile_decider_http import ProfileError, aggregate_rows, generated_png, summarize


def row(words: int, elapsed: float, tokens: int) -> dict:
    return {
        "requested_words": words,
        "http_wall_seconds": elapsed + 0.4,
        "encode_seconds": elapsed / 10,
        "native_roundtrip_seconds": elapsed,
        "state_creation_seconds": elapsed / 20,
        "prefill_seconds": elapsed - 0.1,
        "readout_seconds": 0.01,
        "total_seconds": elapsed - 0.05,
        "input_tokens": tokens,
    }


class ProfileAggregationTests(unittest.TestCase):
    def test_aggregate_discards_warmup_and_keeps_only_summaries(self):
        output = aggregate_rows(
            [
                row(0, 1.0, 10),
                row(0, 2.0, 10),
                row(0, 4.0, 10),
                row(64, 2.0, 120),
                row(64, 3.0, 120),
            ],
            warmups=1,
        )
        self.assertEqual(output[0]["requested_words"], 0)
        self.assertEqual(output[0]["samples"], 2)
        self.assertEqual(
            output[0]["metrics"]["native_roundtrip_seconds"],
            {
                "median": 3.0,
                "min": 2.0,
                "max": 4.0,
            },
        )
        self.assertIn("input_tokens", output[0])
        self.assertNotIn("requested_words", output[0]["metrics"])
        self.assertEqual(output[1]["samples"], 1)

    def test_aggregate_rejects_missing_measured_rows(self):
        with self.assertRaisesRegex(ProfileError, "missing_measurements"):
            aggregate_rows([row(0, 1.0, 10)], warmups=1)

    def test_summarize_returns_only_aggregate_statistics(self):
        self.assertEqual(
            summarize([1.0, 2.0, 3.0]), {"median": 2.0, "min": 1.0, "max": 3.0}
        )

    def test_generated_image_is_a_small_png(self):
        png = generated_png(2, 3)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertLess(len(png), 1024)


if __name__ == "__main__":
    unittest.main()
