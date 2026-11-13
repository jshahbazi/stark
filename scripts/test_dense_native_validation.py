"""Model-free checks for the dense native validation reference geometry."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dense_native_validation import (
    FULL_ATTENTION_LONG_CASE_TOKENS,
    long_case_tokens,
)
from test_hybrid_workers import CheckFailed


class LongCaseTokens(unittest.TestCase):
    def test_active_window_crosses_its_edge(self) -> None:
        self.assertEqual(long_case_tokens({"sliding_window": 4096}), 4101)
        self.assertEqual(long_case_tokens({"sliding_window": 512}), 517)
        self.assertEqual(
            long_case_tokens({"sliding_window": 4096, "use_sliding_window": True}),
            4101,
        )

    def test_full_attention_crosses_the_prefill_chunk(self) -> None:
        for text in (
            {"sliding_window": None},
            {},
            {"sliding_window": 32768, "use_sliding_window": False},
            {"sliding_window": 131072, "use_sliding_window": False},
        ):
            self.assertEqual(long_case_tokens(text), FULL_ATTENTION_LONG_CASE_TOKENS)
        self.assertGreater(FULL_ATTENTION_LONG_CASE_TOKENS, 1024)

    def test_active_window_outside_bounds_is_refused(self) -> None:
        for window in (0, 4097, 32768, 1.5, True, "4096"):
            with self.assertRaises(CheckFailed):
                long_case_tokens({"sliding_window": window})
        with self.assertRaises(CheckFailed):
            long_case_tokens({"sliding_window": 32768, "use_sliding_window": True})


if __name__ == "__main__":
    unittest.main()
