"""Model-free checks of the capture schedules. Run with `python3 -m unittest`."""

import unittest

from schedule import CASES, check_case, replay_boundaries, request_fields, source_chunks


class ScheduleTest(unittest.TestCase):
    def test_archived_stock_schedule(self):
        self.assertEqual(request_fields(4096), {"Start": 4095, "Chunks": [2048, 2044, 3], "FullTarget": True})
        self.assertEqual(replay_boundaries(4096), [2048])

    def test_short_prompts(self):
        for prompt, bulk in ((68, 64), (69, 65), (73, 69), (130, 126), (511, 507)):
            self.assertEqual(source_chunks(prompt), [bulk, 3])
            self.assertEqual(replay_boundaries(prompt), [])

    def test_long_prompts(self):
        self.assertEqual(source_chunks(8800), [2048] * 4 + [604, 3])
        self.assertEqual(replay_boundaries(8800), [4096, 8192])
        self.assertEqual(source_chunks(16000), [2048] * 7 + [1660, 3])
        self.assertEqual(replay_boundaries(16000), [4096, 8192, 14336])
        self.assertEqual(source_chunks(8188), [2048] * 3 + [2040, 3])
        self.assertEqual(source_chunks(4350), [2048, 2048, 250, 3])
        self.assertEqual(source_chunks(4100), [2048, 2048, 3])

    def test_chunks_cover_the_prefix(self):
        for prompt, context, _ in CASES.values():
            check_case(prompt, context)
            fields = request_fields(prompt)
            self.assertEqual(sum(fields["Chunks"]), fields["Start"])

    def test_teacher_rows_past_log_scaling(self):
        prompt, context, _ = CASES["l8800"]
        self.assertGreater(prompt - 1, 8192)
        self.assertLessEqual(prompt + 3, context)


if __name__ == "__main__":
    unittest.main()
