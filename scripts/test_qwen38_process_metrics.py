"""Model-free tests for the private macOS process metrics helper."""

import unittest

from qwen38_process_metrics import delta, snapshot


class ProcessMetricsTests(unittest.TestCase):
    def test_delta_keeps_gauges_and_omits_identity(self):
        before = {"available": True, "_identity": b"a" * 16, "units": {"time": "nanoseconds"},
                  "cumulative": {"user_time": 10, "pageins": 2, "resident_size": 100}}
        after = {"available": True, "_identity": b"a" * 16, "units": {"time": "nanoseconds"},
                 "cumulative": {"user_time": 35, "pageins": 5, "resident_size": 125}}

        result = delta(before, after)

        self.assertTrue(result["available"])
        self.assertEqual(result["delta"], {"user_time": 25, "pageins": 3})
        self.assertEqual(result["gauges"], {"resident_size": 125})
        self.assertNotIn("_identity", result)

    def test_delta_rejects_pid_reuse_and_unavailable_inputs(self):
        available = {"available": True, "_identity": b"a" * 16, "cumulative": {}}
        changed = {"available": True, "_identity": b"b" * 16, "cumulative": {}}

        self.assertEqual(delta(available, changed), {"available": False, "reason": "process_changed"})
        self.assertEqual(delta({"available": False}, available),
                         {"available": False, "reason": "snapshot_unavailable"})

    def test_decreasing_memory_gauge_is_retained(self):
        before = {"available": True, "_identity": b"a" * 16,
                  "cumulative": {"resident_size": 200, "instructions": 10}}
        after = {"available": True, "_identity": b"a" * 16,
                 "cumulative": {"resident_size": 100, "instructions": 9}}

        result = delta(before, after)

        self.assertEqual(result["gauges"], {"resident_size": 100})
        self.assertEqual(result["delta"], {})

    def test_snapshot_failure_is_metadata_only(self):
        self.assertEqual(snapshot(0), {"available": False, "reason": "invalid_pid"})
