"""Model-free tests for the private macOS process metrics helper."""

import unittest
from unittest.mock import patch

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
        for platform in ("darwin", "linux"):
            for pid in (0, -1, True, None, "1", 1.5):
                with self.subTest(platform=platform, pid=pid), \
                        patch("qwen38_process_metrics.sys.platform", platform), \
                        patch("qwen38_process_metrics.ctypes.CDLL") as load_library:
                    self.assertEqual(snapshot(pid), {"available": False, "reason": "invalid_pid"})
                    load_library.assert_not_called()

    def test_snapshot_unsupported_platform_does_not_load_native_library(self):
        with patch("qwen38_process_metrics.sys.platform", "linux"), \
                patch("qwen38_process_metrics.ctypes.CDLL") as load_library:
            self.assertEqual(snapshot(1), {"available": False, "reason": "unsupported"})
            load_library.assert_not_called()

    def test_snapshot_native_failure_does_not_expose_exception(self):
        with patch("qwen38_process_metrics.sys.platform", "darwin"), \
                patch("qwen38_process_metrics.ctypes.CDLL", side_effect=OSError("private detail")):
            self.assertEqual(snapshot(1), {"available": False, "reason": "api_unavailable"})
