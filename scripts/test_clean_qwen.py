"""Check CPU attribution and phase-aware quiet-window admission."""

import unittest
from unittest.mock import patch

from benchmark_clean_qwen import Monitor, accept_timing, cpu_deltas, quiet_window


def process(parent, cpu, name="external", start="one"):
    return {"parent": parent, "cpu": cpu, "name": name, "start": start}


class CpuAttributionTests(unittest.TestCase):
    def test_observe_only_run_cannot_be_certified_as_quiet(self):
        report = {
            "protocol": {},
            "power_log_screen": {"checked": True, "sleep_events": 0},
            "sleep_clock_discrepancy_seconds": 0,
            "blocks": [{"status": "complete"}],
        }
        self.assertTrue(accept_timing(report))
        report["protocol"]["require_quiet_windows"] = False
        self.assertFalse(accept_timing(report))
        report["protocol"]["require_quiet_windows"] = True
        report["blocks"][0]["status"] = "excluded_interference"
        self.assertFalse(accept_timing(report))

    def test_descendants_are_owned_but_xpc_compilers_are_external(self):
        old = {1: process(0, 1), 2: process(1, 1), 3: process(2, 1), 4: process(0, 1)}
        new = {
            1: process(0, 2),
            2: process(1, 2),
            3: process(2, 2),
            4: process(0, 4, "MTLCompilerService"),
        }
        self.assertEqual(
            cpu_deltas(old, new, 0.5, 1),
            [{"pid": 4, "name": "MTLCompilerService", "cpu_percent": 600.0}],
        )

    def test_new_process_and_reused_pid_do_not_disappear(self):
        old = {2: process(0, 100)}
        new = {2: process(0, 0.4, start="two"), 3: process(0, 0.2)}
        values = cpu_deltas(old, new, 0.5, 1)
        self.assertEqual([v["cpu_percent"] for v in values], [80.0, 40.0])

    def test_boundary_interval_is_guarded(self):
        monitor = Monitor()
        monitor.rows = [
            {
                "phases": ["measurement", "teardown"],
                "max_process_cpu_percent": 110,
                "external_cpu_percent": 110,
                "max_model_cpu_percent": 0,
                "thermal_state": 0,
                "interval_seconds": 0.5,
            }
        ]
        self.assertTrue(monitor.result()["cpu_guard_exceeded"])

    def test_load_compiler_peak_is_retained_separately(self):
        monitor = Monitor()
        monitor.rows = [
            {
                "phases": ["load"],
                "max_process_cpu_percent": 600,
                "external_cpu_percent": 600,
                "max_model_cpu_percent": 0,
                "thermal_state": 0,
                "interval_seconds": 0.5,
            },
            {
                "phases": ["measurement"],
                "max_process_cpu_percent": 30,
                "external_cpu_percent": 60,
                "max_model_cpu_percent": 0,
                "thermal_state": 0,
                "interval_seconds": 0.5,
            },
        ]
        self.assertFalse(monitor.result()["cpu_guard_exceeded"])
        self.assertEqual(len(monitor.result()["samples"]), 2)

    def test_busy_gpu_resets_quiet_streak(self):
        monitor = Monitor()
        ticks = iter(range(20))
        quiet = {
            "phases": ["quiet"],
            "max_process_cpu_percent": 0,
            "external_cpu_percent": 0,
            "max_model_cpu_percent": 0,
            "gpu_device_utilization_percent": 0,
            "thermal_state": 0,
            "interval_seconds": 1,
        }

        def sample(_):
            row = dict(quiet)
            row["start_seconds"] = len(monitor.rows)
            row["end_seconds"] = len(monitor.rows) + 1
            if len(monitor.rows) == 1:
                row["gpu_device_utilization_percent"] = 90
            monitor.rows.append(row)

        with (
            patch(
                "benchmark_clean_qwen.time.monotonic", side_effect=lambda: next(ticks)
            ),
            patch("benchmark_clean_qwen.time.sleep", side_effect=sample),
        ):
            quiet_window(monitor, seconds=2, timeout=30)
            self.assertGreaterEqual(len(monitor.rows), 4)

    def test_stale_quiet_rows_cannot_admit_a_window(self):
        monitor = Monitor()
        monitor.rows = [{"phases": ["quiet"]}]
        with (
            patch("benchmark_clean_qwen.time.monotonic", side_effect=range(20)),
            patch("benchmark_clean_qwen.time.sleep"),
            self.assertRaisesRegex(RuntimeError, "quiet_window_timeout"),
        ):
            quiet_window(monitor, seconds=2, timeout=5)


if __name__ == "__main__":
    unittest.main()
