"""CPU-only contract checks for concurrent-wave accounting."""

import threading
import copy
import unittest
from concurrent.futures import ThreadPoolExecutor

from benchmark_stock_serving import serving_config, summaries, validate_config, wave


class ServingTests(unittest.TestCase):
    def config(self):
        return serving_config(
            {
                "source": "source",
                "task_root": "task",
                "context": 8192,
                "engines": [
                    {
                        "kind": "vllm",
                        "command": [
                            "vllm",
                            "serve",
                            "source",
                            "--host",
                            "127.0.0.1",
                            "--port",
                            "{port}",
                            "--served-model-name",
                            "stock-qwen35",
                            "--generation-config",
                            "vllm",
                            "--max-model-len",
                            "{context}",
                            "--language-model-only",
                        ],
                    }
                ],
            }
        )

    def test_cli_and_report_capacity_must_agree(self):
        config = self.config()
        validate_config(config)
        command = config["engine"]["command"]
        command[command.index("--max-num-seqs") + 1] = "1"
        with self.assertRaisesRegex(ValueError, "server_cli_configuration_mismatch"):
            validate_config(config)

    def test_changed_budget_requires_matching_stock_cli(self):
        config = self.config()
        config["gpu_memory_utilization"] = 0.2
        with self.assertRaisesRegex(ValueError, "server_cli_configuration_mismatch"):
            validate_config(config)
        command = config["engine"]["command"]
        command[command.index("--gpu-memory-utilization") + 1] = "0.2"
        validate_config(config)

    def test_zero_waves_and_conflicting_prefix_controls_are_refused(self):
        config = self.config()
        empty = copy.deepcopy(config)
        empty["rounds"] = 0
        with self.assertRaisesRegex(ValueError, "serving_protocol_invalid"):
            validate_config(empty)
        config["engine"]["command"].append("--enable-prefix-caching")
        with self.assertRaisesRegex(ValueError, "server_forbidden_control"):
            validate_config(config)

    def test_config_keeps_one_fixed_eight_slot_server(self):
        base = {
            "source": "source",
            "task_root": "task",
            "context": 8192,
            "engines": [
                {"kind": "stark", "command": ["stark"]},
                {
                    "kind": "vllm",
                    "command": [
                        "vllm",
                        "serve",
                        "source",
                        "--max-num-seqs",
                        "1",
                        "--gpu-memory-utilization",
                        "0.2",
                    ],
                },
            ],
        }
        result = serving_config(base)
        command = result["engine"]["command"]
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "8")
        self.assertEqual(command[command.index("--gpu-memory-utilization") + 1], "0.1")
        self.assertIn("--no-enable-prefix-caching", command)
        self.assertEqual(result["concurrencies"], [1, 4, 8])
        self.assertEqual(base["engines"][1]["command"][4], "1")

    def test_wave_dispatches_concurrently_and_preserves_unknown_counters(self):
        gate = threading.Barrier(4)

        class FakeEngine:
            def generate(self, fixture, context):
                gate.wait(timeout=2)
                return {
                    "input_tokens": 10,
                    "output_tokens": 128,
                    "returned_token_id_count": 128,
                    "finish_reason": "length",
                    "wall_seconds": 0.2,
                    "token_sha256": "same",
                    "cached_tokens": None,
                }

        with ThreadPoolExecutor(max_workers=4) as pool:
            result = wave(FakeEngine(), {"limit": 128}, 8192, 4, pool)
        self.assertTrue(result["fixed_work_eligible"])
        self.assertEqual(result["completed_output_tokens"], 512)
        self.assertEqual(result["cache_counter_unknown_requests"], 4)
        self.assertEqual(result["cache_counter_known_requests"], 0)

    def test_early_stop_cannot_enter_fixed_work_throughput_summary(self):
        class FakeEngine:
            def generate(self, fixture, context):
                return {
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "returned_token_id_count": 2,
                    "finish_reason": "stop",
                    "wall_seconds": 0.2,
                    "token_sha256": "early",
                    "cached_tokens": None,
                }

        with ThreadPoolExecutor(max_workers=1) as pool:
            result = wave(FakeEngine(), {"limit": 128}, 8192, 1, pool)
        self.assertFalse(result["fixed_work_eligible"])
        self.assertIsNone(result["aggregate_output_tokens_per_second"])
        self.assertEqual(
            summaries(
                [
                    {
                        "block": 0,
                        "status": "complete",
                        "waves": [{"concurrency": 1, **result}],
                    }
                ]
            ),
            {},
        )

    def test_request_failure_is_retained_and_invalidates_wave(self):
        class FakeEngine:
            def generate(self, fixture, context):
                raise RuntimeError("worker_timeout")

        with ThreadPoolExecutor(max_workers=1) as pool:
            result = wave(FakeEngine(), {"limit": 128}, 8192, 1, pool)
        self.assertFalse(result["fixed_work_eligible"])
        self.assertEqual(
            result["failed_requests"], [{"request_index": 0, "error": "worker_timeout"}]
        )


if __name__ == "__main__":
    unittest.main()
