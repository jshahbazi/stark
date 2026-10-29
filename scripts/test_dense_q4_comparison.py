from __future__ import annotations

import os
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_stock_engines import HostMonitor
from dense_q4_comparison import (
    ComparisonError,
    assert_same_identity,
    bind_campaign_report,
    engine_order,
    freeze_engines,
    host_guard_result,
    paired_summary,
    sha256_file,
    validate_frozen_engine,
    validate_report,
)
from dense_q4_sequence_reference import REFERENCE_PROTOCOL, prefill_cache_only


def report(engine: str, rate: float = 100.0) -> dict:
    request_seconds = 640 / rate
    return {
        "passed": True,
        "scope": "matched-pp512-tg128-greedy-v1",
        "input_sha256": "a" * 64,
        "input_tokens": 512,
        "returned_tokens": 128,
        "decode_forward_steps": 128,
        "context_tokens": 1024,
        "source_repo": "test/model",
        "source_revision": "b" * 40,
        "source_config_sha256": "c" * 64,
        "source_weights_digest": "d" * 64,
        "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
        "source_weights": [
            {"filename": "model.safetensors", "bytes": 1024, "sha256": "e" * 64}
        ],
        "warmup_runs": 1,
        "measured_runs": 3,
        "backend": engine,
        "median_request_total_seconds": request_seconds,
        "median_work_tokens_per_second": rate,
        "repeated_outputs_identical": True,
        "runs": [
            {
                "run": index,
                "measured": index > 0,
                "input_tokens": 512,
                "returned_tokens": 128,
                "decode_forward_steps": 128,
                "request_total_seconds": request_seconds,
                "work_tokens_per_second": rate,
                "output_sha256": "f" * 64,
            }
            for index in range(4)
        ],
    }


def native_report() -> dict:
    value = report("stark-metal")
    value.update(
        executable_sha256="1" * 64,
        native_weights_sha256="2" * 64,
        native_config_sha256="3" * 64,
        conversion_receipt_sha256="4" * 64,
        activation_dtype="bfloat16",
        device={"name": "test-device"},
    )
    return value


def mlx_report() -> dict:
    value = report("mlx-lm-direct-greedy")
    value.update(
        harness_sha256="5" * 64,
        mlx_lm_revision="6" * 40,
        mlx_version="0.32.3",
        transformers_version="5.7.0",
        floating_parameter_dtypes=["mlx.core.bfloat16"],
        device={"device_name": "test-device"},
    )
    return value


def campaign_binding() -> dict:
    return {
        "identity": None,
        "native_artifacts": {
            "native_weights_sha256": "2" * 64,
            "native_config_sha256": "3" * 64,
            "conversion_receipt_sha256": "4" * 64,
        },
        "engine_profiles": {},
    }


NATIVE_ENGINE = {"id": "candidate_native", "kind": "native", "binary_sha256": "1" * 64}
MLX_ENGINE = {"id": "mlx", "kind": "mlx", "harness_sha256": "5" * 64}


class DenseQ4ComparisonTests(unittest.TestCase):
    def test_sequence_prefill_evaluates_mutated_cache_without_readout(self):
        events = []
        cache = [SimpleNamespace(state="before")]
        token_batch = object()
        readout = object()

        def model(value, *, cache):
            self.assertIs(value, token_batch)
            cache[0].state = "updated-cache"
            events.append("model")
            return readout

        def evaluate(value):
            self.assertIsNot(value, readout)
            self.assertEqual(value, ["updated-cache"])
            events.append("cache-eval")

        mx = SimpleNamespace(
            eval=evaluate, clear_cache=lambda: events.append("clear-cache")
        )
        prefill_cache_only(mx, model, token_batch, cache)
        self.assertEqual(events, ["model", "cache-eval", "clear-cache"])
        self.assertEqual(
            REFERENCE_PROTOCOL, "dense-q4-sequence-v3-cache-state-only-prefill"
        )

    def test_campaign_freezes_programs_and_rejects_mutated_execution_inputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            native = root / "native"
            native.write_bytes(b"native-v1")
            harness = root / "dense_profile_mlx.py"
            harness.write_bytes(b"mlx-harness-v1")
            python = root / "python"
            python.write_bytes(b"python-v1")
            fixture = root / "fixture.json"
            fixture.write_bytes(b"fixture-v1")
            output = root / "output"
            output.mkdir()
            engines = [
                {"id": "candidate_native", "kind": "native", "binary": native},
                {"id": "mlx", "kind": "mlx"},
            ]
            args = Namespace(
                output_dir=output,
                fixture=fixture,
                model=root / "model",
                source=root / "source",
                mlx_python=python,
            )
            with patch("dense_q4_comparison.MLX_HARNESS", harness):
                freeze_engines(engines, args, "unit", sha256_file(fixture))
            native.write_bytes(b"native-v2")
            harness.write_bytes(b"mlx-harness-v2")
            self.assertEqual(engines[0]["binary"].read_bytes(), b"native-v1")
            self.assertEqual(engines[1]["harness"].read_bytes(), b"mlx-harness-v1")
            for engine in engines:
                validate_frozen_engine(engine)
            fixture.write_bytes(b"fixture-v2")
            with self.assertRaises(ComparisonError) as caught:
                validate_frozen_engine(engines[0])
            self.assertEqual(caught.exception.code, "fixture_changed_during_campaign")
            fixture.write_bytes(b"fixture-v1")
            engines[0]["binary"].write_bytes(b"mutated")
            with self.assertRaises(ComparisonError) as caught:
                validate_frozen_engine(engines[0])
            self.assertEqual(
                caught.exception.code, "native_executable_changed_during_campaign"
            )
            engines[1]["harness"].write_bytes(b"mutated")
            with self.assertRaises(ComparisonError) as caught:
                validate_frozen_engine(engines[1])
            self.assertEqual(caught.exception.code, "mlx_harness_changed_during_campaign")

    def test_campaign_identity_is_bound_across_loads_and_engines(self):
        binding = campaign_binding()
        bind_campaign_report(native_report(), NATIVE_ENGINE, binding)
        bind_campaign_report(mlx_report(), MLX_ENGINE, binding)
        changed = native_report()
        changed["input_sha256"] = "7" * 64
        with self.assertRaises(ComparisonError) as caught:
            bind_campaign_report(changed, NATIVE_ENGINE, binding)
        self.assertEqual(caught.exception.code, "campaign_identity_mismatch")

    def test_campaign_rejects_wrong_binary_or_mutated_native_artifact(self):
        for key, code in (
            ("executable_sha256", "native_executable_identity_mismatch"),
            ("native_weights_sha256", "native_artifact_changed_during_campaign"),
            ("native_config_sha256", "native_artifact_changed_during_campaign"),
            ("conversion_receipt_sha256", "native_artifact_changed_during_campaign"),
        ):
            with self.subTest(key=key):
                changed = native_report()
                changed[key] = "7" * 64
                with self.assertRaises(ComparisonError) as caught:
                    bind_campaign_report(changed, NATIVE_ENGINE, campaign_binding())
                self.assertEqual(caught.exception.code, code)

    def test_campaign_rejects_precision_or_cross_load_output_changes(self):
        for key, value in (
            ("activation_dtype", "float32"),
            ("device", {"name": "another-device"}),
        ):
            with self.subTest(key=key):
                binding = campaign_binding()
                bind_campaign_report(native_report(), NATIVE_ENGINE, binding)
                changed = native_report()
                changed[key] = value
                with self.assertRaises(ComparisonError) as caught:
                    bind_campaign_report(changed, NATIVE_ENGINE, binding)
                self.assertEqual(
                    caught.exception.code, "engine_profile_changed_during_campaign"
                )
        binding = campaign_binding()
        bind_campaign_report(native_report(), NATIVE_ENGINE, binding)
        changed = native_report()
        for run in changed["runs"]:
            run["output_sha256"] = "7" * 64
        with self.assertRaises(ComparisonError) as caught:
            bind_campaign_report(changed, NATIVE_ENGINE, binding)
        self.assertEqual(caught.exception.code, "engine_profile_changed_during_campaign")

    def test_campaign_rejects_changed_mlx_harness_and_reference_profile(self):
        changed = mlx_report()
        changed["harness_sha256"] = "7" * 64
        with self.assertRaises(ComparisonError) as caught:
            bind_campaign_report(changed, MLX_ENGINE, campaign_binding())
        self.assertEqual(caught.exception.code, "mlx_harness_identity_mismatch")
        binding = campaign_binding()
        bind_campaign_report(mlx_report(), MLX_ENGINE, binding)
        changed = mlx_report()
        changed["floating_parameter_dtypes"] = ["mlx.core.float32"]
        with self.assertRaises(ComparisonError) as caught:
            bind_campaign_report(changed, MLX_ENGINE, binding)
        self.assertEqual(caught.exception.code, "engine_profile_changed_during_campaign")

    def test_three_engine_orders_keep_position_counts_within_one(self):
        ids = ["baseline", "candidate", "mlx"]
        orders = [engine_order(ids, index) for index in range(4)]
        self.assertEqual(
            orders,
            [
                ["baseline", "candidate", "mlx"],
                ["candidate", "mlx", "baseline"],
                ["mlx", "baseline", "candidate"],
                ["baseline", "mlx", "candidate"],
            ],
        )
        for engine in ids:
            counts = [
                sum(order[position] == engine for order in orders)
                for position in range(3)
            ]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_report_requires_fixed_work_and_rejects_payload_fields(self):
        value = report("native")
        validate_report(value)
        value["runs"][0]["generated_token_ids"] = [1]
        with self.assertRaises(ComparisonError) as caught:
            validate_report(value)
        self.assertEqual(caught.exception.code, "unsafe_harness_field")

    def test_report_rejects_work_count_and_source_identity_mismatches(self):
        value = report("native")
        value["input_tokens"] = 511
        with self.assertRaises(ComparisonError) as caught:
            validate_report(value)
        self.assertEqual(caught.exception.code, "unexpected_work_counts")
        value = report("native")
        value["runs"][2]["decode_forward_steps"] = 127
        with self.assertRaises(ComparisonError) as caught:
            validate_report(value)
        self.assertEqual(caught.exception.code, "unexpected_run_work_counts")
        value = report("native")
        value["source_revision"] = "not-a-revision"
        with self.assertRaises(ComparisonError) as caught:
            validate_report(value)
        self.assertEqual(caught.exception.code, "invalid_source_identity")

    def test_report_rejects_nonrepeatable_outputs(self):
        value = report("native")
        value["runs"][2]["output_sha256"] = "1" * 64
        with self.assertRaises(ComparisonError) as caught:
            validate_report(value)
        self.assertEqual(caught.exception.code, "nonrepeatable_outputs")
        value = report("native")
        value["repeated_outputs_identical"] = False
        with self.assertRaises(ComparisonError) as caught:
            validate_report(value)
        self.assertEqual(caught.exception.code, "nonrepeatable_outputs")

    def test_report_recomputes_median_duration_and_rate(self):
        value = report("native")
        value["median_work_tokens_per_second"] = 123.0
        with self.assertRaises(ComparisonError) as caught:
            validate_report(value)
        self.assertEqual(caught.exception.code, "invalid_timing")
        value = report("native")
        value["median_request_total_seconds"] = 7.0
        with self.assertRaises(ComparisonError) as caught:
            validate_report(value)
        self.assertEqual(caught.exception.code, "invalid_timing")
        value = report("native")
        value["runs"][1]["work_tokens_per_second"] = 999.0
        with self.assertRaises(ComparisonError) as caught:
            validate_report(value)
        self.assertEqual(caught.exception.code, "invalid_timing")

    def test_monitor_excludes_root_children_and_counts_external_processes(self):
        root_pid = os.getpid()
        snapshots = [
            {
                root_pid + 1: (root_pid, 1.0, "dense_profile"),
                root_pid + 2: (root_pid + 1, 1.0, "mlx-worker"),
                root_pid + 3: (1, 1.0, "unrelated"),
                root_pid + 4: (1, 1.0, "mlx-server"),
            },
            {
                root_pid + 1: (root_pid, 2.0, "dense_profile"),
                root_pid + 2: (root_pid + 1, 2.0, "mlx-worker"),
                root_pid + 3: (1, 1.1, "unrelated"),
                root_pid + 4: (1, 1.2, "mlx-server"),
            },
        ]
        waits = iter((False, True))
        monitor = HostMonitor()
        monitor.done = type(
            "Done", (), {"wait": staticmethod(lambda _delay: next(waits))}
        )()
        with (
            patch("benchmark_stock_engines.process_times", side_effect=snapshots),
            patch("benchmark_stock_engines.time.time", side_effect=(10.0, 11.0)),
        ):
            monitor.run()
        result = monitor.result()
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["sampling_failures"], 0)
        self.assertAlmostEqual(result["max_background_process_cpu_percent"], 20.0)
        self.assertAlmostEqual(result["max_background_model_process_cpu_percent"], 20.0)
        self.assertTrue(result["background_model_active"])

    def test_identity_binds_same_source_fixture_and_quantization(self):
        native = report("native")
        mlx = report("mlx")
        assert_same_identity([native, mlx])
        mlx["source_weights_digest"] = "e" * 64
        with self.assertRaises(ComparisonError) as caught:
            assert_same_identity([native, mlx])
        self.assertEqual(caught.exception.code, "engine_identity_mismatch")

    def test_guard_rejects_failed_samples_and_clock_discrepancy(self):
        base = {
            "samples": 2,
            "sampling_failures": 0,
            "cpu_guard_exceeded": False,
            "background_model_active": False,
        }
        self.assertTrue(host_guard_result(base, 10.0, 20.0, 10.0, 20.0)["accepted"])
        failed = dict(base, sampling_failures=1)
        self.assertFalse(host_guard_result(failed, 10.0, 20.0, 10.0, 20.0)["accepted"])
        self.assertFalse(host_guard_result(base, 10.0, 20.0, 10.0, 17.0)["accepted"])

    def test_paired_summary_keeps_every_guard_accepted_pair(self):
        native_guard = {"accepted": True}
        mlx_guard = {"accepted": True}
        blocks = []
        for index, rate in enumerate((100.0, 80.0, 120.0, 90.0)):
            blocks.append(
                {
                    "block": index,
                    "identity_match": True,
                    "engines": {
                        "native": {
                            "status": "complete",
                            "guard": native_guard,
                            "report": report("native", rate),
                        },
                        "mlx": {
                            "status": "complete",
                            "guard": mlx_guard,
                            "report": report("mlx", 100.0),
                        },
                    },
                }
            )
        result = paired_summary(blocks, "native", True)
        self.assertEqual(result["accepted_pair_count"], 4)
        self.assertAlmostEqual(result["median_native_to_mlx_rate_ratio"], 0.95)
        rejected = paired_summary(blocks, "native", False)
        self.assertEqual(rejected["accepted_pair_count"], 0)
        self.assertIsNone(rejected["median_native_to_mlx_rate_ratio"])


if __name__ == "__main__":
    unittest.main()
