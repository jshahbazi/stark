"""Model-free tests for the stock MLX-LM worker adapters."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_stock_engines import Engine


def worker_module():
    path = Path(__file__).with_name("stock_compare_mlx.py")
    spec = importlib.util.spec_from_file_location("stock_compare_mlx_tested", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Response:
    def __init__(self, token, text, finish_reason=None):
        self.token = token
        self.text = text
        self.finish_reason = finish_reason


class StockWorkerTests(unittest.TestCase):
    def test_current_llama_native_termination_schema(self):
        engine = object.__new__(Engine)
        engine.kind = "llama"
        engine.base = "http://unused"
        fixture = {"ids": [1, 2], "limit": 3}
        response = {
            "tokens": [10, 11, 12],
            "content": "fixture",
            "tokens_evaluated": 2,
        }
        for stop_type, reason in (
            ("limit", "length"),
            ("eos", "stop"),
            ("word", "stop"),
        ):
            with (
                self.subTest(stop_type=stop_type),
                patch(
                    "benchmark_stock_engines.http",
                    return_value={**response, "stop_type": stop_type},
                ),
            ):
                result = engine.generate(fixture, 8192)
                self.assertEqual(result["server_stop_type"], stop_type)
                self.assertEqual(result["finish_reason"], reason)
                self.assertEqual(result["fixed_work_eligible"], stop_type == "limit")
        for stop_type in (None, "unknown", "none"):
            with (
                self.subTest(stop_type=stop_type),
                patch(
                    "benchmark_stock_engines.http",
                    return_value={**response, "stop_type": stop_type},
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "unknown_llama_stop_type"):
                    engine.generate(fixture, 8192)

    def test_length_final_response_contributes_actual_last_token(self):
        worker = worker_module()
        mlx_lm = types.ModuleType("mlx_lm")
        mlx_lm.stream_generate = lambda *args, **kwargs: iter(
            [Response(10, "a"), Response(11, "b"), Response(12, "c", "length")]
        )
        sample_utils = types.ModuleType("mlx_lm.sample_utils")
        sample_utils.make_sampler = lambda **kwargs: object()
        old = {
            name: sys.modules.get(name) for name in ("mlx_lm", "mlx_lm.sample_utils")
        }
        sys.modules["mlx_lm"] = mlx_lm
        sys.modules["mlx_lm.sample_utils"] = sample_utils
        try:
            result = worker.generate(
                object(),
                object(),
                {
                    "input_token_ids": [1, 2],
                    "max_new_tokens": 3,
                },
            )
        finally:
            for name, value in old.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
        self.assertEqual(result["generated_token_ids"], [10, 11, 12])
        self.assertEqual(result["output_tokens"], 3)
        self.assertEqual(result["finish_reason"], "length")

    def test_eos_final_response_is_not_counted_as_generated_token(self):
        worker = worker_module()
        mlx_lm = types.ModuleType("mlx_lm")
        mlx_lm.stream_generate = lambda *args, **kwargs: iter(
            [Response(20, "a"), Response(21, "b"), Response(2, "", "stop")]
        )
        sample_utils = types.ModuleType("mlx_lm.sample_utils")
        sample_utils.make_sampler = lambda **kwargs: object()
        old = {
            name: sys.modules.get(name) for name in ("mlx_lm", "mlx_lm.sample_utils")
        }
        sys.modules["mlx_lm"] = mlx_lm
        sys.modules["mlx_lm.sample_utils"] = sample_utils
        try:
            result = worker.generate(
                object(),
                object(),
                {
                    "input_token_ids": [1, 2],
                    "max_new_tokens": 3,
                },
            )
        finally:
            for name, value in old.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
        self.assertEqual(result["generated_token_ids"], [20, 21])
        self.assertEqual(result["output_tokens"], 2)
        self.assertEqual(result["finish_reason"], "stop")

    def test_score_normalizes_candidates_against_full_vocabulary(self):
        worker = worker_module()

        class Scalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

            def __sub__(self, other):
                return Scalar(self.value - other.value)

        class Row:
            def __init__(self, values):
                self.values = values

            def astype(self, _):
                return self

            def __getitem__(self, index):
                return Scalar(self.values[index])

        class Output:
            def __getitem__(self, index):
                return Row([1.0, 2.0, 3.0])

        fake_mx = types.ModuleType("mlx.core")
        fake_mx.float32 = object()
        fake_mx.array = lambda value: value
        fake_mx.logsumexp = lambda row: Scalar(3.4076059644443806)
        fake_mx.eval = lambda *values: None
        cache_module = types.ModuleType("mlx_lm.models.cache")
        cache_module.make_prompt_cache = lambda model: []
        old = {
            name: sys.modules.get(name)
            for name in ("mlx", "mlx.core", "mlx_lm.models.cache")
        }
        fake_mlx = types.ModuleType("mlx")
        fake_mlx.core = fake_mx
        sys.modules["mlx"] = fake_mlx
        sys.modules["mlx.core"] = fake_mx
        sys.modules["mlx_lm.models.cache"] = cache_module
        try:

            class Model:
                def __call__(self, ids, cache=None):
                    return Output()

            result = worker.score(
                Model(),
                object(),
                {
                    "input_token_ids": [1, 2],
                    "candidate_ids": [0, 2],
                },
            )
        finally:
            for name, value in old.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
        self.assertTrue(result["finite_logits"])
        self.assertTrue(result["finite_log_probabilities"])
        self.assertAlmostEqual(
            result["candidate_log_probabilities"][0], -2.4076059644443806
        )
        self.assertAlmostEqual(
            result["candidate_log_probabilities"][1], -0.4076059644443806
        )


if __name__ == "__main__":
    unittest.main()
