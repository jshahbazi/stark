from __future__ import annotations

import json
from argparse import Namespace
from copy import deepcopy

import benchmark_native_prefill as benchmark
import pytest
from stark_convert.chat import ChatCancelled


class FakeWorker:
    def __init__(self, path, _model, _timeout, *, mismatch=False):
        self.mismatch = mismatch and path.name == "candidate"
        self.closed = False
        self.requests = []

    def read(self):
        return {
            "ready": True,
            "mode": "text-generation-v1",
            "streaming": "text-delta-v1",
            "max_context_tokens": 40960,
        }

    def request(self, request, **options):
        self.requests.append(deepcopy(request))
        if request["context_tokens"] <= len(request["input_token_ids"]):
            return {"error": "context_overflow"}
        if options.get("cancelled") is not None:
            options["cancelled"].set()
            raise ChatCancelled("synthetic cancellation")
        ids = [7 + int(self.mismatch)] * min(2, request["max_new_tokens"])
        text = "private-model-response-marker"
        sampled, stop, finish = len(ids), None, "length"
        if "stop_token_ids" in request:
            ids, text, sampled, stop, finish = (
                [],
                "",
                1,
                request["stop_token_ids"][0],
                "stop_token",
            )
        elif "stop" in request:
            ids, text, sampled, finish = [7], "", 1, "stop_sequence"
        if options.get("on_delta"):
            options["on_delta"](text)
        return {
            "generated_token_ids": ids,
            "text": text,
            "sampled_tokens": sampled,
            "stop_token_id": stop,
            "finish_reason": finish,
            "input_tokens": len(request["input_token_ids"]),
            "output_tokens": len(ids),
            "prefill_seconds": 2.0 if self.mismatch else 1.0,
            "decode_seconds": 0.1,
            "generation_seconds": 1.1,
        }

    def close(self):
        self.closed = True


def fixture_factory(_workers, _model, _target, _context, _maximum):
    return [
        {
            "name": name,
            "ids": [3] * count,
            "metadata": {
                "input_tokens": count,
                "tools": tools,
                "messages": messages,
                "tool_results": results,
            },
        }
        for name, count, tools, messages, results in [
            ("short_chat", 4, 0, 2, 0),
            ("long_tools_history", 64, 8, 5, 1),
        ]
    ]


@pytest.fixture
def arguments(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    for name, content in [
        ("config.json", "{}"),
        ("tokenizer.json", "{}"),
        ("model.safetensors", "private-weight-fixture"),
        ("conversion.json", json.dumps({"source_manifest_sha256": "a" * 64})),
    ]:
        (model / name).write_text(content)
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    baseline.write_bytes(b"baseline")
    candidate.write_bytes(b"candidate")
    monkeypatch.setattr(benchmark, "source_seal", lambda: "f" * 64)
    return Namespace(
        model=model,
        baseline=baseline,
        candidate=candidate,
        target_tokens=64,
        context=256,
        tokens=8,
        cycles=1,
        mode="all",
        no_quiet=True,
    )


def test_abba_parity_stops_cancel_overflow_and_report_privacy(arguments):
    workers = []

    def factory(*args):
        worker = FakeWorker(*args)
        workers.append(worker)
        return worker

    report = benchmark.run(
        arguments, worker_factory=factory, fixture_factory=fixture_factory
    )
    assert report["passed"] is True
    assert (
        report["timing_accepted"] is False
    )  # Explicit exploratory mode cannot claim accepted timing.
    assert len(report["timing_rows"]) == 8
    assert [r["worker"] for r in report["timing_rows"][:4]] == list(benchmark.ORDER)
    assert len(report["parity"]) == 10
    assert all(
        row["exact"] and row.get("stop_observed", True) for row in report["parity"]
    )
    assert all(
        value["context_overflow_recoverable"]
        and value["cancelled_before_terminal"]
        and value["recovery_exact"]
        for value in report["lifecycle"].values()
    )
    assert all(worker.closed for worker in workers)
    serialized = json.dumps(report)
    for forbidden in (
        "private-model-response-marker",
        "private-weight-fixture",
        "input_token_ids",
        "generated_token_ids",
        str(arguments.model),
    ):
        assert forbidden not in serialized
    assert any(
        r["sampling"].get("top_k") == 40 and r["seed"] == 17
        for r in workers[0].requests
    )
    assert any(
        r["sampling"].get("mode") == "stochastic"
        and "top_k" not in r["sampling"]
        and r["seed"] == 42
        for r in workers[0].requests
    )


def test_mismatched_candidate_never_passes_even_when_faster(arguments):
    report = benchmark.run(
        arguments,
        worker_factory=lambda *args: FakeWorker(*args, mismatch=True),
        fixture_factory=fixture_factory,
    )
    assert report["passed"] is False and report["timing_accepted"] is False
    assert not all(report["warmup_exact"])
    assert any(not row["exact"] for row in report["timing_rows"])


def test_render_keeps_all_tool_definitions_and_linked_result_history():
    class Template:
        def render(self, **values):
            return json.dumps(values, sort_keys=True)

    text, metadata = benchmark.render_fixture({}, Template(), 10)
    payload = json.loads(text)
    assert metadata == {"messages": 5, "tools": 8, "tool_results": 1}
    assert len(payload["tools"]) == 8
    assert "Reference context. " * 10 in payload["tools"][0]["function"]["description"]
    assistant, result = payload["messages"][2:4]
    assert assistant["tool_calls"][0]["id"] == result["tool_call_id"]
    assert assistant["tool_calls"][0]["function"]["arguments"] == {"key": "reference"}
    assert result["content"] == "Synthetic reference result: preserve all history."


@pytest.mark.parametrize(
    "change", [{"sampled_tokens": 0}, {"output_tokens": 2}, {"text": None}]
)
def test_invalid_native_reply_is_rejected(change):
    reply = {
        "generated_token_ids": [7],
        "text": "private",
        "sampled_tokens": 1,
        "finish_reason": "length",
        "input_tokens": 4,
    }
    with pytest.raises(ValueError, match="generation_invalid"):
        benchmark.checked_reply({**reply, **change}, 4, 8)


@pytest.mark.parametrize(
    "actual,expected",
    [
        (([7, 8], "private-marker", "length", None, 2), {}),
        (
            ([7, 9], "private-marker", "length", None, 2),
            {"same_ids": False, "first_token_mismatch_index": 1},
        ),
        (
            ([7], "private-marker", "length", None, 1),
            {
                "same_ids": False,
                "same_usage": False,
                "first_token_mismatch_index": 1,
            },
        ),
        (
            ([7, 8], "private-change", "length", None, 2),
            {"same_text": False, "first_text_mismatch_index": 8},
        ),
        (
            ([7, 8], "private-marker", "eos", 11, 3),
            {
                "same_finish_reason": False,
                "same_stop_token_id": False,
                "same_usage": False,
            },
        ),
    ],
)
def test_signature_component_diagnostics_locate_differences_without_payloads(
    actual, expected
):
    reference = ([7, 8], "private-marker", "length", None, 2)
    comparison = benchmark.signature_comparison(reference, actual)
    fields = {
        "same_ids",
        "same_text",
        "same_finish_reason",
        "same_stop_token_id",
        "same_usage",
    }
    for key in fields:
        assert comparison[key] is expected.get(key, True)
    for key in ("first_token_mismatch_index", "first_text_mismatch_index"):
        assert comparison[key] == expected.get(key)
    assert comparison["reference_output_tokens"] == 2
    assert comparison["actual_output_tokens"] == len(actual[0])
    assert comparison["reference_sampled_tokens"] == 2
    assert comparison["actual_sampled_tokens"] == actual[4]
    serialized = json.dumps(comparison)
    assert "private-" not in serialized
    assert "[7" not in serialized and "11" not in serialized


def test_generation_signature_excludes_nondeterministic_timing_statistics():
    reply = {
        "generated_token_ids": [7],
        "text": "private-marker",
        "input_tokens": 4,
        "sampled_tokens": 1,
        "output_tokens": 1,
        "finish_reason": "length",
        "prefill_seconds": 1.0,
        "decode_seconds": 0.1,
    }
    reference = benchmark.checked_reply(reply, 4, 8)
    actual = benchmark.checked_reply(
        {**reply, "prefill_seconds": 3.0, "decode_seconds": 0.3}, 4, 8
    )
    assert reference == actual
    assert benchmark.signature_comparison(reference, actual)["same_ids"] is True


def test_cli_refuses_existing_output_or_symlink_without_launch(tmp_path, monkeypatch):
    existing = tmp_path / "existing.json"
    existing.write_text("keep")
    output = tmp_path / "report.json"
    output.symlink_to(existing)
    monkeypatch.setattr(
        benchmark, "run", lambda _args: pytest.fail("must not launch a worker")
    )
    args = [
        "--model",
        str(tmp_path),
        "--baseline",
        str(tmp_path),
        "--candidate",
        str(tmp_path),
        "--output",
        str(output),
    ]
    assert benchmark.main(args) == 1
    assert existing.read_text() == "keep"


@pytest.mark.parametrize(
    "failure,category",
    [
        (RuntimeError("quiet_window_timeout"), "quiet_window_timeout"),
        (RuntimeError("private-runtime-exception-marker"), "operation_failed"),
        (LookupError("private-other-exception-marker"), "operation_failed"),
    ],
)
def test_quiet_guard_failure_writes_sanitized_report_and_closes_workers(
    arguments, monkeypatch, tmp_path, capsys, failure, category
):
    import benchmark_clean_qwen

    workers = []
    monitors = []

    class Monitor:
        def __enter__(self):
            self.closed = False
            monitors.append(self)
            return self

        def __exit__(self, *_args):
            self.closed = True

        def set_phase(self, _phase):
            pass

    def factory(*args):
        worker = FakeWorker(*args)
        workers.append(worker)
        return worker

    quiet_calls = 0

    def quiet_window(*_args, **_kwargs):
        nonlocal quiet_calls
        quiet_calls += 1
        if quiet_calls == 2:
            raise failure

    original_run = benchmark.run
    monkeypatch.setattr(benchmark_clean_qwen, "Monitor", Monitor)
    monkeypatch.setattr(benchmark_clean_qwen, "quiet_window", quiet_window)
    monkeypatch.setattr(
        benchmark_clean_qwen, "paging", lambda: {"pageouts": 0, "swapouts": 0}
    )
    monkeypatch.setattr(
        benchmark,
        "run",
        lambda args: original_run(
            args, worker_factory=factory, fixture_factory=fixture_factory
        ),
    )
    output = tmp_path / "failed-report.json"
    assert (
        benchmark.main(
            [
                "--model",
                str(arguments.model),
                "--baseline",
                str(arguments.baseline),
                "--candidate",
                str(arguments.candidate),
                "--output",
                str(output),
                "--target-tokens",
                "64",
                "--context",
                "256",
                "--tokens",
                "8",
                "--mode",
                "timing",
            ]
        )
        == 1
    )
    expected = {
        "passed": False,
        "timing_accepted": False,
        "stage": "native_prefill_benchmark_failed",
        "failure_category": category,
    }
    report_text = output.read_text()
    captured = capsys.readouterr()
    assert json.loads(report_text) == json.loads(captured.out) == expected
    assert captured.err == ""
    assert "private-" not in report_text + captured.out
    assert output.stat().st_mode & 0o777 == 0o600
    assert quiet_calls == 2 and len(workers) == 2
    assert all(worker.closed for worker in workers)
    assert len(monitors) == 1 and monitors[0].closed


def test_source_drift_disqualifies_a_run(arguments, monkeypatch):
    seals = iter(["a" * 64, "b" * 64])
    monkeypatch.setattr(benchmark, "source_seal", lambda: next(seals))
    arguments.mode = "timing"
    report = benchmark.run(
        arguments, worker_factory=FakeWorker, fixture_factory=fixture_factory
    )
    assert (
        report["source_unchanged"] is False
        and report["passed"] is False
        and report["timing_accepted"] is False
    )


@pytest.mark.parametrize(
    "field",
    [
        "weights_sha256",
        "harness_sha256",
        "rendering_sources_sha256",
        "benchmark_dependencies_sha256",
    ],
)
def test_model_harness_and_rendering_drift_disqualify_run(
    arguments, monkeypatch, field
):
    original = benchmark.input_bindings
    calls = 0

    def bindings(model):
        nonlocal calls
        calls += 1
        value = original(model)
        if calls > 1:
            value[field] = (
                {"changed": "b" * 64}
                if field.endswith("sources_sha256")
                or field == "benchmark_dependencies_sha256"
                else "b" * 64
            )
        return value

    monkeypatch.setattr(benchmark, "input_bindings", bindings)
    arguments.mode = "timing"
    report = benchmark.run(
        arguments, worker_factory=FakeWorker, fixture_factory=fixture_factory
    )
    assert report["passed"] is False and report["timing_accepted"] is False


def test_guard_sleep_and_provenance_dependency_sources_are_bound(arguments):
    bindings = benchmark.input_bindings(arguments.model)
    assert set(bindings["benchmark_dependencies_sha256"]) == {
        "benchmark_clean_qwen",
        "benchmark_stock_engines",
        "benchmark_hybrid_context",
    }
    assert all(
        len(digest) == 64
        for digest in bindings["benchmark_dependencies_sha256"].values()
    )


@pytest.mark.parametrize(
    "phases,cpu_guard_exceeded,accepted",
    [
        ([], False, False),
        ([["measurement", "quiet"]], False, False),
        ([["measurement"]], False, True),
        ([["measurement"], ["measurement", "parity"]], True, False),
    ],
)
def test_quiet_timing_requires_actual_measurement_samples(
    arguments, monkeypatch, phases, cpu_guard_exceeded, accepted
):
    import benchmark_clean_qwen
    import benchmark_stock_engines

    class Monitor:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def set_phase(self, _phase):
            pass

        def result(self):
            return {
                "samples": [
                    {"phases": value, "active_processes": ["private-process-marker"]}
                    for value in phases
                ],
                "measurement_samples": len(phases),
                "sampling_failures": 0,
                "sampling_gap_exceeded": False,
                "cpu_guard_exceeded": cpu_guard_exceeded,
                "background_model_active": False,
                "thermal_guard_exceeded": False,
            }

    monkeypatch.setattr(benchmark_clean_qwen, "Monitor", Monitor)
    monkeypatch.setattr(
        benchmark_clean_qwen, "quiet_window", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        benchmark_clean_qwen, "paging", lambda: {"pageouts": 0, "swapouts": 0}
    )
    monkeypatch.setattr(
        benchmark_stock_engines,
        "sleep_events",
        lambda *_args: {"checked": True, "sleep_events": 0},
    )
    arguments.no_quiet, arguments.mode = False, "timing"
    report = benchmark.run(
        arguments, worker_factory=FakeWorker, fixture_factory=fixture_factory
    )
    assert report["passed"] is True and report["timing_accepted"] is accepted
    assert report["host_guard"]["measurement_samples"] == sum(
        value == ["measurement"] for value in phases
    )
    assert report["host_guard"]["guard_interval_samples"] == len(phases)
    assert report["host_guard"]["cpu_guard_exceeded"] is cpu_guard_exceeded
    assert "private-process-marker" not in json.dumps(report)


def test_late_dependency_drift_after_other_end_bindings_disqualifies_run(
    arguments, monkeypatch
):
    calls = 0
    original = benchmark.benchmark_dependency_bindings

    def dependencies():
        nonlocal calls
        calls += 1
        value = original()
        if calls >= 3:
            value["benchmark_stock_engines"] = "b" * 64
        return value

    monkeypatch.setattr(benchmark, "benchmark_dependency_bindings", dependencies)
    arguments.mode = "timing"
    report = benchmark.run(
        arguments, worker_factory=FakeWorker, fixture_factory=fixture_factory
    )
    assert report["benchmark_dependencies_unchanged"] is False
    assert report["passed"] is False and report["timing_accepted"] is False
