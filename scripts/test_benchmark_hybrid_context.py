from __future__ import annotations

import hashlib
import json
from argparse import Namespace

import pytest
from benchmark_hybrid_context import model_provenance, run
import benchmark_hybrid_context as context_benchmark


class FakeWorker:
    def __init__(self, _worker, _model, _timeout):
        self.generation_requests = []

    def read(self):
        return {"ready": True, "mode": "text-generation-v1", "eos_token_ids": [33]}

    def request(self, request):
        if request["operation"] == "tokenize":
            return {"token_ids": [11, 22]}
        self.generation_requests.append(request)
        return {
            "generated_token_ids": [31, 32],
            "sampled_tokens": 3,
            "finish_reason": "eos",
            "stop_token_id": 33,
            "output_tokens_per_second": 12.0,
        }

    def close(self):
        pass


@pytest.mark.parametrize("sampling_mode", ["greedy", "stochastic"])
def test_run_uses_exact_cycled_prefix_and_emits_only_hashes_and_aggregates(tmp_path, sampling_mode):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "native-source.json").write_text("{}")
    worker_path = tmp_path / "worker"
    worker_path.write_bytes(b"fake executable")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("public corpus text")
    output = tmp_path / "report.json"
    args = Namespace(
        model=model,
        worker=worker_path,
        output=output,
        corpus=corpus,
        input_tokens=5,
        tokens=4,
        replays=3,
        sampling=sampling_mode,
        temperature=1.0,
        top_k=20,
        top_p=0.95,
        seed=42,
    )
    workers = []

    def factory(*args):
        worker = FakeWorker(*args)
        workers.append(worker)
        return worker

    report = run(args, worker_factory=factory)

    assert report["input_tokens"] == 5
    assert report["input_sha256"] == hashlib.sha256(b"[11,22,11,22,11]").hexdigest()
    assert report["context_tokens"] == 9
    assert report["repeat_exact"] is True
    assert "host_metrics_monitor" not in report
    assert "host_metrics" not in report["replays"][0]
    assert report["replays"][0]["sampled_tokens"] == 3
    assert report["replays"][0]["output_tokens"] == 2
    assert report["replays"][0]["finish_reason"] == "eos"
    assert "generated_token_ids" not in json.dumps(report)
    assert "public corpus text" not in json.dumps(report)
    assert len(workers[0].generation_requests) == 3
    assert all(
        request["input_token_ids"] == [11, 22, 11, 22, 11]
        and request["context_tokens"] == 9
        and request["max_new_tokens"] == 4
        for request in workers[0].generation_requests
    )
    expected_sampling = (
        {"mode": "greedy"}
        if sampling_mode == "greedy"
        else {"mode": "stochastic", "temperature": 1.0, "top_k": 20, "top_p": 0.95}
    )
    assert all(request["sampling"] == expected_sampling for request in workers[0].generation_requests)
    if sampling_mode == "stochastic":
        assert report["seed"] == 42
        assert all(request["seed"] == 42 for request in workers[0].generation_requests)
    assert json.loads(output.read_text()) == report


def test_host_metrics_are_opt_in_and_use_fixed_aggregate_fields(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "native-source.json").write_text("{}")
    worker_path = tmp_path / "worker"
    worker_path.write_bytes(b"fake executable")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("public corpus text")
    args = Namespace(
        model=model,
        worker=worker_path,
        output=tmp_path / "report.json",
        corpus=corpus,
        input_tokens=5,
        tokens=4,
        replays=3,
        sampling="greedy",
        temperature=1.0,
        top_k=20,
        top_p=0.95,
        seed=42,
        host_metrics=True,
    )

    class FakeMonitor:
        def __init__(self, _worker_pid):
            self.markers = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def begin_request(self):
            marker = len(self.markers)
            self.markers.append(marker)
            return marker

        def end_request(self, _marker):
            pass

        def request_result(self, marker):
            return {
                "sample_interval_count": marker,
                "gpu_counter_unit": "unverified IORegistry accumulatedGPUTime units",
            }

        def summary(self):
            return {"process_names_and_pids_persisted": False}

    report = run(
        args,
        worker_factory=FakeWorker,
        host_metrics_factory=FakeMonitor,
    )

    assert report["host_metrics_monitor"]["process_names_and_pids_persisted"] is False
    assert [item["host_metrics"]["sample_interval_count"] for item in report["replays"]] == [
        0,
        1,
        2,
    ]
    serialized = json.dumps(report)
    assert "WindowServer" not in serialized
    assert "Claude Helper" not in serialized
    assert '"pid"' not in serialized


def test_native_source_provenance_keeps_legacy_hash_and_labels_receipt(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    receipt = model / "native-source.json"
    receipt.write_text("{}")

    result = model_provenance(model)

    assert result["model_artifact_kind"] == "native_source"
    assert result["model_artifact_receipt_file"] == "native-source.json"
    assert result["model_artifact_receipt_sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert result["source_receipt_sha256"] == result["model_artifact_receipt_sha256"]


def test_conversion_provenance_labels_conversion_and_source_manifest_hash(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    manifest_hash = "a" * 64
    receipt_path = model / "conversion.json"
    receipt_path.write_text(json.dumps({"status": "ok", "source_manifest_sha256": manifest_hash}))

    result = model_provenance(model)

    assert result["model_artifact_kind"] == "stark_conversion"
    assert result["model_artifact_receipt_file"] == "conversion.json"
    assert result["model_artifact_receipt_sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    assert result["source_manifest_sha256"] == manifest_hash
    assert "source_receipt_sha256" not in result


def test_conversion_provenance_rejects_missing_source_manifest_hash(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "conversion.json").write_text(json.dumps({"status": "ok"}))

    with pytest.raises(ValueError, match="conversion_source_manifest_hash_invalid"):
        model_provenance(model)


@pytest.fixture
def context_args(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "native-source.json").write_text("{}")
    worker = tmp_path / "worker"
    worker.write_bytes(b"fake executable")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("public corpus text")
    return Namespace(
        model=model, worker=worker, corpus=corpus, output=tmp_path / "report.json",
        input_tokens=5, tokens=4, replays=5, sampling="greedy", screened=True,
        max_attempts=24, host_metrics=True,
    )


class FakeScreenMonitor:
    def __init__(self):
        self.failures = 0
        self.rows = []
        self.phases = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def set_phase(self, phase):
        self.phases.append(phase)
        if phase.startswith("attempt_"):
            self.rows.append({
                "phases": [phase], "external_cpu_percent": 0,
                "max_model_cpu_percent": 0, "thermal_state": 0,
                "active_processes": [], "interval_seconds": 0.5,
            })


class FakeHostSampler:
    def __init__(self, _pid):
        self.markers = []
        self.ended = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.closed = True

    def begin_request(self):
        marker = len(self.markers)
        self.markers.append(marker)
        return marker

    def end_request(self, marker):
        self.ended.append(marker)

    def request_result(self, marker):
        assert self.closed
        assert marker in self.ended
        return {"sample_interval_count": marker}

    def summary(self):
        return {"process_names_and_pids_persisted": False}


def screened_fakes(monkeypatch, *, rejected=(), fail_admission=None, fail_request=None):
    """Run the REAL shared screen_attempt with synthetic observations/paging."""
    from benchmark_glm_mlx_serve import screen_attempt

    clock = [0.0]
    durations = [100.0, 999.0, 4.0, 8.0, 777.0, 6.0, 2.0]
    rates = [1.0, 999.0, 20.0, 40.0, 777.0, 30.0, 10.0]
    monitor = FakeScreenMonitor()
    host = FakeHostSampler(None)
    admissions = []
    paging_calls = [0]
    workers = []

    class ScriptWorker(FakeWorker):
        def __init__(self, *args):
            super().__init__(*args)
            self.closed = False

        def request(self, request):
            if request["operation"] == "tokenize":
                return super().request(request)
            index = len(self.generation_requests)
            result = super().request(request)
            clock[0] += durations[index % len(durations)]
            if fail_request == index:
                raise RuntimeError("PRIVATE_WORKER_EXCEPTION_PAYLOAD")
            if index in rejected:
                result["generated_token_ids"] = [91, 92]
            result["output_tokens_per_second"] = rates[index % len(rates)]
            result["input_tokens_per_second"] = rates[index % len(rates)] * 10
            result["prefill_seconds"] = 0.5
            result["decode_seconds"] = 0.25
            result["generation_seconds"] = 0.75
            return result

        def close(self):
            self.closed = True

    def factory(*args):
        worker = ScriptWorker(*args)
        workers.append(worker)
        return worker

    def admission(observed):
        assert observed is monitor
        admissions.append(len(workers[0].generation_requests))
        observed.set_phase("quiet")
        if len(admissions) == fail_admission:
            raise TimeoutError("PRIVATE_ADMISSION_EXCEPTION_PAYLOAD")
        return {"streak_seconds": 10.0, "samples": 20}

    def paging():
        call = paging_calls[0]
        paging_calls[0] += 1
        attempt = call // 2
        total = sum(index < attempt for index in rejected)
        return {"pageouts": total + int(call % 2 == 1 and attempt in rejected), "swapouts": 0}

    monkeypatch.setattr(context_benchmark.time, "monotonic", lambda: clock[0])
    # The real shared screen policy's 0.6s sampler drain is simulated; no host
    # process, native worker, clocks, GPU counters or sleeps are used here.
    monkeypatch.setattr(context_benchmark.time, "sleep", lambda _seconds: None)
    kwargs = dict(
        worker_factory=factory, host_metrics_factory=lambda _pid: host,
        monitor_factory=lambda: monitor, paging_reader=paging,
        admission_fn=admission, screen_fn=screen_attempt,
    )
    return kwargs, workers, monitor, host, admissions


def test_screened_rejects_retries_and_aggregates_only_accepted_warm_rows(context_args, monkeypatch):
    kwargs, workers, monitor, host, admissions = screened_fakes(monkeypatch, rejected={1, 4})
    report = run(context_args, **kwargs)

    assert [row["attempt"] for row in report["replays"]] == [0, 2, 3, 5, 6]
    assert [row["attempt"] for row in report["excluded_attempts"]] == [1, 4]
    assert [request["request_id"] for request in workers[0].generation_requests] == list(range(1, 8))
    assert admissions == list(range(1, 7))
    assert report["attempt_count"] == 7
    assert report["warmup_replays"] == 1
    assert report["median_wall_seconds"] == 6.0
    assert report["median_warm_wall_seconds"] == 5.0
    assert report["median_warm_decode_tokens_per_second"] == 25.0
    assert report["median_warm_prefill_tokens_per_second"] == 250.0
    assert report["repeat_exact"] is True
    assert report["all_attempts_repeat_exact"] is False
    assert report["passed"] is True
    for row in report["replays"] + report["excluded_attempts"]:
        assert row["host_metrics"]["sample_interval_count"] == row["attempt"]
        assert len(row["output_sha256"]) == 64
        assert row["warmup"] == (row["attempt"] == 0)
        if not row["warmup"]:
            assert row["admission"]["streak_seconds"] == 10
    for row in report["excluded_attempts"]:
        assert row["exclusion_reason"] == "host_screen_ineligible"
        assert row["host_screen"]["paging_delta"]["pageouts"] == 1
    assert host.ended == list(range(7))
    assert host.closed and monitor.closed and workers[0].closed
    assert json.loads(context_args.output.read_text()) == report
    assert "generated_token_ids" not in context_args.output.read_text()
    assert "public corpus text" not in context_args.output.read_text()


def test_screened_attempt_bound_preserves_partial_receipt_and_cleans_up(context_args, monkeypatch):
    context_args.max_attempts = 5
    kwargs, workers, monitor, host, _ = screened_fakes(monkeypatch, rejected={1, 2, 3, 4})
    with pytest.raises(context_benchmark.BenchmarkFailure, match="native_context_benchmark_failed"):
        run(context_args, **kwargs)
    report = json.loads(context_args.output.read_text())
    assert report["passed"] is False
    assert report["failed_stage"] == "aggregation"
    assert report["error_type"] == "ValueError"
    assert len(report["replays"]) == 1
    assert len(report["excluded_attempts"]) == 4
    assert report["attempt_count"] == 5
    assert "median_warm_wall_seconds" not in report
    assert host.closed and monitor.closed and workers[0].closed
    assert all("host_metrics" in row and "output_sha256" in row for row in report["excluded_attempts"])


@pytest.mark.parametrize("failed_stage", ["quiet_admission", "generation", "host_screen"])
def test_screened_failures_preserve_sanitized_partial_receipt(context_args, monkeypatch, failed_stage):
    kwargs, workers, monitor, host, _ = screened_fakes(
        monkeypatch,
        fail_admission=2 if failed_stage == "quiet_admission" else None,
        fail_request=2 if failed_stage == "generation" else None,
    )
    if failed_stage == "host_screen":
        original = kwargs["screen_fn"]

        def fail_screen(*args):
            if args[1] == "attempt_2":
                raise RuntimeError("PRIVATE_SCREEN_EXCEPTION_PAYLOAD")
            return original(*args)

        kwargs["screen_fn"] = fail_screen
    with pytest.raises(context_benchmark.BenchmarkFailure):
        run(context_args, **kwargs)
    text = context_args.output.read_text()
    report = json.loads(text)
    assert report["passed"] is False
    assert report["error"] == "native_context_benchmark_failed"
    assert report["failed_stage"] == failed_stage
    assert report["error_type"] == ("TimeoutError" if failed_stage == "quiet_admission" else "RuntimeError")
    assert report["failed_attempt"]["attempt"] == 2
    assert len(report["replays"]) == 2
    assert "PRIVATE_" not in text
    assert "generated_token_ids" not in text
    assert "public corpus text" not in text
    if failed_stage != "quiet_admission":
        assert report["failed_attempt"]["host_metrics"]["sample_interval_count"] == 2
        assert host.ended == [0, 1, 2]
    if failed_stage == "host_screen":
        assert len(report["failed_attempt"]["output_sha256"]) == 64
    assert host.closed and monitor.closed and workers[0].closed


def test_screened_cli_failure_is_nonzero_and_has_no_exception_payload(context_args, monkeypatch, capsys):
    kwargs, workers, monitor, host, _ = screened_fakes(monkeypatch, fail_admission=1)
    original = run
    monkeypatch.setattr(context_benchmark, "run", lambda args: original(args, **kwargs))
    result = context_benchmark.main([
        "--model", str(context_args.model), "--worker", str(context_args.worker),
        "--corpus", str(context_args.corpus), "--output", str(context_args.output),
        "--input-tokens", "5", "--tokens", "4", "--screened", "--host-metrics",
    ])
    assert result == 1
    assert json.loads(context_args.output.read_text())["failed_stage"] == "quiet_admission"
    assert "PRIVATE_" not in capsys.readouterr().out
    assert workers[0].closed and monitor.closed and host.closed


def test_default_cli_and_unscreened_policy_preserve_five_replays(context_args, monkeypatch):
    seen = []
    monkeypatch.setattr(context_benchmark, "run", lambda args: seen.append(args) or {"passed": True})
    assert context_benchmark.main([
        "--model", str(context_args.model), "--worker", str(context_args.worker),
        "--corpus", str(context_args.corpus), "--output", str(context_args.output),
    ]) == 0
    assert seen[0].replays == 5 and seen[0].max_attempts == 24
    assert seen[0].screened is False
    context_args.screened = False
    context_args.host_metrics = False

    def forbidden():
        pytest.fail("unscreened mode started the screen monitor")

    report = run(context_args, worker_factory=FakeWorker, monitor_factory=forbidden)
    assert len(report["replays"]) == 5
    assert report["excluded_attempts"] == []
    assert report["warmup_replays"] == 1


def test_verified_weight_metadata_and_chunk_flag_survive_screening(context_args, monkeypatch):
    import benchmark_hybrid

    context_args.verified_weights = context_args.model / "verified.json"
    calls = []
    identity = {"receipt_sha256": "a" * 64, "files": {}}
    monkeypatch.setattr(benchmark_hybrid, "weight_identity", lambda *args: calls.append(args) or identity)
    monkeypatch.setattr(benchmark_hybrid, "hardware_identity", lambda: {"model": "fixture_hardware"})
    monkeypatch.setattr(benchmark_hybrid, "checkout_identity", lambda: {"head": "fixture_source"})
    monkeypatch.setenv("STARK_GLM53_PREFILL_CHUNK", "1")
    monkeypatch.setenv("STARK_GLM53_PREFILL_MMA", "0")
    kwargs, _, _, _, _ = screened_fakes(monkeypatch)
    report = run(context_args, **kwargs)
    assert len(calls) == 2
    assert report["weights"] == identity
    assert report["hardware"] == {"model": "fixture_hardware"}
    assert report["harness_checkout"] == {"head": "fixture_source"}
    assert report["prefill_configuration_flags"]["STARK_GLM53_PREFILL_CHUNK"] == "1"
    assert report["prefill_configuration_flags"]["STARK_GLM53_PREFILL_MMA"] == "0"
    assert len(report["harness_sha256"]) == 64


def test_observer_cleanup_failure_still_closes_worker_and_preserves_metrics(context_args, monkeypatch):
    kwargs, workers, monitor, host, _ = screened_fakes(monkeypatch)

    def fail_stop(*_args):
        host.closed = True
        raise RuntimeError("PRIVATE_CLEANUP_EXCEPTION_PAYLOAD")

    host.__exit__ = fail_stop
    with pytest.raises(context_benchmark.BenchmarkFailure):
        run(context_args, **kwargs)
    text = context_args.output.read_text()
    report = json.loads(text)
    assert report["failed_stage"] == "host_metrics_stop"
    assert report["cleanup_failures"] == [{"stage": "host_metrics_stop", "error_type": "RuntimeError"}]
    assert all("host_metrics" in row for row in report["replays"])
    assert len(report["replays"]) == 5
    assert workers[0].closed and monitor.closed and host.closed
    assert "PRIVATE_" not in text


def test_screen_monitor_diagnostics_whitelist_aggregates_and_age_before_cleanup():
    class Monitor:
        started = 100.0
        rows = [
            {"end_seconds": 5.0, "interval_seconds": 0.6, "external_cpu_percent": 10.0,
             "max_model_cpu_percent": 2.0, "gpu_device_utilization_percent": 3,
             "thermal_state": 0, "active_processes": [{"pid": 123, "name": "PRIVATE_PROCESS", "args": "PRIVATE_ARGS"}],
             "unknown_field": "PRIVATE_HOST_DATA"},
            {"end_seconds": 6.0, "interval_seconds": 0.8, "external_cpu_percent": 30.0,
             "max_model_cpu_percent": 7.0, "gpu_device_utilization_percent": 9,
             "thermal_state": 1, "active_processes": [{"pid": 456, "name": "clang"}]},
        ]

    result = context_benchmark.screen_monitor_diagnostics(Monitor(), now=108.0)
    assert result == {
        "data_available": True, "sample_count": 2, "latest_sample_age_seconds": 2.0,
        "max_sample_interval_seconds": 0.8, "thermal_state_max": 1, "compiler_active": True,
        "external_cpu_percent_min": 10.0, "external_cpu_percent_max": 30.0,
        "external_model_cpu_percent_min": 2.0, "external_model_cpu_percent_max": 7.0,
        "gpu_device_utilization_percent_min": 3, "gpu_device_utilization_percent_max": 9,
    }
    serialized = json.dumps(result)
    assert all(private not in serialized for private in ("PRIVATE_", "pid", "name", "args", "rows", "clang"))


def test_screen_monitor_diagnostics_supports_fake_monitor_without_observations():
    result = context_benchmark.screen_monitor_diagnostics(object(), now=108.0)
    assert result["data_available"] is False
    assert all(value is None for key, value in result.items() if key != "data_available")
    assert context_benchmark.screen_monitor_diagnostics(FakeScreenMonitor())["sample_count"] == 0


@pytest.mark.parametrize("fail_admission", [None, 1])
def test_screen_monitor_diagnostics_persist_on_success_and_partial_receipts(context_args, monkeypatch, fail_admission):
    kwargs, _, monitor, _, _ = screened_fakes(monkeypatch, fail_admission=fail_admission)
    if fail_admission is None:
        run(context_args, **kwargs)
    else:
        with pytest.raises(context_benchmark.BenchmarkFailure):
            run(context_args, **kwargs)
    report = json.loads(context_args.output.read_text())
    diagnostics = report["screening_monitor"]
    assert diagnostics["diagnostic_snapshot_stage"] == "before_observer_cleanup"
    assert diagnostics["data_available"] is True
    assert diagnostics["sample_count"] == (5 if fail_admission is None else 1)
    assert diagnostics["external_cpu_percent_min"] == 0
    assert diagnostics["external_cpu_percent_max"] == 0
    assert diagnostics["max_sample_interval_seconds"] == 0.5
    assert diagnostics["sampling_failures"] == 0
    assert diagnostics["latest_sample_age_seconds"] is None
    assert monitor.closed
