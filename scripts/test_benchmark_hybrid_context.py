from __future__ import annotations

import hashlib
import json
from argparse import Namespace

import pytest
from benchmark_hybrid_context import model_provenance, run, validate_args


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


@pytest.fixture
def context_args(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "native-source.json").write_text("{}")
    worker = tmp_path / "worker"
    worker.write_bytes(b"fake executable")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("synthetic context corpus")
    return Namespace(
        model=model,
        worker=worker,
        output=tmp_path / "report.json",
        corpus=corpus,
        input_tokens=65280,
        tokens=256,
        replays=3,
        sampling="greedy",
    )


@pytest.mark.parametrize(
    "input_tokens, output_tokens, accepted",
    [(65280, 256, True), (65535, 1, True), (65281, 256, False), (65536, 1, False)],
)
def test_context_tool_envelope_includes_exactly_64k(input_tokens, output_tokens, accepted):
    args = Namespace(input_tokens=input_tokens, tokens=output_tokens, replays=3)
    if accepted:
        validate_args(args)
    else:
        with pytest.raises(ValueError, match="^context_limit_exceeded$"):
            validate_args(args)


@pytest.mark.parametrize(
    "advertisement, input_tokens, accepted",
    [
        ({}, 16128, True),
        ({}, 16129, False),
        ({"max_context_tokens": 16384}, 16128, True),
        ({"max_context_tokens": 16384}, 65280, False),
        ({"max_context_tokens": 65536}, 65280, True),
    ],
)
def test_loaded_context_capability_is_checked_before_tokenization(context_args, advertisement, input_tokens, accepted):
    context_args.input_tokens = input_tokens
    workers = []

    class ContextWorker(FakeWorker):
        def __init__(self, *args):
            super().__init__(*args)
            self.operations = []
            self.closed = False
            workers.append(self)

        def read(self):
            return {**super().read(), **advertisement}

        def request(self, request):
            self.operations.append(request["operation"])
            return super().request(request)

        def close(self):
            self.closed = True

    if accepted:
        report = run(context_args, worker_factory=ContextWorker)
        assert report["context_tokens"] == input_tokens + 256
        assert report["input_tokens"] == input_tokens
        assert report["repeat_exact"] is True
        assert workers[0].operations == ["tokenize", "generate", "generate", "generate"]
        assert all(
            len(request["input_token_ids"]) == input_tokens
            and request["context_tokens"] == input_tokens + 256
            and request["max_new_tokens"] == 256
            for request in workers[0].generation_requests
        )
        assert "synthetic context corpus" not in json.dumps(report)
        assert "input_token_ids" not in json.dumps(report)
    else:
        with pytest.raises(ValueError, match="^worker_context_limit_exceeded$"):
            run(context_args, worker_factory=ContextWorker)
        assert workers[0].operations == []
        assert not context_args.output.exists()
    assert workers[0].closed


@pytest.mark.parametrize("limit", [True, False, 0, -1, 65537, 16384.0, "65536", None])
def test_malformed_loaded_context_limit_fails_before_tokenization(context_args, limit):
    workers = []

    class InvalidContextWorker(FakeWorker):
        def __init__(self, *args):
            super().__init__(*args)
            self.operations = []
            self.closed = False
            workers.append(self)

        def read(self):
            return {**super().read(), "max_context_tokens": limit}

        def request(self, request):
            self.operations.append(request["operation"])
            return super().request(request)

        def close(self):
            self.closed = True

    with pytest.raises(ValueError, match="^worker_context_limit_invalid$"):
        run(context_args, worker_factory=InvalidContextWorker)
    assert workers[0].operations == []
    assert workers[0].closed
    assert not context_args.output.exists()


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
