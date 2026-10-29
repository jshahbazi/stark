from __future__ import annotations

import hashlib
import json
from argparse import Namespace

import pytest
from benchmark_hybrid_context import run


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
