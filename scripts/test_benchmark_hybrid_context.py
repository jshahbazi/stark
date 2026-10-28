from __future__ import annotations

import hashlib
import json
from argparse import Namespace

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


def test_run_uses_exact_cycled_prefix_and_emits_only_hashes_and_aggregates(tmp_path):
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
    assert json.loads(output.read_text()) == report
