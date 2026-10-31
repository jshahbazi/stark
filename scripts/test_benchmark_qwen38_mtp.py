from __future__ import annotations

import json
import os
from argparse import Namespace

import benchmark_qwen38_mtp as benchmark
import pytest
from benchmark_qwen38_mtp import run


class FakeWorker:
    verification = "batched-decode-exact-v1"

    def __init__(self, _worker, _model, _timeout, *, env):
        self.env = env
        self.requests = []
        self.closed = False
        self.route_counts = {"off": 0, "on": 0}

    def read(self):
        return {
            "ready": True,
            "mode": "text-generation-v1",
            "eos_token_ids": [99],
            "max_context_tokens": 16384,
            "vocab_size": 100,
            "mtp": {
                "supported": True,
                "max_drafts": 3,
                "sampling": "exact-rejection-v1",
                "verification": self.verification,
            },
        }

    def request(self, request):
        self.requests.append(request)
        if request["operation"] == "tokenize":
            return {"token_ids": [11, 22]}
        enabled = request["mtp_drafts"] > 0
        route = "on" if enabled else "off"
        self.route_counts[route] += 1
        # Stable per route, with deliberately distinct routes to ensure the
        # harness does not demand cross-route trajectory equality.
        generated = [31, 32, 33] if enabled else [41, 42, 43]
        return {
            "generated_token_ids": generated,
            "sampled_tokens": 4,
            "finish_reason": "eos",
            "stop_token_id": 99,
            "output_tokens_per_second": 8.0,
            "decode_seconds": 0.5,
            "mtp": {
                "enabled": enabled,
                "drafts": request["mtp_drafts"],
                "sampling": "exact-rejection-v1",
                "verification": self.verification,
                "stats": {
                    "rounds": 2 if enabled else 0,
                    "proposed": 3 if enabled else 0,
                    "accepted": 2 if enabled else 0,
                    "verified_rows": 3 if enabled else 0,
                    "draft_seconds": 0.1 if enabled else 0.0,
                    "verify_seconds": 0.2 if enabled else 0.0,
                    "rollback_seconds": 0.01 if enabled else 0.0,
                },
            },
        }

    def close(self):
        self.closed = True


def make_args(tmp_path, *, drafts=2, blocks=4, top_k=20):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen4_exp","lily":{"mtp":{}}}')
    (model / "native-source.json").write_text(
        json.dumps(
            {
                "weights_source": {
                    "repo": "fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl",
                    "revision": "0f94842e3682b49acace4daeb4cf2e65a6f94f83",
                }
            }
        )
    )
    worker = tmp_path / "stark-generate"
    worker.write_bytes(b"synthetic worker executable")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("safe corpus material")
    return Namespace(
        model=model,
        worker=worker,
        corpus=corpus,
        output=tmp_path / "report.json",
        input_tokens=5,
        tokens=4,
        blocks=blocks,
        drafts=drafts,
        top_k=top_k,
        top_p=0.95,
        seed=42,
    )


@pytest.mark.parametrize("top_k", [20, 0])
def test_same_worker_pairs_routes_and_emits_private_aggregates(tmp_path, top_k):
    args = make_args(tmp_path, top_k=top_k)
    workers = []

    def factory(*factory_args, **factory_kwargs):
        worker = FakeWorker(*factory_args, **factory_kwargs)
        workers.append(worker)
        return worker

    report = run(args, worker_factory=factory)
    worker = workers[0]
    assert worker.closed
    assert "STARK_QWEN38_PROFILE_OUTPUT" not in worker.env
    assert report["sampling"] == {
        "mode": "stochastic",
        "temperature": 1.0,
        "top_p": 0.95,
        **({} if top_k == 0 else {"top_k": top_k}),
    }
    generated_requests = [
        request for request in worker.requests if request["operation"] == "generate"
    ]
    assert [request["mtp_drafts"] for request in generated_requests[:2]] == [0, 2]
    assert len(generated_requests) == 2 + 2 * args.blocks
    assert all(set(pair["order"]) == {"off", "on"} for pair in report["pairs"])
    leading = [pair["order"][0] for pair in report["pairs"]]
    other = "on" if leading[0] == "off" else "off"
    assert leading == [leading[0], other, other, leading[0]]
    assert all(
        pair["routes"]["off"]["requested_drafts"] == 0
        and pair["routes"]["on"]["requested_drafts"] == 2
        for pair in report["pairs"]
    )
    assert all(
        request["seed"] == 42
        and request["sampling"] == report["sampling"]
        and request["input_token_ids"] == [11, 22, 11, 22, 11]
        for request in worker.requests
        if request["operation"] == "generate"
    )
    assert report["route_summary"]["off"]["deterministic_within_route"]
    assert report["route_summary"]["on"]["deterministic_within_route"]
    assert report["cross_route_hashes_equal"] is False
    assert report["route_summary"]["off"]["pooled_decode_steps"] == 12
    assert report["route_summary"]["on"]["pooled_decode_steps"] == 12
    assert report["route_summary"]["off"]["pooled_decode_tokens_per_second"] == 6
    assert report["route_summary"]["on"]["pooled_decode_tokens_per_second"] == 6
    assert report["route_summary"]["on"]["mtp_totals"]["accepted"] == 8
    serialized = json.dumps(report)
    assert "generated_token_ids" not in serialized
    assert "safe corpus material" not in serialized
    assert "[31, 32, 33]" not in serialized
    assert json.loads(args.output.read_text()) == report
    assert args.output.stat().st_mode & 0o777 == 0o600


def test_rejects_worker_without_mtp_before_tokenization(tmp_path):
    args = make_args(tmp_path)
    worker = FakeWorker(args.worker, args.model, 0, env={})
    worker.read = lambda: {"ready": True, "mode": "text-generation-v1"}
    with pytest.raises(ValueError, match="mtp_not_supported"):
        run(args, worker_factory=lambda *a, **kw: worker)
    assert worker.closed
    assert not args.output.exists()
    assert not worker.requests


def test_refuses_existing_report_without_overwrite(tmp_path):
    args = make_args(tmp_path)
    args.output.write_text("preserve")
    with pytest.raises(ValueError, match="output_exists"):
        run(args)
    assert args.output.read_text() == "preserve"


@pytest.mark.parametrize("blocks", [0, 3, 18])
def test_rejects_unbalanced_or_excessive_block_count(tmp_path, blocks):
    args = make_args(tmp_path, blocks=blocks)
    with pytest.raises(ValueError, match="invalid_benchmark_limits"):
        run(args, worker_factory=FakeWorker)


@pytest.mark.parametrize("tokens", [1, 4097])
def test_rejects_output_limits_outside_benchmark_range(tmp_path, tokens):
    args = make_args(tmp_path)
    args.tokens = tokens
    with pytest.raises(ValueError, match="invalid_benchmark_limits"):
        run(args, worker_factory=FakeWorker)


@pytest.mark.parametrize(
    "bad_reply", ["missing_decode", "out_of_vocab", "off_activity", "wrong_verifier"]
)
def test_rejects_invalid_timing_ids_or_disabled_route_stats(tmp_path, bad_reply):
    args = make_args(tmp_path)

    class InvalidWorker(FakeWorker):
        def request(self, request):
            result = super().request(request)
            if request["operation"] != "generate":
                return result
            if bad_reply == "missing_decode":
                result.pop("decode_seconds")
            elif bad_reply == "out_of_vocab":
                result["generated_token_ids"] = [31, 32, 100]
                result["sampled_tokens"] = 4
            elif bad_reply == "off_activity" and request["mtp_drafts"] == 0:
                result["mtp"]["stats"]["rounds"] = 1
            elif bad_reply == "wrong_verifier":
                result["mtp"]["verification"] = "serial-control"
            return result

    worker = InvalidWorker(args.worker, args.model, 0, env={})
    with pytest.raises(ValueError):
        run(args, worker_factory=lambda *a, **kw: worker)
    assert worker.closed
    assert not args.output.exists()


def test_refuses_worker_without_batched_verifier_identity(tmp_path):
    args = make_args(tmp_path)
    worker = FakeWorker(args.worker, args.model, 0, env={})
    ready = worker.read()
    ready["mtp"].pop("verification")
    worker.read = lambda: ready
    with pytest.raises(ValueError, match="mtp_verifier_unavailable"):
        run(args, worker_factory=lambda *a, **kw: worker)
    assert worker.closed and not worker.requests


def make_official_args(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    (args.model / "native-source.json").unlink()
    (args.model / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "text_config": {"mtp": {}}})
    )
    (args.model / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    f"language_model.mtp.fixture_{i}": "fixture-0" for i in range(72)
                }
            }
        )
    )
    for index in range(111):
        (args.model / f"fixture-{index}").write_bytes(b"model-free fixture")
    paths = list(args.model.iterdir())
    receipt = {
        "schema": "stark-shared-checkpoint-payload-hashes-v1",
        "completed": True,
        "all_manifest_lfs_payloads_verified": True,
        "repository": benchmark.OFFICIAL_REPOSITORY,
        "revision": benchmark.OFFICIAL_REVISION,
        "manifest_sha256": benchmark.OFFICIAL_MANIFEST_SHA256,
        "checkpoint_snapshot": {
            path.name: [
                path.stat().st_size,
                path.stat().st_mtime_ns,
                path.stat().st_ino,
            ]
            for path in paths
        },
        "all_file_sha256": {path.name: benchmark.sha256_file(path) for path in paths},
    }
    args.checkpoint_receipt = tmp_path / "checkpoint-receipt.json"
    args.checkpoint_receipt.write_text(json.dumps(receipt))
    monkeypatch.setattr(
        benchmark,
        "OFFICIAL_RECEIPT_SHA256",
        benchmark.sha256_file(args.checkpoint_receipt),
    )
    return args


def test_official_receipt_and_serial_verifier_are_bound_to_report(
    tmp_path, monkeypatch
):
    args = make_official_args(tmp_path, monkeypatch)

    class SerialWorker(FakeWorker):
        verification = "serial-decode-exact-v1"

    report = run(args, worker_factory=SerialWorker)
    assert report["verification"] == "serial-decode-exact-v1"
    assert all(
        item["mtp"]["verification"] == report["verification"]
        for item in report["requests"]
    )
    assert report["checkpoint"]["repo"] == benchmark.OFFICIAL_REPOSITORY
    assert report["checkpoint"]["revision"] == benchmark.OFFICIAL_REVISION
    assert report["checkpoint"]["manifest_sha256"] == benchmark.OFFICIAL_MANIFEST_SHA256
    assert report["checkpoint_receipt_sha256"] == benchmark.sha256_file(
        args.checkpoint_receipt
    )
    assert report["checkpoint"]["files"] == 113
    assert report["checkpoint"]["payloads_rehashed_this_run"] is False
    assert "source_receipt_sha256" not in report
    assert "checkpoint_snapshot" not in json.dumps(report)


def test_rejects_official_snapshot_without_explicit_receipt(tmp_path, monkeypatch):
    args = make_official_args(tmp_path, monkeypatch)
    args.checkpoint_receipt = None
    with pytest.raises(ValueError, match="model_metadata_unavailable"):
        run(args, worker_factory=FakeWorker)
    assert not args.output.exists()


@pytest.mark.parametrize(
    "change", ["receipt", "file_identity", "metadata_bytes", "manifest", "mtp_index"]
)
def test_rejects_unbound_official_checkpoint(tmp_path, monkeypatch, change):
    args = make_official_args(tmp_path, monkeypatch)
    if change == "receipt":
        args.checkpoint_receipt.write_text(args.checkpoint_receipt.read_text() + " ")
        expected = "checkpoint_receipt_hash_mismatch"
    elif change == "file_identity":
        (args.model / "fixture-0").write_bytes(b"changed fixture length")
        expected = "checkpoint_snapshot_identity_changed"
    elif change == "metadata_bytes":
        path = args.model / "config.json"
        stat = path.stat()
        # Keep the admitted structure and byte count while changing its bytes.
        path.write_text(path.read_text().replace('"mtp": {}', '"mtp":{ }'))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        expected = "checkpoint_metadata_hash_mismatch"
    else:
        receipt = json.loads(args.checkpoint_receipt.read_text())
        if change == "manifest":
            receipt["manifest_sha256"] = "0" * 64
        else:
            path = args.model / "model.safetensors.index.json"
            path.write_text(json.dumps({"weight_map": {}}))
        args.checkpoint_receipt.write_text(json.dumps(receipt))
        monkeypatch.setattr(
            benchmark,
            "OFFICIAL_RECEIPT_SHA256",
            benchmark.sha256_file(args.checkpoint_receipt),
        )
        expected = "unexpected_qwen38_checkpoint"
    with pytest.raises(ValueError, match=expected):
        run(args, worker_factory=FakeWorker)
    assert not args.output.exists()


def test_response_cannot_switch_between_known_exact_verifiers(tmp_path):
    args = make_args(tmp_path)

    class SwitchedWorker(FakeWorker):
        def request(self, request):
            result = super().request(request)
            if request["operation"] == "generate":
                result["mtp"]["verification"] = "serial-decode-exact-v1"
            return result

    worker = SwitchedWorker(args.worker, args.model, 0, env={})
    with pytest.raises(ValueError, match="mtp_verifier_acknowledgement_mismatch"):
        run(args, worker_factory=lambda *a, **kw: worker)
    assert worker.closed and not args.output.exists()
