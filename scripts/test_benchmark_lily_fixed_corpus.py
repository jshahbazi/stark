"""Model-free tests for scripts/benchmark_lily_fixed_corpus.py."""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import benchmark_lily_fixed_corpus as bench
import pytest

# Pinned output of the Python port for Sushi's seed (0x5a5ac0de). A change to
# these hashes means the generator no longer produces the recorded corpus.
FIRST_CHUNKS_SHA256 = (
    (24901, "a730d43470790234cc6da3487ab6cdd2c5b75bbe275ca33c73682c224c62d529"),
    (27232, "d94a9a17d47d8c91b4d02de71dd9248b10615cc24a6a0441e3d4fc80428fe367"),
    (25495, "36a0dc2a6b3b6c34301b403a3df38d0a61663d13875cb10600d3a30ff87dfd05"),
)
SENTINEL_TEXT = "PRIVATE-OUTPUT-SENTINEL"


def chunks(seed=bench.CORPUS_SEED, count=3):
    stream = bench.code_chunks(seed)
    return [next(stream) for _ in range(count)]


def test_splitmix64_matches_the_reference_sequence():
    # Reference values of SplitMix64 seeded with 0 (Vigna's published generator).
    rng = bench.SplitMix64(0)
    assert [rng.next() for _ in range(3)] == [
        0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_code_corpus_is_pinned_by_hash_for_the_sushi_seed():
    for text, (size, digest) in zip(chunks(), FIRST_CHUNKS_SHA256, strict=True):
        data = text.encode()
        assert len(data) == size
        assert hashlib.sha256(data).hexdigest() == digest


def test_code_corpus_is_deterministic_and_seed_dependent():
    first, second = chunks(), chunks()
    assert first == second
    assert chunks(bench.CORPUS_SEED + 1, 1)[0] != first[0]
    assert first[0] != first[1]


def test_code_chunk_shape_matches_sushi_rules():
    for text in chunks():
        size = len(text.encode())
        assert bench.CHUNK_BYTES <= size < 40 * 1024
        assert text.endswith("\n\n")
        assert "export class" in text
    assert chunks(count=1)[0].startswith("// src/")


def test_prose_chunks_are_paragraph_aligned_and_lossless():
    paragraph = ("It is a truth universally acknowledged. " * 20 + "\n\n").encode()
    data = paragraph * 200 + "café end".encode()
    pieces = list(bench.prose_chunks(data))
    assert "".join(pieces).encode() == data
    for piece in pieces[:-1]:
        assert piece.endswith("\n\n")
        assert len(piece.encode()) <= bench.CHUNK_BYTES


def test_prose_chunks_cut_on_bytes_without_a_blank_line():
    data = b"a" * (bench.CHUNK_BYTES + 10)
    pieces = list(bench.prose_chunks(data))
    assert [len(p) for p in pieces] == [bench.CHUNK_BYTES, 10]


@pytest.mark.parametrize("text,want", [
    ("4k", [4096]), ("8k,4k,8192", [4096, 8192]), ("16K, 4096", [4096, 16384]), ("128k", [131072]),
])
def test_rungs_parse_like_sushi(text, want):
    assert bench.parse_rungs(text) == want


@pytest.mark.parametrize("text", ["", "2k", "12k", "136k", "4k,,8k", "x", "-4k"])
def test_rungs_refuse_values_the_ladder_cannot_land_on(text):
    with pytest.raises(bench.BenchError):
        bench.parse_rungs(text)


def test_plan_matches_sushi_ladder():
    reqs = bench.plan(8, [4096, 24576])
    want = [(4096, 8, 0), (4096, 8, 4096), (4096, 8, 4096), (8192, 1, 4096), (16384, 1, 8192),
            (24576, 8, 16384), (24576, 8, 24576), (24576, 8, 24576)]
    assert [(r["prompt_len"], r["max_tokens"], r["base"]) for r in reqs] == want
    full = bench.plan(512, bench.parse_rungs("4k,8k,16k,32k,64k,128k"))
    assert len(full) == 1 + 16 + 6 * 2
    assert sum(r["kind"] == "ladder" for r in full) == 17
    big = bench.plan(8, [32768])
    assert [r["max_tokens"] for r in big[:4]] == [1, 1, 1, 1]
    assert bench.corpus_tokens_needed(bench.parse_rungs("4k,8k,16k,32k,64k,128k")) == 132000


def test_prose_decode_is_capped_at_128():
    assert bench.arm_decode_tokens("prose", 512) == 128
    assert bench.arm_decode_tokens("prose", 64) == 64
    assert bench.arm_decode_tokens("code", 512) == 512


def test_depths_options_and_arms_are_validated():
    assert bench.parse_depths("off,1,2,3") == ["off", "1", "2", "3"]
    assert bench.parse_depths("0,default") == ["off", "default"]
    for bad in ("4", "1,1", "", "x"):
        with pytest.raises(bench.BenchError):
            bench.parse_depths(bad)
    assert bench.parse_request_options(['mtp_policy="adaptive"', "draft_budget=4"]) == {
        "mtp_policy": "adaptive", "draft_budget": 4}
    for bad in (["mtp_drafts=2"], ["cache_scope=\"x\""], ["noequals"], ["k=not json"], ["a=1", "a=2"]):
        with pytest.raises(bench.BenchError):
            bench.parse_request_options(bad)
    assert bench.parse_arms("prose,code") == ["code", "prose"]
    for bad in ("", "code,code", "math"):
        with pytest.raises(bench.BenchError):
            bench.parse_arms(bad)


class FakeWorker:
    """A worker double: tokenizes by bytes and returns sentinel outputs."""

    instances: list = []

    def __init__(self, executable, model, timeout, *, early_stop_every=0, ready=None):
        self.requests = []
        self.early_stop_every = early_stop_every
        self.generated = 0
        self.closed = False
        self.process = Namespace(pid=None)
        self._ready = ready
        FakeWorker.instances.append(self)

    def read(self):
        return self._ready or {
            "ready": True, "mode": "text-generation-v1", "vocab_size": 151936, "max_context_tokens": 262144,
            "eos_token_ids": [151645], "prefix_cache": {"enabled": True, "chunk_tokens": 2048},
            "mtp": {"supported": True, "max_drafts": 3}, "image_input": dict(bench.EXPECTED_IMAGE_READY),
        }

    def request(self, payload):
        self.requests.append(payload)
        op = payload["operation"]
        if op == "tokenize":
            data = payload["text"].encode()
            return {"token_ids": [(data[i] * 7 + data[i + 1]) % 151000 for i in range(0, len(data) - 1, 4)]}
        if op == "reset_cache":
            return {"cache_reset": True}
        self.generated += 1
        count = payload["max_new_tokens"]
        drafts = payload.get("mtp_drafts", 0)
        early = self.early_stop_every and self.generated % self.early_stop_every == 0 and count > 1
        sampled = count // 2 if early else count
        ids = [(i * 31 + 5) % 1000 for i in range(sampled - (1 if early else 0))]
        reused = 2048 if payload.get("cache_scope") and payload["input_token_ids"].__len__() > 4096 else 0
        return {
            "generated_token_ids": ids, "text": SENTINEL_TEXT, "output_text": SENTINEL_TEXT,
            "input_tokens": len(payload["input_token_ids"]), "reused_input_tokens": reused,
            "sampled_tokens": sampled, "output_tokens": len(ids),
            "finish_reason": "eos" if early else "length", "stop_token_id": 151645 if early else None,
            "prefill_seconds": 1.5, "decode_seconds": 0.01 * sampled if sampled > 1 else None,
            "generation_seconds": 2.0,
            "mtp": {"enabled": drafts > 0, "drafts": drafts, "sampling": "exact-rejection-v1",
                    "verification": "batched-affine-decode-exact-v1",
                    "stats": {"rounds": 10 * drafts, "proposed": 20 * drafts, "accepted": 15 * drafts,
                              "verified_rows": 30 * drafts, "draft_seconds": 0.01 * drafts,
                              "verify_seconds": 0.1 * drafts, "rollback_seconds": 0.0}},
        }

    def close(self):
        self.closed = True


def make_args(tmp_path: Path, **overrides) -> Namespace:
    prose = tmp_path / "prose.txt"
    prose.write_text(("Elizabeth walked to Netherfield in the rain. " * 30 + "\n\n") * 120)
    values = dict(
        model=tmp_path / "model", worker=None, output=tmp_path / "report.json", arms="code,prose",
        prose_corpus=prose, rungs="4k", tokens=16, depths="off,1,2,3", request_option=None,
        prefix="scoped", warmup_tokens=4, corpus_seed=bench.CORPUS_SEED, seed=0, timeout=10,
        label="test", observe_host=False, keep_environment=True, require_lily_image=True, progress=False,
    )
    values.update(overrides)
    return Namespace(**values)


def test_run_produces_the_report_schema(tmp_path):
    report = bench.run(make_args(tmp_path), worker_factory=FakeWorker)
    assert report["schema"] == bench.SCHEMA and report["attempt"] == bench.ATTEMPT_ID
    assert report["passed"] is True
    assert report["decode_sample_count"] == 2 * 4 * 3
    assert report["early_stop_count"] == 0
    assert report["protocol"]["decode_tokens"] == {"code": 16, "prose": 16}
    assert report["arms"]["prose"]["matches_sushi_fixture"] is False
    rows = report["summary"]
    depths = ("off", "1", "2", "3")
    assert {(r["arm"], r["depth"]) for r in rows} == {(a, d) for a in ("code", "prose") for d in depths}
    for row in rows:
        assert row["valid_samples"] == 3 and row["decode_tokens_per_second_median"] == pytest.approx(15 / 0.16)
        assert row["sushi_formula_tokens_per_second_median"] == pytest.approx(16 / 0.16)
        if row["depth"] != "off":
            assert row["mtp_totals"]["proposed"] > 0 and row["mtp_totals"]["tokens_per_round"] > 1
    worker = FakeWorker.instances[-1]
    assert worker.closed
    generates = [r for r in worker.requests if r["operation"] == "generate"]
    assert all(r["sampling"] == {"mode": "greedy"} for r in generates)
    assert all("cache_scope" in r for r in generates if r["max_new_tokens"] == 16)
    assert sum(r["operation"] == "reset_cache" for r in worker.requests) == 2 * 4 * 2


def test_extra_request_options_pass_through_and_default_depth_omits_drafts(tmp_path):
    args = make_args(tmp_path, arms="code", depths="default", request_option=['mtp_policy="adaptive"'])
    report = bench.run(args, worker_factory=FakeWorker)
    generates = [r for r in FakeWorker.instances[-1].requests if r["operation"] == "generate"]
    assert all(r["mtp_policy"] == "adaptive" and "mtp_drafts" not in r for r in generates)
    assert report["protocol"]["request_options"] == {"mtp_policy": "adaptive"}


def test_early_eos_stops_are_counted_and_invalid(tmp_path):
    def factory(*a):
        return FakeWorker(*a, early_stop_every=4)

    report = bench.run(make_args(tmp_path, arms="code", depths="off"), worker_factory=factory)
    assert report["passed"] is False
    assert report["early_stop_count"] >= 1
    assert report["finish_reason_counts"]["eos"] == report["early_stop_count"]
    row = report["summary"][0]
    assert row["early_stops"] == report["early_stop_count"]
    assert row["valid_samples"] == 3 - row["early_stops"]


def test_fresh_prefix_mode_sends_no_scope(tmp_path):
    bench.run(make_args(tmp_path, arms="code", depths="off", prefix="fresh"), worker_factory=FakeWorker)
    assert not any("cache_scope" in r for r in FakeWorker.instances[-1].requests)


def test_run_refuses_bad_setups(tmp_path):
    with pytest.raises(bench.BenchError, match="prose_corpus_required"):
        bench.run(make_args(tmp_path, prose_corpus=None), worker_factory=FakeWorker)
    with pytest.raises(bench.BenchError, match="invalid_tokens"):
        bench.run(make_args(tmp_path, tokens=1), worker_factory=FakeWorker)
    args = make_args(tmp_path)
    args.output.write_text("{}")
    with pytest.raises(bench.BenchError, match="output_exists"):
        bench.run(args, worker_factory=FakeWorker)

    def no_image(*a):
        ready = FakeWorker(*a).read()
        ready.pop("image_input")
        return FakeWorker(*a, ready=ready)

    with pytest.raises(bench.BenchError, match="required_image_ready_missing"):
        bench.run(make_args(tmp_path, arms="code", output=tmp_path / "other.json"), worker_factory=no_image)
    assert FakeWorker.instances[-1].closed


def test_report_contains_no_text_or_token_vectors(tmp_path):
    args = make_args(tmp_path)
    report = bench.run(args, worker_factory=FakeWorker)
    bench.write_report(args.output, report)
    serialized = args.output.read_text()
    assert SENTINEL_TEXT not in serialized
    assert "Netherfield" not in serialized and "export class" not in serialized and "Logger" not in serialized
    worker = FakeWorker.instances[-1]
    generate = next(r for r in worker.requests if r["operation"] == "generate")
    prompt = generate["input_token_ids"]
    assert json.dumps(prompt[:8], separators=(",", ":"))[1:-1] not in serialized
    output_ids = [(i * 31 + 5) % 1000 for i in range(16)]
    assert json.dumps(output_ids[:8])[1:-1] not in serialized

    def walk(value):
        if isinstance(value, list):
            assert len(value) <= 64, "no long vectors in the report"
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                assert key not in {"generated_token_ids", "input_token_ids", "text", "output_text", "token_ids"}
                walk(item)

    walk(json.loads(serialized))


def test_write_report_refuses_to_overwrite(tmp_path):
    path = tmp_path / "r.json"
    bench.write_report(path, {"a": 1})
    with pytest.raises(FileExistsError):
        bench.write_report(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}


def test_main_refuses_while_another_worker_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bench, "other_workers", lambda own=None: [12345])
    args = ["--model", str(tmp_path), "--worker", str(tmp_path / "w"), "--output", str(tmp_path / "o.json"),
            "--arms", "code"]
    assert bench.main(args) == 2
    assert json.loads(capsys.readouterr().out)["stage"] == "other_worker_present"
    assert not (tmp_path / "o.json").exists()


def test_main_plan_only_needs_no_worker(capsys):
    assert bench.main(["--plan-only", "--arms", "code", "--rungs", "4k", "--tokens", "512"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [r["max_tokens"] for r in out["code"]] == [512, 512, 512]
