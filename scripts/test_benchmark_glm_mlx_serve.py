from __future__ import annotations

import hashlib
import json
from argparse import Namespace

import benchmark_glm_mlx_serve as benchmark
import benchmark_hybrid
import pytest
from benchmark_hybrid_context import exact_prefix, token_ids_sha256
from tokenizers import Tokenizer, models, pre_tokenizers


@pytest.fixture
def args(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"offline fake weights")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    config = {
        "token_codec_config": {
            "type": "ChatCodecConfig",
            "prompt_template": '{{ bos_token }} {{ messages[0]["content"] }} {{ eos_token }}'
            '{% if add_generation_prompt %} <assistant>{% endif %}',
            "bos_token": "<bos>",
            "eos_token": "<eos>",
        },
    }
    (tokenizer / "config.json").write_text(json.dumps(config))
    (tokenizer / "native-source.json").write_text("{}")
    codec = Tokenizer(models.WordLevel(
        {"[UNK]": 0, "alpha": 1, "beta": 2, "Hello.": 3, "<bos>": 4, "<eos>": 5, "<assistant>": 6},
        unk_token="[UNK]",
    ))
    codec.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    codec.save(str(tokenizer / "tokenizer.json"))
    binary = tmp_path / "fake-mlx-serve"
    binary.write_bytes(b"offline fake executable")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("alpha beta", encoding="utf-8")
    return Namespace(
        model=model, tokenizer=tokenizer, binary=binary,
        weights_manifest=None, engine="llama", screened=False, max_attempts=20,
        replays=2, tokens=32, corpus=corpus, input_tokens=5, context_tokens=None,
    )


@pytest.fixture
def fake_server(monkeypatch):
    calls = {"launches": [], "requests": [], "signals": []}

    class FakeProcess:
        pid = 123456789

        def poll(self):
            return None

        def wait(self, timeout):
            return 0

    def launch(command, **kwargs):
        calls["launches"].append((command, kwargs))
        return FakeProcess()

    def http_json(_base, path, payload=None, timeout=10):
        if path == "/health":
            return {"status": "ok"}
        if path == "/props":
            return {
                "version": "1.2.3-test",
                "settings": {"engine": "llama", "mtp": False, "ignored_private_setting": "never publish"},
            }
        if path == "/v1/models":
            context = int(calls["launches"][0][0][calls["launches"][0][0].index("--ctx-size") + 1])
            return {"data": [{
                "id": "/private/model.gguf", "loaded": True, "state": "ready",
                "context_length": context, "meta": {"engine": "llama"},
            }]}
        if path == "/v1/completions":
            calls["requests"].append(payload)
            return {
                "choices": [{"text": "private fake response", "finish_reason": "length"}],
                "usage": {
                    "prompt_tokens": len(payload["prompt"]), "completion_tokens": payload["max_tokens"],
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            }
        if path == "/metrics.json":
            count = len(calls["requests"])
            return {
                "counters": {"requests_success_total": count},
                "histograms": {
                    "prefill_time_seconds": {"count": count, "sum": count * 0.5},
                    "decode_time_seconds": {"count": count, "sum": count * 2.0},
                },
            }
        raise AssertionError("unexpected_http_path")

    monkeypatch.setattr(benchmark, "http_json", http_json)
    monkeypatch.setattr(benchmark, "free_port", lambda: 12345)
    monkeypatch.setattr(benchmark.subprocess, "Popen", launch)
    monkeypatch.setattr(benchmark.os, "killpg", lambda *values: calls["signals"].append(values))
    monkeypatch.setattr(benchmark_hybrid, "checkout_identity", lambda: {
        "head": "a" * 40, "tracked_diff_sha256": "b" * 64,
        "binding": "Offline fixture; no build provenance claim.",
    })
    return calls


@pytest.mark.parametrize("input_tokens", [1, 5, 2048, 16128])
def test_corpus_request_matches_native_exact_prefix_and_hash(args, fake_server, input_tokens, capsys):
    args.input_tokens = input_tokens
    args.tokens = 256

    report = benchmark.run(args)

    native_ids = exact_prefix([1, 2], input_tokens)
    assert report["passed"] is True
    assert all(request["prompt"] == native_ids for request in fake_server["requests"])
    assert all(request["max_tokens"] == 256 for request in fake_server["requests"])
    assert report["requested_input_tokens"] == report["input_tokens"] == input_tokens
    assert report["requested_output_tokens"] == 256
    assert [row["output_tokens"] for row in report["replays"]] == [256, 256]
    assert [row["reported_cached_tokens"] for row in report["replays"]] == [0, 0]
    assert report["input_token_sha256"] == report["input_sha256"] == token_ids_sha256(native_ids)
    assert report["corpus_token_count"] == 2
    assert report["corpus_sha256"] == hashlib.sha256(b"alpha beta").hexdigest()
    assert report["context_tokens"] == input_tokens + 256
    assert report["requested_context_tokens"] is None
    command, launch_options = fake_server["launches"][0]
    assert command[command.index("--ctx-size") + 1] == str(input_tokens + 256)
    assert command[command.index("--max-tokens") + 1] == "256"
    assert launch_options["stdout"] == launch_options["stderr"] == benchmark.subprocess.DEVNULL
    assert report["mlx_serve_reported_version"] == "1.2.3-test"
    assert report["effective_settings"] == {"engine": "llama", "mtp": False, "version": "1.2.3-test"}
    profile = report["execution_profile"]
    assert profile["dependency_source_sha256"]["benchmark_hybrid_context.py"] == benchmark.sha(
        benchmark.Path(benchmark.__file__).with_name("benchmark_hybrid_context.py")
    )
    assert len(profile["converter_lock_sha256"]) == 64
    assert profile["tokenizers_version"]
    serialized = json.dumps(report)
    for private in ("alpha beta", "private fake response", "never publish", "/private/", '"prompt":', '"token_ids":'):
        assert private not in serialized
    assert "prompt_text_sha256" not in report
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == [
        {"completed_replay": 1}, {"completed_replay": 2},
    ]


def test_legacy_short_fixture_preserves_template_and_context(args, fake_server):
    args.corpus = args.input_tokens = None

    report = benchmark.run(args)

    assert report["passed"] is True
    assert report["context_tokens"] == 512
    assert report["input_tokens"] == 4
    assert report["prompt_mode"] == "rendered_short_fixture"
    assert report["requested_input_tokens"] is None
    assert report["prompt_text_sha256"] == hashlib.sha256(b"<bos> Hello. <eos> <assistant>").hexdigest()
    assert all(request["prompt"] == [4, 3, 5, 6] for request in fake_server["requests"])
    assert "corpus_sha256" not in report
    assert "corpus_token_count" not in report


def test_explicit_context_capacity_is_passed_to_server(args, fake_server):
    args.context_tokens = 4096

    report = benchmark.run(args)

    assert report["passed"] is True
    assert report["context_tokens"] == report["requested_context_tokens"] == 4096
    command = fake_server["launches"][0][0]
    assert command[command.index("--ctx-size") + 1] == "4096"


def test_warm_prefill_and_wall_medians_exclude_warmup(args, fake_server, monkeypatch):
    args.replays = 3
    original_http = benchmark.http_json
    ticks = iter([0, 0, 10, 15, 20, 21, 30, 33])
    prefill_sums = [0, 10, 12, 16]

    def varied_metrics(base, path, payload=None, timeout=10):
        response = original_http(base, path, payload, timeout)
        if path == "/metrics.json":
            count = len(fake_server["requests"])
            response["histograms"]["prefill_time_seconds"]["sum"] = prefill_sums[count]
        return response

    monkeypatch.setattr(benchmark.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(benchmark, "http_json", varied_metrics)

    report = benchmark.run(args)

    assert report["passed"] is True
    assert [row["wall_seconds"] for row in report["replays"]] == [5, 1, 3]
    assert [row["prefill_seconds"] for row in report["replays"]] == [10, 2, 4]
    assert report["median_warm_wall_seconds"] == 2
    assert report["median_warm_prefill_tokens_per_second"] == (5 / 2 + 5 / 4) / 2


@pytest.mark.parametrize("cached_value", [None, "0", False])
def test_missing_or_noninteger_cached_count_is_not_reported_as_observed_zero(
    args, fake_server, monkeypatch, cached_value,
):
    original_http = benchmark.http_json

    def unreported_cache(base, path, payload=None, timeout=10):
        response = original_http(base, path, payload, timeout)
        if path == "/v1/completions":
            if cached_value is None:
                response["usage"].pop("prompt_tokens_details")
            else:
                response["usage"]["prompt_tokens_details"]["cached_tokens"] = cached_value
        return response

    monkeypatch.setattr(benchmark, "http_json", unreported_cache)

    report = benchmark.run(args)

    assert report["passed"] is True
    assert [row["cached_tokens"] for row in report["replays"]] == [0, 0]
    assert all(row["reported_cached_tokens"] is None for row in report["replays"])


@pytest.mark.parametrize(("values", "error"), [
    ({"input_tokens": None}, "corpus_and_input_tokens_required_together"),
    ({"corpus": None}, "corpus_and_input_tokens_required_together"),
    ({"input_tokens": 0}, "invalid_benchmark_limits"),
    ({"input_tokens": True}, "invalid_benchmark_limits"),
    ({"tokens": 257}, "invalid_benchmark_limits"),
    ({"tokens": 7}, "invalid_benchmark_limits"),
    ({"input_tokens": 16384}, "context_limit_exceeded"),
    ({"context_tokens": 16385}, "context_limit_exceeded"),
    ({"context_tokens": 0}, "context_limit_exceeded"),
    ({"context_tokens": 36}, "context_too_small"),
    ({"screened": True, "replays": 8, "max_attempts": 7}, "invalid_screened_attempt_limit"),
])
def test_invalid_limits_reject_before_any_process_or_tokenization(args, monkeypatch, values, error):
    for key, value in values.items():
        setattr(args, key, value)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("must_reject_before_process_or_tokenization")

    monkeypatch.setattr(benchmark.subprocess, "Popen", forbidden)
    monkeypatch.setattr(benchmark, "prepare_input", forbidden)
    with pytest.raises(ValueError, match=f"^{error}$"):
        benchmark.run(args)


@pytest.mark.parametrize(("content", "error"), [
    (b"", "corpus_empty"),
    (b"\xff", "corpus_not_utf8"),
    (b"   \n", "tokenization_invalid"),
])
def test_unusable_corpus_rejects_before_process_launch(args, monkeypatch, content, error):
    args.corpus.write_bytes(content)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("must_reject_before_process")

    monkeypatch.setattr(benchmark.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match=f"^{error}$"):
        benchmark.run(args)


def test_short_fixture_context_checked_after_tokenization_before_process(args, monkeypatch):
    args.corpus = args.input_tokens = None
    args.context_tokens = args.tokens + 3

    def forbidden(*_args, **_kwargs):
        raise AssertionError("must_reject_before_process")

    monkeypatch.setattr(benchmark.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="^context_too_small$"):
        benchmark.run(args)
