from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from argparse import Namespace
from types import SimpleNamespace

import compare_hybrid_score_workers as comparison
import pytest

LEGACY = {"success": True, "encoded_dispatch_calls": 4, "encoded_weight_bytes": 8192}
CURRENT = {**LEGACY, "encoded_q5_dispatch_calls": 1, "encoded_q5_weight_bytes": 2048}


@pytest.fixture(autouse=True)
def clean_probe_environment(monkeypatch):
    for key in (comparison.MMA_PROBE_ENV, "STARK_GLM53_PREFILL_MMA", "STARK_GLM53_PREFILL_CHUNK"):
        monkeypatch.delenv(key, raising=False)


def write_probe(path, rows, mode=0o600):
    raw = b"".join(json.dumps(row, separators=(",", ":")).encode() + b"\n" for row in rows)
    path.write_bytes(raw)
    path.chmod(mode)
    return raw


def assert_rejected(path, code, **kwargs):
    before = path.read_bytes()
    with pytest.raises(comparison.CheckFailed, match=f"^{code}$"):
        comparison.read_mma_probe(path, **kwargs)
    assert path.read_bytes() == before


class FakeTokenizer:
    def __init__(self, ids):
        self.ids = ids
        self.encodings = []

    def encode(self, text, *, add_special_tokens):
        self.encodings.append((text, add_special_tokens))
        return SimpleNamespace(ids=self.ids)

    def get_vocab(self):
        return {str(value): value for value in range(1024)}


@pytest.mark.parametrize("value", ["", "0", "4097", "-1", "1,1", "1,", ",2", "1.0", "True", "1;2", "１２８"])
def test_prefix_length_parser_rejects_invalid_lengths(value):
    with pytest.raises(argparse.ArgumentTypeError, match="^prefix-lengths must be distinct integers from 1 through 4096$"):
        comparison.parse_prefix_lengths(value)


def test_prefix_length_parser_preserves_requested_order_and_context_boundary():
    assert comparison.parse_prefix_lengths("96, 128,256,512") == [96, 128, 256, 512]
    assert comparison.parse_prefix_lengths("512,1") == [512, 1]
    assert comparison.parse_prefix_lengths("128,1024,3840,4096") == [128, 1024, 3840, 4096]


@pytest.mark.parametrize("lengths", [[], [True], [1.0], [0], [513], [1, 1], "128"])
def test_programmatic_prefix_lengths_use_same_validation(lengths):
    with pytest.raises(comparison.CheckFailed, match="^prefix_lengths_invalid$"):
        comparison.validate_prefix_lengths(lengths)


def test_exact_corpus_prefixes_are_one_direct_tokenization_without_template_or_repetition(tmp_path):
    raw = "A public document.\nA second paragraph with café.\n".encode()
    corpus = tmp_path / "public.txt"
    corpus.write_bytes(raw)
    tokenizer = FakeTokenizer(list(range(600)))

    prefixes, corpus_hash = comparison.collect_corpus_prefixes(tokenizer, corpus, [96, 128, 256, 512])

    assert tokenizer.encodings == [(raw.decode(), False)]
    assert [len(prefix) for prefix in prefixes] == [96, 128, 256, 512]
    assert all(prefix == tokenizer.ids[:length] for prefix, length in zip(prefixes, [96, 128, 256, 512]))
    assert corpus_hash == hashlib.sha256(raw).hexdigest()
    assert corpus.read_bytes() == raw


def test_corpus_short_encoding_and_invalid_utf8_fail_without_padding_or_payload(tmp_path):
    corpus = tmp_path / "public.txt"
    corpus.write_text("public text")
    with pytest.raises(comparison.CheckFailed, match="^corpus_tokens_insufficient$"):
        comparison.collect_corpus_prefixes(FakeTokenizer([1, 2]), corpus, [3])
    corpus.write_bytes(b"\xff")
    with pytest.raises(comparison.CheckFailed, match="^corpus_encoding_invalid$"):
        comparison.collect_corpus_prefixes(FakeTokenizer([1, 2]), corpus, [1])


@pytest.mark.parametrize("token", [-1, True, 1024])
def test_corpus_prefix_token_ids_must_fit_tokenizer_vocabulary(tmp_path, token):
    corpus = tmp_path / "public.txt"
    corpus.write_text("public text")
    with pytest.raises(comparison.CheckFailed, match="^token_id_out_of_range$"):
        comparison.collect_corpus_prefixes(FakeTokenizer([token]), corpus, [1])


def test_probe_default_eight_rows_and_expanded_expected_count(tmp_path, monkeypatch):
    path = tmp_path / "probe.jsonl"
    monkeypatch.setenv("STARK_GLM53_PREFILL_MMA", "1")
    monkeypatch.setenv("STARK_GLM53_PREFILL_CHUNK", "8")
    raw = write_probe(path, [CURRENT] * 12)

    result = comparison.read_mma_probe(path, expected_rows=12)

    assert result["request_count"] == 12
    assert result["encoded_dispatch_calls"] == 48
    assert result["encoded_q5_dispatch_calls"] == 12
    assert result["encoded_q6_dispatch_calls"] == 36
    assert result["encoded_weight_bytes"] == 98_304
    assert result["encoded_q5_weight_bytes"] == 24_576
    assert result["encoded_q6_weight_bytes"] == 73_728
    assert result["probe_sha256"] == hashlib.sha256(raw).hexdigest()
    assert path.read_bytes() == raw
    assert_rejected(path, "mma_probe_row_count_invalid")
    write_probe(path, [CURRENT] * 8)
    assert comparison.read_mma_probe(path)["request_count"] == 8


@pytest.mark.parametrize("expected", [0, -1, True, 8.0])
def test_probe_expected_row_count_is_a_positive_integer(tmp_path, expected):
    path = tmp_path / "probe.jsonl"
    write_probe(path, [CURRENT] * 8)
    assert_rejected(path, "mma_probe_expected_row_count_invalid", expected_rows=expected)


def test_probe_legacy_current_and_chunk_policies_remain_strict(tmp_path, monkeypatch):
    path = tmp_path / "probe.jsonl"
    monkeypatch.setenv("STARK_GLM53_PREFILL_MMA", "1")
    monkeypatch.setenv("STARK_GLM53_PREFILL_CHUNK", "4")
    write_probe(path, [LEGACY] * 8)
    result = comparison.read_mma_probe(path)
    assert result["encoded_q5_dispatch_calls"] == 0
    assert result["encoded_q6_dispatch_calls"] == 32
    assert result["encoded_q6_weight_bytes"] == 65_536
    write_probe(path, [{**CURRENT, "encoded_q5_dispatch_calls": 0, "encoded_q5_weight_bytes": 0}] * 8)
    assert comparison.read_mma_probe(path)["encoded_q6_weight_bytes"] == 65_536
    write_probe(path, [CURRENT, *([LEGACY] * 7)])
    assert_rejected(path, "mma_probe_schema_changed")
    monkeypatch.setenv("STARK_GLM53_PREFILL_CHUNK", "8")
    write_probe(path, [LEGACY] * 8)
    assert_rejected(path, "mma_probe_q5_schema_required")
    write_probe(path, [CURRENT] * 8)
    assert comparison.read_mma_probe(path)["encoded_q5_dispatch_calls"] == 8
    monkeypatch.setenv("STARK_GLM53_PREFILL_MMA", "0")
    for row in (LEGACY, CURRENT):
        zero = {key: (True if key == "success" else 0) for key in row}
        write_probe(path, [zero] * 8)
        assert comparison.read_mma_probe(path)["encoded_dispatch_calls"] == 0


@pytest.mark.parametrize(("changes", "code"), [
    ({"extra": []}, "mma_probe_schema_invalid"),
    ({"encoded_dispatch_calls": True}, "mma_probe_counter_invalid"),
    ({"encoded_q5_dispatch_calls": True}, "mma_probe_counter_invalid"),
    ({"encoded_dispatch_calls": 1 << 64}, "mma_probe_counter_invalid"),
    ({"encoded_weight_bytes": 1 << 64}, "mma_probe_counter_invalid"),
    ({"encoded_q5_dispatch_calls": 1 << 64}, "mma_probe_counter_invalid"),
    ({"encoded_q5_weight_bytes": 1 << 64}, "mma_probe_counter_invalid"),
    ({"encoded_q5_weight_bytes": -1}, "mma_probe_counter_invalid"),
    ({"encoded_q5_dispatch_calls": 1.0}, "mma_probe_counter_invalid"),
    ({"encoded_q5_weight_bytes": []}, "mma_probe_counter_invalid"),
    ({"encoded_q5_dispatch_calls": 5}, "mma_probe_subset_invalid"),
    ({"encoded_q5_weight_bytes": 8193}, "mma_probe_subset_invalid"),
    ({"success": False}, "mma_probe_request_failed"),
    ({"success": 1}, "mma_probe_request_failed"),
    ({"encoded_dispatch_calls": 0, "encoded_weight_bytes": 0,
      "encoded_q5_dispatch_calls": 0, "encoded_q5_weight_bytes": 0}, "mma_probe_dispatch_missing"),
    ({"encoded_q5_dispatch_calls": 0, "encoded_q5_weight_bytes": 0}, "mma_probe_q5_dispatch_missing"),
    ({"encoded_q5_dispatch_calls": 4, "encoded_q5_weight_bytes": 8192}, "mma_probe_q6_dispatch_missing"),
    ({"encoded_q5_weight_bytes": 0}, "mma_probe_q5_dispatch_missing"),
    ({"encoded_q5_weight_bytes": 8192}, "mma_probe_q6_dispatch_missing"),
    ({"encoded_dispatch_calls": (1 << 64) - 1}, "mma_probe_total_overflow"),
    ({"encoded_weight_bytes": (1 << 64) - 1}, "mma_probe_total_overflow"),
])
def test_probe_counter_and_schema_rejections_do_not_mutate_private_file(tmp_path, monkeypatch, changes, code):
    monkeypatch.setenv("STARK_GLM53_PREFILL_MMA", "1")
    monkeypatch.setenv("STARK_GLM53_PREFILL_CHUNK", "8")
    path = tmp_path / "probe.jsonl"
    write_probe(path, [{**CURRENT, **changes}, *([CURRENT] * 7)])
    assert_rejected(path, code)


def test_probe_missing_field_rejected(tmp_path):
    path = tmp_path / "probe.jsonl"
    row = {key: value for key, value in CURRENT.items() if key != "encoded_q5_weight_bytes"}
    write_probe(path, [row] * 8)
    assert_rejected(path, "mma_probe_schema_invalid")


@pytest.mark.parametrize("count", [7, 9, 11, 13])
def test_probe_wrong_count_rejected(tmp_path, count):
    path = tmp_path / "probe.jsonl"
    write_probe(path, [CURRENT] * count)
    assert_rejected(path, "mma_probe_row_count_invalid", expected_rows=12 if count > 9 else 8)


def test_probe_file_boundaries_permissions_and_fresh_path(tmp_path, monkeypatch):
    path = tmp_path / "probe.jsonl"
    assert comparison.requested_probe_path() is None
    monkeypatch.setenv(comparison.MMA_PROBE_ENV, str(path))
    assert comparison.requested_probe_path() == path
    assert not path.exists()
    write_probe(path, [CURRENT] * 8, mode=0o644)
    assert_rejected(path, "mma_probe_permissions_invalid")
    with pytest.raises(comparison.CheckFailed, match="^mma_probe_path_not_fresh$"):
        comparison.requested_probe_path()
    write_probe(path, [CURRENT] * 8)
    path.write_bytes(path.read_bytes()[:-1])
    assert_rejected(path, "mma_probe_jsonl_boundary_invalid")
    path.write_bytes(b"{bad}\n" * 8)
    assert_rejected(path, "mma_probe_json_invalid")
    path.write_bytes(b"x" * (comparison.MMA_PROBE_LIMIT + 1))
    assert_rejected(path, "mma_probe_too_large")
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(comparison.CheckFailed, match="^mma_probe_file_invalid$"):
        comparison.read_mma_probe(link)
    missing = tmp_path / "absent"
    dangling = tmp_path / "dangling"
    dangling.symlink_to(missing)
    monkeypatch.setenv(comparison.MMA_PROBE_ENV, str(dangling))
    with pytest.raises(comparison.CheckFailed, match="^mma_probe_path_not_fresh$"):
        comparison.requested_probe_path()
    with pytest.raises(comparison.CheckFailed, match="^mma_probe_missing_or_unreadable$"):
        comparison.read_mma_probe(missing)
    monkeypatch.setenv(comparison.MMA_PROBE_ENV, str(tmp_path / "missing_parent" / "probe"))
    with pytest.raises(comparison.CheckFailed, match="^mma_probe_parent_invalid$"):
        comparison.requested_probe_path()


@pytest.mark.parametrize(("prefix_lengths", "context_tokens"), [
    (None, None), ([96, 128, 256, 512], 512), ([128, 1024, 3840], 4096),
])
def test_run_report_binds_actual_lengths_hashes_and_probe_count_without_payloads(
    tmp_path, monkeypatch, prefix_lengths, context_tokens,
):
    expanded = prefix_lengths is not None
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    reference.write_bytes(b"reference executable")
    candidate.write_bytes(b"candidate executable")
    corpus = tmp_path / "public.txt"
    corpus_text = "Synthetic public corpus used only by the model-free test."
    corpus.write_text(corpus_text)
    output = tmp_path / "report.json"
    probe = tmp_path / "probe.jsonl"
    monkeypatch.setenv(comparison.MMA_PROBE_ENV, str(probe))
    monkeypatch.setenv("STARK_GLM53_PREFILL_MMA", "1")
    monkeypatch.setenv("STARK_GLM53_PREFILL_CHUNK", "8")
    monkeypatch.setenv("STARK_QWEN38_PROFILE_OUTPUT", "private-profile-output")
    monkeypatch.setenv("STARK_NATIVE_PROFILE_OUTPUT", "private-native-profile-output")
    short_inputs = [list(range(14 + row)) for row in range(8)]
    candidates = list(range(16))
    monkeypatch.setattr(comparison, "collect_inputs", lambda _model: ([row[:] for row in short_inputs], candidates, "template_hash"))
    tokenizer = FakeTokenizer([index % 1024 for index in range(4096)])
    monkeypatch.setattr(comparison, "Tokenizer", SimpleNamespace(from_file=lambda _path: tokenizer))
    monkeypatch.setattr(comparison, "weight_identity", lambda *_args: {"weights_sha256": "unchanged"})
    monkeypatch.setattr(comparison, "checkout_identity", lambda: {"source": "synthetic"})
    calls = []

    async def fake_worker(binary, model_path, inputs, candidate_ids, inherited_flags, probe_path=None,
                          context_tokens=comparison.CONTEXT):
        calls.append((binary, [row[:] for row in inputs], candidate_ids, inherited_flags, context_tokens))
        if probe_path is not None:
            write_probe(probe_path, [CURRENT] * len(inputs))
        return [[-2.0 - column / 1000 for column in range(16)] for _row in inputs]

    monkeypatch.setattr(comparison, "run_worker", fake_worker)
    args = Namespace(model=model, reference_worker=reference, candidate_worker=candidate,
                     output=output, verified_weights=None, max_logprob_error=0.001)
    if expanded:
        args.corpus = corpus
        args.prefix_lengths = prefix_lengths
    if context_tokens is not None:
        args.context_tokens = context_tokens
    report = asyncio.run(comparison.run(args))
    expected = short_inputs + ([tokenizer.ids[:length] for length in args.prefix_lengths] if expanded else [])
    assert report["context_count"] == len(expected) == 8 + len(prefix_lengths or [])
    assert report["candidate_count"] == 16
    assert report["score_point_count"] == len(expected) * 16
    assert report["context_tokens"] == (context_tokens or 512)
    assert report["min_input_tokens"] == 14
    assert report["max_input_tokens"] == (max(prefix_lengths) if expanded else 21)
    assert report["total_input_tokens"] == sum(map(len, expected))
    assert report["input_token_ids_sha256"] == comparison.token_hash(expected)
    assert report["candidate_token_ids_sha256"] == comparison.token_hash([candidates])
    assert report["mma_score_probe"]["request_count"] == len(expected)
    assert [call[3] for call in calls] == [False, True]
    assert all(call[1] == expected and call[2] == candidates for call in calls)
    assert all(call[4] == (context_tokens or 512) for call in calls)
    assert report["candidate_flags"]["STARK_GLM53_PREFILL_CHUNK"] == "8"
    assert report["diagnostic_profile_output_requested"] is True
    assert report["native_profile_output_requested"] is True
    if expanded:
        assert report["corpus_sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
        assert report["requested_prefix_lengths"] == prefix_lengths
        assert tokenizer.encodings == [(corpus_text, False)]
    else:
        assert "corpus_sha256" not in report
        assert "requested_prefix_lengths" not in report
        assert tokenizer.encodings == []
        assert report["scope"].startswith("8 public rendered contexts x")
    serialized = output.read_text()
    assert json.loads(serialized) == report
    assert output.stat().st_mode & 0o777 == 0o600
    assert corpus_text not in serialized
    assert '"input_token_ids":' not in serialized
    assert '"candidate_token_ids":' not in serialized
    assert "log_probabilities" not in serialized
    assert json.dumps(expected) not in serialized
    with pytest.raises(comparison.CheckFailed, match="^output_already_exists$"):
        asyncio.run(comparison.run(args))
    assert output.read_text() == serialized


@pytest.mark.parametrize("options", [
    {"corpus": "public.txt"}, {"prefix_lengths": [96]}, {"corpus": "public.txt", "prefix_lengths": [513]},
    {"corpus": "public.txt", "prefix_lengths": [1025], "context_tokens": 1024},
    {"corpus": "public.txt", "prefix_lengths": [4097], "context_tokens": 4096},
])
def test_run_rejects_invalid_prefix_options_before_any_worker_or_model_io(options):
    with pytest.raises(comparison.CheckFailed, match="^(corpus_prefix_options_required_together|prefix_lengths_invalid)$"):
        asyncio.run(comparison.run(Namespace(**options)))


@pytest.mark.parametrize("context_tokens", [0, -1, 4097, True, 512.0])
def test_invalid_context_rejected_before_any_worker_or_model_io(context_tokens):
    with pytest.raises(comparison.CheckFailed, match="^context_tokens_invalid$"):
        asyncio.run(comparison.run(Namespace(context_tokens=context_tokens)))


@pytest.mark.parametrize("failure", [RuntimeError("PRIVATE_EXCEPTION_PAYLOAD"), KeyboardInterrupt()])
def test_cli_sanitizes_unexpected_failures(tmp_path, monkeypatch, capsys, failure):
    async def fail(_args):
        raise failure

    monkeypatch.setattr(comparison, "run", fail)
    monkeypatch.setattr("sys.argv", ["compare", "--model", str(tmp_path), "--reference-worker", "reference",
                                    "--candidate-worker", "candidate", "--output", str(tmp_path / "report")])
    assert comparison.main() == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"passed": False, "check": "score_worker_comparison_failed"}
    assert captured.err == ""
    assert "PRIVATE_EXCEPTION_PAYLOAD" not in captured.out


def test_cli_prefix_pairing_fails_before_execution(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["compare", "--model", "model", "--reference-worker", "reference",
                                    "--candidate-worker", "candidate", "--output", "report", "--prefix-lengths", "96"])
    with pytest.raises(SystemExit) as failure:
        comparison.main()
    assert failure.value.code == 2
    assert "corpus and prefix-lengths must be supplied together" in capsys.readouterr().err


class FakePipe:
    def __init__(self, replies=()):
        self.replies = list(replies)
        self.writes = []
        self.closed = False

    async def readline(self):
        return self.replies.pop(0)

    async def read(self, _limit):
        return b""

    def write(self, data):
        self.writes.append(data)

    async def drain(self):
        pass

    def close(self):
        self.closed = True


@pytest.mark.parametrize("selected_context", [None, 4096])
def test_workers_send_selected_context_and_finish_teardown_before_next_launch(
    tmp_path, monkeypatch, selected_context,
):
    lengths = [2] if selected_context is None else [128, 1024, 3840, 4096]
    inputs = [[index % 1024 for index in range(length)] for length in lengths]
    candidates = list(range(16))
    processes = []
    events = []

    async def create(binary, *_args, **_kwargs):
        if processes:
            assert processes[-1].stdin.closed
            assert processes[-1].waited
        events.append(("launch", comparison.Path(binary).name))
        replies = [{"ready": True, "cache_enabled": True}]
        replies.extend({
            "input_tokens": length, "candidate_token_ids": candidates,
            "log_probabilities": [-2.0] * 16,
        } for length in lengths)
        process = SimpleNamespace(
            stdin=FakePipe(), stdout=FakePipe([json.dumps(reply).encode() + b"\n" for reply in replies]),
            returncode=0, waited=False,
        )

        async def wait():
            process.waited = True
            events.append(("wait", comparison.Path(binary).name))
            return 0

        process.wait = wait
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    async def exercise():
        options = {} if selected_context is None else {"context_tokens": selected_context}
        for name, inherited_flags in (("reference", False), ("candidate", True)):
            rows = await comparison.run_worker(
                tmp_path / name, tmp_path / "model", inputs, candidates, inherited_flags, **options,
            )
            assert len(rows) == len(lengths)

    asyncio.run(exercise())

    assert events == [("launch", "reference"), ("wait", "reference"),
                      ("launch", "candidate"), ("wait", "candidate")]
    for process in processes:
        requests = [json.loads(data) for data in process.stdin.writes]
        assert [request["input_token_ids"] for request in requests] == inputs
        assert all(request["context_tokens"] == (selected_context or 512) for request in requests)
        assert all(request["candidate_token_ids"] == candidates for request in requests)
        assert process.stdin.closed and process.waited


@pytest.mark.parametrize(("context_tokens", "input_tokens", "error"), [
    (512, 513, "input_length_out_of_range"),
    (4096, 4097, "input_length_out_of_range"),
    (4097, 1, "context_tokens_invalid"),
])
def test_worker_limits_rejected_before_process_launch(tmp_path, monkeypatch, context_tokens, input_tokens, error):
    async def forbidden(*_args, **_kwargs):
        raise AssertionError("worker_must_not_launch")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    with pytest.raises(comparison.CheckFailed, match=f"^{error}$"):
        asyncio.run(comparison.run_worker(
            tmp_path / "worker", tmp_path / "model", [[1] * input_tokens], [3], False,
            context_tokens=context_tokens,
        ))


@pytest.mark.parametrize(("reply", "code"), [
    (b"", "worker_reply_boundary_invalid"),
    (b"{}", "worker_reply_boundary_invalid"),
    (b"x" * 1_048_577 + b"\n", "worker_reply_boundary_invalid"),
    (b"[]\n", "worker_reply_invalid"),
])
def test_pipe_worker_private_reply_boundaries(reply, code):
    worker = comparison.PipeWorker(SimpleNamespace(stdout=FakePipe([reply])))
    with pytest.raises(comparison.CheckFailed, match=f"^{code}$"):
        asyncio.run(worker.reply())


@pytest.mark.parametrize("logps", [[True], [float("nan")], [float("inf")], [0.1], []])
def test_worker_rejects_invalid_score_payloads_and_closes_pipe(tmp_path, monkeypatch, logps):
    replies = [{"ready": True, "cache_enabled": True},
               {"input_tokens": 2, "candidate_token_ids": [3], "log_probabilities": logps}]
    process = SimpleNamespace(stdin=FakePipe(), stdout=FakePipe([
        json.dumps(reply).encode() + b"\n" for reply in replies]), returncode=0)
    state = {"waited": False}

    async def wait():
        state["waited"] = True
        return 0

    process.wait = wait

    async def create(*_args, **kwargs):
        assert kwargs["stderr"] == asyncio.subprocess.DEVNULL
        assert not any(key.startswith(("STARK_", "UZU_")) for key in kwargs["env"])
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    with pytest.raises(comparison.CheckFailed, match="^(logprob_values_invalid|logprob_coverage_invalid)$"):
        asyncio.run(comparison.run_worker(tmp_path / "worker", tmp_path / "model", [[1, 2]], [3], False))
    assert process.stdin.closed and state["waited"]


def test_pipe_worker_request_size_and_nan_fail_before_writing():
    pipe = FakePipe()
    worker = comparison.PipeWorker(SimpleNamespace(stdin=pipe))
    with pytest.raises(ValueError):
        asyncio.run(worker.request({"value": float("nan")}))
    with pytest.raises(comparison.CheckFailed, match="^worker_request_boundary_invalid$"):
        asyncio.run(worker.request({"value": "x" * 1_048_577}))
    assert pipe.writes == []
