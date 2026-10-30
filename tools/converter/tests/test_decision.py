"""Generic decision input and native-readout boundaries, using synthetic data."""

import json
from pathlib import Path

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from stark_convert.decision import (
    DecisionInputError,
    DecisionProtocolError,
    ScoreWorker,
    decide,
    validate_inputs,
    worker_context_limit,
)


class Worker:
    def __init__(self, *, tokenize=None, score=None):
        self.requests = []
        self.tokenize = tokenize
        self.score = score

    def request(self, payload):
        self.requests.append(payload)
        if payload["operation"] == "tokenize":
            ids = [ord(payload["text"])] if len(payload["text"]) == 1 else [1, 2, 3]
            if self.tokenize is not None:
                ids = self.tokenize(payload["text"], ids)
            return {"token_ids": ids, "token_count": len(ids)}
        candidates = payload["candidate_token_ids"]
        reply = {
            "input_tokens": len(payload["input_token_ids"]),
            "candidate_token_ids": candidates,
            "candidate_probabilities": [0.25, 0.75],
            "temperature": 1.0,
            "truncated": False,
        }
        if self.score is not None:
            reply.update(self.score)
        return reply


def evaluate(worker, answers=None):
    template = ImmutableSandboxedEnvironment().from_string(
        "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}"
        "thinking={{ enable_thinking }};bos={{ bos_token }};eos={{ eos_token }}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )
    return decide(
        worker,
        {"bos_token": "start", "eos_token": "end"},
        template,
        "synthetic text",
        "synthetic question",
        ["First", "Second"] if answers is None else answers,
        1000,
    )


@pytest.mark.parametrize("value", [None, True, 0, -1, 65537, "65536", 65536.0])
def test_ready_context_limit_rejects_invalid_capabilities(value):
    with pytest.raises(DecisionProtocolError):
        worker_context_limit({"max_context_tokens": value}, legacy=True)


def test_ready_context_limit_preserves_legacy_fallback_and_accepts_bounded_advertisement():
    assert worker_context_limit({}, legacy=True) == 16384
    assert worker_context_limit({"max_context_tokens": 65536}, legacy=True) == 65536
    with pytest.raises(DecisionProtocolError):
        worker_context_limit({})


@pytest.mark.parametrize("advertised,expected", [(None, 16384), (16384, 16384), (65536, 65536)])
def test_score_worker_stores_its_ready_capacity(monkeypatch, advertised, expected):
    reply = {"ready": True}
    if advertised is not None:
        reply["max_context_tokens"] = advertised
    monkeypatch.setattr("stark_convert.chat.Worker.read", lambda self, **kwargs: reply)
    worker = ScoreWorker.__new__(ScoreWorker)
    assert worker.read() == reply
    assert worker.max_context_tokens == expected


def test_score_worker_closes_on_an_invalid_ready_capacity(monkeypatch):
    from stark_convert.chat import ChatError

    monkeypatch.setattr(
        "stark_convert.chat.Worker.read",
        lambda self, **kwargs: {"ready": True, "max_context_tokens": True},
    )
    worker = ScoreWorker.__new__(ScoreWorker)
    closed = []
    worker.close = lambda: closed.append(True)
    with pytest.raises(ChatError, match="invalid scoring context limit"):
        worker.read()
    assert closed == [True]


@pytest.mark.parametrize(
    "limit,length,accepted",
    [(None, 16384, True), (None, 16385, False), (65536, 65535, True), (65536, 65536, True), (65536, 65537, False)],
)
def test_decision_uses_the_worker_context_capability_without_truncation(limit, length, accepted):
    worker = Worker(tokenize=lambda text, ids: ids if len(text) == 1 else [1] * length)
    if limit is not None:
        worker.max_context_tokens = limit
    if accepted:
        assert evaluate(worker)["input_tokens"] == length
        assert worker.requests[-1]["context_tokens"] == length
        assert len(worker.requests[-1]["input_token_ids"]) == length
    else:
        with pytest.raises(DecisionInputError, match="context limit"):
            evaluate(worker)
        assert all(request["operation"] != "score" for request in worker.requests)


def test_decision_uses_model_template_ordered_letter_mapping_and_uncalibrated_readout():
    worker = Worker()
    assert evaluate(worker, [" Second ", "First"]) == {
        "choice_index": 1,
        "answers": ["Second", "First"],
        "probabilities": [0.25, 0.75],
        "input_tokens": 3,
    }
    prompt = worker.requests[-2]["text"]
    assert '"letter": "A", "label": "Second"' in prompt
    assert '"letter": "B", "label": "First"' in prompt
    assert "thinking=False;bos=start;eos=end" in prompt
    assert worker.requests[-1] == {
        "operation": "score",
        "input_token_ids": [1, 2, 3],
        "candidate_token_ids": [65, 66],
        "context_tokens": 3,
        "temperature": 1.0,
    }
    assert all(request.get("add_special_tokens") is False for request in worker.requests[:-1])


def rendering_template(name):
    path = Path(__file__).parent / "fixtures" / "rendering_templates.json"
    return ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(
        json.loads(path.read_text())["templates"][name]
    )


@pytest.mark.parametrize("text", ["synthetic text", "literal </think> in supplied evidence"])
def test_always_on_qwen3_thinking_template_is_rejected_before_native_requests(text):
    worker = Worker()
    with pytest.raises(DecisionInputError, match="starts in reasoning mode"):
        decide(
            worker,
            {"end_of_thinking_tag": "\n</think>"},
            rendering_template("qwen3-thinking"),
            text,
            "synthetic question",
            ["First", "Second"],
            1000,
        )
    assert worker.requests == []


@pytest.mark.parametrize("profile", ["qwen3", "qwen3-instruct", "qwen3.5", "lfm2.5-instruct", "gemma-3"])
@pytest.mark.parametrize("text", ["synthetic text", "literal <think> and <think> without closing tags"])
def test_real_direct_answer_templates_accept_literal_reasoning_tags_in_input(profile, text):
    worker = Worker()
    result = decide(worker, {}, rendering_template(profile), text, "synthetic question", ["First", "Second"], 1000)
    assert result["choice_index"] == 1
    assert worker.requests[-1]["operation"] == "score"


def test_supported_codec_reasoning_field_is_disabled_before_inspecting_assistant_suffix():
    template = ImmutableSandboxedEnvironment().from_string(
        "{% for m in messages %}{{ m.content }}{% endfor %}"
        "{% if add_generation_prompt %}assistant:"
        "{% if thinking is not defined or thinking %}<think>{% endif %}{% endif %}"
    )
    codec = {
        "reasoning_config": {"field_name": "thinking", "reasoning_effort_to_field_value": {"no_reasoning": False}},
        "end_of_thinking_tag": "</think>",
    }
    assert decide(Worker(), codec, template, "text", "question", ["First", "Second"], 1000)["choice_index"] == 1


@pytest.mark.parametrize("profile", ["qwen3", "qwen3.5"])
@pytest.mark.parametrize("default", ["medium", "no_reasoning"])
def test_switchable_real_templates_preserve_direct_answers_with_either_reasoning_default(profile, default):
    codec = {
        "reasoning_config": {
            "default_reasoning_effort": default,
            "field_name": "enable_thinking",
            "reasoning_effort_to_field_value": {"medium": True, "no_reasoning": False},
        },
        "end_of_thinking_tag": "\n</think>",
    }
    result = decide(Worker(), codec, rendering_template(profile), "text", "question", ["First", "Second"], 1000)
    assert result["choice_index"] == 1


@pytest.mark.parametrize(
    "source",
    [
        "{% for m in messages %}{{ m.content }}{% endfor %}assistant:<think>",
        "{% if add_generation_prompt %}different prefix{% endif %}content",
    ],
)
def test_unestablished_assistant_suffix_is_rejected_without_scanning_supplied_text(source):
    worker = Worker()
    with pytest.raises(DecisionInputError, match="cannot establish a direct-answer boundary"):
        decide(
            worker,
            {},
            ImmutableSandboxedEnvironment().from_string(source),
            "<think> supplied text </think>",
            "question",
            ["First", "Second"],
            1000,
        )
    assert worker.requests == []


@pytest.mark.parametrize("count", [2, 16])
def test_choice_count_boundaries(count):
    labels = [str(i) for i in range(count)]
    assert validate_inputs("text", "question", labels) == labels


@pytest.mark.parametrize(
    "text, question, answers",
    [
        (None, "question", ["a", "b"]),
        (" ", "question", ["a", "b"]),
        ("text", None, ["a", "b"]),
        ("text", " ", ["a", "b"]),
        ("text", "question", "a,b"),
        ("text", "question", ["a"]),
        ("text", "question", list(map(str, range(17)))),
        ("text", "question", ["a", ""]),
        ("text", "question", ["a", 2]),
        ("text", "question", ["a", " A "]),
    ],
)
def test_invalid_inputs_fail_without_content_in_error(text, question, answers):
    with pytest.raises(DecisionInputError) as error:
        validate_inputs(text, question, answers)
    assert "question" not in str(error.value) or str(error.value) == "Enter a decision question."


@pytest.mark.parametrize("ids", [[65, 66], [65]])
def test_unsupported_multi_token_or_duplicate_answer_letters_stop_before_scoring(ids):
    worker = Worker(tokenize=lambda text, original: ids if len(text) == 1 else original)
    with pytest.raises(DecisionInputError):
        evaluate(worker)
    assert all(request["operation"] != "score" for request in worker.requests)


@pytest.mark.parametrize("ids", [[], [-1], [1000], [True], [1.5]])
def test_invalid_token_readouts_are_protocol_failures(ids):
    with pytest.raises(DecisionProtocolError):
        evaluate(Worker(tokenize=lambda text, original: ids))


def test_context_overflow_is_rejected_without_truncation_or_native_scoring():
    worker = Worker(tokenize=lambda text, original: [1] * 16385 if len(text) > 1 else original)
    with pytest.raises(DecisionInputError, match="context limit"):
        evaluate(worker)
    assert all(request["operation"] != "score" for request in worker.requests)


def test_model_specific_context_overflow_is_recoverable():
    with pytest.raises(DecisionInputError, match="model's context limit"):
        evaluate(Worker(score={"error": "context_overflow"}))


@pytest.mark.parametrize(
    "reply",
    [
        {"error": "private diagnostic fixture"},
        {"candidate_token_ids": [66, 65]},
        {"candidate_token_ids": [True, 66]},
        {"input_tokens": 2},
        {"input_tokens": True},
        {"truncated": True},
        {"truncated": 0},
        {"temperature": True},
        {"temperature": 1.22},
        {"candidate_probabilities": None},
        {"candidate_probabilities": [1.0]},
        {"candidate_probabilities": [True, 0.0]},
        {"candidate_probabilities": [float("nan"), 0.0]},
        {"candidate_probabilities": [float("inf"), 0.0]},
        {"candidate_probabilities": [-0.25, 1.25]},
        {"candidate_probabilities": [0.1, 0.2]},
    ],
)
def test_unusable_score_readouts_are_protocol_failures(reply):
    with pytest.raises(DecisionProtocolError) as error:
        evaluate(Worker(score=reply))
    assert str(error.value) == ""
