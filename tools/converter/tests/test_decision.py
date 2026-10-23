"""Generic decision input and native-readout boundaries, using synthetic data."""

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from stark_convert.decision import DecisionInputError, DecisionProtocolError, decide, validate_inputs


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
