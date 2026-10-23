"""Generic, uncalibrated letter-choice scoring for the private desktop controller.

This application prompt is implemented independently of model-specific decision
profiles. It uses the selected converted model's chat template and temperature
1.0; it does not inherit any model's published calibration or qualification.
Request content and native readouts remain in memory and private process pipes.
"""

import json
import math
import os
import subprocess

from .chat import Worker

LETTERS = "ABCDEFGHIJKLMNOP"
MAX_CONTEXT_TOKENS = 16384
SYSTEM = (
    "Select the answer that best satisfies the question using the provided text. "
    "Reply with one uppercase option letter only."
)


class DecisionInputError(Exception):
    """A fixed, content-free input or model-compatibility diagnostic."""


class DecisionProtocolError(Exception):
    """An invalid or unsuccessful native response; the worker must be closed."""


class ScoreWorker(Worker):
    def __init__(self, executable, model, timeout):
        self.timeout = timeout
        self.pending = b""
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("STARK_CAPTURE_", "UZU_CAPTURE_"))
            and key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT"}
        }
        self.process = subprocess.Popen(
            [str(executable), "--score-model-path", str(model)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            start_new_session=True,
            env=environment,
        )


def validate_inputs(text, question, answers):
    if not isinstance(text, str) or not text.strip():
        raise DecisionInputError("Enter text to evaluate.")
    if not isinstance(question, str) or not question.strip():
        raise DecisionInputError("Enter a decision question.")
    if not isinstance(answers, list) or not 2 <= len(answers) <= len(LETTERS):
        raise DecisionInputError("Enter between 2 and 16 answers.")
    if any(not isinstance(label, str) or not label.strip() for label in answers):
        raise DecisionInputError("Each answer must contain text.")
    labels = [label.strip() for label in answers]
    if len({label.casefold() for label in labels}) != len(labels):
        raise DecisionInputError("Use distinct answers.")
    return labels


def token_ids(reply, vocab_size):
    ids = reply.get("token_ids")
    if (
        "error" in reply
        or not isinstance(ids, list)
        or not ids
        or type(reply.get("token_count")) is not int
        or reply["token_count"] != len(ids)
        or any(type(token) is not int or not 0 <= token < vocab_size for token in ids)
    ):
        raise DecisionProtocolError
    return ids


def decide(worker, codec, template, text, question, answers, vocab_size):
    labels = validate_inputs(text, question, answers)
    payload = {
        "text": text,
        "question": question,
        "answers": [{"letter": LETTERS[index], "label": label} for index, label in enumerate(labels)],
    }
    prompt = template.render(
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        tools=None,
        add_generation_prompt=True,
        enable_thinking=False,
        bos_token=codec.get("bos_token") or "",
        eos_token=codec.get("eos_token") or "",
    )
    candidates = []
    for letter in LETTERS[: len(labels)]:
        ids = token_ids(
            worker.request({"operation": "tokenize", "text": letter, "add_special_tokens": False}), vocab_size
        )
        if len(ids) != 1:
            raise DecisionInputError("This model cannot score the required single-token answer letters.")
        candidates.append(ids[0])
    if len(set(candidates)) != len(candidates):
        raise DecisionInputError("This model cannot distinguish the required answer letters.")
    ids = token_ids(worker.request({"operation": "tokenize", "text": prompt, "add_special_tokens": False}), vocab_size)
    if len(ids) > MAX_CONTEXT_TOKENS:
        raise DecisionInputError("The decision exceeds the context limit. Use shorter text or fewer answers.")
    reply = worker.request(
        {
            "operation": "score",
            "input_token_ids": ids,
            "candidate_token_ids": candidates,
            "context_tokens": len(ids),
            "temperature": 1.0,
        }
    )
    if reply.get("error") == "context_overflow":
        raise DecisionInputError("The decision exceeds this model's context limit. Use shorter text or fewer answers.")
    probabilities = reply.get("candidate_probabilities")
    returned_candidates = reply.get("candidate_token_ids")
    if (
        "error" in reply
        or not isinstance(returned_candidates, list)
        or any(type(token) is not int for token in returned_candidates)
        or returned_candidates != candidates
        or type(reply.get("input_tokens")) is not int
        or reply["input_tokens"] != len(ids)
        or reply.get("truncated") is not False
        or type(reply.get("temperature")) not in (int, float)
        or reply["temperature"] != 1.0
        or not isinstance(probabilities, list)
        or len(probabilities) != len(labels)
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
            for value in probabilities
        )
        or not math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-6)
    ):
        raise DecisionProtocolError
    return {
        "choice_index": max(range(len(labels)), key=probabilities.__getitem__),
        "answers": labels,
        "probabilities": probabilities,
        "input_tokens": len(ids),
    }
