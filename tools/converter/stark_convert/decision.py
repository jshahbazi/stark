"""Generic, uncalibrated letter-choice scoring for the private desktop controller.

This application prompt is implemented independently of model-specific decision
profiles. It uses the selected converted model's chat template and temperature
1.0; it does not inherit any model's published calibration or qualification.
Request content and native readouts remain in memory and private process pipes.
"""

import json
import math
import os
import re
import subprocess

from .chat import ChatError, Worker, _popen_with_gpu_memory

LETTERS = "ABCDEFGHIJKLMNOP"
MAX_CONTEXT_TOKENS = 16384
MAX_SUPPORTED_CONTEXT_TOKENS = 65536
SYSTEM = (
    "Select the answer that best satisfies the question using the provided text. "
    "Reply with one uppercase option letter only."
)
THINK_TAG = re.compile(r"<(/?)think\s*>", re.IGNORECASE)
BOUNDARY_ERROR = "This model's template cannot establish a direct-answer boundary for decisions."


class DecisionInputError(Exception):
    """A fixed, content-free input or model-compatibility diagnostic."""


class DecisionProtocolError(Exception):
    """An invalid or unsuccessful native response; the worker must be closed."""


def worker_context_limit(ready, *, legacy=False):
    value = ready.get("max_context_tokens", MAX_CONTEXT_TOKENS if legacy else None)
    if type(value) is not int or not 1 <= value <= MAX_SUPPORTED_CONTEXT_TOKENS:
        raise DecisionProtocolError
    return value


class ScoreWorker(Worker):
    def read(self, cancelled=None, request_id=None):
        reply = super().read(cancelled=cancelled, request_id=request_id)
        if reply.get("ready") is True:
            try:
                self.max_context_tokens = worker_context_limit(reply, legacy=True)
            except DecisionProtocolError:
                self.close()
                raise ChatError("Stark returned an invalid scoring context limit.") from None
        return reply

    def __init__(self, executable, model, timeout, gpu_memory_callback=None):
        self.timeout = timeout
        self.pending = b""
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("STARK_CAPTURE_", "UZU_CAPTURE_"))
            and key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT"}
        }
        self.process, telemetry_fd = _popen_with_gpu_memory(
            [str(executable), "--score-model-path", str(model)],
            callback=gpu_memory_callback,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            start_new_session=True,
            env=environment,
        )
        self._closed = False
        try:
            self._start_gpu_memory_reader(telemetry_fd, gpu_memory_callback)
        except BaseException:
            self.close()
            raise


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


def decision_prompt(codec, template, payload):
    """Inspect only the assistant suffix, never reasoning tags in supplied text."""
    arguments = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "tools": None,
        "enable_thinking": False,
        "bos_token": codec.get("bos_token") or "",
        "eos_token": codec.get("eos_token") or "",
    }
    reasoning = codec.get("reasoning_config")
    if reasoning is not None:
        if (
            not isinstance(reasoning, dict)
            or reasoning.get("field_name") not in ("enable_thinking", "thinking")
            or not isinstance(reasoning.get("reasoning_effort_to_field_value"), dict)
            or reasoning["reasoning_effort_to_field_value"].get("no_reasoning") is not False
        ):
            raise DecisionInputError(BOUNDARY_ERROR)
        arguments[reasoning["field_name"]] = False
    end_tag = codec.get("end_of_thinking_tag")
    if end_tag is not None and (not isinstance(end_tag, str) or end_tag.strip() != "</think>"):
        raise DecisionInputError(BOUNDARY_ERROR)
    try:
        prefix = template.render(**arguments, add_generation_prompt=False)
        prompt = template.render(**arguments, add_generation_prompt=True)
    except Exception:
        raise DecisionInputError(BOUNDARY_ERROR) from None
    if not prompt.startswith(prefix) or len(prompt) == len(prefix):
        raise DecisionInputError(BOUNDARY_ERROR)
    depth = 0
    for tag in THINK_TAG.finditer(prompt[len(prefix) :]):
        depth = max(0, depth - 1) if tag.group(1) else depth + 1
    if depth:
        raise DecisionInputError("This model's template starts in reasoning mode and cannot score answers directly.")
    return prompt


def decide(worker, codec, template, text, question, answers, vocab_size):
    labels = validate_inputs(text, question, answers)
    payload = {
        "text": text,
        "question": question,
        "answers": [{"letter": LETTERS[index], "label": label} for index, label in enumerate(labels)],
    }
    prompt = decision_prompt(codec, template, payload)
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
    limit = getattr(worker, "max_context_tokens", MAX_CONTEXT_TOKENS)
    if type(limit) is not int or not 1 <= limit <= MAX_SUPPORTED_CONTEXT_TOKENS:
        raise DecisionProtocolError
    if len(ids) > limit:
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
