"""OneJev qev-labels-v2 prompt protocol for Elephant decision checkpoints.

Adapted from OmniJev/OneJev revision 81ce62f1597c91e46767d4d02d6ac2e18534fe94,
qev/prompt.py, qev/schema.py and qev/answers.py (Apache-2.0). See
_third_party/elephant/PROVENANCE.md. Native template evaluation and image
expansion are performed by Stark's existing private worker paths.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from .decider_protocol import DeciderError, describe

PROMPT_VERSION = "qev-labels-v2"
SYSTEM_PROMPT = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NOT_ONE_TOKEN = {
    "BQ",
    "BZ",
    "CJ",
    "CQ",
    "CZ",
    "DQ",
    "DZ",
    "EJ",
    "EY",
    "FJ",
    "FQ",
    "FV",
    "FZ",
    "GJ",
    "GK",
    "GQ",
    "GZ",
    "HJ",
    "IY",
    "JF",
    "JG",
    "JH",
    "JL",
    "JN",
    "JQ",
}
SLOT_LABELS = tuple(
    list(LETTERS)
    + [a + b for a in LETTERS for b in LETTERS if a + b not in NOT_ONE_TOKEN][: 256 - len(LETTERS)]
)
MAX_SLOTS = len(SLOT_LABELS)
MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
PREFIX_MARKER = "\u0000qev-prefix-end\u0000"
NOUL_DEFAULT_INSTRUCTIONS = "Is the statement true, or is the answer to the question yes?"
NOUL_DEFAULT_TRUE = "the statement is true / the answer is yes"
NOUL_DEFAULT_FALSE = "the statement is false / the answer is no"


def render_entry(entry):
    if entry is None:
        return ""
    if isinstance(entry, str):
        return entry.strip()
    return json.dumps(entry, ensure_ascii=False, indent=2)


def render_state(state):
    return "" if state is None else state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)


def option_entries(question):
    kind = question.get("type")
    if kind == "noul":
        criteria = question.get("criteria") or {}
        return (
            ["yes", "no"],
            [
                render_entry(criteria.get("true")) or NOUL_DEFAULT_TRUE,
                render_entry(criteria.get("false")) or NOUL_DEFAULT_FALSE,
            ],
        )
    if kind == "choice":
        criteria = question["criteria"]
        return list(criteria), [criteria[key] for key in criteria]
    if kind == "score":
        criteria = question["criteria"]
        return [str(index) for index in range(len(criteria))], criteria
    raise DeciderError("Unsupported decision question type.")


def render_question(question):
    kind = question.get("type")
    labels, descriptions = option_entries(question)
    if kind == "noul":
        instructions = render_entry(question.get("instructions")) or NOUL_DEFAULT_INSTRUCTIONS
        names = labels
        header = "Question:"
    elif kind == "choice":
        instructions = render_entry(question.get("instructions")) or "Which option applies to the state?"
        names = labels
        header = "Question:"
    elif kind == "score":
        instructions = render_entry(question.get("instructions")) or "Which level describes the state?"
        names = [f"level {index}" for index in range(len(labels))]
        header = "Rate the state:"
    else:
        raise DeciderError("Unsupported decision question type.")
    lines = []
    for letter, name, description in zip(SLOT_LABELS, names, descriptions, strict=False):
        text = render_entry(description)
        if not text:
            lines.append(f"{letter}. {name}")
        else:
            lines.append(f"{letter}. {name}: {text}")
    letters = list(SLOT_LABELS[: len(labels)])
    answer_line = (
        "Answer with one letter: {letters}." if len(letters) <= len(LETTERS) else "Answer with one label: {letters}."
    )
    return (
        labels,
        f"{header} {instructions}\n\nOptions:\n" + "\n".join(lines) + "\n\n" + answer_line.format(
            letters=", ".join(letters)
        ),
    )


def answer(question, probabilities):
    values = [float(value) for value in probabilities]
    if not values or any(not math.isfinite(value) or value < 0 for value in values) or sum(values) <= 0:
        raise DeciderError("Invalid decision result.")
    values = [value / sum(values) for value in values]
    labels, descriptions = option_entries(question)
    if len(values) != len(labels):
        raise DeciderError("Invalid decision result.")
    if question["type"] == "noul":
        return {"type": "noul", "noul": values[0]}
    best = max(range(len(values)), key=values.__getitem__)
    if question["type"] == "choice":
        confidence = 1.0
        if len(values) > 1:
            uniform = 1 / len(values)
            confidence = (values[best] - uniform) / (1 - uniform)
        return {
            "type": "choice",
            "choice": labels[best],
            "probabilities": dict(zip(labels, values, strict=True)),
            "confidence": max(0.0, min(1.0, confidence)),
        }
    center = (len(values) - 1) / 2
    baseline = sum(abs(index - center) for index in range(len(values))) / len(values)
    distance = sum(probability * abs(index - best) for index, probability in enumerate(values))
    return {
        "type": "score",
        "score": sum(index * probability for index, probability in enumerate(values)),
        "legend": dict(zip(labels, descriptions, strict=True)),
        "probabilities": dict(zip(labels, values, strict=True)),
        "confidence": max(0.0, 1.0 - distance / baseline),
    }


@dataclass(frozen=True, repr=False)
class EncodedQuestion:
    key: str
    input_ids: tuple[int, ...]
    option_count: int
    context_length: int
    block_boundary: int


class Encoder:
    """Render OneJev's one shared-prefix, one-suffix-per-question layout."""

    def __init__(self, tokenizer, template):
        self.tokenizer = tokenizer
        self.template = template

    def validate(self, request):
        if not isinstance(request, dict) or "state" not in request or "questions" not in request:
            raise DeciderError("A decision request needs state and questions.")
        questions = request["questions"]
        if not isinstance(questions, dict) or not questions or not isinstance(request.get("images", []), list):
            raise DeciderError("A decision request needs questions and optional images.")
        for key, question in questions.items():
            if not isinstance(key, str) or not key or not isinstance(question, dict):
                raise DeciderError("Invalid decision question.")
            kind = question.get("type")
            if kind not in {"choice", "noul", "score"}:
                raise DeciderError("Supported decision types are choice, noul and score.")
            if kind == "choice" and (
                not isinstance(question.get("criteria"), dict)
                or not 1 <= len(question["criteria"]) <= MAX_CHOICE_OPTIONS
                or any(not isinstance(name, str) for name in question["criteria"])
            ):
                raise DeciderError("Choice criteria must contain 1 to 255 named options.")
            if kind == "score" and (
                not isinstance(question.get("criteria"), list) or not 2 <= len(question["criteria"]) <= MAX_SCORE_LEVELS
            ):
                raise DeciderError("Score criteria must contain 2 to 10 levels.")
            if kind == "noul" and question.get("criteria") is not None and not isinstance(question["criteria"], dict):
                raise DeciderError("Binary criteria must be an object.")
        try:
            describe(request["state"])
            json.dumps(request["questions"], ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, OverflowError):
            raise DeciderError("Decision content must be JSON serializable.") from None

    def _prompt(self, state, suffix, images):
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": f"<state>\n{render_state(state)}\n</state>\n\n{suffix}"})
        return self.template.render(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}],
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def plan(self, request, max_length=16384):
        self.validate(request)
        if type(max_length) is not int or not 1 <= max_length <= 16384:
            raise DeciderError("Decision context limit must be between 1 and 16384.")
        images = request.get("images", [])
        prefix_prompt = self._prompt(request["state"], PREFIX_MARKER, images)
        cut = prefix_prompt.index(PREFIX_MARKER)
        prefix_text = prefix_prompt[:cut]
        prefix_ids = tuple(self.tokenizer.encode(prefix_text, add_special_tokens=False).ids)
        rendered = []
        for key, question in request["questions"].items():
            labels, suffix = render_question(question)
            full_ids = tuple(
                self.tokenizer.encode(
                    self._prompt(request["state"], suffix, images), add_special_tokens=False
                ).ids
            )
            boundary = len(prefix_ids)
            while boundary > 0 and full_ids[:boundary] != prefix_ids[:boundary]:
                boundary -= 1
            if boundary == 0 or len(full_ids) <= boundary:
                raise DeciderError("Decision prefix boundary is not token-stable.")
            if len(full_ids) > max_length:
                raise DeciderError("Decision input exceeds the context limit; nothing was truncated.")
            rendered.append(EncodedQuestion(key, full_ids, len(labels), boundary, boundary))
        return rendered
