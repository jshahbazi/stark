"""Reasoning controls and in-memory separation of assistant reasoning text."""

import hashlib

from .api_tools import GLM53_TEMPLATE_SHA256
from .template_compat import template_program_sha256

# Same source-bound GLM program as api_tools: zai-org/GLM-5.3-Flash
# metadata revision 04c4e9e95c5da8862dced7e5056455116f83a7e0.
_GLM53_PROGRAM = "55dff1823a10de87fa2f4ca4c0986498cd577ed4d495642f6c371e0abbcc243a"
_OPEN = "<think>"
_CLOSE = "</think>"


def prompt_opens_reasoning(prompt):
    return prompt.rstrip().endswith(_OPEN)


def reasoning_options(codec, *, thinking=False, effort=None):
    """Map controls only to the reviewed GLM effort template or boolean input."""
    source = (codec or {}).get("prompt_template", "")
    reasoning = (codec or {}).get("reasoning_config")
    field = reasoning.get("field_name") if isinstance(reasoning, dict) else None
    if not isinstance(field, str) or field not in {"enable_thinking", "thinking"}:
        field = "enable_thinking"
    try:
        glm = isinstance(source, str) and (
            hashlib.sha256(source.encode("utf-8")).hexdigest() == GLM53_TEMPLATE_SHA256
            or template_program_sha256(source) == _GLM53_PROGRAM
        )
    except UnicodeEncodeError:
        glm = False
    if effort is not None and (not glm or effort not in {"low", "high", "max"}):
        raise ValueError("This model does not support the requested reasoning effort.")
    return {
        field: thinking,
        **({"reasoning_effort": effort or ("max" if thinking else "low")} if glm else {}),
    }


def assistant_message(text, *, initial_thinking=False):
    """Split one leading reasoning block; later literal tags stay visible."""
    start = len(_OPEN) if not initial_thinking and text.startswith(_OPEN) else 0
    if not initial_thinking and not start:
        return {"role": "assistant", "content": text}
    closing = text.find(_CLOSE, start)
    if closing < 0:
        return {"role": "assistant", "content": "", "reasoning_content": text[start:]}
    return {
        "role": "assistant",
        "content": text[closing + len(_CLOSE) :],
        "reasoning_content": text[start:closing],
    }


class ReasoningStream:
    """Keep partial boundary tags private and emit the same split as buffering."""

    def __init__(self, content, reasoning, *, initial_thinking=False):
        self.deliver = {"content": content, "reasoning_content": reasoning}
        self.state = "reasoning_content" if initial_thinking else "opening"
        self.initial_thinking = initial_thinking
        self.tail = ""
        self.emitted = {"content": 0, "reasoning_content": 0}
        self.digests = {key: hashlib.sha256() for key in self.deliver}

    def _emit(self, field, value):
        if value:
            self.deliver[field](value)
            self.emitted[field] += len(value)
            self.digests[field].update(value.encode("utf-8"))

    def feed(self, value):
        value, self.tail = self.tail + value, ""
        if self.state == "opening":
            if _OPEN.startswith(value) and len(value) < len(_OPEN):
                self.tail = value
                return
            if value.startswith(_OPEN):
                value = value[len(_OPEN) :]
                self.state = "reasoning_content"
            else:
                self.state = "content"
        if self.state == "reasoning_content":
            closing = value.find(_CLOSE)
            if closing >= 0:
                self._emit("reasoning_content", value[:closing])
                value = value[closing + len(_CLOSE) :]
                self.state = "content"
            else:
                for size in range(min(len(value), len(_CLOSE) - 1), 0, -1):
                    if _CLOSE.startswith(value[-size:]):
                        self.tail, value = value[-size:], value[:-size]
                        break
                self._emit("reasoning_content", value)
                return
        self._emit("content", value)

    def finish(self, text):
        message = assistant_message(text, initial_thinking=self.initial_thinking)
        for field in self.deliver:
            value = message.get(field, "")
            emitted = self.emitted[field]
            if (
                emitted > len(value)
                or hashlib.sha256(value[:emitted].encode("utf-8")).digest() != self.digests[field].digest()
            ):
                raise ValueError("Invalid worker stream")
            self._emit(field, value[emitted:])
        self.tail = ""
