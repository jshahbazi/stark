"""Validation and normalization for the small OpenAI-compatible API."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from .local_api import APIError

_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_FIELDS = {
    "model",
    "messages",
    "stream",
    "stream_options",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "stop",
    "n",
    "presence_penalty",
    "frequency_penalty",
    "logprobs",
    "top_logprobs",
    "response_format",
    "user",
    "metadata",
}


def _bad(message: str, code: str = "invalid_request") -> APIError:
    return APIError(400, message, code)


def _finite_number(value: Any, message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _bad(message)
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise _bad(message) from None
    if not math.isfinite(number):
        raise _bad(message)
    return number


def _content(value: Any) -> str:
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise _bad("Message content must contain valid Unicode.") from None
        return value
    if not isinstance(value, list):
        raise _bad("Message content must be text or text content parts.")
    pieces = []
    for part in value:
        if (
            not isinstance(part, dict)
            or set(part) != {"type", "text"}
            or part["type"] != "text"
            or not isinstance(part["text"], str)
        ):
            raise _bad("Message content parts must be text objects.")
        pieces.append(_content(part["text"]))
    return "".join(pieces)


def _messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise _bad("messages must be a non-empty array.")
    result = []
    for message in value:
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            raise _bad("Each message requires role and content.")
        if set(message) - {"role", "content", "name", "reasoning_content"}:
            raise _bad("Unsupported message fields.", "unsupported_parameter")
        role = message["role"]
        if role == "developer":
            role = "system"
        if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
            raise _bad("Unsupported message role.", "unsupported_parameter")
        if "name" in message and (
            not isinstance(message["name"], str) or len(message["name"]) > 64 or not _NAME.fullmatch(message["name"])
        ):
            raise _bad("Message name must be an ASCII identifier of at most 64 characters.")
        normalized = {"role": role, "content": _content(message["content"])}
        if "reasoning_content" in message:
            reasoning = message["reasoning_content"]
            if role != "assistant" or not isinstance(reasoning, str):
                raise _bad("reasoning_content must be text on an assistant message.", "unsupported_parameter")
            # Some clients replay this text field even when empty. Preserve it
            # for templates that consume assistant reasoning in the history.
            normalized["reasoning_content"] = _content(reasoning)
        result.append(normalized)
    return result


def parse_chat_request(payload: Any, max_output: int, vocab_size: int) -> dict[str, Any]:
    """Validate a text-only request; user and metadata are accepted then discarded."""
    if not isinstance(payload, dict):
        raise _bad("Request body must be an object.")
    unknown = set(payload) - _FIELDS
    if unknown:
        raise _bad("Unsupported request fields.", "unsupported_parameter")
    model = payload.get("model")
    if model is not None and (not isinstance(model, str) or not model):
        raise _bad("model must be a non-empty string.")
    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise _bad("stream must be a boolean.")
    options = payload.get("stream_options")
    if options is not None and (
        not stream
        or not isinstance(options, dict)
        or set(options) - {"include_usage"}
        or not isinstance(options.get("include_usage", False), bool)
    ):
        raise _bad("stream_options supports only include_usage for streaming requests.")
    n = payload.get("n", 1)
    if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n != 1):
        raise _bad("Only n=1 is supported.", "unsupported_parameter")
    aliases = [payload.get("max_tokens"), payload.get("max_completion_tokens")]
    if aliases[0] is not None and aliases[1] is not None:
        raise _bad("Choose one output token limit.")
    maximum = next((x for x in aliases if x is not None), min(512, max_output))
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1 or maximum > max_output:
        raise _bad("Output token limit is outside the loaded model limit.")
    temperature = payload.get("temperature", 0)
    if temperature is None:
        temperature = 0
    temperature = _finite_number(temperature, "temperature must be finite.")
    if not 0 <= temperature <= 2:
        raise _bad("temperature must be between 0 and 2.")
    top_p = payload.get("top_p")
    if top_p is not None:
        top_p = _finite_number(top_p, "top_p must be finite.")
        if not 0 < top_p <= 1:
            raise _bad("top_p must be greater than 0 and at most 1.")
    seed = payload.get("seed", 0)
    top_k = payload.get("top_k")
    if top_k is not None and (type(top_k) is not int or top_k < 1 or (vocab_size > 0 and top_k > vocab_size)):
        raise _bad("top_k must be a positive vocabulary-bounded integer.")
    min_p = payload.get("min_p")
    if min_p is not None:
        min_p = _finite_number(min_p, "min_p must be finite.")
        if not 0 <= min_p <= 1:
            raise _bad("min_p must be between 0 and 1.")
    if seed is None:
        seed = 0
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**64 - 1:
        raise _bad("seed must be an unsigned 64-bit integer.")
    for key in ("presence_penalty", "frequency_penalty"):
        value = payload.get(key, 0)
        if value is not None and _finite_number(value, f"{key} must be zero.") != 0:
            raise _bad(f"{key} must be zero.", "unsupported_parameter")
    if payload.get("logprobs") not in (None, False):
        raise _bad("logprobs is not supported.", "unsupported_parameter")
    if payload.get("top_logprobs") not in (None, 0):
        raise _bad("top_logprobs is not supported.", "unsupported_parameter")
    response_format = payload.get("response_format")
    if response_format is not None and response_format != {"type": "text"}:
        raise _bad("Only response_format type text is supported.", "unsupported_parameter")
    user = payload.get("user")
    if user is not None and (not isinstance(user, str) or len(user) > 256):
        raise _bad("user must be a string of at most 256 characters.")
    metadata = payload.get("metadata")
    if metadata is not None:
        try:
            valid = (
                isinstance(metadata, dict)
                and len(metadata) <= 64
                and len(json.dumps(metadata, ensure_ascii=False, allow_nan=False)) <= 16384
            )
        except (TypeError, ValueError, RecursionError):
            valid = False
        if not valid:
            raise _bad("metadata must be a bounded object.")
    stop = payload.get("stop", [])
    if stop is None:
        stop = []
    if isinstance(stop, str):
        stop = [stop]
    if (
        not isinstance(stop, list)
        or len(stop) > 4
        or any(not isinstance(item, str) or not item or len(item) > 1024 for item in stop)
    ):
        raise _bad("stop must contain at most four non-empty strings of at most 1024 characters.")
    messages = _messages(payload.get("messages"))
    for item in stop:
        _content(item)
    sampling = {"mode": "greedy"}
    if temperature > 0:
        sampling = {"mode": "stochastic", "temperature": temperature}
        if top_p is not None:
            sampling["top_p"] = top_p
        if top_k is not None:
            sampling["top_k"] = top_k
        if min_p is not None:
            sampling["min_p"] = min_p
    return {"model": model, "messages": messages, "maximum": maximum, "sampling": sampling, "seed": seed, "stop": stop}
