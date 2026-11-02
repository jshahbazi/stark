"""Bounded Chat Completions tools and the pinned Qwen3 tool wire format.

Tools are requests for client execution. JSON schemas are supplied to the model
as hints; this module does not promise schema-constrained generation. The client
must validate execution arguments and obtain any required user approval.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from .local_api import APIError

MAX_TOOLS = 64
MAX_CALLS = 16
MAX_ARGUMENT_BYTES = 64 * 1024
MAX_SCHEMA_BYTES = 256 * 1024
MAX_TOOL_RESULT_BYTES = 256 * 1024
MAX_TOOL_OUTPUT_BYTES = 512 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 16384
_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_CALL_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_PARSER = "qwen3-hermes-v1"
_XML_PARSER = "qwen38-xml-v1"
_QWEN35_XML_PARSER = "qwen35-xml-v1"
_XML_PARSERS = frozenset({_XML_PARSER, _QWEN35_XML_PARSER})
# Official Qwen/Qwen3-0.6B tokenizer_config.json chat_template at revision
# c1899de289a04d12100db370d81485cdf75e47ca, verified 2026-10-31. Admission
# binds the exact template program, not a model name.
QWEN3_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
# Official mlx-community/Qwen3-4B-4bit tokenizer_config.json chat_template at
# revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25, independently reviewed
# for the same Qwen3-Hermes tool-call wire markers and history ordering.
QWEN3_4B_TEMPLATE_SHA256 = "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5"
_SUPPORTED_TEMPLATE_HASHES = frozenset({QWEN3_TEMPLATE_SHA256, QWEN3_4B_TEMPLATE_SHA256})
# fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl tokenizer_config.json chat_template,
# revision 0f94842e3682b49acace4daeb4cf2e65a6f94f83. The same exact program
# is retained in the raw checkpoint's token_codec_config. Unlike Hermes JSON,
# it renders named XML-like function/parameter blocks. No other template is
# admitted by model family/name, and no upstream template text is copied here.
QWEN38_XML_TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
# alibiserikbay/JevK5's converted Qwen3.5 template at revision
# c4f7fdb3aeab5582336406e78d3bef11bf98833d, reviewed in memory against typed
# synthetic history. Its top-level boolean/null values use the exact
# literals True/False/None; nested object/array values still use JSON.
QWEN35_XML_TEMPLATE_SHA256 = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
_XML_TYPES = frozenset({"string", "integer", "number", "boolean", "null", "object", "array"})
_XML_RESERVED = (
    "<tool_call",
    "</tool_call",
    "<function",
    "</function",
    "<parameter",
    "</parameter",
    "<think",
    "</think",
)


def _bad(message: str, code: str = "invalid_request") -> APIError:
    return APIError(400, message, code)


def bounded_json(value: Any, maximum: int, *, label: str) -> None:
    """Bound work before encoding and reject invalid Unicode/non-JSON values."""
    pending = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            raise _bad(f"{label} exceeds the nesting or item limit.")
        if isinstance(item, dict):
            if len(item) > MAX_JSON_NODES:
                raise _bad(f"{label} exceeds the item limit.")
            if any(not isinstance(key, str) for key in item):
                raise _bad(f"{label} must contain JSON object keys.")
            pending.extend((key, depth + 1) for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            if len(item) > MAX_JSON_NODES:
                raise _bad(f"{label} exceeds the item limit.")
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            try:
                if len(item.encode("utf-8")) > maximum:
                    raise _bad(f"{label} exceeds its byte limit.")
            except UnicodeEncodeError:
                raise _bad(f"{label} must contain valid Unicode.") from None
        elif item is not None and type(item) not in (int, float, bool):
            raise _bad(f"{label} must contain JSON values.")
        elif isinstance(item, float) and not math.isfinite(item):
            raise _bad(f"{label} must contain finite numbers.")
    try:
        if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")) > maximum:
            raise _bad(f"{label} exceeds its byte limit.")
    except (ValueError, TypeError, RecursionError, OverflowError):
        raise _bad(f"{label} must be bounded JSON.") from None


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _constant(_value):
    raise ValueError


def strict_arguments(value: Any) -> dict:
    if not isinstance(value, str):
        raise _bad("Function arguments must be a JSON object encoded as a string.")
    try:
        if len(value.encode("utf-8")) > MAX_ARGUMENT_BYTES:
            raise _bad("Function arguments exceed the byte limit.")
        parsed = json.loads(value, object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeEncodeError, ValueError, RecursionError):
        raise _bad("Function arguments must contain one complete, unambiguous JSON object.") from None
    if not isinstance(parsed, dict):
        raise _bad("Function arguments must contain a JSON object.")
    bounded_json(parsed, MAX_ARGUMENT_BYTES, label="Function arguments")
    return parsed


def normalize_calls(value: Any) -> list[dict]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_CALLS:
        raise _bad("Assistant tool_calls must contain between 1 and 16 function calls.")
    result = []
    ids = set()
    for call in value:
        if not isinstance(call, dict) or set(call) != {"id", "type", "function"}:
            raise _bad("Each assistant tool call requires id, type and function.")
        identifier, function = call["id"], call["function"]
        if not isinstance(identifier, str) or not _CALL_ID.fullmatch(identifier) or identifier in ids:
            raise _bad("Tool call IDs must be unique bounded ASCII identifiers.")
        if call["type"] != "function" or not isinstance(function, dict) or set(function) != {"name", "arguments"}:
            raise _bad("Only function tool calls with name and arguments are supported.", "unsupported_parameter")
        if not isinstance(function["name"], str) or not _NAME.fullmatch(function["name"]):
            raise _bad("Function names must be bounded ASCII identifiers.")
        strict_arguments(function["arguments"])
        ids.add(identifier)
        result.append({"id": identifier, "type": "function", "function": dict(function)})
    return result


def validate_history(messages: list[dict]) -> bool:
    """Require complete, unambiguous call/result transactions in client history."""
    pending = {}
    seen = set()
    has_tools = False
    for message in messages:
        if message["role"] == "tool":
            has_tools = True
            identifier = message["tool_call_id"]
            if identifier not in pending:
                raise _bad("Tool results must reference an outstanding assistant tool_call_id.")
            pending.pop(identifier)
        else:
            if pending:
                raise _bad("Return every outstanding tool result before continuing the conversation.")
            for call in message.get("tool_calls", []):
                has_tools = True
                if call["id"] in seen:
                    raise _bad("Tool call IDs must be unique throughout the conversation.")
                seen.add(call["id"])
                pending[call["id"]] = call["function"]["name"]
    if pending:
        raise _bad("Return every outstanding tool result before requesting another completion.")
    return has_tools


def normalize_tool_options(payload: dict) -> dict:
    tools = payload.get("tools")
    if tools is None:
        tools = []
    if not isinstance(tools, list) or len(tools) > MAX_TOOLS:
        raise _bad("tools must contain at most 64 function definitions.")
    bounded_json(tools, MAX_SCHEMA_BYTES, label="Tool definitions")
    names = set()
    normalized = []
    for tool in tools:
        if not isinstance(tool, dict) or set(tool) != {"type", "function"} or tool["type"] != "function":
            raise _bad("Only function tool definitions are supported.", "unsupported_parameter")
        function = tool["function"]
        if not isinstance(function, dict) or set(function) - {"name", "description", "parameters", "strict"}:
            raise _bad("Unsupported function definition fields.", "unsupported_parameter")
        name = function.get("name")
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name in names:
            raise _bad("Function names must be unique bounded ASCII identifiers.")
        if "description" in function and not isinstance(function["description"], str):
            raise _bad("Function description must be a string.")
        parameters = function.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(parameters, dict) or parameters.get("type", "object") != "object":
            raise _bad("Function parameters must describe a JSON object.")
        if "strict" in function and function["strict"] is not False:
            raise _bad("Schema-constrained strict function generation is unavailable.", "unsupported_parameter")
        normalized.append({"type": "function", "function": {**function, "parameters": parameters}})
        names.add(name)
    choice = payload.get("tool_choice", "auto" if tools else "none")
    if isinstance(choice, dict):
        selected = choice.get("function")
        if (
            set(choice) != {"type", "function"}
            or choice["type"] != "function"
            or not isinstance(selected, dict)
            or set(selected) != {"name"}
            or not isinstance(selected["name"], str)
            or selected["name"] not in names
        ):
            raise _bad("Named tool_choice must select a supplied function.")
        choice = {"type": "function", "function": dict(selected)}
    elif not isinstance(choice, str) or choice not in {"auto", "none", "required"}:
        raise _bad("tool_choice must be auto, none, required or a named function.")
    if not tools and choice != "none":
        raise _bad("tool_choice requires function definitions.")
    parallel = payload.get("parallel_tool_calls", True)
    if not isinstance(parallel, bool):
        raise _bad("parallel_tool_calls must be a boolean.")
    return {"tools": normalized, "tool_choice": choice, "parallel_tool_calls": parallel}


def render_tools(request: dict) -> list[dict] | None:
    choice = request["tool_choice"]
    if choice == "none":
        return None
    if isinstance(choice, dict):
        name = choice["function"]["name"]
        return [tool for tool in request["tools"] if tool["function"]["name"] == name]
    return request["tools"] or None


def template_messages(messages: list[dict]) -> list[dict]:
    """Render call arguments as objects and align parallel results by call ID.

    The pinned Qwen template represents calls/results by order, with no call-ID
    field. Ordering each contiguous result group preserves the client's linkage.
    """
    result = []
    order = {}
    index = 0
    while index < len(messages):
        message = messages[index]
        if message["role"] == "tool":
            group = []
            while index < len(messages) and messages[index]["role"] == "tool":
                group.append(dict(messages[index]))
                index += 1
            result.extend(sorted(group, key=lambda item: order[item["tool_call_id"]]))
            continue
        copy = dict(message)
        if "tool_calls" in message:
            # Qwen3-4B's pinned template performs string membership checks on
            # content. OpenAI clients may omit content or send null for a
            # tool-call-only assistant turn, so normalize that case before
            # rendering while preserving any supplied string.
            if copy.get("content") is None:
                copy["content"] = ""
            order = {call["id"]: i for i, call in enumerate(message["tool_calls"])}
            copy["tool_calls"] = [
                {
                    **call,
                    "function": {**call["function"], "arguments": strict_arguments(call["function"]["arguments"])},
                }
                for call in message["tool_calls"]
            ]
        result.append(copy)
        index += 1
    return result


def tool_parser_for(codec: dict | None) -> str | None:
    source = (codec or {}).get("prompt_template")
    if isinstance(source, str):
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if digest in _SUPPORTED_TEMPLATE_HASHES:
            return _PARSER
        if digest == QWEN38_XML_TEMPLATE_SHA256:
            return _XML_PARSER
        if digest == QWEN35_XML_TEMPLATE_SHA256:
            return _QWEN35_XML_PARSER
    return None


def _xml_schema_error(reason="parameter_type"):
    return _bad(
        "This XML tool parser requires direct named parameters with declared or literal-constrained types or bounded "
        "local schema references. Dynamic parameters, root composition, and untyped or unresolved schemas "
        "are unsupported. "
        f"Unsupported schema form: {reason}.",
        "tool_schema_unsupported",
    )


def _xml_types(schema, root, work, memo, depth=0, references=frozenset()):
    work[0] -= 1
    if work[0] < 0 or not isinstance(schema, dict) or depth > MAX_JSON_DEPTH:
        raise _xml_schema_error("resolution_limit_or_invalid_schema")
    declared = schema.get("type")
    if declared is not None:
        values = [declared] if isinstance(declared, str) else declared
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(v, str) or v not in _XML_TYPES for v in values)
        ):
            raise _xml_schema_error("declared_type")
        return frozenset(values)
    if "$ref" in schema:
        reference = schema["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#/") or reference in references:
            raise _xml_schema_error("local_reference")
        identity = (id(root), reference)
        if identity in memo:
            return memo[identity]
        tokens = reference[2:].split("/")
        if len(tokens) > MAX_JSON_DEPTH or any(re.search(r"~(?![01])", token) for token in tokens):
            raise _xml_schema_error("local_reference")
        target = root
        for token in tokens:
            if not isinstance(target, dict):
                raise _xml_schema_error("local_reference")
            target = target.get(token.replace("~1", "/").replace("~0", "~"))
        result = _xml_types(target, root, work, memo, depth + 1, references | {reference})
        memo[identity] = result
        return result
    if "enum" in schema or "const" in schema:
        literals = schema.get("enum", [schema.get("const")])
        if not isinstance(literals, list) or not literals:
            raise _xml_schema_error("literal_constraint")
        # Effect Schema emits enum-only scalar schemas on OpenCode's OpenAI
        # wire. Literal JSON values conclusively determine their representations;
        # this never infers a type from generated model text. Full constraint
        # validation remains the executing client's responsibility.
        return frozenset(
            "null"
            if value is None
            else "boolean"
            if type(value) is bool
            else "integer"
            if type(value) is int
            else "number"
            if type(value) is float
            else "string"
            if isinstance(value, str)
            else "array"
            if isinstance(value, list)
            else "object"
            for value in literals
        )
    alternatives = [schema[key] for key in ("anyOf", "oneOf") if key in schema]
    if len(alternatives) != 1 or not isinstance(alternatives[0], list) or not alternatives[0]:
        raise _xml_schema_error("untyped_or_composed_parameter")
    result = frozenset()
    for candidate in alternatives[0]:
        result |= _xml_types(candidate, root, work, memo, depth + 1, references)
    return result


def _xml_tool_schemas(tools):
    bounded_json(tools, MAX_SCHEMA_BYTES, label="Tool definitions")
    result = {}
    work, memo = [MAX_JSON_NODES], {}
    for tool in tools:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        if not isinstance(function, dict):
            raise _xml_schema_error("function_definition")
        name = function.get("name")
        parameters = function.get("parameters", {"type": "object", "properties": {}})
        if (
            not isinstance(name, str)
            or not _NAME.fullmatch(name)
            or name in result
            or not isinstance(parameters, dict)
        ):
            raise _xml_schema_error("function_or_root_parameters")
        properties = parameters.get("properties", {})
        unsupported_root = {
            "$ref",
            "$dynamicRef",
            "$recursiveRef",
            "allOf",
            "anyOf",
            "oneOf",
            "not",
            "if",
            "then",
            "else",
            "patternProperties",
            "dependentSchemas",
            "dependencies",
            "unevaluatedProperties",
        }
        if parameters.get("type", "object") != "object" or not isinstance(properties, dict):
            raise _xml_schema_error("root_properties")
        if unsupported_root.intersection(parameters):
            raise _xml_schema_error("root_composition_or_dynamic_parameters")
        if parameters.get("additionalProperties", False) is not False:
            raise _xml_schema_error("root_additional_properties")
        if any(not _NAME.fullmatch(key) for key in properties):
            raise _xml_schema_error("parameter_name")
        result[name] = {key: _xml_types(schema, parameters, work, memo) for key, schema in properties.items()}
    return result


def validate_parser_tools(tools: list[dict], parser: str | None) -> None:
    """Reject unsupported XML decoding schemas before queueing/generation.

    This determines parameter representations, not full JSON Schema validity.
    Mixed string/non-string unions are accepted here; an ambiguous generated
    value is rejected later rather than guessed into an executable argument.
    """
    if parser in _XML_PARSERS:
        _xml_tool_schemas(tools)


def _tool_prefix_safe(prefix):
    visible = []
    cursor = 0
    while cursor < len(prefix):
        opening = prefix.find("<think>", cursor)
        closing = prefix.find("</think>", cursor)
        if opening < 0:
            visible.append(prefix[cursor:])
            break
        if 0 <= closing < opening:
            return False
        visible.append(prefix[cursor:opening])
        closing = prefix.find("</think>", opening + len("<think>"))
        nested = prefix.find("<think", opening + len("<think>"))
        if closing < 0 or (nested >= 0 and nested < closing):
            return False
        cursor = closing + len("</think>")
    prefix = "".join(visible)
    if "```" in prefix or "<think" in prefix or "</think" in prefix:
        return False
    quote = None
    escaped = False
    for index, character in enumerate(prefix):
        if escaped:
            escaped = False
        elif quote and character == "\\":
            escaped = True
        elif quote:
            apostrophe = (
                character in {"'", "’"}
                and index > 0
                and index + 1 < len(prefix)
                and prefix[index - 1].isalnum()
                and prefix[index + 1].isalnum()
            )
            if character == quote and not apostrophe:
                quote = None
        elif character in {'"', "`", "“"}:
            quote = "”" if character == "“" else character
        elif character in {"'", "‘"} and (index == 0 or not prefix[index - 1].isalnum()):
            quote = "’" if character == "‘" else character
    return quote is None


def _xml_parameter_value(body, types, invalid, parser):
    if any(marker in body for marker in _XML_RESERVED):
        raise invalid()
    # The pinned renderer adds exactly one LF before and after each value.
    # Inline values are also lossless; a half frame cannot be repaired safely.
    if body.startswith("\n") or body.endswith("\n"):
        if len(body) < 2 or not (body.startswith("\n") and body.endswith("\n")):
            raise invalid()
        body = body[1:-1]
    if types == {"string"}:
        return body
    if parser == _QWEN35_XML_PARSER and body in {"True", "False", "None"}:
        value = {"True": True, "False": False, "None": None}[body]
    else:
        try:
            value = json.loads(body, object_pairs_hook=_pairs, parse_constant=_constant)
        except (ValueError, RecursionError):
            if "string" in types:
                return body
            raise invalid() from None
    kind = (
        "null"
        if value is None
        else "boolean"
        if type(value) is bool
        else "integer"
        if type(value) is int
        else "number"
        if type(value) is float
        else "object"
        if isinstance(value, dict)
        else "array"
        if isinstance(value, list)
        else "string"
    )
    matches = kind in types or (kind == "integer" and "number" in types)
    if "string" in types:
        if matches and kind != "string":
            raise invalid()  # Both the raw string and JSON value fit: no guessed coercion.
        return body
    if not matches:
        raise invalid()
    bounded_json(value, MAX_ARGUMENT_BYTES, label="Function arguments")
    return value


def _parse_xml_calls(text, start, tools, names, call_id_prefix, parallel_tool_calls, invalid, parser):
    schemas = _xml_tool_schemas(tools)
    calls = []
    cursor = start

    def whitespace(position):
        while position < len(text) and text[position].isspace():
            position += 1
        return position

    def named_tag(position, prefix):
        if not text.startswith(prefix, position):
            raise invalid()
        end = text.find(">", position + len(prefix), position + len(prefix) + 65)
        if end < 0:
            raise invalid()
        name = text[position + len(prefix) : end]
        if not _NAME.fullmatch(name):
            raise invalid()
        return name, end + 1

    while cursor < len(text):
        if len(calls) >= MAX_CALLS or not text.startswith("<tool_call>", cursor):
            raise invalid()
        cursor = whitespace(cursor + len("<tool_call>"))
        name, cursor = named_tag(cursor, "<function=")
        if name not in names:
            raise invalid()
        cursor = whitespace(cursor)
        arguments = {}
        argument_bytes = 0
        while text.startswith("<parameter=", cursor):
            parameter, cursor = named_tag(cursor, "<parameter=")
            if parameter not in schemas[name] or parameter in arguments:
                raise invalid()
            end = text.find("</parameter>", cursor, cursor + MAX_ARGUMENT_BYTES + len("</parameter>") + 1)
            if end < 0:
                raise invalid()
            body = text[cursor:end]
            argument_bytes += len(parameter.encode("utf-8")) + len(body.encode("utf-8"))
            if argument_bytes > MAX_ARGUMENT_BYTES:
                raise invalid()
            try:
                arguments[parameter] = _xml_parameter_value(body, schemas[name][parameter], invalid, parser)
            except APIError:
                raise invalid() from None
            cursor = whitespace(end + len("</parameter>"))
        if not text.startswith("</function>", cursor):
            raise invalid()
        cursor = whitespace(cursor + len("</function>"))
        if not text.startswith("</tool_call>", cursor):
            raise invalid()
        try:
            bounded_json(arguments, MAX_ARGUMENT_BYTES, label="Function arguments")
            encoded = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        except (APIError, ValueError, RecursionError, OverflowError):
            raise invalid() from None
        calls.append(
            {
                "id": f"call_{call_id_prefix}_{len(calls)}",
                "type": "function",
                "function": {"name": name, "arguments": encoded},
            }
        )
        cursor = whitespace(cursor + len("</tool_call>"))
    if not parallel_tool_calls and len(calls) != 1:
        raise invalid()
    return {"role": "assistant", "content": text[:start].rstrip() or None, "tool_calls": calls}, "tool_calls"


class ToolTextStream:
    """Stream only text before a possible call; calls require terminal validation.

    Keep marker prefixes and trailing whitespace private until their meaning is
    known. The final parsed content is authoritative, including its rstrip when
    a call follows text. Work is linear in the bounded native output, even for
    a long run of whitespace delivered in many small deltas.
    """

    _markers = ("<tool_call", "</tool_call")

    def __init__(self, deliver):
        self.deliver = deliver
        self.tail = ""
        self.whitespace = []
        self.blocked = False
        self.bytes = 0
        self.emitted = 0
        self.digest = hashlib.sha256()

    def feed(self, text: str) -> None:
        try:
            self.bytes += len(text.encode("utf-8"))
        except UnicodeEncodeError:
            self.blocked = True
        if self.bytes > MAX_TOOL_OUTPUT_BYTES:
            self.blocked = True
        if self.blocked:
            self.tail = ""
            self.whitespace.clear()
            return
        value = self.tail + text
        self.tail = ""
        positions = [position for marker in self._markers if (position := value.find(marker)) >= 0]
        if positions:
            value = value[: min(positions)]
            self.blocked = True
        else:
            # Only a fixed-size suffix can become a marker in the next delta.
            for size in range(min(len(value), max(map(len, self._markers)) - 1), 0, -1):
                if any(marker.startswith(value[-size:]) for marker in self._markers):
                    self.tail, value = value[-size:], value[:-size]
                    break
        visible = value.rstrip()
        if visible:
            self._emit("".join(self.whitespace) + visible)
            self.whitespace.clear()
        if self.blocked:
            self.whitespace.clear()
        elif suffix := value[len(visible) :]:
            self.whitespace.append(suffix)

    def _emit(self, value: str) -> None:
        self.deliver(value)
        self.emitted += len(value)
        self.digest.update(value.encode("utf-8"))

    def finish(self, content: str | None) -> None:
        content = content or ""
        if (
            self.emitted > len(content)
            or hashlib.sha256(content[: self.emitted].encode("utf-8")).digest() != self.digest.digest()
        ):
            raise ValueError("Invalid worker stream")
        if suffix := content[self.emitted :]:
            self._emit(suffix)
        self.tail = ""
        self.whitespace.clear()


def parse_tool_output(
    text: str,
    tools: list[dict],
    *,
    parser: str | None,
    finish_reason: str,
    parallel_tool_calls: bool = True,
    call_id_prefix: str,
    tool_choice: str | dict = "auto",
) -> tuple[dict, str]:
    """Parse only complete calls after generation; never repair model output.

    Streaming callers can emit text prefixes while this path keeps calls private.
    No call may be emitted before the native terminal frame is received.
    """
    try:
        oversized = len(text.encode("utf-8")) > MAX_TOOL_OUTPUT_BYTES
    except UnicodeEncodeError:
        oversized = True
    if oversized:
        raise APIError(502, "The model output exceeded the tool parser's limits.", "invalid_tool_call")
    message = {"role": "assistant", "content": text}
    reason = "length" if finish_reason in {"length", "context"} else "stop"

    def invalid():
        return APIError(
            502, "The model produced an incomplete, invalid or unavailable tool call.", "invalid_tool_call"
        )

    start = text.find("<tool_call")
    if start < 0 and "</tool_call" not in text:
        if parser in _XML_PARSERS and any(
            marker in text for marker in ("<function", "</function", "<parameter", "</parameter")
        ):
            raise invalid()
        if tool_choice == "required" or isinstance(tool_choice, dict):
            raise APIError(502, "The model did not satisfy the requested tool choice.", "tool_choice_not_satisfied")
        return message, reason

    if (
        parser not in {_PARSER, *_XML_PARSERS}
        or tool_choice == "none"
        or not tools
        or finish_reason not in {"eos", "stop_token", "stop"}
    ):
        raise invalid()
    if not isinstance(call_id_prefix, str) or not _CALL_ID.fullmatch(call_id_prefix) or len(call_id_prefix) > 100:
        raise ValueError("Invalid private tool-call ID prefix")
    if start < 0 or "</tool_call" in text[:start] or not _tool_prefix_safe(text[:start]):
        raise invalid()
    names = {tool["function"]["name"] for tool in tools}
    if isinstance(tool_choice, dict):
        names &= {tool_choice["function"]["name"]}
    if parser in _XML_PARSERS:
        return _parse_xml_calls(text, start, tools, names, call_id_prefix, parallel_tool_calls, invalid, parser)
    calls = []
    cursor = start
    decoder = json.JSONDecoder(object_pairs_hook=_pairs, parse_constant=_constant)
    while cursor < len(text):
        if not text.startswith("<tool_call>", cursor):
            raise invalid()
        body_start = cursor + len("<tool_call>")
        while body_start < len(text) and text[body_start].isspace():
            body_start += 1
        if len(calls) >= MAX_CALLS:
            raise invalid()
        try:
            # Decode one JSON value before checking its closing delimiter. A
            # delimiter inside an argument string is ordinary argument data.
            body = text[body_start : body_start + MAX_ARGUMENT_BYTES + 1024]
            call, body_size = decoder.raw_decode(body)
            body = body[:body_size]
            if len(body.encode("utf-8")) > MAX_ARGUMENT_BYTES + 1024:
                raise invalid()
            if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
                raise invalid()
            if (
                not isinstance(call["name"], str)
                or call["name"] not in names
                or not isinstance(call["arguments"], dict)
            ):
                raise invalid()
            bounded_json(call["arguments"], MAX_ARGUMENT_BYTES, label="Function arguments")
            arguments = json.dumps(call["arguments"], ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        except (APIError, ValueError, TypeError, RecursionError, OverflowError):
            raise invalid() from None
        end = body_start + body_size
        while end < len(text) and text[end].isspace():
            end += 1
        if not text.startswith("</tool_call>", end):
            raise invalid()
        calls.append(
            {
                "id": f"call_{call_id_prefix}_{len(calls)}",
                "type": "function",
                "function": {"name": call["name"], "arguments": arguments},
            }
        )
        cursor = end + len("</tool_call>")
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
    if not parallel_tool_calls and len(calls) != 1:
        raise invalid()
    message["content"] = text[:start].rstrip() or None
    message["tool_calls"] = calls
    return message, "tool_calls"
