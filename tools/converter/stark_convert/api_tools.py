"""Bounded Chat Completions tools and reviewed model tool wire formats.

Tools are requests for client execution. JSON schemas are supplied to the model
as hints; this module does not promise schema-constrained generation. The client
must validate execution arguments and obtain any required user approval.
"""

from __future__ import annotations

import hashlib
import json
import keyword
import math
import re
from typing import Any

from .json_budget import check_json_structure
from .local_api import APIError
from .resource_budget import INPUT_BYTES
from .template_compat import MAX_TEMPLATE_BYTES, template_program_sha256
from .tool_protocols import (
    FORMATS,
    TEMPLATE_HASHES,
    TEMPLATE_PROGRAMS,
    adapt_history,
    adapt_tools,
    functiongemma_values_supported,
    validate_functiongemma_tools,
)

MAX_TOOLS = 64
MAX_CALLS = 16
MAX_ARGUMENT_BYTES = INPUT_BYTES
MAX_SCHEMA_BYTES = INPUT_BYTES
MAX_TOOL_OUTPUT_BYTES = INPUT_BYTES
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 16384
_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_CALL_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_PARSER = "qwen3-hermes-v1"
_XML_PARSER = "qwen38-xml-v1"
_QWEN35_XML_PARSER = "qwen35-xml-v1"
_GLM_PARSER = "glm53-xml-v1"
_XML_PARSERS = frozenset({_XML_PARSER, _QWEN35_XML_PARSER, _GLM_PARSER})
_HERMES_PARSERS = frozenset({_PARSER, "smollm3-hermes-v1"})
TOOL_PARSERS = frozenset({*_HERMES_PARSERS, *_XML_PARSERS, *FORMATS})
# Official Qwen/Qwen3-0.6B tokenizer_config.json chat_template at revision
# c1899de289a04d12100db370d81485cdf75e47ca, verified 2026-10-31. Admission
# binds the reviewed template program, not a model name.
QWEN3_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
# Official mlx-community/Qwen3-4B-4bit tokenizer_config.json chat_template at
# revision 4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25, independently reviewed
# for the same Qwen3-Hermes tool-call wire markers and history ordering.
QWEN3_4B_TEMPLATE_SHA256 = "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5"
QWEN25_TEMPLATE_SHA256 = "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
_SUPPORTED_TEMPLATE_HASHES = frozenset({QWEN3_TEMPLATE_SHA256, QWEN3_4B_TEMPLATE_SHA256, QWEN25_TEMPLATE_SHA256})
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
GLM53_TEMPLATE_SHA256 = "34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891"
FLINT_TEMPLATE_SHA256 = "7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466"
# Version-1 Jinja AST fingerprints of the hash-verified sources above.
# They admit only syntax-equivalent programs, independently of model metadata.
_SUPPORTED_TEMPLATE_PROGRAMS = {
    "2537497c84cb9ae2a885de78e7531cd44c62f38dadfcfa7565793ce9e38481ac": _PARSER,
    "813d6e292f35cf32a98357c5382ec57ae39ddb658e5b16559c957b119a2859b5": _PARSER,
    # Qwen/Qwen2.5-0.5B-Instruct, revision
    # 7ae557604adf67be50417f59c2c2f167def9a775. Same Hermes envelope;
    # independently rendered definitions, typed calls and result history.
    "8f92b77faedd343fd1d45a534e1fc93a77efb76adddd98a6819e6cd17a0b9bfa": _PARSER,
    "ece3bd43a4194cae4a2d3b6079214373b84f9083ff6258940d0b03634e93f08e": _XML_PARSER,
    "a0feb6a6a23bced6ef842bcb46dad27dc43738bf9acd5751606f24aa343610d8": _QWEN35_XML_PARSER,
    # zai-org/GLM-5.3-Flash metadata revision
    # 04c4e9e95c5da8862dced7e5056455116f83a7e0: bare function name and
    # paired arg_key/arg_value tags, raw strings and JSON non-string values.
    "55dff1823a10de87fa2f4ca4c0986498cd577ed4d495642f6c371e0abbcc243a": _GLM_PARSER,
    # RecursiveSI/flint-int4, revision 3b93f9053b530488405ae9fca51d4900d6ce21f0.
    # The same raw-string/JSON arg-key/value wire format; independently rendered.
    "c61d6063b6aeb383d9a6e80cb62f48164363c6fabb4369d85005c0ae9b9237b8": _GLM_PARSER,
}
_XML_TYPES = frozenset({"string", "integer", "number", "boolean", "null", "object", "array"})
_XML_UNKNOWN = object()
_XML_ANNOTATIONS = frozenset(
    {"title", "description", "$comment", "default", "examples", "deprecated", "readOnly", "writeOnly"}
)
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
# Fixed categories for rejected generated calls, safe to show clients. They
# never include generated text, tool names, argument names or values.
_INVALID_CALL_REASONS = {
    "limit": "generation stopped at the output or context limit",
    "unfinished": "generation did not end normally",
    "disabled": "this request does not allow tool calls",
    "markup": "tool-call markup appeared outside a complete call",
    "prefix": "text before the call contains a code block, quotation or unclosed reasoning",
    "format": "the call does not follow the model's tool-call format",
    "trailing": "text follows the call",
    "tool": "the call names a tool that this request does not offer or allow",
    "argument": "the call uses an argument that the tool does not define, or repeats one",
    "value": "an argument value does not match the tool's schema",
    "xml_value_json": "[xml_value_json] a non-string XML argument is not valid unambiguous JSON",
    "xml_value_type": "[xml_value_type] a decoded XML argument has a type the schema does not allow",
    "xml_value_ambiguous": "[xml_value_ambiguous] an XML argument could be either a raw string or a non-string value",
    "limits": "the call's arguments exceed the size or nesting limits",
    "count": "the reply has more tool calls than this request allows",
}


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
        encoded = value.encode("utf-8")
        if len(encoded) > MAX_ARGUMENT_BYTES:
            raise _bad("Function arguments exceed the byte limit.")
        check_json_structure(encoded)
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


def template_tools(tools, *, parser=None):
    """Adapt definitions for a reviewed renderer; parsing uses original schemas."""
    return adapt_tools(tools, parser)


def template_messages(messages: list[dict], *, parser=None, tools=None) -> list[dict]:
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
    return adapt_history(result, parser, tools)


def tool_parser_for(codec: dict | None) -> str | None:
    source = (codec or {}).get("prompt_template")
    if not isinstance(source, str) or not source or len(source) > MAX_TEMPLATE_BYTES:
        return None
    try:
        encoded = source.encode("utf-8")
    except UnicodeEncodeError:
        return None
    if len(encoded) > MAX_TEMPLATE_BYTES:
        return None
    digest = hashlib.sha256(encoded).hexdigest()
    if digest in TEMPLATE_HASHES:
        return TEMPLATE_HASHES[digest]
    if digest in _SUPPORTED_TEMPLATE_HASHES:
        return _PARSER
    if digest == QWEN38_XML_TEMPLATE_SHA256:
        return _XML_PARSER
    if digest == QWEN35_XML_TEMPLATE_SHA256:
        return _QWEN35_XML_PARSER
    if digest in {GLM53_TEMPLATE_SHA256, FLINT_TEMPLATE_SHA256}:
        return _GLM_PARSER
    program = template_program_sha256(source)
    return _SUPPORTED_TEMPLATE_PROGRAMS.get(program) or TEMPLATE_PROGRAMS.get(program)


def tool_support_for(codec: dict | None) -> dict:
    """Report template protocol support without claiming model task competence."""
    source = codec.get("prompt_template") if isinstance(codec, dict) else None
    digest = None
    if isinstance(source, str) and 0 < len(source) <= MAX_TEMPLATE_BYTES:
        try:
            encoded = source.encode("utf-8")
            if len(encoded) <= MAX_TEMPLATE_BYTES:
                digest = hashlib.sha256(encoded).hexdigest()
        except UnicodeEncodeError:
            pass
    parser = tool_parser_for(codec) if digest is not None else None
    return {
        "status": "available" if parser else "unavailable" if digest else "unchecked",
        "parser": parser,
        "template_sha256": digest,
        "model_behavior": "unverified",
    }


def _xml_schema_error(reason="parameter_type"):
    return _bad(
        "This XML tool parser requires direct named parameters with declared or literal-constrained types or bounded "
        "local schema references. Dynamic parameters, root composition, and unresolved or unsupported schemas "
        "are unsupported. "
        f"Unsupported schema form: {reason}.",
        "tool_schema_unsupported",
    )


def _xml_types(schema, root, work, memo, depth=0, references=frozenset()):
    work[0] -= 1
    if work[0] < 0 or not isinstance(schema, dict) or depth > MAX_JSON_DEPTH:
        raise _xml_schema_error("resolution_limit_or_invalid_schema")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise _xml_schema_error("literal_constraint")
    declared = schema.get("type")
    if "type" in schema:
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
    if set(schema) <= _XML_ANNOTATIONS:
        if (
            any(key in schema and not isinstance(schema[key], str) for key in ("title", "description", "$comment"))
            or any(key in schema and type(schema[key]) is not bool for key in ("deprecated", "readOnly", "writeOnly"))
            or ("examples" in schema and not isinstance(schema["examples"], list))
        ):
            raise _xml_schema_error("invalid_annotation")
        return _XML_UNKNOWN
    alternatives = [schema[key] for key in ("anyOf", "oneOf") if key in schema]
    if len(alternatives) != 1 or not isinstance(alternatives[0], list) or not alternatives[0]:
        reason = "untyped_all_of" if "allOf" in schema else "untyped_constraint_or_unknown_keyword"
        raise _xml_schema_error(reason)
    result = frozenset()
    unknown = False
    for candidate in alternatives[0]:
        resolved = _xml_types(candidate, root, work, memo, depth + 1, references)
        if resolved is _XML_UNKNOWN:
            unknown = True
        else:
            result |= resolved
    return _XML_UNKNOWN if unknown else result


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
    Empty or annotation-only parameters can remain unused in a definition;
    generating one is rejected because its wire value has no conclusive type.
    """
    if parser in _XML_PARSERS:
        _xml_tool_schemas(tools)
    if parser == "functiongemma-v1":
        validate_functiongemma_tools(tools)
    if parser == "lfm2-pythonic-v1":
        for tool in tools:
            function = tool["function"]
            properties = function.get("parameters", {}).get("properties", {})
            if not isinstance(properties, dict):
                raise _bad("This tool format requires named function parameters.", "tool_schema_unsupported")
            names = [function["name"], *properties]
            if any(
                not isinstance(name, str)
                or len(name) > 64
                or not name.isascii()
                or not name.isidentifier()
                or keyword.iskeyword(name)
                for name in names
            ):
                raise _bad(
                    "This tool format requires Python-style function and parameter identifiers.",
                    "tool_schema_unsupported",
                )


def _tool_prefix_safe(prefix, *, initial_thinking=False):
    if initial_thinking:
        # The trusted renderer may have opened this block in the input prompt.
        # Generated text must close it before an executable call can begin.
        prefix = "<think>" + prefix
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
    if any(marker in prefix for marker in ("```", "~~~", "<!--", "<think", "</think")):
        return False
    if re.search(r"^[ \t]*>|</?(?:pre|code|blockquote|q)(?=[\s/>])", prefix, re.IGNORECASE | re.MULTILINE):
        # Conservatively reject structural quotation/code contexts even when
        # closed before the call. Plain comparisons and similarly named tags
        # are distinct. This bounded check never interprets intent from prose.
        return False
    quote = None
    code_ticks = 0
    escaped = False
    index = 0
    while index < len(prefix):
        character = prefix[index]
        if code_ticks or (character == "`" and quote is None):
            if character == "`":
                end = index + 1
                while end < len(prefix) and prefix[end] == "`":
                    end += 1
                width = end - index
                if not code_ticks:
                    code_ticks = width
                elif width == code_ticks:
                    code_ticks = 0
                index = end
            else:
                index += 1
            # Inline code closes only on a matching run. Quotes and backslash
            # escapes inside it cannot turn an example into an executable call.
            continue
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
        elif character in {'"', "“"}:
            quote = "”" if character == "“" else character
        elif character in {"'", "‘"} and (index == 0 or not prefix[index - 1].isalnum()):
            quote = "’" if character == "‘" else character
        index += 1
    return quote is None and code_ticks == 0


def _xml_parameter_value(body, types, invalid, parser):
    reserved = _XML_RESERVED + (
        ("<arg_key", "</arg_key", "<arg_value", "</arg_value") if parser == _GLM_PARSER else ()
    )
    if any(marker in body for marker in reserved):
        raise invalid("format")
    # The Qwen renderers add exactly one LF before and after each value.
    # Inline values are also lossless; a half frame cannot be repaired safely.
    if parser != _GLM_PARSER and (body.startswith("\n") or body.endswith("\n")):
        if len(body) < 2 or not (body.startswith("\n") and body.endswith("\n")):
            raise invalid("format")
        body = body[1:-1]
    if types == {"string"}:
        return body
    if parser == _QWEN35_XML_PARSER and body in {"True", "False", "None"}:
        value = {"True": True, "False": False, "None": None}[body]
    else:
        try:
            check_json_structure(body.encode("utf-8"))
            value = json.loads(body, object_pairs_hook=_pairs, parse_constant=_constant)
        except (ValueError, RecursionError):
            if "string" in types:
                return body
            raise invalid("xml_value_json") from None
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
            raise invalid("xml_value_ambiguous")  # Both representations fit: no guessed coercion.
        return body
    if not matches:
        raise invalid("xml_value_type")
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
            raise invalid("format")
        end = text.find(">", position + len(prefix), position + len(prefix) + 65)
        if end < 0:
            raise invalid("format")
        name = text[position + len(prefix) : end]
        if not _NAME.fullmatch(name):
            raise invalid("format")
        return name, end + 1

    while cursor < len(text):
        if len(calls) >= MAX_CALLS:
            raise invalid("count")
        if not text.startswith("<tool_call>", cursor):
            raise invalid("trailing" if calls else "format")
        cursor = whitespace(cursor + len("<tool_call>"))
        name, cursor = named_tag(cursor, "<function=")
        if name not in names:
            raise invalid("tool")
        cursor = whitespace(cursor)
        arguments = {}
        argument_bytes = 0
        while text.startswith("<parameter=", cursor):
            parameter, cursor = named_tag(cursor, "<parameter=")
            if parameter not in schemas[name] or parameter in arguments:
                raise invalid("argument")
            end = text.find("</parameter>", cursor, cursor + MAX_ARGUMENT_BYTES + len("</parameter>") + 1)
            if end < 0:
                # A closing tag beyond the bounded window means an oversized value.
                raise invalid("limits" if text.find("</parameter>", cursor) >= 0 else "format")
            body = text[cursor:end]
            argument_bytes += len(parameter.encode("utf-8")) + len(body.encode("utf-8"))
            if argument_bytes > MAX_ARGUMENT_BYTES:
                raise invalid("limits")
            if schemas[name][parameter] is _XML_UNKNOWN:
                raise APIError(
                    502,
                    "The model selected a tool argument whose schema does not determine its XML value type. "
                    "No tool calls from this completion were released.",
                    "tool_argument_type_unsupported",
                )
            try:
                arguments[parameter] = _xml_parameter_value(body, schemas[name][parameter], invalid, parser)
            except APIError as error:
                if error.code == "invalid_tool_call":
                    raise
                raise invalid("limits") from None
            cursor = whitespace(end + len("</parameter>"))
        if not text.startswith("</function>", cursor):
            raise invalid("format")
        cursor = whitespace(cursor + len("</function>"))
        if not text.startswith("</tool_call>", cursor):
            raise invalid("format")
        try:
            bounded_json(arguments, MAX_ARGUMENT_BYTES, label="Function arguments")
        except (APIError, ValueError, RecursionError):
            raise invalid("limits") from None
        try:
            encoded = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        except (ValueError, RecursionError, OverflowError):
            raise invalid("value") from None
        calls.append(
            {
                "id": f"call_{call_id_prefix}_{len(calls)}",
                "type": "function",
                "function": {"name": name, "arguments": encoded},
            }
        )
        cursor = whitespace(cursor + len("</tool_call>"))
    if not parallel_tool_calls and len(calls) != 1:
        raise invalid("count")
    return {"role": "assistant", "content": text[:start].rstrip() or None, "tool_calls": calls}, "tool_calls"


def _parse_glm_calls(text, start, tools, names, call_id_prefix, parallel_tool_calls, invalid):
    """Decode the pinned GLM arg-key/value grammar without repairing output."""
    schemas = _xml_tool_schemas(tools)
    calls = []
    cursor = start
    while cursor < len(text):
        if len(calls) >= MAX_CALLS:
            raise invalid("count")
        if not text.startswith("<tool_call>", cursor):
            raise invalid("trailing" if calls else "format")
        cursor += len("<tool_call>")
        end = text.find("<", cursor, cursor + 66)
        if end < 0:
            raise invalid("format")
        name = text[cursor:end]
        if not _NAME.fullmatch(name):
            raise invalid("format")
        if name not in names:
            raise invalid("tool")
        cursor = end
        arguments = {}
        argument_bytes = 0
        while text.startswith("<arg_key>", cursor):
            cursor += len("<arg_key>")
            end = text.find("</arg_key>", cursor, cursor + 65 + len("</arg_key>"))
            if end < 0:
                raise invalid("format")
            parameter = text[cursor:end]
            if not _NAME.fullmatch(parameter) or parameter not in schemas[name] or parameter in arguments:
                raise invalid("argument")
            cursor = end + len("</arg_key>")
            if not text.startswith("<arg_value>", cursor):
                raise invalid("format")
            cursor += len("<arg_value>")
            end = text.find("</arg_value>", cursor, cursor + MAX_ARGUMENT_BYTES + len("</arg_value>") + 1)
            if end < 0:
                raise invalid("limits" if text.find("</arg_value>", cursor) >= 0 else "format")
            body = text[cursor:end]
            argument_bytes += len(parameter.encode("utf-8")) + len(body.encode("utf-8"))
            if argument_bytes > MAX_ARGUMENT_BYTES:
                raise invalid("limits")
            if schemas[name][parameter] is _XML_UNKNOWN:
                raise APIError(
                    502,
                    "The model selected a tool argument whose schema does not determine its XML value type. "
                    "No tool calls from this completion were released.",
                    "tool_argument_type_unsupported",
                )
            try:
                arguments[parameter] = _xml_parameter_value(body, schemas[name][parameter], invalid, _GLM_PARSER)
                bounded_json(arguments, MAX_ARGUMENT_BYTES, label="Function arguments")
            except APIError as error:
                if error.code in {"invalid_tool_call", "tool_argument_type_unsupported"}:
                    raise
                raise invalid("limits") from None
            cursor = end + len("</arg_value>")
        if not text.startswith("</tool_call>", cursor):
            raise invalid("format")
        calls.append(
            {
                "id": f"call_{call_id_prefix}_{len(calls)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                },
            }
        )
        cursor += len("</tool_call>")
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
    if not parallel_tool_calls and len(calls) != 1:
        raise invalid("count")
    return {"role": "assistant", "content": text[:start].rstrip() or None, "tool_calls": calls}, "tool_calls"


def _pythonic_frame(body, closing, invalid):
    """Find a frame boundary outside quoted literals in one bounded pass."""
    cursor, quote = 0, None
    while cursor < len(body):
        if quote is not None:
            if body[cursor] == "\\":
                cursor += 2
                continue
            if body.startswith(quote, cursor):
                cursor += len(quote)
                quote = None
                continue
        elif body.startswith(closing, cursor):
            return body[:cursor].strip(), body[cursor + len(closing) :]
        elif body[cursor] in "\"'":
            quote = body[cursor] * (3 if body.startswith(body[cursor] * 3, cursor) else 1)
            cursor += len(quote)
            continue
        cursor += 1
    raise invalid("format")


def _parse_additional_output(
    text, tools, parser, finish_reason, parallel, call_id_prefix, choice, initial_thinking, invalid
):
    opening, closing, kind, argument_key = FORMATS[parser]
    start = text.find(opening)
    bare_prefix = "{" if parser == "llama-json-v1" else "["
    bare = parser in {"llama-json-v1", "phi4-json-v1"} and start < 0 and text.lstrip().startswith(bare_prefix)
    if bare:
        start = len(text) - len(text.lstrip())
        body_start = start
    else:
        body_start = start + len(opening)
    if start < 0:
        if any(marker in text for marker in TOOL_MARKERS):
            raise invalid("markup")
        if choice == "required" or isinstance(choice, dict):
            raise APIError(502, "The model did not satisfy the requested tool choice.", "tool_choice_not_satisfied")
        return {"role": "assistant", "content": text}, "length" if finish_reason in {"length", "context"} else "stop"
    if choice == "none" or not tools:
        raise invalid("disabled")
    if finish_reason not in {"eos", "stop_token", "stop"}:
        raise invalid("limit" if finish_reason in {"length", "context"} else "unfinished")
    if not isinstance(call_id_prefix, str) or not _CALL_ID.fullmatch(call_id_prefix) or len(call_id_prefix) > 100:
        raise ValueError("Invalid private tool-call ID prefix")
    prefix = text[:start]
    if any(marker in prefix for marker in TOOL_MARKERS):
        raise invalid("markup")
    if parser != "lfm2-pythonic-v1" and prefix.strip():
        raise invalid("prefix")
    if not _tool_prefix_safe(prefix, initial_thinking=initial_thinking):
        raise invalid("prefix")
    body = text[body_start:].strip()
    suffix = ""
    if kind == "pythonic":
        body, suffix = _pythonic_frame(text[body_start:], closing, invalid)
        if any(marker in suffix for marker in TOOL_MARKERS):
            raise invalid("markup")
    elif closing and not bare:
        if not body.endswith(closing):
            raise invalid("format")
        body = body[: -len(closing)].strip()
    if kind == "pythonic":
        from .tool_pythonic import parse_pythonic_payload

        functions = parse_pythonic_payload(body, invalid=invalid)
    elif kind == "functiongemma":
        from .tool_functiongemma import parse_functiongemma_calls

        functions = parse_functiongemma_calls(text[start:], invalid=invalid)
    else:
        try:
            check_json_structure(body.encode("utf-8"))
            decoded = json.loads(body, object_pairs_hook=_pairs, parse_constant=_constant)
        except (ValueError, RecursionError):
            raise invalid("format") from None
        items = [decoded] if kind == "object" else decoded
        if not isinstance(items, list) or len(items) > MAX_CALLS:
            raise invalid("count")
        if not items:
            if parser != "smollm2-json-v1":
                raise invalid("count")
            if choice == "required" or isinstance(choice, dict):
                raise APIError(
                    502, "The model did not satisfy the requested tool choice.", "tool_choice_not_satisfied"
                )
            return {"role": "assistant", "content": prefix.rstrip() or None}, "stop"
        functions, identifiers = [], set()
        for item in items:
            allowed = {"name", argument_key} | ({"id"} if parser == "mistral-json-v1" else set())
            if not isinstance(item, dict) or set(item) - allowed or not {"name", argument_key} <= set(item):
                raise invalid("format")
            if "id" in item:
                identifier = item["id"]
                if (
                    not isinstance(identifier, str)
                    or re.fullmatch(r"[A-Za-z0-9]{9}", identifier) is None
                    or identifier in identifiers
                ):
                    raise invalid("format")
                identifiers.add(identifier)
            functions.append((item["name"], item[argument_key]))
    names = {tool["function"]["name"] for tool in tools}
    if isinstance(choice, dict):
        names &= {choice["function"]["name"]}
    calls = []
    for name, arguments in functions:
        if not isinstance(name, str) or not _NAME.fullmatch(name) or name not in names:
            raise invalid("tool")
        if not isinstance(arguments, dict):
            raise invalid("format")
        if parser == "functiongemma-v1" and not functiongemma_values_supported(arguments):
            raise invalid("value")
        try:
            bounded_json(arguments, MAX_ARGUMENT_BYTES, label="Function arguments")
        except APIError:
            raise invalid("limits") from None
        calls.append(
            {
                "id": f"call_{call_id_prefix}_{len(calls)}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False, allow_nan=False)},
            }
        )
    if not parallel and len(calls) != 1:
        raise invalid("count")
    return {"role": "assistant", "content": (prefix.rstrip() + suffix) or None, "tool_calls": calls}, "tool_calls"


TOOL_MARKERS = tuple(
    dict.fromkeys(
        [
            "<tool_call",
            "</tool_call",
            *(
                marker
                for opening, closing, _kind, _argument in FORMATS.values()
                for marker in (opening, closing)
                if marker
            ),
        ]
    )
)


class ToolTextStream:
    """Stream only text before a possible call; calls require terminal validation.

    Keep marker prefixes and trailing whitespace private until their meaning is
    known. The final parsed content is authoritative, including its rstrip when
    a call follows text. Work is linear in the bounded native output, even for
    a long run of whitespace delivered in many small deltas.
    """

    _markers = ("<tool_call", "</tool_call")

    def __init__(self, deliver, *, parser=None):
        self.deliver = deliver
        self._markers = TOOL_MARKERS if parser in FORMATS else self._markers
        self._bare_pending = parser in {"llama-json-v1", "phi4-json-v1"}
        self._bare_prefix = "{" if parser == "llama-json-v1" else "["
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
        if self._bare_pending:
            text = self.tail + text
            self.tail = ""
            if not text.strip():
                self.tail = text
                return
            self._bare_pending = False
            if text.lstrip().startswith(self._bare_prefix):
                self.blocked = True
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
    initial_thinking: bool = False,
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

    def invalid(category):
        # Clients see which rule rejected the call, never what the model wrote.
        return APIError(
            502,
            "The model produced an incomplete, invalid or unavailable tool call: "
            + _INVALID_CALL_REASONS[category]
            + ".",
            "invalid_tool_call",
        )

    if parser in FORMATS:
        return _parse_additional_output(
            text,
            tools,
            parser,
            finish_reason,
            parallel_tool_calls,
            call_id_prefix,
            tool_choice,
            initial_thinking,
            invalid,
        )

    start = text.find("<tool_call")
    if start < 0 and "</tool_call" not in text:
        if parser == _GLM_PARSER and any(
            marker in text for marker in ("<arg_key", "</arg_key", "<arg_value", "</arg_value")
        ):
            raise invalid("markup")
        if parser in _XML_PARSERS and any(
            marker in text for marker in ("<function", "</function", "<parameter", "</parameter")
        ):
            raise invalid("markup")
        if tool_choice == "required" or isinstance(tool_choice, dict):
            raise APIError(502, "The model did not satisfy the requested tool choice.", "tool_choice_not_satisfied")
        return message, reason

    if parser not in TOOL_PARSERS or tool_choice == "none" or not tools:
        raise invalid("disabled")
    if finish_reason not in {"eos", "stop_token", "stop"}:
        raise invalid("limit" if reason == "length" else "unfinished")
    if not isinstance(call_id_prefix, str) or not _CALL_ID.fullmatch(call_id_prefix) or len(call_id_prefix) > 100:
        raise ValueError("Invalid private tool-call ID prefix")
    if start < 0 or "</tool_call" in text[:start]:
        raise invalid("markup")
    if parser == _GLM_PARSER and any(
        marker in text[:start] for marker in ("<arg_key", "</arg_key", "<arg_value", "</arg_value")
    ):
        raise invalid("markup")
    if not _tool_prefix_safe(text[:start], initial_thinking=initial_thinking):
        raise invalid("prefix")
    names = {tool["function"]["name"] for tool in tools}
    if isinstance(tool_choice, dict):
        names &= {tool_choice["function"]["name"]}
    if parser == _GLM_PARSER:
        return _parse_glm_calls(text, start, tools, names, call_id_prefix, parallel_tool_calls, invalid)
    if parser in _XML_PARSERS:
        return _parse_xml_calls(text, start, tools, names, call_id_prefix, parallel_tool_calls, invalid, parser)
    calls = []
    cursor = start
    decoder = json.JSONDecoder(object_pairs_hook=_pairs, parse_constant=_constant)
    while cursor < len(text):
        if not text.startswith("<tool_call>", cursor):
            raise invalid("trailing" if calls else "format")
        body_start = cursor + len("<tool_call>")
        while body_start < len(text) and text[body_start].isspace():
            body_start += 1
        if len(calls) >= MAX_CALLS:
            raise invalid("count")
        try:
            # Decode one JSON value before checking its closing delimiter. A
            # delimiter inside an argument string is ordinary argument data.
            body = text[body_start : body_start + MAX_ARGUMENT_BYTES + 1024]
            call, body_size = decoder.raw_decode(body)
            oversized = len(body[:body_size].encode("utf-8")) > MAX_ARGUMENT_BYTES + 1024
        except (ValueError, RecursionError):
            # A value cut off by the bounded window is oversized rather than malformed.
            raise invalid("limits" if len(text) > body_start + len(body) else "format") from None
        if oversized:
            raise invalid("limits")
        if (
            not isinstance(call, dict)
            or set(call) != {"name", "arguments"}
            or not isinstance(call["name"], str)
            or not isinstance(call["arguments"], dict)
        ):
            raise invalid("format")
        if call["name"] not in names:
            raise invalid("tool")
        try:
            bounded_json(call["arguments"], MAX_ARGUMENT_BYTES, label="Function arguments")
        except (APIError, ValueError, TypeError, RecursionError):
            raise invalid("limits") from None
        try:
            arguments = json.dumps(call["arguments"], ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        except (ValueError, TypeError, RecursionError, OverflowError):
            raise invalid("value") from None
        end = body_start + body_size
        while end < len(text) and text[end].isspace():
            end += 1
        if not text.startswith("</tool_call>", end):
            raise invalid("format")
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
        raise invalid("count")
    message["content"] = text[:start].rstrip() or None
    message["tool_calls"] = calls
    return message, "tool_calls"
