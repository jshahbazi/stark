"""Bounded parser for LFM2/LFM2.5 Pythonic function-call payloads.

The payload is data. This module never evaluates or compiles generated source.
"""

from __future__ import annotations

import ast
import io
import json
import math
import re
import tokenize
import warnings

MAX_PAYLOAD_BYTES = 512 * 1024
MAX_CALLS = 16
MAX_ARGUMENT_BYTES = 64 * 1024
MAX_AST_NODES = 16_384
MAX_AST_DEPTH = 64
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z", re.ASCII)
_DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z", re.ASCII)


def parse_pythonic_payload(payload: str, *, invalid) -> list[tuple[str, dict]]:
    """Parse exactly ``[name(key=value,...), ...]`` or return ``invalid`` error.

    ``invalid(category)`` constructs the caller's sanitized APIError. Categories
    are limited to ``format``, ``limits``, and ``value``.
    """

    def reject(category="format"):
        raise invalid(category)

    if not isinstance(payload, str):
        reject()
    try:
        if len(payload.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            reject("limits")
    except UnicodeEncodeError:
        reject()

    try:
        # Bound AST creation before walking; ast.parse only builds syntax and
        # does not execute generated code.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tree = ast.parse(payload, mode="eval")
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError):
        reject()

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(payload).readline))
    except (tokenize.TokenError, IndentationError):
        reject()
    meaningful = [
        t
        for t in tokens
        if t.type
        not in (
            tokenize.ENCODING,
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.COMMENT,
            tokenize.ENDMARKER,
        )
    ]
    if any(t.type == tokenize.COMMENT for t in tokens):
        reject()
    for previous, current in zip(meaningful, meaningful[1:], strict=False):
        if previous.type == tokenize.STRING and current.type == tokenize.STRING:
            reject("value")
        if current.type == tokenize.STRING and previous.type == tokenize.NAME:
            reject("value")

    pending = [(tree, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > MAX_AST_NODES or depth > MAX_AST_DEPTH:
            reject("limits")
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))

    if not isinstance(tree.body, ast.List) or not 1 <= len(tree.body.elts) <= MAX_CALLS:
        reject()

    encoded_payload = payload.encode("utf-8")
    encoded_lines = encoded_payload.splitlines(keepends=True)
    line_offsets = [0]
    for line in encoded_lines:
        line_offsets.append(line_offsets[-1] + len(line))

    def segment(node):
        start = line_offsets[node.lineno - 1] + node.col_offset
        end = line_offsets[node.end_lineno - 1] + node.end_col_offset
        return encoded_payload[start:end].decode("utf-8")

    def value(node):
        if isinstance(node, ast.Constant):
            v = node.value
            source = segment(node)
            if type(v) in (str, bool, int) or v is None:
                if isinstance(v, str) and (not source or source[0] not in "'\""):
                    reject("value")
                if type(v) is bool and source not in {"True", "False"}:
                    reject("value")
                if type(v) is int and (not source or not _DECIMAL.fullmatch(source)):
                    reject("value")
                return v
            if type(v) is float and math.isfinite(v):
                if not source or not _DECIMAL.fullmatch(source):
                    reject("value")
                mantissa = re.split("[eE]", source)[0].lstrip("+-")
                if v == 0.0 and any(ch in "123456789" for ch in mantissa):
                    reject("value")
                return v
            reject("value")
        if isinstance(node, ast.Name) and node.id in {"true", "false", "null"}:
            return {"true": True, "false": False, "null": None}[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            operand = node.operand
            if not isinstance(operand, ast.Constant) or type(operand.value) not in (int, float):
                reject("value")
            source = segment(node)
            unsigned = source[1:] if source and source[0] in "+-" else source
            if not unsigned or not _DECIMAL.fullmatch(unsigned):
                reject("value")
            number = operand.value if isinstance(node.op, ast.UAdd) else -operand.value
            if isinstance(number, float) and not math.isfinite(number):
                reject("value")
            mantissa = re.split("[eE]", unsigned)[0]
            if isinstance(number, float) and number == 0.0 and any(ch in "123456789" for ch in mantissa):
                reject("value")
            return number
        if isinstance(node, (ast.List, ast.Dict)):
            if isinstance(node, ast.List):
                result = [value(item) for item in node.elts]
            else:
                result = {}
                for key_node, item_node in zip(node.keys, node.values, strict=True):
                    if key_node is None:
                        reject("value")
                    key = value(key_node)
                    if not isinstance(key, str) or key in result:
                        reject("value")
                    result[key] = value(item_node)
            try:
                if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")) > MAX_ARGUMENT_BYTES:
                    reject("limits")
            except (UnicodeEncodeError, ValueError, TypeError, RecursionError, OverflowError):
                reject("value")
            return result
        reject("value")

    calls = []
    for call in tree.body.elts:
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            reject()
        name = call.func.id
        if not _IDENTIFIER.fullmatch(name) or call.args:
            reject()
        args = {}
        for kw in call.keywords:
            if kw.arg is None or not _IDENTIFIER.fullmatch(kw.arg) or kw.arg in args:
                reject()
            args[kw.arg] = value(kw.value)
        try:
            pending_values = [(args, 0)]
            value_nodes = 0
            while pending_values:
                item, depth = pending_values.pop()
                value_nodes += 1
                if value_nodes > MAX_AST_NODES or depth > MAX_AST_DEPTH:
                    reject("limits")
                if isinstance(item, dict):
                    pending_values.extend((k, depth + 1) for k in item)
                    pending_values.extend((v, depth + 1) for v in item.values())
                elif isinstance(item, list):
                    pending_values.extend((v, depth + 1) for v in item)
                elif isinstance(item, str):
                    item.encode("utf-8")
            if len(json.dumps(args, ensure_ascii=False, allow_nan=False).encode("utf-8")) > MAX_ARGUMENT_BYTES:
                reject("limits")
        except (UnicodeEncodeError, ValueError, TypeError, RecursionError, OverflowError):
            reject("value")
        calls.append((name, args))
    return calls
