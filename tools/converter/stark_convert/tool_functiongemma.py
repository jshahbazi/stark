"""Strict decoder for FunctionGemma's textual function-call blocks.

This module only decodes the model's bounded wire format. It never resolves or
executes a function. The caller must validate names and schemas before use.
"""

from __future__ import annotations

import json
import math
import re
from array import array
from collections.abc import Callable
from typing import NoReturn

_OPEN = "<start_function_call>call:"
_CLOSE = "<end_function_call>"
_ESCAPE = "<escape>"
_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_KEY_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_RESERVED = (
    "<start_function_call>",
    "<end_function_call>",
    "<start_function_response>",
    "<end_function_response>",
    "<start_function_declaration>",
    "<end_function_declaration>",
    "<escape>",
)

# Match the converter's shared tool parser bounds without importing api_tools
# (which routes to this decoder and would create a cycle).
_MAX_OUTPUT_BYTES = 512 * 1024
_MAX_ARGUMENT_BYTES = 64 * 1024
_MAX_DEPTH = 32
_MAX_NODES = 16_384
_MAX_CALLS = 16
_MAX_NUMBER_CHARS = 256
_MAX_ROOT_KEY_CHARS = 64


class _DecodeError(Exception):
    def __init__(self, category: str):
        self.category = category


class _Parser:
    def __init__(self, text: str, invalid: Callable[[str], Exception]):
        self.text = text
        self.invalid = invalid
        self.cursor = 0
        self.nodes = 0
        self.argument_start = 0
        self.byte_offsets = array("I", [0])
        byte_offset = 0
        for char in text:
            byte_offset += len(char.encode("utf-8"))
            self.byte_offsets.append(byte_offset)

    def fail(self, category: str = "format") -> NoReturn:
        raise _DecodeError(category)

    def skip_space(self) -> None:
        size = len(self.text)
        while self.cursor < size and self.text[self.cursor] in " \t\r\n":
            self.cursor += 1

    def add_node(self, depth: int) -> None:
        self.nodes += 1
        if self.nodes > _MAX_NODES or depth > _MAX_DEPTH:
            self.fail("limits")

    def check_argument_size(self) -> None:
        size = self.byte_offsets[self.cursor] - self.byte_offsets[self.argument_start]
        if size > _MAX_ARGUMENT_BYTES:
            self.fail("limits")

    def parse_name(self) -> str:
        start = self.cursor
        while self.cursor < len(self.text) and self.text[self.cursor] not in "{ \t\r\n":
            self.cursor += 1
        name = self.text[start : self.cursor]
        if not _NAME.fullmatch(name):
            self.fail("format")
        if self.cursor >= len(self.text) or self.text[self.cursor] != "{":
            self.fail("format")
        return name

    def parse_key(self) -> str:
        if self.text.startswith(_ESCAPE, self.cursor):
            self.cursor += len(_ESCAPE)
            end = self.text.find(_ESCAPE, self.cursor)
            if end < 0:
                self.fail("format")
            key = self.text[self.cursor : end]
            if any(marker in key for marker in _RESERVED):
                self.fail("argument")
            self.cursor = end + len(_ESCAPE)
            self.add_node(0)
            return key
        start = self.cursor
        while self.cursor < len(self.text) and self.text[self.cursor] in _KEY_CHARS:
            self.cursor += 1
        key = self.text[start : self.cursor]
        if not _KEY.fullmatch(key):
            self.fail("argument")
        self.add_node(0)
        return key

    def parse_object(self, depth: int, *, counted: bool = False) -> dict:
        if not counted:
            self.add_node(depth)
        if self.cursor >= len(self.text) or self.text[self.cursor] != "{":
            self.fail("format")
        self.cursor += 1
        result = {}
        self.skip_space()
        if self.cursor < len(self.text) and self.text[self.cursor] == "}":
            self.cursor += 1
            return result
        while True:
            self.skip_space()
            key = self.parse_key()
            if depth == 0 and len(key) > _MAX_ROOT_KEY_CHARS:
                self.fail("limits")
            if key in result:
                self.fail("argument")
            self.skip_space()
            if self.cursor >= len(self.text) or self.text[self.cursor] != ":":
                self.fail("format")
            self.cursor += 1
            self.skip_space()
            result[key] = self.parse_value(depth + 1)
            self.check_argument_size()
            self.skip_space()
            if self.cursor >= len(self.text):
                self.fail("format")
            delimiter = self.text[self.cursor]
            self.cursor += 1
            if delimiter == "}":
                return result
            if delimiter != ",":
                self.fail("format")

    def parse_array(self, depth: int, *, counted: bool = False) -> list:
        if not counted:
            self.add_node(depth)
        self.cursor += 1
        result = []
        self.skip_space()
        if self.cursor < len(self.text) and self.text[self.cursor] == "]":
            self.cursor += 1
            return result
        while True:
            self.skip_space()
            result.append(self.parse_value(depth + 1))
            self.check_argument_size()
            self.skip_space()
            if self.cursor >= len(self.text):
                self.fail("format")
            delimiter = self.text[self.cursor]
            self.cursor += 1
            if delimiter == "]":
                return result
            if delimiter != ",":
                self.fail("format")

    def parse_string(self) -> str:
        self.cursor += len(_ESCAPE)
        end = self.text.find(_ESCAPE, self.cursor)
        if end < 0:
            self.fail("format")
        value = self.text[self.cursor : end]
        if any(marker in value for marker in _RESERVED):
            self.fail("format")
        try:
            if len(value.encode("utf-8")) > _MAX_ARGUMENT_BYTES:
                self.fail("limits")
        except UnicodeEncodeError:
            self.fail("value")
        self.cursor = end + len(_ESCAPE)
        return value

    def parse_scalar(self):
        if self.text.startswith("true", self.cursor):
            self.cursor += 4
            return True
        if self.text.startswith("false", self.cursor):
            self.cursor += 5
            return False
        if self.text.startswith("null", self.cursor):
            self.cursor += 4
            return None
        # The public FunctionGemma template's fallback branch stringifies
        # Python None, so its native rendering is this exact literal.
        if self.text.startswith("None", self.cursor):
            self.cursor += 4
            return None
        match = _NUMBER.match(self.text, self.cursor)
        if match is None:
            self.fail("value")
        token = match.group(0)
        if len(token) > _MAX_NUMBER_CHARS:
            self.fail("limits")
        self.cursor = match.end()
        try:
            value = json.loads(token)
        except (ValueError, RecursionError):
            self.fail("value")
        if isinstance(value, float) and not math.isfinite(value):
            self.fail("value")
        return value

    def parse_value(self, depth: int):
        self.add_node(depth)
        if self.cursor >= len(self.text):
            self.fail("format")
        if self.text.startswith(_ESCAPE, self.cursor):
            return self.parse_string()
        char = self.text[self.cursor]
        if char == "{":
            return self.parse_object(depth, counted=True)
        if char == "[":
            return self.parse_array(depth, counted=True)
        return self.parse_scalar()

    def parse_calls(self) -> list[tuple[str, dict]]:
        calls = []
        while True:
            self.skip_space()
            if self.cursor == len(self.text):
                break
            if len(calls) >= _MAX_CALLS:
                self.fail("limits")
            if not self.text.startswith(_OPEN, self.cursor):
                self.fail("trailing" if calls else "format")
            self.cursor += len(_OPEN)
            name = self.parse_name()
            self.argument_start = self.cursor
            arguments = self.parse_object(0)
            self.check_argument_size()
            self.skip_space()
            if not self.text.startswith(_CLOSE, self.cursor):
                self.fail("format")
            self.cursor += len(_CLOSE)
            calls.append((name, arguments))
        if not calls:
            self.fail("format")
        return calls


def parse_functiongemma_calls(text: str, *, invalid: Callable[[str], Exception]) -> list[tuple[str, dict]]:
    """Decode one or more adjacent FunctionGemma call blocks.

    Whitespace may separate complete blocks. No prefix, commentary, or suffix
    is accepted. Tool name and schema checks remain the caller's responsibility.
    """
    if not isinstance(text, str):
        raise invalid("format")
    try:
        if len(text.encode("utf-8")) > _MAX_OUTPUT_BYTES:
            raise invalid("limits")
    except UnicodeEncodeError:
        raise invalid("value") from None
    parser = _Parser(text, invalid)
    try:
        return parser.parse_calls()
    except _DecodeError as error:
        raise invalid(error.category) from None
