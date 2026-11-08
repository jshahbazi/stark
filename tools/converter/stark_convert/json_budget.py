"""Bound JSON structure before allocating its parsed containers.

This scanner is an admission check, not a JSON validator. The caller must still
decode UTF-8 and validate JSON syntax. Long strings and scalar spans are skipped
without creating a second copy of their contents.
"""

import re

MAX_JSON_NODES = 65536
MAX_JSON_DEPTH = 64
_ERROR = "JSON structure exceeds the resource budget."
_NEXT_NONSPACE = re.compile(rb"[^ \t\r\n]")
_SCALAR_END = re.compile(rb'[ \t\r\n,\[\]{}:"]')


def check_json_structure(data: bytes) -> None:
    """Reject excessive nodes or nesting without constructing a JSON tree.

    Containers, scalar values and object keys each count as a node. Syntax
    errors, including invalid escapes and Unicode, are left to the JSON parser.
    """
    position = nodes = depth = 0
    size = len(data)
    while position < size:
        match = _NEXT_NONSPACE.search(data, position)
        if match is None:
            return
        position = match.start()
        byte = data[position]
        position += 1
        if byte in (ord(","), ord(":")):
            continue
        if byte in (ord("]"), ord("}")):
            # Mismatched or excess closing delimiters remain syntax errors.
            depth = max(0, depth - 1)
            continue
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise ValueError(_ERROR)
        if byte in (ord("["), ord("{")):
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise ValueError(_ERROR)
        elif byte == ord('"'):
            while True:
                quote = data.find(b'"', position)
                if quote < 0:
                    return  # The parser will reject the unterminated string.
                slash = quote - 1
                while slash >= position and data[slash] == ord("\\"):
                    slash -= 1
                position = quote + 1
                if (quote - slash - 1) % 2 == 0:
                    break
        else:
            end = _SCALAR_END.search(data, position)
            position = end.start() if end is not None else size
