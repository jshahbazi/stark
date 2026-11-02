"""Bounded, source-only comparison of reviewed Jinja chat programs.

This is syntax equivalence, not a heuristic for recognizing a model family.
No template is compiled or executed here. Literal output and every executable
AST field remain part of the fingerprint; source locations are diagnostics.
"""

import hashlib
import json
from functools import lru_cache

from jinja2 import nodes
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

MAX_TEMPLATE_BYTES = 256 * 1024
MAX_TEMPLATE_NODES = 16384
MAX_TEMPLATE_DEPTH = 128


def chat_template_environment():
    """Keep admission and actual rendering on the same language settings."""
    return ImmutableSandboxedEnvironment(extensions=["jinja2.ext.loopcontrols"], trim_blocks=True, lstrip_blocks=True)


def template_program_sha256(source):
    """Return a versioned AST fingerprint, or fail closed on unsupported input."""
    if not isinstance(source, str) or not source or len(source) > MAX_TEMPLATE_BYTES:
        return None
    try:
        if len(source.encode("utf-8")) > MAX_TEMPLATE_BYTES:
            return None
    except UnicodeEncodeError:
        return None
    # Check the cap before caching. This cache contains at most eight public
    # template programs, never conversation content, and is not persisted.
    return _program_sha256(source)


@lru_cache(maxsize=8)
def _program_sha256(source):
    try:
        root = chat_template_environment().parse(source)
        digest = hashlib.sha256(b"stark-jinja-program-v1\0")
        pending = [(root, 0)]
        work = 0
        while pending:
            item, depth = pending.pop()
            work += 1
            if work > MAX_TEMPLATE_NODES or depth > MAX_TEMPLATE_DEPTH:
                return None
            if isinstance(item, nodes.Node):
                fields = list(item.iter_fields())
                if isinstance(item, nodes.Output):
                    # Comments can split literal data into several nodes. Only
                    # coalesce adjacent TemplateData, with byte-exact contents;
                    # never fold expressions, constants or output statements.
                    merged = []
                    parts = []
                    for child in item.nodes:
                        work += 1
                        if work > MAX_TEMPLATE_NODES:
                            return None
                        if type(child) is nodes.TemplateData:
                            parts.append(child.data)
                        else:
                            if parts:
                                merged.append(nodes.TemplateData("".join(parts)))
                                parts = []
                            merged.append(child)
                    if parts:
                        merged.append(nodes.TemplateData("".join(parts)))
                    fields = [(name, merged if name == "nodes" else value) for name, value in fields]
                record = ["node", type(item).__name__, [name for name, _ in fields]]
                pending.extend((value, depth + 1) for _, value in reversed(fields))
            elif isinstance(item, list):
                record = ["list", len(item)]
                pending.extend((value, depth + 1) for value in reversed(item))
            elif item is None or type(item) in (str, int, float, bool):
                record = [type(item).__name__, item]
            else:
                return None
            digest.update(json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode())
            digest.update(b"\n")
        return digest.hexdigest()
    except (TemplateError, ValueError, TypeError, RecursionError):
        return None
