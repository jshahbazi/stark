"""Private syntax-only checker for generated Python programs.

No supplied source is executed, persisted, or printed. Runtime validation is
reported as unavailable until a restrictive sandbox profile is qualified.
"""

from __future__ import annotations

import ast
import re


_GUI_MODULES = {"tkinter", "pygame", "turtle", "curses", "arcade", "pyglet"}


def _program_source(text: str) -> tuple[str | None, bool]:
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.DOTALL)
    for candidate in sorted(blocks, key=len, reverse=True):
        try:
            ast.parse(candidate)
        except (SyntaxError, ValueError, TypeError):
            continue
        return candidate, True
    try:
        ast.parse(text)
    except (SyntaxError, ValueError, TypeError):
        return None, False
    return text, False


def _has_gui_import(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", 1)[0] in _GUI_MODULES for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".", 1)[0] in _GUI_MODULES:
                return True
    return False


def check_response(text: str) -> dict:
    """Return syntax and GUI-import metadata only; never execute response code."""
    if not isinstance(text, str):
        source, fenced = None, False
    else:
        source, fenced = _program_source(text)
    syntax_valid = source is not None
    return {
        "syntax_valid": syntax_valid,
        "gui_import_detected": _has_gui_import(source) if source is not None else False,
        "runtime_available": False,
        "normal_exit": None,
        "game_end_signal": None,
        "input_calls": 0,
        "replays": 0,
        "sandbox_or_runtime_failure": True,
        "runtime_validation": "not_run_unqualified_sandbox",
        "source_was_fenced": fenced,
    }
