"""Temporary local XML diagnostics, including explicitly authorized arguments."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import threading
from datetime import UTC, datetime
from pathlib import Path

_CATEGORIES = frozenset({"xml_value_json", "xml_value_type", "xml_value_ambiguous"})
_MAX_BYTES = 64 * 1024
_MAX_CAPTURE_CHARACTERS = 1024 * 1024
_MAX_ARGUMENT_LOG_BYTES = 8 * 1024 * 1024
_LOCK = threading.Lock()


def _log_directory() -> Path:
    return Path.home() / "Library/Logs/Stark"


def record_xml_rejection(category: str) -> None:
    """Accept only fixed categories; never accept exception or request payloads.

    Reset the file at 64 KiB. Unavailable storage or lock contention must not
    change the API error, emit exception details or block inference.
    """
    if category not in _CATEGORIES:
        return
    _append_record("tool-diagnostics.jsonl", {"category": category})


def record_xml_argument(category: str, body: str, types: frozenset[str], parser: str) -> None:
    """Temporary capture authorized by the user; never receives a full response.

    Separate from the category-only log. Limit each value to 1,048,576 characters;
    the private file retains at most 8 MiB. Never include prompt/history data.
    """
    if category not in _CATEGORIES:
        return
    _append_record(
        "tool-arguments.jsonl",
        {
            "category": category,
            "argument": body[:_MAX_CAPTURE_CHARACTERS],
            "truncated": len(body) > _MAX_CAPTURE_CHARACTERS,
            "expected_types": sorted(types),
            "parser": parser,
        },
        maximum=_MAX_ARGUMENT_LOG_BYTES,
    )


def _append_record(filename: str, fields: dict, *, maximum: int | None = None) -> None:
    record = (json.dumps({"timestamp": datetime.now(UTC).isoformat(timespec="seconds"), **fields}) + "\n").encode(
        "ascii"
    )
    maximum = _MAX_BYTES if maximum is None else maximum
    with _LOCK:
        directory_fd = descriptor = None
        try:
            directory = _log_directory()
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            if os.fstat(directory_fd).st_uid != os.getuid():
                return
            os.fchmod(directory_fd, 0o700)
            descriptor = os.open(
                filename,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600,
                dir_fd=directory_fd,
            )
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_nlink != 1:
                return
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.fchmod(descriptor, 0o600)
            if os.fstat(descriptor).st_size + len(record) > maximum:
                os.ftruncate(descriptor, 0)
            os.write(descriptor, record)
        except OSError:
            pass
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if directory_fd is not None:
                os.close(directory_fd)
