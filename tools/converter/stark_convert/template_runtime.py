"""Native, disposable evaluation of model-provided chat programs.

Only JSON data crosses the private pipes. Child diagnostics and tracebacks are
discarded. The macOS RSS watchdog is sampled, not a kernel memory ceiling;
allocation guards and a restricted sandbox also reject expansion primitives.
"""

import json
import math
import os
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from subprocess import Popen

from jinja2.exceptions import SecurityError, TemplateError

from .resource_budget import INPUT_BYTES, REQUEST_BUDGET_MIB, TEMPLATE_LIVE_BYTES, WIRE_BYTES

MAX_SOURCE_BYTES = 256 * 1024
MAX_REQUEST_BYTES = WIRE_BYTES
MAX_OUTPUT_BYTES = INPUT_BYTES
MAX_SEQUENCE_ITEMS = 16384
MAX_VALUE_NODES = 65536
MAX_VALUE_DEPTH = 64
MAX_INTEGER_BITS = 8192
MAX_CHILD_RSS = TEMPLATE_LIVE_BYTES
CPU_SECONDS = 2
WALL_SECONDS = 5.0
POLL_SECONDS = 0.01
ERROR = "The model's chat template is unsupported or exceeded its resource limits."


class TemplateRuntimeError(TemplateError):
    """A fixed, content-free evaluation failure."""


class TemplateCancelled(TemplateRuntimeError):
    """The child was reaped after its caller cancelled evaluation."""


class TemplateSizeLimit(TemplateRuntimeError):
    """Input or rendered bytes exceed the request size contract."""


class TemplateHelperMissing(TemplateRuntimeError):
    """The native evaluator must be built or installed before rendering."""


def _check_value(value):
    """Bound expanded containers, including repeated aliases, before conversion."""
    pending = [(value, 0)]
    nodes = size = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > MAX_VALUE_NODES or depth > MAX_VALUE_DEPTH:
            raise TemplateRuntimeError(ERROR)
        if isinstance(item, str):
            size += len(item.encode("utf-8"))
        elif item is None or type(item) is bool:
            size += 5
        elif type(item) is int:
            if item.bit_length() > MAX_INTEGER_BITS:
                raise TemplateRuntimeError(ERROR)
            size += max(1, item.bit_length() // 3 + 2)
        elif type(item) is float:
            if not math.isfinite(item):
                raise TemplateRuntimeError(ERROR)
            size += 32
        elif type(item) in (list, tuple, dict):
            if len(item) > MAX_SEQUENCE_ITEMS:
                raise TemplateRuntimeError(ERROR)
            children = len(item) * (2 if type(item) is dict else 1)
            if nodes + len(pending) + children > MAX_VALUE_NODES:
                raise TemplateRuntimeError(ERROR)
            if type(item) is dict:
                pending.extend((key, depth + 1) for key in item)
                pending.extend((child, depth + 1) for child in item.values())
            else:
                pending.extend((child, depth + 1) for child in item)
            size += len(item) * 4
        else:
            raise TemplateRuntimeError(ERROR)
        if size > MAX_OUTPUT_BYTES:
            raise TemplateSizeLimit(ERROR)
    return size


def _request_bytes(source, arguments):
    _check_value(arguments)
    data = bytearray()
    encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for part in encoder.iterencode({"source": source, "arguments": arguments}):
        encoded = part.encode("utf-8")
        if len(data) + len(encoded) > MAX_REQUEST_BYTES:
            raise TemplateSizeLimit(ERROR)
        data.extend(encoded)
    return data


def _template_executable():
    """Resolve the trusted installed helper or this checkout's release build."""
    candidates = [
        Path(sys.executable).parent / "stark-template",
        Path(sys.prefix).parent / "stark-template",
    ]
    repository = Path(__file__).resolve().parents[3]
    if (repository / "Cargo.toml").is_file() and (repository / "tools/converter").is_dir():
        candidates.append(repository / "target/release/stark-template")
    found = shutil.which("stark-template")
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise TemplateHelperMissing("Native chat-template helper is missing; rebuild or reinstall Stark.")


class IsolatedTemplate:
    """A source-only handle; compilation and every render use a fresh Rust helper."""

    def __init__(self, source):
        if not isinstance(source, str) or not source or len(source) > MAX_SOURCE_BYTES:
            raise TemplateRuntimeError(ERROR)
        try:
            if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
                raise TemplateRuntimeError(ERROR)
        except UnicodeError:
            raise TemplateRuntimeError(ERROR) from None
        self._source = source
        self._executable = _template_executable()
        self._evaluate(None, None)

    def render(self, **arguments):
        return self.render_with_cancellation(None, **arguments)

    def render_with_cancellation(self, cancelled, **arguments):
        return self._evaluate(arguments, cancelled)

    def _evaluate(self, arguments, cancelled):
        import psutil

        child = None
        deadline = time.monotonic() + WALL_SECONDS
        try:
            if cancelled is not None and cancelled.is_set():
                raise TemplateCancelled(ERROR)
            data = _request_bytes(self._source, arguments)
            child = Popen(
                [str(self._executable)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                close_fds=True,
                start_new_session=True,
                env={**os.environ, "STARK_REQUEST_BUDGET_MIB": str(REQUEST_BUDGET_MIB)},
            )
            monitored = psutil.Process(child.pid)
            os.set_blocking(child.stdin.fileno(), False)
            os.set_blocking(child.stdout.fileno(), False)
            written = 0
            output = bytearray()
            eof = False
            while not eof or child.poll() is None:
                if cancelled is not None and cancelled.is_set():
                    raise TemplateCancelled(ERROR)
                if time.monotonic() >= deadline:
                    raise TemplateRuntimeError(ERROR)
                try:
                    if monitored.memory_info().rss > MAX_CHILD_RSS:
                        raise TemplateRuntimeError(ERROR)
                except psutil.NoSuchProcess:
                    pass
                reads, writes, _ = select.select(
                    [] if eof else [child.stdout],
                    [child.stdin] if written < len(data) else [],
                    [],
                    POLL_SECONDS,
                )
                if writes:
                    try:
                        written += os.write(child.stdin.fileno(), memoryview(data)[written : written + 65536])
                    except BrokenPipeError:
                        written = len(data)
                    if written == len(data):
                        child.stdin.close()
                if reads:
                    chunk = os.read(child.stdout.fileno(), 65536)
                    if not chunk:
                        eof = True
                    elif len(output) + len(chunk) <= MAX_OUTPUT_BYTES + 1:
                        output.extend(chunk)
                    else:
                        raise TemplateRuntimeError(ERROR)
            if child.returncode == 0 and output[:1] == b"O":
                return output[1:].decode("utf-8")
            if output == b"S":
                raise SecurityError("The model's chat template violated the sandbox.")
            if output == b"L":
                raise TemplateSizeLimit(ERROR)
            raise TemplateRuntimeError(ERROR)
        except (OSError, ValueError, UnicodeError, psutil.Error):
            raise TemplateRuntimeError(ERROR) from None
        finally:
            if child is not None:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                child.wait()
                child.stdin.close()
                child.stdout.close()
