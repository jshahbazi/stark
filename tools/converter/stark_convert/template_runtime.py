"""Disposable, resource-bounded evaluation of model-provided chat programs.

Only JSON data crosses the private pipes. Child diagnostics and tracebacks are
discarded. The macOS RSS watchdog is sampled, not a kernel memory ceiling;
allocation guards and a restricted sandbox also reject expansion primitives.
"""

import json
import math
import os
import select
import signal
import subprocess
import sys
import time
from subprocess import Popen

from jinja2.exceptions import SecurityError, TemplateError

MAX_SOURCE_BYTES = 256 * 1024
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_OUTPUT_BYTES = 1024 * 1024
MAX_SEQUENCE_ITEMS = 16384
MAX_VALUE_NODES = 65536
MAX_VALUE_DEPTH = 64
MAX_INTEGER_BITS = 8192
MAX_CHILD_RSS = 256 * 1024 * 1024
MAX_EXPANSION_BYTES = 16 * 1024 * 1024
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


class _ExpansionBudget:
    """Conservative cumulative allocation/work units for one child evaluation."""

    def __init__(self, limit=MAX_EXPANSION_BYTES):
        self.limit = limit
        self.used = 0

    def charge(self, amount):
        if amount < 0 or amount > self.limit - self.used:
            raise TemplateRuntimeError(ERROR)
        self.used += amount


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


class IsolatedTemplate:
    """A source-only handle; compilation and every render run in a fresh child."""

    def __init__(self, source):
        if not isinstance(source, str) or not source or len(source) > MAX_SOURCE_BYTES:
            raise TemplateRuntimeError(ERROR)
        try:
            if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
                raise TemplateRuntimeError(ERROR)
        except UnicodeError:
            raise TemplateRuntimeError(ERROR) from None
        self._source = source
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
                [sys.executable, "-I", "-B", "-m", "stark_convert.template_runtime"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                close_fds=True,
                start_new_session=True,
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


def _child_environment(*, expansion_budget=None):
    from collections.abc import Iterator, MappingView
    from functools import wraps

    from jinja2 import Undefined, nodes, pass_context
    from jinja2.compiler import CodeGenerator
    from jinja2.runtime import Context, EvalContext, LoopContext, Macro
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from jinja2.utils import Namespace

    budget = _ExpansionBudget() if expansion_budget is None else expansion_budget

    def checked_argument(value):
        if isinstance(value, (Iterator, MappingView)):
            return bounded_list(value)
        if isinstance(value, Undefined):
            return value
        if isinstance(value, Namespace):
            _check_value(object.__getattribute__(value, "_Namespace__attrs"))
        elif isinstance(value, (LoopContext, Macro, range)):
            pass
        else:
            _check_value(value)
        return value

    def charged_result(value):
        value = checked_argument(value)
        if isinstance(value, (Undefined, LoopContext, Macro, range)):
            budget.charge(64)
        elif isinstance(value, Namespace):
            budget.charge(64 + 4 * _check_value(object.__getattribute__(value, "_Namespace__attrs")))
        else:
            budget.charge(64 + 4 * _check_value(value))
        return value

    def text(value):
        if isinstance(value, Undefined):
            return str(value)
        if isinstance(value, Namespace):
            budget.charge(64 + 6 * _check_value(object.__getattribute__(value, "_Namespace__attrs")))
        elif isinstance(value, (LoopContext, Macro, range)):
            budget.charge(64)
        else:
            size = _check_value(value)
            if type(value) is not str:
                budget.charge(64 + 6 * size)
        result = str(value)
        if len(result.encode("utf-8")) > MAX_OUTPUT_BYTES:
            raise TemplateSizeLimit(ERROR)
        return result

    class OutputBuffer(list):
        def __init__(self):
            budget.charge(64)
            self.size = 0

        def append(self, value):
            value = text(value)
            self.size += len(value.encode("utf-8"))
            if self.size > MAX_OUTPUT_BYTES or len(self) >= MAX_VALUE_NODES:
                raise TemplateSizeLimit(ERROR)
            budget.charge(8 + len(value.encode("utf-8")))
            super().append(value)

        def extend(self, values):
            for value in values:
                self.append(value)

    def concatenate(values):
        buffer = OutputBuffer()
        buffer.extend(values)
        budget.charge(64 + 4 * buffer.size)
        return "".join(buffer)

    class BoundedCodeGenerator(CodeGenerator):
        def buffer(self, frame):
            frame.buffer = self.temporary_identifier()
            self.writeline(f"{frame.buffer} = environment.output_buffer()")

        def visit_Output(self, node, frame):
            # Jinja normally evaluates an entire tuple before buffer.extend.
            # Emit one append/yield at a time so the cap runs between nodes.
            for child in node.nodes:
                super().visit_Output(nodes.Output([child]).set_lineno(node.lineno), frame)

        def visit_Concat(self, node, frame):
            self.write("environment.lazy_concat((")
            for child in node.nodes:
                self.write("lambda: (")
                self.visit(child, frame)
                self.write("), ")
            self.write("))")

        def visit_Getitem(self, node, frame):
            if not isinstance(node.arg, nodes.Slice):
                return super().visit_Getitem(node, frame)
            # Jinja emits Python slicing directly, bypassing getitem and the
            # callable guards. Charge the copy before evaluating that slice.
            self.write("environment.slice_value(")
            self.visit(node.node, frame)
            for part in (node.arg.start, node.arg.stop, node.arg.step):
                self.write(", ")
                if part is None:
                    self.write("None")
                else:
                    self.visit(part, frame)
            self.write(")")

        def visit_List(self, node, frame):
            self.write(f"environment.literal({len(node.items)}, False, lambda: (")
            super().visit_List(node, frame)
            self.write("))")

        def visit_Tuple(self, node, frame):
            if node.ctx != "load":
                return super().visit_Tuple(node, frame)
            self.write(f"environment.literal({len(node.items)}, False, lambda: (")
            super().visit_Tuple(node, frame)
            self.write("))")

        def visit_Dict(self, node, frame):
            self.write(f"environment.literal({len(node.items)}, True, lambda: (")
            super().visit_Dict(node, frame)
            self.write("))")

    class ResourceSandbox(ImmutableSandboxedEnvironment):
        intercepted_binops = frozenset({"+", "-", "*", "/", "//", "**", "%"})
        intercepted_unops = frozenset({"+", "-"})
        code_generator_class = BoundedCodeGenerator
        concat = staticmethod(concatenate)
        output_buffer = OutputBuffer

        @staticmethod
        def lazy_concat(thunks):
            return concatenate(thunk() for thunk in thunks)

        @staticmethod
        def slice_value(value, start, stop, step):
            if not isinstance(value, str) and type(value) not in (list, tuple, range):
                raise TemplateRuntimeError(ERROR)
            index = slice(start, stop, step)
            length = len(range(*index.indices(len(value))))
            width = 4 if isinstance(value, str) else 8
            budget.charge(64 + (0 if type(value) is range else width * length))
            return checked_argument(value[index])

        @staticmethod
        def literal(count, mapping, factory):
            if count > MAX_SEQUENCE_ITEMS:
                raise TemplateRuntimeError(ERROR)
            budget.charge(64 + (64 if mapping else 16) * count)
            result = factory()
            _check_value(result)
            return result

        def call_binop(self, context, operator, left, right):
            allocation = 64
            if operator == "+" and isinstance(left, (str, list, tuple)):
                if type(left) is not type(right) or len(left) + len(right) > MAX_OUTPUT_BYTES:
                    raise TemplateRuntimeError(ERROR)
                allocation += (4 if isinstance(left, str) else 8) * (len(left) + len(right))
            elif operator == "*":
                sequence, count = (left, right) if isinstance(left, (str, list, tuple)) else (right, left)
                if isinstance(sequence, (str, list, tuple)):
                    limit = MAX_OUTPUT_BYTES if isinstance(sequence, str) else MAX_SEQUENCE_ITEMS
                    if type(count) is not int or len(sequence) * max(0, count) > limit:
                        raise TemplateRuntimeError(ERROR)
                    allocation += (4 if isinstance(sequence, str) else 8) * len(sequence) * max(0, count)
            elif operator == "**":
                if type(left) is not int or type(right) is not int or right < 0:
                    raise TemplateRuntimeError(ERROR)
                if right > MAX_INTEGER_BITS or max(1, left.bit_length()) * right > MAX_INTEGER_BITS:
                    raise TemplateRuntimeError(ERROR)
                allocation += (max(1, left.bit_length()) * right + 7) // 8
            elif operator == "%" and isinstance(left, str):
                raise TemplateRuntimeError(ERROR)
            if type(left) is int and type(right) is int and operator != "**":
                allocation += (left.bit_length() + right.bit_length() + 7) // 8
            budget.charge(allocation)
            result = super().call_binop(context, operator, left, right)
            _check_value(result)
            return result

        def call_unop(self, context, operator, value):
            budget.charge(64 + (_check_value(value) if type(value) is int else 32))
            return super().call_unop(context, operator, value)

        def call(self, context, obj, *args, **kwargs):
            args = tuple(checked_argument(value) for value in args)
            kwargs = {key: checked_argument(value) for key, value in kwargs.items()}
            if any(obj is function for function in self.globals.values()) or isinstance(obj, (Macro, LoopContext)):
                return charged_result(super().call(context, obj, *args, **kwargs))
            owner, name = getattr(obj, "__self__", None), getattr(obj, "__name__", None)
            methods = {
                str: {
                    "startswith",
                    "endswith",
                    "strip",
                    "lstrip",
                    "rstrip",
                    "split",
                    "rsplit",
                    "lower",
                    "upper",
                    "capitalize",
                    "casefold",
                    "title",
                    "swapcase",
                    "count",
                    "find",
                    "rfind",
                    "index",
                    "rindex",
                    "isdigit",
                    "isalpha",
                    "isalnum",
                    "isspace",
                    "islower",
                    "isupper",
                    "isdecimal",
                    "isnumeric",
                },
                dict: {"get", "items", "keys", "values"},
                list: {"count", "index"},
                tuple: {"count", "index"},
            }
            if name not in methods.get(type(owner), set()):
                # Preserve Jinja's SecurityError for unsafe attributes/calls.
                if not self.is_safe_callable(obj):
                    return super().call(context, obj, *args, **kwargs)
                raise TemplateRuntimeError(ERROR)
            return charged_result(super().call(context, obj, *args, **kwargs))

    @pass_context
    def finalize(_context, value):
        # Context-dependent finalization prevents Jinja's Output constant-fold
        # path from evaluating model expressions while generating Python code.
        return text(value)

    environment = ResourceSandbox(
        extensions=["jinja2.ext.loopcontrols"],
        trim_blocks=True,
        lstrip_blocks=True,
        optimized=False,
        finalize=finalize,
    )

    def bounded_range(*args):
        result = range(*args)
        if len(result) > MAX_SEQUENCE_ITEMS:
            raise TemplateRuntimeError(ERROR)
        return result

    def bounded_dict(*args, **kwargs):
        result = dict(*args, **kwargs)
        _check_value(result)
        return result

    def reject(_message):
        raise TemplateRuntimeError(ERROR)

    environment.globals = {
        "range": bounded_range,
        "dict": bounded_dict,
        "namespace": Namespace,
        "raise_exception": reject,
    }
    safe_filters = {
        "abs",
        "capitalize",
        "default",
        "d",
        "escape",
        "e",
        "first",
        "float",
        "forceescape",
        "int",
        "last",
        "length",
        "count",
        "lower",
        "map",
        "max",
        "min",
        "reject",
        "rejectattr",
        "round",
        "safe",
        "select",
        "selectattr",
        "sort",
        "dictsort",
        "items",
        "reverse",
        "striptags",
        "title",
        "trim",
        "unique",
        "upper",
    }

    def checked_filter(function):
        @wraps(function)
        def invoke(*args, **kwargs):
            args = tuple(
                value if isinstance(value, (ResourceSandbox, Context, EvalContext)) else checked_argument(value)
                for value in args
            )
            kwargs = {key: checked_argument(value) for key, value in kwargs.items()}
            return charged_result(function(*args, **kwargs))

        return invoke

    environment.filters = {
        key: checked_filter(value) for key, value in environment.filters.items() if key in safe_filters
    }

    def bounded_list(values):
        result = []
        for value in values:
            if len(result) >= MAX_SEQUENCE_ITEMS:
                raise TemplateRuntimeError(ERROR)
            budget.charge(16)
            result.append(value)
        _check_value(result)
        return result

    def bounded_join(values, delimiter="", attribute=None):
        from jinja2.filters import make_attrgetter

        if attribute is not None:
            values = map(make_attrgetter(environment, attribute), values)
        delimiter = text(delimiter)
        return concatenate(
            part
            for index, item in enumerate(bounded_list(values))
            for part in ((delimiter, text(item)) if index else (text(item),))
        )

    def bounded_replace(value, old, new, count=None):
        value, old, new = text(value), text(old), text(new)
        count = value.count(old) if count is None or count < 0 else min(count, value.count(old))
        if len(value) + max(0, len(new) - len(old)) * count > MAX_OUTPUT_BYTES:
            raise TemplateRuntimeError(ERROR)
        budget.charge(64 + 4 * (len(value) + max(0, len(new) - len(old)) * count))
        return value.replace(old, new, count)

    def bounded_tojson(value, indent=None):
        _check_value(value)
        if indent is not None and (type(indent) is not int or not 0 <= indent <= 16):
            raise TemplateRuntimeError(ERROR)
        encoder = json.JSONEncoder(ensure_ascii=True, allow_nan=False, sort_keys=True, indent=indent)
        result = concatenate(encoder.iterencode(value))
        for old, new in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"), ("'", "\\u0027")):
            result = bounded_replace(result, old, new)
        return text(result)

    environment.filters.update(
        {
            "join": bounded_join,
            "list": bounded_list,
            "replace": bounded_replace,
            "string": text,
            "tojson": bounded_tojson,
        }
    )

    def compile_source(source):
        root = environment.parse(source)
        pending = [(root, 0)]
        count = 0
        while pending:
            node, depth = pending.pop()
            count += 1
            if count > 16384 or depth > 128:
                raise TemplateRuntimeError(ERROR)
            if isinstance(node, (nodes.EvalContextModifier, nodes.ScopedEvalContextModifier)):
                raise TemplateRuntimeError(ERROR)
            pending.extend((child, depth + 1) for child in node.iter_child_nodes())
        return environment.from_string(root)

    return compile_source


def _child_main():
    import resource

    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        if sys.platform != "darwin":
            resource.setrlimit(resource.RLIMIT_AS, (MAX_CHILD_RSS, MAX_CHILD_RSS))
        data = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(data) > MAX_REQUEST_BYTES:
            raise TemplateRuntimeError(ERROR)
        request = json.loads(data)
        source, arguments = request["source"], request["arguments"]
        if not isinstance(source, str) or not source or len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
            raise TemplateRuntimeError(ERROR)
        _check_value(arguments)
        template = _child_environment()(source)
        output = bytearray(b"O")
        if arguments is not None:
            if type(arguments) is not dict:
                raise TemplateRuntimeError(ERROR)
            for chunk in template.generate(**arguments):
                chunk = chunk.encode("utf-8")
                if len(output) + len(chunk) > MAX_OUTPUT_BYTES + 1:
                    raise TemplateSizeLimit(ERROR)
                output.extend(chunk)
        sys.stdout.buffer.write(output)
        return 0
    except SecurityError:
        sys.stdout.buffer.write(b"S")
    except TemplateSizeLimit:
        sys.stdout.buffer.write(b"L")
    except BaseException:
        sys.stdout.buffer.write(b"E")
    return 1


if __name__ == "__main__":
    raise SystemExit(_child_main())
