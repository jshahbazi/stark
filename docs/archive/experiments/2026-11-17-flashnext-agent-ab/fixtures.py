"""Synthetic coding-agent worktrees and detached, bounded behavior grading.

Only task specifications and starter code enter the agent worktree. Grading
copies two expected regular source files into a fresh directory. The macOS
sandbox denies file writes, network, and child processes. The separate grader
is intended for ordinary generated Python, not deliberate result spoofing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap


TASKS = {
    "ledger": (
        "Wallet reconciliation",
        """Compute account balances from ledger entries. Amounts are integer cents.
        Each account starts at zero. A credit adds its amount, a debit subtracts
        it, and a transfer subtracts from its account and adds to its `to`
        account. Return integer balances for all touched accounts, including zero
        balances. Keep the existing API and CLI. Ordinary entries have positive
        amounts and valid account names.""",
        """Make transaction IDs idempotent across credits, debits and transfers.
        Identical repeated transaction dictionaries count only once. Raise
        ValueError if the same ID occurs with any different field or value.
        Entries without IDs remain independent. Preserve previous behavior.""",
        """Add reversals. A row with kind 'reverse', its own optional 'id', and
        'of' naming a prior non-reversal transaction undoes the original postings.
        The target may be reversed once. Unknown, self, reversal-of-reversal and
        second reversals raise ValueError. An identical retry of a reversal with
        the same ID is idempotent. A reversal preserves touched zero balances.""",
    ),
    "schedule": (
        "Room availability",
        """Return maximal free half-open integer intervals [start, end] in the
        supplied window. Input bookings can overlap, be adjacent, duplicate,
        arrive unsorted, or extend outside the window. Ignore empty bookings.
        Clip all busy time to the window. An empty window has no free intervals.""",
        """Bookings may include a nonnegative integer 'setup' and 'cleanup'.
        They reserve that much time immediately before and after a booking.
        Empty bookings stay empty. Omitted padding is zero. Reject negative
        padding with ValueError. Preserve the API.""",
        """Support optional 'capacity' on the query (default one) and 'units'
        on bookings (default one). Positive integer units consume room capacity
        during the whole padded half-open booking. A minute is free if the total
        units consumed is strictly less than capacity. Reject nonpositive
        capacity or units. At simultaneous endpoints, use the net change.
        The window can span a billion minutes. Preserve the output shape.""",
    ),
    "catalog": (
        "Product selection",
        """Return product IDs in search order. With 'query', all whitespace-
        separated terms must appear as case-insensitive substrings of the product
        name or tags. Missing query is empty. With 'max_price', filter prices
        above the inclusive limit. Sort by ascending integer price then ID.""",
        """Products can declare 'aliases' and queries can use an 'exclude'
        list of IDs. Search names, tags and aliases. Normalize Unicode using
        NFKC then casefold, for both query and search fields. Missing aliases
        and exclude mean empty lists. Preserve earlier ranking and filters.""",
        """Add optional positive integer 'limit' and optional 'after' cursor
        as [price, id]. Filter by the strict ascending sort-key comparison
        with the cursor, even if it refers to a row the filters excluded or a
        row absent from inventory. Apply limit last. Reject nonpositive limit.
        Preserve all earlier filtering and normalization rules.""",
    ),
    "inventory": (
        "Warehouse availability",
        """Compute available inventory by SKU. Each receipt adds quantity and
        each shipment subtracts quantity, independently by SKU. An unknown SKU
        starts at zero. Return integer quantities for touched SKUs, including
        zero. Reject a shipment if it would make its SKU negative. Rows are
        chronological. Quantities are positive integers.""",
        """Add 'reserve' events with a unique 'ref', 'sku' and 'qty'. A reservation
        reduces available inventory. 'release' takes a 'ref' and restores all
        remaining quantity on that reservation, then closes it. Unknown,
        closed, duplicate refs, or a reserve larger than availability raise
        ValueError. Keep refs globally unique after closure.""",
        """Add partial fulfillment and partial release. A 'fulfill' specifies an
        active 'ref' and a positive 'qty'; it ships reserved stock and reduces
        the reservation without changing available units. A 'release' with a
        positive 'qty' releases only that much, and without qty still releases
        all. Reject operations larger than remaining and close reservations at
        zero. Preserve previous behavior and the public API.""",
    ),
}

STARTER = {
    "ledger": (
        "reconcile", "entries",
        """
        from collections import defaultdict

        def reconcile(entries):
            """ + '"""Return cents by account for a chronological entry list."""' + """
            balances = defaultdict(int)
            for row in entries:
                account = row["account"]
                if row["kind"] == "credit":
                    balances[account] += row["amount"]
                elif row["kind"] == "debit":
                    balances[account] -= row["amount"]
                else:
                    raise ValueError("unknown ledger kind")
            return dict(balances)
        """,
        '{"entries":[{"kind":"credit","account":"cash","amount":90},{"kind":"debit","account":"cash","amount":20}]}',
        {"cash": 70},
    ),
    "schedule": (
        "availability", "request",
        """
        def availability(request):
            """ + '"""Return the free intervals for one room and query window."""' + """
            start, end = request["window"]
            bookings = request.get("bookings", [])
            if not bookings:
                return [[start, end]] if start < end else []
            first = bookings[0]
            return [[start, first["start"]]] if start < first["start"] else []
        """,
        '{"window":[0,20],"bookings":[]}', [[0, 20]],
    ),
    "catalog": (
        "search", "request",
        """
        def search(request):
            """ + '"""Return SKU identifiers for the requested search."""' + """
            rows = request["products"]
            query = request.get("query", "")
            return [row["id"] for row in rows if query in row["name"]]
        """,
        '{"products":[{"id":"r2","name":"Rope","price":120,"tags":["climbing"]}],"query":"Rope"}',
        ["r2"],
    ),
    "inventory": (
        "available", "entries",
        """
        from collections import defaultdict

        def available(entries):
            """ + '"""Return available counts by SKU after applying warehouse events."""' + """
            counts = defaultdict(int)
            for event in entries:
                sku = event["sku"]
                if event["kind"] == "receive":
                    counts[sku] += event["qty"]
                elif event["kind"] == "ship":
                    counts[sku] -= event["qty"]
                else:
                    raise ValueError("unknown warehouse event")
            return dict(counts)
        """,
        '{"entries":[{"kind":"receive","sku":"R","qty":8},{"kind":"ship","sku":"R","qty":2}]}',
        {"R": 6},
    ),
}

OPERATING_NOTES = {
    "ledger": """
    ## Input and output

    The accounting export sends a chronological `entries` array. A basic row
    has `kind`, `account`, and `amount`. Account identifiers are opaque,
    case-sensitive strings. Do arithmetic in integer cents. The export covers
    only the window being reconciled, so starting at zero and returning a
    negative balance are both valid. The caller will merge these balances into
    its own reports.

    The command receives its request through stdin, with no filename argument.
    The library can also be imported by a larger batch command. Keep output
    deterministic for identical inputs. This module has no durable state.
    """,
    "schedule": """
    ## Input and output

    Both the query's `window` pair and each booking's `start` and `end`
    use integer offsets on the same time axis. The schedule exporter handles
    time zones before calling this utility. A reservation ending where
    another begins does not leave a free interval of length zero.

    The library returns a list of two-element JSON arrays in time order.
    Clients display each array as a continuous slot and expect adjacent free
    intervals to be merged. A request may include previously loaded bookings
    before and after the selected window.
    """,
    "catalog": """
    ## Input and output

    A request has a `products` list and optional filters. Each product
    has an opaque, case-sensitive string `id`, display `name`, integer
    `price`, and a list of `tags`. The ID is the stable value used by
    downstream inventory operations. Search never rewrites the ID or
    the input documents.

    The client consumes an ordered array of IDs and looks up the display
    records itself. Returning an empty array is normal. Prices use one
    currency in integer minor units. Sorting follows the ordinary comparison
    of the original IDs among products with equal numeric price.
    """,
    "inventory": """
    ## Input and output

    The nightly warehouse export sends an `entries` list in event order.
    Each SKU is an opaque, case-sensitive string. The result is the available
    quantity keyed by SKU. Include a tracked SKU after it reaches zero so the
    consumer can distinguish it from a SKU with no events.

    The library computes one batch using local state. It does not retain
    data between calls. Events earlier in the sequence can affect the
    validity of later events. On validation failure the per-batch CLI
    reports an error and writes no partial result to stdout.
    """,
}


def prepare(task_id: str, repo: Path) -> tuple[str, ...]:
    """Create one clean participant workspace and return three ordered turns."""
    title, *phases = TASKS[task_id]
    name, argument, implementation, sample, expected = STARTER[task_id]
    repo = Path(repo)
    repo.mkdir(parents=True, exist_ok=True)
    if any(repo.iterdir()):
        raise ValueError("fixture directory must be empty")
    arg = 'payload["entries"]' if argument == "entries" else "payload"
    cli = f"""\
import json
import sys
from service import {name}

def main():
    payload = json.load(sys.stdin)
    try:
        result = {name}({arg})
    except ValueError:
        print("invalid request", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, sort_keys=True)
    print()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""
    starter = textwrap.dedent(implementation).lstrip()
    repo.joinpath("service.py").write_text(starter, encoding="utf-8")
    repo.joinpath("cli.py").write_text(cli, encoding="utf-8")
    repo.joinpath("example.json").write_text(sample + "\n", encoding="utf-8")
    repo.joinpath("README.md").write_text(
        f"# {title}\n\nThis utility reads one JSON request from stdin and writes one JSON result. "
        f"Run `python3 cli.py < example.json` for the existing happy path. "
        f"The library entry point is `service.{name}({argument})`. "
        "Validation errors use ValueError in the library and exit status 2 in the CLI. "
        "Use the Python standard library and preserve both entry points.\n\n"
        + textwrap.dedent(OPERATING_NOTES[task_id]).strip()
        + "\n",
        encoding="utf-8",
    )
    repo.joinpath("test_smoke.py").write_text(
        f"import json\nimport unittest\nfrom pathlib import Path\nfrom service import {name}\n\n"
        f"class Smoke(unittest.TestCase):\n"
        f"    def test_existing_example(self):\n"
        f"        payload = json.loads(Path(__file__).with_name('example.json').read_text())\n"
        f"        self.assertEqual({name}({arg}), {expected!r})\n",
        encoding="utf-8",
    )
    suffix = "\n\nImplement the change in this workspace. Run relevant tests and summarize the result."
    return tuple(" ".join(textwrap.dedent(spec).split()) + suffix for spec in phases)


_ERROR = {"__expected_value_error__": True}


def _entry(kind, account, amount, **fields):
    return dict(kind=kind, account=account, amount=amount, **fields)


def _event(kind, sku, qty, **fields):
    return dict(kind=kind, sku=sku, qty=qty, **fields)


def _cases(task_id):
    if task_id == "ledger":
        credit = _entry("credit", "A", 9, id="c")
        transfer = _entry("transfer", "A", 7, to="b", id="t")
        debit = _entry("debit", "b", 4, id="d")
        reverse = {"kind": "reverse", "id": "r", "of": "t"}
        return (
            [([], {}), ([credit, transfer, debit], {"A": 2, "b": 3}),
             ([transfer], {"A": -7, "b": 7}),
             ([_entry("transfer", "A", 8, to="A")], {"A": 0})],
            [([credit, credit, transfer, transfer], {"A": 2, "b": 7}),
             ([credit, dict(credit, amount=10)], _ERROR),
             ([credit, dict(transfer, id="c")], _ERROR),
             ([_entry("credit", "a", 6)] * 3, {"a": 18})],
            [([credit, transfer, reverse, reverse], {"A": 9, "b": 0}),
             ([credit, transfer, reverse, dict(reverse, id="new")], _ERROR),
             ([reverse, transfer], _ERROR), ([dict(reverse, of="r")], _ERROR),
             ([transfer, reverse, dict(reverse, id="other", of="r")], _ERROR),
             ([credit, {"kind": "reverse", "of": "c"}], {"A": 0})],
        )
    if task_id == "schedule":
        def case(bookings, expected, window=(0, 20), **fields):
            return (dict(window=list(window), bookings=bookings, **fields), expected)
        def b(start, end, **fields):
            return dict(start=start, end=end, **fields)
        return (
            [case([b(12, 15), b(2, 8), b(6, 12), b(2, 8)], [[0, 2], [15, 20]]),
             case([b(-5, 3), b(15, 21), b(10, 10)], [[3, 15]]),
             case([b(-7, 28)], []), case([], [], window=(7, 7)),
             case([b(22, 33), b(-6, -1)], [[0, 20]])],
            [case([b(4, 6, setup=2, cleanup=3), b(11, 17, setup=2, cleanup=8)], [[0, 2]]),
             case([b(4, 4, setup=4, cleanup=4)], [[0, 20]]),
             case([b(4, 9, cleanup=-1)], _ERROR),
             case([b(-4, -1, cleanup=2), b(21, 24, setup=2)], [[1, 19]])],
            [case([b(2, 9), b(4, 12), b(9, 14)], [[0, 4], [12, 20]], capacity=2),
             case([b(2, 9, units=3), b(9, 12, units=3)], [[0, 2], [12, 20]], capacity=3),
             case([b(8, 10, units=2, setup=5), b(6, 8)], [[0, 6], [8, 10**9]],
                  window=(0, 10**9), capacity=3),
             case([b(1, 2, units=0)], _ERROR, capacity=2),
             case([], _ERROR, capacity=0)],
        )
    if task_id == "catalog":
        products = [
            dict(id="b", name="Steel Mug", tags=["permanent"], price=8, aliases=["ＳＴＲＡẞＥ"]),
            dict(id="a", name="Travel Steel Mug", tags=["BLUE"], price=8, aliases=["Nomad"]),
            dict(id="x", name="Thermos", tags=["Élite", "blue", "steel"], price=11, aliases=["Café"]),
            dict(id="z", name="Tea Cup", tags=["Porcelain"], price=4),
        ]
        def case(expected, **fields):
            # Keep later-stage fields out of earlier cases.
            base = [dict(row) for row in products]
            return (dict(products=base, **fields), expected)
        return (
            [case(["a", "x"], query="blue  steel"),
             case(["z", "a", "b"], max_price=8),
             case(["a", "b"], query="MUG", max_price=8),
             case(["z", "a", "b", "x"], query=" \t "),
             case([], query="blue steel mug", max_price=7)],
            [case(["b"], query="strasse"), case(["x"], query="cafe\u0301"),
             case(["x"], query="STEEL blue", exclude=["a"]),
             case(["a"], query="nomad STEEL", exclude=["unknown"])],
            [case(["b", "x"], query="steel", after=[8, "a"]),
             case(["x"], query="blue", after=[8, "b"], limit=1),
             case(["a"], after=[6, "never-present"], limit=1),
             case([], limit=3, after=[15, "unknown"]),
             case(_ERROR, limit=0)],
        )
    if task_id == "inventory":
        receipt = _event("receive", "A", 12)
        reserve = _event("reserve", "A", 9, ref="r")
        def r(kind, **fields):
            return dict(kind=kind, ref="r", **fields)
        return (
            [([], {}), ([receipt, _event("ship", "A", 12)], {"A": 0}),
             ([receipt, _event("ship", "A", 3), _event("receive", "B", 8)], {"A": 9, "B": 8}),
             ([_event("ship", "missing", 1)], _ERROR),
             ([receipt, _event("ship", "A", 13), receipt], _ERROR)],
            [([receipt, reserve], {"A": 3}),
             ([receipt, reserve, r("release")], {"A": 12}),
             ([receipt, reserve, r("release"), reserve], _ERROR),
             ([receipt, r("release")], _ERROR),
             ([receipt, reserve, _event("ship", "A", 4)], _ERROR),
             ([receipt, _event("reserve", "A", 13, ref="other")], _ERROR)],
            [([receipt, reserve, r("fulfill", qty=4), r("release", qty=2), r("release")], {"A": 8}),
             ([receipt, reserve, r("fulfill", qty=9), r("release")], _ERROR),
             ([receipt, reserve, r("release", qty=3), r("fulfill", qty=7)], _ERROR),
             ([receipt, reserve, r("release", qty=0)], _ERROR),
             ([receipt, reserve, r("fulfill", qty=4), _event("ship", "A", 2)], {"A": 1})],
        )
    raise KeyError(task_id)


_RUNNER = """
import contextlib
import io
import json
import resource
import runpy
import sys

resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
sys.path.insert(0, sys.argv[1])
payload = json.loads(sys.stdin.read())
passed = 0

class Sink(io.TextIOBase):
    def write(self, value):
        return len(value)

class BoundedText(io.StringIO):
    def write(self, value):
        if self.tell() + len(value) > 16384:
            raise ValueError("CLI output limit exceeded")
        return super().write(value)

def observe(fn, arg):
    try:
        return fn(arg)
    except ValueError:
        return payload["error"]
    except Exception:
        return None

with contextlib.redirect_stdout(Sink()), contextlib.redirect_stderr(Sink()):
    try:
        from service import __dict__ as service
        fn = service[payload["name"]]
        for arg, expected in payload["cases"]:
            passed += observe(fn, arg) == expected
        previous_stdin = sys.stdin
        try:
            sys.stdin = io.StringIO(payload["example"])
            sink = BoundedText()
            code = 0
            with contextlib.redirect_stdout(sink):
                try:
                    runpy.run_module("cli", run_name="__main__")
                except SystemExit as error:
                    code = error.code
            passed += code in (None, 0) and json.loads(sink.getvalue()) == payload["example_expected"]
        finally:
            sys.stdin = previous_stdin
    except Exception:
        pass
print(json.dumps({"passed": passed}))
"""


def _stage_source(repo: Path, work: Path) -> None:
    for name in ("service.py", "cli.py"):
        fd = os.open(repo / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 131072 or info.st_nlink != 1:
                raise ValueError("fixture source must be a bounded ordinary file")
            contents = source.read(131073)
            if len(contents) > 131072:
                raise ValueError("fixture source exceeded its byte limit")
        (work / name).write_bytes(contents)


def grade(task_id: str, repo: Path, phase: int) -> dict:
    """Run hidden cumulative cases and return aggregate counts only."""
    if phase not in (1, 2, 3):
        raise ValueError("phase must be 1, 2 or 3")
    name, _, _, example, expected = STARTER[task_id]
    cases = [case for stage in _cases(task_id)[:phase] for case in stage]
    total = len(cases) + 1
    payload = json.dumps(
        dict(name=name, cases=cases, error=_ERROR, example=example, example_expected=expected)
    ).encode()
    result = {"passed": 0, "total": total, "status": "sandbox_error"}
    sandbox = Path("/usr/bin/sandbox-exec")
    if sys.platform != "darwin" or not sandbox.is_file():
        return result
    prefix = Path(sys.base_prefix).resolve()
    launcher = prefix / "Resources/Python.app/Contents/MacOS/Python"
    executable = str(launcher if launcher.is_file() else Path(sys.executable).resolve())
    with tempfile.TemporaryDirectory(prefix="stark-agent-grade-") as temporary:
        work = Path(temporary).resolve()
        try:
            _stage_source(Path(repo).resolve(), work)
        except (OSError, ValueError):
            return dict(result, status="source_error")
        paths = (str(work), str(prefix), "/System", "/usr/lib", "/private/var/db/dyld")
        profile = (
            "(version 1) (deny default) (allow sysctl-read) "
            + f"(allow process-exec (literal {json.dumps(executable)})) "
            + '(allow file-read-metadata) (allow file-read-data (literal "/")) '
            + "(allow file-read-data file-map-executable "
            + " ".join(f"(subpath {json.dumps(path)})" for path in paths)
            + ")"
        )
        with tempfile.TemporaryFile() as output:
            try:
                child = subprocess.Popen(
                    [str(sandbox), "-p", profile, executable, "-I", "-B", "-c", _RUNNER, str(work)],
                    cwd=work,
                    env={"PATH": "/usr/bin:/bin", "LC_CTYPE": "UTF-8"},
                    stdin=subprocess.PIPE,
                    stdout=output,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                try:
                    child.communicate(payload, timeout=6)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.communicate()
                    return dict(result, status="timeout")
                output.seek(0)
                if child.returncode:
                    return dict(result, status="child_error")
                report = json.loads(output.read(1024))
            except (OSError, ValueError):
                return result
    if not isinstance(report, dict) or type(report.get("passed")) is not int:
        return result
    passed = report["passed"]
    if not 0 <= passed <= total:
        return result
    return {"passed": passed, "total": total, "status": "ok"}
