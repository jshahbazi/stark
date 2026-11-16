"""Independent, bounded acceptance checks for the manual user-test project.

Only aggregate booleans and the tested program's digest leave this module.
Generated stdout/stderr are bounded in memory and discarded. Running generated
Python is not a sandbox: invoke this only for the disposable user-test workspace.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import random
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

STAGES = ("build", "revise", "recover")
CASE_TIMEOUT = 3.0
TOTAL_TIMEOUT = 30.0
MAX_OUTPUT = 65_536
MAX_PROGRAM = 1_048_576


def fixture_rows() -> list[dict]:
    """Public sample data; acceptance also uses freshly varied private cases."""
    return [
        {"id": "chairs", "category": "equipment", "amount_cents": 12500},
        {"id": "coffee", "category": "food", "amount_cents": 3250},
        {"id": "snacks", "category": "food", "amount_cents": 1750},
        {"id": "flyers", "category": "printing", "amount_cents": 900},
    ]


def _program_bytes(path: Path) -> bytes | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_PROGRAM:
                return None
            return handle.read(MAX_PROGRAM + 1)
    except OSError:
        return None


def _run(
    program: Path, source: Path, arguments: list[str], directory: Path, deadline: float
) -> tuple[int, bytes] | None:
    """Drain both pipes with an output limit; always reap the process group."""
    remaining = min(CASE_TIMEOUT, deadline - time.monotonic())
    if remaining <= 0:
        return None
    environment = {
        "PATH": os.defpath,
        "HOME": str(directory),
        "TMPDIR": str(directory),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    child = None
    try:
        child = subprocess.Popen(
            [sys.executable, "-I", "-B", str(program), str(source), *arguments],
            cwd=directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        output = bytearray()
        total = 0
        stop = time.monotonic() + remaining
        with selectors.DefaultSelector() as selector:
            for pipe in (child.stdout, child.stderr):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ)
            while selector.get_map():
                if time.monotonic() >= stop:
                    return None
                for key, _mask in selector.select(
                    min(0.05, max(0, stop - time.monotonic()))
                ):
                    data = os.read(key.fileobj.fileno(), 8192)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    total += len(data)
                    if total > MAX_OUTPUT:
                        return None
                    if key.fileobj is child.stdout:
                        output.extend(data)
            return child.wait(timeout=max(0.01, stop - time.monotonic())), bytes(output)
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            for pipe in (child.stdout, child.stderr):
                if pipe is not None:
                    pipe.close()


def _expected(rows: list[dict], category: str | None = None) -> dict:
    selected = (
        rows
        if category is None
        else [row for row in rows if row["category"] == category]
    )
    totals = {}
    for row in selected:
        totals[row["category"]] = totals.get(row["category"], 0) + row["amount_cents"]
    return {
        "count": len(selected),
        "total_cents": sum(totals.values()),
        "by_category": totals,
    }


def _matches_json(output: bytes, expected: dict) -> bool:
    try:
        # Python considers True == 1; round-trip with canonical JSON preserves
        # the contract's integer and boolean distinction.
        actual = json.loads(output)
        return json.dumps(actual, sort_keys=True) == json.dumps(
            expected, sort_keys=True
        )
    except (ValueError, UnicodeError, RecursionError):
        return False


def _input_unchanged(path: Path, encoded: bytes) -> bool:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            return (
                stat.S_ISREG(info.st_mode)
                and info.st_size == len(encoded)
                and handle.read(len(encoded) + 1) == encoded
            )
    except OSError:
        return False


def verify_project(workspace: Path, stage: str) -> dict:
    """Check a project without trusting its own tests or emitting its content."""
    if stage not in STAGES:
        raise ValueError("unknown_stage")
    workspace = workspace.resolve()
    program = workspace / "ledger.py"
    checks = []

    def record(identifier: str, passed: bool) -> None:
        checks.append({"id": identifier, "passed": bool(passed)})

    source = None if program.is_symlink() else _program_bytes(program)
    valid = (
        source is not None
        and len(source) <= MAX_PROGRAM
        and program.resolve().parent == workspace
    )
    record("regular_program", valid)
    digest = hashlib.sha256(source).hexdigest() if valid else None
    if not valid:
        return {"passed": False, "checks": checks, "artifact_sha256": digest}

    rng = random.SystemRandom()
    categories = ["Food", "food", "rental space", "printing, signs"]
    rows = [
        {
            "id": f"entry-{index}-{rng.randrange(10**9)}",
            "category": categories[index % 4],
            "amount_cents": rng.randrange(1, 100_000),
        }
        for index in range(9)
    ]
    rows.append({"id": "free", "category": "Food", "amount_cents": 0})
    deadline = time.monotonic() + TOTAL_TIMEOUT
    with tempfile.TemporaryDirectory(prefix="stark-user-verify-") as temporary:
        directory = Path(temporary)
        inputs = directory / "expenses.json"

        def case(
            identifier: str,
            data: object,
            arguments: list[str],
            expected: dict | list | None,
        ) -> None:
            encoded = (
                data.encode() if isinstance(data, str) else json.dumps(data).encode()
            )
            try:
                inputs.unlink(missing_ok=True)
                with inputs.open("xb") as handle:
                    handle.write(encoded)
            except OSError:
                record(identifier, False)
                return
            result = _run(program, inputs, arguments, directory, deadline)
            unchanged = _input_unchanged(inputs, encoded)
            passed = False
            if result is not None:
                code, output = result
                if expected is None:
                    # Invalid input must fail without a success payload.
                    try:
                        json.loads(output)
                        json_output = True
                    except (ValueError, UnicodeError, RecursionError):
                        json_output = False
                    passed = code != 0 and not json_output
                elif isinstance(expected, dict):
                    passed = code == 0 and _matches_json(output, expected)
                else:
                    try:
                        passed = (
                            code == 0
                            and list(csv.reader(io.StringIO(output.decode())))
                            == expected
                        )
                    except (ValueError, UnicodeError, csv.Error):
                        passed = False
            record(identifier, passed and unchanged)

        case("sample_json", fixture_rows(), [], _expected(fixture_rows()))
        case("varied_json", rows, [], _expected(rows))
        case("empty_json", [], [], _expected([]))
        if stage in ("revise", "recover"):
            case("explicit_json", rows, ["--format", "json"], _expected(rows))
            for category, name in (
                ("Food", "exact_filter"),
                ("missing", "empty_filter"),
            ):
                case(name, rows, ["--category", category], _expected(rows, category))
            for category, name in (
                (None, "csv_totals"),
                ("food", "csv_filter"),
                ("missing", "csv_empty_filter"),
            ):
                expected = _expected(rows, category)["by_category"]
                csv_rows = [
                    ["category", "total_cents"],
                    *[[key, str(value)] for key, value in sorted(expected.items())],
                ]
                arguments = ["--format", "csv"] + (
                    [] if category is None else ["--category", category]
                )
                case(name, rows, arguments, csv_rows)
        if stage == "recover":
            bad_inputs = {
                "invalid_json": "{broken",
                "non_list": {"expenses": rows},
                "non_object_row": [5],
                "missing_field": [{"id": "a", "category": "food"}],
                "duplicate_id": [rows[0], rows[0]],
                "duplicate_id_different_fields": [
                    rows[0],
                    {"id": rows[0]["id"], "category": "other", "amount_cents": 7},
                ],
                "boolean_amount": [
                    {"id": "a", "category": "food", "amount_cents": True}
                ],
                "negative_amount": [
                    {"id": "a", "category": "food", "amount_cents": -1}
                ],
                "fractional_amount": [
                    {"id": "a", "category": "food", "amount_cents": 1.5}
                ],
                "non_string_id": [{"id": 1, "category": "food", "amount_cents": 1}],
                "non_string_category": [{"id": "a", "category": 1, "amount_cents": 1}],
            }
            for name, data in bad_inputs.items():
                case(name, data, [], None)
        record(
            "program_unchanged",
            not program.is_symlink() and _program_bytes(program) == source,
        )
    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "artifact_sha256": digest,
    }
