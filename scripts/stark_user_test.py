#!/usr/bin/env python3
"""Opt-in desktop user-test fixtures and aggregate evidence. Never drives inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import re
import shutil
import stat
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from user_simulation_project import verify_project

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runtime/user-simulation"
MANIFEST = "run.json"
CLIENTS = ("openwork", "opencode", "pi")
STAGES = ("build", "revise", "recover")
COMMON_STEPS = (
    "connection",
    "model",
    "stark_activity",
    "build_tools",
    "revise_tools",
    "failed_tool",
    "recover_tools",
    "cancel",
    "cancel_recovery",
    "cleanup",
)
LIFECYCLE_STEPS = {
    "openwork": ("disconnect_denied", "reconnect", "reconnect_recovery"),
    "opencode": ("disconnect_denied", "reconnect", "reconnect_recovery"),
    "pi": ("session_restart", "restart_recovery"),
}
STEP_NAMES = tuple(dict.fromkeys(COMMON_STEPS + sum(LIFECYCLE_STEPS.values(), ())))
FIXTURE = [
    {"id": "e1", "category": "food", "amount_cents": 1250},
    {"id": "e2", "category": "travel", "amount_cents": 2800},
    {"id": "e3", "category": "food", "amount_cents": 375},
]
BRIEFS = {
    "build": """Build a small event-expense ledger in this folder using Python 3's standard library.
Read expenses.json before implementing ledger.py. Run it as:
python3 ledger.py INPUT
INPUT is a JSON array of objects with string id, string category, and integer amount_cents.
Print exactly one JSON object with count (number of rows), total_cents, and by_category
(an object mapping each category to its total cents). Keep cents as integers. An empty
array produces count 0, total_cents 0, and by_category {}. Do not modify inputs.
Add your own tests and a short README. Run your tests and the CLI, inspect the actual
results, and finish the task. Work only in this disposable project; use no network,
packages, other providers, or files outside it. Use the selected Stark model.
""",
    "revise": """Extend the expense ledger we just built. Add --category CATEGORY for exact category
filtering, including a category with no matches. Add --format json|csv, default json.
CSV must contain the header category,total_cents and one row per selected category,
sorted by category ascending. JSON must retain the same count/total_cents/by_category
shape, computed after filtering. Preserve the original CLI, integer cents, empty
input behavior and input files. Update and run your tests, run both formats, and
inspect the results. Continue in this same conversation and project.
""",
    "recover": """Harden the ledger against bad expense files. First run the harmless failing command
python3 ledger.py missing-expenses.json and inspect its failure. Then reject malformed
JSON, a non-array input, a non-object row, missing fields, non-string id/category,
duplicate ids, and amount_cents that is a boolean, negative, or not an integer.
Invalid inputs must exit nonzero and print no successful JSON result to stdout.
Preserve valid JSON/CSV/filter behavior and never change inputs. Add and run regression
tests, then run the valid expenses.json command to demonstrate recovery. Continue in
this same conversation and project. Use only this project and the selected Stark model.
""",
}


class UserTestError(Exception):
    """Only fixed diagnostic codes cross the CLI boundary."""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(run: Path, value: dict) -> None:
    temporary = run / ".run.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(run / MANIFEST)


def load(run: Path) -> dict:
    # Runs are intentionally confined to this checkout's ignored private root.
    if run.is_symlink() or run.resolve().parent != RUNS.resolve():
        raise UserTestError("invalid_run_directory")
    path = run / MANIFEST
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise UserTestError("invalid_run_manifest")
    value = json.loads(path.read_text())
    if value.get("schema") != 1 or value.get("id") != run.name:
        raise UserTestError("invalid_run_manifest")
    if not value.get("clients") or set(value["clients"]) - set(CLIENTS):
        raise UserTestError("invalid_run_clients")
    for client in value["clients"]:
        workspace = run / client
        if workspace.is_symlink() or not workspace.is_dir():
            raise UserTestError("invalid_workspace")
    return value


def running_app_paths() -> dict[str, Path]:
    """Read running bundle locations without launching/activating any app."""
    script = """ObjC.import('AppKit');
    var apps = $.NSWorkspace.sharedWorkspace.runningApplications;
    var result = [];
    for (var i = 0; i < apps.count; i++) {
        var app = apps.objectAtIndex(i);
        if (app.bundleURL && app.bundleIdentifier) result.push({
            id: ObjC.unwrap(app.bundleIdentifier), path: ObjC.unwrap(app.bundleURL.path)});
    }
    JSON.stringify(result);"""
    identifiers = {
        "stark": "com.starkengine.menubar",
        "openwork": "com.differentai.openwork",
        "opencode": "ai.opencode.desktop",
    }
    try:
        result = subprocess.run(
            ["/usr/bin/osascript", "-l", "JavaScript", "-e", script],
            capture_output=True,
            check=False,
            timeout=10,
        )
        apps = json.loads(result.stdout) if result.returncode == 0 else []
        paths = {}
        for name, identifier in identifiers.items():
            matches = {
                Path(app["path"]).resolve() for app in apps if app["id"] == identifier
            }
            if len(matches) == 1:
                paths[name] = matches.pop()
        return paths
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return {}


def app_identity(name: str, app: Path | None) -> dict:
    if app is None:
        return {"version": "unavailable", "executable_sha256": None, "app_path": None}
    try:
        app = app.resolve()
        info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
        executable = app / "Contents/MacOS" / info["CFBundleExecutable"]
        identity = {
            "app_path": str(app),
            "version": str(info.get("CFBundleShortVersionString", "unknown")),
            "executable_sha256": digest(executable),
        }
        if name == "stark":
            runtime = app / "Contents/Resources/runtime"
            files = [
                runtime / "libexec" / name
                for name in ("stark-generate", "stark-template")
            ]
            files.append(runtime / "libexec/python/bin/python3.13")
            files += [
                p
                for p in runtime.rglob("*")
                if p.is_file() and p.suffix in (".py", ".so", ".dylib")
            ]
            if not any("stark_convert" in p.parts for p in files):
                raise ValueError("missing_backend")
            identity["runtime_sha256"] = {
                str(p.relative_to(runtime)): digest(p) for p in sorted(set(files))
            }
        return identity
    except (OSError, KeyError, ValueError, plistlib.InvalidFileException):
        return {
            "version": "unavailable",
            "executable_sha256": None,
            "app_path": str(app),
        }


def installed_identities(bound: dict | None = None) -> dict:
    paths = (
        running_app_paths()
        if bound is None
        else {
            name: Path(value["app_path"])
            for name, value in bound.items()
            if value.get("app_path")
        }
    )
    result = {
        name: app_identity(name, paths.get(name))
        for name in ("stark", "openwork", "opencode")
    }
    pi = (
        shutil.which("pi")
        if bound is None
        else bound.get("pi", {}).get("executable_path")
    )
    try:
        result["pi"] = (
            {
                "executable_path": str(Path(pi).resolve()),
                "executable_sha256": digest(Path(pi).resolve()),
            }
            if pi
            else {}
        )
    except OSError:
        result["pi"] = {}
    return result


def available_builds(builds: dict, clients: tuple | list) -> bool:
    return all(
        builds.get(name, {}).get("executable_sha256") for name in ("stark", *clients)
    ) and bool(builds.get("stark", {}).get("runtime_sha256"))


def prepare(clients: list[str]) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True, mode=0o700)
    run = RUNS / str(uuid.uuid4())
    run.mkdir(mode=0o700)
    state = {
        "schema": 1,
        "id": run.name,
        "created": now(),
        "clients": {},
        "harness_sha256": digest(Path(__file__)),
        "oracle_sha256": digest(Path(__file__).with_name("user_simulation_project.py")),
        "installed": {},
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, check=False
    )
    sha = revision.stdout.decode().strip()
    state["source_commit"] = sha if re.fullmatch(r"[a-f0-9]{40}", sha) else "unknown"
    changed = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, check=False
    )
    state["source_dirty"] = bool(changed.stdout)
    for client in clients:
        workspace = run / client
        workspace.mkdir(mode=0o700)
        (workspace / "expenses.json").write_text(json.dumps(FIXTURE, indent=2) + "\n")
        (workspace / "BRIEF.md").write_text(BRIEFS["build"])
        state["clients"][client] = {
            "identity": None,
            "observations": {},
            "verifications": {},
            "fixture_sha256": digest(workspace / "expenses.json"),
        }
    save(run, state)
    return run


def record_identity(
    state: dict,
    client: str,
    model: str,
    context: int,
    artifact: str | None,
    mtp: str,
    images: str,
    client_version: str,
) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.:/@+ -]{1,200}", model):
        raise UserTestError("invalid_model_id")
    if not re.fullmatch(r"[A-Za-z0-9_.+ -]{1,80}", client_version):
        raise UserTestError("invalid_client_version")
    if context < 1 or context > 10_000_000:
        raise UserTestError("invalid_context")
    if artifact is not None and not re.fullmatch(r"[a-f0-9]{64}", artifact):
        raise UserTestError("invalid_artifact_hash")
    value = {
        "model_id": model,
        "context_tokens": context,
        "artifact_sha256": artifact,
        "mtp": mtp,
        "images": images,
        "client_version": client_version,
    }
    entry = state["clients"][client]
    if entry["identity"] is not None and entry["identity"] != value:
        raise UserTestError("identity_changed_start_new_run")
    current = installed_identities()
    if not available_builds(current, [client]):
        raise UserTestError("tested_application_unavailable")
    for name in ("stark", client):
        if name in state["installed"] and state["installed"][name] != current[name]:
            raise UserTestError("tested_application_changed_start_new_run")
        state["installed"][name] = current[name]
    entry["identity"] = value


def observe(state: dict, client: str, step: str, outcome: str) -> None:
    if step not in COMMON_STEPS + LIFECYCLE_STEPS[client]:
        raise UserTestError("step_not_applicable")
    observations = state["clients"][client]["observations"]
    prerequisites = {
        "cancel_recovery": "cancel",
        "reconnect": "disconnect_denied",
        "reconnect_recovery": "reconnect",
        "restart_recovery": "session_restart",
    }
    if outcome == "pass" and step in prerequisites:
        prior = observations.get(prerequisites[step], [{}])[-1]
        if prior.get("outcome") != "pass":
            raise UserTestError("observe_predecessor_first")
    invalidates = {
        "cancel": ("cancel_recovery",),
        "disconnect_denied": ("reconnect", "reconnect_recovery"),
        "reconnect": ("reconnect_recovery",),
        "session_restart": ("restart_recovery",),
    }
    for later in invalidates.get(step, ()):
        if later in observations:
            observations[later].append({"outcome": "pending", "at": now()})
    observations.setdefault(step, []).append({"outcome": outcome, "at": now()})


def verify(run: Path, state: dict, client: str, stage: str) -> dict:
    if not tooling_unchanged(state):
        raise UserTestError("harness_changed_start_new_run")
    current = installed_identities()
    if not available_builds(current, [client]) or any(
        current.get(name) != state["installed"].get(name) for name in ("stark", client)
    ):
        raise UserTestError("tested_application_changed_start_new_run")
    entry = state["clients"][client]
    required = ("connection", "model", "stark_activity", stage + "_tools")
    if stage == "recover":
        required += ("failed_tool",)
    if not entry["identity"] or any(
        entry["observations"].get(step, [{}])[-1].get("outcome") != "pass"
        for step in required
    ):
        raise UserTestError("observe_identity_and_tools_first")
    for previous in STAGES[: STAGES.index(stage)]:
        prior = entry["verifications"].get(previous, [{}])[-1]
        if not prior.get("passed") or prior.get("superseded"):
            raise UserTestError("previous_stage_not_passed")
    result = verify_project(run / client, stage)
    sample = run / client / "expenses.json"
    unchanged = (
        sample.is_file()
        and not sample.is_symlink()
        and digest(sample) == entry["fixture_sha256"]
    )
    result["checks"].append({"id": "public_input_unchanged", "passed": unchanged})
    result["passed"] = result["passed"] and unchanged
    result["at"] = now()
    entry["verifications"].setdefault(stage, []).append(result)
    # Later acceptance must be repeated after an earlier stage is rerun.
    for later in STAGES[STAGES.index(stage) + 1 :]:
        for attempt in entry["verifications"].get(later, []):
            attempt["superseded"] = True
    return result


def tooling_unchanged(state: dict) -> bool:
    return (
        digest(Path(__file__)) == state["harness_sha256"]
        and digest(Path(__file__).with_name("user_simulation_project.py"))
        == state["oracle_sha256"]
    )


def summarize(run: Path, state: dict) -> dict:
    result = {
        key: state[key]
        for key in (
            "id",
            "source_commit",
            "source_dirty",
            "harness_sha256",
            "oracle_sha256",
            "installed",
        )
    }
    result["clients"] = {}
    for client, entry in state["clients"].items():
        checks = {}
        for step in COMMON_STEPS + LIFECYCLE_STEPS[client]:
            checks["ui_" + step] = (
                entry["observations"].get(step, [{}])[-1].get("outcome", "pending")
            )
        for stage in STAGES:
            attempt = entry["verifications"].get(stage, [{}])[-1]
            checks["project_" + stage] = (
                "pending"
                if not attempt or attempt.get("superseded")
                else "pass"
                if attempt.get("passed")
                else "fail"
            )
        checks["identity"] = "pass" if entry["identity"] else "pending"
        artifact = run / client / "ledger.py"
        latest = entry["verifications"].get("recover", [{}])[-1]
        current = (
            digest(artifact)
            if artifact.is_file() and not artifact.is_symlink()
            else None
        )
        checks["final_artifact"] = (
            "pass"
            if current and current == latest.get("artifact_sha256")
            else "pending"
        )
        sample = run / client / "expenses.json"
        checks["public_input_unchanged"] = (
            "pass"
            if sample.is_file()
            and not sample.is_symlink()
            and digest(sample) == entry["fixture_sha256"]
            else "fail"
        )
        status = (
            "failed"
            if "fail" in checks.values()
            else "passed"
            if set(checks.values()) == {"pass"}
            else "incomplete"
        )
        result["clients"][client] = {
            "status": status,
            "identity": entry["identity"],
            "checks": checks,
            "artifact_sha256": current,
            "verification_attempts": {
                s: len(v) for s, v in entry["verifications"].items()
            },
        }
    statuses = {entry["status"] for entry in result["clients"].values()}
    result["status"] = (
        "failed"
        if "failed" in statuses
        else "passed"
        if statuses == {"passed"}
        else "incomplete"
    )
    result["harness_unchanged"] = tooling_unchanged(state)
    current_installed = installed_identities(state["installed"])
    result["installed_unchanged"] = available_builds(
        state["installed"], list(state["clients"])
    ) and all(
        current_installed.get(client) == state["installed"].get(client)
        for client in ("stark", *state["clients"])
    )
    result["evidence_valid"] = (
        result["harness_unchanged"] and result["installed_unchanged"]
    )
    if not result["evidence_valid"] and result["status"] != "failed":
        result["status"] = "incomplete"
    return result


def report(run: Path, state: dict) -> dict:
    result = summarize(run, state)
    (run / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Stark desktop user simulation",
        "",
        f"Outcome: {result['status']}",
        "",
        f"Harness source: `{state['source_commit']}`",
        "",
        f"Helpers unchanged: {result['harness_unchanged']}. Selected installed apps unchanged: {result['installed_unchanged']}.",
        f"Evidence identities valid: {result['evidence_valid']}.",
        "",
        "UI checks are operator observations. Project checks are independently executed.",
        "",
        "| Client | Outcome | Project checks | UI observations |",
        "| --- | --- | --- | --- |",
    ]
    for client, value in result["clients"].items():
        checks = value["checks"]
        projects = sum(checks["project_" + s] == "pass" for s in STAGES)
        ui = [v for k, v in checks.items() if k.startswith("ui_")]
        lines.append(
            f"| {client} | {value['status']} | {projects}/3 | {ui.count('pass')}/{len(ui)} |"
        )
    lines += [
        "",
        "See summary.json for build identities, hashes and individual outcomes.",
        "",
        "This run covers the selected clients, artifact and task only. Pi lifecycle covers session restart;",
        "it does not test managed revocation. No transcript or raw subprocess output is recorded.",
    ]
    (run / "report.md").write_text("\n".join(lines) + "\n")
    return result


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser(
        "prepare", help="Create private workspaces; never starts an app or inference"
    )
    create.add_argument("--clients", nargs="+", choices=CLIENTS, default=list(CLIENTS))
    for command in ("identity", "observe", "verify", "report", "brief"):
        child = sub.add_parser(command)
        child.add_argument("--run", type=Path, required=True)
        if command != "report":
            child.add_argument("--client", choices=CLIENTS, required=True)
        if command in ("verify", "brief"):
            child.add_argument("--stage", choices=STAGES, required=True)
        if command == "identity":
            child.add_argument("--model", required=True)
            child.add_argument("--context", type=int, required=True)
            child.add_argument("--artifact-sha256")
            child.add_argument(
                "--mtp", choices=("on", "off", "unavailable", "unknown"), required=True
            )
            child.add_argument(
                "--images", choices=("loaded", "text-only", "unknown"), required=True
            )
            child.add_argument("--client-version", required=True)
        if command == "observe":
            child.add_argument("--step", choices=STEP_NAMES, required=True)
            child.add_argument(
                "--outcome", choices=("pass", "fail", "blocked"), required=True
            )
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            print(prepare(list(dict.fromkeys(args.clients))))
            return 0
        run = args.run.absolute()
        state = load(run)
        if hasattr(args, "client") and args.client not in state["clients"]:
            raise UserTestError("client_not_in_run")
        if args.command == "brief":
            print(BRIEFS[args.stage])
            return 0
        if args.command == "identity":
            record_identity(
                state,
                args.client,
                args.model,
                args.context,
                args.artifact_sha256,
                args.mtp,
                args.images,
                args.client_version,
            )
        elif args.command == "observe":
            observe(state, args.client, args.step, args.outcome)
        elif args.command == "verify":
            verified = verify(run, state, args.client, args.stage)
            save(run, state)
            print(json.dumps(verified, sort_keys=True))
            return 0 if verified["passed"] else 1
        elif args.command == "report":
            result = report(run, state)
            print(json.dumps(result, sort_keys=True))
            return {"passed": 0, "failed": 1, "incomplete": 2}[result["status"]]
        save(run, state)
        print("recorded")
        return 0
    except UserTestError as error:
        print(str(error), file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError):
        print("user_test_operation_failed", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
