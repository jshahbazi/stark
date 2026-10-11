#!/usr/bin/env python3
# /// script
# dependencies = ["PyYAML==6.0.3"]
# requires-python = ">=3.12,<3.13"
# ///

"""Portable repository policy checks used by CI with a locked YAML parser."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOTS = {"target", "models", "runtime", "test-artifacts", "artifacts", ".venv"}
WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".ggml")
INHERITED_DISABLED = {
    "release.yml",
    "run-benchmark.yml",
    "run-stark-cargo-command.yml",
    "tests.yml",
}
APPROVED_RUNNERS = {"ubuntu-24.04", "macos-26"}
# Every tracked supported lockfile must have an explicit OSV scan input in
# security.yml. Update this inventory and both scanner/gate inputs together.
SCANNED_LOCKFILES = {
    "Cargo.lock",
    "crates/legacy/download-manager/examples/wasm/Cargo.lock",
    "crates/legacy/stark/bindings/python/uv.lock",
    "crates/legacy/stark/bindings/typescript/pnpm-lock.yaml",
    "scripts/ci/check_repository.py.lock",
    "tools/uv.lock",
    "tools/converter/uv.lock",
    "tools/converter/lalamo/uv.lock",
}
SHA = re.compile(r"^[0-9a-fA-F]{40}$")


class UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects ambiguous duplicate mapping keys."""

    def construct_mapping(self, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    return [name for name in result.stdout.decode().split("\0") if name]


def check_tracked_policy(files: list[str]) -> list[str]:
    errors: list[str] = []
    for name in files:
        path = Path(name)
        if path.parts and path.parts[0] in PRIVATE_ROOTS:
            errors.append(f"private root directory is tracked: {name}")
        if path.name.startswith(".env") and path.name not in {".env.example", ".envrc"}:
            errors.append(f"environment file is tracked: {name}")
        if path.name.lower().endswith(WEIGHT_SUFFIXES):
            errors.append(f"model weight file is tracked: {name}")
    return errors


def check_lockfile_coverage(files: list[str]) -> list[str]:
    lockfiles = {
        name for name in files
        if Path(name).name in {"Cargo.lock", "uv.lock", "pnpm-lock.yaml"}
        or Path(name).name.endswith(".py.lock")
    }
    errors = [f"lockfile has no declared security scan: {name}"
              for name in sorted(lockfiles - SCANNED_LOCKFILES)]
    errors.extend(f"declared security lockfile is missing: {name}"
                  for name in sorted(SCANNED_LOCKFILES - lockfiles))
    return errors


def workflow_files() -> list[Path]:
    paths = list((ROOT / ".github" / "workflows").glob("*.yml"))
    paths += list((ROOT / ".github" / "workflows").glob("*.yaml"))
    paths += list((ROOT / ".github" / "actions").glob("**/*.yml"))
    paths += list((ROOT / ".github" / "actions").glob("**/*.yaml"))
    return sorted(paths)


def active_jobs(document: object) -> list[tuple[str, object]]:
    if not isinstance(document, dict):
        return []
    jobs = document.get("jobs", {})
    if not isinstance(jobs, dict):
        return []
    return [(str(name), job) for name, job in jobs.items()]


def job_is_disabled(job: object) -> bool:
    if not isinstance(job, dict):
        return False
    condition = job.get("if")
    return condition is False or str(condition).strip().lower() == "false"


def iter_uses(value: object, seen: set[int] | None = None):
    if seen is None:
        seen = set()
    if isinstance(value, (dict, list)):
        if id(value) in seen:
            return
        seen.add(id(value))
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key) == "uses" and isinstance(child, str):
                yield child
            yield from iter_uses(child, seen)
    elif isinstance(value, list):
        for child in value:
            yield from iter_uses(child, seen)


def approved_runner(job: object) -> bool:
    if not isinstance(job, dict):
        return False
    runner = job.get("runs-on")
    if isinstance(runner, str) and runner in APPROVED_RUNNERS:
        return True
    if runner != "${{ matrix.os }}":
        return False
    strategy = job.get("strategy")
    matrix = strategy.get("matrix") if isinstance(strategy, dict) else None
    values = matrix.get("os") if isinstance(matrix, dict) else None
    if not isinstance(values, list) or not values or not all(
        isinstance(value, str) and value in APPROVED_RUNNERS for value in values
    ):
        return False
    includes = matrix.get("include", []) if isinstance(matrix, dict) else []
    if not isinstance(includes, list):
        return False
    return all(
        isinstance(entry, dict)
        and ("os" not in entry or entry["os"] in APPROVED_RUNNERS)
        for entry in includes
    )


def has_trigger(triggers: object, name: str) -> bool:
    if isinstance(triggers, dict):
        return name in triggers
    if isinstance(triggers, list):
        return name in triggers
    return triggers == name


def has_secrets_mapping(value: object, seen: set[int] | None = None) -> bool:
    if seen is None:
        seen = set()
    if isinstance(value, (dict, list)):
        if id(value) in seen:
            return False
        seen.add(id(value))
    if isinstance(value, dict):
        if any(str(key) == "secrets" for key in value):
            return True
        return any(has_secrets_mapping(child, seen) for child in value.values())
    if isinstance(value, list):
        return any(has_secrets_mapping(child, seen) for child in value)
    return False


def check_workflows(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError:
            relative = path.as_posix()
        try:
            document = yaml.load(text, Loader=UniqueKeyLoader)
        except (yaml.YAMLError, RecursionError, MemoryError, TypeError, ValueError) as exc:
            errors.append(f"invalid workflow YAML: {relative} ({type(exc).__name__})")
            continue
        for reference in iter_uses(document):
            if reference.startswith("./"):
                continue
            # Docker actions may use an immutable image digest.
            if re.fullmatch(r"docker://[^@\s]+@sha256:[0-9a-fA-F]{64}", reference):
                continue
            if "@" not in reference or not SHA.fullmatch(reference.rsplit("@", 1)[1]):
                errors.append(f"unpinned action reference in {relative}")

        if path.parent.name == "workflows" and path.name in INHERITED_DISABLED:
            jobs = active_jobs(document)
            if not jobs:
                errors.append(f"inherited workflow has no jobs section: {relative}")
            else:
                for job, definition in jobs:
                    if not job_is_disabled(definition):
                        errors.append(f"inherited workflow job is not disabled: {relative}:{job}")

        # Disabled inherited jobs retain historical references. All active
        # workflows must avoid untrusted pull_request_target and secrets.
        disabled = path.parent.name == "workflows" and path.name in INHERITED_DISABLED
        if not disabled:
            if isinstance(document, dict) and True in document:
                triggers = document[True]
            elif isinstance(document, dict):
                triggers = document.get("on")
            else:
                triggers = None
            if has_trigger(triggers, "pull_request_target"):
                errors.append(f"pull_request_target is forbidden in active workflow: {relative}")
            if re.search(r"\bsecrets\s*(?:\.[A-Za-z_][A-Za-z0-9_]*|\[['\"][A-Za-z_][A-Za-z0-9_]*['\"]\])", text):
                errors.append(f"secrets reference is forbidden in active workflow: {relative}")
            if has_secrets_mapping(document):
                errors.append(f"secrets mapping is forbidden in active workflow: {relative}")
            for job, definition in active_jobs(document):
                if not approved_runner(definition):
                    errors.append(f"unsupported runner configuration in active workflow: {relative}:{job}")
    return errors


def main() -> int:
    files = tracked_files()
    errors = check_tracked_policy(files)
    # git ls-files includes pending working-tree deletions until they are staged.
    errors.extend(check_lockfile_coverage([name for name in files if (ROOT / name).is_file()]))
    errors.extend(check_workflows(workflow_files()))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("repository policy checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
