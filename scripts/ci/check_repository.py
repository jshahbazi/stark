#!/usr/bin/env python3
# /// script
# dependencies = ["PyYAML==6.0.3"]
# requires-python = ">=3.12,<3.13"
# ///

"""Portable repository policy checks used by CI with a locked YAML parser."""

from __future__ import annotations

import argparse
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
    "scripts/ci/check_repository.py.lock",
    "tools/uv.lock",
    "tools/converter/uv.lock",
    "tools/stock_benchmark/mlx/uv.lock",
    "tools/stock_benchmark/mlx_current/uv.lock",
    "tools/stock_benchmark/vllm/uv.lock",
}
SHA = re.compile(r"^[0-9a-fA-F]{40}$")
SECRET_REFERENCE = re.compile(r"\bsecrets\s*(?:\.|\[)")
WORKFLOW_EXPRESSION = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)
SIGNING_SECRETS = {
    "MACOS_CERTIFICATE_P12_BASE64",
    "MACOS_CERTIFICATE_PASSWORD",
    "MACOS_SIGNING_IDENTITY",
    "APPLE_API_KEY_P8_BASE64",
    "APPLE_API_KEY_ID",
    "APPLE_API_ISSUER_ID",
}
SIGNING_CONDITION = (
    "github.repository == 'jshahbazi/stark' && github.ref == 'refs/heads/main'"
)


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


def expression(value: object) -> str:
    result = str(value).strip()
    if result.startswith("${{") and result.endswith("}}"):
        result = result[3:-2].strip()
    return " ".join(result.split())


def has_secret_reference(value: str) -> bool:
    return bool(SECRET_REFERENCE.search(value)) or any(
        re.search(r"\bsecrets\b", item) for item in WORKFLOW_EXPRESSION.findall(value)
    )


def secret_references(value: object, path: tuple = (), ancestors: frozenset = frozenset()):
    """Visit each alias location, rejecting recursive YAML objects."""
    if isinstance(value, (dict, list)):
        if id(value) in ancestors:
            raise ValueError("recursive workflow object")
        ancestors = ancestors | {id(value)}
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and has_secret_reference(key):
                yield path + (key,), key
            yield from secret_references(child, path + (key,), ancestors)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from secret_references(child, path + (index,), ancestors)
    elif isinstance(value, str) and has_secret_reference(value):
        yield path, value


def check_signing_workflow(document: object, triggers: object) -> list[str]:
    """Allow Apple secrets only in the protected manual Stark signing job."""
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["signing workflow must be a mapping"]
    if not isinstance(triggers, dict) or set(triggers) != {"workflow_dispatch"}:
        errors.append("signing workflow must have only a manual dispatch trigger")
    if document.get("permissions") != {"contents": "read"}:
        errors.append("signing workflow must default to read-only repository access")
    jobs = document.get("jobs")
    if not isinstance(jobs, dict) or set(jobs) != {"sign", "publish"}:
        return errors + ["signing workflow must separate sign and publish jobs"]
    sign, publish = jobs["sign"], jobs["publish"]
    if not isinstance(sign, dict) or not isinstance(publish, dict):
        return errors + ["signing workflow jobs must be mappings"]
    if expression(sign.get("if")) != SIGNING_CONDITION:
        errors.append("signing job must require the Stark repository and main branch")
    if sign.get("environment") != "macos-release":
        errors.append("signing job must use the protected macos-release environment")
    if sign.get("runs-on") != "macos-26":
        errors.append("signing job must use the approved Mac runner")
    if sign.get("permissions", {"contents": "read"}) != {"contents": "read"}:
        errors.append("signing job must have read-only repository access")
    if expression(publish.get("if")) != SIGNING_CONDITION:
        errors.append("publishing job must require the Stark repository and main branch")
    if publish.get("needs") not in ("sign", ["sign"]):
        errors.append("publishing job must depend on successful signing")
    if publish.get("permissions") != {"contents": "write"}:
        errors.append("publishing job must have only release upload permission")
    if "environment" in publish:
        errors.append("publishing job must not enter a credential environment")

    steps = sign.get("steps")
    if not isinstance(steps, list):
        return errors + ["signing job must contain credential setup and cleanup steps"]
    allowed: dict[tuple, str] = {}
    imports: list[int] = []
    signing: list[int] = []
    cleanup: list[int] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        run = str(step.get("run", "")).strip()
        environment = step.get("env", {})
        if step.get("name") == "Import temporary Apple credentials":
            imports.append(index)
            if run != "python .github/scripts/apple_credentials.py setup":
                errors.append("credential import must use the reviewed setup helper")
            expected = {name: f"${{{{ secrets.{name} }}}}" for name in SIGNING_SECRETS}
            if environment != expected:
                errors.append("credential import must receive only the six Apple secrets")
            for name in SIGNING_SECRETS:
                allowed[("jobs", "sign", "steps", index, "env", name)] = f"secrets.{name}"
        elif isinstance(environment, dict) and "MACOS_SIGNING_IDENTITY" in environment:
            signing.append(index)
            if "scripts/sign_distribution.py" not in run:
                errors.append("signing identity must be scoped to the reviewed signer")
            if environment["MACOS_SIGNING_IDENTITY"] != "${{ secrets.MACOS_SIGNING_IDENTITY }}":
                errors.append("signer must receive the configured Apple signing identity")
            allowed[("jobs", "sign", "steps", index, "env", "MACOS_SIGNING_IDENTITY")] = (
                "secrets.MACOS_SIGNING_IDENTITY"
            )
        if run == "python .github/scripts/apple_credentials.py cleanup":
            cleanup.append(index)
            if expression(step.get("if")) != "always()":
                errors.append("credential cleanup must run unconditionally")
    if len(imports) != 1 or len(signing) != 1 or len(cleanup) != 1:
        errors.append("signing job must have one credential import, signer and cleanup")
    elif not imports[0] < signing[0] < cleanup[0]:
        errors.append("credential cleanup must follow import and signing")
    try:
        for path, value in secret_references(document):
            if allowed.get(path) != expression(value):
                errors.append("Apple secrets must remain scoped to approved signing steps")
    except ValueError:
        errors.append("signing workflow must not contain recursive YAML objects")
    return errors


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

        # Disabled inherited jobs retain historical references. The dedicated
        # protected manual signing job is the sole active Apple-secret consumer.
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
            if relative == ".github/workflows/signed-release.yml":
                errors.extend(f"{error}: {relative}" for error in check_signing_workflow(document, triggers))
            elif has_secret_reference(text):
                errors.append(f"secrets reference is forbidden in active workflow: {relative}")
            if has_secrets_mapping(document):
                errors.append(f"secrets mapping is forbidden in active workflow: {relative}")
            for job, definition in active_jobs(document):
                if not approved_runner(definition):
                    errors.append(f"unsupported runner configuration in active workflow: {relative}:{job}")
    return errors


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="run CI policy tests with the locked YAML parser")
    options = parser.parse_args(arguments)
    if options.test:
        import unittest

        suite = unittest.defaultTestLoader.discover(str(ROOT / "scripts" / "ci"), pattern="test_*.py")
        return 0 if unittest.TextTestRunner().run(suite).wasSuccessful() else 1
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
