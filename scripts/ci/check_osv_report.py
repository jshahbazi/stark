#!/usr/bin/env python3
"""Gate OSV lockfile JSON, retaining explicit maintenance and accepted-risk warnings."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from check_dependency_exceptions import ExceptionPolicy, load_policy, validate_preflight

ROOT = Path(__file__).resolve().parents[2]
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+!-]{0,127}$")
GIT_COMMIT = re.compile(r"^[0-9A-Fa-f]{40}$")
SCHEMA_VERSION = re.compile(r"^1\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
TIMESTAMP = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z$")
PACKAGE_NAMES = {
    "crates.io": re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,213}$"),
    "PyPI": re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,213}$"),
    "npm": re.compile(r"^(?:@[A-Za-z0-9._~-]+/)?[A-Za-z0-9._~-]{1,214}$"),
}


class InvalidReport(ValueError):
    """An authored failure with no report contents in its diagnostic."""


def require(condition: bool) -> None:
    if not condition:
        raise InvalidReport("osv_report_invalid")


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        require(key not in result)
        result[key] = value
    return result


def reject_constant(_value: str) -> None:
    raise InvalidReport("osv_report_invalid")


def canonical_name(ecosystem: str, name: str) -> str:
    if ecosystem == "PyPI":
        return re.sub(r"[-_.]+", "-", name).lower()
    return name


def lockfile_ecosystem(path: Path) -> str:
    if path.name == "Cargo.lock":
        return "crates.io"
    if path.name == "uv.lock" or path.name.endswith(".py.lock"):
        return "PyPI"
    if path.name == "pnpm-lock.yaml":
        return "npm"
    raise InvalidReport("osv_report_invalid")


def expected_sources(root: Path, lockfiles: list[str]) -> dict[Path, str]:
    require(bool(lockfiles))
    sources: dict[Path, str] = {}
    for filename in lockfiles:
        require(isinstance(filename, str) and bool(filename) and "\0" not in filename)
        path = (root / filename).resolve()
        require(path.is_relative_to(root) and path not in sources)
        sources[path] = lockfile_ecosystem(path)
    return sources


def package_identity(package: object, ecosystem: str) -> tuple[str, str, str]:
    require(isinstance(package, dict))
    required = {"ecosystem", "name", "version"}
    require(required <= set(package) <= required | {"commit"})
    if "commit" in package:
        require(isinstance(package["commit"], str) and GIT_COMMIT.fullmatch(package["commit"]) is not None)
    require(package["ecosystem"] == ecosystem)
    name, version = package["name"], package["version"]
    require(isinstance(name, str) and PACKAGE_NAMES[ecosystem].fullmatch(name) is not None)
    # OSV --all-packages retains local uv projects with an empty version.
    require(isinstance(version, str) and (
        (not version and ecosystem == "PyPI") or VERSION.fullmatch(version) is not None
    ))
    return ecosystem, canonical_name(ecosystem, name), version


def advisory_identity(advisory: object) -> tuple[str, set[str]]:
    require(isinstance(advisory, dict))
    schema_version = advisory.get("schema_version", "1.0.0")
    require(isinstance(schema_version, str) and SCHEMA_VERSION.fullmatch(schema_version) is not None)
    for field in ("modified", "published", "withdrawn"):
        if field == "modified" or field in advisory:
            value = advisory.get(field)
            require(isinstance(value, str) and TIMESTAMP.fullmatch(value) is not None)
            datetime.fromisoformat(value)
    for field in ("summary", "details"):
        if field in advisory:
            require(isinstance(advisory[field], str))
    if "database_specific" in advisory:
        require(isinstance(advisory["database_specific"], dict))
    identifier = advisory.get("id")
    require(isinstance(identifier, str) and IDENTIFIER.fullmatch(identifier) is not None)
    aliases = advisory.get("aliases", [])
    require(isinstance(aliases, list))
    require(all(isinstance(alias, str) and IDENTIFIER.fullmatch(alias) is not None for alias in aliases))
    require(len(aliases) == len(set(aliases)))
    return identifier, {identifier, *aliases}


def check_affected_ranges(entry: dict) -> None:
    ranges, versions = entry.get("ranges", []), entry.get("versions", [])
    require(isinstance(ranges, list) and isinstance(versions, list))
    require(bool(ranges) or bool(versions))
    require(all(isinstance(version, str) and bool(version) for version in versions))
    for interval in ranges:
        require(isinstance(interval, dict))
        require(interval.get("type") in {"SEMVER", "ECOSYSTEM", "GIT"})
        if interval["type"] == "GIT":
            require(isinstance(interval.get("repo"), str) and bool(interval["repo"]))
        events = interval.get("events")
        require(isinstance(events, list) and bool(events))
        for event in events:
            require(isinstance(event, dict) and len(event) == 1)
            require(set(event).issubset({"introduced", "fixed", "last_affected", "limit"}))
            require(all(isinstance(value, str) and bool(value) for value in event.values()))
        if "database_specific" in interval:
            require(isinstance(interval["database_specific"], dict))
    if "ecosystem_specific" in entry:
        require(isinstance(entry["ecosystem_specific"], dict))


def maintenance_only(advisory: dict, identity: tuple[str, str, str]) -> bool:
    ecosystem, name, version = identity
    require(bool(version))
    affected = advisory.get("affected")
    require(isinstance(affected, list) and bool(affected))
    matched: list[dict] = []
    for entry in affected:
        require(isinstance(entry, dict))
        package = entry.get("package")
        require(isinstance(package, dict))
        affected_ecosystem, affected_name = package.get("ecosystem"), package.get("name")
        require(isinstance(affected_ecosystem, str) and bool(affected_ecosystem))
        require(isinstance(affected_name, str) and bool(affected_name))
        check_affected_ranges(entry)
        specific = entry.get("database_specific", {})
        require(isinstance(specific, dict))
        if affected_ecosystem == ecosystem and canonical_name(ecosystem, affected_name) == name:
            matched.append(specific)
    require(bool(matched))
    # The pinned scanner matches dependency versions against advisory ranges.
    # Requiring every same-package entry to be maintenance-only prevents an
    # unrelated affected entry or mixed classification from downgrading a finding.
    return all(entry.get("informational") == "unmaintained" for entry in matched)


def check_groups(groups: object, identifiers: set[str]) -> None:
    require(isinstance(groups, list))
    for group in groups:
        require(isinstance(group, dict))
        values = group.get("ids")
        require(isinstance(values, list) and bool(values))
        require(all(isinstance(value, str) and value in identifiers for value in values))


def validate_report(
    document: object, root: Path, lockfiles: list[str], exceptions: ExceptionPolicy | None = None
) -> dict:
    root = root.resolve()
    expected = expected_sources(root, lockfiles)
    require(isinstance(document, dict))
    require(set(document).issubset({"results", "experimental_config"}))
    if "experimental_config" in document:
        require(isinstance(document["experimental_config"], dict))
    results = document.get("results")
    require(isinstance(results, list) and bool(results))
    seen_sources: set[Path] = set()
    notices: set[tuple[str, str, str, str]] = set()
    failures: set[tuple[str, str, str, str]] = set()
    accepted: set[tuple[str, str, str, str]] = set()
    package_count = 0
    for result in results:
        require(isinstance(result, dict) and set(result) == {"source", "packages"})
        source = result["source"]
        require(isinstance(source, dict) and set(source) == {"path", "type"})
        require(source["type"] == "lockfile")
        filename = source["path"]
        require(isinstance(filename, str) and bool(filename) and "\0" not in filename)
        path = (root / filename).resolve()
        require(path in expected and path not in seen_sources)
        seen_sources.add(path)
        packages = result["packages"]
        require(isinstance(packages, list) and bool(packages))
        seen_packages: set[tuple[str, str, str]] = set()
        for package in packages:
            require(isinstance(package, dict))
            require(set(package).issubset({"package", "vulnerabilities", "groups", "dependency_groups"}))
            identity = package_identity(package.get("package"), expected[path])
            require(identity not in seen_packages)
            seen_packages.add(identity)
            package_count += 1
            if "dependency_groups" in package:
                require(isinstance(package["dependency_groups"], list))
                require(all(isinstance(group, str) for group in package["dependency_groups"]))
            vulnerabilities = package.get("vulnerabilities", [])
            require(isinstance(vulnerabilities, list))
            identifiers: set[str] = set()
            advisory_ids: set[str] = set()
            for advisory in vulnerabilities:
                identifier, aliases = advisory_identity(advisory)
                require(identifier not in advisory_ids)
                advisory_ids.add(identifier)
                identifiers.update(aliases)
                finding = (*identity, identifier)
                # Validate every affected entry before considering an exception.
                maintenance = maintenance_only(advisory, identity)
                if exceptions is not None and exceptions.accepts(root, path, identity, identifier):
                    accepted.add(finding)
                elif maintenance:
                    notices.add(finding)
                else:
                    failures.add(finding)
            if "groups" in package:
                check_groups(package["groups"], identifiers)
    require(seen_sources == set(expected))
    return {"sources": len(seen_sources), "packages": package_count,
            "maintenance": sorted(notices), "blocking": sorted(failures), "accepted": sorted(accepted)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--lockfile", action="append", required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--exceptions", type=Path)
    arguments = parser.parse_args(argv)
    try:
        document = json.loads(arguments.report.read_text(encoding="utf-8"),
                              object_pairs_hook=unique_object, parse_constant=reject_constant)
        exceptions = load_policy(arguments.exceptions) if arguments.exceptions is not None else None
        if exceptions is not None:
            validate_preflight(exceptions, arguments.root)
        result = validate_report(document, arguments.root, arguments.lockfile, exceptions)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, RecursionError, MemoryError):
        print("ERROR: osv_report_invalid", file=sys.stderr)
        return 2
    for kind, label in (("maintenance", "WARNING maintenance"), ("accepted", "WARNING accepted-risk"),
                        ("blocking", "ERROR advisory")):
        for ecosystem, name, version, identifier in result[kind]:
            print(f"{label}: {ecosystem} {name}@{version} {identifier}")
    print(f"OSV summary: sources={result['sources']} packages={result['packages']} "
          f"maintenance={len(result['maintenance'])} accepted={len(result['accepted'])} blocking={len(result['blocking'])}")
    return 1 if result["blocking"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
