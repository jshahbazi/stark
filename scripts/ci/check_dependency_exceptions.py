#!/usr/bin/env python3
"""Validate the explicitly accepted, temporary rustls testing exception."""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = Path("scripts/ci/dependency-exceptions.toml")
ADVISORIES = frozenset(f"RUSTSEC-2026-{number:04d}" for number in range(346, 350))
LOCKFILES = frozenset({"Cargo.lock", "crates/legacy/download-manager/examples/wasm/Cargo.lock"})
REGISTRY = "registry+https://github.com/rust-lang/crates.io-index"
CHECKSUM = "0d41d731c7d2f962d1ccc364cec258de3c0e93b38c2fb3ba97ac74513048d634"


class InvalidPolicy(ValueError):
    """A fixed diagnostic that never includes input file contents."""


def require(condition: bool) -> None:
    if not condition:
        raise InvalidPolicy("dependency_exception_invalid")


@dataclass(frozen=True)
class ExceptionPolicy:
    ecosystem: str
    name: str
    version: str
    source: str
    checksum: str
    advisories: frozenset[str]
    lockfiles: frozenset[str]
    expires: date
    rationale: str

    def accepts(self, root: Path, lockfile: Path, identity: tuple[str, str, str], advisory: str) -> bool:
        """Match the canonical advisory ID and exact package/source scope."""
        relative = lockfile.resolve().relative_to(root.resolve()).as_posix()
        return (
            relative in self.lockfiles
            and identity == (self.ecosystem, self.name, self.version)
            and advisory in self.advisories
        )


def load_policy(path: Path, *, today: date | None = None) -> ExceptionPolicy:
    """Load only this reviewed scope, with an exclusive UTC expiry date."""
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    require(set(document) == {"schema_version", "exceptions"} and type(document["schema_version"]) is int)
    require(document["schema_version"] == 1)
    records = document["exceptions"]
    require(isinstance(records, list) and len(records) == 1 and isinstance(records[0], dict))
    record = records[0]
    require(set(record) == {
        "ecosystem", "name", "version", "source", "checksum", "advisories", "lockfiles", "expires", "rationale"
    })
    require((record["ecosystem"], record["name"], record["version"]) == ("crates.io", "rustls", "0.23.45"))
    require(record["source"] == REGISTRY and record["checksum"] == CHECKSUM)
    for key, expected in (("advisories", ADVISORIES), ("lockfiles", LOCKFILES)):
        values = record[key]
        require(isinstance(values, list) and all(isinstance(value, str) for value in values))
        require(len(values) == len(expected) and frozenset(values) == expected)
    require(record["expires"] == "2026-11-14")
    expires = date.fromisoformat(record["expires"])
    require(isinstance(record["rationale"], str) and bool(record["rationale"].strip()))
    require("\n" not in record["rationale"] and "\r" not in record["rationale"])
    if (today or datetime.now(timezone.utc).date()) >= expires:
        raise InvalidPolicy("dependency_exception_expired")
    return ExceptionPolicy(
        record["ecosystem"], record["name"], record["version"], record["source"], record["checksum"],
        frozenset(record["advisories"]), frozenset(record["lockfiles"]), expires, record["rationale"],
    )


def validate_preflight(policy: ExceptionPolicy, root: Path) -> None:
    """Bind the exception to both complete lockfiles and cargo-deny's ignores."""
    root = root.resolve()
    for filename in policy.lockfiles:
        path = root / filename
        require(path.resolve() == path and path.is_file())
        with path.open("rb") as stream:
            lock = tomllib.load(stream)
        packages = lock.get("package")
        require(isinstance(packages, list) and all(isinstance(package, dict) for package in packages))
        rustls = [package for package in packages if package.get("name") == policy.name]
        require(len(rustls) == 1)
        for package in rustls:
            require((package.get("version"), package.get("source"), package.get("checksum")) == (
                policy.version, policy.source, policy.checksum
            ))
    with (root / "deny.toml").open("rb") as stream:
        deny = tomllib.load(stream)
    advisories = deny.get("advisories")
    require(isinstance(advisories, dict))
    ignored = advisories.get("ignore")
    require(isinstance(ignored, list) and len(ignored) == len(policy.advisories))
    identifiers = []
    for entry in ignored:
        require(isinstance(entry, dict) and set(entry) == {"id", "reason"})
        require(isinstance(entry["id"], str) and entry["reason"] == policy.rationale)
        identifiers.append(entry["id"])
    require(frozenset(identifiers) == policy.advisories)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--exceptions", type=Path)
    arguments = parser.parse_args(argv)
    try:
        policy = load_policy(arguments.exceptions or arguments.root / POLICY_PATH)
        validate_preflight(policy, arguments.root)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, RecursionError, MemoryError) as error:
        reason = str(error) if isinstance(error, InvalidPolicy) else "dependency_exception_invalid"
        print(f"ERROR: {reason}", file=sys.stderr)
        return 2
    print(f"WARNING accepted-risk: {policy.ecosystem} {policy.name}@{policy.version} "
          f"advisories={len(policy.advisories)} expires={policy.expires.isoformat()} UTC (exclusive)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
