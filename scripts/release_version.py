"""Select the next immutable distribution version from fetched stable Git tags."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


def next_version(baseline: str, tags: list[str], bump: str) -> str:
    if bump not in ("major", "minor", "patch"):
        raise ValueError("invalid version bump")
    if VERSION.fullmatch(baseline) is None:
        raise ValueError("invalid source version")
    highest = tuple(map(int, baseline.split(".")))
    for tag in tags:
        if tag.startswith("v") and VERSION.fullmatch(tag[1:]):
            highest = max(highest, tuple(map(int, tag[1:].split("."))))
    major, minor, patch = highest
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def existing_release_source(version: str) -> str:
    if VERSION.fullmatch(version) is None:
        raise ValueError("invalid existing version")
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/v{version}^{{commit}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("invalid release revision")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return revision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bump", required=True, choices=("major", "minor", "patch"))
    parser.add_argument(
        "--existing-version",
        default="",
        help="Add a DMG to this existing stable release",
    )
    options = parser.parse_args()
    try:
        baseline = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"][
            "package"
        ]["version"]
        tags = subprocess.run(
            ["git", "tag", "--list", "v*"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        version = options.existing_version or next_version(baseline, tags, options.bump)
        revision = (
            existing_release_source(version) if options.existing_version else None
        )
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
            output.write(f"version={version}\n")
            if revision is not None:
                output.write(f"source_revision={revision}\n")
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError):
        parser.exit(1, "Unable to select release version.\n")
    print(f"Selected distribution version: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
