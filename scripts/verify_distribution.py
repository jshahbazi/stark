#!/usr/bin/env python3
"""Verify a release archive outside the checkout, without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class VerificationFailed(Exception):
    """A content-free release verification failure."""


def require(condition: bool, stage: str) -> None:
    if not condition:
        raise VerificationFailed(stage)


def run(
    arguments: list[str | Path],
    environment: dict[str, str],
    cwd: Path,
    *,
    expected: int = 0,
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        arguments,
        env=environment,
        cwd=cwd,
        input=b"",
        capture_output=True,
        timeout=180,
        check=False,
    )
    require(result.returncode == expected, "command_failed")
    return result


def verify(archive: Path, version: str, expected_source: str) -> dict:
    archive = archive.resolve()
    with archive.open("rb") as archive_stream:
        digest = hashlib.file_digest(archive_stream, "sha256").hexdigest()
    with tarfile.open(archive) as contents:
        stream = contents.extractfile("stark/share/stark/manifest.json")
        require(stream is not None, "manifest_missing")
        raw_manifest = stream.read()
    manifest = json.loads(raw_manifest)
    require(
        manifest["schema_version"] == 1 and manifest["version"] == version,
        "bundle_identity",
    )
    require(manifest["source"]["revision"] == expected_source, "source_identity")
    require(manifest["source"]["dirty"] is False, "dirty_source")
    require(manifest["target"] == "aarch64-apple-darwin", "target_identity")
    converter_version = tomllib.loads(
        (ROOT / "tools/converter/pyproject.toml").read_text()
    )["project"]["version"]
    checks = []
    with tempfile.TemporaryDirectory(prefix="stark release verification ") as temporary:
        working = Path(temporary)
        prefix = working / "installation with spaces"
        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(working / "home"),
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": str(working / "empty-hub"),
            "PYTHONPATH": "/invalid-developer-python",
            "PYTHONHOME": "/invalid-developer-python",
        }
        Path(environment["HOME"]).mkdir()
        run(
            ["/bin/sh", ROOT / "install.sh", "--archive", archive, "--prefix", prefix],
            environment,
            working,
        )
        checks.append("archive_install_and_checksum")
        launcher = prefix / "bin/stark"
        run([launcher, "doctor"], environment, working)
        require(
            run([launcher, "--version"], environment, working).stdout.strip()
            == f"stark {version}".encode(),
            "launcher_version",
        )
        require(
            run(
                [prefix / "bin/stark-convert", "--version"], environment, working
            ).stdout.strip()
            == f"stark-convert {converter_version}".encode(),
            "converter_version",
        )
        checks.append("isolated_doctor_and_cli_identity")
        bundle = (prefix / "lib/stark/current").resolve()
        python = bundle / "libexec/python/bin/python3.13"
        run(
            [
                python,
                "-I",
                "-B",
                "-c",
                (
                    "import importlib.util; "
                    "assert importlib.util.find_spec('pip') is None; "
                    "assert importlib.util.find_spec('ensurepip') is None"
                ),
            ],
            environment,
            working,
        )
        checks.append("bootstrap_package_installers_absent")
        for command in ("score", "generate"):
            reply = run([launcher, command], environment, working, expected=1)
            require(
                not reply.stderr
                and json.loads(reply.stdout)
                == {"ready": False, "error": "invalid_arguments"},
                "worker_protocol",
            )
        checks.append("model_free_native_worker_protocol")
        # Use the same tiny synthetic checkpoint as the converter's source tests.
        # Only the parent verification process imports test helpers; the installed
        # converter receives local files and an isolated, offline environment.
        sys.path.insert(0, str(ROOT / "tools/converter/tests"))
        from test_cli import source_fixture

        source = source_fixture(working / "downloaded model", sharded=True)
        receipt = json.loads(
            run(
                [
                    launcher,
                    "convert",
                    source,
                    "--metadata-profile",
                    "hf-generic",
                    "--offline",
                    "--output",
                    working / "converted model",
                ],
                environment,
                working,
            ).stdout
        )
        require(receipt["status"] == "ok", "offline_conversion")
        checks.append("offline_synthetic_conversion_outside_checkout")
        for name, expected_hash in manifest["files_sha256"].items():
            path = (bundle / name).resolve()
            require(path.is_relative_to(bundle), "manifest_path")
            with path.open("rb") as entry:
                require(
                    hashlib.file_digest(entry, "sha256").hexdigest() == expected_hash,
                    "file_integrity",
                )
        checks.append("all_manifest_file_hashes")
        run(
            ["/bin/sh", ROOT / "install.sh", "--archive", archive, "--prefix", prefix],
            environment,
            working,
        )
        checks.append("exact_archive_reinstall")
        moved = working / "relocated installation"
        prefix.rename(moved)
        launcher = moved / "bin/stark"
        run([launcher, "doctor"], environment, working)
        run([launcher, "convert", "--version"], environment, working)
        checks.append("complete_prefix_relocation")
        model_marker = moved / "models/kept"
        model_marker.parent.mkdir()
        model_marker.write_text("synthetic preservation marker\n")
        run([launcher, "uninstall"], environment, working)
        require(
            model_marker.is_file() and not (moved / "lib/stark").exists(),
            "uninstall_preservation",
        )
        checks.append("uninstall_preserves_models")
    return {
        "status": "passed",
        "version": version,
        "source_revision": expected_source,
        "archive_sha256": digest,
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "source_files_manifest_sha256": manifest["source"]["files_manifest_sha256"],
        "file_count": len(manifest["files_sha256"]),
        "checks": checks,
        "models_loaded": 0,
        "native_inference_calls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    try:
        result = verify(options.archive, options.version, options.expected_source)
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    # Release checks must never emit exception dumps or private protocol data.
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001
        print(
            f"distribution verification failed ({type(error).__name__})",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "version": result["version"],
                "checks": len(result["checks"]),
                "archive_sha256": result["archive_sha256"],
                "models_loaded": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
