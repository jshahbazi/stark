#!/usr/bin/env python3
"""Sign and notarize a verified Stark bundle without rebuilding its engine.

Only aggregate status and hashes leave this command. Apple credentials remain
in the selected Keychain; no certificate export or plaintext password is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import build_distribution as distribution

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "share/stark/manifest.json"
RECIPE_FILES = ("scripts/sign_distribution.py", "scripts/build_distribution.py")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
MACHO_MAGIC = {
    b"\xcf\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xce",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
}


class SigningError(Exception):
    """A fixed, content-free signing failure."""


def require(condition: bool, stage: str) -> None:
    if not condition:
        raise SigningError(stage)


def environment() -> dict[str, str]:
    result = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "C",
        "LC_ALL": "C",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for name in ("HOME", "TMPDIR"):
        if name in os.environ:
            result[name] = os.environ[name]
    return result


def run(
    arguments: list[str | Path],
    stage: str,
    env: dict[str, str],
    *,
    cwd: Path = ROOT,
    timeout: int = 120,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            arguments,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SigningError(stage) from error
    require(result.returncode == 0, stage)
    return result


def relative_path(name: str) -> PurePosixPath:
    require(
        bool(name)
        and not name.startswith("/")
        and "\\" not in name
        and not any(ord(character) < 32 for character in name),
        "archive_path_invalid",
    )
    parts = name.split("/")
    require(all(part not in {"", ".", ".."} for part in parts), "archive_path_invalid")
    return PurePosixPath(*parts)


def extract_verified(
    archive: Path, digest: str, source: str, directory: Path
) -> tuple[Path, dict]:
    require(SHA256.fullmatch(digest) is not None, "input_digest_invalid")
    require(REVISION.fullmatch(source) is not None, "input_source_invalid")
    require(distribution.sha256(archive) == digest, "input_archive_hash_mismatch")
    with tarfile.open(archive) as contents:
        entries = contents.getmembers()
        require(len(entries) <= 100_000, "archive_too_many_entries")
        seen = set()
        total = 0
        for member in entries:
            name = member.name.rstrip("/") if member.isdir() else member.name
            path = relative_path(name)
            require(path.parts[0] == "stark", "archive_root_invalid")
            require(member.isfile() or member.isdir(), "archive_entry_type_invalid")
            require(path not in seen, "archive_duplicate_entry")
            seen.add(path)
            total += member.size
            require(total <= 2 * 1024**3, "archive_too_large")
        for member in entries:
            name = member.name.rstrip("/") if member.isdir() else member.name
            destination = directory / relative_path(name)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                destination.chmod(0o755)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                stream = contents.extractfile(member)
                require(stream is not None, "archive_file_unreadable")
                with stream, destination.open("xb") as output:
                    shutil.copyfileobj(stream, output)
                destination.chmod(member.mode & 0o755)
    bundle = directory / "stark"
    manifest = json.loads((bundle / MANIFEST).read_bytes())
    require(
        type(manifest.get("schema_version")) is int
        and manifest["schema_version"] == 1
        and manifest.get("target") == distribution.TARGET
        and manifest.get("minimum_macos") == distribution.MIN_MACOS,
        "input_manifest_identity",
    )
    require(
        manifest.get("source", {}).get("revision") == source
        and manifest["source"].get("dirty") is False,
        "input_source_identity",
    )
    require(
        (bundle / "share/stark/VERSION").read_text().strip() == manifest.get("version"),
        "input_version_identity",
    )
    hashes = manifest.get("files_sha256")
    require(isinstance(hashes, dict), "input_file_manifest_invalid")
    for name, expected in hashes.items():
        relative_path(name)
        require(
            isinstance(expected, str) and SHA256.fullmatch(expected) is not None,
            "input_file_hash_invalid",
        )
    actual = distribution.file_manifest(bundle)
    actual.pop(MANIFEST, None)
    require(actual == hashes, "input_file_integrity")
    return bundle, manifest


def resolve_identity(
    selected: str, keychain: Path | None, env: dict[str, str]
) -> dict[str, str]:
    arguments = ["/usr/bin/security", "find-identity", "-v", "-p", "codesigning"]
    if keychain is not None:
        arguments.append(str(keychain))
    identities = run(arguments, "signing_identity_lookup", env).stdout.decode()
    matches = []
    for line in identities.splitlines():
        match = re.fullmatch(r'\s*\d+\) ([0-9A-Fa-f]{40}) "(.+)"\s*', line)
        if match is None:
            continue
        certificate, name = match.groups()
        team = re.fullmatch(r"Developer ID Application: .+ \(([A-Z0-9]{10})\)", name)
        if team is not None and selected in {name, certificate, certificate.lower()}:
            matches.append({"sha1": certificate.upper(), "team_id": team[1]})
    require(len(matches) == 1, "developer_id_application_identity_required")
    return matches[0]


def macho_kind(path: Path) -> str | None:
    """Identify executable images, preserving non-loadable build object files."""
    thin = {
        b"\xcf\xfa\xed\xfe": "little",
        b"\xfe\xed\xfa\xcf": "big",
        b"\xce\xfa\xed\xfe": "little",
        b"\xfe\xed\xfa\xce": "big",
    }
    fat = {
        b"\xca\xfe\xba\xbe": ("big", 20),
        b"\xbe\xba\xfe\xca": ("little", 20),
        b"\xca\xfe\xba\xbf": ("big", 32),
        b"\xbf\xba\xfe\xca": ("little", 32),
    }
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(16)
        magic = header[:4]
        if magic not in MACHO_MAGIC:
            return None
        if magic in thin:
            require(len(header) == 16, "mach_o_header_invalid")
            types = {int.from_bytes(header[12:16], thin[magic])}
        else:
            byte_order, entry_size = fat[magic]
            count = int.from_bytes(header[4:8], byte_order)
            require(0 < count <= 64, "mach_o_header_invalid")
            stream.seek(8)
            table = stream.read(count * entry_size)
            require(len(table) == count * entry_size, "mach_o_header_invalid")
            types = set()
            for index in range(count):
                entry = table[index * entry_size : (index + 1) * entry_size]
                integer_size = 4 if entry_size == 20 else 8
                offset = int.from_bytes(entry[8 : 8 + integer_size], byte_order)
                size = int.from_bytes(
                    entry[8 + integer_size : 8 + 2 * integer_size], byte_order
                )
                require(
                    offset >= 8 + len(table)
                    and size >= 16
                    and offset + size <= file_size,
                    "mach_o_header_invalid",
                )
                stream.seek(offset)
                image = stream.read(16)
                require(image[:4] in thin, "mach_o_header_invalid")
                types.add(int.from_bytes(image[12:16], thin[image[:4]]))
    require(len(types) == 1, "mach_o_slice_type_mismatch")
    file_type = next(iter(types))
    if file_type == 1:  # MH_OBJECT cannot execute or be dynamically loaded.
        return "object"
    require(file_type in {2, 6, 7, 8, 11}, "mach_o_image_type_unsupported")
    return "runtime"


def macho_files(bundle: Path) -> list[Path]:
    result = []
    for path in sorted(bundle.rglob("*")):
        require(not path.is_symlink(), "bundle_contains_symlink")
        if path.is_file():
            if macho_kind(path) == "runtime":
                result.append(path)
        else:
            require(path.is_dir(), "bundle_contains_special_file")
    required = {
        "libexec/stark-score",
        "libexec/stark-generate",
        "libexec/python/bin/python3.13",
    }
    names = {path.relative_to(bundle).as_posix() for path in result}
    require(required <= names, "required_mach_o_missing")
    require(
        any(
            path.name.startswith("libpython3.13") and path.suffix == ".dylib"
            for path in result
        ),
        "python_library_missing",
    )
    return result


def verify_signature(
    path: Path,
    identity: dict[str, str],
    env: dict[str, str],
    *,
    notarized: bool = False,
) -> None:
    requirement = (
        "=anchor apple generic and "
        "certificate leaf[field.1.2.840.113635.100.6.1.13] exists and "
        f'certificate leaf = H"{identity["sha1"]}"'
    )
    verification = ["/usr/bin/codesign", "--verify", "--strict", "--all-architectures"]
    if notarized:
        verification.append("--check-notarization")
    run(
        [*verification, "-R", requirement, path],
        "code_signature_invalid",
        env,
    )
    detail = run(
        ["/usr/bin/codesign", "--display", "--verbose=4", path],
        "code_signature_metadata",
        env,
    ).stderr.decode(errors="replace")
    flags = re.search(r"(?m)^CodeDirectory\b[^\n]*\bflags=0x([0-9a-fA-F]+)\(", detail)
    require(
        flags is not None and int(flags[1], 16) & 0x10000 != 0,
        "hardened_runtime_missing",
    )
    require(
        re.search(rf"(?m)^TeamIdentifier={re.escape(identity['team_id'])}$", detail)
        is not None,
        "signing_team_mismatch",
    )
    timestamp = re.search(r"(?m)^Timestamp=(.+)$", detail)
    require(
        timestamp is not None
        and timestamp[1].strip().lower() not in {"none", "not set"},
        "secure_timestamp_missing",
    )
    entitlements = run(
        ["/usr/bin/codesign", "--display", "--entitlements", ":-", path],
        "code_signature_entitlements",
        env,
    )
    require(
        b"<key>" not in entitlements.stdout + entitlements.stderr,
        "unexpected_code_entitlements",
    )


def sign_bundle(
    bundle: Path, identity: dict[str, str], keychain: Path | None, env: dict[str, str]
) -> list[str]:
    paths = macho_files(bundle)
    for path in paths:
        relative = path.relative_to(bundle).as_posix()
        suffix = hashlib.sha256(relative.encode()).hexdigest()[:20]
        arguments = [
            "/usr/bin/codesign",
            "--force",
            "--sign",
            identity["sha1"],
            "--options",
            "runtime",
            "--timestamp",
            "--identifier",
            f"io.github.jshahbazi.stark.native.{suffix}",
        ]
        if keychain is not None:
            arguments.extend(["--keychain", str(keychain)])
        arguments.append(str(path))
        run(arguments, "code_signing_failed", env)
        verify_signature(path, identity, env)
    return [path.relative_to(bundle).as_posix() for path in paths]


def recipe_provenance(env: dict[str, str], allow_dirty: bool) -> dict:
    revision = (
        run(["/usr/bin/git", "rev-parse", "HEAD"], "recipe_revision", env)
        .stdout.decode()
        .strip()
    )
    require(REVISION.fullmatch(revision) is not None, "recipe_revision_invalid")
    status = run(
        [
            "/usr/bin/git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *RECIPE_FILES,
        ],
        "recipe_status",
        env,
    ).stdout
    require(not status or allow_dirty, "signing_recipe_has_changes")
    return {
        "revision": revision,
        "dirty": bool(status),
        "files_sha256": {
            name: distribution.sha256(ROOT / name) for name in RECIPE_FILES
        },
    }


def notarize(
    archive: Path, profile: str, keychain: Path | None, env: dict[str, str]
) -> dict:
    arguments = [
        "/usr/bin/xcrun",
        "notarytool",
        "submit",
        str(archive),
        "--keychain-profile",
        profile,
        "--wait",
        "--timeout",
        "30m",
        "--output-format",
        "json",
    ]
    if keychain is not None:
        arguments.extend(["--keychain", str(keychain)])
    result = json.loads(run(arguments, "notarization_failed", env, timeout=1860).stdout)
    require(result.get("status") == "Accepted", "notarization_not_accepted")
    submission = result.get("id")
    require(
        isinstance(submission, str)
        and re.fullmatch(
            r"[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}", submission
        )
        is not None,
        "notarization_receipt_invalid",
    )
    return {"status": "accepted", "submission_id": submission}


def publish(outputs: list[tuple[Path, Path]]) -> None:
    created = []
    try:
        for staged, destination in outputs:
            os.link(staged, destination)
            created.append(destination)
    except BaseException:
        for destination in reversed(created):
            destination.unlink()
        raise


def sign(options: argparse.Namespace) -> dict:
    require(
        platform.system() == "Darwin" and platform.machine() == "arm64",
        "signing_requires_apple_silicon_macos",
    )
    require(
        options.sign_only or bool(options.notary_profile), "notary_profile_required"
    )
    env = environment()
    recipe = recipe_provenance(env, allow_dirty=options.sign_only)
    output = options.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = f"stark-{options.version}-{distribution.TARGET}"
    if options.sign_only:
        stem += "-signed-unnotarized"
    receipt_name = (
        f"{stem}.signing.json" if options.sign_only else f"{stem}.notarization.json"
    )
    names = [f"{stem}.tar.gz", f"{stem}.tar.gz.sha256", receipt_name]
    if not options.sign_only:
        names.extend([f"{stem}.zip", f"{stem}.zip.sha256"])
    require(
        all(not (output / name).exists() for name in names),
        "distribution_output_already_exists",
    )
    with tempfile.TemporaryDirectory(prefix=".stark-signing-", dir=output) as temporary:
        work = Path(temporary)
        payload = work / "payload"
        payload.mkdir()
        bundle, manifest = extract_verified(
            options.archive.resolve(),
            options.input_sha256,
            options.expected_source,
            payload,
        )
        require(
            manifest["version"] != options.version, "new_distribution_version_required"
        )
        identity = resolve_identity(options.sign_identity, options.keychain, env)
        (bundle / "share/stark/VERSION").write_text(options.version + "\n")
        signed_paths = sign_bundle(bundle, identity, options.keychain, env)
        try:
            distribution.verify_relocation(bundle, env)
        except distribution.BuildError as error:
            raise SigningError("signed_runtime_relocation_failed") from error
        for name in signed_paths:
            verify_signature(bundle / name, identity, env)
        manifest["version"] = options.version
        manifest["distribution_source"] = recipe
        manifest["signing"] = {
            "certificate_sha1": identity["sha1"],
            "team_id": identity["team_id"],
            "hardened_runtime": True,
            "secure_timestamp": True,
            "mach_o_files": signed_paths,
            "input_archive_sha256": options.input_sha256,
            "non_loadable_mach_o_objects": [
                path.relative_to(bundle).as_posix()
                for path in sorted(bundle.rglob("*"))
                if path.is_file() and macho_kind(path) == "object"
            ],
        }
        (bundle / MANIFEST).unlink()
        manifest["files_sha256"] = distribution.file_manifest(bundle)
        (bundle / MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        archive = work / f"{stem}.tar.gz"
        digest = distribution.create_archive(bundle, archive)
        receipt = {
            "schema_version": 1,
            "version": options.version,
            "source_revision": options.expected_source,
            "distribution_source": recipe,
            "input_archive_sha256": options.input_sha256,
            "archive_sha256": digest,
            "certificate_sha1": identity["sha1"],
            "team_id": identity["team_id"],
            "mach_o_count": len(signed_paths),
            "hardened_runtime": True,
            "secure_timestamp": True,
            "models_loaded": 0,
            "native_inference_calls": 0,
            "notarization": {"status": "not_requested"},
        }
        if not options.sign_only:
            zipped = work / f"{stem}.zip"
            run(
                ["/usr/bin/ditto", "-c", "-k", "--keepParent", bundle, zipped],
                "notary_zip_failed",
                env,
            )
            receipt["zip_sha256"] = distribution.sha256(zipped)
            receipt["notarization"] = notarize(
                zipped, options.notary_profile, options.keychain, env
            )
            for name in (
                "libexec/stark-score",
                "libexec/stark-generate",
                "libexec/python/bin/python3.13",
            ):
                verify_signature(bundle / name, identity, env, notarized=True)
            receipt["notarization"]["standalone_executable_tickets"] = "verified_online"
            (work / f"{stem}.zip.sha256").write_text(
                f"{receipt['zip_sha256']}  {zipped.name}\n"
            )
        require(
            recipe_provenance(env, allow_dirty=options.sign_only) == recipe,
            "signing_recipe_changed",
        )
        (work / receipt_name).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        publish([(work / name, output / name) for name in names])
    print(
        json.dumps(
            {
                "version": options.version,
                "status": "signed_unnotarized"
                if options.sign_only
                else "signed_and_notarized",
                "archive_sha256": digest,
                "mach_o_count": len(signed_paths),
                "models_loaded": 0,
            },
            sort_keys=True,
        )
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--version", required=True, type=distribution.validate_version)
    parser.add_argument("--sign-identity", required=True)
    parser.add_argument("--notary-profile")
    parser.add_argument("--keychain", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/distributions"
    )
    parser.add_argument(
        "--sign-only",
        action="store_true",
        help="development check; emit explicitly unnotarized output names",
    )
    options = parser.parse_args()
    try:
        sign(options)
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001
        message = (
            str(error)
            if isinstance(error, SigningError)
            else "distribution_signing_failed"
        )
        print(f"ERROR: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
