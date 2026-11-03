#!/usr/bin/env python3
"""Build the local, self-contained Stark menu bar application."""

from __future__ import annotations

import argparse
import ctypes
import email.parser
import hashlib
import json
import os
import platform
import plistlib
import re
import runpy
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import sign_distribution as signing

ROOT = Path(__file__).resolve().parent.parent
APP_ID = "com.starkengine.menubar"
MIN_MACOS = "26.4"
RUNTIME_MANIFEST = Path("share/stark/manifest.json")
PYTHON = Path("libexec/python/bin/python3.13")
CONVERTER = Path("libexec/python/lib/python3.13/site-packages/stark_convert")
OPENWORK_CONNECTOR_LOCK = ROOT / "apps/menu-bar/Resources/OpenWorkConnector.lock.json"
OPENWORK_CONNECTOR_REPOSITORY = "https://github.com/jshahbazi/stark-openwork"
OPENWORK_CONNECTOR_FILES = (
    "package.json",
    "index.js",
    "server.js",
    "connector.js",
    "protocol.js",
    "LICENSE",
)
MAX_CONNECTOR_FILE_BYTES = 256 * 1024
MAX_CONNECTOR_TOTAL_BYTES = 1024 * 1024


class BuildError(Exception):
    """Fixed, content-free build diagnostic."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_openwork_connector_lock(path: Path | None = None) -> dict:
    """Accept only a fixed first-party source and a complete immutable pin."""
    path = path or OPENWORK_CONNECTOR_LOCK

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError
            result[key] = value
        return result

    descriptor = -1
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
        )
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size > 16 * 1024
        ):
            raise ValueError
        data = bytearray()
        while True:
            block = os.read(descriptor, 16 * 1024 + 1 - len(data))
            if not block:
                break
            data.extend(block)
            if len(data) > 16 * 1024:
                raise ValueError
        after = os.fstat(descriptor)
        if (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or len(data) != info.st_size:
            raise ValueError
        lock = json.loads(data, object_pairs_hook=unique_object)
        if not isinstance(lock, dict) or set(lock) != {
            "schema_version",
            "protocol_version",
            "repository",
            "revision",
            "files_sha256",
        }:
            raise ValueError
        if (
            type(lock["schema_version"]) is not int
            or lock["schema_version"] != 1
            or type(lock["protocol_version"]) is not int
            or lock["protocol_version"] != 1
            or lock["repository"] != OPENWORK_CONNECTOR_REPOSITORY
            or not isinstance(lock["files_sha256"], dict)
        ):
            raise ValueError
        files = lock["files_sha256"]
        if not set(files).issubset(OPENWORK_CONNECTOR_FILES) or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in files.values()
        ):
            raise ValueError
        revision = lock["revision"]
        if revision is not None and (
            not isinstance(revision, str)
            or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or set(files) != set(OPENWORK_CONNECTOR_FILES)
        ):
            raise ValueError
        return {**lock, "lock_sha256": hashlib.sha256(data).hexdigest()}
    except (OSError, ValueError, TypeError, UnicodeError):
        raise BuildError("openwork_connector_lock_invalid") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_openwork_connector_file(directory: int, name: str) -> bytes:
    if name not in OPENWORK_CONNECTOR_FILES:
        raise BuildError("openwork_connector_source_invalid")
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
            dir_fd=directory,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & 0o133
            or before.st_size < 1
            or before.st_size > MAX_CONNECTOR_FILE_BYTES
        ):
            raise ValueError
        data = bytearray()
        while True:
            block = os.read(
                descriptor, min(8192, MAX_CONNECTOR_FILE_BYTES + 1 - len(data))
            )
            if not block:
                break
            data.extend(block)
            if len(data) > MAX_CONNECTOR_FILE_BYTES:
                raise ValueError
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or len(data) != before.st_size:
            raise ValueError
        return bytes(data)
    except (OSError, ValueError):
        raise BuildError("openwork_connector_source_invalid") from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _download_openwork_connector_file(revision: str, name: str) -> bytes:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_args, **_kwargs):
            return None

    if (
        not re.fullmatch(r"[0-9a-f]{40}", revision)
        or name not in OPENWORK_CONNECTOR_FILES
    ):
        raise BuildError("openwork_connector_download_failed")
    url = (
        f"https://raw.githubusercontent.com/jshahbazi/stark-openwork/{revision}/{name}"
    )
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Stark-application-builder",
            "Accept": "application/octet-stream",
        },
    )
    try:
        with urllib.request.build_opener(NoRedirect()).open(
            request, timeout=30
        ) as response:
            if response.status != 200 or response.geturl() != url:
                raise ValueError
            size = response.headers.get("Content-Length")
            if size is not None and not 1 <= int(size) <= MAX_CONNECTOR_FILE_BYTES:
                raise ValueError
            data = response.read(MAX_CONNECTOR_FILE_BYTES + 1)
            if not 1 <= len(data) <= MAX_CONNECTOR_FILE_BYTES:
                raise ValueError
            return data
    except (OSError, ValueError, urllib.error.URLError):
        raise BuildError("openwork_connector_download_failed") from None


def install_openwork_connector(
    resources: Path,
    *,
    source: Path | None = None,
    allow_development: bool = False,
    lockfile: Path | None = None,
) -> dict:
    """Bundle only public, reviewed connector files; never execute package scripts."""
    lock_path = lockfile or OPENWORK_CONNECTOR_LOCK
    lock = read_openwork_connector_lock(lock_path)
    pinned = lock["revision"] is not None
    if allow_development and source is None:
        raise BuildError("openwork_connector_development_source_required")
    if not pinned and not (source is not None and allow_development):
        raise BuildError("openwork_connector_lock_unpinned")
    payload: dict[str, bytes] = {}
    if source is not None:
        directory = -1
        try:
            source = source.expanduser().absolute()
            if ".." in source.parts or source.is_symlink():
                raise ValueError
            directory = os.open(
                source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            )
            info = os.fstat(directory)
            if info.st_mode & 0o022:
                raise ValueError
            payload = {
                name: _read_openwork_connector_file(directory, name)
                for name in OPENWORK_CONNECTOR_FILES
            }
        except (OSError, ValueError):
            raise BuildError("openwork_connector_source_invalid") from None
        finally:
            if directory >= 0:
                os.close(directory)
    else:
        payload = {
            name: _download_openwork_connector_file(lock["revision"], name)
            for name in OPENWORK_CONNECTOR_FILES
        }
    if sum(map(len, payload.values())) > MAX_CONNECTOR_TOTAL_BYTES:
        raise BuildError("openwork_connector_source_invalid")
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    if pinned and hashes != lock["files_sha256"]:
        raise BuildError("openwork_connector_hash_mismatch")
    try:
        for data in payload.values():
            if b"\0" in data:
                raise ValueError
            data.decode("utf-8")
        package = json.loads(payload["package.json"])
        if (
            not isinstance(package, dict)
            or package.get("name") != "@stark/openwork-provider"
            or package.get("type") != "module"
            or package.get("main") != "./server.js"
            or package.get("exports") != {".": "./index.js", "./server": "./server.js"}
            or any(
                package.get(name)
                for name in ("dependencies", "optionalDependencies", "peerDependencies")
            )
        ):
            raise ValueError
    except (ValueError, TypeError, UnicodeError):
        raise BuildError("openwork_connector_package_invalid") from None
    target = resources / "OpenWorkConnector"
    if (
        resources.is_symlink()
        or not resources.is_dir()
        or target.exists()
        or target.is_symlink()
    ):
        raise BuildError("openwork_connector_destination_invalid")
    target_descriptor = -1
    target_identity = None
    try:
        target.mkdir(mode=0o755)
        target_descriptor = os.open(
            target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        target_identity = os.fstat(target_descriptor)
        for name, data in payload.items():
            destination = target / name
            destination.write_bytes(data)
            destination.chmod(0o644)
    except OSError:
        try:
            current = target.lstat()
            if (
                target_identity is not None
                and stat.S_ISDIR(current.st_mode)
                and (current.st_dev, current.st_ino)
                == (target_identity.st_dev, target_identity.st_ino)
            ):
                shutil.rmtree(target)
        except OSError:
            pass
        raise BuildError("openwork_connector_copy_failed") from None
    finally:
        if target_descriptor >= 0:
            os.close(target_descriptor)
    return {
        "repository": OPENWORK_CONNECTOR_REPOSITORY,
        "revision": lock["revision"],
        "protocol_version": lock["protocol_version"],
        "development_override": not pinned,
        "source_mode": "local" if source is not None else "download",
        "lock_sha256": lock["lock_sha256"],
        "files_sha256": hashes,
        "resources_sha256": {
            f"OpenWorkConnector/{name}": digest for name, digest in hashes.items()
        },
    }


def run(command: list[str], stage: str) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        raise BuildError(stage + "_unavailable") from None
    if result.returncode:
        raise BuildError(stage + "_failed")
    return result.stdout.decode("utf-8", errors="replace").strip()


def read_plist(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            value = plistlib.load(stream)
        if not isinstance(value, dict):
            raise TypeError
        return value
    except (OSError, TypeError, ValueError, plistlib.InvalidFileException):
        raise BuildError("app_metadata_invalid") from None


def set_application_version(
    info: dict, version: str | None, build_number: str | None = None
) -> None:
    """Keep local build identity separate from versions reserved for releases."""
    if (
        version is not None
        and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[.-][A-Za-z0-9]+)*", version)
        is None
    ):
        raise BuildError("application_version_invalid")
    if build_number is None:
        build_number = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    if re.fullmatch(r"[1-9][0-9]{0,19}", build_number) is None:
        raise BuildError("application_build_number_invalid")
    info["CFBundleShortVersionString"] = "0.0.0"
    info["CFBundleVersion"] = build_number
    info["StarkBuildChannel"] = "development"


def verify_runtime(runtime: Path) -> dict:
    """Check the immutable input before recording local app overlays separately."""
    try:
        manifest = json.loads((runtime / RUNTIME_MANIFEST).read_text())
        files = manifest["files_sha256"]
        if not isinstance(files, dict) or not files:
            raise ValueError
        for name, expected in files.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError
            path = runtime / relative
            if path.is_symlink() or not path.is_file() or sha256(path) != expected:
                raise ValueError
        allowed_extra = {RUNTIME_MANIFEST.as_posix()}
        archive_marker = runtime / ".archive-sha256"
        if archive_marker.exists():
            if not re.fullmatch(r"[0-9a-f]{64}\n?", archive_marker.read_text()):
                raise ValueError
            allowed_extra.add(".archive-sha256")
        for path in runtime.rglob("*"):
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise ValueError
            if path.is_file() and path.relative_to(runtime).as_posix() not in {
                *files,
                *allowed_extra,
            }:
                raise ValueError
        if not (runtime / PYTHON).is_file() or not (runtime / CONVERTER).is_dir():
            raise ValueError
        return manifest
    except (OSError, KeyError, TypeError, ValueError):
        raise BuildError("runtime_manifest_verification_failed") from None


def source_identity() -> dict:
    tracked = run(["git", "ls-files", "-z"], "source_files").split("\0")
    paths = [ROOT / name for name in tracked if name and (ROOT / name).is_file()]
    paths += list((ROOT / "apps/menu-bar").rglob("*"))
    paths += [
        path
        for path in (ROOT / "tools/converter/stark_convert").rglob("*")
        if "__pycache__" not in path.parts and path.suffix != ".pyc"
    ]
    paths += [
        ROOT / "scripts/build_menubar.py",
        ROOT / "tools/converter/stark_convert/desktop.py",
    ]
    files = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(set(paths))
        if path.is_file() and not path.is_symlink()
    }
    return {
        "revision": run(["git", "rev-parse", "HEAD"], "source_revision"),
        "dirty": bool(run(["git", "status", "--porcelain"], "source_status")),
        "files_sha256": files,
    }


def copy_runtime_for_application(
    runtime: Path, embedded: Path, manifest: dict
) -> list[str]:
    """Derive a CLI runtime from a combined install without embedding another app."""
    excluded = ["app"] if (runtime / "app").exists() else []

    def ignore(directory: str, _names: list[str]) -> list[str]:
        return excluded if Path(directory) == runtime else []

    shutil.copytree(runtime, embedded, symlinks=False, ignore=ignore)
    (embedded / ".archive-sha256").unlink(missing_ok=True)
    if excluded:
        derived = json.loads(json.dumps(manifest))
        (embedded / RUNTIME_MANIFEST).unlink()
        derived["files_sha256"] = {
            path.relative_to(embedded).as_posix(): sha256(path)
            for path in sorted(embedded.rglob("*"))
            if path.is_file()
        }
        derived["application_embedding"] = {
            "parent_manifest_sha256": sha256(runtime / RUNTIME_MANIFEST),
            "excluded_payload": excluded,
        }
        (embedded / RUNTIME_MANIFEST).write_text(
            json.dumps(derived, indent=2, sort_keys=True) + "\n"
        )
    return excluded


def verify_worker(worker: Path, kind: str = "generation") -> None:
    try:
        reply = subprocess.run(
            [str(worker), *(["--self-test"] if kind == "template" else [])],
            input=b"",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
        expected_reply = (
            {"status": "ok", "protocol": "stark-template-v1"}
            if kind == "template"
            else {"ready": False, "error": "invalid_arguments"}
        )
        if (
            reply.returncode != (0 if kind == "template" else 1)
            or json.loads(reply.stdout) != expected_reply
        ):
            raise ValueError
    except (OSError, ValueError, subprocess.TimeoutExpired):
        raise BuildError(f"{kind}_worker_self_check_failed") from None


def install_native_notices(embedded: Path, temporary: Path) -> dict:
    """Refresh notices for current workers instead of inheriting an old CLI set."""
    distribution = runpy.run_path(str(ROOT / "scripts/build_distribution.py"))
    cargo_name = shutil.which("cargo")
    cargo = Path(cargo_name) if cargo_name else Path.home() / ".cargo/bin/cargo"
    if not cargo.is_file():
        raise BuildError("cargo_required_for_current_native_notices")
    environment = distribution["clean_environment"]()
    environment["PATH"] = str(cargo.parent) + os.pathsep + environment.get("PATH", "")
    overlay = temporary / "native-notice-overlay"
    (overlay / "share/stark").mkdir(parents=True)
    try:
        distribution["dependency_licenses"](overlay, str(cargo), environment)
    except distribution["BuildError"]:
        raise BuildError("native_dependency_license_collection_failed") from None
    hashes = {
        path.relative_to(overlay).as_posix(): sha256(path)
        for path in sorted(overlay.rglob("*"))
        if path.is_file()
    }
    if not hashes:
        raise BuildError("native_dependency_notices_missing")
    shutil.copytree(overlay, embedded, dirs_exist_ok=True)
    return {
        "rust_version": distribution["RUST_VERSION"],
        "cargo_lock_sha256": sha256(ROOT / "Cargo.lock"),
        "scope": "current workers' locked normal dependency metadata and source adaptations; may include unused optional code",
        "files_sha256": hashes,
    }


def verify_worker_build_record(record_path: Path, worker: Path, error: str) -> None:
    try:
        record = json.loads(record_path.read_text())
        if record.get("worker_sha256") != sha256(worker):
            raise ValueError
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        raise BuildError(error) from None


def desktop_requirements(lockfile: Path) -> tuple[str, dict[str, str]]:
    """Select the chat renderer's complete dependency closure from uv.lock."""
    try:
        packages = tomllib.loads(lockfile.read_text())["package"]
        selected = {}
        for name in ("jinja2", "markupsafe"):
            matches = [package for package in packages if package.get("name") == name]
            if len(matches) != 1:
                raise ValueError
            package = matches[0]
            version = package["version"]
            if not isinstance(version, str) or not re.fullmatch(
                r"[0-9]+(?:\.[0-9]+)+", version
            ):
                raise ValueError
            dependencies = {
                dependency["name"] for dependency in package.get("dependencies", [])
            }
            if dependencies != ({"markupsafe"} if name == "jinja2" else set()):
                raise ValueError
            hashes = {wheel["hash"] for wheel in package["wheels"]}
            if not hashes or any(
                not re.fullmatch(r"sha256:[0-9a-f]{64}", value) for value in hashes
            ):
                raise ValueError
            selected[name] = (version, sorted(hashes))
        requirements = "".join(
            name
            + "=="
            + version
            + " "
            + " ".join("--hash=" + digest for digest in hashes)
            + "\n"
            for name, (version, hashes) in selected.items()
        )
        return requirements, {
            name: version for name, (version, _hashes) in selected.items()
        }
    except (OSError, KeyError, TypeError, ValueError):
        raise BuildError("desktop_dependency_lock_invalid") from None


def install_desktop_dependencies(embedded: Path, temporary: Path) -> dict:
    uv = shutil.which("uv")
    if uv is None:
        raise BuildError("uv_required_for_locked_desktop_dependencies")
    requirements, versions = desktop_requirements(ROOT / "tools/converter/uv.lock")
    requirement_file = temporary / "menu-bar-python-requirements.txt"
    requirement_file.write_text(requirements)
    overlay = temporary / "desktop-python-overlay"
    run(
        [
            uv,
            "--no-config",
            "pip",
            "install",
            "--python",
            str(embedded / PYTHON),
            "--target",
            str(overlay),
            "--no-deps",
            "--only-binary",
            ":all:",
            "--require-hashes",
            "--requirements",
            str(requirement_file),
        ],
        "locked_desktop_dependencies",
    )
    remove_python_target_lock(overlay)
    # Preserve wheel license/metadata files, while discarding local cache paths
    # and bytecode that is unrelated to the source-bound package artifact.
    for path in list(overlay.rglob("*")):
        if path.is_file() and (
            path.name in {"direct_url.json", "uv_cache.json"} or path.suffix == ".pyc"
        ):
            path.unlink()
    for path in sorted(overlay.rglob("__pycache__"), reverse=True):
        shutil.rmtree(path)
    site_packages = embedded / CONVERTER.parent
    added_files = {}
    for name, version in versions.items():
        metadata = overlay / f"{name}-{version}.dist-info"
        if not metadata.is_dir() or not any(
            "license" in path.name.lower() for path in metadata.rglob("*")
        ):
            raise BuildError("desktop_dependency_license_missing")
        for existing in [
            site_packages / name,
            *site_packages.glob(name + "-*.dist-info"),
        ]:
            if existing.is_dir():
                shutil.rmtree(existing)
    for path in overlay.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise BuildError("desktop_dependency_overlay_invalid")
        if path.is_file():
            destination = site_packages / path.relative_to(overlay)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            added_files[destination.relative_to(embedded).as_posix()] = sha256(
                destination
            )
    shutil.copy2(requirement_file, embedded.parent / requirement_file.name)
    return {
        "versions": versions,
        "requirements_sha256": sha256(requirement_file),
        "files_sha256": added_files,
    }


def remove_python_target_lock(overlay: Path) -> None:
    """Discard uv's empty target-directory lock after its installer has exited."""
    lock = overlay / ".lock"
    if not lock.exists() and not lock.is_symlink():
        return
    if lock.is_symlink() or not lock.is_file() or lock.stat().st_size != 0:
        raise BuildError("python_target_lock_invalid")
    lock.unlink()


def converter_snapshot(project: Path, destination: Path) -> dict:
    """Build from a fresh, source-bound package tree rather than stale build/ files."""
    try:
        configuration = tomllib.loads((project / "pyproject.toml").read_text())
        build = configuration["build-system"]
        if build != {
            "requires": ["setuptools==83.0.0"],
            "build-backend": "setuptools.build_meta",
        }:
            raise ValueError
        source_package = project / "stark_convert"
        if (source_package / "_build_inputs").exists():
            raise ValueError
        provenance_module = runpy.run_path(str(source_package / "_build_metadata.py"))
        source_inputs = (*provenance_module["SOURCE_INPUTS"], "uv.lock")
        readme = configuration["project"]["readme"]
        if not isinstance(readme, str):
            raise TypeError
        files = {}
        for path in sorted(source_package.rglob("*")):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise ValueError
            if path.is_file():
                relative = path.relative_to(project)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                files[relative.as_posix()] = sha256(target)
        for name in (*source_inputs, readme):
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError
            path = project / relative
            if path.is_symlink() or not path.is_file():
                raise ValueError
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            files[relative.as_posix()] = sha256(target)
        try:
            tooling = provenance_module["toolchain_provenance"](destination)
        except provenance_module["ConversionError"]:
            raise ValueError from None
        return {
            "project_name": configuration["project"]["name"],
            "version": configuration["project"]["version"],
            "source_inputs": list(source_inputs),
            "source_files_sha256": files,
            "toolchain_provenance": tooling,
            "build_system": build,
        }
    except (OSError, KeyError, TypeError, ValueError):
        raise BuildError("converter_snapshot_invalid") from None


def validate_converter_wheel(wheel: Path, source: dict) -> dict:
    """Reject stale modules, missing provenance capsules or stripped license metadata."""
    metadata = f"stark_convert-{source['version']}.dist-info"
    try:
        with zipfile.ZipFile(wheel) as archive:
            members = archive.namelist()
            if len(members) != len(set(members)):
                raise ValueError
            for entry in archive.infolist():
                relative = Path(entry.filename)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or stat.S_ISLNK(entry.external_attr >> 16)
                ):
                    raise ValueError
            expected_package = {
                name: digest
                for name, digest in source["source_files_sha256"].items()
                if name.startswith("stark_convert/")
            }
            expected_capsule = {
                "stark_convert/_build_inputs/" + name: source["source_files_sha256"][
                    name
                ]
                for name in source["source_inputs"]
            }
            actual_package = {
                name
                for name in members
                if name.startswith("stark_convert/") and not name.endswith("/")
            }
            if actual_package != {*expected_package, *expected_capsule}:
                raise ValueError
            for name, digest in {**expected_package, **expected_capsule}.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    raise ValueError
            license_path = metadata + "/licenses/LICENSE.lalamo"
            wheel_metadata = email.parser.Parser().parsestr(
                archive.read(metadata + "/METADATA").decode()
            )
            if (
                metadata + "/METADATA" not in members
                or metadata + "/WHEEL" not in members
                or metadata + "/RECORD" not in members
                or wheel_metadata["Name"] != source["project_name"]
                or wheel_metadata["Version"] != source["version"]
                or "LICENSE.lalamo" not in wheel_metadata.get_all("License-File", [])
                or hashlib.sha256(archive.read(license_path)).hexdigest()
                != source["source_files_sha256"]["LICENSE.lalamo"]
            ):
                raise ValueError
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        raise BuildError("converter_wheel_validation_failed") from None
    return {
        **source,
        "wheel_filename": wheel.name,
        "wheel_sha256": sha256(wheel),
    }


def install_converter(embedded: Path, temporary: Path) -> dict:
    uv = shutil.which("uv")
    if uv is None:
        raise BuildError("uv_required_for_current_converter")
    snapshot = temporary / "converter-source"
    source = converter_snapshot(ROOT / "tools/converter", snapshot)
    wheels = temporary / "converter-wheels"
    run(
        [
            uv,
            "--no-config",
            "build",
            str(snapshot),
            "--wheel",
            "--python",
            str(embedded / PYTHON),
            "--out-dir",
            str(wheels),
            "--no-build-logs",
        ],
        "current_converter_wheel",
    )
    artifacts = list(wheels.glob("*.whl"))
    if len(artifacts) != 1 or not artifacts[0].name.startswith("stark_convert-"):
        raise BuildError("converter_wheel_layout_invalid")
    record = validate_converter_wheel(artifacts[0], source)
    overlay = temporary / "converter-python-overlay"
    run(
        [
            uv,
            "--no-config",
            "pip",
            "install",
            "--python",
            str(embedded / PYTHON),
            "--target",
            str(overlay),
            "--no-deps",
            str(artifacts[0]),
        ],
        "current_converter_install",
    )
    remove_python_target_lock(overlay)
    for path in list(overlay.rglob("*")):
        if path.is_file() and (
            path.name in {"direct_url.json", "uv_cache.json"} or path.suffix == ".pyc"
        ):
            path.unlink()
    for path in sorted(overlay.rglob("__pycache__"), reverse=True):
        shutil.rmtree(path)
    # Console scripts contain staging paths; the app invokes Python modules.
    shutil.rmtree(overlay / "bin", ignore_errors=True)
    metadata_name = f"stark_convert-{source['version']}.dist-info"
    if {path.name for path in overlay.iterdir()} != {"stark_convert", metadata_name}:
        raise BuildError("converter_install_layout_invalid")
    for path in overlay.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise BuildError("converter_install_layout_invalid")
    site_packages = embedded / CONVERTER.parent
    for previous in [
        embedded / CONVERTER,
        *site_packages.glob("stark_convert-*.dist-info"),
    ]:
        if previous.is_dir():
            shutil.rmtree(previous)
    files = {}
    for path in overlay.rglob("*"):
        if path.is_file():
            target = site_packages / path.relative_to(overlay)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            files[target.relative_to(embedded).as_posix()] = sha256(target)
    return {**record, "files_sha256": files}


def make_icon(source: Path, resources: Path, temporary: Path) -> None:
    iconset = temporary / "Stark.iconset"
    iconset.mkdir()
    # sips fits the complete supplied artwork, preserving its aspect ratio.
    square = temporary / "Stark-square.png"
    run(
        ["/usr/bin/sips", "-s", "format", "png", str(source), "--out", str(square)],
        "icon_convert",
    )
    run(
        [
            "/usr/bin/sips",
            "--padToHeightWidth",
            "1274",
            "1274",
            "--padColor",
            "D0D0D0",
            str(square),
        ],
        "icon_pad",
    )
    for size in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            filename = f"icon_{size}x{size}" + ("@2x" if scale == 2 else "") + ".png"
            pixels = str(size * scale)
            run(
                [
                    "/usr/bin/sips",
                    "-z",
                    pixels,
                    pixels,
                    str(square),
                    "--out",
                    str(iconset / filename),
                ],
                "icon_resize",
            )
    run(
        [
            "/usr/bin/iconutil",
            "-c",
            "icns",
            str(iconset),
            "-o",
            str(resources / "Stark.icns"),
        ],
        "icon_package",
    )


def app_running(destination: Path) -> bool:
    executable = str(destination / "Contents/MacOS/Stark")
    listing = run(["/bin/ps", "-axo", "comm="], "running_apps")
    return executable in (line.strip() for line in listing.splitlines())


def stop_app(destination: Path) -> None:
    installer = runpy.run_path(str(ROOT / "scripts/install_app.py"))
    try:
        installer["stop_app"](destination)
    except installer["AppError"]:
        raise BuildError("app_termination_failed_previous_installation_kept") from None


def exchange_paths(first: Path, second: Path) -> None:
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    rename = library.renamex_np
    rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(os.fsencode(first), os.fsencode(second), 0x00000002) != 0:
        raise BuildError("app_atomic_exchange_failed")


def install_app(app: Path, destination: Path) -> None:
    """Stage a complete app, then atomically replace only a matching Stark app."""
    if destination.is_symlink():
        raise BuildError("install_destination_is_symlink")
    if destination.exists() and (
        not destination.is_dir()
        or read_plist(destination / "Contents/Info.plist").get("CFBundleIdentifier")
        != APP_ID
    ):
        raise BuildError("install_destination_is_unrelated")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".stark-install-", dir=destination.parent
    ) as temporary_name:
        stage = Path(temporary_name) / "Stark.app"
        shutil.copytree(app, stage, symlinks=False)
        run(
            ["/usr/bin/codesign", "--verify", "--deep", "--strict", str(stage)],
            "staged_signature",
        )
        if destination.exists():
            stop_app(destination)
            exchange_paths(stage, destination)
        else:
            stage.rename(destination)


def resolve_signing_identity(
    selected: str | None, keychain: Path | None, *, ad_hoc: bool = False
) -> dict[str, str] | None:
    """Require a unique Developer ID by default; never silently change to ad-hoc."""
    if ad_hoc:
        if selected is not None or keychain is not None:
            raise BuildError("ad_hoc_signing_options_conflict")
        return None
    try:
        return signing.resolve_identity(selected, keychain, signing.environment())
    except signing.SigningError as error:
        raise BuildError(str(error)) from None


def sign_application(
    app: Path, identity: dict[str, str] | None, keychain: Path | None
) -> dict:
    """Sign the staged app before publication, retaining source-bound build hashes."""
    manifest_path = app / "Contents/Resources/menu-bar-manifest.json"
    if identity is None:
        run(
            ["/usr/bin/codesign", "--force", "--deep", "--sign", "-", str(app)],
            "app_signature",
        )
    else:
        env = signing.environment()
        try:
            # Seal each embedded executable first. --deep signing can preserve
            # stale ad-hoc signatures in an otherwise Developer ID signed app.
            signing.sign_bundle(
                app / "Contents/Resources/runtime", identity, keychain, env
            )
            signing.refresh_signed_application_manifest(app, identity)
            arguments = [
                "/usr/bin/codesign",
                "--force",
                "--sign",
                identity["sha1"],
                "--options",
                "runtime",
                "--timestamp",
                "--identifier",
                APP_ID,
            ]
            if keychain is not None:
                arguments.extend(["--keychain", str(keychain)])
            run([*arguments, str(app)], "app_signature")
            signing.verify_signature(app, identity, env)
        except signing.SigningError as error:
            raise BuildError(str(error)) from None
    run(
        ["/usr/bin/codesign", "--verify", "--deep", "--strict", str(app)],
        "app_signature_verification",
    )
    return json.loads(manifest_path.read_bytes())


def build(options: argparse.Namespace) -> Path:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise BuildError("build_requires_native_apple_silicon_macos")
    keychain = getattr(options, "keychain", None)
    if keychain is not None:
        keychain = keychain.expanduser().resolve()
    identity = resolve_signing_identity(
        getattr(options, "sign_identity", None),
        keychain,
        ad_hoc=getattr(options, "ad_hoc", False),
    )
    runtime = options.runtime.expanduser().resolve()
    worker = options.worker.expanduser().resolve()
    score_worker = options.score_worker.expanduser().resolve()
    template_worker = options.template_worker.expanduser().resolve()
    worker_record_path = getattr(options, "worker_build_record", None)
    if worker_record_path is not None:
        worker_record_path = worker_record_path.expanduser().resolve()
        verify_worker_build_record(
            worker_record_path, worker, "worker_build_record_hash_mismatch"
        )
    score_worker_record_path = getattr(options, "score_worker_build_record", None)
    if score_worker_record_path is not None:
        score_worker_record_path = score_worker_record_path.expanduser().resolve()
        verify_worker_build_record(
            score_worker_record_path,
            score_worker,
            "score_worker_build_record_hash_mismatch",
        )
    runtime_manifest = verify_runtime(runtime)
    if (
        not worker.is_file()
        or "arm64"
        not in run(
            ["/usr/bin/lipo", "-archs", str(worker)], "worker_architecture"
        ).split()
    ):
        raise BuildError("arm64_generation_worker_missing")
    verify_worker(worker)
    if (
        not score_worker.is_file()
        or "arm64"
        not in run(
            ["/usr/bin/lipo", "-archs", str(score_worker)], "score_worker_architecture"
        ).split()
    ):
        raise BuildError("arm64_score_worker_missing")
    verify_worker(score_worker, "score")
    if (
        not template_worker.is_file()
        or "arm64"
        not in run(
            ["/usr/bin/lipo", "-archs", str(template_worker)],
            "template_worker_architecture",
        ).split()
    ):
        raise BuildError("arm64_template_worker_missing")
    verify_worker(template_worker, "template")
    source_directory = ROOT / "apps/menu-bar"
    sources = sorted((source_directory / "Sources").glob("*.swift"))
    if not sources or not (ROOT / "tools/converter/stark_convert/desktop.py").is_file():
        raise BuildError("menu_bar_sources_missing")
    artwork = source_directory / "Resources/Stark.jpg"
    template = source_directory / "Resources/MenuBarTemplate.png"
    if not artwork.is_file() or not template.is_file():
        raise BuildError("menu_bar_artwork_missing")
    output = options.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    app = output / "Stark.app"
    if app.is_symlink():
        raise BuildError("output_app_is_symlink")
    if (
        app.exists()
        and read_plist(app / "Contents/Info.plist").get("CFBundleIdentifier") != APP_ID
    ):
        raise BuildError("output_app_is_unrelated")
    with tempfile.TemporaryDirectory(
        prefix=".stark-build-", dir=output
    ) as temporary_name:
        temporary = Path(temporary_name)
        staged = temporary / "Stark.app"
        contents = staged / "Contents"
        executable = contents / "MacOS/Stark"
        executable.parent.mkdir(parents=True)
        resources = contents / "Resources"
        resources.mkdir()
        openwork_connector = install_openwork_connector(
            resources,
            source=getattr(options, "openwork_connector", None),
            allow_development=getattr(options, "allow_development_connector", False),
        )
        for helper in ("update_app.py", "install_app.py"):
            shutil.copy2(ROOT / "scripts" / helper, resources / helper)
        embedded = resources / "runtime"
        excluded_runtime_payload = copy_runtime_for_application(
            runtime, embedded, runtime_manifest
        )
        shutil.copy2(worker, embedded / "libexec/stark-generate")
        shutil.copy2(score_worker, embedded / "libexec/stark-score")
        shutil.copy2(template_worker, embedded / "libexec/stark-template")
        native_notices = install_native_notices(embedded, temporary)
        converter_package = install_converter(embedded, temporary)
        desktop_dependencies = install_desktop_dependencies(embedded, temporary)
        shutil.copy2(artwork, resources / "Stark.jpg")
        shutil.copy2(template, resources / "MenuBarTemplate.png")
        if worker_record_path is not None:
            shutil.copy2(worker_record_path, resources / "worker-build.json")
        if score_worker_record_path is not None:
            shutil.copy2(
                score_worker_record_path, resources / "score-worker-build.json"
            )
        make_icon(artwork, resources, temporary)
        info = read_plist(source_directory / "Info.plist")
        set_application_version(
            info,
            getattr(options, "version", None),
            getattr(options, "build_number", None),
        )
        with (contents / "Info.plist").open("wb") as stream:
            plistlib.dump(info, stream)
        run(
            [
                "/usr/bin/xcrun",
                "swiftc",
                "-O",
                "-target",
                "arm64-apple-macos" + MIN_MACOS,
                "-framework",
                "AppKit",
                "-framework",
                "SwiftUI",
                "-framework",
                "Metal",
                *map(str, sources),
                "-o",
                str(executable),
            ],
            "swift_build",
        )
        inspection = (
            "import inspect,json; import stark_convert.desktop,stark_convert.chat,stark_convert.decision,jinja2; "
            "import stark_convert.desktop_media,stark_convert.media,stark_convert.vision_runtime; "
            "import stark_convert.image_checks,stark_convert.vision_family,stark_convert.model_support,stark_convert.model_inventory,stark_convert.model_catalog,stark_convert.model_removal; "
            "assert callable(stark_convert.model_support.support_matrix); "
            "assert callable(stark_convert.model_inventory.model_display); "
            "assert callable(stark_convert.model_catalog.family_catalog); "
            "assert callable(stark_convert.model_removal.prepare_removal); "
            "from stark_convert import cli,progress; "
            "from stark_convert.provenance import toolchain_provenance; "
            "from stark_convert.template_runtime import IsolatedTemplate; "
            "assert IsolatedTemplate('{{ value }}').render(value='synthetic control') == 'synthetic control'; "
            "assert 'progress' in inspect.signature(cli.convert).parameters; "
            "assert callable(progress.FDProgressSink); "
            "print(json.dumps(toolchain_provenance(),sort_keys=True))"
        )
        installed_tooling = json.loads(
            run(
                [str(embedded / PYTHON), "-I", "-B", "-c", inspection],
                "embedded_python_imports",
            )
        )
        if installed_tooling != converter_package["toolchain_provenance"]:
            raise BuildError("installed_converter_provenance_mismatch")
        response = run(
            [
                str(embedded / PYTHON),
                "-I",
                "-B",
                "-m",
                "stark_convert.desktop",
                "--help",
            ],
            "desktop_entrypoint",
        )
        if (
            "--worker" not in response
            or "--score-worker" not in response
            or "--data-dir" not in response
            or "--attachment-root" not in response
        ):
            raise BuildError("desktop_entrypoint_contract_failed")
        manifest = {
            "schema_version": 1,
            "application_id": APP_ID,
            "minimum_macos": MIN_MACOS,
            "distribution_version": getattr(options, "version", None),
            "application_version": info["CFBundleShortVersionString"],
            "build_number": info["CFBundleVersion"],
            "build_channel": info["StarkBuildChannel"],
            "source": source_identity(),
            "base_runtime": {
                "manifest_sha256": sha256(runtime / RUNTIME_MANIFEST),
                "embedded_manifest_sha256": sha256(embedded / RUNTIME_MANIFEST),
                "excluded_payload": excluded_runtime_payload,
                "version": runtime_manifest.get("version"),
                "source": runtime_manifest.get("source", {}),
                "note": "The embedded runtime manifest identifies the CLI snapshot; a combined parent app is excluded. App overlays are recorded here.",
            },
            "overlays_sha256": {
                "libexec/stark-generate": sha256(embedded / "libexec/stark-generate"),
                "libexec/stark-score": sha256(embedded / "libexec/stark-score"),
                "libexec/stark-template": sha256(embedded / "libexec/stark-template"),
                **native_notices["files_sha256"],
                **converter_package["files_sha256"],
                **desktop_dependencies["files_sha256"],
            },
            "converter_package": {
                key: value
                for key, value in converter_package.items()
                if key != "files_sha256"
            },
            "native_license_notices": {
                key: value
                for key, value in native_notices.items()
                if key != "files_sha256"
            },
            "desktop_python_dependencies": {
                "versions": desktop_dependencies["versions"],
                "converter_lock_sha256": sha256(ROOT / "tools/converter/uv.lock"),
                "requirements_sha256": desktop_dependencies["requirements_sha256"],
                "requirement_file": "menu-bar-python-requirements.txt",
            },
            "openwork_connector": {
                key: value
                for key, value in openwork_connector.items()
                if key != "resources_sha256"
            },
            "resources_sha256": {
                **{
                    name: sha256(resources / name)
                    for name in (
                        "Stark.jpg",
                        "MenuBarTemplate.png",
                        "Stark.icns",
                        "update_app.py",
                        "install_app.py",
                    )
                },
                **openwork_connector["resources_sha256"],
            },
            "executable_before_signing_sha256": sha256(executable),
            "validation": {
                "base_runtime_hashes": "passed",
                "worker_invalid_arguments": "passed",
                "score_worker_invalid_arguments": "passed",
                "current_native_dependency_and_adaptation_notices": "passed",
                "python_imports": "passed",
                "converter_progress_api": "passed",
                "converter_toolchain_provenance": "passed",
                "desktop_entrypoint": "passed",
                "image_attachment_imports_and_entrypoint": "passed",
                "openwork_connector_resources": "passed",
                "openwork_connector_lock": "development_override"
                if openwork_connector["development_override"]
                else "passed",
                "model_loaded": False,
            },
            "signature": "local ad-hoc; no Developer ID signature or notarization",
        }
        if worker_record_path is not None:
            manifest["worker_build_record_sha256"] = sha256(
                resources / "worker-build.json"
            )
        if score_worker_record_path is not None:
            manifest["score_worker_build_record_sha256"] = sha256(
                resources / "score-worker-build.json"
            )
        (resources / "menu-bar-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        manifest = sign_application(staged, identity, keychain)
        if app.exists():
            stop_app(app)
            exchange_paths(staged, app)
        else:
            staged.rename(app)
    external_manifest = {
        **manifest,
        "signed_executable_sha256": sha256(app / "Contents/MacOS/Stark"),
        "embedded_manifest_sha256": sha256(
            app / "Contents/Resources/menu-bar-manifest.json"
        ),
        "signature_verification": "passed",
    }
    (output / "Stark.app.manifest.json").write_text(
        json.dumps(external_manifest, indent=2, sort_keys=True) + "\n"
    )
    if options.install:
        destination = options.install_dir.expanduser().resolve() / "Stark.app"
        install_app(app, destination)
    else:
        destination = app
    if options.open:
        run(["/usr/bin/open", str(destination)], "open_app")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path.home() / ".local/lib/stark/current",
        help="Existing standalone CLI bundle (copied without changing it)",
    )
    parser.add_argument(
        "--worker",
        type=Path,
        default=ROOT / "target/release/stark-generate",
        help="Generation worker to embed",
    )
    parser.add_argument(
        "--score-worker",
        type=Path,
        default=ROOT / "target/release/stark-score",
        help="Probability scoring worker to embed",
    )
    parser.add_argument(
        "--template-worker",
        type=Path,
        default=ROOT / "target/release/stark-template",
        help="Native chat-template evaluator to embed",
    )
    parser.add_argument(
        "--worker-build-record",
        type=Path,
        help="Optional source-bound build record for a development worker",
    )
    parser.add_argument(
        "--score-worker-build-record",
        type=Path,
        help="Optional source-bound build record for a development scoring worker",
    )
    parser.add_argument(
        "--openwork-connector",
        type=Path,
        help="Local first-party connector source; pinned hashes are verified",
    )
    parser.add_argument(
        "--allow-development-connector",
        action="store_true",
        help="Allow local connector source while its lock is explicitly unpinned; records a development override",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/menu-bar")
    signing_options = parser.add_mutually_exclusive_group()
    signing_options.add_argument(
        "--sign-identity",
        help="Developer ID Application name or certificate SHA-1 (default: the unique available Developer ID)",
    )
    signing_options.add_argument(
        "--ad-hoc",
        action="store_true",
        help="Explicitly use an ad-hoc signature for isolated development; rebuilds may require Keychain approval",
    )
    parser.add_argument(
        "--keychain",
        type=Path,
        help="Keychain containing the Developer ID identity (default: user search list)",
    )
    parser.add_argument(
        "--version",
        help="archive identifier recorded in build provenance; app stays a development build",
    )
    parser.add_argument(
        "--build-number",
        help="positive numeric local build identifier (default: UTC timestamp with microseconds)",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install into ~/Applications, or --install-dir",
    )
    parser.add_argument(
        "--install-dir", type=Path, default=Path.home() / "Applications"
    )
    parser.add_argument(
        "--open", action="store_true", help="Open the built or installed app"
    )
    options = parser.parse_args(argv)
    try:
        destination = build(options)
    except (BuildError, OSError, ValueError):
        error = sys.exc_info()[1]
        message = (
            str(error)
            if isinstance(error, BuildError)
            else "build_filesystem_operation_failed"
        )
        print("Stark menu bar: " + message, file=sys.stderr)
        return 1
    print(str(destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
