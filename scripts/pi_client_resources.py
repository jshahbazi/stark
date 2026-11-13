"""Stage the public, hash-pinned Pi client resources into Stark.app."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from pathlib import Path

REPOSITORY = "https://github.com/jshahbazi/stark-agent-clients"
FILES = (
    "package.json",
    "LICENSE",
    "src/config.mjs",
    "src/guarded-fetch.mjs",
    "extensions/stark-stream-guard.mjs",
    "src/desktop-setup.mjs",
    "src/desktop-launch.mjs",
    "src/default-profile.mjs",
)
MAX_FILE_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 1024 * 1024
LOCK_LIMIT_BYTES = 16 * 1024
LOCK_PATH = (
    Path(__file__).resolve().parent.parent
    / "apps/menu-bar/Resources/PiClient.lock.json"
)


class PiClientResourceError(Exception):
    """A fixed, content-free resource packaging error."""


def _fail(code: str) -> None:
    raise PiClientResourceError(code)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def read_lock(path: Path = LOCK_PATH) -> dict:
    """Read and validate the complete immutable public-resource pin."""
    descriptor = -1
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 1
            or before.st_size > LOCK_LIMIT_BYTES
        ):
            raise ValueError
        data = bytearray()
        while len(data) <= LOCK_LIMIT_BYTES:
            block = os.read(descriptor, min(4096, LOCK_LIMIT_BYTES + 1 - len(data)))
            if not block:
                break
            data.extend(block)
        after = os.fstat(descriptor)
        if len(data) != before.st_size or (
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
        ):
            raise ValueError
        lock = json.loads(data, object_pairs_hook=_unique_object)
        if not isinstance(lock, dict) or set(lock) != {
            "schema_version",
            "repository",
            "revision",
            "files_sha256",
        }:
            raise ValueError
        files = lock["files_sha256"]
        if (
            type(lock["schema_version"]) is not int
            or lock["schema_version"] != 1
            or lock["repository"] != REPOSITORY
            or not isinstance(lock["revision"], str)
            or not re.fullmatch(r"[0-9a-f]{40}", lock["revision"])
            or not isinstance(files, dict)
            or set(files) != set(FILES)
            or any(
                not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                for digest in files.values()
            )
        ):
            raise ValueError
        return {**lock, "lock_sha256": hashlib.sha256(data).hexdigest()}
    except (OSError, ValueError, TypeError, UnicodeError):
        _fail("pi_client_lock_invalid")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _open_source_root(source: Path) -> int:
    try:
        absolute = source.expanduser().absolute()
        if ".." in absolute.parts or absolute.is_symlink():
            raise ValueError
        descriptor = os.open(
            absolute, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o022:
            os.close(descriptor)
            raise ValueError
        return descriptor
    except (OSError, ValueError):
        _fail("pi_client_source_invalid")


def _read_source_file(root_descriptor: int, name: str) -> bytes:
    descriptor = -1
    parent = -1
    try:
        parts = name.split("/")
        if name not in FILES or any(part in ("", ".", "..") for part in parts):
            raise ValueError
        parent = os.dup(root_descriptor)
        for component in parts[:-1]:
            child = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent,
            )
            os.close(parent)
            parent = child
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
            dir_fd=parent,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & 0o133
            or before.st_size < 1
            or before.st_size > MAX_FILE_BYTES
        ):
            raise ValueError
        data = bytearray()
        while len(data) <= MAX_FILE_BYTES:
            block = os.read(descriptor, min(8192, MAX_FILE_BYTES + 1 - len(data)))
            if not block:
                break
            data.extend(block)
        after = os.fstat(descriptor)
        if len(data) != before.st_size or (
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
        ):
            raise ValueError
        return bytes(data)
    except (OSError, ValueError):
        _fail("pi_client_source_invalid")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if parent >= 0:
            os.close(parent)


def _write_payload(resources: Path, payload: dict[str, bytes]) -> None:
    target = resources / "PiClient"
    if (
        resources.is_symlink()
        or not resources.is_dir()
        or target.exists()
        or target.is_symlink()
    ):
        _fail("pi_client_destination_invalid")
    target_descriptor = -1
    target_identity = None
    try:
        target.mkdir(mode=0o755)
        target_descriptor = os.open(
            target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        target_identity = os.fstat(target_descriptor)
        for name, data in payload.items():
            parts = name.split("/")
            parent = os.dup(target_descriptor)
            try:
                for component in parts[:-1]:
                    try:
                        os.mkdir(component, mode=0o755, dir_fd=parent)
                    except FileExistsError:
                        pass
                    child = os.open(
                        component,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=parent,
                    )
                    os.close(parent)
                    parent = child
                descriptor = os.open(
                    parts[-1],
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o644,
                    dir_fd=parent,
                )
                try:
                    with os.fdopen(descriptor, "wb", closefd=False) as stream:
                        stream.write(data)
                        stream.flush()
                        os.fsync(descriptor)
                    os.fchmod(descriptor, 0o644)
                finally:
                    os.close(descriptor)
            finally:
                os.close(parent)
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
        _fail("pi_client_copy_failed")
    finally:
        if target_descriptor >= 0:
            os.close(target_descriptor)


def install_pi_client(
    resources: Path,
    *,
    source: Path | None,
    lockfile: Path = LOCK_PATH,
) -> dict:
    """Copy only the exact public file closure authorized by the lock."""
    lock = read_lock(lockfile)
    snapshot = source is None
    if snapshot:
        source = Path(lockfile).parent / "PiClient"
        if not source.exists() and not source.is_symlink():
            _fail("pi_client_snapshot_missing")
    root = _open_source_root(source)
    try:
        payload = {name: _read_source_file(root, name) for name in FILES}
    finally:
        os.close(root)
    if sum(map(len, payload.values())) > MAX_TOTAL_BYTES:
        _fail("pi_client_source_invalid")
    try:
        for data in payload.values():
            if b"\0" in data:
                raise ValueError
            data.decode("utf-8")
        package = json.loads(payload["package.json"])
        if (
            not isinstance(package, dict)
            or package.get("name") != "stark-agent-clients"
            or package.get("type") != "module"
            or package.get("private") is not True
        ):
            raise ValueError
    except (ValueError, TypeError, UnicodeError):
        _fail("pi_client_package_invalid")
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    if hashes != lock["files_sha256"]:
        _fail("pi_client_hash_mismatch")
    _write_payload(resources, payload)
    return {
        "repository": REPOSITORY,
        "revision": lock["revision"],
        "lock_sha256": lock["lock_sha256"],
        "files_sha256": hashes,
        "resources_sha256": {
            f"PiClient/{name}": digest for name, digest in hashes.items()
        },
    }
