#!/usr/bin/env python3
"""Activate a combined Stark CLI/app installation using its private Python."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import plistlib
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

APP_ID = "com.starkengine.menubar"
STATE_NAME = "managed-app.json"
OWNER_NAME = ".Stark.app.installation.json"
COMMANDS = {
    "stark": "../lib/stark/current/bin/stark",
    "stark-convert": "../lib/stark/current/bin/stark",
    "stark-score": "../lib/stark/current/libexec/stark-score",
    "stark-generate": "../lib/stark/current/libexec/stark-generate",
}


class AppError(Exception):
    """A content-free installation failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AppError(message)


def absolute(path: Path) -> Path:
    require(path.is_absolute(), "installation paths must be absolute")
    require(not any(ord(c) < 32 for c in str(path)), "installation path is invalid")
    return path.resolve()


def command(arguments: list[str | Path], message: str) -> bytes:
    try:
        result = subprocess.run(
            arguments,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise AppError(message) from error
    require(result.returncode == 0, message)
    return result.stdout


def verify_signature(app: Path) -> None:
    command(
        ["/usr/bin/codesign", "--verify", "--deep", "--strict", app],
        "app signature verification failed",
    )


def running(app: Path) -> bool:
    output = command(
        ["/bin/ps", "-axo", "comm="], "cannot inspect running applications"
    )
    executable = str(app / "Contents/MacOS/Stark")
    return executable in (
        line.strip() for line in output.decode(errors="replace").splitlines()
    )


def validate(app: Path) -> str:
    require(not app.is_symlink() and app.is_dir(), "app must be a physical directory")
    try:
        with (app / "Contents/Info.plist").open("rb") as stream:
            info = plistlib.load(stream)
    except (OSError, ValueError, plistlib.InvalidFileException) as error:
        raise AppError("app metadata is invalid") from error
    require(
        isinstance(info, dict)
        and info.get("CFBundleIdentifier") == APP_ID
        and info.get("CFBundleExecutable") == "Stark",
        "refusing to replace an unrelated application",
    )
    executable = app / "Contents/MacOS/Stark"
    require(
        executable.is_file() and os.access(executable, os.X_OK),
        "app executable is missing",
    )
    digest = hashlib.sha256()
    for entry in sorted(app.rglob("*")):
        mode = entry.lstat().st_mode
        require(
            stat.S_ISDIR(mode) or stat.S_ISREG(mode),
            "app contains unsupported filesystem entries",
        )
        digest.update(entry.relative_to(app).as_posix().encode() + b"\0")
        digest.update(str(stat.S_IMODE(mode)).encode() + b"\0")
        if stat.S_ISREG(mode):
            with entry.open("rb") as stream:
                digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


def rename(first: Path, second: Path, *, exchange: bool = False) -> None:
    """Use Darwin's atomic exchange or exclusive rename; never overwrite a race."""
    require(sys.platform == "darwin", "atomic app installation requires macOS")
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    operation = library.renamex_np
    operation.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    operation.restype = ctypes.c_int
    flags = 0x00000002 if exchange else 0x00000004
    require(
        operation(os.fsencode(first), os.fsencode(second), flags) == 0,
        "atomic app replacement failed",
    )


@contextlib.contextmanager
def mutation():
    """Defer termination until a filesystem change and its rollback flag agree."""
    previous = signal.pthread_sigmask(
        signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    )
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def read_record(path: Path) -> dict | None:
    require(not path.is_symlink(), "managed app record must not be a symlink")
    if not path.exists():
        return None
    require(
        path.is_file() and path.stat().st_size <= 4096, "managed app record is invalid"
    )
    try:
        record = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise AppError("managed app record is invalid") from error
    require(
        isinstance(record, dict)
        and set(record)
        == {"schema_version", "installation_id", "path", "app_id", "sha256"}
        and record["schema_version"] == 1
        and record["app_id"] == APP_ID
        and isinstance(record["installation_id"], str)
        and re.fullmatch(r"[0-9a-f]{32}", record["installation_id"]) is not None
        and isinstance(record["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None
        and isinstance(record["path"], str)
        and Path(record["path"]).is_absolute()
        and Path(record["path"]).name == "Stark.app",
        "managed app record is invalid",
    )
    return record


def write_bytes(path: Path, content: bytes | None) -> None:
    require(not path.is_symlink(), "managed app record must not be a symlink")
    if content is None:
        path.unlink(missing_ok=True)
        return
    descriptor, name = tempfile.mkstemp(prefix=".stark-record-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def record_bytes(record: dict) -> bytes:
    return (json.dumps(record, sort_keys=True) + "\n").encode()


def paths(prefix: Path, app_dir: Path) -> tuple[Path, Path, Path]:
    prefix, app_dir = absolute(prefix), absolute(app_dir)
    target = app_dir / "Stark.app"
    require(
        prefix != Path("/") and app_dir != Path("/"), "installation root is invalid"
    )
    require(
        not target.is_relative_to(prefix) and not prefix.is_relative_to(target),
        "app and CLI installation paths must not overlap",
    )
    require(not target.is_symlink(), "installed app path must not be a symlink")
    return prefix, app_dir, target


def owned_target(state: dict | None, owner: dict | None, target: Path) -> None:
    if state is not None:
        require(
            state["path"] == str(target),
            "app directory changed; uninstall the previous app first",
        )
        require(owner == state, "app ownership record does not match this installation")
    else:
        require(owner is None, "app belongs to another Stark installation")
    if target.exists():
        current = validate(target)
        if state is not None:
            require(current == state["sha256"], "installed app changed; preserving it")
        require(not running(target), "installed app is running; quit Stark first")


def switch_current(base: Path, destination: str | None) -> None:
    current = base / "current"
    if destination is None:
        current.unlink(missing_ok=True)
        return
    temporary = base / (".current-" + uuid.uuid4().hex)
    try:
        temporary.symlink_to(destination)
        os.replace(temporary, current)
    finally:
        temporary.unlink(missing_ok=True)


def install(bundle: Path, prefix: Path, app_dir: Path) -> None:
    prefix, app_dir, target = paths(prefix, app_dir)
    bundle = absolute(bundle)
    base = prefix / "lib/stark"
    require(
        bundle.parent == base / "releases" and not bundle.is_symlink(),
        "CLI release path is invalid",
    )
    source = bundle / "app/Stark.app"
    desired_digest = validate(source)
    verify_signature(source)
    state_path, owner_path = base / STATE_NAME, app_dir / OWNER_NAME
    state = read_record(state_path)
    app_dir.mkdir(parents=True, exist_ok=True)
    lock = app_dir / ".Stark.app.install-lock"
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError as error:
        raise AppError("another app installation is in progress") from error
    temporary: Path | None = None
    preserve_backup = False
    try:
        owner = read_record(owner_path)
        owned_target(state, owner, target)
        previous_digest = validate(target) if target.exists() else None
        previous_current = None
        current = base / "current"
        if current.exists() or current.is_symlink():
            require(current.is_symlink(), "current must be a managed version link")
            previous_current = os.readlink(current)
            require(
                re.fullmatch(
                    r"releases/[0-9]+\.[0-9]+\.[0-9]+[0-9A-Za-z.-]*", previous_current
                )
                is not None,
                "current version link is invalid",
            )
        for name, destination in COMMANDS.items():
            link = prefix / "bin" / name
            if link.exists() or link.is_symlink():
                require(
                    link.is_symlink() and os.readlink(link) == destination,
                    "refusing to replace another command installation",
                )
        temporary = Path(tempfile.mkdtemp(prefix=".stark-install-", dir=app_dir))
        staged = temporary / "Stark.app"
        shutil.copytree(source, staged)
        require(validate(staged) == desired_digest, "staged app differs from release")
        verify_signature(staged)
        previous_state = state_path.read_bytes() if state is not None else None
        previous_owner = owner_path.read_bytes() if owner is not None else None
        installed = False
        activated = False
        created_links: list[Path] = []
        record = {
            "schema_version": 1,
            "installation_id": state["installation_id"] if state else uuid.uuid4().hex,
            "path": str(target),
            "app_id": APP_ID,
            "sha256": desired_digest,
        }
        try:
            owned_target(state, read_record(owner_path), target)
            require(
                (validate(target) if target.exists() else None) == previous_digest,
                "installed app changed during staging",
            )
            with mutation():
                rename(staged, target, exchange=previous_digest is not None)
                installed = True
            for name, destination in COMMANDS.items():
                link = prefix / "bin" / name
                if not link.is_symlink():
                    with mutation():
                        link.symlink_to(destination)
                        created_links.append(link)
            with mutation():
                # Restoring the old link is safe even if replacement fails.
                activated = True
                switch_current(base, "releases/" + bundle.name)
            write_bytes(state_path, record_bytes(record))
            write_bytes(owner_path, record_bytes(record))
        except BaseException:
            rollback = None
            try:
                rollback = signal.pthread_sigmask(
                    signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
                )
                if activated:
                    switch_current(base, previous_current)
                for link in reversed(created_links):
                    link.unlink()
                if installed:
                    require(
                        validate(target) == desired_digest,
                        "app changed during rollback",
                    )
                    if previous_digest is not None:
                        rename(staged, target, exchange=True)
                    else:
                        rename(target, staged)
                write_bytes(state_path, previous_state)
                write_bytes(owner_path, previous_owner)
            except BaseException as error:
                preserve_backup = True
                raise AppError(
                    "installation rollback failed; private app backup was preserved"
                ) from error
            finally:
                if rollback is not None:
                    signal.pthread_sigmask(signal.SIG_SETMASK, rollback)
            raise
    finally:
        if temporary is not None and not preserve_backup:
            shutil.rmtree(temporary, ignore_errors=True)
        with contextlib.suppress(OSError):
            lock.rmdir()


def uninstall(prefix: Path) -> None:
    prefix = absolute(prefix)
    base = prefix / "lib/stark"
    state_path = base / STATE_NAME
    state = read_record(state_path)
    if state is None:
        return
    target = Path(state["path"])
    _, app_dir, target = paths(prefix, target.parent)
    owner_path = app_dir / OWNER_NAME
    lock = app_dir / ".Stark.app.install-lock"
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError as error:
        raise AppError("another app installation is in progress") from error
    temporary: Path | None = None
    preserve_backup = False
    try:
        owner = read_record(owner_path)
        owned_target(state, owner, target)
        temporary = Path(tempfile.mkdtemp(prefix=".stark-uninstall-", dir=app_dir))
        removed = temporary / "Stark.app"
        moved = False
        try:
            if target.exists():
                with mutation():
                    rename(target, removed)
                    moved = True
            owner_path.unlink()
            state_path.unlink()
        except BaseException:
            rollback = None
            try:
                rollback = signal.pthread_sigmask(
                    signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
                )
                if moved:
                    rename(removed, target)
                write_bytes(owner_path, record_bytes(owner))
                write_bytes(state_path, record_bytes(state))
            except BaseException as error:
                preserve_backup = True
                raise AppError(
                    "uninstall rollback failed; private app backup was preserved"
                ) from error
            finally:
                if rollback is not None:
                    signal.pthread_sigmask(signal.SIG_SETMASK, rollback)
            raise
    finally:
        if temporary is not None and not preserve_backup:
            shutil.rmtree(temporary, ignore_errors=True)
        with contextlib.suppress(OSError):
            lock.rmdir()


def interrupted(_number: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    install_parser = commands.add_parser("install")
    install_parser.add_argument("--bundle", type=Path, required=True)
    install_parser.add_argument("--prefix", type=Path, required=True)
    install_parser.add_argument("--app-dir", type=Path, required=True)
    uninstall_parser = commands.add_parser("uninstall")
    uninstall_parser.add_argument("--prefix", type=Path, required=True)
    options = parser.parse_args()
    for number in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, interrupted)
    try:
        if options.operation == "install":
            install(options.bundle, options.prefix, options.app_dir)
        else:
            uninstall(options.prefix)
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - content-free CLI boundary
        message = (
            str(error) if isinstance(error, AppError) else "app installation failed"
        )
        print("stark install: " + message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
