#!/usr/bin/env python3
"""Apply a verified app update or install the app from its disk image.

The running app copies this helper and install_app.py into a private workspace.
Python runs from the verified candidate, which is retained until the helper exits.
Only fixed status messages are persisted; runtime/model data is never inspected.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import plistlib
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import install_app as installer

TEAM_ID = "S5BT2PRN5B"
SYSTEM_APPLICATIONS = Path("/Applications")


def physical_path(path: Path, message: str) -> Path:
    installer.require(
        path.is_absolute()
        and not any(ord(character) < 32 for character in str(path))
        and not any(part.is_symlink() for part in (path, *path.parents)),
        message,
    )
    return path.resolve()


def release_version(candidate: Path) -> str:
    try:
        info = plistlib.loads((candidate / "Contents/Info.plist").read_bytes())
    except (OSError, ValueError, plistlib.InvalidFileException) as error:
        raise installer.AppError("app metadata is invalid") from error
    version = info.get("CFBundleShortVersionString") if isinstance(info, dict) else None
    installer.require(
        isinstance(version, str)
        and re.fullmatch(
            r"(?:0|[1-9][0-9]{0,8})(?:\.(?:0|[1-9][0-9]{0,8})){2}", version
        )
        is not None,
        "app release version is invalid",
    )
    return version


def application_processes() -> list[list[str]]:
    output = installer.command(
        ["/bin/ps", "-axo", "uid=,comm="], "cannot inspect running applications"
    )
    return [
        fields
        for line in output.decode(errors="replace").splitlines()
        if len(fields := line.strip().split(maxsplit=1)) == 2
    ]


def running_installations() -> list[Path]:
    """Find same-user processes with the exact physical Stark app executable."""
    applications = set()
    for uid, name in application_processes():
        if uid != str(os.getuid()):
            continue
        executable = Path(name)
        if not executable.is_absolute() or executable.parts[-4:] != (
            "Stark.app",
            "Contents",
            "MacOS",
            "Stark",
        ):
            continue
        app = physical_path(executable.parents[2], "running app path is invalid")
        installer.validate(app)
        installer.verify_signature(app)
        applications.add(app)
    installer.require(
        len(applications) <= 1,
        "multiple Stark copies are running; quit them and retry installation",
    )
    return sorted(applications)


def require_same_user_session(target: Path) -> None:
    executable = str(target / "Contents/MacOS/Stark")
    installer.require(
        not any(
            uid != str(os.getuid()) and name == executable
            for uid, name in application_processes()
        ),
        "Stark is running in another user session; quit it before installation",
    )


def wait_for_launch(target: Path, timeout: float = 10) -> None:
    executable = str(target / "Contents/MacOS/Stark")
    deadline = time.monotonic() + timeout
    while True:
        if any(
            uid == str(os.getuid()) and name == executable
            for uid, name in application_processes()
        ):
            return
        installer.require(
            time.monotonic() < deadline,
            "installed app did not start; previous installation was kept",
        )
        time.sleep(0.1)


def writable_target(target: Path) -> None:
    physical_path(target, "installation destination is invalid")
    installer.require(target.name == "Stark.app", "installation destination is invalid")
    installer.require(
        target.parent.is_dir()
        and os.access(target.parent, os.W_OK | os.X_OK)
        and (not target.exists() or os.access(target, os.W_OK | os.X_OK)),
        "installation destination is not writable",
    )


def dmg_target() -> Path:
    applications = running_installations()
    if applications:
        target = applications[0]
    else:
        user_applications = Path.home() / "Applications"
        defaults = (SYSTEM_APPLICATIONS / "Stark.app", user_applications / "Stark.app")
        target = next(
            (path for path in defaults if path.exists() or path.is_symlink()), None
        )
        if target is None:
            system = physical_path(
                SYSTEM_APPLICATIONS, "installation destination is invalid"
            )
            if system.is_dir() and os.access(system, os.W_OK | os.X_OK):
                target = system / "Stark.app"
            else:
                physical_path(user_applications, "installation destination is invalid")
                user_applications.mkdir(parents=True, exist_ok=True)
                target = user_applications / "Stark.app"
    target = physical_path(target, "installation destination is invalid")
    writable_target(target)
    if target.exists():
        installer.validate(target)
        installer.verify_signature(target)
    return target


def verify_release(app: Path, version: str) -> str:
    digest = installer.validate(app)
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    installer.require(
        info.get("CFBundleShortVersionString") == version,
        "update version does not match release",
    )
    requirement = (
        '=anchor apple generic and identifier "com.starkengine.menubar" and '
        "certificate leaf[field.1.2.840.113635.100.6.1.13] exists and "
        f'certificate leaf[subject.OU] = "{TEAM_ID}"'
    )
    installer.command(
        ["/usr/bin/codesign", "--verify", "--deep", "--strict", "-R", requirement, app],
        "update signature verification failed",
    )
    installer.command(
        ["/usr/sbin/spctl", "--assess", "--type", "execute", app],
        "update did not pass Gatekeeper",
    )
    return digest


def ownership(target: Path) -> tuple[Path, dict, Path] | None:
    owner_path = target.parent / installer.OWNER_NAME
    owner = installer.read_record(owner_path)
    if owner is None:
        return None
    # Schema 2 names its CLI state. Older default-prefix installations can be
    # identified by exact equality of both records, without searching prefixes.
    state_path = Path(
        owner.get("state_path", Path.home() / ".local/lib/stark/managed-app.json")
    )
    state = installer.read_record(state_path)
    installer.require(
        state == owner and owner["path"] == str(target),
        "cannot locate this app's CLI installation; upgrade once with its installer",
    )
    installer.require(
        installer.validate(target) == owner["sha256"],
        "installed app changed; preserving it",
    )
    return state_path, owner, owner_path


def wait_for_exit(pid: int, target: Path, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        if not alive and not installer.running(target):
            return
        installer.require(
            time.monotonic() < deadline, "Stark did not exit; update was not installed"
        )
        time.sleep(0.1)


def apply_update(
    candidate: Path,
    target: Path,
    version: str,
    pid: int | None,
    workspace: Path,
) -> None:
    from_dmg = pid is None
    installer.require(
        target.is_absolute() and target.name == "Stark.app" and not target.is_symlink(),
        "update destination is invalid",
    )
    if from_dmg:
        candidate = physical_path(candidate, "installation source is invalid")
        target = physical_path(target, "installation destination is invalid")
        installer.require(
            not candidate.is_relative_to(target)
            and not target.is_relative_to(candidate),
            "installation source and destination must not overlap",
        )
        writable_target(target)
    previous_digest = installer.validate(target) if target.exists() else None
    installer.require(
        from_dmg or previous_digest is not None, "installed app is missing"
    )
    desired_digest = verify_release(candidate, version)
    lock = target.parent / ".Stark.app.install-lock"
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError as error:
        raise installer.AppError("another app installation is in progress") from error
    stage_directory = None
    preserve_backup = False
    try:
        owner = ownership(target)
        stage_directory = Path(
            tempfile.mkdtemp(prefix=".stark-update-", dir=target.parent)
        )
        staged = stage_directory / "Stark.app"
        shutil.copytree(candidate, staged)
        installer.require(
            verify_release(staged, version) == desired_digest, "staged update changed"
        )
        # The app quits only after staging, signature checks and ownership checks.
        if from_dmg:
            writable_target(target)
            installer.require(
                (installer.validate(target) if target.exists() else None)
                == previous_digest,
                "installed app changed during update",
            )
            installer.require(
                ownership(target) == owner, "app ownership changed during update"
            )
            applications = running_installations()
            installer.require(
                not applications or applications == [target],
                "running Stark installation changed; retry installation",
            )
            require_same_user_session(target)
        (workspace / "ready").write_text("ready\n")
        if from_dmg:
            installer.stop_app(target)
        else:
            wait_for_exit(pid, target)
        installer.require(
            verify_release(staged, version) == desired_digest,
            "staged update changed before activation",
        )
        installer.require(
            (installer.validate(target) if target.exists() else None)
            == previous_digest,
            "installed app changed during update",
        )
        installer.require(
            ownership(target) == owner, "app ownership changed during update"
        )
        previous_records = {}
        if owner is not None:
            state_path, record, owner_path = owner
            previous_records = {
                path: path.read_bytes() for path in (state_path, owner_path)
            }
            updated = installer.record_bytes({**record, "sha256": desired_digest})
        exchanged = False
        try:
            with installer.mutation():
                installer.rename(staged, target, exchange=previous_digest is not None)
                exchanged = True
                for path in previous_records:
                    installer.write_bytes(path, updated)
            installer.command(
                ["/usr/bin/open", "-n", target], "updated app could not launch"
            )
            if from_dmg:
                wait_for_launch(target)
        except BaseException:
            try:
                with installer.mutation():
                    if exchanged:
                        installer.require(
                            installer.validate(target) == desired_digest,
                            "app changed during rollback",
                        )
                        if previous_digest is not None:
                            installer.rename(staged, target, exchange=True)
                        else:
                            installer.rename(target, staged)
                    for path, content in previous_records.items():
                        installer.write_bytes(path, content)
            except BaseException as error:
                preserve_backup = True
                raise installer.AppError(
                    "update rollback failed; private app backup was preserved"
                ) from error
            # main() persists the failure before reopening the previous app so
            # its startup can reliably display the diagnostic.
            raise
    finally:
        if stage_directory is not None and not preserve_backup:
            shutil.rmtree(stage_directory, ignore_errors=True)
        with contextlib.suppress(OSError):
            lock.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--version")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--install-from-dmg", action="store_true")
    options = parser.parse_args()
    target = options.target
    workspace_checked = not options.install_from_dmg
    try:
        if options.install_from_dmg:
            installer.require(
                options.target is None
                and options.version is None
                and options.pid is None,
                "disk image installation arguments are invalid",
            )
            workspace = physical_path(
                options.workspace, "installation workspace is invalid"
            )
            installer.require(
                workspace.is_dir()
                and workspace.stat().st_uid == os.getuid()
                and workspace.stat().st_mode & 0o077 == 0,
                "installation workspace must be private",
            )
            workspace_checked = True
            candidate = physical_path(
                options.candidate, "installation source is invalid"
            )
            installer.require(
                candidate.name == "Stark.app", "installation source is invalid"
            )
            version = release_version(candidate)
            # Verify the source before discovering or creating any destination.
            verify_release(candidate, version)
            target = dmg_target()
        else:
            installer.require(
                options.pid is not None and options.pid > 1,
                "update process is invalid",
            )
            installer.require(
                target is not None and options.version is not None,
                "update arguments are invalid",
            )
            candidate, version = options.candidate, options.version
        apply_update(
            candidate,
            target,
            version,
            options.pid,
            options.workspace,
        )
    except Exception as error:  # noqa: BLE001 - content-free subprocess boundary
        message = (
            str(error)
            if isinstance(error, installer.AppError)
            else "app update failed; previous installation was kept"
        )
        if workspace_checked:
            with contextlib.suppress(OSError):
                (options.workspace / "result.json").write_text(
                    json.dumps({"error": message}) + "\n"
                )
        if options.install_from_dmg:
            print(json.dumps({"error": message}))
        if (
            target is not None
            and target.exists()
            and (options.workspace / "ready").exists()
        ):
            with contextlib.suppress(Exception):
                if not installer.running(target):
                    installer.command(
                        ["/usr/bin/open", "-n", target],
                        "previous app could not launch",
                    )
        return 1
    if options.install_from_dmg:
        print(json.dumps({"path": str(target), "version": version}))
    shutil.rmtree(options.workspace, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
