#!/usr/bin/env python3
"""Apply a verified app update after the menu bar app has exited.

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
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import install_app as installer

TEAM_ID = "S5BT2PRN5B"


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
    candidate: Path, target: Path, version: str, pid: int, workspace: Path
) -> None:
    installer.require(
        target.is_absolute() and target.name == "Stark.app" and not target.is_symlink(),
        "update destination is invalid",
    )
    previous_digest = installer.validate(target)
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
        (workspace / "ready").write_text("ready\n")
        wait_for_exit(pid, target)
        installer.require(
            verify_release(staged, version) == desired_digest,
            "staged update changed before activation",
        )
        installer.require(
            installer.validate(target) == previous_digest,
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
                installer.rename(staged, target, exchange=True)
                exchanged = True
                for path in previous_records:
                    installer.write_bytes(path, updated)
            installer.command(
                ["/usr/bin/open", "-n", target], "updated app could not launch"
            )
        except BaseException:
            try:
                with installer.mutation():
                    if exchanged:
                        installer.require(
                            installer.validate(target) == desired_digest,
                            "app changed during rollback",
                        )
                        installer.rename(staged, target, exchange=True)
                    for path, content in previous_records.items():
                        installer.write_bytes(path, content)
            except BaseException as error:
                preserve_backup = True
                raise installer.AppError(
                    "update rollback failed; private app backup was preserved"
                ) from error
            with contextlib.suppress(installer.AppError):
                installer.command(
                    ["/usr/bin/open", "-n", target], "previous app could not launch"
                )
            raise
    finally:
        if stage_directory is not None and not preserve_backup:
            shutil.rmtree(stage_directory, ignore_errors=True)
        with contextlib.suppress(OSError):
            lock.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    options = parser.parse_args()
    try:
        installer.require(options.pid > 1, "update process is invalid")
        apply_update(
            options.candidate,
            options.target,
            options.version,
            options.pid,
            options.workspace,
        )
    except Exception as error:  # noqa: BLE001 - content-free subprocess boundary
        message = (
            str(error)
            if isinstance(error, installer.AppError)
            else "app update failed; previous installation was kept"
        )
        (options.workspace / "result.json").write_text(
            json.dumps({"error": message}) + "\n"
        )
        if (options.workspace / "ready").exists():
            with contextlib.suppress(Exception):
                if not installer.running(options.target):
                    installer.command(
                        ["/usr/bin/open", "-n", options.target],
                        "previous app could not launch",
                    )
        return 1
    shutil.rmtree(options.workspace, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
