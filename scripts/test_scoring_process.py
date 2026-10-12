"""Model-free lifecycle checks for the standalone native validation owner.

Run with: uv run --python 3.12.13 python scripts/test_scoring_process.py --reaper runtime/stark-validation-reap
No child output, paths or raw exceptions are emitted; only authored scalar codes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from scoring_process import ProcessError, ProcessOwner

CURRENT_STAGE = "startup"
PROCESS_ERROR_CODES = frozenset({
    "short_lived_exit_invalid", "short_lived_ownership_unsettled", "pipe_transfer_invalid",
    "clean_eof_ownership_unsettled", "explicit_stop_ownership_unsettled", "concurrent_launch_not_rejected",
    "concurrent_instance_count_invalid", "launch_cancellation_not_delivered", "cancelled_launch_ownership_unsettled",
    "cancelled_close_ownership_unsettled", "exec_failure_not_reported", "exec_failure_ownership_unsettled",
    "wrapper_loss_not_reported", "wrapper_loss_settlement_claimed", "wrapper_loss_close_claimed",
})


def require(condition, code):
    if not condition:
        raise ProcessError(code)


async def launch(owner, executable, arguments=()):
    return await owner.launch(Path(executable), list(arguments), Path("/"), {"PATH": "/usr/bin:/bin"})


async def checks(reaper):
    global CURRENT_STAGE
    passed = []
    CURRENT_STAGE = "short_lived_exit"
    for executable, expected in (("/usr/bin/true", 0), ("/usr/bin/false", 1)):
        owner = ProcessOwner(reaper)
        try:
            process = await launch(owner, executable)
            require(await process.wait() == expected, "short_lived_exit_invalid")
            await process.stop()
            require(not owner.instances, "short_lived_ownership_unsettled")
        finally:
            await owner.close()
    passed.append("short_lived_exit")
    CURRENT_STAGE = "pipes_and_clean_eof"
    owner = ProcessOwner(reaper)
    try:
        process = await launch(owner, "/bin/cat")
        process.stdin.write(b"fixture-check\n")
        await process.stdin.drain()
        require(await process.stdout.readline() == b"fixture-check\n", "pipe_transfer_invalid")
        process.stdin.close()
        await process.stdin.wait_closed()
        require(await process.wait() == 0 and not owner.instances, "clean_eof_ownership_unsettled")
        await process.stop()
    finally:
        await owner.close()
    passed.append("pipes_and_clean_eof")
    CURRENT_STAGE = "explicit_stop"
    owner = ProcessOwner(reaper)
    try:
        process = await launch(owner, "/bin/sleep", ("60",))
        require(await process.stop() == -15 and not owner.instances, "explicit_stop_ownership_unsettled")
    finally:
        await owner.close()
    passed.append("explicit_stop")
    CURRENT_STAGE = "concurrent_launch_rejected"
    owner = ProcessOwner(reaper)
    try:
        first = asyncio.create_task(launch(owner, "/bin/sleep", ("60",)))
        await asyncio.sleep(0)
        try:
            await launch(owner, "/bin/sleep", ("60",))
        except ProcessError:
            pass
        else:
            raise ProcessError("concurrent_launch_not_rejected")
        process = await first
        require(len(owner.instances) == 1, "concurrent_instance_count_invalid")
        await process.stop()
    finally:
        await owner.close()
    passed.append("concurrent_launch_rejected")
    CURRENT_STAGE = "launch_cancellation"
    owner = ProcessOwner(reaper)
    try:
        task = asyncio.create_task(launch(owner, "/bin/sleep", ("60",)))
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise ProcessError("launch_cancellation_not_delivered")
        require(not owner.instances and not owner.launches, "cancelled_launch_ownership_unsettled")
    finally:
        await owner.close()
    passed.append("launch_cancellation")
    CURRENT_STAGE = "cleanup_cancellation"
    owner = ProcessOwner(reaper)
    await launch(owner, "/bin/sleep", ("60",))
    task = asyncio.create_task(owner.close())
    await asyncio.sleep(0)
    task.cancel()
    await task
    require(not owner.instances and not owner.launches, "cancelled_close_ownership_unsettled")
    passed.append("cleanup_cancellation")
    CURRENT_STAGE = "exec_failure"
    owner = ProcessOwner(reaper)
    try:
        try:
            await launch(owner, "/nonexistent-validation-worker")
        except ProcessError:
            pass
        else:
            raise ProcessError("exec_failure_not_reported")
        require(not owner.instances, "exec_failure_ownership_unsettled")
    finally:
        await owner.close()
    passed.append("exec_failure")
    CURRENT_STAGE = "wrapper_loss_retained"
    owner = ProcessOwner(Path("/usr/bin/false"))
    try:
        await launch(owner, "/usr/bin/true")
    except ProcessError:
        pass
    else:
        raise ProcessError("wrapper_loss_not_reported")
    require(len(owner.instances) == 1 and next(iter(owner.instances.values())).ownership_lost,
            "wrapper_loss_settlement_claimed")
    try:
        await owner.close()
    except ProcessError:
        pass
    else:
        raise ProcessError("wrapper_loss_close_claimed")
    passed.append("wrapper_loss_retained")
    return passed


def main():
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--reaper", type=Path, required=True)
        args = parser.parse_args()
        results = asyncio.run(asyncio.wait_for(checks(args.reaper.absolute()), 30))
        print(json.dumps({"status": "passed", "checks": results, "model_free": True}), flush=True)
        return 0
    except ProcessError as failure:
        code = str(failure)
        error = code if code in PROCESS_ERROR_CODES else "process_lifecycle_check_failed"
        print(json.dumps({"status": "failed", "error": error, "check": CURRENT_STAGE, "model_free": True}), flush=True)
        return 1
    except (Exception, KeyboardInterrupt):
        print(json.dumps({"status": "failed", "error": "process_lifecycle_check_failed", "check": CURRENT_STAGE,
                          "model_free": True}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
