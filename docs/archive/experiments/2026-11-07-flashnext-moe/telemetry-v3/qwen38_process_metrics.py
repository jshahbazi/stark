"""Small, privacy-preserving proc_pid_rusage sampler for a worker process.

The PID is supplied by the caller and is never returned or stored. This uses
the macOS RUSAGE_INFO_V4 ABI; counters retain SDK raw units except memory
gauges (bytes). `mach_timebase_info` converts the raw time counters from
Mach absolute-time ticks to seconds.
"""

from __future__ import annotations

import ctypes
import os
import struct
import sys
from typing import Any

RUSAGE_INFO_V4 = 4


class _RusageInfoV4(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups",
            "pageins", "wired_size", "resident_size", "phys_footprint",
            "proc_start_abstime", "proc_exit_abstime", "child_user_time",
            "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups",
            "child_pageins", "child_elapsed_abstime", "diskio_bytesread",
            "diskio_byteswritten", "cpu_time_qos_default",
            "cpu_time_qos_maintenance", "cpu_time_qos_background",
            "cpu_time_qos_utility", "cpu_time_qos_legacy",
            "cpu_time_qos_user_initiated", "cpu_time_qos_user_interactive",
            "billed_system_time", "serviced_system_time", "logical_writes",
            "lifetime_max_phys_footprint", "instructions", "cycles",
            "billed_energy", "serviced_energy", "interval_max_phys_footprint",
            "runnable_time",
        )
    ]


_COUNTERS = tuple(name for name, _ in _RusageInfoV4._fields_[1:])
_GAUGES = {"wired_size", "resident_size", "phys_footprint", "lifetime_max_phys_footprint", "interval_max_phys_footprint"}
_TIME_FIELDS = {
    "user_time", "system_time", "child_user_time", "child_system_time",
    "cpu_time_qos_default", "cpu_time_qos_maintenance", "cpu_time_qos_background",
    "cpu_time_qos_utility", "cpu_time_qos_legacy", "cpu_time_qos_user_initiated",
    "cpu_time_qos_user_interactive", "billed_system_time", "serviced_system_time",
    "runnable_time",
}
_TIME_FIELDS = tuple(sorted(_TIME_FIELDS))


class _MachTimebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


def _mach_timebase(libsystem) -> tuple[int, int] | None:
    try:
        function = libsystem.mach_timebase_info
        function.argtypes = [ctypes.POINTER(_MachTimebase)]
        function.restype = ctypes.c_int
        value = _MachTimebase()
        if function(ctypes.byref(value)) != 0 or not value.denom:
            return None
        return int(value.numer), int(value.denom)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def snapshot(pid: int) -> dict[str, Any]:
    """Return a cumulative snapshot; failed APIs return availability only."""
    unavailable = {"available": False, "reason": "unsupported"}
    if sys.platform != "darwin":
        return unavailable
    if type(pid) is not int or pid <= 0:
        return {"available": False, "reason": "invalid_pid"}
    try:
        libproc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        bsd_info = ctypes.create_string_buffer(1024)
        proc_pidinfo = libproc.proc_pidinfo
        proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        proc_pidinfo.restype = ctypes.c_int
        received = proc_pidinfo(pid, 3, 0, bsd_info, len(bsd_info))
        if received < 24:
            return {"available": False, "reason": "api_unavailable"}
        target_uid = struct.unpack_from("=I", bsd_info.raw, 20)[0]
        if target_uid != os.getuid():
            return {"available": False, "reason": "not_own_user"}
        function = libproc.proc_pid_rusage
        # Although the SDK typedef spells this as rusage_info_t * (void **),
        # the API writes the fixed-size flavor record into caller storage.
        function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        function.restype = ctypes.c_int
        info = _RusageInfoV4()
        result = function(pid, RUSAGE_INFO_V4, ctypes.byref(info))
        if result != 0:
            return {"available": False, "reason": "api_unavailable"}
        values = {name: int(getattr(info, name)) for name in _COUNTERS}
        timebase = _mach_timebase(libproc)
        seconds = None
        if timebase is not None:
            numer, denom = timebase
            seconds = {name: values[name] * numer / denom / 1_000_000_000 for name in _TIME_FIELDS}
        return {
            "available": True,
            "units": {"time_raw": "Mach absolute-time ticks", "time_seconds": "seconds", "memory": "bytes",
                      "other": "raw SDK counters"},
            "cumulative": values,
            "cpu_seconds": seconds,
            "timebase": {"numer": timebase[0], "denom": timebase[1]} if timebase else None,
            "_identity": bytes(info.uuid),
        }
    except (AttributeError, OSError, TypeError, ValueError):
        return {"available": False, "reason": "api_unavailable"}


def delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Return interval counter deltas and latest memory gauges.

    UUID is held only as an in-memory identity guard and omitted from output.
    """
    if not before.get("available") or not after.get("available"):
        return {"available": False, "reason": "snapshot_unavailable"}
    if before.get("_identity") != after.get("_identity"):
        return {"available": False, "reason": "process_changed"}
    old, new = before.get("cumulative", {}), after.get("cumulative", {})
    counters: dict[str, int] = {}
    gauges: dict[str, int] = {}
    for name in _COUNTERS:
        if name not in old or name not in new or (name not in _GAUGES and new[name] < old[name]):
            continue
        (gauges if name in _GAUGES else counters)[name] = new[name] if name in _GAUGES else new[name] - old[name]
    cpu_seconds = None
    if before.get("cpu_seconds") is not None and after.get("cpu_seconds") is not None:
        cpu_seconds = {
            name: max(0.0, after["cpu_seconds"][name] - before["cpu_seconds"][name])
            for name in _TIME_FIELDS
            if name in before["cpu_seconds"] and name in after["cpu_seconds"]
        }
    return {
        "available": True,
        "units": after.get("units", {}),
        "delta": counters,
        "cpu_seconds": cpu_seconds,
        "gauges": gauges,
        "time_fields": list(_TIME_FIELDS),
    }
