"""Process-wide request and template resource budgets.

``STARK_REQUEST_BUDGET_MIB`` is read once when the package is imported so one
process cannot change limits halfway through a request lifecycle.
"""

from __future__ import annotations

import os
import sys

import psutil

# Match the native allocation-size bound for the largest derived allowance,
# 32 MiB + 32B. Content has no fixed product ceiling.
MAX_BUDGET_MIB = (sys.maxsize - 32 * 1024 * 1024) // (32 * 1024 * 1024)


def available_memory_bytes():
    """Current host headroom; failure uses the conservative automatic floor."""
    try:
        return max(0, int(psutil.virtual_memory().available))
    except (OSError, ValueError, psutil.Error):
        return 0


def automatic_budget_mib(available_bytes):
    # Size the content envelope from a conservative share of host headroom.
    # Derived framing, parsed values and native execution need additional RAM;
    # this is not a reservation, RSS ceiling or model-capacity guarantee.
    return max(16, min(MAX_BUDGET_MIB, available_bytes // (128 * 1024 * 1024)))


def json_parse_headroom_bytes(raw_bytes):
    """Conservative transient JSON allowance; this is not an RSS guarantee.

    Decoding bytes can hold a four-byte-wide Unicode source and its parsed
    values together. Small control/retry frames retain their existing path.
    """
    return 12 * raw_bytes if raw_bytes > 1024 * 1024 else 0


_SETTING = os.environ.get("STARK_REQUEST_BUDGET_MIB", str(automatic_budget_mib(available_memory_bytes())))
if (
    not _SETTING.isascii()
    or not _SETTING.isdecimal()
    or not 1 <= len(_SETTING) <= len(str(MAX_BUDGET_MIB))
    or not 1 <= int(_SETTING) <= MAX_BUDGET_MIB
):
    raise SystemExit(
        "Invalid STARK_REQUEST_BUDGET_MIB; expected a positive decimal integer with representable byte allowances."
    )

REQUEST_BUDGET_MIB = int(_SETTING)
INPUT_BYTES = REQUEST_BUDGET_MIB * 1024 * 1024
WIRE_BYTES = 6 * INPUT_BYTES + 2 * 1024 * 1024
PENDING_BYTES = 2 * (WIRE_BYTES + 6 * 1024 * 1024)
TEMPLATE_LIVE_BYTES = 128 * 1024 * 1024 + 8 * INPUT_BYTES
TEMPLATE_CUMULATIVE_BYTES = 32 * 1024 * 1024 + 32 * INPUT_BYTES
