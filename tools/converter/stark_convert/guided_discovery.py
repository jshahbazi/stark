"""Bounded metadata-only catalog scan, isolated from the loaded model worker."""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .model_catalog import MAX_ITEMS, family_catalog
from .model_support import support_matrix

FAMILY_SECONDS = 10
MAX_FAMILY_PAGES = 10
FAMILY_WORKERS = 3
STARTUP_SERIALIZATION_MARGIN_SECONDS = 10


def discovery_bounds(family_count=None):
    """Return (maximum downloadable rows, subprocess timeout) for this scan."""
    if family_count is None:
        family_count = sum(
            family["modes"]["text"]["status"] == "conditional" for family in support_matrix()["families"]
        )
    waves = math.ceil(family_count / FAMILY_WORKERS)
    return (
        family_count * MAX_ITEMS,
        waves * FAMILY_SECONDS + STARTUP_SERIALIZATION_MARGIN_SECONDS,
    )


def _discover_family(catalog, family, task="general"):
    """Pass unavailable pages while sharing one family's metadata budget."""
    deadline = time.monotonic() + FAMILY_SECONDS
    items = {}
    offset = 0
    report = None
    for page in range(MAX_FAMILY_PAGES):
        seconds = FAMILY_SECONDS if page == 0 else deadline - time.monotonic()
        if seconds <= 0:
            break
        options = {"planning": True, "deadline_seconds": seconds, "limit": MAX_ITEMS, "ranking_task": task}
        if offset:
            options["offset"] = offset
        try:
            report = catalog(family, **options)
        except Exception:
            if report is None:
                raise
            return {"status": "error", "has_more": True, "items": list(items.values())}
        for item in report["items"]:
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("downloadable") is True:
                items[item["id"]] = item
        if (
            report["status"] != "ready"
            or not report.get("has_more", False)
            or any(item.get("downloadable") is True for item in items.values())
        ):
            break
        next_offset = report.get("next_offset")
        if type(next_offset) is not int or not offset < next_offset <= 10_000_000:
            break
        offset = next_offset
    return {**report, "items": list(items.values())}


def discover(*, catalog=family_catalog, task="general"):
    if task not in {"general", "coding"}:
        raise ValueError("Unsupported guided discovery task")
    # At most three families share the active lookup slots. Each family passes
    # unavailable pages within its own total deadline and metadata page budget.
    families = [
        family["family_id"]
        for family in support_matrix()["families"]
        if family["modes"]["text"]["status"] == "conditional"
    ]
    items, checked, incomplete, failed = {}, 0, False, 0
    with ThreadPoolExecutor(max_workers=FAMILY_WORKERS) as pool:
        futures = {pool.submit(_discover_family, catalog, family, task): family for family in families}
        for future in as_completed(futures):
            try:
                report = future.result()
                checked += 1
                partial_failure = bool(report.get("partial_failure", False))
                failed += int(report["status"] != "ready" or partial_failure)
                incomplete |= (
                    report["status"] != "ready"
                    or partial_failure
                    or report.get("incomplete", False)
                    or report.get("has_more", False)
                )
                for item in report["items"]:
                    if isinstance(item, dict) and isinstance(item.get("id"), str):
                        items[item["id"]] = item
            except Exception:
                # Never return exception text, paths, credentials or Hub payloads.
                incomplete = True
                failed += 1
    return {
        "schema_version": 1,
        "items": list(items.values()),
        "families_checked": checked,
        "incomplete": incomplete,
        "failed_families": failed,
        "bounded_scan": True,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("general", "coding"), default="general")
    arguments = parser.parse_args()
    print(json.dumps(discover(task=arguments.task), allow_nan=False))


if __name__ == "__main__":
    main()
