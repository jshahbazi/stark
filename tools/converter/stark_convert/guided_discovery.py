"""Bounded metadata-only catalog scan, isolated from the loaded model worker."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from .model_catalog import family_catalog
from .model_support import support_matrix


def discover(*, catalog=family_catalog):
    # One candidate window per generation family, with at most three active
    # family lookups. The desktop parent enforces the overall process deadline.
    families = [
        family["family_id"]
        for family in support_matrix()["families"]
        if family["modes"]["text"]["status"] == "conditional"
    ]
    items, checked, incomplete, failed = {}, 0, False, 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(catalog, family, planning=True, deadline_seconds=10, limit=30): family for family in families
        }
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
    print(json.dumps(discover(), allow_nan=False))


if __name__ == "__main__":
    main()
