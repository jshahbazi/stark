"""Own only image/cache IDs introduced by one serialized benchmark attempt.

No command in this module deletes volumes, containers, baseline images or
baseline cache records. All command diagnostics are discarded. Receipts retain
infrastructure IDs, fixed categories, counters and filesystem capacity only.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

MIN_FREE_BYTES = 12 * 1024**3
_IMAGE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CACHE = re.compile(r"[a-z0-9]{20,64}\Z")


class ResourceError(ValueError):
    """Fixed content-free benchmark resource failure."""


def require_disk_space(path):
    Path(path).mkdir(parents=True, exist_ok=True, mode=0o700)
    if shutil.disk_usage(path).free < MIN_FREE_BYTES:
        raise ResourceError("insufficient_disk_space")


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _lines(raw):
    result = []
    try:
        for line in raw.splitlines():
            if line.strip():
                value = json.loads(line)
                result.extend(value if isinstance(value, list) else [value])
    except (ValueError, TypeError):
        raise ResourceError("resource_snapshot_failed") from None
    if not all(isinstance(value, dict) for value in result):
        raise ResourceError("resource_snapshot_failed")
    return result


class DockerResources:
    """Narrow, argument-list-only Docker metadata and exact-ID cleanup client."""

    @staticmethod
    def command(arguments, *, mutation=False):
        try:
            reply = subprocess.run(["docker", *arguments], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   text=True, timeout=30 if mutation else 15, check=False)
        except (OSError, subprocess.TimeoutExpired):
            raise ResourceError("resource_cleanup_failed" if mutation else "resource_snapshot_failed") from None
        if reply.returncode != 0 or len(reply.stdout) > 16 * 1024**2:
            raise ResourceError("resource_cleanup_failed" if mutation else "resource_snapshot_failed")
        return reply.stdout

    def builder(self):
        rows = _lines(self.command(["buildx", "ls", "--format=json"]))
        requested = os.environ.get("BUILDX_BUILDER")
        selected = [row for row in rows if row.get("Name") == requested] if requested else [
            row for row in rows if row.get("Current") is True
        ]
        if len(selected) != 1:
            raise ResourceError("resource_snapshot_failed")
        row = selected[0]
        name = row.get("Name")
        nodes = row.get("Nodes", [])
        if (not isinstance(nodes, list) or not all(isinstance(node, dict) for node in nodes)
                or not all(isinstance(node.get("IDs"), list) for node in nodes)):
            raise ResourceError("resource_snapshot_failed")
        identifiers = [identifier for node in nodes for identifier in node["IDs"]]
        if not all(isinstance(value, str) for value in identifiers):
            raise ResourceError("resource_snapshot_failed")
        workers = sorted(set(identifiers))
        if (not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) is None
                or not workers or not all(isinstance(value, str) for value in workers)):
            raise ResourceError("resource_snapshot_failed")
        return {"name": name, "workers": workers}

    def images(self):
        values = set(self.command(["image", "ls", "--all", "--quiet", "--no-trunc"]).split())
        if not all(_IMAGE.fullmatch(value) is not None for value in values):
            raise ResourceError("resource_snapshot_failed")
        return values

    def volumes(self):
        # Metadata only: this module has no volume deletion path.
        return set(self.command(["volume", "ls", "--quiet"]).split())

    def cache(self, builder, filters=()):
        arguments = ["buildx", "du", "--builder", builder["name"], "--format=json"]
        for selector in filters:
            arguments.extend(["--filter", selector])
        result = {}
        for row in _lines(self.command(arguments)):
            identifier = row.get("ID")
            parents = row.get("Parents") or []
            if (not isinstance(identifier, str) or _CACHE.fullmatch(identifier) is None
                    or not isinstance(parents, list) or not all(isinstance(value, str) for value in parents)
                    or not isinstance(row.get("Type"), str)
                    or any(type(row.get(name)) is not bool for name in ("Shared", "Mutable", "Reclaimable"))):
                raise ResourceError("resource_snapshot_failed")
            result[identifier] = {name: row.get(name) for name in ("Shared", "Mutable", "Reclaimable", "Type")}
            result[identifier]["Parents"] = parents
        return result

    def image_in_use(self, identifier):
        return bool(self.command(["ps", "--all", "--no-trunc", "--filter", "ancestor=" + identifier,
                                  "--format", "{{.ID}}"]).strip())

    def remove_image(self, identifier):
        if _IMAGE.fullmatch(identifier) is None:
            raise ResourceError("resource_cleanup_failed")
        # Do not let Docker recursively remove any untagged parent image IDs.
        self.command(["image", "rm", "--no-prune", identifier], mutation=True)

    def prune_cache(self, builder, identifier):
        if _CACHE.fullmatch(identifier) is None:
            raise ResourceError("resource_cleanup_failed")
        # BuildKit models these booleans as empty-valued presence fields.
        # Regex ^$ selects only present private/immutable fields. '=false'
        # would not express negation. Active records are never pruned by BuildKit.
        self.command(["buildx", "prune", "--builder", builder["name"], "--force",
                      "--filter", "id=" + identifier, "--filter", "private~=^$",
                      "--filter", "immutable~=^$"], mutation=True)


def eligible(record):
    return (record.get("Shared") is False and record.get("Mutable") is False
            and record.get("Reclaimable") is True and record.get("Type") not in {"internal", "frontend"})


def _cache_order(records, identifiers):
    depths = {}

    def depth(identifier, seen):
        if identifier in seen:
            raise ResourceError("resource_snapshot_failed")
        if identifier not in depths:
            parents = [value for value in records[identifier]["Parents"] if value in identifiers]
            depths[identifier] = 1 + max((depth(value, seen | {identifier}) for value in parents), default=0)
        return depths[identifier]

    return sorted(identifiers, key=lambda value: (-depth(value, set()), value))


def cleanup_introduced(client, builder, before_images, before_cache, before_volumes, *, report=None):
    if client.builder() != builder or not before_volumes <= client.volumes():
        raise ResourceError("resource_identity_changed")
    report = {} if report is None else report
    report.update(new_image_ids=sorted(client.images() - before_images), removed_image_ids=[],
                  retained_image_ids=[], new_cache_ids=[], removed_cache_ids=[], retained_cache_ids=[],
                  image_removal_attempted_ids=[], cache_prune_attempted_ids=[])
    failed = False
    for identifier in report["new_image_ids"]:
        try:
            if client.image_in_use(identifier):
                report["retained_image_ids"].append(identifier)
                failed = True
                continue
            report["image_removal_attempted_ids"].append(identifier)
            client.remove_image(identifier)
            if identifier in client.images():
                raise ResourceError("resource_cleanup_failed")
            report["removed_image_ids"].append(identifier)
        except ResourceError:
            report["retained_image_ids"].append(identifier)
            failed = True
    # Removing owned images can make their own layers unshared. Re-snapshot
    # after image cleanup, while the pre-task cache IDs remain protected.
    after = client.cache(builder)
    introduced = set(after) - set(before_cache)
    report["new_cache_ids"] = sorted(introduced)
    for identifier in _cache_order(after, introduced):
        if not eligible(after[identifier]):
            report["retained_cache_ids"].append(identifier)
            continue
        try:
            current = client.cache(builder, ["id=" + identifier, "private~=^$", "immutable~=^$"])
            if set(current) != {identifier} or not eligible(current[identifier]):
                report["retained_cache_ids"].append(identifier)
                continue
            report["cache_prune_attempted_ids"].append(identifier)
            client.prune_cache(builder, identifier)
            if identifier in client.cache(builder, ["id=" + identifier]):
                report["retained_cache_ids"].append(identifier)
            else:
                report["removed_cache_ids"].append(identifier)
        except ResourceError:
            report["retained_cache_ids"].append(identifier)
            failed = True
    report["cleanup_failed"] = failed
    return report


@contextlib.contextmanager
def task_resources(path, *, client=None):
    """Run inside the runner's serial benchmark lock, once per task attempt."""
    path = Path(path)
    require_disk_space(path.parent)
    client = client or DockerResources()
    builder = client.builder()
    before_images, before_cache, before_volumes = client.images(), client.cache(builder), client.volumes()
    receipt = {"schema_version": 1, "started_at": datetime.now(UTC).isoformat(), "state": "running",
               "builder": builder, "protected_image_ids": sorted(before_images),
               "protected_cache_ids": sorted(before_cache), "protected_volume_count": len(before_volumes),
               "protected_volumes_sha256": hashlib.sha256(json.dumps(sorted(before_volumes)).encode()).hexdigest(),
               "host_free_before_bytes": shutil.disk_usage(path.parent).free}
    _write(path, receipt)
    body_failed = False
    try:
        yield
    except BaseException:
        body_failed = True
        raise
    finally:
        failure = None
        try:
            cleanup_introduced(client, builder, before_images, before_cache, before_volumes, report=receipt)
            if receipt["cleanup_failed"]:
                failure = "resource_cleanup_failed"
        except ResourceError as error:
            failure = str(error)
        receipt.update(state="cleanup_failed" if failure else "finished", failure=failure,
                       body_failed=body_failed, finished_at=datetime.now(UTC).isoformat(),
                       host_free_after_bytes=shutil.disk_usage(path.parent).free)
        _write(path, receipt)
        if failure and not body_failed:
            raise ResourceError(failure)
