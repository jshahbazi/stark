"""Disk cache of catalogue verdicts for pinned repository revisions.

A verdict is derived from one immutable revision by this converter's own
checks, so it stays valid until the revision or the converter changes. Entries
hold public repository metadata only and never grant installation rights:
installing repeats the pinned preflight.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path

SCHEMA_VERSION = 1
# A verdict for a fixed revision and converter build is deterministic.
SETTLED_SECONDS = 30 * 24 * 3600
# Access-dependent outcomes (gated or missing files, metadata borrowed from
# another repository's moving head) are retried soon.
VOLATILE_SECONDS = 3600
# How long a confirmed head revision is assumed not to have moved.
HEAD_SECONDS = 600
MAX_ENTRY_BYTES = 64 * 1024


@lru_cache(maxsize=1)
def converter_fingerprint():
    """Identify the installed converter sources that produce verdicts."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        status = path.stat()
        digest.update(f"{path.relative_to(root).as_posix()}\0{status.st_size}\0{status.st_mtime_ns}\n".encode())
    return digest.hexdigest()


class VerdictCache:
    """One family's verdicts, keyed by repository and bound to a revision."""

    def __init__(self, root, family, *, planning=False, fingerprint=None, clock=time.time):
        self.directory = Path(root) / family / ("planning" if planning else "browse")
        self.fingerprint = fingerprint if fingerprint is not None else converter_fingerprint()
        self.clock = clock

    def _path(self, repo):
        return self.directory / (hashlib.sha256(repo.casefold().encode()).hexdigest()[:32] + ".json")

    def _entry(self, repo):
        """Return this repository's current entry, or ``None``."""
        try:
            path = self._path(repo)
            if path.stat().st_size > MAX_ENTRY_BYTES:
                return None
            entry = json.loads(path.read_bytes())
            item = entry["item"]
            lifetime = SETTLED_SECONDS if entry["settled"] is True else VOLATILE_SECONDS
            if (
                entry["schema_version"] != SCHEMA_VERSION
                or entry["fingerprint"] != self.fingerprint
                or entry["repo"] != repo
                or not isinstance(entry["revision"], str)
                or type(entry["checked"]) is not int
                or type(entry["confirmed"]) is not int
                or not 0 <= self.clock() - entry["checked"] < lifetime
                or not (item is None or isinstance(item, dict))
            ):
                return None
            return entry
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write(self, repo, entry):
        """Replace an entry. A failed write only costs a later recheck."""
        temporary = None
        try:
            payload = json.dumps(entry, sort_keys=True).encode()
            if len(payload) > MAX_ENTRY_BYTES:
                return
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            handle, temporary = tempfile.mkstemp(dir=self.directory, prefix=".verdict-")
            with os.fdopen(handle, "wb") as stream:
                stream.write(payload)
            os.replace(temporary, self._path(repo))
            temporary = None
        except (OSError, ValueError, TypeError):
            pass
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def load(self, repo, revision):
        """Return ``(True, item_or_None)`` for a current verdict on this revision, else ``(False, None)``."""
        entry = self._entry(repo)
        if entry is None or entry["revision"] != revision:
            return False, None
        return True, entry["item"]

    def load_recent(self, repo):
        """Return a current verdict whose revision was confirmed as the head moments ago."""
        entry = self._entry(repo)
        if entry is None or not 0 <= self.clock() - entry["confirmed"] < HEAD_SECONDS:
            return False, None
        return True, entry["item"]

    def confirm(self, repo, revision):
        """Note that the stored revision is still the repository's head."""
        entry = self._entry(repo)
        if entry is not None and entry["revision"] == revision:
            self._write(repo, {**entry, "confirmed": int(self.clock())})

    def store(self, repo, revision, item, *, settled=True):
        """Record the verdict for a revision that is the repository's head now."""
        now = int(self.clock())
        self._write(
            repo,
            {
                "schema_version": SCHEMA_VERSION,
                "fingerprint": self.fingerprint,
                "repo": repo,
                "revision": revision,
                "checked": now,
                "confirmed": now,
                "settled": bool(settled),
                "item": item,
            },
        )
