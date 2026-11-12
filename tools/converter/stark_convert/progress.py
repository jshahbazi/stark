"""Private, content-free byte progress for model installation.

Only phase names, fixed states and numeric counts leave this module. Download
counts describe source bytes available locally, including reusable cached bytes.
"""

from __future__ import annotations

import errno
import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub.utils import tqdm as hf_tqdm

ProgressCallback = Callable[[dict[str, Any]], None]

CONVERTER_FAILURE_CATEGORIES = frozenset(
    {
        "disk_full",
        "permission_denied",
        "unsupported_format_or_config",
        "invalid_checkpoint",
        "conversion_failed",
    }
)


def converter_failure_category(error: BaseException) -> str:
    """Map failures to a fixed, content-free diagnostic category."""
    error_name = type(error).__name__
    error_number = getattr(error, "errno", None)
    if error_number in {errno.ENOSPC, getattr(errno, "EDQUOT", -1)}:
        return "disk_full"
    if isinstance(error, PermissionError) or error_number in {errno.EPERM, errno.EACCES}:
        return "permission_denied"
    if error_name == "UnsupportedConversion":
        return "unsupported_format_or_config"
    if error_name == "ConversionError":
        return "invalid_checkpoint"
    return "conversion_failed"


class FDProgressSink:
    """Own a duplicate of an inherited private pipe; never use stdout/stderr."""

    def __init__(self, descriptor: int):
        if type(descriptor) is not int or descriptor < 3:
            raise ValueError("invalid_progress_descriptor")
        self.descriptor = os.dup(descriptor)

    def __call__(self, event: dict) -> None:
        if self.descriptor is None:
            return
        data = memoryview((json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n").encode())
        try:
            while data:
                written = os.write(self.descriptor, data)
                if not written:
                    raise BrokenPipeError
                data = data[written:]
        except OSError:
            # Progress is advisory. The owning app independently cancels the
            # converter process group if its GUI or request goes away.
            self.close()

    def close(self) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None


class ProgressReporter:
    """Throttle intermediate byte counts while always publishing state changes."""

    def __init__(self, callback: ProgressCallback, *, clock=time.monotonic, interval: float = 0.1):
        self.callback = callback
        self.clock = clock
        self.interval = interval
        self.lock = threading.RLock()
        self.last: dict[str, tuple[float, dict]] = {}

    def emit(
        self,
        phase: str,
        state: str,
        completed: int = 0,
        total: int | None = None,
        *,
        transferred_bytes: int | None = None,
        force=False,
    ) -> None:
        if phase not in {"download", "conversion"} or state not in {"waiting", "running", "complete", "skipped"}:
            raise ValueError("invalid_progress_state")
        if type(completed) is not int or completed < 0 or total is not None and (type(total) is not int or total < 0):
            raise ValueError("invalid_progress_counts")
        if total is not None and completed > total:
            raise ValueError("invalid_progress_counts")
        event = {"phase": phase, "state": state, "completed_bytes": completed, "total_bytes": total}
        if transferred_bytes is not None:
            if phase != "download" or type(transferred_bytes) is not int or not 0 <= transferred_bytes <= 2**63 - 1:
                raise ValueError("invalid_progress_counts")
            event["transferred_bytes"] = transferred_bytes
        with self.lock:
            now = self.clock()
            previous = self.last.get(phase)
            changed = previous is None or any(previous[1][key] != event[key] for key in ("state", "total_bytes"))
            changed = (
                changed
                or previous is not None
                and ("transferred_bytes" in previous[1]) != ("transferred_bytes" in event)
            )
            if force or changed or now - previous[0] >= self.interval:
                self.callback(event)
                self.last[phase] = (now, event)

    def failure(self, category: str) -> None:
        """Publish one allowlisted, content-free converter failure category."""
        if category not in CONVERTER_FAILURE_CATEGORIES:
            raise ValueError("invalid_converter_failure_category")
        self.callback({"phase": "failure", "category": category})


def reporter(progress: ProgressCallback | ProgressReporter | None) -> ProgressReporter | None:
    return progress if progress is None or isinstance(progress, ProgressReporter) else ProgressReporter(progress)


class DownloadProgress:
    """Aggregate selected immutable files without exposing their identities."""

    def __init__(self, progress: ProgressReporter):
        self.progress = progress
        self.lock = threading.RLock()
        self.sizes: dict[tuple, int | None] = {}
        self.completed: dict[tuple, int] = {}
        self.paths: dict[tuple, Path] = {}
        self.selected: set[tuple] | None = None
        self.progress.emit("download", "running", force=True)

    def _emit(self, *, force=False) -> None:
        keys = self.selected if self.selected is not None else self.sizes.keys()
        completed = sum(self.completed.get(key, 0) for key in keys)
        total = None
        if self.selected is not None and all(self.sizes.get(key) is not None for key in keys):
            total = sum(self.sizes[key] for key in keys)
        self.progress.emit("download", "running", completed, total, force=force)

    def count(self, key: tuple, completed: int, total: int | None = None) -> None:
        with self.lock:
            if type(total) is int and total >= 0:
                self.sizes[key] = total
            size = self.sizes.get(key)
            value = max(0, completed)
            if size is not None:
                value = min(value, size)
            # HTTP retries may restart an ignored Range response. Keep the UI
            # monotonic without counting those retransferred bytes twice.
            self.completed[key] = max(self.completed.get(key, 0), value)
            self._emit()

    def bar_class(self, key: tuple):
        owner = self

        class SilentProgressBar:
            def __init__(self, *args, total=None, initial=0, **kwargs):
                self.total = total if type(total) is int and total >= 0 else owner.sizes.get(key)
                self.n = initial if type(initial) is int and initial >= 0 else 0
                owner.count(key, self.n, self.total)

            def update(self, amount=1):
                if type(amount) is int:
                    self.n = max(0, self.n + amount)
                    owner.count(key, self.n, self.total)

            def update_transfer(self, _amount=1):
                # HF Xet discovers this method and uses a single bar. File
                # reconstruction is the source-byte count; network transfers
                # are a separate counter and must not be added to it.
                pass

            def set_postfix_str(self, *_args, **_kwargs):
                pass

            def set_transfer_postfix_str(self, *_args, **_kwargs):
                pass

            def refresh(self, *_args, **_kwargs):
                pass

            def close(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        return SilentProgressBar

    def prepare(self, hub, repo: str, filename: str, revision: str) -> tuple:
        key = (repo, revision, filename)
        with self.lock:
            if key in self.sizes:
                return key
        info = hub.hf_hub_download(
            repo_id=repo, filename=filename, revision=revision, local_files_only=False, dry_run=True
        )
        size = getattr(info, "file_size", None)
        size = size if type(size) is int and size >= 0 else None
        with self.lock:
            self.sizes[key] = size
            self.completed[key] = size if getattr(info, "is_cached", False) and size is not None else 0
            self._emit()
        return key

    def select(self, hub, files: list[tuple[str, str, str]]) -> None:
        keys = {self.prepare(hub, repo, filename, revision) for repo, filename, revision in files}
        with self.lock:
            self.selected = keys
            self._emit(force=True)

    def download(self, hub, repo: str, filename: str, revision: str) -> Path:
        key = self.prepare(hub, repo, filename, revision)
        if key in self.paths:
            return self.paths[key]
        path = Path(
            hub.hf_hub_download(
                repo_id=repo,
                filename=filename,
                revision=revision,
                local_files_only=False,
                tqdm_class=self.bar_class(key),
            )
        )
        size = path.stat().st_size
        self.count(key, size, size)
        self.paths[key] = path
        return path

    def complete(self) -> None:
        with self.lock:
            keys = self.selected if self.selected is not None else self.sizes.keys()
            total = sum(self.completed.get(key, 0) for key in keys)
            self.progress.emit("download", "complete", total, total, force=True)


class SnapshotDownloadProgress:
    """Capture snapshot reconstruction bytes without exposing file or repo names.

    ``snapshot_download`` uses its aggregate reconstruction bar for bytes made
    available locally, and distinct bars for network transfer and file counts.
    Reconstruction bytes are capped by the caller's independently pinned
    maximum. Transfer bytes count positive received-byte updates, including
    retransmissions, and have no source-size cap or denominator.
    With a completed-files reader, its count replaces reconstruction progress
    and the pinned maximum becomes the fixed completion denominator.
    """

    BAR_NAME = "huggingface_hub.snapshot_download"
    TRANSFER_BAR_NAME = "huggingface_hub.snapshot_download.transfer"

    def __init__(self, progress: ProgressReporter, max_bytes: int, *, completed_files=None):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("invalid_snapshot_progress_limit")
        self.progress = progress
        self.max_bytes = max_bytes
        self.completed_files = completed_files
        self.completed_file_bytes = 0
        self.completed_bytes = 0
        self._reconstruction_raw_bytes = 0
        self.transferred_bytes = 0
        self.lock = threading.RLock()

    def _count_reconstruction(self, amount: int, *, force: bool = False) -> None:
        if type(amount) is not int:
            return
        with self.lock:
            self._reconstruction_raw_bytes = max(0, self._reconstruction_raw_bytes + amount)
            self.completed_bytes = min(
                self.max_bytes,
                max(self.completed_bytes, self._reconstruction_raw_bytes),
            )
            self._emit(force=force)

    def _count_transfer(self, amount: int, *, force: bool = False) -> None:
        if type(amount) is not int or amount <= 0:
            return
        with self.lock:
            self.transferred_bytes += amount
            self._emit(force=force)

    def _emit(self, *, force: bool = False) -> None:
        with self.lock:
            if self.completed_files is not None:
                self.completed_file_bytes = min(self.max_bytes, max(self.completed_file_bytes, self.completed_files()))
                self.completed_bytes = self.completed_file_bytes
            self.progress.emit(
                "download",
                "running",
                self.completed_bytes,
                self.max_bytes if self.completed_files is not None else None,
                transferred_bytes=self.transferred_bytes,
                force=force,
            )

    def bar_class(self):
        owner = self

        class SilentSnapshotProgressBar(hf_tqdm):
            def __init__(self, *args, **kwargs):
                self._capture_reconstruction = False
                self._capture_transfer = False
                self._closed = False
                name = kwargs.get("name")
                initial = kwargs.get("initial", 0)
                # The Hub factory may pass disable=False for a custom subclass.
                # Force silence independently of environment and TTY settings.
                kwargs["disable"] = True
                super().__init__(*args, **kwargs)
                self._capture_reconstruction = name == owner.BAR_NAME
                self._capture_transfer = name == owner.TRANSFER_BAR_NAME
                self._last_n = initial if type(initial) is int and initial >= 0 else 0
                self._last_n_lock = threading.Lock()
                if self._capture_reconstruction and self._last_n:
                    owner._count_reconstruction(self._last_n)
                # Initial transfer counts may describe a resumed file's bytes;
                # only live updates establish bytes received in this attempt.

            def update(self, amount=1):
                if not (self._capture_reconstruction or self._capture_transfer):
                    if owner.completed_files is not None:
                        owner._emit(force=True)
                    return
                if type(amount) is not int:
                    return
                with self._last_n_lock:
                    before = self._last_n
                    self._last_n = max(0, before + amount)
                    self.n = self._last_n
                    delta = self._last_n - before
                    if self._capture_reconstruction and delta:
                        owner._count_reconstruction(delta)
                    elif self._capture_transfer and delta > 0:
                        owner._count_transfer(delta)

            def update_transfer(self, _amount=1):
                # Network bytes can differ from reconstructed source bytes.
                return None

            def set_transfer_postfix_str(self, *_args, **_kwargs):
                return None

            def close(self):
                super().close()
                if (self._capture_reconstruction or self._capture_transfer) and not self._closed:
                    self._closed = True
                    owner._emit(force=True)

        return SilentSnapshotProgressBar

    def close(self) -> None:
        """Force the latest indeterminate count after a snapshot call returns."""
        with self.lock:
            self._emit(force=True)
