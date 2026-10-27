"""Claim opaque attachments from the paired Swift process's private directory."""

from __future__ import annotations

import os
import re
import stat
import threading
from pathlib import Path

from .media import MAX_IMAGE_BYTES, ImageData, MediaError

_ID = re.compile(r"[0-9a-f]{32}\Z")


class DesktopAttachments:
    """Pin one initially empty owner-only directory; never accept client paths."""

    def __init__(self, root: Path):
        self._lock = threading.Lock()
        self._fd = None
        descriptor = None
        try:
            if not root.is_absolute() or root.resolve(strict=True) != root:
                raise OSError
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            info = os.fstat(descriptor)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700 or os.listdir(descriptor):
                raise OSError
            self._fd = descriptor
        except OSError:
            if descriptor is not None:
                os.close(descriptor)
            raise MediaError("media_store_unavailable") from None

    @staticmethod
    def _private(info):
        return (
            stat.S_ISREG(info.st_mode)
            and info.st_uid == os.getuid()
            and stat.S_IMODE(info.st_mode) == 0o600
            and info.st_nlink == 1
        )

    def claim(self, reference: object) -> ImageData:
        if (
            not isinstance(reference, dict)
            or set(reference) != {"id", "media_type", "byte_length"}
            or not isinstance(reference["id"], str)
            or _ID.fullmatch(reference["id"]) is None
            or reference["media_type"] not in ("image/png", "image/jpeg")
            or type(reference["byte_length"]) is not int
            or not 0 < reference["byte_length"] <= MAX_IMAGE_BYTES
        ):
            raise MediaError("invalid_image")
        with self._lock:
            descriptor = None
            try:
                if self._fd is None:
                    raise OSError
                name = reference["id"]
                descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self._fd)
                before = os.fstat(descriptor)
                if not self._private(before) or before.st_size != reference["byte_length"]:
                    raise OSError
                chunks = []
                remaining = before.st_size + 1
                while remaining:
                    chunk = os.read(descriptor, min(65536, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                data = b"".join(chunks)
                after = os.fstat(descriptor)
                linked = os.stat(name, dir_fd=self._fd, follow_symlinks=False)
                if (
                    any(
                        getattr(before, key) != getattr(after, key)
                        for key in (
                            "st_dev",
                            "st_ino",
                            "st_mode",
                            "st_uid",
                            "st_gid",
                            "st_nlink",
                            "st_size",
                            "st_mtime_ns",
                            "st_ctime_ns",
                        )
                    )
                    or (linked.st_dev, linked.st_ino) != (before.st_dev, before.st_ino)
                    or len(data) != before.st_size
                ):
                    raise OSError
                # Remove the pathname before any image decoding or inference.
                os.unlink(name, dir_fd=self._fd)
                media_type = reference["media_type"]
                valid = (
                    data.startswith(b"\x89PNG\r\n\x1a\n")
                    if media_type == "image/png"
                    else data.startswith(b"\xff\xd8\xff") and data.endswith(b"\xff\xd9")
                )
                if not valid:
                    raise MediaError("invalid_image")
                return ImageData(data=data, media_type=media_type)
            except OSError:
                raise MediaError("invalid_image") from None
            finally:
                if descriptor is not None:
                    os.close(descriptor)

    def clear(self):
        """Clean only opaque private regular files in the pinned directory."""
        with self._lock:
            if self._fd is None:
                return
            try:
                for name in os.listdir(self._fd):
                    if _ID.fullmatch(name) is None:
                        continue
                    try:
                        info = os.stat(name, dir_fd=self._fd, follow_symlinks=False)
                        if self._private(info):
                            os.unlink(name, dir_fd=self._fd)
                    except FileNotFoundError:
                        pass
            except OSError:
                raise MediaError("media_store_unavailable") from None

    def close(self):
        try:
            self.clear()
        finally:
            with self._lock:
                if self._fd is not None:
                    descriptor, self._fd = self._fd, None
                    try:
                        os.close(descriptor)
                    except OSError:
                        raise MediaError("media_store_unavailable") from None
