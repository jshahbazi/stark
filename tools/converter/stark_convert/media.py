"""Bounded parsing and private staging for static image request content."""

from __future__ import annotations

import base64
import binascii
import os
import re
import secrets
import stat
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

MAX_IMAGE_BYTES = 4 * 1024 * 1024
_MAX_ENCODED_BYTES = 4 * ((MAX_IMAGE_BYTES + 2) // 3)
_DATA_URL_PREFIX = "data:image/"
_DATA_URL = re.compile(r"data:image/(jpeg|png);base64,([A-Za-z0-9+/]*={0,2})\Z")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class MediaError(ValueError):
    """A content-free image validation or staging error."""

    _MESSAGES = {
        "image_too_large": "Image exceeds the supported size limit.",
        "invalid_image_url": "Image URL must be an inline PNG or JPEG data URL.",
        "invalid_image_detail": "Image detail must be omitted or set to auto.",
        "invalid_image": "Image data is invalid or does not match its media type.",
        "media_store_unavailable": "Image staging is unavailable.",
    }

    def __init__(self, code: str):
        if code not in self._MESSAGES:
            code = "invalid_image"
        self.code = code
        self.status = 413 if code == "image_too_large" else 400 if code.startswith("invalid_") else 503
        super().__init__(self._MESSAGES[code])


@dataclass(frozen=True, slots=True, repr=False)
class ImageData:
    """Decoded image bytes with a validated MIME type; never includes a path or digest."""

    data: bytes
    media_type: str


def parse_image_url(value: object) -> ImageData:
    """Parse one strict inline PNG/JPEG URL without decoding image pixels."""
    if not isinstance(value, dict) or set(value) not in ({"url"}, {"url", "detail"}):
        raise MediaError("invalid_image_url")
    if "detail" in value and value["detail"] != "auto":
        raise MediaError("invalid_image_detail")

    url = value["url"]
    if not isinstance(url, str):
        raise MediaError("invalid_image_url")
    # Bound work before ASCII validation, regex matching, or base64 decoding.
    if len(url) > len(_DATA_URL_PREFIX) + len("jpeg;base64,") + _MAX_ENCODED_BYTES:
        raise MediaError("image_too_large")
    try:
        url.encode("ascii")
    except UnicodeEncodeError:
        raise MediaError("invalid_image_url") from None

    match = _DATA_URL.fullmatch(url)
    if match is None:
        raise MediaError("invalid_image_url")
    subtype, encoded = match.groups()
    if not encoded:
        raise MediaError("invalid_image")
    if len(encoded) > _MAX_ENCODED_BYTES:
        raise MediaError("image_too_large")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise MediaError("invalid_image") from None
    if base64.b64encode(decoded).decode("ascii") != encoded:
        raise MediaError("invalid_image")
    if not decoded:
        raise MediaError("invalid_image")
    if len(decoded) > MAX_IMAGE_BYTES:
        raise MediaError("image_too_large")

    media_type = "image/jpeg" if subtype == "jpeg" else "image/png"
    if media_type == "image/png":
        valid_magic = decoded.startswith(_PNG_SIGNATURE)
    else:
        valid_magic = decoded.startswith(b"\xff\xd8\xff") and decoded.endswith(b"\xff\xd9")
    if not valid_magic:
        raise MediaError("invalid_image")
    return ImageData(data=decoded, media_type=media_type)


class PrivateMediaStore:
    """Per-worker private directory for short-lived image files."""

    def __init__(self):
        try:
            self._temporary = tempfile.TemporaryDirectory(prefix="stark-media-")
            self._root = Path(self._temporary.name)
            os.chmod(self._root, 0o700)
        except OSError:
            temporary = getattr(self, "_temporary", None)
            if temporary is not None:
                try:
                    temporary.cleanup()
                except OSError:
                    pass
            raise MediaError("media_store_unavailable") from None
        self._closed = False
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        """Directory to pass only to the paired private worker process."""
        return self._root

    @contextmanager
    def stage(self, image: ImageData) -> Iterator[dict[str, str | int]]:
        """Write an image under an opaque identifier and unlink it on every exit."""
        if not isinstance(image, ImageData) or not isinstance(image.data, bytes):
            raise MediaError("invalid_image")
        if image.media_type not in {"image/jpeg", "image/png"}:
            raise MediaError("invalid_image")
        if not image.data or len(image.data) > MAX_IMAGE_BYTES:
            raise MediaError("image_too_large" if image.data else "invalid_image")
        if image.media_type == "image/png":
            valid_magic = image.data.startswith(_PNG_SIGNATURE)
        else:
            valid_magic = image.data.startswith(b"\xff\xd8\xff") and image.data.endswith(b"\xff\xd9")
        if not valid_magic:
            raise MediaError("invalid_image")

        with self._lock:
            if self._closed:
                raise MediaError("media_store_unavailable")
            identifier = secrets.token_hex(16)
            path = self._root / identifier
            fd = None
            try:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                flags |= getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(path, flags, 0o600)
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise OSError
                os.fchmod(fd, 0o600)
                view = memoryview(image.data)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError
                    view = view[written:]
                os.close(fd)
                fd = None
            except OSError:
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise MediaError("media_store_unavailable") from None

        try:
            yield {"id": identifier, "byte_length": len(image.data), "media_type": image.media_type}
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                raise MediaError("media_store_unavailable") from None

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._temporary.cleanup()
        except OSError:
            raise MediaError("media_store_unavailable") from None

    def __enter__(self) -> PrivateMediaStore:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()
