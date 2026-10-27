"""Automatic local image execution checks; never a reference-parity certificate."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import secrets
import stat
import struct
import sys
import threading
import time
import zlib
from pathlib import Path

import psutil

from ._build_metadata import file_hash
from .media import ImageData
from .vision_runtime import (
    _ARTIFACT_HASHES,
    FAMILY_PROFILE,
    _current_os_version,
    family_image_input_contract,
    image_input_contract,
)

CHECK_VERSION = 1
_MAX_RECEIPT_BYTES = 65536
_INVALIDATED = set()
_CHECKS = {
    "text_generation": True,
    "image_generation": True,
    "image_token_accounting": True,
    "recoverable_image_error": True,
    "fresh_image_replay": True,
    "staging_cleanup": True,
}


class ImageCheckError(Exception):
    """A fixed, content-free local validation failure."""


def preflight_family_memory(model: Path, config: dict) -> None:
    """Reject an obviously insufficient budget before starting the native loader.

    Available memory is an observation, not a reservation or peak-memory bound.
    The native loader and operation-specific allocation guards remain authoritative.
    """
    contract = config.get("artifact_contract", {})
    if not isinstance(contract, dict) or contract.get("schema_version") != 3:
        return
    family = contract.get("family")
    if (
        not isinstance(family, dict)
        or family != {"profile": FAMILY_PROFILE, "version": 1}
        or type(family.get("version")) is not int
    ):
        raise ImageCheckError("This image model family is not supported.")
    try:
        size = (model / "model.safetensors").stat().st_size
        available = psutil.virtual_memory().available
        if size <= 0 or type(available) is not int or available < size + 512 * 1024 * 1024:
            raise ImageCheckError("Not enough available memory to check this image model.")
    except OSError:
        raise ImageCheckError("Cannot inspect this image model's memory requirements.") from None


def _canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _dependency_versions() -> dict:
    return {name: importlib.metadata.version(name) for name in ("jinja2", "markupsafe", "tokenizers", "psutil")}


def check_identity(worker) -> dict:
    """Bind checks to admitted files, live implementation, policy, device and OS."""
    contract = worker.image_input
    if image_input_contract(contract) is None or not family_image_input_contract(contract):
        raise ImageCheckError("This worker does not implement the supported image family.")
    observed = file_hash(worker.executable)
    if observed != worker._launch_worker_sha256:
        raise ImageCheckError("The image worker changed while loading.")
    os_version = _current_os_version()
    if os_version is None:
        raise ImageCheckError("Cannot identify this image runtime.")
    package = Path(__file__).parent
    checker = hashlib.sha256()
    for name in ("image_checks.py", "chat.py", "vision_runtime.py", "media.py", "_build_metadata.py"):
        checker.update(name.encode())
        checker.update(bytes.fromhex(file_hash(package / name)))
    return {
        "check_version": CHECK_VERSION,
        "checker_sha256": checker.hexdigest(),
        "python_version": list(sys.version_info[:3]),
        "runtime_dependencies": _dependency_versions(),
        "family_profile": FAMILY_PROFILE,
        "family_version": 1,
        "artifact_schema_version": contract["artifact_schema_version"],
        "artifact_hashes": {key: contract[key] for key in sorted(_ARTIFACT_HASHES)},
        "worker_sha256": observed,
        "runtime_identity": dict(contract["runtime_identity"]),
        "os_version": os_version,
        "limits": {
            key: contract[key]
            for key in (
                "max_images",
                "max_compressed_bytes",
                "max_source_pixels",
                "max_processed_pixels",
                "max_patches",
                "max_image_tokens",
            )
        },
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError()
        result[key] = value
    return result


class CheckStore:
    """Owner-private, descriptor-pinned metadata receipts; no model content."""

    def __init__(self, root: Path):
        self._fd = None
        self._lock = threading.Lock()
        descriptor = None
        try:
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            info = os.fstat(descriptor)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise OSError
            self._fd = descriptor
        except OSError:
            if descriptor is not None:
                os.close(descriptor)
            raise ImageCheckError("Cannot open private image-check metadata.") from None

    @staticmethod
    def key(identity: dict) -> str:
        return hashlib.sha256(_canonical(identity)).hexdigest()

    def load(self, identity: dict) -> bool:
        name = self.key(identity) + ".json"
        descriptor = None
        with self._lock:
            if self._fd is None or self.key(identity) in _INVALIDATED:
                return False
            try:
                descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self._fd)
                before = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_uid != os.getuid()
                    or stat.S_IMODE(before.st_mode) != 0o600
                    or before.st_nlink != 1
                    or not 0 < before.st_size <= _MAX_RECEIPT_BYTES
                ):
                    return False
                with os.fdopen(descriptor, "rb") as stream:
                    descriptor = None
                    data = stream.read(_MAX_RECEIPT_BYTES + 1)
                    after = os.fstat(stream.fileno())
                if len(data) != before.st_size or any(
                    getattr(before, key) != getattr(after, key)
                    for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
                ):
                    return False
                value = json.loads(data, object_pairs_hook=_unique_object)
                expected = {
                    "schema_version": 1,
                    "evidence_level": "locally_tested",
                    "identity": identity,
                    "checks": _CHECKS,
                }
                return isinstance(value, dict) and _canonical(value) == _canonical(expected)
            except (OSError, ValueError, TypeError):
                return False
            finally:
                if descriptor is not None:
                    os.close(descriptor)

    def save(self, identity: dict) -> None:
        name = self.key(identity) + ".json"
        temporary = ".check-" + secrets.token_hex(16)
        with self._lock:
            if self._fd is None:
                raise ImageCheckError("Private image-check metadata is closed.")
            descriptor = None
            created = False
            try:
                descriptor = os.open(
                    temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self._fd
                )
                created = True
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    descriptor = None
                    stream.write(
                        _canonical(
                            {
                                "schema_version": 1,
                                "evidence_level": "locally_tested",
                                "identity": identity,
                                "checks": _CHECKS,
                            }
                        )
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, name, src_dir_fd=self._fd, dst_dir_fd=self._fd)
                _INVALIDATED.discard(self.key(identity))
            except OSError:
                raise ImageCheckError("Cannot save private image-check metadata.") from None
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                if created:
                    try:
                        os.unlink(temporary, dir_fd=self._fd)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        raise ImageCheckError("Cannot clean private image-check metadata.") from None

    def invalidate(self, identity: dict) -> None:
        _INVALIDATED.add(self.key(identity))
        with self._lock:
            if self._fd is None:
                raise ImageCheckError("Private image-check metadata is closed.")
            try:
                os.unlink(self.key(identity) + ".json", dir_fd=self._fd)
            except FileNotFoundError:
                pass
            except OSError:
                raise ImageCheckError("Cannot clear private image-check metadata.") from None

    def close(self):
        with self._lock:
            if self._fd is not None:
                descriptor, self._fd = self._fd, None
                try:
                    os.close(descriptor)
                except OSError:
                    raise ImageCheckError("Cannot close private image-check metadata.") from None


def _fixture() -> ImageData:
    """One fixed public synthetic RGB image, generated without a second decoder."""
    side = 256

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * side) * side))
        + chunk(b"IEND", b"")
    )
    return ImageData(data, "image/png")


def _valid_reply(reply: dict, maximum: int) -> bool:
    return (
        isinstance(reply, dict)
        and "error" not in reply
        and isinstance(reply.get("text"), str)
        and bool(reply["text"].strip())
        and type(reply.get("input_tokens")) is int
        and reply["input_tokens"] > 0
        and type(reply.get("output_tokens")) is int
        and 0 < reply["output_tokens"] <= maximum
        and reply.get("finish_reason") in {"eos", "stop_token", "stop_sequence", "length", "context"}
    )


def verify_images(worker, ready: dict, *, directory: Path | None = None) -> bool:
    """Run or reuse local checks. Return whether an existing receipt was reused."""
    from .chat import ChatImageError, load_template

    store = None
    identity = None
    try:
        identity = check_identity(worker)
        context = ready.get("max_context_tokens")
        maximum = ready.get("max_new_tokens")
        if type(context) is not int or context < 512 or type(maximum) is not int or maximum < 1:
            raise ImageCheckError("The image worker reports unsupported token limits.")
        if directory is not None:
            store = CheckStore(directory)
            if store.load(identity):
                if (
                    worker.process.poll() is not None
                    or getattr(worker, "_closed", False)
                    or check_identity(worker) != identity
                ):
                    raise ImageCheckError("Image checks were interrupted.")
                worker._image_check_identity = identity
                return True
        codec, template = load_template(worker.model_path, expected_hashes=worker.image_input)
        context, maximum = min(context, 2048), min(maximum, 8)
        options = {"context_tokens": context, "max_new_tokens": maximum, "sampling": {"mode": "greedy"}}
        worker._verification_deadline = time.monotonic() + min(worker.timeout, 120)

        def prompt(image=False):
            content = (
                [{"type": "image"}, {"type": "text", "text": "Describe the image briefly."}] if image else "Hello."
            )
            return template.render(
                messages=[{"role": "user", "content": content}],
                tools=None,
                add_generation_prompt=True,
                enable_thinking=False,
                bos_token=codec.get("bos_token") or "",
                eos_token=codec.get("eos_token") or "",
            )

        worker._image_deadline = worker._verification_deadline
        text = worker.request({"operation": "generate", "rendered_prompt": prompt(), **options})
        if not _valid_reply(text, maximum) or text["input_tokens"] + maximum > context:
            raise ImageCheckError("The model did not pass the text execution check.")
        image_prompt = prompt(True)
        tokens = worker.request({"operation": "tokenize", "text": image_prompt, "add_special_tokens": False})
        base_count = tokens.get("token_count")
        if (
            "error" in tokens
            or type(base_count) is not int
            or base_count < 3
            or not isinstance(tokens.get("token_ids"), list)
            or len(tokens["token_ids"]) != base_count
            or any(type(token) is not int or not 0 <= token < 248320 for token in tokens["token_ids"])
        ):
            raise ImageCheckError("The model did not pass image prompt token accounting.")
        try:
            worker.request_image(image_prompt, ImageData(b"\x89PNG\r\n\x1a\n\0", "image/png"), options)
        except ChatImageError:
            pass
        else:
            raise ImageCheckError("The model did not reject an invalid image safely.")
        counts = []
        for _ in range(2):
            reply = worker.request_image(image_prompt, _fixture(), options)
            if not _valid_reply(reply, maximum) or reply["input_tokens"] + maximum > context:
                raise ImageCheckError("The model did not pass the image execution check.")
            image_tokens = getattr(worker, "_last_image_tokens", None)
            if type(image_tokens) is not int or image_tokens != 64 or reply["input_tokens"] != base_count + 63:
                raise ImageCheckError("The model did not pass image token accounting.")
            if list(worker.media.root.iterdir()):
                raise ImageCheckError("The model did not clear staged image data.")
            counts.append((reply["input_tokens"], image_tokens))
        if counts[0] != counts[1] or check_identity(worker) != identity:
            raise ImageCheckError("Image check identity or fresh-request accounting changed.")
        if worker.process.poll() is not None or getattr(worker, "_closed", False):
            raise ImageCheckError("Image checks were interrupted.")
        if store is not None:
            store.save(identity)
        worker._image_check_identity = identity
        return False
    except BaseException:
        if store is not None and identity is not None:
            store.invalidate(identity)
        raise
    finally:
        worker._verification_deadline = None
        worker._image_deadline = None
        if store is not None:
            store.close()


def invalidate_checks(worker) -> None:
    identity, directory = getattr(worker, "_image_check_identity", None), getattr(worker, "image_check_dir", None)
    worker._image_check_identity = None
    if identity is not None:
        _INVALIDATED.add(CheckStore.key(identity))
    if identity is not None and directory is not None:
        store = CheckStore(directory)
        try:
            store.invalidate(identity)
        finally:
            store.close()
