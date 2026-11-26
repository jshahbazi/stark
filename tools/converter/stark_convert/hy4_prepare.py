#!/usr/bin/env python3
"""Verify the original AngelSlim Q3_K shards and stage Stark's small sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import struct
import tempfile
from contextlib import ExitStack
from pathlib import Path

from huggingface_hub import hf_hub_download

from .gguf_import import read_gguf_metadata

TRUNK = "Hy4-flash-Q3_K-00001-of-00002.gguf"
SOURCE_REPOSITORY = "AngelSlim/Hy4-flash-GGUF"
SOURCE_REVISION = "c50d4dd30be42f2b91138e5731de0e2f32ad56b2"
ARCHITECTURE = "hy4_flash"
SOURCE_MODEL_TYPE = "hy_v4_flash"
QUANTIZATION = "Q3_K"
SOURCE_RECEIPT = ".stark-hy4-source.json"
GGUF = {
    TRUNK: (
        92_010_389_504,
        "987b2902813ad94919e3e46e375bd8144d36113cced62d25f13013ccc88a3abb",
    ),
    "Hy4-flash-Q3_K-00002-of-00002.gguf": (
        63_360_616_128,
        "ecd6a0ecd7f258b1ee9211bf566261ae33efc1743ad40f58cb5004cf7b051839",
    ),
    "mmproj-Hy4-flash-bf16.gguf": (
        1_044_397_440,
        "8c33a48ef57480abf4854126964058a2015b5da3b773b14c847c995cf8899bda",
    ),
}
TENCENT_REVISION = "eccd3123eec73f2040f4c9ed0de1883593470510"
METADATA_REPOSITORY = "tencent/Hy4-flash"
METADATA = {
    "config.json": "1f56ac1a3063f926e91ec43842bfdc0fd6d4401ef91bcdecd1fc27dd506ea968",
    "tokenizer.json": "2e3f3d22065c2be5765ce99ab2410411ae338b32ce809ea20dcdb61d00198e83",
    "processor_config.json": "cc5cc63da33e0ebf53b8ac035879982ba18c47965f82a5f341bc82d98759dc1d",
    "chat_template.jinja": "11b12531403da5726b7e1f6003f479043b05999e6746d39af949cbcdfd715c35",
}
MAP_NAME = "engram_token_map.bin"
MAP_SHA = "99ed7bbd800d9d9c087aa2cbbf0fcc497839c519045f6f3bf9aca177cc1769b5"


def verify(path: Path, expected: str) -> None:
    with path.open("rb") as reader:
        if hashlib.file_digest(reader, "sha256").hexdigest() != expected:
            raise ValueError(f"Invalid digest for {path.name}")


def stage(path: Path, data: bytes, expected: str) -> None:
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Invalid source digest for {path.name}")
    if path.exists():
        verify(path, expected)
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as target:
        temporary = Path(target.name)
        try:
            target.write(data)
            target.flush()
            try:
                os.link(temporary, path)
            except FileExistsError:
                verify(path, expected)
        finally:
            temporary.unlink()


def fingerprint(metadata: os.stat_result) -> dict[str, int]:
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("Hy4 source must be a regular file")
    mtime_sec, mtime_nsec = divmod(metadata.st_mtime_ns, 1_000_000_000)
    ctime_sec, ctime_nsec = divmod(metadata.st_ctime_ns, 1_000_000_000)
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "size": metadata.st_size,
        "mtime_sec": mtime_sec,
        "mtime_nsec": mtime_nsec,
        "ctime_sec": ctime_sec,
        "ctime_nsec": ctime_nsec,
    }


def publish_receipt(model: Path, files: list[dict]) -> None:
    data = {
        "schema_version": 1,
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "files": files,
    }
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=model, delete=False) as target:
        temporary = Path(target.name)
        try:
            json.dump(data, target, separators=(",", ":"), sort_keys=True)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
            os.replace(temporary, model / SOURCE_RECEIPT)
        finally:
            temporary.unlink(missing_ok=True)


def prepare_sidecars(model: Path) -> None:
    metadata = read_gguf_metadata(model / TRUNK)
    token_map = metadata.get("hy_v4_flash.engram.token_map")
    if metadata.get("general.architecture") != "hy_v4_flash" or not isinstance(token_map, list):
        raise ValueError("Missing Hy4 Engram metadata")
    if len(token_map) != 120_832 or any(type(i) is not int or not -(1 << 31) <= i < (1 << 31) for i in token_map):
        raise ValueError("Invalid Hy4 Engram token map")
    stage(model / MAP_NAME, struct.pack(f"<{len(token_map)}i", *token_map), MAP_SHA)
    for filename, digest in METADATA.items():
        path = model / filename
        if path.exists():
            verify(path, digest)
            continue
        source = Path(hf_hub_download("tencent/Hy4-flash", filename, revision=TENCENT_REVISION))
        stage(path, source.read_bytes(), digest)
    print("Hy4 Q3_K sidecars verified")


def prepare(model: Path) -> None:
    # Hugging Face snapshots can point to blobs through symlinks. Verify the
    # opened regular files, then recheck both the handles and their names.
    with ExitStack() as stack:
        guards = []
        receipts = []
        for name, (size, digest) in GGUF.items():
            path = model / name
            descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            source = stack.enter_context(os.fdopen(descriptor, "rb"))
            verified = fingerprint(os.fstat(descriptor))
            if verified["size"] != size:
                raise ValueError(f"Invalid size for {name}")
            if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                raise ValueError(f"Invalid digest for {name}")
            if fingerprint(os.fstat(descriptor)) != verified:
                raise ValueError(f"Source changed during verification for {name}")
            guards.append((path, descriptor, verified))
            receipts.append({"name": name, "sha256": digest, "fingerprint": verified})
            print(f"Verified {name}", flush=True)
        prepare_sidecars(model)
        for path, descriptor, verified in guards:
            if fingerprint(os.fstat(descriptor)) != verified or fingerprint(path.stat()) != verified:
                raise ValueError(f"Source changed during preparation for {path.name}")
        publish_receipt(model, receipts)
    print("Hy4 Q3_K source receipt recorded")


def prepared_identity(model: Path) -> dict | None:
    """Recognize the exact prepared source without rehashing model weights."""
    try:
        receipt_path = model / SOURCE_RECEIPT
        receipt_stat = receipt_path.stat(follow_symlinks=False)
        if (
            receipt_path.is_symlink()
            or not stat.S_ISREG(receipt_stat.st_mode)
            or receipt_stat.st_uid != os.geteuid()
            or receipt_stat.st_mode & 0o077
            or receipt_stat.st_nlink != 1
            or receipt_stat.st_size > 8192
        ):
            return None
        receipt = json.loads(receipt_path.read_bytes())
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {"schema_version", "repository", "revision", "files"}
            or receipt["schema_version"] != 1
            or receipt["repository"] != SOURCE_REPOSITORY
            or receipt["revision"] != SOURCE_REVISION
            or not isinstance(receipt["files"], list)
            or len(receipt["files"]) != len(GGUF)
        ):
            return None
        fingerprint_keys = {"device", "inode", "size", "mtime_sec", "mtime_nsec", "ctime_sec", "ctime_nsec"}
        for record, (name, (size, digest)) in zip(receipt["files"], GGUF.items(), strict=True):
            path = model / name
            expected_fingerprint = record.get("fingerprint") if isinstance(record, dict) else None
            if (
                not isinstance(record, dict)
                or set(record) != {"name", "sha256", "fingerprint"}
                or record.get("name") != name
                or record.get("sha256") != digest
                or not isinstance(expected_fingerprint, dict)
                or set(expected_fingerprint) != fingerprint_keys
                or any(type(value) is not int for value in expected_fingerprint.values())
                or expected_fingerprint.get("size") != size
                or path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != size
                or fingerprint(path.stat()) != expected_fingerprint
            ):
                return None
        for name, digest in METADATA.items():
            verify(model / name, digest)
        verify(model / MAP_NAME, MAP_SHA)
        return {
            "repo": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "architecture": ARCHITECTURE,
            "weight_bytes": sum(size for size, _digest in GGUF.values()),
            "quantization": QUANTIZATION,
        }
    except (OSError, ValueError, TypeError, KeyError, UnicodeError):
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Directory containing the original three GGUF files")
    args = parser.parse_args()
    try:
        prepare(args.model)
    except OSError as exc:
        raise SystemExit(f"Hy4 preparation failed: {exc.strerror}") from None
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
