"""Verify the pinned GLM-5.3 Flash UD-Q2_K_XL GGUF payloads once, streaming."""

# The CLI boundary sanitizes unexpected failures without dumping exception text.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path

WEIGHTS_REPO = "unsloth/GLM-5.3-Flash-GGUF"
WEIGHTS_REVISION = "a38483c8cd5df544f53d70fb281afe97369d5ab6"
METADATA_REPO = "zai-org/GLM-5.3-Flash"
METADATA_REVISION = "04c4e9e95c5da8862dced7e5056455116f83a7e0"
QUANTIZATION = "UD-Q2_K_XL"
ARCHITECTURE = "glm5_next"
CHUNK_BYTES = 4 * 1024 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024

# LFS object SHA-256 values from the pinned Hugging Face tree metadata.
SHARDS = {
    "GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf": (
        9_429_984,
        "2f32eba33ebc75a715e9c85a010faac4c67542e38e8970bd961d544bfe815f9e",
    ),
    "GLM-5.3-Flash-UD-Q2_K_XL-00002-of-00004.gguf": (
        49_294_975_936,
        "f4a9e1ab13d5d9620f5590c9a4aba4c169e6707a1554f3be3fe3112f80a66825",
    ),
    "GLM-5.3-Flash-UD-Q2_K_XL-00003-of-00004.gguf": (
        49_949_266_048,
        "330a8ad76c787b3ab6df062dd30abea7aafe6a0e3d5860e8720dfa4abb2434a5",
    ),
    "GLM-5.3-Flash-UD-Q2_K_XL-00004-of-00004.gguf": (
        9_466_399_584,
        "5c294f42edc5d69cf00a79ab444e61f0742e9933105fc5c85f97da54dab8946f",
    ),
}
TOTAL_WEIGHT_BYTES = sum(size for size, _ in SHARDS.values())
STAT_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")


class VerificationError(Exception):
    pass


def _snapshot(file_stat: os.stat_result) -> dict[str, int]:
    return {
        "device": file_stat.st_dev,
        "inode": file_stat.st_ino,
        "bytes": file_stat.st_size,
        "mtime_ns": file_stat.st_mtime_ns,
        "ctime_ns": file_stat.st_ctime_ns,
    }


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return all(getattr(left, field) == getattr(right, field) for field in STAT_FIELDS)


def _open_regular(path: Path, *, max_bytes: int | None = None) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
        before = os.fstat(fd)
    except OSError as error:
        raise VerificationError from error
    if not stat.S_ISREG(before.st_mode) or (max_bytes is not None and before.st_size > max_bytes):
        os.close(fd)
        raise VerificationError
    return fd, before


def _read_json(path: Path) -> tuple[dict, str]:
    fd, before = _open_regular(path, max_bytes=MAX_JSON_BYTES)
    try:
        digest = hashlib.sha256()
        chunks = []
        with os.fdopen(fd, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
                chunks.append(block)
                digest.update(block)
        after = os.fstat(fd)
        path_after = os.stat(path, follow_symlinks=False)
        if not _same_stat(before, after) or not _same_stat(after, path_after):
            raise VerificationError
        try:
            value = json.loads(b"".join(chunks))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VerificationError from error
        if not isinstance(value, dict):
            raise VerificationError
        return value, digest.hexdigest()
    finally:
        os.close(fd)


def _hash_regular(path: Path) -> tuple[str, dict[str, dict[str, int]]]:
    fd, before = _open_regular(path)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
                digest.update(block)
        after = os.fstat(fd)
        path_after = os.stat(path, follow_symlinks=False)
        if not _same_stat(before, after) or not _same_stat(after, path_after):
            raise VerificationError
        return digest.hexdigest(), {"before": _snapshot(before), "after": _snapshot(after)}
    finally:
        os.close(fd)


def _hash_shard(path: Path, model_digest) -> tuple[str, int, dict]:
    fd, before = _open_regular(path)
    try:
        digest = hashlib.sha256()
        name_bytes = path.name.encode("utf-8")
        model_digest.update(len(name_bytes).to_bytes(4, "big"))
        model_digest.update(name_bytes)
        model_digest.update(before.st_size.to_bytes(8, "big"))
        with os.fdopen(fd, "rb", closefd=False) as stream:
            while True:
                block = stream.read(CHUNK_BYTES)
                if not block:
                    break
                digest.update(block)
                model_digest.update(block)
        after = os.fstat(fd)
        path_after = os.stat(path, follow_symlinks=False)
        if not _same_stat(before, after) or not _same_stat(after, path_after):
            raise VerificationError
        return digest.hexdigest(), before.st_size, {"before": _snapshot(before), "after": _snapshot(after)}
    finally:
        os.close(fd)


def _metadata_pins(native: dict, config: dict, config_sha256: str) -> bool:
    weight_source = native.get("weights_source")
    metadata_source = native.get("metadata_source")
    hybrid = config.get("stark_native_hybrid")
    return (
        native.get("architecture") == ARCHITECTURE
        and native.get("quantization") == QUANTIZATION
        and native.get("config_sha256") == config_sha256
        and native.get("weight_bytes") == TOTAL_WEIGHT_BYTES
        and weight_source == {"repo": WEIGHTS_REPO, "revision": WEIGHTS_REVISION}
        and metadata_source == {"repo": METADATA_REPO, "revision": METADATA_REVISION}
        and config.get("model_type") == ARCHITECTURE
        and isinstance(hybrid, dict)
        and hybrid.get("source") == WEIGHTS_REPO
        and hybrid.get("revision") == WEIGHTS_REVISION
    )


def verify(model: Path, output: Path) -> tuple[dict | None, int, int]:
    if output.exists() or output.is_symlink():
        raise VerificationError
    if model.is_symlink() or not model.is_dir():
        raise VerificationError

    mismatches = 0
    shards_verified = 0
    bytes_verified = 0
    native, native_sha256 = _read_json(model / "native-source.json")
    config, config_sha256 = _read_json(model / "config.json")
    tokenizer_sha256, tokenizer_stat = _hash_regular(model / "tokenizer.json")
    if not _metadata_pins(native, config, config_sha256):
        mismatches += 1

    expected_names = set(SHARDS)
    actual_names = {path.name for path in model.glob("*.gguf")}
    if actual_names != expected_names:
        mismatches += 1

    files: dict[str, dict] = {}
    model_digest = hashlib.sha256(b"stark-glm53-ordered-shards-v1\0")
    for name, (expected_bytes, expected_sha256) in SHARDS.items():
        path = model / name
        try:
            actual_sha256, actual_bytes, stat_guard = _hash_shard(path, model_digest)
            matched = actual_sha256 == expected_sha256 and actual_bytes == expected_bytes
            shards_verified += 1
            bytes_verified += actual_bytes
            if not matched:
                mismatches += 1
            files[name] = {
                "matched": matched,
                "sha256": actual_sha256,
                "bytes": actual_bytes,
                "mtime_ns": stat_guard["before"]["mtime_ns"],
                "inode": stat_guard["before"]["inode"],
                "stat_guard": stat_guard,
            }
        except VerificationError:
            mismatches += 1

    if bytes_verified != TOTAL_WEIGHT_BYTES:
        mismatches += 1
    if mismatches:
        return None, shards_verified, mismatches

    source_config_sha256 = config_sha256
    receipt = {
        "schema_version": 1,
        "passed": True,
        "scope": "pinned_glm53_ud_q2_k_xl_gguf_integrity_and_source_metadata",
        "weights_source": {"repo": WEIGHTS_REPO, "revision": WEIGHTS_REVISION},
        "metadata_source": {"repo": METADATA_REPO, "revision": METADATA_REVISION},
        "quantization": QUANTIZATION,
        "architecture": ARCHITECTURE,
        "weight_bytes": TOTAL_WEIGHT_BYTES,
        "files": files,
        "model_sha256": model_digest.hexdigest(),
        "model_sha256_algorithm": "SHA-256 over ordered filename-length, UTF-8 filename, file-length, and shard bytes; shard files read once in 4 MiB chunks.",
        "config_sha256": source_config_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "native_source_sha256": native_sha256,
        "tokenizer_stat_guard": tokenizer_stat,
        "verifier_sha256": _hash_regular(Path(__file__))[0],
    }
    return receipt, shards_verified, 0


def write_private_new(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as error:
        raise VerificationError from error
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt, shards_verified, mismatches = verify(args.model, args.output)
        if receipt is not None:
            write_private_new(args.output, receipt)
            print(json.dumps({
                "passed": True,
                "shards_verified": shards_verified,
                "bytes_verified": TOTAL_WEIGHT_BYTES,
                "mismatches": 0,
            }, sort_keys=True))
            return 0
        print(json.dumps({
            "passed": False,
            "shards_verified": shards_verified,
            "bytes_verified": 0,
            "mismatches": max(1, mismatches),
        }, sort_keys=True))
        return 1
    except Exception:
        print(json.dumps({
            "passed": False,
            "shards_verified": 0,
            "bytes_verified": 0,
            "mismatches": 1,
        }, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
