"""Losslessly align safetensors payloads in a new, atomically published model."""

import argparse
import ctypes
import hashlib
import json
import math
import os
import shutil
import stat
import struct
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

BUFFER_BYTES = 8 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024 * 1024
PROGRESS_BYTES = 1024 * 1024 * 1024
RECEIPT = "alignment-receipt.json"
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "U32": 4,
    "I32": 4,
    "U64": 8,
    "I64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2FNUZ": 1,
}


class AlignmentError(Exception):
    """A fixed diagnostic label, never model contents or an exception dump."""


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AlignmentError("duplicate_json_key")
        result[key] = value
    return result


def parse_json(data):
    try:
        return json.loads(data, object_pairs_hook=unique_object)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise AlignmentError("invalid_json") from error


def signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@dataclass(frozen=True)
class Container:
    path: Path
    header: bytes
    payload_bytes: int
    source_signature: tuple
    tensor_names: tuple

    @property
    def padding_bytes(self):
        return (-(8 + len(self.header))) % 16


def inspect_container(path):
    if path.is_symlink():
        raise AlignmentError("source_symlink_not_supported")
    with path.open("rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise AlignmentError("source_not_regular_file")
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise AlignmentError("truncated_header_prefix")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if not 0 < header_bytes <= MAX_HEADER_BYTES or 8 + header_bytes > info.st_size:
            raise AlignmentError("invalid_header_size")
        header = stream.read(header_bytes)
        if len(header) != header_bytes:
            raise AlignmentError("truncated_header")
    document = parse_json(header)
    if not isinstance(document, dict):
        raise AlignmentError("invalid_header_object")
    payload_bytes = info.st_size - 8 - header_bytes
    intervals = []
    names = []
    for name, tensor in document.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in tensor.items()
            ):
                raise AlignmentError("invalid_container_metadata")
            continue
        if not name or not isinstance(tensor, dict):
            raise AlignmentError("invalid_tensor_descriptor")
        dtype = tensor.get("dtype")
        shape = tensor.get("shape")
        offsets = tensor.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
            raise AlignmentError("unsupported_tensor_dtype")
        if (
            not isinstance(shape, list)
            or len(shape) > 128
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
        ):
            raise AlignmentError("invalid_tensor_shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
        ):
            raise AlignmentError("invalid_tensor_offsets")
        begin, end = offsets
        if not 0 <= begin <= end <= payload_bytes:
            raise AlignmentError("tensor_span_out_of_bounds")
        if math.prod(shape) * DTYPE_BYTES[dtype] != end - begin:
            raise AlignmentError("tensor_shape_size_mismatch")
        intervals.append((begin, end))
        names.append(name)
    if not names:
        raise AlignmentError("empty_tensor_inventory")
    previous = 0
    for begin, end in sorted(intervals):
        if begin != previous:
            raise AlignmentError("noncontiguous_tensor_payload")
        previous = end
    if previous != payload_bytes:
        raise AlignmentError("unreferenced_payload_bytes")
    return Container(path, header, payload_bytes, signature(info), tuple(names))


def emit(event):
    print(json.dumps(event, sort_keys=True), flush=True)


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(BUFFER_BYTES):
            digest.update(block)
    return digest.hexdigest()


def copy_metadata_file(source, destination):
    try:
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=BUFFER_BYTES)
            outgoing.flush()
            os.fsync(outgoing.fileno())
    except FileExistsError as error:
        raise AlignmentError("metadata_destination_collision") from error
    shutil.copystat(source, destination, follow_symlinks=False)


def verify_copy(container, destination, source_payload_sha256, progress):
    copied = inspect_container(destination)
    expected_header = container.header + b" " * container.padding_bytes
    if (
        copied.header != expected_header
        or copied.payload_bytes != container.payload_bytes
    ):
        raise AlignmentError("derived_container_changed")
    whole = hashlib.sha256()
    payload = hashlib.sha256()
    with destination.open("rb") as stream:
        whole.update(stream.read(8 + len(copied.header)))
        read_bytes = 0
        next_progress = PROGRESS_BYTES
        while block := stream.read(BUFFER_BYTES):
            whole.update(block)
            payload.update(block)
            read_bytes += len(block)
            if read_bytes >= next_progress:
                progress(
                    {
                        "event": "verify_progress",
                        "file": destination.name,
                        "payload_bytes": read_bytes,
                        "total_payload_bytes": copied.payload_bytes,
                    }
                )
                next_progress += PROGRESS_BYTES
    if (
        read_bytes != container.payload_bytes
        or payload.hexdigest() != source_payload_sha256
    ):
        raise AlignmentError("payload_hash_mismatch")
    return whole.hexdigest()


def copy_container(container, destination, progress):
    original = hashlib.sha256()
    derived = hashlib.sha256()
    payload = hashlib.sha256()
    original_prefix = struct.pack("<Q", len(container.header)) + container.header
    derived_prefix = struct.pack("<Q", len(container.header) + container.padding_bytes)
    derived_prefix += container.header + b" " * container.padding_bytes
    with container.path.open("rb") as source, destination.open("xb") as target:
        if signature(os.fstat(source.fileno())) != container.source_signature:
            raise AlignmentError("source_changed")
        if source.read(len(original_prefix)) != original_prefix:
            raise AlignmentError("source_header_changed")
        original.update(original_prefix)
        derived.update(derived_prefix)
        target.write(derived_prefix)
        copied_bytes = 0
        next_progress = PROGRESS_BYTES
        while block := source.read(BUFFER_BYTES):
            original.update(block)
            derived.update(block)
            payload.update(block)
            target.write(block)
            copied_bytes += len(block)
            if copied_bytes >= next_progress:
                progress(
                    {
                        "event": "copy_progress",
                        "file": destination.name,
                        "payload_bytes": copied_bytes,
                        "total_payload_bytes": container.payload_bytes,
                    }
                )
                next_progress += PROGRESS_BYTES
        if (
            copied_bytes != container.payload_bytes
            or signature(os.fstat(source.fileno())) != container.source_signature
        ):
            raise AlignmentError("source_changed")
        target.flush()
        os.fsync(target.fileno())
    payload_sha256 = payload.hexdigest()
    verified_sha256 = verify_copy(container, destination, payload_sha256, progress)
    if verified_sha256 != derived.hexdigest():
        raise AlignmentError("derived_file_hash_mismatch")
    if signature(container.path.stat()) != container.source_signature:
        raise AlignmentError("source_changed")
    shutil.copystat(container.path, destination, follow_symlinks=False)
    return {
        "file": container.path.name,
        "original_file_sha256": original.hexdigest(),
        "derived_file_sha256": verified_sha256,
        "payload_sha256": payload_sha256,
        "original_header_sha256": hashlib.sha256(container.header).hexdigest(),
        "derived_header_sha256": hashlib.sha256(
            container.header + b" " * container.padding_bytes
        ).hexdigest(),
        "payload_bytes": container.payload_bytes,
        "original_header_bytes": len(container.header),
        "derived_header_bytes": len(container.header) + container.padding_bytes,
        "padding_bytes": container.padding_bytes,
        "tensor_count": len(container.tensor_names),
        "data_offsets_unchanged": True,
    }


def publish_new_directory(staging, output):
    """Atomically publish without replacing even an existing empty directory."""
    if sys.platform == "win32":
        os.rename(staging, output)
        return
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        # Darwin sys/stdio.h: RENAME_EXCL = 0x00000004.
        rename = library.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        arguments = (os.fsencode(staging), os.fsencode(output), 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        # AT_FDCWD and RENAME_NOREPLACE.
        arguments = (-100, os.fsencode(staging), -100, os.fsencode(output), 1)
    else:
        raise AlignmentError("exclusive_atomic_publish_unavailable")
    rename.restype = ctypes.c_int
    if rename(*arguments) != 0:
        raise OSError(ctypes.get_errno(), "exclusive_atomic_publish_failed")


def align_model(model, output, progress=emit):
    model = Path(model).resolve(strict=True)
    output = Path(output).absolute()
    if not model.is_dir():
        raise AlignmentError("model_not_directory")
    if os.path.lexists(output):
        raise AlignmentError("destination_exists")
    if output.resolve().is_relative_to(model):
        raise AlignmentError("destination_inside_source")
    if not output.parent.is_dir():
        raise AlignmentError("destination_parent_missing")
    containers = []
    metadata = []
    metadata_signatures = {}
    names = {}
    destination_names = {RECEIPT.casefold(), "native-source.json"}
    for path in sorted(model.iterdir()):
        if path.name.startswith("."):
            if path.suffix in {".safetensors", ".gguf"}:
                raise AlignmentError("hidden_weight_shard")
            continue
        if path.is_symlink() or not path.is_file():
            raise AlignmentError("unsupported_source_entry")
        if path.name.casefold() in {RECEIPT.casefold(), "native-source.original.json"}:
            raise AlignmentError("source_already_has_alignment_receipt")
        destination_name = (
            "native-source.original.json"
            if path.name == "native-source.json"
            else path.name
        )
        if destination_name.casefold() in destination_names:
            raise AlignmentError("source_destination_name_collision")
        destination_names.add(destination_name.casefold())
        if path.suffix == ".safetensors":
            container = inspect_container(path)
            for name in container.tensor_names:
                if name in names:
                    raise AlignmentError("duplicate_tensor_across_shards")
                names[name] = path.name
            containers.append(container)
        else:
            if path.stat().st_size > MAX_METADATA_BYTES:
                raise AlignmentError("metadata_size_exceeded")
            metadata.append(path)
            metadata_signatures[path.name] = signature(path.stat())
    if not containers:
        raise AlignmentError("no_safetensors_shards")
    original_receipt = None
    original_receipt_path = model / "native-source.json"
    if original_receipt_path.exists():
        if original_receipt_path.stat().st_size > 1024 * 1024:
            raise AlignmentError("source_receipt_size_exceeded")
        original_receipt = parse_json(original_receipt_path.read_bytes())
        if not isinstance(original_receipt, dict):
            raise AlignmentError("invalid_source_receipt")
    index = model / "model.safetensors.index.json"
    if index.exists():
        document = parse_json(index.read_bytes())
        if not isinstance(document, dict) or document.get("weight_map") != names:
            raise AlignmentError("index_inventory_mismatch")
    required_bytes = sum(
        item.source_signature[2] + item.padding_bytes for item in containers
    )
    required_bytes += sum(path.stat().st_size for path in metadata)
    if shutil.disk_usage(output.parent).free < required_bytes:
        raise AlignmentError("insufficient_destination_space")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.alignment-", dir=output.parent)
    )
    published = False
    try:
        shard_receipts = []
        for container in containers:
            item = copy_container(container, staging / container.path.name, progress)
            shard_receipts.append(item)
            progress({"event": "verified_shard", **item})
        metadata_receipts = []
        for path in metadata:
            before = signature(path.stat())
            if before != metadata_signatures[path.name]:
                raise AlignmentError("source_metadata_changed")
            original_sha256 = hash_file(path)
            destination_name = (
                "native-source.original.json"
                if path.name == "native-source.json"
                else path.name
            )
            copy_metadata_file(path, staging / destination_name)
            if (
                signature(path.stat()) != before
                or hash_file(staging / destination_name) != original_sha256
            ):
                raise AlignmentError("metadata_copy_mismatch")
            metadata_receipts.append(
                {
                    "file": destination_name,
                    "source_file": path.name,
                    "sha256": original_sha256,
                    "bytes": before[2],
                }
            )
        for container in containers:
            if signature(container.path.stat()) != container.source_signature:
                raise AlignmentError("source_changed")
        if any(
            signature(path.stat()) != metadata_signatures[path.name]
            for path in metadata
        ):
            raise AlignmentError("source_metadata_changed")
        original_receipt_sha256 = next(
            (
                item["sha256"]
                for item in metadata_receipts
                if item["source_file"] == "native-source.json"
            ),
            None,
        )
        receipt = {
            "schema_version": 1,
            "operation": "safetensors_header_padding_only",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_model": str(model),
            "payload_alignment_bytes": 16,
            "tensor_payload_bytes_unchanged": True,
            "data_offsets_unchanged": True,
            "native_source_receipt_preserved": original_receipt is not None,
            "original_native_source": {
                "file": "native-source.original.json",
                "sha256": original_receipt_sha256,
            }
            if original_receipt is not None
            else None,
            "provenance": "Derived containers only. The original source receipt is preserved verbatim as native-source.original.json. The derived source receipt updates container size, binds this alignment receipt, and resets native loading and execution to unverified for the new layout.",
            "shards": shard_receipts,
            "metadata": metadata_receipts,
        }
        receipt_path = staging / RECEIPT
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if original_receipt is not None:
            derived_receipt = {
                **original_receipt,
                "native_loading": "unverified",
                "execution": "unverified",
                "weight_bytes": sum(
                    item.source_signature[2] + item.padding_bytes for item in containers
                ),
                "storage_alignment": {
                    "schema_version": 1,
                    "receipt": RECEIPT,
                    "receipt_sha256": hash_file(receipt_path),
                    "original_receipt": "native-source.original.json",
                    "original_receipt_sha256": original_receipt_sha256,
                },
            }
            with (staging / "native-source.json").open("x", encoding="utf-8") as stream:
                json.dump(derived_receipt, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        publish_new_directory(staging, output)
        published = True
        progress(
            {
                "event": "alignment_complete",
                "shards": len(containers),
                "payload_bytes": sum(item.payload_bytes for item in containers),
                "padding_bytes": sum(item.padding_bytes for item in containers),
                "receipt_sha256": hash_file(output / RECEIPT),
            }
        )
        return receipt
    finally:
        if not published:
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        align_model(args.model, args.output)
    except AlignmentError as error:
        emit({"event": "alignment_failed", "reason": str(error)})
        return 1
    except OSError as error:
        emit(
            {
                "event": "alignment_failed",
                "reason": "filesystem_operation_failed",
                "errno": error.errno,
            }
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
