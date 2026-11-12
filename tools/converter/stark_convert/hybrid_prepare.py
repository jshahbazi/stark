"""Prepare pinned original-format weights for Stark's native hybrid worker.

This path preserves quantized bytes. It downloads metadata and weights only;
repository Python, speculative sidecars and foreign inference engines are never
requested or executed. Native load and execution remain separate checks.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import struct
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from . import qwen38_external
from .cli import _publish
from .progress import FDProgressSink, ProgressCallback, ProgressReporter, SnapshotDownloadProgress, reporter
from .sources import SourceSnapshot, _verified_hashes, huggingface_repository
from .streaming import ConversionError, UnsupportedConversion, load_checkpoint, read_safetensors_header

SOURCES = {
    "flint-int4": {
        "architecture": "flint",
        "weights_repo": "RecursiveSI/flint-int4",
        "weights_revision": "3b93f9053b530488405ae9fca51d4900d6ce21f0",
        "weights_patterns": ["*.safetensors", "README.md"],
        "metadata_repo": "RecursiveSI/flint-int4",
        "metadata_revision": "3b93f9053b530488405ae9fca51d4900d6ce21f0",
        "weight_bytes": 22_568_912_600,
        "quantization": "MLX affine 4-bit, 8-bit embeddings",
        "weight_shards": 5,
    },
    "glm53-q2": {
        "architecture": "glm5_next",
        "weights_repo": "unsloth/GLM-5.3-Flash-GGUF",
        "weights_revision": "a38483c8cd5df544f53d70fb281afe97369d5ab6",
        "weights_patterns": ["UD-Q2_K_XL/*.gguf", "README.md"],
        "metadata_repo": "zai-org/GLM-5.3-Flash",
        "metadata_revision": "04c4e9e95c5da8862dced7e5056455116f83a7e0",
        "weight_bytes": 108_720_071_552,
        "quantization": "UD-Q2_K_XL",
        "weight_shards": 4,
    },
    "qwen38-mixed": {
        "architecture": "qwen4_exp",
        "weights_repo": "pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit",
        "weights_revision": "b2c422f3c643e36f04227a64d61796b44a4b1029",
        "weights_patterns": ["*.safetensors", "model.safetensors.index.json", "README.md"],
        "metadata_repo": "pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit",
        "metadata_revision": "b2c422f3c643e36f04227a64d61796b44a4b1029",
        "weight_bytes": 106_195_031_237,
        "quantization": "MLX mixed 4/8-bit",
    },
    "qwen38-lily-q4-xl": {
        "architecture": "qwen4_exp",
        "weights_repo": "fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl",
        "weights_revision": "0f94842e3682b49acace4daeb4cf2e65a6f94f83",
        "weights_patterns": [
            "model-*-of-*.safetensors",
            "model.safetensors.index.json",
            "README.md",
            "LICENSE",
            "vocab.json",
            "merges.txt",
        ],
        "metadata_repo": "fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl",
        "metadata_revision": "0f94842e3682b49acace4daeb4cf2e65a6f94f83",
        "weight_bytes": 106_905_155_192,
        "weight_shards": 55,
        "quantization": "Lily q4-xl (Q4 GDN; Q8 attn/shared/head/embed)",
        "required_files": ["LICENSE", "README.md"],
    },
    "qwen38-mlx-serve-mixed": {
        "architecture": "qwen4_exp",
        "weights_repo": "ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit",
        "weights_revision": "788c07a8725642bccf13a0a3b8a21954ab84b4f2",
        "weights_patterns": ["*.safetensors", "model.safetensors.index.json", "ngram_table.bin", "README.md"],
        "metadata_repo": "ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit",
        "metadata_revision": "788c07a8725642bccf13a0a3b8a21954ab84b4f2",
        "weight_bytes": 107_303_406_192,
        "weight_shards": 101,
        "ngram_bytes": 32_000_153_976,
        "ngram_in_weight_bytes": True,
        "quantization": "MLX-Serve mixed 4/8-bit",
    },
}
_METADATA = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
    "LICENSE",
    "LICENSE.txt",
]


def _json(path: Path, limit: int = 1024 * 1024):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("invalid metadata file")
    try:
        return json.loads(path.read_bytes())
    except RecursionError as error:
        raise ValueError("metadata JSON nesting exceeds limit") from error


def _ids(value):
    if type(value) is int:
        return [value]
    if isinstance(value, list) and value and all(type(item) is int for item in value):
        return value
    raise ValueError("invalid EOS metadata")


def finish_metadata(directory: Path, choice: str) -> dict:
    """Bind the preserved architecture and tokenizer template for CLI chat."""
    receipt = _finish_metadata(directory, SOURCES[choice])
    (directory / "native-source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def _finish_metadata(directory: Path, source: dict) -> dict:
    config = _json(directory / "config.json")
    if config.get("model_type") != source["architecture"]:
        raise ValueError("source architecture changed")
    text = config.get("text_config", config)
    tokenizer = _json(directory / "tokenizer_config.json", 2 * 1024 * 1024)
    template_file = directory / "chat_template.jinja"
    if template_file.is_file() and template_file.stat().st_size > 256 * 1024:
        raise ValueError("excessive chat template")
    template = template_file.read_text() if template_file.is_file() else tokenizer.get("chat_template")
    if isinstance(template, list):
        template = next((item.get("template") for item in template if item.get("name") == "default"), None)
    if not isinstance(template, str) or not 0 < len(template.encode()) <= 256 * 1024:
        raise ValueError("missing or excessive chat template")
    stops = _ids(text.get("eos_token_id"))
    generation_path = directory / "generation_config.json"
    if generation_path.is_file():
        generation = _json(generation_path)
        if generation.get("eos_token_id") is not None:
            stops.extend(_ids(generation["eos_token_id"]))
    vocab = text.get("vocab_size")
    if type(vocab) is not int or not 0 < vocab <= 1_000_000 or any(not 0 <= value < vocab for value in stops):
        raise ValueError("invalid model vocabulary or EOS IDs")
    text["eos_token_id"] = list(dict.fromkeys(stops))
    codec = {"type": "ChatCodecConfig", "prompt_template": template}
    for field in ("bos_token", "eos_token"):
        token = tokenizer.get(field)
        if isinstance(token, dict):
            token = token.get("content")
        if token is not None:
            if not isinstance(token, str) or not 0 < len(token.encode()) <= 8192:
                raise ValueError("invalid special-token metadata")
            codec[field] = token
    config["token_codec_config"] = codec
    # Runtime selection is by parsed architecture. This marker is solely an
    # informational receipt; it never grants image or scoring capabilities.
    config["stark_native_hybrid"] = {
        "schema_version": 1,
        "source": source["weights_repo"],
        "revision": source["weights_revision"],
    }
    encoded = (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode()
    (directory / "config.json").write_bytes(encoded)
    return {
        "schema_version": 1,
        "artifact_mode": "text",
        "architecture": source["architecture"],
        "quantization": source["quantization"],
        "weights_source": {"repo": source["weights_repo"], "revision": source["weights_revision"]},
        "metadata_source": {"repo": source["metadata_repo"], "revision": source["metadata_revision"]},
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "native_loading": "unverified",
        "execution": "unverified",
        "weight_bytes": sum(
            path.stat().st_size for path in directory.iterdir() if path.suffix in {".gguf", ".safetensors"}
        )
        + (source.get("ngram_bytes", 0) if source.get("ngram_in_weight_bytes") else 0),
    }


_EXTERNAL_LAYOUT = "qwen38-external-affine"


def _copy_snapshot_file(source: Path, target: Path) -> None:
    # A same-volume APFS clone isolates the installed bytes from the source
    # without requiring a second allocation for a hundred-gigabyte checkpoint.
    if sys.platform == "darwin":
        clone = ctypes.CDLL(None, use_errno=True).clonefile
        clone.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
        if clone(os.fsencode(source), os.fsencode(target), 0) == 0:
            return
        code = ctypes.get_errno()
        if code not in {errno.EXDEV, errno.ENOTSUP, errno.ENOSYS}:
            raise OSError(code, "Cannot clone model file")
    shutil.copyfile(source, target)


def stage_external_qwen(snapshot: SourceSnapshot, staging: Path) -> dict:
    """Prepare an independently owned native artifact from a bound source."""
    snapshot.assert_unchanged(verify_content=False)
    records = snapshot.manifest["files"]
    targets = []
    for record in records:
        name = record["filename"]
        target = staging / name
        _copy_snapshot_file(snapshot.resolve_file(name), target)
        target.chmod(0o600)
        targets.append(target)
    _, hashes = _verified_hashes(targets)
    if hashes != [record["sha256"] for record in records]:
        raise ConversionError("External Qwen3.8 source changed during preparation")
    snapshot.assert_unchanged(verify_content=False)
    tensors = load_checkpoint(staging)
    table = staging / qwen38_external.NGRAM_TABLE
    _, planes = read_safetensors_header(table)
    qwen38_external.validate_layout(
        _json(staging / "config.json"),
        {name: (info.dtype, info.shape) for name, info in tensors.items()},
        {name: (info.dtype, info.shape) for name, info in planes.items()},
    )
    source = {
        "architecture": "qwen4_exp",
        "weights_repo": snapshot.repo,
        "weights_revision": snapshot.revision,
        "metadata_repo": snapshot.repo,
        "metadata_revision": snapshot.revision,
        "quantization": qwen38_external.QUANTIZATION,
        "ngram_bytes": table.stat().st_size,
        "ngram_in_weight_bytes": True,
    }
    receipt = {
        **_finish_metadata(staging, source),
        **snapshot.receipt(),
        "layout": _EXTERNAL_LAYOUT,
        "weight_files": {
            record["filename"]: {"size": record["size"], "sha256": record["sha256"]}
            for record in records
            if record["role"] in {"weights", "weight_index", "ngram_table"}
        },
        "status": "prepared",
    }
    (staging / "native-source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if prepared_identity(staging) is None:
        raise ConversionError("External Qwen3.8 prepared identity could not be verified")
    return receipt


def stage_retained_qwen(snapshot: SourceSnapshot, staging: Path) -> dict:
    """Preserve a compatible Flash Next checkpoint and prepare its image route.

    The original tensors, MTP heads and recurrent-state implementation remain
    unchanged. Image transactions still bypass text-prefix state in the worker.
    """
    from . import qwen38_retained
    from .hybrid_vision import PROCESSOR_BYTES, prepare_image_artifact
    from .hybrid_vision import _read as read_image_metadata

    if snapshot.artifact_mode != "vision-language":
        raise UnsupportedConversion("Retained Flash Next image preparation requires image mode")
    if (
        not isinstance(snapshot.repo, str)
        or huggingface_repository(snapshot.repo) != snapshot.repo
        or not isinstance(snapshot.revision, str)
        or len(snapshot.revision) != 40
        or any(character not in "0123456789abcdef" for character in snapshot.revision)
    ):
        raise UnsupportedConversion("Retained Flash Next images require an immutable Hugging Face source revision")
    snapshot.assert_unchanged(verify_content=False)
    records = snapshot.manifest["files"]
    targets = []
    for record in records:
        name = record["filename"]
        target = staging / name
        _copy_snapshot_file(snapshot.resolve_file(name), target)
        target.chmod(0o600)
        targets.append(target)
    _, hashes = _verified_hashes(targets)
    if hashes != [record["sha256"] for record in records]:
        raise ConversionError("Retained Flash Next source changed during preparation")
    snapshot.assert_unchanged(verify_content=False)
    config = _json(staging / "config.json")
    processing_names = [record["filename"] for record in records]
    derived_processor = qwen38_retained.family_default_processor_metadata(config, processing_names)
    if derived_processor is not None:
        processor = staging / "preprocessor_config.json"
        with processor.open("xb") as stream:
            stream.write(PROCESSOR_BYTES)
        processor.chmod(0o600)
        processing_names.append(processor.name)
    qwen38_retained.validate_processing_metadata(
        processing_names, lambda name, maximum: read_image_metadata(staging / name, maximum)
    )
    tensors = load_checkpoint(staging)
    table = staging / qwen38_external.NGRAM_TABLE
    planes = read_safetensors_header(table)[1] if qwen38_external.is_external_affine(config) else None
    qwen38_retained.validate_layout(config, tensors, planes)
    if qwen38_retained.auxiliary_metadata_filenames(config):
        from . import qwen38_kquant

        qwen38_kquant.validate_manifest(config, _json(staging / qwen38_kquant.PLE_MANIFEST), tensors)
    index = staging / qwen38_external.INDEX
    if not index.exists():
        index.write_text(
            json.dumps(
                {"weight_map": {name: info.path.name for name, info in sorted(tensors.items())}}, sort_keys=True
            )
            + "\n"
        )
    weight_files = {
        record["filename"]: {"size": record["size"], "sha256": record["sha256"]}
        for record in records
        if record["role"] in {"weights", "weight_index", "ngram_table", "ple_manifest"}
    }
    weight_files[index.name] = {"size": index.stat().st_size, "sha256": hashlib.sha256(index.read_bytes()).hexdigest()}
    source = {
        "architecture": "qwen4_exp",
        "weights_repo": snapshot.repo,
        "weights_revision": snapshot.revision,
        "metadata_repo": snapshot.repo,
        "metadata_revision": snapshot.revision,
        "quantization": qwen38_retained.quantization_name(config),
        "ngram_bytes": table.stat().st_size if table.exists() else 0,
        "ngram_in_weight_bytes": table.exists(),
    }
    receipt = {
        **_finish_metadata(staging, source),
        **snapshot.receipt(),
        "layout": qwen38_retained.LAYOUT,
        "weight_files": weight_files,
        "status": "prepared",
    }
    if derived_processor is not None:
        receipt["derived_metadata"] = {"preprocessor_config.json": derived_processor}
    (staging / "native-source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if prepared_identity(staging) is None:
        raise ConversionError("Retained Flash Next prepared identity could not be verified")
    prepare_image_artifact(staging)
    return receipt


def _sha256_text(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _aligned_storage_identity(directory: Path, receipt: dict, source: dict, files: list[Path], size: int) -> bool:
    """Check bounded container provenance; payload hashes remain recorded claims.

    This enables the native loader to inspect a losslessly repacked artifact.
    It does not rehash model payloads or qualify native loading or execution.
    """
    extension = receipt.get("storage_alignment")
    if not isinstance(extension, dict) or extension.get("schema_version") != 1:
        return False
    if (
        extension.get("receipt") != "alignment-receipt.json"
        or extension.get("original_receipt") != "native-source.original.json"
        or any(path.suffix != ".safetensors" for path in files)
    ):
        return False
    alignment_path = directory / "alignment-receipt.json"
    alignment = _json(alignment_path, 4 * 1024 * 1024)
    if extension.get("receipt_sha256") != hashlib.sha256(alignment_path.read_bytes()).hexdigest():
        return False
    original_path = directory / "native-source.original.json"
    original = _json(original_path)
    original_sha256 = hashlib.sha256(original_path.read_bytes()).hexdigest()
    if (
        extension.get("original_receipt_sha256") != original_sha256
        or alignment.get("original_native_source") != {"file": original_path.name, "sha256": original_sha256}
        or type(original.get("weight_bytes")) is not int
        or original.get("weight_bytes") != source["weight_bytes"]
        or "storage_alignment" in original
        or any(receipt.get(field) != "unverified" for field in ("native_loading", "execution"))
        or {
            key: value for key, value in original.items() if key not in {"weight_bytes", "native_loading", "execution"}
        }
        != {
            key: value
            for key, value in receipt.items()
            if key not in {"weight_bytes", "storage_alignment", "native_loading", "execution"}
        }
        or alignment.get("schema_version") != 1
        or alignment.get("operation") != "safetensors_header_padding_only"
        or alignment.get("payload_alignment_bytes") != 16
        or alignment.get("tensor_payload_bytes_unchanged") is not True
        or alignment.get("data_offsets_unchanged") is not True
        or alignment.get("native_source_receipt_preserved") is not True
    ):
        return False
    shards = alignment.get("shards")
    if not isinstance(shards, list) or len(shards) != len(files):
        return False
    by_name = {path.name: path for path in files}
    seen = set()
    original_size = 0
    derived_size = 0
    for shard in shards:
        if not isinstance(shard, dict) or shard.get("file") not in by_name or shard["file"] in seen:
            return False
        seen.add(shard["file"])
        fields = ("original_header_bytes", "derived_header_bytes", "padding_bytes", "payload_bytes", "tensor_count")
        if any(type(shard.get(field)) is not int for field in fields):
            return False
        old_header, new_header, padding, payload, tensor_count = (shard[field] for field in fields)
        if (
            not 0 < old_header <= 64 * 1024 * 1024
            or not 0 <= padding <= 15
            or new_header != old_header + padding
            or padding != (-(8 + old_header)) % 16
            or payload < 0
            or tensor_count <= 0
            or shard.get("data_offsets_unchanged") is not True
            or any(
                not _sha256_text(shard.get(field))
                for field in (
                    "original_file_sha256",
                    "derived_file_sha256",
                    "payload_sha256",
                    "original_header_sha256",
                    "derived_header_sha256",
                )
            )
        ):
            return False
        path = by_name[shard["file"]]
        if path.stat().st_size != 8 + new_header + payload:
            return False
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8 or struct.unpack("<Q", prefix)[0] != new_header:
                return False
            header = stream.read(new_header)
        if len(header) != new_header or (padding and header[-padding:] != b" " * padding):
            return False
        original_header = header[:old_header]
        if (
            hashlib.sha256(header).hexdigest() != shard["derived_header_sha256"]
            or hashlib.sha256(original_header).hexdigest() != shard["original_header_sha256"]
        ):
            return False
        inventory = json.loads(original_header)
        if not isinstance(inventory, dict) or sum(name != "__metadata__" for name in inventory) != tensor_count:
            return False
        original_size += 8 + old_header + payload
        derived_size += 8 + new_header + payload
    return original_size == source["weight_bytes"] and derived_size == size


def prepared_identity(directory: Path) -> dict | None:
    """Recognize a prepared local folder; the native loader checks tensors.

    This bounded inspection enables importing the preserved-format directory.
    It does not mark the checkpoint executed or grant image qualification.
    """
    try:
        config_path = directory / "config.json"
        config = _json(config_path)
        receipt = _json(directory / "native-source.json")
        marker = config.get("stark_native_hybrid")
        if not isinstance(marker, dict) or marker.get("schema_version") != 1:
            return None
        if receipt.get("layout") == _EXTERNAL_LAYOUT:
            return _external_identity(directory, config, receipt, marker)
        from . import qwen38_retained

        if receipt.get("layout") == qwen38_retained.LAYOUT:
            return _external_identity(directory, config, receipt, marker, retained=True)
        source = next(
            (
                value
                for value in SOURCES.values()
                if value["weights_repo"] == marker.get("source")
                and value["weights_revision"] == marker.get("revision")
            ),
            None,
        )
        if source is None or config.get("model_type") != source["architecture"]:
            return None
        if (
            receipt.get("schema_version") != 1
            or receipt.get("artifact_mode") != "text"
            or receipt.get("architecture") != source["architecture"]
            or receipt.get("config_sha256") != hashlib.sha256(config_path.read_bytes()).hexdigest()
            or receipt.get("weights_source")
            != {"repo": source["weights_repo"], "revision": source["weights_revision"]}
        ):
            return None
        codec = config.get("token_codec_config")
        if not isinstance(codec, dict) or codec.get("type") != "ChatCodecConfig" or not codec.get("prompt_template"):
            return None
        extension = ".gguf" if source["architecture"] == "glm5_next" else ".safetensors"
        files = list(directory.glob("*" + extension))
        if not 1 <= len(files) <= 512 or any(path.is_symlink() or not path.is_file() for path in files):
            return None
        if "weight_shards" in source and len(files) != source["weight_shards"]:
            return None
        required_files = source.get("required_files", [])
        if any(not (directory / name).is_file() or (directory / name).is_symlink() for name in required_files):
            return None
        size = sum(path.stat().st_size for path in files)
        if source.get("ngram_in_weight_bytes"):
            size += source["ngram_bytes"]
        if receipt.get("weight_bytes") != size:
            return None
        if "storage_alignment" in receipt:
            if not _aligned_storage_identity(directory, receipt, source, files, size):
                return None
        elif size != source["weight_bytes"]:
            return None
        if not (directory / "tokenizer.json").is_file():
            return None
        if "ngram_bytes" in source:
            table = directory / "ngram_table.bin"
            if table.is_symlink() or not table.is_file() or table.stat().st_size != source["ngram_bytes"]:
                return None
        return {
            "repo": source["weights_repo"],
            "revision": source["weights_revision"],
            "architecture": source["architecture"],
            "weight_bytes": size,
            "quantization": source["quantization"],
        }
    except (OSError, ValueError, TypeError, AttributeError, RecursionError):
        return None


def _external_identity(directory: Path, config: dict, receipt: dict, marker: dict, *, retained=False) -> dict | None:
    from . import qwen38_retained

    repo, revision = marker.get("source"), marker.get("revision")
    identity = {"repo": repo, "revision": revision}
    codec = config.get("token_codec_config")
    external = qwen38_external.is_external_affine(config)
    supported = qwen38_retained.is_retained_affine(config) if retained else external
    quantization = qwen38_retained.quantization_name(config) if retained else qwen38_external.QUANTIZATION
    if (
        not supported
        or receipt.get("schema_version") != 1
        or receipt.get("artifact_mode") != ("vision-language" if retained else "text")
        or receipt.get("architecture") != "qwen4_exp"
        or receipt.get("quantization") != quantization
        or receipt.get("config_sha256") != hashlib.sha256((directory / "config.json").read_bytes()).hexdigest()
        or receipt.get("weights_source") != identity
        or receipt.get("metadata_source") != identity
        or repo is not None
        and huggingface_repository(repo) != repo
        or revision is not None
        and (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        )
        or not isinstance(codec, dict)
        or codec.get("type") != "ChatCodecConfig"
        or not isinstance(codec.get("prompt_template"), str)
        or not codec["prompt_template"]
        or (directory / "tokenizer.json").is_symlink()
        or not (directory / "tokenizer.json").is_file()
    ):
        return None
    records = receipt.get("weight_files")
    shards = sorted(path.name for path in directory.glob("*.safetensors"))
    index = directory / qwen38_external.INDEX
    indexed = index.exists() or index.is_symlink()
    names = {*shards, *([qwen38_external.NGRAM_TABLE] if external else [])}
    auxiliary = qwen38_retained.auxiliary_metadata_filenames(config) if retained else ()
    names.update(auxiliary)
    if indexed:
        names.add(index.name)
    if not qwen38_external.supported_shards(shards, indexed) or not isinstance(records, dict) or set(records) != names:
        return None
    total = 0
    for name, record in records.items():
        path = directory / name
        if (
            not isinstance(record, dict)
            or not _sha256_text(record.get("sha256"))
            or type(record.get("size")) is not int
            or record["size"] <= 0
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["size"]
        ):
            return None
        if name not in (qwen38_external.INDEX, *auxiliary):
            total += record["size"]
        elif record["size"] > 1048576 or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            return None
    if indexed and (
        index.stat().st_size > 1024 * 1024
        or hashlib.sha256(index.read_bytes()).hexdigest() != records[index.name]["sha256"]
    ):
        return None
    if total != receipt.get("weight_bytes"):
        return None
    return {
        **identity,
        "architecture": "qwen4_exp",
        "weight_bytes": total,
        "quantization": quantization,
    }


def completed_model_bytes(stage: Path, source: dict) -> int:
    """Count completed Hub files at the pinned revision, never incomplete allocations."""
    completed = 0
    for path in stage.rglob("*"):
        if path.suffix not in {".safetensors", ".gguf"} and not (
            path.name == "ngram_table.bin" and source.get("ngram_in_weight_bytes")
        ):
            continue
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(stage)
        metadata = stage / ".cache/huggingface/download" / (str(relative) + ".metadata")
        try:
            if metadata.is_symlink() or metadata.stat().st_size > 4096:
                continue
            lines = metadata.read_text().splitlines()
            if len(lines) != 3 or lines[0] != source["weights_revision"] or not lines[1]:
                continue
            info = path.stat()
            timestamp = float(lines[2])
            if not math.isfinite(timestamp) or timestamp < info.st_mtime:
                continue
            completed += info.st_size
        except (OSError, ValueError, UnicodeError):
            continue
    return min(completed, source["weight_bytes"])


def prepare(
    choice: str,
    output: Path,
    *,
    offline: bool = False,
    progress: ProgressCallback | ProgressReporter | None = None,
    artifact_mode: str = "text",
) -> dict:
    if artifact_mode not in {"text", "vision-language"} or (
        artifact_mode == "vision-language" and choice not in {"qwen38-mlx-serve-mixed", "qwen38-lily-q4-xl"}
    ):
        raise ValueError("unsupported native artifact mode")
    progress = reporter(progress)
    source = SOURCES[choice]
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + ".native-download")
    if stage.is_symlink():
        raise ValueError("invalid download staging directory")
    stage.mkdir(mode=0o700, exist_ok=True)
    marker = stage / "stark-download.json"
    identity = {"choice": choice, "source": source}
    if marker.exists():
        if _json(marker) != identity:
            raise ValueError("staging source mismatch")
    elif any(stage.iterdir()):
        raise ValueError("download staging directory is not owned by this operation")
    else:
        marker.write_text(json.dumps(identity, sort_keys=True))
    disk_bytes = source["weight_bytes"] + (0 if source.get("ngram_in_weight_bytes") else source.get("ngram_bytes", 0))
    if not offline and shutil.disk_usage(output.parent).free < disk_bytes + 4 * 1024**3:
        # Account for a resumable completed shard already in our owned staging
        # directory. Hugging Face handles its own incomplete-file recovery.
        present = sum(path.stat().st_size for path in stage.rglob("*") if path.is_file() and not path.is_symlink())
        if shutil.disk_usage(output.parent).free + present < disk_bytes + 4 * 1024**3:
            raise ValueError("insufficient disk space for the pinned checkpoint")
    completed_gguf = sorted(stage.glob("*.gguf"))
    gguf_complete = (
        choice == "glm53-q2"
        and len(completed_gguf) == 4
        and sum(path.stat().st_size for path in completed_gguf) == source["weight_bytes"]
    )
    if choice == "glm53-q2" and completed_gguf and not gguf_complete:
        # A process can stop between the individual flattening renames below.
        # Restore those completed shards to Hugging Face's original paths so
        # its verified local download records remain usable on this retry.
        nested = stage / "UD-Q2_K_XL"
        if nested.is_symlink():
            raise ValueError("invalid GGUF staging directory")
        nested.mkdir(exist_ok=True)
        for shard in completed_gguf:
            target = nested / shard.name
            if shard.is_symlink() or not shard.is_file() or target.exists():
                raise ValueError("conflicting GGUF resume state")
            shard.rename(target)
    download_options = {}
    if progress is not None:
        # Read reconstruction callbacks from the locked Hub implementation;
        # preallocated incomplete-file sizes are not evidence of progress.
        progress.emit("download", "running", force=True)
        progress.emit("conversion", "waiting", force=True)
        capture = SnapshotDownloadProgress(
            progress,
            source["weight_bytes"],
            completed_files=lambda: source["weight_bytes"] if gguf_complete else completed_model_bytes(stage, source),
        )
        capture.close()
        download_options["tqdm_class"] = capture.bar_class()
    if not gguf_complete:
        snapshot_download(
            source["weights_repo"],
            revision=source["weights_revision"],
            allow_patterns=source["weights_patterns"],
            local_dir=stage,
            local_files_only=offline,
            max_workers=3,
            **download_options,
        )
    snapshot_download(
        source["metadata_repo"],
        revision=source["metadata_revision"],
        allow_patterns=_METADATA,
        local_dir=stage,
        local_files_only=offline,
        max_workers=2,
        **download_options,
    )
    if choice == "glm53-q2" and not gguf_complete:
        shards = sorted((stage / "UD-Q2_K_XL").glob("*.gguf"))
        if len(shards) != 4:
            raise ValueError("incomplete pinned GGUF split set")
        for shard in shards:
            shard.rename(stage / shard.name)
        (stage / "UD-Q2_K_XL").rmdir()
    weights = [path for path in stage.iterdir() if path.suffix in {".gguf", ".safetensors"}]
    if not weights or any(path.is_symlink() or not path.is_file() for path in weights):
        raise ValueError("invalid prepared weight files")
    if "weight_shards" in source and len(weights) != source["weight_shards"]:
        raise ValueError("incomplete pinned weight shard set")
    weight_bytes = sum(path.stat().st_size for path in weights)
    expected_shards_bytes = source["weight_bytes"] - (
        source.get("ngram_bytes", 0) if source.get("ngram_in_weight_bytes") else 0
    )
    if weight_bytes != expected_shards_bytes:
        raise ValueError("pinned weight size mismatch")
    if "ngram_bytes" in source:
        table = stage / "ngram_table.bin"
        if table.is_symlink() or not table.is_file() or table.stat().st_size != source["ngram_bytes"]:
            raise ValueError("incomplete pinned n-gram table")
        if source.get("ngram_in_weight_bytes"):
            weight_bytes += source["ngram_bytes"]
    if progress is not None:
        progress.emit("download", "complete", weight_bytes, weight_bytes, force=True)
        progress.emit("conversion", "running", force=True)
    receipt = finish_metadata(stage, choice)
    if artifact_mode == "vision-language":
        from .hybrid_vision import prepare_image_artifact

        prepare_image_artifact(stage, choice)
    # Publish only the completed, metadata-bound directory. The download remains
    # resumable on all preceding failures and is never treated as a ready model.
    _publish(stage, output)
    (output / marker.name).unlink()
    if progress is not None:
        progress.emit("conversion", "complete", force=True)
    return {**receipt, "weight_bytes": weight_bytes, "status": "prepared"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(SOURCES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--artifact-mode", choices=("text", "vision-language"), default="text")
    parser.add_argument("--upgrade-images", action="store_true", help="Upgrade the existing --output folder in place")
    parser.add_argument("--progress-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    progress_sink = None
    try:
        if args.progress_fd is not None:
            progress_sink = FDProgressSink(args.progress_fd)
        if args.upgrade_images:
            from .hybrid_vision import prepare_image_artifact

            receipt = prepare_image_artifact(args.output, args.model)
        else:
            options = {"artifact_mode": args.artifact_mode} if args.artifact_mode != "text" else {}
            receipt = prepare(args.model, args.output, offline=args.offline, progress=progress_sink, **options)
    except (Exception, KeyboardInterrupt):
        print(
            "Native model preparation did not complete; its owned download directory can be retried.", file=sys.stderr
        )
        return 1
    finally:
        if progress_sink is not None:
            progress_sink.close()
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
