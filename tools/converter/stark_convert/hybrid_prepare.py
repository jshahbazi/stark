"""Prepare pinned original-format weights for Stark's native hybrid worker.

This path preserves quantized bytes. It downloads metadata and weights only;
repository Python, speculative sidecars and foreign inference engines are never
requested or executed. Native load and execution remain separate checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from .cli import _publish

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
    return json.loads(path.read_bytes())


def _ids(value):
    if type(value) is int:
        return [value]
    if isinstance(value, list) and value and all(type(item) is int for item in value):
        return value
    raise ValueError("invalid EOS metadata")


def finish_metadata(directory: Path, choice: str) -> dict:
    """Bind the preserved architecture and tokenizer template for CLI chat."""
    source = SOURCES[choice]
    config = _json(directory / "config.json")
    if config.get("model_type") != source["architecture"]:
        raise ValueError("source architecture changed")
    text = config.get("text_config", config)
    tokenizer = _json(directory / "tokenizer_config.json", 2 * 1024 * 1024)
    template_file = directory / "chat_template.jinja"
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
    receipt = {
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
        ),
    }
    (directory / "native-source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


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
        size = sum(path.stat().st_size for path in files)
        if size != source["weight_bytes"] or receipt.get("weight_bytes") != size:
            return None
        if not (directory / "tokenizer.json").is_file():
            return None
        return {
            "repo": source["weights_repo"],
            "revision": source["weights_revision"],
            "architecture": source["architecture"],
            "weight_bytes": size,
            "quantization": source["quantization"],
        }
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def prepare(choice: str, output: Path, *, offline: bool = False) -> dict:
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
    if not offline and shutil.disk_usage(output.parent).free < source["weight_bytes"] + 4 * 1024**3:
        # Account for a resumable completed shard already in our owned staging
        # directory. Hugging Face handles its own incomplete-file recovery.
        present = sum(path.stat().st_size for path in stage.rglob("*") if path.is_file() and not path.is_symlink())
        if shutil.disk_usage(output.parent).free + present < source["weight_bytes"] + 4 * 1024**3:
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
    if not gguf_complete:
        snapshot_download(
            source["weights_repo"],
            revision=source["weights_revision"],
            allow_patterns=source["weights_patterns"],
            local_dir=stage,
            local_files_only=offline,
            max_workers=3,
        )
    snapshot_download(
        source["metadata_repo"],
        revision=source["metadata_revision"],
        allow_patterns=_METADATA,
        local_dir=stage,
        local_files_only=offline,
        max_workers=2,
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
    if weight_bytes != source["weight_bytes"]:
        raise ValueError("pinned weight size mismatch")
    receipt = finish_metadata(stage, choice)
    # Publish only the completed, metadata-bound directory. The download remains
    # resumable on all preceding failures and is never treated as a ready model.
    _publish(stage, output)
    (output / marker.name).unlink()
    return {**receipt, "weight_bytes": weight_bytes, "status": "prepared"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(SOURCES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    try:
        receipt = prepare(args.model, args.output, offline=args.offline)
    except (Exception, KeyboardInterrupt):
        print(
            "Native model preparation did not complete; its owned download directory can be retried.", file=sys.stderr
        )
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
