"""Pinned local/Hugging Face source admission and install helper for Decider Flash v1.3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from .decider import SOURCE_REPO, SOURCE_REVISION, validate_decision_metadata
from .decider import prepare as prepare_decider
from .progress import DownloadProgress, FDProgressSink, ProgressReporter, reporter
from .streaming import authorize_source_path, read_regular_bytes

FAMILY_ID = "decider-flash-v1.3"
DISK_MARGIN = 15 * 1024**3
_FILE_PINS = {
    "LICENSE": ("git_blob_sha1", "f938136e3adacfd92be087f6e113b5d6d97f678f", 11544),
    "NOTICE": ("git_blob_sha1", "675e00d0e7554437fc0f139aeafa23c9a2b0ea56", 132),
    "chat_template.jinja": ("git_blob_sha1", "a585dec894e63da457d9440ec6aa7caa16d20860", 7756),
    "config.json": ("git_blob_sha1", "cc17ee553587bd774dd2d02415ea1ac0e0316104", 2898),
    "decision_config.json": ("git_blob_sha1", "c3302d2a1b3f5c7dc31cc866a3ac279be0295756", 4687),
    "joint_config.json": ("git_blob_sha1", "5f2b195316a2ed2dffcaa449746154f057c277b0", 273),
    "model-00001-of-00002.safetensors": (
        "sha256",
        "0048d1ddf2c5754fc05540fccdbc4ac50407e5cf14a35326be01cac0756b4de6",
        4972946624,
    ),
    "model-00002-of-00002.safetensors": (
        "sha256",
        "40e0c0ccb3248a59db0724ffd1ad1c81fc935d9f474e6fd52e9df32af4690608",
        4105669472,
    ),
    "model.safetensors.index.json": (
        "git_blob_sha1",
        "3d9b681bfe92f1e0d1cff660d6cecf9a199627fa",
        61898,
    ),
    "processor_config.json": ("git_blob_sha1", "424539fa61b2d9d23bf0e481ca3078e9ff404b12", 1219),
    "readout.safetensors": (
        "sha256",
        "46358a2e2639f1956561687ba261dd9b37506f878e90eb5eabc15f114532bec3",
        1305712,
    ),
    "tokenizer.json": (
        "sha256",
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        19989325,
    ),
    "tokenizer_config.json": ("git_blob_sha1", "875b7b0a3aabbd4e5e2888802dfbd5624a4a2060", 1164),
}
_EXPECTED_SHARDS = {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
_EXPECTED_IMAGE_PROCESSOR = {
    "do_convert_rgb": True,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "merge_size": 2,
    "patch_size": 16,
    "resample": 3,
    "rescale_factor": 1 / 255,
    "size": {"longest_edge": 1638400, "shortest_edge": 65536},
    "temporal_patch_size": 2,
    "image_processor_type": "Qwen2VLImageProcessor",
}


def _json_file(path: Path, maximum: int = 2 * 1024 * 1024) -> dict:
    data = read_regular_bytes(authorize_source_path(path.parent, path))
    if not 0 < len(data) <= maximum:
        raise ValueError("metadata_size")
    value = json.loads(data, parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()))
    if not isinstance(value, dict):
        raise ValueError("metadata_shape")
    return value


def _validate_metadata(source: Path) -> tuple[dict, dict]:
    config = _json_file(source / "config.json")
    decision = _json_file(source / "decision_config.json")
    joint = _json_file(source / "joint_config.json")
    processor = _json_file(source / "processor_config.json")
    if (
        config.get("model_type") != "qwen3_5"
        or config.get("architectures") != ["Qwen3_5Model"]
        or config.get("tie_word_embeddings") is not True
        or config.get("auto_map") is not None
        or not isinstance(config.get("text_config"), dict)
        or not isinstance(config.get("vision_config"), dict)
    ):
        raise ValueError("model_config")
    validate_decision_metadata(decision, joint)
    if (
        processor.get("processor_class") != "Qwen3VLProcessor"
        or processor.get("image_processor") != _EXPECTED_IMAGE_PROCESSOR
    ):
        raise ValueError("image_processor")
    text_rope = config["text_config"].get("rope_parameters", {})
    vision_rope = config["vision_config"].get("rope_parameters", {})
    sections = text_rope.get("mrope_section") if isinstance(text_rope, dict) else None
    if (
        text_rope.get("mrope_interleaved") is not True
        or not isinstance(sections, list)
        or len(sections) != 3
        or any(type(value) is not int or value < 0 for value in sections)
        or vision_rope != {"rope_theta": 10000.0, "rope_type": "axial"}
    ):
        raise ValueError("rope_config")
    index = _json_file(source / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map or set(weight_map.values()) != _EXPECTED_SHARDS:
        raise ValueError("shard_index")
    if not all(isinstance(key, str) and key for key in weight_map):
        raise ValueError("shard_index")
    return config, decision


def is_decider_source(source, *, kind: str, revision: str | None = None) -> bool:
    """Recognize only the pinned source profile; generic Qwen3.5 stays unrelated."""
    if kind not in {"local", "huggingface"} or revision not in (None, SOURCE_REVISION):
        return False
    if kind == "huggingface":
        return str(source) == SOURCE_REPO
    path = Path(source).expanduser()
    if not path.is_dir():
        return False
    try:
        _validate_metadata(path)
        return True
    except Exception:
        return False


def _source_args(source, kind: str, revision: str | None):
    if kind not in {"local", "huggingface"}:
        raise ValueError("source_kind")
    if revision not in (None, SOURCE_REVISION):
        raise ValueError("source_revision")
    if kind == "huggingface":
        if source != SOURCE_REPO:
            raise ValueError("source_repo")
        return SOURCE_REPO, SOURCE_REVISION
    path = Path(source).expanduser().resolve(strict=True)
    if not path.is_dir() or not is_decider_source(path, kind="local", revision=revision):
        raise ValueError("source_profile")
    return path, SOURCE_REVISION


def _manifest_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
    if algorithm == "git_blob_sha1":
        size = path.stat().st_size
        digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_files(source: Path) -> int:
    total = 0
    for name, (algorithm, identity, expected_size) in _FILE_PINS.items():
        path = source / name
        try:
            path = authorize_source_path(source, path)
        except Exception:
            raise ValueError("source_file") from None
        size = path.stat().st_size
        if size != expected_size or _manifest_digest(path, algorithm) != identity:
            raise ValueError("source_identity")
        total += size
    return total


def _verify_hf_files(source: Path) -> int:
    total = 0
    for name, (algorithm, identity, expected_size) in _FILE_PINS.items():
        path = (source / name).resolve(strict=True)
        if not path.is_file() or path.stat().st_size != expected_size:
            raise ValueError("source_file")
        if _manifest_digest(path, algorithm) != identity:
            raise ValueError("source_identity")
        total += expected_size
    return total


def _disk_check(path: Path, additional: int, output_bytes: int) -> None:
    while not path.exists():
        path = path.parent
    if additional + output_bytes + DISK_MARGIN > shutil.disk_usage(path).free:
        raise OSError("disk_space")


def _preflight_report(source, kind: str, revision: str | None = None) -> dict:
    try:
        normalized, pinned = _source_args(source, kind, revision)
        if kind == "local":
            _validate_metadata(normalized)
            _, output_size = 0, sum(_FILE_PINS[name][2] for name in _FILE_PINS if name.endswith(".safetensors"))
            _disk_check(normalized.parent, 0, output_size)
        else:
            _validate_hf_metadata(normalized, pinned)
            output_size = sum(_FILE_PINS[name][2] for name in _FILE_PINS if name.endswith(".safetensors"))
            additional = _hf_missing_bytes(normalized, pinned)
            cache = Path.home() / ".cache" / "huggingface"
            _disk_check(cache, additional, output_size)
        return {
            "schema_version": 1,
            "status": "conditional",
            "family_id": FAMILY_ID,
            "mode": "decision",
            "model_kind": "decision",
            "message": "This pinned Decider source can be prepared for Decision mode with one image.",
            "source": {"kind": kind, "repo": SOURCE_REPO, "revision": pinned},
            "modes": {"text": "unsupported", "decision": "conditional", "vision-language": "unsupported"},
            "capabilities": {"decision": True, "images": True, "max_images": 1},
        }
    except OSError as error:
        reason = "insufficient_disk_space" if str(error) == "disk_space" else "source_unavailable"
        message = (
            "Free disk space before installing this model."
            if reason == "insufficient_disk_space"
            else "The pinned source is unavailable."
        )
    except Exception:
        reason, message = "unsupported_source", "This source does not match the pinned Decider Flash v1.3 profile."
    return {
        "schema_version": 1,
        "status": "unsupported",
        "family_id": FAMILY_ID,
        "mode": "decision",
        "model_kind": "decision",
        "message": message,
        "reason": reason,
        "source": {"kind": kind, "repo": SOURCE_REPO if kind == "huggingface" else None, "revision": revision},
        "modes": {"text": "unsupported", "decision": "unsupported", "vision-language": "unsupported"},
        "capabilities": {"decision": False, "images": False, "max_images": 0},
    }


def _hub(hub=None):
    if hub is not None:
        return hub
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import huggingface_hub

    return huggingface_hub


def _downloaded_path(hub, name: str, revision: str) -> Path:
    return Path(hub.hf_hub_download(repo_id=SOURCE_REPO, filename=name, revision=revision, local_files_only=False))


def _validate_hf_metadata(repo: str, revision: str, hub=None) -> Path:
    hub = _hub(hub)
    paths = [
        _downloaded_path(hub, name, revision)
        for name in _FILE_PINS
        if not name.endswith(".safetensors") and name != "tokenizer.json"
    ]
    tokenizer_path = _downloaded_path(hub, "tokenizer.json", revision)
    paths.append(tokenizer_path)
    roots = {path.parent.resolve() for path in paths}
    if len(roots) != 1:
        raise ValueError("snapshot_layout")
    root = roots.pop()
    if root.name != revision or root.parent.name != "snapshots":
        raise ValueError("snapshot_revision")
    for path in paths:
        if path.is_symlink() or not path.is_file():
            # HF cache snapshot files are symlinks to immutable blob objects;
            # their canonical targets are checked through the cache API.
            if not path.is_symlink() or not path.resolve(strict=True).is_file():
                raise ValueError("snapshot_file")
        algorithm, identity, expected_size = _FILE_PINS[path.name]
        if path.stat().st_size != expected_size or _manifest_digest(path.resolve(), algorithm) != identity:
            raise ValueError("source_identity")
    _validate_metadata(root)
    return root


def _hf_missing_bytes(repo: str, revision: str, hub=None) -> int:
    hub = _hub(hub)
    total = 0
    for name, (_algorithm, _identity, expected_size) in _FILE_PINS.items():
        result = hub.hf_hub_download(
            repo_id=repo, filename=name, revision=revision, local_files_only=False, dry_run=True
        )
        size = getattr(result, "file_size", expected_size)
        if type(size) is not int or size != expected_size:
            raise ValueError("source_size")
        if not getattr(result, "is_cached", False):
            total += size
    return total


def preflight(source, *, kind: str, mode: str = "decision", revision: str | None = None, hub=None) -> dict:
    if mode != "decision":
        return _preflight_failure(kind, "Decider Flash supports Decision mode only.", "unsupported_mode")
    if kind == "huggingface" and source == SOURCE_REPO and revision in (None, SOURCE_REVISION):
        try:
            _validate_hf_metadata(source, SOURCE_REVISION, hub=hub)
            output_size = sum(_FILE_PINS[name][2] for name in _FILE_PINS if name.endswith(".safetensors"))
            _disk_check(
                Path.home() / ".cache" / "huggingface",
                _hf_missing_bytes(source, SOURCE_REVISION, hub=hub),
                output_size,
            )
            return _preflight_success(kind, SOURCE_REPO, SOURCE_REVISION)
        except OSError:
            return _preflight_failure(kind, "Free disk space before installing this model.", "insufficient_disk_space")
        except Exception:
            return _preflight_failure(
                kind, "The pinned source metadata or image protocol did not validate.", "unsupported_source"
            )
    if kind == "local":
        try:
            path, pinned = _source_args(source, kind, revision)
            _validate_metadata(path)
            output_size = sum(_FILE_PINS[name][2] for name in _FILE_PINS if name.endswith(".safetensors"))
            _disk_check(Path.home(), 0, output_size)
            return _preflight_success(kind, SOURCE_REPO, pinned)
        except OSError:
            return _preflight_failure(kind, "Free disk space before installing this model.", "insufficient_disk_space")
        except Exception:
            return _preflight_failure(
                kind, "The local source metadata or image protocol did not validate.", "unsupported_source"
            )
    return _preflight_report(source, kind, revision)


def _preflight_success(kind, repo, revision):
    return {
        "schema_version": 1,
        "status": "conditional",
        "family_id": FAMILY_ID,
        "mode": "decision",
        "model_kind": "decision",
        "message": "This pinned Decider source can be prepared for Decision mode with one image.",
        "source": {"kind": kind, "repo": repo, "revision": revision},
        "modes": {"text": "unsupported", "decision": "conditional", "vision-language": "unsupported"},
        "capabilities": {"decision": True, "images": True, "max_images": 1},
    }


def _preflight_failure(kind, message, reason):
    return {
        "schema_version": 1,
        "status": "unsupported",
        "family_id": FAMILY_ID,
        "mode": "decision",
        "model_kind": "decision",
        "message": message,
        "reason": reason,
        "source": {"kind": kind, "repo": SOURCE_REPO if kind == "huggingface" else None, "revision": None},
        "modes": {"text": "unsupported", "decision": "unsupported", "vision-language": "unsupported"},
        "capabilities": {"decision": False, "images": False, "max_images": 0},
    }


def install(source, *, kind: str, output, revision: str | None = None, progress=None, hub=None) -> dict:
    """Verify the pinned source, then create the native Decision artifact."""
    reporter_ = reporter(progress)
    normalized, pinned = _source_args(source, kind, revision)
    output = Path(output).expanduser().absolute()
    if output.exists():
        raise ValueError("output_exists")
    if kind == "local":
        source_root = normalized
        output_size = sum(_FILE_PINS[name][2] for name in _FILE_PINS if name.endswith(".safetensors"))
        _disk_check(output.parent, 0, output_size)
        if reporter_ is not None:
            reporter_.emit("download", "skipped", force=True)
    else:
        hub = _hub(hub)
        output_size = sum(_FILE_PINS[name][2] for name in _FILE_PINS if name.endswith(".safetensors"))
        _disk_check(
            Path.home() / ".cache" / "huggingface", _hf_missing_bytes(SOURCE_REPO, pinned, hub=hub), output_size
        )
        _disk_check(output.parent, 0, output_size)
        selected = [(SOURCE_REPO, name, pinned) for name in _FILE_PINS]
        download = DownloadProgress(reporter_) if reporter_ is not None else None
        if download is not None:
            download.select(hub, selected)
            for _repo, name, _revision in selected:
                download.download(hub, SOURCE_REPO, name, pinned)
        else:
            for name in _FILE_PINS:
                _downloaded_path(hub, name, pinned)
        source_root = _validate_hf_metadata(SOURCE_REPO, pinned, hub=hub)
    if reporter_ is not None:
        reporter_.emit("conversion", "running", force=True)
    total = _verify_files(source_root) if kind == "local" else _verify_hf_files(source_root)
    result = prepare_decider(source_root, output, revision=pinned)
    if reporter_ is not None:
        reporter_.emit("conversion", "complete", total, total, force=True)
    return {"prepared": True, "type": result["type"], "images": True, "revision": pinned}


def _cli(argv=None):
    parser = argparse.ArgumentParser(description="Install the pinned Decider Flash v1.3 native artifact.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("source")
        command.add_argument("--kind", choices=("local", "huggingface"), required=True)
        command.add_argument("--revision")
        if name == "preflight":
            command.add_argument("--mode", default="decision")
        else:
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--progress-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    progress_sink = None
    try:
        if args.command == "preflight":
            report = preflight(args.source, kind=args.kind, mode=args.mode, revision=args.revision)
            print(json.dumps(report, separators=(",", ":")))
            return 0
        progress_sink = FDProgressSink(args.progress_fd) if args.progress_fd is not None else None
        progress_reporter = ProgressReporter(progress_sink) if progress_sink is not None else None
        result = install(
            args.source, kind=args.kind, output=args.output, revision=args.revision, progress=progress_reporter
        )
        print(json.dumps(result, separators=(",", ":")))
        return 0
    except Exception:
        print(json.dumps({"prepared": False, "error_code": "source_or_conversion_failed"}, separators=(",", ":")))
        return 2
    finally:
        if progress_sink is not None:
            progress_sink.close()


if __name__ == "__main__":
    raise SystemExit(_cli())
