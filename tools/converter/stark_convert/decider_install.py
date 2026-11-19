"""Pinned local/Hugging Face source admission and install helpers for Decider v1.3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from .decider import SOURCE_REPO, validate_decision_metadata
from .decider import prepare as prepare_decider
from .decider_specs import (
    DECIDER_PROFILES,
    FLASH_DECIDER,
    DeciderSourceProfile,
    decider_profile,
)
from .progress import DownloadProgress, FDProgressSink, ProgressReporter, reporter
from .streaming import authorize_source_path, read_regular_bytes

FAMILY_ID = FLASH_DECIDER.family_id
SOURCE_REVISION = FLASH_DECIDER.revision
DISK_MARGIN = 15 * 1024**3
_FILE_PINS = FLASH_DECIDER.file_pins
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


def _pins(profile: DeciderSourceProfile) -> dict[str, tuple[str, str, int]]:
    # Preserve the Flash alias because model-free tests replace it with tiny pins.
    return _FILE_PINS if profile.family_id == FLASH_DECIDER.family_id else profile.file_pins


def _json_file(path: Path, maximum: int = 2 * 1024 * 1024) -> dict:
    data = read_regular_bytes(authorize_source_path(path.parent, path))
    if not 0 < len(data) <= maximum:
        raise ValueError("metadata_size")
    value = json.loads(data, parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()))
    if not isinstance(value, dict):
        raise ValueError("metadata_shape")
    return value


def _validate_metadata(
    source: Path, profile: DeciderSourceProfile | None = None
) -> tuple[dict, dict, DeciderSourceProfile]:
    config = _json_file(source / "config.json")
    decision = _json_file(source / "decision_config.json")
    joint = _json_file(source / "joint_config.json")
    processor = _json_file(source / "processor_config.json")
    if (
        config.get("model_type") != "qwen3_5"
        or config.get("architectures") != ["Qwen3_5Model"]
        or config.get("auto_map") is not None
        or not isinstance(config.get("text_config"), dict)
        or not isinstance(config.get("vision_config"), dict)
    ):
        raise ValueError("model_config")
    candidates = [profile] if profile is not None else list(DECIDER_PROFILES)
    profile = next(
        (
            candidate
            for candidate in candidates
            if decision.get("base_model") == candidate.base_model
            and config.get("tie_word_embeddings") is candidate.tied_word_embeddings
        ),
        None,
    )
    if profile is None:
        raise ValueError("model_profile")
    validate_decision_metadata(decision, joint, profile=profile)
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
    if not isinstance(weight_map, dict) or not weight_map or set(weight_map.values()) != profile.shards:
        raise ValueError("shard_index")
    if not all(isinstance(key, str) and key for key in weight_map):
        raise ValueError("shard_index")
    return config, decision, profile


def is_decider_source(source, *, kind: str, revision: str | None = None) -> bool:
    """Recognize only the pinned source profile; generic Qwen3.5 stays unrelated."""
    if kind not in {"local", "huggingface"}:
        return False
    if kind == "huggingface":
        return decider_profile(str(source), revision=revision) is not None
    path = Path(source).expanduser()
    if not path.is_dir():
        return False
    try:
        return source_profile(path, kind="local", revision=revision) is not None
    except Exception:
        return False


def source_profile(source, *, kind: str, revision: str | None = None) -> DeciderSourceProfile | None:
    try:
        if kind == "huggingface":
            return decider_profile(str(source), revision=revision)
        path = Path(source).expanduser()
        if not path.is_dir():
            return None
        _config, _decision, profile = _validate_metadata(path)
        return profile if revision in (None, profile.revision) else None
    except Exception:
        return None


def _source_args(source, kind: str, revision: str | None):
    if kind not in {"local", "huggingface"}:
        raise ValueError("source_kind")
    if kind == "huggingface":
        profile = source_profile(source, kind=kind, revision=revision)
        if profile is None:
            raise ValueError("source_repo")
        return profile.repo, profile.revision, profile
    path = Path(source).expanduser().resolve(strict=True)
    profile = source_profile(path, kind=kind, revision=revision)
    if not path.is_dir() or profile is None:
        raise ValueError("source_profile")
    return path, profile.revision, profile


def _manifest_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
    if algorithm == "git_blob_sha1":
        size = path.stat().st_size
        digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_files(source: Path, profile: DeciderSourceProfile = FLASH_DECIDER) -> int:
    total = 0
    for name, (algorithm, identity, expected_size) in _pins(profile).items():
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


def _verify_hf_files(source: Path, profile: DeciderSourceProfile = FLASH_DECIDER) -> int:
    total = 0
    for name, (algorithm, identity, expected_size) in _pins(profile).items():
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
    profile = source_profile(source, kind=kind, revision=revision)
    try:
        normalized, pinned, profile = _source_args(source, kind, revision)
        if kind == "local":
            _validate_metadata(normalized, profile)
            output_size = sum(
                _pins(profile)[name][2] for name in _pins(profile) if name.endswith(".safetensors")
            )
            _disk_check(normalized.parent, 0, output_size)
        else:
            _validate_hf_metadata(profile)
            output_size = sum(
                _pins(profile)[name][2] for name in _pins(profile) if name.endswith(".safetensors")
            )
            additional = _hf_missing_bytes(profile)
            cache = Path.home() / ".cache" / "huggingface"
            _disk_check(cache, additional, output_size)
        return {
            "schema_version": 1,
            "status": "conditional",
            "family_id": profile.family_id,
            "mode": "decision",
            "model_kind": "decision",
            "message": f"This pinned {profile.name} source can be prepared for Decision mode with one image.",
            "source": {"kind": kind, "repo": profile.repo, "revision": pinned},
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
        reason, message = "unsupported_source", "This source does not match a pinned Decider v1.3 profile."
    return {
        "schema_version": 1,
        "status": "unsupported",
        "family_id": profile.family_id if profile is not None else FAMILY_ID,
        "mode": "decision",
        "model_kind": "decision",
        "message": message,
        "reason": reason,
        "source": {
            "kind": kind,
            "repo": profile.repo if profile is not None else SOURCE_REPO if kind == "huggingface" else None,
            "revision": revision,
        },
        "modes": {"text": "unsupported", "decision": "unsupported", "vision-language": "unsupported"},
        "capabilities": {"decision": False, "images": False, "max_images": 0},
    }


def _hub(hub=None):
    if hub is not None:
        return hub
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import huggingface_hub

    return huggingface_hub


def _downloaded_path(hub, profile: DeciderSourceProfile, name: str) -> Path:
    return Path(
        hub.hf_hub_download(
            repo_id=profile.repo, filename=name, revision=profile.revision, local_files_only=False
        )
    )


def _validate_hf_metadata(
    profile_or_repo: DeciderSourceProfile | str = FLASH_DECIDER,
    revision: str | None = None,
    hub=None,
) -> Path:
    profile = (
        profile_or_repo
        if isinstance(profile_or_repo, DeciderSourceProfile)
        else decider_profile(profile_or_repo, revision=revision)
    )
    if profile is None:
        raise ValueError("source_profile")
    hub = _hub(hub)
    paths = [
        _downloaded_path(hub, profile, name)
        for name in _pins(profile)
        if not name.endswith(".safetensors") and name != "tokenizer.json"
    ]
    tokenizer_path = _downloaded_path(hub, profile, "tokenizer.json")
    paths.append(tokenizer_path)
    roots = {path.parent.resolve() for path in paths}
    if len(roots) != 1:
        raise ValueError("snapshot_layout")
    root = roots.pop()
    if root.name != profile.revision or root.parent.name != "snapshots":
        raise ValueError("snapshot_revision")
    for path in paths:
        if path.is_symlink() or not path.is_file():
            # HF cache snapshot files are symlinks to immutable blob objects;
            # their canonical targets are checked through the cache API.
            if not path.is_symlink() or not path.resolve(strict=True).is_file():
                raise ValueError("snapshot_file")
        algorithm, identity, expected_size = _pins(profile)[path.name]
        if path.stat().st_size != expected_size or _manifest_digest(path.resolve(), algorithm) != identity:
            raise ValueError("source_identity")
    _validate_metadata(root, profile)
    return root


def _hf_missing_bytes(
    profile_or_repo: DeciderSourceProfile | str = FLASH_DECIDER,
    revision: str | None = None,
    hub=None,
) -> int:
    profile = (
        profile_or_repo
        if isinstance(profile_or_repo, DeciderSourceProfile)
        else decider_profile(profile_or_repo, revision=revision)
    )
    if profile is None:
        raise ValueError("source_profile")
    hub = _hub(hub)
    total = 0
    for name, (_algorithm, _identity, expected_size) in _pins(profile).items():
        result = hub.hf_hub_download(
            repo_id=profile.repo,
            filename=name,
            revision=profile.revision,
            local_files_only=False,
            dry_run=True,
        )
        size = getattr(result, "file_size", expected_size)
        if type(size) is not int or size != expected_size:
            raise ValueError("source_size")
        if not getattr(result, "is_cached", False):
            total += size
    return total


def preflight(source, *, kind: str, mode: str = "decision", revision: str | None = None, hub=None) -> dict:
    profile = source_profile(source, kind=kind, revision=revision)
    if mode != "decision":
        return _preflight_failure(
            kind, "Decider supports Decision mode only.", "unsupported_mode", profile=profile
        )
    if kind == "huggingface" and profile is not None:
        try:
            _validate_hf_metadata(profile, hub=hub)
            output_size = sum(
                _pins(profile)[name][2] for name in _pins(profile) if name.endswith(".safetensors")
            )
            _disk_check(
                Path.home() / ".cache" / "huggingface",
                _hf_missing_bytes(profile, hub=hub),
                output_size,
            )
            return _preflight_success(kind, profile)
        except OSError:
            return _preflight_failure(
                kind, "Free disk space before installing this model.", "insufficient_disk_space", profile=profile
            )
        except Exception:
            return _preflight_failure(
                kind,
                "The pinned source metadata or image protocol did not validate.",
                "unsupported_source",
                profile=profile,
            )
    if kind == "local":
        try:
            path, _pinned, profile = _source_args(source, kind, revision)
            _validate_metadata(path, profile)
            output_size = sum(
                _pins(profile)[name][2] for name in _pins(profile) if name.endswith(".safetensors")
            )
            _disk_check(Path.home(), 0, output_size)
            return _preflight_success(kind, profile)
        except OSError:
            return _preflight_failure(
                kind, "Free disk space before installing this model.", "insufficient_disk_space", profile=profile
            )
        except Exception:
            return _preflight_failure(
                kind,
                "The local source metadata or image protocol did not validate.",
                "unsupported_source",
                profile=profile,
            )
    return _preflight_report(source, kind, revision)


def _preflight_success(kind, profile):
    return {
        "schema_version": 1,
        "status": "conditional",
        "family_id": profile.family_id,
        "mode": "decision",
        "model_kind": "decision",
        "message": f"This pinned {profile.name} source can be prepared for Decision mode with one image.",
        "source": {"kind": kind, "repo": profile.repo, "revision": profile.revision},
        "modes": {"text": "unsupported", "decision": "conditional", "vision-language": "unsupported"},
        "capabilities": {"decision": True, "images": True, "max_images": 1},
    }


def _preflight_failure(kind, message, reason, *, profile: DeciderSourceProfile | None = None):
    return {
        "schema_version": 1,
        "status": "unsupported",
        "family_id": profile.family_id if profile is not None else FAMILY_ID,
        "mode": "decision",
        "model_kind": "decision",
        "message": message,
        "reason": reason,
        "source": {
            "kind": kind,
            "repo": profile.repo if profile is not None else SOURCE_REPO if kind == "huggingface" else None,
            "revision": None,
        },
        "modes": {"text": "unsupported", "decision": "unsupported", "vision-language": "unsupported"},
        "capabilities": {"decision": False, "images": False, "max_images": 0},
    }


def install(source, *, kind: str, output, revision: str | None = None, progress=None, hub=None) -> dict:
    """Verify the pinned source, then create the native Decision artifact."""
    reporter_ = reporter(progress)
    normalized, pinned, profile = _source_args(source, kind, revision)
    output = Path(output).expanduser().absolute()
    if output.exists():
        raise ValueError("output_exists")
    if kind == "local":
        source_root = normalized
        output_size = sum(
            _pins(profile)[name][2] for name in _pins(profile) if name.endswith(".safetensors")
        )
        _disk_check(output.parent, 0, output_size)
        if reporter_ is not None:
            reporter_.emit("download", "skipped", force=True)
    else:
        hub = _hub(hub)
        output_size = sum(
            _pins(profile)[name][2] for name in _pins(profile) if name.endswith(".safetensors")
        )
        _disk_check(
            Path.home() / ".cache" / "huggingface", _hf_missing_bytes(profile, hub=hub), output_size
        )
        _disk_check(output.parent, 0, output_size)
        selected = [(profile.repo, name, pinned) for name in _pins(profile)]
        download = DownloadProgress(reporter_) if reporter_ is not None else None
        if download is not None:
            download.select(hub, selected)
            for _repo, name, _revision in selected:
                download.download(hub, profile.repo, name, pinned)
        else:
            for name in _pins(profile):
                _downloaded_path(hub, profile, name)
        source_root = _validate_hf_metadata(profile, hub=hub)
    if reporter_ is not None:
        reporter_.emit("conversion", "running", force=True)
    total = (
        _verify_files(source_root, profile) if kind == "local" else _verify_hf_files(source_root, profile)
    )
    result = prepare_decider(source_root, output, revision=pinned, profile=profile)
    if reporter_ is not None:
        reporter_.emit("conversion", "complete", total, total, force=True)
    return {"prepared": True, "type": result["type"], "images": True, "revision": pinned}


def _cli(argv=None):
    parser = argparse.ArgumentParser(description="Install a pinned Decider v1.3 native artifact.")
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
