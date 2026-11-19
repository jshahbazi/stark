"""Pinned local/Hugging Face source admission and installation for Elephant 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from .elephant import SOURCE_REPO, SOURCE_REVISION, prepare, validate_source_metadata
from .progress import DownloadProgress, FDProgressSink, ProgressReporter, reporter
from .streaming import authorize_source_path, read_regular_bytes

FAMILY_ID = "elephant-1"
DISK_MARGIN = 24 * 1024**3
_FILE_PINS = {
    "README.md": ("git_blob_sha1", "8589c01ff0c76249b2afafc6faf61600363902fa", 5475),
    "chat_template.jinja": ("git_blob_sha1", "a585dec894e63da457d9440ec6aa7caa16d20860", 7756),
    "config.json": ("git_blob_sha1", "98295053fab64c0a71b1c2bbba06c2e31ed33cfb", 3146),
    "constraints.txt": ("git_blob_sha1", "d5bbbdf8baf9a81153bf1aae0ddbb98ecfd91011", 1416),
    "engine.py": ("git_blob_sha1", "a4e20f94387e9fe35c8ab6d9842a9f70e40122bd", 2799),
    "generation_config.json": ("git_blob_sha1", "5db0e6e75e3df4676fba6e8e0cd836f1b125e862", 116),
    "model.safetensors": (
        "sha256",
        "682f02348dfe4617dd21bba2915c12e805de8cf16327a61e895a831ea681d0ba",
        18819722392,
    ),
    "processor_config.json": ("git_blob_sha1", "43c4343ec493f200e452f87a7c6649ecdfcce1ac", 1220),
    "readout.json": ("git_blob_sha1", "444b7bf8313d43415e97988a79740a93154fdd3b", 63),
    "serve.patch": ("git_blob_sha1", "7a432e8d254a9838b04ca3a155547ec3b3c70e78", 41045),
    "setup.sh": ("git_blob_sha1", "1d00a2676f8f1da5d25295bf727a85491ff95d51", 1285),
    "tokenizer.json": (
        "sha256",
        "a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f",
        19989339,
    ),
    "tokenizer_config.json": ("git_blob_sha1", "a60ae1a905ad7a2edaa995dbb0fc81a1c4fc42ee", 1505),
}


def _json_file(path: Path, maximum: int = 2 * 1024 * 1024):
    data = read_regular_bytes(authorize_source_path(path.parent, path))
    if not 0 < len(data) <= maximum:
        raise ValueError("metadata_size")
    value = json.loads(data, parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()))
    if not isinstance(value, dict):
        raise ValueError("metadata_shape")
    return value


def _validate_metadata(source: Path):
    config = _json_file(source / "config.json")
    readout = _json_file(source / "readout.json")
    processor = _json_file(source / "processor_config.json")
    validate_source_metadata(config, readout, processor)
    return config


def is_elephant_source(source, *, kind: str, revision: str | None = None):
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


def _source_args(source, kind, revision):
    if kind not in {"local", "huggingface"} or revision not in (None, SOURCE_REVISION):
        raise ValueError("source_profile")
    if kind == "huggingface":
        if source != SOURCE_REPO:
            raise ValueError("source_repo")
        return SOURCE_REPO, SOURCE_REVISION
    path = Path(source).expanduser().resolve(strict=True)
    if not path.is_dir() or not is_elephant_source(path, kind="local", revision=revision):
        raise ValueError("source_profile")
    return path, SOURCE_REVISION


def _manifest_digest(path: Path, algorithm: str):
    digest = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
    if algorithm == "git_blob_sha1":
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_files(source: Path):
    for name, (algorithm, identity, expected_size) in _FILE_PINS.items():
        path = authorize_source_path(source, source / name)
        if path.stat().st_size != expected_size or _manifest_digest(path, algorithm) != identity:
            raise ValueError("source_identity")
    return sum(item[2] for item in _FILE_PINS.values())


def _hub(hub=None):
    if hub is not None:
        return hub
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import huggingface_hub

    return huggingface_hub


def _downloaded_path(hub, name, revision):
    return Path(hub.hf_hub_download(repo_id=SOURCE_REPO, filename=name, revision=revision))


def _validate_hf_metadata(repo, revision, hub=None):
    hub = _hub(hub)
    paths = [
        _downloaded_path(hub, name, revision)
        for name in _FILE_PINS
        if name not in {"model.safetensors", "tokenizer.json"}
    ]
    paths.append(_downloaded_path(hub, "tokenizer.json", revision))
    roots = {path.parent.resolve() for path in paths}
    if len(roots) != 1:
        raise ValueError("snapshot_layout")
    root = roots.pop()
    if root.name != revision or root.parent.name != "snapshots":
        raise ValueError("snapshot_revision")
    for path in paths:
        pin = _FILE_PINS[path.name]
        if not path.is_file() and not path.resolve(strict=True).is_file():
            raise ValueError("snapshot_file")
        if path.stat().st_size != pin[2] or _manifest_digest(path.resolve(), pin[0]) != pin[1]:
            raise ValueError("source_identity")
    _validate_metadata(root)
    return root


def _hf_missing_bytes(repo, revision, hub=None):
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


def _disk_check(path: Path, additional, output_bytes):
    while not path.exists():
        path = path.parent
    if additional + output_bytes + DISK_MARGIN > shutil.disk_usage(path).free:
        raise OSError("disk_space")


def _success(kind):
    return {
        "schema_version": 1,
        "status": "conditional",
        "family_id": FAMILY_ID,
        "mode": "decision",
        "model_kind": "decision",
        "message": "This pinned Elephant 1 source can be prepared for Decision mode with one image.",
        "source": {"kind": kind, "repo": SOURCE_REPO, "revision": SOURCE_REVISION},
        "modes": {"text": "unsupported", "decision": "conditional", "vision-language": "unsupported"},
        "capabilities": {"decision": True, "images": True, "max_images": 1},
    }


def _failure(kind, message, reason):
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


def preflight(source, *, kind, mode="decision", revision=None, hub=None):
    if mode != "decision":
        return _failure(kind, "Elephant 1 supports Decision mode only.", "unsupported_mode")
    try:
        normalized, pinned = _source_args(source, kind, revision)
        output_size = _FILE_PINS["model.safetensors"][2]
        if kind == "local":
            _validate_metadata(normalized)
            _disk_check(Path(normalized).parent, 0, output_size)
        else:
            _validate_hf_metadata(normalized, pinned, hub=hub)
            _disk_check(Path.home() / ".cache" / "huggingface", _hf_missing_bytes(normalized, pinned, hub), output_size)
        return _success(kind)
    except OSError:
        return _failure(kind, "Free disk space before installing this model.", "insufficient_disk_space")
    except Exception:
        return _failure(kind, "This source does not match the pinned Elephant 1 profile.", "unsupported_source")


def install(source, *, kind, output, revision=None, progress=None, hub=None):
    reporter_ = reporter(progress)
    normalized, pinned = _source_args(source, kind, revision)
    output = Path(output).expanduser().absolute()
    if output.exists():
        raise ValueError("output_exists")
    output_size = _FILE_PINS["model.safetensors"][2]
    if kind == "local":
        source_root = normalized
        _disk_check(output.parent, 0, output_size)
        if reporter_ is not None:
            reporter_.emit("download", "skipped", force=True)
    else:
        hub = _hub(hub)
        _disk_check(Path.home() / ".cache" / "huggingface", _hf_missing_bytes(SOURCE_REPO, pinned, hub), output_size)
        _disk_check(output.parent, 0, output_size)
        download = DownloadProgress(reporter_) if reporter_ is not None else None
        if download is not None:
            download.select(hub, [(SOURCE_REPO, name, pinned) for name in _FILE_PINS])
        for name in _FILE_PINS:
            if download is not None:
                download.download(hub, SOURCE_REPO, name, pinned)
            else:
                _downloaded_path(hub, name, pinned)
        source_root = _validate_hf_metadata(SOURCE_REPO, pinned, hub=hub)
    if reporter_ is not None:
        reporter_.emit("conversion", "running", force=True)
    _verify_files(source_root)
    result = prepare(source_root, output, revision=pinned)
    if reporter_ is not None:
        reporter_.emit("conversion", "complete", output_size, output_size, force=True)
    return {"prepared": True, "type": result["type"], "images": True, "revision": pinned}


def _cli(argv=None):
    parser = argparse.ArgumentParser(description="Install the pinned Elephant 1 native artifact.")
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
    progress = None
    if args.command == "prepare" and args.progress_fd is not None:
        progress = ProgressReporter(FDProgressSink(args.progress_fd))
    if args.command == "preflight":
        result = preflight(args.source, kind=args.kind, mode=args.mode, revision=args.revision)
    else:
        result = install(
            args.source,
            kind=args.kind,
            output=args.output,
            revision=args.revision,
            progress=progress,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    _cli()
