#!/usr/bin/env python3
"""Private JSONL bridge to the pinned Perplexity Decider reference package.

The caller supplies an already-downloaded, immutable model snapshot. This
process never downloads weights and writes only protocol records to stdout.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import string
import sys
from pathlib import Path
from typing import Any

SOURCE_REVISION = "d06fb3f2d97e0a52325a0d463c450cf0338f44fe"
SOURCE_SHA256 = {
    "__init__.py": "7e95504afcec8696365b936721f80a9013046a70a4d8c25802079eb04ea14d48",
    "fast.py": "867392d787151402314028151ca47c8ba89cad969d5b70f2e83c9633abfad2f2",
    "locate.py": "272a40ba42e83b437022b09fa3e5fe28bf0160b3d1905fa274589f6cfae8eeb8",
    "model.py": "8b57c37a27365a8a12fc3b4a1853908c7c7d519bfba36871f40e092f1d77f1bf",
    "protocol.py": "412292fba7f9e92f70f3071935f090e381910d1f6279380e728dc1c6b8380e6c",
    "py.typed": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "types.py": "f13de0c018e76eb37fd9d7c1fbecc2aa811bb54f650ccc69861c901b65866892",
}


def _protocol_write(value: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    sys.stdout.flush()


@contextlib.contextmanager
def _quiet_output():
    """Suppress Python and native-library writes on stdout/stderr temporarily."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(sys.stdout.fileno())
    saved_stderr = os.dup(sys.stderr.fileno())
    with open(os.devnull, "w", encoding="utf-8") as sink:
        try:
            os.dup2(sink.fileno(), sys.stdout.fileno())
            os.dup2(sink.fileno(), sys.stderr.fileno())
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                yield
        finally:
            os.dup2(saved_stdout, sys.stdout.fileno())
            os.dup2(saved_stderr, sys.stderr.fileno())
            os.close(saved_stdout)
            os.close(saved_stderr)


def _quiet_call(function, *args, **kwargs):
    with _quiet_output():
        return function(*args, **kwargs)


def _pinned_source_digest(snapshot: Path) -> str | None:
    """Verify the upstream package against its pinned source-file hashes."""
    source_root = snapshot / "source" / "src" / "pplx_decider"
    manifest_path = snapshot / "release-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("repo_id") != "perplexity-ai/pplx-decider-flash-v1.3":
            return None
        files = manifest["files"]
        digest = hashlib.sha256()
        for name, expected_hash in sorted(SOURCE_SHA256.items()):
            path = source_root / name
            relative = f"source/src/pplx_decider/{name}"
            content = path.read_bytes()
            actual_hash = hashlib.sha256(content).hexdigest()
            if actual_hash != expected_hash or files[relative]["sha256"] != expected_hash:
                return None
            digest.update(relative.encode("utf-8") + b"\0" + bytes.fromhex(actual_hash))
        return digest.hexdigest()
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _load_decider(snapshot: Path, source: Path, device: str, cpu_threads: int, max_length: int):
    sys.path.insert(0, str(source))
    from pplx_decider import Decider

    return Decider.from_pretrained(
        snapshot,
        device=device,
        fast=False,
        locate=False,
        cpu_threads=cpu_threads,
        max_length=max_length,
    )


def _load_encoder(snapshot: Path, source: Path, cpu_threads: int):
    sys.path.insert(0, str(source))
    import torch
    from pplx_decider.protocol import LEGACY_MAX_PIXELS, MAX_OPTIONS, MIN_PIXELS, Encoder, PROTOCOL, load_json
    from transformers import AutoProcessor

    joint_path = snapshot / "joint_config.json"
    protocol = load_json(joint_path) if joint_path.exists() else {}
    if protocol != PROTOCOL:
        raise ValueError
    decision = load_json(snapshot / "decision_config.json")
    if decision.get("attention_mode") != "noncausal_full_attention" or decision.get("pooling", "last") != "last":
        raise ValueError
    temperature = float(decision["temperature"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError
    codes = list(decision["codes"])
    torch.set_num_threads(cpu_threads)
    processor = AutoProcessor.from_pretrained(str(snapshot))
    processor.tokenizer.padding_side = "left"
    max_pixels = int(decision.get("max_pixels") or LEGACY_MAX_PIXELS)
    processor.image_processor.size = {"shortest_edge": MIN_PIXELS, "longest_edge": max_pixels}
    tokenizer = processor.tokenizer
    if len(codes) != MAX_OPTIONS or any(
        not isinstance(code, str) or not set(code) <= set(string.ascii_uppercase)
        for code in codes
    ) or any(
        tokenizer.encode(code, add_special_tokens=False) != [token_id]
        for code, token_id in zip(codes, decision["token_ids"], strict=True)
    ):
        raise ValueError
    return Encoder(processor, codes), max_pixels


def _encoding_reply(encoder, request, max_length):
    chunks = encoder.plan(request, max_length, mode="separate")
    return {
        "encodings": [
            {
                "keys": list(chunk.keys),
                "input_token_ids": chunk.input_ids.tolist(),
                "readout_positions": chunk.markers.tolist(),
                "option_counts": list(chunk.counts),
                "segments": list(chunk.segments),
                "image_grid_thw": None if chunk.image_grid_thw is None else chunk.image_grid_thw.tolist(),
            }
            for chunk in chunks
        ]
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path, help="local immutable HF snapshot directory")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--encoding-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.cpu_threads < 1 or args.cpu_threads > 32:
        _protocol_write({"error": "invalid_cpu_threads"})
        return 2
    if args.max_length < 1:
        _protocol_write({"error": "invalid_max_length"})
        return 2
    if not args.snapshot.is_dir():
        _protocol_write({"error": "snapshot_missing"})
        return 2
    source = args.snapshot / "source" / "src"
    source_digest = _pinned_source_digest(args.snapshot)
    if source_digest is None:
        _protocol_write({"error": "reference_source_mismatch"})
        return 2

    if args.encoding_only:
        try:
            encoder, max_pixels = _quiet_call(_load_encoder, args.snapshot, source, args.cpu_threads)
        except Exception:
            _protocol_write({"error": "reference_encoder_initialization_failed"})
            return 3
        _protocol_write({
            "ready": True,
            "encoding_only": True,
            "model_loaded": False,
            "max_length": args.max_length,
            "max_pixels": max_pixels,
            "source_revision": SOURCE_REVISION,
            "source_sha256": source_digest,
        })
        for line in sys.stdin:
            try:
                envelope = json.loads(line)
                request = envelope.get("request") if isinstance(envelope, dict) else None
                if not isinstance(request, dict):
                    raise ValueError
                result = _quiet_call(_encoding_reply, encoder, request, args.max_length)
                _protocol_write(result)
            except Exception:
                _protocol_write({"error": "reference_encoding_failed"})
        return 0

    try:
        decider = _quiet_call(
            _load_decider,
            args.snapshot,
            source,
            args.device,
            args.cpu_threads,
            args.max_length,
        )
    except Exception:
        _protocol_write({"error": "reference_initialization_failed"})
        return 3

    _protocol_write({
        "ready": True,
        "device": args.device,
        "dtype": str(decider.backbone.dtype),
        "source_revision": SOURCE_REVISION,
        "source_sha256": source_digest,
    })
    for line in sys.stdin:
        try:
            envelope = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            _protocol_write({"error": "invalid_json"})
            continue
        request = envelope.get("request") if isinstance(envelope, dict) else None
        if not isinstance(request, dict):
            _protocol_write({"error": "invalid_request"})
            continue
        try:
            response = _quiet_call(decider.decide, request)
            _protocol_write({"response": response})
        except Exception:
            _protocol_write({"error": "reference_inference_failed"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
