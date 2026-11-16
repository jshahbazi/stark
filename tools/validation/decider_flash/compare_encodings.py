"""Private, no-model differential check of upstream and Stark token encoders."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

SOURCE_REVISION = "d06fb3f2d97e0a52325a0d463c450cf0338f44fe"
MAX_LENGTH = 4096


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _child(command: list[str], requests: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = "".join(json.dumps({"request": row}, ensure_ascii=False, allow_nan=False) + "\n" for row in requests)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        text=True,
    )
    try:
        stdout, _ = process.communicate(payload, timeout=1200)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise ValueError("encoder_timeout") from None
    if process.returncode != 0:
        raise ValueError("encoder_child_failed")
    try:
        records = [json.loads(line) for line in stdout.splitlines()]
    except (ValueError, UnicodeError):
        raise ValueError("encoder_protocol_invalid") from None
    if len(records) != len(requests) + 1:
        raise ValueError("encoder_record_count_mismatch")
    ready, encoded = records[0], records[1:]
    if ready.get("ready") is not True or ready.get("encoding_only") is not True or ready.get("model_loaded") is not False:
        raise ValueError("encoder_not_ready")
    if any("encodings" not in record for record in encoded):
        raise ValueError("encoder_request_failed")
    return ready, encoded


def _expand_image_placeholder(
    ids: list[int], grid: list[list[int]] | None, image_token: int, merge_size: int
) -> list[int]:
    positions = [index for index, token in enumerate(ids) if token == image_token]
    if grid is None:
        if positions:
            raise ValueError("image_placeholder_without_processor_grid")
        return ids
    if len(grid) != 1 or len(positions) != 1:
        raise ValueError("unsupported_image_grid_or_placeholder_count")
    temporal, height, width = (int(value) for value in grid[0])
    divisor = merge_size * merge_size
    patches = temporal * height * width
    if min(temporal, height, width, merge_size) < 1 or patches % divisor:
        raise ValueError("invalid_image_grid_geometry")
    count = patches // divisor
    position = positions[0]
    return ids[:position] + [image_token] * count + ids[position + 1 :]


def _ids_hash(rows: list[list[int]]) -> str:
    digest = hashlib.sha256()
    for ids in rows:
        digest.update(struct.pack("<I", len(ids)))
        for token in ids:
            digest.update(struct.pack("<I", token))
    return digest.hexdigest()


def compare(reference: list[dict[str, Any]], native: list[dict[str, Any]], requests: list[dict[str, Any]],
            image_token: int, merge_size: int) -> dict[str, Any]:
    counts = {
        "reference_sequences": 0,
        "native_sequences": 0,
        "image_sequences": 0,
        "token_mismatches": 0,
        "marker_mismatches": 0,
        "segment_mismatches": 0,
        "key_mismatches": 0,
        "option_count_mismatches": 0,
    }
    reference_rows: list[list[int]] = []
    native_rows: list[list[int]] = []
    for request, ref_record, native_record in zip(requests, reference, native, strict=True):
        ref_rows = ref_record["encodings"]
        native_encodings = native_record["encodings"]
        counts["reference_sequences"] += len(ref_rows)
        counts["native_sequences"] += len(native_encodings)
        if len(ref_rows) != len(native_encodings):
            counts["key_mismatches"] += abs(len(ref_rows) - len(native_encodings)) or 1
            continue
        has_image = bool(request.get("images"))
        if has_image:
            counts["image_sequences"] += len(ref_rows)
        for ref, local in zip(ref_rows, native_encodings, strict=True):
            if ref["keys"] != local["keys"]:
                counts["key_mismatches"] += 1
            if ref["option_counts"] != local["option_counts"]:
                counts["option_count_mismatches"] += 1
            reference_ids = [int(value) for value in ref["input_token_ids"]]
            native_ids = [int(value) for value in local["input_token_ids"]]
            expanded_native = _expand_image_placeholder(native_ids, ref["image_grid_thw"], image_token, merge_size)
            reference_rows.append(reference_ids)
            native_rows.append(expanded_native)
            if reference_ids != expanded_native:
                counts["token_mismatches"] += 1
            adjustment = len(expanded_native) - len(native_ids)
            native_markers = [int(value) + (adjustment if local["input_token_ids"].index(image_token)
                                              < int(value) else 0)
                              for value in local["readout_positions"]] if has_image else local["readout_positions"]
            if [int(value) for value in ref["readout_positions"]] != [int(value) for value in native_markers]:
                counts["marker_mismatches"] += 1
            native_segments = list(local["segments"])
            if has_image and native_segments:
                native_segments[0] += adjustment
            if [int(value) for value in ref["segments"]] != [int(value) for value in native_segments]:
                counts["segment_mismatches"] += 1
    passed = all(value == 0 for key, value in counts.items() if key.endswith("mismatches"))
    return {
        "passed": passed,
        **counts,
        "compared_tokens": sum(len(row) for row in reference_rows),
        "reference_ids_sha256": _ids_hash(reference_rows),
        "expanded_native_ids_sha256": _ids_hash(native_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--fixtures", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    snapshot = (args.snapshot or root / "runtime/decider-flash-v13/source").resolve()
    fixtures = (args.fixtures or root / "runtime/decider-flash-v13/fixtures.json").resolve()
    try:
        fixture = json.loads(fixtures.read_text(encoding="utf-8"))
        cases = fixture["cases"]
        requests = [case["request"] for case in cases]
        if not requests:
            raise ValueError("empty_fixture_set")
        reference_command = [
            "uv", "run", "--project", str(root / "tools/validation/decider_flash"), "--locked",
            "python", str(Path(__file__).with_name("runner.py")), "--snapshot", str(snapshot),
            "--device", "cpu", "--max-length", str(MAX_LENGTH), "--encoding-only",
        ]
        native_command = [
            "uv", "run", "--project", str(root / "tools/converter"), "--locked", "python",
            str(Path(__file__).with_name("native_encoder.py")), "--snapshot", str(snapshot),
            "--max-length", str(MAX_LENGTH),
        ]
        reference_ready, reference_rows = _child(reference_command, requests)
        native_ready, native_rows = _child(native_command, requests)
        if reference_ready.get("source_revision") != SOURCE_REVISION:
            raise ValueError("reference_revision_mismatch")
        model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        processor_config = json.loads((snapshot / "processor_config.json").read_text(encoding="utf-8"))
        image_token = model_config["image_token_id"]
        merge_size = processor_config["image_processor"]["merge_size"]
        result = compare(reference_rows, native_rows, requests, image_token, merge_size)
        result.update({
            "source_revision": SOURCE_REVISION,
            "reference_source_sha256": reference_ready["source_sha256"],
            "source_manifest_sha256": _hash_file(snapshot / "release-manifest.json"),
            "source_metadata_sha256": {
                name: _hash_file(snapshot / name)
                for name in (
                    "config.json", "decision_config.json", "joint_config.json", "processor_config.json",
                    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                )
            },
            "fixture_sha256": _hash_file(fixtures),
            "reference_runner_sha256": _hash_file(Path(__file__).with_name("runner.py")),
            "native_encoder_sha256": _hash_file(Path(__file__).with_name("native_encoder.py")),
            "reference_lock_sha256": _hash_file(root / "tools/validation/decider_flash/uv.lock"),
            "native_protocol_sha256": _hash_file(root / "tools/converter/stark_convert/decider_protocol.py"),
            "native_template_sha256": _hash_file(root / "tools/converter/stark_convert/template_runtime.py"),
            "native_lock_sha256": _hash_file(root / "tools/converter/uv.lock"),
            "fixture_cases": len(requests),
            "native_ready": native_ready["ready"] is True,
        })
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["passed"] else 1
    except Exception as error:
        code = str(error) if isinstance(error, ValueError) and str(error).isascii() else "encoding_comparison_failed"
        print(json.dumps({"passed": False, "error": code}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
