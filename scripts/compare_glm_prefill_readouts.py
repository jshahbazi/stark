"""Compare full GLM readouts in RAM between existing T8 and selected prefill.

Use the optimized recipe wrapper. Native arrays travel through anonymous pipes
only; the persisted receipt contains aggregate errors, dispatch counts and hashes.
The sequential profiled timings are diagnostic, not performance qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import select
import struct
import subprocess
import sys
import time
from array import array
from pathlib import Path

from benchmark_hybrid import (
    checkout_identity,
    configuration_flags,
    hardware_identity,
    weight_identity,
)

NEW_FLAGS = (
    "STARK_GLM53_PREFILL_WIDE",
    "STARK_GLM53_CONTROL_PREFILL_BATCH",
    "STARK_GLM53_KDA_REGISTER_PREFILL",
)
MAX_ARM_BYTES = 128 * 1024 * 1024
ATOL = RTOL = 1e-3


def require(ok, label):
    if not ok:
        raise ValueError(label)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_exact(fd, count, deadline):
    data = bytearray()
    while len(data) < count:
        remaining = deadline - time.monotonic()
        require(remaining > 0 and select.select([fd], [], [], remaining)[0], "transport_timeout")
        chunk = os.read(fd, min(count - len(data), 1024 * 1024))
        require(bool(chunk), "transport_truncated")
        data.extend(chunk)
    return data


def run_arm(args, env, label):
    read_fd, write_fd = os.pipe()
    process = None
    deadline = time.monotonic() + args.timeout
    try:
        process = subprocess.Popen(
            [str(args.example.resolve()), str(args.model.resolve()), str(args.corpus.resolve()),
             ",".join(map(str, args.lengths)), str(write_fd)],
            pass_fds=(write_fd,), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
        )
        os.close(write_fd)
        write_fd = None
        require(read_exact(read_fd, 8, deadline) == b"SPRF001\n", "transport_magic_invalid")
        cases = struct.unpack("<I", read_exact(read_fd, 4, deadline))[0]
        require(cases == len(args.lengths), "transport_case_count_invalid")
        rows = []
        total_bytes = 0
        for length in args.lengths:
            metadata_bytes = struct.unpack("<I", read_exact(read_fd, 4, deadline))[0]
            require(0 < metadata_bytes <= 65_536, "transport_metadata_bound_invalid")
            metadata = json.loads(read_exact(read_fd, metadata_bytes, deadline))
            require(isinstance(metadata, dict), "transport_metadata_invalid")
            width, count = metadata.get("vocabulary"), metadata.get("readout_rows")
            require(type(width) is int and 0 < width <= 1_000_000 and count == 5
                    and metadata.get("input_tokens") == length, "transport_dimensions_invalid")
            byte_count = width * count * 4
            total_bytes += byte_count
            require(total_bytes <= MAX_ARM_BYTES, "transport_payload_bound_exceeded")
            raw = read_exact(read_fd, byte_count, deadline)
            digest = hashlib.sha256(raw).hexdigest()
            values = array("f")
            values.frombytes(raw)
            if sys.byteorder != "little":
                values.byteswap()
            require(all(math.isfinite(value) for value in values), "transport_nonfinite_readout")
            rows.append((metadata, values, digest))
            print(json.dumps({"arm": label, "completed_input_tokens": length}), flush=True)
        require(process.wait(timeout=max(0.001, deadline - time.monotonic())) == 0, "native_process_failed")
        remaining = deadline - time.monotonic()
        require(remaining > 0 and select.select([read_fd], [], [], remaining)[0], "transport_shutdown_timeout")
        require(os.read(read_fd, 1) == b"", "transport_trailing_payload")
        return rows
    finally:
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def routes_observed(metadata, flags):
    count = lambda name: metadata.get("dispatch", {}).get(name, {}).get("bytes", 0)
    wide = flags.get(NEW_FLAGS[0])
    if wide is not None:
        require(count("native.glm.prefill.wide_tokens") == metadata["input_tokens"] - 1,
                "wide_route_not_observed")
    else:
        require(count("native.glm.prefill.chunk_tokens") > 0, "t8_route_not_observed")
    for flag, counter in zip(NEW_FLAGS[1:], ("glm.control.prefill.batch.tokens", "glm.kda.prefill.register.tokens"), strict=True):
        require((count(counter) > 0) == (flags.get(flag) == "1"), "requested_route_not_observed")


def compare(reference, candidate):
    before, a, a_hash = reference
    after, b, b_hash = candidate
    for key in ("input_tokens", "input_sha256", "corpus_tokens", "readout_rows", "vocabulary"):
        require(before[key] == after[key], "arm_identity_mismatch")
    max_abs = squared_error = squared_reference = 0.0
    close = True
    for expected, actual in zip(a, b, strict=True):
        error = abs(expected - actual)
        max_abs = max(max_abs, error)
        squared_error += error * error
        squared_reference += expected * expected
        close &= error <= ATOL + RTOL * abs(expected)
    width = before["vocabulary"]
    agreement = sum(
        max(range(width), key=lambda i: a[start + i]) == max(range(width), key=lambda i: b[start + i])
        for start in range(0, len(a), width)
    )
    return {"input_tokens": before["input_tokens"], "input_sha256": before["input_sha256"],
            "corpus_tokens": before["corpus_tokens"], "cyclic_corpus": before["cyclic_corpus"],
            "readout_rows": before["readout_rows"], "vocabulary": width,
            "max_absolute_error": max_abs,
            "relative_l2_error": math.sqrt(squared_error / max(squared_reference, sys.float_info.min)),
            "all_close": close, "greedy_agreement": agreement, "exact": a_hash == b_hash,
            "reference_sha256": a_hash, "candidate_sha256": b_hash,
            "reference_dispatch": before["dispatch"], "candidate_dispatch": after["dispatch"],
            "reference_diagnostic_seconds": before["prefill_seconds"],
            "candidate_diagnostic_seconds": after["prefill_seconds"],
            "passed": close and agreement == before["readout_rows"]}


def run(args):
    require(getattr(args, "verified_weights", None) is not None, "verified_weight_receipt_required")
    require(1 <= len(args.lengths) <= 16 and all(2 <= n <= 4096 for n in args.lengths), "invalid_lengths")
    require(1 <= args.timeout <= 86_400 and args.corpus.stat().st_size <= 8 * 1024 * 1024, "invalid_limits")
    flags = configuration_flags()
    require(flags.get("STARK_GLM53_PREFILL_MMA") == "1" and flags.get("STARK_GLM53_PREFILL_CHUNK") == "8",
            "optimized_t8_recipe_required")
    require(flags.get(NEW_FLAGS[0]) is not None or any(flags.get(key) == "1" for key in NEW_FLAGS[1:]),
            "candidate_selectors_required")
    require(flags[NEW_FLAGS[0]] in (None, "16", "32", "64", "128")
            and all(flags[key] in (None, "0", "1") for key in NEW_FLAGS[1:]), "invalid_candidate_selectors")
    model_config = json.loads((args.model / "config.json").read_bytes())
    require(isinstance(model_config, dict) and model_config.get("model_type") == "glm5_next", "native_glm_model_required")
    receipt = args.model / "native-source.json"
    require(receipt.is_file(), "native_source_receipt_required")
    sources = {"harness_sha256": Path(__file__), "example_source_sha256":
               Path(__file__).resolve().parents[1] / "crates/stark-engine/examples/hybrid_prefill_readouts.rs"}
    source_identity = {key: sha(path) for key, path in sources.items()}
    identity = {"example_sha256": sha(args.example), "corpus_sha256": sha(args.corpus),
                "configuration_sha256": sha(args.model / "config.json"),
                "tokenizer_sha256": sha(args.model / "tokenizer.json"),
                "model_receipt_sha256": sha(receipt)}
    weights = weight_identity(args.model, args.verified_weights)
    reference_env = {key: value for key, value in os.environ.items() if key not in NEW_FLAGS}
    reference = run_arm(args, reference_env, "existing_t8")
    candidate = run_arm(args, dict(os.environ), "candidate")
    reference_flags = {key: None if key in NEW_FLAGS else value for key, value in flags.items()}
    for metadata, _, _ in reference:
        routes_observed(metadata, reference_flags)
    for metadata, _, _ in candidate:
        routes_observed(metadata, flags)
    require(identity == {"example_sha256": sha(args.example), "corpus_sha256": sha(args.corpus),
                         "configuration_sha256": sha(args.model / "config.json"),
                         "tokenizer_sha256": sha(args.model / "tokenizer.json"),
                         "model_receipt_sha256": sha(receipt)}, "identity_changed")
    require(weight_identity(args.model, args.verified_weights) == weights, "weights_changed")
    require(source_identity == {key: sha(path) for key, path in sources.items()}, "validation_sources_changed")
    cases = [compare(a, b) for a, b in zip(reference, candidate, strict=True)]
    return {"schema": "stark-glm-prefill-readouts-v1", **identity,
            **source_identity,
            "harness_checkout": checkout_identity(), "hardware": hardware_identity(), "weights": weights,
            "configuration_flags": flags, "reference_configuration_flags": reference_flags,
            "absolute_tolerance": ATOL, "relative_tolerance": RTOL,
            "scope": "five full-vocabulary rows per fixed prefix; no answer-quality or performance qualification",
            "cases": cases, "passed": all(case["passed"] for case in cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "example", "corpus", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--lengths", type=lambda value: list(map(int, value.split(","))), default=[513, 1221, 2049])
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--verified-weights", type=Path, required=True,
                        help="Prior verified GGUF hashes; guard shard identity before and after both arms.")
    args = parser.parse_args()
    try:
        result = run(args)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        result = {"passed": False, "error": "matched_readout_validation_failed"}
    try:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as output:
            json.dump(result, output, sort_keys=True, allow_nan=False)
            output.write("\n")
    except (OSError, ValueError):
        print(json.dumps({"passed": False, "error": "aggregate_receipt_write_failed"}))
        return 1
    print(json.dumps({"passed": result["passed"], "cases": len(result.get("cases", []))}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
