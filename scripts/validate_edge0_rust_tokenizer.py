#!/usr/bin/env python3
"""Private-pipe Rust 0.22.2 tokenizer qualification against pinned GGUF source.

Only aggregate counts/hashes are persisted. This helper exercises the locked
Rust parser and incremental decoder without loading an inference model. It
does not qualify the generation worker, template/parser/tool path or a model.
"""

from __future__ import annotations

import argparse
import itertools
import json
import select
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from validate_edge0_tokenizer import (
    BOUNDARY_CORPUS,
    CORPUS_SHA256,
    LLAMA_REVISION,
    ROOT,
    Oracle,
    PropertyOracle,
    QualificationError,
    identity,
    private_output,
    write_new,
)

sys.path.insert(0, str(ROOT / "tools/converter"))
from stark_convert import edge0_native_tokenizer as native
from stark_convert.edge0_source import fetch_skeleton_metadata

PROOF_SHA256 = "d74501849a329954290a58bb376bc3d59b513cc5af01a04b1da40e3e94052ca0"
DEFAULT_RUNTIME = ROOT / "runtime/edge0-native-tokenizer-2026-11-12"


def require(condition, code):
    if not condition:
        raise QualificationError(code)


def batches(values):
    iterator = iter(values)
    while batch := list(itertools.islice(iterator, 1024)):
        yield batch


class Probe:
    def __init__(self, binary, program):
        self.process = subprocess.Popen(
            [str(binary), str(program)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        ready = self.read()
        require(
            ready == {"ready": True, "tokenizers": "0.22.2"}, "probe-ready-mismatch"
        )

    def read(self):
        require(
            bool(select.select([self.process.stdout], [], [], 45)[0]), "probe-timeout"
        )
        line = self.process.stdout.readline(32 * 1024 * 1024 + 1)
        require(bool(line) and len(line) <= 32 * 1024 * 1024, "probe-response-size")
        result = json.loads(line)
        require(isinstance(result, dict), "probe-response-shape")
        return result

    def request(self, operation, **values):
        data = (
            json.dumps(
                {"operation": operation, **values},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        require(len(data) <= 16 * 1024 * 1024, "probe-request-size")
        self.process.stdin.write(data)
        self.process.stdin.flush()
        return self.read()

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)


def bindings(args):
    proof = json.loads(args.python_proof.read_text())
    require(
        identity(args.python_proof)["sha256"] == PROOF_SHA256, "python-proof-identity"
    )
    require(proof["comparison"]["passed"] is True, "python-proof-not-passed")
    require(proof["oracle"]["source_revision"] == LLAMA_REVISION, "oracle-source-pin")
    files = {
        args.oracle_library: proof["oracle"]["library"],
        args.metadata_only_gguf: proof["oracle"]["metadata_only_gguf"],
        args.property_library: proof["property_oracle"]["library"],
        ROOT / "scripts/fixtures/edge0_tokenizer_oracle.cpp": proof["oracle"][
            "tracked_bridge_source"
        ],
        ROOT / "tools/converter/stark_convert/edge0_native_tokenizer.py": proof[
            "candidate_source"
        ],
    }
    for name, expected in proof["oracle"]["dependencies"].items():
        files[args.oracle_library.parent / name] = expected
    for path, expected in files.items():
        require(identity(path) == expected, "oracle-or-program-identity-mismatch")
    require(
        identity(args.tokenizer)["sha256"] == proof["comparison"]["candidate_sha256"],
        "derived-tokenizer-identity",
    )
    lock = tomllib.loads((ROOT / "Cargo.lock").read_text())
    versions = sorted(
        {p["version"] for p in lock["package"] if p["name"] == "tokenizers"}
    )
    require(versions == ["0.22.2"], "cargo-tokenizers-version")
    files.update(
        {
            args.probe: identity(args.probe),
            args.tokenizer: identity(args.tokenizer),
            ROOT / "Cargo.lock": identity(ROOT / "Cargo.lock"),
            ROOT / "crates/stark-engine/examples/edge0_tokenizer_probe.rs": identity(
                ROOT / "crates/stark-engine/examples/edge0_tokenizer_probe.rs"
            ),
            Path(__file__): identity(Path(__file__)),
            args.python_proof: identity(args.python_proof),
        }
    )
    return {str(path): expected for path, expected in files.items()}


def qualify(args):
    private_output(args.output)
    registration_path = args.output.with_suffix(".registration.json")
    private_output(registration_path)
    sealed_files = bindings(args)
    metadata = native.sealed_metadata(
        args.source, fetch_skeleton_metadata(args.source, offline=True)
    )
    corpus = native.corpus_identity(metadata)
    require(
        corpus["sha256"] == CORPUS_SHA256
        and corpus["cases"] == {"encode": 1942, "decode": 248327},
        "corpus-mismatch",
    )
    require(
        native.boundary_corpus_identity(metadata) == BOUNDARY_CORPUS,
        "boundary-corpus-mismatch",
    )
    registration = {
        "attempt": "20261112-EDGE0-05-rust-tokenizer",
        "registered_at": datetime.now(UTC).isoformat(),
        "corpus": corpus,
        "boundary_corpus": BOUNDARY_CORPUS,
        "files": sealed_files,
        "batch_limit": 1024,
        "encoding_cases": 3884,
        "decode_cases": 496654,
        "stream_policy": "ordinary encode corpus only; all token classes normal, complete strict UTF8; skip false/true",
        "negative_controls": args.negative_controls,
        "scope": "Rust 0.22.2 parser, encode/decode/incremental decode helper; no model or generation worker",
    }
    write_new(registration_path, registration)
    oracle = probe = None
    try:
        oracle = Oracle(args.oracle_library, args.metadata_only_gguf)
        properties = PropertyOracle(
            args.property_library, metadata["descriptor"]["pre_pattern"]
        )
        probe = Probe(args.probe, args.tokenizer)
        mismatches = {
            "encode": 0,
            "boundaries": 0,
            "decode_valid_utf8": 0,
            "decode_utf8_replacement": 0,
            "stream": 0,
        }
        counts = dict.fromkeys(mismatches, 0)
        texts = list(native.encode_corpus(metadata))
        references = [oracle.encode(text, parse_special=True) for text in texts]
        for special in (False, True):
            for offset in range(0, len(texts), 1024):
                actual = probe.request(
                    "encode", texts=texts[offset : offset + 1024], special=special
                ).get("ids")
                expected = references[offset : offset + 1024]
                require(
                    isinstance(actual, list) and len(actual) == len(expected),
                    "probe-encode-shape",
                )
                mismatches["encode"] += sum(
                    a != b for a, b in zip(actual, expected, strict=True)
                )
                counts["encode"] += len(expected)
        boundary_texts = list(native.boundary_corpus(metadata))
        boundary_refs = [properties.pretokenize(text) for text in boundary_texts]
        for offset in range(0, len(boundary_texts), 1024):
            actual = probe.request(
                "boundaries", texts=boundary_texts[offset : offset + 1024]
            ).get("boundaries")
            expected = boundary_refs[offset : offset + 1024]
            require(
                isinstance(actual, list) and len(actual) == len(expected),
                "probe-boundary-shape",
            )
            mismatches["boundaries"] += sum(
                a != b for a, b in zip(actual, expected, strict=True)
            )
            counts["boundaries"] += len(expected)
        raw = {
            render: [oracle.piece(i, special=render) for i in range(oracle.n_vocab)]
            for render in (False, True)
        }
        for skip in (False, True):
            for ids in batches(native.decode_corpus(metadata)):
                actual = probe.request("decode", ids=ids, skip=skip).get("texts")
                require(
                    isinstance(actual, list) and len(actual) == len(ids),
                    "probe-decode-shape",
                )
                for sequence, observed in zip(ids, actual, strict=True):
                    piece = b"".join(raw[not skip][i] for i in sequence)
                    try:
                        expected = piece.decode("utf-8", errors="strict")
                        kind = "decode_valid_utf8"
                    except UnicodeDecodeError:
                        expected, kind = (
                            piece.decode("utf-8", errors="replace"),
                            "decode_utf8_replacement",
                        )
                    mismatches[kind] += observed != expected
                    counts[kind] += 1
        streams, stream_refs = [], []
        for ids in references:
            if any(metadata["types"][i] != 1 for i in ids):
                continue
            try:
                expected = b"".join(raw[True][i] for i in ids).decode(
                    "utf-8", errors="strict"
                )
            except UnicodeDecodeError:
                continue
            streams.append(ids)
            stream_refs.append(expected)
        for skip in (False, True):
            for offset in range(0, len(streams), 1024):
                actual = probe.request(
                    "stream", ids=streams[offset : offset + 1024], skip=skip
                ).get("texts")
                expected = stream_refs[offset : offset + 1024]
                require(
                    isinstance(actual, list) and len(actual) == len(expected),
                    "probe-stream-shape",
                )
                mismatches["stream"] += sum(
                    a != b for a, b in zip(actual, expected, strict=True)
                )
                counts["stream"] += len(expected)
        negatives = {}
        if args.negative_controls:
            program = json.loads(args.tokenizer.read_text())
            control_ids = [
                [i] for i, kind in enumerate(metadata["types"]) if kind in (3, 4, 5)
            ]
            for name, mutant in native.negative_programs(program, metadata=metadata):
                require(
                    probe.request("replace", program=mutant).get("replaced") is True,
                    "probe-replace-failed",
                )
                errors = {"encode": 0, "boundaries": 0, "decode": 0}
                for offset in range(0, len(texts), 1024):
                    actual = probe.request(
                        "encode", texts=texts[offset : offset + 1024], special=False
                    )["ids"]
                    errors["encode"] += sum(
                        a != b
                        for a, b in zip(
                            actual, references[offset : offset + 1024], strict=True
                        )
                    )
                for offset in range(0, len(boundary_texts), 1024):
                    actual = probe.request(
                        "boundaries", texts=boundary_texts[offset : offset + 1024]
                    )["boundaries"]
                    errors["boundaries"] += sum(
                        a != b
                        for a, b in zip(
                            actual, boundary_refs[offset : offset + 1024], strict=True
                        )
                    )
                for skip in (False, True):
                    actual = probe.request("decode", ids=control_ids, skip=skip)[
                        "texts"
                    ]
                    expected = [
                        b"".join(raw[not skip][i] for i in ids).decode(
                            "utf-8", errors="replace"
                        )
                        for ids in control_ids
                    ]
                    errors["decode"] += sum(
                        a != b for a, b in zip(actual, expected, strict=True)
                    )
                negatives[name] = {"detected": any(errors.values()), **errors}
        require(
            all(
                identity(Path(path)) == expected
                for path, expected in sealed_files.items()
            ),
            "files-changed-after-registration",
        )
        passed = not any(mismatches.values()) and all(
            v["detected"] for v in negatives.values()
        )
        report = {
            "status": "compared",
            "passed": passed,
            "counts": counts,
            "mismatches": mismatches,
            "negative_controls": negatives,
            "registration": identity(registration_path),
            "native_rust_parser_validated": passed,
            "helper_incremental_decoder_validated": passed,
            "generation_worker_validated": False,
            "model_inference_validated": False,
            "template_tool_parser_validated": False,
        }
        write_new(args.output, report)
        print(
            json.dumps(
                {
                    "status": "compared",
                    "passed": passed,
                    "report": identity(args.output),
                },
                sort_keys=True,
            )
        )
        return 0 if passed else 1
    except Exception as error:
        code = (
            str(error)
            if isinstance(error, QualificationError)
            else type(error).__name__
        )
        write_new(
            args.output,
            {
                "status": "error",
                "passed": False,
                "failure_code": code,
                "registration": identity(registration_path),
            },
        )
        raise
    finally:
        if probe is not None:
            probe.close()
        if oracle is not None:
            oracle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "runtime/edge0-native-v2-evaluation"
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "runtime/edge0-native-v2-evaluation/tokenizer.json",
    )
    parser.add_argument(
        "--oracle-library", type=Path, default=DEFAULT_RUNTIME / "libedge0_oracle.dylib"
    )
    parser.add_argument(
        "--property-library",
        type=Path,
        default=DEFAULT_RUNTIME / "libedge0_oracle_v3.dylib",
    )
    parser.add_argument(
        "--metadata-only-gguf",
        type=Path,
        default=DEFAULT_RUNTIME / "edge0-metadata-only.gguf",
    )
    parser.add_argument(
        "--python-proof",
        type=Path,
        default=DEFAULT_RUNTIME / "qualifier-v2-final-1943-envbound.json",
    )
    parser.add_argument("--negative-controls", action="store_true")
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    try:
        return qualify(args)
    except Exception as error:  # noqa: BLE001 -- Never expose text-bearing exceptions.
        code = (
            str(error)
            if isinstance(error, QualificationError)
            else type(error).__name__
        )
        print(json.dumps({"status": "error", "passed": False, "failure_code": code}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
