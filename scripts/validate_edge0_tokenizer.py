#!/usr/bin/env python3
"""Private, metadata-only Edge0 tokenizer qualification with a prebuilt CPU oracle.

Run with locked converter dependencies. Build the exact llama.cpp pin and the
Stark bridge using scripts/fixtures/edge0_tokenizer_oracle.cpp's recipe; this
script fetches/builds nothing and never publishes a tokenizer. Explicit paths
bind a source staging directory, oracle bridge, metadata-only GGUF and build
receipt. Results and the preregistration receipt must be new ignored files.
Text, token vectors and exception text never leave the comparison subprocess.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import importlib
import importlib.metadata
import importlib.util
import json
import multiprocessing
import os
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/converter"))

from stark_convert._build_metadata import file_hash, open_regular
from stark_convert.edge0_source import (
    EDGE0_REVISION,
    SKELETON_HEADER_SHA256,
    fetch_skeleton_metadata,
)

LLAMA_REVISION = "3cf03257f219afbe7334045ff7c6a06ac68c627d"
SOURCE_ARCHIVE_SHA256 = (
    "c076d7534afa0e5d0ec2a0d425b11e791c16f3de0d727221aea071cef156a280"
)
SOURCE_TREE_SHA256 = "2cfb4bf2251a0ca61e56f6e906af85ed655173ad0b67717e9fbe7d3dc780cca2"
METADATA_GGUF_SHA256 = (
    "f84c1aeca75a7c8672d00dc6110b51a59af4ae7edc1ba8ed31bb615978494e37"
)
CORPUS_SHA256 = "bea133e4fee577f39d8f23750d460e7b788aa4e2f748337185c0df50d0fd3d7d"
BOUNDARY_CORPUS = {
    "sha256": "bf94af69b727318542ee1933d9553e0f8a7085e1c7b05b424e396e6be488c1fd",
    "cases": 1943,
    "additional_witness_sha256": "826e174be5effb10384a99e534fbde86afab0af48f2e242f7cb9b6261cddc9c4",
}


class QualificationError(Exception):
    """Only fixed, content-free failure codes may be reported."""


def identity(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": file_hash(path)}


def proof_environment() -> dict:
    """Bind the executed Python decoder; Rust worker policy is source evidence."""
    import tokenizers

    lock_path = ROOT / "tools/converter/uv.lock"
    lock = tomllib.loads(lock_path.read_text())
    packages = [
        package for package in lock["package"] if package["name"] == "tokenizers"
    ]
    distribution = importlib.metadata.distribution("tokenizers")
    if (
        sys.implementation.name != "cpython"
        or sys.version_info[:3] != (3, 13, 13)
        or len(packages) != 1
        or packages[0]["version"] != "0.23.3"
        or tokenizers.__version__ != packages[0]["version"]
        or distribution.version != packages[0]["version"]
        or (ROOT / "tools/converter/.python-version").read_text().strip() != "3.13.13"
    ):
        raise QualificationError("proof-python-or-tokenizers-version-mismatch")
    backend = importlib.import_module("tokenizers.tokenizers")
    extension = Path(backend.__file__).resolve(strict=True)
    owned = [
        entry
        for entry in distribution.files or []
        if str(entry).startswith("tokenizers/")
        and Path(distribution.locate_file(entry)).resolve() == extension
    ]
    if len(owned) != 1 or owned[0].hash is None or owned[0].hash.mode != "sha256":
        raise QualificationError("proof-tokenizers-backend-distribution-mismatch")
    backend_identity = identity(extension)
    record_hash = (
        base64.urlsafe_b64encode(bytes.fromhex(backend_identity["sha256"]))
        .decode()
        .rstrip("=")
    )
    if owned[0].hash.value != record_hash:
        raise QualificationError("proof-tokenizers-backend-record-mismatch")
    rust_lock = ROOT / "Cargo.lock"
    rust = tomllib.loads(rust_lock.read_text())
    rust_versions = sorted(
        {
            package["version"]
            for package in rust["package"]
            if package["name"] == "tokenizers"
        }
    )
    return {
        "scope": "evaluation-preparation-python-tokenizer-only",
        "python": {
            "implementation": sys.implementation.name,
            "version": ".".join(map(str, sys.version_info[:3])),
            "executable": identity(Path(sys.executable).resolve(strict=True)),
        },
        "tokenizers": {
            "module_version": tokenizers.__version__,
            "distribution_version": distribution.version,
            "module_source": identity(Path(tokenizers.__file__).resolve(strict=True)),
            "compiled_extension": backend_identity,
            "distribution_record_verified": True,
            "uv_locked_version": packages[0]["version"],
            "uv_lock": identity(lock_path),
        },
        "rust_worker": {
            "cargo_locked_tokenizers_versions": rust_versions,
            "cargo_lock": identity(rust_lock),
            "evidence": "source-policy-inspection-only",
            "runtime_executed": False,
            "native_parser_equivalence_validated": False,
            "generation_stream_validated": False,
        },
    }


def read_receipt(path: Path) -> dict:
    with open_regular(path) as stream:
        data = stream.read(1_000_001)
    if len(data) > 1_000_000:
        raise QualificationError("oversized-oracle-receipt")
    return json.loads(data)


def candidate_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "stark_convert._edge0_qualification_candidate", path
    )
    if spec is None or spec.loader is None:
        raise QualificationError("candidate-module-unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Oracle:
    """FFI for the committed Stark-authored bridge; no inference context exists."""

    def __init__(self, library: Path, gguf: Path):
        self.lib = ctypes.CDLL(str(library))
        sigs = {
            "edge0_load": ([ctypes.c_char_p], ctypes.c_void_p),
            "edge0_free": ([ctypes.c_void_p], None),
            "edge0_n_vocab": ([ctypes.c_void_p], ctypes.c_int32),
            "edge0_encode": (
                [
                    ctypes.c_void_p,
                    ctypes.c_char_p,
                    ctypes.c_int32,
                    ctypes.POINTER(ctypes.c_int32),
                    ctypes.c_int32,
                    ctypes.c_bool,
                ],
                ctypes.c_int32,
            ),
            "edge0_piece": (
                [
                    ctypes.c_void_p,
                    ctypes.c_int32,
                    ctypes.c_void_p,
                    ctypes.c_int32,
                    ctypes.c_bool,
                ],
                ctypes.c_int32,
            ),
            "edge0_attr": ([ctypes.c_void_p, ctypes.c_int32], ctypes.c_int32),
            "edge0_is_eog": ([ctypes.c_void_p, ctypes.c_int32], ctypes.c_bool),
            "edge0_special": ([ctypes.c_void_p, ctypes.c_int32], ctypes.c_int32),
        }
        for name, (args, result) in sigs.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result
        self.model = self.lib.edge0_load(os.fsencode(gguf))
        if not self.model:
            raise QualificationError("vocabulary-only-load-failed")
        self.n_vocab = self.lib.edge0_n_vocab(self.model)

    def encode(self, text: str, parse_special: bool = True) -> list[int]:
        data = text.encode()
        n = self.lib.edge0_encode(self.model, data, len(data), None, 0, parse_special)
        capacity = max(1, -n)
        output = (ctypes.c_int32 * capacity)()
        n = self.lib.edge0_encode(
            self.model, data, len(data), output, capacity, parse_special
        )
        if not 0 <= n <= capacity:
            raise QualificationError("oracle-encoding-capacity-failed")
        return list(output[:n])

    def piece(self, token: int, special: bool = True) -> bytes:
        if not 0 <= token < self.n_vocab:
            raise QualificationError("oracle-token-out-of-range")
        n = self.lib.edge0_piece(self.model, token, None, 0, special)
        capacity = max(1, -n)
        output = ctypes.create_string_buffer(capacity)
        n = self.lib.edge0_piece(self.model, token, output, capacity, special)
        if not 0 <= n <= capacity:
            raise QualificationError("oracle-decoding-capacity-failed")
        return output.raw[:n]

    def attr(self, token: int) -> int:
        return self.lib.edge0_attr(self.model, token)

    def is_eog(self, token: int) -> bool:
        return self.lib.edge0_is_eog(self.model, token)

    def special_ids(self) -> dict:
        names = (
            "bos",
            "eos",
            "eot",
            "sep",
            "nl",
            "pad",
            "mask",
            "fim_pre",
            "fim_suf",
            "fim_mid",
            "fim_pad",
            "fim_rep",
            "fim_sep",
        )
        return {
            name: self.lib.edge0_special(self.model, i) for i, name in enumerate(names)
        }

    def close(self):
        if self.model:
            self.lib.edge0_free(self.model)
            self.model = None


class PropertyOracle:
    def __init__(self, library: Path, pattern: str):
        self.lib = ctypes.CDLL(str(library))
        self.pattern = pattern.encode()
        self.lib.edge0_unicode_flags.argtypes = [
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_uint32,
        ]
        self.lib.edge0_unicode_flags.restype = ctypes.c_int32
        self.lib.edge0_split_lengths.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_int32,
        ]
        self.lib.edge0_split_lengths.restype = ctypes.c_int32

    def flags(self) -> bytes:
        output = (ctypes.c_uint16 * 0x110000)()
        if self.lib.edge0_unicode_flags(output, len(output)) != len(output):
            raise QualificationError("unicode-property-oracle-failed")
        return bytes(output)

    def pretokenize(self, text: str) -> list[int]:
        data = text.encode()
        n = self.lib.edge0_split_lengths(
            data, len(data), self.pattern, len(self.pattern), None, 0
        )
        if n == -2147483648:
            raise QualificationError("native-splitter-failed")
        capacity = max(1, -n)
        output = (ctypes.c_uint32 * capacity)()
        n = self.lib.edge0_split_lengths(
            data, len(data), self.pattern, len(self.pattern), output, capacity
        )
        if not 0 <= n <= capacity:
            raise QualificationError("native-splitter-capacity-failed")
        return list(output[:n])


def verify_oracle(args) -> dict:
    receipt = read_receipt(args.oracle_receipt)
    if (
        receipt.get("source_revision") != LLAMA_REVISION
        or receipt.get("source_tree_manifest_sha256") != SOURCE_TREE_SHA256
        or receipt.get("original_header", {}).get("sha256") != SKELETON_HEADER_SHA256
        or receipt.get("local_files", {}).get("llama-source.tar.gz", {}).get("sha256")
        != SOURCE_ARCHIVE_SHA256
    ):
        raise QualificationError("oracle-source-identity-mismatch")
    if identity(args.metadata_only_gguf) != {
        "bytes": 10945120,
        "sha256": METADATA_GGUF_SHA256,
    }:
        raise QualificationError("metadata-only-gguf-identity-mismatch")
    files = receipt["local_files"]
    if files.get("oracle.cpp") != {
        "bytes": 4176,
        "sha256": "6bcce07c37928babf8b7b7ec61af24609b0c95a0d10ea6bc2fb1d203e899db11",
    }:
        raise QualificationError("oracle-bridge-source-identity-mismatch")
    required_disabled = {
        "GGML_METAL",
        "GGML_ACCELERATE",
        "GGML_BLAS",
        "GGML_CUDA",
        "GGML_VULKAN",
        "GGML_OPENMP",
        "LLAMA_BUILD_TESTS",
        "LLAMA_BUILD_TOOLS",
        "LLAMA_BUILD_SERVER",
        "LLAMA_BUILD_APP",
    }
    load = receipt.get("load", {})
    if (
        not required_disabled.issubset(receipt.get("build", {}).get("disabled", []))
        or load.get("vocab_only") is not True
        or load.get("n_gpu_layers") != 0
        or load.get("llama_context_created") is not False
    ):
        raise QualificationError("oracle-cpu-vocabulary-only-contract-mismatch")
    if identity(args.oracle_library) != files["libedge0_oracle.dylib"]:
        raise QualificationError("oracle-library-identity-mismatch")
    deps = {}
    for name in (
        "libllama.dylib",
        "libggml.dylib",
        "libggml-cpu.dylib",
        "libggml-base.dylib",
    ):
        relative = f"build/bin/{name}"
        path = args.oracle_library.parent / relative
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(args.oracle_library.parent):
            raise QualificationError("oracle-dependency-path-escape")
        deps[relative] = identity(resolved)
        if deps[relative] != files[relative]:
            raise QualificationError("oracle-dependency-identity-mismatch")
    return {
        "source_revision": LLAMA_REVISION,
        "source_tree_manifest_sha256": SOURCE_TREE_SHA256,
        "receipt": identity(args.oracle_receipt),
        "library": identity(args.oracle_library),
        "dependencies": deps,
        "metadata_only_gguf": identity(args.metadata_only_gguf),
        "run_bridge_source": files["oracle.cpp"],
        "tracked_bridge_source": identity(
            ROOT / "scripts/fixtures/edge0_tokenizer_oracle.cpp"
        ),
    }


def private_output(path: Path) -> None:
    if path.exists() or not path.is_relative_to(ROOT):
        raise QualificationError("output-must-be-new-and-ignored")
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(path)], cwd=ROOT, check=False
    )
    if ignored.returncode:
        raise QualificationError("output-must-be-new-and-ignored")


def verify_property_oracle(args) -> dict | None:
    if args.candidate_revision == "edge0-native-tokenizer-v1":
        if (
            args.property_oracle_library is not None
            or args.property_oracle_receipt is not None
        ):
            raise QualificationError("v1-does-not-use-property-oracle")
        return None
    if args.candidate_revision != "edge0-native-tokenizer-v2-unicode":
        raise QualificationError("unsupported-candidate-revision")
    if args.property_oracle_library is None or args.property_oracle_receipt is None:
        raise QualificationError("unicode-candidate-needs-property-oracle")
    receipt = read_receipt(args.property_oracle_receipt)
    if (
        receipt.get("source_revision") != LLAMA_REVISION
        or receipt.get("original_oracle_report_sha256")
        != file_hash(args.oracle_receipt)
        or receipt.get("splitter", {}).get("function")
        != "unicode_regex_split(text,{exact_source_pattern},false)"
    ):
        raise QualificationError("property-oracle-source-identity-mismatch")
    expected = receipt["files"]["library"]
    if (
        identity(args.property_oracle_library)
        != {"bytes": expected["bytes"], "sha256": expected["sha256"]}
        or args.property_oracle_library.parent != args.oracle_library.parent
    ):
        raise QualificationError("property-oracle-library-identity-mismatch")
    return {
        "receipt": identity(args.property_oracle_receipt),
        "library": identity(args.property_oracle_library),
        "bridge_source": receipt["files"]["bridge"],
        "source_revision": LLAMA_REVISION,
        "flags_sha256": "ac10d2cb7eea9e4686334ca6cf7e4d37fa586504ed793702ac81f49b55d7fdd4",
    }


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def child_compare(arguments: dict, pipe) -> None:
    oracle = None
    try:
        if proof_environment() != arguments["proof_environment"]:
            raise QualificationError("proof-environment-changed-after-registration")
        if (
            file_hash(Path(arguments["candidate_source"]))
            != arguments["candidate_sha256"]
        ):
            raise QualificationError("candidate-changed-after-registration")
        module = candidate_module(Path(arguments["candidate_source"]))
        metadata = module.sealed_metadata(
            Path(arguments["source"]),
            fetch_skeleton_metadata(Path(arguments["source"]), offline=True),
        )
        oracle = Oracle(
            Path(arguments["oracle_library"]), Path(arguments["metadata_only_gguf"])
        )
        options = {}
        if arguments.get("property_oracle_library"):
            properties = PropertyOracle(
                Path(arguments["property_oracle_library"]),
                metadata["descriptor"]["pre_pattern"],
            )
            with open_regular(ROOT / "LICENSE.ggml-hybrid") as stream:
                license_text = stream.read(16384)
            flags = properties.flags()
            asset = module.derive_unicode_asset(flags, license_text)
            proof = module.verify_unicode_membership(asset, flags)
            options = {
                "unicode_asset": asset,
                "unicode_membership_proof": proof,
                "pretokenizer_reference": properties.pretokenize,
            }
        result = module.compare_candidate(
            metadata,
            oracle,
            registered_corpus_sha256=CORPUS_SHA256,
            worker_utf8_policy_verified=True,
            timeout_seconds=120,
            **options,
        )
        if options and result.get("pretokenizer_boundary_corpus") != BOUNDARY_CORPUS:
            raise QualificationError("comparison-boundary-corpus-mismatch")
        if (
            file_hash(Path(arguments["candidate_source"]))
            != arguments["candidate_sha256"]
        ):
            raise QualificationError("candidate-changed-during-comparison")
        if proof_environment() != arguments["proof_environment"]:
            raise QualificationError("proof-environment-changed-during-comparison")
        pipe.send({"status": "compared", "comparison": result})
    except Exception as error:  # noqa: BLE001 -- Do not expose text-bearing exception dumps.
        # Exception text, source strings and token vectors are deliberately omitted.
        code = (
            str(error)
            if isinstance(error, QualificationError)
            else type(error).__name__
        )
        pipe.send({"status": "error", "failure_code": code})
    finally:
        if oracle is not None:
            oracle.close()
        pipe.close()


def qualify(args) -> int:
    private_output(args.output)
    registration_path = args.output.with_suffix(".registration.json")
    private_output(registration_path)
    environment = proof_environment()
    bindings = verify_oracle(args)
    property_bindings = verify_property_oracle(args)
    module = candidate_module(args.candidate_source)
    supported = {
        module.CANDIDATE_REVISION,
        getattr(module, "UNICODE_CANDIDATE_REVISION", None),
    }
    if args.candidate_revision not in supported:
        raise QualificationError("candidate-revision-mismatch")
    metadata = module.sealed_metadata(
        args.source, fetch_skeleton_metadata(args.source, offline=True)
    )
    corpus = module.corpus_identity(metadata)
    if corpus["sha256"] != CORPUS_SHA256 or corpus["cases"] != {
        "encode": 1942,
        "decode": 248327,
    }:
        raise QualificationError("preregistered-corpus-mismatch")
    if (
        corpus["encode_modes"]
        != [
            {"add_special_tokens": False, "parse_special": True},
            {"add_special_tokens": True, "parse_special": True},
        ]
        or corpus["decode_skip_special_modes"] != [False, True]
        or corpus["raw_piece_modes"] != [False, True]
        or corpus["comparison_timeout_seconds"] != 120
        or corpus["text_policy"] != "concatenate-native-pieces-then-utf8-replace"
    ):
        raise QualificationError("preregistered-mode-matrix-mismatch")
    boundary_corpus = None
    if property_bindings is not None:
        boundary_corpus = module.boundary_corpus_identity(metadata)
        if boundary_corpus != BOUNDARY_CORPUS:
            raise QualificationError("preregistered-boundary-corpus-mismatch")
    registration = {
        "schema": 1,
        "registered_at": datetime.now(UTC).isoformat(),
        "candidate_revision": args.candidate_revision,
        "candidate_source": identity(args.candidate_source),
        "source_revision": EDGE0_REVISION,
        "header_sha256": SKELETON_HEADER_SHA256,
        "corpus": corpus,
        "pretokenizer_boundary_corpus": boundary_corpus,
        "oracle": bindings,
        "property_oracle": property_bindings,
        "qualifier": identity(Path(__file__)),
        "proof_environment": environment,
        "worker_source": {
            name: identity(ROOT / name)
            for name in (
                "crates/stark-generate/src/generation.rs",
                "crates/stark-generate/src/native.rs",
                "crates/stark-generate/src/hybrid.rs",
                "Cargo.lock",
                "tools/converter/uv.lock",
            )
        },
    }
    write_new(registration_path, registration)
    context = multiprocessing.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    values = {
        name: str(getattr(args, name))
        for name in (
            "candidate_source",
            "source",
            "oracle_library",
            "metadata_only_gguf",
        )
    }
    values["candidate_sha256"] = registration["candidate_source"]["sha256"]
    values["proof_environment"] = environment
    if args.property_oracle_library is not None:
        values["property_oracle_library"] = str(args.property_oracle_library)
    process = context.Process(target=child_compare, args=(values, sending))
    process.start()
    sending.close()
    if receiving.poll(120):
        try:
            result = receiving.recv()
        except EOFError:
            result = {"status": "error", "failure_code": "comparison-process-exited"}
        process.join(2)
    else:
        result = {"status": "error", "failure_code": "comparison-timeout-120-seconds"}
    if process.is_alive():
        process.terminate()
        process.join(5)
    receiving.close()
    if verify_oracle(args) != bindings:
        result = {"status": "error", "failure_code": "oracle-changed-during-comparison"}
    if verify_property_oracle(args) != property_bindings:
        result = {
            "status": "error",
            "failure_code": "property-oracle-changed-during-comparison",
        }
    if proof_environment() != environment:
        result = {
            "status": "error",
            "failure_code": "proof-environment-changed-during-comparison",
        }
    report = {
        "schema": 1,
        "completed_at": datetime.now(UTC).isoformat(),
        "registration_receipt": identity(registration_path),
        **registration,
        **result,
        "publication_enabled": False,
        "inference_validated": False,
        "generation_stream_validated": False,
        "rust_worker_native_parser_equivalence_validated": False,
        "worker_text_boundary": "concatenate-native-pieces-then-utf8-replace",
        "upstream_frontend_equivalence_claimed": False,
    }
    write_new(args.output, report)
    passed = result.get("comparison", {}).get("passed", False)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": passed,
                "report_sha256": file_hash(args.output),
            }
        )
    )
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "oracle-library",
        "metadata-only-gguf",
        "oracle-receipt",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--candidate-source",
        type=Path,
        default=ROOT / "tools/converter/stark_convert/edge0_native_tokenizer.py",
    )
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--property-oracle-library", type=Path)
    parser.add_argument("--property-oracle-receipt", type=Path)
    args = parser.parse_args()
    for name in (
        "source",
        "oracle_library",
        "metadata_only_gguf",
        "oracle_receipt",
        "output",
        "candidate_source",
    ):
        setattr(args, name, getattr(args, name).absolute().resolve())
    for name in ("property_oracle_library", "property_oracle_receipt"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).absolute().resolve())
    try:
        return qualify(args)
    except Exception as error:  # noqa: BLE001 -- CLI failures must remain content-free.
        code = (
            str(error)
            if isinstance(error, QualificationError)
            else type(error).__name__
        )
        print(json.dumps({"status": "error", "passed": False, "failure_code": code}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
