"""Register and evaluate a private synthetic Edge0 component oracle.

Run with the locked converter environment. This reads no checkpoint tensors,
loads no model, and never imports candidate Rust arithmetic. The only native
code is the pinned GGML quantizer/decoder behind a CPU-only C bridge.
The fixture covers the latent stage, source routing and synthetic shared/readout
composition; it contains no attention, GDN, PLE or whole-model reference.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import random
import struct
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "runtime/edge0-forward-2026-11-12"
GGML = REPO / "runtime/edge0-native-tokenizer-2026-11-12"
SOURCE = GGML / "llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d"
BRIDGE = REPO / "scripts/fixtures/edge0_forward_reference.cpp"
GEOMETRY = {
    "H": 256,
    "ELL": 256,
    "I": 256,
    "G": 3,
    "E": 5,
    "K": 3,
    "T": 3,
    "readout": 17,
}
SEED = 0xED6E003
LABELS = [0, 1, 0, 2, 1]
LOGITS = [
    [-0.5, 0.75, -1.0, 1.5, 0.25],
    [0.5, 0.5, 0.5, -1.0, -2.0],
    [0.25, -0.5, 0.75, -1.0, 1.5],
]
FLAGS = ["-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off", "-dynamiclib"]


def identity(path):
    data = Path(path).read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def source_route(logits, k):
    """Generic pinned router_top10.cu scalar semantics, including F32 p."""
    mx = max(logits)
    ex = [math.exp(float(v) - float(mx)) for v in logits]
    total = 0.0
    for v in ex:
        total += v
    inv = f32(1.0 / total)
    probabilities = [f32(v * inv) for v in ex]
    ids = sorted(range(len(logits)), key=lambda e: (-probabilities[e], e))[:k]
    selected_sum = 0.0
    for e in ids:
        selected_sum += probabilities[e]
    divisor = max(selected_sum, 2.0**-14)
    return ids, [f32(probabilities[e] / divisor) for e in ids]


def matvec(matrix, vector):
    # Python scalar double products and math.fsum; no BLAS and no Rust.
    values = [float(v) for v in vector]
    output, sums = [], []
    for row in matrix:
        products = [float(a) * b for a, b in zip(row, values, strict=True)]
        output.append(math.fsum(products))
        sums.append(math.fsum(abs(v) for v in products))
    return np.asarray(output, dtype="<f8"), np.asarray(sums, dtype="<f8")


class Oracle:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(library))
        ptr = ctypes.c_void_p
        self.lib.edge0_forward_row_bytes.argtypes = [ctypes.c_int, ctypes.c_size_t]
        self.lib.edge0_forward_row_bytes.restype = ctypes.c_size_t
        for name in ("pack", "unpack", "activation"):
            fn = getattr(self.lib, "edge0_forward_" + name)
            fn.argtypes = [ctypes.c_int, ptr, ctypes.c_size_t, ptr]
            fn.restype = ctypes.c_int

    def matrix(self, kind, values):
        row_bytes = self.lib.edge0_forward_row_bytes(kind, values.shape[1])
        raw, decoded = bytearray(), []
        for row in values:
            packed = ctypes.create_string_buffer(row_bytes)
            row = np.ascontiguousarray(row, dtype="<f4")
            assert (
                self.lib.edge0_forward_pack(kind, row.ctypes.data, len(row), packed)
                == 0
            )
            out = np.empty(len(row), dtype="<f4")
            assert (
                self.lib.edge0_forward_unpack(kind, packed, len(row), out.ctypes.data)
                == 0
            )
            raw.extend(packed.raw)
            decoded.append(out)
        return bytes(raw), np.asarray(decoded, dtype="<f4")

    def activation(self, kind, vector):
        source = np.ascontiguousarray(vector, dtype="<f4")
        out = np.empty(len(source), dtype="<f4")
        assert (
            self.lib.edge0_forward_activation(
                kind, source.ctypes.data, len(source), out.ctypes.data
            )
            == 0
        )
        return out.astype("<f8")


def metrics(reference, actual):
    delta = actual - reference
    return {
        "maximum_absolute_error": float(np.max(np.abs(delta))),
        "relative_l2_error": float(
            np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-300)
        ),
        "finite": bool(np.isfinite(actual).all()),
    }


def generate():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    registration = RUNTIME / "fixture-registration.json"
    if registration.exists():
        raise SystemExit(
            "Existing registration preserved; use a distinct attempt for changed fixtures"
        )
    library = RUNTIME / "libedge0_forward_reference.dylib"
    command = [
        "clang++",
        *FLAGS,
        str(BRIDGE),
        "-I",
        str(SOURCE / "ggml/include"),
        "-I",
        str(SOURCE / "ggml/src"),
        "-L",
        str(GGML / "build/bin"),
        "-lggml-base",
        "-Wl,-rpath," + str(GGML / "build/bin"),
        "-o",
        str(library),
    ]
    subprocess.run(command, check=True, capture_output=True)
    control = {
        "attempt": "20261112-EDGE0-05",
        "fixture_revision": "edge0-forward-v1",
        "seed": SEED,
        "geometry": GEOMETRY,
        "labels": LABELS,
        "input_policy": "Python Random(seed) uniform[-1,1], each row RMS-normalized, then F32",
        "weight_policy": "same RNG stream; uniform[-sqrt(3/cols),sqrt(3/cols)], F32 then pinned quantizer",
        "matrix_order": "d groups, u groups, per-expert gate/up/down, shared gate/up/down/scalar, readout",
        "scale_policy": "stored s[j] cycles [0.5,1.25,2.0]; a_g=(D_g x)/s exactly once",
        "router_input_sha256": hashlib.sha256(
            np.asarray(LOGITS, dtype="<f4").tobytes()
        ).hexdigest(),
        "routing": "full softmax double exp/ascending sum, F32 reciprocal and p; stable descending p ties ID ascending; selected double sum, clamp2^-14, divide to F32",
        "reduction": "per-expert group U; weighted accumulation in original routed order; add full-width shared expert; independent linear readout",
        "thresholds": {
            "final_maximum_absolute_error": 0.002,
            "final_relative_l2_error": 0.0002,
            "per_matrix": "2e-5 * max(sum(abs(products)), 1)",
        },
        "source_revision": "b7900cbb322926ccebdcd0c0c5a1280497ecff22",
        "llama_revision": "3cf03257f219afbe7334045ff7c6a06ac68c627d",
        "scope": "synthetic latent/shared/readout component; no trunk, checkpoint, model or GPU",
        "activation_control": "pinned GGML CPU Q8_0 for D/U/shared/readout and IQ4_NL down; Q8_K for Q3_K gate/up; F64 accumulation; does not emulate CUDA Q8_1, CUDA fast math or device reductions",
        "compiler_flags": FLAGS,
        "programs": {
            str(p.relative_to(REPO)): identity(p)
            for p in (
                Path(__file__),
                BRIDGE,
                library,
                SOURCE / "ggml/src/ggml-quants.c",
                SOURCE / "ggml/src/ggml-common.h",
                SOURCE / "ggml/src/ggml-cpu/ggml-cpu.c",
                GGML / "build/bin/libggml-base.dylib",
            )
        },
        "sources": {
            str(p.relative_to(REPO)): identity(p)
            for p in sorted((RUNTIME / "source").rglob("*"))
            if p.is_file()
        },
        "candidate_comparison_run": False,
    }
    # Freeze geometry, RNG, scales, thresholds, source and programs before any
    # fixture weight decoding, F64 forward output or candidate invocation.
    write_json(registration, control)
    oracle = Oracle(library)
    rng = random.Random(SEED)
    inputs = []
    for _ in range(GEOMETRY["T"]):
        row = [rng.uniform(-1, 1) for _ in range(256)]
        rms = math.sqrt(math.fsum(v * v for v in row) / len(row))
        inputs.append([f32(v / rms) for v in row])
    inputs = np.asarray(inputs, dtype="<f4")
    matrix_specs = [(f"d.{g}", 8, 256, 256) for g in range(3)]
    matrix_specs += [(f"u.{g}", 8, 256, 256) for g in range(3)]
    matrix_specs += [
        (f"{part}.{e}", kind, 256, 256)
        for e in range(5)
        for part, kind in (("gate", 11), ("up", 11), ("down", 20))
    ]
    matrix_specs += [
        ("shared_gate", 8, 256, 256),
        ("shared_up", 8, 256, 256),
        ("shared_down", 8, 256, 256),
        ("shared_scalar", 8, 1, 256),
        ("readout", 8, 17, 256),
    ]
    manifest = {"geometry": GEOMETRY, "matrices": [], "arrays": {}}
    matrices = {}

    def array(name, value, dtype):
        data = np.asarray(value, dtype=dtype)
        path = RUNTIME / (name + ".bin")
        path.write_bytes(data.tobytes())
        manifest["arrays"][name] = {
            "path": path.name,
            "shape": list(data.shape),
            "dtype": dtype,
            **identity(path),
        }

    for name, kind, rows, cols in matrix_specs:
        bound = math.sqrt(3.0 / cols)
        values = np.asarray(
            [[rng.uniform(-bound, bound) for _ in range(cols)] for _ in range(rows)],
            dtype="<f4",
        )
        raw, decoded = oracle.matrix(kind, values)
        path = RUNTIME / (name + ".ggml.bin")
        path.write_bytes(raw)
        matrices[name] = decoded
        manifest["matrices"].append(
            {
                "name": name,
                "type": kind,
                "rows": rows,
                "cols": cols,
                "path": path.name,
                **identity(path),
            }
        )
    scales = np.asarray(([0.5, 1.25, 2.0] * 86)[:256], dtype="<f4")
    routes, weights = zip(*(source_route(row, 3) for row in LOGITS), strict=True)
    assert list(routes) == [[3, 1, 4], [0, 1, 2], [4, 2, 0]]
    array("x.f32", inputs, "<f4")
    array("scales.f32", scales, "<f4")
    array("labels.u32", LABELS, "<u4")
    array("router_logits.f32", LOGITS, "<f4")
    array("routes.u32", routes, "<u4")
    array("route_weights.f32", weights, "<f4")
    # Seal every input and input manifest BEFORE evaluating expected outputs.
    write_json(RUNTIME / "inputs.json", manifest)
    write_json(
        RUNTIME / "input-seal.json",
        {
            "registration": identity(registration),
            "inputs": identity(RUNTIME / "inputs.json"),
            "candidate_comparison_run": False,
        },
    )

    expected = {"arrays": {}, "matrix_checks": []}

    def evaluate(quantized):
        outputs = []

        def mv(name, vector, tag, act_type=8):
            if quantized:
                vector = oracle.activation(act_type, vector)
            result, absolute_sums = matvec(matrices[name], vector)
            if not quantized:
                for suffix, values in (("", result), (".sumabs", absolute_sums)):
                    path = RUNTIME / (tag + suffix + ".f64.bin")
                    path.write_bytes(values.astype("<f8").tobytes())
                    expected["arrays"][tag + suffix] = {
                        "path": path.name,
                        "shape": list(values.shape),
                        "dtype": "<f8",
                        **identity(path),
                    }
                expected["matrix_checks"].append(
                    {
                        "matrix": name,
                        "input_tag": tag,
                        "output": tag,
                        "sumabs": tag + ".sumabs",
                    }
                )
            return result

        for t, x in enumerate(inputs):
            prefix = f"t{t}."
            latent = [mv(f"d.{g}", x, prefix + f"d.{g}") / scales for g in range(3)]
            mixed = np.zeros(256, dtype="<f8")
            for r, e in enumerate(routes[t]):
                g = LABELS[e]
                gate = mv(f"gate.{e}", latent[g], prefix + f"r{r}.gate", 15)
                up = mv(f"up.{e}", latent[g], prefix + f"r{r}.up", 15)
                hidden = gate / (1.0 + np.exp(-gate)) * up
                down = mv(f"down.{e}", hidden, prefix + f"r{r}.down")
                result = mv(f"u.{g}", down, prefix + f"r{r}.u")
                # Each expert U is applied before this ordered reduction.
                for h in range(256):
                    mixed[h] += float(weights[t][r]) * result[h]
            if not quantized:
                path = RUNTIME / (prefix + "latent.f64.bin")
                path.write_bytes(mixed.tobytes())
                expected["arrays"][prefix + "latent"] = {
                    "path": path.name,
                    "shape": [256],
                    "dtype": "<f8",
                    **identity(path),
                }
            sg = mv("shared_gate", x, prefix + "shared_gate")
            su = mv("shared_up", x, prefix + "shared_up")
            shared = mv(
                "shared_down", sg / (1.0 + np.exp(-sg)) * su, prefix + "shared_down"
            )
            scalar = mv("shared_scalar", x, prefix + "shared_scalar")[0]
            combined = mixed + shared / (1.0 + math.exp(-scalar))
            outputs.append(mv("readout", combined, prefix + "readout"))
        return np.asarray(outputs, dtype="<f8")

    floating = evaluate(False)
    quantized = evaluate(True)
    assert np.isfinite(floating).all() and np.isfinite(quantized).all()
    for name, values in (
        ("expected-readout.f64", floating),
        ("cpu-activation-control-readout.f64", quantized),
    ):
        path = RUNTIME / (name + ".bin")
        path.write_bytes(values.tobytes())
        expected["arrays"][name] = {
            "path": path.name,
            "shape": list(values.shape),
            "dtype": "<f8",
            **identity(path),
        }
    write_json(RUNTIME / "expected.json", expected)
    report = {
        "registration": identity(registration),
        "input_seal": identity(RUNTIME / "input-seal.json"),
        "expected_manifest": identity(RUNTIME / "expected.json"),
        "cases": 3,
        "matrix_operations": len(expected["matrix_checks"]),
        "finite": True,
        "floating_vs_source_cpu_activation_control": metrics(floating, quantized),
        "candidate_comparison_run": False,
        "whole_model_parity": False,
        "cuda_activation_emulation": False,
        "real_layer_control_run": False,
    }
    write_json(RUNTIME / "reference-report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    generate()
