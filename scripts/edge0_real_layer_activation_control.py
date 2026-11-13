"""Bounded, CPU-only layer-0 activation-value control on source weight spans.

The input is a hypothetical normalized synthetic vector, never a hidden state
obtained from inference. No vectors, probabilities or decoded weights persist.
This measures floating versus quantized activation values over selected actual
weights; source CUDA arithmetic and whole-model equivalence are excluded.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import random
import resource
import sys
from pathlib import Path

import numpy as np
from edge0_native_fixture import REPO, RUNTIME, Oracle, identity, metrics, source_route
from edge0_source_activation_control import q81_values

sys.path.insert(0, str(REPO / "tools/converter"))
from stark_convert.edge0_source import (
    EDGE0_FILES,
    read_gguf,
    read_native_experts,
    read_pack_index,
)

SOURCE = REPO / "runtime/edge0-native-v2-evaluation"
SEED = 0xED6E004
PREPARED_SHA256 = "8dc700fe5cd5f6644ed9f062eb391ce32cb0a09b76a2e19ec44539bf89200a5e"
PAYLOAD_LIMIT = 110_000_000
SCRATCH_LIMIT = 1 << 30


def require(value, code):
    if not value:
        raise RuntimeError(code)


def write_new(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def bf16(values):
    bits = np.asarray(values, dtype="<f4").view("<u4")
    bits = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return bits.view("<f4").astype("<f8")


def silu(values):
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = values[positive] / (1 + np.exp(-values[positive]))
    exp = np.exp(values[~positive])
    output[~positive] = values[~positive] * exp / (1 + exp)
    return output


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1 + exp)


def file_state(path):
    value = path.stat()
    return {"size": value.st_size, "inode": value.st_ino, "mtime_ns": value.st_mtime_ns}


class Spans:
    def __init__(self, oracle):
        self.oracle = oracle
        self.total = 0
        self.reads = []

    def raw(self, filename, offset, count):
        require(self.total + count <= PAYLOAD_LIMIT, "payload-limit")
        path = SOURCE / filename
        require(0 <= offset <= path.stat().st_size - count, "span-bounds")
        fd = os.open(path, os.O_RDONLY)
        try:
            data = os.pread(fd, count, offset)
        finally:
            os.close(fd)
        require(len(data) == count, "short-span")
        self.total += count
        self.reads.append(
            {
                "file": filename,
                "offset": offset,
                "bytes": count,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        return data

    def matrix(self, filename, offset, rows, cols, kind):
        if kind == "bf16":
            data = self.raw(filename, offset, rows * cols * 2)
            words = np.frombuffer(data, dtype="<u2").astype("<u4") << 16
            result = words.view("<f4").reshape(rows, cols)
        else:
            row_bytes = self.oracle.lib.edge0_forward_row_bytes(kind, cols)
            require(row_bytes > 0, "unsupported-row")
            data = self.raw(filename, offset, rows * row_bytes)
            result = np.empty((rows, cols), dtype="<f4")
            raw = ctypes.create_string_buffer(data)
            for r in range(rows):
                ptr = ctypes.c_void_p(ctypes.addressof(raw) + r * row_bytes)
                require(
                    self.oracle.lib.edge0_forward_unpack(
                        kind, ptr, cols, result[r].ctypes.data
                    )
                    == 0,
                    "decode-failed",
                )
        require(np.isfinite(result).all(), "nonfinite-weight")
        require(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < SCRATCH_LIMIT,
            "rss-limit",
        )
        return result


def run(args):
    output = args.output
    registration = output.with_suffix(".registration.json")
    require(
        not output.exists() and not registration.exists(), "existing-attempt-preserved"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = SOURCE / "native-source.json"
    require(
        identity(receipt_path)["sha256"] == PREPARED_SHA256, "prepared-receipt-identity"
    )
    receipt = json.loads(receipt_path.read_text())
    for name in ("index.txt", "native_experts.txt", "labels.bin", "scales/l0.f32"):
        require(
            identity(SOURCE / name)
            == {
                "bytes": EDGE0_FILES[name]["size"],
                "sha256": EDGE0_FILES[name]["sha256"],
            },
            "metadata-identity",
        )
    # Header/index reads contain no tensor payload and happen before registration.
    skeleton = read_gguf(SOURCE / "model-skeleton.gguf")
    pack = read_pack_index(SOURCE / "index.txt", EDGE0_FILES["dense.bin"]["size"])
    experts = read_native_experts(
        SOURCE / "native_experts.txt", EDGE0_FILES["experts.bin"]["size"]
    )
    names = [
        "blk.0.ffn_gate_inp.weight",
        "blk.0.ffn_gate_inp_shexp.weight",
        "blk.0.ffn_gate_shexp.weight",
        "blk.0.ffn_up_shexp.weight",
        "blk.0.ffn_down_shexp.weight",
    ]
    paths = [
        Path(__file__),
        REPO / "scripts/edge0_native_fixture.py",
        REPO / "scripts/edge0_source_activation_control.py",
        REPO / "scripts/fixtures/edge0_forward_reference.cpp",
        RUNTIME / "libedge0_forward_reference.dylib",
        SOURCE / "native-source.json",
        SOURCE / "index.txt",
        SOURCE / "native_experts.txt",
        SOURCE / "labels.bin",
        SOURCE / "scales/l0.f32",
    ]
    payload_files = (
        "latent_dense_q8.bin",
        "experts.bin",
        "dense.bin",
        "model-skeleton.gguf",
    )
    states = {name: file_state(SOURCE / name) for name in payload_files}
    for name in payload_files:
        require(states[name]["size"] == EDGE0_FILES[name]["size"], "payload-file-size")
    rng = random.Random(SEED)
    x = np.asarray([rng.uniform(-1, 1) for _ in range(2560)], dtype="<f8")
    x = (x / math.sqrt(math.fsum(float(v * v) for v in x) / 2560)).astype("<f4")
    policy = {
        "attempt": "20261112-EDGE0-05-real-layer0-activation-v1",
        "seed": SEED,
        "scope": "actual layer0 latent FFN and full-width shared expert weights with hypothetical RMS-normalized synthetic x; no actual trunk hidden state or whole-model inference",
        "geometry": {"H": 2560, "ELL": 1792, "I": 640, "G": 8, "E": 512, "K": 10},
        "input": {
            "generator": "Python Random(seed), uniform[-1,1], F64 RMS normalization then F32",
            "sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        },
        "routing": "actual router BF16 weights; BF16-rounded synthetic x; F64 NumPy dot then logits cast F32, pinned full-softmax/stable-top10/renorm; same IDs and weights held fixed across all three policies",
        "floating": "decoded stored weights; F64 matvec/SwiGLU/sigmoid/reduction, stored s reciprocal applied once after D",
        "cpu_control": "Q8_0 activation values D/U/shared; Q8_K expert gate/up, Q8_0 expert down; F64 arithmetic retained",
        "q81_control": "source Q8_1 activation values D/U/shared; CPU Q8_K expert gate/up, Q8_0 expert down; F64 arithmetic retained",
        "shared_scalar": "actual BF16 scalar weights and BF16-rounded synthetic x, F64 dot/sigmoid held identical across policies",
        "reduction": "perexpert U then original routed-order weighted sum, add fullwidth shared expert",
        "limits": "no CUDA instructions/reductions/fastmath equivalence, source model state, quality or wholemodel parity; no readout head",
        "caps": {
            "actual_tensor_payload_bytes": PAYLOAD_LIMIT,
            "peak_process_rss_bytes": SCRATCH_LIMIT,
        },
        "source_spans": {
            name: vars(
                pack[name] if pack[name].kind != "q8_0" else skeleton.tensors[name]
            )
            for name in names
        },
        "selected_span_policy": "read full router then exact top10 expert records; only D/U groups referenced by these routes, each matrix once; shared tensors once; no other payload",
        "files": {str(p): identity(p) for p in paths},
        "payload_file_states": states,
        "prepared_source_file_identities": {
            name: receipt["files"][name] for name in payload_files
        },
        "numpy": np.__version__,
        "blas_threads_requested": 1,
        "persisted_data": "counts, hashes, metrics and byte spans only; no vector/probability/decoded weight payloads",
    }
    write_new(registration, policy)
    try:
        oracle = Oracle(RUNTIME / "libedge0_forward_reference.dylib")
        spans = Spans(oracle)
        router = pack[names[0]]
        router_matrix = spans.matrix("dense.bin", router.offset, 512, 2560, "bf16")
        router_values = router_matrix.astype("<f8") @ bf16(x)
        ids, weights = source_route(router_values.astype("<f4").tolist(), 10)
        del router_matrix, router_values
        labels = np.frombuffer((SOURCE / "labels.bin").read_bytes(), dtype="<u4")[:512]
        scales = np.frombuffer((SOURCE / "scales/l0.f32").read_bytes(), dtype="<f4")
        require(np.isfinite(scales).all() and (scales > 0).all(), "invalid-scales")
        groups = sorted({int(labels[e]) for e in ids})
        policies = ("floating", "cpu", "q81")

        def activation(vector, policy_name, kind=8):
            if policy_name == "floating":
                return np.asarray(vector, dtype="<f8")
            if policy_name == "q81" and kind == 8:
                return q81_values(vector)
            return oracle.activation(kind, vector)

        def multiply(matrix, vector):
            result = matrix.astype("<f8") @ vector
            require(np.isfinite(result).all(), "nonfinite-result")
            require(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < SCRATCH_LIMIT,
                "rss-limit",
            )
            return result

        d_group_bytes = 1792 * (2560 // 32) * 34
        u_group_bytes = 2560 * (1792 // 32) * 34
        latent = {name: {} for name in policies}
        stage_metrics = {"D": [], "expert_down": [], "U": [], "shared": []}
        for group in groups:
            matrix = spans.matrix(
                "latent_dense_q8.bin", group * d_group_bytes, 1792, 2560, 8
            )
            for name in policies:
                latent[name][group] = multiply(matrix, activation(x, name)) / scales
            stage_metrics["D"].append(
                {
                    name: metrics(latent["floating"][group], latent[name][group])
                    for name in policies[1:]
                }
            )
            del matrix
        expert_outputs = {name: [] for name in policies}
        for e in ids:
            base = e * experts["record_bytes"]
            gate = spans.matrix("experts.bin", base, 640, 1792, 11)
            up = spans.matrix(
                "experts.bin", base + experts["gate_bytes"], 640, 1792, 11
            )
            hidden = {}
            for name in policies:
                a = activation(latent[name][int(labels[e])], name, 15)
                hidden[name] = silu(multiply(gate, a)) * multiply(up, a)
            del gate, up
            down = spans.matrix(
                "experts.bin",
                base + experts["gate_bytes"] + experts["up_bytes"],
                1792,
                640,
                20,
            )
            for name in policies:
                # IQ4_NL CPU traits consume Q8_0, including the hybrid Q8_1 control.
                a = (
                    hidden[name]
                    if name == "floating"
                    else oracle.activation(8, hidden[name])
                )
                expert_outputs[name].append(multiply(down, a))
            stage_metrics["expert_down"].append(
                {
                    name: metrics(
                        expert_outputs["floating"][-1], expert_outputs[name][-1]
                    )
                    for name in policies[1:]
                }
            )
            del down, hidden
        expanded = {name: [None] * 10 for name in policies}
        for group in groups:
            matrix = spans.matrix(
                "latent_dense_q8.bin",
                8 * d_group_bytes + group * u_group_bytes,
                2560,
                1792,
                8,
            )
            for r, e in enumerate(ids):
                if labels[e] != group:
                    continue
                for name in policies:
                    expanded[name][r] = multiply(
                        matrix, activation(expert_outputs[name][r], name)
                    )
                stage_metrics["U"].append(
                    {
                        name: metrics(expanded["floating"][r], expanded[name][r])
                        for name in policies[1:]
                    }
                )
            del matrix
        mixed = {name: np.zeros(2560, dtype="<f8") for name in policies}
        for r in range(10):
            for name in policies:
                mixed[name] += float(weights[r]) * expanded[name][r]
        gate_ref, up_ref, down_ref = (skeleton.tensors[names[i]] for i in (2, 3, 4))
        gate = spans.matrix("model-skeleton.gguf", gate_ref.offset, 640, 2560, 8)
        up = spans.matrix("model-skeleton.gguf", up_ref.offset, 640, 2560, 8)
        hidden = {}
        for name in policies:
            a = activation(x, name)
            hidden[name] = silu(multiply(gate, a)) * multiply(up, a)
        del gate, up
        down = spans.matrix("model-skeleton.gguf", down_ref.offset, 2560, 640, 8)
        shared = {
            name: multiply(down, activation(hidden[name], name)) for name in policies
        }
        del down, hidden
        scalar_ref = pack[names[1]]
        scalar_matrix = spans.matrix("dense.bin", scalar_ref.offset, 1, 2560, "bf16")
        scalar = sigmoid(float(multiply(scalar_matrix, bf16(x))[0]))
        for name in policies:
            mixed[name] += scalar * shared[name]
        stage_metrics["shared"].append(
            {name: metrics(shared["floating"], shared[name]) for name in policies[1:]}
        )

        def aggregate(values):
            return {
                name: {
                    "maximum_absolute_error": max(
                        v[name]["maximum_absolute_error"] for v in values
                    ),
                    "maximum_case_relative_l2_error": max(
                        v[name]["relative_l2_error"] for v in values
                    ),
                    "all_finite": all(v[name]["finite"] for v in values),
                }
                for name in policies[1:]
            }

        require(
            all(file_state(SOURCE / name) == state for name, state in states.items()),
            "source-state-changed",
        )
        require(
            all(identity(Path(path)) == seal for path, seal in policy["files"].items()),
            "program-or-metadata-changed",
        )
        report = {
            "status": "compared",
            "registration": identity(registration),
            "layer": 0,
            "input_rows": 1,
            "routed_experts": 10,
            "selected_group_count": len(groups),
            "route_ids_sha256": hashlib.sha256(
                np.asarray(ids, dtype="<u4").tobytes()
            ).hexdigest(),
            "route_weights_sha256": hashlib.sha256(
                np.asarray(weights, dtype="<f4").tobytes()
            ).hexdigest(),
            "final_floating_vs_quantized_activation_values": {
                name: metrics(mixed["floating"], mixed[name]) for name in policies[1:]
            },
            "stage_metrics": {
                stage: aggregate(values) for stage, values in stage_metrics.items()
            },
            "output_hashes": {
                name: hashlib.sha256(values.astype("<f8").tobytes()).hexdigest()
                for name, values in mixed.items()
            },
            "actual_payload_bytes_read": spans.total,
            "selected_spans": spans.reads,
            "peak_process_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "hypothetical_synthetic_input": True,
            "route_held_fixed": True,
            "source_cuda_arithmetic_parity": False,
            "whole_model_parity": False,
            "model_loaded": False,
        }
        write_new(output, report)
        print(
            json.dumps(
                {
                    "status": "compared",
                    "report": identity(output),
                    "actual_payload_bytes_read": spans.total,
                    "final_differences": report[
                        "final_floating_vs_quantized_activation_values"
                    ],
                },
                sort_keys=True,
            )
        )
    except Exception as error:
        write_new(
            output,
            {
                "status": "error",
                "failure_code": str(error)
                if type(error) is RuntimeError
                else type(error).__name__,
                "registration": identity(registration),
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
        return 0
    except Exception as error:  # noqa: BLE001 -- Content-free failure codes only.
        print(
            json.dumps(
                {
                    "status": "error",
                    "failure_code": str(error)
                    if type(error) is RuntimeError
                    else type(error).__name__,
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
