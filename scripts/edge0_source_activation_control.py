"""Preregister a source Q8_1 value control over the sealed private fixture.

This measures activation quantization only, retaining independent F64 matvec
and nonlinearities. It does not emulate CUDA reductions, fast math, a whole
source layer, or the source's BF16 shared scalar gate.
"""

import ctypes
import json
import math
import struct
from pathlib import Path

import numpy as np
from edge0_native_fixture import (
    RUNTIME,
    Oracle,
    f32,
    identity,
    matvec,
    metrics,
    write_json,
)


def q81_values(values):
    """Pinned iq_kernels.cu::q8_1_store/q8_1_finite.hpp value semantics."""
    values = np.asarray(values, dtype="<f4")
    output = []
    for offset in range(0, len(values), 32):
        block = [float(x) for x in values[offset : offset + 32]]
        amax = max(abs(v) for v in block)
        d = f32(amax / 127.0)
        d = math.copysign(min(abs(d), 65504.0), d)
        stored_d = struct.unpack("<e", struct.pack("<e", d))[0]
        for x in block:
            ratio = f32(x / d) if amax else 0.0
            q = math.floor(ratio + 0.5) if ratio >= 0 else math.ceil(ratio - 0.5)
            output.append(stored_d * max(-127, min(127, q)))
    return np.asarray(output, dtype="<f8")


def main():
    registration = RUNTIME / "source-activation-control-registration.json"
    if registration.exists():
        raise SystemExit("Existing control registration preserved")
    paths = [
        Path(__file__),
        RUNTIME / "fixture-registration.json",
        RUNTIME / "input-seal.json",
        RUNTIME / "inputs.json",
        RUNTIME / "expected.json",
        RUNTIME / "source/edge0-strata/include/strata/kernels/q8_1_finite.hpp",
    ]
    write_json(
        registration,
        {
            "revision": "edge0-forward-source-activation-control-v1",
            "controls": "unchanged sealed corpus/matrices/scales/routing; Q8_1 at D/U/shared/readout; CPU Q8_K gate/up and Q8_0 IQ4_NL down",
            "arithmetic": "host F32 division, half-stored d, ties away from zero int8, symmetric decoded activation values; independent F64 products/accumulation/nonlinearities",
            "limits": "activation value quantization only; no CUDA instruction/fastmath/reduction equivalence; no BF16 shared scalar gate emulation; no real layer or whole-model comparison",
            "files": {str(p): identity(p) for p in paths},
            "candidate_comparison_run": False,
        },
    )
    manifest = json.loads((RUNTIME / "inputs.json").read_text())
    expected = json.loads((RUNTIME / "expected.json").read_text())
    oracle = Oracle(RUNTIME / "libedge0_forward_reference.dylib")
    matrices = {}
    for entry in manifest["matrices"]:
        data = (RUNTIME / entry["path"]).read_bytes()
        assert identity(RUNTIME / entry["path"])["sha256"] == entry["sha256"]
        cols, rows = entry["cols"], entry["rows"]
        row_bytes = len(data) // rows
        decoded = []
        for r in range(rows):
            raw = ctypes.create_string_buffer(data[r * row_bytes : (r + 1) * row_bytes])
            out = np.empty(cols, dtype="<f4")
            assert (
                oracle.lib.edge0_forward_unpack(
                    entry["type"], raw, cols, out.ctypes.data
                )
                == 0
            )
            decoded.append(out)
        matrices[entry["name"]] = decoded

    def array(name):
        entry = manifest["arrays"][name]
        path = RUNTIME / entry["path"]
        assert identity(path)["sha256"] == entry["sha256"]
        return np.frombuffer(path.read_bytes(), dtype=entry["dtype"]).reshape(
            entry["shape"]
        )

    x, scales = array("x.f32"), array("scales.f32")
    labels, routes, weights = (
        array("labels.u32"),
        array("routes.u32"),
        array("route_weights.f32"),
    )

    def mv(name, vector, act=9):
        vector = q81_values(vector) if act == 9 else oracle.activation(act, vector)
        return matvec(matrices[name], vector)[0]

    outputs = []
    for t, values in enumerate(x):
        latent = [mv(f"d.{g}", values) / scales for g in range(3)]
        mixed = np.zeros(256, dtype="<f8")
        for r, e in enumerate(routes[t]):
            group = labels[e]
            gate, up = (
                mv(f"gate.{e}", latent[group], 15),
                mv(f"up.{e}", latent[group], 15),
            )
            hidden = gate / (1.0 + np.exp(-gate)) * up
            down = mv(f"down.{e}", hidden, 8)
            result = mv(f"u.{group}", down)
            for h in range(256):
                mixed[h] += float(weights[t][r]) * result[h]
        gate, up = mv("shared_gate", values), mv("shared_up", values)
        shared = mv("shared_down", gate / (1.0 + np.exp(-gate)) * up)
        # Retain the same synthetic scalar gate, isolating activation controls.
        scalar = matvec(matrices["shared_scalar"], values)[0][0]
        combined = mixed + shared / (1.0 + math.exp(-scalar))
        outputs.append(mv("readout", combined))
    outputs = np.asarray(outputs, dtype="<f8")
    reference_entry = expected["arrays"]["expected-readout.f64"]
    reference = np.frombuffer(
        (RUNTIME / reference_entry["path"]).read_bytes(), dtype="<f8"
    ).reshape(3, 17)
    path = RUNTIME / "source-activation-control-readout.f64.bin"
    path.write_bytes(outputs.tobytes())
    report = {
        "registration": identity(registration),
        "output": identity(path),
        "cases": 3,
        "floating_vs_source_activation_values": metrics(reference, outputs),
        "cuda_arithmetic_parity": False,
        "whole_model_parity": False,
        "real_layer_control_run": False,
    }
    write_json(RUNTIME / "source-activation-control-report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
