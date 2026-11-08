"""Create tiny private independent scalar-rescale exponential controls."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import decimal
import hashlib
import json
import os
import platform
import struct
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(output, trace_path, base_fixture=None):
    import numpy as np
    import torch

    revision = "5c4886908584029761b579af026dcfb627c84070"
    if (
        torch.__version__.split("+")[0] != "2.14.1"
        or torch.version.git_version != revision
    ):
        raise ValueError("reference_identity_changed")
    output = output.resolve()
    if (
        output.exists()
        or subprocess.run(
            ["git", "check-ignore", "--quiet", "--", str(output)],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        ).returncode
    ):
        raise ValueError("output_must_be_new_and_ignored")
    observed = []
    observed_source = []
    trace_hash = None
    if trace_path:
        trace = json.loads(trace_path.read_text())
        if trace["backend"] != "cpu":
            raise ValueError("requires_source_cpu_trace")
        trace_hash = digest(trace_path)
        states = {}
        for record in trace["records"]:
            if record["stage"] == "state":
                path = trace_path.parent / record["file"]
                if digest(path) != record["sha256"]:
                    raise ValueError("trace_hash_changed")
                states[record["key_start"], record["pair"]] = struct.unpack(
                    "<3f", path.read_bytes()
                )
        if not 0 < len(states) <= 30:
            raise ValueError("trace_extent_outside_bounds")
        for (tile, pair), state in states.items():
            old = states[tile - 512, pair][0] if tile else -float("inf")
            observed.append(np.float32(old - state[0]))
            observed_source.append(np.float32(state[2]))
    rng = np.random.default_rng(98122)
    # The goldens are scalar libc and high-precision Decimal, independent of the
    # candidate's compensated range reduction and polynomial implementation.
    inputs = np.concatenate(
        (
            np.array(observed, dtype=np.float32),
            rng.uniform(-104, 0, 4096).astype(np.float32),
            np.linspace(-104, 0, 4097, dtype=np.float32),
            np.array([0, -float("inf")], dtype=np.float32),
        )
    )
    base_manifest = None
    if base_fixture:
        base_manifest = json.loads((base_fixture / "manifest.json").read_text())
        if (
            digest(base_fixture / "manifest.json")
            != "fb77820532a29404499e2f17287d48464061c1b1d7bb5a693dbeae04d6972ad6"
        ):
            raise ValueError("base_fixture_identity_changed")
        record = base_manifest["records"]["input"]
        path = base_fixture / record["file"]
        if digest(path) != record["sha256"]:
            raise ValueError("base_input_changed")
        inputs = np.concatenate(
            (np.fromfile(path, dtype="<f4"), np.array(observed, dtype=np.float32))
        )
    library = ctypes.CDLL(None)
    expf = library.expf
    expf.argtypes, expf.restype = [ctypes.c_float], ctypes.c_float
    source = np.array([expf(float(x)) for x in inputs], dtype=np.float32)
    actual_observed = (
        source[-len(observed) :] if base_fixture else source[: len(observed)]
    )
    if observed and np.any(
        actual_observed.view(np.uint32) != np.array(observed_source).view(np.uint32)
    ):
        raise ValueError("observed_source_factor_changed")
    with decimal.localcontext() as ctx:
        ctx.prec = 100

        def rounded_decimal(value):
            # Select by exact Decimal distances to adjacent F32 values rather
            # than relying on a potentially double-rounded F64 conversion.
            estimate = np.float32(float(value))
            bits = int(estimate.view(np.uint32))
            neighbors = range(max(0, bits - 1), bits + 2)
            best = min(
                neighbors,
                key=lambda b: (
                    abs(
                        value
                        - decimal.Decimal(
                            float(np.array([b], dtype=np.uint32).view(np.float32)[0])
                        )
                    ),
                    b & 1,
                ),
            )
            return np.array([best], dtype=np.uint32).view(np.float32)[0]

        nearest = np.array(
            [
                rounded_decimal(decimal.Decimal(float(x)).exp())
                if np.isfinite(x)
                else 0.0
                for x in inputs
            ],
            dtype=np.float32,
        )
    differences = source.view(np.uint32).astype(np.int64) - nearest.view(
        np.uint32
    ).astype(np.int64)
    if np.max(np.abs(differences)) > 1 or (
        not base_fixture and np.any(differences[: len(observed)])
    ):
        raise ValueError("source_nearest_control_changed")
    if base_manifest:
        for name, actual in (("source", source), ("nearest", nearest)):
            r = base_manifest["records"][name]
            path = base_fixture / r["file"]
            expected = np.fromfile(path, dtype="<f4")
            if digest(path) != r["sha256"] or np.any(
                expected.view(np.uint32) != actual[: len(expected)].view(np.uint32)
            ):
                raise ValueError("base_goldens_changed")
    output.mkdir(mode=0o700, parents=True)
    records = {}
    for name, values in (("input", inputs), ("source", source), ("nearest", nearest)):
        path = output / f"{name}.f32"
        path.write_bytes(values.astype("<f4").tobytes())
        records[name] = {"file": path.name, "sha256": digest(path)}
    total = sum((output / r["file"]).stat().st_size for r in records.values())
    if total > 256 * 1024:
        raise ValueError("fixture_byte_cap_exceeded")
    manifest = {
        "scope": "independent-larger-vision-source-rescale-exp"
        if base_fixture
        else "model-free-larger-vision-scalar-rescale-exp",
        "oracle": "decimal-exp-100-digits-and-source-libm-expf",
        "torch_source_revision": revision,
        "producer_sha256": digest(__file__),
        "source_platform": platform.platform(),
        "count": len(inputs),
        "bytes": total,
        "observed_count": len(observed),
        "observed_cpu_trace_sha256": trace_hash,
        "source_nearest_differences": int(np.count_nonzero(differences)),
        "maximum_source_nearest_ulp": int(np.max(np.abs(differences))),
        "records": records,
    }
    if base_fixture:
        manifest["base_manifest_sha256"] = digest(base_fixture / "manifest.json")
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return {
        "passed": True,
        "cases": len(inputs),
        "bytes": total,
        "source_nearest_differences": manifest["source_nearest_differences"],
        "manifest_sha256": digest(output / "manifest.json"),
    }


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observed-trace", type=Path)
    parser.add_argument("--base-fixture", type=Path)
    args = parser.parse_args()
    try:
        with (
            open(os.devnull, "w") as quiet,
            contextlib.redirect_stdout(quiet),
            contextlib.redirect_stderr(quiet),
        ):
            result = run(args.output, args.observed_trace, args.base_fixture)
    except Exception as error:  # noqa: BLE001 - keep private fixtures content-free
        print(json.dumps({"passed": False, "error_type": type(error).__name__}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
