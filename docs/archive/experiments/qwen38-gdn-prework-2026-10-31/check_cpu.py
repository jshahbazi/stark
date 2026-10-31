#!/usr/bin/env python3
"""Audit fixed-shape ownership and compile Metal offline; never open a GPU.

This checks indexing, source provenance and compilation. It does not establish
numerical parity or performance on a Metal device.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
METAL_ROOT = ROOT / "vendor/lily-core/src/kernels/metal"
HK, HV, D, SPLIT, LANES, SIMD_GROUPS = 16, 48, 128, 4, 32, 8
C = (2 * HK + HV) * D


def check_ownership() -> dict[str, int]:
    """Enumerate physical writes and compare with independent dense ranges."""
    channels: list[int] = []
    histories: list[int] = []
    gates: list[int] = []
    for group in range(2 * HK + HV):
        for lane in range(LANES):
            for component in range(4):
                channel = group * D + lane * 4 + component
                channels.append(channel)
                histories.extend(channel * 3 + tap for tap in range(3))
            if group >= 2 * HK and lane == 31:
                gates.extend(2 * (group - 2 * HK) + i for i in range(2))
    assert sorted(channels) == list(range(C))
    assert sorted(histories) == list(range(3 * C))
    assert sorted(gates) == list(range(2 * HV))

    # Independent reference history: a dense stream of tagged tokens per
    # channel. Candidate shifts each disjoint channel's three-element history.
    window = list(range(3 * C))
    reference = [window[c * 3 : c * 3 + 3] for c in range(C)]
    for step in range(4):
        for channel in reversed(channels):
            start = channel * 3
            token = (step + 1) * 3 * C + channel
            window[start] = window[start + 1]
            window[start + 1] = window[start + 2]
            window[start + 2] = token
        for channel, history in enumerate(reference):
            reference[channel] = history[1:] + [(step + 1) * 3 * C + channel]
        assert window == [value for history in reference for value in history]

    state_owners = bytearray(HV * D * D)
    raw_owners = bytearray(HV * D)
    rows_per_group = D // SPLIT
    rows_per_simd = rows_per_group // SIMD_GROUPS
    for group in range(HV * SPLIT):
        head, part = divmod(group, SPLIT)
        key_head = head // (HV // HK)
        assert 0 <= key_head < HK
        for simd in range(SIMD_GROUPS):
            for row in range(rows_per_simd):
                value = part * rows_per_group + simd * rows_per_simd + row
                raw_owners[head * D + value] += 1
                for lane in range(LANES):
                    for component in range(4):
                        key = lane * 4 + component
                        state_owners[(head * D + value) * D + key] += 1
    assert all(count == 1 for count in state_owners)
    assert all(count == 1 for count in raw_owners)
    return {
        "convolution_channel_owners": len(channels),
        "history_element_owners": len(histories),
        "gate_element_owners": len(gates),
        "state_element_owners": len(state_owners),
        "raw_element_owners": len(raw_owners),
        "history_steps_checked": 4,
        "baseline_qk_normalizations_per_layer": HV * SPLIT * 2,
        "prepared_qk_normalizations_per_layer": HK * 2,
        "baseline_gate_pairs_per_layer": HV * SPLIT,
        "prepared_gate_pairs_per_layer": HV,
        "prepared_qkv_bytes": C * 2,
        "additional_gate_scratch_bytes": HV * 2 * 4,
    }


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    prototype = HERE / "prepared.metal"
    source = prototype.read_text()
    hashes = {
        name: source_hash(METAL_ROOT / name)
        for name in ("gdn.metal", "official_gdn_k1_helper.metal")
    }
    for digest in hashes.values():
        assert digest in source, "prototype source provenance needs review"
    entrypoints = re.findall(r"kernel void (\w+)\(", source)
    assert set(entrypoints) == {
        "qwen38_gdn_conv_prepare_research",
        "qwen38_gdn_recurrence_research",
        "qwen38_gdn_recurrence_staged_research",
    }
    assert "device bfloat* prepared [[buffer(3)]]" in source
    assert source.count("device const bfloat* prepared [[buffer(0)]]") == 2
    flags = [
        "-std=metal3.1",
        "-fmetal-math-mode=safe",
        "-fmetal-math-fp32-functions=fast",
        "-Wno-c++17-extensions",
        "-Werror",
    ]
    # xcrun metal is an offline compiler. No executable, model, device context
    # or GPU runtime is created. Compiler products remain temporary.
    with tempfile.TemporaryDirectory(prefix="stark-gdn-prework-") as directory:
        subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", *flags, "-c", str(prototype),
             "-o", str(Path(directory) / "prepared.air")],
            check=True,
        )
        ir = Path(directory) / "prepared.ll"
        subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", *flags, "-S", "-emit-llvm",
             str(prototype), "-o", str(ir)],
            check=True,
        )
        ir_text = ir.read_text()
        for name in entrypoints:
            assert f"@{name}(" in ir_text, f"missing compiled entrypoint: {name}"
        ir_hash = source_hash(ir)
    compiler = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "--version"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()[0]
    print(json.dumps({
        "scope": "CPU ownership and offline compilation only",
        "ownership": check_ownership(),
        "compiler": compiler,
        "flags": flags,
        "entrypoints": entrypoints,
        "prototype_sha256": source_hash(prototype),
        "harness_sha256": source_hash(ROOT / "vendor/lily-core/tests/gdn_prework_research.rs"),
        "reference_source_sha256": hashes,
        "air_ir_sha256": ir_hash,
    }, indent=2))


if __name__ == "__main__":
    main()
