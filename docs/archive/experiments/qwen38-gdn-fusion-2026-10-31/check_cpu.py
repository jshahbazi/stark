#!/usr/bin/env python3
"""Enumerate fusion ownership and compile both capture policies offline."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ownership(threads: int) -> dict[str, int]:
    state = bytearray(48 * 128 * 128)
    raw = bytearray(48 * 128)
    gated = bytearray(48 * 128)
    groups = threads // 32
    rows = 128 // groups
    for head in range(48):
        key_head = head // 3
        assert 0 <= key_head < 16
        for sg in range(groups):
            for j in range(rows):
                row = sg * rows + j
                raw[head * 128 + row] += 1
                for lane in range(32):
                    for i in range(4):
                        state[(head * 128 + row) * 128 + lane * 4 + i] += 1
        for lane in range(32):
            for i in range(4):
                gated[head * 128 + lane * 4 + i] += 1
    assert all(x == 1 for x in state)
    assert all(x == 1 for x in raw)
    assert all(x == 1 for x in gated)
    return {
        "threadgroups": 48, "threads_per_group": threads,
        "rows_per_simdgroup": rows, "state_scalars_per_thread": rows * 4,
        "state_unique_writes": len(state), "raw_unique_writes": len(raw),
        "gated_unique_writes": len(gated), "threadgroup_scratch_bytes": 1800,
    }


def main() -> None:
    shader = HERE / "fused.metal"
    production = ROOT / "vendor/lily-core/src/kernels/metal/gdn.metal"
    helpers = production.with_name("official_gdn_k1_helper.metal")
    text = shader.read_text()
    assert sha(production) in text and sha(helpers) in text
    flags = [
        "-std=metal3.1", "-fmetal-math-mode=safe", "-fmetal-math-fp32-functions=fast",
        "-Wno-c++17-extensions", "-Werror",
    ]
    compiles = []
    with tempfile.TemporaryDirectory(prefix="stark-gdn-fusion-") as temp:
        for capture in (0, 1):
            air = Path(temp) / f"fusion-{capture}.air"
            ir = Path(temp) / f"fusion-{capture}.ll"
            command = ["xcrun", "-sdk", "macosx", "metal", *flags, f"-DRESEARCH_CAPTURE_RAW={capture}"]
            subprocess.run([*command, "-c", str(shader), "-o", str(air)], check=True)
            subprocess.run([*command, "-S", "-emit-llvm", str(shader), "-o", str(ir)], check=True)
            ir_text = ir.read_text()
            for threads in (256, 512):
                assert f"@qwen38_gdn_fused{threads}_research(" in ir_text
            compiles.append({"capture_raw": bool(capture), "ir_sha256": sha(ir)})
    print(json.dumps({
        "scope": "CPU ownership and offline compilation only",
        "ownership": {str(nt): ownership(nt) for nt in (256, 512)},
        "compiler": subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "--version"],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()[0],
        "flags": flags, "compiles": compiles,
        "shader_sha256": sha(shader), "check_script_sha256": sha(Path(__file__)),
        "reference_source_sha256": sha(production), "reference_helper_sha256": sha(helpers),
        "harness_sha256": sha(ROOT / "vendor/lily-core/tests/gdn_fusion_research.rs"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
