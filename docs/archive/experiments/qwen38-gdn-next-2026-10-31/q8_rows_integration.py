"""Patch only an isolated, frozen official-MTP source archive for this trial.

No production checkout, weights, runtime selectors, builds or GPU operations.
The caller owns archival/manifests, dedicated Cargo targets and measurements.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

BASE_COMMIT = "959ec981354acab313a9a8d709933fb2f2649b17"
MODULE_REL = Path("vendor/lily-core/src/kernels/official_q8_projection.rs")
METAL_REL = Path("vendor/lily-core/src/kernels/metal/qwen38_gdn_q8_rows_research.metal")
MODULE_SHA256 = "a7b311927a75a99bfe8987288fb91b703e1b6c0ae09c46beda0204d62298e01b"
SHADER = Path(__file__).with_name("q8_rows.metal")
RESEARCH_ROOT = Path(__file__).resolve().parents[4]

ADDITION = """
        // Fixed research route in this copied source only. M1 continues to
        // use the original ordinary projection; no environment selector.
        if rows > 1 {
            let entry = match rows {
                2 => "qwen38_q8_gdn_rows_2_research",
                3 => "qwen38_q8_gdn_rows_3_research",
                4 => "qwen38_q8_gdn_rows_4_research",
                _ => unreachable!(),
            };
            let reuse = ctx.pipeline_with_math_mode(
                entry,
                QWEN38_GDN_Q8_ROWS_RESEARCH,
                MslVersion::V4_0,
                MslMathMode::Safe,
            )?;
            if reuse.thread_execution_width() == 32 {
                return pass.dispatch_at(
                    &reuse,
                    &[
                        w.codes.binding(), w.scales.binding(), w.biases.binding(),
                        x.binding(), y.binding(),
                    ],
                    &[&u32_bytes(k), &u32_bytes(n)],
                    Grid::Threadgroups {
                        groups: (1, n / 8, 1),
                        threadgroup: (32, 2, 1),
                    },
                );
            }
        }
"""


def patch_source(source_root: Path) -> list[str]:
    """Return changed relative paths; refuse a source other than frozen 959ec981."""
    if (source_root / ".git").exists() or not source_root.resolve().is_relative_to(
        (RESEARCH_ROOT / "runtime").resolve()
    ):
        raise ValueError("q8_rows_requires_isolated_runtime_archive")
    module = source_root / MODULE_REL
    raw = module.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MODULE_SHA256:
        raise ValueError("q8_rows_frozen_module_identity")
    source = raw.decode("utf-8")
    source_anchor = 'const SOURCE: &str = include_str!("metal/official_q8_gdn_decode.metal");'
    route_anchor = "    if stock {\n        quant::check_quant(w)?;"
    if source.count(source_anchor) != 1 or source.count(route_anchor) != 1:
        raise ValueError("q8_rows_integration_anchor")
    source = source.replace(
        source_anchor,
        source_anchor + '\nconst QWEN38_GDN_Q8_ROWS_RESEARCH: &str = '
        'include_str!("metal/qwen38_gdn_q8_rows_research.metal");',
        1,
    )
    source = source.replace(route_anchor, route_anchor + ADDITION, 1)
    metal = source_root / METAL_REL
    if metal.exists():
        raise ValueError("q8_rows_shader_destination_exists")
    metal.parent.mkdir(parents=True, exist_ok=True)
    metal.write_bytes(SHADER.read_bytes())
    module.write_text(source, encoding="utf-8", newline="\n")
    return [MODULE_REL.as_posix(), METAL_REL.as_posix()]
