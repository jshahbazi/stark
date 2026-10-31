// SPDX-License-Identifier: Apache-2.0
use super::*;

#[test]
fn math_mode_pipeline_cache_keeps_default_and_safe_separate() -> Result<()> {
    let ctx = MetalContext::new()?;
    const SOURCE: &str = "#include <metal_stdlib>\nusing namespace metal;\nkernel void stark_math_mode_cache_probe(device float* out [[buffer(0)]]) { out[0] = 0.0f; }\n";
    let (before, _) = ctx.build_stats();
    let _ = ctx.pipeline("stark_math_mode_cache_probe", SOURCE, MslVersion::V3_1)?;
    ensure!(ctx.build_stats().0 == before + 1, "default pipeline was not cached");
    let _ = ctx.pipeline_with_math_mode(
        "stark_math_mode_cache_probe",
        SOURCE,
        MslVersion::V3_1,
        MslMathMode::Safe,
    )?;
    ensure!(
        ctx.build_stats().0 == before + 2,
        "safe pipeline reused default cache entry"
    );
    let _ = ctx.pipeline_with_math_mode(
        "stark_math_mode_cache_probe",
        SOURCE,
        MslVersion::V3_1,
        MslMathMode::Safe,
    )?;
    let _ = ctx.pipeline("stark_math_mode_cache_probe", SOURCE, MslVersion::V3_1)?;
    ensure!(
        ctx.build_stats().0 == before + 2,
        "math mode cache entries were not retained"
    );
    Ok(())
}
