// SPDX-License-Identifier: Apache-2.0
// Synthetic BF16 controls; failures expose aggregate errors only.
use super::*;

fn bf(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

fn exact(actual: &[f32], expected: &[f32], category: &str) -> Result<()> {
    ensure!(actual.len() == expected.len(), "{category}: length mismatch");
    let mismatch_count =
        actual.iter().zip(expected).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let signed_zero_count = actual
        .iter()
        .zip(expected)
        .filter(|(a, b)| **a == 0.0 && **b == 0.0 && a.to_bits() != b.to_bits())
        .count();
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0f32, f32::max);
    ensure!(
        mismatch_count == 0,
        "{category}: mismatches={mismatch_count}, signed_zero_mismatches={signed_zero_count}, max_abs={max_abs}"
    );
    Ok(())
}

#[test]
fn official_ordinary_rms_matches_bf16_normalize_then_gain_at_capi_and_head_widths()
-> Result<()> {
    let ctx = MetalContext::new()?;
    for (h, m, eps) in
        [(128, 1, 0.0), (128, 3, 1e-6), (256, 3, 1e-6), (2560, 1, 1e-6), (2560, 3, 0.0)]
    {
        let values: Vec<f32> =
            (0..m * h).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }).collect();
        let gains: Vec<f32> =
            (0..h).map(|i| bf(0.5 + (i % 192) as f32 / 128.0)).collect();
        let x = Tensor::from_f32_as_bf16(&ctx, &values, &[m, h])?;
        let w = Tensor::from_f32_as_bf16(&ctx, &gains, &[h])?;
        let out = Tensor::zeros(&ctx, &[m, h], DType::BF16)?;
        let in_place = Tensor::from_f32_as_bf16(&ctx, &values, &[m, h])?;
        let pass = ctx.begin_concurrent()?;
        rmsnorm_bf16_with_rounding(&ctx, &pass, &x, &w, &out, eps, 0.0, true)?;
        rmsnorm_bf16_with_rounding(
            &ctx, &pass, &in_place, &w, &in_place, eps, 0.0, true,
        )?;
        pass.commit()?.wait()?;
        let inv = 1.0 / (2.5f32 + eps).sqrt();
        let two: Vec<f32> = values
            .iter()
            .enumerate()
            .map(|(i, x)| bf(bf(*x * inv) * gains[i % h]))
            .collect();
        let one: Vec<f32> = values
            .iter()
            .enumerate()
            .map(|(i, x)| bf(*x * inv * gains[i % h]))
            .collect();
        ensure!(
            two.iter().zip(&one).any(|(a, b)| a.to_bits() != b.to_bits()),
            "ordinary RMS fixture failed to distinguish gain rounding"
        );
        exact(&out.to_f32()?, &two, "official ordinary RMS two-round chain")?;
        exact(&in_place.to_f32()?, &two, "official ordinary RMS in-place chain")?;
    }
    Ok(())
}

#[test]
fn official_ordinary_rms_rejects_f32_gain_and_delta_policy_and_preserves_legacy()
-> Result<()> {
    let ctx = MetalContext::new()?;
    let h = 128usize;
    let values: Vec<f32> = (0..h).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }).collect();
    let gains: Vec<f32> = (0..h).map(|i| bf(0.5 + (i % 96) as f32 / 128.0)).collect();
    let x = Tensor::from_f32_as_bf16(&ctx, &values, &[1, h])?;
    let w = Tensor::from_f32_as_bf16(&ctx, &gains, &[h])?;
    let f32_w = Tensor::from_f32(&ctx, &gains, &[h])?;
    let legacy = Tensor::zeros(&ctx, &[1, h], DType::BF16)?;
    let explicit_legacy = Tensor::zeros(&ctx, &[1, h], DType::BF16)?;
    let pass = ctx.begin_concurrent()?;
    ensure!(
        rmsnorm_bf16_with_rounding(&ctx, &pass, &x, &f32_w, &legacy, 1e-6, 0.0, true)
            .is_err(),
        "official ordinary RMS silently accepted promoting F32 gain"
    );
    ensure!(
        rmsnorm_bf16_with_rounding(&ctx, &pass, &x, &w, &legacy, 1e-6, 1.0, true)
            .is_err(),
        "official ordinary RMS accepted delta-gain policy"
    );
    rmsnorm_bf16(&ctx, &pass, &x, &w, &legacy, 1e-6, 1.0)?;
    rmsnorm_bf16_with_rounding(
        &ctx,
        &pass,
        &x,
        &w,
        &explicit_legacy,
        1e-6,
        1.0,
        false,
    )?;
    pass.commit()?.wait()?;
    exact(
        &legacy.to_f32()?,
        &explicit_legacy.to_f32()?,
        "legacy ordinary RMS entry changed",
    )
}

#[test]
fn official_ordinary_rms_preserves_raw_folded_gain_signed_zero() -> Result<()> {
    let ctx = MetalContext::new()?;
    let h = 128usize;
    let values: Vec<f32> =
        (0..h).map(|i| if i % 2 == 0 { -1.0 } else { 2.0 }).collect();
    let gains: Vec<f32> = (0..h)
        .map(|i| {
            if i % 3 == 0 {
                -0.0
            } else if i % 3 == 1 {
                0.0
            } else {
                1.25
            }
        })
        .collect();
    let x = Tensor::from_f32_as_bf16(&ctx, &values, &[1, h])?;
    let w = Tensor::from_f32_as_bf16(&ctx, &gains, &[h])?;
    let out = Tensor::zeros(&ctx, &[1, h], DType::BF16)?;
    let pass = ctx.begin_concurrent()?;
    rmsnorm_bf16_with_rounding(&ctx, &pass, &x, &w, &out, 1e-6, 0.0, true)?;
    pass.commit()?.wait()?;
    let inv = 1.0 / (2.5f32 + 1e-6).sqrt();
    let expected: Vec<f32> =
        values.iter().zip(&gains).map(|(x, w)| bf(bf(*x * inv) * *w)).collect();
    exact(&out.to_f32()?, &expected, "official ordinary RMS raw signed-zero gain")
}
