// SPDX-License-Identifier: Apache-2.0
// Synthetic correctness controls for unchanged BF16 checkpoint gate bytes.
use super::*;

#[test]
fn direct_bf16_gate_matches_host_and_preserves_weight_bits() -> Result<()> {
    let ctx = MetalContext::new()?;
    let (m, n, k) = (3, 4, 128);
    let values: Vec<f32> =
        (0..m * k).map(|i| (i as i32 % 17 - 8) as f32 / 64.0).collect();
    let weights: Vec<f32> =
        (0..n * k).map(|i| (i as i32 % 13 - 6) as f32 / 32.0).collect();
    let x = Tensor::from_f32_as_bf16(&ctx, &values, &[m, k])?;
    let w = QuantWeights::bf16(Tensor::from_f32_as_bf16(&ctx, &weights, &[n, k])?)?;
    let original = w.codes.raw_bytes().to_vec();
    let y = Tensor::zeros(&ctx, &[m, n], DType::BF16)?;
    let yf = Tensor::zeros(&ctx, &[n], DType::F32)?;
    let pass = ctx.begin_concurrent()?;
    dense_bf16_project(&ctx, &pass, &x, &w.codes, &y)?;
    gemv_quant(&ctx, &pass, &w, &x.view(0, &[k])?, &yf)?;
    pass.commit()?.wait()?;
    let actual = y.to_f32()?;
    let actual_f = yf.to_f32()?;
    for row in 0..m {
        for out in 0..n {
            let expected: f32 =
                (0..k).map(|col| values[row * k + col] * weights[out * k + col]).sum();
            ensure!(
                actual[row * n + out].to_bits()
                    == half::bf16::from_f32(expected).to_f32().to_bits(),
                "BF16 gate projection mismatch"
            );
            if row == 0 {
                ensure!(
                    actual_f[out].to_bits() == expected.to_bits(),
                    "F32 gate projection mismatch"
                );
            }
        }
    }
    ensure!(w.codes.raw_bytes() == original, "BF16 gate changed its source values");
    Ok(())
}

#[test]
fn dense_hc_gate_uses_folded_or_delta_norm_gain_explicitly() -> Result<()> {
    let ctx = MetalContext::new()?;
    let (h, g, n, m) = (128, 4, 4, 2);
    let k = h * g;
    let values: Vec<f32> =
        (0..m * k).map(|i| (i as i32 % 19 - 9) as f32 / 16.0).collect();
    let norm: Vec<f32> = (0..k).map(|i| 1.0 + (i % 7) as f32 / 64.0).collect();
    let weights: Vec<f32> =
        (0..n * k).map(|i| (i as i32 % 11 - 5) as f32 / 128.0).collect();
    let x = Tensor::from_f32_as_bf16(&ctx, &values, &[m, k])?;
    let norm_tensor = Tensor::from_f32_as_bf16(&ctx, &norm, &[k])?;
    let w = QuantWeights::bf16(Tensor::from_f32_as_bf16(&ctx, &weights, &[n, k])?)?;
    for bias in [0.0, 1.0] {
        let y = Tensor::zeros(&ctx, &[m, n], DType::BF16)?;
        let pass = ctx.begin_concurrent()?;
        hc_dense_bf16_inject(&ctx, &pass, &x, &norm_tensor, &w, &y, h, g, 1e-6, bias)?;
        pass.commit()?.wait()?;
        let actual = y.to_f32()?;
        for row in 0..m {
            for out in 0..n {
                let mut expected = 0.0;
                for stream in 0..g {
                    let sumsq: f32 =
                        (0..h).map(|j| values[row * k + stream * h + j].powi(2)).sum();
                    let dot: f32 = (0..h)
                        .map(|j| {
                            let col = stream * h + j;
                            values[row * k + col]
                                * (bias + norm[col])
                                * weights[out * k + col]
                        })
                        .sum();
                    expected += dot / (sumsq / h as f32 + 1e-6).sqrt();
                }
                ensure!(
                    (actual[row * n + out] - expected).abs() < 0.005,
                    "HC BF16 gate norm mismatch"
                );
            }
        }
    }
    Ok(())
}
