// SPDX-License-Identifier: Apache-2.0
// Synthetic, aggregate-only controls for official external-MLX GDN arithmetic.
use super::*;

fn bf(value: f32) -> f32 {
    half::bf16::from_f32(value).to_f32()
}

fn sigmoid(value: f32) -> f32 {
    let exponential = bf(value.abs().exp());
    let denominator = bf(1.0 + exponential);
    let y = bf(1.0 / denominator);
    if value < 0.0 { y } else { bf(1.0 - y) }
}

fn exact(actual: &[f32], expected: &[f32], category: &str) -> Result<()> {
    ensure!(actual.len() == expected.len(), "{category}: length mismatch");
    let mismatches =
        actual.iter().zip(expected).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let signed_zeros = actual
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
        mismatches == 0,
        "{category}: mismatches={mismatches}, signed_zero_mismatches={signed_zeros}, max_abs={max_abs}"
    );
    Ok(())
}

#[test]
fn official_gdn_qk_rms_epsilon_and_two_rounds_match_small_dyadic_witnesses()
-> Result<()> {
    let ctx = MetalContext::new()?;
    let (m, hk, h, d) = (3, 1, 3, GDN_HEAD_DIM);
    let width = (2 * hk + h) * d;
    let mut inputs = vec![0.0; m * width];
    for row in 1..m {
        for col in 0..d {
            inputs[row * width + col] = if row == 1 {
                1.0 / 4096.0
            } else {
                if col % 2 == 0 { 1.0 } else { 2.0 }
            };
            inputs[row * width + d + col] = if row == 1 {
                1.0 / 8192.0
            } else {
                if col % 2 == 0 { -1.0 } else { 2.0 }
            };
        }
    }
    let qkv = Tensor::from_f32_as_bf16(&ctx, &inputs, &[m, width])?;
    let official = Tensor::zeros(&ctx, &[m, 2 * hk * d], DType::BF16)?;
    let legacy = Tensor::zeros(&ctx, &[m, 2 * hk * d], DType::BF16)?;
    let scale = 1.0 / (d as f32).sqrt();
    let pass = ctx.begin_concurrent()?;
    gdn_qk_rms_mlx(&ctx, &pass, &qkv, &official, scale, hk, h)?;
    gdn_qk_l2norm(&ctx, &pass, &qkv, &legacy, scale, hk, h)?;
    pass.commit()?.wait()?;
    let mut expected = Vec::new();
    for row in 0..m {
        for part in 0..2 {
            let values = &inputs[row * width + part * d..row * width + (part + 1) * d];
            let mean = values.iter().map(|x| x * x).sum::<f32>() / d as f32;
            let inv = 1.0 / (mean + 1e-6).sqrt();
            let scalar = if part == 0 { bf(1.0 / d as f32) } else { bf(scale) };
            expected.extend(values.iter().map(|x| bf(bf(*x * inv) * scalar)));
        }
    }
    let actual = official.to_f32()?;
    exact(&actual, &expected, "official GDN RMS Q/K")?;
    let old = legacy.to_f32()?;
    ensure!(
        actual[d * 2..d * 4]
            .iter()
            .zip(&old[d * 2..d * 4])
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "GDN epsilon fixture did not distinguish the old norm"
    );
    ensure!(
        actual[..d * 2].iter().all(|x| *x == 0.0 && x.is_finite()),
        "zero Q/K rows are not finite zeros"
    );
    Ok(())
}

#[test]
fn official_gdn_beta_matches_capi_qualified_bf16_sigmoid_and_decay_stays_f32()
-> Result<()> {
    let ctx = MetalContext::new()?;
    // Qualified by stock MLX C API report fd273290f1a9f2a3568b0110d51e926b223728c11ff9ecfb195747a0704be0e5.
    let cases: [u16; 32] = [
        0x0000, 0x8000, 0x3ebf, 0xbebf, 0x3ec0, 0xbec0, 0x3ec1, 0xbec1, 0x3f9f, 0xbf9f,
        0x3fa0, 0xbfa0, 0x3fff, 0xbfff, 0x4000, 0xc000, 0x4001, 0xc001, 0x401f, 0xc01f,
        0x4020, 0xc020, 0x4021, 0xc021, 0x40ff, 0xc0ff, 0x4100, 0xc100, 0x4180, 0xc180,
        0x42a0, 0xc2a0,
    ];
    let values: Vec<f32> =
        cases.iter().map(|v| half::bf16::from_bits(*v).to_f32()).collect();
    let h = cases.len();
    let a = Tensor::from_f32_as_bf16(&ctx, &vec![-10.0; h], &[1, h])?;
    let b = Tensor::from_f32_as_bf16(&ctx, &values, &[1, h])?;
    let a_log = Tensor::zeros(&ctx, &[h], DType::F32)?;
    let dt = Tensor::zeros(&ctx, &[h], DType::BF16)?;
    let decay = Tensor::zeros(&ctx, &[1, h], DType::F32)?;
    let beta = Tensor::zeros(&ctx, &[1, h], DType::F32)?;
    let pass = ctx.begin_concurrent()?;
    gdn_gates_mlx(&ctx, &pass, &a, &b, &a_log, &dt, &decay, &beta)?;
    pass.commit()?.wait()?;
    exact(
        &beta.to_f32()?,
        &values.iter().map(|x| sigmoid(*x)).collect::<Vec<_>>(),
        "official GDN beta",
    )?;
    ensure!(
        decay
            .to_f32()?
            .iter()
            .all(|x| x.is_finite() && *x > 0.0 && *x < 1.0 && bf(*x) == 1.0),
        "official GDN decay lost F32 near-one gate precision"
    );
    Ok(())
}

#[test]
fn official_gdn_conv_prefill_and_steps_match_materialized_bf16_chain_across_tile()
-> Result<()> {
    let ctx = MetalContext::new()?;
    let (m, c, kd) = (65, 32, 4);
    let history: Vec<f32> =
        (0..c * (kd - 1)).map(|i| bf((i as i32 % 17 - 8) as f32 / 16.0)).collect();
    let values: Vec<f32> =
        (0..m * c).map(|i| bf((i as i32 % 29 - 14) as f32 / 16.0)).collect();
    let weights: Vec<f32> =
        (0..kd * c).map(|i| [-0.5, 0.375, 0.625, -0.875][i / c]).collect();
    let window_in = Tensor::from_f32_as_bf16(&ctx, &history, &[c, kd - 1])?;
    let window_out = Tensor::zeros(&ctx, &[c, kd - 1], DType::BF16)?;
    let window_steps = Tensor::from_f32_as_bf16(&ctx, &history, &[c, kd - 1])?;
    let x = Tensor::from_f32_as_bf16(&ctx, &values, &[m, c])?;
    let w = Tensor::from_f32_as_bf16(&ctx, &weights, &[kd, c])?;
    let full = Tensor::zeros(&ctx, &[m, c], DType::BF16)?;
    let steps = Tensor::zeros(&ctx, &[m, c], DType::BF16)?;
    let pass = ctx.begin_concurrent()?;
    conv1d_prefill_with_rounding(
        &ctx,
        &pass,
        &window_in,
        &window_out,
        &x,
        &w,
        &full,
        true,
    )?;
    for row in 0..m {
        conv1d_step_with_rounding(
            &ctx,
            &pass,
            &window_steps,
            &x.view(row * c, &[c])?,
            &w,
            &steps.view(row * c, &[c])?,
            true,
        )?;
        pass.level_barrier(&[&window_steps])?;
    }
    pass.commit()?.wait()?;
    let mut window = history;
    let mut expected = Vec::new();
    for row in 0..m {
        for channel in 0..c {
            let mut sum = 0.0;
            for tap in 0..kd - 1 {
                sum += window[channel * (kd - 1) + tap] * weights[tap * c + channel];
            }
            sum += values[row * c + channel] * weights[(kd - 1) * c + channel];
            let conv = bf(sum);
            expected.push(bf(conv * sigmoid(conv)));
            for old in 0..kd - 2 {
                window[channel * (kd - 1) + old] = window[channel * (kd - 1) + old + 1];
            }
            window[channel * (kd - 1) + kd - 2] = values[row * c + channel];
        }
    }
    exact(&full.to_f32()?, &expected, "official GDN conv prefill")?;
    exact(&steps.to_f32()?, &expected, "official GDN conv steps")?;
    exact(&window_out.to_f32()?, &window, "official GDN prefill window")?;
    exact(&window_steps.to_f32()?, &window, "official GDN step window")
}

#[test]
fn official_gdn_normgate_rounds_normalization_gain_and_sigmoid_separately() -> Result<()>
{
    let ctx = MetalContext::new()?;
    let (m, d) = (2, GDN_HEAD_DIM);
    let y: Vec<f32> = (0..m * d).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }).collect();
    let z: Vec<f32> =
        (0..m * d).map(|i| bf((i as i32 % 37 - 18) as f32 / 8.0)).collect();
    let gains: Vec<f32> = (0..d).map(|i| bf(0.5 + (i % 96) as f32 / 64.0)).collect();
    let input = Tensor::from_f32_as_bf16(&ctx, &y, &[m, d])?;
    let gate = Tensor::from_f32_as_bf16(&ctx, &z, &[m, d])?;
    let weight = Tensor::from_f32(&ctx, &gains, &[d])?;
    let output = Tensor::zeros(&ctx, &[m, d], DType::BF16)?;
    let pass = ctx.begin_concurrent()?;
    gated_rmsnorm_with_rounding(
        &ctx,
        &pass,
        &input,
        &gate,
        &weight,
        &output,
        1e-6,
        GdnGate::Sigmoid,
        true,
    )?;
    pass.commit()?.wait()?;
    let inv = 1.0 / (2.5f32 + 1e-6).sqrt();
    let expected: Vec<f32> = y
        .iter()
        .enumerate()
        .map(|(i, x)| bf(bf(gains[i % d] * bf(*x * inv)) * sigmoid(z[i])))
        .collect();
    exact(&output.to_f32()?, &expected, "official GDN norm-gate")
}

#[test]
fn official_gdn_serial_prefill_matches_steps_and_host_sparse_state_at_17_and_129()
-> Result<()> {
    let ctx = MetalContext::new()?;
    // Keep the production three-value-heads-per-key ratio while giving every
    // BF16 row a four-byte-aligned offset for Tensor::view.
    let (hk, h, d) = (2, 6, GDN_HEAD_DIM);
    let width = (2 * hk + h) * d;
    let scale = 1.0 / (d as f32).sqrt();
    for m in [17, 129] {
        let mut values = vec![0.0; m * width];
        for row in 0..m {
            for key_head in 0..hk {
                values[row * width + key_head * d] = 1.0;
                values[row * width + (hk + key_head) * d] = 1.0;
            }
            for head in 0..h {
                for col in 0..d {
                    values[row * width + (2 * hk + head) * d + col] =
                        bf(((row + head + col) as i32 % 17 - 8) as f32 / 16.0);
                }
            }
        }
        let mut warm = vec![0.0; h * d * d];
        for head in 0..h {
            for col in 0..d {
                warm[head * d * d + col] = ((head + col) as i32 % 7 - 3) as f32 / 16.0;
            }
        }
        let qkv = Tensor::from_f32_as_bf16(&ctx, &values, &[m, width])?;
        let a = Tensor::from_f32_as_bf16(&ctx, &vec![-80.0; m * h], &[m, h])?;
        let b = Tensor::zeros(&ctx, &[m, h], DType::BF16)?;
        let a_log = Tensor::zeros(&ctx, &[h], DType::F32)?;
        let dt = Tensor::zeros(&ctx, &[h], DType::BF16)?;
        let state = Tensor::from_f32(&ctx, &warm, &[h, d, d])?;
        let state_steps = Tensor::from_f32(&ctx, &warm, &[h, d, d])?;
        let qk = Tensor::zeros(&ctx, &[m, 2 * hk * d], DType::BF16)?;
        let decay = Tensor::zeros(&ctx, &[m, h], DType::F32)?;
        let beta = Tensor::zeros(&ctx, &[m, h], DType::F32)?;
        let chunks: Vec<Tensor> = gdn_chunk_staging_shapes(m, h)
            .into_iter()
            .map(|(shape, dtype)| Tensor::zeros(&ctx, &shape, dtype))
            .collect::<Result<_>>()?;
        // Providing valid WY storage proves official mode must actively bypass
        // the ordinary >=128 default, rather than rely on absent staging.
        let staging = GdnRegscanStaging {
            qk_norm: &qk,
            decay: &decay,
            beta: &beta,
            chunk: Some(GdnChunkStaging {
                w: &chunks[0],
                u: &chunks[1],
                p: &chunks[2],
                g: &chunks[3],
            }),
        };
        let raw = Tensor::zeros(&ctx, &[m, h, d], DType::BF16)?;
        let z = Tensor::zeros(&ctx, &[m, h, d], DType::BF16)?;
        let weight = Tensor::from_f32(&ctx, &vec![1.0; d], &[d])?;
        let full = Tensor::zeros(&ctx, &[m, h, d], DType::BF16)?;
        let steps = Tensor::zeros(&ctx, &[m, h, d], DType::BF16)?;
        let pass = ctx.begin_concurrent()?;
        gdn_prefill_mid_with_rounding(
            &ctx, &pass, &qkv, &a, &b, &a_log, &dt, &staging, &state, &raw, scale, hk,
            None, true,
        )?;
        pass.level_barrier(&[&raw, &state])?;
        gated_rmsnorm_with_rounding(
            &ctx,
            &pass,
            &raw,
            &z,
            &weight,
            &full,
            1e-6,
            GdnGate::Sigmoid,
            true,
        )?;
        for row in 0..m {
            gdn_step_gated_fused_with_rounding(
                &ctx,
                &pass,
                &qkv.view(row * width, &[width])?,
                &a.view(row * h, &[h])?,
                &b.view(row * h, &[h])?,
                &a_log,
                &dt,
                &state_steps,
                &z.view(row * h * d, &[h, d])?,
                &weight,
                &steps.view(row * h * d, &[h, d])?,
                scale,
                hk,
                1e-6,
                GdnGate::Sigmoid,
                true,
            )?;
            pass.level_barrier(&[&state_steps])?;
        }
        pass.commit()?.wait()?;
        let q = bf(bf(1.0 / (1.0 / d as f32 + 1e-6).sqrt()) * bf(1.0 / d as f32));
        let k = bf(bf(1.0 / (1.0 / d as f32 + 1e-6).sqrt()) * bf(scale));
        ensure!(k == 1.0, "sparse recurrence witness needs an exact unit key");
        let mut expected_state = warm;
        let mut expected_raw = Vec::new();
        for row in 0..m {
            for head in 0..h {
                for col in 0..d {
                    let at = head * d * d + col;
                    let delta = (values[row * width + (2 * hk + head) * d + col]
                        - expected_state[at])
                        * 0.5;
                    expected_state[at] += delta;
                    expected_raw.push(bf(expected_state[at] * q));
                }
            }
        }
        exact(&raw.to_f32()?, &expected_raw, "official GDN raw sparse readout")?;
        exact(&state.to_f32()?, &expected_state, "official GDN F32 sparse state")?;
        exact(&state_steps.to_f32()?, &expected_state, "official GDN step F32 state")?;
        exact(
            &steps.to_f32()?,
            &full.to_f32()?,
            "official GDN prefill/step gated readout",
        )?;
    }
    Ok(())
}

#[test]
fn official_gdn_decode_decays_dense_state_before_key_prediction() -> Result<()> {
    let ctx = MetalContext::new()?;
    let (hk, h, d) = (1, 3, GDN_HEAD_DIM);
    let width = (2 * hk + h) * d;
    let scale = 1.0 / (d as f32).sqrt();
    let mut values = vec![0.0; width];
    for col in 0..d {
        values[col] = bf((col as i32 % 17 - 8) as f32 / 16.0);
        values[d + col] = bf((col as i32 % 23 - 11) as f32 / 16.0);
    }
    for head in 0..h {
        for col in 0..d {
            values[(2 + head) * d + col] =
                bf(((head + col) as i32 % 13 - 6) as f32 / 16.0);
        }
    }
    let warm: Vec<f32> = (0..h * d * d)
        .map(|i| (i as i32 % 257 - 128) as f32 / 128.0 + (i % 11) as f32 / 4096.0)
        .collect();
    let qkv = Tensor::from_f32_as_bf16(&ctx, &values, &[1, width])?;
    let a = Tensor::zeros(&ctx, &[1, h], DType::BF16)?;
    let b = Tensor::zeros(&ctx, &[1, h], DType::BF16)?;
    let a_log = Tensor::from_f32(&ctx, &vec![0.5; h], &[h])?;
    let dt = Tensor::zeros(&ctx, &[h], DType::BF16)?;
    let qk = Tensor::zeros(&ctx, &[1, 2 * hk * d], DType::BF16)?;
    let decay = Tensor::zeros(&ctx, &[1, h], DType::F32)?;
    let beta = Tensor::zeros(&ctx, &[1, h], DType::F32)?;
    // Obtain the exact new prework's F32 gate/BF16 key values in RAM so
    // this witness isolates recurrence order from exp/rsqrt differences.
    let pass = ctx.begin_concurrent()?;
    gdn_qk_rms_mlx(&ctx, &pass, &qkv, &qk, scale, hk, h)?;
    gdn_gates_mlx(&ctx, &pass, &a, &b, &a_log, &dt, &decay, &beta)?;
    pass.commit()?.wait()?;
    let keys = qk.to_f32()?;
    let decay_values = decay.to_f32()?;
    let betas = beta.to_f32()?;
    ensure!(
        decay_values.iter().all(|g| *g > 0.0 && *g < 1.0 && *g != 0.5),
        "dense witness needs nontrivial F32 decay"
    );
    let mut expected = warm.clone();
    let mut wrong = warm.clone();
    for head in 0..h {
        for col in 0..d {
            let g = decay_values[head];
            let mut prediction = 0.0;
            let mut wrong_prediction = 0.0;
            for key in 0..d {
                let at = (head * d + key) * d + col;
                let decayed = warm[at] * g;
                prediction += decayed * keys[d + key];
                wrong_prediction += (keys[d + key] * warm[at]) * g;
            }
            let value = values[(2 + head) * d + col];
            let delta = (value - prediction) * betas[head];
            let wrong_delta = (value - wrong_prediction) * betas[head];
            for key in 0..d {
                let at = (head * d + key) * d + col;
                expected[at] = warm[at] * g + keys[d + key] * delta;
                wrong[at] = warm[at] * g + keys[d + key] * wrong_delta;
            }
        }
    }
    ensure!(
        expected.iter().zip(&wrong).any(|(a, b)| a.to_bits() != b.to_bits()),
        "dense recurrence fixture did not distinguish decay-before-dot"
    );
    let state = Tensor::from_f32(&ctx, &warm, &[h, d, d])?;
    let z = Tensor::zeros(&ctx, &[h, d], DType::BF16)?;
    let weight = Tensor::from_f32(&ctx, &vec![1.0; d], &[d])?;
    let output = Tensor::zeros(&ctx, &[h, d], DType::BF16)?;
    let pass = ctx.begin_concurrent()?;
    gdn_step_gated_fused_with_rounding(
        &ctx,
        &pass,
        &qkv,
        &a,
        &b,
        &a_log,
        &dt,
        &state,
        &z,
        &weight,
        &output,
        scale,
        hk,
        1e-6,
        GdnGate::Sigmoid,
        true,
    )?;
    pass.commit()?.wait()?;
    exact(&state.to_f32()?, &expected, "official GDN dense state decay order")?;
    ensure!(
        output.to_f32()?.iter().all(|x| x.is_finite()),
        "dense GDN readout is not finite"
    );
    Ok(())
}
