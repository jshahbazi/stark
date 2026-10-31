// SPDX-License-Identifier: Apache-2.0
// Synthetic operation-chain controls for the admitted external MLX contract.
// Root runs these on Metal. Never print tensor contents on failure.
use super::*;
use crate::kernels::ple;

fn bf(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

// Source-specific contracts: primitive MLX unary_ops.h::Sigmoid and the
// served hc_decode2 custom narrow F32 expression deliberately differ.
fn sig(x: f32) -> f32 {
    let exponential = bf(x.abs().exp());
    let denominator = bf(1.0 + exponential);
    let y = bf(1.0 / denominator);
    if x < 0.0 { y } else { bf(1.0 - y) }
}

fn narrow_sig(x: f32) -> f32 {
    bf(1.0 / (1.0 + (-x).exp()))
}

fn exact(actual: &[f32], expected: &[f32], category: &str) -> Result<()> {
    let mismatch_count =
        actual.iter().zip(expected).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let signed_zero_count = actual
        .iter()
        .zip(expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits() && **a == 0.0 && **b == 0.0)
        .count();
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0f32, f32::max);
    // Only aggregate diagnostics are allowed; never include tensor contents,
    // element indices or individual reference/output pairs in the error.
    let ordered_bf16 = |x: f32| -> u16 {
        let bits = half::bf16::from_f32(x).to_bits();
        if bits & 0x8000 != 0 { !bits } else { bits | 0x8000 }
    };
    let max_bf16_ulp = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| ordered_bf16(*a).abs_diff(ordered_bf16(*b)))
        .max()
        .unwrap_or(0);
    ensure!(
        actual.len() == expected.len() && mismatch_count == 0,
        "{category}: actual_len={}, expected_len={}, mismatches={mismatch_count}, signed_zero_mismatches={signed_zero_count}, max_abs={max_abs}, max_bf16_ulp={max_bf16_ulp}",
        actual.len(),
        expected.len(),
    );
    Ok(())
}

#[test]
fn official_grouped_norm_rounds_before_gain_and_preserves_legacy_entry() -> Result<()> {
    let ctx = MetalContext::new()?;
    let (h, g, m) = (128, 4, 3);
    // Dyadic squares sum exactly in F32 in any reduction order. Nontrivial
    // gains expose the missing unweighted-normalization rounding boundary.
    let values: Vec<f32> =
        (0..m * g * h).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }).collect();
    let gains: Vec<f32> =
        (0..g * h).map(|i| bf(0.5 + (i % 192) as f32 / 128.0)).collect();
    let x = Tensor::from_f32_as_bf16(&ctx, &values, &[m, g * h])?;
    let w = Tensor::from_f32_as_bf16(&ctx, &gains, &[g * h])?;
    let output = Tensor::zeros(&ctx, &[m, g * h], DType::BF16)?;
    let legacy = Tensor::zeros(&ctx, &[m, g * h], DType::BF16)?;
    let explicit_legacy = Tensor::zeros(&ctx, &[m, g * h], DType::BF16)?;
    let pass = ctx.begin_concurrent()?;
    rmsnorm_grouped_bf16_with_rounding(
        &ctx, &pass, &x, &w, &output, h, g, 0.0, 0.0, true,
    )?;
    rmsnorm_grouped_bf16(&ctx, &pass, &x, &w, &legacy, h, g, 0.0, 0.0)?;
    rmsnorm_grouped_bf16_with_rounding(
        &ctx,
        &pass,
        &x,
        &w,
        &explicit_legacy,
        h,
        g,
        0.0,
        0.0,
        false,
    )?;
    ensure!(
        rmsnorm_grouped_bf16_with_rounding(
            &ctx, &pass, &x, &w, &output, h, g, 0.0, 1.0, true
        )
        .is_err(),
        "official norm accepted delta convention"
    );
    pass.commit()?.wait()?;
    let inv = 1.0 / 2.5f32.sqrt();
    let reference: Vec<f32> = values
        .iter()
        .enumerate()
        .map(|(i, x)| bf(bf(*x * inv) * gains[i % (g * h)]))
        .collect();
    let original_formula: Vec<f32> = values
        .iter()
        .enumerate()
        .map(|(i, x)| bf(*x * inv * gains[i % (g * h)]))
        .collect();
    ensure!(
        reference
            .iter()
            .zip(&original_formula)
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "grouped norm fixture does not distinguish rounding policies"
    );
    exact(&output.to_f32()?, &reference, "official grouped norm rounding mismatch")?;
    exact(
        &legacy.to_f32()?,
        &explicit_legacy.to_f32()?,
        "legacy grouped norm entry changed",
    )
}

#[test]
fn official_hc_tails_match_bf16_reference_chains() -> Result<()> {
    let ctx = MetalContext::new()?;
    let g = 4;
    for (m, h, primitive, sequential) in [
        (2, 32, false, false),
        (17, 32, true, true),
        (1, 32, true, false),
        (17, 32, true, false),
        (17, 2560, true, false),
    ] {
        let sigmoid = if primitive { sig } else { narrow_sig };
        let up: Vec<f32> =
            (0..m * g * h).map(|i| [-1.25, 0.375, 2.5, -2.0][(i / h) % g]).collect();
        let normalized: Vec<f32> =
            (0..m * g * h).map(|i| bf((i as i32 % 37 - 18) as f32 / 16.0)).collect();
        let raw: Vec<f32> =
            (0..m * g).map(|i| [-6.0, -1.5, 2.5, 10.0][i % g]).collect();
        let branch: Vec<f32> =
            (0..m * h).map(|i| bf((i as i32 % 29 - 14) as f32 / 16.0)).collect();
        let initial: Vec<f32> =
            (0..m * g * h).map(|i| bf((i as i32 % 31 - 15) as f32 / 32.0)).collect();
        let u = Tensor::from_f32_as_bf16(&ctx, &up, &[m, g * h])?;
        let n = Tensor::from_f32_as_bf16(&ctx, &normalized, &[m, g * h])?;
        let mixed = Tensor::zeros(&ctx, &[m, h], DType::BF16)?;
        let inj = Tensor::from_f32_as_bf16(&ctx, &raw, &[m, g])?;
        let b = Tensor::from_f32_as_bf16(&ctx, &branch, &[m, h])?;
        let hyper = Tensor::from_f32_as_bf16(&ctx, &initial, &[m, g * h])?;
        let act = Tensor::zeros(&ctx, &[m, g], DType::BF16)?;
        let pass = ctx.begin_concurrent()?;
        hc_mix_bf16_with_rounding(
            &ctx, &pass, &u, &n, &mixed, h, g, true, primitive, sequential,
        )?;
        hc_inject_bf16_with_rounding(
            &ctx, &pass, &hyper, &b, &inj, h, g, true, primitive,
        )?;
        silu_scaled_bf16_with_rounding(&ctx, &pass, &inj, &act, 0.25, true, primitive)?;
        pass.commit()?.wait()?;
        let mut expected_mix = Vec::new();
        let mut old_f32_mean = Vec::new();
        for row in 0..m {
            for col in 0..h {
                let mut sum = 0.0;
                let mut f32_sum = 0.0;
                for stream in 0..g {
                    let at = (row * g + stream) * h + col;
                    let product = bf(sigmoid(up[at]) * normalized[at]);
                    f32_sum += product;
                    sum = if primitive { bf(sum + product) } else { sum + product };
                }
                expected_mix.push(bf(bf(sum) * bf(1.0 / g as f32)));
                old_f32_mean.push(bf(bf(f32_sum) * bf(1.0 / g as f32)));
            }
        }
        if primitive && !sequential {
            ensure!(
                expected_mix
                    .iter()
                    .zip(&old_f32_mean)
                    .any(|(a, b)| a.to_bits() != b.to_bits()),
                "primitive mean fixture does not distinguish BF16 and F32 accumulators"
            );
        }
        let expected_act: Vec<f32> = raw
            .iter()
            .map(|x| {
                let v = bf(*x * 0.25);
                bf(v * sigmoid(v))
            })
            .collect();
        let expected_hyper: Vec<f32> = initial
            .iter()
            .enumerate()
            .map(|(at, x)| {
                let row = at / (g * h);
                let stream = at % (g * h) / h;
                let gate = bf(2.0 * sigmoid(bf(raw[row * g + stream] * 0.25)));
                bf(*x + bf(branch[row * h + at % h] * gate))
            })
            .collect();
        exact(
            &mixed.to_f32()?,
            &expected_mix,
            if sequential {
                "official HC prefill sequential mix rounding mismatch (M=17,H=32,G=4)"
            } else if primitive {
                "official HC primitive fallback BF16 mean mismatch"
            } else {
                "official HC custom decode mix rounding mismatch (M=2,H=32,G=4)"
            },
        )?;
        exact(
            &act.to_f32()?,
            &expected_act,
            "official HC activation rounding mismatch",
        )?;
        exact(
            &hyper.to_f32()?,
            &expected_hyper,
            "official HC write rounding mismatch",
        )?;
    }
    Ok(())
}

#[test]
fn official_ple_gate_matches_materialized_bf16_chain() -> Result<()> {
    let ctx = MetalContext::new()?;
    let (h, g, m) = (128, 4, 2);
    let key: Vec<f32> = (0..m * g * h)
        .map(|i| if i / h == 0 { 0.0 } else { bf(0.75 + (i % 31) as f32 / 128.0) })
        .collect();
    let query: Vec<f32> =
        (0..m * g * h).map(|i| bf(-0.625 + (i % 47) as f32 / 128.0)).collect();
    let value: Vec<f32> =
        (0..m * h).map(|i| bf((i as i32 % 23 - 11) as f32 / 16.0)).collect();
    let k = Tensor::from_f32_as_bf16(&ctx, &key, &[m, g * h])?;
    let q = Tensor::from_f32_as_bf16(&ctx, &query, &[m, g * h])?;
    let v = Tensor::from_f32_as_bf16(&ctx, &value, &[m, h])?;
    let y = Tensor::zeros(&ctx, &[m, g * h], DType::BF16)?;
    let pass = ctx.begin_concurrent()?;
    ple::ple_gate_value_bf16_with_rounding(&ctx, &pass, &k, &q, &v, &y, h, g, true)?;
    pass.commit()?.wait()?;
    let mut reference = vec![0.0; m * g * h];
    for seg in 0..m * g {
        let total: f32 =
            (0..h).map(|i| bf(key[seg * h + i] * query[seg * h + i])).sum();
        let scaled = bf(bf(total) * bf(1.0 / (h as f32).sqrt()));
        let magnitude = bf(scaled.abs().max(bf(1e-6)).sqrt());
        let signed = bf(if scaled > 0.0 {
            magnitude
        } else if scaled < 0.0 {
            -magnitude
        } else {
            0.0
        });
        let gate = sig(signed);
        for i in 0..h {
            reference[seg * h + i] = bf(gate * value[seg / g * h + i]);
        }
    }
    exact(&y.to_f32()?, &reference, "official PLE gate chain mismatch")
}

#[test]
fn official_ple_conv_prefill_and_steps_match_host_rounding_and_state() -> Result<()> {
    let ctx = MetalContext::new()?;
    let (m, c, kd, dilation) = (7, 32, 4, 3);
    let state = (kd - 1) * dilation;
    let initial_window: Vec<f32> =
        (0..c * state).map(|i| bf((i as i32 % 17 - 8) as f32 / 16.0)).collect();
    let weights: Vec<f32> =
        (0..kd * c).map(|i| [-0.5, 0.375, 0.625, -0.875][i / c]).collect();
    let values: Vec<f32> =
        (0..m * c).map(|i| bf((i as i32 % 29 - 14) as f32 / 16.0)).collect();
    let base: Vec<f32> =
        (0..m * c).map(|i| bf((i as i32 % 23 - 11) as f32 / 32.0)).collect();
    let initial: Vec<f32> =
        (0..m * c).map(|i| bf((i as i32 % 19 - 9) as f32 / 32.0)).collect();
    let wi = Tensor::from_f32_as_bf16(&ctx, &initial_window, &[c, state])?;
    let wo = Tensor::zeros(&ctx, &[c, state], DType::BF16)?;
    let ws = Tensor::from_f32_as_bf16(&ctx, &initial_window, &[c, state])?;
    let x = Tensor::from_f32_as_bf16(&ctx, &values, &[m, c])?;
    let w = Tensor::from_f32_as_bf16(&ctx, &weights, &[kd, c])?;
    let b = Tensor::from_f32_as_bf16(&ctx, &base, &[m, c])?;
    let full = Tensor::from_f32_as_bf16(&ctx, &initial, &[m, c])?;
    let steps = Tensor::from_f32_as_bf16(&ctx, &initial, &[m, c])?;
    let pass = ctx.begin_concurrent()?;
    ple::ple_conv1d_prefill_with_rounding(
        &ctx, &pass, &wi, &wo, &x, &w, &b, &full, dilation, true,
    )?;
    for row in 0..m {
        ple::ple_conv1d_step_with_rounding(
            &ctx,
            &pass,
            &ws,
            &x.view(row * c, &[c])?,
            &w,
            &b.view(row * c, &[c])?,
            &steps.view(row * c, &[c])?,
            dilation,
            true,
        )?;
        pass.level_barrier(&[&ws])?;
    }
    pass.commit()?.wait()?;
    let mut reference = initial.clone();
    let mut window = initial_window;
    let mut rounding_witness = false;
    for row in 0..m {
        for channel in 0..c {
            let at = row * c + channel;
            let mut sum = 0.0;
            for tap in 0..kd - 1 {
                sum += window[channel * state + tap * dilation]
                    * weights[tap * c + channel];
            }
            sum += values[at] * weights[(kd - 1) * c + channel];
            let conv = bf(sum);
            let act = bf(conv * sig(conv));
            reference[at] = bf(initial[at] + bf(base[at] + act));
            let legacy = bf(initial[at] + base[at] + sum / (1.0 + (-sum).exp()));
            rounding_witness |= reference[at].to_bits() != legacy.to_bits();
            for old in 0..state - 1 {
                window[channel * state + old] = window[channel * state + old + 1];
            }
            window[channel * state + state - 1] = values[at];
        }
    }
    ensure!(
        rounding_witness,
        "PLE conv fixture does not distinguish rounding policies"
    );
    exact(&full.to_f32()?, &reference, "official PLE prefill rounding mismatch")?;
    exact(&steps.to_f32()?, &reference, "official PLE step rounding mismatch")?;
    exact(&wo.to_f32()?, &window, "official PLE prefill state mismatch")?;
    exact(&ws.to_f32()?, &window, "official PLE step state mismatch")
}

#[test]
fn official_sigmoid_modes_match_source_chains_on_adversarial_bf16_inputs() -> Result<()>
{
    let ctx = MetalContext::new()?;
    // Neighbours and both signs expose exp/denominator rounding and the
    // stable primitive's positive-side subtraction. No fixture values log.
    let cases: [u16; 32] = [
        0x0000, 0x8000, 0x3ebf, 0xbebf, 0x3ec0, 0xbec0, 0x3ec1, 0xbec1, 0x3f9f, 0xbf9f,
        0x3fa0, 0xbfa0, 0x3fff, 0xbfff, 0x4000, 0xc000, 0x4001, 0xc001, 0x401f, 0xc01f,
        0x4020, 0xc020, 0x4021, 0xc021, 0x40ff, 0xc0ff, 0x4100, 0xc100, 0x4180, 0xc180,
        0x42a0, 0xc2a0,
    ];
    let h = cases.len();
    let inputs: Vec<f32> =
        cases.iter().map(|v| half::bf16::from_bits(*v).to_f32()).collect();
    let primitive: Vec<f32> = inputs.iter().map(|v| sig(*v)).collect();
    let narrow: Vec<f32> = inputs.iter().map(|v| narrow_sig(*v)).collect();
    ensure!(
        primitive.iter().zip(&narrow).any(|(a, b)| a.to_bits() != b.to_bits()),
        "sigmoid mode fixture does not distinguish serving branches"
    );
    let x = Tensor::from_f32_as_bf16(&ctx, &inputs, &[1, h])?;
    let ones = Tensor::from_f32_as_bf16(&ctx, &vec![1.0; h], &[1, h])?;
    let yp = Tensor::zeros(&ctx, &[1, h], DType::BF16)?;
    let yn = Tensor::zeros(&ctx, &[1, h], DType::BF16)?;
    let pass = ctx.begin_concurrent()?;
    hc_mix_bf16_with_rounding(&ctx, &pass, &x, &ones, &yp, h, 1, true, true, false)?;
    hc_mix_bf16_with_rounding(&ctx, &pass, &x, &ones, &yn, h, 1, true, false, false)?;
    pass.commit()?.wait()?;
    exact(
        &yp.to_f32()?,
        &primitive,
        "official primitive sigmoid adversarial mismatch",
    )?;
    exact(&yn.to_f32()?, &narrow, "official custom sigmoid adversarial mismatch")?;

    // Small-column Sum<BF16> initializes positive zero before combining its
    // streams; hc_prefill_mix starts from its first product. All-negative-zero
    // products distinguish their exact source contracts without magnitude noise.
    let zero_up = Tensor::zeros(&ctx, &[1, 4 * h], DType::BF16)?;
    let negative_zero =
        Tensor::from_f32_as_bf16(&ctx, &vec![-0.0; 4 * h], &[1, 4 * h])?;
    let primitive_zero = Tensor::zeros(&ctx, &[1, h], DType::BF16)?;
    let prefill_zero = Tensor::zeros(&ctx, &[1, h], DType::BF16)?;
    let pass = ctx.begin_concurrent()?;
    hc_mix_bf16_with_rounding(
        &ctx,
        &pass,
        &zero_up,
        &negative_zero,
        &primitive_zero,
        h,
        4,
        true,
        true,
        false,
    )?;
    hc_mix_bf16_with_rounding(
        &ctx,
        &pass,
        &zero_up,
        &negative_zero,
        &prefill_zero,
        h,
        4,
        true,
        true,
        true,
    )?;
    pass.commit()?.wait()?;
    exact(
        &primitive_zero.to_f32()?,
        &vec![0.0; h],
        "official primitive mean positive-zero initialization mismatch",
    )?;
    exact(
        &prefill_zero.to_f32()?,
        &vec![-0.0; h],
        "official prefill mean first-product sign mismatch",
    )
}

#[test]
fn official_hc_served_sigmoid_admission_checks_complete_descriptors() -> Result<()> {
    let ctx = MetalContext::new()?;
    let q8 = |n: usize, k: usize| -> Result<QuantWeights> {
        Ok(QuantWeights {
            codes: Tensor::zeros(&ctx, &[n, k / 4], DType::U32)?,
            scales: Tensor::zeros(&ctx, &[n, k / 64], DType::BF16)?,
            biases: Tensor::zeros(&ctx, &[n, k / 64], DType::BF16)?,
            group_size: 64,
            bits: 8,
        })
    };
    let norm = Tensor::zeros(&ctx, &[10240], DType::BF16)?;
    let down = q8(320, 10240)?;
    let mut up = q8(10240, 320)?;
    let inject = QuantWeights::bf16(Tensor::zeros(&ctx, &[4, 10240], DType::BF16)?)?;
    for rows in [1, 16] {
        ensure!(
            official_narrow_sigmoid_supported(
                &norm,
                &down,
                &up,
                Some(&inject),
                2560,
                4,
                rows
            ),
            "admitted official HC descriptor rejected"
        );
        // The final mixer's ordinary read supports INJ=0 in hc_decode2.
        ensure!(
            official_narrow_sigmoid_supported(&norm, &down, &up, None, 2560, 4, rows),
            "admitted no-inject final HC rejected"
        );
    }
    for rows in [0, 17] {
        ensure!(
            !official_narrow_sigmoid_supported(&norm, &down, &up, None, 2560, 4, rows),
            "HC row fallback missed"
        );
    }
    ensure!(
        !official_narrow_sigmoid_supported(&norm, &down, &up, None, 128, 4, 1),
        "HC geometry fallback missed"
    );
    ensure!(
        !official_narrow_sigmoid_supported(&norm, &down, &up, Some(&down), 2560, 4, 1),
        "HC inject dtype fallback missed"
    );
    up.group_size = 32;
    ensure!(
        !official_narrow_sigmoid_supported(&norm, &down, &up, None, 2560, 4, 1),
        "HC group fallback missed"
    );
    up.group_size = 64;
    up.bits = 4;
    ensure!(
        !official_narrow_sigmoid_supported(&norm, &down, &up, None, 2560, 4, 1),
        "HC bit-width fallback missed"
    );
    Ok(())
}
