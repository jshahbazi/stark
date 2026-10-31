// SPDX-License-Identifier: Apache-2.0
// Synthetic readout controls. Only hashes/counts/aggregate distances are printed.
use super::*;
use crate::sha256::{hex, sha256};
use crate::weights::QuantWeights;

fn bf(value: f32) -> f32 {
    half::bf16::from_f32(value).to_f32()
}
fn hash(t: &Tensor) -> String {
    hex(&sha256(t.raw_bytes()))
}
fn exact(a: &[f32], b: &[f32], category: &str) -> Result<()> {
    ensure!(a.len() == b.len(), "{category}: length");
    let count = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
    ensure!(count == 0, "{category}: mismatch_count={count}");
    Ok(())
}

fn stock_fixture(ctx: &MetalContext, n: usize) -> Result<(QuantWeights, Tensor)> {
    let (k, seed) = (2560, 0x384064usize);
    let x: Vec<f32> = (0..k)
        .map(|i| bf((((i * 17 + seed) % 71) as i32 - 35) as f32 / 23.0))
        .collect();
    let mut codes = Vec::<u32>::new();
    for r in 0..n {
        for word in 0..k / 8 {
            let mut packed = 0u32;
            for lane in 0..8 {
                let col = word * 8 + lane;
                packed |=
                    (((r * 7 + col * 3 + col / 64 + seed) % 16) as u32) << (4 * lane);
            }
            codes.push(packed);
        }
    }
    let scales: Vec<f32> = (0..n)
        .flat_map(|r| {
            (0..k / 64)
                .map(move |g| bf(((r * 11 + g * 5 + seed) % 23 + 1) as f32 / 512.0))
        })
        .collect();
    let biases: Vec<f32> = (0..n)
        .flat_map(|r| {
            (0..k / 64).map(move |g| {
                bf((((r * 3 + g * 7 + seed) % 19) as i32 - 9) as f32 / 256.0)
            })
        })
        .collect();
    Ok((
        QuantWeights {
            codes: Tensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&codes),
                &[n, k / 8],
                DType::U32,
            )?,
            scales: Tensor::from_f32_as_bf16(ctx, &scales, &[n, k / 64])?,
            biases: Tensor::from_f32_as_bf16(ctx, &biases, &[n, k / 64])?,
            group_size: 64,
            bits: 4,
        },
        Tensor::from_f32_as_bf16(ctx, &x, &[1, k])?,
    ))
}

#[test]
fn official_head_verifier_rows_match_ordinary_bytes() -> Result<()> {
    let ctx = MetalContext::new()?;
    let k = 2560;
    let mut checked = 0;
    // The odd output width exercises the ordinary per-row fallback. The
    // aligned widths use the fused weight-reuse kernel for M2 through M4.
    for n in [128, 256, 129] {
        let (w, _) = stock_fixture(&ctx, n)?;
        for m in 1..=4 {
            let values: Vec<f32> = (0..m * k)
                .map(|i| {
                    let row = i / k;
                    bf((((i * 23 + row * 37 + 17) % 131) as i32 - 65) as f32 / 29.0)
                })
                .collect();
            let mut padded = vec![bf(19.0); k];
            padded.extend_from_slice(&values);
            padded.extend_from_slice(&vec![bf(19.0); k]);
            let x_storage = Tensor::from_f32_as_bf16(&ctx, &padded, &[m + 2, k])?;
            let x = x_storage.view(k, &[m, k])?;
            let storage = Tensor::from_f32(&ctx, &vec![17.0; m * n + 8], &[m * n + 8])?;
            let out = storage.view(4, &[m, n])?;
            let reference = Tensor::zeros(&ctx, &[m, n], DType::F32)?;
            let pass = ctx.begin_concurrent()?;
            gemv_quant_logits_rows_decode_exact(&ctx, &pass, &w, &x, &out, true)?;
            for row in 0..m {
                gemv_quant_logits(
                    &ctx,
                    &pass,
                    &w,
                    &x.view(row * k, &[k])?,
                    &reference.view(row * n, &[n])?,
                    true,
                )?;
            }
            pass.commit()?.wait()?;
            exact(
                &out.to_f32()?,
                &reference.to_f32()?,
                "official verifier head versus ordinary decode",
            )?;
            let guarded = storage.to_f32()?;
            ensure!(
                guarded[..4].iter().chain(&guarded[4 + m * n..]).all(|v| *v == 17.0),
                "official verifier head changed output guards"
            );
            checked += m * n;
        }
    }
    println!(
        "official_head_exact_verifier_rows compared_f32_values={checked} differing_bits=0"
    );
    Ok(())
}

#[test]
fn official_head_fused_precision_and_strict_stock_q4_hash_controls() -> Result<()> {
    // Actual stock CAPI report8da8c196e2cbd772300c290563aa5841fb3840548a3b05c4b278795639bb8616.
    // These are strict hash gates, not a relaxed numerical comparison. A
    // dot-order discrepancy stays a recorded failure, distinct from RNE policy.
    let ctx = MetalContext::new()?;
    let mut stock_matches = 0;
    for (case, n) in [128, 256].into_iter().enumerate() {
        let (w, x) = stock_fixture(&ctx, n)?;
        ensure!(
            hash(&x)
                == "7bf98253ab6cb64197f92f791052e7633c2cc25c9e9f504f4e9d0de9c123f5b7",
            "stock head activation input hash"
        );
        let expected_codes = [
            "99924e6102bc2d9343d206264aa075f3b0b46d2fb6ac1b886166aa19288c0522",
            "34ab9bc681832a7bcfc1b1eb7e9d911d7637b04229d3a16ca8f1fd46fecdc78b",
        ];
        let expected_scales = [
            "c912cffcc4b9d3d32bb13c927dbed8744be3768e9fd8a1683edd16297d43587a",
            "6085d30b0674457452b520c5927442cda52ffb55b07a757375be9eedeeaec539",
        ];
        let expected_biases = [
            "86c8a780384f57067a4832d9a19c1727c080dcaa288079fdc85a08c5461280a6",
            "b2f64d5f07f5be0070faa3394bc89fa309bd7b9106caa7025bb76e64e89562b4",
        ];
        ensure!(
            hash(&w.codes) == expected_codes[case]
                && hash(&w.scales) == expected_scales[case]
                && hash(&w.biases) == expected_biases[case],
            "stock head packed weight input hashes"
        );
        let native = Tensor::zeros(&ctx, &[n], DType::F32)?;
        let matrix_storage = Tensor::from_f32(&ctx, &vec![17.0; n + 8], &[n + 8])?;
        let matrix = matrix_storage.view(4, &[1, n])?;
        let no_dequant = Tensor::zeros(&ctx, &[0], DType::BF16)?;
        let reference = Tensor::zeros(&ctx, &[n], DType::F32)?;
        let previous_fused = Tensor::zeros(&ctx, &[n], DType::F32)?;
        let pipeline = ctx.pipeline_with_math_mode(
            "gemv_q4_bf16_f32out",
            HEAD_SOURCE,
            MslVersion::V3_1,
            crate::metal::MslMathMode::Safe,
        )?;
        let pass = ctx.begin_concurrent()?;
        pass.dispatch_at(
            &pipeline,
            &[
                w.codes.binding(),
                w.scales.binding(),
                w.biases.binding(),
                x.binding(),
                reference.binding(),
            ],
            &[&u32_bytes(2560), &u32_bytes(64)],
            Grid::Threadgroups { groups: (n, 1, 1), threadgroup: (32, 1, 1) },
        )?;
        let previous_pipeline = ctx.pipeline_with_math_mode(
            "gemv_q4_bf16_bf16values_f32out",
            HEAD_SOURCE,
            MslVersion::V3_1,
            crate::metal::MslMathMode::Safe,
        )?;
        pass.dispatch_at(
            &previous_pipeline,
            &[
                w.codes.binding(),
                w.scales.binding(),
                w.biases.binding(),
                x.binding(),
                previous_fused.binding(),
            ],
            &[&u32_bytes(2560), &u32_bytes(64)],
            Grid::Threadgroups { groups: (n, 1, 1), threadgroup: (32, 1, 1) },
        )?;
        gemv_quant_logits(&ctx, &pass, &w, &x, &native, true)?;
        crate::moe_ffn::project_lm_logits(
            &ctx,
            &pass,
            &x,
            &w,
            &matrix,
            &no_dequant,
            true,
        )?;
        pass.commit()?.wait()?;
        exact(
            &matrix.to_f32()?,
            &native.to_f32()?,
            "same-input nonuniform M1 vector/matrix head policy",
        )?;
        let guarded = matrix_storage.to_f32()?;
        ensure!(
            guarded[..4].iter().chain(&guarded[4 + n..]).all(|v| *v == 17.0),
            "M1 head matrix output guards changed"
        );
        let raw = reference.to_f32()?;
        let rounded: Vec<f32> = raw.iter().map(|x| bf(*x)).collect();
        exact(
            &previous_fused.to_f32()?,
            &rounded,
            "preserved previous native-body BF16-value epilogue",
        )?;
        let port_values = native.to_f32()?;
        let previous_values = previous_fused.to_f32()?;
        let source_order_changes = port_values
            .iter()
            .zip(&previous_values)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let max_abs_source_order_diff = port_values
            .iter()
            .zip(&previous_values)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Ordered BF16 bit positions make the diagnostic cross-zero aware.
        let ordered_bf16 = |v: f32| {
            let b = half::bf16::from_f32(v).to_bits();
            if b & 0x8000 != 0 { u32::from(!b) } else { u32::from(b | 0x8000) }
        };
        let max_bf16_ulps_source_order_diff = port_values
            .iter()
            .zip(&previous_values)
            .map(|(a, b)| ordered_bf16(*a).abs_diff(ordered_bf16(*b)))
            .max()
            .unwrap_or(0);
        let changes = raw
            .iter()
            .zip(&rounded)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        ensure!(changes > 0, "head fixture did not witness final output rounding");
        let expected = [
            "fa371b9385aee19e833449e6b66d3acbda04ed9d99ab6e91480b66855826129c",
            "d1ee092dabcb03cd6164b891f8345e5201f516bdf0fc9768e122831100154460",
        ];
        let actual = hash(&native);
        ensure!(hash(&matrix) == actual, "same-input M1 head output hash mismatch");
        let matched = actual == expected[case];
        stock_matches += usize::from(matched);
        println!(
            "head_stock_q4_comparison n={n} bf16_round_changed_count={changes} previous_native_vs_stock_source_port_changed_count={source_order_changes} previous_native_vs_stock_source_port_max_abs={max_abs_source_order_diff} previous_native_vs_stock_source_port_max_bf16_ulps={max_bf16_ulps_source_order_diff} exact_stock_hash_match={matched} same_input_M1_vector_matrix_exact=true actual_f32_buffer_sha256={actual} expected_f32_buffer_sha256={}",
            expected[case]
        );
    }
    ensure!(
        stock_matches == 2,
        "strict stock head hashes: matched_cases={stock_matches}/2; do not replace this gate with tolerance"
    );
    Ok(())
}

#[test]
fn official_head_rounding_has_argmax_tie_and_probability_distance_witness() -> Result<()>
{
    let ctx = MetalContext::new()?;
    let values = [1.0001f32, 1.0002, -1.0, 0.0];
    let storage = Tensor::from_f32(
        &ctx,
        &[17.0, values[0], values[1], values[2], values[3], 17.0],
        &[6],
    )?;
    let out = storage.view(1, &[4])?;
    let gpu_argmax_partials = Tensor::zeros(
        &ctx,
        &[crate::kernels::elementwise::ARGMAX_GROUPS * 2],
        DType::U32,
    )?;
    let gpu_argmax = Tensor::zeros(&ctx, &[1], DType::U32)?;
    let pass = ctx.begin_concurrent()?;
    round_logits_bf16_values(&ctx, &pass, &out)?;
    pass.level_barrier(&[&out])?;
    crate::kernels::elementwise::argmax_f32(
        &ctx,
        &pass,
        &out,
        &gpu_argmax_partials,
        &gpu_argmax,
    )?;
    pass.commit()?.wait()?;
    ensure!(
        gpu_argmax.to_u32()? == vec![0],
        "existing GPU argmax did not select the first BF16-tied value"
    );
    let actual = out.to_f32()?;
    let expected: Vec<f32> = values.iter().map(|v| bf(*v)).collect();
    exact(&actual, &expected, "in-place head rounding")?;
    let all = storage.to_f32()?;
    ensure!(all[0] == 17.0 && all[5] == 17.0, "head rounding touched guards");
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold(
                (0, f32::NEG_INFINITY),
                |best, (i, x)| if *x > best.1 { (i, *x) } else { best },
            )
            .0
    };
    ensure!(
        argmax(&values) == 1
            && argmax(&actual) == 0
            && actual[0].to_bits() == actual[1].to_bits(),
        "BF16 tie/ascending-id argmax witness missing"
    );
    let softmax = |v: &[f32]| {
        let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let e: Vec<f64> = v.iter().map(|x| (*x as f64 - max).exp()).collect();
        let total: f64 = e.iter().sum();
        e.into_iter().map(|x| x / total).collect::<Vec<_>>()
    };
    let before = softmax(&values);
    let after = softmax(&actual);
    let tv: f64 =
        before.iter().zip(&after).map(|(a, b)| (a - b).abs()).sum::<f64>() * 0.5;
    ensure!(tv.is_finite() && tv > 0.0, "probability-distance witness missing");
    println!(
        "head_rounding_witness changed_values=2 argmax_changed=true top_tie_count=2 existing_GPU_argmax_first_tie=true softmax_f64_diagnostic_total_variation={tv}"
    );
    Ok(())
}

fn fallback_fixture(
    ctx: &MetalContext,
    bits: usize,
    gs: usize,
    n: usize,
    k: usize,
) -> Result<(QuantWeights, Vec<f32>)> {
    if bits == 16 {
        let dense: Vec<f32> =
            (0..n * k).map(|i| (i as i32 % 17 - 8) as f32 / 16.0).collect();
        return Ok((
            QuantWeights {
                codes: Tensor::from_f32_as_bf16(ctx, &dense, &[n, k])?,
                scales: Tensor::zeros(ctx, &[0], DType::BF16)?,
                biases: Tensor::zeros(ctx, &[0], DType::BF16)?,
                group_size: 0,
                bits,
            },
            dense,
        ));
    }
    let pack = 32 / bits;
    let mask = (1u32 << bits) - 1;
    let mut codes = vec![0u32; n * k / pack];
    let mut dense = vec![0.0; n * k];
    for r in 0..n {
        for col in 0..k {
            let q = ((r * 7 + col * 3) as u32) & mask;
            codes[r * k / pack + col / pack] |= q << ((col % pack) * bits);
            dense[r * k + col] = q as f32 / 256.0 - 0.125;
        }
    }
    Ok((
        QuantWeights {
            codes: Tensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&codes),
                &[n, k / pack],
                DType::U32,
            )?,
            scales: Tensor::from_f32_as_bf16(
                ctx,
                &vec![1.0 / 256.0; n * k / gs],
                &[n, k / gs],
            )?,
            biases: Tensor::from_f32_as_bf16(
                ctx,
                &vec![-0.125; n * k / gs],
                &[n, k / gs],
            )?,
            group_size: gs,
            bits,
        },
        dense,
    ))
}

#[test]
fn official_head_matrix_quantization_fallbacks_and_row_views_match_dyadic_reference()
-> Result<()> {
    let ctx = MetalContext::new()?;
    let k = 128;
    for (bits, gs, m, n) in [
        (4, 128, 2, 7),
        (8, 32, 2, 7),
        (16, 0, 2, 7),
        (4, 64, 17, 7),
        (4, 64, 2, 128),
        (8, 32, 2, 128),
    ] {
        let (w, dense) = fallback_fixture(&ctx, bits, gs, n, k)?;
        let values: Vec<f32> =
            (0..m * k).map(|i| (i as i32 % 19 - 9) as f32 / 16.0).collect();
        let a = Tensor::from_f32_as_bf16(&ctx, &values, &[m, k])?;
        let storage =
            Tensor::from_f32(&ctx, &vec![17.0; 4 + m * n + 4], &[4 + m * n + 4])?;
        let out = storage.view(4, &[m, n])?;
        let no_dequant = Tensor::zeros(&ctx, &[0], DType::BF16)?;
        let pass = ctx.begin_concurrent()?;
        crate::moe_ffn::project_lm_logits(
            &ctx,
            &pass,
            &a,
            &w,
            &out,
            &no_dequant,
            true,
        )?;
        pass.commit()?.wait()?;
        let mut expected = Vec::new();
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for j in 0..k {
                    sum += values[row * k + j] * dense[col * k + j];
                }
                expected.push(bf(sum));
            }
        }
        exact(&out.to_f32()?, &expected, "official head conditional fallback")?;
        let all = storage.to_f32()?;
        ensure!(
            all[..4].iter().chain(&all[4 + m * n..]).all(|v| *v == 17.0),
            "head fallback row-view guards changed"
        );
    }
    Ok(())
}

#[test]
fn official_head_stock_qmv_fast_admission_bounds_signed_row_offsets() {
    const K: usize = 2560;
    let max_rows = (i32::MAX as usize / (K / 2)) / 8 * 8;
    assert_eq!(max_rows, 1_677_720);
    for n in [8, 128, 256, 248_320, max_rows] {
        assert!(official_head_stock_qmv_fast_supported(n, K));
        let last_tile_row = n - 4;
        assert!(i32::try_from(last_tile_row * (K / 2)).is_ok());
    }
    // The first otherwise-admitted tile beyond the span cap overflows the
    // stock signed byte-row expression; no large allocation is required.
    let first_rejected = max_rows + 8;
    assert!(i32::try_from((first_rejected - 4) * (K / 2)).is_err());
    for n in [0, 7, 9, first_rejected, usize::MAX - 7] {
        assert!(!official_head_stock_qmv_fast_supported(n, K));
    }
    for k in [0, 128, 2_559, 2_561] {
        assert!(!official_head_stock_qmv_fast_supported(8, k));
    }
}
