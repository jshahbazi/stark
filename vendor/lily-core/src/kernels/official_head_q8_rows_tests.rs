// SPDX-License-Identifier: Apache-2.0
//! Native generic-Q8 head controls. Synthetic tensors remain in memory;
//! diagnostics contain aggregate counts and GPU timing summaries only.
use super::*;

fn next_word(seed: &mut u32) -> u32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed
}

fn weights(ctx: &MetalContext, n: usize, k: usize) -> Result<QuantWeights> {
    let mut seed = 0x4838_8b64;
    let storage_rows = n + 2;
    let codes: Vec<u32> =
        (0..storage_rows * k / 4).map(|_| next_word(&mut seed)).collect();
    let scales: Vec<f32> = (0..storage_rows * k / 64)
        .map(|_| (next_word(&mut seed) % 127 + 1) as f32 / 65_536.0)
        .collect();
    let biases: Vec<f32> = (0..storage_rows * k / 64)
        .map(|_| (next_word(&mut seed) % 257) as f32 / 4096.0 - 0.03125)
        .collect();
    QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[storage_rows, k / 4],
            DType::U32,
        )?,
        scales: Tensor::from_f32_as_bf16(ctx, &scales, &[storage_rows, k / 64])?,
        biases: Tensor::from_f32_as_bf16(ctx, &biases, &[storage_rows, k / 64])?,
        group_size: 64,
        bits: 8,
    }
    .view_rows(2, n)
}

fn input(ctx: &MetalContext, rows: usize, k: usize) -> Result<Tensor> {
    let mut seed = 0x3812_3497;
    let values: Vec<f32> = (0..(rows + 2) * k)
        .map(|i| {
            if i % 31 == 0 {
                if i % 2 == 0 { 0.0 } else { -0.0 }
            } else {
                (next_word(&mut seed) % 4097) as f32 / 1024.0 - 2.0
            }
        })
        .collect();
    Tensor::from_f32_as_bf16(ctx, &values, &[rows + 2, k])?.view(k, &[rows, k])
}

fn guarded_output(
    ctx: &MetalContext,
    rows: usize,
    n: usize,
) -> Result<(Tensor, Tensor)> {
    let storage = Tensor::zeros(ctx, &[rows * n + 16], DType::F32)?;
    storage.write_bytes(&vec![0xa5; storage.byte_len()])?;
    let output = storage.view(8, &[rows, n])?;
    Ok((storage, output))
}

fn check_guards(storage: &Tensor) -> Result<()> {
    let bytes = storage.raw_bytes();
    ensure!(
        bytes[..32].iter().chain(&bytes[bytes.len() - 32..]).all(|b| *b == 0xa5),
        "Q8 verifier head changed output guards"
    );
    Ok(())
}

fn exact(candidate: &Tensor, reference: &Tensor) -> Result<()> {
    ensure!(
        candidate.raw_bytes() == reference.raw_bytes(),
        "Q8 verifier head differs from ordinary generic Q8 target"
    );
    Ok(())
}

fn run_gpu_seconds(
    ctx: &MetalContext,
    w: &QuantWeights,
    x: &Tensor,
    output: &Tensor,
    candidate: Option<bool>,
) -> Result<f64> {
    let (rows, k, n) = (x.shape()[0], w.in_features(), w.out_features());
    let pass = ctx.begin_concurrent()?;
    if let Some(reuse) = candidate {
        gemv_quant_logits_q8_rows_candidate(ctx, &pass, w, x, output, reuse)?;
    } else {
        for row in 0..rows {
            gemv_quant_logits(
                ctx,
                &pass,
                w,
                &x.view(row * k, &[k])?,
                &output.view(row * n, &[n])?,
                true,
            )?;
            pass.level_barrier(&[output])?;
        }
    }
    let completed = pass.commit()?.wait_retain()?;
    let timing = completed.timing()?;
    let seconds = timing.gpu_end_secs - timing.gpu_start_secs;
    ensure!(
        seconds.is_finite() && seconds > 0.0,
        "Q8 verifier head GPU timing unavailable"
    );
    Ok(seconds)
}

#[test]
fn official_q8_head_verifier_rows_match_ordinary_bytes() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut compared_values = 0;
    for (n, k) in [(17, 128), (256, 2560)] {
        let w = weights(&ctx, n, k)?;
        for rows in 1..=4 {
            let x = input(&ctx, rows, k)?;
            let reference = Tensor::zeros(&ctx, &[rows, n], DType::F32)?;
            run_gpu_seconds(&ctx, &w, &x, &reference, None)?;
            for reuse in [false, true] {
                let (storage, candidate) = guarded_output(&ctx, rows, n)?;
                run_gpu_seconds(&ctx, &w, &x, &candidate, Some(reuse))?;
                exact(&candidate, &reference)?;
                check_guards(&storage)?;
                compared_values += rows * n;
            }
        }
    }
    println!(
        "official_q8_head_exact_verifier compared_f32_values={compared_values} differing_bits=0"
    );
    Ok(())
}

/// Full actual head geometry, synthetic unchanged Q8/group64 storage. Each
/// candidate passes complete readout and guard controls before/after timing.
#[test]
#[ignore = "exclusive physical Apple Silicon GPU; full-vocabulary Q8 component timing"]
fn official_q8_head_verifier_full_vocabulary_timing() -> Result<()> {
    let ctx = MetalContext::new()?;
    let (n, k) = (248_320, 2560);
    let w = weights(&ctx, n, k)?;
    let weight_bytes = w.codes.byte_len() + w.scales.byte_len() + w.biases.byte_len();
    for rows in 1..=4 {
        let x = input(&ctx, rows, k)?;
        let reference = Tensor::zeros(&ctx, &[rows, n], DType::F32)?;
        let (storage, candidate) = guarded_output(&ctx, rows, n)?;
        run_gpu_seconds(&ctx, &w, &x, &reference, None)?;
        for reuse in [false, true] {
            run_gpu_seconds(&ctx, &w, &x, &candidate, Some(reuse))?;
            exact(&candidate, &reference)?;
            check_guards(&storage)?;
        }
        let warmup = std::time::Instant::now();
        let mut warmup_calls = 0;
        while warmup.elapsed() < std::time::Duration::from_secs(2) {
            for mode in [None, Some(false), Some(true)] {
                run_gpu_seconds(&ctx, &w, &x, &candidate, mode)?;
                warmup_calls += 1;
            }
        }
        for reuse in [false, true] {
            let mut reference_samples = Vec::new();
            let mut candidate_samples = Vec::new();
            const WARMUP_BLOCKS: usize = 2;
            const MEASURED_BLOCKS: usize = 8;
            for block in 0..WARMUP_BLOCKS + MEASURED_BLOCKS {
                let order = if block % 2 == 0 {
                    [false, true, true, false]
                } else {
                    [true, false, false, true]
                };
                for is_candidate in order {
                    let seconds = run_gpu_seconds(
                        &ctx,
                        &w,
                        &x,
                        if is_candidate { &candidate } else { &reference },
                        is_candidate.then_some(reuse),
                    )?;
                    if block >= WARMUP_BLOCKS {
                        if is_candidate {
                            candidate_samples.push(seconds * 1e6);
                        } else {
                            reference_samples.push(seconds * 1e6);
                        }
                    }
                }
            }
            exact(&candidate, &reference)?;
            check_guards(&storage)?;
            let mean =
                |samples: &[f64]| samples.iter().sum::<f64>() / samples.len() as f64;
            let reference_mean = mean(&reference_samples);
            let candidate_mean = mean(&candidate_samples);
            let median = |samples: &[f64]| {
                let mut sorted = samples.to_vec();
                sorted.sort_by(f64::total_cmp);
                (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
            };
            println!(
                "{}",
                serde_json::json!({
                    "scope":"synthetic_official_generic_q8_head_full_vocabulary",
                    "candidate":if reuse { "weight_reuse_fused_round" } else { "row_axis_then_round" },
                    "timer":"metal_command_buffer_gpu_end_minus_start_seconds",
                    "math_policy":"generic_q8_platform_default_integer_bf16_value_round",
                    "rows":rows,"input_width":k,"vocabulary_size":n,"weight_bytes":weight_bytes,
                    "initial_warmup_seconds_at_least":2,"initial_warmup_calls":warmup_calls,
                    "excluded_abba_blocks":WARMUP_BLOCKS,"measured_abba_blocks":MEASURED_BLOCKS,
                    "measured_calls_per_route":reference_samples.len(),
                    "reference_mean_gpu_us":reference_mean,"candidate_mean_gpu_us":candidate_mean,
                    "reference_median_gpu_us":median(&reference_samples),
                    "candidate_median_gpu_us":median(&candidate_samples),
                    "candidate_gpu_speedup":reference_mean/candidate_mean,
                    "compared_f32_values_per_control":rows*n,
                    "compared_bytes_per_control":rows*n*4,"differing_bits":0,
                    "nonzero_input_weight_output_offsets":true,"output_guards_unchanged":true,
                })
            );
        }
    }
    Ok(())
}
