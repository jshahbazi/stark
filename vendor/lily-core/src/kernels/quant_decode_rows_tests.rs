// SPDX-License-Identifier: Apache-2.0
//! Bit controls against ordinary GEMV, including production dense widths.

use super::*;

fn word(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn weights(
    ctx: &MetalContext,
    state: &mut u32,
    bits: usize,
    n: usize,
    k: usize,
) -> Result<QuantWeights> {
    let packed = 32 / bits;
    let rows = n + 2;
    let codes: Vec<u32> = (0..rows * k / packed).map(|_| word(state)).collect();
    let scales: Vec<f32> =
        (0..rows * k / 64).map(|_| (word(state) % 127 + 1) as f32 / 65_536.0).collect();
    let biases: Vec<f32> = (0..rows * k / 64)
        .map(|_| (word(state) % 257) as f32 / 4096.0 - 0.03125)
        .collect();
    QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[rows, k / packed],
            DType::U32,
        )?,
        scales: Tensor::from_f32_as_bf16(ctx, &scales, &[rows, k / 64])?,
        biases: Tensor::from_f32_as_bf16(ctx, &biases, &[rows, k / 64])?,
        group_size: 64,
        bits,
    }
    .view_rows(2, n)
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn decode_rows_gemv_matches_ordinary_bytes() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut seed = 0x7123_8b93;
    for (bits, k, n, dtype) in [
        (4, 2560, 1280, DType::BF16),
        (4, 640, 2560, DType::BF16),
        (4, 2560, 8192, DType::F32),
        (8, 2560, 512, DType::F32),
        (8, 2560, 1, DType::F32),
        (8, 2560, 1280, DType::BF16),
        (4, 128, 17, DType::BF16),
        (8, 2560, 248320, DType::F32),
    ] {
        let w = weights(&ctx, &mut seed, bits, n, k)?;
        for m in 1..=4 {
            let values: Vec<f32> = (0..m * k)
                .map(|i| {
                    if i % 31 == 0 {
                        if i % 2 == 0 { 0.0 } else { -0.0 }
                    } else {
                        (word(&mut seed) % 4097) as f32 / 1024.0 - 2.0
                    }
                })
                .collect();
            let input = Tensor::from_f32_as_bf16(&ctx, &values, &[m, k])?;
            // Separate row allocations also cover odd BF16 widths: a
            // contiguous second-row view would have a non-four-byte offset.
            let reference: Vec<_> = (0..m)
                .map(|_| Tensor::zeros(&ctx, &[n], dtype))
                .collect::<Result<_>>()?;
            let guarded = Tensor::zeros(&ctx, &[m * n + 16], dtype)?;
            guarded.write_bytes(&vec![0xa5; guarded.byte_len()])?;
            let candidate = guarded.view(8, &[m, n])?;
            let pass = ctx.begin_concurrent()?;
            for row in 0..m {
                gemv_quant(
                    &ctx,
                    &pass,
                    &w,
                    &input.view(row * k, &[k])?,
                    &reference[row],
                )?;
            }
            gemv_quant_rows_decode_exact(&ctx, &pass, &w, &input, &candidate)?;
            pass.end()?.commit()?.wait()?;
            let candidate_bytes = candidate.raw_bytes();
            let row_bytes = n * dtype.size();
            ensure!(
                reference.iter().enumerate().all(|(row, expected)| {
                    expected.raw_bytes()
                        == &candidate_bytes[row * row_bytes..(row + 1) * row_bytes]
                }),
                "row-batched GEMV differs from ordinary decode"
            );
            let bytes = guarded.raw_bytes();
            let pad = 8 * dtype.size();
            ensure!(
                bytes[..pad]
                    .iter()
                    .chain(&bytes[bytes.len() - pad..])
                    .all(|&b| b == 0xa5),
                "row projection wrote outside its view"
            );
        }
    }
    Ok(())
}

fn projection_chain_gpu_seconds(
    ctx: &MetalContext,
    weights: &[QuantWeights],
    input: &Tensor,
    output: &Tensor,
    batched: bool,
) -> Result<f64> {
    let m = input.shape()[0];
    let k = input.shape()[1];
    let n = output.shape()[1];
    let pass = ctx.begin_concurrent()?;
    if batched {
        for w in weights {
            gemv_quant_rows_decode_exact(ctx, &pass, w, input, output)?;
            pass.level_barrier(&[output])?;
        }
    } else {
        // Ordinary decode reads every layer before advancing to another
        // token. The candidate consumes all rows while a layer is current.
        for row in 0..m {
            let x = input.view(row * k, &[k])?;
            let y = output.view(row * n, &[n])?;
            for w in weights {
                gemv_quant(ctx, &pass, w, &x, &y)?;
                pass.level_barrier(&[output])?;
            }
        }
    }
    let completed = pass.commit()?.wait_retain()?;
    let timing = completed.timing()?;
    let seconds = timing.gpu_end_secs - timing.gpu_start_secs;
    ensure!(seconds.is_finite() && seconds > 0.0, "projection timing unavailable");
    Ok(seconds)
}

/// Isolated projections, not an end-to-end generation benchmark. Distinct
/// shared/GDN layer weights exceed cache capacity; the full head itself is
/// larger than cache. Every candidate is byte checked before timed pairs.
#[test]
#[ignore = "physical Apple Silicon; alternating component timings only"]
fn decode_rows_projection_streaming_timing() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut seed = 0x4f73_a991;
    for (name, bits, k, n, dtype, layers) in [
        ("shared_gate_up_stream", 4, 2560, 1280, DType::BF16, 48),
        ("gdn_input_stream", 4, 2560, 16480, DType::BF16, 4),
        ("vocabulary_head", 8, 2560, 248320, DType::F32, 1),
    ] {
        let weights: Vec<_> = (0..layers)
            .map(|_| weights(&ctx, &mut seed, bits, n, k))
            .collect::<Result<_>>()?;
        let weight_bytes: usize = weights
            .iter()
            .map(|w| w.codes.byte_len() + w.scales.byte_len() + w.biases.byte_len())
            .sum();
        for m in 2..=4 {
            let values: Vec<_> = (0..m * k)
                .map(|_| (word(&mut seed) % 4097) as f32 / 1024.0 - 2.0)
                .collect();
            let input = Tensor::from_f32_as_bf16(&ctx, &values, &[m, k])?;
            let reference = Tensor::zeros(&ctx, &[m, n], dtype)?;
            let candidate = Tensor::zeros(&ctx, &[m, n], dtype)?;
            for w in &weights {
                let pass = ctx.begin_concurrent()?;
                for row in 0..m {
                    gemv_quant(
                        &ctx,
                        &pass,
                        w,
                        &input.view(row * k, &[k])?,
                        &reference.view(row * n, &[n])?,
                    )?;
                }
                gemv_quant_rows_decode_exact(&ctx, &pass, w, &input, &candidate)?;
                pass.commit()?.wait()?;
                ensure!(
                    reference.raw_bytes() == candidate.raw_bytes(),
                    "timed projection failed byte control"
                );
            }
            let warmup = std::time::Instant::now();
            let mut samples = [Vec::new(), Vec::new()];
            let mut round = 0;
            let mut measured = 0;
            loop {
                let timed = warmup.elapsed() >= std::time::Duration::from_secs(2);
                for order in 0..2 {
                    let variant = (round + order) % 2;
                    let output = if variant == 0 { &reference } else { &candidate };
                    let seconds = projection_chain_gpu_seconds(
                        &ctx,
                        &weights,
                        &input,
                        output,
                        variant == 1,
                    )?;
                    if timed {
                        samples[variant].push(seconds * 1e6);
                    }
                }
                round += 1;
                if timed {
                    measured += 1;
                    if measured == 30 {
                        break;
                    }
                }
            }
            let separate_mean = samples[0].iter().sum::<f64>() / 30.0;
            let batched_mean = samples[1].iter().sum::<f64>() / 30.0;
            eprintln!(
                "{}",
                serde_json::json!({
                    "scope":"synthetic_decode_projection_streaming",
                    "shape":name,"bits":bits,"input_width":k,"output_width":n,
                    "rows":m,"layers":layers,"weight_bytes":weight_bytes,"pairs":30,
                    "reference_gpu_us":samples[0],"batched_gpu_us":samples[1],
                    "reference_mean_gpu_us":separate_mean,"batched_mean_gpu_us":batched_mean,
                    "speedup":separate_mean / batched_mean,"byte_control":true,
                })
            );
        }
    }
    Ok(())
}
