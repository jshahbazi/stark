// Original Stark controls for the Q4 GDN projection/convolution fusion.
// Quantized dots and the reference convolution use the pinned generic APIs.
// Synthetic tensors only; reports contain aggregate counts, hashes and times.
use super::*;
use crate::kernels::quant::{gemv_q4_gdn_conv_step, gemv_quant};
use crate::weights::QuantWeights;

fn q4_weight(
    ctx: &MetalContext,
    random: &mut u32,
    n: usize,
    k: usize,
) -> Result<QuantWeights> {
    let allocated = n + 2;
    let codes: Vec<u32> = (0..allocated * k / 8).map(|_| next_word(random)).collect();
    let scales = random_values(random, allocated * k / 64, 0.02, 0.001);
    let biases = random_values(random, allocated * k / 64, 0.125, -0.125);
    QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[allocated, k / 8],
            DType::U32,
        )?,
        scales: Tensor::from_f32_as_bf16(ctx, &scales, &[allocated, k / 64])?,
        biases: Tensor::from_f32_as_bf16(ctx, &biases, &[allocated, k / 64])?,
        group_size: 64,
        bits: 4,
    }
    .view_rows(2, n)
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn fused_projection_convolution_is_bit_identical() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut random = 0x4744_5031u32;
    let mut reference_hash = Sha256::new();
    let mut candidate_hash = Sha256::new();
    let mut cases = 0;
    let mut checked_bytes = 0;
    // Production QKV/Z/A/B stack, then a small projection whose convolution
    // boundary divides a row pair (row C-1 is convolved; row C is raw-only).
    for (n, k, c) in [(16480, 2560, 10240), (258, 128, 191)] {
        let weight = q4_weight(&ctx, &mut random, n, k)?;
        let conv_w = offset_input(
            &ctx,
            &random_values(&mut random, 4 * c, 1.0, -0.5),
            DType::BF16,
        )?
        .view(0, &[4, c])?;
        for style in 0..3 {
            let initial = random_values(&mut random, 3 * c, 2.0, -1.0);
            let reference_window = GuardedTensor::new(&ctx, &initial, DType::BF16)?;
            let candidate_window = GuardedTensor::new(&ctx, &initial, DType::BF16)?;
            let reference_raw = GuardedTensor::new(&ctx, &vec![0.0; n], DType::BF16)?;
            let candidate_raw = GuardedTensor::new(&ctx, &vec![0.0; n], DType::BF16)?;
            let reference_conv = GuardedTensor::new(&ctx, &vec![0.0; c], DType::BF16)?;
            let candidate_conv = GuardedTensor::new(&ctx, &vec![0.0; c], DType::BF16)?;
            let reference_qkv = reference_raw.value.view(0, &[c])?;
            for step in 0..16 {
                let mut values = random_values(&mut random, k, 2.0, -1.0);
                for (i, value) in values.iter_mut().enumerate() {
                    match style {
                        1 => *value = 0.0,
                        2 => {
                            *value = [0.0, -0.0, 2.0, -2.0, 0.5, -0.5][(i + step) % 6];
                        }
                        _ => {}
                    }
                }
                let x = offset_input(&ctx, &values, DType::BF16)?;
                let pass = ctx.begin_concurrent()?;
                gemv_quant(&ctx, &pass, &weight, &x, &reference_raw.value)?;
                pass.level_barrier(&[&reference_raw.value])?;
                conv1d_step(
                    &ctx,
                    &pass,
                    &reference_window.value,
                    &reference_qkv,
                    &conv_w,
                    &reference_conv.value,
                )?;
                pass.level_barrier(&[&reference_window.value, &reference_conv.value])?;
                gemv_q4_gdn_conv_step(
                    &ctx,
                    &pass,
                    &weight,
                    &x,
                    &candidate_raw.value,
                    &candidate_window.value,
                    &conv_w,
                    &candidate_conv.value,
                )?;
                pass.commit_wait()?;
                for (field, reference, candidate) in [
                    ("raw_projection", &reference_raw, &candidate_raw),
                    ("convolution", &reference_conv, &candidate_conv),
                    ("window", &reference_window, &candidate_window),
                ] {
                    reference.check_canaries()?;
                    candidate.check_canaries()?;
                    ensure!(
                        reference.value.raw_bytes() == candidate.value.raw_bytes(),
                        "GDN fusion mismatch: field={field} n={n} k={k} c={c} style={style} step={step}"
                    );
                    ensure!(
                        candidate.value.to_f32()?.iter().all(|v| v.is_finite()),
                        "GDN fusion output/window is nonfinite"
                    );
                    reference_hash.update(reference.value.raw_bytes());
                    candidate_hash.update(candidate.value.raw_bytes());
                    checked_bytes += reference.value.byte_len();
                }
            }
            cases += 1;
        }
    }
    eprintln!(
        "GDN fusion exact controls: cases={cases} steps_per_case=16 checked_bytes={checked_bytes} reference_sha256={} candidate_sha256={}",
        hex(&reference_hash.finish()),
        hex(&candidate_hash.finish())
    );
    Ok(())
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device; diagnostic timing only"]
fn fused_projection_convolution_streaming_chain_timing() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut random = 0x4744_5032u32;
    let (n, k, c, layers) = (16480, 2560, 10240, 36);
    let x = offset_input(&ctx, &random_values(&mut random, k, 2.0, -1.0), DType::BF16)?;
    let mut weights = Vec::new();
    let mut windows = [Vec::new(), Vec::new()];
    let mut weight_bytes = 0;
    for _ in 0..layers {
        let weight = q4_weight(&ctx, &mut random, n, k)?;
        weight_bytes += weight.codes.byte_len()
            + weight.scales.byte_len()
            + weight.biases.byte_len();
        let conv_w = offset_input(
            &ctx,
            &random_values(&mut random, 4 * c, 1.0, -0.5),
            DType::BF16,
        )?
        .view(0, &[4, c])?;
        weights.push((weight, conv_w));
        let initial = random_values(&mut random, 3 * c, 2.0, -1.0);
        for variant in &mut windows {
            variant.push(GuardedTensor::new(&ctx, &initial, DType::BF16)?);
        }
    }
    let raw = GuardedTensor::new(&ctx, &vec![0.0; n], DType::BF16)?;
    let qkv = raw.value.view(0, &[c])?;
    let conv = GuardedTensor::new(&ctx, &vec![0.0; c], DType::BF16)?;
    let mut times = [Vec::new(), Vec::new()];
    let warmup = std::time::Instant::now();
    let mut round = 0;
    let mut measured_rounds = 0;
    loop {
        let timed = warmup.elapsed() >= std::time::Duration::from_secs(2);
        for order in 0..2 {
            let variant = (round + order) % 2;
            let pass = ctx.begin_concurrent()?;
            for ((weight, conv_w), window) in weights.iter().zip(&windows[variant]) {
                if variant == 0 {
                    gemv_quant(&ctx, &pass, weight, &x, &raw.value)?;
                    pass.level_barrier(&[&raw.value])?;
                    conv1d_step(&ctx, &pass, &window.value, &qkv, conv_w, &conv.value)?;
                } else {
                    gemv_q4_gdn_conv_step(
                        &ctx,
                        &pass,
                        weight,
                        &x,
                        &raw.value,
                        &window.value,
                        conv_w,
                        &conv.value,
                    )?;
                }
                pass.level_barrier(&[&raw.value, &window.value, &conv.value])?;
            }
            let completed = pass.commit()?.wait_retain()?;
            let timing = completed.timing()?;
            if timed {
                times[variant].push(
                    (timing.gpu_end_secs - timing.gpu_start_secs) * 1e6 / layers as f64,
                );
            }
        }
        round += 1;
        if timed {
            measured_rounds += 1;
            if measured_rounds == 16 {
                break;
            }
        }
    }
    for (variant, samples) in times.iter_mut().enumerate() {
        samples.sort_unstable_by(f64::total_cmp);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let median = (samples[7] + samples[8]) * 0.5;
        eprintln!(
            "GDN fusion streaming chain: fused={} layers={layers} weight_bytes={weight_bytes} samples={} mean_us_per_layer={mean:.6} median_us_per_layer={median:.6}",
            variant != 0,
            samples.len(),
        );
    }
    Ok(())
}
