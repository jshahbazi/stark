// Modified by Stark: cached-activation controls adapted from Lily's
// tests/unit/kernels/hc.rs at e22d58fa7cc488493fb1b5d810a198bd29730aeb.
// Upstream tests and kernels are Apache-2.0; see LICENSE, NOTICE and UPSTREAM.md.
// Synthetic tensors only. Never print tensor values, even on a failure.
use super::*;
use crate::sha256::{Sha256, hex};

fn next_word(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn random_values(state: &mut u32, len: usize, scale: f32, bias: f32) -> Vec<f32> {
    (0..len).map(|_| (next_word(state) % 4096) as f32 * scale / 4096.0 + bias).collect()
}

fn q8_weight(
    ctx: &MetalContext,
    state: &mut u32,
    n: usize,
    k: usize,
    gs: usize,
) -> Result<QuantWeights> {
    // Two leading rows exercise nonzero, 16-byte-aligned code offsets.
    let allocated = n + 2;
    let codes: Vec<u32> = (0..allocated * k / 4).map(|_| next_word(state)).collect();
    let scales = random_values(state, allocated * k / gs, 0.0095, 0.0005);
    let biases = random_values(state, allocated * k / gs, 1.0, -1.0);
    QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[allocated, k / 4],
            DType::U32,
        )?,
        scales: Tensor::from_f32_as_bf16(ctx, &scales, &[allocated, k / gs])?,
        biases: Tensor::from_f32_as_bf16(ctx, &biases, &[allocated, k / gs])?,
        group_size: gs,
        bits: 8,
    }
    .view_rows(2, n)
}

struct GuardedTensor {
    storage: Tensor,
    value: Tensor,
}

impl GuardedTensor {
    fn new(ctx: &MetalContext, n: usize, dtype: DType) -> Result<Self> {
        let storage = Tensor::zeros(ctx, &[n + 4], dtype)?;
        storage.write_bytes(&vec![0xa5; storage.byte_len()])?;
        let value = storage.view(2, &[1, n])?;
        Ok(Self { storage, value })
    }

    fn check_canaries(&self) -> Result<()> {
        let bytes = self.storage.raw_bytes();
        let padding = 2 * self.storage.dtype().size();
        ensure!(
            bytes[..padding]
                .iter()
                .chain(&bytes[bytes.len() - padding..])
                .all(|b| *b == 0xa5),
            "HC kernel wrote outside its output view"
        );
        Ok(())
    }
}

struct ReadOutputs {
    down: GuardedTensor,
    inj: GuardedTensor,
    inv_rms: GuardedTensor,
    act: GuardedTensor,
    mixed: GuardedTensor,
}

impl ReadOutputs {
    fn new(ctx: &MetalContext, h: usize, g: usize, r: usize) -> Result<Self> {
        Ok(Self {
            down: GuardedTensor::new(ctx, r, DType::BF16)?,
            inj: GuardedTensor::new(ctx, g, DType::BF16)?,
            inv_rms: GuardedTensor::new(ctx, g, DType::F32)?,
            act: GuardedTensor::new(ctx, r, DType::BF16)?,
            mixed: GuardedTensor::new(ctx, h, DType::BF16)?,
        })
    }

    fn fields(&self) -> [(&str, &GuardedTensor); 5] {
        [
            ("down", &self.down),
            ("inject", &self.inj),
            ("inv_rms", &self.inv_rms),
            ("activation", &self.act),
            ("mixed", &self.mixed),
        ]
    }
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn cached_read_is_bit_identical_to_single_row() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut state = 0x4843_4d31u32;
    let mut reference_hash = Sha256::new();
    let mut candidate_hash = Sha256::new();
    let mut cases = 0;
    let mut checked_bytes = 0;
    // Production with/without injection, output-column tails, the maximum
    // stream count, and short/non-64 quantization groups.
    for (h, g, r, gs, with_inject) in [
        (2560, 4, 320, 64, true),
        (2560, 4, 320, 64, false),
        (192, 3, 128, 64, true),
        (208, 4, 192, 64, true),
        (64, 8, 64, 64, true),
        (256, 1, 64, 64, false),
        (176, 1, 80, 16, true),
    ] {
        let k = g * h;
        let down_w = q8_weight(&ctx, &mut state, r, k, gs)?;
        let up_w = q8_weight(&ctx, &mut state, k, r, gs)?;
        let inject = if with_inject {
            Some(q8_weight(&ctx, &mut state, g, k, gs)?)
        } else {
            None
        };
        let norm_values = random_values(&mut state, k + 4, 1.0, -0.5);
        let norm_storage = Tensor::from_f32_as_bf16(&ctx, &norm_values, &[k + 4])?;
        let norm = norm_storage.view(2, &[k])?;
        for style in 0..4 {
            let mut values = random_values(&mut state, k + 4, 6.0, -3.0);
            for (i, v) in values.iter_mut().enumerate() {
                match style {
                    1 => *v = 0.0,
                    2 => *v = if i.is_multiple_of(2) { 32.0 } else { -32.0 },
                    3 => {
                        *v = half::bf16::from_bits(
                            [0x0000, 0x8000, 0x0001, 0x8001, 0x3f80, 0xbf80][i % 6],
                        )
                        .to_f32();
                    }
                    _ => {}
                }
            }
            let hyper_storage = Tensor::from_f32_as_bf16(&ctx, &values, &[k + 4])?;
            let hyper = hyper_storage.view(2, &[k])?;
            let hyper_rows = hyper.view(0, &[1, k])?;
            let reference = ReadOutputs::new(&ctx, h, g, r)?;
            let candidate = ReadOutputs::new(&ctx, h, g, r)?;
            let pass = ctx.begin_concurrent()?;
            hc_read_down_q8(
                &ctx,
                &pass,
                &hyper,
                &norm,
                &down_w,
                inject.as_ref(),
                &reference.down.value,
                &reference.inj.value,
                &reference.inv_rms.value,
                h,
                g,
                1e-6,
                1.0,
            )?;
            pass.level_barrier(&[
                &reference.down.value,
                &reference.inj.value,
                &reference.inv_rms.value,
            ])?;
            hc_read_up_mix_q8(
                &ctx,
                &pass,
                &up_w,
                &reference.down.value,
                &hyper,
                &norm,
                &reference.inv_rms.value,
                &reference.mixed.value,
                h,
                g,
                1.0,
            )?;
            silu_scaled_bf16(
                &ctx,
                &pass,
                &reference.down.value,
                &reference.act.value,
                1.0 / g as f32,
            )?;
            pass.level_barrier(&[&reference.mixed.value, &reference.act.value])?;
            hc_read_down_q8_rows(
                &ctx,
                &pass,
                &hyper_rows,
                &norm,
                &down_w,
                inject.as_ref(),
                &candidate.down.value,
                &candidate.inj.value,
                &candidate.inv_rms.value,
                &candidate.act.value,
                h,
                g,
                1e-6,
                1.0,
            )?;
            pass.level_barrier(&[
                &candidate.down.value,
                &candidate.inj.value,
                &candidate.inv_rms.value,
                &candidate.act.value,
            ])?;
            hc_read_up_mix_q8_cached(
                &ctx,
                &pass,
                &up_w,
                &candidate.act.value,
                &hyper,
                &norm,
                &candidate.inv_rms.value,
                &candidate.mixed.value,
                h,
                g,
                1.0,
            )?;
            pass.commit_wait()?;
            for ((field, a), (_, b)) in
                reference.fields().into_iter().zip(candidate.fields())
            {
                a.check_canaries()?;
                b.check_canaries()?;
                ensure!(
                    a.value.raw_bytes() == b.value.raw_bytes(),
                    "HC cached mismatch: field={field} h={h} g={g} r={r} gs={gs} inject={with_inject} style={style}"
                );
                reference_hash.update(a.value.raw_bytes());
                candidate_hash.update(b.value.raw_bytes());
                checked_bytes += a.value.byte_len();
            }
            ensure!(
                candidate
                    .inv_rms
                    .value
                    .to_f32()?
                    .iter()
                    .all(|v| v.is_finite() && *v > 0.0),
                "HC cached inverse RMS was not finite and positive"
            );
            cases += 1;
        }
    }
    eprintln!(
        "HC cached exact controls: cases={cases} checked_bytes={checked_bytes} reference_sha256={} candidate_sha256={}",
        hex(&reference_hash.finish()),
        hex(&candidate_hash.finish())
    );
    Ok(())
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device; diagnostic timing only"]
fn cached_read_streaming_chain_timing() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut random = 0x4843_4241u32;
    let (h, g, r, gs, reads) = (2560, 4, 320, 64, 97);
    let k = g * h;
    let hyper = Tensor::from_f32_as_bf16(
        &ctx,
        &random_values(&mut random, k, 6.0, -3.0),
        &[k],
    )?;
    let hyper_rows = hyper.view(0, &[1, k])?;
    let mut weights = Vec::new();
    let mut weight_bytes = 0;
    // All 97 reads own distinct weights, matching the model's HC count.
    // The aggregate 680 MB prevents a single weight set staying cache-hot.
    for index in 0..reads {
        let norm = Tensor::from_f32_as_bf16(
            &ctx,
            &random_values(&mut random, k, 1.0, -0.5),
            &[k],
        )?;
        let down = q8_weight(&ctx, &mut random, r, k, gs)?;
        let up = q8_weight(&ctx, &mut random, k, r, gs)?;
        let inject = if index + 1 == reads {
            None
        } else {
            Some(q8_weight(&ctx, &mut random, g, k, gs)?)
        };
        for weight in [&down, &up].into_iter().chain(inject.as_ref()) {
            weight_bytes += weight.codes.byte_len()
                + weight.scales.byte_len()
                + weight.biases.byte_len();
        }
        weights.push((norm, down, up, inject));
    }
    let out = ReadOutputs::new(&ctx, h, g, r)?;
    let mut times = [Vec::new(), Vec::new()];
    let warmup = std::time::Instant::now();
    let mut round = 0;
    let mut measured_rounds = 0;
    loop {
        let timed = warmup.elapsed() >= std::time::Duration::from_secs(2);
        // AB then BA on successive rounds, using the same immutable weights.
        for order in 0..2 {
            let variant = (round + order) % 2;
            let pass = ctx.begin_concurrent()?;
            for (norm, down, up, inject) in &weights {
                if variant == 0 {
                    hc_read_down_q8(
                        &ctx,
                        &pass,
                        &hyper,
                        norm,
                        down,
                        inject.as_ref(),
                        &out.down.value,
                        &out.inj.value,
                        &out.inv_rms.value,
                        h,
                        g,
                        1e-6,
                        1.0,
                    )?;
                } else {
                    hc_read_down_q8_rows(
                        &ctx,
                        &pass,
                        &hyper_rows,
                        norm,
                        down,
                        inject.as_ref(),
                        &out.down.value,
                        &out.inj.value,
                        &out.inv_rms.value,
                        &out.act.value,
                        h,
                        g,
                        1e-6,
                        1.0,
                    )?;
                }
                pass.level_barrier(&[
                    &out.down.value,
                    &out.inj.value,
                    &out.inv_rms.value,
                    &out.act.value,
                ])?;
                if variant == 0 {
                    hc_read_up_mix_q8(
                        &ctx,
                        &pass,
                        up,
                        &out.down.value,
                        &hyper,
                        norm,
                        &out.inv_rms.value,
                        &out.mixed.value,
                        h,
                        g,
                        1.0,
                    )?;
                } else {
                    hc_read_up_mix_q8_cached(
                        &ctx,
                        &pass,
                        up,
                        &out.act.value,
                        &hyper,
                        norm,
                        &out.inv_rms.value,
                        &out.mixed.value,
                        h,
                        g,
                        1.0,
                    )?;
                }
                pass.level_barrier(&[&out.mixed.value])?;
            }
            let completed = pass.commit()?.wait_retain()?;
            let timing = completed.timing()?;
            if timed {
                times[variant].push(
                    (timing.gpu_end_secs - timing.gpu_start_secs) * 1e6 / reads as f64,
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
            "HC cached streaming chain: cached={} reads={reads} weight_bytes={weight_bytes} samples={} mean_us_per_read={mean:.6} median_us_per_read={median:.6}",
            variant != 0,
            samples.len(),
        );
    }
    Ok(())
}
