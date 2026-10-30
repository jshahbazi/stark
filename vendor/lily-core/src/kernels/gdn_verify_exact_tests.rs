// Original Stark short-GDN verification controls; synthetic operands only.
use super::*;
use crate::kernels::elementwise::copy_words;
use crate::sha256::{Sha256, hex};

fn next(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn values(random: &mut u32, count: usize, scale: f32, bias: f32) -> Vec<f32> {
    (0..count).map(|_| (next(random) % 4096) as f32 * scale / 4096.0 + bias).collect()
}

struct Guarded {
    storage: Tensor,
    tensor: Tensor,
}

impl Guarded {
    fn new(
        ctx: &MetalContext,
        values: &[f32],
        dtype: DType,
        shape: &[usize],
    ) -> Result<Self> {
        let storage = Tensor::zeros(ctx, &[values.len() + 4], dtype)?;
        storage.write_bytes(&vec![0xa5; storage.byte_len()])?;
        let tensor = storage.view(2, shape)?;
        match dtype {
            DType::F32 => tensor.write_bytes(bytemuck::cast_slice(values))?,
            DType::BF16 => {
                let rounded: Vec<_> =
                    values.iter().map(|&v| half::bf16::from_f32(v)).collect();
                tensor.write_bytes(bytemuck::cast_slice(&rounded))?;
            }
            _ => anyhow::bail!("unsupported GDN control dtype"),
        }
        Ok(Self { storage, tensor })
    }

    fn canaries(&self) -> Result<()> {
        let bytes = self.storage.raw_bytes();
        let n = 2 * self.storage.dtype().size();
        ensure!(
            bytes[..n].iter().chain(&bytes[bytes.len() - n..]).all(|&b| b == 0xa5),
            "GDN verifier wrote outside view"
        );
        Ok(())
    }
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn gdn_verification_batch_matches_decode_bytes() -> Result<()> {
    check_kernel("gdn_verify_gated_exact")
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn gdn_register_verification_matches_decode_bytes() -> Result<()> {
    check_kernel("gdn_verify_gated_exact_register")
}

fn check_kernel(kernel: &'static str) -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut random = 0x4744_4e56;
    let mut reference_hash = Sha256::new();
    let mut candidate_hash = Sha256::new();
    let mut checked_bytes = 0;
    let mut cases = 0;
    for (hk, h) in [(2, 6), (16, 48)] {
        let d = GDN_HEAD_DIM;
        let c = (2 * hk + h) * d;
        let width = h * d;
        let state_words = h * d * d;
        for m in 1..=4 {
            for gate in [GdnGate::Sigmoid, GdnGate::Silu] {
                for style in 0..4 {
                    let mut raw = values(&mut random, m * c, 2.0, -1.0);
                    let mut a_values = values(&mut random, m * h, 2.0, -1.0);
                    let mut b_values = values(&mut random, m * h, 2.0, -1.0);
                    let mut z_values = values(&mut random, m * width, 2.0, -1.0);
                    let mut initial = values(&mut random, state_words, 0.5, -0.25);
                    if style == 1 {
                        for row in raw.chunks_exact_mut(c) {
                            row[..2 * hk * d].fill(0.0);
                        }
                        initial.fill(0.0);
                    } else if style == 2 {
                        for (i, a) in a_values.iter_mut().enumerate() {
                            *a = if i.is_multiple_of(2) { 40.0 } else { -40.0 };
                        }
                        for (i, b) in b_values.iter_mut().enumerate() {
                            *b = if i.is_multiple_of(2) { -40.0 } else { 40.0 };
                        }
                        for (i, z) in z_values.iter_mut().enumerate() {
                            *z = if i.is_multiple_of(2) { 40.0 } else { -40.0 };
                        }
                    } else if style == 3 {
                        for (i, x) in raw.iter_mut().enumerate() {
                            *x = [0.0, -0.0, 1e-30, -1e-30, 64.0, -64.0, 0.5, -0.5]
                                [i % 8];
                        }
                    }
                    let qkv = Guarded::new(&ctx, &raw, DType::BF16, &[m, c])?;
                    let a = Guarded::new(&ctx, &a_values, DType::BF16, &[m, h])?;
                    let b = Guarded::new(&ctx, &b_values, DType::BF16, &[m, h])?;
                    let z = Guarded::new(&ctx, &z_values, DType::BF16, &[m, width])?;
                    let a_log = Guarded::new(
                        &ctx,
                        &values(&mut random, h, 2.5, -2.0),
                        DType::F32,
                        &[h],
                    )?;
                    let dt_bias = Guarded::new(
                        &ctx,
                        &values(&mut random, h, 1.0, -0.5),
                        DType::BF16,
                        &[h],
                    )?;
                    let norm = Guarded::new(
                        &ctx,
                        &values(&mut random, d, 1.0, 0.5),
                        DType::F32,
                        &[d],
                    )?;
                    let reference_state =
                        Guarded::new(&ctx, &initial, DType::F32, &[h, d, d])?;
                    let candidate_state =
                        Guarded::new(&ctx, &initial, DType::F32, &[h, d, d])?;
                    let reference_out = Guarded::new(
                        &ctx,
                        &vec![0.0; m * width],
                        DType::BF16,
                        &[m, width],
                    )?;
                    let candidate_out = Guarded::new(
                        &ctx,
                        &vec![0.0; m * width],
                        DType::BF16,
                        &[m, width],
                    )?;
                    let captures = (m - 1).max(1);
                    let reference_mid = Guarded::new(
                        &ctx,
                        &vec![0.0; captures * state_words],
                        DType::F32,
                        &[captures, h, d, d],
                    )?;
                    let candidate_mid = Guarded::new(
                        &ctx,
                        &vec![0.0; captures * state_words],
                        DType::F32,
                        &[captures, h, d, d],
                    )?;
                    let pass = ctx.begin_concurrent()?;
                    for row in 0..m {
                        gdn_step_gated_fused(
                            &ctx,
                            &pass,
                            &qkv.tensor.view(row * c, &[c])?,
                            &a.tensor.view(row * h, &[h])?,
                            &b.tensor.view(row * h, &[h])?,
                            &a_log.tensor,
                            &dt_bias.tensor,
                            &reference_state.tensor,
                            &z.tensor.view(row * width, &[width])?,
                            &norm.tensor,
                            &reference_out.tensor.view(row * width, &[width])?,
                            1.0 / (d as f32).sqrt(),
                            hk,
                            1e-6,
                            gate,
                        )?;
                        pass.level_barrier(&[
                            &reference_state.tensor,
                            &reference_out.tensor,
                        ])?;
                        if row + 1 < m {
                            copy_words(
                                &ctx,
                                &pass,
                                &reference_state.tensor,
                                &reference_mid
                                    .tensor
                                    .view(row * state_words, &[h, d, d])?,
                            )?;
                            pass.level_barrier(&[&reference_mid.tensor])?;
                        }
                    }
                    gdn_verify_gated_exact_named(
                        &ctx,
                        &pass,
                        &qkv.tensor,
                        &a.tensor,
                        &b.tensor,
                        &a_log.tensor,
                        &dt_bias.tensor,
                        &candidate_state.tensor,
                        &z.tensor,
                        &norm.tensor,
                        &candidate_out.tensor,
                        1.0 / (d as f32).sqrt(),
                        hk,
                        1e-6,
                        gate,
                        (m > 1).then_some(&candidate_mid.tensor),
                        kernel,
                    )?;
                    pass.commit_wait()?;
                    for (field, reference, candidate) in [
                        ("state", &reference_state, &candidate_state),
                        ("output", &reference_out, &candidate_out),
                        ("mid", &reference_mid, &candidate_mid),
                    ] {
                        reference.canaries()?;
                        candidate.canaries()?;
                        ensure!(
                            reference.tensor.raw_bytes()
                                == candidate.tensor.raw_bytes(),
                            "GDN verify mismatch field={field} m={m} h={h} style={style} gate={gate:?}"
                        );
                        ensure!(
                            candidate.tensor.to_f32()?.iter().all(|v| v.is_finite()),
                            "nonfinite GDN verify result"
                        );
                        reference_hash.update(reference.tensor.raw_bytes());
                        candidate_hash.update(candidate.tensor.raw_bytes());
                        checked_bytes += reference.tensor.byte_len();
                    }
                    cases += 1;
                }
            }
        }
    }
    eprintln!(
        "GDN verify exact: kernel={kernel} cases={cases} checked_bytes={checked_bytes} reference_sha256={} candidate_sha256={}",
        hex(&reference_hash.finish()),
        hex(&candidate_hash.finish())
    );
    Ok(())
}

#[test]
#[ignore = "requires an exclusive physical Apple Silicon GPU slot; component timing only"]
fn gdn_verification_streaming_chain_timing() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut random = 0x4744_4e54;
    let (h, hk, d, layers) = (48, 16, GDN_HEAD_DIM, 36);
    let c = (2 * hk + h) * d;
    let width = h * d;
    let state_words = h * d * d;
    for m in [2, 4] {
        let qkv = Tensor::from_f32_as_bf16(
            &ctx,
            &values(&mut random, m * c, 2.0, -1.0),
            &[m, c],
        )?;
        let a = Tensor::from_f32_as_bf16(
            &ctx,
            &values(&mut random, m * h, 2.0, -1.0),
            &[m, h],
        )?;
        let b = Tensor::from_f32_as_bf16(
            &ctx,
            &values(&mut random, m * h, 2.0, -1.0),
            &[m, h],
        )?;
        let z = Tensor::from_f32_as_bf16(
            &ctx,
            &values(&mut random, m * width, 2.0, -1.0),
            &[m, width],
        )?;
        let a_log = Tensor::from_f32(&ctx, &values(&mut random, h, 2.5, -2.0), &[h])?;
        let dt_bias =
            Tensor::from_f32_as_bf16(&ctx, &values(&mut random, h, 1.0, -0.5), &[h])?;
        let norm = Tensor::from_f32(&ctx, &values(&mut random, d, 1.0, 0.5), &[d])?;
        let out = Tensor::zeros(&ctx, &[m, width], DType::BF16)?;
        let initial = values(&mut random, state_words, 0.5, -0.25);
        let mut states = [Vec::new(), Vec::new(), Vec::new()];
        for _ in 0..layers {
            for variant in &mut states {
                variant.push((
                    Tensor::from_f32(&ctx, &initial, &[h, d, d])?,
                    Tensor::zeros(&ctx, &[m - 1, h, d, d], DType::F32)?,
                ));
            }
        }
        let mut samples = [Vec::new(), Vec::new(), Vec::new()];
        let warmup = std::time::Instant::now();
        let mut round = 0;
        let mut measured = 0;
        loop {
            let timed = warmup.elapsed() >= std::time::Duration::from_secs(2);
            for order in 0..3 {
                let variant = (round + order) % 3;
                let pass = ctx.begin_concurrent()?;
                for (state, mid) in &states[variant] {
                    if variant == 0 {
                        for row in 0..m {
                            gdn_step_gated_fused(
                                &ctx,
                                &pass,
                                &qkv.view(row * c, &[c])?,
                                &a.view(row * h, &[h])?,
                                &b.view(row * h, &[h])?,
                                &a_log,
                                &dt_bias,
                                state,
                                &z.view(row * width, &[width])?,
                                &norm,
                                &out.view(row * width, &[width])?,
                                1.0 / (d as f32).sqrt(),
                                hk,
                                1e-6,
                                GdnGate::Sigmoid,
                            )?;
                            pass.level_barrier(&[state, &out])?;
                            if row + 1 < m {
                                copy_words(
                                    &ctx,
                                    &pass,
                                    state,
                                    &mid.view(row * state_words, &[h, d, d])?,
                                )?;
                                pass.level_barrier(&[mid])?;
                            }
                        }
                    } else {
                        let kernel = if variant == 1 {
                            "gdn_verify_gated_exact"
                        } else {
                            "gdn_verify_gated_exact_register"
                        };
                        gdn_verify_gated_exact_named(
                            &ctx,
                            &pass,
                            &qkv,
                            &a,
                            &b,
                            &a_log,
                            &dt_bias,
                            state,
                            &z,
                            &norm,
                            &out,
                            1.0 / (d as f32).sqrt(),
                            hk,
                            1e-6,
                            GdnGate::Sigmoid,
                            Some(mid),
                            kernel,
                        )?;
                        pass.level_barrier(&[state, mid, &out])?;
                    }
                }
                let complete = pass.commit()?.wait_retain()?;
                let timing = complete.timing()?;
                if timed {
                    samples[variant].push(
                        (timing.gpu_end_secs - timing.gpu_start_secs) * 1e6
                            / layers as f64,
                    );
                }
            }
            round += 1;
            if timed {
                measured += 1;
                if measured == 18 {
                    break;
                }
            }
        }
        for (variant, times) in samples.iter_mut().enumerate() {
            times.sort_unstable_by(f64::total_cmp);
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            let median = (times[8] + times[9]) * 0.5;
            eprintln!(
                "GDN verify streaming: m={m} variant={variant} layers={layers} state_bytes={} mid_bytes={} samples={} mean_us_per_layer={mean:.6} median_us_per_layer={median:.6}",
                layers * state_words * 4,
                layers * (m - 1) * state_words * 4,
                times.len()
            );
        }
    }
    Ok(())
}
