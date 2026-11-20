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

/// The shipped register verifier against ordinary decode: outputs and final
/// state, the untouched pre-verify state, the in-place form, and replay
/// rollback from the pre-verify state to every accepted prefix, at every row
/// count from one to nine. Three independent random draws per shape; besides
/// ordinary values the styles cover zero q/k rows, saturated gates, signed
/// zeros and extremes, subnormal-heavy state and inputs, and cancellation
/// (large alternating-sign state and keys whose dot products cancel).
#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn gdn_register_verification_matches_decode_bytes() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut reference_hash = Sha256::new();
    let mut candidate_hash = Sha256::new();
    let mut checked_bytes = 0;
    let mut cases = 0;
    let mut replay_cases = 0;
    for draw in 0..3u32 {
    let mut random = 0x4744_4e52 ^ draw.wrapping_mul(0x9e37_79b9);
    for (hk, h) in [(2, 6), (16, 48)] {
        let d = GDN_HEAD_DIM;
        let c = (2 * hk + h) * d;
        let width = h * d;
        let state_words = h * d * d;
        for m in 1..=9 {
            for gate in [GdnGate::Sigmoid, GdnGate::Silu] {
                for style in 0..6 {
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
                    } else if style == 4 {
                        // Subnormal F32 state and BF16-subnormal activations
                        // (BF16 shares F32's exponent range).
                        let tiny = f32::from_bits(1 << 16);
                        for (i, x) in raw.iter_mut().enumerate() {
                            if i % 3 != 0 {
                                *x *= tiny * (1 + (next(&mut random) % 64)) as f32;
                            }
                        }
                        for (i, s) in initial.iter_mut().enumerate() {
                            if i % 2 == 0 {
                                *s = f32::from_bits(next(&mut random) % 0x0080_0000)
                                    * if i % 4 == 0 { 1.0 } else { -1.0 };
                            }
                        }
                    } else if style == 5 {
                        // Cancellation: alternating-sign large state and keys,
                        // values near the resulting prediction.
                        for (i, s) in initial.iter_mut().enumerate() {
                            *s = if i % 2 == 0 { 3.0 } else { -3.0 }
                                + (next(&mut random) % 8) as f32 * 1e-3;
                        }
                        for row in raw.chunks_exact_mut(c) {
                            for (i, x) in row[..2 * hk * d].iter_mut().enumerate() {
                                *x = if i % 2 == 0 { 8.0 } else { -8.0 };
                            }
                            for x in &mut row[2 * hk * d..] {
                                *x = (next(&mut random) % 3) as f32 * 1e-2 - 1e-2;
                            }
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
                    let zeros = vec![0.0; state_words];
                    let reference_state =
                        Guarded::new(&ctx, &initial, DType::F32, &[h, d, d])?;
                    let pre_state = Guarded::new(&ctx, &initial, DType::F32, &[h, d, d])?;
                    let final_state = Guarded::new(&ctx, &zeros, DType::F32, &[h, d, d])?;
                    let in_place = Guarded::new(&ctx, &initial, DType::F32, &[h, d, d])?;
                    let replayed: Vec<_> = (0..m)
                        .map(|_| Guarded::new(&ctx, &zeros, DType::F32, &[h, d, d]))
                        .collect::<Result<_>>()?;
                    let blank = vec![0.0; m * width];
                    let reference_out =
                        Guarded::new(&ctx, &blank, DType::BF16, &[m, width])?;
                    let candidate_out =
                        Guarded::new(&ctx, &blank, DType::BF16, &[m, width])?;
                    let in_place_out =
                        Guarded::new(&ctx, &blank, DType::BF16, &[m, width])?;
                    let sink = Guarded::new(&ctx, &blank, DType::BF16, &[m, width])?;
                    let reference_prefix: Vec<_> = (0..m)
                        .map(|_| Guarded::new(&ctx, &zeros, DType::F32, &[h, d, d]))
                        .collect::<Result<_>>()?;
                    let scale = 1.0 / (d as f32).sqrt();
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
                            scale,
                            hk,
                            1e-6,
                            gate,
                        )?;
                        pass.level_barrier(&[
                            &reference_state.tensor,
                            &reference_out.tensor,
                        ])?;
                        copy_words(
                            &ctx,
                            &pass,
                            &reference_state.tensor,
                            &reference_prefix[row].tensor,
                        )?;
                        pass.level_barrier(&[&reference_prefix[row].tensor])?;
                    }
                    gdn_verify_gated_exact(
                        &ctx,
                        &pass,
                        &qkv.tensor,
                        &a.tensor,
                        &b.tensor,
                        &a_log.tensor,
                        &dt_bias.tensor,
                        &pre_state.tensor,
                        &final_state.tensor,
                        &z.tensor,
                        &norm.tensor,
                        &candidate_out.tensor,
                        scale,
                        hk,
                        1e-6,
                        gate,
                    )?;
                    gdn_verify_gated_exact(
                        &ctx,
                        &pass,
                        &qkv.tensor,
                        &a.tensor,
                        &b.tensor,
                        &a_log.tensor,
                        &dt_bias.tensor,
                        &in_place.tensor,
                        &in_place.tensor,
                        &z.tensor,
                        &norm.tensor,
                        &in_place_out.tensor,
                        scale,
                        hk,
                        1e-6,
                        gate,
                    )?;
                    pass.level_barrier(&[&final_state.tensor, &in_place.tensor])?;
                    for (n, target) in replayed.iter().enumerate().map(|(i, t)| (i + 1, t))
                    {
                        gdn_replay_rows(
                            &ctx,
                            &pass,
                            GdnStateLayout::KeyValue,
                            &GdnReplayRows {
                                qkv: &qkv.tensor.view(0, &[n, c])?,
                                a: &a.tensor.view(0, &[n, h])?,
                                b: &b.tensor.view(0, &[n, h])?,
                            },
                            &a_log.tensor,
                            &dt_bias.tensor,
                            &norm.tensor,
                            &pre_state.tensor,
                            &target.tensor,
                            &sink.tensor,
                            scale,
                            hk,
                            1e-6,
                            gate,
                        )?;
                        pass.level_barrier(&[&target.tensor, &sink.tensor])?;
                    }
                    pass.commit_wait()?;
                    let initial_bytes =
                        Guarded::new(&ctx, &initial, DType::F32, &[h, d, d])?;
                    let mut pairs = vec![
                        ("pre-verify state", &initial_bytes, &pre_state),
                        ("final state", &reference_state, &final_state),
                        ("in-place state", &reference_state, &in_place),
                        ("output", &reference_out, &candidate_out),
                        ("in-place output", &reference_out, &in_place_out),
                    ];
                    for (expected, target) in reference_prefix.iter().zip(&replayed) {
                        pairs.push(("replayed prefix state", expected, target));
                        replay_cases += 1;
                    }
                    for (field, reference, candidate) in pairs {
                        reference.canaries()?;
                        candidate.canaries()?;
                        ensure!(
                            reference.tensor.raw_bytes()
                                == candidate.tensor.raw_bytes(),
                            "GDN register verify mismatch field={field} m={m} h={h} style={style} gate={gate:?}"
                        );
                        ensure!(
                            candidate.tensor.to_f32()?.iter().all(|v| v.is_finite()),
                            "nonfinite GDN verify result"
                        );
                        reference_hash.update(reference.tensor.raw_bytes());
                        candidate_hash.update(candidate.tensor.raw_bytes());
                        checked_bytes += reference.tensor.byte_len();
                    }
                    sink.canaries()?;
                    cases += 1;
                }
            }
        }
    }
    }
    eprintln!(
        "GDN register verify replay exact: cases={cases} replay_cases={replay_cases} checked_bytes={checked_bytes} reference_sha256={} candidate_sha256={}",
        hex(&reference_hash.finish()),
        hex(&candidate_hash.finish())
    );
    Ok(())
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

/// Synthetic 36-layer component timing of the verify recurrence (Lily
/// geometry, distinct state allocations per layer): `m` ordinary decode
/// steps in place, against one exact verify dispatch that reads the
/// pre-verify state and writes the final state to the other buffer (the
/// replay-rollback form; no per-row captures), and against the replay of
/// `m - 1` rows a partial accept costs in the finish pass. The variant
/// order rotates each round.
#[test]
#[ignore = "requires an exclusive physical Apple Silicon GPU slot; component timing only"]
fn gdn_verification_streaming_chain_timing() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut random = 0x4744_4e54;
    let (h, hk, d, layers) = (48, 16, GDN_HEAD_DIM, 36);
    let c = (2 * hk + h) * d;
    let width = h * d;
    let state_words = h * d * d;
    const VARIANTS: usize = 3;
    for m in [1, 2, 3, 4, 9] {
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
        let sink = Tensor::zeros(&ctx, &[m, width], DType::BF16)?;
        let initial = values(&mut random, state_words, 0.5, -0.25);
        let states: Vec<(Tensor, Tensor)> = (0..layers)
            .map(|_| {
                Ok((
                    Tensor::from_f32(&ctx, &initial, &[h, d, d])?,
                    Tensor::zeros(&ctx, &[h, d, d], DType::F32)?,
                ))
            })
            .collect::<Result<_>>()?;
        let scale = 1.0 / (d as f32).sqrt();
        let mut samples = [Vec::new(), Vec::new(), Vec::new()];
        let warmup = std::time::Instant::now();
        let mut round = 0;
        let mut measured = 0;
        loop {
            let timed = warmup.elapsed() >= std::time::Duration::from_secs(2);
            for order in 0..VARIANTS {
                let variant = (round + order) % VARIANTS;
                if variant == 2 && m == 1 {
                    continue;
                }
                let pass = ctx.begin_concurrent()?;
                for (state, other) in &states {
                    match variant {
                        0 => {
                            for row in 0..m {
                                gdn_step_gated_fused(
                                    &ctx,
                                    &pass,
                                    &qkv.view(row * c, &[c])?,
                                    &a.view(row * h, &[h])?,
                                    &b.view(row * h, &[h])?,
                                    &a_log,
                                    &dt_bias,
                                    other,
                                    &z.view(row * width, &[width])?,
                                    &norm,
                                    &out.view(row * width, &[width])?,
                                    scale,
                                    hk,
                                    1e-6,
                                    GdnGate::Sigmoid,
                                )?;
                                pass.level_barrier(&[other, &out])?;
                            }
                        }
                        1 => {
                            gdn_verify_gated_exact(
                                &ctx,
                                &pass,
                                &qkv,
                                &a,
                                &b,
                                &a_log,
                                &dt_bias,
                                state,
                                other,
                                &z,
                                &norm,
                                &out,
                                scale,
                                hk,
                                1e-6,
                                GdnGate::Sigmoid,
                            )?;
                            pass.level_barrier(&[other, &out])?;
                        }
                        2 => {
                            let n = m - 1;
                            gdn_replay_rows(
                                &ctx,
                                &pass,
                                GdnStateLayout::KeyValue,
                                &GdnReplayRows {
                                    qkv: &qkv.view(0, &[n, c])?,
                                    a: &a.view(0, &[n, h])?,
                                    b: &b.view(0, &[n, h])?,
                                },
                                &a_log,
                                &dt_bias,
                                &norm,
                                state,
                                other,
                                &sink,
                                scale,
                                hk,
                                1e-6,
                                GdnGate::Sigmoid,
                            )?;
                            pass.level_barrier(&[other, &sink])?;
                        }
                        _ => unreachable!(),
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
            if times.is_empty() {
                continue;
            }
            times.sort_unstable_by(f64::total_cmp);
            let mean = times.iter().sum::<f64>() / times.len() as f64;
            let median = (times[8] + times[9]) * 0.5;
            let name = ["decode_steps", "verify_final_only", "replay_m_minus_1"][variant];
            eprintln!(
                "GDN verify recurrence timing: m={m} variant={name} layers={layers} state_bytes={} samples={} mean_us_per_layer={mean:.3} median_us_per_layer={median:.3}",
                layers * state_words * 4,
                times.len()
            );
        }
    }
    Ok(())
}
