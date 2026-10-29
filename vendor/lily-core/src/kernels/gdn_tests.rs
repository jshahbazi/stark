// Original Stark controls for the gated-output wrapper of Lily's pinned GDN.
// The test-only Metal wrapper preserves the original shared raw/barrier path.
// Only synthetic tensors and aggregate results/hashes are used.
use super::*;
use crate::sha256::{Sha256, hex};

const REFERENCE_NAME: &str = "stark_gdn_step_gated_reference";
const REFERENCE_WRAPPER: &str = include_str!("metal/stark_gdn_step_reference.metal");

#[path = "gdn_projection_tests.rs"]
mod projection;

fn next_word(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn random_values(state: &mut u32, len: usize, scale: f32, bias: f32) -> Vec<f32> {
    (0..len).map(|_| (next_word(state) % 4096) as f32 * scale / 4096.0 + bias).collect()
}

fn offset_input(ctx: &MetalContext, values: &[f32], dtype: DType) -> Result<Tensor> {
    let mut padded = vec![0.0; values.len() + 4];
    padded[2..values.len() + 2].copy_from_slice(values);
    let shape = [padded.len()];
    let storage = match dtype {
        DType::BF16 => Tensor::from_f32_as_bf16(ctx, &padded, &shape)?,
        DType::F32 => Tensor::from_f32(ctx, &padded, &shape)?,
        _ => anyhow::bail!("unsupported GDN test input dtype"),
    };
    storage.view(2, &[values.len()])
}

struct GuardedTensor {
    storage: Tensor,
    value: Tensor,
}

impl GuardedTensor {
    fn new(ctx: &MetalContext, values: &[f32], dtype: DType) -> Result<Self> {
        let storage = Tensor::zeros(ctx, &[values.len() + 4], dtype)?;
        storage.write_bytes(&vec![0xa5; storage.byte_len()])?;
        let value = storage.view(2, &[values.len()])?;
        match dtype {
            DType::BF16 => {
                let rounded: Vec<half::bf16> =
                    values.iter().map(|v| half::bf16::from_f32(*v)).collect();
                value.write_bytes(bytemuck::cast_slice(&rounded))?;
            }
            DType::F32 => value.write_bytes(bytemuck::cast_slice(values))?,
            _ => anyhow::bail!("unsupported GDN test output dtype"),
        }
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
            "GDN kernel wrote outside its output view"
        );
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn reference_step(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    source: &str,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    z: &Tensor,
    norm_w: &Tensor,
    out: &Tensor,
    hk: usize,
    gate: GdnGate,
) -> Result<()> {
    let h = a.numel();
    let dim = GDN_HEAD_DIM;
    let k_bytes = hk * dim * qkv.dtype().size();
    let (buffer, offset) = qkv.binding();
    let pipeline = ctx.pipeline(REFERENCE_NAME, source, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            (buffer, offset),
            (buffer, offset + k_bytes),
            (buffer, offset + 2 * k_bytes),
            a.binding(),
            b.binding(),
            a_log.binding(),
            dt_bias.binding(),
            state.binding(),
            z.binding(),
            norm_w.binding(),
            out.binding(),
        ],
        &[
            &(1.0 / (dim as f32).sqrt()).to_ne_bytes(),
            &u32_bytes(h / hk),
            &1e-6f32.to_ne_bytes(),
            &u32_bytes(gate as usize),
        ],
        Grid::Threadgroups { groups: (h, 1, 1), threadgroup: (dim, 1, 1) },
    )
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn local_output_rounding_preserves_gdn_recurrence_exactly() -> Result<()> {
    let ctx = MetalContext::new()?;
    // The name differs because the pipeline cache is keyed by function name
    // and language version. The copied wrapper retains shared raw[tid], its
    // barrier, and its readback; gdn_step_body remains the production helper.
    let reference_source = format!("{SOURCE}\n{REFERENCE_WRAPPER}");
    let mut random = 0x4744_4e31u32;
    let mut reference_hash = Sha256::new();
    let mut candidate_hash = Sha256::new();
    let mut cases = 0;
    let mut checked_bytes = 0;
    for (hk, h) in [(2, 6), (16, 48)] {
        let dim = GDN_HEAD_DIM;
        let k_part = hk * dim;
        let v_part = h * dim;
        let qkv_len = 2 * k_part + v_part;
        for gate in [GdnGate::Sigmoid, GdnGate::Silu] {
            for style in 0..3 {
                let initial = random_values(&mut random, h * dim * dim, 0.5, -0.25);
                let reference_state = GuardedTensor::new(&ctx, &initial, DType::F32)?;
                let candidate_state = GuardedTensor::new(&ctx, &initial, DType::F32)?;
                let reference_out =
                    GuardedTensor::new(&ctx, &vec![0.0; v_part], DType::BF16)?;
                let candidate_out =
                    GuardedTensor::new(&ctx, &vec![0.0; v_part], DType::BF16)?;
                let a_log = offset_input(
                    &ctx,
                    &random_values(&mut random, h, 2.5, -2.0),
                    DType::F32,
                )?;
                let dt_bias = offset_input(
                    &ctx,
                    &random_values(&mut random, h, 1.0, -0.5),
                    DType::BF16,
                )?;
                let norm_w = offset_input(
                    &ctx,
                    &random_values(&mut random, dim, 1.0, 0.5),
                    DType::F32,
                )?;
                for step in 0..16 {
                    let mut qkv_values = random_values(&mut random, qkv_len, 2.0, -1.0);
                    if style == 1 {
                        qkv_values[..2 * k_part].fill(0.0);
                    }
                    let mut a_values = random_values(&mut random, h, 2.0, -1.0);
                    let mut b_values = random_values(&mut random, h, 2.0, -1.0);
                    let mut z_values = random_values(&mut random, v_part, 2.0, -1.0);
                    if style == 2 {
                        for (i, value) in a_values.iter_mut().enumerate() {
                            *value =
                                if (i + step).is_multiple_of(2) { 40.0 } else { -40.0 };
                        }
                        for (i, value) in b_values.iter_mut().enumerate() {
                            *value =
                                if (i + step).is_multiple_of(2) { -40.0 } else { 40.0 };
                        }
                        for (i, value) in z_values.iter_mut().enumerate() {
                            *value =
                                if (i + step).is_multiple_of(2) { 40.0 } else { -40.0 };
                        }
                    }
                    let qkv = offset_input(&ctx, &qkv_values, DType::BF16)?;
                    let a = offset_input(&ctx, &a_values, DType::BF16)?;
                    let b = offset_input(&ctx, &b_values, DType::BF16)?;
                    let z = offset_input(&ctx, &z_values, DType::BF16)?;
                    let pass = ctx.begin_concurrent()?;
                    reference_step(
                        &ctx,
                        &pass,
                        &reference_source,
                        &qkv,
                        &a,
                        &b,
                        &a_log,
                        &dt_bias,
                        &reference_state.value,
                        &z,
                        &norm_w,
                        &reference_out.value,
                        hk,
                        gate,
                    )?;
                    gdn_step_gated_fused(
                        &ctx,
                        &pass,
                        &qkv,
                        &a,
                        &b,
                        &a_log,
                        &dt_bias,
                        &candidate_state.value,
                        &z,
                        &norm_w,
                        &candidate_out.value,
                        1.0 / (dim as f32).sqrt(),
                        hk,
                        1e-6,
                        gate,
                    )?;
                    pass.commit_wait()?;
                    for (field, reference, candidate) in [
                        ("state", &reference_state, &candidate_state),
                        ("output", &reference_out, &candidate_out),
                    ] {
                        reference.check_canaries()?;
                        candidate.check_canaries()?;
                        ensure!(
                            reference.value.raw_bytes() == candidate.value.raw_bytes(),
                            "GDN local-rounding mismatch: field={field} hk={hk} h={h} gate={} style={style} step={step}",
                            gate as usize,
                        );
                        ensure!(
                            candidate.value.to_f32()?.iter().all(|v| v.is_finite()),
                            "GDN local-rounding output/state is nonfinite"
                        );
                        // Compare every state byte after every step. Hash the
                        // outputs and final state once per case to keep this
                        // debug-build control bounded on the 3 MiB state.
                        if field == "output" || step == 15 {
                            reference_hash.update(reference.value.raw_bytes());
                            candidate_hash.update(candidate.value.raw_bytes());
                        }
                        checked_bytes += reference.value.byte_len();
                    }
                }
                cases += 1;
            }
        }
    }
    eprintln!(
        "GDN local-rounding exact controls: cases={cases} steps_per_case=16 checked_bytes={checked_bytes} reference_sha256={} candidate_sha256={}",
        hex(&reference_hash.finish()),
        hex(&candidate_hash.finish())
    );
    Ok(())
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device; diagnostic timing only"]
fn local_output_rounding_streaming_chain_timing() -> Result<()> {
    let ctx = MetalContext::new()?;
    let reference_source = format!("{SOURCE}\n{REFERENCE_WRAPPER}");
    let mut random = 0x4744_4e32u32;
    let (hk, h, dim, layers) = (16, 48, GDN_HEAD_DIM, 36);
    let k_part = hk * dim;
    let v_part = h * dim;
    let qkv = offset_input(
        &ctx,
        &random_values(&mut random, 2 * k_part + v_part, 2.0, -1.0),
        DType::BF16,
    )?;
    let a = offset_input(&ctx, &random_values(&mut random, h, 2.0, -1.0), DType::BF16)?;
    let b = offset_input(&ctx, &random_values(&mut random, h, 2.0, -1.0), DType::BF16)?;
    let z = offset_input(
        &ctx,
        &random_values(&mut random, v_part, 2.0, -1.0),
        DType::BF16,
    )?;
    let a_log =
        offset_input(&ctx, &random_values(&mut random, h, 2.0, -4.0), DType::F32)?;
    let dt_bias =
        offset_input(&ctx, &random_values(&mut random, h, 1.0, -0.5), DType::BF16)?;
    let norm_w =
        offset_input(&ctx, &random_values(&mut random, dim, 1.0, 0.5), DType::F32)?;
    let mut states = [Vec::new(), Vec::new()];
    // Separate 36-layer F32 state sets for the reference and candidate.
    // Each set is 113,246,208 payload bytes, matching the complete model.
    for _ in 0..layers {
        let initial = random_values(&mut random, h * dim * dim, 0.5, -0.25);
        for variant in &mut states {
            variant.push(GuardedTensor::new(&ctx, &initial, DType::F32)?);
        }
    }
    let out = GuardedTensor::new(&ctx, &vec![0.0; v_part], DType::BF16)?;
    let mut times = [Vec::new(), Vec::new()];
    let warmup = std::time::Instant::now();
    let mut round = 0;
    let mut measured_rounds = 0;
    loop {
        let timed = warmup.elapsed() >= std::time::Duration::from_secs(2);
        for order in 0..2 {
            let variant = (round + order) % 2;
            let pass = ctx.begin_concurrent()?;
            for state in &states[variant] {
                if variant == 0 {
                    reference_step(
                        &ctx,
                        &pass,
                        &reference_source,
                        &qkv,
                        &a,
                        &b,
                        &a_log,
                        &dt_bias,
                        &state.value,
                        &z,
                        &norm_w,
                        &out.value,
                        hk,
                        GdnGate::Sigmoid,
                    )?;
                } else {
                    gdn_step_gated_fused(
                        &ctx,
                        &pass,
                        &qkv,
                        &a,
                        &b,
                        &a_log,
                        &dt_bias,
                        &state.value,
                        &z,
                        &norm_w,
                        &out.value,
                        1.0 / (dim as f32).sqrt(),
                        hk,
                        1e-6,
                        GdnGate::Sigmoid,
                    )?;
                }
                pass.level_barrier(&[&state.value, &out.value])?;
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
            "GDN local-rounding streaming chain: local={} layers={layers} state_bytes_per_variant={} samples={} mean_us_per_layer={mean:.6} median_us_per_layer={median:.6}",
            variant != 0,
            layers * h * dim * dim * DType::F32.size(),
            samples.len(),
        );
    }
    Ok(())
}
