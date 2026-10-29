// Original Stark controls for down/combine and HC injection fusion.
// Synthetic tensors only; reports contain counts, hashes and timing aggregates.
use super::*;
use crate::kernels::hc::hc_inject_bf16;
use crate::sha256::{Sha256, hex};

fn next_word(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn random_values(state: &mut u32, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| (next_word(state) % 4096) as f32 * scale / 4096.0 - scale * 0.5)
        .collect()
}

fn down_weight(
    ctx: &MetalContext,
    state: &mut u32,
    experts: usize,
    h: usize,
    k: usize,
    gs: usize,
) -> Result<QuantWeights> {
    let rows = experts * h;
    let allocated = rows + 2;
    let codes: Vec<u32> = (0..allocated * k / 8).map(|_| next_word(state)).collect();
    let scales: Vec<f32> = random_values(state, allocated * k / gs, 0.02)
        .into_iter()
        .map(|v| v.abs() + 0.001)
        .collect();
    let biases = random_values(state, allocated * k / gs, 0.125);
    QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[allocated, k / 8],
            DType::U32,
        )?,
        scales: Tensor::from_f32_as_bf16(ctx, &scales, &[allocated, k / gs])?,
        biases: Tensor::from_f32_as_bf16(ctx, &biases, &[allocated, k / gs])?,
        group_size: gs,
        bits: 4,
    }
    .view_rows(2, rows)
}

struct Guarded {
    storage: Tensor,
    value: Tensor,
}

impl Guarded {
    fn new(ctx: &MetalContext, values: &[f32], dtype: DType) -> Result<Self> {
        let storage = Tensor::zeros(ctx, &[values.len() + 16], dtype)?;
        storage.write_bytes(&vec![0xa5; storage.byte_len()])?;
        let value = storage.view(8, &[values.len()])?;
        let contents = match dtype {
            DType::BF16 => Tensor::from_f32_as_bf16(ctx, values, &[values.len()])?,
            DType::F32 => Tensor::from_f32(ctx, values, &[values.len()])?,
            _ => anyhow::bail!("unsupported guarded fixture dtype"),
        };
        value.write_bytes(contents.raw_bytes())?;
        Ok(Self { storage, value })
    }

    fn check_canaries(&self) -> Result<()> {
        let bytes = self.storage.raw_bytes();
        let pad = 8 * self.storage.dtype().size();
        ensure!(
            bytes[..pad].iter().chain(&bytes[bytes.len() - pad..]).all(|b| *b == 0xa5),
            "MoE injection wrote outside the output view"
        );
        Ok(())
    }
}

fn indices(ctx: &MetalContext, ids: &[u32]) -> Result<Tensor> {
    let mut padded = vec![0u32; ids.len() + 8];
    padded[4..4 + ids.len()].copy_from_slice(ids);
    Tensor::from_bytes(ctx, bytemuck::cast_slice(&padded), &[padded.len()], DType::U32)?
        .view(4, &[ids.len()])
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn fused_down_injection_is_bit_identical() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut state = 0x4d49_4e31u32;
    let mut reference_hash = Sha256::new();
    let mut candidate_hash = Sha256::new();
    let mut cases = 0;
    let mut checked_bytes = 0;
    for (h, k, g, slots, gs, shared) in [
        (2560, 640, 4, 10, 64, true),
        (2560, 640, 4, 10, 64, false),
        (18, 128, 3, 1, 32, true),
        (34, 128, 8, 16, 128, true),
        (16, 64, 1, 3, 64, false),
        (258, 192, 4, 7, 64, true),
    ] {
        let experts = slots.max(4);
        let w = down_weight(&ctx, &mut state, experts, h, k, gs)?;
        for style in 0..6 {
            // Duplicate and permuted routed experts are valid inputs.
            let ids: Vec<u32> =
                (0..slots).map(|i| ((i * 3 + style) % experts) as u32).collect();
            let ids = indices(&ctx, &ids)?;
            let score_values: Vec<f32> =
                (0..slots).map(|i| (i + 1) as f32 / (slots * slots) as f32).collect();
            let scores = Guarded::new(&ctx, &score_values, DType::F32)?;
            let gate = Guarded::new(
                &ctx,
                &[[-256.0, 256.0, -2.0, -0.5, 0.5, 2.0][style]],
                DType::F32,
            )?;
            let gate_patterns = [
                -256.0,
                256.0,
                -2.0,
                -0.5,
                0.5,
                2.0,
                0.0,
                -0.0,
                half::bf16::from_bits(0x0001).to_f32(),
                half::bf16::from_bits(0x8001).to_f32(),
                0.1234,
                -0.1234,
            ];
            let inj_values: Vec<f32> = (0..g)
                .map(|i| gate_patterns[(i + 2 * style) % gate_patterns.len()])
                .collect();
            let inj = Guarded::new(&ctx, &inj_values, DType::BF16)?;
            let initial = random_values(&mut state, g * h, 4.0);
            let reference_hyper = Guarded::new(&ctx, &initial, DType::BF16)?;
            let candidate_hyper = Guarded::new(&ctx, &initial, DType::BF16)?;
            let reference_out = Guarded::new(&ctx, &vec![0.0; h], DType::BF16)?;
            let candidate_out = Guarded::new(&ctx, &vec![0.0; h], DType::BF16)?;
            for step in 0..16 {
                let mut values = random_values(&mut state, slots * k, 4.0);
                for (i, v) in values.iter_mut().enumerate() {
                    match style {
                        1 => *v = if i.is_multiple_of(2) { 0.0 } else { -0.0 },
                        2 => {
                            *v = half::bf16::from_bits(
                                [0x0001, 0x8001, 0x3f80, 0xbf80][(i + step) % 4],
                            )
                            .to_f32()
                        }
                        3 => *v = [1e10, -1e10, 32.0, -32.0][(i + step) % 4],
                        _ => {}
                    }
                }
                let x = Guarded::new(&ctx, &values, DType::BF16)?;
                let shared_out = Guarded::new(
                    &ctx,
                    &random_values(&mut state, h, 4.0),
                    DType::BF16,
                )?;
                let shared_args = shared.then_some((&shared_out.value, &gate.value));
                let pass = ctx.begin_concurrent()?;
                moe_gather_gemv_down_combine(
                    &ctx,
                    &pass,
                    &w,
                    h,
                    &x.value,
                    &ids,
                    &scores.value,
                    shared_args,
                    &reference_out.value,
                )?;
                pass.level_barrier(&[&reference_out.value])?;
                hc_inject_bf16(
                    &ctx,
                    &pass,
                    &reference_hyper.value,
                    &reference_out.value,
                    &inj.value,
                    h,
                    g,
                )?;
                pass.level_barrier(&[&reference_hyper.value])?;
                moe_gather_gemv_down_combine_inject(
                    &ctx,
                    &pass,
                    &w,
                    h,
                    &x.value,
                    &ids,
                    &scores.value,
                    shared_args,
                    &candidate_out.value,
                    HcInjection {
                        hyper: &candidate_hyper.value,
                        logits: &inj.value,
                        groups: g,
                    },
                )?;
                pass.commit_wait()?;
                for (field, reference, candidate) in [
                    ("branch", &reference_out, &candidate_out),
                    ("hyper", &reference_hyper, &candidate_hyper),
                ] {
                    reference.check_canaries()?;
                    candidate.check_canaries()?;
                    ensure!(
                        reference.value.raw_bytes() == candidate.value.raw_bytes(),
                        "MoE injection mismatch: field={field} h={h} k={k} g={g} slots={slots} style={style} step={step}"
                    );
                    ensure!(
                        candidate.value.to_f32()?.iter().all(|v| v.is_finite()),
                        "MoE injection output is nonfinite"
                    );
                    reference_hash.update(reference.value.raw_bytes());
                    candidate_hash.update(candidate.value.raw_bytes());
                    checked_bytes += reference.value.byte_len();
                }
                x.check_canaries()?;
                shared_out.check_canaries()?;
            }
            scores.check_canaries()?;
            gate.check_canaries()?;
            inj.check_canaries()?;
            cases += 1;
        }
    }
    eprintln!(
        "MoE injection exact controls: cases={cases} steps_per_case=16 checked_bytes={checked_bytes} reference_sha256={} candidate_sha256={}",
        hex(&reference_hash.finish()),
        hex(&candidate_hash.finish())
    );
    Ok(())
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device; diagnostic timing only"]
fn fused_down_injection_streaming_chain_timing() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut state = 0x4d49_4e32u32;
    let (h, k, g, slots, gs, layers) = (2560, 640, 4, 10, 64, 48);
    let x =
        Guarded::new(&ctx, &random_values(&mut state, slots * k, 4.0), DType::BF16)?;
    let ids = indices(&ctx, &(0..slots as u32).collect::<Vec<_>>())?;
    let scores = Guarded::new(&ctx, &vec![1.0 / slots as f32; slots], DType::F32)?;
    let shared_out =
        Guarded::new(&ctx, &random_values(&mut state, h, 4.0), DType::BF16)?;
    let gate = Guarded::new(&ctx, &[0.5], DType::F32)?;
    let inj = Guarded::new(&ctx, &[-0.5, 0.0, 0.5, 1.0], DType::BF16)?;
    let out = Guarded::new(&ctx, &vec![0.0; h], DType::BF16)?;
    let mut weights = Vec::new();
    let mut hypers = [Vec::new(), Vec::new()];
    let mut weight_bytes = 0;
    for _ in 0..layers {
        let w = down_weight(&ctx, &mut state, slots, h, k, gs)?;
        weight_bytes += w.codes.byte_len() + w.scales.byte_len() + w.biases.byte_len();
        weights.push(w);
        let initial = random_values(&mut state, g * h, 4.0);
        for variant in &mut hypers {
            variant.push(Guarded::new(&ctx, &initial, DType::BF16)?);
        }
    }
    let warmup = std::time::Instant::now();
    let mut times = [Vec::new(), Vec::new()];
    let mut round = 0;
    let mut measured = 0;
    loop {
        let timed = warmup.elapsed() >= std::time::Duration::from_secs(2);
        for order in 0..2 {
            let variant = (round + order) % 2;
            let pass = ctx.begin_concurrent()?;
            for (w, hyper) in weights.iter().zip(&hypers[variant]) {
                let shared = Some((&shared_out.value, &gate.value));
                if variant == 0 {
                    moe_gather_gemv_down_combine(
                        &ctx,
                        &pass,
                        w,
                        h,
                        &x.value,
                        &ids,
                        &scores.value,
                        shared,
                        &out.value,
                    )?;
                    pass.level_barrier(&[&out.value])?;
                    hc_inject_bf16(
                        &ctx,
                        &pass,
                        &hyper.value,
                        &out.value,
                        &inj.value,
                        h,
                        g,
                    )?;
                } else {
                    moe_gather_gemv_down_combine_inject(
                        &ctx,
                        &pass,
                        w,
                        h,
                        &x.value,
                        &ids,
                        &scores.value,
                        shared,
                        &out.value,
                        HcInjection {
                            hyper: &hyper.value,
                            logits: &inj.value,
                            groups: g,
                        },
                    )?;
                }
                pass.level_barrier(&[&out.value, &hyper.value])?;
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
            measured += 1;
            if measured == 16 {
                break;
            }
        }
    }
    for (variant, samples) in times.iter_mut().enumerate() {
        // These are scalar synthetic-kernel timings, never model values.
        // Preserve acquisition order before sorting for the median.
        eprintln!(
            "MoE injection ordered GPU timings: fused={} microseconds_per_layer={samples:?}",
            variant != 0
        );
        samples.sort_unstable_by(f64::total_cmp);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let median = (samples[7] + samples[8]) * 0.5;
        eprintln!(
            "MoE injection streaming chain: fused={} layers={layers} weight_bytes={weight_bytes} samples={} mean_us_per_layer={mean:.6} median_us_per_layer={median:.6}",
            variant != 0,
            samples.len()
        );
    }
    Ok(())
}
