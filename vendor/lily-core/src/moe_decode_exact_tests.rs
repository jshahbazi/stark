// SPDX-License-Identifier: Apache-2.0
//! Complete MoE controls at production hidden/expert widths.

use super::*;
use crate::weights::{MlpWeights, QuantWeights};

fn word(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn weights(
    ctx: &MetalContext,
    seed: &mut u32,
    bits: usize,
    n: usize,
    k: usize,
) -> Result<QuantWeights> {
    let words = k / (32 / bits);
    let codes: Vec<_> = (0..n * words).map(|_| word(seed)).collect();
    let scales: Vec<_> =
        (0..n * k / 64).map(|_| (word(seed) % 127 + 1) as f32 / 65_536.0).collect();
    let biases: Vec<_> =
        (0..n * k / 64).map(|_| (word(seed) % 257) as f32 / 4096.0 - 0.03125).collect();
    Ok(QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[n, words],
            DType::U32,
        )?,
        scales: Tensor::from_f32_as_bf16(ctx, &scales, &[n, k / 64])?,
        biases: Tensor::from_f32_as_bf16(ctx, &biases, &[n, k / 64])?,
        bits,
        group_size: 64,
    })
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn decode_exact_moe_rows_match_ordinary_bytes() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut seed = 0x39f2_17c1;
    let (h, inter, e, top_k) = (2560, 640, 16, 10);
    let shared_gu = weights(&ctx, &mut seed, 4, 2 * inter, h)?;
    let w = MoeWeights {
        gate: weights(&ctx, &mut seed, 8, e, h)?,
        expert_gate: weights(&ctx, &mut seed, 4, e * inter, h)?,
        expert_up: weights(&ctx, &mut seed, 4, e * inter, h)?,
        expert_down: weights(&ctx, &mut seed, 4, e * h, inter)?,
        shared_gate: weights(&ctx, &mut seed, 8, 1, h)?,
        shared: MlpWeights {
            gate_proj: shared_gu.view_rows(0, inter)?,
            up_proj: shared_gu.view_rows(inter, inter)?,
            gate_up_proj: shared_gu,
            down_proj: weights(&ctx, &mut seed, 4, h, inter)?,
        },
        slot_of: None,
    };
    for renorm in [true, false] {
        let dims = MoeDims {
            hidden: h,
            moe_intermediate: inter,
            num_experts: e,
            top_k,
            norm_topk_prob: renorm,
        };
        let decode = MoeScratch::new(&ctx, &dims)?;
        for m in 1..=4 {
            let values: Vec<_> = (0..m * h)
                .map(|_| (word(&mut seed) % 4097) as f32 / 1024.0 - 2.0)
                .collect();
            let input = Tensor::from_f32_as_bf16(&ctx, &values, &[m, h])?;
            let reference = Tensor::zeros(&ctx, &[m, h], DType::BF16)?;
            let candidate = Tensor::zeros(&ctx, &[m, h], DType::BF16)?;
            let gu = Tensor::zeros(&ctx, &[m, 2 * inter], DType::BF16)?;
            let act = Tensor::zeros(&ctx, &[m, inter], DType::BF16)?;
            let cgu = Tensor::zeros(&ctx, &[m, 2 * inter], DType::BF16)?;
            let cgate = Tensor::zeros(&ctx, &[m, inter], DType::BF16)?;
            let cup = Tensor::zeros(&ctx, &[m, inter], DType::BF16)?;
            let cact = Tensor::zeros(&ctx, &[m, inter], DType::BF16)?;
            let staging = Tensor::zeros(&ctx, &[1], DType::BF16)?;
            let rows = PrefillMoeScratch::new(&ctx, &dims, m)?.chunk(m)?;
            let pass = ctx.begin_concurrent()?;
            for row in 0..m {
                decode_moe(
                    &ctx,
                    &pass,
                    &dims,
                    &w,
                    &DecodeMoeIo {
                        x: &input.view(row * h, &[h])?,
                        out: &reference.view(row * h, &[h])?,
                        mlp_gu: &gu.view(row * 2 * inter, &[2 * inter])?,
                        mlp_gate: &gu.view(row * 2 * inter, &[inter])?,
                        mlp_up: &gu.view(row * 2 * inter + inter, &[inter])?,
                        mlp_act: &act.view(row * inter, &[inter])?,
                        hc_injection: None,
                    },
                    &decode,
                )?;
                pass.level_barrier(&[])?;
            }
            prefill_moe_decode_exact(
                &ctx,
                &pass,
                &dims,
                &w,
                &PrefillMoeIo {
                    x: &input,
                    out: &candidate,
                    stack: &cgu,
                    mlp_gate: &cgate,
                    mlp_up: &cup,
                    mlp_act: &cact,
                    dequant: &staging,
                },
                &rows,
                &decode,
            )?;
            pass.end()?.commit()?.wait()?;
            ensure!(
                reference.raw_bytes() == candidate.raw_bytes(),
                "row-batched MoE differs from ordinary decode"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn batched_moe_injection_matches_ordinary_fused_bytes() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut seed = 0x6a71_8f93;
    let (h, k, e, top_k) = (2560, 640, 16, 10);
    let w = weights(&ctx, &mut seed, 4, e * h + 2, k)?.view_rows(2, e * h)?;
    let mut cases = 0;
    let mut compared_bytes = 0;
    for g in [1, 4, 8] {
        for m in 1..=4 {
            for style in 0..6 {
                let x_values: Vec<_> = (0..m * top_k * k)
                    .map(|_| (word(&mut seed) % 4097) as f32 / 1024.0 - 2.0)
                    .collect();
                let x = Tensor::from_f32_as_bf16(&ctx, &x_values, &[m * top_k, k])?;
                let ids_values: Vec<_> = (0..m * top_k)
                    .map(|slot| ((slot * 3 + style) % e) as u32)
                    .collect();
                let ids = Tensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&ids_values),
                    &[m, top_k],
                    DType::U32,
                )?;
                let scores_values: Vec<_> = (0..m * top_k)
                    .map(|slot| (slot % top_k + 1) as f32 / 97.0)
                    .collect();
                let scores = Tensor::from_f32(&ctx, &scores_values, &[m, top_k])?;
                let shared_values: Vec<_> = (0..m * h)
                    .map(|_| (word(&mut seed) % 4097) as f32 / 1024.0 - 2.0)
                    .collect();
                let shared = Tensor::from_f32_as_bf16(&ctx, &shared_values, &[m, h])?;
                let gate_values: Vec<_> = (0..m)
                    .map(|r| {
                        [-256.0, 256.0, -0.1234, 0.1234, -0.0, 0.0][(r + style) % 6]
                    })
                    .collect();
                let gate = Tensor::from_f32(&ctx, &gate_values, &[m])?;
                let initial_values: Vec<_> = (0..m * g * h)
                    .map(|i| match style {
                        1 => {
                            if i % 2 == 0 {
                                0.0
                            } else {
                                -0.0
                            }
                        }
                        2 => {
                            if i % 2 == 0 {
                                32.0
                            } else {
                                -32.0
                            }
                        }
                        3 => half::bf16::from_bits([1, 0x8001, 0x3f80, 0xbf80][i % 4])
                            .to_f32(),
                        _ => (word(&mut seed) % 4097) as f32 / 1024.0 - 2.0,
                    })
                    .collect();
                let reference_hyper =
                    Tensor::from_f32_as_bf16(&ctx, &initial_values, &[m, g * h])?;
                let hyper_storage =
                    Tensor::zeros(&ctx, &[m * g * h + 16], DType::BF16)?;
                hyper_storage.write_bytes(&vec![0xa5; hyper_storage.byte_len()])?;
                let candidate_hyper = hyper_storage.view(8, &[m, g * h])?;
                candidate_hyper.write_bytes(reference_hyper.raw_bytes())?;
                let inj_values: Vec<_> = (0..m * g)
                    .map(|i| [-256.0, 256.0, -0.5, 0.5, -0.0, 0.0][(i + style) % 6])
                    .collect();
                let inj = Tensor::from_f32_as_bf16(&ctx, &inj_values, &[m, g])?;
                // Odd stream counts have legal row pointers in the shader,
                // but Tensor views require four-byte-aligned offsets.
                let row_inj: Vec<_> = inj_values
                    .chunks_exact(g)
                    .map(|v| Tensor::from_f32_as_bf16(&ctx, v, &[g]))
                    .collect::<Result<_>>()?;
                let reference = Tensor::zeros(&ctx, &[m, h], DType::BF16)?;
                let candidate_storage =
                    Tensor::zeros(&ctx, &[m * h + 16], DType::BF16)?;
                candidate_storage
                    .write_bytes(&vec![0xa5; candidate_storage.byte_len()])?;
                let candidate = candidate_storage.view(8, &[m, h])?;
                let pass = ctx.begin_concurrent()?;
                for row in 0..m {
                    moe::moe_gather_gemv_down_combine_inject(
                        &ctx,
                        &pass,
                        &w,
                        h,
                        &x.view(row * top_k * k, &[top_k, k])?,
                        &ids.view(row * top_k, &[top_k])?,
                        &scores.view(row * top_k, &[top_k])?,
                        Some((&shared.view(row * h, &[h])?, &gate.view(row, &[1])?)),
                        &reference.view(row * h, &[h])?,
                        moe::HcInjection {
                            hyper: &reference_hyper.view(row * g * h, &[g * h])?,
                            logits: &row_inj[row],
                            groups: g,
                        },
                    )?;
                }
                moe::moe_down_combine_inject_decode_rows(
                    &ctx,
                    &pass,
                    &w,
                    h,
                    &x,
                    &ids,
                    &scores,
                    &shared,
                    &gate,
                    &candidate,
                    top_k,
                    moe::HcInjection {
                        hyper: &candidate_hyper,
                        logits: &inj,
                        groups: g,
                    },
                )?;
                pass.commit()?.wait()?;
                ensure!(
                    reference.raw_bytes() == candidate.raw_bytes(),
                    "batched fused MoE branch differs"
                );
                ensure!(
                    reference_hyper.raw_bytes() == candidate_hyper.raw_bytes(),
                    "batched fused HC output differs"
                );
                for storage in [&hyper_storage, &candidate_storage] {
                    let bytes = storage.raw_bytes();
                    ensure!(
                        bytes[..16]
                            .iter()
                            .chain(&bytes[bytes.len() - 16..])
                            .all(|&b| b == 0xa5),
                        "fused MoE wrote outside output view"
                    );
                }
                cases += 1;
                compared_bytes += reference.byte_len() + reference_hyper.byte_len();
            }
        }
    }
    eprintln!(
        "{}",
        serde_json::json!({"scope":"batched_moe_hc_injection_byte_control","cases":cases,"compared_bytes":compared_bytes,"unequal_bytes":0})
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple Silicon; alternating component timings only"]
fn batched_moe_injection_streaming_timing() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut seed = 0x498a_30e1;
    let (h, k, e, top_k, g, layers) = (2560, 640, 10, 10, 4, 48);
    let weight_set: Vec<_> = (0..layers)
        .map(|_| weights(&ctx, &mut seed, 4, e * h, k))
        .collect::<Result<_>>()?;
    let weight_bytes: usize = weight_set
        .iter()
        .map(|w| w.codes.byte_len() + w.scales.byte_len() + w.biases.byte_len())
        .sum();
    for m in [2, 3] {
        let x_values: Vec<_> = (0..m * top_k * k)
            .map(|_| (word(&mut seed) % 4097) as f32 / 1024.0 - 2.0)
            .collect();
        let x = Tensor::from_f32_as_bf16(&ctx, &x_values, &[m * top_k, k])?;
        let ids_values: Vec<_> = (0..m * top_k).map(|i| (i % top_k) as u32).collect();
        let ids = Tensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&ids_values),
            &[m, top_k],
            DType::U32,
        )?;
        let scores = Tensor::from_f32(&ctx, &vec![0.1; m * top_k], &[m, top_k])?;
        let shared = Tensor::from_f32_as_bf16(&ctx, &vec![0.125; m * h], &[m, h])?;
        let gate = Tensor::from_f32(&ctx, &vec![0.1234; m], &[m])?;
        let inj = Tensor::from_f32_as_bf16(&ctx, &vec![0.25; m * g], &[m, g])?;
        let initial =
            Tensor::from_f32_as_bf16(&ctx, &vec![0.5; m * g * h], &[m, g * h])?;
        let reference_hyper = Tensor::zeros(&ctx, &[m, g * h], DType::BF16)?;
        let candidate_hyper = Tensor::zeros(&ctx, &[m, g * h], DType::BF16)?;
        let out = Tensor::zeros(&ctx, &[m, h], DType::BF16)?;
        let run = |fused: bool| -> Result<f64> {
            let hyper = if fused { &candidate_hyper } else { &reference_hyper };
            hyper.write_bytes(initial.raw_bytes())?;
            let pass = ctx.begin_concurrent()?;
            for w in &weight_set {
                if fused {
                    moe::moe_down_combine_inject_decode_rows(
                        &ctx,
                        &pass,
                        w,
                        h,
                        &x,
                        &ids,
                        &scores,
                        &shared,
                        &gate,
                        &out,
                        top_k,
                        moe::HcInjection { hyper, logits: &inj, groups: g },
                    )?;
                } else {
                    moe::moe_down_combine_decode_rows(
                        &ctx, &pass, w, h, &x, &ids, &scores, &shared, &gate, &out,
                        top_k,
                    )?;
                    pass.level_barrier(&[&out])?;
                    crate::kernels::hc::hc_inject_bf16(
                        &ctx, &pass, hyper, &out, &inj, h, g,
                    )?;
                }
                pass.level_barrier(&[&out, hyper])?;
            }
            let done = pass.commit()?.wait_retain()?;
            let t = done.timing()?;
            Ok((t.gpu_end_secs - t.gpu_start_secs) * 1e6)
        };
        run(false)?;
        run(true)?;
        ensure!(
            reference_hyper.raw_bytes() == candidate_hyper.raw_bytes(),
            "timed fused MoE fails byte control"
        );
        let warmup = std::time::Instant::now();
        let mut samples = [Vec::new(), Vec::new()];
        let mut round = 0;
        let mut measured = 0;
        loop {
            let timed = warmup.elapsed() >= std::time::Duration::from_secs(2);
            for order in 0..2 {
                let variant = (round + order) % 2;
                let us = run(variant == 1)?;
                if timed {
                    samples[variant].push(us);
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
        let separate = samples[0].iter().sum::<f64>() / 30.0;
        let fused = samples[1].iter().sum::<f64>() / 30.0;
        eprintln!(
            "{}",
            serde_json::json!({"scope":"synthetic_batched_moe_hc_injection_streaming","rows":m,"layers":layers,"weight_bytes":weight_bytes,"pairs":30,
            "reference_gpu_us":samples[0],"fused_gpu_us":samples[1],"reference_mean_gpu_us":separate,"fused_mean_gpu_us":fused,"speedup":separate/fused,"byte_control":true})
        );
    }
    Ok(())
}
