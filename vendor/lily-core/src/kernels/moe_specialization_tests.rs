// SPDX-License-Identifier: Apache-2.0
//! Original Stark controls for the fixed Flash-Next gate/up and shared-down paths.
//! Synthetic inputs stay in memory; only counts, hashes and timings are emitted.

use super::*;
use crate::kernels::quant;
use crate::sha256::{Sha256, hex};

fn word(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn weight(
    ctx: &MetalContext,
    seed: &mut u32,
    bits: usize,
    n: usize,
    k: usize,
) -> Result<QuantWeights> {
    let rows = n + 2;
    let codes: Vec<_> = (0..rows * k / (32 / bits)).map(|_| word(seed)).collect();
    let scales: Vec<_> =
        (0..rows * k / 64).map(|_| (word(seed) % 127 + 1) as f32 / 65536.0).collect();
    let biases: Vec<_> = (0..rows * k / 64)
        .map(|_| (word(seed) % 257) as f32 / 4096.0 - 0.03125)
        .collect();
    QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[rows, k / (32 / bits)],
            DType::U32,
        )?,
        scales: Tensor::from_f32_as_bf16(ctx, &scales, &[rows, k / 64])?,
        biases: Tensor::from_f32_as_bf16(ctx, &biases, &[rows, k / 64])?,
        bits,
        group_size: 64,
    }
    .view_rows(2, n)
}

fn input(ctx: &MetalContext, seed: &mut u32, n: usize, style: usize) -> Result<Tensor> {
    let values: Vec<_> = (0..n)
        .map(|i| match style {
            1 => {
                if i % 2 == 0 {
                    0.0
                } else {
                    -0.0
                }
            }
            2 => {
                half::bf16::from_bits([0x0001, 0x8001, 0x007f, 0x807f][i % 4]).to_f32()
            }
            3 => [64.0, -64.0, 1.0, -1.0][i % 4],
            _ => (word(seed) % 8193) as f32 / 2048.0 - 2.0,
        })
        .collect();
    let padded = Tensor::zeros(ctx, &[n + 16], DType::BF16)?;
    let value = padded.view(8, &[n])?;
    value.write_bytes(Tensor::from_f32_as_bf16(ctx, &values, &[n])?.raw_bytes())?;
    Ok(value)
}

fn output(ctx: &MetalContext, n: usize) -> Result<(Tensor, Tensor)> {
    let storage = Tensor::zeros(ctx, &[n + 16], DType::BF16)?;
    storage.write_bytes(&vec![0xa5; storage.byte_len()])?;
    let value = storage.view(8, &[n])?;
    Ok((storage, value))
}

fn canaries(storage: &Tensor) -> Result<()> {
    let bytes = storage.raw_bytes();
    ensure!(
        bytes[..16].iter().chain(&bytes[bytes.len() - 16..]).all(|b| *b == 0xa5),
        "specialized projection wrote outside its output view"
    );
    Ok(())
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn flashnext_gate_up_shared_down_match_reference_bytes() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut seed = 0x77a1_3bc2;
    let gate = weight(&ctx, &mut seed, 4, 16 * 640, 2560)?;
    let up = weight(&ctx, &mut seed, 4, 16 * 640, 2560)?;
    let shared = weight(&ctx, &mut seed, 8, 2560, 640)?;
    let mut reference_hash = Sha256::new();
    let mut candidate_hash = Sha256::new();
    let mut compared = 0;
    for style in 0..4 {
        for (slots, repeated) in [(1, false), (10, false), (16, false), (10, true)] {
            // Permuted/duplicate routes and offset views exercise real gather addressing.
            let ids: Vec<_> = (0..slots)
                .map(|i| ((i * 3 + style) % if repeated { 4 } else { 16 }) as u32)
                .collect();
            let id_storage = Tensor::zeros(&ctx, &[slots + 8], DType::U32)?;
            let indices = id_storage.view(4, &[slots])?;
            indices.write_bytes(bytemuck::cast_slice(&ids))?;
            let x = input(&ctx, &mut seed, 2560, style)?;
            let act = input(&ctx, &mut seed, 640, style)?;
            let (rs, reference) = output(&ctx, slots * 640)?;
            let (cs, candidate) = output(&ctx, slots * 640)?;
            let (rds, reference_down) = output(&ctx, 2560)?;
            let (cds, candidate_down) = output(&ctx, 2560)?;
            let pass = ctx.begin_concurrent()?;
            moe_gather_gemv_gate_up_named(
                &ctx,
                &pass,
                "moe_gather_gemv_q4_gate_up",
                &gate,
                &up,
                640,
                &x,
                &indices,
                &reference,
            )?;
            quant::gemv_quant(&ctx, &pass, &shared, &act, &reference_down)?;
            pass.level_barrier(&[&reference, &reference_down])?;
            quant::gemv_quant(&ctx, &pass, &shared, &act, &candidate_down)?;
            moe_gather_gemv_gate_up(
                &ctx, &pass, &gate, &up, 640, &x, &indices, &candidate,
            )?;
            pass.end()?.commit()?.wait()?;
            for storage in [&rs, &cs, &rds, &cds] {
                canaries(storage)?;
            }
            for (reference, candidate) in
                [(&reference, &candidate), (&reference_down, &candidate_down)]
            {
                ensure!(
                    reference.raw_bytes() == candidate.raw_bytes(),
                    "fixed projection numerical mismatch"
                );
                reference_hash.update(reference.raw_bytes());
                candidate_hash.update(candidate.raw_bytes());
                compared += reference.numel();
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({"cases":16,"compared_bf16_values":compared,
        "reference_sha256":hex(&reference_hash.finish()),"candidate_sha256":hex(&candidate_hash.finish())})
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple Silicon; alternating streaming component timing"]
fn flashnext_gate_up_shared_down_streaming_timing() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut seed = 0x2ab3_6d82;
    // Rotate sixteen independent banks instead of repeatedly measuring one
    // tiny-weight fixture. These are synthetic, not model weights.
    let banks = (0..16)
        .map(|_| {
            Ok((
                weight(&ctx, &mut seed, 4, 16 * 640, 2560)?,
                weight(&ctx, &mut seed, 4, 16 * 640, 2560)?,
                weight(&ctx, &mut seed, 8, 2560, 640)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let x = input(&ctx, &mut seed, 2560, 0)?;
    let act = input(&ctx, &mut seed, 640, 0)?;
    let ids: Vec<u32> = (0..10).map(|i| (i * 3 % 16) as u32).collect();
    let indices =
        Tensor::from_bytes(&ctx, bytemuck::cast_slice(&ids), &[10], DType::U32)?;
    let gate_out = Tensor::zeros(&ctx, &[10 * 640], DType::BF16)?;
    let down_out = Tensor::zeros(&ctx, &[2560], DType::BF16)?;
    let measure = |specialized: bool, shared: bool| -> Result<f64> {
        let pass = ctx.begin_concurrent()?;
        for (gate, up, down) in &banks {
            if shared && specialized {
                quant::gemv_quant(&ctx, &pass, down, &act, &down_out)?;
            }
            moe_gather_gemv_gate_up_named(
                &ctx,
                &pass,
                if specialized {
                    "stark_moe_gate_up_h2560_i640_g64"
                } else {
                    "moe_gather_gemv_q4_gate_up"
                },
                gate,
                up,
                640,
                &x,
                &indices,
                &gate_out,
            )?;
            if shared && !specialized {
                quant::gemv_quant(&ctx, &pass, down, &act, &down_out)?;
            }
            pass.level_barrier(&[&gate_out, &down_out])?;
        }
        let completed = pass.commit()?.wait_retain()?;
        let timing = completed.timing()?;
        let seconds = timing.gpu_end_secs - timing.gpu_start_secs;
        ensure!(seconds.is_finite() && seconds > 0.0, "component timing unavailable");
        Ok(seconds / banks.len() as f64)
    };
    let mut results = Vec::new();
    for shared in [false, true] {
        for _ in 0..4 {
            measure(false, shared)?;
            measure(true, shared)?;
        }
        let mut pairs = Vec::new();
        for block in 0..16 {
            let candidate_first = [false, true, true, false][block % 4];
            let first = measure(candidate_first, shared)?;
            let second = measure(!candidate_first, shared)?;
            let (reference, candidate) =
                if candidate_first { (second, first) } else { (first, second) };
            pairs.push(serde_json::json!({"reference_seconds_per_layer":reference,
                "candidate_seconds_per_layer":candidate,"speedup":reference/candidate}));
        }
        results.push(serde_json::json!({"includes_shared_down":shared,"pairs":pairs}));
    }
    println!(
        "{}",
        serde_json::json!({"scope":"synthetic_streaming_component_only",
        "banks":banks.len(),"results":results})
    );
    Ok(())
}
