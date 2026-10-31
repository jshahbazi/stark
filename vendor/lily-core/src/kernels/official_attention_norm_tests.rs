// SPDX-License-Identifier: Apache-2.0
// Synthetic Q/K controls: fused decode must honor the same explicit gain
// convention as the ordinary RMSNorm/RoPE/cache writers, including Q8 KV.
use super::*;
use crate::kernels::norm::rmsnorm_bf16;

#[test]
fn official_folded_and_lily_delta_fused_qk_match_unfused_with_q8_and_bf16_kv()
-> Result<()> {
    let ctx = MetalContext::new()?;
    let (heads, d, capacity, pos, rot) = (2, 256, 64, 37, 64);
    let theta = 10_000_000.0;
    let eps = 1e-6;
    let delta = 5;
    let qg_values: Vec<f32> =
        (0..heads * 2 * d).map(|i| (i as i32 % 61 - 30) as f32 / 16.0).collect();
    let k_values: Vec<f32> =
        (0..heads * d).map(|i| (i as i32 % 47 - 23) as f32 / 16.0).collect();
    let gains: Vec<f32> = (0..d).map(|i| 0.75 + (i % 29) as f32 / 64.0).collect();
    let qg = Tensor::from_f32_as_bf16(&ctx, &qg_values, &[heads, 2 * d])?;
    let k = Tensor::from_f32_as_bf16(&ctx, &k_values, &[heads, d])?;
    let w = Tensor::from_f32_as_bf16(&ctx, &gains, &[d])?;
    for bias in [0.0, 1.0] {
        for format in [KvFormat::Bf16, KvFormat::Q8] {
            let q = Tensor::zeros(&ctx, &[heads, d], DType::BF16)?;
            let gate = Tensor::zeros(&ctx, &[heads, d], DType::BF16)?;
            let qr = Tensor::zeros(&ctx, &[heads, d], DType::BF16)?;
            let gr = Tensor::zeros(&ctx, &[heads, d], DType::BF16)?;
            let kr = Tensor::zeros(&ctx, &[heads, d], DType::BF16)?;
            let cache = KvCache::zeros(&ctx, format, heads, capacity, d)?;
            let reference = KvCache::zeros(&ctx, format, heads, capacity, d)?;
            let pass = ctx.begin_concurrent()?;
            split_q_gate(&ctx, &pass, &qg, &qr, &gr)?;
            rmsnorm_bf16(&ctx, &pass, &k, &w, &kr, eps, bias)?;
            pass.level_barrier(&[&qr, &kr])?;
            rmsnorm_bf16(&ctx, &pass, &qr, &w, &qr, eps, bias)?;
            pass.level_barrier(&[&qr])?;
            rope_neox(&ctx, &pass, &qr, heads, rot, pos, theta, Rope::Delta(delta))?;
            rope_neox(&ctx, &pass, &kr, heads, rot, pos, theta, Rope::Delta(delta))?;
            pass.level_barrier(&[&kr])?;
            scatter_kv(&ctx, &pass, &reference, &kr, pos)?;
            q_norm_rope_split_decode_with_bias(
                &ctx, &pass, &qg, &w, &q, &gate, rot, pos, theta, eps, delta, bias,
            )?;
            k_norm_rope_scatter_decode_with_bias(
                &ctx, &pass, &k, &w, &cache, rot, pos, theta, eps, delta, bias,
            )?;
            pass.commit_wait()?;
            ensure!(
                gate.raw_bytes() == gr.raw_bytes(),
                "fused Q/gate split changed gate values"
            );
            ensure!(
                q.raw_bytes() == qr.raw_bytes(),
                "fused Q norm/RoPE does not honor gain convention"
            );
            ensure!(
                cache.values.raw_bytes() == reference.values.raw_bytes(),
                "fused K cache does not honor gain convention"
            );
            if let (Some(actual), Some(expected)) = (&cache.scales, &reference.scales) {
                ensure!(
                    actual.raw_bytes() == expected.raw_bytes(),
                    "fused K Q8 scales mismatch"
                );
            }
        }
    }
    Ok(())
}
