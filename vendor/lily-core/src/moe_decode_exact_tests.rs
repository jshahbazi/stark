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
