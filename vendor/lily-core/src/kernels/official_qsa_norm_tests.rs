// SPDX-License-Identifier: Apache-2.0
// Synthetic zero-gain regression for folded indexer query/block-key norms.
use super::*;

#[test]
fn official_folded_qsa_zero_gain_covers_query_block_keys_and_all_pipelines()
-> Result<()> {
    let ctx = MetalContext::new()?;
    // Compile both ordinary-position and M-RoPE folded entry points. The
    // text execution below additionally exercises the actual dispatch ABI.
    for name in [
        "qsa_prep_q_bf16",
        "qsa_prep_q_mrope_bf16",
        "qsa_block_keys_bf16",
        "qsa_block_keys_mrope_bf16",
    ] {
        let (folded_name, source) = norm_shader(name, 0.0)?;
        ctx.pipeline(folded_name, source, MslVersion::V3_1)?;
    }

    let (rows, heads, d, ratio, rot) = (2, 4, 128, 4, 64);
    let qk = Tensor::from_f32_as_bf16(
        &ctx,
        &vec![1.0; rows * (heads + 1) * d],
        &[rows, (heads + 1) * d],
    )?;
    let keys = Tensor::from_f32_as_bf16(&ctx, &vec![1.0; 8 * d], &[8, d])?;
    let weight = Tensor::zeros(&ctx, &[d], DType::BF16)?;
    for bias in [0.0, 1.0] {
        let query = Tensor::zeros(&ctx, &[rows, heads, d], DType::BF16)?;
        let blocks = Tensor::zeros(&ctx, &[2, d], DType::BF16)?;
        let pass = ctx.begin_concurrent()?;
        qsa_prep_q_with_bias(
            &ctx,
            &pass,
            &qk,
            &weight,
            &query,
            heads,
            rot,
            4,
            10_000_000.0,
            1e-6,
            Rope::Delta(0),
            bias,
        )?;
        qsa_block_keys_with_bias(
            &ctx,
            &pass,
            &keys,
            &weight,
            &blocks,
            ratio,
            1,
            1,
            rot,
            10_000_000.0,
            1e-6,
            Rope::Delta(0),
            bias,
        )?;
        pass.commit_wait()?;
        let query = query.to_f32()?;
        let block = blocks.view(d, &[d])?.to_f32()?;
        if bias == 0.0 {
            ensure!(
                query.iter().all(|&x| x == 0.0),
                "folded QSA query ignored zero gain"
            );
            ensure!(
                block.iter().all(|&x| x == 0.0),
                "folded QSA block key ignored zero gain"
            );
        } else {
            ensure!(
                query.iter().any(|x| x.abs() > 0.25),
                "delta QSA query lost its unit gain"
            );
            ensure!(
                block.iter().any(|x| x.abs() > 0.25),
                "delta QSA block key lost its unit gain"
            );
        }
    }
    Ok(())
}
