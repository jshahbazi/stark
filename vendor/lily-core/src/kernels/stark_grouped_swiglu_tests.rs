// SPDX-License-Identifier: Apache-2.0
use super::*;
use crate::kernels::elementwise::silu_mul_bf16;

fn next(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn bank(
    ctx: &MetalContext,
    seed: &mut u32,
    k: usize,
    n: usize,
) -> Result<QuantWeights> {
    let codes: Vec<_> = (0..2 * n * k / 8).map(|_| next(seed)).collect();
    let scales: Vec<_> =
        (0..2 * n * k / 64).map(|_| (next(seed) % 127 + 1) as f32 / 65536.0).collect();
    let biases: Vec<_> = (0..2 * n * k / 64)
        .map(|_| (next(seed) % 257) as f32 / 4096.0 - 0.03125)
        .collect();
    Ok(QuantWeights {
        codes: Tensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&codes),
            &[2 * n, k / 8],
            DType::U32,
        )?,
        scales: Tensor::from_f32_as_bf16(ctx, &scales, &[2 * n, k / 64])?,
        biases: Tensor::from_f32_as_bf16(ctx, &biases, &[2 * n, k / 64])?,
        group_size: 64,
        bits: 4,
    })
}

#[test]
#[ignore = "requires an Apple GPU with Metal 4 tensor operations"]
fn grouped_swiglu_matches_separate_projections() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut seed = 0x7a53_98f1;
    let mut compared_bytes = 0;
    let mut cases = 0;
    for (k, n) in [(64, 64), (192, 128), (2560, 640)] {
        let gate = bank(&ctx, &mut seed, k, n)?;
        let up = bank(&ctx, &mut seed, k, n)?;
        for tile in [MoeTile::T32Sg4, MoeTile::T64Sg4] {
            // More than one K step, both expert offsets, each tail height,
            // valid blocks beyond a sentinel, and untouched output guards.
            let spans = [tile.rows(), tile.rows() / 2, tile.rows() / 4 - 1];
            let rows = spans.iter().sum::<usize>();
            let input: Vec<_> = (0..rows * k)
                .map(|_| (next(&mut seed) % 4097) as f32 / 1024.0 - 2.0)
                .collect();
            let x = Tensor::from_f32_as_bf16(&ctx, &input, &[rows, k])?;
            let mut map = vec![[0u32; 4]];
            let mut start = 0;
            for (span_index, span) in spans.into_iter().enumerate() {
                for col in (0..n).step_by(64) {
                    map.push([
                        start as u32,
                        (span_index % 2 * n + col) as u32,
                        col as u32,
                        (start + span) as u32,
                    ]);
                }
                start += span;
            }
            let blocks = Tensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&map),
                &[map.len(), 4],
                DType::U32,
            )?;
            let eg = Tensor::zeros(&ctx, &[rows, n], DType::BF16)?;
            let eu = Tensor::zeros(&ctx, &[rows, n], DType::BF16)?;
            let reference = Tensor::zeros(&ctx, &[rows, n], DType::BF16)?;
            let guard = Tensor::zeros(&ctx, &[rows * n + 16], DType::BF16)?;
            guard.write_bytes(&vec![0xa5; guard.byte_len()])?;
            let candidate = guard.view(8, &[rows, n])?;
            let pass = ctx.begin_concurrent()?;
            gemm_q4_grouped_nt(&ctx, &pass, &x, &gate, &eg, &blocks, map.len(), tile)?;
            gemm_q4_grouped_nt(&ctx, &pass, &x, &up, &eu, &blocks, map.len(), tile)?;
            pass.level_barrier(&[&eg, &eu])?;
            silu_mul_bf16(&ctx, &pass, &eg, &eu, &reference)?;
            gemm_q4_grouped_swiglu(
                &ctx,
                &pass,
                &x,
                &gate,
                &up,
                &candidate,
                &blocks,
                map.len(),
                tile,
            )?;
            pass.end()?.commit()?.wait()?;
            ensure!(
                candidate.raw_bytes() == reference.raw_bytes(),
                "grouped SwiGLU changed BF16 values at {k}×{n}, {tile:?}"
            );
            let bytes = guard.raw_bytes();
            ensure!(
                bytes[..16]
                    .iter()
                    .chain(&bytes[bytes.len() - 16..])
                    .all(|&b| b == 0xa5),
                "grouped SwiGLU changed output guards"
            );
            cases += 1;
            compared_bytes += rows * n * DType::BF16.size();
        }
    }
    println!(
        "{{\"grouped_swiglu\":\"exact\",\"cases\":{cases},\"compared_bytes\":{compared_bytes}}}"
    );
    Ok(())
}
