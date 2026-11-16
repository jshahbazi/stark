// SPDX-License-Identifier: Apache-2.0
//! Exact finite controls for default projection reuse and its fallbacks.
use super::*;

fn word(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn projection_reuse_matches_ordinary_bytes() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut seed = 0x7a53_98f1;
    let mut compared_bytes = 0usize;
    let mut cases = 0usize;
    for (bits, k, n, dtype, group_size, patterns) in [
        (4, 128, 18, DType::BF16, 64, 4),
        (8, 128, 17, DType::BF16, 64, 4),
        (8, 128, 17, DType::F32, 64, 4),
        // Fallbacks: odd Q4 width, Q4/F32 and other admitted group sizes.
        (4, 128, 17, DType::BF16, 64, 4),
        (4, 128, 18, DType::F32, 64, 4),
        (4, 128, 18, DType::BF16, 32, 4),
        (4, 128, 18, DType::BF16, 128, 4),
        (8, 128, 17, DType::BF16, 16, 4),
        (8, 128, 17, DType::F32, 16, 4),
        (8, 128, 17, DType::BF16, 32, 4),
        (8, 128, 17, DType::F32, 32, 4),
        (8, 128, 17, DType::BF16, 128, 4),
        (8, 128, 17, DType::F32, 128, 4),
        (4, 6144, 2560, DType::BF16, 64, 1), // GDN output
        (8, 2560, 1280, DType::BF16, 64, 1), // shared gate/up
        (8, 640, 2560, DType::BF16, 64, 1),  // shared down
        (8, 2560, 13312, DType::BF16, 64, 1), // attention joined QKV
        (8, 6144, 2560, DType::BF16, 64, 1), // attention output
        (8, 2560, 640, DType::BF16, 64, 1),  // attention indexer QK
        (8, 2560, 512, DType::F32, 64, 1),   // router
        (8, 2560, 248320, DType::F32, 64, 1), // vocabulary
    ] {
        let storage_n = n + 2;
        let packed = 32 / bits;
        let codes: Vec<u32> =
            (0..storage_n * k / packed).map(|_| word(&mut seed)).collect();
        let scales: Vec<f32> = (0..storage_n * k / group_size)
            .map(|_| (word(&mut seed) % 127 + 1) as f32 / 65536.0)
            .collect();
        let biases: Vec<f32> = (0..storage_n * k / group_size)
            .map(|_| (word(&mut seed) % 257) as f32 / 4096.0 - 0.03125)
            .collect();
        let w = QuantWeights {
            codes: Tensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&codes),
                &[storage_n, k / packed],
                DType::U32,
            )?,
            scales: Tensor::from_f32_as_bf16(
                &ctx,
                &scales,
                &[storage_n, k / group_size],
            )?,
            biases: Tensor::from_f32_as_bf16(
                &ctx,
                &biases,
                &[storage_n, k / group_size],
            )?,
            group_size,
            bits,
        }
        .view_rows(2, n)?;
        for m in 1..=4 {
            for pattern in 0..patterns {
                let values: Vec<f32> = (0..(m + 2) * k)
                    .map(|i| match pattern {
                        0 => (word(&mut seed) % 4097) as f32 / 1024.0 - 2.0,
                        1 => {
                            if i % 2 == 0 {
                                0.0
                            } else {
                                -0.0
                            }
                        }
                        2 => half::bf16::from_bits([1, 0x8001, 0x3f80, 0xbf80][i % 4])
                            .to_f32(),
                        _ => {
                            if i % 2 == 0 {
                                256.0
                            } else {
                                -256.0
                            }
                        }
                    })
                    .collect();
                let x = Tensor::from_f32_as_bf16(&ctx, &values, &[m + 2, k])?
                    .view(k, &[m, k])?;
                let guard = Tensor::zeros(&ctx, &[m * n + 16], dtype)?;
                guard.write_bytes(&vec![0xa5; guard.byte_len()])?;
                let candidate = guard.view(8, &[m, n])?;
                // Separate row allocations admit odd BF16 widths as well.
                let reference: Vec<_> = (0..m)
                    .map(|_| Tensor::zeros(&ctx, &[n], dtype))
                    .collect::<Result<_>>()?;
                let pass = ctx.begin_concurrent()?;
                for row in 0..m {
                    gemv_quant(
                        &ctx,
                        &pass,
                        &w,
                        &x.view(row * k, &[k])?,
                        &reference[row],
                    )?;
                }
                // Exercise the production dispatch, including all fallbacks.
                gemv_quant_rows_decode_exact(&ctx, &pass, &w, &x, &candidate)?;
                pass.end()?.commit()?.wait()?;
                let candidate_bytes = candidate.raw_bytes();
                let row_bytes = n * dtype.size();
                ensure!(
                    reference.iter().enumerate().all(|(row, expected)| expected
                        .raw_bytes()
                        == &candidate_bytes[row * row_bytes..(row + 1) * row_bytes]),
                    "projection reuse changed exact output"
                );
                let bytes = guard.raw_bytes();
                let pad = 8 * dtype.size();
                ensure!(
                    bytes[..pad]
                        .iter()
                        .chain(&bytes[bytes.len() - pad..])
                        .all(|&b| b == 0xa5),
                    "projection reuse changed guards"
                );
                compared_bytes += m * n * dtype.size();
                cases += 1;
            }
        }
    }
    println!(
        "{{\"projection_reuse\":\"exact\",\"cases\":{cases},\"compared_bytes\":{compared_bytes}}}"
    );
    Ok(())
}
