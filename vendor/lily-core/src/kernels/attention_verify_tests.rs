// SPDX-License-Identifier: Apache-2.0
//! Synthetic exact controls for verifier query/key preparation.

use super::*;

fn values(n: usize, seed: &mut u32, style: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 17;
            *seed ^= *seed << 5;
            match style {
                1 => 0.0,
                2 => {
                    if i % 2 == 0 {
                        32.0
                    } else {
                        -32.0
                    }
                }
                3 => {
                    half::bf16::from_bits([0, 0x8000, 1, 0x8001, 0x3f80, 0xbf80][i % 6])
                        .to_f32()
                }
                _ => ((*seed % 8192) as f32 - 4096.0) / 1024.0,
            }
        })
        .collect()
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn verifier_query_key_rows_match_decode_bytes() -> Result<()> {
    let ctx = MetalContext::new()?;
    let mut seed = 0xa7f1_57e9;
    let mut cases = 0usize;
    let mut checked_bytes = 0usize;
    let d = 256;
    let nq = 16;
    let kvh = 2;
    let norm = Tensor::from_f32_as_bf16(&ctx, &values(d, &mut seed, 0), &[d])?;
    for m in 1..=4 {
        for pos in [0usize, 2047, 4095, 16380] {
            for style in 0..4 {
                let rot = if style % 2 == 0 { 64 } else { 256 };
                let delta = if style % 2 == 0 { 0 } else { 7 };
                let qg = Tensor::from_f32_as_bf16(
                    &ctx,
                    &values(m * nq * d * 2, &mut seed, style),
                    &[m, nq, 2 * d],
                )?;
                let k = Tensor::from_f32_as_bf16(
                    &ctx,
                    &values(m * kvh * d, &mut seed, style),
                    &[m, kvh, d],
                )?;
                for format in [KvFormat::Bf16, KvFormat::Q8] {
                    let q_ref_store =
                        Tensor::zeros(&ctx, &[m * nq * d + 4], DType::BF16)?;
                    let q_new_store =
                        Tensor::zeros(&ctx, &[m * nq * d + 4], DType::BF16)?;
                    let g_ref_store =
                        Tensor::zeros(&ctx, &[m * nq * d + 4], DType::BF16)?;
                    let g_new_store =
                        Tensor::zeros(&ctx, &[m * nq * d + 4], DType::BF16)?;
                    for t in [&q_ref_store, &q_new_store, &g_ref_store, &g_new_store] {
                        t.write_bytes(&vec![0xa5; t.byte_len()])?;
                    }
                    let q_ref = q_ref_store.view(2, &[m, nq, d])?;
                    let q_new = q_new_store.view(2, &[m, nq, d])?;
                    let g_ref = g_ref_store.view(2, &[m, nq, d])?;
                    let g_new = g_new_store.view(2, &[m, nq, d])?;
                    let capacity = pos + m + 3;
                    let c_ref = KvCache::zeros(&ctx, format, kvh, capacity, d)?;
                    let c_new = KvCache::zeros(&ctx, format, kvh, capacity, d)?;
                    for t in c_ref.tensors().chain(c_new.tensors()) {
                        t.write_bytes(&vec![0x3d; t.byte_len()])?;
                    }
                    let pass = ctx.begin_concurrent()?;
                    for row in 0..m {
                        q_norm_rope_split_decode(
                            &ctx,
                            &pass,
                            &qg.view(row * nq * d * 2, &[nq, 2 * d])?,
                            &norm,
                            &q_ref.view(row * nq * d, &[nq, d])?,
                            &g_ref.view(row * nq * d, &[nq, d])?,
                            rot,
                            pos + row,
                            1e7,
                            1e-6,
                            delta,
                        )?;
                        k_norm_rope_scatter_decode(
                            &ctx,
                            &pass,
                            &k.view(row * kvh * d, &[kvh, d])?,
                            &norm,
                            &c_ref,
                            rot,
                            pos + row,
                            1e7,
                            1e-6,
                            delta,
                        )?;
                    }
                    q_norm_rope_split_decode_rows(
                        &ctx, &pass, &qg, &norm, &q_new, &g_new, rot, pos, 1e7, 1e-6,
                        delta,
                    )?;
                    k_norm_rope_scatter_decode_rows(
                        &ctx, &pass, &k, &norm, &c_new, rot, pos, 1e7, 1e-6, delta,
                    )?;
                    pass.end()?.commit()?.wait()?;
                    for (a, b) in
                        [(&q_ref_store, &q_new_store), (&g_ref_store, &g_new_store)]
                    {
                        ensure!(
                            a.raw_bytes() == b.raw_bytes(),
                            "query/gate row preparation mismatch"
                        );
                        checked_bytes += a.byte_len();
                    }
                    for (a, b) in c_ref.tensors().zip(c_new.tensors()) {
                        ensure!(
                            a.raw_bytes() == b.raw_bytes(),
                            "key cache row preparation mismatch"
                        );
                        checked_bytes += a.byte_len();
                    }
                    cases += 1;
                }
            }
        }
    }
    println!(
        "{{\"attention_verify_cases\":{cases},\"compared_bytes\":{checked_bytes},\"unequal_bytes\":0}}"
    );
    Ok(())
}
