// Copyright (c) 2023-2024 Apple Inc.
// Exact affine64 encode oracle adapted from MLX quantized.h 64ea011c.
// Synthetic values only; no checkpoint or request tensors are persisted.
use super::*;
use half::bf16;

fn encode_group(values: &[f32]) -> ([u8; 64], [u16; 2]) {
    assert_eq!(values.len(), 64);
    let values: Vec<f32> = values.iter().map(|&x| bf16::from_f32(x).to_f32()).collect();
    let low = values.iter().copied().fold(f32::INFINITY, f32::min);
    let high = values.iter().copied().fold(0.0f32, f32::max);
    let mut scale = ((high - low) / 255.0f32).max(1e-7f32);
    let side = low.abs() > high.abs();
    scale = if side { scale } else { -scale };
    let edge = if side { low } else { high };
    let q0 = (edge / scale).round();
    let bias = if q0 == 0.0 { 0.0 } else { edge };
    if q0 != 0.0 {
        scale = edge / q0;
    }
    let mut codes = [0u8; 64];
    for (to, value) in codes.iter_mut().zip(values) {
        *to = ((value - bias) / scale).round().min(255.0) as u8;
    }
    (codes, [bf16::from_f32(scale).to_bits(), bf16::from_f32(bias).to_bits()])
}

fn exact_attention_bytes(
    actual: &Tensor,
    expected: &Tensor,
    category: &str,
) -> Result<()> {
    let (a, b) = (actual.raw_bytes(), expected.raw_bytes());
    ensure!(a.len() == b.len(), "{category}: length mismatch");
    let mismatches =
        a.chunks_exact(2).zip(b.chunks_exact(2)).filter(|(x, y)| x != y).count();
    let max_abs = actual
        .to_f32()?
        .iter()
        .zip(expected.to_f32()?)
        .map(|(x, y)| (*x - y).abs())
        .fold(0.0f32, f32::max);
    ensure!(a == b, "{category}: bf16_mismatches={mismatches}, max_abs={max_abs}");
    Ok(())
}

#[test]
fn affine64_layout_and_edge_cases() {
    assert_eq!(KvFormat::Q8Affine64.row_bytes(256), 272);
    assert_eq!(KvFormat::Q8Affine64.row_bytes(256), KvFormat::Q8.row_bytes(256));
    for value in [0.0f32, -0.0, 2.0, -2.0, 1e-9, -1e-9] {
        let (codes, meta) = encode_group(&[value; 64]);
        assert!(codes.iter().all(|&x| x == 0));
        assert!(bf16::from_bits(meta[0]).to_f32().is_finite());
    }
    let (_, positive) = encode_group(&[2.0; 64]);
    let (_, negative) = encode_group(&[-2.0; 64]);
    assert!(bf16::from_bits(positive[0]).to_f32() < 0.0);
    assert!(bf16::from_bits(negative[0]).to_f32() > 0.0);
    assert_eq!(positive[1], bf16::from_f32(2.0).to_bits());
    assert_eq!(negative[1], bf16::from_f32(-2.0).to_bits());
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn affine64_scatter_codes_and_metadata_match_cpu_oracle() -> Result<()> {
    let ctx = MetalContext::new()?;
    let (heads, rows, d, capacity, pos) = (2, 3, 256, 64, 37);
    let mut input = vec![0.0f32; heads * rows * d];
    for (i, value) in input.iter_mut().enumerate() {
        let group = i / 64;
        *value = match group % 8 {
            0 => 0.0,
            1 => 2.0,
            2 => -2.0,
            3 => 1e-9,
            4 => -1e-9,
            _ => ((i * 37 % 257) as f32 - 128.0) / (11 + group % 9) as f32,
        };
    }
    let source = Tensor::from_f32_as_bf16(&ctx, &input, &[rows, heads, d])?;
    let cache = KvCache::zeros(&ctx, KvFormat::Q8Affine64, heads, capacity, d)?;
    assert_eq!(cache.kv().format()?, KvFormat::Q8Affine64);
    let pass = ctx.begin_concurrent()?;
    scatter_kv(&ctx, &pass, &cache, &source, pos)?;
    pass.commit_wait()?;
    let mut expected_codes = vec![0u8; heads * capacity * d];
    let mut expected_metadata = vec![0u16; heads * capacity * (d / 32)];
    for m in 0..rows {
        for h in 0..heads {
            for g in 0..d / 64 {
                let begin = (m * heads + h) * d + 64 * g;
                let (codes, metadata) = encode_group(&input[begin..begin + 64]);
                let cache_row = h * capacity + pos + m;
                expected_codes[cache_row * d + g * 64..cache_row * d + (g + 1) * 64]
                    .copy_from_slice(&codes);
                expected_metadata
                    [cache_row * (d / 32) + 2 * g..cache_row * (d / 32) + 2 * g + 2]
                    .copy_from_slice(&metadata);
            }
        }
    }
    ensure!(
        cache.values.raw_bytes() == expected_codes,
        "affine64 code or unwritten-row mismatch"
    );
    ensure!(
        cache.scales.as_ref().unwrap().raw_bytes()
            == bytemuck::cast_slice::<u16, u8>(&expected_metadata),
        "affine64 metadata or unwritten-row mismatch"
    );
    Ok(())
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn affine64_dequant_matches_pinned_mlx_single_round_rule() -> Result<()> {
    let ctx = MetalContext::new()?;
    let (heads, rows, d, capacity, pos) = (2, 3, 256, 64, 37);
    let input: Vec<f32> = (0..heads * rows * d)
        .map(|i| ((i * 31 % 251) as f32 - 125.0) / (13 + i / 64 % 7) as f32)
        .collect();
    let source = Tensor::from_f32_as_bf16(&ctx, &input, &[rows, heads, d])?;
    let cache = KvCache::zeros(&ctx, KvFormat::Q8Affine64, heads, capacity, d)?;
    let mut expected = vec![bf16::from_f32(321.0); heads * capacity * d];
    for h in 0..heads {
        for token in 0..pos + rows {
            expected[(h * capacity + token) * d..(h * capacity + token + 1) * d]
                .fill(bf16::from_f32(0.0));
        }
    }
    for m in 0..rows {
        for h in 0..heads {
            for g in 0..d / 64 {
                let begin = (m * heads + h) * d + 64 * g;
                let (codes, metadata) = encode_group(&input[begin..begin + 64]);
                let scale = bf16::from_bits(metadata[0]).to_f32();
                let bias = bf16::from_bits(metadata[1]).to_f32();
                let out = ((h * capacity + pos + m) * d) + 64 * g;
                for i in 0..64 {
                    expected[out + i] =
                        bf16::from_f32(f32::from(codes[i]) * scale + bias);
                }
            }
        }
    }
    let output = Tensor::from_f32_as_bf16(
        &ctx,
        &vec![321.0; heads * capacity * d],
        &[heads, capacity, d],
    )?;
    let pass = ctx.begin_concurrent()?;
    scatter_kv(&ctx, &pass, &cache, &source, pos)?;
    pass.level_barrier(&[&cache.values, cache.scales.as_ref().unwrap()])?;
    kv_dequant_q8(&ctx, &pass, &cache, &output, pos + rows)?;
    pass.commit_wait()?;
    ensure!(
        output.raw_bytes() == bytemuck::cast_slice::<bf16, u8>(&expected),
        "affine64 dequant or output canary mismatch"
    );
    Ok(())
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn affine64_fused_k_unit_gain_matches_separate_norm_rope_scatter() -> Result<()> {
    // Unit folded gains deliberately isolate cache/RoPE integration from
    // the separately qualified non-unit-gain RMS rounding policy.
    let ctx = MetalContext::new()?;
    let (heads, d, capacity, pos, rot) = (2, 256, 64, 37, 64);
    let values: Vec<f32> =
        (0..heads * d).map(|i| ((i * 23 % 239) as f32 - 119.0) / 31.0).collect();
    let k = Tensor::from_f32_as_bf16(&ctx, &values, &[heads, d])?;
    let w = Tensor::from_f32_as_bf16(&ctx, &vec![1.0; d], &[d])?;
    let normed = Tensor::zeros(&ctx, &[heads, d], DType::BF16)?;
    let direct = KvCache::zeros(&ctx, KvFormat::Q8Affine64, heads, capacity, d)?;
    let reference = KvCache::zeros(&ctx, KvFormat::Q8Affine64, heads, capacity, d)?;
    let pass = ctx.begin_concurrent()?;
    crate::kernels::norm::rmsnorm_bf16(&ctx, &pass, &k, &w, &normed, 1e-6, 0.0)?;
    pass.level_barrier(&[&normed])?;
    rope_neox(&ctx, &pass, &normed, heads, rot, pos, 10_000_000.0, Rope::Delta(5))?;
    pass.level_barrier(&[&normed])?;
    scatter_kv(&ctx, &pass, &reference, &normed, pos)?;
    k_norm_rope_scatter_decode_with_bias(
        &ctx,
        &pass,
        &k,
        &w,
        &direct,
        rot,
        pos,
        10_000_000.0,
        1e-6,
        5,
        0.0,
    )?;
    pass.commit_wait()?;
    ensure!(
        direct.values.raw_bytes() == reference.values.raw_bytes(),
        "affine64 fused K codes mismatch"
    );
    ensure!(
        direct.scales.as_ref().unwrap().raw_bytes()
            == reference.scales.as_ref().unwrap().raw_bytes(),
        "affine64 fused K metadata mismatch"
    );
    Ok(())
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn affine64_attention_reads_match_materialized_bf16_cache() -> Result<()> {
    use crate::kernels::qsa::{self, SparseSplitScratch};
    let ctx = MetalContext::new()?;
    let (heads, nq, d, capacity, rows) = (2, 16, 256, 512, 320);
    let key_values: Vec<f32> =
        (0..rows * heads * d).map(|i| ((i * 29 % 257) as f32 - 128.0) / 91.0).collect();
    let value_values: Vec<f32> =
        (0..rows * heads * d).map(|i| ((i * 43 % 263) as f32 - 131.0) / 37.0).collect();
    let keys = Tensor::from_f32_as_bf16(&ctx, &key_values, &[rows, heads, d])?;
    let values = Tensor::from_f32_as_bf16(&ctx, &value_values, &[rows, heads, d])?;
    let key_cache = KvCache::zeros(&ctx, KvFormat::Q8Affine64, heads, capacity, d)?;
    let value_cache = KvCache::zeros(&ctx, KvFormat::Q8Affine64, heads, capacity, d)?;
    let dense_keys = Tensor::zeros(&ctx, &[heads, capacity, d], DType::BF16)?;
    let dense_values = Tensor::zeros(&ctx, &[heads, capacity, d], DType::BF16)?;
    let pass = ctx.begin_concurrent()?;
    scatter_kv(&ctx, &pass, &key_cache, &keys, 0)?;
    scatter_kv(&ctx, &pass, &value_cache, &values, 0)?;
    pass.level_barrier(&[
        &key_cache.values,
        key_cache.scales.as_ref().unwrap(),
        &value_cache.values,
        value_cache.scales.as_ref().unwrap(),
    ])?;
    kv_dequant_q8(&ctx, &pass, &key_cache, &dense_keys, rows)?;
    kv_dequant_q8(&ctx, &pass, &value_cache, &dense_values, rows)?;
    pass.commit_wait()?;
    let scale = 1.0 / 16.0;
    let queries: Vec<f32> =
        (0..nq * d).map(|i| ((i * 13 % 193) as f32 - 96.0) / 157.0).collect();
    let q = Tensor::from_f32_as_bf16(&ctx, &queries, &[nq, d])?;
    let direct = Tensor::zeros(&ctx, &[nq, d], DType::BF16)?;
    let materialized = Tensor::zeros(&ctx, &[nq, d], DType::BF16)?;
    let splits = rows.div_ceil(SDPA_SPLIT);
    let partials_a = Tensor::zeros(&ctx, &[nq, splits, d], DType::F32)?;
    let stats_a = Tensor::zeros(&ctx, &[nq, splits, 2], DType::F32)?;
    let partials_b = Tensor::zeros(&ctx, &[nq, splits, d], DType::F32)?;
    let stats_b = Tensor::zeros(&ctx, &[nq, splits, 2], DType::F32)?;
    let pass = ctx.begin_concurrent()?;
    sdpa_decode(
        &ctx,
        &pass,
        &q,
        &key_cache,
        &value_cache,
        &direct,
        rows,
        scale,
        Some((&partials_a, &stats_a)),
    )?;
    sdpa_decode(
        &ctx,
        &pass,
        &q,
        &dense_keys,
        &dense_values,
        &materialized,
        rows,
        scale,
        Some((&partials_b, &stats_b)),
    )?;
    pass.commit_wait()?;
    exact_attention_bytes(
        &direct,
        &materialized,
        "affine64 dense attention differs from its bf16 reconstruction",
    )?;
    for qb in [1usize, 16] {
        let base_pos = rows - qb;
        let k_max = 128;
        let ratio = 4;
        let queries: Vec<f32> =
            (0..qb * nq * d).map(|i| ((i * 17 % 197) as f32 - 98.0) / 163.0).collect();
        let query = Tensor::from_f32_as_bf16(&ctx, &queries, &[qb, nq, d])?;
        let mut selected = vec![0u32; qb * k_max];
        let mut n_selected = vec![0u32; qb];
        for qi in 0..qb {
            let nb = qsa::visible_blocks(base_pos + qi, ratio);
            n_selected[qi] = nb as u32;
            for b in 0..nb {
                selected[qi * k_max + b] = b as u32;
            }
        }
        let sel = Tensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&selected),
            &[qb, k_max],
            DType::U32,
        )?;
        let n_sel = Tensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&n_selected),
            &[qb],
            DType::U32,
        )?;
        let slots = qsa::split_scratch_slots(qb, k_max, ratio);
        let p_a = Tensor::zeros(&ctx, &[slots, nq, d], DType::F32)?;
        let p_b = Tensor::zeros(&ctx, &[slots, nq, d], DType::F32)?;
        let s_a = Tensor::zeros(&ctx, &[slots, nq, 2], DType::F32)?;
        let s_b = Tensor::zeros(&ctx, &[slots, nq, 2], DType::F32)?;
        let out_a = Tensor::zeros(&ctx, &[qb, nq, d], DType::BF16)?;
        let out_b = Tensor::zeros(&ctx, &[qb, nq, d], DType::BF16)?;
        let pass = ctx.begin_concurrent()?;
        qsa::qsa_attention(
            &ctx,
            &pass,
            &query,
            &key_cache,
            &value_cache,
            &sel,
            &n_sel,
            &out_a,
            &SparseSplitScratch { partials: &p_a, stats: &s_a },
            qb,
            k_max,
            ratio,
            base_pos,
            scale,
        )?;
        qsa::qsa_attention(
            &ctx,
            &pass,
            &query,
            &dense_keys,
            &dense_values,
            &sel,
            &n_sel,
            &out_b,
            &SparseSplitScratch { partials: &p_b, stats: &s_b },
            qb,
            k_max,
            ratio,
            base_pos,
            scale,
        )?;
        pass.commit_wait()?;
        exact_attention_bytes(
            &out_a,
            &out_b,
            "affine64 sparse split differs from its bf16 reconstruction",
        )?;
        if qb == 16 {
            let pass = ctx.begin_concurrent()?;
            qsa::qsa_attention_query(
                &ctx,
                &pass,
                &query,
                &key_cache,
                &value_cache,
                &sel,
                &n_sel,
                &out_a,
                qb,
                k_max,
                ratio,
                base_pos,
                scale,
            )?;
            qsa::qsa_attention_query(
                &ctx,
                &pass,
                &query,
                &dense_keys,
                &dense_values,
                &sel,
                &n_sel,
                &out_b,
                qb,
                k_max,
                ratio,
                base_pos,
                scale,
            )?;
            pass.commit_wait()?;
            exact_attention_bytes(
                &out_a,
                &out_b,
                "affine64 sparse query differs from its bf16 reconstruction",
            )?;
        }
    }
    Ok(())
}
