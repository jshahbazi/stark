// SPDX-License-Identifier: Apache-2.0
// Synthetic controls only. Failure diagnostics contain aggregate counts/hashes.
use half::bf16;

use super::*;
use crate::kernels::{norm::rmsnorm_bf16_with_rounding, qsa};

fn bf(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
fn exact(a: &[u8], b: &[u8], category: &str) -> Result<()> {
    ensure!(a.len() == b.len(), "{category}: length mismatch");
    let mismatches = a.iter().zip(b).filter(|(x, y)| x != y).count();
    ensure!(mismatches == 0, "{category}: byte_mismatches={mismatches}");
    Ok(())
}

// Test-only SHA-256 binds device bytes to the actual stock-library receipt;
// no dependency, external command, output vector or fixture file is needed.
fn sha256(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
        0x1f83d9ab, 0x5be0cd19,
    ];
    let mut padded = bytes.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&((bytes.len() as u64) * 8).to_be_bytes());
    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(chunk[4 * i..4 * i + 4].try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7)
                ^ w[i - 15].rotate_right(18)
                ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17)
                ^ w[i - 2].rotate_right(19)
                ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (s, v) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *s = s.wrapping_add(v);
        }
    }
    state.iter().map(|x| format!("{x:08x}")).collect()
}

#[test]
fn official_fused_rms_fixture_sha256_known_answers() {
    assert_eq!(
        sha256(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn official_ordinary_rms_matches_recorded_stock_gpu_bf16_hashes() -> Result<()> {
    // Actual stock MLX GPU C API report a76b2034... at pinned 64ea011c.
    // These are RAW BF16-gain output hashes. F32 gains return F32 and are
    // intentionally refused below; their BF16-cast hashes are not raw hashes.
    let ctx = MetalContext::new()?;
    for (m, h, eps, expected) in [
        (
            1usize,
            128usize,
            0.0,
            "4a30a2e2d0107a7b611fcd295882576cef08105794eeb829b32c8d8dd407eda3",
        ),
        (
            3,
            128,
            1e-6,
            "ed0dac788e82a2f43d999490f96da33f3029a97c113da87c153bf8e4082b161a",
        ),
        (
            1,
            2560,
            1e-6,
            "a383af7a7daafa70d68f3b28ea8a33c26742bb7acb0f68e18c06968d7f89842e",
        ),
        (
            3,
            2560,
            0.0,
            "8f51277b157ee276c48c29389a87ab2b35f5250998f1e7c7d6fbc99d2b68a879",
        ),
    ] {
        let values: Vec<f32> =
            (0..m * h).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }).collect();
        let gains: Vec<f32> =
            (0..h).map(|i| bf(0.5 + (i % 192) as f32 / 128.0)).collect();
        let x = Tensor::from_f32_as_bf16(&ctx, &values, &[m, h])?;
        let w = Tensor::from_f32_as_bf16(&ctx, &gains, &[h])?;
        let fw = Tensor::from_f32(&ctx, &gains, &[h])?;
        let out = Tensor::zeros(&ctx, &[m, h], DType::BF16)?;
        let pass = ctx.begin_concurrent()?;
        ensure!(
            rmsnorm_bf16_with_rounding(&ctx, &pass, &x, &fw, &out, eps, 0.0, true)
                .is_err(),
            "stock promoting F32 gain was silently narrowed"
        );
        rmsnorm_bf16_with_rounding(&ctx, &pass, &x, &w, &out, eps, 0.0, true)?;
        pass.commit_wait()?;
        ensure!(
            sha256(out.raw_bytes()) == expected,
            "stock raw BF16 RMS hash mismatch: rows={m}, width={h}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn official_fused_qk_two_round_gain_matches_standalone_and_cache_encode() -> Result<()>
{
    for nq in [16usize, 24] {
        official_fused_qk_two_round_gain_matches_standalone_and_cache_encode_for_heads(
            nq,
        )?;
    }
    Ok(())
}

fn official_fused_qk_two_round_gain_matches_standalone_and_cache_encode_for_heads(
    nq: usize,
) -> Result<()> {
    let ctx = MetalContext::new()?;
    let (kvh, d, cap, pos) = (2usize, 256usize, 64usize, 37usize);
    let qg_values: Vec<f32> =
        (0..nq * 2 * d).map(|i| if i % d % 2 == 0 { 1.0 } else { 2.0 }).collect();
    let k_values: Vec<f32> =
        (0..kvh * d).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }).collect();
    let gains: Vec<f32> = (0..d).map(|i| bf(0.5 + (i % 192) as f32 / 128.0)).collect();
    let qg = Tensor::from_f32_as_bf16(&ctx, &qg_values, &[nq, 2 * d])?;
    let k = Tensor::from_f32_as_bf16(&ctx, &k_values, &[kvh, d])?;
    let w = Tensor::from_f32_as_bf16(&ctx, &gains, &[d])?;
    for format in [KvFormat::Bf16, KvFormat::Q8, KvFormat::Q8Affine64] {
        let q = Tensor::zeros(&ctx, &[nq, d], DType::BF16)?;
        let gate = Tensor::zeros(&ctx, &[nq, d], DType::BF16)?;
        let qr = Tensor::zeros(&ctx, &[nq, d], DType::BF16)?;
        let gr = Tensor::zeros(&ctx, &[nq, d], DType::BF16)?;
        let kr = Tensor::zeros(&ctx, &[kvh, d], DType::BF16)?;
        let cache = KvCache::zeros(&ctx, format, kvh, cap, d)?;
        let reference = KvCache::zeros(&ctx, format, kvh, cap, d)?;
        let pass = ctx.begin_concurrent()?;
        split_q_gate(&ctx, &pass, &qg, &qr, &gr)?;
        rmsnorm_bf16_with_rounding(&ctx, &pass, &k, &w, &kr, 1e-6, 0.0, true)?;
        pass.level_barrier(&[&qr, &kr])?;
        rmsnorm_bf16_with_rounding(&ctx, &pass, &qr, &w, &qr, 1e-6, 0.0, true)?;
        pass.level_barrier(&[&qr])?;
        scatter_kv(&ctx, &pass, &reference, &kr, pos)?;
        q_norm_rope_split_decode_with_rounding(
            &ctx,
            &pass,
            &qg,
            &w,
            &q,
            &gate,
            0,
            pos,
            10_000_000.0,
            1e-6,
            5,
            0.0,
            true,
        )?;
        k_norm_rope_scatter_decode_with_rounding(
            &ctx,
            &pass,
            &k,
            &w,
            &cache,
            0,
            pos,
            10_000_000.0,
            1e-6,
            5,
            0.0,
            true,
        )?;
        pass.commit_wait()?;
        exact(gate.raw_bytes(), gr.raw_bytes(), "official Q gate split")?;
        exact(q.raw_bytes(), qr.raw_bytes(), "official Q two-round norm")?;
        exact(
            cache.values.raw_bytes(),
            reference.values.raw_bytes(),
            "official K cache code",
        )?;
        if let (Some(a), Some(b)) = (&cache.scales, &reference.scales) {
            exact(a.raw_bytes(), b.raw_bytes(), "official K cache metadata")?;
        }
        let inv = 1.0 / (2.5f32 + 1e-6).sqrt();
        let correct: Vec<f32> = k_values
            .iter()
            .enumerate()
            .map(|(i, x)| bf(bf(*x * inv) * gains[i % d]))
            .collect();
        let wrong: Vec<f32> = k_values
            .iter()
            .enumerate()
            .map(|(i, x)| bf(*x * inv * gains[i % d]))
            .collect();
        ensure!(
            correct.iter().zip(wrong).any(|(a, b)| a.to_bits() != b.to_bits()),
            "Q/K non-unit gain mutation witness absent"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn official_qsa_all_four_norm_routes_match_stock_two_round_chain() -> Result<()> {
    let ctx = MetalContext::new()?;
    let (m, nh, d, ratio) = (3usize, 4usize, 128usize, 4usize);
    let values: Vec<f32> =
        (0..m * (nh + 1) * d).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }).collect();
    let gains: Vec<f32> = (0..d).map(|i| bf(0.5 + (i % 192) as f32 / 128.0)).collect();
    let qk = Tensor::from_f32_as_bf16(&ctx, &values, &[m, (nh + 1) * d])?;
    let keys: Vec<f32> =
        (0..12 * d).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }).collect();
    let cache = Tensor::from_f32_as_bf16(&ctx, &keys, &[12, d])?;
    let w = Tensor::from_f32_as_bf16(&ctx, &gains, &[d])?;
    let positions: Vec<u32> = (0..12).flat_map(|i| [i, i + 1, i + 2]).collect();
    let positions = Tensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&positions),
        &[12, 3],
        DType::U32,
    )?;
    let inv = 1.0 / (2.5f32 + 1e-6).sqrt();
    let row: Vec<bf16> = (0..d)
        .map(|i| {
            bf16::from_f32(bf(bf(if i % 2 == 0 { inv } else { 2.0 * inv }) * gains[i]))
        })
        .collect();
    for rope in [Rope::Delta(5), Rope::Rows { positions: &positions, base: 0 }] {
        let q = Tensor::zeros(&ctx, &[m, nh, d], DType::BF16)?;
        let blocks = Tensor::zeros(&ctx, &[3, d], DType::BF16)?;
        let pass = ctx.begin_concurrent()?;
        qsa::qsa_prep_q_with_rounding(
            &ctx,
            &pass,
            &qk,
            &w,
            &q,
            nh,
            0,
            0,
            10_000_000.0,
            1e-6,
            rope,
            0.0,
            true,
        )?;
        qsa::qsa_block_keys_with_rounding(
            &ctx,
            &pass,
            &cache,
            &w,
            &blocks,
            ratio,
            1,
            2,
            0,
            10_000_000.0,
            1e-6,
            rope,
            0.0,
            true,
        )?;
        pass.commit_wait()?;
        let qr: Vec<bf16> = (0..m * nh).flat_map(|_| row.iter().copied()).collect();
        exact(q.raw_bytes(), bytemuck::cast_slice(&qr), "official QSA query norm")?;
        let mut br = vec![bf16::from_f32(0.0); d];
        br.extend(row.iter().copied());
        br.extend(row.iter().copied());
        exact(
            blocks.raw_bytes(),
            bytemuck::cast_slice(&br),
            "official QSA block norm",
        )?;
    }
    Ok(())
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn official_fused_norms_reject_promoting_or_delta_gains_and_preserve_false()
-> Result<()> {
    let ctx = MetalContext::new()?;
    let qg = Tensor::from_f32_as_bf16(&ctx, &vec![1.0; 512], &[1, 512])?;
    let k = Tensor::from_f32_as_bf16(&ctx, &vec![1.0; 256], &[1, 256])?;
    let w = Tensor::from_f32_as_bf16(&ctx, &vec![1.25; 256], &[256])?;
    let fw = Tensor::from_f32(&ctx, &vec![1.25; 256], &[256])?;
    let q = Tensor::zeros(&ctx, &[1, 256], DType::BF16)?;
    let g = Tensor::zeros(&ctx, &[1, 256], DType::BF16)?;
    let qr = Tensor::zeros(&ctx, &[1, 256], DType::BF16)?;
    let gr = Tensor::zeros(&ctx, &[1, 256], DType::BF16)?;
    let cache = KvCache::zeros(&ctx, KvFormat::Q8Affine64, 1, 64, 256)?;
    let iqk = Tensor::from_f32_as_bf16(&ctx, &vec![1.0; 256], &[1, 256])?;
    let iw = Tensor::from_f32(&ctx, &vec![1.25; 128], &[128])?;
    let io = Tensor::zeros(&ctx, &[1, 128], DType::BF16)?;
    let ik = Tensor::from_f32_as_bf16(&ctx, &vec![1.0; 4 * 128], &[4, 128])?;
    let pass = ctx.begin_concurrent()?;
    ensure!(
        q_norm_rope_split_decode_with_rounding(
            &ctx, &pass, &qg, &fw, &q, &g, 0, 0, 1e7, 1e-6, 0, 0.0, true
        )
        .is_err(),
        "official Q accepted F32 gain"
    );
    ensure!(
        k_norm_rope_scatter_decode_with_rounding(
            &ctx, &pass, &k, &fw, &cache, 0, 0, 1e7, 1e-6, 0, 0.0, true
        )
        .is_err(),
        "official K accepted F32 gain"
    );
    ensure!(
        q_norm_rope_split_decode_with_rounding(
            &ctx, &pass, &qg, &w, &q, &g, 0, 0, 1e7, 1e-6, 0, 1.0, true
        )
        .is_err(),
        "official Q accepted delta gain"
    );
    ensure!(
        qsa::qsa_prep_q_with_rounding(
            &ctx,
            &pass,
            &iqk,
            &iw,
            &io,
            1,
            0,
            0,
            1e7,
            1e-6,
            Rope::Delta(0),
            0.0,
            true
        )
        .is_err(),
        "official QSA query accepted F32 gain"
    );
    ensure!(
        qsa::qsa_block_keys_with_rounding(
            &ctx,
            &pass,
            &ik,
            &iw,
            &io,
            4,
            0,
            1,
            0,
            1e7,
            1e-6,
            Rope::Delta(0),
            0.0,
            true
        )
        .is_err(),
        "official QSA block accepted F32 gain"
    );
    q_norm_rope_split_decode_with_bias(
        &ctx, &pass, &qg, &w, &qr, &gr, 0, 0, 1e7, 1e-6, 0, 1.0,
    )?;
    q_norm_rope_split_decode_with_rounding(
        &ctx, &pass, &qg, &w, &q, &g, 0, 0, 1e7, 1e-6, 0, 1.0, false,
    )?;
    pass.commit_wait()?;
    exact(q.raw_bytes(), qr.raw_bytes(), "legacy false Q dispatch")?;
    exact(g.raw_bytes(), gr.raw_bytes(), "legacy false gate dispatch")?;
    Ok(())
}

#[test]
#[ignore = "requires a physical Apple Silicon Metal device"]
fn official_affine_dequant_rounding_visible_in_dense_and_both_sparse_routes()
-> Result<()> {
    let ctx = MetalContext::new()?;
    let (kvh, nq, d, cap) = (2usize, 16usize, 256usize, 32usize);
    let (code, scale, bias) = (131u8, 0.003936767578125f32, -0.5f32);
    let unrounded = f32::from(code) * scale + bias;
    let right = bf(unrounded);
    let wrong = bf(bf(f32::from(code) * scale) + bias);
    ensure!(
        right.to_bits() != wrong.to_bits() && right.to_bits() != unrounded.to_bits(),
        "affine dequant mutation witnesses absent"
    );
    let metadata: Vec<bf16> = (0..kvh * cap * (d / 64))
        .flat_map(|_| [bf16::from_f32(scale), bf16::from_f32(bias)])
        .collect();
    let codes = vec![code; kvh * cap * d];
    let cache = KvCache {
        values: Tensor::from_bytes(&ctx, &codes, &[kvh, cap, d], DType::I8)?,
        scales: Some(Tensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&metadata),
            &[kvh, cap, d / 32],
            DType::BF16,
        )?),
    };
    let queries: Vec<f32> =
        (0..nq * d).map(|i| if i % d == 0 { 1.0 } else { 0.0 }).collect();
    let q = Tensor::from_f32_as_bf16(&ctx, &queries, &[nq, d])?;
    let out = Tensor::zeros(&ctx, &[nq, d], DType::BF16)?;
    let partials = Tensor::zeros(&ctx, &[nq, 1, d], DType::F32)?;
    let stats = Tensor::zeros(&ctx, &[nq, 1, 2], DType::F32)?;
    let pass = ctx.begin_concurrent()?;
    sdpa_decode(
        &ctx,
        &pass,
        &q,
        &cache,
        &cache,
        &out,
        1,
        1.0,
        Some((&partials, &stats)),
    )?;
    pass.commit_wait()?;
    ensure!(
        out.to_f32()?.iter().all(|x| x.to_bits() == right.to_bits()),
        "dense affine V materialization mismatch"
    );
    ensure!(
        stats.to_f32()?.chunks_exact(2).all(|x| x[0].to_bits() == right.to_bits()),
        "dense affine K dot lost final BF16 materialization"
    );
    for qb in [1usize, 16] {
        let queries: Vec<f32> =
            (0..qb * nq * d).map(|i| if i % d == 0 { 1.0 } else { 0.0 }).collect();
        let q = Tensor::from_f32_as_bf16(&ctx, &queries, &[qb, nq, d])?;
        let sel = Tensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&vec![0u32; qb]),
            &[qb, 1],
            DType::U32,
        )?;
        let nsel = Tensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&vec![0u32; qb]),
            &[qb],
            DType::U32,
        )?;
        let slots = qsa::split_scratch_slots(qb, 1, 32);
        let p = Tensor::zeros(&ctx, &[slots, nq, d], DType::F32)?;
        let s = Tensor::zeros(&ctx, &[slots, nq, 2], DType::F32)?;
        let o = Tensor::zeros(&ctx, &[qb, nq, d], DType::BF16)?;
        let pass = ctx.begin_concurrent()?;
        qsa::qsa_attention(
            &ctx,
            &pass,
            &q,
            &cache,
            &cache,
            &sel,
            &nsel,
            &o,
            &qsa::SparseSplitScratch { partials: &p, stats: &s },
            qb,
            1,
            32,
            0,
            1.0,
        )?;
        pass.commit_wait()?;
        ensure!(
            o.to_f32()?.iter().all(|x| x.to_bits() == right.to_bits()),
            "sparse split affine V materialization mismatch"
        );
        let used = qb * nq * 2;
        ensure!(
            s.to_f32()?[..used]
                .chunks_exact(2)
                .all(|x| x[0].to_bits() == right.to_bits()),
            "sparse split affine K dot lost final BF16 materialization"
        );
        if qb == 16 {
            let pass = ctx.begin_concurrent()?;
            qsa::qsa_attention_query(
                &ctx, &pass, &q, &cache, &cache, &sel, &nsel, &o, qb, 1, 32, 0, 1.0,
            )?;
            pass.commit_wait()?;
            ensure!(
                o.to_f32()?.iter().all(|x| x.to_bits() == right.to_bits()),
                "sparse query affine V materialization mismatch"
            );
        }
    }
    Ok(())
}
