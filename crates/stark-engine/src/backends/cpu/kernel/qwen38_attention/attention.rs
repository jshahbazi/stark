//! CPU references for Qwen3.8's standard-backend compressed attention kernels.

use half::{bf16, f16};
use stark_engine_macros::kernel;

const D: usize = 256;
const ID: usize = 128;
const SPLIT: usize = 64;

fn norm(
    values: &[f32],
    weight: *const f32,
    epsilon: f32,
) -> Vec<f32> {
    let sum = values.iter().fold(0.0f32, |a, &v| a + v * v);
    let inverse = (sum / values.len() as f32 + epsilon).sqrt().recip();
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| unsafe { bf16::from_f32(v * inverse * (1.0 + *weight.add(i))).to_f32() })
        .collect()
}

fn rope(
    values: &mut [f32],
    rotary: u32,
    position: u32,
    theta: f32,
    rounded_products: bool,
) {
    let half = rotary as usize / 2;
    for i in 0..half {
        let angle = position as f32 * theta.powf(-2.0 * i as f32 / rotary as f32);
        let (mut sine, mut cosine) = angle.sin_cos();
        let (lo, hi) = (values[i], values[half + i]);
        let (left, right) = if rounded_products {
            sine = bf16::from_f32(sine).to_f32();
            cosine = bf16::from_f32(cosine).to_f32();
            (
                bf16::from_f32(lo * cosine).to_f32() - bf16::from_f32(hi * sine).to_f32(),
                bf16::from_f32(hi * cosine).to_f32() + bf16::from_f32(lo * sine).to_f32(),
            )
        } else {
            (lo * cosine - hi * sine, hi * cosine + lo * sine)
        };
        values[i] = bf16::from_f32(left).to_f32();
        values[half + i] = bf16::from_f32(right).to_f32();
    }
}

fn quantize(
    values: &[f32],
    codes: *mut i8,
    scales: *mut f16,
) {
    for (group, values) in values.chunks_exact(32).enumerate() {
        let maximum = values.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let scale = f16::from_f32((maximum / 127.0).min(65504.0));
        let inverse = if scale.to_f32() == 0.0 {
            0.0
        } else {
            scale.to_f32().recip()
        };
        unsafe {
            *scales.add(group) = scale;
        }
        for (i, &value) in values.iter().enumerate() {
            unsafe {
                *codes.add(group * 32 + i) = (value * inverse).round_ties_even().clamp(-127.0, 127.0) as i8;
            }
        }
    }
}

#[kernel(Qwen38QsaPrepareQuery)]
pub fn prepare_query(
    qg: *const bf16,
    w: *const f32,
    q: *mut bf16,
    gate: *mut bf16,
    query_heads: u32,
    rot: u32,
    pos: u32,
    theta: f32,
    eps: f32,
) {
    for head in 0..query_heads as usize {
        let values: Vec<f32> = (0..D).map(|i| unsafe { (*qg.add(head * 2 * D + i)).to_f32() }).collect();
        let mut values = norm(&values, w, eps);
        rope(&mut values, rot, pos, theta, false);
        for i in 0..D {
            unsafe {
                *q.add(head * D + i) = bf16::from_f32(values[i]);
                *gate.add(head * D + i) = *qg.add(head * 2 * D + D + i);
            }
        }
    }
}

#[kernel(Qwen38QsaPrepareKey)]
pub fn prepare_key(
    k: *const bf16,
    w: *const f32,
    cache: *mut i8,
    scales: *mut f16,
    kv_heads: u32,
    max_seq: u32,
    rot: u32,
    pos: u32,
    theta: f32,
    eps: f32,
) {
    for head in 0..kv_heads as usize {
        let values: Vec<f32> = (0..D).map(|i| unsafe { (*k.add(head * D + i)).to_f32() }).collect();
        let mut values = norm(&values, w, eps);
        rope(&mut values, rot, pos, theta, false);
        let row = head * max_seq as usize + pos as usize;
        unsafe {
            quantize(&values, cache.add(row * D), scales.add(row * (D / 32)));
        }
    }
}

#[kernel(Qwen38QsaStoreValue)]
pub fn store_value(
    rows: *const bf16,
    cache: *mut i8,
    scales: *mut f16,
    kv_heads: u32,
    max_seq: u32,
    pos: u32,
) {
    for head in 0..kv_heads as usize {
        let values: Vec<f32> = (0..D).map(|i| unsafe { (*rows.add(head * D + i)).to_f32() }).collect();
        let row = head * max_seq as usize + pos as usize;
        unsafe {
            quantize(&values, cache.add(row * D), scales.add(row * (D / 32)));
        }
    }
}

#[kernel(Qwen38QsaPrepareIndex)]
pub fn prepare_index(
    qk: *const bf16,
    w: *const f32,
    q: *mut bf16,
    raw: *mut bf16,
    index_heads: u32,
    rot: u32,
    pos: u32,
    theta: f32,
    eps: f32,
) {
    for head in 0..index_heads as usize {
        let values: Vec<f32> = (0..ID).map(|i| unsafe { (*qk.add(head * ID + i)).to_f32() }).collect();
        let mut values = norm(&values, w, eps);
        rope(&mut values, rot, pos, theta, true);
        for (i, value) in values.iter().enumerate() {
            unsafe {
                *q.add(head * ID + i) = bf16::from_f32(*value);
            }
        }
    }
    for i in 0..ID {
        unsafe {
            *raw.add(pos as usize * ID + i) = *qk.add(index_heads as usize * ID + i);
        }
    }
}

#[kernel(Qwen38QsaBlockKeys)]
pub fn block_keys(
    cache: *const bf16,
    w: *const f32,
    blk: *mut bf16,
    ratio: u32,
    first_block: u32,
    rot: u32,
    theta: f32,
    eps: f32,
) {
    let values: Vec<f32> = (0..ID)
        .map(|i| {
            let mut sum = 0.0f32;
            for row in 0..ratio as usize {
                unsafe {
                    sum += (*cache.add((first_block as usize * ratio as usize + row) * ID + i)).to_f32();
                }
            }
            bf16::from_f32(sum / ratio as f32).to_f32()
        })
        .collect();
    let mut values = norm(&values, w, eps);
    rope(&mut values, rot, first_block * ratio, theta, true);
    for (i, value) in values.iter().enumerate() {
        unsafe {
            *blk.add(first_block as usize * ID + i) = bf16::from_f32(*value);
        }
    }
}

#[kernel(Qwen38QsaScores)]
pub fn scores(
    q: *const bf16,
    blk: *const bf16,
    scores: *mut f32,
    index_heads: u32,
    nb_max: u32,
) {
    for block in 0..nb_max as usize {
        let mut score = 0.0f32;
        for head in 0..index_heads as usize {
            let mut sum = 0.0f32;
            for i in 0..ID {
                unsafe {
                    sum += (*q.add(head * ID + i)).to_f32() * (*blk.add(block * ID + i)).to_f32();
                }
            }
            score += sum.max(0.0);
        }
        unsafe {
            *scores.add(block) = score * (ID as f32).sqrt().recip();
        }
    }
}

#[kernel(Qwen38QsaSelect)]
pub fn select(
    scores: *const f32,
    sel: *mut u32,
    n_sel: *mut u32,
    nb_max: u32,
    k_max: u32,
) {
    let count = nb_max.min(k_max) as usize;
    let mut indices: Vec<u32> = (0..nb_max).collect();
    if nb_max > k_max {
        indices.sort_unstable_by(|&a, &b| unsafe {
            let left = (*scores.add(a as usize)).max(0.0).to_bits();
            let right = (*scores.add(b as usize)).max(0.0).to_bits();
            right.cmp(&left).then_with(|| a.cmp(&b))
        });
        indices.truncate(count);
        indices.sort_unstable();
    }
    for (i, &index) in indices.iter().take(count).enumerate() {
        unsafe {
            *sel.add(i) = index;
        }
    }
    unsafe {
        *n_sel = count as u32;
    }
}

#[allow(clippy::too_many_arguments)]
#[kernel(Qwen38QsaAttention)]
pub fn attention(
    q: *const bf16,
    k_cache: *const i8,
    v_cache: *const i8,
    k_scales: *const f16,
    v_scales: *const f16,
    sel: *const u32,
    n_sel: *const u32,
    partials: *mut f32,
    stats: *mut f32,
    max_seq: u32,
    group: u32,
    splits: u32,
    k_max: u32,
    ratio: u32,
    base_pos: u32,
    scale: f32,
    query_heads: u32,
    head_groups: u32,
    kv_heads: u32,
) {
    let _ = (k_max, head_groups, kv_heads);
    let blocks = unsafe { *n_sel };
    let tail_start = (base_pos + 1) / ratio * ratio;
    let total = blocks * ratio + (base_pos + 1 - tail_start);
    let token_at = |slot: usize| -> usize {
        if slot < (blocks * ratio) as usize {
            unsafe { *sel.add(slot / ratio as usize) as usize * ratio as usize + slot % ratio as usize }
        } else {
            tail_start as usize + slot - (blocks * ratio) as usize
        }
    };
    for head in 0..query_heads as usize {
        let kv = head / group as usize;
        for split in 0..splits as usize {
            let first = split * SPLIT;
            let count = (total as usize).saturating_sub(first).min(SPLIT);
            let stat_at = (head * splits as usize + split) * 2;
            let out_at = (head * splits as usize + split) * D;
            if count == 0 {
                unsafe {
                    *stats.add(stat_at) = f32::NEG_INFINITY;
                    *stats.add(stat_at + 1) = 0.0;
                }
                for i in 0..D {
                    unsafe {
                        *partials.add(out_at + i) = 0.0;
                    }
                }
                continue;
            }
            let mut probabilities = [0.0f32; SPLIT];
            let mut maximum = f32::NEG_INFINITY;
            for (p, probability) in probabilities[..count].iter_mut().enumerate() {
                let token = token_at(first + p);
                let cache_row = kv * max_seq as usize + token;
                let mut sum = 0.0f32;
                for i in 0..D {
                    unsafe {
                        sum += (*q.add(head * D + i)).to_f32()
                            * (*k_cache.add(cache_row * D + i)) as f32
                            * (*k_scales.add(cache_row * (D / 32) + i / 32)).to_f32();
                    }
                }
                *probability = sum * scale;
                maximum = maximum.max(*probability);
            }
            let mut denominator = 0.0f32;
            for probability in &mut probabilities[..count] {
                *probability = (*probability - maximum).exp();
                denominator += *probability;
            }
            unsafe {
                *stats.add(stat_at) = maximum;
                *stats.add(stat_at + 1) = denominator;
            }
            for i in 0..D {
                let mut sum = 0.0f32;
                for (p, &probability) in probabilities[..count].iter().enumerate() {
                    let cache_row = kv * max_seq as usize + token_at(first + p);
                    unsafe {
                        sum += probability
                            * (*v_cache.add(cache_row * D + i)) as f32
                            * (*v_scales.add(cache_row * (D / 32) + i / 32)).to_f32();
                    }
                }
                unsafe {
                    *partials.add(out_at + i) = sum;
                }
            }
        }
    }
}

#[kernel(Qwen38QsaCombine)]
pub fn combine(
    partials: *const f32,
    stats: *const f32,
    gate: *const bf16,
    out: *mut bf16,
    query_heads: u32,
    splits: u32,
) {
    for head in 0..query_heads as usize {
        let mut maximum = f32::NEG_INFINITY;
        for split in 0..splits as usize {
            unsafe {
                maximum = maximum.max(*stats.add((head * splits as usize + split) * 2));
            }
        }
        let mut denominator = 0.0f32;
        for split in 0..splits as usize {
            unsafe {
                let at = (head * splits as usize + split) * 2;
                denominator += *stats.add(at + 1) * (*stats.add(at) - maximum).exp();
            }
        }
        for i in 0..D {
            let mut sum = 0.0f32;
            for split in 0..splits as usize {
                unsafe {
                    sum += (*stats.add((head * splits as usize + split) * 2) - maximum).exp()
                        * *partials.add((head * splits as usize + split) * D + i);
                }
            }
            unsafe {
                let rounded = bf16::from_f32(sum / denominator).to_f32();
                let gate = 1.0 / (1.0 + (-(*gate.add(head * D + i)).to_f32()).exp());
                *out.add(head * D + i) = bf16::from_f32(gate * rounded);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn selection_uses_earliest_ties_and_returns_ascending_blocks() {
        let scores = [0.0, 2.0, 2.0, 1.0, 2.0, 0.0];
        let mut selected = [u32::MAX; 3];
        let mut count = u32::MAX;
        select(scores.as_ptr(), selected.as_mut_ptr(), &mut count, 6, 2);
        assert_eq!(count, 2);
        assert_eq!(&selected[..2], &[1, 2]);
        assert_eq!(selected[2], u32::MAX);
        select(scores.as_ptr(), selected.as_mut_ptr(), &mut count, 0, 3);
        assert_eq!(count, 0);
    }

    #[stark_test]
    fn q8_groups_have_zero_and_finite_saturating_scales() {
        let mut codes = [99i8; 64];
        let mut scales = [f16::NAN; 2];
        let mut values = [0.0f32; 64];
        values[32..].fill(bf16::MAX.to_f32());
        quantize(&values, codes.as_mut_ptr(), scales.as_mut_ptr());
        assert_eq!(&codes[..32], &[0i8; 32]);
        assert_eq!(scales[0], f16::ZERO);
        assert_eq!(scales[1], f16::MAX);
        assert_eq!(&codes[32..], &[127i8; 32]);
    }

    #[stark_test]
    fn sparse_attention_reads_selected_blocks_and_causal_tail_only() {
        let q = [bf16::ZERO; D];
        let keys = [0i8; 6 * D];
        let mut values = [0i8; 6 * D];
        for row in 0..6 {
            values[row * D..(row + 1) * D].fill((row + 1) as i8);
        }
        values[2 * D..4 * D].fill(100);
        let scales = [f16::ONE; 6 * (D / 32)];
        let mut partials = [f32::NAN; D];
        let mut stats = [f32::NAN; 2];
        let mut output = [bf16::NAN; D];
        let gate = [bf16::ZERO; D];
        let selected = [0u32, 2];
        attention(
            q.as_ptr(),
            keys.as_ptr(),
            values.as_ptr(),
            scales.as_ptr(),
            scales.as_ptr(),
            selected.as_ptr(),
            &2,
            partials.as_mut_ptr(),
            stats.as_mut_ptr(),
            6,
            1,
            1,
            2,
            2,
            5,
            0.0625,
            1,
            1,
            1,
        );
        combine(partials.as_ptr(), stats.as_ptr(), gate.as_ptr(), output.as_mut_ptr(), 1, 1);
        assert!(output.iter().all(|&v| v == bf16::from_f32(1.75)));
        attention(
            q.as_ptr(),
            keys.as_ptr(),
            values.as_ptr(),
            scales.as_ptr(),
            scales.as_ptr(),
            selected.as_ptr(),
            &1,
            partials.as_mut_ptr(),
            stats.as_mut_ptr(),
            6,
            1,
            1,
            2,
            2,
            4,
            0.0625,
            1,
            1,
            1,
        );
        combine(partials.as_ptr(), stats.as_ptr(), gate.as_ptr(), output.as_mut_ptr(), 1, 1);
        let expected = bf16::from_f32(0.5 * bf16::from_f32(8.0 / 3.0).to_f32());
        assert!(output.iter().all(|&v| v == expected));
    }
}
