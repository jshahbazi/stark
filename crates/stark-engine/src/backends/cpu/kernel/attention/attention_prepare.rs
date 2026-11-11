use half::bf16;
use num_traits::Float;
use stark_engine_macros::kernel;

use crate::array::ArrayElement;

fn apply_rope<ElementT: ArrayElement + Float>(
    head: *const ElementT,
    cosines: *const (impl ArrayElement + Float),
    sines: *const (impl ArrayElement + Float),
    batch_idx: usize,
    head_dim_idx: usize,
    rope_dim: usize,
    reference_bf16: bool,
) -> ElementT {
    let half_rope_dim = rope_dim / 2;
    let paired_idx = if head_dim_idx < half_rope_dim {
        head_dim_idx + half_rope_dim
    } else {
        head_dim_idx - half_rope_dim
    };
    let input = unsafe { (*head.add(head_dim_idx)).to_f32().unwrap() };
    let paired = unsafe { (*head.add(paired_idx)).to_f32().unwrap() };
    let signed_paired = if head_dim_idx < half_rope_dim {
        -paired
    } else {
        paired
    };
    let cos_val = unsafe { (*cosines.add(batch_idx * rope_dim + head_dim_idx)).to_f32().unwrap() };
    let sin_val = unsafe { (*sines.add(batch_idx * rope_dim + head_dim_idx)).to_f32().unwrap() };

    if reference_bf16 {
        // The pinned text rotary operation receives BF16 coefficients and
        // materializes both BF16 products before its final BF16 addition.
        let cosine = bf16::from_f32(cos_val).to_f32();
        let sine = bf16::from_f32(sin_val).to_f32();
        let first = bf16::from_f32(input * cosine).to_f32();
        let second = bf16::from_f32(signed_paired * sine).to_f32();
        return ElementT::from(first + second).unwrap();
    }

    ElementT::from(input * cos_val + signed_paired * sin_val).unwrap()
}

// Mathematical CPU reference for the explicit MLX coefficient profile. Metal
// uses its source fast trigonometric intrinsics; cross-backend comparisons allow
// their final BF16 rounding difference rather than substituting cached tables.
fn apply_source_rope<ElementT: ArrayElement + Float>(
    head: *const ElementT,
    position: u32,
    head_dim_idx: usize,
    rope_dim: usize,
    base: f32,
    rotary_dim: u32,
    proportional: bool,
    yarn: bool,
    yarn_parameters: Option<*const f32>,
) -> ElementT {
    let half_dim = rope_dim / 2;
    let pair = head_dim_idx % half_dim;
    let fraction = pair as f32 / half_dim as f32;
    let inverse_frequency = if proportional {
        if 2 * pair < rotary_dim as usize {
            {
                let mut frequency = base.powf(fraction);
                if yarn {
                    let p = yarn_parameters.unwrap();
                    let (factor, low, high) = unsafe { (*p, *p.add(1), *p.add(2)) };
                    let inter = factor * frequency;
                    let mask = 1.0 - ((pair as f32 - low) / (high - low)).clamp(0.0, 1.0);
                    frequency = (inter * frequency) / (inter * mask + frequency * (1.0 - mask));
                }
                1.0 / frequency
            }
        } else {
            0.0
        }
    } else {
        (-fraction * base).exp2()
    };
    let theta = position as f32 * inverse_frequency;
    let mut x1 = unsafe { (*head.add(pair)).to_f32().unwrap() };
    let mut x2 = unsafe { (*head.add(pair + half_dim)).to_f32().unwrap() };
    if yarn {
        let scale = unsafe { *yarn_parameters.unwrap().add(3) };
        x1 = ElementT::from(x1 * scale).unwrap().to_f32().unwrap();
        x2 = ElementT::from(x2 * scale).unwrap().to_f32().unwrap();
    }
    let value = if head_dim_idx < half_dim {
        x1 * theta.cos() - x2 * theta.sin()
    } else {
        x1 * theta.sin() + x2 * theta.cos()
    };
    ElementT::from(value).unwrap()
}

/// KV-only preparation of full packed QKVG, never tightly packed appended KV.
#[kernel(AttentionPrepareKVOnly)]
#[variants(ElementT, bf16)]
#[variants(RopeT, f32)]
pub fn attention_prepare_kv_only<ElementT: ArrayElement + Float, RopeT: ArrayElement + Float>(
    qkvg: *const ElementT,
    keys: *mut ElementT,
    values: *mut ElementT,
    #[optional(has_rope)] cosines: Option<*const RopeT>,
    #[optional(has_rope)] sines: Option<*const RopeT>,
    #[optional(source_rope)] source_positions: Option<*const u32>,
    #[optional(source_rope)] source_base: Option<f32>,
    #[optional(source_rope)] source_rotary_dim: Option<u32>,
    #[optional(source_yarn)] source_yarn_parameters: Option<*const f32>,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    #[optional(has_rope)] rope_dim: Option<u32>,
    kv_token_offset: u32,
    input_row_stride: u32,
    batch_dim: u32,
    #[specialize] has_rope: bool,
    #[specialize] reference_bf16: bool,
    #[specialize] source_rope: bool,
    #[specialize] source_proportional: bool,
    #[specialize] source_yarn: bool,
) {
    assert!(!source_rope || (has_rope && !reference_bf16), "invalid source RoPE profile");
    assert!(!source_yarn || (source_rope && source_proportional));
    assert_eq!(source_yarn_parameters.is_some(), source_yarn);
    assert_eq!(source_positions.is_some(), source_rope);
    assert_eq!(source_base.is_some(), source_rope);
    assert_eq!(source_rotary_dim.is_some(), source_rope);
    assert!(num_kv_heads > 0, "KV-only prepare requires nonzero KV heads");
    assert!(head_dim > 0, "KV-only prepare requires nonzero head_dim");
    assert_eq!(cosines.is_some(), has_rope, "KV-only prepare cosines presence mismatch");
    assert_eq!(sines.is_some(), has_rope, "KV-only prepare sines presence mismatch");
    assert_eq!(rope_dim.is_some(), has_rope, "KV-only prepare rope_dim presence mismatch");
    let (cosines, sines, rope_dim) = if let (Some(cosines), Some(sines), Some(rope_dim)) = (cosines, sines, rope_dim) {
        assert!(rope_dim > 0 && rope_dim <= head_dim && rope_dim.is_multiple_of(2), "invalid KV-only RoPE dimension");
        (cosines, sines, rope_dim as usize)
    } else {
        (std::ptr::null(), std::ptr::null(), 0)
    };
    let num_q_heads = num_q_heads as usize;
    let num_kv_heads = num_kv_heads as usize;
    let head_dim = head_dim as usize;
    let batch_dim = batch_dim as usize;
    let input_row_stride = input_row_stride as usize;
    let kv_token_offset = kv_token_offset as usize;
    let total_heads = num_q_heads + 2 * num_kv_heads;
    assert!(input_row_stride >= total_heads * head_dim, "KV-only prepare row stride is too small");

    for batch_idx in 0..batch_dim {
        for head_idx in num_q_heads..total_heads {
            let qkvg_head = unsafe { qkvg.add(batch_idx * input_row_stride + head_idx * head_dim) };
            let is_key = head_idx < num_q_heads + num_kv_heads;
            for head_dim_idx in 0..head_dim {
                let mut element = unsafe { *qkvg_head.add(head_dim_idx) };
                if has_rope && head_dim_idx < rope_dim && is_key {
                    element = if source_rope {
                        apply_source_rope(
                            qkvg_head,
                            unsafe { *source_positions.unwrap().add(batch_idx) },
                            head_dim_idx,
                            rope_dim,
                            source_base.unwrap(),
                            source_rotary_dim.unwrap(),
                            source_proportional,
                            source_yarn,
                            source_yarn_parameters,
                        )
                    } else {
                        apply_rope(qkvg_head, cosines, sines, batch_idx, head_dim_idx, rope_dim, reference_bf16)
                    };
                }
                if is_key {
                    let key_offset = (kv_token_offset + batch_idx) * num_kv_heads * head_dim
                        + (head_idx - num_q_heads) * head_dim
                        + head_dim_idx;
                    unsafe {
                        *keys.add(key_offset) = element;
                    }
                } else {
                    let value_offset = (kv_token_offset + batch_idx) * num_kv_heads * head_dim
                        + (head_idx - num_q_heads - num_kv_heads) * head_dim
                        + head_dim_idx;
                    unsafe {
                        *values.add(value_offset) = element;
                    }
                }
            }
        }
    }
}

#[kernel(AttentionPrepare)]
#[variants(ElementT, bf16)]
#[variants(RopeT, f32)]
pub fn attention_prepare<ElementT: ArrayElement + Float, RopeT: ArrayElement + Float>(
    qkvg: *const ElementT,
    queries: *mut ElementT,
    #[optional(has_kv)] keys: Option<*mut ElementT>,
    #[optional(has_kv)] values: Option<*mut ElementT>,
    #[optional(has_rope)] cosines: Option<*const RopeT>,
    #[optional(has_rope)] sines: Option<*const RopeT>,
    #[optional(source_rope)] source_positions: Option<*const u32>,
    #[optional(source_rope)] source_base: Option<f32>,
    #[optional(source_rope)] source_rotary_dim: Option<u32>,
    #[optional(source_yarn)] source_yarn_parameters: Option<*const f32>,
    num_q_heads: u32,
    #[optional(has_kv)] num_kv_heads: Option<u32>,
    head_dim: u32,
    #[optional(has_rope)] rope_dim: Option<u32>,
    #[optional(has_kv)] kv_token_offset: Option<u32>,
    input_row_stride: u32,
    batch_dim: u32,
    #[specialize] has_kv: bool,
    #[specialize] has_rope: bool,
    #[specialize] reference_bf16: bool,
    #[specialize] source_rope: bool,
    #[specialize] source_proportional: bool,
    #[specialize] source_yarn: bool,
) {
    assert!(!source_rope || (has_rope && !reference_bf16), "invalid source RoPE profile");
    assert!(!source_yarn || (source_rope && source_proportional));
    assert_eq!(source_yarn_parameters.is_some(), source_yarn);
    assert_eq!(source_positions.is_some(), source_rope);
    assert_eq!(source_base.is_some(), source_rope);
    assert_eq!(source_rotary_dim.is_some(), source_rope);
    assert!(num_q_heads > 0 || has_kv, "attention prepare without KV requires at least one query head");
    assert!(head_dim > 0, "attention prepare requires nonzero head_dim");
    assert_eq!(keys.is_some(), has_kv, "attention prepare keys presence mismatch");
    assert_eq!(values.is_some(), has_kv, "attention prepare values presence mismatch");
    assert_eq!(num_kv_heads.is_some(), has_kv, "attention prepare num_kv_heads presence mismatch");
    assert_eq!(kv_token_offset.is_some(), has_kv, "attention prepare kv_token_offset presence mismatch");
    assert_eq!(cosines.is_some(), has_rope, "attention prepare cosines presence mismatch");
    assert_eq!(sines.is_some(), has_rope, "attention prepare sines presence mismatch");
    assert_eq!(rope_dim.is_some(), has_rope, "attention prepare rope_dim presence mismatch");

    let (keys, values, num_kv_heads, kv_token_offset) =
        if let (Some(keys), Some(values), Some(num_kv_heads), Some(kv_token_offset)) =
            (keys, values, num_kv_heads, kv_token_offset)
        {
            assert!(num_kv_heads > 0, "attention prepare has_kv requires nonzero num_kv_heads");
            (keys, values, num_kv_heads as usize, kv_token_offset as usize)
        } else {
            (std::ptr::null_mut(), std::ptr::null_mut(), 0, 0)
        };

    let (cosines, sines, rope_dim) = if let (Some(cosines), Some(sines), Some(rope_dim)) = (cosines, sines, rope_dim) {
        assert!(rope_dim > 0, "attention prepare has_rope requires nonzero rope_dim");
        assert!(rope_dim <= head_dim, "attention prepare rope_dim exceeds head_dim");
        assert!(rope_dim.is_multiple_of(2), "attention prepare rope_dim must be even");
        (cosines, sines, rope_dim as usize)
    } else {
        (std::ptr::null(), std::ptr::null(), 0)
    };

    let num_q_heads = num_q_heads as usize;
    let head_dim = head_dim as usize;
    let batch_dim = batch_dim as usize;
    let input_row_stride = input_row_stride as usize;
    let total_heads = if has_kv {
        num_q_heads + num_kv_heads * 2
    } else {
        num_q_heads
    };
    assert!(input_row_stride >= total_heads * head_dim, "attention prepare row stride is too small");

    for batch_idx in 0..batch_dim {
        for head_idx in 0..total_heads {
            let qkvg_head = unsafe { qkvg.add(batch_idx * input_row_stride + head_idx * head_dim) };
            let is_query = !has_kv || head_idx < num_q_heads;
            let is_key = has_kv && head_idx >= num_q_heads && head_idx < num_q_heads + num_kv_heads;

            for head_dim_idx in 0..head_dim {
                let mut element = unsafe { *qkvg_head.add(head_dim_idx) };
                if has_rope && head_dim_idx < rope_dim && (is_query || is_key) {
                    element = if source_rope {
                        apply_source_rope(
                            qkvg_head,
                            unsafe { *source_positions.unwrap().add(batch_idx) },
                            head_dim_idx,
                            rope_dim,
                            source_base.unwrap(),
                            source_rotary_dim.unwrap(),
                            source_proportional,
                            source_yarn,
                            source_yarn_parameters,
                        )
                    } else {
                        apply_rope(qkvg_head, cosines, sines, batch_idx, head_dim_idx, rope_dim, reference_bf16)
                    };
                }

                if is_query {
                    let query_offset = head_idx * batch_dim * head_dim + batch_idx * head_dim + head_dim_idx;
                    unsafe {
                        *queries.add(query_offset) = element;
                    }
                } else if is_key {
                    let key_offset = (kv_token_offset + batch_idx) * num_kv_heads * head_dim
                        + (head_idx - num_q_heads) * head_dim
                        + head_dim_idx;
                    unsafe {
                        *keys.add(key_offset) = element;
                    }
                } else {
                    let value_offset = (kv_token_offset + batch_idx) * num_kv_heads * head_dim
                        + (head_idx - num_q_heads - num_kv_heads) * head_dim
                        + head_dim_idx;
                    unsafe {
                        *values.add(value_offset) = element;
                    }
                }
            }
        }
    }
}
