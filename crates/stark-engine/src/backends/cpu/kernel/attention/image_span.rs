//! Bounded online attention for a single complete bidirectional image block.
//!
//! Mask semantics follow Transformers v4.53.3 Gemma3 token_type_ids_mask_function
//! and masking_utils.create_sliding_window_causal_mask: OR the same-image block
//! with the ordinary causal/window mask. No score matrix is materialized.

use half::bf16;
use num_traits::Float;
use stark_engine_macros::kernel;

use crate::array::ArrayElement;

#[allow(clippy::too_many_arguments)]
pub(crate) fn image_key_visible(
    query: u32,
    key: u32,
    prefix: u32,
    ring_offset: u32,
    has_ring: u32,
    window: u32,
    image_start: u32,
    image_end: u32,
    image_respects_window: u32,
) -> bool {
    let key_position = if key < prefix && has_ring != 0 {
        (prefix + key - ring_offset) % prefix
    } else {
        key
    };
    let query_position = prefix + query;
    let causal = key_position <= query_position && (window == 0 || query_position - key_position < window);
    let same_image = key >= prefix
        && query >= image_start
        && query < image_end
        && key - prefix >= image_start
        && key - prefix < image_end;
    let within_lower_window = window == 0 || key_position > query_position || query_position - key_position < window;
    (causal || same_image) && (image_respects_window == 0 || within_lower_window)
}

#[kernel(AttentionImageSpan)]
#[variants(T, f32, bf16)]
#[variants(HEAD_DIM, 64, 128, 256, 512)]
fn attention_image_span<T: ArrayElement + Float, const HEAD_DIM: u32>(
    queries: *const T,
    keys: *const T,
    values: *const T,
    output: *mut T,
    gqa_factor: u32,
    num_heads: u32,
    suffix_length: u32,
    prefix_length: u32,
    ring_offset: u32,
    has_ring: u32,
    sliding_window_size: u32,
    image_start: u32,
    image_end: u32,
    image_respects_window: u32,
    scale: f32,
) {
    let width = HEAD_DIM as usize;
    let kv_stride = (num_heads / gqa_factor) as usize * width;
    for head in 0..num_heads {
        for row in 0..suffix_length {
            let q_offset = (head * suffix_length + row) as usize * width;
            let kv_offset = (head / gqa_factor) as usize * width;
            let mut accumulator = vec![0.0f32; width];
            let mut maximum = f32::NEG_INFINITY;
            let mut denominator = 0.0f32;
            for key in 0..prefix_length + suffix_length {
                if !image_key_visible(
                    row,
                    key,
                    prefix_length,
                    ring_offset,
                    has_ring,
                    sliding_window_size,
                    image_start,
                    image_end,
                    image_respects_window,
                ) {
                    continue;
                }
                let offset = key as usize * kv_stride + kv_offset;
                let mut score = 0.0;
                for dim in 0..width {
                    score += unsafe { (*queries.add(q_offset + dim)).to_f32().unwrap() }
                        * unsafe { (*keys.add(offset + dim)).to_f32().unwrap() };
                }
                score *= scale;
                let next_max = maximum.max(score);
                let prior_factor = (maximum - next_max).exp();
                let factor = (score - next_max).exp();
                denominator = denominator * prior_factor + factor;
                for (dim, accumulator) in accumulator.iter_mut().enumerate() {
                    *accumulator =
                        *accumulator * prior_factor + factor * unsafe { (*values.add(offset + dim)).to_f32().unwrap() };
                }
                maximum = next_max;
            }
            let out_offset = (row * num_heads + head) as usize * width;
            for (dim, value) in accumulator.into_iter().enumerate() {
                unsafe {
                    *output.add(out_offset + dim) = T::from(value / denominator).unwrap();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn same_image_visibility_extends_causal_and_window_masks_only_inside_block() {
        // A two-token image at batch rows 1..3. Enumerate the mask independently.
        let expected = [
            [true, true, true, false, false, false],
            [false, true, true, true, true, false],
            [false, false, true, true, true, false],
            [false, false, false, true, true, true],
        ];
        for (query, row) in expected.into_iter().enumerate() {
            for (key, expected) in row.into_iter().enumerate() {
                assert_eq!(image_key_visible(query as u32, key as u32, 2, 0, 0, 3, 1, 3, 0), expected);
            }
        }
        assert!(image_key_visible(1, 4, 2, 0, 0, 1, 1, 3, 0));
        assert!(!image_key_visible(0, 4, 2, 0, 0, 0, 1, 3, 0));
        assert!(!image_key_visible(1, 5, 2, 0, 0, 0, 1, 3, 0));
    }

    #[stark_test]
    fn image_attention_respects_ring_chronology() {
        // Ring slots [0,1,2] hold logical positions [2,0,1]. Only the newest
        // prefix row and current row fit this two-token sliding window.
        assert!(image_key_visible(0, 0, 3, 1, 1, 2, 1, 3, 0));
        assert!(!image_key_visible(0, 1, 3, 1, 1, 2, 1, 3, 0));
        assert!(!image_key_visible(0, 2, 3, 1, 1, 2, 1, 3, 0));
        assert!(image_key_visible(1, 5, 3, 1, 1, 2, 1, 3, 0));
    }
}
