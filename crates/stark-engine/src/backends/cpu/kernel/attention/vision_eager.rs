use half::bf16;
use stark_engine_macros::kernel;

/// Pinned BF16 text attention rounds the dot product and its scalar multiply
/// separately, then adds a BF16 causal mask before FP32 softmax.
#[kernel(ReferenceCausalScoresBF16)]
pub fn reference_causal_scores_bf16(
    input: *const f32,
    output: *mut bf16,
    key_count: u32,
    rows: u32,
    prefix_len: u32,
    scale: f32,
) {
    for row in 0..rows as usize {
        for column in 0..key_count as usize {
            let index = row * key_count as usize + column;
            let value = if column <= prefix_len as usize + row {
                bf16::from_f32(unsafe { bf16::from_f32(*input.add(index)).to_f32() } * scale)
            } else {
                bf16::from_bits(0xff7f)
            };
            unsafe {
                *output.add(index) = value;
            }
        }
    }
}

use crate::backends::common::vision_math::vision_exp;

/// The pinned FP32 softmax reduces four lanes, folds crosswise, and multiplies
/// by the reciprocal before the eager attention's single BF16 conversion.
#[kernel(VisionSoftmaxBF16)]
pub fn vision_softmax_bf16(
    values: *mut bf16,
    row_length: u32,
    rows: u32,
) {
    for row in 0..rows as usize {
        let start = row * row_length as usize;
        let mut maximum = f32::NEG_INFINITY;
        for column in 0..row_length as usize {
            maximum = maximum.max(unsafe { (*values.add(start + column)).to_f32() });
        }
        let mut partial = [0.0f32; 4];
        for column in 0..row_length as usize {
            partial[column % 4] += vision_exp(unsafe { (*values.add(start + column)).to_f32() } - maximum);
        }
        let total = if row_length >= 4 {
            (partial[0] + partial[2]) + (partial[1] + partial[3])
        } else {
            partial[..row_length as usize].iter().copied().reduce(|sum, x| sum + x).unwrap_or(0.0)
        };
        let inverse_total = total.recip();
        for column in 0..row_length as usize {
            let probability = vision_exp(unsafe { (*values.add(start + column)).to_f32() } - maximum) * inverse_total;
            unsafe {
                *values.add(start + column) = bf16::from_f32(probability);
            }
        }
    }
}

/// Interleave a contiguous [row, head_dim] result into [row, heads, head_dim].
#[kernel(VisionHeadCopyBF16)]
pub fn vision_head_copy_bf16(
    input: *const bf16,
    output: *mut bf16,
    length: u32,
    head_dim: u32,
    model_dim: u32,
    head_index: u32,
) {
    for index in 0..length as usize {
        let destination = index / head_dim as usize * model_dim as usize
            + head_index as usize * head_dim as usize
            + index % head_dim as usize;
        unsafe {
            *output.add(destination) = *input.add(index);
        }
    }
}
