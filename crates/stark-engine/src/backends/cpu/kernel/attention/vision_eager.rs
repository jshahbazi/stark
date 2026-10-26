use half::bf16;
use stark_engine_macros::kernel;

/// Pinned eager vision attention keeps the complete softmax in FP32 before
/// converting normalized probabilities to BF16 once.
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
        let mut total = 0.0f32;
        for column in 0..row_length as usize {
            total += (unsafe { (*values.add(start + column)).to_f32() } - maximum).exp();
        }
        for column in 0..row_length as usize {
            let probability = (unsafe { (*values.add(start + column)).to_f32() } - maximum).exp() / total;
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
