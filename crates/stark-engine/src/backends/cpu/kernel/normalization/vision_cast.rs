use half::bf16;
use stark_engine_macros::kernel;

/// Preserve the vision LayerNorm's FP32 affine+bias result until its final cast.
#[kernel(VisionCastBF16)]
pub fn vision_cast_bf16(
    input: *const f32,
    output: *mut bf16,
    length: u32,
) {
    for index in 0..length as usize {
        unsafe {
            *output.add(index) = bf16::from_f32(*input.add(index));
        }
    }
}

/// Preserve source BF16 bias values while applying the complete FP32 affine
/// before the vision operation's single final BF16 conversion.
#[kernel(VisionBiasCastBF16)]
pub fn vision_bias_cast_bf16(
    input: *const f32,
    bias: *const bf16,
    output: *mut bf16,
    row_width: u32,
    length: u32,
) {
    for index in 0..length as usize {
        unsafe {
            *output.add(index) = bf16::from_f32(*input.add(index) + (*bias.add(index % row_width as usize)).to_f32());
        }
    }
}
