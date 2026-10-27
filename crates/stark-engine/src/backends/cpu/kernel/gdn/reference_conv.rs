use half::bf16;
use stark_engine_macros::kernel;

use crate::backends::common::vision_math::vision_exp;

/// Exact-artifact causal BF16 conv1d boundary followed by source SiLU.
/// The immutable history snapshot is disjoint from the next-state output.
#[kernel(ReferenceGDNConvBF16)]
pub fn reference_gdn_conv_bf16(
    input: *const bf16,
    history: *const f32,
    weights: *const f32,
    output: *mut bf16,
    state_out: *mut f32,
    rows: u32,
    kernel_size: u32,
    input_row_stride: u32,
    output_stride: u32,
    state_stride: u32,
    conv_dim: u32,
) {
    assert!(rows > 0 && kernel_size == state_stride + 1);
    assert!(input_row_stride >= output_stride && output_stride >= conv_dim);
    let (rows, taps, input_stride, output_stride, state_stride, conv_dim) = (
        rows as usize,
        kernel_size as usize,
        input_row_stride as usize,
        output_stride as usize,
        state_stride as usize,
        conv_dim as usize,
    );
    for row in 0..rows {
        for channel in 0..output_stride {
            if channel >= conv_dim {
                unsafe {
                    *output.add(row * output_stride + channel) = *input.add(row * input_stride + channel);
                }
                continue;
            }
            let mut sum = 0.0f32;
            for tap in 0..taps {
                let padded = row + tap;
                let sample = if padded < state_stride {
                    unsafe { *history.add(channel * state_stride + padded) }
                } else {
                    unsafe { (*input.add((padded - state_stride) * input_stride + channel)).to_f32() }
                };
                sum = unsafe { *weights.add(channel * taps + tap) }.mul_add(sample, sum);
            }
            let convolution = bf16::from_f32(sum).to_f32();
            let activated = convolution / (1.0 + vision_exp(-convolution));
            unsafe {
                *output.add(row * output_stride + channel) = bf16::from_f32(activated);
            }
        }
    }
    for channel in 0..conv_dim {
        for tap in 0..state_stride {
            let padded = rows + tap;
            let sample = if padded < state_stride {
                unsafe { *history.add(channel * state_stride + padded) }
            } else {
                unsafe { (*input.add((padded - state_stride) * input_stride + channel)).to_f32() }
            };
            unsafe {
                *state_out.add(channel * state_stride + tap) = sample;
            }
        }
    }
}
