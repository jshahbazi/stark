use half::bf16;
use num_traits::Float;
use stark_engine_macros::kernel;

use crate::{array::ArrayElement, backends::common::gpu_types::ActivationType};

// CPU reference for the direct scan. Match Conv1dPack + DeltaNetConvScan,
// including retained history when the suffix is shorter than the state.
#[kernel(DeltaNetConvDirect)]
#[variants(T, f32, bf16)]
pub fn delta_net_conv_direct<T: ArrayElement + Float>(
    input: *const T,
    state_in: *const f32,
    conv_weight: *const f32,
    #[optional(has_bias)] bias: Option<*const f32>,
    output: *mut T,
    state_out: *mut f32,
    suffix_len: u32,
    kernel_size: u32,
    row_stride: u32,
    output_stride: u32,
    state_stride: u32,
    conv_dim: u32,
    #[specialize] has_bias: bool,
) {
    let suffix_len = suffix_len as usize;
    let kernel_size = kernel_size as usize;
    let row_stride = row_stride as usize;
    let output_stride = output_stride as usize;
    let state_stride = state_stride as usize;
    let conv_dim = conv_dim as usize;
    for token in 0..suffix_len {
        for channel in 0..output_stride {
            if channel >= conv_dim {
                unsafe { *output.add(token * output_stride + channel) = *input.add(token * row_stride + channel) };
                continue;
            }
            let mut acc = if has_bias {
                unsafe { *bias.unwrap().add(channel) }
            } else {
                0.0f32
            };
            for tap in 0..kernel_size {
                let padded_row = token + tap;
                let sample = if padded_row < state_stride {
                    unsafe { *state_in.add(channel * state_stride + padded_row) }
                } else {
                    unsafe { (*input.add((padded_row - state_stride) * row_stride + channel)).to_f32().unwrap() }
                };
                acc += unsafe { *conv_weight.add(channel * kernel_size + tap) } * sample;
            }
            unsafe {
                *output.add(token * output_stride + channel) = T::from(ActivationType::SILU.activate(acc)).unwrap();
            }
        }
    }
    for channel in 0..conv_dim {
        for tap in 0..state_stride {
            let padded_row = suffix_len + tap;
            let sample = if padded_row < state_stride {
                unsafe { *state_in.add(channel * state_stride + padded_row) }
            } else {
                unsafe { (*input.add((padded_row - state_stride) * row_stride + channel)).to_f32().unwrap() }
            };
            unsafe { *state_out.add(channel * state_stride + tap) = sample };
        }
    }
}
