#include <metal_stdlib>
#include "../activation/activations.h"
#include "../common/dsl.h"

using namespace metal;

// The original projection and history snapshot are immutable and disjoint from
// the output projection and next state. Preserve the packed scan's tap order,
// F32 accumulation, SiLU, and final cast without materializing F32 token rows.
template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(DeltaNetConvDirect)(
    device const T* input,
    device const float* state_in,
    device const float* conv_weight,
    device const float* bias OPTIONAL(has_bias),
    device T* output,
    device float* state_out,
    constant const uint& suffix_len,
    constant const uint& kernel_size,
    constant const uint& row_stride,
    constant const uint& output_stride,
    constant const uint& state_stride,
    constant const uint& conv_dim,
    constant const uint& reference_bf16,
    const bool has_bias SPECIALIZE,
    const uint token_idx AXIS(suffix_len + kernel_size - 1, 1),
    const uint channel_idx AXIS(row_stride, 32)
) {
  if (token_idx < suffix_len) {
    if (channel_idx >= output_stride) {
      return;
    }
    const size_t dst = size_t(token_idx) * output_stride + channel_idx;
    if (channel_idx >= conv_dim) {
      output[dst] = input[size_t(token_idx) * row_stride + channel_idx];
      return;
    }
    const device float* weight_row = conv_weight + channel_idx * kernel_size;
    float acc = has_bias ? float(bias[channel_idx]) : 0.0f;
    for (uint tap = 0; tap < kernel_size; ++tap) {
      const uint padded_row = token_idx + tap;
      const float sample = padded_row < state_stride
          ? state_in[channel_idx * state_stride + padded_row]
          : float(input[(padded_row - state_stride) * row_stride + channel_idx]);
      acc += float(weight_row[tap]) * sample;
    }
    const float convolution = reference_bf16 != 0 ? float(bfloat(acc)) : acc;
    output[dst] = static_cast<T>(activate_silu(convolution));
  } else if (channel_idx < conv_dim) {
    const uint tap = token_idx - suffix_len;
    // Small suffixes can retain history samples in their next state.
    const float sample = token_idx < state_stride
        ? state_in[channel_idx * state_stride + token_idx]
        : float(input[(token_idx - state_stride) * row_stride + channel_idx]);
    state_out[channel_idx * state_stride + tap] = sample;
  }
}
