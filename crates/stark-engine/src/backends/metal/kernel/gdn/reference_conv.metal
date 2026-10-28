#include <metal_stdlib>
#include "../common/dsl.h"

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include "../common/vision_exp.h"

// BF16 causal conv1d is rounded before source SiLU x/(1+exp(-x)). The
// history snapshot and input projection remain immutable throughout dispatch.
PUBLIC KERNEL(ReferenceGDNConvBF16)(
    const device bfloat* input,
    const device float* history,
    const device float* weights,
    device bfloat* output,
    device float* state_out,
    constant uint& rows,
    constant uint& kernel_size,
    constant uint& input_row_stride,
    constant uint& output_stride,
    constant uint& state_stride,
    constant uint& conv_dim,
    const uint token AXIS(rows + state_stride, 1),
    const uint channel AXIS(output_stride, 32)
) {
  if (token < rows) {
    const uint destination = token * output_stride + channel;
    if (channel >= conv_dim) {
      output[destination] = input[token * input_row_stride + channel];
      return;
    }
    float sum = 0.0f;
    for (uint tap = 0; tap < kernel_size; ++tap) {
      const uint padded = token + tap;
      const float sample = padded < state_stride ? history[channel * state_stride + padded]
          : float(input[(padded - state_stride) * input_row_stride + channel]);
      sum = fma(weights[channel * kernel_size + tap], sample, sum);
    }
    const float convolution = float(bfloat(sum));
    output[destination] = bfloat(convolution / (1.0f + vision_exp(-convolution)));
  } else if (channel < conv_dim) {
    const uint tap = token - rows;
    const float sample = token < state_stride ? history[channel * state_stride + token]
        : float(input[(token - state_stride) * input_row_stride + channel]);
    state_out[channel * state_stride + tap] = sample;
  }
}
