// Adapted from MLX v0.32.3 affine qmv_wide_impl and dequantize.
// Source revision: 64ea011cb65f14d9ce2737e60db9a4ae91ed7441.
// Copyright (c) 2023-2024 Apple Inc. Licensed under the MIT license.
// The upstream license is retained in LICENSE.mlx-reference.
#include "../../common/dsl.h"
#include <metal_stdlib>

using namespace metal;

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(on)

// Source-profile prefill intentionally keeps dequantized weights and dot
// products in F32. Single-token decoding has a different reduction contract.
template <uint GROUP_SIZE, uint VECTORS>
VARIANTS(GROUP_SIZE, 16, 32, 64, 128)
VARIANTS(VECTORS, 2, 3, 4, 5)
KERNEL(SourceWideQmv)(
    const device uchar* weights,
    const device bfloat* scales,
    const device bfloat* biases,
    const device bfloat* input,
    device bfloat* output,
    const constant uint& k,
    const constant uint& n,
    const constant uint& m,
    const constant uint& weight_row_bytes,
    const constant uint& scale_output_stride,
    const constant uint& scale_group_stride,
    const uint vector_tile GROUPS(m.div_ceil(VECTORS)),
    const uint row_tile GROUPS(n.div_ceil(8)),
    const uint lane THREADS(32),
    const uint simd_group THREADS(2)
) {
  const uint k_lane = lane % 8;
  const uint output_row = row_tile * 8 + simd_group * 4 + lane / 8;
  const uint row = min(output_row, n - 1);
  const uint first_vector = vector_tile * VECTORS;
  const device uchar* w = weights + row * weight_row_bytes;
  const device bfloat* s = scales + row * scale_output_stride;
  const device bfloat* b = biases + row * scale_output_stride;
  float result[VECTORS] = {0};

  for (uint group = k_lane; group < k / GROUP_SIZE; group += 8) {
    const float scale = float(s[group * scale_group_stride]);
    const float bias = float(b[group * scale_group_stride]);
#pragma unroll
    for (uint subchunk = 0; subchunk < GROUP_SIZE / 8; ++subchunk) {
      const uint k0 = group * GROUP_SIZE + subchunk * 8;
      const device uchar* packed = w + k0 / 2;
      float decoded[8];
#pragma unroll
      for (uint i = 0; i < 4; ++i) {
        decoded[2 * i] = scale * float(packed[i] & 0x0f) + bias;
        decoded[2 * i + 1] = (scale / 16.0f) * float(packed[i] & 0xf0) + bias;
      }
#pragma unroll
      for (uint v = 0; v < VECTORS; ++v) {
        const device bfloat* x = input + min(first_vector + v, m - 1) * k + k0;
        float partial = 0;
#pragma unroll
        for (uint i = 0; i < 8; ++i) {
          partial += float(x[i]) * decoded[i];
        }
        result[v] += partial;
      }
    }
  }
#pragma unroll
  for (uint v = 0; v < VECTORS; ++v) {
    result[v] += simd_shuffle_down(result[v], 4);
    result[v] += simd_shuffle_down(result[v], 2);
    result[v] += simd_shuffle_down(result[v], 1);
  }
  if (k_lane == 0 && output_row < n) {
#pragma unroll
    for (uint v = 0; v < VECTORS; ++v) {
      if (first_vector + v < m) {
        output[(first_vector + v) * n + output_row] = bfloat(result[v]);
      }
    }
  }
}
