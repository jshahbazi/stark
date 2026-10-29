// Adapted from MLX v0.32.3 softmax.h, revision
// 64ea011cb65f14d9ce2737e60db9a4ae91ed7441.
// Copyright (c) 2023-2024 Apple Inc. MIT license: LICENSE.mlx-reference.
#include "../common/dsl.h"
#include <metal_stdlib>

using namespace metal;

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(on)

// Source-profile BF16 probabilities use a global maximum followed by a
// separate F32 exponential reduction. Online rebasing changes BF16 ties.
template <uint THREAD_COUNT>
VARIANTS(THREAD_COUNT, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512,
         544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024)
KERNEL(SourceBF16Softmax)(
    device bfloat* values,
    constant uint& row_width,
    constant uint& row_count,
    threadgroup float partial_maxima[32],
    threadgroup float partial_sums[32],
    const uint row GROUPS(row_count),
    const uint thread_index THREADS(THREAD_COUNT)
) {
  const uint lane = thread_index % 32;
  const uint group = thread_index / 32;
  const uint groups = THREAD_COUNT / 32;
  device bfloat* input = values + ulong(row) * row_width;
  float maximum = -INFINITY;
  for (uint base = thread_index * 4; base < row_width; base += THREAD_COUNT * 4) {
#pragma unroll
    for (uint i = 0; i < 4; ++i) {
      if (base + i < row_width) {
        maximum = max(maximum, float(input[base + i]));
      }
    }
  }
  maximum = simd_max(maximum);
  if (lane == 0) {
    partial_maxima[group] = maximum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (group == 0) {
    const float group_maximum = lane < groups ? partial_maxima[lane] : -INFINITY;
    const float row_maximum = simd_max(group_maximum);
    if (lane == 0) {
      partial_maxima[0] = row_maximum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maximum = partial_maxima[0];

  float sum = 0.0f;
  for (uint base = thread_index * 4; base < row_width; base += THREAD_COUNT * 4) {
#pragma unroll
    for (uint i = 0; i < 4; ++i) {
      if (base + i < row_width) {
        sum += fast::exp(float(input[base + i]) - maximum);
      }
    }
  }
  sum = simd_sum(sum);
  if (lane == 0) {
    partial_sums[group] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (group == 0) {
    const float group_sum = lane < groups ? partial_sums[lane] : 0.0f;
    const float row_sum = simd_sum(group_sum);
    if (lane == 0) {
      partial_sums[0] = row_sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inverse = 1.0f / partial_sums[0];
  for (uint base = thread_index * 4; base < row_width; base += THREAD_COUNT * 4) {
#pragma unroll
    for (uint i = 0; i < 4; ++i) {
      if (base + i < row_width) {
        const float probability = fast::exp(float(input[base + i]) - maximum) * inverse;
        input[base + i] = bfloat(probability);
      }
    }
  }
}
