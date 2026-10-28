// Numerical compatibility implementation of the BF16 RowwiseMoments reduction
// schedule in PyTorch v2.14.1, aten/src/ATen/native/cpu/moments_utils.h and
// layer_norm_kernel.cpp. See scripts/licenses/distribution/pytorch-LICENSE for BSD notices.

#include <metal_stdlib>
#include "../common/dsl.h"

// Only explicit fma calls in the Welford reduction may contract. The reference
// affine rounds each FP32 multiply/add before its single final BF16 conversion.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

struct VisionMoments {
  uint count;
  float4 mean;
  float4 squared_deviation;
};

static void vision_combine_moments(thread VisionMoments& target, VisionMoments other) {
  const uint old_count = target.count;
  const uint count = old_count + other.count;
  const float fraction = count == 0 ? 0.0f : float(other.count) / float(count);
  const float4 delta = other.mean - target.mean;
  const float4 weighted_delta = fraction * delta;
  const float4 old_delta = delta * float(old_count);
  const float4 deviations = target.squared_deviation + other.squared_deviation;
  target.mean = target.mean + weighted_delta;
  target.squared_deviation = fma(old_delta, weighted_delta, deviations);
  target.count = count;
}

PUBLIC KERNEL(VisionLayerNormBF16)(
    const device bfloat* input,
    const device bfloat* weight,
    const device bfloat* bias,
    device bfloat* output,
    constant uint& rows,
    constant uint& width,
    constant float& epsilon,
    const uint row AXIS(rows, 32)
) {
  const device bfloat* row_input = input + row * width;
  const uint vectors = width / 8;
  const uint chunks = (vectors + 15) / 16;
  uint depth = 0;
  for (uint power = 1; power < chunks; power <<= 1) {
    ++depth;
  }
  // VisionTowerConfig bounds a row below 2^20 elements, hence depth <= 13.
  VisionMoments stack[16] = {};
  for (uint chunk = 0; chunk < chunks; ++chunk) {
    const uint count = min(16u, vectors - chunk * 16);
    VisionMoments halves[2] = {};
    for (uint index = 0; index < count; ++index) {
      const float fraction = 1.0f / float(index + 1);
      for (uint part = 0; part < 2; ++part) {
        const uint start = chunk * 128 + index * 8 + part * 4;
        const float4 value = float4(float(row_input[start]), float(row_input[start + 1]),
                                   float(row_input[start + 2]), float(row_input[start + 3]));
        const float4 delta = value - halves[part].mean;
        halves[part].mean = fma(delta, fraction, halves[part].mean);
        const float4 centered = value - halves[part].mean;
        halves[part].squared_deviation = fma(delta, centered, halves[part].squared_deviation);
      }
    }
    halves[0].count = count;
    halves[1].count = count;
    vision_combine_moments(stack[0], halves[0]);
    vision_combine_moments(stack[0], halves[1]);
    uint mask = chunk + 1;
    uint level = 1;
    while (level < depth && (mask & 1) == 0) {
      vision_combine_moments(stack[level], stack[level - 1]);
      stack[level - 1] = VisionMoments{};
      mask >>= 1;
      ++level;
    }
  }
  for (uint level = 1; level < depth; ++level) {
    vision_combine_moments(stack[0], stack[level]);
  }
  uint count = 0;
  float mean = 0.0f;
  float squared_deviation = 0.0f;
  for (uint index = vectors * 8; index < width; ++index) {
    const float value = float(row_input[index]);
    const float delta = value - mean;
    ++count;
    mean += delta / float(count);
    squared_deviation = fma(delta, value - mean, squared_deviation);
  }
  const uint lane_count = vectors * 2;
  for (uint lane = 0; lane < 4; ++lane) {
    const uint total = count + lane_count;
    const float fraction = total == 0 ? 0.0f : float(lane_count) / float(total);
    const float delta = stack[0].mean[lane] - mean;
    // The pinned scalar AddMoments contracts these expressions, unlike its
    // vector combine mean update and final affine.
    mean = fma(fraction, delta, mean);
    const float weighted_square = delta * delta * fraction;
    const float correction = fma(weighted_square, float(count), stack[0].squared_deviation[lane]);
    squared_deviation += correction;
    count = total;
  }
  const float inverse_std = 1.0f / metal::precise::sqrt(squared_deviation / float(width) + epsilon);
  const float offset = -inverse_std * mean;
  for (uint index = 0; index < width; ++index) {
    const float scaled = float(row_input[index]) * inverse_std;
    const float centered = scaled + offset;
    const float affine = centered * float(weight[index]);
    output[row * width + index] = bfloat(affine + float(bias[index]));
  }
}
