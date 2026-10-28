#include <metal_stdlib>
#include "../common/dsl.h"

// F32 row-sum cascade adapted from PyTorch commit
// 5c4886908584029761b579af026dcfb627c84070, SumKernel.cpp.
// Mean division follows ReduceOps.cpp; sqrt/reciprocal order follows the
// pinned ARM NEON vec128_float_neon.h implementation at the same revision.
// See scripts/licenses/distribution/pytorch-LICENSE for BSD terms/copyrights.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

PUBLIC KERNEL(ReferenceRMSBF16)(
    const device bfloat* input OPTIONAL(!in_place),
    const device float* scales,
    device bfloat* output,
    device bfloat* shortcut OPTIONAL(copy_to_shortcut),
    constant uint& batch_size,
    constant uint& element_count,
    constant uint& row_stride,
    constant float& epsilon,
    constant float& scale_offset,
    constant uint& head_offset,
    constant uint& head_count,
    const bool in_place SPECIALIZE,
    const bool copy_to_shortcut SPECIALIZE,
    const bool residual_add SPECIALIZE,
    const uint index AXIS(batch_size * head_count, 32)
) {
  if (in_place) input = reinterpret_cast<const device bfloat*>(output);
  const uint base = (index / head_count) * row_stride + (head_offset + index % head_count) * element_count;
  if (copy_to_shortcut) {
    for (uint channel = 0; channel < element_count; ++channel) {
      bfloat value = input[base + channel];
      if (residual_add) value = bfloat(float(value) + float(shortcut[base + channel]));
      shortcut[base + channel] = value;
    }
  }
  const device bfloat* source = residual_add ? shortcut : input;
  const uint count = element_count / 16;
  const uint ceil_log2 = count > 1 ? 32 - clz(count - 1) : 0;
  const uint power = max(4u, ceil_log2 / 4);
  const uint step = 1u << power;
  const uint mask = step - 1;
  float4 levels[4][4];
  for (uint level = 0; level < 4; ++level)
    for (uint reg = 0; reg < 4; ++reg) levels[level][reg] = 0.0f;
  uint vector_index = 0;
  while (vector_index + step <= count) {
    for (uint j = 0; j < step; ++j, ++vector_index) {
      for (uint reg = 0; reg < 4; ++reg) {
        const uint offset = base + vector_index * 16 + reg * 4;
        const float4 x = float4(float(source[offset]), float(source[offset + 1]), float(source[offset + 2]), float(source[offset + 3]));
        levels[0][reg] += x * x;
      }
    }
    for (uint level = 1; level < 4; ++level) {
      for (uint reg = 0; reg < 4; ++reg) {
        levels[level][reg] += levels[level - 1][reg];
        levels[level - 1][reg] = 0.0f;
      }
      if ((vector_index & (mask << (level * power))) != 0) break;
    }
  }
  while (vector_index < count) {
    for (uint reg = 0; reg < 4; ++reg) {
      const uint offset = base + vector_index * 16 + reg * 4;
      const float4 x = float4(float(source[offset]), float(source[offset + 1]), float(source[offset + 2]), float(source[offset + 3]));
      levels[0][reg] += x * x;
    }
    ++vector_index;
  }
  for (uint level = 1; level < 4; ++level)
    for (uint reg = 0; reg < 4; ++reg) levels[0][reg] += levels[level][reg];
  float4 lanes = levels[0][0];
  for (uint reg = 1; reg < 4; ++reg) lanes += levels[0][reg];
  float sum = 0.0f;
  for (uint lane = 0; lane < 4; ++lane) sum += lanes[lane];
  const float mean = sum / float(element_count);
  const float inverse = 1.0f / metal::precise::sqrt(mean + epsilon);
  for (uint channel = 0; channel < element_count; ++channel) {
    const float normalized = float(source[base + channel]) * inverse;
    const float gain = scales[channel] + scale_offset;
    output[base + channel] = bfloat(normalized * gain);
  }
}
