#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"

using namespace metal;
#define GRAIN_SIZE 4

// Source profile follows MLX v0.32.3 rms_norm.metal (Apple Inc., MIT).
// Copyright 2024 Apple Inc. Notice retained in LICENSE.mlx-reference.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(on)
template <typename InputT>
METAL_FUNC float source_qkv_four(const device InputT* input, uint base, uint count) {
  float accum = 0.0f;
  for (uint j=0;j<4;++j) {
    const float x=base+j<count ? float(input[base+j]) : 0.0f;
    accum += x*x;
  }
  return accum;
}
METAL_FUNC float source_qkv_inverse(float sum,uint count,float epsilon) {
  const float mean=sum/float(count);
  return metal::precise::rsqrt(mean+epsilon);
}
#pragma METAL fp math_mode(fast)
#pragma METAL fp contract(on)

// QKV norm: normalize per-head vectors (small head_dim) efficiently.
//
// Strategy:
// - One SIMD-group (32 threads) processes one head.
// - One threadgroup (one SIMD-group) is dispatched per head.
template <typename InputT, typename ScaleT, typename OutputT, typename AccumT>
VARIANTS(InputT, float, bfloat)
VARIANTS(ScaleT, float, bfloat)
VARIANTS(OutputT, float, bfloat)
VARIANTS(AccumT, float)
PUBLIC KERNEL(QKVNorm)(
    const device InputT* qkvg_input OPTIONAL(!in_place),
    const device ScaleT* scales OPTIONAL(has_scales),
    device OutputT* qkvg_output,
    constant uint& batch_size,
    constant uint& input_row_stride,
    constant uint& head_dim,
    constant float& epsilon,
    constant float& scale_offset,
    constant uint& head_offset,
    constant uint& head_count,
    constant bool& full_layer,
    const uint batch_idx GROUPS(batch_size),
    const uint head_idx GROUPS(head_count),
    const uint lane_id THREADS(METAL_SIMD_SIZE),
    const bool in_place SPECIALIZE,
    const bool has_scales SPECIALIZE,
    const bool mlx_bf16_reduction SPECIALIZE
) {
  if (in_place) {
    qkvg_input = (const device InputT*)qkvg_output;
  }

  if (head_count == 0u || head_dim == 0u)
    return;

  const ulong slice_offset =
      (ulong)batch_idx * (ulong)input_row_stride + (ulong)(head_offset + head_idx) * (ulong)head_dim;

  const device InputT* input_data = qkvg_input + slice_offset;
  const device ScaleT* scales_data = scales;
  device OutputT* output_data = qkvg_output + slice_offset;
  const uint element_count = head_dim;

  AccumT partial_sum = static_cast<AccumT>(0.0f);

  AccumT rms_norm;
  if (mlx_bf16_reduction) {
    // Emulate the source logical SIMD groups in one hardware SIMD group.
    // Each logical group reduces four contiguous values/lane; lane g then
    // contributes group g's subtotal to the source final padded reduction.
    float selected = 0.0f;
    for (uint group=0;group<(element_count+127)/128;++group) {
      const uint base=group*128+lane_id*4;
      const float group_sum=simd_sum(source_qkv_four(input_data,base,element_count));
      if (lane_id==group) selected=group_sum;
    }
    const float total=simd_sum(selected);
    rms_norm=source_qkv_inverse(total,element_count,epsilon);
  } else {
  // Sum of squares: each lane processes GRAIN_SIZE elements per iteration.
  for (uint base_i = lane_id * GRAIN_SIZE; base_i < element_count; base_i += METAL_SIMD_SIZE * GRAIN_SIZE) {
    AccumT vals[GRAIN_SIZE];
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      vals[j] = (i < element_count) ? static_cast<AccumT>(input_data[i]) : 0.0f;
    }
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      partial_sum += vals[j] * vals[j];
    }
  }

  // SIMD-group reduction.
  AccumT total_sum = simd_sum(partial_sum);

  // RMS factor.
  AccumT mean_square = static_cast<AccumT>(total_sum) / static_cast<AccumT>(element_count);
  rms_norm = rsqrt(mean_square + static_cast<AccumT>(epsilon));

  }

  // Normalize + scale.
  for (uint base_i = lane_id * GRAIN_SIZE; base_i < element_count; base_i += METAL_SIMD_SIZE * GRAIN_SIZE) {
    AccumT vals[GRAIN_SIZE];
    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      vals[j] = (i < element_count) ? static_cast<AccumT>(input_data[i]) : 0.0f;
    }

    for (uint j = 0; j < GRAIN_SIZE; ++j) {
      uint i = base_i + j;
      if (i >= element_count)
        continue;

      AccumT normalized_high = vals[j] * rms_norm;

      if (!has_scales) {
        output_data[i] = static_cast<OutputT>(normalized_high);
      } else if (full_layer) {
        AccumT scale_value_high = static_cast<AccumT>(scales_data[i]) + static_cast<AccumT>(scale_offset);
        output_data[i] = static_cast<OutputT>(normalized_high * scale_value_high);
      } else {
        OutputT normalized_low = static_cast<OutputT>(normalized_high);
        OutputT scale_value_low =
            static_cast<OutputT>(static_cast<AccumT>(scales_data[i]) + static_cast<AccumT>(scale_offset));
        OutputT product_low = normalized_low * scale_value_low;
        output_data[i] = static_cast<OutputT>(product_low);
      }
    }
  }
}
