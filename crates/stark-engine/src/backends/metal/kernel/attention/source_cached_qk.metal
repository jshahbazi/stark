// Adapted from MLX v0.32.3 gemv.h GemvWide, revision
// 64ea011cb65f14d9ce2737e60db9a4ae91ed7441.
// Copyright (c) 2023-2024 Apple Inc. MIT license: LICENSE.mlx-reference.
#include "../common/dsl.h"
#include <metal_stdlib>

using namespace metal;

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(on)

// The qualified source cache has eight contiguous D512 query heads and one
// contiguous KV head. Preserve the source float4 dots, eight-load staging,
// local fold and halving shuffle before the single BF16 score store.
template <uint K_LANES>
VARIANTS(K_LANES, 16, 32)
KERNEL(SourceCachedQK)(
    const device bfloat* queries,
    const device bfloat* keys,
    device bfloat* scores,
    const constant uint& sequence_length,
    const uint vector_tile GROUPS(2),
    const uint row_tile GROUPS(sequence_length.div_ceil(4)),
    const uint lane THREADS(32),
    const uint simd_group THREADS(K_LANES / 8)
) {
  constexpr uint VECTORS = 4;
  constexpr uint DIM = 512;
  constexpr uint UNROLL = 8;
  constexpr uint ROWS_PER_SIMDGROUP = 32 / K_LANES;
  constexpr uint VECTOR4_COUNT = DIM / 4;
  constexpr uint MAIN_COUNT = VECTOR4_COUNT - VECTOR4_COUNT % (K_LANES * UNROLL);
  const uint k_lane = lane % K_LANES;
  const uint output_row = row_tile * 4 + ROWS_PER_SIMDGROUP * simd_group + lane / K_LANES;
  const uint row = min(output_row, sequence_length - 1);
  const device bfloat4* w4 = reinterpret_cast<const device bfloat4*>(keys + ulong(row) * DIM);
  const device bfloat4* x4 = reinterpret_cast<const device bfloat4*>(queries);
  const uint vector0 = vector_tile * VECTORS;
  float result[VECTORS] = {0};

  for (uint base = 0; base < MAIN_COUNT; base += K_LANES * UNROLL) {
    float4 weights[UNROLL];
#pragma unroll
    for (uint i = 0; i < UNROLL; ++i) {
      weights[i] = float4(w4[base + i * K_LANES + k_lane]);
    }
#pragma unroll
    for (uint v = 0; v < VECTORS; ++v) {
      float partial = 0;
#pragma unroll
      for (uint i = 0; i < UNROLL; ++i) {
        partial += dot(weights[i], float4(x4[(vector0 + v) * VECTOR4_COUNT + base + i * K_LANES + k_lane]));
      }
      result[v] += partial;
    }
  }
  for (uint index = MAIN_COUNT + k_lane; index < VECTOR4_COUNT; index += K_LANES) {
    const float4 weights = float4(w4[index]);
#pragma unroll
    for (uint v = 0; v < VECTORS; ++v) {
      result[v] += dot(weights, float4(x4[(vector0 + v) * VECTOR4_COUNT + index]));
    }
  }
#pragma unroll
  for (uint v = 0; v < VECTORS; ++v) {
#pragma unroll
    for (ushort offset = K_LANES / 2; offset >= 1; offset >>= 1) {
      result[v] += simd_shuffle_down(result[v], offset);
    }
  }
  if (k_lane == 0 && output_row < sequence_length) {
#pragma unroll
    for (uint v = 0; v < VECTORS; ++v) {
      scores[ulong(vector0 + v) * sequence_length + output_row] = bfloat(result[v]);
    }
  }
}
