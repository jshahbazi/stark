// SPDX-License-Identifier: MIT
// Copyright © 2026 Apple Inc. (LICENSE.mlx-reference).
// Adapted from MLX gemv.h at fc344c986c4169d99f76b5d95ae6b4a58a359a63.

#include <metal_stdlib>
using namespace metal;

// Share each weight load across independent source-order dot products. Keep the
// serial BF16 projection's K-lane order and independent FP32 accumulators.
template <uint Batch>
static inline void ink_matvec_prefill_shared(
    device const bfloat4 *weight, thread const device bfloat4 *const *inputs,
    thread device bfloat *const *outputs, uint3 shape, uint group, uint simd, uint lane) {
    const uint lanes = shape.y <= 64 ? 32 : 16;
    const uint row = group * 4 + simd * (32 / lanes) + lane / lanes;
    const uint k_lane = lane % lanes;
    const uint cols = shape.x / 4;
    const uint stride = lanes * 8;
    const uint end = cols - cols % stride;
    const device bfloat4 *weights = weight + min(row, shape.y - 1) * cols;
    float sums[Batch] = {0};
    for (uint base = 0; base < end; base += stride) {
        float4 loaded[8];
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) loaded[i] = float4(weights[base + i * lanes + k_lane]);
#pragma clang loop unroll(full)
        for (uint vector = 0; vector < Batch; ++vector) {
            float block = 0.0f;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8; ++i) {
                const uint col = base + i * lanes + k_lane;
                block += dot(loaded[i], float4(inputs[vector][col]));
            }
            sums[vector] += block;
        }
    }
    for (uint col = end + k_lane; col < cols; col += lanes) {
        const float4 loaded = float4(weights[col]);
#pragma clang loop unroll(full)
        for (uint vector = 0; vector < Batch; ++vector) {
            sums[vector] += dot(loaded, float4(inputs[vector][col]));
        }
    }
#pragma clang loop unroll(full)
    for (uint vector = 0; vector < Batch; ++vector) {
        for (uint delta = lanes / 2; delta > 0; delta >>= 1) {
            sums[vector] += simd_shuffle_down(sums[vector], ushort(delta));
        }
        if (k_lane == 0 && row < shape.y && vector < shape.z) outputs[vector][row] = bfloat(sums[vector]);
    }
}

kernel void ink_matvec_prefill(
    device const bfloat4 *weight [[buffer(0)]],
    device const bfloat4 *x0 [[buffer(1)]], device const bfloat4 *x1 [[buffer(2)]],
    device const bfloat4 *x2 [[buffer(3)]], device const bfloat4 *x3 [[buffer(4)]],
    device bfloat *y0 [[buffer(5)]], device bfloat *y1 [[buffer(6)]],
    device bfloat *y2 [[buffer(7)]], device bfloat *y3 [[buffer(8)]],
    constant uint3 &shape [[buffer(9)]], uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    device const bfloat4 *inputs[] = {x0, x1, x2, x3};
    device bfloat *outputs[] = {y0, y1, y2, y3};
    ink_matvec_prefill_shared<4>(weight, inputs, outputs, shape, group, simd, lane);
}

kernel void ink_matvec_prefill_wide(
    device const bfloat4 *weight [[buffer(0)]],
    device const bfloat4 *x0 [[buffer(1)]], device const bfloat4 *x1 [[buffer(2)]],
    device const bfloat4 *x2 [[buffer(3)]], device const bfloat4 *x3 [[buffer(4)]],
    device const bfloat4 *x4 [[buffer(5)]], device const bfloat4 *x5 [[buffer(6)]],
    device const bfloat4 *x6 [[buffer(7)]], device const bfloat4 *x7 [[buffer(8)]],
    device bfloat *y0 [[buffer(9)]], device bfloat *y1 [[buffer(10)]],
    device bfloat *y2 [[buffer(11)]], device bfloat *y3 [[buffer(12)]],
    device bfloat *y4 [[buffer(13)]], device bfloat *y5 [[buffer(14)]],
    device bfloat *y6 [[buffer(15)]], device bfloat *y7 [[buffer(16)]],
    constant uint3 &shape [[buffer(17)]], uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    device const bfloat4 *inputs[] = {x0, x1, x2, x3, x4, x5, x6, x7};
    device bfloat *outputs[] = {y0, y1, y2, y3, y4, y5, y6, y7};
    ink_matvec_prefill_shared<8>(weight, inputs, outputs, shape, group, simd, lane);
}
