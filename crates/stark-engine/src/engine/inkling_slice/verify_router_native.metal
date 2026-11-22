// SPDX-License-Identifier: MIT
// Copyright © 2025 Apple Inc.
// Retains the 2048 + 256 router split of the source-derived MLX path in
// prefill_dense_source.metal, upstream fc344c986c4169d99f76b5d95ae6b4a58a359a63.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void ink_verify_router_native(
    device bfloat *input [[buffer(0)]], device bfloat *weight [[buffer(1)]],
    device float *partials [[buffer(2)]], constant uint &rows [[buffer(3)]],
    uint2 tile [[threadgroup_position_in_grid]]) {
    constexpr int width = 2304, experts = 128, major = 2048;
    constexpr int tile_rows = 16, columns = 8;
    const uint base = tile.y * major, channels = tile.y == 0 ? major : width - major;
    const uint expert = tile.x * columns;
    using dims = dextents<int, 2>;
    using strides = array<int, 2>;
    using operand = tensor<device bfloat, dims, tensor_inline>;
    using result = tensor<device float, dims, tensor_inline>;
    auto a = operand(input + base, dims(int(channels), int(rows)), strides{1, width});
    auto b = operand(weight + expert * width + base, dims(int(channels), columns), strides{1, width});
    auto c = result(partials + tile.y * rows * experts + expert, dims(columns, int(rows)), strides{1, experts});
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        tile_rows, columns, int(dynamic_extent), false, true, true);
    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroup> product;
    product.run(a, b, c);
}

kernel void ink_verify_router_join(
    device const float *partials [[buffer(0)]], device float *logits [[buffer(1)]],
    constant uint &words [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    logits[i] = partials[i] + partials[words + i];
}
