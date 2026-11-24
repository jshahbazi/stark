// SPDX-License-Identifier: MIT
// Copyright © 2026 Apple Inc.
// Adapted from MLX steel_gemm_thin_nax.h at
// fc344c986c4169d99f76b5d95ae6b4a58a359a63 (LICENSE.mlx-reference).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// Short target projections run as an eight-row GEMM even for one live row.
// Each SIMD group accumulates every fourth K tile into its own FP32 result.
kernel void ink_short_projection(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weights [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant uint2 &shape [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint shard [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int tile_m = 32;
    constexpr int tile_n = 32;
    constexpr int tile_k = 64;
    constexpr int live_m = 8;
    constexpr int shards = 4;
    constexpr int tile_size = tile_m * tile_n;
    const int width = int(shape.x);
    const int outputs = int(shape.y);
    const int first = int(group) * tile_n;
    // Keep the reference's final full-width B view when it overlaps a prior
    // tile. Only write the columns that belong to this group.
    const int start = min(first, max(0, outputs - tile_n));
    const int columns = min(tile_n, outputs - start);

    using dims = dextents<int, 2>;
    using strides = array<int, 2>;
    using device_bf16 = tensor<device bfloat, dims, tensor_inline>;
    using group_bf16 = tensor<threadgroup bfloat, dims, tensor_inline>;
    using group_f32 = tensor<threadgroup float, dims, tensor_inline>;
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        tile_m, tile_n, tile_k, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroup> mma;
    using a_tile = decltype(metal::declval<group_bf16>().slice<tile_k, tile_m>(0, 0));
    using b_tile = decltype(metal::declval<device_bf16>().slice<tile_k, tile_n>(0, 0));
    auto acc = mma.get_destination_cooperative_tensor<a_tile, b_tile, float>();
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;

    threadgroup bfloat staged[shards][live_m * tile_k];
    threadgroup float partials[shards - 1][tile_size];
    // The seven padded rows stay zero through all K tiles.
    for (int i = tile_k + int(lane); i < live_m * tile_k; i += 32) {
        staged[shard][i] = bfloat(0);
    }
    const int iterations = (width + shards * tile_k - 1) / (shards * tile_k);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if ((iteration & 3) == 0) threadgroup_barrier(mem_flags::mem_none);
        const int k = (iteration * shards + int(shard)) * tile_k;
        if (k >= width) continue;
        const int count = min(tile_k, width - k);
        for (int i = int(lane); i < count; i += 32) staged[shard][i] = input[k + i];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        group_bf16 a(staged[shard], dims(count, live_m), strides{1, tile_k});
        device_bf16 b(const_cast<device bfloat *>(weights) + ulong(start) * width + k,
                      dims(count, columns), strides{1, width});
        mma.run(a, b, acc);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (shard > 0) {
        group_f32 part(partials[shard - 1], dims(tile_n, tile_m), strides{1, tile_n});
        acc.store(part);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (shard != 0) return;
    for (int i = 0; i < shards - 1; ++i) {
        group_f32 part(partials[i], dims(tile_n, tile_m), strides{1, tile_n});
        auto values = mma.get_destination_cooperative_tensor<a_tile, b_tile, float>();
        values.load(part);
        for (ushort j = 0; j < acc.get_capacity(); ++j) acc[j] += values[j];
    }
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        auto index = acc.get_multidimensional_index(i);
        int column = start + index[0];
        if (index[1] == 0 && column >= first && column < outputs) output[column] = bfloat(acc[i]);
    }
}
