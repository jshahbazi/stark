// SPDX-License-Identifier: MIT
// Copyright © 2025 Apple Inc.
// Register fragment layout adapted from MLX steel/gemm/nax.h at
// fc344c986c4169d99f76b5d95ae6b4a58a359a63 (LICENSE.mlx-reference).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

// Each SIMD group expands 32 channels across as many as 16 prompt rows.
kernel void ink_prefill_controller_expand(
    device const bfloat *compressed [[buffer(0)]],
    device const bfloat *gain [[buffer(1)]],
    device const bfloat *forget [[buffer(2)]],
    device const float *input_message [[buffer(3)]],
    device float *mixed [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rank = 64;
    constexpr uint columns = 32;
    constexpr uint fragment_elements = 8;
    constexpr uint width = 2304;
    const uint first = tile.x * columns;
    const uint first_row = tile.y * 16;

    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        16, columns, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroup> mma;
    auto a = mma.get_left_input_cooperative_tensor<float, float, float>();
    auto b = mma.get_right_input_cooperative_tensor<float, float, float>();
    using a_type = remove_addrspace_t<decltype(a)>;
    using b_type = remove_addrspace_t<decltype(b)>;
    auto g = mma.get_destination_cooperative_tensor<a_type, b_type, float>();
    auto f = mma.get_destination_cooperative_tensor<a_type, b_type, float>();
    for (ushort i = 0; i < g.get_capacity(); ++i) {
        g[i] = 0.0f;
        f[i] = 0.0f;
    }

    const uint base_row = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const uint base_column = (((lane >> 2) & 2) | (lane & 1)) * 4;
    for (uint k = 0; k < rank; k += 16) {
        for (ushort i = 0; i < fragment_elements; ++i) {
            const uint row = base_row + (i >> 2) * 8;
            const uint column = base_column + (i & 3);
            const uint index = (first + row) * rank + k + column;
            const uint prompt_row = first_row + row;
            a[i] = prompt_row < rows ? float(compressed[prompt_row * rank + k + column]) : 0.0f;
            b[i] = float(gain[index]);
            b[fragment_elements + i] = float(gain[index + 16 * rank]);
        }
        mma.run(a, b, g);
        for (ushort i = 0; i < fragment_elements; ++i) {
            const uint row = base_row + (i >> 2) * 8;
            const uint column = base_column + (i & 3);
            const uint index = (first + row) * rank + k + column;
            b[i] = float(forget[index]);
            b[fragment_elements + i] = float(forget[index + 16 * rank]);
        }
        mma.run(a, b, f);
    }

    // The compiled target gate uses separate FP32 products followed by an add.
    {
#pragma clang fp contract(off) reassociate(off)
        for (ushort i = 0; i < g.get_capacity(); ++i) {
            const auto coordinate = g.get_multidimensional_index(i);
            const uint prompt_row = first_row + coordinate[1];
            if (prompt_row < rows) {
                const uint channel = first + coordinate[0];
                const float keep = 1.0f - metal::precise::tanh(f[i]);
                const float admit = metal::precise::tanh(g[i]);
                const device float *input = input_message + prompt_row * 2 * width;
                mixed[prompt_row * width + channel] = input[channel] * keep + input[width + channel] * admit;
            }
        }
    }
}

kernel void ink_prefill_controller_pack(
    device const float *input [[buffer(0)]],
    device const float *message [[buffer(1)]],
    device const bfloat *compressed [[buffer(2)]],
    device float *input_message [[buffer(3)]],
    device bfloat *packed_compressed [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    constexpr uint width = 2304;
    input_message[i] = input[i];
    input_message[width + i] = message[i];
    if (i < 64) packed_compressed[i] = compressed[i];
}
