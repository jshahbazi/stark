// SPDX-License-Identifier: MIT
// Copyright © 2025 Apple Inc.
// SIMD fragment coordinates from MLX steel/gemm/nax.h at
// fc344c986c4169d99f76b5d95ae6b4a58a359a63, via Stark's Inkling prefill_dense_source.metal.
// MIT notice: LICENSE.mlx-reference at the repository root.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// Four consecutive Q8_0 codes reside in a 34-byte block with an F16 scale.
// A block starts at a two-byte boundary, so scalar ushort loads are required.
static inline half4 stark_hy4_q8_fragment(device const uchar* row, uint k) {
    device const uchar* block = row + (k / 32) * 34;
    device const ushort* codes = (device const ushort*)(block + 2 + k % 32);
    return half4(float4(as_type<char4>(ushort2(codes[0], codes[1]))) * rdh(block));
}

// Each SIMD group computes 32 tokens by 32 output channels in two 16 x 32
// accumulators that share the same weight fragment. Convert matched F32
// activations and decoded weights to F16 once per fragment, then accumulate
// in F32. This keeps both operands in registers through the tensor operation.
kernel void stark_hy4_flash_dense_q8_packed(
    device const uchar* weights [[buffer(0)]], device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]], constant Hy4FlashDenseMmArgs& shape [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (gemv_heads) {
        weights += ulong(group.z) * GEMV_HW;
        input += ulong(group.z) * GEMV_HX;
        output += ulong(group.z) * GEMV_HY;
    }

    constexpr uint M = 16, N = 32, K = 16;
    const uint first_token = group.y * (4 * M) + (simd / 2) * (2 * M);
    const uint first_channel = group.x * (2 * N) + (simd % 2) * N;
    if (first_token >= shape.n_tok || first_channel >= shape.n_out) return;

    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        M, N, K, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroup> product;
    auto activation = product.get_left_input_cooperative_tensor<half, half, float>();
    auto weight = product.get_right_input_cooperative_tensor<half, half, float>();
    auto upper = product.get_destination_cooperative_tensor<remove_addrspace_t<decltype(activation)>,
        remove_addrspace_t<decltype(weight)>, float>();
    auto lower = product.get_destination_cooperative_tensor<remove_addrspace_t<decltype(activation)>,
        remove_addrspace_t<decltype(weight)>, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < upper.get_capacity(); ++i) {
        upper[i] = 0.0f;
        lower[i] = 0.0f;
    }

    const uint row = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const uint column = (((lane >> 2) & 2) | (lane & 1)) * 4;
    for (uint k = 0; k < shape.n_in; k += K) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 4; ++r) {
            const uint channel = first_channel + row + 8 * r;
            const half4 values = channel < shape.n_out ?
                stark_hy4_q8_fragment(weights + ulong(channel) * shape.row_bytes, k + column) : half4(0);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) weight[4 * r + i] = values[i];
        }
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 2; ++r) {
            const uint token = first_token + row + 8 * r;
            const half4 values = token < shape.n_tok ?
                half4(*(device const float4*)(input + ulong(token) * shape.x_stride + k + column)) : half4(0);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 4; ++i) activation[4 * r + i] = values[i];
        }
        product.run(activation, weight, upper);
        if (first_token + M < shape.n_tok) {
            #pragma clang loop unroll(full)
            for (uint r = 0; r < 2; ++r) {
                const uint token = first_token + M + row + 8 * r;
                const half4 values = token < shape.n_tok ?
                    half4(*(device const float4*)(input + ulong(token) * shape.x_stride + k + column)) : half4(0);
                #pragma clang loop unroll(full)
                for (uint i = 0; i < 4; ++i) activation[4 * r + i] = values[i];
            }
            product.run(activation, weight, lower);
        }
    }

    #pragma clang loop unroll(full)
    for (ushort i = 0; i < upper.get_capacity(); ++i) {
        const auto at = upper.get_multidimensional_index(i);
        const uint channel = first_channel + at[0], token = first_token + at[1];
        if (channel < shape.n_out) {
            if (token < shape.n_tok) output[ulong(token) * shape.y_stride + channel] = upper[i];
            if (token + M < shape.n_tok) output[ulong(token + M) * shape.y_stride + channel] = lower[i];
        }
    }
}
