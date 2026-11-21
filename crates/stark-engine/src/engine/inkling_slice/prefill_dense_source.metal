// SPDX-License-Identifier: MIT
// Copyright © 2025 Apple Inc.
// Register fragments and router arithmetic adapted from MLX at
// fc344c986c4169d99f76b5d95ae6b4a58a359a63 (LICENSE.mlx-reference).
// See steel/gemm/nax.h, binary.h, unary.h and softmax.h.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

template <typename Compute, typename Output>
static void ink_prefill_matrix(
    device const bfloat *input, device const bfloat *weight, device Output *output,
    uint4 shape, uint2 tile, uint lane) {
    constexpr uint M = 16, N = 32, K = 16;
    const uint column = tile.x * N, row = tile.y * M;
    const uint width = shape.x, columns = shape.y, rows = shape.z, split = shape.w;

    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        M, N, K, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroup> product;
    auto a = product.get_left_input_cooperative_tensor<Compute, Compute, float>();
    auto b = product.get_right_input_cooperative_tensor<Compute, Compute, float>();
    auto acc = product.get_destination_cooperative_tensor<remove_addrspace_t<decltype(a)>,
        remove_addrspace_t<decltype(b)>, float>();
    float total[M * N / 32] = {};

    const uint row_base = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const uint k_base = (((lane >> 2) & 2) | (lane & 1)) * 4;
    for (uint part = 0; part < width; part += split) {
        for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
        for (uint k = part; k < min(part + split, width); k += K) {
            for (ushort i = 0; i < 8; ++i) {
                const uint m = row_base + (i >> 2) * 8;
                const uint d = k + k_base + (i & 3);
                a[i] = row + m < rows ? input[(row + m) * width + d] : bfloat(0);
                b[i] = column + m < columns ? weight[(column + m) * width + d] : bfloat(0);
                b[i + 8] = column + m + 16 < columns ? weight[(column + m + 16) * width + d] : bfloat(0);
            }
            product.run(a, b, acc);
        }
        for (ushort i = 0; i < acc.get_capacity(); ++i) total[i] += acc[i];
    }

    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        const auto at = acc.get_multidimensional_index(i);
        if (row + at[1] < rows && column + at[0] < columns) {
            output[(row + at[1]) * columns + column + at[0]] = Output(total[i]);
        }
    }
}

kernel void ink_prefill_dense_source(
    device const bfloat *input [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]], constant uint4 &shape [[buffer(3)]],
    uint2 tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    ink_prefill_matrix<bfloat>(input, weight, output, shape, tile, lane);
}

kernel void ink_prefill_router_source(
    device const bfloat *input [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &rows [[buffer(3)]],
    uint2 tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    ink_prefill_matrix<float>(input, weight, output, uint4(2304, 128, rows, rows <= 1024 ? 2048 : 2304), tile, lane);
}



// Keep the source tensor operation and K order. Only the live row's input
// registers are loaded inside the loop. Each weight fragment is aligned to
// four BF16 words on this fixed 2,304-wide router.
kernel void ink_singleton_router_source(
    device const bfloat *input [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device float *output [[buffer(2)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint M = 16, N = 32, K = 16, Width = 2304;
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        M, N, K, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroup> product;
    auto a = product.get_left_input_cooperative_tensor<float, float, float>();
    auto b = product.get_right_input_cooperative_tensor<float, float, float>();
    auto acc = product.get_destination_cooperative_tensor<remove_addrspace_t<decltype(a)>,
        remove_addrspace_t<decltype(b)>, float>();
    for (ushort i = 0; i < a.get_capacity(); ++i) a[i] = 0.0f;
    float total[M * N / 32] = {};
    const uint row_base = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const uint k_base = (((lane >> 2) & 2) | (lane & 1)) * 4;
    const uint column = tile * N + row_base;
    const device bfloat4 *w0 = (const device bfloat4 *)(weight + column * Width + k_base);
    const device bfloat4 *w1 = w0 + 8 * Width / 4;
    const device bfloat4 *w2 = w0 + 16 * Width / 4;
    const device bfloat4 *w3 = w0 + 24 * Width / 4;
    const device bfloat4 *source = (const device bfloat4 *)(input + k_base);
    for (uint part = 0; part < Width; part += 2048) {
        for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
        for (uint k = part / 4; k < min(part + 2048, Width) / 4; k += K / 4) {
            const bfloat4 x0 = w0[k], x1 = w1[k], x2 = w2[k], x3 = w3[k];
            const bfloat4 row = row_base == 0 ? source[k] : bfloat4(0);
            for (ushort i = 0; i < 4; ++i) {
                if (row_base == 0) a[i] = row[i];
                b[i] = x0[i];
                b[i + 4] = x1[i];
                b[i + 8] = x2[i];
                b[i + 12] = x3[i];
            }
            product.run(a, b, acc);
        }
        for (ushort i = 0; i < acc.get_capacity(); ++i) total[i] += acc[i];
    }
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        const auto at = acc.get_multidimensional_index(i);
        if (at[1] == 0) output[tile * N + at[0]] = total[i];
    }
}

kernel void ink_prefill_select_source(
    device const float *all_logits [[buffer(0)]], device const float *bias [[buffer(1)]],
    device uint *all_ids [[buffer(2)]], device float *all_weights [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const device float *logits = all_logits + row * 128;
    threadgroup float choices[128];
    const float x = logits[lane];
    const float low = 1.0f / (1.0f + precise::exp(abs(x)));
    choices[lane] = (x < 0.0f ? low : 1.0f - low) + bias[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane != 0) return;

    float scores[8], peak = -INFINITY;
    for (uint route = 0; route < 8; ++route) {
        uint selected = 0;
        for (uint expert = 1; expert < 128; ++expert) {
            if (choices[expert] > choices[selected]) selected = expert;
        }
        choices[selected] = -INFINITY;
        all_ids[row * 8 + route] = selected;
        const float x = -logits[selected];
#ifdef INK_PREFILL_LOG_WEIGHTS
        scores[route] = x;
#else
        const float largest = max(x, 0.0f);
        const float e = exp(min(x, 0.0f) - largest);
        const float p = 1.0f + e;
        const float correction = p == 1.0f ? e : e * (log(p) / (p - 1.0f));
        scores[route] = -(largest + correction);
        peak = max(peak, scores[route]);
#endif
    }
#ifdef INK_PREFILL_LOG_WEIGHTS
    for (uint route = 0; route < 8; ++route) all_weights[row * 8 + route] = scores[route];
#else
    for (uint route = 0; route < 8; ++route) scores[route] = fast::exp(scores[route] - peak);
    const float first = ((scores[0] + scores[1]) + scores[2]) + scores[3];
    const float second = ((scores[4] + scores[5]) + scores[6]) + scores[7];
    const float inverse = 1.0f / (first + second);
    for (uint route = 0; route < 8; ++route) all_weights[row * 8 + route] = (scores[route] * inverse) * 8.0f;
#endif
}

#ifdef INK_PREFILL_LOG_WEIGHTS
kernel void ink_prefill_negate_route_weights(device float *weights [[buffer(0)]], uint pos [[thread_position_in_grid]]) {
    weights[pos] = -weights[pos];
}

kernel void ink_prefill_scale_route_weights(
    device float *weights [[buffer(0)]], constant uint &count [[buffer(1)]], uint pos [[thread_position_in_grid]]) {
    if (pos < count) weights[pos] *= 8.0f;
}
#endif
