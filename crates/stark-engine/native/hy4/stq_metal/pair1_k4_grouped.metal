// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Theinruj Toranavikrai
// Copyright (c) 2026 David Dalcu
// Copyright (c) 2025 Apple Inc.
// PAIR1 decoding and pair-major register feeds adapted from Sushi at
// b7b7975cc017acba1b4c0877d65543c4c5a4a381, src/exl3/kernel_source.zig and
// src/exl3/expert_exl3_kernels.zig. MMA coordinates follow MLX steel/gemm/nax.h
// at fc344c986c4169d99f76b5d95ae6b4a58a359a63. See LICENSE and LICENSE.mlx.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

static inline half2 stark_hy4_pair1_k4(uint code) {
    const uint product = (code & 0xffffu) * 0xe2c7fc0du;
    const uint rotated = rotate(product, 19u);
    const half4 h = as_type<half4>(uint2(product & 0x8f408fa0u, rotated & 0x8f118fc9u));
    return half2(h.x + h.y, h.z + h.w) * 4096.0h;
}

// Each lane owns two rows of four weights from the 16x16 tile. The odd
// window encodes the pair, including the tail bits when it wraps at word 0.
static inline vec<half, 8> stark_hy4_pair1_k4_fragment(device const uint* words, uint lane) {
    const uint index = (lane & 16u) + ((lane & 7u) << 1u);
    const uint2 current = *(device const uint2*)(words + index);
    const ulong w0 = (ulong(words[(index + 31u) & 31u]) << 32u) | ulong(current.x);
    const ulong w1 = (ulong(current.x) << 32u) | ulong(current.y);
    const uint shift = 24u - (lane & 8u);
    const half2 a = stark_hy4_pair1_k4(uint(w0 >> shift));
    const half2 b = stark_hy4_pair1_k4(uint(w1 >> shift));
    const half2 c = stark_hy4_pair1_k4(uint(w0 >> (shift - 16u)));
    const half2 d = stark_hy4_pair1_k4(uint(w1 >> (shift - 16u)));
    return vec<half, 8>(a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y);
}

template <typename Fragment>
static inline void stark_hy4_stq_load_activation(thread Fragment& target,
    device const half* row0, device const half* row1, uint offset) {
    const half4 a = *(device const half4*)(row0 + offset);
    const half4 b = *(device const half4*)(row1 + offset);
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        target[i] = a[i];
        target[i + 4] = b[i];
    }
}

template <typename Fragment>
static inline void stark_hy4_stq_store(const thread Fragment& source, device half* output,
    uint channel, uint first_row, uint live_rows, uint out_width) {
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < source.get_capacity(); ++i) {
        const auto at = source.get_multidimensional_index(i);
        if (at[1] < live_rows) {
            output[ulong(first_row + at[1]) * out_width + channel + at[0]] = half(source[i]);
        }
    }
}

template <bool TwoTiles>
static inline void stark_hy4_pair1_k4_project(device const uint* weights,
    device const half* input, device half* output, uint3 job, uint2 shape, uint channel, uint lane) {
    constexpr uint K = 16, Rows = 16, PairWords = 64;
    constexpr auto descriptor = matmul2d_descriptor(Rows, 32, K, false, true, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroup> product;
    auto left = product.get_left_input_cooperative_tensor<half, half, float>();
    auto right = product.get_right_input_cooperative_tensor<half, half, float>();
    using Left = remove_addrspace_t<decltype(left)>;
    using Right = remove_addrspace_t<decltype(right)>;
    auto top = product.get_destination_cooperative_tensor<Left, Right, float>();
    auto bottom = product.get_destination_cooperative_tensor<Left, Right, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < top.get_capacity(); ++i) {
        top[i] = 0.0f;
        if constexpr (TwoTiles) bottom[i] = 0.0f;
    }

    const uint row = ((lane >> 2u) & 4u) | ((lane >> 1u) & 3u);
    const uint col = (((lane >> 2u) & 2u) | (lane & 1u)) * 4u;
    const uint last = job.z - 1u;
    device const half* input_start = input + ulong(job.y) * shape.x + col;
    device const half* top0 = input_start + ulong(min(row, last)) * shape.x;
    device const half* top1 = input_start + ulong(min(row + 8u, last)) * shape.x;
    device const half* bottom0 = input_start + ulong(min(row + Rows, last)) * shape.x;
    device const half* bottom1 = input_start + ulong(min(row + Rows + 8u, last)) * shape.x;
    const ulong pair = ulong(job.x) * (shape.y / 32u) + channel / 32u;
    device const uint* pair_tiles = weights + pair * (shape.x / K) * PairWords;

    #pragma clang loop unroll_count(2)
    for (uint k = 0; k < shape.x; k += K, pair_tiles += PairWords) {
        const auto w0 = stark_hy4_pair1_k4_fragment(pair_tiles, lane);
        const auto w1 = stark_hy4_pair1_k4_fragment(pair_tiles + PairWords / 2u, lane);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) {
            right[i] = w0[i];
            right[i + 8] = w1[i];
        }
        stark_hy4_stq_load_activation(left, top0, top1, k);
        product.run(left, right, top);
        if constexpr (TwoTiles) {
            stark_hy4_stq_load_activation(left, bottom0, bottom1, k);
            product.run(left, right, bottom);
        }
    }

    stark_hy4_stq_store(top, output, channel, job.y, min(job.z, Rows), shape.y);
    if constexpr (TwoTiles) {
        stark_hy4_stq_store(bottom, output, channel, job.y + Rows, job.z - Rows, shape.y);
    }
}

// shape is (input channels, output channels), aligned to 16 and 32 respectively.
// The weight buffer is aligned to eight bytes and holds unmodified little-endian
// [expert][output tile pair][input tile][2][32 uint32] data. Input contains F16
// rows after SUH/H128 preparation. Output is F16 before H128/SVH finishing.
// jobs are uint4(expert, first sorted row, live rows, reserved). Live row
// ranges are disjoint and in bounds for both activation and output buffers.
// The encoder dispatches (ceil(output channels / 128), jobs, 1) groups of 128.
kernel void stark_hy4_stq_pair1_k4_grouped(
    device const uint* weights [[buffer(0)]], device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]], device const uint4* jobs [[buffer(3)]],
    constant uint2& shape [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint3 job = jobs[group.y].xyz;
    const uint channel = group.x * 128u + sg * 32u;
    if (job.z == 0u || job.z > 32u || channel >= shape.y) return;
    if (job.z > 16u) {
        stark_hy4_pair1_k4_project<true>(weights, input, output, job, shape, channel, lane);
    } else {
        stark_hy4_pair1_k4_project<false>(weights, input, output, job, shape, channel, lane);
    }
}
