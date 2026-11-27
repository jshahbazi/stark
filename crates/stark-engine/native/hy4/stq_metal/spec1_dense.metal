// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Theinruj Toranavikrai
// Copyright (c) 2026 David Dalcu
// Copyright (c) 2025 Apple Inc.
// SPEC1 codebook rows and trellis lane order follow Sushi v1.8.1 at
// b7b7975cc017acba1b4c0877d65543c4c5a4a381, src/exl3/codebook_tables.zig and
// src/exl3/expert_exl3_kernels.zig. MMA coordinates follow MLX steel/gemm/nax.h
// at fc344c986c4169d99f76b5d95ae6b4a58a359a63. See LICENSE and LICENSE.mlx.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

template <uint Rate>
static inline half2 stark_hy4_spec1_window(ulong words, uint first_shift, uint second_shift) {
    // spec1-w15 (37f72431), rows n=64 and n=96. Round the half sum
    // before scaling so its binary16 bits match the pack's scalar codec.
    constexpr uint multiplier = Rate == 4 ? 0x3fed4cefu : 0xd0003751u;
    constexpr uint mask = Rate == 4 ? 0x8ff08ff0u : 0x8ff98fffu;
    const uint2 code = uint2(uint(words >> first_shift), uint(words >> second_shift)) & 0x7fffu;
    const half4 h = as_type<half4>((code * multiplier) & mask);
    return half2(h.x + h.y, h.z + h.w) * 4096.0h;
}

template <uint Rate>
static inline vec<half, 8> stark_hy4_spec1_fragment(device const uint* words, uint lane) {
    constexpr uint tile_words = Rate * 8;
    // Sixteen consecutive trellis positions start at a word boundary at
    // both rates. Adjacent lane quads start two positions into the group.
    const uint base = ((lane & 16u) / 2u + (lane & 7u)) * (Rate / 2u);
    const uint offset = (lane & 8u) != 0u ? 2u * Rate : 0u;
    const uint previous = words[base == 0u ? tile_words - 1u : base - 1u];
    const uint x = words[base], y = words[base + 1u];
    const ulong xy = (ulong(x) << 32u) | ulong(y);
    const ulong px = (ulong(previous) << 32u) | ulong(x);
    half2 a, b, c, d;
    if constexpr (Rate == 4) {
        a = stark_hy4_spec1_window<Rate>(px, 28u - offset, 24u - offset);
        b = stark_hy4_spec1_window<Rate>(xy, 28u - offset, 24u - offset);
        c = stark_hy4_spec1_window<Rate>(px, 12u - offset, 8u - offset);
        d = stark_hy4_spec1_window<Rate>(xy, 12u - offset, 8u - offset);
    } else {
        const ulong yz = (ulong(y) << 32u) | ulong(words[base + 2u]);
        a = stark_hy4_spec1_window<Rate>(px, 26u - offset, 20u - offset);
        b = stark_hy4_spec1_window<Rate>(yz, 42u - offset, 36u - offset);
        c = stark_hy4_spec1_window<Rate>(xy, 34u - offset, 28u - offset);
        d = stark_hy4_spec1_window<Rate>(yz, 18u - offset, 12u - offset);
    }
    return vec<half, 8>(a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y);
}

template <typename Fragment>
static inline void stark_hy4_spec1_load(thread Fragment& target, device const half* a, device const half* b, uint k) {
    const half4 first = *(device const half4*)(a + k);
    const half4 second = *(device const half4*)(b + k);
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        target[i] = first[i];
        target[i + 4] = second[i];
    }
}

template <typename Fragment, typename Output>
static inline void stark_hy4_spec1_store(const thread Fragment& source, device Output* output,
    uint channel, uint first_row, uint live_rows, uint width) {
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < source.get_capacity(); ++i) {
        const auto at = source.get_multidimensional_index(i);
        if (at[1] < live_rows) output[ulong(first_row + at[1]) * width + channel + at[0]] = Output(source[i]);
    }
}

template <uint Rate, bool TwoTiles, typename Output>
static inline void stark_hy4_spec1_project(device const uint* weights, device const half* input,
    device Output* output, uint2 widths, uint first_row, uint live_rows, uint channel, uint lane,
    uint begin_k, uint end_k) {
    constexpr uint K = 16, Rows = 16, TileWords = Rate * 8;
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
    const uint last = live_rows - 1u;
    device const half* start = input + ulong(first_row) * widths.x + col;
    device const half* top0 = start + ulong(min(row, last)) * widths.x;
    device const half* top1 = start + ulong(min(row + 8u, last)) * widths.x;
    device const half* bottom0 = start + ulong(min(row + Rows, last)) * widths.x;
    device const half* bottom1 = start + ulong(min(row + Rows + 8u, last)) * widths.x;
    const ulong tile_stride = ulong(widths.y / 16u) * TileWords;
    device const uint* tiles = weights + (begin_k / K) * tile_stride + (channel / K) * TileWords;

    #pragma clang loop unroll_count(2)
    for (uint k = begin_k; k < end_k; k += K, tiles += tile_stride) {
        const auto w0 = stark_hy4_spec1_fragment<Rate>(tiles, lane);
        const auto w1 = stark_hy4_spec1_fragment<Rate>(tiles + TileWords, lane);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) {
            right[i] = w0[i];
            right[i + 8] = w1[i];
        }
        stark_hy4_spec1_load(left, top0, top1, k);
        product.run(left, right, top);
        if constexpr (TwoTiles) {
            stark_hy4_spec1_load(left, bottom0, bottom1, k);
            product.run(left, right, bottom);
        }
    }

    stark_hy4_spec1_store(top, output, channel, first_row, min(live_rows, Rows), widths.y);
    if constexpr (TwoTiles) {
        stark_hy4_spec1_store(bottom, output, channel, first_row + Rows, live_rows - Rows, widths.y);
    }
}

// shape is (input channels, output channels, rows, reserved). Input channels
// are a multiple of 16 and output channels a multiple of 32. Weights are the
// original little-endian [input tile][output tile][64/96 U16] words. Input is
// eight-byte-aligned F16 after SUH/H128. The F16 output precedes H128/SVH.
// Launch (ceil(output channels / 128), ceil(rows / 32), 1) groups of 128.
template <uint Rate>
kernel void stark_hy4_stq_spec1_dense(
    device const uint* weights [[buffer(0)]], device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]], constant uint4& shape [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint channel = group.x * 128u + sg * 32u;
    const uint first_row = group.y * 32u;
    if (first_row >= shape.z || channel >= shape.y) return;
    const uint live_rows = min(32u, shape.z - first_row);
    if (live_rows > 16u) {
        stark_hy4_spec1_project<Rate, true>(weights, input, output, shape.xy, first_row, live_rows, channel, lane, 0u, shape.x);
    } else {
        stark_hy4_spec1_project<Rate, false>(weights, input, output, shape.xy, first_row, live_rows, channel, lane, 0u, shape.x);
    }
}

template [[host_name("stark_hy4_stq_spec1_k4_dense")]] kernel decltype(stark_hy4_stq_spec1_dense<4>)
    stark_hy4_stq_spec1_dense<4>;
template [[host_name("stark_hy4_stq_spec1_k6_dense")]] kernel decltype(stark_hy4_stq_spec1_dense<6>)
    stark_hy4_stq_spec1_dense<6>;

// The narrow NAX projection partitions its contiguous K tiles across threadgroups,
// as in Sushi v1.8.1 src/exl3/project_rows.zig. Each split writes an F32 plane
// [rows, output channels]. The F16 round occurs after the ordered split reduction.
// shape is (input channels, output channels, rows <= 16, split count).
// Launch (output channels / 128, split count, 1) groups of 128.
kernel void stark_hy4_stq_spec1_k6_split(
    device const uint* weights [[buffer(0)]], device const half* input [[buffer(1)]],
    device float* partials [[buffer(2)]], constant uint4& shape [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint channel = group.x * 128u + sg * 32u;
    const uint tiles = shape.x / 16u;
    const uint begin_k = (group.y * tiles / shape.w) * 16u;
    const uint end_k = ((group.y + 1u) * tiles / shape.w) * 16u;
    device float* plane = partials + ulong(group.y) * shape.z * shape.y;
    stark_hy4_spec1_project<6, false>(weights, input, plane, shape.xy, 0u, shape.z, channel, lane, begin_k, end_k);
}
