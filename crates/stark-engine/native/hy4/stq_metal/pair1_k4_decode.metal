// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Theinruj Toranavikrai
// Copyright (c) 2026 David Dalcu
// Lane funnel and reduction follow Sushi v1.8.1 at
// b7b7975cc017acba1b4c0877d65543c4c5a4a381, src/exl3/kernel_source.zig and
// src/exl3/expert_exl3_kernels.zig. See LICENSE.
#include <metal_stdlib>
using namespace metal;

static inline float2 stark_hy4_pair1_decode_window(uint code) {
    const uint product = (code & 0xffffu) * 0xe2c7fc0du;
    const half4 h = as_type<half4>(uint2(product & 0x8f408fa0u, rotate(product, 19u) & 0x8f118fc9u));
    return float2(half2(h.x + h.y, h.z + h.w));
}

static inline void stark_hy4_pair1_decode_weights(device const uint* tile, uint lane,
                                                 thread float4& low, thread float4& high) {
    const ulong words = (ulong(tile[(lane + 31u) & 31u]) << 32u) | tile[lane];
    low = float4(stark_hy4_pair1_decode_window(uint(words >> 24u)),
                 stark_hy4_pair1_decode_window(uint(words >> 16u)));
    high = float4(stark_hy4_pair1_decode_window(uint(words >> 8u)),
                  stark_hy4_pair1_decode_window(uint(words)));
}

template <uint Rows>
static inline void stark_hy4_pair1_decode_tile(device const uint* weights, device const half* input,
    device half* output, uint expert, uint first_row, uint pair, uint2 shape, uint sg, uint lane,
    threadgroup float* partial) {
    constexpr uint KTile = 16, TileWords = 32, Groups = 4;
    const uint it = shape.x / KTile;
    const uint k0 = (lane & 3u) * 2u;
    const uint output_col = lane >> 2u;
    const ulong weight_tile = (ulong(expert) * (shape.y / 32u) + pair) * it + sg;
    device const uint* packed = weights + weight_tile * (2u * TileWords);
    device const half* activation = input + ulong(first_row) * shape.x + sg * KTile + k0;
    float4 lo[Rows][2] = {}, hi[Rows][2] = {};
    #pragma clang loop unroll_count(2)
    for (uint k = sg; k < it; k += Groups) {
        float4 lhs[Rows];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < Rows; ++r) {
            device const half* x = activation + ulong(r) * shape.x;
            lhs[r] = float4(float2(*(device const half2*)x), float2(*(device const half2*)(x + 8u)));
        }
        #pragma clang loop unroll(full)
        for (uint tile = 0; tile < 2; ++tile) {
            float4 low, high;
            stark_hy4_pair1_decode_weights(packed + tile * TileWords, lane, low, high);
            #pragma clang loop unroll(full)
            for (uint r = 0; r < Rows; ++r) {
                lo[r][tile] = fma(lhs[r], low, lo[r][tile]);
                hi[r][tile] = fma(lhs[r], high, hi[r][tile]);
            }
        }
        packed += Groups * 2u * TileWords;
        activation += Groups * KTile;
    }

    #pragma clang loop unroll(full)
    for (uint r = 0; r < Rows; ++r) {
        #pragma clang loop unroll(full)
        for (uint tile = 0; tile < 2; ++tile) {
            const float4 a = lo[r][tile], b = hi[r][tile];
            float2 sum = float2((a.x + a.y) + (a.z + a.w), (b.x + b.y) + (b.z + b.w));
            sum += simd_shuffle_xor(sum, 1u);
            sum += simd_shuffle_xor(sum, 2u);
            if ((lane & 3u) == 0u) {
                const uint index = (r * Groups + sg) * 32u + tile * 16u + output_col;
                partial[index] = sum.x;
                partial[index + 8u] = sum.y;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < Rows) {
        const uint index = sg * Groups * 32u + lane;
        float sum = 0.0f;
        #pragma clang loop unroll(full)
        for (uint g = 0; g < Groups; ++g) sum += partial[index + g * 32u];
        output[ulong(first_row + sg) * shape.y + pair * 32u + lane] = half(sum * 4096.0f);
    }
}

// Prototype for narrow verification batches. Uses the grouped jobs ABI and
// assigns K tiles to four SIMD groups for every 32-output pair of tiles.
// Grid (output channels / 32, jobs, ceil(max job rows / 2)) groups of 128.
// The original little-endian packed bank and prepared/result rows are shared
// with the MMA kernel. Input width is a multiple of 64, output a multiple of 32.
kernel void stark_hy4_stq_pair1_k4_decode(
    device const uint* weights [[buffer(0)]], device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]], device const uint4* jobs [[buffer(3)]],
    constant uint2& shape [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint4 job = jobs[group.y];
    const uint member = group.z * 2u;
    if (member >= job.z) return;
    threadgroup float partial[2 * 4 * 32];
    if (member + 1u < job.z) {
        stark_hy4_pair1_decode_tile<2>(weights, input, output, job.x, job.y + member, group.x, shape, sg, lane, partial);
    } else {
        stark_hy4_pair1_decode_tile<1>(weights, input, output, job.x, job.y + member, group.x, shape, sg, lane, partial);
    }
}
