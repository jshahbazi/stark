// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Theinruj Toranavikrai
// Copyright (c) 2026 David Dalcu
// SPEC1 K6 window and tensor layout follow Sushi v1.8.1 at
// b7b7975cc017acba1b4c0877d65543c4c5a4a381, src/exl3/codebook_tables.zig,
// src/exl3/expert_exl3.zig, and src/hy4flash/hy4_embed.zig. See LICENSE.
#include <metal_stdlib>
using namespace metal;

static inline uint stark_hy4_embed_code(device const uint* tile, uint position) {
    const uint end = 6u * (position + 1u);
    const uint word = (end - 1u) / 32u;
    const uint previous = word == 0u ? 47u : word - 1u;
    const uint shift = (0u - end) & 31u;
    return ((tile[word] >> shift) | ((tile[previous] << 1u) << (31u - shift))) & 0x7fffu;
}

static inline half2 stark_hy4_embed_pair(device const uint* tile, uint position) {
    // The tensor-core permutation places output columns c and c+8 four
    // trellis positions apart for a fixed input feature.
    const uint2 code = uint2(stark_hy4_embed_code(tile, position), stark_hy4_embed_code(tile, position + 4u));
    const half4 h = as_type<half4>((code * 0xd0003751u) & 0x8ff98fffu);
    return half2(h.x + h.y, h.z + h.w) * 4096.0h;
}

static inline float stark_hy4_embed_hadamard(float value, uint position, threadgroup float* workspace) {
    #pragma clang loop unroll(full)
    for (uint step = 1; step < 32; step <<= 1) {
        const float peer = simd_shuffle_xor(value, ushort(step));
        value = position & step ? peer - value : value + peer;
    }
    workspace[position] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float peer32 = workspace[position ^ 32u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = position & 32u ? peer32 - value : value + peer32;
    workspace[position] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float peer64 = workspace[position ^ 64u];
    return position & 64u ? peer64 - value : value + peer64;
}

// Decode the requested 128-ID block directly from [IT, OT, 96×U16]. The
// output-side Hadamard selects one column, then the input-side Hadamard
// transforms it in place. Each group handles 128 features for one token.
// Moving the output-side transform first leaves the two independent
// axes mathematically equivalent with a different F32 reduction order.
kernel void stark_hy4_stq_embedding_k6(
    device const uint* trellis [[buffer(0)]], device const half* suh [[buffer(1)]],
    device const half* svh [[buffer(2)]], device const int* ids [[buffer(3)]],
    device float* output [[buffer(4)]], constant uint2& shape [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], uint position [[thread_index_in_threadgroup]]) {
    const uint token = uint(ids[group.y]);
    const uint feature = group.x * 128u + position;
    const ulong out = ulong(group.y) * shape.x + feature;
    if (token >= shape.y) {
        output[out] = 0.0f;
        return;
    }

    threadgroup float workspace[128];
    const uint input_tile = feature / 16u;
    const uint in_tile = feature & 15u;
    const uint token_block = token / 128u;
    const uint origin = ((in_tile & 7u) / 2u) * 8u + ((in_tile & 8u) / 4u) + (in_tile & 1u);
    device const uint* tiles = trellis + (ulong(input_tile) * (shape.y / 16u) + token_block * 8u) * 48u;
    float sum = 0.0f;
    for (uint tile = 0; tile < 8; ++tile) {
        #pragma clang loop unroll(full)
        for (uint c = 0; c < 8; ++c) {
            const half2 weights = stark_hy4_embed_pair(tiles + tile * 48u, origin + c * 32u);
            const float pair = float(weights.x) + ((token & 8u) == 0u ? float(weights.y) : -float(weights.y));
            sum += (popcount((tile * 16u + c) & token) & 1u) == 0u ? pair : -pair;
        }
    }
    const float rotated = stark_hy4_embed_hadamard(sum, position, workspace);
    output[out] = (rotated * (1.0f / 128.0f)) * (float(suh[feature]) * float(svh[token]));
}
