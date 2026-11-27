// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Theinruj Toranavikrai
// Copyright (c) 2026 David Dalcu
// Normalization order follows Sushi v1.8.1, src/exl3/expert_exl3_kernels.zig
// at b7b7975cc017acba1b4c0877d65543c4c5a4a381. See LICENSE.
#include <metal_stdlib>

using namespace metal;

constant bool STARK_HY4_STQ_ROUND_BF16 [[function_constant(0)]];

static inline float stark_hy4_stq_projection_boundary(float value) {
    if (is_function_constant_defined(STARK_HY4_STQ_ROUND_BF16) && STARK_HY4_STQ_ROUND_BF16)
        return float(bfloat(value));
    return value;
}

static inline float stark_hy4_stq_hadamard128(float value, uint position, threadgroup float* workspace) {
    // Preserve the ordered sum and difference of every butterfly. The first
    // five stages stay within a SIMD group, then two stages cross groups.
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
    return (position & 64u ? peer64 - value : value + peer64) * 0.08838834764831845f;
}

// One threadgroup per (128-channel block, sorted expert entry). The input
// map holds (expert ordinal, source row) for the grouped projection jobs.
kernel void stark_hy4_stq_pair1_prepare(
    device const float* input [[buffer(0)]], device const half* scales [[buffer(1)]],
    device const uint2* row_map [[buffer(2)]], device half* output [[buffer(3)]],
    constant uint& width [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
    uint position [[thread_index_in_threadgroup]]) {
    threadgroup float workspace[128];
    const uint2 entry = row_map[group.y];
    const uint channel = group.x * 128u + position;
    const float value = stark_hy4_stq_projection_boundary(input[ulong(entry.y) * width + channel]) *
                        float(scales[ulong(entry.x) * width + channel]);
    output[ulong(group.y) * width + channel] = half(stark_hy4_stq_hadamard128(value, position, workspace));
}

kernel void stark_hy4_stq_pair1_finish(
    device const half* input [[buffer(0)]], device const half* scales [[buffer(1)]],
    device const uint2* row_map [[buffer(2)]], device float* output [[buffer(3)]],
    constant uint& width [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
    uint position [[thread_index_in_threadgroup]]) {
    threadgroup float workspace[128];
    const uint expert = row_map[group.y].x;
    const uint channel = group.x * 128u + position;
    const ulong offset = ulong(group.y) * width + channel;
    const float value = stark_hy4_stq_hadamard128(float(input[offset]), position, workspace);
    output[offset] = stark_hy4_stq_projection_boundary(value * float(scales[ulong(expert) * width + channel]));
}

// Finish both projections, apply the routed SwiGLU clamp, and prepare the
// down projection's input without materializing gate and up in F32 buffers.
kernel void stark_hy4_stq_pair1_prepare_down(
    device const half* gate [[buffer(0)]], device const half* up [[buffer(1)]],
    device const half* gate_svh [[buffer(2)]], device const half* up_svh [[buffer(3)]],
    device const half* down_suh [[buffer(4)]], device const uint2* row_map [[buffer(5)]],
    device half* prepared [[buffer(6)]], constant uint& width [[buffer(7)]],
    constant float& limit [[buffer(8)]], uint3 group [[threadgroup_position_in_grid]],
    uint position [[thread_index_in_threadgroup]]) {
    threadgroup float gate_work[128], up_work[128], down_work[128];
    const uint channel = group.x * 128u + position;
    const ulong route_index = ulong(group.y) * width + channel;
    const ulong scale_index = ulong(row_map[group.y].x) * width + channel;
    const float g = stark_hy4_stq_hadamard128(float(gate[route_index]), position, gate_work) * float(gate_svh[scale_index]);
    const float u = stark_hy4_stq_hadamard128(float(up[route_index]), position, up_work) * float(up_svh[scale_index]);
    const float gv = min(g, limit);
    const float activation = (gv * (1.0f / (1.0f + precise::exp(-gv)))) * clamp(u, -limit, limit);
    const float weighted = activation * float(down_suh[scale_index]);
    prepared[route_index] = half(stark_hy4_stq_hadamard128(weighted, position, down_work));
}

kernel void stark_hy4_stq_pair1_combine(
    device const float* routed [[buffer(0)]], device const uint* inverse [[buffer(1)]],
    device const float* route_weights [[buffer(2)]], device float* output [[buffer(3)]],
    constant uint& width [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
    uint position [[thread_index_in_threadgroup]]) {
    const uint channel = group.x * 128u + position;
    const uint base = group.y * 16u;
    float sum = 0.0f;
    for (uint slot = 0; slot < 16u; ++slot) {
        const uint index = inverse[base + slot];
        if (index != 0xffffffffu) sum = fma(route_weights[base + slot], routed[ulong(index) * width + channel], sum);
    }
    output[ulong(group.y) * width + channel] = sum;
}
