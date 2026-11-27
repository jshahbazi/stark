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

// The attention gate's finish, sigmoid product and output projection prepare
// share the same 128-channel block. Keep the two Hadamard workspaces and all
// precision boundaries while avoiding the intermediate F32 planes.
static inline half stark_hy4_stq_apply_attention_gate(
    half raw_gate, float gate_scale, float value, float input_scale, uint lane,
    threadgroup float* gate_work, threadgroup float* output_work) {
    const float gate = stark_hy4_stq_projection_boundary(
        stark_hy4_stq_hadamard128(float(raw_gate), lane, gate_work) * gate_scale);
    const float gated = value * (1.0f / (1.0f + precise::exp(-gate)));
    const float source = stark_hy4_stq_projection_boundary(gated) * input_scale;
    return half(stark_hy4_stq_hadamard128(source, lane, output_work));
}

kernel void stark_hy4_stq_attention_gate_prepare(
    device const half* raw_gate [[buffer(0)]], device const half* gate_scale [[buffer(1)]],
    device const float* value [[buffer(2)]], device const half* input_scale [[buffer(3)]],
    device half* prepared [[buffer(4)]], constant uint& width [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint channel_in_group [[thread_index_in_threadgroup]]) {
    threadgroup float gate_work[128];
    threadgroup float output_work[128];
    const uint channel = group.x * 128u + channel_in_group;
    const ulong index = ulong(group.y) * width + channel;
    prepared[index] = stark_hy4_stq_apply_attention_gate(raw_gate[index], float(gate_scale[channel]),
        value[index], float(input_scale[channel]), channel_in_group, gate_work, output_work);
}

static inline half stark_hy4_stq_spec1_split_round(device const float* partials, ulong offset,
                                                    ulong plane_size, uint splits) {
    float sum = partials[offset];
    for (uint split = 1u; split < splits; ++split) sum += partials[ulong(split) * plane_size + offset];
    return half(sum);
}

// The verifier preserves the same ordered split reduction, F16 product,
// gate, and output preparation as the three separate dispatches.
kernel void stark_hy4_stq_attention_gate_prepare_split(
    device const float* partials [[buffer(0)]], device const half* gate_scale [[buffer(1)]],
    device const float* value [[buffer(2)]], device const half* input_scale [[buffer(3)]],
    device half* prepared [[buffer(4)]], constant uint4& shape [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float gate_work[128];
    threadgroup float output_work[128];
    const uint channel = group.x * 128u + lane;
    const ulong index = ulong(group.y) * shape.y + channel;
    const half raw_gate = stark_hy4_stq_spec1_split_round(partials, index, ulong(shape.y) * shape.z, shape.w);
    prepared[index] = stark_hy4_stq_apply_attention_gate(raw_gate, float(gate_scale[channel]),
        value[index], float(input_scale[channel]), lane, gate_work, output_work);
}

// SPEC1 split-K restores the F16 product boundary before H128/SVH. A split
// plane holds shape.z rows of shape.y channels. Launch (shape.y/128, shape.z, 1)
// groups of 128 with shape.w partials per output element.
kernel void stark_hy4_stq_spec1_split_finish(
    device const float* partials [[buffer(0)]], device const half* scales [[buffer(1)]],
    device float* output [[buffer(2)]], constant uint4& shape [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]], uint position [[thread_index_in_threadgroup]]) {
    threadgroup float workspace[128];
    const uint channel = group.x * 128u + position;
    const ulong offset = ulong(group.y) * shape.y + channel;
    const ulong plane_size = ulong(shape.z) * shape.y;
    const float value = stark_hy4_stq_hadamard128(
        float(stark_hy4_stq_spec1_split_round(partials, offset, plane_size, shape.w)), position, workspace);
    output[offset] = stark_hy4_stq_projection_boundary(value * float(scales[channel]));
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

// The inverse-indexed finish follows Sushi v1.8.1's sorted_windows.zig and
// REDUCE_SOURCE in expert_exl3_kernels.zig. Each SIMD group finishes one
// route's H128/SVH in F32. Group zero reduces the original router slots in
// their original order with the separate Stark combine's explicit FMA.
kernel void stark_hy4_stq_pair1_finish_reduce(
    device const half* raw [[buffer(0)]], device const half* svh [[buffer(1)]],
    device const uint2* row_map [[buffer(2)]], device const uint* inverse [[buffer(3)]],
    device const float* route_weights [[buffer(4)]], device float* output [[buffer(5)]],
    constant uint& width [[buffer(6)]], uint3 group [[threadgroup_position_in_grid]],
    uint slot [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float values[16 * 128];
    const uint base = group.y * 16u;
    const uint row = inverse[base + slot];
    const uint channel = group.x * 128u + lane;
    if (row != 0xffffffffu) {
        const ulong at = ulong(row) * width + channel;
        float4 v = float4(raw[at], raw[at + 32u], raw[at + 64u], raw[at + 96u]);
        #pragma clang loop unroll(full)
        for (ushort step = 1; step < 32; step <<= 1) {
            const float4 peer = float4(simd_shuffle_xor(v.x, step), simd_shuffle_xor(v.y, step),
                                       simd_shuffle_xor(v.z, step), simd_shuffle_xor(v.w, step));
            v = lane & step ? peer - v : v + peer;
        }
        const float a = v.x + v.y, b = v.x - v.y, c = v.z + v.w, d = v.z - v.w;
        v = float4(a + c, b + d, a - c, b - d) * 0.08838834764831845f;
        const ulong scale = ulong(row_map[row].x) * width + channel;
        v *= float4(svh[scale], svh[scale + 32u], svh[scale + 64u], svh[scale + 96u]);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) values[slot * 128u + lane + 32u * i] = v[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slot == 0u) {
        float4 sum = 0.0f;
        for (uint i = 0; i < 16u; ++i) {
            if (inverse[base + i] == 0xffffffffu) continue;
            const uint at = i * 128u + lane;
            const float4 v = float4(values[at], values[at + 32u], values[at + 64u], values[at + 96u]);
            sum = fma(float4(route_weights[base + i]), v, sum);
        }
        const ulong at = ulong(group.y) * width + channel;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) output[at + 32u * i] = sum[i];
    }
}

static inline float4 stark_hy4_stq_hadamard_lane4(float4 value, ushort lane) {
    #pragma clang loop unroll(full)
    for (ushort step = 1; step < 32; step <<= 1) {
        const float4 peer(simd_shuffle_xor(value.x, step), simd_shuffle_xor(value.y, step),
                          simd_shuffle_xor(value.z, step), simd_shuffle_xor(value.w, step));
        value = lane & step ? peer - value : value + peer;
    }
    const float a = value.x + value.y, b = value.x - value.y;
    const float c = value.z + value.w, d = value.z - value.w;
    return float4(a + c, b + d, a - c, b - d) * 0.08838834764831845f;
}

template<typename T>
static inline float4 stark_hy4_stq_lane4_load(device const T* base) {
    return float4(base[0], base[32], base[64], base[96]);
}

static inline void stark_hy4_stq_lane4_store(device half* base, float4 v) {
    base[0] = half(v.x); base[32] = half(v.y); base[64] = half(v.z); base[96] = half(v.w);
}

// Finish the dense or shared gate and up projections, then prepare the
// down projection without storing three intermediate F32 planes.
kernel void stark_hy4_stq_ffn_prepare_down(
    device const half* gate [[buffer(0)]], device const half* up [[buffer(1)]],
    device const half* gate_svh [[buffer(2)]], device const half* up_svh [[buffer(3)]],
    device const half* down_suh [[buffer(4)]], device half* prepared [[buffer(5)]],
    constant uint& width [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    ushort simd [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint channel = (group.x * 4u + simd) * 128u + lane;
    const ulong offset = ulong(group.y) * width + channel;
    float4 g = stark_hy4_stq_hadamard_lane4(stark_hy4_stq_lane4_load(gate + offset), lane) *
               stark_hy4_stq_lane4_load(gate_svh + channel);
    float4 u = stark_hy4_stq_hadamard_lane4(stark_hy4_stq_lane4_load(up + offset), lane) *
               stark_hy4_stq_lane4_load(up_svh + channel);
    // Match hy4_flash_swiglu_clamp with an infinite limit and its division order.
    float4 activation;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        const float gv = min(stark_hy4_stq_projection_boundary(g[i]), INFINITY);
        const float uv = clamp(stark_hy4_stq_projection_boundary(u[i]), -INFINITY, INFINITY);
        activation[i] = stark_hy4_stq_projection_boundary((gv / (1.0f + precise::exp(-gv))) * uv);
    }
    activation *= stark_hy4_stq_lane4_load(down_suh + channel);
    stark_hy4_stq_lane4_store(prepared + offset, stark_hy4_stq_hadamard_lane4(activation, lane));
}

kernel void stark_hy4_stq_pair1_prepare_register(
    device const float* input [[buffer(0)]], device const half* scales [[buffer(1)]],
    device const uint2* row_map [[buffer(2)]], device half* output [[buffer(3)]],
    constant uint& width [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
    ushort simd [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint channel = (group.x * 4u + simd) * 128u + lane;
    const uint2 entry = row_map[group.y];
    float4 values = stark_hy4_stq_lane4_load(input + ulong(entry.y) * width + channel);
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) values[i] = stark_hy4_stq_projection_boundary(values[i]);
    values *= stark_hy4_stq_lane4_load(scales + ulong(entry.x) * width + channel);
    stark_hy4_stq_lane4_store(output + ulong(group.y) * width + channel,
                             stark_hy4_stq_hadamard_lane4(values, lane));
}

kernel void stark_hy4_stq_pair1_prepare_down_register(
    device const half* gate [[buffer(0)]], device const half* up [[buffer(1)]],
    device const half* gate_svh [[buffer(2)]], device const half* up_svh [[buffer(3)]],
    device const half* down_suh [[buffer(4)]], device const uint2* row_map [[buffer(5)]],
    device half* prepared [[buffer(6)]], constant uint& width [[buffer(7)]],
    constant float& limit [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]],
    ushort simd [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint channel = (group.x * 4u + simd) * 128u + lane;
    const ulong route_index = ulong(group.y) * width + channel;
    const ulong scale_index = ulong(row_map[group.y].x) * width + channel;
    const float4 g = stark_hy4_stq_hadamard_lane4(stark_hy4_stq_lane4_load(gate + route_index), lane) *
                     stark_hy4_stq_lane4_load(gate_svh + scale_index);
    const float4 u = stark_hy4_stq_hadamard_lane4(stark_hy4_stq_lane4_load(up + route_index), lane) *
                     stark_hy4_stq_lane4_load(up_svh + scale_index);
    const float4 gv = min(g, limit);
    const float4 activation = (gv * (1.0f / (1.0f + precise::exp(-gv)))) * clamp(u, -limit, limit);
    const float4 weighted = activation * stark_hy4_stq_lane4_load(down_suh + scale_index);
    stark_hy4_stq_lane4_store(prepared + route_index, stark_hy4_stq_hadamard_lane4(weighted, lane));
}
