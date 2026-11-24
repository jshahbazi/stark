// SPDX-License-Identifier: Apache-2.0
// Implements Inkling 2 Mini's vision equations and BF16 stage boundaries.
// Architecture reference: Ollama 6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8.
#include <metal_stdlib>
using namespace metal;

constant uint visual_width = 1024;
constant uint visual_head_dim = 64;

kernel void inkling_visual_linear(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device const bfloat *bias [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant uint4 &p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint column = group.x * 4 + sg;
    if (column >= p.y) return;
    uint row = group.y;
    device const bfloat *x = input + row * p.x;
    device const bfloat *w = weight + column * p.x;
    float4 sum(0.0f);
    for (uint k = lane * 4; k < p.x; k += 128) {
        float4 a = float4(*reinterpret_cast<device const bfloat4 *>(x + k));
        float4 b = float4(*reinterpret_cast<device const bfloat4 *>(w + k));
        sum = fma(a, b, sum);
    }
    float value = simd_sum((sum.x + sum.y) + (sum.z + sum.w));
    if (lane == 0) {
        output[row * p.y + column] = bfloat(value + (p.w ? float(bias[column]) : 0.0f));
    }
}

struct PositionSample {
    uint4 index;
    float4 coefficient;
};

kernel void inkling_visual_position(
    device const bfloat *patch [[buffer(0)]],
    device const bfloat *position [[buffer(1)]],
    device const bfloat *class_embedding [[buffer(2)]],
    device const bfloat *class_position [[buffer(3)]],
    device const PositionSample *samples [[buffer(4)]],
    device bfloat *output [[buffer(5)]],
    constant uint &count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
#pragma clang fp contract(off)
    if (gid >= count * visual_width) return;
    uint row = gid / visual_width;
    uint channel = gid % visual_width;
    if (row == 0) {
        output[gid] = bfloat(float(class_embedding[channel]) + float(class_position[channel]));
        return;
    }
    PositionSample sample = samples[row - 1];
    float pos = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        pos += float(position[sample.index[i] * visual_width + channel]) * sample.coefficient[i];
    }
    float rounded = float(bfloat(pos));
    output[gid] = bfloat(float(patch[(row - 1) * visual_width + channel]) + rounded);
}

// The source normalizes the centered values in F32 and rounds before the
// BF16 affine transform. Each lane handles eight contiguous channels.
kernel void inkling_visual_norm(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]],
    device const bfloat *bias [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
#pragma clang fp contract(off)
    threadgroup float means[4];
    threadgroup float variances[4];
    uint base = row * visual_width + tid * 8;
    float x[8];
    float sum = 0.0f;
    for (uint j = 0; j < 8; ++j) {
        x[j] = float(input[base + j]);
        sum += x[j];
    }
    sum = simd_sum(sum);
    if (lane == 0) means[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = simd_sum(lane < 4 ? means[lane] : 0.0f) * (1.0f / visual_width);
    sum = 0.0f;
    for (uint j = 0; j < 8; ++j) {
        x[j] -= mean;
        sum += x[j] * x[j];
    }
    sum = simd_sum(sum);
    if (lane == 0) variances[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inverse = metal::precise::rsqrt(simd_sum(lane < 4 ? variances[lane] : 0.0f) * (1.0f / visual_width) + 1.0e-5f);
    for (uint j = 0; j < 8; ++j) {
        uint channel = tid * 8 + j;
        float unit = float(bfloat(x[j] * inverse));
        float affine = float(bfloat(unit * float(weight[channel])));
        output[base + j] = bfloat(affine + float(bias[channel]));
    }
}

kernel void inkling_visual_rotary(
    device bfloat *queries [[buffer(0)]],
    device bfloat *keys [[buffer(1)]],
    device const float2 *phases [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
#pragma clang fp contract(off)
    if (gid >= count * (visual_width / 2)) return;
    uint row = gid / (visual_width / 2);
    float2 phase = phases[row * (visual_head_dim / 2) + gid % (visual_head_dim / 2)];
    for (uint kind = 0; kind < 2; ++kind) {
        device bfloat *values = (kind ? keys : queries) + gid * 2;
        float even = float(values[0]), odd = float(values[1]);
        values[0] = bfloat(even * phase.x - odd * phase.y);
        values[1] = bfloat(even * phase.y + odd * phase.x);
    }
}

kernel void inkling_visual_attention(
    device const bfloat *q [[buffer(0)]],
    device const bfloat *k [[buffer(1)]],
    device const bfloat *v [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    // Four SIMD groups stream every fourth key for one query/head. This has
    // linear workspace and retains F32 accumulators until the result store.
    threadgroup float maxima[4], masses[4];
    threadgroup float2 values[4 * 32];
    uint column = group.x * visual_head_dim + lane * 2;
    uint query = group.y * visual_width + column;
    float2 qv = float2(float(q[query]), float(q[query + 1])) * (1.0f / 8.0f);
    float max_score = -INFINITY, mass = 0.0f;
    float2 attention(0.0f);
    for (uint row = sg; row < count; row += 4) {
        uint i = row * visual_width + column;
        float score = simd_sum(qv.x * float(k[i]) + qv.y * float(k[i + 1]));
        float peak = max(max_score, score);
        float old = fast::exp(max_score - peak);
        float next = fast::exp(score - peak);
        attention = fma(attention, float2(old), float2(float(v[i]), float(v[i + 1])) * next);
        mass = fma(mass, old, next);
        max_score = peak;
    }
    if (lane == 0) {
        maxima[sg] = max_score;
        masses[sg] = mass;
    }
    values[sg * 32 + lane] = attention;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    float peak = max(max(maxima[0], maxima[1]), max(maxima[2], maxima[3]));
    attention = float2(0.0f);
    mass = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        float coefficient = fast::exp(maxima[i] - peak);
        attention += coefficient * values[i * 32 + lane];
        mass += coefficient * masses[i];
    }
    float2 result = attention / mass;
    output[query] = bfloat(result.x);
    output[query + 1] = bfloat(result.y);
}

kernel void inkling_visual_residual(
    device bfloat *state [[buffer(0)]],
    device const bfloat *change [[buffer(1)]],
    device const bfloat *scale [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count * visual_width) return;
    float term = float(bfloat(float(change[gid]) * float(scale[gid % visual_width])));
    state[gid] = bfloat(float(state[gid]) + term);
}

kernel void inkling_visual_gelu(
    device bfloat *values [[buffer(0)]],
    constant uint &elements [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= elements) return;
    float x = float(values[gid]);
    float argument = float(bfloat(x * float(bfloat(0.7071067811865476f))));
    float cdf = float(bfloat(erf_precise(argument)));
    float factor = float(bfloat(1.0f + cdf));
    values[gid] = bfloat(float(bfloat(0.5f * x)) * factor);
}

kernel void inkling_visual_merge(
    device const bfloat *features [[buffer(0)]],
    device bfloat *merged [[buffer(1)]],
    constant uint2 &grid [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint patch = gid / visual_width;
    if (patch >= grid.x * grid.y * 4) return;
    uint block = patch / 4;
    uint inner = patch % 4;
    uint y = 2 * (block / grid.x) + inner / 2;
    uint x = 2 * (block % grid.x) + inner % 2;
    uint source = 1 + y * (2 * grid.x) + x;
    merged[gid] = features[source * visual_width + gid % visual_width];
}
