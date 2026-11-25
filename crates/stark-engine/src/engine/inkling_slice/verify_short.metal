// SPDX-License-Identifier: MIT
/*
Verifier rows below 1,024 visible keys, all rows in one launch. Each row runs
the 32-shard short path of ink_attention_source_parts and
ink_attention_source_gate (attention_source.metal) unchanged; see that
translation unit for provenance and the license text. The short-path
arithmetic follows Apple's MLX sdpa_vector at
fc344c986c4169d99f76b5d95ae6b4a58a359a63 (Copyright © 2024 Apple Inc.,
LICENSE.mlx-reference) and Ollama mlx/relative_attention.go at
6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8 (MIT, Copyright (c) Ollama).
*/

#include <metal_stdlib>
using namespace metal;

constant uint ink_short_part = 130;
constant uint ink_short_heads = 24;
constant uint ink_short_shards = 32;
constant uint ink_short_distances = 1024;
constant uint ink_short_kv_width = 1024;

// Grid: (8, heads * rows), 128 threads. Ordinary shape: x = first row
// position, y = cache capacity. Fork shape adds z = physical row count and
// w = the sibling's logical depth.
kernel void ink_verify_short_parts(
    device const bfloat *q0 [[buffer(0)]], device const bfloat *q1 [[buffer(1)]],
    device const bfloat *q2 [[buffer(2)]], device const bfloat *q3 [[buffer(3)]],
    device const bfloat *t0 [[buffer(4)]], device const bfloat *t1 [[buffer(5)]],
    device const bfloat *t2 [[buffer(6)]], device const bfloat *t3 [[buffer(7)]],
    device const bfloat *s0 [[buffer(8)]], device const bfloat *s1 [[buffer(9)]],
    device const bfloat *s2 [[buffer(10)]], device const bfloat *s3 [[buffer(11)]],
    device const bfloat *keys [[buffer(12)]], device const bfloat *values [[buffer(13)]],
    device float *all_parts [[buffer(14)]],
#ifdef INK_VERIFY_FORK
    constant uint4 &shape [[buffer(15)]],
#else
    constant uint2 &shape [[buffer(15)]],
#endif
    uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    device const bfloat *queries[] = {q0, q1, q2, q3};
    device const bfloat *tables[] = {t0, t1, t2, t3};
    device const bfloat *sinks[] = {s0, s1, s2, s3};
    const uint row = group.y / ink_short_heads;
    const uint head = group.y % ink_short_heads;
#ifdef INK_VERIFY_FORK
    const bool sibling = row + 1 == shape.z;
    const uint logical_depth = sibling ? shape.w : row;
#else
    const uint logical_depth = row;
#endif
    device const bfloat *q = queries[row];
    device const bfloat *table = tables[row];
    device float *parts = all_parts + row * ink_short_heads * ink_short_shards * ink_short_part;
    uint shard = group.x * 4 + sg;
    uint visible = min(shape.x + logical_depth + 1, shape.y);
    uint first = shape.x + logical_depth + 1 - visible;
    uint component = lane * 4;
    uint kv_component = (head / 3) * 128 + component;
    float4 query = float4(*reinterpret_cast<device const bfloat4 *>(q + head * 128 + component))
        * 0.08838834764831845f;
    float sink_logit = float(sinks[row][head]);
    float peak = shard == 0 ? 0.0f : -FLT_MAX;
    float weight = shard == 0 ? 1.0f : 0.0f;
    float4 numerator(0.0f);
    for (uint i = shard; i < visible; i += ink_short_shards) {
        const uint logical_position = first + i;
#ifdef INK_VERIFY_FORK
        const uint physical_position = sibling && logical_position == shape.x + logical_depth
            ? shape.x + shape.z - 1 : logical_position;
#else
        const uint physical_position = logical_position;
#endif
        uint base = (physical_position % shape.y) * ink_short_kv_width + kv_component;
        float4 key = float4(*reinterpret_cast<device const bfloat4 *>(keys + base));
        float dot = 0.0f;
        for (uint j = 0; j < 4; ++j) dot += query[j] * key[j];
        uint distance = visible - i - 1;
        float bias = distance < ink_short_distances
            ? float(table[head * ink_short_distances + distance]) : 0.0f;
        float score = simd_sum(dot) + (bias - sink_logit);
        float next = max(peak, score);
        float prior = fast::exp(peak - next);
        float probability = fast::exp(score - next);
        float4 value = float4(*reinterpret_cast<device const bfloat4 *>(values + base));
        numerator = numerator * prior + probability * value;
        weight = weight * prior + probability;
        peak = next;
    }
    device float *part = parts + (head * ink_short_shards + shard) * ink_short_part;
    for (uint j = 0; j < 4; ++j) part[component + j] = numerator[j];
    if (lane == 0) {
        part[128] = peak;
        part[129] = weight;
    }
}

// Grid: (8, heads * rows), 128 threads.
kernel void ink_verify_short_gate(
    device const float *all_parts [[buffer(0)]], device const bfloat *g0 [[buffer(1)]],
    device const bfloat *g1 [[buffer(2)]], device const bfloat *g2 [[buffer(3)]],
    device const bfloat *g3 [[buffer(4)]], device bfloat *o0 [[buffer(5)]],
    device bfloat *o1 [[buffer(6)]], device bfloat *o2 [[buffer(7)]],
    device bfloat *o3 [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    device const bfloat *gates[] = {g0, g1, g2, g3};
    device bfloat *outputs[] = {o0, o1, o2, o3};
    const uint row = group.y / ink_short_heads;
    uint head = group.y % ink_short_heads;
    device const bfloat *gate = gates[row];
    device bfloat *result = outputs[row];
    device const float *parts = all_parts + row * ink_short_heads * ink_short_shards * ink_short_part;
    uint component = (group.x * 4 + sg) * 4;
    device const float *head_parts = parts + head * ink_short_shards * ink_short_part;

    float peak = -INFINITY;
    for (uint shard = lane; shard < ink_short_shards; shard += 32) {
        peak = max(peak, head_parts[shard * ink_short_part + 128]);
    }
    peak = simd_max(peak);

    float weight = 0.0f;
    float4 numerator(0.0f);
    for (uint shard = lane; shard < ink_short_shards; shard += 32) {
        device const float *part = head_parts + shard * ink_short_part;
        float factor = fast::exp(part[128] - peak);
        weight = fma(part[129], factor, weight);
        for (uint j = 0; j < 4; ++j) {
            numerator[j] = fma(part[component + j], factor, numerator[j]);
        }
    }
    weight = simd_sum(weight);
    for (uint j = 0; j < 4; ++j) {
        float answer = simd_sum(numerator[j]) / weight;
        if (lane == 0) {
            uint index = head * 128 + component + j;
            float g = float(gate[index]);
            float y = 1.0f / (1.0f + metal::precise::exp(abs(g)));
            result[index] = bfloat(answer * (g < 0.0f ? y : 1.0f - y));
        }
    }
}
