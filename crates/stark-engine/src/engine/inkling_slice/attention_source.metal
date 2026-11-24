// SPDX-License-Identifier: MIT
/*
The long path follows Ollama mlx/relative_attention.go at
6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8. The Stark version uses single-query
dispatch, token-major KV rings, interleaved FP32 numerators and statistics,
and a fused output gate. The short-path arithmetic follows Apple's MLX
sdpa_vector at fc344c986c4169d99f76b5d95ae6b4a58a359a63
(Copyright © 2024 Apple Inc., LICENSE.mlx-reference).

MIT License

Copyright (c) Ollama

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <metal_stdlib>
using namespace metal;

constant uint ink_source_part_width = 130;
constant uint ink_source_distances = 1024;
constant uint ink_source_kv_width = 1024;

kernel void ink_attention_source_parts(
    device const bfloat *q [[buffer(0)]], device const bfloat *keys [[buffer(1)]],
    device const bfloat *values [[buffer(2)]], device const bfloat *table [[buffer(3)]],
    device const bfloat *sink [[buffer(4)]], device float *parts [[buffer(5)]],
    constant uint4 &p [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint shard = group.x * 4 + sg;
    uint head = group.y;
    uint visible = min(p.x + 1, p.z == 0 ? p.y : min(p.y, p.z));
    uint first = p.x + 1 - visible;
    uint shards = p.w;
    uint component = lane * 4;
    uint kv_component = (head / 3) * 128 + component;
    float4 query = float4(*reinterpret_cast<device const bfloat4 *>(q + head * 128 + component))
        * 0.08838834764831845f;
    float sink_logit = float(sink[head]);

    // Include the zero-logit, zero-value sink once, in the first shard.
    float peak = shard == 0 ? 0.0f : (shards == 32 ? -FLT_MAX : -INFINITY);
    float weight = shard == 0 ? 1.0f : 0.0f;
    float4 numerator(0.0f);
    for (uint i = shard; i < visible; i += shards) {
        uint base = ((first + i) % p.y) * ink_source_kv_width + kv_component;
        float4 key = float4(*reinterpret_cast<device const bfloat4 *>(keys + base));
        float dot = 0.0f;
        if (shards == 32) {
            for (uint j = 0; j < 4; ++j) dot += query[j] * key[j];
        } else {
            for (uint j = 0; j < 4; ++j) dot = fma(query[j], key[j], dot);
        }
        uint distance = visible - i - 1;
        float bias = distance < ink_source_distances
            ? float(table[head * ink_source_distances + distance]) : 0.0f;
        float score = simd_sum(dot) + (bias - sink_logit);
        float next = max(peak, score);
        float prior = fast::exp(peak - next);
        float probability = fast::exp(score - next);
        float4 value = float4(*reinterpret_cast<device const bfloat4 *>(values + base));
        if (shards == 32) {
            numerator = numerator * prior + probability * value;
            weight = weight * prior + probability;
        } else {
            numerator = fma(numerator, float4(prior), probability * value);
            weight = fma(weight, prior, probability);
        }
        peak = next;
    }

    device float *part = parts + (head * shards + shard) * ink_source_part_width;
    for (uint j = 0; j < 4; ++j) part[component + j] = numerator[j];
    if (lane == 0) {
        part[128] = peak;
        part[129] = weight;
    }
}

kernel void ink_attention_source_gate(
    device const float *parts [[buffer(0)]], device const bfloat *gate [[buffer(1)]],
    device bfloat *result [[buffer(2)]], constant uint2 &p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint head = group.y;
    uint component = (group.x * 4 + sg) * 4;
    device const float *head_parts = parts + head * p.y * ink_source_part_width;

    float peak = -INFINITY;
    for (uint shard = lane; shard < p.x; shard += 32) {
        peak = max(peak, head_parts[shard * ink_source_part_width + 128]);
    }
    peak = simd_max(peak);

    float weight = 0.0f;
    float4 numerator(0.0f);
    for (uint shard = lane; shard < p.x; shard += 32) {
        device const float *part = head_parts + shard * ink_source_part_width;
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
