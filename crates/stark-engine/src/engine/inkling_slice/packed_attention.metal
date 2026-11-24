// SPDX-License-Identifier: Apache-2.0 AND MIT
/*
The long path follows Ollama mlx/relative_attention.go at
6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8. The Stark version uses packed-query
dispatch, token-major KV rings, interleaved FP32 numerators and statistics,
and a fused output gate. The short-path arithmetic follows Apple's MLX
sdpa_vector at fc344c986c4169d99f76b5d95ae6b4a58a359a63
(Copyright © 2024 Apple Inc., LICENSE.mlx-reference).
Convolution and head normalization are batched from Stark serial.metal
(Apache-2.0).

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

constant uint heads = 24, head_size = 128, distances = 1024;
constant uint kv = 1024;
constant uint part_width = 130, max_shards = 64;

// Read preceding packed rows directly. The carried history stays read-only
// until a separate commit, allowing all rows to run at once.
kernel void ink_packed_conv(
    device const bfloat *x [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device const bfloat *dynamic [[buffer(2)]], device const bfloat *history [[buffer(3)]],
    device bfloat *output [[buffer(4)]], constant uint4 &p [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const uint width = p.x, row = tid.y, column = tid.x;
    if (column >= width || row >= p.w) return;
    const uint position = p.y + row;
    const uint index = row * width + column;
    const bool is_dynamic = p.z != 0;
    const float current = float(x[index]);
    float result = 0;
    for (uint tap = 0; tap < 4; ++tap) {
        const uint lag = is_dynamic ? tap : 3 - tap;
        if (position < lag) continue;
        const float value = row >= lag ? float(x[(row - lag) * width + column])
            : float(history[((position - lag) % 3) * width + column]);
        const float coefficient = is_dynamic
            ? float(weight[tap * width + column]) + float(dynamic[(row * (width / 128) + column / 128) * 4 + tap])
            : float(weight[column * 4 + tap]);
#ifdef INK_PACKED_SOURCE_CONV
        {
#pragma clang fp contract(off) reassociate(off)
            result += coefficient * value;
        }
#else
        result = fma(coefficient, value, result);
#endif
    }
    output[index] = bfloat((result + current) * (is_dynamic ? 1.0f / 3.0f : 0.447213595499958f));
}

kernel void ink_packed_commit_conv(
    device const bfloat *input [[buffer(0)]], device bfloat *history [[buffer(1)]],
    constant uint4 &p [[buffer(2)]], uint2 tid [[thread_position_in_grid]]
) {
    const uint rows = min(3u, p.w);
    if (tid.x >= p.x || tid.y >= rows) return;
    const uint row = p.w - rows + tid.y;
    history[((p.y + row) % 3) * p.x + tid.x] = input[row * p.x + tid.x];
}

kernel void ink_packed_heads(
    device bfloat *q [[buffer(0)]], device const bfloat *k [[buffer(1)]],
    device const bfloat *v [[buffer(2)]], device bfloat *relative [[buffer(3)]],
    device const bfloat *q_scale [[buffer(4)]], device const bfloat *r_scale [[buffer(5)]],
    device bfloat *keys [[buffer(6)]], device bfloat *values [[buffer(7)]],
    constant uint2 &p [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = 4 * group.x + sg;
    if (row >= 64) return;
    bool is_q = row < 24, is_k = row >= 24 && row < 32, is_v = row >= 32 && row < 40;
    bool is_rel = row >= 40;
    uint head = is_q ? row : is_k ? row - 24 : is_v ? row - 32 : row - 40;
    uint width = is_rel ? 16 : 128;
    device const bfloat *input = is_q ? q : is_k ? k : is_v ? v : relative;
    device const bfloat *gain = is_q ? q_scale : r_scale;
    const uint token = group.y;
    const uint position = p.x + token;
    const uint token_width = is_q ? 24 * 128 : is_rel ? 24 * 16 : kv;
    const uint token_offset = token * token_width;
    float4 vals = 0;
    float norm = 0;
    for (uint j = 0; j < width / 32 + uint(is_rel); ++j) {
#ifdef INK_PACKED_SOURCE_HEADS
        const uint col = is_rel ? lane : 4 * lane + j;
#else
        uint col = lane + j * 32;
#endif
        if (col < width) {
            uint index = head * width + col;
            float value = float(input[token_offset + index]);
            if (is_q || is_rel) value *= float(gain[index]);
#ifdef INK_PACKED_SOURCE_HEADS
            if (is_v) norm = fma(value, value, norm);
            else {
#pragma clang fp contract(off) reassociate(off)
                norm += value * value;
            }
#else
            norm = fma(value, value, norm);
#endif
            vals[j] = value;
        }
    }
#ifdef INK_PACKED_SOURCE_HEADS
    // MLX takes a serial reduction for the 16-channel relative head. Its
    // 128-channel contiguous reductions use four neighboring reads per lane.
    if (is_rel) {
        float total = 0;
        if (lane == 0) {
#pragma clang fp contract(off) reassociate(off)
            const uint row_offset = token_offset + head * 16;
            for (uint i = 0; i < 16; ++i) {
                const float x = float(input[row_offset + i]) * float(gain[head * 16 + i]);
                total += x * x;
            }
        }
        norm = simd_broadcast(total, 0);
    } else {
        norm = simd_sum(norm);
    }
#else
    norm = simd_sum(norm);
#endif
    float limit = is_rel ? 8.0f : 22.62741699797f;
#ifdef INK_PACKED_SOURCE_HEADS
    float scale = is_v ? precise::rsqrt(norm / 128.0f + 1e-6f) : min(1.0f, limit / (precise::sqrt(norm) + 1e-6f));
#else
    float scale = is_v ? rsqrt(norm / 128.0f + 1e-6f) : min(1.0f, limit / (sqrt(norm) + 1e-6f));
#endif
    if ((is_q || is_rel) && p.y == 0 && position >= 8192) {
        scale *= ink_long_position_gain(position);
    }
    for (uint j = 0; j < width / 32 + uint(is_rel); ++j) {
#ifdef INK_PACKED_SOURCE_HEADS
        const uint col = is_rel ? lane : 4 * lane + j;
#else
        uint col = lane + j * 32;
#endif
        if (col < width) {
            uint index = head * width + col;
            bfloat result = bfloat(vals[j] * scale);
            if (is_q) q[token_offset + index] = result;
            else if (is_k) keys[token_offset + index] = result;
            else if (is_v) values[token_offset + index] = result;
            else relative[token_offset + index] = result;
        }
    }
}

// Copy only the latest write for each physical slot if the chunk wraps.
kernel void ink_packed_commit_kv(
    device const bfloat *current_keys [[buffer(0)]], device const bfloat *current_values [[buffer(1)]],
    device bfloat *keys [[buffer(2)]], device bfloat *values [[buffer(3)]],
    constant uint4 &p [[buffer(4)]], // start, rows, cache slots, local window or zero
    uint2 tid [[thread_position_in_grid]]) {
    const uint copy_rows = min(p.y, p.z);
    if (tid.x >= kv || tid.y >= copy_rows) return;
    const uint row = p.y - copy_rows + tid.y;
    const uint source = row * kv + tid.x;
    const uint target = ((p.x + row) % p.z) * kv + tid.x;
    keys[target] = current_keys[source];
    values[target] = current_values[source];
}

kernel void ink_packed_chronological_kv(
    device const bfloat *ring_keys [[buffer(0)]], device const bfloat *ring_values [[buffer(1)]],
    device const bfloat *keys [[buffer(2)]], device const bfloat *values [[buffer(3)]],
    device bfloat *ordered_keys [[buffer(4)]], device bfloat *ordered_values [[buffer(5)]],
    constant uint *p [[buffer(6)]], // start, rows, ring slots, aligned start, first visible
    uint2 tid [[thread_position_in_grid]]) {
    const uint token = p[3] + tid.y, column = tid.x;
    bfloat key = 0, value = 0;
    if (token >= p[4] && token < p[0] + p[1]) {
        if (token < p[0]) {
            const uint source = (token % p[2]) * kv + column;
            key = ring_keys[source];
            value = ring_values[source];
        } else {
            const uint source = (token - p[0]) * kv + column;
            key = keys[source];
            value = values[source];
        }
    }
    ordered_keys[tid.y * kv + column] = key;
    ordered_values[tid.y * kv + column] = value;
}

kernel void ink_packed_attention_table(
    device const bfloat *relative [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device bfloat *table [[buffer(2)]], uint2 tid [[thread_position_in_grid]]) {
    float acc = 0.0f;
    for (uint i = 0; i < 16; ++i)
        acc = fma(float(relative[tid.y * 16 + i]), float(weight[i * distances + tid.x]), acc);
    table[tid.y * distances + tid.x] = bfloat(acc);
}

kernel void ink_packed_attention_parts(
    device const bfloat *queries [[buffer(0)]], device const bfloat *keys [[buffer(1)]],
    device const bfloat *values [[buffer(2)]], device const bfloat *tables [[buffer(3)]],
    device const bfloat *sinks [[buffer(4)]], device float *parts [[buffer(5)]],
    device const bfloat *current_keys [[buffer(6)]], device const bfloat *current_values [[buffer(7)]],
    constant uint *p [[buffer(8)]], // start, rows, cache slots, window, query offset
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint shard = group[0] * 4 + sg, head = group.y % heads;
    const uint query_head = p[4] * heads + group.y;
    const uint position = p[0] + query_head / heads;
    const uint visible = min(position + 1, p[3] == 0 ? p[2] : min(p[2], p[3])), first = position + 1 - visible;
    const uint shards = visible < distances ? 32 : 64;
    if (shard >= shards) return;
    const uint component = lane * 4, kv_component = (head / 3) * head_size + component;
    const float4 query = float4(*reinterpret_cast<device const bfloat4 *>(
        queries + query_head * head_size + component)) * 0.08838834764831845f;
    const float sink = float(sinks[query_head]);

    float peak = shard == 0 ? 0.0f : (shards == 32 ? -FLT_MAX : -INFINITY);
    float weight = shard == 0 ? 1.0f : 0.0f;
    float4 numerator(0.0f);
    for (uint i = shard; i < visible; i += shards) {
        const uint absolute = first + i;
        const bool current = absolute >= p[0];
        const uint base = (current ? absolute - p[0] : absolute % p[2]) * kv + kv_component;
        const auto key_data = current ? current_keys : keys;
        const auto value_data = current ? current_values : values;
        const float4 key = float4(*reinterpret_cast<device const bfloat4 *>(key_data + base));
        float dot = 0.0f;
        if (shards == 32) {
            for (uint j = 0; j < 4; ++j) dot += query[j] * key[j];
        } else {
            for (uint j = 0; j < 4; ++j) dot = fma(query[j], key[j], dot);
        }
        const uint distance = visible - i - 1;
        const float bias = distance < distances ? float(tables[query_head * distances + distance]) : 0.0f;
        const float score = simd_sum(dot) + (bias - sink);
        const float next = max(peak, score);
        const float prior = fast::exp(peak - next), probability = fast::exp(score - next);
        const float4 value = float4(*reinterpret_cast<device const bfloat4 *>(value_data + base));
        if (shards == 32) {
            numerator = numerator * prior + probability * value;
            weight = weight * prior + probability;
        } else {
            numerator = fma(numerator, float4(prior), probability * value);
            weight = fma(weight, prior, probability);
        }
        peak = next;
    }

    device float *part = parts + (group.y * max_shards + shard) * part_width;
    for (uint j = 0; j < 4; ++j) part[component + j] = numerator[j];
    if (lane == 0) {
        part[128] = peak;
        part[129] = weight;
    }
}

kernel void ink_packed_attention_gate(
    device const float *parts [[buffer(0)]], device const bfloat *gate [[buffer(1)]],
    device bfloat *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint query_head = p[4] * heads + group.y;
    const uint visible = min(p[0] + query_head / heads + 1, p[3] == 0 ? p[2] : min(p[2], p[3]));
    const uint shards = visible < distances ? 32 : 64;
    const uint component = (group[0] * 4 + sg) * 4;
    const device float *head_parts = parts + group.y * max_shards * part_width;
    float peak = -INFINITY;
    for (uint shard = lane; shard < shards; shard += 32)
        peak = max(peak, head_parts[shard * part_width + 128]);
    peak = simd_max(peak);

    float weight = 0.0f;
    float4 numerator(0.0f);
    for (uint shard = lane; shard < shards; shard += 32) {
        const device float *part = head_parts + shard * part_width;
        const float factor = fast::exp(part[128] - peak);
        weight = fma(part[129], factor, weight);
        for (uint j = 0; j < 4; ++j) numerator[j] = fma(part[component + j], factor, numerator[j]);
    }
    weight = simd_sum(weight);
    for (uint j = 0; j < 4; ++j) {
        const float answer = simd_sum(numerator[j]) / weight;
        if (lane == 0) {
            const uint index = query_head * head_size + component + j;
            const float g = float(gate[index]);
            const float y = 1.0f / (1.0f + metal::precise::exp(abs(g)));
            output[index] = bfloat(answer * (g < 0.0f ? y : 1.0f - y));
        }
    }
}
