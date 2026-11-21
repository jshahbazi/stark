// SPDX-License-Identifier: Apache-2.0
// Convert the packed graph's operands into the pinned MLX diagnostic ABI.
#include <metal_stdlib>
using namespace metal;

constant uint head_width = 128, query_heads = 24, key_heads = 8, distances = 1024;
constant uint query_width = head_width * query_heads, key_width = head_width * key_heads;

kernel void ink_global_probe_query(
    device const bfloat *input [[buffer(0)]], device float *query [[buffer(1)]],
    constant uint2 &p [[buffer(2)]], uint2 tid [[thread_position_in_grid]]) {
    const uint head = tid.x / head_width, channel = tid.x % head_width;
    query[(head * p.x + tid.y) * head_width + channel] = float(input[(p.y + tid.y) * query_width + tid.x]);
}

kernel void ink_global_probe_kv(
    device const bfloat *ring_k [[buffer(0)]], device const bfloat *ring_v [[buffer(1)]],
    device const bfloat *current_k [[buffer(2)]], device const bfloat *current_v [[buffer(3)]],
    device float *keys [[buffer(4)]], device float *values [[buffer(5)]],
    constant uint2 &p [[buffer(6)]], uint2 tid [[thread_position_in_grid]]) {
    const uint head = tid.x / head_width, channel = tid.x % head_width;
    const uint output = (head * p.y + tid.y) * head_width + channel;
    if (tid.y < p.x) {
        keys[output] = float(ring_k[tid.y * key_width + tid.x]);
        values[output] = float(ring_v[tid.y * key_width + tid.x]);
    } else {
        keys[output] = float(current_k[(tid.y - p.x) * key_width + tid.x]);
        values[output] = float(current_v[(tid.y - p.x) * key_width + tid.x]);
    }
}

kernel void ink_global_probe_mask(
    device const bfloat *table [[buffer(0)]], device const bfloat *sink [[buffer(1)]],
    device float *mask [[buffer(2)]], constant uint4 &p [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
    const uint queries = p.x, keys = p.y, offset = p.z;
    if (tid.x >= keys) return;
    const uint head = tid.y / queries, row = tid.y % queries;
    const int distance = int(keys - queries + row) - int(tid.x);
    const uint query_head = (offset + row) * query_heads + head;
    const float relative = distance >= 0 && distance < int(distances)
        ? float(table[query_head * distances + distance]) : 0.0f;
    mask[tid.y * keys + tid.x] = distance < 0 ? -INFINITY : relative - float(sink[query_head]);
}

kernel void ink_global_probe_gate(
    device const float *input [[buffer(0)]], device const bfloat *gate [[buffer(1)]],
    device bfloat *output [[buffer(2)]], constant uint2 &p [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
    const uint at = (p.y + tid.y) * query_width + tid.x;
    const float g = float(gate[at]);
    const float y = 1.0f / (1.0f + precise::exp(abs(g)));
    output[at] = bfloat(input[tid.y * query_width + tid.x] * (g < 0.0f ? y : 1.0f - y));
}
