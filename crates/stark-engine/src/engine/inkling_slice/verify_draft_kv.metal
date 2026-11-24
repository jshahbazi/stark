// SPDX-License-Identifier: MIT
// Copyright (c) Ollama
// Adapts the stock DSpark attention recurrence in dspark.metal.
kernel void ink_draft_kv_four_rows(
    device const bfloat *queries [[buffer(0)]], device const bfloat *keys [[buffer(1)]],
    device const bfloat *values [[buffer(2)]], device const bfloat *block_keys [[buffer(3)]],
    device const bfloat *block_values [[buffer(4)]], device float *parts [[buffer(5)]],
    constant uint3 &p [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint batch = 4;
    const uint head = group.y % heads;
    const uint row = group.y / heads * batch;
    const uint offset = head_dim * (head / 3) + lane * 4;
    float4 q[batch], output[batch] = {};
    float high[batch], total[batch] = {};
#pragma clang loop unroll(full)
    for (uint j = 0; j < batch; ++j) {
        const uint view = (row + j) * heads + head;
        q[j] = float4(*reinterpret_cast<device const bfloat4 *>(
            queries + view * head_dim + lane * 4)) * 0.08838834765f;
        high[j] = -INFINITY;
    }
    for (uint i = p.x + group.x; i < p.y + rows; i += shards) {
        const bool committed = i < p.y;
        const uint index = (committed ? i % p.z : i - p.y) * kv_width + offset;
        const float4 key = float4(*reinterpret_cast<device const bfloat4 *>(
            (committed ? keys : block_keys) + index));
        const float4 value = float4(*reinterpret_cast<device const bfloat4 *>(
            (committed ? values : block_values) + index));
#pragma clang loop unroll(full)
        for (uint j = 0; j < batch; ++j) {
            const float score = simd_sum(dot(q[j], key));
            const float next_high = max(high[j], score);
            const float keep = exp(high[j] - next_high);
            const float add = exp(score - next_high);
            output[j] = output[j] * keep + value * add;
            total[j] = total[j] * keep + add;
            high[j] = next_high;
        }
    }
#pragma clang loop unroll(full)
    for (uint j = 0; j < batch; ++j) {
        const uint view = (row + j) * heads + head;
        device float *dest = parts + (view * shards + group.x) * part_width;
        for (uint i = 0; i < 4; ++i) dest[lane * 4 + i] = output[j][i];
        if (lane == 0) {
            dest[head_dim] = high[j];
            dest[head_dim + 1] = total[j];
        }
    }
}
