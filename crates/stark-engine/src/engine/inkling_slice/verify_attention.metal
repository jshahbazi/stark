// SPDX-License-Identifier: MIT
// Short-row variant of attention_source.metal, retaining its Ollama and MLX
// arithmetic and license. See that translation unit for provenance.
#include <metal_stdlib>
using namespace metal;

struct InkLocalQueries {
    device const bfloat *q[4];
    device const bfloat *relative[4];
    device const bfloat *sink[4];
};

template <uint Rows>
inline void ink_verify_shared_local(
    const thread InkLocalQueries &input,
    device const bfloat *keys, device const bfloat *values, device const bfloat *overwritten,
    device float *parts, uint start, uint whole_rows, uint row_start, uint head, uint physical_shard, uint lane
) {
    constexpr uint Capacity = 1024, Heads = 24, Shards = 64, Part = 130;
    const uint begin = start + row_start;
    const uint first = begin + 1 - Capacity;
    const uint component = lane * 4;
    const uint kv_component = (head / 3) * 128 + component;
    float4 queries[Rows], numerators[Rows];
    float sink[Rows], peak[Rows], mass[Rows];
    uint shard[Rows];
#pragma clang loop unroll(full)
    for (uint row = 0; row < Rows; ++row) {
        queries[row] = float4(*reinterpret_cast<device const bfloat4 *>(input.q[row_start + row] + head * 128 + component))
            * 0.08838834764831845f;
        sink[row] = float(input.sink[row_start + row][head]);
        // A workgroup follows one absolute-key residue class. Shift that
        // class into each query's original logical shard before reducing.
        shard[row] = (physical_shard - first - row) & (Shards - 1);
        peak[row] = shard[row] == 0 ? 0.0f : -INFINITY;
        mass[row] = shard[row] == 0 ? 1.0f : 0.0f;
        numerators[row] = float4(0.0f);
    }

    uint first_key = (first & ~(Shards - 1)) + physical_shard;
    if (first_key < first) first_key += Shards;
    for (uint key_position = first_key; key_position < begin + Rows; key_position += Shards) {
        // Later provisional writes can displace keys still visible to an
        // earlier query. Use the same pre-write slots saved for rollback.
        const uint displaced = key_position + Capacity - start;
        float4 key, value;
        if (key_position < start && displaced < whole_rows) {
            const uint saved = displaced * Capacity + kv_component;
            key = float4(*reinterpret_cast<device const bfloat4 *>(overwritten + saved));
            value = float4(*reinterpret_cast<device const bfloat4 *>(overwritten + 4 * Capacity + saved));
        } else {
            const uint live = (key_position & (Capacity - 1)) * Capacity + kv_component;
            key = float4(*reinterpret_cast<device const bfloat4 *>(keys + live));
            value = float4(*reinterpret_cast<device const bfloat4 *>(values + live));
        }
#pragma clang loop unroll(full)
        for (uint row = 0; row < Rows; ++row) {
            if (key_position < first + row || key_position > begin + row) continue;
            float dot = 0.0f;
            for (uint j = 0; j < 4; ++j) dot = fma(queries[row][j], key[j], dot);
            const uint distance = begin + row - key_position;
            const float bias = float(input.relative[row_start + row][head * Capacity + distance]);
            const float score = simd_sum(dot) + (bias - sink[row]);
            const float next = max(peak[row], score);
            const float prior = fast::exp(peak[row] - next);
            const float probability = fast::exp(score - next);
            numerators[row] = fma(numerators[row], float4(prior), probability * value);
            mass[row] = fma(mass[row], prior, probability);
            peak[row] = next;
        }
    }
#pragma clang loop unroll(full)
    for (uint row = 0; row < Rows; ++row) {
        device float *part = parts + (((row_start + row) * Heads + head) * Shards + shard[row]) * Part;
        for (uint j = 0; j < 4; ++j) part[component + j] = numerators[row][j];
        if (lane == 0) {
            part[128] = peak[row];
            part[129] = mass[row];
        }
    }
}

kernel void ink_verify_local_parts(
    device const bfloat *q0 [[buffer(0)]], device const bfloat *q1 [[buffer(1)]],
    device const bfloat *q2 [[buffer(2)]], device const bfloat *q3 [[buffer(3)]],
    device const bfloat *r0 [[buffer(4)]], device const bfloat *r1 [[buffer(5)]],
    device const bfloat *r2 [[buffer(6)]], device const bfloat *r3 [[buffer(7)]],
    device const bfloat *s0 [[buffer(8)]], device const bfloat *s1 [[buffer(9)]],
    device const bfloat *s2 [[buffer(10)]], device const bfloat *s3 [[buffer(11)]],
    device const bfloat *keys [[buffer(12)]], device const bfloat *values [[buffer(13)]],
    device const bfloat *overwritten [[buffer(14)]], device float *parts [[buffer(15)]],
    constant uint4 &shape [[buffer(16)]], uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    const InkLocalQueries input = {{q0, q1, q2, q3}, {r0, r1, r2, r3}, {s0, s1, s2, s3}};
    const uint shard = group.x * 4 + sg;
    const uint head = group.y % 24, row_start = (group.y / 24) * shape.z;
    const uint live = min(shape.z, shape.y - row_start);
    switch (live) {
        case 1: ink_verify_shared_local<1>(input, keys, values, overwritten, parts, shape.x, shape.y, row_start, head, shard, lane); break;
        case 2: ink_verify_shared_local<2>(input, keys, values, overwritten, parts, shape.x, shape.y, row_start, head, shard, lane); break;
        case 3: ink_verify_shared_local<3>(input, keys, values, overwritten, parts, shape.x, shape.y, row_start, head, shard, lane); break;
        default: ink_verify_shared_local<4>(input, keys, values, overwritten, parts, shape.x, shape.y, row_start, head, shard, lane); break;
    }
}
