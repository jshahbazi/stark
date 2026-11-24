// SPDX-License-Identifier: MIT
// Adapts the local verifier recurrence in verify_attention.metal.

inline void ink_verify_local_three(
    const thread InkLocalQueries &input,
    device const bfloat *keys, device const bfloat *values, device const bfloat *overwritten,
    device float *parts, uint start, uint whole_rows, uint row_start, uint first_head, uint physical_shard, uint lane
) {
    constexpr uint GroupHeads = 3, Rows = 1;
    constexpr uint Capacity = 1024, Heads = 24, Shards = 64, Part = 130;
    const uint begin = start + row_start;
    const uint first = begin + 1 - Capacity;
    const uint component = lane * 4;
    const uint kv_component = (first_head / 3) * 128 + component;
    float4 queries[Rows][GroupHeads], numerators[Rows][GroupHeads];
    float sink[Rows][GroupHeads], peak[Rows][GroupHeads], mass[Rows][GroupHeads];
    uint shard[Rows];
#pragma clang loop unroll(full)
    for (uint row = 0; row < Rows; ++row) {
        // A workgroup follows one absolute-key residue class. Shift that
        // class into each query's original logical shard before reducing.
        shard[row] = (physical_shard - first - row) & (Shards - 1);
#pragma clang loop unroll(full)
        for (uint h = 0; h < GroupHeads; ++h) {
            const uint head = first_head + h;
            queries[row][h] = float4(*reinterpret_cast<device const bfloat4 *>(
                input.q[row_start + row] + head * 128 + component)) * 0.08838834764831845f;
            sink[row][h] = float(input.sink[row_start + row][head]);
            peak[row][h] = shard[row] == 0 ? 0.0f : -INFINITY;
            mass[row][h] = shard[row] == 0 ? 1.0f : 0.0f;
            numerators[row][h] = float4(0.0f);
        }
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
            const uint distance = begin + row - key_position;
#pragma clang loop unroll(full)
            for (uint h = 0; h < GroupHeads; ++h) {
                float dot = 0.0f;
                for (uint j = 0; j < 4; ++j) dot = fma(queries[row][h][j], key[j], dot);
                const float bias = float(input.relative[row_start + row][(first_head + h) * Capacity + distance]);
                const float score = simd_sum(dot) + (bias - sink[row][h]);
                const float next = max(peak[row][h], score);
                const float prior = fast::exp(peak[row][h] - next);
                const float probability = fast::exp(score - next);
                numerators[row][h] = fma(numerators[row][h], float4(prior), probability * value);
                mass[row][h] = fma(mass[row][h], prior, probability);
                peak[row][h] = next;
            }
        }
    }
#pragma clang loop unroll(full)
    for (uint row = 0; row < Rows; ++row) {
#pragma clang loop unroll(full)
        for (uint h = 0; h < GroupHeads; ++h) {
            device float *part = parts + (((row_start + row) * Heads + first_head + h) * Shards + shard[row]) * Part;
            for (uint j = 0; j < 4; ++j) part[component + j] = numerators[row][h][j];
            if (lane == 0) {
                part[128] = peak[row][h];
                part[129] = mass[row][h];
            }
        }
    }
}

kernel void ink_verify_local_three_parts(
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
    const uint row = group.y / 8;
    const uint head = (group.y % 8) * 3;
    ink_verify_local_three(input, keys, values, overwritten, parts, shape.x, shape.y, row, head, shard, lane);
}
