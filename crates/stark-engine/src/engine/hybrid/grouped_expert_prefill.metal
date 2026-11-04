// SPDX-License-Identifier: Apache-2.0
// Original Stark cross-token IQ expert batching. Existing GGML tables/layouts
// and Stark's selected grouped32/vector8 arithmetic retain LICENSE.ggml-hybrid.
// A decoded weight is consumed by four independent input rows before advancing.

kernel void stark_expert_copy_indices(device const uint *input [[buffer(0)]],
    device uint *output [[buffer(1)]], constant uint *p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i < p[0]) output[i] = input[i];
}

// p = experts, routes, selected. Every original route slot is listed exactly
// once; descriptors retain its slot so the final mixture keeps router order.
kernel void stark_expert_lists(device const uint *indices [[buffer(0)]],
    device uint *counts [[buffer(1)]], device uint *lists [[buffer(2)]],
    constant uint *p [[buffer(3)]], device atomic_uint *failure [[buffer(30)]],
    uint expert [[thread_position_in_grid]]) {
    if (expert >= p[0]) return;
    uint count = 0;
    for (uint route = 0; route < p[1]; ++route) {
        const uint value = indices[route];
        if (expert == 0 && value >= p[0]) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        if (value == expert) lists[expert * p[1] + count++] = route;
    }
    counts[expert] = count;
}

// jobs = actual_count, followed by [expert, route0, route1, route2, route3].
// The route-count capacity is an upper bound even for all-singleton experts.
kernel void stark_expert_jobs(device const uint *counts [[buffer(0)]],
    device const uint *lists [[buffer(1)]], device uint *jobs [[buffer(2)]],
    constant uint *p [[buffer(3)]], device atomic_uint *failure [[buffer(30)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    uint count = 0;
    for (uint expert = 0; expert < p[0]; ++expert) {
        if (counts[expert] > p[1]) { atomic_store_explicit(failure, 1u, memory_order_relaxed); continue; }
        for (uint first = 0; first < counts[expert]; first += 4) {
            if (count >= p[1]) { atomic_store_explicit(failure, 1u, memory_order_relaxed); break; }
            const uint offset = 1 + 5 * count++;
            jobs[offset] = expert;
            for (uint row = 0; row < 4; ++row)
                jobs[offset + 1 + row] = first + row < counts[expert]
                    ? lists[expert * p[1] + first + row] : 0xffffffffu;
        }
    }
    jobs[0] = count;
}

kernel void stark_expert_indirect(device const uint *jobs [[buffer(0)]],
    device uint *args [[buffer(1)]], constant uint *p [[buffer(2)]],
    device atomic_uint *failure [[buffer(30)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    const uint count = jobs[0];
    if (count > p[1]) { atomic_store_explicit(failure, 1u, memory_order_relaxed); args[0] = args[1] = args[2] = 0; return; }
    args[0] = (p[0] + 7) / 8;
    args[1] = count;
    args[2] = 1;
}

// p = rows, columns, experts, input_batched, selected, route_count.
kernel void stark_expert_iq_grouped(device const uchar *codes [[buffer(0)]],
    device const float *input [[buffer(1)]], device float *output [[buffer(2)]],
    device const uint *jobs [[buffer(3)]], device const uint *bounds [[buffer(4)]],
    constant uint *p [[buffer(5)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint job = 1 + 5 * grid.y;
    const uint expert = jobs[job];
    const uint first_row = grid.x * 8 + sg * 4;
    if (expert >= p[2]) { if (lane == 0) atomic_store_explicit(failure, 1u, memory_order_relaxed); return; }
    uint routes[4], sources[4];
    bool valid[4], fallback[4];
    for (uint b = 0; b < 4; ++b) {
        routes[b] = jobs[job + 1 + b];
        valid[b] = routes[b] < p[5];
        sources[b] = p[3] ? routes[b] : routes[b] / p[4];
        fallback[b] = valid[b] && bounds[sources[b]] != 0;
    }
    threadgroup ulong lookup2[512];
    threadgroup uint lookup3[256];
    threadgroup uchar signs_table[128];
    const uint tid = sg * 32 + lane;
    if (hybrid_packed_kind == 17)
        for (uint i = tid; i < 512; i += 64) lookup2[i] = iq2xs_grid[i];
    else
        for (uint i = tid; i < 256; i += 64) lookup3[i] = iq3xxs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs_table[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong row_bytes = ulong(p[1] >> 8) * hybrid_packed_block_bytes;
    float sums[4][4] = {{0.0f}};
    for (uint col = lane * 32; col < p[1]; col += 1024) {
        const uint part = (col & 255) >> 5;
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 4; ++r) {
            if (first_row + r >= p[0]) continue;
            device const uchar *block = codes + (ulong(expert) * p[0] + first_row + r) * row_bytes
                + ulong(col >> 8) * hybrid_packed_block_bytes;
            if (hybrid_packed_kind == 17) {
                const uint scales = block[66 + part];
                const float d1 = hybrid_iq_grouped32_half<true>(block) * (0.5f + float(scales & 15));
                const float d2 = hybrid_iq_grouped32_half<true>(block) * (0.5f + float(scales >> 4));
                float a[4] = {0.0f}, bsum[4] = {0.0f};
                #pragma clang loop unroll(full)
                for (uint l = 0; l < 4; ++l) {
                    const uint q = hybrid_iq_grouped32_u16<true>(block + 2 + part * 8 + l * 2);
                    const ulong entry = lookup2[q & 511];
                    const uint signs = signs_table[q >> 9];
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 8; ++j) {
                        const float magnitude = float((entry >> (8 * j)) & 255ul);
                        const float weight = (signs & (1 << j)) ? -magnitude : magnitude;
                        #pragma clang loop unroll(full)
                        for (uint b = 0; b < 4; ++b) {
                            if (!valid[b] || fallback[b]) continue;
                            // Stream one scalar after decoding the shared
                            // weight. Do not retain four 32-value input rows
                            // across output rows in source-level locals.
                            const float value = input[ulong(sources[b]) * p[1] + col + 8 * l + j];
                            if (l < 2) a[b] += value * weight;
                            else bsum[b] += value * weight;
                        }
                    }
                }
                #pragma clang loop unroll(full)
                for (uint b = 0; b < 4; ++b)
                    if (valid[b] && !fallback[b]) sums[b][r] += d1 * a[b] + d2 * bsum[b];
            } else {
                const uint ss = hybrid_iq_grouped32_aux<true>(block + 66 + part * 4);
                const float d = hybrid_iq_grouped32_half<true>(block) * (0.5f + float(ss >> 28));
                float2 a[4] = {float2(0.0f)};
                #pragma clang loop unroll(full)
                for (uint l = 0; l < 4; ++l) {
                    const uint at = 2 + part * 8 + 2 * l;
                    const uint entry1 = lookup3[block[at]], entry2 = lookup3[block[at + 1]];
                    const uint signs = signs_table[(ss >> (7 * l)) & 127];
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 4; ++j) {
                        const float m1 = float((entry1 >> (8 * j)) & 255u), m2 = float((entry2 >> (8 * j)) & 255u);
                        const float w1 = (signs & (1 << j)) ? -m1 : m1;
                        const float w2 = (signs & (1 << (j + 4))) ? -m2 : m2;
                        #pragma clang loop unroll(full)
                        for (uint b = 0; b < 4; ++b) {
                            if (!valid[b] || fallback[b]) continue;
                            const ulong source = ulong(sources[b]) * p[1] + col + 8 * l + j;
                            a[b].x += input[source] * w1;
                            a[b].y += input[source + 4] * w2;
                        }
                    }
                }
                #pragma clang loop unroll(full)
                for (uint b = 0; b < 4; ++b)
                    if (valid[b] && !fallback[b]) sums[b][r] += d * (a[b].x + a[b].y);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint b = 0; b < 4; ++b) {
        if (!valid[b]) continue;
        if (!fallback[b]) {
            #pragma clang loop unroll(full)
            for (uint r = 0; r < 4; ++r) {
                const float total = simd_sum(sums[b][r]) * (hybrid_packed_kind == 17 ? 0.25f : 0.5f);
                if (lane == 0 && first_row + r < p[0]) output[ulong(routes[b]) * p[0] + first_row + r] = total;
            }
            continue;
        }
        // Only this incoming row takes the source vector8 fallback. Normal
        // rows in the same job retain their selected grouped32 chronology.
        float slow[4] = {0.0f};
        for (uint col = lane * 8; col < p[1]; col += 256) {
            const uint part = (col & 255) >> 5, l = (col & 31) >> 3;
            float values[8];
            for (uint j = 0; j < 8; ++j) values[j] = input[ulong(sources[b]) * p[1] + col + j];
            for (uint r = 0; r < 4; ++r) {
                if (first_row + r >= p[0]) continue;
                device const uchar *block = codes + (ulong(expert) * p[0] + first_row + r) * row_bytes
                    + ulong(col >> 8) * hybrid_packed_block_bytes;
                ulong entry; uint signs; float scale;
                if (hybrid_packed_kind == 17) {
                    const uint q = hybrid_u16(block + 2 + part * 8 + l * 2), scales = block[66 + part];
                    entry = lookup2[q & 511]; signs = signs_table[q >> 9];
                    scale = hybrid_half(block) * (0.5f + float((scales >> (l < 2 ? 0 : 4)) & 15)) * 0.25f;
                } else {
                    const uint ss = hybrid_u32(block + 66 + part * 4), at = 2 + part * 8 + 2 * l;
                    entry = ulong(lookup3[block[at]]) | (ulong(lookup3[block[at + 1]]) << 32);
                    signs = signs_table[(ss >> (7 * l)) & 127];
                    scale = hybrid_half(block) * (0.5f + float(ss >> 28)) * 0.5f;
                }
                for (uint j = 0; j < 8; ++j) {
                    const float weight = scale * float((entry >> (8 * j)) & 255ul) * ((signs & (1 << j)) ? -1.0f : 1.0f);
                    slow[r] += weight * values[j];
                }
            }
        }
        for (uint r = 0; r < 4; ++r) {
            const float total = simd_sum(slow[r]);
            if (lane == 0 && first_row + r < p[0]) output[ulong(routes[b]) * p[0] + first_row + r] = total;
        }
    }
}
