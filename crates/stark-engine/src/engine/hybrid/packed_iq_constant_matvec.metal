// SPDX-License-Identifier: Apache-2.0
// Test-only constant-table variant of Stark's IQ grouped32 schedule. Layout,
// upstream revision and MIT attribution remain in packed_iq_grouped32_matvec.metal
// and LICENSE.ggml-hybrid. Only table storage/addressing changes here.

inline void hybrid_iq_constant_rows(device const uchar *codes, device const float *input,
    device float *output, constant uint *p, device const uint *indices,
    device atomic_uint *failure, uint2 grid, uint lane, uint sg) {
    const uint first_row = grid.x * 8 + sg * 4;
    const uint batch = grid.y;
    const uint expert = indices[batch];
    if (expert >= p[2]) {
        if (lane == 0) {
            for (uint r = 0; r < 4; ++r)
                if (first_row + r < p[0]) output[batch * p[0] + first_row + r] = 0.0f;
            atomic_store_explicit(failure, 1u, memory_order_relaxed);
        }
        return;
    }
    const ulong row_bytes = ulong(p[1] >> 8) * hybrid_packed_block_bytes;
    const ulong input_offset = p[3] ? ulong(batch) * p[1] : 0;
    float sums[4] = {0.0f};
    for (uint col = lane * 32; col < p[1]; col += 1024) {
        float values[32];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 32; ++j) values[j] = input[input_offset + col + j];
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
                float sum1 = 0.0f, sum2 = 0.0f;
                #pragma clang loop unroll(full)
                for (uint l = 0; l < 2; ++l) {
                    const uint q = hybrid_iq_grouped32_u16<true>(block + 2 + part * 8 + l * 2);
                    const uint2 entry = as_type<uint2>(iq2xs_grid[q & 511]);
                    const uint signs = ksigns[q >> 9];
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 8; ++j) {
                        const uchar v = uchar((j < 4 ? entry.x : entry.y) >> ((j & 3) * 8));
                        sum1 = hybrid_iq_grouped32_add<1>(sum1, values[8 * l + j], v, (signs & (1 << j)) != 0);
                    }
                }
                #pragma clang loop unroll(full)
                for (uint l = 2; l < 4; ++l) {
                    const uint q = hybrid_iq_grouped32_u16<true>(block + 2 + part * 8 + l * 2);
                    const uint2 entry = as_type<uint2>(iq2xs_grid[q & 511]);
                    const uint signs = ksigns[q >> 9];
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 8; ++j) {
                        const uchar v = uchar((j < 4 ? entry.x : entry.y) >> ((j & 3) * 8));
                        sum2 = hybrid_iq_grouped32_add<1>(sum2, values[8 * l + j], v, (signs & (1 << j)) != 0);
                    }
                }
                sums[r] += d1 * sum1 + d2 * sum2;
            } else {
                const uint ss = hybrid_iq_grouped32_aux<true>(block + 66 + part * 4);
                const float d = hybrid_iq_grouped32_half<true>(block) * (0.5f + float(ss >> 28));
                float2 sum = float2(0.0f);
                #pragma clang loop unroll(full)
                for (uint l = 0; l < 4; ++l) {
                    const uint at = 2 + part * 8 + 2 * l;
                    const uint entry1 = iq3xxs_grid[block[at]];
                    const uint entry2 = iq3xxs_grid[block[at + 1]];
                    const uint signs = ksigns[(ss >> (7 * l)) & 127];
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 4; ++j) {
                        sum.x = hybrid_iq_grouped32_add<1>(sum.x, values[8 * l + j], uchar(entry1 >> (j * 8)), (signs & (1 << j)) != 0);
                        sum.y = hybrid_iq_grouped32_add<1>(sum.y, values[8 * l + j + 4], uchar(entry2 >> (j * 8)), (signs & (1 << (j + 4))) != 0);
                    }
                }
                sums[r] += d * (sum.x + sum.y);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 4; ++r) {
        const float total = simd_sum(sums[r]) * (hybrid_packed_kind == 17 ? 0.25f : 0.5f);
        if (lane == 0 && first_row + r < p[0]) output[batch * p[0] + first_row + r] = total;
    }
}

kernel void hybrid_graph_iq_constant_iq2xs(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device const uint *bounds [[buffer(7)]],
    device atomic_uint *failure [[buffer(30)]], uint2 grid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    if (bounds[p[3] ? grid.y : 0] != 0) {
        hybrid_iq_multirow_rows(codes, input, output, p, indices, failure, grid, lane, sg);
        return;
    }
    hybrid_iq_constant_rows(codes, input, output, p, indices, failure, grid, lane, sg);
}

kernel void hybrid_graph_iq_constant_iq3xxs(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device const uint *bounds [[buffer(7)]],
    device atomic_uint *failure [[buffer(30)]], uint2 grid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    if (bounds[p[3] ? grid.y : 0] != 0) {
        hybrid_iq_multirow_rows(codes, input, output, p, indices, failure, grid, lane, sg);
        return;
    }
    hybrid_iq_constant_rows(codes, input, output, p, indices, failure, grid, lane, sg);
}
