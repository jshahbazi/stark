// Original Stark IQ2_XS/IQ3_XXS multirow SIMD schedule. GGML layouts/tables
// retain LICENSE.ggml-hybrid provenance. The four-row/two-SIMD geometry is
// motivated by llama.cpp b11429 (d81235049384534c167caea52b85a694f6103d14),
// ggml/src/ggml-metal/ggml-metal-impl.h lines 19-20,73-75,81-83.
// This geometry research pin is separate from the existing GGML decode pin;
// no upstream kernel body is copied here.
// Every row keeps the existing vector8 multiply/add chronology. Only input
// loads are shared across two/four independent row accumulators per lane.
constant uint hybrid_iq_rows_per_simd [[function_constant(71)]];

inline void hybrid_iq_multirow_rows(
    device const uchar *codes, device const float *input,
    device float *output, constant uint *p,
    device const uint *indices, device atomic_uint *failure,
    uint2 grid, uint lane, uint simd_group) {
    const uint first_row = grid.x * (2 * hybrid_iq_rows_per_simd) + simd_group * hybrid_iq_rows_per_simd;
    if (first_row >= p[0]) return;
    const uint batch = grid.y;
    const uint expert = indices[batch];
    if (expert >= p[2]) {
        if (lane == 0) {
            #pragma clang loop unroll(full)
            for (uint r = 0; r < hybrid_iq_rows_per_simd; ++r) {
                if (first_row + r < p[0]) output[batch * p[0] + first_row + r] = 0.0f;
            }
            atomic_store_explicit(failure, 1u, memory_order_relaxed);
        }
        return;
    }
    const ulong row_bytes = ulong(p[1] >> 8) * hybrid_packed_block_bytes;
    const ulong first_offset = (ulong(expert) * p[0] + first_row) * row_bytes;
    const ulong input_offset = p[3] ? ulong(batch) * p[1] : 0;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint col = lane * 8; col < p[1]; col += 256) {
        float values[8];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j) values[j] = input[input_offset + col + j];
        const uint part = (col & 255) >> 5;
        const uint l = (col & 31) >> 3;
        device const uchar *first_block = codes + first_offset + ulong(col >> 8) * hybrid_packed_block_bytes;
        #pragma clang loop unroll(full)
        for (uint r = 0; r < hybrid_iq_rows_per_simd; ++r) {
            if (first_row + r >= p[0]) continue;
            device const uchar *block = first_block + r * row_bytes;
            uint signs;
            ulong grid_values;
            float scale;
            if (hybrid_packed_kind == 17) {
                const uint q = hybrid_u16(block + 2 + part * 8 + l * 2);
                const uint scales = block[66 + part];
                grid_values = iq2xs_grid[q & 511];
                signs = ksigns[q >> 9];
                scale = hybrid_half(block) * (0.5f + float((scales >> (l < 2 ? 0 : 4)) & 15)) * 0.25f;
            } else {
                const uint ss = hybrid_u32(block + 66 + part * 4);
                signs = ksigns[(ss >> (7 * l)) & 127];
                const uint at = 2 + part * 8 + 2 * l;
                grid_values = ulong(iq3xxs_grid[block[at]]) | (ulong(iq3xxs_grid[block[at + 1]]) << 32);
                scale = hybrid_half(block) * (0.5f + float(ss >> 28)) * 0.5f;
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) {
                const float weight = scale * float((grid_values >> (8 * j)) & 255ul) * ((signs & (1 << j)) ? -1.0f : 1.0f);
                sums[r] += weight * values[j];
            }
        }
    }
    #pragma clang loop unroll(full)
    for (uint r = 0; r < hybrid_iq_rows_per_simd; ++r) {
        const float total = simd_sum(sums[r]);
        if (lane == 0 && first_row + r < p[0]) output[batch * p[0] + first_row + r] = total;
    }
}

kernel void hybrid_graph_iq_indexed_multirow(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    hybrid_iq_multirow_rows(codes, input, output, p, indices, failure, grid, lane, simd_group);
}
