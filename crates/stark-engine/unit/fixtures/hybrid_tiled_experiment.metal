// Four-row scheduling inspired by Stark's existing Uzu-derived GEMV tiles.
// This independent schedule preserves the hybrid kernels' per-row arithmetic.
// GGML decoding helpers come from packed_matvec.metal; see LICENSE.ggml-hybrid.

inline void hybrid_tiled_write_sum(
    thread const float *sums,
    device float *output,
    uint first_row,
    uint valid_rows,
    uint lane,
    uint simd_lane,
    uint simd_group,
    threadgroup float *partial
) {
    // Keep exactly the baseline's eight SIMD partials and their final order
    // for every output row. All lanes participate even in a partial tail tile.
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 4; ++r) {
        float total = simd_sum(sums[r]);
        if (simd_lane == 0) partial[r * 8 + simd_group] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 4; ++r) {
            if (r < valid_rows) {
                float result = 0;
                for (uint i = 0; i < 8; ++i) result += partial[r * 8 + i];
                output[first_row + r] = result;
            }
        }
    }
}

kernel void hybrid_packed_matvec_tiled(
    device const uchar *codes_base [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint first_row = tile * 4;
    if (first_row >= p.rows) return;
    const uint valid_rows = min(4u, p.rows - first_row);
    const uint kind = hybrid_packed_kind;
    const uint block = hybrid_packed_block;
    const uint block_bytes = hybrid_packed_block_bytes;
    const uint block_shift = block == 256 ? 8 : (block == 32 ? 5 : 0);
    const ulong row_stride = ulong(p.columns >> block_shift) * block_bytes;
    device const uchar *codes = codes_base + offsets.x;

    ulong row_offsets[4];
    ulong input_offsets[4];
    float sums[4] = {0, 0, 0, 0};
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 4; ++r) {
        const uint row = first_row + r;
        row_offsets[r] = ulong(row) * row_stride;
        // Packed matrices reserve group for the number of rows per batch.
        input_offsets[r] = p.group ? ulong(row / p.group) * p.columns : 0;
    }
    // A tile may straddle batches when their row count is not a multiple of
    // four. Only reuse the activation when every valid row has the same input.
    const bool shared_input = input_offsets[0] == input_offsets[valid_rows - 1];
    for (uint col = lane; col < p.columns; col += 256) {
        const ulong block_offset = ulong(col >> block_shift) * block_bytes;
        const uint in_block = col & (block - 1);
        const float activation = shared_input ? input[input_offsets[0] + col] : 0;
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 4; ++r) {
            if (r < valid_rows) {
                const float x = shared_input ? activation : input[input_offsets[r] + col];
                sums[r] += hybrid_quant(codes + row_offsets[r] + block_offset, kind, in_block) * x;
            }
        }
    }
    threadgroup float partial[32];
    hybrid_tiled_write_sum(sums, output, first_row, valid_rows, lane, simd_lane, simd_group, partial);
}

kernel void hybrid_affine_matvec_tiled(
    device const uchar *codes_base [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant HybridMatrixParams &p [[buffer(3)]],
    device const uchar *scales_base [[buffer(4)]],
    device const uchar *biases_base [[buffer(5)]],
    constant uint4 &offsets [[buffer(6)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint first_row = tile * 4;
    if (first_row >= p.rows) return;
    const uint valid_rows = min(4u, p.rows - first_row);
    const uint bits = hybrid_affine_bits;
    const uint group_size = hybrid_affine_group;
    const uint scale_kind = hybrid_affine_scale_kind;
    const uint item_bytes = scale_kind == 0 ? 4 : 2;
    const uint packed_shift = bits == 4 ? 1 : 0;
    const uint group_shift = group_size == 128 ? 7 : (group_size == 64 ? 6 : 5);
    device const uchar *codes = codes_base + offsets.x;
    device const uchar *scales = scales_base + offsets.y;
    device const uchar *biases = biases_base + offsets.z;

    ulong code_offsets[4];
    ulong plane_offsets[4];
    ulong input_offsets[4];
    float sums[4] = {0, 0, 0, 0};
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 4; ++r) {
        const uint row = first_row + r;
        const ulong row_element = ulong(row) * p.columns;
        code_offsets[r] = row_element >> packed_shift;
        plane_offsets[r] = row_element >> group_shift;
        // Affine matrices reserve block for the number of rows per batch.
        input_offsets[r] = p.block ? ulong(row / p.block) * p.columns : 0;
    }
    const bool shared_input = input_offsets[0] == input_offsets[valid_rows - 1];
    for (uint col = lane; col < p.columns; col += 256) {
        const uint code_shift = bits == 4 ? (col & 1) * 4 : 0;
        const uint code_column = col >> packed_shift;
        const uint group_column = col >> group_shift;
        const float activation = shared_input ? input[input_offsets[0] + col] : 0;
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 4; ++r) {
            if (r < valid_rows) {
                const uint q = (codes[code_offsets[r] + code_column] >> code_shift) & ((1u << bits) - 1);
                const ulong group = plane_offsets[r] + group_column;
                const float value = hybrid_float(scales + group * item_bytes, scale_kind) * float(q)
                    + hybrid_float(biases + group * item_bytes, scale_kind);
                const float x = shared_input ? activation : input[input_offsets[r] + col];
                sums[r] += value * x;
            }
        }
    }
    threadgroup float partial[32];
    hybrid_tiled_write_sum(sums, output, first_row, valid_rows, lane, simd_lane, simd_group, partial);
}
