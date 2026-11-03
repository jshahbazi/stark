// Original Stark SIMD-row scheduling for GGML IQ4_XS. Storage layout and
// codebook retain the provenance recorded in LICENSE.ggml-hybrid.
// Reuse metadata for eight neighbors, preserving each decoded weight's F32
// operation order. One SIMD group owns a row, changing the summation tree.

inline float hybrid_iq4_dot_vector8(device const uchar *codes, device const float *input,
    ulong row_offset, ulong input_offset, uint columns, uint lane) {
    float sum = 0;
    for (uint col = lane * 8; col < columns; col += 256) {
        device const uchar *block = codes + row_offset + ulong(col >> 8) * hybrid_packed_block_bytes;
        const uint within = col & 255;
        const uint part = within >> 5;
        const uint low = (block[4 + (part >> 1)] >> (4 * (part & 1))) & 15;
        const int subscale = int(low | (((hybrid_u16(block + 2) >> (part * 2)) & 3) << 4)) - 32;
        const float scale = hybrid_half(block) * float(subscale);
        device const uchar *at = block + 8 + part * 16 + (within & 15);
        const uint2 packed = uint2(hybrid_u32(at), hybrid_u32(at + 4));
        const uint shift = within & 16 ? 4 : 0;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j) {
            const uint word = j < 4 ? packed.x : packed.y;
            const uint raw = (word >> (8 * (j & 3))) & 255;
            const float weight = scale * float(iq4values[(raw >> shift) & 15]);
            sum += weight * input[input_offset + col + j];
        }
    }
    return simd_sum(sum);
}

kernel void hybrid_iq4_matvec_vector8(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = tile * 8 + simd_group;
    if (row >= p.rows) return;
    const ulong row_offset = ulong(row) * (p.columns >> 8) * hybrid_packed_block_bytes;
    const ulong input_offset = p.group ? ulong(row / p.group) * p.columns : 0;
    const float total = hybrid_iq4_dot_vector8(codes_base + offsets.x, input, row_offset, input_offset, p.columns, lane);
    if (lane == 0) output[row] = total;
}

kernel void hybrid_graph_iq4_indexed_vector8(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = grid.x * 8 + simd_group;
    if (row >= p[0]) return;
    const uint batch = grid.y;
    const uint expert = indices[batch];
    if (expert >= p[2]) {
        if (lane == 0) {
            output[batch * p[0] + row] = 0;
            atomic_store_explicit(failure, 1u, memory_order_relaxed);
        }
        return;
    }
    const ulong at = (ulong(expert) * p[0] + row) * (p[1] >> 8) * hybrid_packed_block_bytes;
    const ulong input_offset = p[3] ? ulong(batch) * p[1] : 0;
    const float total = hybrid_iq4_dot_vector8(codes, input, at, input_offset, p[1], lane);
    if (lane == 0) output[batch * p[0] + row] = total;
}
