// Original Stark generic SIMD-row scheduling experiment. Source-bound packed
// decoding helpers and format attribution remain in packed_matvec.metal.
kernel void hybrid_packed_simd_vector8_generic(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = tile * 8 + simd_group;
    if (row >= p.rows) return;
    device const uchar *codes = codes_base + offsets.x;
    const uint block_shift = hybrid_packed_block == 256 ? 8 : (hybrid_packed_block == 32 ? 5 : 0);
    const ulong row_offset = ulong(row) * (p.columns >> block_shift) * hybrid_packed_block_bytes;
    const ulong input_offset = p.group ? ulong(row / p.group) * p.columns : 0;
    float sum = 0;
    for (uint first = lane * 8; first < p.columns; first += 256) {
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j) {
            const uint col = first + j;
            if (col < p.columns) {
                const ulong block_offset = ulong(col >> block_shift) * hybrid_packed_block_bytes;
                const float w = hybrid_quant(codes + row_offset + block_offset,
                    hybrid_packed_kind, col & (hybrid_packed_block - 1));
                sum += w * input[input_offset + col];
            }
        }
    }
    const float total = simd_sum(sum);
    if (lane == 0) output[row] = total;
}
