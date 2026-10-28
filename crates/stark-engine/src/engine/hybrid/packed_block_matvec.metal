// Original Stark block/lane GEMV for GGML Q5_K and Q6_K.
// GGML conventions and helper attribution remain in packed_matvec.metal.

kernel void hybrid_packed_block_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = tile * 2 + simd_group;
    if (row >= p.rows) return;
    const uint kind = hybrid_packed_kind;
    device const uchar *codes = codes_base + offsets.x;
    const uint block_shift = kind == 8 ? 5 : 8;
    const ulong row_offset = ulong(row) * (p.columns >> block_shift) * hybrid_packed_block_bytes;
    const ulong input_offset = p.group ? ulong(row / p.group) * p.columns : 0;
    float sum = 0;
    if (kind == 8) {
        for (uint col = lane; col < p.columns; col += 32) {
            device const uchar *block = codes + row_offset + ulong(col >> 5) * 34;
            const float w = hybrid_half(block) * float(as_type<char>(block[2 + lane]));
            sum += w * input[input_offset + col];
        }
    } else {
        for (uint first = 0; first < p.columns; first += 256) {
            device const uchar *block = codes + row_offset + ulong(first >> 8) * hybrid_packed_block_bytes;
            if (kind == 13) {
                const float d = hybrid_half(block);
                const float dmin = hybrid_half(block + 2);
                const uint high = block[16 + lane];
                #pragma clang loop unroll(full)
                for (uint part = 0; part < 4; ++part) {
                    const uint low = block[48 + part * 32 + lane];
                    const uint q0 = (low & 15) | (((high >> (part * 2)) & 1) << 4);
                    const uint q1 = (low >> 4) | (((high >> (part * 2 + 1)) & 1) << 4);
                    const uint2 sm0 = hybrid_scale_min(block + 4, part * 2);
                    const uint2 sm1 = hybrid_scale_min(block + 4, part * 2 + 1);
                    const float w0 = d * float(sm0.x) * float(q0) - dmin * float(sm0.y);
                    const float w1 = d * float(sm1.x) * float(q1) - dmin * float(sm1.y);
                    sum += w0 * input[input_offset + first + part * 64 + lane];
                    sum += w1 * input[input_offset + first + part * 64 + lane + 32];
                }
            } else {
                const float d = hybrid_half(block + 208);
                #pragma clang loop unroll(full)
                for (uint half_block = 0; half_block < 2; ++half_block) {
                    const uint low0 = block[half_block * 64 + lane];
                    const uint low1 = block[half_block * 64 + lane + 32];
                    const uint high = block[128 + half_block * 32 + lane];
                    #pragma clang loop unroll(full)
                    for (uint part = 0; part < 4; ++part) {
                        const uint low = part & 1 ? low1 : low0;
                        const int q = int(((low >> (part >= 2 ? 4 : 0)) & 15)
                            | (((high >> (part * 2)) & 3) << 4)) - 32;
                        const float scale = float(as_type<char>(block[192 + half_block * 8 + lane / 16 + part * 2]));
                        const float w = d * scale * float(q);
                        sum += w * input[input_offset + first + half_block * 128 + part * 32 + lane];
                    }
                }
            }
        }
    }
    const float total = simd_sum(sum);
    if (lane == 0) output[row] = total;
}
