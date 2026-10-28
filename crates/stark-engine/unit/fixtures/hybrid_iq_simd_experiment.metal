// Original Stark eight-value decoding experiment. GGML format conventions and
// source-bound lookup tables come from packed_matvec.metal/LICENSE.ggml-hybrid.
kernel void hybrid_iq_simd_vector8(
    device const uchar *codes_base [[buffer(0)]],
    device const float *input [[buffer(1)]], device float *output [[buffer(2)]],
    constant HybridMatrixParams &p [[buffer(3)]], constant uint4 &offsets [[buffer(6)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = tile * 8 + simd_group;
    if (row >= p.rows) return;
    const uint kind = hybrid_packed_kind;
    device const uchar *codes = codes_base + offsets.x;
    const ulong row_offset = ulong(row) * (p.columns >> 8) * hybrid_packed_block_bytes;
    const ulong input_offset = p.group ? ulong(row / p.group) * p.columns : 0;
    float sum = 0;
    for (uint col = lane * 8; col < p.columns; col += 256) {
        device const uchar *block = codes + row_offset + ulong(col >> 8) * hybrid_packed_block_bytes;
        const uint part = (col & 255) >> 5;
        const uint l = (col & 31) >> 3;
        uint signs;
        ulong grid;
        float scale;
        if (kind == 17) {
            const uint q = hybrid_u16(block + 2 + part * 8 + l * 2);
            const uint scales = block[66 + part];
            grid = iq2xs_grid[q & 511];
            signs = ksigns[q >> 9];
            scale = hybrid_half(block) * (0.5f + float((scales >> (l < 2 ? 0 : 4)) & 15)) * 0.25f;
        } else {
            const uint ss = hybrid_u32(block + 66 + part * 4);
            signs = ksigns[(ss >> (7 * l)) & 127];
            const uint at = 2 + part * 8 + 2 * l;
            grid = ulong(iq3xxs_grid[block[at]]) | (ulong(iq3xxs_grid[block[at + 1]]) << 32);
            scale = hybrid_half(block) * (0.5f + float(ss >> 28)) * 0.5f;
        }
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j) {
            const float w = scale * float((grid >> (8 * j)) & 255ul) * ((signs & (1 << j)) ? -1.0f : 1.0f);
            sum += w * input[input_offset + col + j];
        }
    }
    const float total = simd_sum(sum);
    if (lane == 0) output[row] = total;
}
