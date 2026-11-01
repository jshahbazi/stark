// Original Stark Q6_K component experiment. GGML storage/layout provenance
// remains in packed_matvec.metal/LICENSE.ggml-hybrid. Research motivation:
// llama.cpp b11429, d81235049384534c167caea52b85a694f6103d14,
// ggml/src/ggml-metal/kernels/mul_mv.metal lines 1822-1850 (block-local d
// factoring and independent accumulators). No upstream kernel body is copied.
// The old coalesced lane addressing is retained. This experiment changes F32
// association; stored codes are unchanged and the original row is the oracle.
constant uint hybrid_q6_partial_chains [[function_constant(72)]];

inline float hybrid_q6_coalesced_factored_dot(device const uchar *codes, device const float *input,
    ulong row_offset, ulong input_offset, uint columns, uint lane) {
    float sum = 0.0f;
    bool reference_required = false;
    for (uint first = 0; first < columns; first += 256) {
        device const uchar *block = codes + row_offset + ulong(first >> 8) * hybrid_packed_block_bytes;
        float4 partial = float4(0.0f);
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
                const float x = input[input_offset + first + half_block * 128 + part * 32 + lane];
                reference_required = reference_required || abs(x) > 1e20f;
                const uint chain = (half_block * 4 + part) & (hybrid_q6_partial_chains - 1);
                partial[chain] += scale * float(q) * x;
            }
        }
        const float local = hybrid_q6_partial_chains == 2
            ? partial.x + partial.y : (partial.x + partial.y) + (partial.z + partial.w);
        sum += hybrid_half(block + 208) * local;
    }
    // Finite decoded weights are bounded below 2.7e8. At |input|<=1e20,
    // even 8M absolute products remain below 2.2e35. Larger values run the
    // original multiply/add row so factoring cannot hide an old overflow.
    if (simd_any(reference_required)) {
        return hybrid_packed_block_dot(codes, input, row_offset, input_offset, columns, lane);
    }
    return simd_sum(sum);
}

kernel void hybrid_q6_coalesced_factored_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = tile * 2 + simd_group;
    if (row >= p.rows) return;
    const ulong row_offset = ulong(row) * (p.columns >> 8) * hybrid_packed_block_bytes;
    const ulong input_offset = p.group ? ulong(row / p.group) * p.columns : 0;
    const float total = hybrid_q6_coalesced_factored_dot(codes_base + offsets.x, input, row_offset, input_offset, p.columns, lane);
    if (lane == 0) output[row] = total;
}
