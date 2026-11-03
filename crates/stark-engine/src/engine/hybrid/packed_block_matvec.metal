// Original Stark block/lane GEMV for GGML Q4_K, Q5_K and Q6_K.
// GGML conventions and helper attribution remain in packed_matvec.metal.

// Test-only grouped load specialization. Every participating block/row stride
// is even; its caller must also prove an even effective encoded-buffer offset.
template<bool TypedHalf>
inline float hybrid_grouped_half(device const uchar *p) {
    if (TypedHalf) return float(*(device const half *)p);
    return hybrid_half(p);
}

inline float hybrid_packed_block_dot(device const uchar *codes, device const float *input,
    ulong row_offset, ulong input_offset, uint columns, uint lane) {
    const uint kind = hybrid_packed_kind;
    float sum = 0;
    if (kind == 8) {
        for (uint col = lane; col < columns; col += 32) {
            device const uchar *block = codes + row_offset + ulong(col >> 5) * 34;
            const float w = hybrid_half(block) * float(as_type<char>(block[2 + lane]));
            sum += w * input[input_offset + col];
        }
    } else {
        for (uint first = 0; first < columns; first += 256) {
            device const uchar *block = codes + row_offset + ulong(first >> 8) * hybrid_packed_block_bytes;
            if (kind == 12 || kind == 13) {
                const float d = hybrid_half(block);
                const float dmin = hybrid_half(block + 2);
                const uint high = kind == 13 ? block[16 + lane] : 0;
                #pragma clang loop unroll(full)
                for (uint part = 0; part < 4; ++part) {
                    const uint low = block[(kind == 13 ? 48 : 16) + part * 32 + lane];
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
    return simd_sum(sum);
}

kernel void hybrid_packed_block_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = tile * 2 + simd_group;
    if (row >= p.rows) return;
    const uint block_shift = hybrid_packed_kind == 8 ? 5 : 8;
    const ulong row_offset = ulong(row) * (p.columns >> block_shift) * hybrid_packed_block_bytes;
    const ulong input_offset = p.group ? ulong(row / p.group) * p.columns : 0;
    const float total = hybrid_packed_block_dot(codes_base + offsets.x, input, row_offset, input_offset, p.columns, lane);
    if (lane == 0) output[row] = total;
}

// Same per-row arithmetic as the selected-view kernel above. Only the row's
// source address comes from a bounds-checked device routing index.
kernel void hybrid_graph_packed_indexed_block(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = grid.x * 2 + simd_group;
    if (row >= p[0]) return;
    const uint batch = grid.y;
    const uint expert = indices[batch];
    if (expert >= p[2]) {
        if (lane == 0) {
            output[batch * p[0] + row] = 0.0f;
            atomic_store_explicit(failure, 1u, memory_order_relaxed);
        }
        return;
    }
    const ulong at = (ulong(expert) * p[0] + row) * (p[1] >> 8) * hybrid_packed_block_bytes;
    const ulong input_offset = p[3] ? ulong(batch) * p[1] : 0;
    const float total = hybrid_packed_block_dot(codes, input, at, input_offset, p[1], lane);
    if (lane == 0) output[batch * p[0] + row] = total;
}
