// Original Stark group-factored GGML GEMV. GGML layout provenance remains in
// packed_matvec.metal and LICENSE.ggml-hybrid. Factoring is separately
// qualified: it preserves stored codes but changes F32 multiplication/sum
// association. Schedule research: mlx-kquant (MIT), revision
// 91b642ea0f779b0bbd559d9fbd3b9f403d5f195b, kq_quantized_kquants.h.
// No external implementation is included here.

inline float hybrid_packed_factored_dot(device const uchar *codes, device const float *input,
    ulong row_offset, ulong input_offset, uint columns, uint lane) {
    const uint kind = hybrid_packed_kind;
    const uint within = lane * 8;
    float sum = 0.0f;
    bool reference_required = false;
    for (uint first = 0; first < columns; first += 256) {
        device const uchar *block = codes + row_offset + ulong(first >> 8) * hybrid_packed_block_bytes;
        float code_dot = 0.0f;
        float input_sum = 0.0f;
        if (kind == 12 || kind == 13) {
            const uint group = within >> 5;
            const uint l = within & 31;
            const uint part = group >> 1;
            const uint hi = group & 1;
            const uint2 sm = hybrid_scale_min(block + 4, group);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) {
                const uint packed = block[(kind == 13 ? 48 : 16) + part * 32 + l + j];
                uint q = (packed >> (hi * 4)) & 15;
                if (kind == 13) q |= ((uint(block[16 + l + j]) >> group) & 1) << 4;
                const float x = input[input_offset + first + within + j];
                reference_required = reference_required || abs(x) > 1e20f;
                code_dot += float(q) * x;
                input_sum += x;
            }
            sum += hybrid_half(block) * float(sm.x) * code_dot
                - hybrid_half(block + 2) * float(sm.y) * input_sum;
        } else {
            const uint half_block = within >> 7;
            const uint part = (within & 127) >> 5;
            const uint l = within & 31;
            const float scale = float(as_type<char>(block[192 + half_block * 8 + l / 16 + part * 2]));
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) {
                const uint low = block[half_block * 64 + (part & 1) * 32 + l + j];
                const uint high = block[128 + half_block * 32 + l + j];
                const int q = int(((low >> (part >= 2 ? 4 : 0)) & 15)
                    | (((high >> (part * 2)) & 3) << 4)) - 32;
                const float x = input[input_offset + first + within + j];
                reference_required = reference_required || abs(x) > 1e20f;
                code_dot += float(q) * x;
            }
            sum += hybrid_half(block + 208) * scale * code_dot;
        }
    }
    // At bounded inputs, the largest finite GGML weight is below 2.7e8.
    // Even 8M products of magnitude 1e20 have an absolute sum below 2.2e35.
    // For larger operands, evaluate the original row before any result is
    // stored: algebraic factoring must not conceal an overflowing product.
    if (simd_any(reference_required)) {
        return hybrid_packed_block_dot(codes, input, row_offset, input_offset, columns, lane);
    }
    return simd_sum(sum);
}

kernel void hybrid_packed_factored_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = tile * 2 + simd_group;
    if (row >= p.rows) return;
    const ulong row_offset = ulong(row) * (p.columns >> 8) * hybrid_packed_block_bytes;
    const ulong input_offset = p.group ? ulong(row / p.group) * p.columns : 0;
    const float total = hybrid_packed_factored_dot(codes_base + offsets.x, input, row_offset, input_offset, p.columns, lane);
    if (lane == 0) output[row] = total;
}

kernel void hybrid_graph_packed_indexed_factored(
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
    const float total = hybrid_packed_factored_dot(codes, input, at, input_offset, p[1], lane);
    if (lane == 0) output[batch * p[0] + row] = total;
}
