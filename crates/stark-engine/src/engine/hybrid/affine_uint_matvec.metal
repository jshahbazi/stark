// SPDX-License-Identifier: Apache-2.0
// Stark graph-only aligned load variants of the vector8 affine kernels.
// Host admission proves four-byte code and natural scale/bias alignment.
// Decoding, lane ownership and every F32 arithmetic operation are unchanged.

inline float hybrid_graph_uint_float(device const uchar *base, ulong index, uint kind) {
    if (kind == 0) return as_type<float>(((device const uint *)base)[index]);
    ushort bits = ((device const ushort *)base)[index];
    if (kind == 30) return as_type<float>(uint(bits) << 16);
    return float(as_type<half>(bits));
}

kernel void hybrid_graph_affine_matvec_uint(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    device const uchar *scales_base [[buffer(4)]], device const uchar *biases_base [[buffer(5)]],
    constant uint4 &offsets [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint row = tile * 8 + simd_group;
    if (row >= p.rows) return;
    const uint packed_shift = hybrid_affine_bits == 4 ? 1 : 0;
    const uint group_shift = hybrid_affine_group == 128 ? 7 : (hybrid_affine_group == 64 ? 6 : 5);
    const ulong row_element = ulong(row) * p.columns;
    const ulong code_offset = row_element >> packed_shift;
    const ulong plane_offset = row_element >> group_shift;
    const ulong input_offset = p.block ? ulong(row / p.block) * p.columns : 0;
    device const uchar *codes = codes_base + offsets.x;
    device const uchar *scales = scales_base + offsets.y;
    device const uchar *biases = biases_base + offsets.z;
    float sum = 0;
    for (uint col = lane * 8; col < p.columns; col += 256) {
        const ulong group = plane_offset + (col >> group_shift);
        const float scale = hybrid_graph_uint_float(scales, group, hybrid_affine_scale_kind);
        const float bias = hybrid_graph_uint_float(biases, group, hybrid_affine_scale_kind);
        device const uint *packed = (device const uint *)(codes + code_offset + (col >> packed_shift));
        const uint low = packed[0];
        const uint high = hybrid_affine_bits == 8 ? packed[1] : 0;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j) {
            const uint word = hybrid_affine_bits == 4 || j < 4 ? low : high;
            const uint shift = hybrid_affine_bits == 4 ? j * 4 : (j & 3) * 8;
            const uint q = (word >> shift) & ((1u << hybrid_affine_bits) - 1);
            const float w = scale * float(q) + bias;
            sum += w * input[input_offset + col + j];
        }
    }
    const float total = simd_sum(sum);
    if (lane == 0) {
        if (!isfinite(total)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        output[row] = total;
    }
}

kernel void hybrid_graph_affine_indexed_uint(device const uchar *codes [[buffer(0)]],
    device const float *input [[buffer(1)]], device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uchar *scales [[buffer(4)]], device const uchar *biases [[buffer(5)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = grid.x * 8 + sg;
    uint batch = grid.y;
    if (row >= p[0]) return;
    uint expert = indices[batch];
    if (expert >= p[2]) {
        if (!lane) {
            output[batch * p[0] + row] = 0.0f;
            atomic_store_explicit(failure, 1u, memory_order_relaxed);
        }
        return;
    }
    uint shift = hybrid_affine_bits == 4 ? 1 : 0;
    uint gs = hybrid_affine_group == 128 ? 7 : (hybrid_affine_group == 64 ? 6 : 5);
    ulong element = (ulong(expert) * p[0] + row) * p[1];
    ulong code_offset = element >> shift;
    ulong plane_offset = element >> gs;
    ulong input_offset = p[3] ? ulong(batch) * p[1] : 0;
    float sum = 0.0f;
    for (uint col = lane * 8; col < p[1]; col += 256) {
        ulong group = plane_offset + (col >> gs);
        float scale = hybrid_graph_uint_float(scales, group, hybrid_affine_scale_kind);
        float bias = hybrid_graph_uint_float(biases, group, hybrid_affine_scale_kind);
        device const uint *packed = (device const uint *)(codes + code_offset + (col >> shift));
        uint low = packed[0];
        uint high = hybrid_affine_bits == 8 ? packed[1] : 0;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j) {
            uint word = hybrid_affine_bits == 4 || j < 4 ? low : high;
            uint bit = hybrid_affine_bits == 4 ? j * 4 : (j & 3) * 8;
            uint q = (word >> bit) & ((1u << hybrid_affine_bits) - 1);
            sum += (scale * float(q) + bias) * input[input_offset + col + j];
        }
    }
    float total = simd_sum(sum);
    if (!lane) {
        if (!isfinite(total)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        output[batch * p[0] + row] = total;
    }
}
