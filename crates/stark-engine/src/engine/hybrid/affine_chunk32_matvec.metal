// SPDX-License-Identifier: Apache-2.0
// Graph-only affine schedule experiment. A lane consumes 32 adjacent values,
// sharing one metadata pair across them; lanes advance by 1024 columns.
// Dequantization is unchanged. The lane assignment changes F32 sum order.
template <bool VectorLoads, bool Factored = false>
inline float hybrid_graph_affine_chunk32_sum(device const uchar *codes,
    device const float *input, device const uchar *scales, device const uchar *biases,
    ulong row_element, ulong input_offset, uint columns, uint lane) {
    uint packed_shift = hybrid_affine_bits == 4 ? 1 : 0;
    uint group_shift = hybrid_affine_group == 128 ? 7 : (hybrid_affine_group == 64 ? 6 : 5);
    ulong code_offset = row_element >> packed_shift;
    ulong plane_offset = row_element >> group_shift;
    float sum = 0.0f;
    for (uint col = lane * 32; col < columns; col += 1024) {
        ulong group = plane_offset + (col >> group_shift);
        float scale = hybrid_graph_uint_float(scales, group, hybrid_affine_scale_kind);
        float bias = hybrid_graph_uint_float(biases, group, hybrid_affine_scale_kind);
        device const uint *packed = (device const uint *)(codes + code_offset + (col >> packed_shift));
        uint words[8];
        if (VectorLoads) {
            uint4 first = ((device const uint4 *)packed)[0];
            words[0] = first.x; words[1] = first.y; words[2] = first.z; words[3] = first.w;
            if (hybrid_affine_bits == 8) {
                uint4 second = ((device const uint4 *)packed)[1];
                words[4] = second.x; words[5] = second.y; words[6] = second.z; words[7] = second.w;
            }
        } else {
            #pragma clang loop unroll(full)
            for (uint word = 0; word < (hybrid_affine_bits == 4 ? 4u : 8u); ++word) words[word] = packed[word];
        }
        if (Factored) {
            float qdot = 0.0f;
            float xsum = 0.0f;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 32; ++j) {
                uint word = words[hybrid_affine_bits == 4 ? j >> 3 : j >> 2];
                uint shift = hybrid_affine_bits == 4 ? (j & 7) * 4 : (j & 3) * 8;
                uint q = (word >> shift) & ((1u << hybrid_affine_bits) - 1);
                float x = input[input_offset + col + j];
                qdot += float(q) * x;
                xsum += x;
            }
            float qpart = scale * qdot;
            float bpart = bias * xsum;
            float term = qpart + bpart;
            float next = sum + term;
            // Factoring can overflow on otherwise valid zero/cancelled
            // weights. Recompute such chunks in the original product order.
            // Severe finite cancellation takes the same conservative fallback.
            float cancellation_limit = 8.0f * 1.1920928955078125e-7f * (abs(qpart) + abs(bpart));
            bool cancellation = abs(term) <= cancellation_limit;
            if (isfinite(qdot) && isfinite(xsum) && isfinite(qpart) && isfinite(bpart)
                && isfinite(term) && isfinite(next) && !cancellation) {
                sum = next;
                continue;
            }
        }
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 32; ++j) {
            uint word = words[hybrid_affine_bits == 4 ? j >> 3 : j >> 2];
            uint shift = hybrid_affine_bits == 4 ? (j & 7) * 4 : (j & 3) * 8;
            uint q = (word >> shift) & ((1u << hybrid_affine_bits) - 1);
            float weight = scale * float(q) + bias;
            sum += weight * input[input_offset + col + j];
        }
    }
    return sum;
}

kernel void hybrid_graph_affine_matvec_chunk32(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    device const uchar *scales [[buffer(4)]], device const uchar *biases [[buffer(5)]],
    constant uint4 &offsets [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = tile * 8 + sg;
    if (row >= p.rows) return;
    float sum = hybrid_graph_affine_chunk32_sum<false>(codes + offsets.x, input, scales + offsets.y, biases + offsets.z,
        ulong(row) * p.columns, p.block ? ulong(row / p.block) * p.columns : 0, p.columns, lane);
    float total = simd_sum(sum);
    if (!lane) {
        if (!isfinite(total)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        output[row] = total;
    }
}

kernel void hybrid_graph_affine_indexed_chunk32(device const uchar *codes [[buffer(0)]],
    device const float *input [[buffer(1)]], device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uchar *scales [[buffer(4)]], device const uchar *biases [[buffer(5)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = grid.x * 8 + sg, batch = grid.y;
    if (row >= p[0]) return;
    uint expert = indices[batch];
    if (expert >= p[2]) {
        if (!lane) {
            output[batch * p[0] + row] = 0.0f;
            atomic_store_explicit(failure, 1u, memory_order_relaxed);
        }
        return;
    }
    float sum = hybrid_graph_affine_chunk32_sum<false>(codes, input, scales, biases,
        (ulong(expert) * p[0] + row) * p[1], p[3] ? ulong(batch) * p[1] : 0, p[1], lane);
    float total = simd_sum(sum);
    if (!lane) {
        if (!isfinite(total)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        output[batch * p[0] + row] = total;
    }
}

// These entrypoints require a 16-byte aligned code span. Whole 32-value
// chunks preserve that alignment for every row under both 4- and 8-bit codes.
kernel void hybrid_graph_affine_matvec_chunk32_uint4(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    device const uchar *scales [[buffer(4)]], device const uchar *biases [[buffer(5)]],
    constant uint4 &offsets [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = tile * 8 + sg;
    if (row >= p.rows) return;
    float sum = hybrid_graph_affine_chunk32_sum<true>(codes + offsets.x, input, scales + offsets.y, biases + offsets.z,
        ulong(row) * p.columns, p.block ? ulong(row / p.block) * p.columns : 0, p.columns, lane);
    float total = simd_sum(sum);
    if (!lane) {
        if (!isfinite(total)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        output[row] = total;
    }
}

kernel void hybrid_graph_affine_indexed_chunk32_uint4(device const uchar *codes [[buffer(0)]],
    device const float *input [[buffer(1)]], device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uchar *scales [[buffer(4)]], device const uchar *biases [[buffer(5)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = grid.x * 8 + sg, batch = grid.y;
    if (row >= p[0]) return;
    uint expert = indices[batch];
    if (expert >= p[2]) {
        if (!lane) {
            output[batch * p[0] + row] = 0.0f;
            atomic_store_explicit(failure, 1u, memory_order_relaxed);
        }
        return;
    }
    float sum = hybrid_graph_affine_chunk32_sum<true>(codes, input, scales, biases,
        (ulong(expert) * p[0] + row) * p[1], p[3] ? ulong(batch) * p[1] : 0, p[1], lane);
    float total = simd_sum(sum);
    if (!lane) {
        if (!isfinite(total)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        output[batch * p[0] + row] = total;
    }
}

// Opt-in F32 factored dot experiment; the default entrypoints above retain
// their original dequantized-product order and precision.
kernel void hybrid_graph_affine_matvec_factored(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    device const uchar *scales [[buffer(4)]], device const uchar *biases [[buffer(5)]],
    constant uint4 &offsets [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = tile * 8 + sg;
    if (row >= p.rows) return;
    float sum = hybrid_graph_affine_chunk32_sum<true, true>(codes + offsets.x, input, scales + offsets.y, biases + offsets.z,
        ulong(row) * p.columns, p.block ? ulong(row / p.block) * p.columns : 0, p.columns, lane);
    float total = simd_sum(sum);
    if (!lane) {
        if (!isfinite(total)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        output[row] = total;
    }
}

kernel void hybrid_graph_affine_indexed_factored(device const uchar *codes [[buffer(0)]],
    device const float *input [[buffer(1)]], device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uchar *scales [[buffer(4)]], device const uchar *biases [[buffer(5)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = grid.x * 8 + sg, batch = grid.y;
    if (row >= p[0]) return;
    uint expert = indices[batch];
    if (expert >= p[2]) {
        if (!lane) {
            output[batch * p[0] + row] = 0.0f;
            atomic_store_explicit(failure, 1u, memory_order_relaxed);
        }
        return;
    }
    float sum = hybrid_graph_affine_chunk32_sum<true, true>(codes, input, scales, biases,
        (ulong(expert) * p[0] + row) * p[1], p[3] ? ulong(batch) * p[1] : 0, p[1], lane);
    float total = simd_sum(sum);
    if (!lane) {
        if (!isfinite(total)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        output[batch * p[0] + row] = total;
    }
}
