// SPDX-License-Identifier: Apache-2.0
// Original Stark deferred decode operators. F32 RMS has a separately qualified
// SIMD reduction; the ordered diagnostic retains the scalar reference order.

kernel void hybrid_graph_finite(device const float *input [[buffer(0)]],
    constant uint *p [[buffer(1)]], device atomic_uint *failure [[buffer(30)]],
    uint i [[thread_position_in_grid]]) {
    if (i < p[0] && !isfinite(input[i])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
}

kernel void hybrid_graph_copy(device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]], constant uint *p [[buffer(2)]],
    device atomic_uint *failure [[buffer(30)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0]) return;
    if (!isfinite(input[i % p[1]])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    output[i] = input[i % p[1]];
}

kernel void hybrid_graph_add(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint *p [[buffer(3)]],
    device atomic_uint *failure [[buffer(30)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0]) return;
    float value = a[i] + b[i];
    if (!isfinite(a[i]) || !isfinite(b[i]) || !isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    out[i] = value;
}

kernel void hybrid_graph_rms_ordered(device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device atomic_uint *failure [[buffer(30)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (group >= p[0]) return;
    uint width = p[1];
    threadgroup float scale;
    // Coalesce and prefetch device reads into registers. The first SIMD group
    // broadcasts values in chronological order; each lane keeps the same F32
    // accumulator, avoiding both partial-sum reassociation and serial LDS loads.
    if (tid < 32) {
        float sum = 0.0f;
        for (uint start = 0; start < width; start += 256) {
            float values[8];
            #pragma clang loop unroll(full)
            for (uint block = 0; block < 8; ++block) {
                uint d = start + block * 32 + tid;
                values[block] = d < width ? input[group * width + d] : 0.0f;
                if (d < width && !isfinite(values[block])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
            }
            #pragma clang loop unroll(full)
            for (uint block = 0; block < 8; ++block) {
                uint at = start + block * 32;
                uint count = at < width ? min(32u, width - at) : 0;
                #pragma clang loop unroll(full)
                for (uint lane = 0; lane < 32; ++lane) {
                    float value = simd_shuffle(values[block], lane);
                    if (lane < count) sum += value * value;
                }
            }
        }
        if (tid == 0) scale = 1.0f / sqrt(sum / float(width) + as_type<float>(p[2]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < width; d += 256) {
        uint index = group * width + d;
        float w = weight[p[3] == width ? d : index];
        float value = input[index] * scale * w;
        if (!isfinite(w) || !isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        output[index] = value;
    }
}

// Standard F32 RMS reduction. Its SIMD/tree order differs from the scalar
// reference; wide F64 oracles and complete source-model gates qualify it.
kernel void hybrid_graph_rms(device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device atomic_uint *failure [[buffer(30)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    if (group >= p[0]) return;
    uint width = p[1];
    float sum = 0.0f;
    for (uint d = tid; d < width; d += 256) {
        float value = input[group * width + d];
        if (!isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        sum += value * value;
    }
    threadgroup float partial[8];
    threadgroup float scale;
    float total = simd_sum(sum);
    if (lane == 0) partial[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        total = simd_sum(lane < 8 ? partial[lane] : 0.0f);
        if (lane == 0) scale = 1.0f / sqrt(total / float(width) + as_type<float>(p[2]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < width; d += 256) {
        uint index = group * width + d;
        float w = weight[p[3] == width ? d : index];
        float value = input[index] * scale * w;
        if (!isfinite(w) || !isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        output[index] = value;
    }
}

kernel void hybrid_graph_swiglu(device const float *gate [[buffer(0)]], device const float *up [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint *p [[buffer(3)]],
    device atomic_uint *failure [[buffer(30)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0]) return;
    float value = hybrid_ffn_activation(gate[i], up[i], as_type<float>(p[1]));
    if (!isfinite(gate[i]) || !isfinite(up[i]) || !isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    out[i] = value;
}

kernel void hybrid_graph_weighted_sum(device const float *experts [[buffer(0)]], device const float *prob [[buffer(1)]],
    device const float *shared [[buffer(2)]], device const float *shared_gate [[buffer(3)]],
    device float *out [[buffer(4)]], constant uint *p [[buffer(5)]],
    device atomic_uint *failure [[buffer(30)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0]) return;
    float value = 0.0f;
    for (uint e = 0; e < p[1]; ++e) {
        float x = experts[e * p[0] + i];
        if (!isfinite(x) || !isfinite(prob[e])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        value += prob[e] * x;
    }
    value += shared_gate[0] * shared[i];
    if (!isfinite(shared[i]) || !isfinite(shared_gate[0]) || !isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    out[i] = value;
}



// Whole-bank affine gather preserves the existing vector8 GEMV lane schedule.
// Expert indices are validated on device before forming any weight address.
kernel void hybrid_graph_affine_indexed(device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uchar *scales [[buffer(4)]], device const uchar *biases [[buffer(5)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row = grid.x * 8 + sg;
    uint batch = grid.y;
    if (row >= p[0]) return;
    uint expert = indices[batch];
    if (expert >= p[2]) { if (!lane) { output[batch * p[0] + row] = 0.0f; atomic_store_explicit(failure, 1u, memory_order_relaxed); } return; }
    uint item = hybrid_affine_scale_kind == 0 ? 4 : 2;
    uint shift = hybrid_affine_bits == 4 ? 1 : 0;
    uint gs = hybrid_affine_group == 128 ? 7 : (hybrid_affine_group == 64 ? 6 : 5);
    ulong element = (ulong(expert) * p[0] + row) * p[1];
    ulong code_offset = element >> shift;
    ulong plane_offset = element >> gs;
    ulong input_offset = p[3] ? ulong(batch) * p[1] : 0;
    float sum = 0.0f;
    for (uint col = lane * 8; col < p[1]; col += 256) {
        ulong group = plane_offset + (col >> gs);
        float scale = hybrid_float(scales + group * item, hybrid_affine_scale_kind);
        float bias = hybrid_float(biases + group * item, hybrid_affine_scale_kind);
        device const uchar *packed = codes + code_offset + (col >> shift);
        uint low = hybrid_u32(packed);
        uint high = hybrid_affine_bits == 8 ? hybrid_u32(packed + 4) : 0;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j) {
            uint word = hybrid_affine_bits == 4 || j < 4 ? low : high;
            uint bit = hybrid_affine_bits == 4 ? j * 4 : (j & 3) * 8;
            uint q = (word >> bit) & ((1u << hybrid_affine_bits) - 1);
            sum += (scale * float(q) + bias) * input[input_offset + col + j];
        }
    }
    float total = simd_sum(sum);
    if (!lane) { if (!isfinite(total)) atomic_store_explicit(failure, 1u, memory_order_relaxed); output[batch * p[0] + row] = total; }
}

kernel void hybrid_graph_packed_indexed(device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    uint row = grid.x, batch = grid.y;
    uint expert = indices[batch];
    if (expert >= p[2]) { if (!tid) { output[batch * p[0] + row] = 0.0f; atomic_store_explicit(failure, 1u, memory_order_relaxed); } return; }
    ulong at = (ulong(expert) * p[0] + row) * (p[1] / hybrid_packed_block) * hybrid_packed_block_bytes;
    float sum = 0.0f;
    for (uint col = tid; col < p[1]; col += 256) {
        device const uchar *block = codes + at + ulong(col / hybrid_packed_block) * hybrid_packed_block_bytes;
        sum += hybrid_quant(block, hybrid_packed_kind, col % hybrid_packed_block) * input[(p[3] ? batch * p[1] : 0) + col];
    }
    threadgroup float partial[8];
    hybrid_write_sum(sum, output + batch * p[0], row, tid, simd_lane, simd_group, partial);
}
