// SPDX-License-Identifier: Apache-2.0
// Original Stark deferred decode operators. F32 RMS has a separately qualified
// SIMD reduction; the ordered diagnostic retains the scalar reference order.

kernel void hybrid_graph_finite(device const float *input [[buffer(0)]],
    constant uint *p [[buffer(1)]], device atomic_uint *failure [[buffer(30)]],
    uint i [[thread_position_in_grid]]) {
    if (i < p[0] && !isfinite(input[i])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
}

// Exact-span finite scans. Each group owns one <=1024-value tile; host-side
// range union never fills holes between checked views or scans padding.
kernel void hybrid_graph_finite_tiles(device const float *input [[buffer(0)]],
    constant uint *p [[buffer(1)]], device atomic_uint *failure [[buffer(30)]],
    uint tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint start = p[tile * 2], count = p[tile * 2 + 1];
    for (uint i = tid; i < count; i += 256) {
        if (!isfinite(input[start + i])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    }
}

// Integer IEEE ordering preserves subnormals even when floating comparisons
// flush them. Normalize both zeros to one key; finite negatives reverse their
// bits and nonnegative values flip their sign bit. No logit arithmetic occurs.
inline uint hybrid_greedy_key(uint bits) {
    if ((bits & 0x7fffffffu) == 0) bits = 0;
    return bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
}

inline uint hybrid_greedy_bits(uint key) {
    // An empty/all-poisoned tile already set failure. Its sentinel remains
    // nonfinite when interpreted by any existing exact-span finite checks.
    if (key == 0) return 0xff800000u;
    return key & 0x80000000u ? key ^ 0x80000000u : ~key;
}

inline void hybrid_greedy_reduce(thread uint &best, thread uint &index,
    threadgroup uint *values, threadgroup uint *indices, uint lane, uint sg) {
    const uint maximum = simd_max(best);
    const uint first = simd_min(best == maximum ? index : 0xffffffffu);
    if (lane == 0) {
        values[sg] = maximum;
        indices[sg] = first;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        best = lane < 8 ? values[lane] : 0;
        index = lane < 8 ? indices[lane] : 0xffffffffu;
        const uint final_maximum = simd_max(best);
        index = simd_min(best == final_maximum ? index : 0xffffffffu);
        best = final_maximum;
    }
}

kernel void hybrid_graph_greedy_partials(device const uint *input [[buffer(0)]],
    device uint *output [[buffer(1)]], constant uint *p [[buffer(2)]],
    device atomic_uint *failure [[buffer(30)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    uint best = 0;
    uint index = 0xffffffffu;
    const uint end = min((group + 1) * 1024, p[0]);
    for (uint i = group * 1024 + tid; i < end; i += 256) {
        const uint bits = input[i];
        const uint key = hybrid_greedy_key(bits);
        if ((bits & 0x7f800000u) == 0x7f800000u) {
            atomic_store_explicit(failure, 1u, memory_order_relaxed);
        } else if (key > best || (key == best && i < index)) {
            best = key;
            index = i;
        }
    }
    threadgroup uint values[8];
    threadgroup uint indices[8];
    hybrid_greedy_reduce(best, index, values, indices, lane, sg);
    if (tid == 0) {
        // Keep original finite-value bits, rather than ordering-key bits,
        // in the first plane. Index bits are <=8M (also finite as F32), and
        // this private intermediate is consumed only by the integer reader.
        output[group] = hybrid_greedy_bits(best);
        output[p[1] + group] = index == 0xffffffffu ? 0u : index;
    }
}

kernel void hybrid_graph_greedy_finish(device const uint *input [[buffer(0)]],
    device float *output [[buffer(1)]], constant uint *p [[buffer(2)]],
    device atomic_uint *failure [[buffer(30)]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    uint best = 0;
    uint index = 0xffffffffu;
    for (uint i = tid; i < p[0]; i += 256) {
        const uint bits = input[i];
        const uint key = hybrid_greedy_key(bits);
        const uint candidate = input[p[0] + i];
        if ((bits & 0x7f800000u) == 0x7f800000u) {
            atomic_store_explicit(failure, 1u, memory_order_relaxed);
        } else if (key > best || (key == best && candidate < index)) {
            best = key;
            index = candidate;
        }
    }
    threadgroup uint values[8];
    threadgroup uint indices[8];
    hybrid_greedy_reduce(best, index, values, indices, lane, sg);
    if (tid == 0) output[0] = index == 0xffffffffu ? 0.0f : float(index);
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

// Prefetch independent strided loads, then consume them in the original F32
// lane chronology. The SIMD and threadgroup reduction tree remains unchanged.
kernel void hybrid_graph_rms_prefetch(device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device atomic_uint *failure [[buffer(30)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    if (group >= p[0]) return;
    const uint width = p[1];
    float sum = 0.0f;
    uint d = tid;
    for (; d + 7 * 256 < width; d += 8 * 256) {
        float values[8];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) values[i] = input[group * width + d + i * 256];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) {
            if (!isfinite(values[i])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
            sum += values[i] * values[i];
        }
    }
    for (; d < width; d += 256) {
        const float value = input[group * width + d];
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
    d = tid;
    for (; d + 7 * 256 < width; d += 8 * 256) {
        float values[8], weights[8];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) {
            const uint at = d + i * 256;
            values[i] = input[group * width + at];
            weights[i] = weight[p[3] == width ? at : group * width + at];
        }
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) {
            const float value = values[i] * scale * weights[i];
            if (!isfinite(weights[i]) || !isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
            output[group * width + d + i * 256] = value;
        }
    }
    for (; d < width; d += 256) {
        const uint index = group * width + d;
        const float w = weight[p[3] == width ? d : index];
        const float value = input[index] * scale * w;
        if (!isfinite(w) || !isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        output[index] = value;
    }
}

// Split wide norms across compute units. The F32 partial-sum association is
// separately qualified; input, weight and final-value failure checks remain.
kernel void hybrid_graph_rms_partials(device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]], constant uint *p [[buffer(2)]],
    device atomic_uint *failure [[buffer(30)]], uint tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint group = tile / p[4];
    const uint start = (tile % p[4]) * 1024;
    float sum = 0.0f;
    for (uint d = start + tid; d < min(start + 1024, p[1]); d += 256) {
        const float value = input[group * p[1] + d];
        if (!isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        sum += value * value;
    }
    threadgroup float partial[8];
    const float total = simd_sum(sum);
    if (lane == 0) partial[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float result = simd_sum(lane < 8 ? partial[lane] : 0.0f);
        if (lane == 0) output[tile] = result;
    }
}

kernel void hybrid_graph_rms_apply(device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]], device const float *partial [[buffer(2)]],
    device float *output [[buffer(3)]], constant uint *p [[buffer(4)]],
    device atomic_uint *failure [[buffer(30)]], uint tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint tiles = (p[1] + 255) / 256;
    const uint group = tile / tiles;
    threadgroup float scale;
    threadgroup float sums[8];
    const float sum = tid < p[4] ? partial[group * p[4] + tid] : 0.0f;
    const float total = simd_sum(sum);
    if (lane == 0) sums[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float result = simd_sum(lane < 8 ? sums[lane] : 0.0f);
        if (lane == 0) scale = 1.0f / sqrt(result / float(p[1]) + as_type<float>(p[2]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint d = (tile % tiles) * 256 + tid;
    if (d < p[1]) {
        const uint index = group * p[1] + d;
        const float w = weight[p[3] == p[1] ? d : index];
        const float value = input[index] * scale * w;
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

kernel void hybrid_graph_qwen_inject(device const float *residual [[buffer(0)]], device const float *branch [[buffer(1)]],
    device const float *gates [[buffer(2)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]],
    device atomic_uint *failure [[buffer(30)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0] * p[1]) return;
    float value = residual[i] + gates[i / p[0]] * branch[i % p[0]];
    if (!isfinite(residual[i]) || !isfinite(gates[i / p[0]]) || !isfinite(branch[i % p[0]]) || !isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    out[i] = value;
}

kernel void hybrid_graph_qwen_hc_activate(device float *down [[buffer(0)]], constant uint *p [[buffer(1)]],
    device atomic_uint *failure [[buffer(30)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0]) return;
    float x = down[i] / float(p[1]);
    if (!isfinite(down[i])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    down[i] = x * hybrid_ffn_sigmoid(x);
}

kernel void hybrid_graph_qwen_hc_mix(device const float *gates [[buffer(0)]], device const float *norm [[buffer(1)]],
    device const float *inject [[buffer(2)]], device float *out [[buffer(3)]], constant uint *p [[buffer(4)]],
    device atomic_uint *failure [[buffer(30)]], uint i [[thread_position_in_grid]]) {
    if (i < p[0]) {
        float value = 0.0f;
        for (uint stream = 0; stream < p[1]; ++stream) {
            uint at = stream * p[0] + i;
            if (!isfinite(gates[at]) || !isfinite(norm[at])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
            value += (hybrid_ffn_sigmoid(gates[at]) * norm[at]) / float(p[1]);
        }
        out[i] = value;
    }
    if (p[2] && i < p[1]) {
        if (!isfinite(inject[i])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        out[p[0] + i] = 2.0f * hybrid_ffn_sigmoid(inject[i] / float(p[1]));
    }
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
