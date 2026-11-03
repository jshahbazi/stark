// SPDX-License-Identifier: Apache-2.0
// Original Stark optional mHC fusion. Exact stream/Sinkhorn chronology;
// normal projection and RMS reductions use wider parallel trees.
kernel void glm_control_fused_parallel(
    device const float *projection [[buffer(0)]], device const float *weights [[buffer(1)]],
    device const float *residual [[buffer(2)]], device const float *normalization [[buffer(3)]],
    device float *mix [[buffer(4)]], device float *weighted [[buffer(5)]],
    device float *output [[buffer(6)]], constant uint *raw [[buffer(7)]],
    device atomic_uint *failure [[buffer(8)]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    constant GlmControlParameters &p = *((constant GlmControlParameters *)raw);
    threadgroup float reduced[24];
    threadgroup float partial[24];
    threadgroup uint risk[24];
    threadgroup uint fallback;
    threadgroup float inverse;
    float part = projection[sg * 32 + lane];
    bool ordered = simd_any(!isfinite(part) || fabs(part) > 1.0e20f);
    float row_sum = simd_sum(part);
    if (lane == 0) {
        if (ordered) {
            row_sum = 0.0f;
            for (uint i = 0; i < 32; ++i) row_sum += projection[sg * 32 + i];
        }
        reduced[sg] = row_sum;
        if (!isfinite(row_sum)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint count = p.streams;
        const uint cells = count * count;
        device const float *scale = weights + (2 + count) * count;
        if (lane < count) {
            float pre = glm_sigmoid(reduced[lane]
                                    * scale[0] + weights[lane]) + p.epsilon;
            float post = 2.0f * glm_sigmoid(reduced[count + lane]
                                          * scale[1] + weights[count + lane]);
            mix[lane] = pre;
            mix[count + lane] = post;
            if (!isfinite(pre) || !isfinite(post)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
        }
        const uint row = lane < cells ? lane / count : 0;
        const uint col = lane < cells ? lane % count : 0;
        const uint index = 2 * count + row * count + col;
        float value = reduced[index] * scale[2] + weights[index];
        float maximum = -INFINITY;
        for (uint i = 0; i < count; ++i) maximum = max(maximum, simd_shuffle(value, row * count + i));
        value = exp(value - maximum);
        float denominator = 0.0f;
        for (uint i = 0; i < count; ++i) denominator += simd_shuffle(value, row * count + i);
        value = value / denominator + p.epsilon;
        for (uint iteration = 0; iteration < p.iterations; ++iteration) {
            if (iteration > 0) {
                denominator = 0.0f;
                for (uint i = 0; i < count; ++i) denominator += simd_shuffle(value, row * count + i);
                denominator += p.epsilon;
                value /= denominator;
            }
            denominator = 0.0f;
            for (uint i = 0; i < count; ++i) denominator += simd_shuffle(value, i * count + col);
            denominator += p.epsilon;
            value /= denominator;
        }
        if (lane < cells) {
            mix[2 * count + lane] = value;
            if (!isfinite(value)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    bool large = false;
    for (uint d = tid; d < 4096; d += 768) {
        float sum = 0.0f;
        for (uint stream = 0; stream < 4; ++stream)
            sum += mix[stream] * residual[stream * 4096 + d];
        weighted[d] = sum;
        large |= !isfinite(sum) || fabs(sum) > 1.0e15f;
        if (!isfinite(sum)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
    }
    bool group_large = simd_any(large);
    if (lane == 0) risk[sg] = group_large ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint any_large = simd_any(lane < 24 ? risk[lane] != 0u : false) ? 1u : 0u;
        if (lane == 0) fallback = any_large;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The fallback uses exactly the old 256-thread, eight-SIMD F32 tree,
    // including its overflow-to-zero-scale behavior.
    float sum_square = 0.0f;
    if (fallback != 0) {
        if (tid < 256) for (uint d = tid; d < 4096; d += 256) {
            float value = weighted[d]; sum_square += value * value;
        }
    } else {
        for (uint d = tid; d < 4096; d += 768) {
            float value = weighted[d]; sum_square += value * value;
        }
    }
    float total = simd_sum(sum_square);
    if (lane == 0) partial[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        total = simd_sum(lane < (fallback != 0 ? 8u : 24u) ? partial[lane] : 0.0f);
        if (lane == 0) inverse = 1.0f / sqrt(total / 4096.0f + as_type<float>(raw[5]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < 4096; d += 768) {
        float value = weighted[d] * inverse * normalization[d];
        output[d] = value;
        if (!isfinite(normalization[d]) || !isfinite(value))
            atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
    }
}
