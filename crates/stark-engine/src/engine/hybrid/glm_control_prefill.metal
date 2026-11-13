// SPDX-License-Identifier: Apache-2.0
// Original Stark mHC token-major batching, adapted from glm_metal.metal,
// glm_control_split.metal and glm_control_fused_parallel.metal in this tree.
// Only token addressing changes; each row retains its selected F32 schedule.


kernel void glm_control_prefill_project(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint *raw [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constant GlmControlParameters &p = *((constant GlmControlParameters *)raw);
    const uint rows = (2 + p.streams) * p.streams;
    const uint token = tile / rows;
    const uint row = tile % rows;
    input += token * p.columns;
    output += token * rows;

    uint end = p.columns / 16 * 16;
    device const float *weight = weights + row * p.columns;
    float accumulator = 0.0f;
    if (lane < 16) {
        uint i = lane;
        for (; i + 7 * 16 < end; i += 8 * 16) {
            // Independent loads can overlap their latency. Consume each of
            // the eight pairs in its original NEON accumulator chronology;
            // no partial sums or products are combined out of order.
            float w[8], x[8];
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) {
                w[j] = weight[i + j * 16];
                x[j] = input[i + j * 16];
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) accumulator += w[j] * x[j];
        }
        for (; i < end; i += 16) accumulator += weight[i] * input[i];
    }
    float x = (simd_shuffle(accumulator, 0) + simd_shuffle(accumulator, 4))
        + (simd_shuffle(accumulator, 8) + simd_shuffle(accumulator, 12));
    float y = (simd_shuffle(accumulator, 1) + simd_shuffle(accumulator, 5))
        + (simd_shuffle(accumulator, 9) + simd_shuffle(accumulator, 13));
    float z = (simd_shuffle(accumulator, 2) + simd_shuffle(accumulator, 6))
        + (simd_shuffle(accumulator, 10) + simd_shuffle(accumulator, 14));
    float w = (simd_shuffle(accumulator, 3) + simd_shuffle(accumulator, 7))
        + (simd_shuffle(accumulator, 11) + simd_shuffle(accumulator, 15));
    if (lane == 0) {
        float sum = (x + y) + (z + w);
        for (uint i = end; i < p.columns; ++i) sum += weight[i] * input[i];
        output[row] = sum;
        if (!isfinite(sum)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
    }
}

kernel void glm_control_prefill_project_parts(
    device const float *input [[buffer(0)]], device const float *weights [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *raw [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constant GlmControlParameters &p = *((constant GlmControlParameters *)raw);
    const uint tiles = (2 + p.streams) * p.streams * raw[6];
    const uint token = tile / tiles;
    glm_control_project_parts(input + token * p.columns, weights,
        output + token * tiles, p, failure, tile % tiles, lane, raw[6]);
}

kernel void glm_control_prefill_rms_project_split32(
    device const float *input [[buffer(0)]], device const float *weights [[buffer(1)]],
    device const float *partials [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint *p [[buffer(4)]], device atomic_uint *failure [[buffer(5)]],
    uint batch_tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint tiles = (2 + p[1]) * p[1] * 32;
    const uint token = batch_tile / tiles;
    const uint tile = batch_tile % tiles;
    input += token * p[3];
    partials += token * 16;
    output += token * tiles;

    const uint row = tile / 32;
    const uint part = tile % 32;
    const float total_square = simd_sum(lane < 16 ? partials[lane] : 0.0f);
    const float scale = 1.0f / sqrt(total_square / float(p[3]) + as_type<float>(p[5]));
    const uint width = p[3] / 32;
    const uint end = (part + 1) * width;
    float sum = 0.0f;
    for (uint col = part * width + lane; col < end; col += 32) {
        const float normalized = input[col] * scale;
        sum += weights[row * p[3] + col] * normalized;
    }
    const float total = simd_sum(sum);
    if (lane == 0) {
        output[tile] = total;
        if (!isfinite(total)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
    }
}

kernel void glm_control_prefill_mix(
    device const float *projection [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *mix [[buffer(2)]],
    constant uint *raw [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint token [[threadgroup_position_in_grid]]) {
    constant GlmControlParameters &p = *((constant GlmControlParameters *)raw);
    const uint rows = (2 + p.streams) * p.streams;
    projection += token * rows;
    mix += token * rows;

    threadgroup float comb[64];
    threadgroup float denominators[8];
    uint count = p.streams;
    uint cells = count * count;
    device const float *base = weights;
    device const float *scale = weights + (2 + count) * count;
    if (tid < count) {
        float pre = glm_sigmoid(projection[tid] * scale[0] + base[tid]) + p.epsilon;
        float post = 2.0f * glm_sigmoid(projection[count+tid] * scale[1] + base[count+tid]);
        mix[tid] = pre;
        mix[count+tid] = post;
        if (!isfinite(pre) || !isfinite(post)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);

        // Rows are independent. Each lane retains the original chronological
        // max, exp and sum order within its row, using registers for logits.
        float row[8];
        float maximum = -INFINITY;
        for (uint col = 0; col < count; ++col) {
            uint i = 2 * count + tid * count + col;
            row[col] = projection[i] * scale[2] + base[i];
            maximum = max(maximum, row[col]);
        }
        float denominator = 0.0f;
        for (uint col = 0; col < count; ++col) {
            row[col] = exp(row[col] - maximum);
            denominator += row[col];
        }
        for (uint col = 0; col < count; ++col) comb[tid * count + col] = row[col] / denominator + p.epsilon;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint iteration = 0; iteration < p.iterations; ++iteration) {
        if (iteration > 0) {
            if (tid < count) {
                float denominator = 0.0f;
                for (uint col = 0; col < count; ++col) denominator += comb[tid * count + col];
                denominators[tid] = denominator + p.epsilon;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < cells) comb[tid] /= denominators[tid / count];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid < count) {
            float denominator = 0.0f;
            for (uint row = 0; row < count; ++row) denominator += comb[row * count + tid];
            denominators[tid] = denominator + p.epsilon;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < cells) comb[tid] /= denominators[tid % count];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < cells) {
        float value = comb[tid];
        mix[2 * count + tid] = value;
        if (!isfinite(value)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
    }
}

kernel void glm_control_prefill_mix_simd(
    device const float *projection [[buffer(0)]], device const float *weights [[buffer(1)]],
    device float *mix [[buffer(2)]], constant uint *raw [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]], uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constant GlmControlParameters &p = *((constant GlmControlParameters *)raw);
    const uint rows = (2 + p.streams) * p.streams;
    glm_control_mix_one_simd(projection + token * rows * raw[6], weights,
        mix + token * rows, p, failure, lane, raw[6]);
}

kernel void glm_control_prefill_input(
    device const float *residual [[buffer(0)]],
    device const float *mix [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint *raw [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    constant GlmControlParameters &p = *((constant GlmControlParameters *)raw);
    if (index >= raw[7] * p.hidden) return;
    const uint token = index / p.hidden;
    const uint d = index % p.hidden;
    residual += token * p.columns;
    mix += token * (2 + p.streams) * p.streams;
    output += token * p.hidden;

    if (d >= p.hidden) return;
    float sum = 0.0f;
    for (uint stream = 0; stream < p.streams; ++stream) sum += mix[stream] * residual[stream * p.hidden + d];
    output[d] = sum;
    if (!isfinite(sum)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
}

kernel void glm_control_prefill_post(
    device const float *residual [[buffer(0)]],
    device const float *output [[buffer(1)]],
    device const float *mix [[buffer(2)]],
    device float *next [[buffer(3)]],
    constant uint *raw [[buffer(4)]],
    device atomic_uint *failure [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    constant GlmControlParameters &p = *((constant GlmControlParameters *)raw);
    if (index >= raw[7] * p.columns) return;
    const uint token = index / p.columns;
    const uint i = index % p.columns;
    residual += token * p.columns;
    output += token * p.hidden;
    mix += token * (2 + p.streams) * p.streams;
    next += token * p.columns;

    if (i >= p.columns) return;
    uint dst = i / p.hidden;
    uint d = i % p.hidden;
    float sum = mix[p.streams + dst] * output[d];
    for (uint src = 0; src < p.streams; ++src) sum += mix[2 * p.streams + src * p.streams + dst] * residual[src * p.hidden + d];
    next[i] = sum;
    if (!isfinite(sum)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
}

kernel void glm_control_prefill_fused_parallel(
    device const float *projection [[buffer(0)]], device const float *weights [[buffer(1)]],
    device const float *residual [[buffer(2)]], device const float *normalization [[buffer(3)]],
    device float *mix [[buffer(4)]], device float *weighted [[buffer(5)]],
    device float *output [[buffer(6)]], constant uint *raw [[buffer(7)]],
    device atomic_uint *failure [[buffer(8)]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint token [[threadgroup_position_in_grid]]) {
    constant GlmControlParameters &p = *((constant GlmControlParameters *)raw);
    const uint rows = (2 + p.streams) * p.streams;
    projection += token * rows * 32;
    residual += token * p.columns;
    mix += token * rows;
    weighted += token * p.hidden;
    output += token * p.hidden;
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
