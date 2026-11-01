// SPDX-License-Identifier: Apache-2.0
// Original Stark mHC schedules. The split projection changes F32 association;
// the single-SIMD Sinkhorn retains chronological row and column summation.

inline void glm_control_project_parts(
    device const float *input, device const float *weights, device float *output,
    constant GlmControlParameters &p, device atomic_uint *failure, uint tile, uint lane,
    uint chunks) {
    const uint row = tile / chunks;
    const uint part = tile % chunks;
    const uint width = (p.columns + chunks - 1) / chunks;
    const uint end = min((part + 1) * width, p.columns);
    float sum = 0.0f;
    for (uint col = part * width + lane; col < end; col += 32)
        sum += weights[row * p.columns + col] * input[col];
    const float total = simd_sum(sum);
    if (lane == 0) {
        output[tile] = total;
        if (!isfinite(total)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
    }
}

#define GLM_CONTROL_PROJECT_PARTS(NAME, CHUNKS) \
kernel void NAME( \
    device const float *input [[buffer(0)]], device const float *weights [[buffer(1)]], \
    device float *output [[buffer(2)]], constant GlmControlParameters &p [[buffer(3)]], \
    device atomic_uint *failure [[buffer(4)]], uint tile [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]]) { \
    glm_control_project_parts(input, weights, output, p, failure, tile, lane, CHUNKS); \
}

GLM_CONTROL_PROJECT_PARTS(glm_control_project_split, 8)
GLM_CONTROL_PROJECT_PARTS(glm_control_project_split16, 16)
GLM_CONTROL_PROJECT_PARTS(glm_control_project_split32, 32)

// Actual GLM53 geometry only: [4096,4,iterations,16384,epsilon,rms_epsilon].
// The first SIMD sum is the existing 16-part parallel RMS apply reduction;
// adding its seven zero SIMD partials leaves the same nonnegative F32 bits.
// Keep each normalized input as a separate F32 value before the original
// split32 projection product and chronological lane accumulation.
kernel void glm_control_rms_project_split32(
    device const float *input [[buffer(0)]], device const float *weights [[buffer(1)]],
    device const float *partials [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint *p [[buffer(4)]], device atomic_uint *failure [[buffer(5)]],
    uint tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
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

inline float glm_split_control_value(device const float *projection, uint row, uint chunks,
                                     device atomic_uint *failure) {
    float value;
    if (chunks > 1) {
        value = 0.0f;
        for (uint part = 0; part < chunks; ++part) value += projection[row * chunks + part];
    } else value = projection[row];
    if (!isfinite(value)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
    return value;
}

inline void glm_control_mix_one_simd(
    device const float *projection, device const float *weights, device float *mix,
    constant GlmControlParameters &p, device atomic_uint *failure, uint lane, uint chunks) {
    const uint count = p.streams;
    const uint cells = count * count;
    device const float *scale = weights + (2 + count) * count;
    if (lane < count) {
        float pre = glm_sigmoid(glm_split_control_value(projection, lane, chunks, failure)
                                * scale[0] + weights[lane]) + p.epsilon;
        float post = 2.0f * glm_sigmoid(glm_split_control_value(projection, count + lane, chunks, failure)
                                      * scale[1] + weights[count + lane]);
        mix[lane] = pre;
        mix[count + lane] = post;
        if (!isfinite(pre) || !isfinite(post)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
    }
    const uint row = lane < cells ? lane / count : 0;
    const uint col = lane < cells ? lane % count : 0;
    const uint index = 2 * count + row * count + col;
    float value = glm_split_control_value(projection, index, chunks, failure) * scale[2] + weights[index];
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

kernel void glm_control_mix_simd(
    device const float *projection [[buffer(0)]], device const float *weights [[buffer(1)]],
    device float *mix [[buffer(2)]], constant GlmControlParameters &p [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]], uint lane [[thread_index_in_simdgroup]]) {
    glm_control_mix_one_simd(projection, weights, mix, p, failure, lane, 1);
}

kernel void glm_control_mix_split_simd(
    device const float *projection [[buffer(0)]], device const float *weights [[buffer(1)]],
    device float *mix [[buffer(2)]], constant GlmControlParameters &p [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]], uint lane [[thread_index_in_simdgroup]]) {
    glm_control_mix_one_simd(projection, weights, mix, p, failure, lane, 8);
}

#define GLM_CONTROL_MIX_PARTS(NAME, CHUNKS) \
kernel void NAME( \
    device const float *projection [[buffer(0)]], device const float *weights [[buffer(1)]], \
    device float *mix [[buffer(2)]], constant GlmControlParameters &p [[buffer(3)]], \
    device atomic_uint *failure [[buffer(4)]], uint lane [[thread_index_in_simdgroup]]) { \
    glm_control_mix_one_simd(projection, weights, mix, p, failure, lane, CHUNKS); \
}

GLM_CONTROL_MIX_PARTS(glm_control_mix_split16_simd, 16)
GLM_CONTROL_MIX_PARTS(glm_control_mix_split32_simd, 32)
