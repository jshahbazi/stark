// SPDX-License-Identifier: Apache-2.0
// Original Stark parallel Ripple head preparation kernels. The head's RMS
// sum keeps the scalar reference order; lanes share only its completed scale.
kernel void ripple_graph_attention_prepare_parallel(
    device const float *queries [[buffer(0)]], device const float *keys [[buffer(1)]],
    device const float *values [[buffer(2)]], device const float *q_norm [[buffer(3)]],
    device const float *k_norm [[buffer(4)]], device float *normalized_queries [[buffer(5)]],
    device float *key_cache [[buffer(6)]], device float *value_cache [[buffer(7)]],
    constant uint *p [[buffer(8)]], device atomic_uint *failure [[buffer(30)]],
    uint task [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint heads = p[0], kv_heads = p[1], dim = p[2], position = p[3], slot = p[4];
    if (task >= heads + kv_heads) return;
    bool query = task < heads;
    uint head = query ? task : task - heads;
    uint source = head * dim;
    uint destination = query ? source : slot * kv_heads * dim + source;
    device const float *input = query ? queries : keys;
    device const float *norm = query ? q_norm : k_norm;
    device float *output = query ? normalized_queries : key_cache;
    float scale = 0.0f;
    if (lane == 0) {
        float sum = 0.0f;
        for (uint d = 0; d < dim; ++d) sum += input[source + d] * input[source + d];
        scale = 1.0f / sqrt(sum / float(dim) + as_type<float>(p[6]));
        if (!isfinite(sum) || !isfinite(scale)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    }
    scale = simd_broadcast(scale, 0);
    if (p[5]) {
        uint half_dim = dim / 2;
        for (uint rotary = lane; rotary < half_dim; rotary += 32) {
            uint upper = rotary + half_dim;
            float lower_value = input[source + rotary] * scale * norm[rotary];
            float upper_value = input[source + upper] * scale * norm[upper];
            float phase = float(position) * pow(as_type<float>(p[7]), -2.0f * float(rotary) / float(dim));
            float cosine;
            float sine = sincos(phase, cosine);
            float low = lower_value * cosine - upper_value * sine;
            float high = lower_value * sine + upper_value * cosine;
            output[destination + rotary] = low;
            output[destination + upper] = high;
            if (!isfinite(low) || !isfinite(high)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        }
    } else {
        for (uint d = lane; d < dim; d += 32) {
            float value = input[source + d] * scale * norm[d];
            output[destination + d] = value;
            if (!isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        }
    }
    if (!query) {
        for (uint d = lane; d < dim; d += 32) {
            float value = values[source + d];
            value_cache[destination + d] = value;
            if (!isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        }
    }
}

// Scalar max and denominator loops retain the reference operation order.
// A single SIMD group shares only the independent exponentials/divisions.
kernel void ripple_graph_attention_softmax_parallel(
    device float *scores [[buffer(0)]], constant uint *p [[buffer(1)]],
    device atomic_uint *failure [[buffer(30)]], uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint count = p[0], heads = p[1], capacity = p[4];
    if (head >= heads) return;
    uint base = head * capacity;
    float maximum = -INFINITY;
    if (lane == 0) {
        for (uint token = 0; token < count; ++token) maximum = max(maximum, scores[base + token]);
    }
    maximum = simd_broadcast(maximum, 0);
    for (uint token = lane; token < count; token += 32) scores[base + token] = exp(scores[base + token] - maximum);
    simdgroup_barrier(mem_flags::mem_device);
    float denominator = 0.0f;
    if (lane == 0) {
        for (uint token = 0; token < count; ++token) denominator += scores[base + token];
        if (!isfinite(maximum) || !isfinite(denominator) || denominator <= 0.0f)
            atomic_store_explicit(failure, 1u, memory_order_relaxed);
    }
    denominator = simd_broadcast(denominator, 0);
    for (uint token = lane; token < count; token += 32) scores[base + token] /= denominator;
}
