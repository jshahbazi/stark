// SPDX-License-Identifier: Apache-2.0
// Independent native Ripple control kernels. F32 state/activations and stable
// softmax retain the model's fixed embedding value seed, asymmetric FFN clamp,
// split-half rotary sliding layers, and global layers without rotary position.

inline float ripple_graph_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + exp(-x));
    float e = exp(x);
    return e / (1.0f + e);
}

kernel void ripple_graph_value_input(device const float *seed [[buffer(0)]],
    device const float *input [[buffer(1)]], device float *output [[buffer(2)]],
    constant uint *p [[buffer(3)]], device atomic_uint *failure [[buffer(30)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= p[0]) return;
    float value = seed[i] + as_type<float>(p[1]) * (input[i] - seed[i]);
    output[i] = value;
    if (!isfinite(seed[i]) || !isfinite(input[i]) || !isfinite(value))
        atomic_store_explicit(failure, 1u, memory_order_relaxed);
}

kernel void ripple_graph_attention_prepare(device const float *queries [[buffer(0)]],
    device const float *keys [[buffer(1)]], device const float *values [[buffer(2)]],
    device const float *q_norm [[buffer(3)]], device const float *k_norm [[buffer(4)]],
    device float *normalized_queries [[buffer(5)]], device float *key_cache [[buffer(6)]],
    device float *value_cache [[buffer(7)]], constant uint *p [[buffer(8)]],
    device atomic_uint *failure [[buffer(30)]], uint task [[thread_position_in_grid]]) {
    uint heads = p[0], kv_heads = p[1], dim = p[2], position = p[3], slot = p[4];
    if (task >= heads + kv_heads) return;
    bool query = task < heads;
    uint head = query ? task : task - heads;
    uint source = head * dim;
    uint destination = query ? source : slot * kv_heads * dim + source;
    device const float *input = query ? queries : keys;
    device const float *norm = query ? q_norm : k_norm;
    device float *output = query ? normalized_queries : key_cache;
    float sum = 0.0f;
    for (uint d = 0; d < dim; ++d) sum += input[source + d] * input[source + d];
    float scale = 1.0f / sqrt(sum / float(dim) + as_type<float>(p[6]));
    if (!isfinite(sum) || !isfinite(scale)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    for (uint d = 0; d < dim; ++d) {
        float value = input[source + d] * scale * norm[d];
        if (p[5]) {
            uint half_dim = dim / 2;
            uint rotary = d % half_dim;
            uint partner = d < half_dim ? d + half_dim : d - half_dim;
            float companion = input[source + partner] * scale * norm[partner];
            float phase = float(position) * pow(as_type<float>(p[7]), -2.0f * float(rotary) / float(dim));
            float cosine;
            float sine = sincos(phase, cosine);
            value = d < half_dim ? value * cosine - companion * sine : companion * sine + value * cosine;
        }
        output[destination + d] = value;
        if (!isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        if (!query) {
            float value = values[source + d];
            value_cache[destination + d] = value;
            if (!isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        }
    }
}

kernel void ripple_graph_attention_scores(device const float *queries [[buffer(0)]],
    device const float *keys [[buffer(1)]], device float *scores [[buffer(2)]],
    constant uint *p [[buffer(3)]], device atomic_uint *failure [[buffer(30)]],
    uint task [[thread_position_in_grid]]) {
    uint count = p[0], heads = p[1], kv_heads = p[2], dim = p[3], capacity = p[4], start = p[5];
    uint head = task / count, token = task % count;
    if (head >= heads) return;
    uint key_base = ((start + token) % capacity) * kv_heads * dim + head / (heads / kv_heads) * dim;
    float score = 0.0f;
    for (uint d = 0; d < dim; ++d) score += queries[head * dim + d] * keys[key_base + d];
    score *= 1.0f / sqrt(float(dim));
    scores[head * capacity + token] = score;
    if (!isfinite(score)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
}

kernel void ripple_graph_attention_softmax(device float *scores [[buffer(0)]],
    constant uint *p [[buffer(1)]], device atomic_uint *failure [[buffer(30)]],
    uint head [[thread_position_in_grid]]) {
    uint count = p[0], heads = p[1], capacity = p[4];
    if (head >= heads) return;
    uint base = head * capacity;
    float maximum = -INFINITY;
    for (uint token = 0; token < count; ++token) maximum = max(maximum, scores[base + token]);
    float denominator = 0.0f;
    for (uint token = 0; token < count; ++token) {
        float value = exp(scores[base + token] - maximum);
        scores[base + token] = value;
        denominator += value;
    }
    if (!isfinite(maximum) || !isfinite(denominator) || denominator <= 0.0f)
        atomic_store_explicit(failure, 1u, memory_order_relaxed);
    for (uint token = 0; token < count; ++token) scores[base + token] /= denominator;
}

kernel void ripple_graph_attention_output(device const float *probabilities [[buffer(0)]],
    device const float *values [[buffer(1)]], device const float *gates [[buffer(2)]],
    device float *output [[buffer(3)]], constant uint *p [[buffer(4)]],
    device atomic_uint *failure [[buffer(30)]], uint task [[thread_position_in_grid]]) {
    uint count = p[0], heads = p[1], kv_heads = p[2], dim = p[3], capacity = p[4], start = p[5];
    uint head = task / dim, d = task % dim;
    if (head >= heads) return;
    uint kv_base = head / (heads / kv_heads) * dim + d;
    float value = 0.0f;
    for (uint token = 0; token < count; ++token) {
        uint slot = (start + token) % capacity;
        value += probabilities[head * capacity + token] * values[slot * kv_heads * dim + kv_base];
    }
    if (p[6]) value *= ripple_graph_sigmoid(gates[task]);
    output[task] = value;
    if (!isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
}

kernel void ripple_graph_swiglu(device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]], device float *output [[buffer(2)]],
    constant uint *p [[buffer(3)]], device atomic_uint *failure [[buffer(30)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= p[0]) return;
    float g = gate[i], u = up[i], limit = as_type<float>(p[1]);
    if (!isfinite(g) || !isfinite(u)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    if (limit > 0.0f) { g = min(g, limit); u = clamp(u, -limit, limit); }
    float value = (g * ripple_graph_sigmoid(g)) * u;
    output[i] = value;
    if (!isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
}
