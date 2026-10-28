// SPDX-License-Identifier: Apache-2.0
// Original Stark F32 split-K attention. A group owns one head and 256 token
// positions. Split maxima/denominators preserve stable softmax across splits;
// normalized split values avoid additional unnormalized-value overflow.

kernel void qwen_graph_qsa_split_attention(device const float *query [[buffer(0)]],
    device const float *keys [[buffer(1)]], device const float *values [[buffer(2)]],
    device const uint *visible [[buffer(3)]], device float *partials [[buffer(4)]],
    constant uint *p [[buffer(5)]], device atomic_uint *failure [[buffer(30)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    uint count = p[0], heads = p[1], dim = p[2], kv_heads = p[3], splits = p[4];
    uint head = group / splits, split = group % splits;
    if (head >= heads) return;
    uint kv_head = head / (heads / kv_heads);
    uint start = split * 256;
    threadgroup float scores[256], probabilities[256], reduction[8];
    threadgroup float maximum, denominator;
    for (uint local = sg; local < 256; local += 8) {
        uint token = start + local;
        float score = -INFINITY;
        if (token < count && visible[token]) {
            float sum = 0.0f;
            bool finite = true;
            uint key_base = token * kv_heads * dim + kv_head * dim;
            for (uint d = lane; d < dim; d += 32) {
                float q = query[head * dim + d], k = keys[key_base + d];
                finite = finite && isfinite(q) && isfinite(k);
                sum += q * k;
            }
            if (!finite) atomic_store_explicit(failure, 1u, memory_order_relaxed);
            score = simd_sum(sum) / sqrt(float(dim));
            if (isnan(score) || score == INFINITY) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        }
        if (!lane) scores[local] = score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_maximum = simd_max(scores[tid]);
    if (!lane) reduction[sg] = local_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        local_maximum = simd_max(lane < 8 ? reduction[lane] : -INFINITY);
        if (!lane) maximum = local_maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float probability = isfinite(maximum) ? exp(scores[tid] - maximum) : 0.0f;
    probabilities[tid] = probability;
    float sum = simd_sum(probability);
    if (!lane) reduction[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        sum = simd_sum(lane < 8 ? reduction[lane] : 0.0f);
        if (!lane) denominator = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    probabilities[tid] = denominator > 0.0f ? probability / denominator : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint output_base = group * (dim + 2);
    if (!tid) {
        partials[output_base] = maximum;
        partials[output_base + 1] = denominator;
        if (!isfinite(denominator)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    }
    for (uint d = tid; d < dim; d += 256) {
        float value = 0.0f;
        for (uint local = 0; local < 256 && start + local < count; ++local) {
            uint token = start + local;
            // Consume even a zero-probability visible value: NaN/Inf must not
            // be hidden by exponential underflow or an empty-score split.
            if (visible[token]) value += probabilities[local] * values[token * kv_heads * dim + kv_head * dim + d];
        }
        partials[output_base + 2 + d] = value;
        if (!isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    }
}

kernel void qwen_graph_qsa_combine_attention(device const float *partials [[buffer(0)]],
    device const float *gates [[buffer(1)]], device float *output [[buffer(2)]],
    constant uint *p [[buffer(3)]], device atomic_uint *failure [[buffer(30)]],
    uint task [[thread_position_in_grid]]) {
    uint heads = p[0], dim = p[1], splits = p[2];
    uint head = task / dim;
    if (head >= heads) return;
    uint d = task % dim;
    float maximum = -INFINITY;
    for (uint split = 0; split < splits; ++split) maximum = max(maximum, partials[(head * splits + split) * (dim + 2)]);
    float denominator = 0.0f;
    for (uint split = 0; split < splits; ++split) {
        uint base = (head * splits + split) * (dim + 2);
        float split_denominator = partials[base + 1];
        if (split_denominator > 0.0f) {
            float scale = exp(partials[base] - maximum);
            denominator += split_denominator * scale;
        }
    }
    if (!isfinite(maximum) || !isfinite(denominator) || denominator <= 0.0f || !isfinite(gates[task])) {
        atomic_store_explicit(failure, 1u, memory_order_relaxed);
        output[task] = 0.0f;
        return;
    }
    float value = 0.0f;
    for (uint split = 0; split < splits; ++split) {
        uint base = (head * splits + split) * (dim + 2);
        float split_denominator = partials[base + 1];
        if (split_denominator > 0.0f) {
            float weight = (split_denominator * exp(partials[base] - maximum)) / denominator;
            value += partials[base + 2 + d] * weight;
        }
    }
    value *= qwen_graph_sigmoid_value(gates[task]);
    output[task] = value;
    if (!isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
}
