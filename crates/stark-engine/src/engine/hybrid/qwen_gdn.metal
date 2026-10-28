// SPDX-License-Identifier: Apache-2.0
// Original Stark Qwen GDN F32 recurrence. Each lane owns one value row;
// dot products and normalization sums retain the scalar reference order.

struct QwenGdnParameters {
    uint key_heads, key_dim, value_heads, value_dim, conv_width, eps_bits, sigmoid_output, width;
};

// Metal has no log1p intrinsic. Compensate the rounded 1+x before taking log;
// x is exp(-abs(argument)), so this helper's admitted range is [0,1].
inline float hybrid_qwen_log1p(float x) {
    float sum = 1.0f + x;
    if (sum == 1.0f) return x;
    return log(sum) * (x / (sum - 1.0f));
}

kernel void hybrid_qwen_gdn_convolve(device const float *input [[buffer(0)]],
    device const float *controls [[buffer(1)]], device float *history [[buffer(2)]],
    device float *output [[buffer(3)]], constant QwenGdnParameters &p [[buffer(4)]],
    device atomic_uint *failure [[buffer(6)]], uint channel [[thread_position_in_grid]]) {
    if (channel >= p.width) return;
    float value = input[channel] * controls[channel * p.conv_width + p.conv_width - 1];
    for (uint tap = 0; tap < p.conv_width - 1; ++tap) {
        value += history[channel * (p.conv_width - 1) + tap] * controls[channel * p.conv_width + tap];
    }
    for (uint tap = 0; tap < p.conv_width - 2; ++tap) {
        history[channel * (p.conv_width - 1) + tap] = history[channel * (p.conv_width - 1) + tap + 1];
    }
    history[channel * (p.conv_width - 1) + p.conv_width - 2] = input[channel];
    float activated = value * hybrid_ffn_sigmoid(value);
    if (!isfinite(input[channel]) || !isfinite(activated)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    output[channel] = activated;
}

kernel void hybrid_qwen_gdn_recurrent(device const float *mixed [[buffer(0)]],
    device const float *z [[buffer(1)]], device const float *a [[buffer(2)]],
    device const float *b [[buffer(3)]], device const float *controls [[buffer(4)]],
    device float *state [[buffer(5)]], device atomic_uint *failure [[buffer(6)]],
    device float *output [[buffer(7)]], constant QwenGdnParameters &p [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float q[1024];
    threadgroup float k[1024];
    threadgroup float values[1024];
    threadgroup float scales[5];
    uint key_width = p.key_heads * p.key_dim;
    uint key_head = head / (p.value_heads / p.key_heads);
    if (lane < p.key_dim) {
        q[lane] = mixed[key_head * p.key_dim + lane];
        k[lane] = mixed[key_width + key_head * p.key_dim + lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (uint i = 0; i < p.key_dim; ++i) {
            q_sum += q[i] * q[i];
            k_sum += k[i] * k[i];
        }
        scales[0] = (1.0f / sqrt(float(p.key_dim))) / sqrt(q_sum + 1e-6f);
        scales[1] = 1.0f / sqrt(k_sum + 1e-6f);
        uint control_offset = p.width * p.conv_width;
        float argument = a[head] + controls[control_offset + p.value_heads + head];
        float softplus = max(argument, 0.0f) + hybrid_qwen_log1p(exp(-abs(argument)));
        scales[2] = exp(-exp(controls[control_offset + head]) * softplus);
        scales[3] = hybrid_ffn_sigmoid(b[head]);
        if (!isfinite(a[head]) || !isfinite(b[head]) || !isfinite(scales[2])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < p.key_dim) {
        q[lane] *= scales[0];
        k[lane] *= scales[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < p.value_dim) {
        uint index = head * p.value_dim + lane;
        // Value channels are adjacent in physical storage. Neighboring lanes
        // access neighboring floats at each key coordinate while every row's
        // scalar accumulation order stays unchanged.
        device float *row = state + head * p.value_dim * p.key_dim + lane;
        for (uint i = 0; i < p.key_dim; ++i) row[i * p.value_dim] *= scales[2];
        float prediction = 0.0f;
        for (uint i = 0; i < p.key_dim; ++i) prediction += row[i * p.value_dim] * k[i];
        float delta = (mixed[2 * key_width + index] - prediction) * scales[3];
        for (uint i = 0; i < p.key_dim; ++i) row[i * p.value_dim] += k[i] * delta;
        float value = 0.0f;
        for (uint i = 0; i < p.key_dim; ++i) value += row[i * p.value_dim] * q[i];
        values[lane] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float square_sum = 0.0f;
        for (uint i = 0; i < p.value_dim; ++i) square_sum += values[i] * values[i];
        scales[4] = 1.0f / sqrt(square_sum / float(p.value_dim) + as_type<float>(p.eps_bits));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < p.value_dim) {
        uint index = head * p.value_dim + lane;
        uint norm_offset = p.width * p.conv_width + 2 * p.value_heads;
        float gate = hybrid_ffn_sigmoid(z[index]);
        if (!p.sigmoid_output) gate *= z[index];
        float value = ((values[lane] * scales[4]) * controls[norm_offset + lane]) * gate;
        if (!isfinite(z[index]) || !isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        output[index] = value;
    }
}

// Original Stark blocked recurrence. Retain eight decayed coordinates at a
// time so prediction and output keep their ordered F32 sums while state needs
// two reads and one write instead of four reads and two writes. The independent
// staged kernel above remains available for numerical and performance replay.
kernel void hybrid_qwen_gdn_recurrent_fused(device const float *mixed [[buffer(0)]],
    device const float *z [[buffer(1)]], device const float *a [[buffer(2)]],
    device const float *b [[buffer(3)]], device const float *controls [[buffer(4)]],
    device float *state [[buffer(5)]], device atomic_uint *failure [[buffer(6)]],
    device float *output [[buffer(7)]], constant QwenGdnParameters &p [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float q[1024];
    threadgroup float k[1024];
    threadgroup float values[1024];
    threadgroup float scales[5];
    uint key_width = p.key_heads * p.key_dim;
    uint key_head = head / (p.value_heads / p.key_heads);
    if (lane < p.key_dim) {
        q[lane] = mixed[key_head * p.key_dim + lane];
        k[lane] = mixed[key_width + key_head * p.key_dim + lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (uint i = 0; i < p.key_dim; ++i) {
            q_sum += q[i] * q[i];
            k_sum += k[i] * k[i];
        }
        scales[0] = (1.0f / sqrt(float(p.key_dim))) / sqrt(q_sum + 1e-6f);
        scales[1] = 1.0f / sqrt(k_sum + 1e-6f);
        uint control_offset = p.width * p.conv_width;
        float argument = a[head] + controls[control_offset + p.value_heads + head];
        float softplus = max(argument, 0.0f) + hybrid_qwen_log1p(exp(-abs(argument)));
        scales[2] = exp(-exp(controls[control_offset + head]) * softplus);
        scales[3] = hybrid_ffn_sigmoid(b[head]);
        if (!isfinite(a[head]) || !isfinite(b[head]) || !isfinite(scales[2])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < p.key_dim) {
        q[lane] *= scales[0];
        k[lane] *= scales[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < p.value_dim) {
        uint index = head * p.value_dim + lane;
        // Value channels are adjacent in physical storage. Neighboring lanes
        // access neighboring floats at each key coordinate while every row's
        // scalar accumulation order stays unchanged.
        device float *row = state + head * p.value_dim * p.key_dim + lane;
        float prediction = 0.0f;
        for (uint start = 0; start < p.key_dim; start += 8) {
            float prefetched[8];
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) {
                uint i = start + j;
                prefetched[j] = i < p.key_dim ? row[i * p.value_dim] * scales[2] : 0.0f;
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) {
                uint i = start + j;
                if (i < p.key_dim) prediction += prefetched[j] * k[i];
            }
        }
        float delta = (mixed[2 * key_width + index] - prediction) * scales[3];
        float value = 0.0f;
        for (uint start = 0; start < p.key_dim; start += 8) {
            float prefetched[8];
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) {
                uint i = start + j;
                prefetched[j] = i < p.key_dim ? row[i * p.value_dim] * scales[2] : 0.0f;
            }
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) {
                uint i = start + j;
                if (i < p.key_dim) {
                    float updated = prefetched[j] + k[i] * delta;
                    row[i * p.value_dim] = updated;
                    value += updated * q[i];
                }
            }
        }
        values[lane] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float square_sum = 0.0f;
        for (uint i = 0; i < p.value_dim; ++i) square_sum += values[i] * values[i];
        scales[4] = 1.0f / sqrt(square_sum / float(p.value_dim) + as_type<float>(p.eps_bits));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < p.value_dim) {
        uint index = head * p.value_dim + lane;
        uint norm_offset = p.width * p.conv_width + 2 * p.value_heads;
        float gate = hybrid_ffn_sigmoid(z[index]);
        if (!p.sigmoid_output) gate *= z[index];
        float value = ((values[lane] * scales[4]) * controls[norm_offset + lane]) * gate;
        if (!isfinite(z[index]) || !isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        output[index] = value;
    }
}
