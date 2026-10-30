// SPDX-License-Identifier: Apache-2.0
// Stark native FFN scheduling. Arithmetic follows hybrid/math.rs and the
// existing ordered CPU mixture; packed projections retain their kernels.

inline float hybrid_ffn_sigmoid(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + exp(-value));
    float exponential = exp(value);
    return exponential / (1.0f + exponential);
}

inline float hybrid_ffn_activation(float gate, float up, float limit) {
    if (limit > 0.0f) {
        gate = min(gate, limit);
        up = clamp(up, -limit, limit);
    }
    return (gate * hybrid_ffn_sigmoid(gate)) * up;
}

kernel void hybrid_ffn_activate(device float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]], device float *shared_gate [[buffer(2)]],
    device const float *shared_up [[buffer(3)]], constant uint4 &sizes [[buffer(4)]],
    constant float2 &limits [[buffer(5)]], device atomic_uint *failure [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index < sizes.x) {
        float value = hybrid_ffn_activation(gate[index], up[index], limits.x);
        if (!isfinite(gate[index]) || !isfinite(up[index]) || !isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        gate[index] = value;
    }
    if (index < sizes.y) {
        float value = hybrid_ffn_activation(shared_gate[index], shared_up[index], limits.y);
        if (!isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        shared_gate[index] = value;
    }
}

kernel void hybrid_ffn_reduce(device const float *experts [[buffer(0)]],
    device const float *shared [[buffer(1)]], device const float *probabilities [[buffer(2)]],
    device float *output [[buffer(3)]], constant uint2 &sizes [[buffer(4)]],
    constant float &shared_gate [[buffer(5)]], device atomic_uint *failure [[buffer(6)]],
    uint channel [[thread_position_in_grid]]) {
    if (channel >= sizes.x) return;
    float value = 0.0f;
    for (uint expert = 0; expert < sizes.y; ++expert) {
        if (!isfinite(experts[expert * sizes.x + channel])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        value += probabilities[expert] * experts[expert * sizes.x + channel];
    }
    if (!isfinite(shared[channel])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    output[channel] = value + shared_gate * shared[channel];
}
