// SPDX-License-Identifier: Apache-2.0
// Historical Stark F32 Qwen HC graph kernels, compiled only by reference tests.

kernel void hybrid_qwen_hc_activate(device float *reduced [[buffer(0)]],
    constant uint2 &sizes [[buffer(1)]], device atomic_uint *failure [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= sizes.x) return;
    float value = reduced[index] / float(sizes.y);
    if (!isfinite(reduced[index])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    reduced[index] = value * hybrid_ffn_sigmoid(value);
}

kernel void hybrid_qwen_hc_mix(device const float *gates [[buffer(0)]],
    device const float *normalized [[buffer(1)]], device const float *inject [[buffer(2)]],
    device float *output [[buffer(3)]], constant uint4 &sizes [[buffer(4)]],
    device atomic_uint *failure [[buffer(6)]], uint channel [[thread_position_in_grid]]) {
    if (channel < sizes.x) {
        float value = 0.0f;
        for (uint stream = 0; stream < sizes.y; ++stream) {
            uint index = stream * sizes.x + channel;
            if (!isfinite(gates[index])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
            value += (hybrid_ffn_sigmoid(gates[index]) * normalized[index]) / float(sizes.y);
        }
        output[channel] = value;
    }
    if (sizes.z && channel < sizes.y) {
        if (!isfinite(inject[channel])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        output[sizes.x + channel] = 2.0f * hybrid_ffn_sigmoid(inject[channel] / float(sizes.y));
    }
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
