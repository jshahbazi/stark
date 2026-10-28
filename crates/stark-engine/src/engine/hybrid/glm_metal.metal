// SPDX-License-Identifier: Apache-2.0
// Original Stark GLM KDA decode, matching the scalar contract in glm.rs.
// The recurrence retains [head, value, key] F32 state.
// Each lane owns one value row; reductions retain the scalar reference order.
#include <metal_stdlib>
using namespace metal;

static float glm_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + exp(-x));
    float e = exp(x);
    return e / (1.0f + e);
}

struct GlmKdaParameters {
    uint heads;
    uint dim;
    uint conv_width;
    uint inner;
    float rms_eps;
    float lower_bound;
};

kernel void glm_kda_decode(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *state [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GlmKdaParameters &p [[buffer(4)]],
    device atomic_uint *failure [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float q[256];
    threadgroup float k[256];
    threadgroup float v[256];
    threadgroup float decay[256];
    threadgroup float values[256];
    threadgroup float scales[3];
    const uint history_length = p.inner * (p.conv_width - 1);
    const uint coefficient_length = p.inner * p.conv_width;
    const uint channel = head * p.dim + lane;
    if (lane < p.dim) {
        for (uint part = 0; part < 5; ++part) {
            if (!isfinite(input[part * p.inner + channel])) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
        }
        if (!isfinite(input[5 * p.inner + head])) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
        float conv[3];
        for (uint part = 0; part < 3; ++part) {
            device float *history = state + part * history_length + channel * (p.conv_width - 1);
            device const float *coefficients = weight + part * coefficient_length + channel * p.conv_width;
            float sum = 0.0f;
            for (uint i = 0; i < p.conv_width - 1; ++i) sum += history[i] * coefficients[i];
            float x = input[part * p.inner + channel];
            sum += x * coefficients[p.conv_width - 1];
            if (!isfinite(sum)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
            conv[part] = sum * glm_sigmoid(sum);
            for (uint i = 0; i < p.conv_width - 2; ++i) history[i] = history[i + 1];
            history[p.conv_width - 2] = x;
        }
        q[lane] = conv[0];
        k[lane] = conv[1];
        v[lane] = conv[2];
        device const float *dt = weight + 3 * coefficient_length;
        device const float *a = dt + p.inner + p.dim;
        float g = p.lower_bound * glm_sigmoid(-a[head] * (input[3 * p.inner + channel] + dt[channel]));
        decay[lane] = exp(g);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (uint i = 0; i < p.dim; ++i) {
            q_sum += q[i] * q[i];
            k_sum += k[i] * k[i];
        }
        scales[0] = sqrt(q_sum + 1e-6f);
        scales[1] = sqrt(k_sum + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < p.dim) {
        q[lane] /= scales[0];
        k[lane] /= scales[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < p.dim) {
        device float *row = state + 3 * history_length + (head * p.dim + lane) * p.dim;
        for (uint i = 0; i < p.dim; ++i) row[i] *= decay[i];
        float prediction = 0.0f;
        for (uint i = 0; i < p.dim; ++i) prediction += row[i] * k[i];
        float beta = glm_sigmoid(input[5 * p.inner + head]);
        float delta = (v[lane] - prediction) * beta;
        for (uint i = 0; i < p.dim; ++i) row[i] += k[i] * delta;
        float sum = 0.0f;
        for (uint i = 0; i < p.dim; ++i) sum += row[i] * q[i];
        values[lane] = sum * (1.0f / sqrt(float(p.dim)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < p.dim; ++i) sum += values[i] * values[i];
        scales[2] = 1.0f / sqrt(sum / float(p.dim) + p.rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < p.dim) {
        device const float *norm = weight + 3 * coefficient_length + p.inner;
        output[channel] = values[lane] * scales[2] * norm[lane] * glm_sigmoid(input[4 * p.inner + channel]);
        if (!isfinite(output[channel])) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
    }
}
