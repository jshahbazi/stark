// SPDX-License-Identifier: Apache-2.0
// Original Stark F32 multi-token KDA. Algorithmic reference: mlx-stream
// ac3b772b61bce66450862bca697dbb742c35453f, src/arch/glm5_next/kda_percore.zig
// and kernels/kda_percore.metal (the latter credits oMLX 0.7.0, Copyright oMLX
// contributors, Apache-2.0). No generated upstream kernel text is copied.
// See docs/validation/GLM_KDA_PREFILL.md for precision and reduction policy.
#include <metal_stdlib>
using namespace metal;

struct GlmKdaPrefillParameters {
    uint heads, dim, conv_width, inner;
    float rms_eps, lower_bound;
    // policy bit0: transposed backing; bit1: selected source parallel norm.
    uint tokens, policy, groups, rows;
};

static float glm_kp_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + exp(-x));
    const float e = exp(x);
    return e / (1.0f + e);
}

// Select an incoming history value or an earlier row of this same chunk.
// The original history remains untouched until all convolution outputs exist.
static float glm_kp_past(device const float *input, device const float *history,
                         uint channel, int time, constant GlmKdaPrefillParameters &p) {
    return time < 0 ? history[channel * (p.conv_width - 1) + uint(time + int(p.conv_width - 1))]
                    : input[uint(time) * p.inner + channel];
}

kernel void glm_kda_prefill_prepare(
    device const float *input_q [[buffer(0)]], device const float *input_k [[buffer(1)]],
    device const float *input_v [[buffer(2)]], device const float *forget [[buffer(3)]],
    device const float *beta [[buffer(4)]], device const float *weight [[buffer(5)]],
    device float *state [[buffer(6)]], device float *prepared [[buffer(7)]],
    device atomic_uint *failure [[buffer(8)]],
    constant GlmKdaPrefillParameters &p [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float q[128], k[128], scales[2];
    threadgroup float partial_a[4], partial_b[4];
    threadgroup uint risks[4];
    const uint channel = head * 128 + lane;
    const uint history_length = p.inner * (p.conv_width - 1);
    const uint coefficient_length = p.inner * p.conv_width;
    const uint stride = 4 * p.inner + p.heads;
    device const float *inputs[3] = {input_q, input_k, input_v};
    for (uint token = 0; token < p.tokens; ++token) {
        float conv[3];
        for (uint part = 0; part < 3; ++part) {
            device const float *history = state + part * history_length;
            device const float *coefficients = weight + part * coefficient_length + channel * p.conv_width;
            float sum = 0.0f;
            for (uint i = 0; i < p.conv_width - 1; ++i)
                sum += glm_kp_past(inputs[part], history, channel, int(token + i) - int(p.conv_width - 1), p) * coefficients[i];
            const float x = inputs[part][token * p.inner + channel];
            sum += x * coefficients[p.conv_width - 1];
            conv[part] = sum * glm_kp_sigmoid(sum);
            if (!isfinite(x) || !isfinite(sum) || !isfinite(conv[part]))
                atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
        }
        q[lane] = conv[0];
        k[lane] = conv[1];
        const uint offset = token * stride;
        prepared[offset + 2 * p.inner + channel] = conv[2];
        const float f = forget[token * p.inner + channel];
        const float dt = weight[3 * coefficient_length + channel];
        const float a = weight[3 * coefficient_length + p.inner + p.dim + head];
        const float decay = exp(p.lower_bound * glm_kp_sigmoid(-a * (f + dt)));
        prepared[offset + 3 * p.inner + channel] = decay;
        if (!isfinite(f) || !isfinite(decay)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
        if (lane == 0) {
            const float b = beta[token * p.heads + head];
            prepared[offset + 4 * p.inner + head] = glm_kp_sigmoid(b);
            if (!isfinite(b)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
        }
        if (p.policy & 2u) {
            // The identical existing helper preserves the selected SIMD tree
            // and its joint ordered fallback for extreme q or k values.
            glm_kda_parallel_norm<true,false>(q,k,partial_a,partial_b,risks,scales,lane,p.dim,1e-6f);
        } else {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) {
                float qs = 0.0f, ks = 0.0f;
                for (uint key = 0; key < p.dim; ++key) { qs += q[key] * q[key]; ks += k[key] * k[key]; }
                scales[0] = sqrt(qs + 1e-6f);
                scales[1] = sqrt(ks + 1e-6f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        prepared[offset + channel] = q[lane] / scales[0];
        prepared[offset + p.inner + channel] = k[lane] / scales[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint part = 0; part < 3; ++part) {
        device float *history = state + part * history_length;
        // Forward writes cannot overwrite a still-needed incoming element:
        // when T < history width each source index is T+i, strictly later.
        for (uint i = 0; i < p.conv_width - 1; ++i)
            history[channel * (p.conv_width - 1) + i] = glm_kp_past(inputs[part], history, channel,
                int(p.tokens + i) - int(p.conv_width - 1), p);
    }
}

kernel void glm_kda_prefill_recur(
    device const float *prepared [[buffer(0)]], device float *state [[buffer(1)]],
    device float *values [[buffer(2)]], device atomic_uint *failure [[buffer(3)]],
    constant GlmKdaPrefillParameters &p [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
    // 28 KiB fixed threadgroup storage. Eight threads share a row and each
    // owns sixteen F32 state elements for the entire submitted token span.
    threadgroup float qs[2][8][128], ks[2][8][128], ds[2][8][128];
    threadgroup float vs[8][128], bs[2][8];
    const uint total = p.inner;
    const uint first = 4 * ((group * (total / 4)) / p.groups);
    const uint end = 4 * (((group + 1) * (total / 4)) / p.groups);
    const uint rows = end - first;
    const uint first_head = first / 128;
    const uint heads = (end - 1) / 128 - first_head + 1;
    const uint row = tid / 8, segment = tid % 8;
    const uint subgroup_base = simd_lane / 8 * 8;
    const bool active = row < rows;
    const uint global_row = first + (active ? row : 0);
    const uint head = global_row / 128, value = global_row % 128;
    const uint slot = head - first_head;
    const uint key0 = segment * 16;
    const uint state_base = 3 * p.inner * (p.conv_width - 1) + head * 128 * 128;
    const uint stride = 4 * p.inner + p.heads;
    float4 held[4];
    #pragma unroll
    for (uint vector = 0; vector < 4; ++vector) {
        #pragma unroll
        for (uint component = 0; component < 4; ++component) {
            const uint key = key0 + vector * 4 + component;
            const float s = active ? state[state_base + ((p.policy & 1u) ? key * 128 + value : value * 128 + key)] : 0.0f;
            held[vector][component] = s;
            if (active && !isfinite(s)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
        }
    }
    for (uint tile = 0; tile < p.tokens; tile += 8) {
        const uint count = min(8u, p.tokens - tile);
        for (uint i = tid; i < heads * count * 128; i += p.rows * 8) {
            const uint h = i / (count * 128), t = i / 128 % count, key = i % 128;
            const uint offset = (tile + t) * stride + (first_head + h) * 128 + key;
            qs[h][t][key] = prepared[offset];
            ks[h][t][key] = prepared[offset + p.inner];
            ds[h][t][key] = prepared[offset + 3 * p.inner];
        }
        for (uint i = tid; i < count * rows; i += p.rows * 8) {
            const uint t = i / rows, r = i % rows;
            vs[t][r] = prepared[(tile + t) * stride + 2 * p.inner + first + r];
        }
        for (uint i = tid; i < heads * count; i += p.rows * 8)
            bs[i / count][i % count] = prepared[(tile + i % count) * stride + 4 * p.inner + first_head + i / count];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint token = 0; token < count; ++token) {
            float prediction = 0.0f;
            #pragma unroll
            for (uint owner = 0; owner < 8; ++owner) {
                if (segment == owner) {
                    #pragma unroll
                    for (uint vector = 0; vector < 4; ++vector) {
                        #pragma unroll
                        for (uint component = 0; component < 4; ++component) {
                            const uint key = key0 + vector * 4 + component;
                            const float decayed = held[vector][component] * ds[slot][token][key];
                            held[vector][component] = decayed;
                            prediction += decayed * ks[slot][token][key];
                        }
                    }
                }
                // Pass the running sum, never a sum of segment partials.
                // This retains all 128 chronological F32 additions while
                // each lane's sixteen state elements remain register-local.
                prediction = simd_shuffle(prediction, subgroup_base + owner);
            }
            const float delta = ((active ? vs[token][row] : 0.0f) - prediction) * bs[slot][token];
            float sum = 0.0f;
            #pragma unroll
            for (uint owner = 0; owner < 8; ++owner) {
                if (segment == owner) {
                    #pragma unroll
                    for (uint vector = 0; vector < 4; ++vector) {
                        #pragma unroll
                        for (uint component = 0; component < 4; ++component) {
                            const uint key = key0 + vector * 4 + component;
                            const float updated = held[vector][component] + ks[slot][token][key] * delta;
                            held[vector][component] = updated;
                            sum += updated * qs[slot][token][key];
                        }
                    }
                }
                sum = simd_shuffle(sum, subgroup_base + owner);
            }
            if (active && segment == 0) {
                const float result = sum * (1.0f / sqrt(float(p.dim)));
                values[(tile + token) * p.inner + global_row] = result;
                if (!isfinite(result) || !isfinite(delta)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        #pragma unroll
        for (uint vector = 0; vector < 4; ++vector) {
            #pragma unroll
            for (uint component = 0; component < 4; ++component) {
                const uint key = key0 + vector * 4 + component;
                const float s = held[vector][component];
                state[state_base + ((p.policy & 1u) ? key * 128 + value : value * 128 + key)] = s;
                if (!isfinite(s)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
            }
        }
    }
}

kernel void glm_kda_prefill_finish(
    device const float *values [[buffer(0)]], device const float *gate [[buffer(1)]],
    device const float *weight [[buffer(2)]], device float *output [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]], constant GlmKdaPrefillParameters &p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float scales[1];
    threadgroup float norm_values[128], partial_a[4], partial_b[4];
    threadgroup uint risks[4];
    const uint token = row / p.heads, head = row % p.heads;
    const uint base = token * p.inner + head * 128;
    if (p.policy & 2u) {
        norm_values[lane] = values[base + lane];
        glm_kda_parallel_norm<false,true>(norm_values,norm_values,partial_a,partial_b,risks,scales,lane,p.dim,p.rms_eps);
    } else {
        if (lane == 0) {
            float sum = 0.0f;
            for (uint i = 0; i < p.dim; ++i) sum += values[base + i] * values[base + i];
            scales[0] = 1.0f / sqrt(sum / float(p.dim) + p.rms_eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float g = gate[base + lane];
    const float norm = weight[3 * p.inner * p.conv_width + p.inner + lane];
    const float result = values[base + lane] * scales[0] * norm * glm_kp_sigmoid(g);
    output[base + lane] = result;
    if (!isfinite(g) || !isfinite(result)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
}
