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

// Persistent graph controls. Reductions deliberately retain the scalar
// reference order; no activation or recurrent state is narrowed to BF16.
struct GlmControlParameters {
    uint hidden;
    uint streams;
    uint iterations;
    uint columns;
    float epsilon;
};

kernel void glm_control_project_scalar(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant GlmControlParameters &p [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= (2 + p.streams) * p.streams) return;
    float4 s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    uint end = p.columns / 16 * 16;
    device const float *weight = weights + row * p.columns;
    for (uint i = 0; i < end; i += 16) {
        s0 += float4(weight[i], weight[i+1], weight[i+2], weight[i+3]) * float4(input[i], input[i+1], input[i+2], input[i+3]);
        s1 += float4(weight[i+4], weight[i+5], weight[i+6], weight[i+7]) * float4(input[i+4], input[i+5], input[i+6], input[i+7]);
        s2 += float4(weight[i+8], weight[i+9], weight[i+10], weight[i+11]) * float4(input[i+8], input[i+9], input[i+10], input[i+11]);
        s3 += float4(weight[i+12], weight[i+13], weight[i+14], weight[i+15]) * float4(input[i+12], input[i+13], input[i+14], input[i+15]);
    }
    float4 sums = (s0 + s1) + (s2 + s3);
    float sum = (sums.x + sums.y) + (sums.z + sums.w);
    for (uint i = end; i < p.columns; ++i) sum += weight[i] * input[i];
    if (!isfinite(sum)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
    output[row] = sum;
}

kernel void glm_control_project_unprefetched(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant GlmControlParameters &p [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    // One SIMD group owns a row. The first 16 lanes retain the independent
    // accumulator streams of the CPU's four NEON float4 accumulators.
    uint end = p.columns / 16 * 16;
    device const float *weight = weights + row * p.columns;
    float accumulator = 0.0f;
    if (lane < 16) {
        for (uint i = lane; i < end; i += 16) accumulator += weight[i] * input[i];
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

kernel void glm_control_project(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant GlmControlParameters &p [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
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

kernel void glm_control_mix_scalar(
    device const float *projection [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *mix [[buffer(2)]],
    constant GlmControlParameters &p [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    uint count = (2 + p.streams) * p.streams;
    device const float *base = weights;
    device const float *scale = weights + count;
    for (uint i = 0; i < p.streams; ++i) {
        mix[i] = glm_sigmoid(projection[i] * scale[0] + base[i]) + p.epsilon;
        mix[p.streams+i] = 2.0f * glm_sigmoid(projection[p.streams+i] * scale[1] + base[p.streams+i]);
    }
    device float *comb = mix + 2 * p.streams;
    for (uint row = 0; row < p.streams; ++row) {
        float maximum = -INFINITY;
        for (uint col = 0; col < p.streams; ++col) {
            uint i = 2 * p.streams + row * p.streams + col;
            comb[row * p.streams + col] = projection[i] * scale[2] + base[i];
            maximum = max(maximum, comb[row * p.streams + col]);
        }
        float denominator = 0.0f;
        for (uint col = 0; col < p.streams; ++col) {
            comb[row * p.streams + col] = exp(comb[row * p.streams + col] - maximum);
            denominator += comb[row * p.streams + col];
        }
        for (uint col = 0; col < p.streams; ++col) comb[row * p.streams + col] = comb[row * p.streams + col] / denominator + p.epsilon;
    }
    for (uint iteration = 0; iteration < p.iterations; ++iteration) {
        if (iteration > 0) {
            for (uint row = 0; row < p.streams; ++row) {
                float denominator = 0.0f;
                for (uint col = 0; col < p.streams; ++col) denominator += comb[row * p.streams + col];
                denominator += p.epsilon;
                for (uint col = 0; col < p.streams; ++col) comb[row * p.streams + col] /= denominator;
            }
        }
        for (uint col = 0; col < p.streams; ++col) {
            float denominator = 0.0f;
            for (uint row = 0; row < p.streams; ++row) denominator += comb[row * p.streams + col];
            denominator += p.epsilon;
            for (uint row = 0; row < p.streams; ++row) comb[row * p.streams + col] /= denominator;
        }
    }
    for (uint i = 0; i < count; ++i) if (!isfinite(mix[i])) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
}

kernel void glm_control_mix(
    device const float *projection [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *mix [[buffer(2)]],
    constant GlmControlParameters &p [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]) {
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

kernel void glm_control_input(
    device const float *residual [[buffer(0)]],
    device const float *mix [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant GlmControlParameters &p [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]],
    uint d [[thread_position_in_grid]]) {
    if (d >= p.hidden) return;
    float sum = 0.0f;
    for (uint stream = 0; stream < p.streams; ++stream) sum += mix[stream] * residual[stream * p.hidden + d];
    output[d] = sum;
    if (!isfinite(sum)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
}

kernel void glm_control_post(
    device const float *residual [[buffer(0)]],
    device const float *output [[buffer(1)]],
    device const float *mix [[buffer(2)]],
    device float *next [[buffer(3)]],
    constant GlmControlParameters &p [[buffer(4)]],
    device atomic_uint *failure [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= p.columns) return;
    uint dst = i / p.hidden;
    uint d = i % p.hidden;
    float sum = mix[p.streams + dst] * output[d];
    for (uint src = 0; src < p.streams; ++src) sum += mix[2 * p.streams + src * p.streams + dst] * residual[src * p.hidden + d];
    next[i] = sum;
    if (!isfinite(sum)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
}

struct GlmAttentionParameters {
    uint tokens;
    uint rank;
    uint heads;
    uint index_dim;
    uint index_heads;
    uint pool;
    uint pools;
    uint top_pools;
    uint sort_size;
    uint sort_span;
    uint sort_stride;
    float scale;
};

kernel void glm_index_pool(
    device const float *keys [[buffer(0)]],
    device const float *gates [[buffer(1)]],
    device const float *ape [[buffer(2)]],
    device float *pooled [[buffer(3)]],
    constant GlmAttentionParameters &p [[buffer(4)]],
    device atomic_uint *failure [[buffer(5)]],
    uint channel [[thread_position_in_grid]]) {
    if (channel >= p.index_dim) return;
    float probabilities[64];
    float maximum = -INFINITY;
    for (uint i = 0; i < p.pool; ++i) {
        probabilities[i] = gates[i * p.index_dim + channel] + ape[i * p.index_dim + channel];
        maximum = max(maximum, probabilities[i]);
    }
    float denominator = 0.0f;
    for (uint i = 0; i < p.pool; ++i) {
        probabilities[i] = exp(probabilities[i] - maximum);
        denominator += probabilities[i];
    }
    float sum = 0.0f;
    for (uint i = 0; i < p.pool; ++i) sum += (probabilities[i] / denominator) * keys[i * p.index_dim + channel];
    pooled[channel] = sum;
    if (!isfinite(sum) || !isfinite(denominator) || denominator <= 0.0f) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
}

kernel void glm_index_scores(
    device const float *queries [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *keys [[buffer(2)]],
    device float *scores [[buffer(3)]],
    device uint *indices [[buffer(4)]],
    constant GlmAttentionParameters &p [[buffer(5)]],
    device atomic_uint *failure [[buffer(6)]],
    uint group [[thread_position_in_grid]]) {
    if (group >= p.sort_size) return;
    float sum = -INFINITY;
    if (group < p.pools) {
        sum = 0.0f;
        float scale = 1.0f / sqrt(float(p.index_dim * p.index_heads));
        for (uint head = 0; head < p.index_heads; ++head) {
            float dot = 0.0f;
            for (uint d = 0; d < p.index_dim; ++d) dot += queries[head * p.index_dim + d] * keys[group * p.index_dim + d];
            sum += max(dot, 0.0f) * weights[head] * scale;
        }
        if (!isfinite(sum)) {
            atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
            // A failed score must still be safe to sort before the owning
            // command observes its sticky status. Stable -infinity ties keep
            // padding behind every ordinary pool.
            sum = -INFINITY;
        }
    }
    scores[group] = sum;
    indices[group] = group;
}

static bool glm_precedes(float a, uint ai, float b, uint bi) {
    // All ordinary index scores are finite. Explicit signbit keeps the same
    // total_cmp ordering for signed zero as the CPU selector.
    return a > b || (a == b && (signbit(a) != signbit(b) ? !signbit(a) : ai < bi));
}

kernel void glm_index_sort(
    device float *scores [[buffer(0)]],
    device uint *indices [[buffer(1)]],
    constant GlmAttentionParameters &p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    uint j = i ^ p.sort_stride;
    if (i >= p.sort_size || j <= i || j >= p.sort_size) return;
    bool ascending = (i & p.sort_span) == 0;
    bool before = glm_precedes(scores[i], indices[i], scores[j], indices[j]);
    if (before != ascending) {
        float score = scores[i]; scores[i] = scores[j]; scores[j] = score;
        uint index = indices[i]; indices[i] = indices[j]; indices[j] = index;
    }
}

kernel void glm_index_mask(
    device const uint *indices [[buffer(0)]],
    device uint *mask [[buffer(1)]],
    constant GlmAttentionParameters &p [[buffer(2)]],
    device atomic_uint *failure [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= p.top_pools) return;
    uint index = indices[i];
    if (index >= p.pools) {
        atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
        return;
    }
    mask[index] = 1;
}

kernel void glm_mla_scores(
    device const float *query [[buffer(0)]],
    device const float *latents [[buffer(1)]],
    device const uint *mask [[buffer(2)]],
    device float *scores [[buffer(3)]],
    constant GlmAttentionParameters &p [[buffer(4)]],
    device atomic_uint *failure [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= p.heads * p.tokens) return;
    uint position = i % p.tokens;
    uint head = i / p.tokens;
    if (p.top_pools < p.pools && position < p.pools * p.pool && mask[position / p.pool] == 0) {
        scores[i] = -INFINITY;
        return;
    }
    float sum = 0.0f;
    for (uint d = 0; d < p.rank; ++d) sum += query[head * p.rank + d] * latents[position * p.rank + d];
    scores[i] = sum * p.scale;
    if (!isfinite(scores[i])) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
}

kernel void glm_mla_softmax(
    device float *scores [[buffer(0)]],
    constant GlmAttentionParameters &p [[buffer(1)]],
    device atomic_uint *failure [[buffer(2)]],
    uint head [[thread_position_in_grid]]) {
    if (head >= p.heads) return;
    device float *row = scores + head * p.tokens;
    float maximum = -INFINITY;
    for (uint i = 0; i < p.tokens; ++i) maximum = max(maximum, row[i]);
    float sum = 0.0f;
    for (uint i = 0; i < p.tokens; ++i) {
        row[i] = exp(row[i] - maximum);
        sum += row[i];
    }
    for (uint i = 0; i < p.tokens; ++i) row[i] /= sum;
    if (!isfinite(sum) || sum <= 0.0f) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
}

kernel void glm_mla_values(
    device const float *probabilities [[buffer(0)]],
    device const float *latents [[buffer(1)]],
    device const uint *mask [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant GlmAttentionParameters &p [[buffer(4)]],
    device atomic_uint *failure [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= p.heads * p.rank) return;
    uint head = i / p.rank;
    uint d = i % p.rank;
    float sum = 0.0f;
    for (uint position = 0; position < p.tokens; ++position) {
        if (p.top_pools < p.pools && position < p.pools * p.pool && mask[position / p.pool] == 0) continue;
        sum += probabilities[head * p.tokens + position] * latents[position * p.rank + d];
    }
    output[i] = sum;
    if (!isfinite(sum)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
}

kernel void glm_index_layer_norm(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &length [[buffer(4)]],
    device atomic_uint *failure [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float moments[2];
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < length; ++i) sum += input[i];
        float mean = sum / float(length);
        float variance = 0.0f;
        for (uint i = 0; i < length; ++i) variance += (input[i] - mean) * (input[i] - mean);
        moments[0] = mean;
        moments[1] = sqrt(variance / float(length) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < length; i += 256) {
        float v = (input[i] - moments[0]) / moments[1] * weight[i] + bias[i];
        output[i] = v;
        if (!isfinite(v)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
    }
}

kernel void glm_stream_mean(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant GlmControlParameters &p [[buffer(2)]],
    uint d [[thread_position_in_grid]]) {
    if (d >= p.hidden) return;
    float sum = 0.0f;
    for (uint stream = 0; stream < p.streams; ++stream) sum += input[stream * p.hidden + d];
    output[d] = sum / float(p.streams);
}

kernel void glm_kda_graph_status(
    device const uint *status [[buffer(0)]],
    constant uint &unused [[buffer(1)]],
    device atomic_uint *failure [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i == 0 && status[0] != 0) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
}
