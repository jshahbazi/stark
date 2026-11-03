// SPDX-License-Identifier: Apache-2.0
// Original Stark GLM KDA decode, matching the scalar contract in glm.rs.
// The reference retains [head, value, key] F32 state. The optional transposed
// backing layout coalesces adjacent value lanes without changing key order.
// Each lane owns one value row; reductions retain the scalar reference order.
#include <metal_stdlib>
using namespace metal;

#ifndef STARK_GLM_KDA_TRANSPOSED
#define STARK_GLM_KDA_TRANSPOSED 0
#endif
#ifndef STARK_GLM_KDA_FUSED_STATE
#define STARK_GLM_KDA_FUSED_STATE 0
#endif
#ifndef STARK_GLM_KDA_PARALLEL_NORM
#define STARK_GLM_KDA_PARALLEL_NORM 0
#endif


static float glm_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + exp(-x));
    float e = exp(x);
    return e / (1.0f + e);
}

// Ascending IEEE total-order key, including the -0/+0 distinction. Router
// NaNs are rejected before selection; equal keys retain ascending expert IDs.
static uint glm_ordered_key(float value) {
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

kernel void glm_router_scores(
    device const float *logits [[buffer(0)]], device const float *bias [[buffer(1)]],
    device float *probabilities [[buffer(2)]], device float *choices [[buffer(3)]],
    constant uint *p [[buffer(4)]], device atomic_uint *failure [[buffer(5)]],
    uint expert [[thread_position_in_grid]]) {
    if (expert >= p[0]) return;
    float probability = glm_sigmoid(logits[expert]);
    float choice = probability + bias[expert];
    if (!isfinite(logits[expert]) || !isfinite(probability) || isnan(choice)) {
        atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
        probability = 0.0f;
        choice = -INFINITY;
    }
    probabilities[expert] = probability;
    choices[expert] = choice;
}

kernel void glm_router_select(
    device const float *probabilities [[buffer(0)]], device const float *choices [[buffer(1)]],
    device uint *indices [[buffer(2)]], device float *mixture [[buffer(3)]],
    constant uint *p [[buffer(4)]], device atomic_uint *failure [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]]) {
    if (tid != 0) return;
    for (uint rank = 0; rank < p[1]; ++rank) {
        bool found = false;
        uint best = 0;
        uint key = 0;
        for (uint expert = 0; expert < p[0]; ++expert) {
            bool used = false;
            for (uint previous = 0; previous < rank; ++previous) used |= indices[previous] == expert;
            if (used) continue;
            const uint candidate = glm_ordered_key(choices[expert]);
            if (!found || candidate > key) { found = true; best = expert; key = candidate; }
        }
        if (!found) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
        indices[rank] = best;
    }
    float denominator = 0.0f;
    for (uint rank = 0; rank < p[1]; ++rank) denominator += probabilities[indices[rank]];
    denominator += 1e-20f;
    const float scale = as_type<float>(p[2]);
    for (uint rank = 0; rank < p[1]; ++rank) {
        const float value = probabilities[indices[rank]] / denominator * scale;
        if (!isfinite(value) || value < 0.0f) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
        mixture[rank] = value;
    }
}

// One block computes probabilities and repeatedly reduces the stable top-k.
// Keep the selected-order normalization on one lane, matching glm_router_select.
// This replaces its serial expert scans with SIMD reductions and threadgroup
// storage; no probability arithmetic or comparison key is changed.
kernel void glm_router_parallel(
    device const float *logits [[buffer(0)]], device const float *bias [[buffer(1)]],
    device uint *indices [[buffer(2)]], device float *mixture [[buffer(3)]],
    constant uint *p [[buffer(4)]], device atomic_uint *failure [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup float probabilities[2048];
    threadgroup uint keys[2048];
    threadgroup uint group_keys[8];
    threadgroup uint group_ids[8];
    threadgroup uint selected[64];
    for (uint expert = tid; expert < p[0]; expert += 256) {
        float probability = glm_sigmoid(logits[expert]);
        float choice = probability + bias[expert];
        if (!isfinite(logits[expert]) || !isfinite(probability) || isnan(choice)) {
            atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
            probability = 0.0f;
            choice = -INFINITY;
        }
        probabilities[expert] = probability;
        keys[expert] = glm_ordered_key(choice);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint rank = 0; rank < p[1]; ++rank) {
        uint best_key = 0;
        uint best_id = 0xffffffffu;
        for (uint expert = tid; expert < p[0]; expert += 256) {
            const uint key = keys[expert];
            if (key != 0 && (key > best_key || (key == best_key && expert < best_id))) {
                best_key = key;
                best_id = expert;
            }
        }
        const uint simd_key = simd_max(best_key);
        const uint simd_id = simd_min(best_key == simd_key ? best_id : 0xffffffffu);
        if (lane == 0) { group_keys[simd_group] = simd_key; group_ids[simd_group] = simd_id; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint winner_key = 0;
            uint winner = 0xffffffffu;
            for (uint group = 0; group < 8; ++group) {
                const uint key = group_keys[group];
                const uint id = group_ids[group];
                if (key > winner_key || (key == winner_key && id < winner)) {
                    winner_key = key;
                    winner = id;
                }
            }
            if (winner >= p[0]) {
                atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
                winner = 0;
            }
            selected[rank] = winner;
            indices[rank] = winner;
            // Every non-NaN IEEE comparison key, including -infinity, is
            // positive; zero is an unambiguous already-selected sentinel.
            keys[winner] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float denominator = 0.0f;
        for (uint rank = 0; rank < p[1]; ++rank) denominator += probabilities[selected[rank]];
        denominator += 1e-20f;
        const float scale = as_type<float>(p[2]);
        for (uint rank = 0; rank < p[1]; ++rank) {
            const float value = probabilities[selected[rank]] / denominator * scale;
            if (!isfinite(value) || value < 0.0f) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
            mixture[rank] = value;
        }
    }
}

struct GlmKdaParameters {
    uint heads;
    uint dim;
    uint conv_width;
    uint inner;
    float rms_eps;
    float lower_bound;
};

#if STARK_GLM_KDA_PARALLEL_NORM
// Original Stark optional norm reduction. The normal tree changes F32
// association. Each extreme vector retains its original chronological sum.
// Threadgroup storage belongs to the kernel, never this inline helper.
template<bool Pair, bool Rms>
inline void glm_kda_parallel_norm(
    threadgroup const float *a, threadgroup const float *b,
    threadgroup float *partial_a, threadgroup float *partial_b,
    threadgroup uint *risks, threadgroup float *scales,
    uint tid, uint width, float epsilon) {
    const uint lane = tid % 32;
    const uint sg = tid / 32;
    const uint groups = (width + 31) / 32;
    const float av = tid < width ? a[tid] : 0.0f;
    const float bv = Pair && tid < width ? b[tid] : 0.0f;
    const bool risky_a = simd_any(!isfinite(av) || fabs(av) > 1e15f);
    const bool risky_b = simd_any(Pair && (!isfinite(bv) || fabs(bv) > 1e15f));
    const float sum_a = simd_sum(av * av);
    const float sum_b = simd_sum(bv * bv);
    if (lane == 0) {
        partial_a[sg] = sum_a;
        partial_b[sg] = sum_b;
        risks[sg] = (risky_a ? 1u : 0u) | (risky_b ? 2u : 0u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const bool ordered_a = simd_any(lane < groups ? risks[lane] != 0 : false);
        const bool ordered_b = Pair && ordered_a;
        float total_a = simd_sum(lane < groups ? partial_a[lane] : 0.0f);
        float total_b = simd_sum(lane < groups ? partial_b[lane] : 0.0f);
        if (lane == 0) {
            if (ordered_a) {
                total_a = 0.0f;
                for (uint i = 0; i < width; ++i) total_a += a[i] * a[i];
            }
            if (Pair && ordered_b) {
                total_b = 0.0f;
                for (uint i = 0; i < width; ++i) total_b += b[i] * b[i];
            }
            scales[0] = Rms ? 1.0f / sqrt(total_a / float(width) + epsilon) : sqrt(total_a + epsilon);
            if (Pair) scales[1] = Rms ? 1.0f / sqrt(total_b / float(width) + epsilon) : sqrt(total_b + epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Test fixture for the only changed arithmetic. All values outside width are
// masked; the original lane0 reference and candidate share sqrt/divide policy.
kernel void glm_kda_parallel_norm_probe(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant uint *p [[buffer(2)]], device atomic_uint *failure [[buffer(30)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float values[256];
    threadgroup float partial_a[8], partial_b[8], scales[2];
    threadgroup uint risks[8];
    values[tid] = tid < p[0] ? input[tid] : 0.0f;
    if (tid < p[0] && !isfinite(values[tid])) atomic_fetch_or_explicit(failure,1u,memory_order_relaxed);
    if (p[2]) {
        if (p[1]) glm_kda_parallel_norm<false,true>(values,values,partial_a,partial_b,risks,scales,tid,p[0],as_type<float>(p[3]));
        else glm_kda_parallel_norm<false,false>(values,values,partial_a,partial_b,risks,scales,tid,p[0],as_type<float>(p[3]));
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float sum = 0.0f;
            for (uint i=0;i<p[0];++i) sum += values[i]*values[i];
            scales[0] = p[1] ? 1.0f/sqrt(sum/float(p[0])+as_type<float>(p[3]))
                            : sqrt(sum+as_type<float>(p[3]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < p[0]) {
        float value = p[1] ? values[tid]*scales[0] : values[tid]/scales[0];
        output[tid] = value;
        if (!isfinite(value)) atomic_fetch_or_explicit(failure,1u,memory_order_relaxed);
    }
}
#endif

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
    #if STARK_GLM_KDA_PARALLEL_NORM
    threadgroup float norm_partial_a[8], norm_partial_b[8];
    threadgroup uint norm_risks[8];
    #endif
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
    #if STARK_GLM_KDA_PARALLEL_NORM
    glm_kda_parallel_norm<true,false>(q,k,norm_partial_a,norm_partial_b,norm_risks,scales,lane,p.dim,1e-6f);
    #else
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
    #endif
    if (lane < p.dim) {
        q[lane] /= scales[0];
        k[lane] /= scales[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < p.dim) {
        const uint stride = STARK_GLM_KDA_TRANSPOSED ? p.dim : 1;
        device float *row = state + 3 * history_length + head * p.dim * p.dim
            + (STARK_GLM_KDA_TRANSPOSED ? lane : lane * p.dim);
        float prediction = 0.0f;
        #if STARK_GLM_KDA_FUSED_STATE
        for (uint i = 0; i < p.dim; ++i) {
            const float decayed = row[i * stride] * decay[i];
            row[i * stride] = decayed;
            prediction += decayed * k[i];
        }
        #else
        for (uint i = 0; i < p.dim; ++i) row[i * stride] *= decay[i];
        for (uint i = 0; i < p.dim; ++i) prediction += row[i * stride] * k[i];
        #endif
        float beta = glm_sigmoid(input[5 * p.inner + head]);
        float delta = (v[lane] - prediction) * beta;
        float sum = 0.0f;
        #if STARK_GLM_KDA_FUSED_STATE
        for (uint i = 0; i < p.dim; ++i) {
            const float updated = row[i * stride] + k[i] * delta;
            row[i * stride] = updated;
            sum += updated * q[i];
        }
        #else
        for (uint i = 0; i < p.dim; ++i) row[i * stride] += k[i] * delta;
        for (uint i = 0; i < p.dim; ++i) sum += row[i * stride] * q[i];
        #endif
        values[lane] = sum * (1.0f / sqrt(float(p.dim)));
    }
    #if STARK_GLM_KDA_PARALLEL_NORM
    glm_kda_parallel_norm<false,true>(values,values,norm_partial_a,norm_partial_b,norm_risks,scales+2,lane,p.dim,p.rms_eps);
    #else
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < p.dim; ++i) sum += values[i] * values[i];
        scales[2] = 1.0f / sqrt(sum / float(p.dim) + p.rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #endif
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
