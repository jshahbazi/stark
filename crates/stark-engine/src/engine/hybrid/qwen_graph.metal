// SPDX-License-Identifier: Apache-2.0
// Original scalar-order Qwen graph kernels. Projections and n-gram gathers remain
// in the existing weight and host paths; these kernels own only control work.

inline float qwen_graph_sigmoid_value(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + exp(-x));
    float e = exp(x);
    return e / (1.0f + e);
}

kernel void qwen_graph_sigmoid(device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]], constant uint *p [[buffer(2)]],
    device atomic_uint *failure [[buffer(30)]], uint index [[thread_position_in_grid]]) {
    if (index >= p[0]) return;
    float value = qwen_graph_sigmoid_value(input[index]);
    output[index] = value;
    if (!isfinite(input[index]) || !isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
}

kernel void qwen_graph_inject(device const float *residual [[buffer(0)]],
    device const float *output [[buffer(1)]], device const float *gates [[buffer(2)]],
    device float *result [[buffer(3)]], constant uint *p [[buffer(4)]],
    device atomic_uint *failure [[buffer(30)]], uint index [[thread_position_in_grid]]) {
    uint streams = p[0], dim = p[1];
    uint width = streams * dim;
    if (index >= width) return;
    float value = residual[index] + gates[index / dim] * output[index % dim];
    result[index] = value;
    if (!isfinite(residual[index]) || !isfinite(output[index % dim]) ||
        !isfinite(gates[index / dim]) || !isfinite(value)) {
        atomic_store_explicit(failure, 1u, memory_order_relaxed);
    }
}

inline void qwen_graph_rope(device float *x, uint offset, uint rotary_dim,
    uint position, float theta) {
    uint half_width = rotary_dim / 2;
    for (uint i = 0; i < half_width; ++i) {
        float phase = float(position) * pow(theta, -2.0f * float(i) / float(rotary_dim));
        float cosine;
        float sine = sincos(phase, cosine);
        float left = x[offset + i];
        float right = x[offset + i + half_width];
        x[offset + i] = left * cosine - right * sine;
        x[offset + i + half_width] = right * cosine + left * sine;
    }
}

inline int qwen_graph_total_key(float value) {
    uint bits = as_type<uint>(value);
    uint mask = (as_type<int>(bits) >> 31) == 0 ? 0u : 0x7fffffffu;
    return as_type<int>(bits ^ mask);
}

// Inputs: index projection [index_heads+1,index_dim], q/gate [heads,2*head_dim],
// K/V [kv_heads,head_dim], followed by q and k norm weights. The raw index key
// is appended at the zero-based position; Q is current-only, K/V caches are token-major.
kernel void qwen_graph_qsa_prepare(device const float *index_projection [[buffer(0)]],
    device const float *query_gate [[buffer(1)]], device const float *key_input [[buffer(2)]],
    device const float *value_input [[buffer(3)]], device const float *query_norm [[buffer(4)]],
    device const float *key_norm [[buffer(5)]], device float *index_cache [[buffer(6)]],
    device float *query_cache [[buffer(7)]], device float *key_cache [[buffer(8)]],
    device float *value_cache [[buffer(9)]], device float *gates [[buffer(10)]],
    constant uint *p [[buffer(11)]],
    device atomic_uint *failure [[buffer(30)]], uint tid [[thread_position_in_grid]]) {
    uint count = p[0], index_heads = p[1], index_dim = p[2], heads = p[3];
    uint head_dim = p[4], kv_heads = p[5], rotary_dim = p[6];
    uint position = p[7];
    float eps = as_type<float>(p[8]), theta = as_type<float>(p[9]);
    if (count == 0 || position + 1 != count) {
        atomic_store_explicit(failure, 1u, memory_order_relaxed);
        return;
    }
    // Independent query heads, KV heads, and the raw index append execute as
    // separate tasks. Every head retains its original ordered F32 reduction.
    if (tid == 0) {
        uint index_width = index_heads * index_dim;
        for (uint i = 0; i < (index_heads + 1) * index_dim; ++i) {
            if (!isfinite(index_projection[i])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        }
        for (uint i = 0; i < index_dim; ++i) {
            index_cache[position * index_dim + i] = index_projection[index_width + i];
        }
    } else if (tid <= heads) {
        uint head = tid - 1;
        uint src = head * 2 * head_dim;
        float sum = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            float x = query_gate[src + i];
            sum += x * x;
        }
        float scale = 1.0f / sqrt(sum / float(head_dim) + eps);
        uint dst = head * head_dim;
        for (uint i = 0; i < head_dim; ++i) {
            query_cache[dst + i] = query_gate[src + i] * scale * query_norm[i];
            gates[dst + i] = query_gate[src + head_dim + i];
        }
        qwen_graph_rope(query_cache, dst, rotary_dim, position, theta);
        for (uint i = 0; i < head_dim; ++i) {
            if (!isfinite(query_cache[dst + i]) || !isfinite(gates[dst + i])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        }
    } else if (tid <= heads + kv_heads) {
        uint head = tid - heads - 1;
        uint dst = position * kv_heads * head_dim + head * head_dim;
        uint src = head * head_dim;
        float sum = 0.0f;
        for (uint i = 0; i < head_dim; ++i) {
            float x = key_input[src + i];
            sum += x * x;
        }
        float scale = 1.0f / sqrt(sum / float(head_dim) + eps);
        for (uint i = 0; i < head_dim; ++i) key_cache[dst + i] = key_input[src + i] * scale * key_norm[i];
        qwen_graph_rope(key_cache, dst, rotary_dim, position, theta);
        for (uint i = 0; i < head_dim; ++i) {
            float value = value_input[src + i];
            value_cache[position * kv_heads * head_dim + src + i] = value;
            if (!isfinite(value) || !isfinite(key_cache[dst + i])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        }
    }
}

// Sparse QSA visibility. A single scalar invocation intentionally preserves the
// Rust pooling, per-head dot, and stable top-k order for the first graph version.
kernel void qwen_graph_qsa_visible(device const float *index_projection [[buffer(0)]],
    device const float *index_cache [[buffer(1)]], device const float *query_norm [[buffer(2)]],
    device const float *key_norm [[buffer(3)]], device uint *visible_mask [[buffer(4)]],
    device float *block_scores [[buffer(5)]], device uint *top_ids [[buffer(6)]],
    device float *top_scores [[buffer(7)]], constant uint *p [[buffer(8)]],
    device atomic_uint *failure [[buffer(30)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    uint count = p[0], index_heads = p[1], dim = p[2], rotary_dim = p[3];
    uint position = p[4], ratio = p[7], budget = p[8];
    float eps = as_type<float>(p[5]), theta = as_type<float>(p[6]);
    if (count == 0 || ratio == 0 || budget < ratio) {
        atomic_store_explicit(failure, 1u, memory_order_relaxed);
        return;
    }
    if (count <= budget) {
        for (uint token = 0; token < count; ++token) visible_mask[token] = 1;
        return;
    }
    uint blocks = count / ratio;
    uint keep_count = min(budget / ratio, blocks);
    for (uint slot = 0; slot < keep_count; ++slot) {
        top_ids[slot] = UINT_MAX;
        top_scores[slot] = -INFINITY;
    }
    // Normalize and rotate each query head, then score each complete pooled block.
    for (uint block = 0; block < blocks; ++block) {
        float score = 0.0f;
        for (uint head = 0; head < index_heads; ++head) {
            uint qbase = head * dim;
            float qsum = 0.0f;
            float pooled_square_sum = 0.0f;
            for (uint i = 0; i < dim; ++i) {
                float x = index_projection[qbase + i];
                qsum += x * x;
            }
            float qscale = 1.0f / sqrt(qsum / float(dim) + eps);
            float head_score = 0.0f;
            // Compute pooled-key RMS and dot in scalar order using one dimension
            // at a time; the pooling sum divides each key before accumulation.
            for (uint i = 0; i < dim; ++i) {
                float pooled = 0.0f;
                for (uint token = block * ratio; token < (block + 1) * ratio; ++token) {
                    pooled += index_cache[token * dim + i] / float(ratio);
                }
                pooled_square_sum += pooled * pooled;
                // Defer normalized pooled components until the pooled RMS is known.
            }
            float pooled_scale = 1.0f / sqrt(pooled_square_sum / float(dim) + eps);
            // Recompute each pooled component (same sequence) for normalization,
            // RoPE, and dot accumulation without a large per-thread array.
            for (uint i = 0; i < dim; ++i) {
                float pooled = 0.0f;
                for (uint token = block * ratio; token < (block + 1) * ratio; ++token) {
                    pooled += index_cache[token * dim + i] / float(ratio);
                }
                pooled = pooled * pooled_scale * key_norm[i];
                float rotated = pooled;
                if (i < rotary_dim / 2 || (i >= rotary_dim / 2 && i < rotary_dim)) {
                    uint r = i < rotary_dim / 2 ? i : i - rotary_dim / 2;
                    float phase = float(block * ratio) * pow(theta, -2.0f * float(r) / float(rotary_dim));
                    float cosine;
                    float sine = sincos(phase, cosine);
                    // Need both paired components. Recompute the paired normalized
                    // pooled element directly, retaining the Rust half-split RoPE.
                    uint partner = i < rotary_dim / 2 ? i + rotary_dim / 2 : i - rotary_dim / 2;
                    float paired = 0.0f;
                    for (uint token = block * ratio; token < (block + 1) * ratio; ++token) {
                        paired += index_cache[token * dim + partner] / float(ratio);
                    }
                    paired = paired * pooled_scale * key_norm[partner];
                    rotated = i < rotary_dim / 2 ? pooled * cosine - paired * sine : paired * sine + pooled * cosine;
                }
                float q = index_projection[qbase + i] * qscale * query_norm[i];
                if (i < rotary_dim / 2 || (i >= rotary_dim / 2 && i < rotary_dim)) {
                    uint r = i < rotary_dim / 2 ? i : i - rotary_dim / 2;
                    float phase = float(position) * pow(theta, -2.0f * float(r) / float(rotary_dim));
                    float cosine;
                    float sine = sincos(phase, cosine);
                    uint partner = i < rotary_dim / 2 ? i + rotary_dim / 2 : i - rotary_dim / 2;
                    float other = index_projection[qbase + partner] * qscale * query_norm[partner];
                    q = i < rotary_dim / 2 ? q * cosine - other * sine : other * sine + q * cosine;
                }
                head_score += q * rotated;
            }
            score += max(head_score, 0.0f);
        }
        score /= sqrt(float(dim));
        if (isnan(score)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        block_scores[block] = score;
        // Insertion order implements descending total_cmp followed by ascending id.
        uint insertion = keep_count;
        for (uint slot = 0; slot < keep_count; ++slot) {
            float old = top_scores[slot];
            uint old_id = top_ids[slot];
            int score_key = qwen_graph_total_key(score);
            int old_key = qwen_graph_total_key(old);
            if (score_key > old_key || (score_key == old_key && block < old_id)) { insertion = slot; break; }
        }
        if (insertion < keep_count) {
            for (uint slot = keep_count - 1; slot > insertion; --slot) {
                top_scores[slot] = top_scores[slot - 1];
                top_ids[slot] = top_ids[slot - 1];
            }
            top_scores[insertion] = score;
            top_ids[insertion] = block;
        }
    }
    for (uint token = 0; token < count; ++token) {
        uint block = token / ratio;
        bool keep = token >= blocks * ratio;
        if (!keep) {
            for (uint slot = 0; slot < keep_count; ++slot) keep |= top_ids[slot] == block;
        }
        visible_mask[token] = keep ? 1u : 0u;
    }
}

// Separate independent score dots, ordered softmax reductions and independent
// value channels. Scratch remains [heads,capacity]; arithmetic within every dot
// and visible-token reduction follows the scalar reference order.
kernel void qwen_graph_qsa_scores(device const float *query [[buffer(0)]],
    device const float *key_cache [[buffer(1)]], device const uint *visible_mask [[buffer(2)]],
    device float *scores [[buffer(3)]], constant uint *p [[buffer(4)]],
    device atomic_uint *failure [[buffer(30)]], uint task [[thread_position_in_grid]]) {
    uint count = p[0], heads = p[1], dim = p[2], kv_heads = p[3], capacity = p[4];
    uint head = task / count, token = task % count;
    if (head >= heads) return;
    if (!visible_mask[token]) { scores[head * capacity + token] = -INFINITY; return; }
    uint kv_head = head / (heads / kv_heads);
    uint qbase = head * dim, kbase = kv_head * dim;
    float score = 0.0f;
    for (uint i = 0; i < dim; ++i) score += query[qbase + i] * key_cache[token * (kv_heads * dim) + kbase + i];
    score /= sqrt(float(dim));
    scores[head * capacity + token] = score;
    if (isnan(score) || score == INFINITY) atomic_store_explicit(failure, 1u, memory_order_relaxed);
}

kernel void qwen_graph_qsa_softmax(device const float *scores [[buffer(0)]],
    device const uint *visible_mask [[buffer(1)]], device float *probabilities [[buffer(2)]],
    constant uint *p [[buffer(3)]], device atomic_uint *failure [[buffer(30)]],
    uint head [[thread_position_in_grid]]) {
    uint count = p[0], heads = p[1], capacity = p[2];
    if (head >= heads) return;
    float max_score = -INFINITY;
    for (uint token = 0; token < count; ++token) {
        if (visible_mask[token]) max_score = max(max_score, scores[head * capacity + token]);
    }
    float sum = 0.0f;
    for (uint token = 0; token < count; ++token) {
        float value = visible_mask[token] ? exp(scores[head * capacity + token] - max_score) : 0.0f;
        probabilities[head * capacity + token] = value;
        sum += value;
    }
    if (!isfinite(sum) || sum <= 0.0f) {
        atomic_store_explicit(failure, 1u, memory_order_relaxed);
        return;
    }
    for (uint token = 0; token < count; ++token) probabilities[head * capacity + token] /= sum;
}

kernel void qwen_graph_qsa_output(device const float *probabilities [[buffer(0)]],
    device const float *value_cache [[buffer(1)]], device const float *gates [[buffer(2)]],
    device const uint *visible_mask [[buffer(3)]], device float *output [[buffer(4)]],
    constant uint *p [[buffer(5)]], device atomic_uint *failure [[buffer(30)]],
    uint task [[thread_position_in_grid]]) {
    uint count = p[0], heads = p[1], dim = p[2], kv_heads = p[3], capacity = p[4];
    uint head = task / dim, channel = task % dim;
    if (head >= heads) return;
    uint kv_head = head / (heads / kv_heads);
    float value = 0.0f;
    for (uint token = 0; token < count; ++token) {
        if (visible_mask[token]) {
            value += probabilities[head * capacity + token] * value_cache[token * (kv_heads * dim) + kv_head * dim + channel];
        }
    }
    value *= qwen_graph_sigmoid_value(gates[task]);
    output[task] = value;
    if (!isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
}

// Projections are supplied by the packed projection path. History is chronological
// [time,channel], with (kernel-1)*dilation prior rows, and is advanced once/token.
kernel void qwen_graph_ple(device const float *key [[buffer(0)]],
    device const float *value [[buffer(1)]], device const float *query [[buffer(2)]],
    device const float *key_norm [[buffer(3)]], device const float *query_norm [[buffer(4)]],
    device const float *conv_norm [[buffer(5)]], device const float *conv_weights [[buffer(6)]],
    device float *history [[buffer(7)]], device float *output [[buffer(8)]],
    device float *scratch [[buffer(9)]], constant uint *p [[buffer(10)]],
    device atomic_uint *failure [[buffer(30)]], uint stream [[thread_position_in_grid]]) {
    uint streams = p[0], dim = p[1], kernel_size = p[2], dilation = p[3];
    float eps = as_type<float>(p[4]);
    if (streams == 0 || dim == 0 || kernel_size < 2 || dilation == 0) {
        atomic_store_explicit(failure, 1u, memory_order_relaxed);
        return;
    }
    if (stream >= streams) return;
    uint base = stream * dim;
    float key_sum = 0.0f, query_sum = 0.0f;
    for (uint i = 0; i < dim; ++i) {
        key_sum += key[base + i] * key[base + i];
        query_sum += query[base + i] * query[base + i];
    }
    float key_scale = 1.0f / sqrt(key_sum / float(dim) + eps);
    float query_scale = 1.0f / sqrt(query_sum / float(dim) + eps);
    float dot = 0.0f;
    for (uint i = 0; i < dim; ++i) {
        float normalized_key = key[base + i] * key_scale * key_norm[base + i];
        float normalized_query = query[base + i] * query_scale * query_norm[base + i];
        dot += normalized_key * normalized_query;
    }
    float score = dot / sqrt(float(dim));
    float sign = score > 0.0f ? 1.0f : (score < 0.0f ? -1.0f : 0.0f);
    float gate = qwen_graph_sigmoid_value(sign * sqrt(max(abs(score), 1e-6f)));
    for (uint i = 0; i < dim; ++i) output[base + i] = gate * value[i];
    float sum = 0.0f;
    for (uint i = 0; i < dim; ++i) sum += output[base + i] * output[base + i];
    float scale = 1.0f / sqrt(sum / float(dim) + eps);
    for (uint i = 0; i < dim; ++i) {
        float normalized = output[base + i] * scale * conv_norm[base + i];
        scratch[base + i] = normalized;
        if (!isfinite(key[base + i]) || !isfinite(query[base + i]) || !isfinite(value[i]) ||
            !isfinite(output[base + i]) || !isfinite(normalized)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    }
}

// Each channel owns its chronological history column, so convolution and
// shifting have no cross-channel dependency or in-place copy race.
kernel void qwen_graph_ple_convolve(device const float *conv_weights [[buffer(0)]],
    device float *history [[buffer(1)]], device float *output [[buffer(2)]],
    device const float *scratch [[buffer(3)]], constant uint *p [[buffer(4)]],
    device atomic_uint *failure [[buffer(30)]], uint channel [[thread_position_in_grid]]) {
    uint width = p[0] * p[1], kernel_size = p[2], dilation = p[3];
    if (channel >= width) return;
    float sum = scratch[channel] * conv_weights[channel * kernel_size + kernel_size - 1];
    for (uint tap = 0; tap < kernel_size - 1; ++tap) {
        sum += history[(tap * dilation) * width + channel] * conv_weights[channel * kernel_size + tap];
    }
    float activated = sum * qwen_graph_sigmoid_value(sum);
    output[channel] += activated;
    if (!isfinite(output[channel]) || !isfinite(activated)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    uint history_rows = (kernel_size - 1) * dilation;
    for (uint row = 0; row + 1 < history_rows; ++row) {
        history[row * width + channel] = history[(row + 1) * width + channel];
    }
    history[(history_rows - 1) * width + channel] = scratch[channel];
}

// Qwen routes on raw scores: top-k uses descending total_cmp and ascending id
// ties, while softmax uses the selected original scores in that selected order.
kernel void qwen_graph_router(device const float *scores [[buffer(0)]],
    device uint *selected_ids [[buffer(1)]], device float *selected_probs [[buffer(2)]],
    constant uint *p [[buffer(3)]], device atomic_uint *failure [[buffer(30)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    uint experts = p[0], k = p[1];
    if (k == 0 || k > experts) { atomic_store_explicit(failure, 1u, memory_order_relaxed); return; }
    // The command arena is uninitialized. Even a rejected router input must
    // write deterministic invalid indices before later encoded gathers run.
    for (uint slot = 0; slot < k; ++slot) { selected_ids[slot] = UINT_MAX; selected_probs[slot] = -INFINITY; }
    for (uint expert = 0; expert < experts; ++expert) {
        if (isnan(scores[expert])) { atomic_store_explicit(failure, 1u, memory_order_relaxed); return; }
    }
    for (uint expert = 0; expert < experts; ++expert) {
        float score = scores[expert];
        uint insertion = k;
        for (uint slot = 0; slot < k; ++slot) {
            float old = selected_probs[slot];
            uint old_id = selected_ids[slot];
            int score_key = qwen_graph_total_key(score);
            int old_key = qwen_graph_total_key(old);
            if (score_key > old_key || (score_key == old_key && expert < old_id)) { insertion = slot; break; }
        }
        if (insertion < k) {
            for (uint slot = k - 1; slot > insertion; --slot) {
                selected_probs[slot] = selected_probs[slot - 1];
                selected_ids[slot] = selected_ids[slot - 1];
            }
            selected_probs[insertion] = score;
            selected_ids[insertion] = expert;
        }
    }
    float maximum = selected_probs[0];
    float sum = 0.0f;
    for (uint slot = 0; slot < k; ++slot) {
        float probability = exp(selected_probs[slot] - maximum);
        selected_probs[slot] = probability;
        sum += probability;
    }
    if (!isfinite(sum) || sum <= 0.0f) { atomic_store_explicit(failure, 1u, memory_order_relaxed); return; }
    for (uint slot = 0; slot < k; ++slot) selected_probs[slot] /= sum;
}
