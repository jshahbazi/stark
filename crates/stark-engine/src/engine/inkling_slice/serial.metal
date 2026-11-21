// SPDX-License-Identifier: Apache-2.0
#include <metal_stdlib>
using namespace metal;

// Original Inkling 2 Mini tensor geometry. Source revisions are recorded in
// docs/INKLING_2_MINI_PLAN.md. Every packed and projected activation is BF16.
// Only the checkpoint's residual mixes, gates, reductions and SDPA stay F32.
constant uint ink_width = 2304;
constant uint ink_kv_width = 1024;
constant uint ink_table = 1024;
constant uint ink_key_tile = 32;

kernel void ink_matvec(
    device const bfloat *x [[buffer(0)]], device const bfloat *w [[buffer(1)]],
    device bfloat *y [[buffer(2)]], constant uint2 &shape [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row = group * 4 + simd;
    if (row >= shape.y) return;
    float4 accum = 0;
    device const bfloat *weights = w + row * shape.x;
    for (uint col = lane * 4; col < shape.x; col += 128) {
        accum = fma(float4(*reinterpret_cast<device const bfloat4 *>(x + col)),
                    float4(*reinterpret_cast<device const bfloat4 *>(weights + col)), accum);
    }
    float sum = simd_sum((accum.x + accum.y) + (accum.z + accum.w));
    if (lane == 0) y[row] = bfloat(sum);
}

// Each source is RMS-normalized, including the current FP32 residual partial.
// At a block boundary the pending attention output becomes the single partial
// while the old partial is appended to the frozen block list.
kernel void ink_mix(
    device const float *sources [[buffer(0)]], device const float *partial [[buffer(1)]],
    device const float *queries [[buffer(2)]], device float *result [[buffer(3)]],
    constant uint2 &args [[buffer(4)]], uint q [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float sums[8];
    threadgroup float dots[8];
    threadgroup float scales[5];
    threadgroup float scores[5];
    uint count = args.x + args.y;
    for (uint s = 0; s < count; ++s) {
        device const float *row = s == args.x ? partial : sources + s * ink_width;
        float norm = 0, dot = 0;
        for (uint i = tid; i < ink_width; i += 256) {
            float x = row[i];
            norm = fma(x, x, norm);
            if (count > 1) dot = fma(x, queries[q * ink_width + i], dot);
        }
        norm = simd_sum(norm);
        dot = simd_sum(dot);
        if (lane == 0) {
            sums[simd] = norm;
            dots[simd] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float sum = 0, dp = 0;
            for (uint j = 0; j < 8; ++j) {
                sum += sums[j];
                dp += dots[j];
            }
            scales[s] = rsqrt(sum / float(ink_width) + 1e-6f);
            scores[s] = dp * scales[s] / 48.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float peak = -INFINITY;
        for (uint s = 0; s < count; ++s) peak = max(peak, scores[s]);
        float sum = 0;
        for (uint s = 0; s < count; ++s) {
            scores[s] = exp(scores[s] - peak);
            sum += scores[s];
        }
        for (uint s = 0; s < count; ++s) scores[s] *= scales[s] / sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < ink_width; i += 256) {
        float out = 0;
        for (uint s = 0; s < count; ++s) {
            device const float *row = s == args.x ? partial : sources + s * ink_width;
            out = fma(scores[s], row[i], out);
        }
        result[q * ink_width + i] = out;
    }
}

kernel void ink_norm(
    device const float *x [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device bfloat *out [[buffer(2)]], constant uint &weighted [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float sums[8];
    threadgroup float scale;
    float sum = 0;
    x += row * ink_width;
    out += row * ink_width;
    for (uint i = tid; i < ink_width; i += 256) sum = fma(x[i], x[i], sum);
    sum = simd_sum(sum);
    if (lane == 0) sums[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        sum = 0;
        for (uint i = 0; i < 8; ++i) sum += sums[i];
        scale = rsqrt(sum / float(ink_width) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < ink_width; i += 256) {
        float value = x[i] * scale;
        out[i] = bfloat(weighted ? value * float(weight[i]) : value);
    }
}

kernel void ink_controller_expand(
    device const bfloat *compressed [[buffer(0)]], device const bfloat *gain [[buffer(1)]],
    device const bfloat *forget [[buffer(2)]], device const float *input [[buffer(3)]],
    device const float *message [[buffer(4)]], device float *mixed [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint index = group * 4 + sg;
    uint col = index * 64 + lane;
    float a = float(compressed[lane]), b = float(compressed[lane + 32]);
    float g = simd_sum(fma(a, float(gain[col]), b * float(gain[col + 32])));
    float f = simd_sum(fma(a, float(forget[col]), b * float(forget[col + 32])));
    if (lane == 0) mixed[index] = input[index] * (1.0f - tanh(f)) + message[index] * tanh(g);
}

kernel void ink_conv(
    device const bfloat *x [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device const bfloat *dynamic [[buffer(2)]], device bfloat *history [[buffer(3)]],
    device bfloat *output [[buffer(4)]], constant uint3 &p [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
    uint width = p.x, pos = p.y;
    if (i >= width) return;
    bool is_dynamic = p.z != 0;
    float current = float(x[i]), result = 0;
    for (uint tap = 0; tap < 4; ++tap) {
        uint lag = is_dynamic ? tap : 3 - tap;
        if (pos < lag) continue;
        float value = lag == 0 ? current : float(history[((pos - lag) % 3) * width + i]);
        float coef = is_dynamic ? float(weight[tap * width + i]) + float(dynamic[(i / 128) * 4 + tap])
                                : float(weight[i * 4 + tap]);
        result = fma(coef, value, result);
    }
    history[(pos % 3) * width + i] = x[i];
    output[i] = bfloat((result + current) * (is_dynamic ? 1.0f / 3.0f : 0.447213595499958f));
}

// Runs on complete SIMD groups. Each SIMD group owns a Q, K, V or relative row.
kernel void ink_heads_and_cache(
    device bfloat *q [[buffer(0)]], device const bfloat *k [[buffer(1)]],
    device const bfloat *v [[buffer(2)]], device bfloat *relative [[buffer(3)]],
    device const bfloat *q_scale [[buffer(4)]], device const bfloat *r_scale [[buffer(5)]],
    device bfloat *keys [[buffer(6)]], device bfloat *values [[buffer(7)]],
    constant uint3 &p [[buffer(8)]], uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = 4 * group + sg;
    if (row >= 64) return;
    bool is_q = row < 24, is_k = row >= 24 && row < 32, is_v = row >= 32 && row < 40;
    bool is_rel = row >= 40;
    uint head = is_q ? row : is_k ? row - 24 : is_v ? row - 32 : row - 40;
    uint width = is_rel ? 16 : 128;
    device const bfloat *input = is_q ? q : is_k ? k : is_v ? v : relative;
    device const bfloat *gain = is_q ? q_scale : r_scale;
    float4 vals = 0;
    float norm = 0;
    for (uint j = 0; j < width / 32 + uint(is_rel); ++j) {
        uint col = lane + j * 32;
        if (col < width) {
            uint index = head * width + col;
            float value = float(input[index]);
            if (is_q || is_rel) value *= float(gain[index]);
            norm = fma(value, value, norm);
            vals[j] = value;
        }
    }
    norm = simd_sum(norm);
    float limit = is_rel ? 8.0f : 22.62741699797f;
    float scale = is_v ? rsqrt(norm / 128.0f + 1e-6f) : min(1.0f, limit / (sqrt(norm) + 1e-6f));
    if ((is_q || is_rel) && p.z == 0 && p.x >= 8192) {
        scale *= 1.0f + 0.1f * log(float(p.x + 1) / 8192.0f);
    }
    uint offset = (p.x % p.y) * ink_kv_width;
    for (uint j = 0; j < width / 32 + uint(is_rel); ++j) {
        uint col = lane + j * 32;
        if (col < width) {
            uint index = head * width + col;
            bfloat result = bfloat(vals[j] * scale);
            if (is_q) q[index] = result;
            else if (is_k) keys[offset + index] = result;
            else if (is_v) values[offset + index] = result;
            else relative[index] = result;
        }
    }
}

kernel void ink_relative_table(
    device const bfloat *x [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device bfloat *out [[buffer(2)]], uint tid [[thread_position_in_grid]]
) {
    uint head = tid / ink_table, distance = tid % ink_table;
    float acc = 0;
    for (uint i = 0; i < 16; ++i) acc = fma(float(x[head * 16 + i]), float(weight[i * ink_table + distance]), acc);
    out[tid] = bfloat(acc);
}

// In prefill this needs a separate query-tiled path. Here there is one live
// query, and each group reduces 32 keys without expanding the BF16 history.
kernel void ink_attention_parts(
    device const bfloat *q [[buffer(0)]], device const bfloat *keys [[buffer(1)]],
    device const bfloat *values [[buffer(2)]], device const bfloat *table [[buffer(3)]],
    device const bfloat *sink [[buffer(4)]], device float *parts [[buffer(5)]],
    constant uint3 &p [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint head = group.y, kv = head / 3;
    uint visible = min(p.x + 1, p.z == 0 ? p.y : p.z);
    uint first = p.x + 1 - visible;
    uint tile = group.x * ink_key_tile;
    uint count = min(ink_key_tile, visible - tile);
    threadgroup float probabilities[ink_key_tile];
    threadgroup float peak;
    threadgroup float denom;
    float4 query = float4(*reinterpret_cast<device const bfloat4 *>(q + head * 128 + lane * 4));
    for (uint j = sg; j < count; j += 4) {
        uint pos = first + tile + j;
        uint base = (pos % p.y) * ink_kv_width + kv * 128;
        float4 key = float4(*reinterpret_cast<device const bfloat4 *>(keys + base + lane * 4));
        float score = simd_sum(dot(query, key)) * 0.08838834764831845f;
        uint distance = p.x - pos;
        if (distance < ink_table) score += float(table[head * ink_table + distance]);
        if (lane == 0) probabilities[j] = score - float(sink[head]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float score = lane < count ? probabilities[lane] : -INFINITY;
        float largest = simd_max(score);
        float prob = exp(score - largest);
        if (lane < count) probabilities[lane] = prob;
        float total = simd_sum(prob);
        if (lane == 0) {
            peak = largest;
            denom = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0;
    for (uint j = 0; j < count; ++j) {
        uint base = ((first + tile + j) % p.y) * ink_kv_width + kv * 128;
        sum = fma(probabilities[j], float(values[base + tid]), sum);
    }
    uint part = (head * ((p.y + ink_key_tile - 1) / ink_key_tile) + group.x) * 130;
    parts[part + tid] = sum;
    if (tid == 0) {
        parts[part + 128] = peak;
        parts[part + 129] = denom;
    }
}

kernel void ink_attention_gate(
    device const float *parts [[buffer(0)]], device const bfloat *gate [[buffer(1)]],
    device bfloat *result [[buffer(2)]], constant uint2 &p [[buffer(3)]],
    uint head [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float scales[136];
    threadgroup float inverse;
    device const float *head_parts = parts + head * p.y * 130;
    if (tid == 0) {
        float peak = 0; // zero-logit sink
        for (uint i = 0; i < p.x; ++i) peak = max(peak, head_parts[i * 130 + 128]);
        float denom = exp(-peak);
        for (uint i = 0; i < p.x; ++i) {
            float weight = exp(head_parts[i * 130 + 128] - peak);
            scales[i] = weight;
            denom = fma(weight, head_parts[i * 130 + 129], denom);
        }
        inverse = 1.0f / denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0;
    for (uint i = 0; i < p.x; ++i) sum = fma(scales[i], head_parts[i * 130 + tid], sum);
    uint index = head * 128 + tid;
    result[index] = bfloat(sum * inverse / (1.0f + exp(-float(gate[index]))));
}

kernel void ink_after_attention(
    device float *partial [[buffer(0)]], device const bfloat *attn [[buffer(1)]],
    constant uint &boundary [[buffer(2)]], uint i [[thread_position_in_grid]]
) {
    if (i < ink_width) partial[i] = (boundary ? 0.0f : partial[i]) + float(attn[i]);
}

kernel void ink_after_experts(
    device float *partial [[buffer(0)]], device const bfloat *ffn [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < ink_width) partial[i] = float(bfloat(partial[i] + float(ffn[i])));
}

kernel void ink_copy(device const float *src [[buffer(0)]], device float *dst [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    dst[i] = src[i];
}
