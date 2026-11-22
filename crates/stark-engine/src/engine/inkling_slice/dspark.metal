// SPDX-License-Identifier: MIT
// Copyright (c) Ollama
// The folded vocabulary tile adapts the Metal tensor layout in Ollama
// 6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8, mlx/bf16_matmul.go.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

constant uint width = 2304;
constant uint rank = 256;
constant uint heads = 24;
constant uint head_dim = 128;
constant uint kv_width = 1024;
constant uint rows = 8;
constant uint shards = 32;
constant uint part_width = head_dim + 2;

kernel void ink_draft_mix(
    device const bfloat *features [[buffer(0)]], device const bfloat *weights [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint &layer [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
#pragma clang fp contract(off) reassociate(off)
    out[i] = (float(features[i]) * float(weights[layer * 4])
        + float(features[width + i]) * float(weights[layer * 4 + 1]))
        + (float(features[2 * width + i]) * float(weights[layer * 4 + 2])
        + float(features[3 * width + i]) * float(weights[layer * 4 + 3]));
}

// Q/K normalization and nontraditional RoPE keep the source's BF16 rounding.
kernel void ink_draft_rope(
    device const bfloat *in [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device bfloat *out [[buffer(2)]], constant uint3 &p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off) reassociate(off)
    const uint offset = group.y * p.x + group.x * head_dim;
    const uint square_index = 4 * lane;
    float norm = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        const float value = float(in[offset + square_index + i]);
        norm += value * value;
    }
    const float scale = metal::precise::rsqrt(simd_sum(norm) / float(head_dim) + 1.0e-6f);
    const uint position = p.y + group.y;
    const uint dest = (p.z == 0 ? group.y : position % p.z) * p.x + group.x * head_dim;
    for (uint i = lane; i < head_dim / 2; i += 32) {
        const float theta = float(position) * exp(-log(1000000.0f) * float(i) / 64.0f);
        const float cosine = cos(theta);
        const float sine = sin(theta);
        const uint j = i + head_dim / 2;
        const float first = float(bfloat(float(weight[i]) * (float(in[offset + i]) * scale)));
        const float second = float(bfloat(float(weight[j]) * (float(in[offset + j]) * scale)));
        out[dest + i] = bfloat(first * cosine - second * sine);
        out[dest + j] = bfloat(first * sine + second * cosine);
    }
}

kernel void ink_draft_copy_value(
    device const bfloat *in [[buffer(0)]], device bfloat *cache [[buffer(1)]],
    constant uint &slot [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    cache[slot * kv_width + i] = in[i];
}

kernel void ink_draft_block(
    device const float *anchor [[buffer(0)]], device const bfloat *mask [[buffer(1)]],
    device float *out [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
    const uint row = i.y;
    const uint column = i.x;
    out[row * width + column] = row == 0 ? anchor[column] : float(mask[column]);
}

kernel void ink_draft_round(
    device const float *in [[buffer(0)]], device bfloat *out [[buffer(1)]],
    constant uint &elements [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < elements) out[i] = bfloat(in[i]);
}

kernel void ink_draft_fold_input(
    device const bfloat *in [[buffer(0)]], device bfloat *out [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    out[i] = in[(i % width) * rank + i / width];
}

kernel void ink_draft_fold(
    device const bfloat *latent [[buffer(0)]], device const bfloat *target [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint &columns [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]]) {
    constexpr int latent_tile = 16, vocab_tile = 32;
    using dims = dextents<int, 2>;
    using strides = array<int, 2>;
    const uint first_token = group.x * vocab_tile;
    const uint first_latent = group.y * latent_tile;
    tensor<device bfloat, dims, tensor_inline> input(
        const_cast<device bfloat *>(latent) + size_t(first_latent) * width, dims(width, latent_tile),
        strides{1, width});
    tensor<device bfloat, dims, tensor_inline> weights(
        const_cast<device bfloat *>(target) + size_t(first_token) * width, dims(width, vocab_tile),
        strides{1, width});
    tensor<device float, dims, tensor_inline> output(
        out + size_t(first_latent) * columns + first_token, dims(vocab_tile, latent_tile), strides{1, int(columns)});
    constexpr mpp::tensor_ops::matmul2d_descriptor tile(
        latent_tile, vocab_tile, int(dynamic_extent), false, true, true);
    mpp::tensor_ops::matmul2d<tile, execution_simdgroup> product;
    product.run(input, weights, output);
}

kernel void ink_draft_fold_output(
    device const float *in [[buffer(0)]], device bfloat *out [[buffer(1)]],
    constant uint &columns [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float tile[32][33];
    const uint token = 32 * group.x;
    const uint latent = 32 * group.y;
    const uint x = tid % 32;
    const uint y = tid / 32;
    for (uint row = y; row < 32; row += 8)
        tile[row][x] = in[(latent + row) * columns + token + x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = y; row < 32; row += 8)
        out[(token + row) * rank + latent + x] = bfloat(tile[x][row]);
}

kernel void ink_draft_fold_bias(
    device const bfloat *latent [[buffer(0)]], device const bfloat *target [[buffer(1)]],
    device bfloat *out [[buffer(2)]], uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint token = 4 * group + simd;
    target += size_t(token) * width;
    float4 dot = 0;
    for (uint i = lane * 4; i < width; i += 128) {
        const float4 a = float4(*reinterpret_cast<device const bfloat4 *>(latent + i));
        const float4 b = float4(*reinterpret_cast<device const bfloat4 *>(target + i));
        dot = fma(a, b, dot);
    }
    const float result = simd_sum((dot.x + dot.y) + (dot.z + dot.w));
    if (lane == 0) out[token] = bfloat(result);
}

// The eight block rows see the full block and either the full committed
// context or the window measured from the anchor. Proposals never write KV.
kernel void ink_draft_attend_parts(
    device const bfloat *queries [[buffer(0)]], device const bfloat *keys [[buffer(1)]],
    device const bfloat *values [[buffer(2)]], device const bfloat *block_keys [[buffer(3)]],
    device const bfloat *block_values [[buffer(4)]], device float *parts [[buffer(5)]],
    constant uint3 &p [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint head = group.y % heads;
    const uint row = group.y / heads;
    const uint offset = head_dim * (head / 3) + lane * 4;
    const float4 q = float4(*reinterpret_cast<device const bfloat4 *>(
        queries + (row * heads + head) * head_dim + lane * 4)) * 0.08838834765f;
    float4 output = 0;
    float high = -INFINITY;
    float total = 0.0f;
    for (uint i = p.x + group.x; i < p.y + rows; i += shards) {
        const bool committed = i < p.y;
        const uint index = (committed ? i % p.z : i - p.y) * kv_width + offset;
        const device bfloat *key = (committed ? keys : block_keys) + index;
        const device bfloat *value = (committed ? values : block_values) + index;
        const float score = simd_sum(dot(q, float4(*reinterpret_cast<device const bfloat4 *>(key))));
        const float next_high = max(high, score);
        const float keep = exp(high - next_high);
        const float add = exp(score - next_high);
        output = output * keep + float4(*reinterpret_cast<device const bfloat4 *>(value)) * add;
        total = total * keep + add;
        high = next_high;
    }
    device float *dest = parts + (group.y * shards + group.x) * part_width;
    for (uint i = 0; i < 4; ++i) dest[lane * 4 + i] = output[i];
    if (lane == 0) {
        dest[head_dim] = high;
        dest[head_dim + 1] = total;
    }
}

kernel void ink_draft_attend_join(
    device const float *parts [[buffer(0)]], device bfloat *out [[buffer(1)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float weights[shards];
    threadgroup float inverse;
    const device float *row = parts + group * shards * part_width;
    if (tid < shards) {
        const device float *part = row + tid * part_width;
        const float high = simd_max(part[head_dim]);
        const float weight = exp(part[head_dim] - high);
        weights[tid] = weight;
        const float sum = simd_sum(weight * part[head_dim + 1]);
        if (tid == 0) inverse = 1.0f / sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float value = 0;
    for (uint i = 0; i < shards; ++i) value += row[i * part_width + tid] * weights[i];
    out[group * head_dim + tid] = bfloat(value * inverse);
}

kernel void ink_draft_residual(
    device float *residual [[buffer(0)]], device const bfloat *update [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    residual[i] = float(bfloat(residual[i])) + float(update[i]);
}

kernel void ink_draft_swiglu(
    device const bfloat *gate [[buffer(0)]], device const bfloat *up [[buffer(1)]],
    device bfloat *out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    const float value = float(gate[i]);
    out[i] = bfloat((value / (1.0f + exp(-value))) * float(up[i]));
}

kernel void ink_draft_rnn_input(
    device const bfloat *previous [[buffer(0)]], device const bfloat *hidden [[buffer(1)]],
    device const bfloat *feature [[buffer(2)]], device bfloat *out [[buffer(3)]],
    constant uint &anchor [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    // One bounded proposal starts with a fresh, zero recurrent state.
    if (i < rank) out[i] = bfloat(0.0f);
    else if (i < 2 * rank) out[i] = previous[anchor * rank + i - rank];
    else if (i < 2 * rank + width) out[i] = hidden[i - 2 * rank];
    else out[i] = feature[i - 2 * rank - width];
}

kernel void ink_draft_rnn_latent(
    device const float *gates [[buffer(0)]], device const bfloat *bias [[buffer(1)]],
    device bfloat *out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    const uint gate = 2 * rank + i;
    out[i] = bfloat(metal::precise::tanh(float(bfloat(gates[gate] + float(bias[gate])))));
}

#ifdef INK_DRAFT_MARKOV_BLOCK
kernel void ink_draft_markov_input(
    device const bfloat *previous [[buffer(0)]], device const bfloat *hidden [[buffer(1)]],
    device const bfloat *feature [[buffer(2)]], device bfloat *out [[buffer(3)]],
    device const bfloat *state [[buffer(4)]], device const uint *winners [[buffer(5)]],
    constant uint3 &step [[buffer(6)]], uint i [[thread_position_in_grid]]) {
    const uint token = step.x == 0 ? step.y : winners[step.x - 1];
    if (i < rank) out[i] = step.x == 0 ? bfloat(0.0f) : state[i];
    else if (i < 2 * rank) out[i] = previous[(token < step.z ? token : 0u) * rank + i - rank];
    else if (i < 2 * rank + width) out[i] = hidden[i - 2 * rank];
    else out[i] = feature[i - 2 * rank - width];
}

kernel void ink_draft_markov_state(
    device const float *gates [[buffer(0)]], device const bfloat *bias [[buffer(1)]],
    device bfloat *latent [[buffer(2)]], device bfloat *state [[buffer(3)]],
    constant uint &row [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    const float keep_logit = float(bfloat(gates[i] + float(bias[i])));
    const float keep = float(bfloat(1.0f / (1.0f + metal::precise::exp(-keep_logit))));
    const float update = float(bfloat(metal::precise::tanh(float(bfloat(
        gates[rank + i] + float(bias[rank + i]))))));
    const float retain = row == 0 ? 0.0f : float(bfloat(keep * float(state[i])));
    const float replace = float(bfloat(float(bfloat(1.0f - keep)) * update));
    state[i] = bfloat(retain + replace);
    const uint gate = 2 * rank + i;
    latent[i] = bfloat(metal::precise::tanh(float(bfloat(gates[gate] + float(bias[gate])))));
}

#endif

inline bool draft_better(float score, uint token, float best, uint best_token) {
    return score > best || (score == best && token < best_token);
}

inline float draft_rank_score(
    device const float *base, device const bfloat *folded, device const bfloat *bias,
    device const bfloat *latent, uint token, uint lane) {
    float sum = 0.0f;
    for (uint i = lane; i < rank; i += 32)
        sum = fma(float(folded[token * rank + i]), float(latent[i]), sum);
    const float correction = simd_sum(sum) + float(bias[token]);
    const float corrected = float(bfloat(float(bfloat(base[token])) + float(bfloat(correction))));
    return float(bfloat(30.0f * metal::precise::tanh(corrected / 30.0f)));
}

#ifdef INK_DRAFT_EXPORT_LOGITS
kernel void ink_draft_export_scores(
    device const float *base [[buffer(0)]], device const bfloat *folded [[buffer(1)]],
    device const bfloat *bias [[buffer(2)]], device const bfloat *latent [[buffer(3)]],
    device float *scores [[buffer(4)]], constant uint &vocab [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint token = group * 4 + simd;
    if (token < vocab) {
        const float score = draft_rank_score(base, folded, bias, latent, token, lane);
        if (lane == 0)
            scores[token] = score;
    }
}
#endif

kernel void ink_draft_rank(
    device const float *base [[buffer(0)]], device const bfloat *folded [[buffer(1)]],
    device const bfloat *bias [[buffer(2)]], device const bfloat *latent [[buffer(3)]],
    device uint2 *winners [[buffer(4)]], constant uint &vocab [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial_scores[4];
    threadgroup uint partial_tokens[4];
    const uint token = group * 4 + simd;
    float score = -INFINITY;
    if (token < vocab)
        score = draft_rank_score(base, folded, bias, latent, token, lane);
    if (lane == 0) {
        partial_scores[simd] = isfinite(score) ? score : -INFINITY;
        partial_tokens[simd] = isfinite(score) ? token : UINT_MAX;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float best = partial_scores[0];
        uint winner = partial_tokens[0];
        for (uint i = 1; i < 4; ++i)
            if (draft_better(partial_scores[i], partial_tokens[i], best, winner)) {
                best = partial_scores[i];
                winner = partial_tokens[i];
            }
        winners[group] = uint2(as_type<uint>(best), winner);
    }
}

#ifdef INK_DRAFT_SHORTLIST
// Each SIMD group selects two candidates using the base head and reranks
// them with the same BF16 recurrence. The complete target still verifies
// every proposal and applies its usual sampling policy.
kernel void ink_draft_shortlist(
    device const float *base [[buffer(0)]], device const bfloat *folded [[buffer(1)]],
    device const bfloat *bias [[buffer(2)]], device const bfloat *latent [[buffer(3)]],
    device uint2 *winners [[buffer(4)]], constant uint &vocab [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint token = group * 128 + simd * 32 + lane;
    const float value = token < vocab ? float(bfloat(base[token])) : -INFINITY;
    const bool valid = token < vocab && !isnan(value);
    const float first_max = simd_max(valid ? value : -INFINITY);
    const uint first = simd_min(valid && value == first_max ? token : UINT_MAX);
    const bool remaining = valid && token != first;
    const float second_max = simd_max(remaining ? value : -INFINITY);
    const uint second = simd_min(remaining && value == second_max ? token : UINT_MAX);

    float best = -INFINITY;
    uint winner = UINT_MAX;
    if (first < vocab) {
        const float score = draft_rank_score(base, folded, bias, latent, first, lane);
        if (isfinite(score)) {
            best = score;
            winner = first;
        }
    }
    if (second < vocab) {
        const float score = draft_rank_score(base, folded, bias, latent, second, lane);
        if (isfinite(score) && draft_better(score, second, best, winner)) {
            best = score;
            winner = second;
        }
    }
    if (lane == 0) winners[group * 4 + simd] = uint2(as_type<uint>(best), winner);
}
#endif

kernel void ink_draft_winner(
    device const uint2 *groups [[buffer(0)]], device uint *out [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float scores[256];
    threadgroup uint tokens[256];
    float best = -INFINITY;
    uint winner = UINT_MAX;
    for (uint i = tid; i < count; i += 256) {
        const uint2 entry = groups[i];
        const float score = as_type<float>(entry.x);
        if (draft_better(score, entry.y, best, winner)) {
            best = score;
            winner = entry.y;
        }
    }
    scores[tid] = best;
    tokens[tid] = winner;
    for (uint step = 128; step > 0; step /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < step && draft_better(scores[tid + step], tokens[tid + step], scores[tid], tokens[tid])) {
            scores[tid] = scores[tid + step];
            tokens[tid] = tokens[tid + step];
        }
    }
    if (tid == 0) out[0] = tokens[0];
}
