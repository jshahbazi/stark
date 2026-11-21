// SPDX-License-Identifier: Apache-2.0
// Original Stark implementation of the publisher's Inkling 2 Mini packed expert
// format and activation policy. Numerical controls and source revisions:
// docs/archive/experiments/2026-11-20-inkling-mini/NVFP4_PROBE.md.

struct InkExpertShape {
    uint width;
    uint intermediate;
    float denominator;
};

constant float ink_fp4_table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
constant float ink_fp4_midpoints[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};

inline float ink_sigmoid(float value) {
    const float exponential = exp(-abs(value));
    return value >= 0.0f ? 1.0f / (1.0f + exponential) : exponential / (1.0f + exponential);
}

kernel void ink_router_logits(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weights [[buffer(1)]],
    device float *logits [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    uint expert [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const device bfloat *weight = weights + expert * width;
    float dot = 0.0f;
    for (uint i = lane; i < width; i += 32) {
        dot = fma(float(input[i]), float(weight[i]), dot);
    }
    const float sum = simd_sum(dot);
    if (lane == 0) logits[expert] = sum;
}

kernel void ink_router_select(
    device const float *logits [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    device uint *ids [[buffer(2)]],
    device float *weights [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float choices[128];
    choices[lane] = ink_sigmoid(logits[lane]) + bias[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane != 0) return;

    float scores[8];
    float max_score = -INFINITY;
    for (uint route = 0; route < 8; ++route) {
        uint selected = 0;
        for (uint expert = 1; expert < 128; ++expert) {
            if (choices[expert] > choices[selected]) selected = expert;
        }
        choices[selected] = -INFINITY;
        ids[route] = selected;
        const float x = logits[selected];
        // log(sigmoid(x)), evaluated without a positive exponential argument.
        const float exponential = exp(-abs(x));
        const float one_plus = 1.0f + exponential;
        const float softplus = one_plus == 1.0f ? exponential : log(one_plus) * (exponential / (one_plus - 1.0f));
        scores[route] = min(x, 0.0f) - softplus;
        max_score = max(max_score, scores[route]);
    }
    float total = 0.0f;
    for (uint route = 0; route < 8; ++route) {
        scores[route] = exp(scores[route] - max_score);
        total += scores[route];
    }
    for (uint route = 0; route < 8; ++route) weights[route] = scores[route] * (8.0f / total);
}

// Positive finite E4M3 decode. Bit 7 carries the sign for signed FP8.
inline float ink_e4m3_positive(uint code) {
    const uint exponent = code >> 3;
    const uint mantissa = code & 7u;
    return exponent == 0
        ? float(mantissa) * 0x1p-9f
        : as_type<float>(((exponent + 120u) << 23) | (mantissa << 20));
}

inline float ink_round_e4m3(float positive) {
    // Nearest, ties to even. Cap before rounding to avoid the reserved NaN code.
    const float value = min(positive, 448.0f);
    const uint exponent = (as_type<uint>(value) >> 23) & 255u;
    const float step = exponent < 121u ? 0x1p-9f : as_type<float>((exponent - 3u) << 23);
    return rint(value / step) * step;
}

inline float ink_round_fp4(float value) {
    const float magnitude = abs(value);
    uint code = 0;
    for (uint i = 0; i < 7; ++i) {
        code += magnitude > ink_fp4_midpoints[i] || (magnitude == ink_fp4_midpoints[i] && (i & 1u));
    }
    return copysign(ink_fp4_table[code], value);
}

kernel void ink_nvfp4_activation(
    device const bfloat *input [[buffer(0)]],
    device float *inner [[buffer(1)]],
    device float *amax [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
#pragma clang fp contract(off) reassociate(off)
    threadgroup float maxima[8];
    const device bfloat *source = input + row * width;
    float local_max = 0.0f;
    for (uint i = lane; i < width; i += 256) local_max = max(local_max, abs(float(source[i])));
    const float simd_maximum = simd_max(local_max);
    if (simd_lane == 0) maxima[simd_group] = simd_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float row_max = simd_max(simd_lane < 8 ? maxima[simd_lane] : 0.0f);
    if (lane == 0) amax[row] = row_max;

    const float outer = row_max * (1.0f / 1536.0f);
    const float normalizer = min(1536.0f / row_max, FLT_MAX);
    for (uint block = lane; block < width / 16; block += 256) {
        const uint first = block * 16;
        float values[16];
        float maximum = 0.0f;
        for (uint i = 0; i < 16; ++i) {
            values[i] = float(source[first + i]);
            maximum = max(maximum, abs(values[i]));
        }

        const float scale6 = ink_round_e4m3(maximum * (normalizer * (1.0f / 6.0f)));
        const float scale4 = ink_round_e4m3(maximum * (normalizer * 0.25f));
        const float denom6 = scale6 * (1.0f / normalizer);
        const float denom4 = scale4 * (1.0f / normalizer);
        const float inverse6 = min(1.0f / denom6, FLT_MAX);
        const float inverse4 = min(1.0f / denom4, FLT_MAX);
        float candidate6[16];
        float candidate4[16];
        float error6 = 0.0f;
        float error4 = 0.0f;
        for (uint i = 0; i < 16; ++i) {
            candidate6[i] = ink_round_fp4(values[i] * inverse6) * scale6;
            candidate4[i] = ink_round_fp4(values[i] * inverse4) * scale4;
            error6 += abs(values[i] - candidate6[i] * outer);
            error4 += abs(values[i] - candidate4[i] * outer);
        }
        for (uint i = 0; i < 16; ++i) {
            inner[row * width + first + i] = error4 < error6 ? candidate4[i] : candidate6[i];
        }
    }
}

inline float ink_fp4_code(uint nibble) {
    return (nibble & 8u) ? -ink_fp4_table[nibble & 7u] : ink_fp4_table[nibble];
}

// A lane handles one original 16-channel block at a time. The publisher
// stores consecutive channels in a byte's low then high nibble, with the
// E4M3 block scales in a separate plane. Inputs restore their row scale
// after the complete SIMD reduction.
inline float ink_nvfp4_lane_dot(
    device const float *input, device const uchar *codes, device const uchar *scales,
    uint width, uint lane) {
    float sum = 0.0f;
    for (uint block = lane; block < width / 16; block += 32) {
        const uint2 packed = reinterpret_cast<device const uint2 *>(codes)[block];
        const float scale = ink_e4m3_positive(scales[block]);
        const device float *values = input + block * 16;
        float block_dot = 0.0f;
        for (uint byte = 0; byte < 4; ++byte) {
            const uint lo = packed.x >> (byte * 8);
            const uint hi = packed.y >> (byte * 8);
            block_dot = fma(ink_fp4_code(lo & 15u), values[byte * 2], block_dot);
            block_dot = fma(ink_fp4_code((lo >> 4) & 15u), values[byte * 2 + 1], block_dot);
            block_dot = fma(ink_fp4_code(hi & 15u), values[byte * 2 + 8], block_dot);
            block_dot = fma(ink_fp4_code((hi >> 4) & 15u), values[byte * 2 + 9], block_dot);
        }
        sum = fma(block_dot, scale, sum);
    }
    return sum;
}

kernel void ink_nvfp4_up(
    device const float *inner [[buffer(0)]],
    device const float *amax [[buffer(1)]],
    device const uint *ids [[buffer(2)]],
    device const float *route_weights [[buffer(3)]],
    device const uchar *codes [[buffer(4)]],
    device const uchar *scales [[buffer(5)]],
    device const float *expert_max [[buffer(6)]],
    device bfloat *middle [[buffer(7)]],
    constant InkExpertShape &shape [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint column = group.x * 4 + simd_group;
    const uint route = group.y;
    if (column >= shape.intermediate) return;
    const uint expert = ids[route];
    const uint row = (expert * shape.intermediate + column) * 2;
    const uint code_stride = shape.width / 2;
    const uint scale_stride = shape.width / 16;
    const float gate_dot = simd_sum(ink_nvfp4_lane_dot(
        inner, codes + row * code_stride, scales + row * scale_stride, shape.width, lane));
    const float up_dot = simd_sum(ink_nvfp4_lane_dot(
        inner, codes + (row + 1) * code_stride, scales + (row + 1) * scale_stride, shape.width, lane));
    if (lane == 0) {
        const float outer = (amax[0] * expert_max[expert]) / (1536.0f * shape.denominator);
        const float gate = float(bfloat(gate_dot * outer));
        const float up = float(bfloat(up_dot * outer));
        const float act = (4.0f * tanh(gate * 0.25f)) * ink_sigmoid(gate) * (25.0f * tanh(up / 25.0f));
        middle[route * shape.intermediate + column] = bfloat(act * route_weights[route]);
    }
}

kernel void ink_nvfp4_down(
    device const float *inner [[buffer(0)]],
    device const float *amax [[buffer(1)]],
    device const uint *ids [[buffer(2)]],
    device const uchar *codes [[buffer(3)]],
    device const uchar *scales [[buffer(4)]],
    device const float *expert_max [[buffer(5)]],
    device bfloat *down [[buffer(6)]],
    constant InkExpertShape &shape [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint column = group.x * 4 + simd_group;
    const uint route = group.y;
    if (column >= shape.width) return;
    const uint expert = ids[route];
    const uint row = expert * shape.width + column;
    const uint code_stride = shape.intermediate / 2;
    const uint scale_stride = shape.intermediate / 16;
    const float dot = simd_sum(ink_nvfp4_lane_dot(
        inner + route * shape.intermediate, codes + row * code_stride, scales + row * scale_stride,
        shape.intermediate, lane));
    if (lane == 0) {
        const float outer = (amax[route] * expert_max[expert]) / (1536.0f * shape.denominator);
        down[route * shape.width + column] = bfloat(dot * outer);
    }
}

kernel void ink_expert_sum(
    device const bfloat *down [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant uint &width [[buffer(2)]],
    uint column [[thread_position_in_grid]]) {
    if (column >= width) return;
    float sum = 0.0f;
    for (uint route = 0; route < 8; ++route) sum += float(down[route * width + column]);
    output[column] = bfloat(sum);
}
