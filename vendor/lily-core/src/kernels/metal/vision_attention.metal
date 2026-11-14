// Vision-only streaming source-order attention. PyTorch5c488690 schedules
// and SLEEF5a1d179d exp; notices retained in NOTICE/LICENSE.sleef.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)


// Include after the caller selects safe FP32 math and disables contraction.
// SLEEF expf_u10, revision 5a1d179df9cf652951b59010a2d2075372d67f68.
// Copyright Naoki Shibata and contributors, Boost Software License 1.0;
// see scripts/licenses/distribution/sleef-LICENSE.txt.
static float vision_exp(float x) {
  if (x < -104.0f) return 0.0f;
  if (x > 100.0f) return INFINITY;
  const float q = rint(x * 1.4426950408889634f);
  float s = fma(q, -0.693145751953125f, x);
  s = fma(q, -1.428606765330187e-6f, s);
  float u = 0.00019852761761285365f;
  u = fma(u, s, 0.0013930435525253415f);
  u = fma(u, s, 0.008333360776305199f);
  u = fma(u, s, 0.041666485369205475f);
  u = fma(u, s, 0.1666666716337204f);
  u = fma(u, s, 0.5f);
  u = 1.0f + fma(s * s, u, s);
  const int first_exponent = int(q) >> 1;
  const float first = as_type<float>(uint(first_exponent + 127) << 23);
  const float second = as_type<float>(uint(int(q) - first_exponent + 127) << 23);
  return (u * first) * second;
}

static inline float source_attention_round(float x) {
    const uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) == 0x7f800000u) return x;
    return as_type<float>((bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u);
}

// One SIMD lane per source accumulator. D72 is aligned64 plus the separate
// eight-term/four-lane tail, matching the independently measured scalar body.
static inline float2 source_attention_score(
    device const bfloat* qkv, uint query, uint key, uint head, float scale, uint lane) {
    const ulong qb = ulong(query) * 3456 + head * 72;
    const ulong kb = ulong(key) * 3456 + 1152 + head * 72;
    float sum = fma(float(qkv[qb + lane]), float(qkv[kb + lane]), 0.0f);
    sum = fma(float(qkv[qb + lane + 32]), float(qkv[kb + lane + 32]), sum);
    for (uint offset = 16; offset >= 4; offset >>= 1) {
        const float other = simd_shuffle_down(sum, offset);
        if (lane < offset) sum = sum + other;
    }
    const float pair = sum + simd_shuffle_down(sum, 1);
    const float other_pair = simd_shuffle_down(pair, 2);
    float tail = 0.0f;
    if (lane < 4) {
        tail = fma(float(qkv[qb + 64 + lane]), float(qkv[kb + 64 + lane]), tail);
        tail = fma(float(qkv[qb + 68 + lane]), float(qkv[kb + 68 + lane]), tail);
    }
    const float tail_pair = tail + simd_shuffle_down(tail, 1);
    const float other_tail_pair = simd_shuffle_down(tail_pair, 2);
    float rounded = 0.0f, scaled = 0.0f;
    if (lane == 0) {
        const float dot = (pair + other_pair) + (tail_pair + other_tail_pair);
        rounded = source_attention_round(dot);
        scaled = source_attention_round(rounded * scale);
    }
    return float2(simd_shuffle(rounded, 0), simd_shuffle(scaled, 0));
}

template <bool TRACE>
static void source_attention_body(
    device const bfloat* qkv, device bfloat* out, device float* stages,
    uint N, float scale, uint2 group, uint sg, uint lane) {
    const uint query = group.x * 4 + sg, head = group.y;
    if (query >= N) return;
    float maximum = -INFINITY;
    for (uint key = 0; key < N; ++key) {
        const float2 score = source_attention_score(qkv, query, key, head, scale, lane);
        maximum = max(maximum, score.y);
        if constexpr (TRACE) {
            if (lane == 0) {
                const ulong index = (ulong(head) * N + query) * N + key;
                const ulong plane = ulong(16) * N * N;
                stages[index] = score.x;
                stages[plane + index] = score.y;
            }
        }
    }
    // Keep each source modulo-four sum across the complete key range, then
    // perform the pinned crosswise fold and reciprocal. No per-tile sums.
    float partial = 0.0f;
    for (uint key = 0; key < N; ++key) {
        const float score = source_attention_score(qkv, query, key, head, scale, lane).y;
        float exponential = 0.0f;
        if (lane == 0) exponential = vision_exp(score - maximum);
        exponential = simd_shuffle(exponential, 0);
        if (lane == key % 4) partial = partial + exponential;
    }
    const float cross = partial + simd_shuffle_down(partial, 2);
    const float other_cross = simd_shuffle(cross, 1);
    float inverse = 0.0f;
    if (lane == 0) inverse = 1.0f / (cross + other_cross);
    inverse = simd_shuffle(inverse, 0);
    float4 products[3] = {};
    for (uint key = 0; key < N; ++key) {
        const float score = source_attention_score(qkv, query, key, head, scale, lane).y;
        float probability = 0.0f;
        if (lane == 0) probability = source_attention_round(vision_exp(score - maximum) * inverse);
        probability = simd_shuffle(probability, 0);
        if constexpr (TRACE) {
            if (lane == 0) {
                const ulong index = (ulong(head) * N + query) * N + key;
                stages[2 * ulong(16) * N * N + index] = probability;
            }
        }
        for (uint part = 0; part < 3; ++part) {
            const uint channel = lane + 32 * part;
            if (channel < 72) {
                const float value = float(qkv[ulong(key) * 3456 + 2304 + head * 72 + channel]);
                products[part][key % 4] = fma(probability, value, products[part][key % 4]);
            }
        }
    }
    for (uint part = 0; part < 3; ++part) {
        const uint channel = lane + 32 * part;
        if (channel < 72) {
            const float4 sums = products[part];
            out[ulong(query) * 1152 + head * 72 + channel] = bfloat(((sums[0] + sums[1]) + sums[2]) + sums[3]);
        }
    }
}

kernel void vision_attention_source_bf16(
    device const bfloat* qkv [[buffer(0)]], device bfloat* out [[buffer(1)]],
    constant uint& N [[buffer(2)]], constant float& scale [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    source_attention_body<false>(qkv, out, nullptr, N, scale, group, sg, lane);
}
