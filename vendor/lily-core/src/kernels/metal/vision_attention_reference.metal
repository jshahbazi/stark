// Test-only source-order attention controls. Dot/softmax schedules follow
// PyTorch5c488690 and the main-engine vision bodies; BSD/MIT/Boost notices
// remain in scripts/licenses/distribution and vendor NOTICE/LICENSE.sleef.
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

static float reference_attention_round(float value) {
    const uint bits = as_type<uint>(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) return value;
    return as_type<float>((bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u);
}

kernel void vision_attention_order_qk(
    device const bfloat* qkv [[buffer(0)]], device bfloat* qk [[buffer(1)]],
    device bfloat* scaled [[buffer(2)]], constant uint& N [[buffer(3)]],
    constant float& scale [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    if (index >= 16 * N * N) return;
    const uint head = index / (N * N), query = (index / N) % N, key = index % N;
    const ulong qb = ulong(query) * 3456 + head * 72;
    const ulong kb = ulong(key) * 3456 + 1152 + head * 72;
    float4 sums[8] = {};
    for (uint start = 0; start < 64; start += 32) {
        for (uint reg = 0; reg < 8; ++reg) {
            for (uint lane = 0; lane < 4; ++lane) {
                const uint inner = start + reg * 4 + lane;
                sums[reg][lane] = fma(float(qkv[qb + inner]), float(qkv[kb + inner]), sums[reg][lane]);
            }
        }
    }
    for (uint offset = 4; offset != 0; offset >>= 1)
        for (uint reg = 0; reg < offset; ++reg) sums[reg] = sums[reg] + sums[reg + offset];
    float dot = (sums[0][0] + sums[0][1]) + (sums[0][2] + sums[0][3]);
    float4 tail = 0.0f;
    for (uint part = 0; part < 2; ++part) {
        for (uint lane = 0; lane < 4; ++lane) {
            const uint inner = 64 + part * 4 + lane;
            tail[lane] = fma(float(qkv[qb + inner]), float(qkv[kb + inner]), tail[lane]);
        }
    }
    dot += (tail[0] + tail[1]) + (tail[2] + tail[3]);
    const float rounded = reference_attention_round(dot);
    qk[index] = bfloat(rounded);
    scaled[index] = bfloat(reference_attention_round(rounded * scale));
}

kernel void vision_attention_order_softmax(
    device const bfloat* scaled [[buffer(0)]], device bfloat* probabilities [[buffer(1)]],
    constant uint& N [[buffer(2)]], uint row [[thread_position_in_grid]]) {
    if (row >= 16 * N) return;
    const ulong base = ulong(row) * N;
    float maximum = -INFINITY;
    for (uint column = 0; column < N; ++column) maximum = max(maximum, float(scaled[base + column]));
    float4 partial = 0.0f;
    for (uint column = 0; column < N; ++column) partial[column % 4] += vision_exp(float(scaled[base + column]) - maximum);
    const float total = (partial[0] + partial[2]) + (partial[1] + partial[3]);
    const float inverse = 1.0f / total;
    for (uint column = 0; column < N; ++column)
        probabilities[base + column] = bfloat(vision_exp(float(scaled[base + column]) - maximum) * inverse);
}

kernel void vision_attention_order_scale(
    device const bfloat* qk [[buffer(0)]], device bfloat* scaled [[buffer(1)]],
    constant uint& length [[buffer(2)]], constant float& scale [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < length) scaled[index] = bfloat(reference_attention_round(float(qk[index]) * scale));
}

kernel void vision_attention_order_pv(
    device const bfloat* qkv [[buffer(0)]], device const bfloat* probabilities [[buffer(1)]],
    device bfloat* output [[buffer(2)]], constant uint& N [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= N * 1152) return;
    const uint query = index / 1152, head = (index % 1152) / 72, channel = index % 72;
    const ulong base = (ulong(head) * N + query) * N;
    float4 sums = 0.0f;
    for (uint start = 0; start < N; start += 4) {
        for (uint lane = 0; lane < 4; ++lane) {
            const uint key = start + lane;
            const ulong value_index = ulong(key) * 3456 + 2304 + head * 72 + channel;
            sums[lane] = fma(float(probabilities[base + key]), float(qkv[value_index]), sums[lane]);
        }
    }
    output[index] = bfloat(((sums[0] + sums[1]) + sums[2]) + sums[3]);
}
