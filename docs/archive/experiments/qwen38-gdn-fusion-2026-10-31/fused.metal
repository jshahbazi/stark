// RESEARCH ONLY. No production dispatch selects this source.
// Recurrence plus BF16 raw boundary plus original output RMS norm/sigmoid gate.
// Arithmetic fragments copied from the source hashes below; only per-head
// ownership and communication change. Convolution remains unchanged.
//
// ABI: buffers0..5=convolved QKV,a,b,A_log,dt_bias,F32 state; buffer6=
// diagnostic BF16 raw (unused unless RESEARCH_CAPTURE_RAW=1); buffers7..9=
// BF16 z,F32 norm weights,BF16 gated output; constants10..11=Kscale,eps.
// Dispatch48groups x256 or512threads matching the named entry. One group owns
// all128 rows of one value head. 256threads keep16rows/thread (64F32 state
// scalars), 512threads keep8rows/thread (32scalars). Norm uses SG0 exactly.
// State, raw (diagnostic only), gated output and inputs must not alias.
//
// GDN decode recurrence lineage: MLX-LM gated_delta.py, Copyright (c) 2023
// Apple Inc., MIT License. ValueKey activation boundaries: mlx-serve 26.10.3,
// revision 046b647686419df06c6c7132fb1d15d76185c3e5, src/gdn_decode.zig
// K1_SOURCE, Copyright (c) 2026 David Dalcu, MIT License. The K1 helper
// functions below are self-contained copies from the separately named helper.
// The
// convolve, recurrence, stock log1p, and norm fragments below are adapted from
// vendor/lily-core/src/kernels/metal/gdn.metal; see vendor/lily-core/NOTICE
// and vendor/lily-core/LICENSE.mlx-serve for retained upstream notices.
// Exact upstream inputs used for the helper fragments:
//   official_gdn_k1_helper.metal SHA-256
//   6cfe8926c91f02bf14dbca0c4e68a36f8b27cc381bcb9b23e9ff2185e226b78e
//   gdn.metal SHA-256
//   78ca01774e89e1fea05190234171704cc6a2bca4b24415cee2aa45cdd5ac6844
//
// MIT License
// Copyright (c) 2023 Apple Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// MIT License
// Copyright (c) 2026 David Dalcu
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Self-contained copies of the three production-used K1 helper functions.
// Source: official_gdn_k1_helper.metal at the SHA-256 recorded above.
static inline float official_gdn_k1_round_bf16(float x) {
    const uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) == 0x7f800000u) return x;
    const uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return as_type<float>(rounded & 0xffff0000u);
}

template <uint MODE>
static inline float official_gdn_k1_sigmoid_f32(float x) {
    float ex;
    if constexpr (MODE == 1) ex = metal::precise::exp(metal::abs(x));
    else if constexpr (MODE == 2) ex = metal::fast::exp(metal::abs(x));
    else ex = metal::exp(metal::abs(x));
    const float e = official_gdn_k1_round_bf16(ex);
    const float d = official_gdn_k1_round_bf16(1.0f + e);
    const float y = official_gdn_k1_round_bf16(1.0f / d);
    return x < 0.0f ? y : official_gdn_k1_round_bf16(1.0f - y);
}

template <uint MODE>
static inline float official_gdn_k1_silu_f32(float acc) {
    const float x = official_gdn_k1_round_bf16(acc);
    const float sig = official_gdn_k1_sigmoid_f32<MODE>(x);
    return official_gdn_k1_round_bf16(x * sig);
}


#ifndef RESEARCH_CAPTURE_RAW
#define RESEARCH_CAPTURE_RAW 0
#endif

static inline bfloat official_gdn_scaled_rms(float value, float inv, bfloat scale) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const bfloat normalized = bfloat(value * inv);
    return bfloat(float(normalized) * float(scale));
}

inline float gdn_value_key_stock_log1p(float x) {
    float xp1 = 1.0f + x;
    if (xp1 == metal::numeric_limits<float>::max()) { return metal::numeric_limits<float>::max(); }
    if (xp1 == 1.0f) { return x; }
    return x * (metal::log(xp1) / (xp1 - 1.0f));
}

template <uint NT>
static inline void research_gdn_fused_body(
    device const bfloat* qkv,
    device const bfloat* a_in,
    device const bfloat* b_in,
    device const float* A_log,
    device const bfloat* dt_bias,
    device float* state,
    device bfloat* y,
    device const bfloat* z,
    device const float* norm_w,
    device bfloat* gated,
    constant float& k_scale_f32,
    constant float& eps,
    uint tg, uint lane, uint sg,
    threadgroup float* qs, threadgroup float* ks, threadgroup float* vs,
    threadgroup float* gb, threadgroup bfloat* raw_head) {
    using T = bfloat;
    constexpr int HK = 16, HV = 48, DK = 128, DV = 128;
    constexpr int SPLIT = 1, NSG = NT / 32;
    constexpr int RB = DV / SPLIT, R = RB / NSG, GRP = HV / HK;
    const T q_scale = T(1.0f / float(DK));
    const T k_scale = T(k_scale_f32);
    uint hv = tg / SPLIT;
    uint part = tg % SPLIT;
    uint hk = hv / GRP;
    uint row0 = part * RB + sg * R;
    float st[R][4];
    for (int j = 0; j < R; ++j) {
        uint base = (hv * DV + row0 + j) * DK + lane * 4;
        for (int i = 0; i < 4; ++i) st[j][i] = state[base + i];
    }
    // The native qualified convolution produces exactly the BF16 act[i]
    // boundary from stock K1. Never update convolution history in V tiles.
    if (sg < 3) {
        uint cb = sg == 0 ? hk * DK : (sg == 1 ? HK * DK + hk * DK : 2 * HK * DK + hv * DV);
        T act[4];
        float sumsq = 0.0f;
        for (int i = 0; i < 4; ++i) {
            act[i] = qkv[cb + lane * 4 + i];
            float v = float(act[i]);
            sumsq += v * v;
        }
        if (sg < 2) {
            sumsq = simd_sum(sumsq);
            float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f);
            const T scale = sg == 0 ? q_scale : k_scale;
            threadgroup float* dst = sg == 0 ? qs : ks;
            // Materialize the normalized BF16 row and separate BF16 scalar
            // product, retaining the source's measured rounding boundaries.
            for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = float(official_gdn_scaled_rms(float(act[i]), inv, scale));
        } else {
            for (int i = 0; i < 4; ++i) vs[lane * 4 + i] = float(act[i]);
        }
    }
    if (sg == (NSG > 3 ? 3 : 0) && lane == 31) {
        const T bv = b_in[hv];
        gb[1] = official_gdn_k1_sigmoid_f32<0>(float(bv));
        const float apd = float(a_in[hv]) + float(dt_bias[hv]);
        float sp = gdn_value_key_stock_log1p(metal::precise::exp(apd));
        float ea = metal::precise::exp(A_log[hv]);
        gb[0] = metal::precise::exp(-(ea * sp));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float kk[4], qq[4];
    for (int i = 0; i < 4; ++i) { kk[i] = ks[lane * 4 + i]; qq[i] = qs[lane * 4 + i]; }
    const float g = gb[0], beta = gb[1];
    for (int j = 0; j < R; ++j) {
        uint dv = row0 + j;
        float kv_mem = 0.0f;
        for (int i = 0; i < 4; ++i) { st[j][i] = st[j][i] * g; kv_mem += st[j][i] * kk[i]; }
        kv_mem = simd_sum(kv_mem);
        float delta = (vs[dv] - kv_mem) * beta;
        float out = 0.0f;
        for (int i = 0; i < 4; ++i) { st[j][i] = st[j][i] + kk[i] * delta; out += st[j][i] * qq[i]; }
        out = simd_sum(out);
        uint base = (hv * DV + dv) * DK + lane * 4;
        for (int i = 0; i < 4; ++i) state[base + i] = st[j][i];
        if (lane == 0) {
            const bfloat rounded = static_cast<T>(out);
            raw_head[dv] = rounded;
#if RESEARCH_CAPTURE_RAW
            y[hv * DV + dv] = rounded;
#endif
        }
    }
    // All 128 BF16 readouts must be visible to the original one-SIMD norm tree.
    // No device raw-output reload or cross-threadgroup communication is used.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const uint base = hv * 128 + lane * 4;
        float xs[4];
        float sumsq = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            xs[i] = float(raw_head[lane * 4 + i]);
            sumsq += xs[i] * xs[i];
        }
        sumsq = simd_sum(sumsq);
        const float inv = metal::precise::rsqrt(sumsq / 128.0f + eps);
        for (uint i = 0; i < 4; ++i) {
            const float normed = official_gdn_k1_round_bf16(norm_w[lane * 4 + i] * official_gdn_k1_round_bf16(xs[i] * inv));
            const float sig = official_gdn_k1_sigmoid_f32<0>(float(z[base + i]));
            gated[base + i] = bfloat(official_gdn_k1_round_bf16(normed * sig));
        }
    }
}

kernel void qwen38_gdn_fused256_research(
    device const bfloat* qkv [[buffer(0)]],
    device const bfloat* a_in [[buffer(1)]],
    device const bfloat* b_in [[buffer(2)]],
    device const float* A_log [[buffer(3)]],
    device const bfloat* dt_bias [[buffer(4)]],
    device float* state [[buffer(5)]],
    device bfloat* y [[buffer(6)]],
    device const bfloat* z [[buffer(7)]],
    device const float* norm_w [[buffer(8)]],
    device bfloat* gated [[buffer(9)]],
    constant float& k_scale_f32 [[buffer(10)]],
    constant float& eps [[buffer(11)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float qs[128], ks[128], vs[128], gb[2];
    threadgroup bfloat raw_head[128];
    research_gdn_fused_body<256>(
        qkv, a_in, b_in, A_log, dt_bias, state, y, z, norm_w,
        gated, k_scale_f32, eps, tg, lane, sg, qs, ks, vs, gb, raw_head);
}

kernel void qwen38_gdn_fused512_research(
    device const bfloat* qkv [[buffer(0)]],
    device const bfloat* a_in [[buffer(1)]],
    device const bfloat* b_in [[buffer(2)]],
    device const float* A_log [[buffer(3)]],
    device const bfloat* dt_bias [[buffer(4)]],
    device float* state [[buffer(5)]],
    device bfloat* y [[buffer(6)]],
    device const bfloat* z [[buffer(7)]],
    device const float* norm_w [[buffer(8)]],
    device bfloat* gated [[buffer(9)]],
    constant float& k_scale_f32 [[buffer(10)]],
    constant float& eps [[buffer(11)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float qs[128], ks[128], vs[128], gb[2];
    threadgroup bfloat raw_head[128];
    research_gdn_fused_body<512>(
        qkv, a_in, b_in, A_log, dt_bias, state, y, z, norm_w,
        gated, k_scale_f32, eps, tg, lane, sg, qs, ks, vs, gb, raw_head);
}
