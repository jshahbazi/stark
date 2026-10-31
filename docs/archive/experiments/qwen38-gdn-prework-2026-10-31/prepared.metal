// RESEARCH ONLY. No production caller, Rust wrapper, or environment selector
// exists for this prototype. Dispatch and scratch ABI are documented below.
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

// ABI for qwen38_gdn_conv_prepare_research:
//   buffer 0: mutable BF16 conv_window [10240,3], channel-major history
//   buffer 1: BF16 raw in_proj_qkv [10240]
//   buffer 2: BF16 conv weights [4,10240], tap-major
//   buffer 3: BF16 prepared_qkv [10240], stock-rounded Q/K and convolved V
//   buffer 4: BF16 a [48], buffer 5: BF16 b [48]
//   buffer 6: F32 A_log [48], buffer 7: BF16 dt_bias [48]
//   buffer 8: F32 gate_pairs [96], [head*2+0]=decay, [head*2+1]=beta
//   buffer 9: the caller's original F32 K normalization scale
// Dispatch exactly 80 threadgroups of 32 threads: TG 0..15 Q heads,
// 16..31 K heads, 32..79 V heads. Every lane handles four contiguous values.
// Q/K normalization and gate expressions follow stock source order; intended
// numerical equivalence remains unqualified until source-bound GPU controls.
// Scratch outputs must not alias inputs, weights, state, or the history.

static inline bfloat research_gdn_scaled_rms(float value, float inv, bfloat scale) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const bfloat normalized = bfloat(value * inv);
    return bfloat(float(normalized) * float(scale));
}

// Stock ValueKey helper copied without changing branch or expression order.
inline float research_gdn_stock_log1p(float x) {
    float xp1 = 1.0f + x;
    if (xp1 == metal::numeric_limits<float>::max()) { return metal::numeric_limits<float>::max(); }
    if (xp1 == 1.0f) { return x; }
    return x * (metal::log(xp1) / (xp1 - 1.0f));
}

kernel void qwen38_gdn_conv_prepare_research(
    device bfloat* window [[buffer(0)]],
    device const bfloat* x [[buffer(1)]],
    device const bfloat* w [[buffer(2)]],
    device bfloat* prepared [[buffer(3)]],
    device const bfloat* a [[buffer(4)]],
    device const bfloat* b [[buffer(5)]],
    device const float* A_log [[buffer(6)]],
    device const bfloat* dt_bias [[buffer(7)]],
    device float* gate_pairs [[buffer(8)]],
    constant float& k_scale_f32 [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint HK = 16, D = 128, C = 10240, TAPS = 3;
    const bool is_q = tg < HK;
    const bool is_k = tg >= HK && tg < 2 * HK;
    const uint head = is_q ? tg : (is_k ? tg - HK : tg - 2 * HK);
    const uint channel_base = is_q ? head * D : (is_k ? (HK + head) * D : (2 * HK + head) * D);
    const uint lane_base = channel_base + lane * 4;
    bfloat act[4];
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint tap = 0; tap < TAPS; ++tap) {
        for (uint i = 0; i < 4; ++i) {
            const uint c = lane_base + i;
            acc[i] += float(window[c * TAPS + tap]) * float(w[tap * C + c]);
        }
    }
    for (uint i = 0; i < 4; ++i) {
        const uint c = lane_base + i;
        acc[i] += float(x[c]) * float(w[TAPS * C + c]);
        act[i] = bfloat(official_gdn_k1_silu_f32<0>(acc[i]));
    }
    // The original convolution advances each channel exactly once after its
    // output has been formed. Q, K and V channel ownership is disjoint.
    for (uint i = 0; i < 4; ++i) {
        const uint c = lane_base + i;
        window[c * TAPS] = window[c * TAPS + 1];
        window[c * TAPS + 1] = window[c * TAPS + 2];
        window[c * TAPS + 2] = x[c];
    }

    if (is_q || is_k) {
        float sumsq = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            const float v = float(act[i]);
            sumsq += v * v;
        }
        sumsq = simd_sum(sumsq);
        const float inv = metal::precise::rsqrt(sumsq / float(D) + 1e-6f);
        const bfloat scale = is_q ? bfloat(1.0f / float(D)) : bfloat(k_scale_f32);
        for (uint i = 0; i < 4; ++i) {
            prepared[lane_base + i] = research_gdn_scaled_rms(float(act[i]), inv, scale);
        }
    } else {
        for (uint i = 0; i < 4; ++i) prepared[lane_base + i] = act[i];
        if (lane == 31) {
            const bfloat bv = b[head];
            gate_pairs[head * 2 + 1] = official_gdn_k1_sigmoid_f32<0>(float(bv));
            const float apd = float(a[head]) + float(dt_bias[head]);
            const float sp = research_gdn_stock_log1p(metal::precise::exp(apd));
            const float ea = metal::precise::exp(A_log[head]);
            gate_pairs[head * 2] = metal::precise::exp(-(ea * sp));
        }
    }
}

// ABI for qwen38_gdn_recurrence_research:
//   buffer 0: BF16 prepared_qkv [10240] from the preceding kernel
//   buffer 1: F32 gate_pairs [96] from the preceding kernel
//   buffer 2: mutable F32 state [48,128,128], physical [H,V,K]
//   buffer 3: BF16 raw output [48,128]
// Dispatch exactly 192 threadgroups (48 heads x four V partitions), 256
// threads/TG. This retains the production VK row partition, per-lane four-K
// state register tile, loop order, SIMD reductions, state write order and BF16
// raw boundary. Prepared scratch must remain live until this dispatch ends.
kernel void qwen38_gdn_recurrence_research(
    device const bfloat* prepared [[buffer(0)]],
    device const float* gate_pairs [[buffer(1)]],
    device float* state [[buffer(2)]],
    device bfloat* y [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int HK = 16, HV = 48, DK = 128, DV = 128;
    constexpr int NT = 256, SPLIT = 4, NSG = NT / 32;
    constexpr int RB = DV / SPLIT, R = RB / NSG, GRP = HV / HK;
    const uint hv = tg / SPLIT;
    const uint part = tg % SPLIT;
    const uint hk = hv / GRP;
    const uint row0 = part * RB + sg * R;
    float st[R][4];
    for (int j = 0; j < R; ++j) {
        const uint base = (hv * DV + row0 + j) * DK + lane * 4;
        for (int i = 0; i < 4; ++i) st[j][i] = state[base + i];
    }

    const uint q_base = hk * DK + lane * 4;
    const uint k_base = HK * DK + hk * DK + lane * 4;
    float kk[4], qq[4];
    for (int i = 0; i < 4; ++i) {
        qq[i] = float(prepared[q_base + i]);
        kk[i] = float(prepared[k_base + i]);
    }
    float g = lane == 31 ? gate_pairs[hv * 2] : 0.0f;
    float beta = lane == 31 ? gate_pairs[hv * 2 + 1] : 0.0f;
    g = simd_broadcast(g, 31);
    beta = simd_broadcast(beta, 31);
    for (int j = 0; j < R; ++j) {
        const uint dv = row0 + j;
        float kv_mem = 0.0f;
        for (int i = 0; i < 4; ++i) {
            st[j][i] = st[j][i] * g;
            kv_mem += st[j][i] * kk[i];
        }
        kv_mem = simd_sum(kv_mem);
        float v = lane == 31 ? float(prepared[2 * HK * DK + hv * DV + dv]) : 0.0f;
        v = simd_broadcast(v, 31);
        const float delta = (v - kv_mem) * beta;
        float out = 0.0f;
        for (int i = 0; i < 4; ++i) {
            st[j][i] = st[j][i] + kk[i] * delta;
            out += st[j][i] * qq[i];
        }
        out = simd_sum(out);
        const uint base = (hv * DV + dv) * DK + lane * 4;
        for (int i = 0; i < 4; ++i) state[base + i] = st[j][i];
        if (lane == 0) y[hv * DV + dv] = bfloat(out);
    }
}

// Staged control variant: same prepared-QKV and gates as the direct variant,
// but restores the original recurrence's Q/K/V threadgroup staging
// and threadgroup barrier. This isolates prework savings from the direct
// variant's repeated prepared-Q/K device loads. ABI and grid match the direct
// recurrence kernel above; scratch lifetime and non-aliasing rules also match.
kernel void qwen38_gdn_recurrence_staged_research(
    device const bfloat* prepared [[buffer(0)]],
    device const float* gate_pairs [[buffer(1)]],
    device float* state [[buffer(2)]],
    device bfloat* y [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    constexpr int HK = 16, HV = 48, DK = 128, DV = 128;
    constexpr int NT = 256, SPLIT = 4, NSG = NT / 32;
    constexpr int RB = DV / SPLIT, R = RB / NSG, GRP = HV / HK;
    const uint hv = tg / SPLIT;
    const uint part = tg % SPLIT;
    const uint hk = hv / GRP;
    const uint row0 = part * RB + sg * R;
    threadgroup float qs[DK], ks[DK], vs[DV];
    threadgroup float gb[2];
    float st[R][4];
    for (int j = 0; j < R; ++j) {
        const uint base = (hv * DV + row0 + j) * DK + lane * 4;
        for (int i = 0; i < 4; ++i) st[j][i] = state[base + i];
    }
    if (sg < 3) {
        const uint base = sg == 0 ? hk * DK : (sg == 1 ? HK * DK + hk * DK : 2 * HK * DK + hv * DV);
        threadgroup float* dst = sg == 0 ? qs : (sg == 1 ? ks : vs);
        for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = float(prepared[base + lane * 4 + i]);
    }
    if (sg == 3 && lane == 31) {
        gb[0] = gate_pairs[hv * 2];
        gb[1] = gate_pairs[hv * 2 + 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float kk[4], qq[4];
    for (int i = 0; i < 4; ++i) {
        kk[i] = ks[lane * 4 + i];
        qq[i] = qs[lane * 4 + i];
    }
    const float g = gb[0], beta = gb[1];
    for (int j = 0; j < R; ++j) {
        const uint dv = row0 + j;
        float kv_mem = 0.0f;
        for (int i = 0; i < 4; ++i) {
            st[j][i] = st[j][i] * g;
            kv_mem += st[j][i] * kk[i];
        }
        kv_mem = simd_sum(kv_mem);
        const float delta = (vs[dv] - kv_mem) * beta;
        float out = 0.0f;
        for (int i = 0; i < 4; ++i) {
            st[j][i] = st[j][i] + kk[i] * delta;
            out += st[j][i] * qq[i];
        }
        out = simd_sum(out);
        const uint base = (hv * DV + dv) * DK + lane * 4;
        for (int i = 0; i < 4; ++i) state[base + i] = st[j][i];
        if (lane == 0) y[hv * DV + dv] = bfloat(out);
    }
}
