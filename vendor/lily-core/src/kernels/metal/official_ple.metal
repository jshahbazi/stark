// SPDX-License-Identifier: Apache-2.0
// Official external-MLX arithmetic policy; Lily entry points remain unchanged.
// Safe math and disabled contraction preserve each materialized BF16 operation;
// fast math can otherwise fold a BF16 product round-trip into the following add.
#include <metal_stdlib>
using namespace metal;
#define TG 256

#define PLE_MAX_CONTEXT 16

// MLX 64ea011cb65f14d9ce2737e60db9a4ae91ed7441:
// unary_ops.h::Sigmoid and bf16_math.h use a stable abs formulation with
// BF16 exp, denominator, reciprocal and positive-side subtraction.
static inline bfloat mlx_primitive_sigmoid(bfloat x) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const bfloat exponential = bfloat(precise::exp(abs(float(x))));
    const bfloat denominator = bfloat(1.0f + float(exponential));
    const bfloat y = bfloat(1.0f / float(denominator));
    return float(x) < 0.0f ? y : bfloat(1.0f - float(y));
}

// gate_g = <key[r, g], query[r, g]> / sqrt(H), signed-sqrt'ed, then
// gated[r, g*H + i] = sigmoid(gate_g) * value[r, i]. One threadgroup per (row, g).
kernel void ple_gate_value_bf16_mlx(device const bfloat* key   [[buffer(0)]],  // [rows, G*H] normed
                                device const bfloat* query [[buffer(1)]],  // [rows, G*H] normed
                                device const bfloat* value [[buffer(2)]],  // [rows, H]
                                device bfloat*       gated [[buffer(3)]],  // [rows, G*H]
                                constant uint&       H     [[buffer(4)]],
                                constant uint&       G     [[buffer(5)]],
                                constant float&      inv_sqrt_h [[buffer(6)]],
                                uint seg  [[threadgroup_position_in_grid]],
                                uint tid  [[thread_index_in_threadgroup]],
                                uint sg   [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    threadgroup float partial[TG / 32];
    threadgroup float gate_s;

    const ulong base = (ulong)seg * H;
    const ulong vrow = (ulong)(seg / G) * H;
    float acc = 0.0f;
    for (uint i = tid; i < H; i += TG) {
        acc += float(bfloat(float(key[base + i]) * float(query[base + i])));
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < TG / 32; ++i) {
            total += partial[i];
        }
        const bfloat reduced = bfloat(total);
        const bfloat gate = bfloat(float(reduced) * float(bfloat(inv_sqrt_h)));
        const bfloat mag = bfloat(sqrt(max(abs(float(gate)), float(bfloat(1e-6f)))));
        const float sign = float(gate) > 0.0f ? 1.0f : (float(gate) < 0.0f ? -1.0f : 0.0f);
        const bfloat signed_gate = bfloat(sign * float(mag));
        gate_s = float(mlx_primitive_sigmoid(signed_gate));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float s = gate_s;
    for (uint i = tid; i < H; i += TG) {
        gated[base + i] = bfloat(s * float(value[vrow + i]));
    }
}

// Dilated causal depthwise conv over x[M, C] (kernel KD, dilation DIL), then
// hyper[m, c] += base[m, c] + silu(conv). The window holds the S = (KD-1)*DIL
// inputs before the chunk, oldest first; win_out receives the post-chunk window.
kernel void ple_conv1d_prefill_bf16_mlx(device const bfloat* win_in  [[buffer(0)]],  // [C, S]
                                    device bfloat*       win_out [[buffer(1)]],  // [C, S]
                                    device const bfloat* x       [[buffer(2)]],  // [M, C]
                                    device const bfloat* w       [[buffer(3)]],  // [KD, C]
                                    device const bfloat* base    [[buffer(4)]],  // [M, C]
                                    device bfloat*       hyper   [[buffer(5)]],  // [M, C]
                                    constant uint&       C       [[buffer(6)]],
                                    constant uint&       KD      [[buffer(7)]],
                                    constant uint&       DIL     [[buffer(8)]],
                                    constant uint&       M       [[buffer(9)]],
                                    uint c [[thread_position_in_grid]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const uint S = (KD - 1) * DIL;
    float win[PLE_MAX_CONTEXT];
    for (uint s = 0; s < S; ++s) {
        win[s] = float(win_in[c * S + s]);
    }
    for (uint m = 0; m < M; ++m) {
        const float xc = float(x[(ulong)m * C + c]);
        float acc = 0.0f;
        for (uint t = 0; t + 1 < KD; ++t) {
            // Tap t looks back (KD-1-t)*DIL tokens; the window's newest entry
            // is one token back.
            acc += win[S - (KD - 1 - t) * DIL] * float(w[t * C + c]);
        }
        acc += xc * float(w[(KD - 1) * C + c]);
        const bfloat conv = bfloat(acc);
        const bfloat sig = mlx_primitive_sigmoid(conv);
        const bfloat act = bfloat(float(conv) * float(sig));
        const ulong at = (ulong)m * C + c;
        const bfloat addend = bfloat(float(base[at]) + float(act));
        hyper[at] = bfloat(float(hyper[at]) + float(addend));
        for (uint s = 0; s + 1 < S; ++s) {
            win[s] = win[s + 1];
        }
        win[S - 1] = xc;
    }
    for (uint s = 0; s < S; ++s) {
        win_out[c * S + s] = bfloat(win[s]);
    }
}

// Single-token variant of the above; the window is shifted in place.
kernel void ple_conv1d_step_bf16_mlx(device bfloat*       window [[buffer(0)]],  // [C, S]
                                 device const bfloat* x      [[buffer(1)]],  // [C]
                                 device const bfloat* w      [[buffer(2)]],  // [KD, C]
                                 device const bfloat* base   [[buffer(3)]],  // [C]
                                 device bfloat*       hyper  [[buffer(4)]],  // [C]
                                 constant uint&       C      [[buffer(5)]],
                                 constant uint&       KD     [[buffer(6)]],
                                 constant uint&       DIL    [[buffer(7)]],
                                 uint c [[thread_position_in_grid]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const uint S = (KD - 1) * DIL;
    const float xc = float(x[c]);
    float acc = 0.0f;
    for (uint t = 0; t + 1 < KD; ++t) {
        acc += float(window[c * S + S - (KD - 1 - t) * DIL]) * float(w[t * C + c]);
    }
    acc += xc * float(w[(KD - 1) * C + c]);
    const bfloat conv = bfloat(acc);
    const bfloat sig = mlx_primitive_sigmoid(conv);
    const bfloat act = bfloat(float(conv) * float(sig));
    const bfloat addend = bfloat(float(base[c]) + float(act));
    hyper[c] = bfloat(float(hyper[c]) + float(addend));
    for (uint s = 0; s + 1 < S; ++s) {
        window[c * S + s] = window[c * S + s + 1];
    }
    window[c * S + S - 1] = bfloat(xc);
}
