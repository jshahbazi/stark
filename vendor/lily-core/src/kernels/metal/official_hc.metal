// SPDX-License-Identifier: Apache-2.0
// Official external-MLX arithmetic policy; Lily entry points remain unchanged.
// Safe math and disabled contraction preserve each materialized BF16 operation;
// fast math can otherwise fold a BF16 product round-trip into the following add.
#include <metal_stdlib>
using namespace metal;
#define TG 256

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

template <bool Primitive>
static inline bfloat mlx_hc_sigmoid(bfloat x) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    // hc_decode2.zig's served narrow kernel explicitly computes in F32.
    return Primitive ? mlx_primitive_sigmoid(x)
                     : bfloat(1.0f / (1.0f + exp(-float(x))));
}

kernel void rmsnorm_grouped_bf16_mlx(device const bfloat* x   [[buffer(0)]],
                                 device const bfloat* w   [[buffer(1)]],
                                 device bfloat*       out [[buffer(2)]],
                                 constant uint&       H   [[buffer(3)]],
                                 constant uint&       G   [[buffer(4)]],
                                 constant float&      eps [[buffer(5)]],
                                 constant float&      w_bias [[buffer(6)]],
                                 uint seg  [[threadgroup_position_in_grid]],
                                 uint tid  [[thread_index_in_threadgroup]],
                                 uint sg   [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    threadgroup float partial[TG / 32];
    threadgroup float inv_rms;

    const ulong base = (ulong)seg * H;
    const uint w0 = (seg % G) * H;
    float acc = 0.0f;
    for (uint i = tid; i < H; i += TG) {
        float v = float(x[base + i]);
        acc += v * v;
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
        inv_rms = precise::rsqrt(total / float(H) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < H; i += TG) {
        const float gain = float(w[w0 + i]);
        const bfloat normalized = bfloat(float(x[base + i]) * inv_rms);
        out[base + i] = bfloat(float(normalized) * gain);
    }
}

template <bool Primitive>
static inline void mlx_hc_silu(device const bfloat* x, device bfloat* out,
                               float inv_scale, uint gid) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const bfloat v = bfloat(float(x[gid]) * inv_scale);
    out[gid] = bfloat(float(v) * float(mlx_hc_sigmoid<Primitive>(v)));
}
#define MLX_HC_SILU(NAME, PRIMITIVE) \
kernel void NAME(device const bfloat* x [[buffer(0)]], \
    device bfloat* out [[buffer(1)]], constant float& inv_scale [[buffer(2)]], \
    uint gid [[thread_position_in_grid]]) { mlx_hc_silu<PRIMITIVE>(x, out, inv_scale, gid); }
MLX_HC_SILU(silu_scaled_bf16_mlx, false)
MLX_HC_SILU(silu_scaled_bf16_mlx_primitive, true)

// MLX narrow decode rounds the sum once; its prefill kernel rounds each
// stream addition. Both round sigmoid, product and reciprocal multiply.
template <bool Sequential, bool Primitive, bool FirstProduct>
static inline void mlx_hc_mix(device const bfloat* up, device const bfloat* hn,
    device bfloat* mixed, uint H, uint G, uint2 gid) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const uint i = gid.x;
    const ulong row = ulong(gid.y) * G * H;
    float sum = 0.0f;
    for (uint g = 0; g < G; ++g) {
        const bfloat sig = mlx_hc_sigmoid<Primitive>(up[row + g * H + i]);
        const bfloat product = bfloat(float(sig) * float(hn[row + g * H + i]));
        sum = Sequential ? (FirstProduct && g == 0 ? float(product) : float(bfloat(sum + float(product))))
                         : sum + float(product);
    }
    mixed[ulong(gid.y) * H + i] =
        bfloat(float(bfloat(sum)) * float(bfloat(1.0f / float(G))));
}
#define MLX_HC_MIX(NAME, SEQUENTIAL, PRIMITIVE, FIRST_PRODUCT) \
kernel void NAME(device const bfloat* up [[buffer(0)]], \
    device const bfloat* hn [[buffer(1)]], device bfloat* mixed [[buffer(2)]], \
    constant uint& H [[buffer(3)]], constant uint& G [[buffer(4)]], \
    uint2 gid [[thread_position_in_grid]]) { mlx_hc_mix<SEQUENTIAL, PRIMITIVE, FIRST_PRODUCT>(up, hn, mixed, H, G, gid); }
MLX_HC_MIX(hc_mix_bf16_mlx, false, false, false)
// The primitive BF16 column mean also rounds its stream accumulator after
// each addition. Only the eligible custom narrow HC kernel sums in F32.
// MLX 64ea011c reduction/ops.h Sum<BF16> and reduce_col.h col_reduce_small
// start from positive zero; hc_prefill_mix.metal starts with its first product.
// Preserve their different signed-zero results.
MLX_HC_MIX(hc_mix_bf16_mlx_primitive, true, true, false)
MLX_HC_MIX(hc_mix_bf16_mlx_prefill, true, true, true)

template <bool Primitive>
static inline void mlx_hc_inject(device bfloat* hyper, device const bfloat* branch,
    device const bfloat* inj, uint H, uint G, float inv_g, uint2 gid) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const uint col = gid.x;
    const uint g = col / H;
    const uint i = col % H;
    const ulong r = gid.y;
    const bfloat logit = bfloat(float(inj[r * G + g]) * inv_g);
    const bfloat weight = bfloat(2.0f * float(mlx_hc_sigmoid<Primitive>(logit)));
    const ulong at = r * G * H + col;
    const bfloat delta = bfloat(float(branch[r * H + i]) * float(weight));
    hyper[at] = bfloat(float(hyper[at]) + float(delta));
}
#define MLX_HC_INJECT(NAME, PRIMITIVE) \
kernel void NAME(device bfloat* hyper [[buffer(0)]], \
    device const bfloat* branch [[buffer(1)]], device const bfloat* inj [[buffer(2)]], \
    constant uint& H [[buffer(3)]], constant uint& G [[buffer(4)]], \
    constant float& inv_g [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) { \
    mlx_hc_inject<PRIMITIVE>(hyper, branch, inj, H, G, inv_g, gid); }
MLX_HC_INJECT(hc_inject_bf16_mlx, false)
MLX_HC_INJECT(hc_inject_bf16_mlx_primitive, true)
