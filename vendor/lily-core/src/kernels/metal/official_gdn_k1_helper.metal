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
//
// Source contract: mlx-serve 046b647686419df06c6c7132fb1d15d76185c3e5,
// gdn_decode.zig K1_SOURCE; inline BF16 sigmoid and SiLU. MLX BF16 wrappers
// round each arithmetic result. Stark port uses integer RNE barriers so native
// bfloat promotion/fast-math cannot erase intermediate BF16 boundaries.
// MODE0 keeps the literal non-precise metal::exp(float) intrinsic. MODE1/2
// are explicit diagnostic controls only; they do not select production policy.

#include <metal_stdlib>
using namespace metal;

static inline float official_gdn_k1_round_bf16(float x) {
    const uint bits = as_type<uint>(x);
    // Preserve infinities/NaNs. All oracle inputs are finite; exp can overflow.
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

kernel void official_gdn_k1_helper_exhaustive(
    device const bfloat* x [[buffer(0)]],
    device bfloat* default_sig [[buffer(1)]],
    device bfloat* default_silu [[buffer(2)]],
    device bfloat* precise_sig [[buffer(3)]],
    device bfloat* precise_silu [[buffer(4)]],
    device bfloat* fast_sig [[buffer(5)]],
    device bfloat* fast_silu [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    const float value = float(x[i]);
    default_sig[i] = bfloat(official_gdn_k1_sigmoid_f32<0>(value));
    default_silu[i] = bfloat(official_gdn_k1_silu_f32<0>(value));
    precise_sig[i] = bfloat(official_gdn_k1_sigmoid_f32<1>(value));
    precise_silu[i] = bfloat(official_gdn_k1_silu_f32<1>(value));
    fast_sig[i] = bfloat(official_gdn_k1_sigmoid_f32<2>(value));
    fast_silu[i] = bfloat(official_gdn_k1_silu_f32<2>(value));
}
