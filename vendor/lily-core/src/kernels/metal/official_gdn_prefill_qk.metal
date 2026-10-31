// SPDX-License-Identifier: MIT
// Copyright (c) 2026 David Dalcu
// Normalization fragment adapted from MLX-Serve
// 046b647686419df06c6c7132fb1d15d76185c3e5, transformer.zig
// GDN_PREWORK_SOURCE. See LICENSE.mlx-serve for the retained MIT notice.
// Modified by Stark: consume already-activated native QKV, fixed geometry,
// native buffer/grid bindings, and interleaved Q/K output row stride.
// The four-element sum, SIMD reduction and BF16 rounding expressions follow
// the served prework; conv, gates, recurrence and norm are not implemented here.
#include <metal_stdlib>
using namespace metal;

kernel void gdn_prefill_qk_value_key_contiguous_mlx(
    device const bfloat* qkv [[buffer(0)]],
    device bfloat* qk [[buffer(1)]],
    constant float& k_scale_f32 [[buffer(2)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 local [[thread_position_in_threadgroup]]) {
    using T = bfloat;
    constexpr uint HK = 16, HV = 48, DK = 128, C = (2 * HK + HV) * DK;
    const uint lane = local.x;
    const size_t row = group.x;
    const uint logical_head = group.y;
    const bool is_q = logical_head < HK;
    const uint head = is_q ? logical_head : logical_head - HK;
    const uint channel_base = is_q ? head * DK : HK * DK + head * DK;
    const T q_scale = T(1.0f / float(DK));
    const T k_scale = T(k_scale_f32);
    constexpr float qk_eps = 1e-6f;
    T activated[4];
    float sumsq = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        activated[i] = qkv[row * C + channel_base + lane * 4 + i];
        float value = float(activated[i]);
        sumsq += value * value;
    }
    sumsq = simd_sum(sumsq);
    float inv = metal::precise::rsqrt(sumsq / float(DK) + qk_eps);
    const T scale = is_q ? q_scale : k_scale;
    size_t out_base = (row * (2 * HK) + logical_head) * DK + lane * 4;
    for (uint i = 0; i < 4; ++i) {
        const T rms = T(1) * T(float(activated[i]) * inv);
        const T value = scale * rms;
        qk[out_base + i] = value;
    }
}
