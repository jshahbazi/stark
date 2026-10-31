// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT
// Adapted from MLX 64ea011cb65f14d9ce2737e60db9a4ae91ed7441,
// rms_norm.metal::rms_single_row. See LICENSE.mlx and UPSTREAM.md.
// D threads remain live for the following RoPE/cache writer. Only ceil(D/4)
// threads read four consecutive values; all other RMS workers contribute +0.
// The 32-slot zero-padded second SIMD reduction is the pinned RMS tree.
static inline void official_rms_row(
    float value, device const bfloat* w, uint D, float eps,
    uint tid, uint sg, uint lane, threadgroup float* partial,
    threadgroup float* inv_rms, threadgroup bfloat* normed) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    normed[tid] = bfloat(value);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        const uint col = tid * 4 + i;
        const float x = col < D ? float(normed[col]) : 0.0f;
        acc += x * x;
    }
    acc = simd_sum(acc);
    if (sg == 0) partial[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partial[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float total = simd_sum(partial[lane]);
        if (lane == 0) *inv_rms = precise::rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bfloat normalized = bfloat(value * *inv_rms);
    normed[tid] = bfloat(float(normalized) * float(w[tid]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
