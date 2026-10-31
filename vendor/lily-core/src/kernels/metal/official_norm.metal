// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: MIT
// Adapted by Stark from MLX 64ea011cb65f14d9ce2737e60db9a4ae91ed7441,
// mlx/backend/metal/kernels/rms_norm.metal::rms_single_row.
// The MIT License
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <metal_stdlib>
using namespace metal;

// Explicit BF16-only official policy. Four consecutive elements per thread,
// an F32 SIMD sum, a zero-padded F32 partial plane and a second SIMD sum
// follow the pinned single-row kernel. Wider looped RMS remains unqualified.
kernel void rmsnorm_bf16_mlx(device const bfloat* x [[buffer(0)]],
                            device const bfloat* w [[buffer(1)]],
                            device bfloat* out [[buffer(2)]],
                            constant uint& H [[buffer(3)]],
                            constant float& eps [[buffer(4)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    threadgroup float partial[32];
    threadgroup float inv_rms;
    const ulong base = ulong(row) * H;
    float values[4];
    float acc = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        const uint col = tid * 4 + i;
        values[i] = col < H ? float(x[base + col]) : 0.0f;
        acc += values[i] * values[i];
    }
    acc = simd_sum(acc);
    if (sg == 0) partial[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) partial[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float total = simd_sum(partial[lane]);
        if (lane == 0) inv_rms = precise::rsqrt(total / float(H) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
        const uint col = tid * 4 + i;
        if (col < H) {
            const bfloat normalized = bfloat(values[i] * inv_rms);
            out[base + col] = bfloat(float(w[col]) * float(normalized));
        }
    }
}
