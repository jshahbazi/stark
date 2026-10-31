// RESEARCH ONLY: explicit Q8 weight reuse across up to four input rows.
// Adapted from vendor/lily-core/src/kernels/metal/official_q8_gdn_decode.metal,
// SHA-256 21465bbcf61db5342186429c9c4405693fc2205cb5c8730af9f57296adcdfa44.
// That source is a scoped MLX 64ea011cb65f14d9ce2737e60db9a4ae91ed7441 port.
// Per-input-row expressions, group64 metadata, SIMD reduction and BF16 output
// boundary are retained. New: hoist each eight-byte code fragment and affine
// pair once, then apply it to ROWS independent vectors in the same SIMD group.
// Source correspondence is not numerical qualification; strict GPU controls
// must pass before this is used in a model experiment.
//
// Copyright © 2023-2024 Apple Inc.
// MIT License
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

// Fixed-shape ABI, identical buffers/constants to the stock projection:
// 0 U32 codes [N,K/4]; 1 BF16 scales [N,K/64]; 2 BF16 biases [N,K/64];
// 3 BF16 input [ROWS,K]; 4 BF16 output [ROWS,N]; 5 constant int K; 6 int N.
// Dispatch exactly (1,N/8,1) threadgroups, (32,2,1) threads, Safe Metal 4.
// Admitted shapes are (N,K)=(16480,2560) and (2560,6144), Q8/group64.
// ROWS=1 is a diagnostic entry only; model integration retains stock for M1.
// All inputs/outputs must be nonaliasing and live through dispatch completion.

template <int ROWS>
inline void research_q8_gdn_rows_impl(
    const device uint32_t* codes,
    const device bfloat* scales,
    const device bfloat* biases,
    const device bfloat* x,
    device bfloat* y,
    const constant int& K,
    const constant int& N,
    uint3 tid,
    uint simd_gid,
    uint simd_lid) {
    constexpr int VALUES = 8;
    constexpr int BLOCK = VALUES * 32;
    constexpr int OUTPUTS = 4;
    const int out_row = tid.y * 8 + simd_gid * OUTPUTS;
    const int group_stride = K / 64;
    const device uint8_t* ws = (const device uint8_t*)codes
        + out_row * K + simd_lid * VALUES;
    scales += out_row * group_stride + simd_lid / 8;
    biases += out_row * group_stride + simd_lid / 8;
    const device bfloat* vectors[ROWS];
    for (int r = 0; r < ROWS; ++r) vectors[r] = x + r * K + simd_lid * VALUES;
    float result[ROWS][OUTPUTS] = {};

    for (int k = 0; k < K; k += BLOCK) {
        float x_thread[ROWS][VALUES];
        float sums[ROWS];
        for (int r = 0; r < ROWS; ++r) {
            // Exact stock Q8 load_vector specialization: sum is updated in
            // source order before this lane's value is assigned to its tile.
            float sum = 0;
            for (int i = 0; i < VALUES; ++i) {
                sum += vectors[r][i];
                x_thread[r][i] = vectors[r][i];
            }
            sums[r] = sum;
        }
        for (int row = 0; row < OUTPUTS; ++row) {
            const device uint8_t* wl = ws + row * K;
            const float s = scales[row * group_stride];
            const float b = biases[row * group_stride];
            float weight_thread[VALUES];
            for (int i = 0; i < VALUES; ++i) weight_thread[i] = float(wl[i]);
            for (int r = 0; r < ROWS; ++r) {
                // Exact stock qdot<8,8>: eight ordered multiply-adds, then
                // scale*accum + sum*bias, then the block result accumulation.
                float accum = 0;
                for (int i = 0; i < VALUES; ++i) {
                    accum += x_thread[r][i] * weight_thread[i];
                }
                result[r][row] += s * accum + sums[r] * b;
            }
        }
        ws += BLOCK;
        scales += BLOCK / 64;
        biases += BLOCK / 64;
        for (int r = 0; r < ROWS; ++r) vectors[r] += BLOCK;
    }
    for (int r = 0; r < ROWS; ++r) {
        for (int row = 0; row < OUTPUTS; ++row) {
            result[r][row] = simd_sum(result[r][row]);
            if (simd_lid == 0) y[r * N + out_row + row] = bfloat(result[r][row]);
        }
    }
}

#define RESEARCH_Q8_ROWS_ENTRY(ROWS) \
kernel void qwen38_q8_gdn_rows_##ROWS##_research( \
    device const uint* codes [[buffer(0)]], \
    device const bfloat* scales [[buffer(1)]], \
    device const bfloat* biases [[buffer(2)]], \
    device const bfloat* x [[buffer(3)]], \
    device bfloat* y [[buffer(4)]], \
    constant int& K [[buffer(5)]], \
    constant int& N [[buffer(6)]], \
    uint3 tid [[threadgroup_position_in_grid]], \
    uint simd_gid [[simdgroup_index_in_threadgroup]], \
    uint simd_lid [[thread_index_in_simdgroup]]) { \
    research_q8_gdn_rows_impl<ROWS>( \
        codes, scales, biases, x, y, K, N, tid, simd_gid, simd_lid); \
}

RESEARCH_Q8_ROWS_ENTRY(1)
RESEARCH_Q8_ROWS_ENTRY(2)
RESEARCH_Q8_ROWS_ENTRY(3)
RESEARCH_Q8_ROWS_ENTRY(4)
