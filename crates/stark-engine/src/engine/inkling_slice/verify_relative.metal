// SPDX-License-Identifier: MIT
// Adapted from Ollama mlx/relative_attention_prefill.go (6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8).
// Source file SHA-256 c120b7e40520aefbc3d5bf008f72a3bd8be68e2624aafb66032dad486ebe5a6e.
// The table product uses register fragments from MLX steel/gemm/nax.h,
// copyright © 2025 Apple Inc., at fc344c986c4169d99f76b5d95ae6b4a58a359a63 (LICENSE.mlx-reference).
// Table projection register order from MLX steel/gemm/nax.h at
// fc344c986c4169d99f76b5d95ae6b4a58a359a63, Copyright © 2025 Apple Inc. (LICENSE.mlx-reference).

/*
MIT License

Copyright (c) Ollama

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

namespace {
constant constexpr int query_rows = 16;
constant constexpr int head_width = 128;
constant constexpr int query_heads = 24;
constant constexpr int key_heads = 8;
constant constexpr int window = 1024;
constant constexpr int kv_width = key_heads * head_width;
constant constexpr int query_width = query_heads * head_width;
constant constexpr int groups = 4;
constant constexpr int columns = head_width / groups;
} // namespace

kernel void ink_verify_relative_tables(
    device const bfloat *r0 [[buffer(0)]], device const bfloat *r1 [[buffer(1)]],
    device const bfloat *r2 [[buffer(2)]], device const bfloat *r3 [[buffer(3)]],
    device const bfloat *weight [[buffer(4)]],
    device bfloat *t0 [[buffer(5)]], device bfloat *t1 [[buffer(6)]],
    device bfloat *t2 [[buffer(7)]], device bfloat *t3 [[buffer(8)]],
    uint2 tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint M = 16, N = 32, K = 16;
    constexpr uint query_head_count = 24;
    device const bfloat *inputs[] = {r0, r1, r2, r3};
    device bfloat *outputs[] = {t0, t1, t2, t3};
    device const bfloat *input = inputs[tile.y / 2];
    device bfloat *output = outputs[tile.y / 2];
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        M, N, K, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroup> product;
    auto a = product.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto b = product.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto acc = product.get_destination_cooperative_tensor<remove_addrspace_t<decltype(a)>,
        remove_addrspace_t<decltype(b)>, float>();

    const uint row = (tile.y % 2) * M;
    const uint column = tile.x * N;
    const uint row_base = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const uint k_base = (((lane >> 2) & 2) | (lane & 1)) * 4;
    for (ushort i = 0; i < 8; ++i) {
        const uint m = row_base + (i >> 2) * 8;
        const uint d = k_base + (i & 3);
        a[i] = row + m < query_head_count ? input[(row + m) * K + d] : bfloat(0);
        b[i] = weight[d * window + column + m];
        b[i + 8] = weight[d * window + column + m + 16];
    }
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
    product.run(a, b, acc);
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        const auto at = acc.get_multidimensional_index(i);
        if (row + at[1] < query_head_count) {
            output[(row + at[1]) * window + column + at[0]] = bfloat(acc[i]);
        }
    }
}
