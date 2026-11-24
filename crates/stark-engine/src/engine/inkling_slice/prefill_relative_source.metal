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

// Dispatch (32, ceil(query_head_count / 16)) threadgroups of 32 threads.
// Input is [query_head_count, 16], weight [16, 1024], output [query_head_count, 1024].
// query_head_count is the number of packed token/head rows, including all 24 heads per token.
kernel void ink_prefill_relative_table_source(
    device const bfloat *input [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]], constant uint &query_head_count [[buffer(3)]],
    uint2 tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint M = 16, N = 32, K = 16;
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        M, N, K, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroup> product;
    auto a = product.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto b = product.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto acc = product.get_destination_cooperative_tensor<remove_addrspace_t<decltype(a)>,
        remove_addrspace_t<decltype(b)>, float>();

    const uint row = tile.y * M;
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

// Ring and backing are both [token][KV head][channel]. Backing begins at
// floor(key_start / 128) * 128 and ends at round_up(key_end, 128).
// Dispatch (kv_width / 4, copy_rows) threads. Gather the preceding span
// before the chunk writes its KV, and gather the chunk span after those writes.
// Each dispatch also zeros its share of the aligned border rows.
kernel void ink_prefill_relative_gather(
    device const bfloat4 *ring_k [[buffer(0)]], device const bfloat4 *ring_v [[buffer(1)]],
    device bfloat4 *backing_k [[buffer(2)]], device bfloat4 *backing_v [[buffer(3)]],
    constant int4 &p [[buffer(4)]], // backing_start, copy_start, key_start, key_end (exclusive)
    uint2 tid [[thread_position_in_grid]]
) {
    if (tid.x >= kv_width / 4) return;
    const int token = p.y + int(tid.y);
    const int target = (token - p.x) * (kv_width / 4) + int(tid.x);
    if (token < p.z || token >= p.w) {
        backing_k[target] = bfloat4(0);
        backing_v[target] = bfloat4(0);
    } else {
        const int source = (token % window) * (kv_width / 4) + int(tid.x);
        backing_k[target] = ring_k[source];
        backing_v[target] = ring_v[source];
    }
}

// All query-side inputs and outputs are BF16, packed by query row and head.
// Queries, gate and attended have 128 channels, tables have 1024 distances,
// and sinks have one channel. Dispatch (ceil(queries / 16), 24) threadgroups
// of 128 threads. The chronological KV backing must cover each query's
// visible window and contain zeros in its aligned leading/trailing borders.
kernel void ink_prefill_relative_source(
    device const bfloat *query [[buffer(0)]],
    device bfloat *key [[buffer(1)]], device bfloat *value [[buffer(2)]],
    device const bfloat *table [[buffer(3)]], device const bfloat *sink [[buffer(4)]],
    device const bfloat *gate [[buffer(5)]], device bfloat *attended [[buffer(6)]],
    constant int4 &p [[buffer(7)]], // query_start, query_count, key_start, key_count
    uint2 position [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint group [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    using namespace mpp::tensor_ops;
    using dims = dextents<int, 2>;
    using strides = array<int, 2>;
    constexpr matmul2d_descriptor dot_shape(query_rows, columns, head_width, false, true, true);
    constexpr matmul2d_descriptor value_shape(query_rows, columns, head_width, false, false, true);
    matmul2d<dot_shape, execution_simdgroup> dot;
    matmul2d<value_shape, execution_simdgroup> weighted_value;

    const int query_start = p.x;
    const int query_count = p.y;
    const int key_start = p.z;
    const int key_end = key_start + p.w;
    const int head = int(position.y);
    const int first_row = int(position.x) * query_rows;
    const int live_rows = min(query_rows, query_count - first_row);
    const int column_start = int(group) * columns;
    const int absolute_query = query_start + first_row;
    const int newest = (absolute_query + live_rows - 1) / head_width;
    const int oldest = max(key_start, absolute_query + 1 - window) / head_width;
    const int first_block = key_start / head_width;

#ifdef INK_RELATIVE_BF16_QUERY
    // BF16 queries: every product is exact in F32 as before; only the
    // tensor unit's accumulation path differs from the F32-operand reference.
    typedef bfloat query_word;
#else
    typedef float query_word;
#endif
    threadgroup query_word widened_query[query_rows * head_width];
    threadgroup float tile[query_rows * head_width];
    threadgroup float scale_prior[query_rows], denominator[query_rows], sink_weight[query_rows];
    threadgroup bfloat probabilities[query_rows * head_width];
    float accumulator[query_rows] = {};
    float peaks[query_rows / groups], denominators[query_rows / groups] = {};
    for (int i = 0; i < query_rows / groups; ++i) peaks[i] = -INFINITY;
    for (int row = 0; row < query_rows; ++row) {
        const int query_offset = ((first_row + row) * query_heads + head) * head_width;
        widened_query[row * head_width + tid] = row < live_rows ? query_word(query[query_offset + tid]) : query_word(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tensor<threadgroup query_word, dims, tensor_inline> queries(
        widened_query, dims(head_width, query_rows), strides{1, head_width});
    tensor<threadgroup float, dims, tensor_inline> output(
        tile + column_start, dims(columns, query_rows), strides{1, head_width});
    tensor<threadgroup bfloat, dims, tensor_inline> weights(
        probabilities, dims(head_width, query_rows), strides{1, head_width});

    for (int block = newest; block >= oldest; --block) {
        const int kv_head = head / (query_heads / key_heads);
        const int base = (block - first_block) * head_width * kv_width + kv_head * head_width;
        tensor<device bfloat, dims, tensor_inline> keys(
            key + base + column_start * kv_width, dims(head_width, columns), strides{1, kv_width});
        dot.run(queries, keys, output);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each SIMD group normalizes four complete rows in the source order.
        for (int row = int(group); row < query_rows; row += groups) {
            const int query_head = (first_row + row) * query_heads + head;
            float scores[4], local_max = -INFINITY;
            for (int j = 0; j < 4; ++j) {
                const int column = int(lane) + j * 32;
                const int absolute_key = block * head_width + column;
                const int distance = absolute_query + row - absolute_key;
                scores[j] = -INFINITY;
                if (row < live_rows && absolute_key >= key_start && absolute_key < key_end &&
                    distance >= 0 && distance < window) {
                    const float relative = float(table[query_head * window + distance]);
                    scores[j] = fma(tile[row * head_width + column], 0.08838834764831845f, relative);
                }
                local_max = max(local_max, scores[j]);
            }
            const int i = row / groups;
            const float peak = max(peaks[i], simd_max(local_max));
            const float previous = isfinite(peaks[i]) ? fast::exp(peaks[i] - peak) : 0.0f;
            float local_sum = 0.0f;
            for (int j = 0; j < 4; ++j) {
                const float weight = isfinite(scores[j]) ? fast::exp(scores[j] - peak) : 0.0f;
                probabilities[row * head_width + lane + j * 32] = bfloat(weight);
                local_sum += weight;
            }
            denominators[i] = fma(denominators[i], previous, simd_sum(local_sum));
            peaks[i] = peak;
            if (lane == 0) {
                scale_prior[row] = previous;
                if (block == oldest && row < live_rows) {
                    const float gap = float(sink[query_head]) - peak;
                    const float inverse = 1.0f / (denominators[i] + fast::exp(gap));
                    denominator[row] = 1.0f;
                    sink_weight[row] = inverse;
                    if (inverse < FLT_MIN) {
                        // Rescale before applying a sink whose reciprocal would flush.
                        denominator[row] = denominators[i];
                        sink_weight[row] = 1.0f / (1.0f + fast::exp(gap - fast::log(denominators[i])));
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        tensor<device bfloat, dims, tensor_inline> values(
            value + base + column_start, dims(columns, head_width), strides{1, kv_width});
        weighted_value.run(weights, values, output);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int row = 0; row < query_rows; ++row) {
            accumulator[row] = fma(
                accumulator[row], scale_prior[row], tile[row * head_width + column_start + lane]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int row = 0; row < live_rows; ++row) {
        const int out = (first_row + row) * query_width + head * head_width + column_start + int(lane);
        const float answer = (accumulator[row] / denominator[row]) * sink_weight[row];
        const float g = float(gate[out]);
        const float y = 1.0f / (1.0f + precise::exp(abs(g)));
        attended[out] = bfloat(answer * (g < 0.0f ? y : 1.0f - y));
    }
}
