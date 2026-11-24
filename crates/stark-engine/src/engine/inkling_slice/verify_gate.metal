// SPDX-License-Identifier: MIT
/*
The long path follows Ollama mlx/relative_attention.go at
6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8. The Stark version uses single-query
dispatch, token-major KV rings, interleaved FP32 numerators and statistics,
and a fused output gate. The short-path arithmetic follows Apple's MLX
sdpa_vector at fc344c986c4169d99f76b5d95ae6b4a58a359a63
(Copyright © 2024 Apple Inc., LICENSE.mlx-reference).

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
using namespace metal;

constant uint ink_gated_part = 130;
constant uint ink_gated_heads = 24;
constant uint ink_gated_shards = 64;

inline void ink_gate_wrapped(
    device const float *parts, device const bfloat *gate, device bfloat *result,
    uint head, uint component, uint lane
) {
    const device float *head_parts = parts + head * ink_gated_shards * ink_gated_part;
    float peak = -INFINITY;
    for (uint shard = lane; shard < ink_gated_shards; shard += 32)
        peak = max(peak, head_parts[shard * ink_gated_part + 128]);
    peak = simd_max(peak);
    float weight = 0.f;
    float4 numerator(0.f);
    for (uint shard = lane; shard < ink_gated_shards; shard += 32) {
        const device float *part = head_parts + shard * ink_gated_part;
        const float factor = fast::exp(part[128] - peak);
        weight = fma(part[129], factor, weight);
        for (uint j = 0; j < 4; ++j) numerator[j] = fma(part[component+j], factor, numerator[j]);
    }
    weight = simd_sum(weight);
    for (uint j = 0; j < 4; ++j) {
        const float answer = simd_sum(numerator[j]) / weight;
        if (lane == 0) {
            const uint index = head * 128 + component + j;
            const float g = float(gate[index]);
            const float y = 1.0f / (1.0f + metal::precise::exp(abs(g)));
            result[index] = bfloat(answer * (g < 0.0f ? y : 1.0f-y));
        }
    }
}

#define INK_GATE_SIG \
    device const float *parts [[buffer(0)]], device const bfloat *g0 [[buffer(1)]], \
    device const bfloat *g1 [[buffer(2)]], device const bfloat *g2 [[buffer(3)]], \
    device const bfloat *g3 [[buffer(4)]], device bfloat *o0 [[buffer(5)]], \
    device bfloat *o1 [[buffer(6)]], device bfloat *o2 [[buffer(7)]], \
    device bfloat *o3 [[buffer(8)]], uint2 tile [[threadgroup_position_in_grid]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]

kernel void ink_verify_gate_batched(INK_GATE_SIG) {
    const uint row = tile.y / ink_gated_heads;
    const uint head = tile.y % ink_gated_heads;
    device const bfloat *gates[] = {g0,g1,g2,g3};
    device bfloat *outputs[] = {o0,o1,o2,o3};
    ink_gate_wrapped(parts + row * ink_gated_heads * ink_gated_shards * ink_gated_part,
        gates[row], outputs[row], head, (tile.x*4 + sg)*4, lane);
}
