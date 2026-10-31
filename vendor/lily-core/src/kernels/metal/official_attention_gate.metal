// SPDX-License-Identifier: MIT
// Copyright (c) 2026 David Dalcu
// Adapted by Stark from MLX-Serve 046b647686419df06c6c7132fb1d15d76185c3e5,
// transformer.zig::STRIDED_SIGMOID_GATE_MUL_SOURCE. See LICENSE.mlx-serve.
// The table producer is the already-qualified MoE scratch producer; this
// kernel introduces no exponential approximation or per-token table creation.
#include <metal_stdlib>
using namespace metal;
kernel void official_attention_sigmoid_mul_bf16(
    device const bfloat* gate [[buffer(0)]],
    device const bfloat* x [[buffer(1)]],
    device const bfloat* table [[buffer(2)]],
    device bfloat* out [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const bfloat sigmoid = table[as_type<ushort>(gate[gid])];
    out[gid] = bfloat(float(x[gid]) * float(sigmoid));
}
