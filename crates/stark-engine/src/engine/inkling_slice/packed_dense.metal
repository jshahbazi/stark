// SPDX-License-Identifier: MIT
// Copyright © 2026 Apple Inc. (LICENSE.mlx-reference).
// Uses the MLX-derived dot in prefill.metal.

kernel void ink_packed_matvec(
    device const bfloat4 *weight [[buffer(0)]], device const bfloat4 *x [[buffer(1)]],
    device bfloat *y [[buffer(2)]], constant uint4 &shape [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint batch = 8;
    const uint live = min(batch, shape.z - batch * group.y);
    device const bfloat4 *inputs[batch];
    device bfloat *outputs[batch];
    for (uint i = 0; i < batch; ++i) {
        const uint row = batch * group.y + min(i, live - 1);
        inputs[i] = x + row * (shape.x / 4);
        outputs[i] = y + row * shape.y;
    }
    ink_matvec_prefill_shared<batch>(weight, inputs, outputs, uint3(shape.xy, live), group.x, simd, lane);
}
