// SPDX-License-Identifier: Apache-2.0
#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/softmax.h"
#include "mlx/backend/metal/kernels/binary_ops.h"
#include "mlx/backend/metal/kernels/binary.h"
instantiate_kernel("block_softmax_float32", softmax_single_row, float)
instantiate_kernel("vs_LogAddExpfloat32", binary_vs, float, float, LogAddExp, 1)

// Match MLX LogAddExp, four reads per lane, SIMD normalization and scaling
// without materializing the intermediate logits in device buffers.
kernel void ink_singleton_route_weights(
    device const float *logits [[buffer(0)]], device const float *bias [[buffer(1)]],
    device uint *ids [[buffer(2)]], device float *weights [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float choices[128];
    threadgroup uint selected[8];
    const float x = logits[lane];
    const float low = 1.0f / (1.0f + precise::exp(abs(x)));
    choices[lane] = (x < 0.0f ? low : 1.0f - low) + bias[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        for (uint route = 0; route < 8; ++route) {
            uint expert = 0;
            for (uint j = 1; j < 128; ++j) {
                if (choices[j] > choices[expert]) expert = j;
            }
            choices[expert] = -INFINITY;
            selected[route] = expert;
            ids[route] = expert;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane >= 32) return;

    float scores[4];
    float local_max = Limits<float>::finite_min;
    for (uint j = 0; j < 4; ++j) {
        scores[j] = lane < 2 ? -LogAddExp{}(-logits[selected[lane * 4 + j]], 0.0f) : -INFINITY;
        local_max = max(local_max, scores[j]);
    }
    const float peak = simd_max(local_max);
    float total = 0.0f;
    for (uint j = 0; j < 4; ++j) {
        scores[j] = softmax_exp(scores[j] - peak);
        total += scores[j];
    }
    const float inverse = 1.0f / simd_sum(total);
    if (lane < 2) {
        for (uint j = 0; j < 4; ++j) weights[lane * 4 + j] = (scores[j] * inverse) * 8.0f;
    }
}
