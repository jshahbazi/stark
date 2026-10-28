// SPDX-License-Identifier: Apache-2.0
// Original prefetch-free Stark graph RMS, retained only as a bitwise oracle.
kernel void hybrid_graph_rms_scalar_reference(device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]], device float *output [[buffer(2)]],
    constant uint *p [[buffer(3)]], device atomic_uint *failure [[buffer(30)]],
    uint group [[thread_position_in_grid]]) {
    if (group >= p[0]) return;
    uint width = p[1];
    float sum = 0.0f;
    for (uint d = 0; d < width; ++d) {
        float value = input[group * width + d];
        if (!isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        sum += value * value;
    }
    float scale = 1.0f / sqrt(sum / float(width) + as_type<float>(p[2]));
    for (uint d = 0; d < width; ++d) {
        uint index = group * width + d;
        float w = weight[p[3] == width ? d : index];
        float value = input[index] * scale * w;
        if (!isfinite(w) || !isfinite(value)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
        output[index] = value;
    }
}
