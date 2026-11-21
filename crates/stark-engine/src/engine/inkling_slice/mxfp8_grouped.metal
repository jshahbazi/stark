// SPDX-License-Identifier: Apache-2.0

// Share a decoded weight across up to four routes. Each row retains the
// singleton lane's FP32 recurrence and SIMD reduction.
template <uint Width>
inline float4 ink_mxfp8_group_dot(
    device const float *input, uint4 rows, uint count,
    device const uchar *weights, device const uchar *exponents, uint lane) {
    float4 sum = 0.0f;
    for (uint start = 0; start < Width; start += 32) {
        const uint exponent = uint(exponents[start / 32]);
        float scale = exponent == 0u ? 0x1p-127f : as_type<float>(exponent << 23);
        if (exponent == 255u) scale = NAN;
        const uint i = start + lane;
        const float weight = ink_mx_e4m3(weights[i]) * scale;
        sum.x = fma(weight, input[rows.x * Width + i], sum.x);
        if (count > 1) sum.y = fma(weight, input[rows.y * Width + i], sum.y);
        if (count > 2) sum.z = fma(weight, input[rows.z * Width + i], sum.z);
        if (count > 3) sum.w = fma(weight, input[rows.w * Width + i], sum.w);
    }
    return float4(simd_sum(sum.x), simd_sum(sum.y), simd_sum(sum.z), simd_sum(sum.w));
}

kernel void ink_group_mxfp8_up(
    device const float *input [[buffer(0)]], device const uint *order [[buffer(1)]],
    device const uint *runs [[buffer(2)]], device const float *route_weights [[buffer(3)]],
    device const uchar *weights [[buffer(4)]], device const uchar *exponents [[buffer(5)]],
    device bfloat *output [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint WIDTH = 2304;
    constexpr uint MID = 1152;
    const uint start = runs[group.y * 3], rows = runs[group.y * 3 + 1];
    if (rows == 0) return;
    const uint column = group.x * 4 + sg;
    const uint weight_row = (runs[group.y * 3 + 2] * MID + column) * 2;
    const uint scale_stride = WIDTH / 32;
    for (uint offset = 0; offset < rows; offset += 4) {
        const uint count = min(4u, rows - offset);
        uint4 routes = 0;
        for (uint i = 0; i < count; ++i) routes[i] = order[start + offset + i];
        const uint4 inputs = routes / 8u;
        const float4 gates = ink_mxfp8_group_dot<WIDTH>(
            input, inputs, count, weights + weight_row * WIDTH, exponents + weight_row * scale_stride, lane);
        const float4 ups = ink_mxfp8_group_dot<WIDTH>(
            input, inputs, count, weights + (weight_row + 1) * WIDTH, exponents + (weight_row + 1) * scale_stride, lane);
        if (lane == 0) {
            for (uint i = 0; i < count; ++i) {
                const float gate = float(bfloat(gates[i]));
                const float up = float(bfloat(ups[i]));
                const float sigmoid = 1.0f / (1.0f + metal::precise::exp(-gate));
                const float gate_cap = 4.0f * metal::precise::tanh(gate * 0.25f);
                const float up_cap = 25.0f * metal::precise::tanh(up / 25.0f);
                output[(start + offset + i) * MID + column] =
                    bfloat(((gate_cap * sigmoid) * up_cap) * route_weights[routes[i]]);
            }
        }
    }
}

kernel void ink_group_mxfp8_down(
    device const float *input [[buffer(0)]], device const uint *order [[buffer(1)]],
    device const uint *runs [[buffer(2)]], device const uchar *weights [[buffer(3)]],
    device const uchar *exponents [[buffer(4)]], device bfloat *output [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint WIDTH = 2304;
    constexpr uint MID = 1152;
    const uint start = runs[group.y * 3], rows = runs[group.y * 3 + 1];
    if (rows == 0) return;
    const uint column = group.x * 4 + sg;
    const uint weight_row = runs[group.y * 3 + 2] * WIDTH + column;
    for (uint offset = 0; offset < rows; offset += 4) {
        const uint sorted = start + offset;
        const uint count = min(4u, rows - offset);
        const float4 dot = ink_mxfp8_group_dot<MID>(
            input, sorted + uint4(0, 1, 2, 3), count,
            weights + weight_row * MID, exponents + weight_row * (MID / 32), lane);
        if (lane == 0) {
            for (uint i = 0; i < count; ++i)
                output[order[sorted + i] * WIDTH + column] = bfloat(dot[i]);
        }
    }
}
