// SPDX-License-Identifier: Apache-2.0
// Original Inkling 2 Mini MXFP8 activation and checkpoint weight format.
// Equations checked against Ollama 6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8
// mlxrunner/model/inkling2/experts.go.

kernel void ink_mxfp8_activation(
    device const bfloat *input [[buffer(0)]], device float *quantized [[buffer(1)]],
    constant uint &width [[buffer(2)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off) reassociate(off)
    const uint i = group.y * width + group.x * 32 + lane;
    const float x = float(input[i]);
    const float raw_scale = simd_max(abs(x)) * (1.0f / 448.0f);
    const uint bits = as_type<uint>(raw_scale);
    const uint exponent = min(254u, (bits >> 23) + uint((bits & 0x7fffffu) != 0));
    const float minimum = 0x1p-127f;
    const float scale = raw_scale > minimum ? as_type<float>(exponent << 23) : minimum;
    const float inverse = raw_scale > minimum ? 1.0f / scale : 0x1p127f;
    const float rounded = ink_round_e4m3(abs(x * inverse));
    quantized[i] = copysign(rounded, x) * scale;
}

inline float ink_mx_e4m3(uchar encoded) {
    const uint code = uint(encoded) & 0x7fu;
    // E4M3FN reserves both signs of 0x7f for NaN.
    const float magnitude = code == 0x7fu ? NAN : ink_e4m3_positive(code);
    return (encoded & 0x80u) ? -magnitude : magnitude;
}

inline float ink_mxfp8_dot(
    device const float *input, device const uchar *weights, device const uchar *exponents,
    uint width, uint lane) {
    float sum = 0.0f;
    for (uint start = 0; start < width; start += 32) {
        uint exponent = uint(exponents[start / 32]);
        float scale = exponent == 0u ? 0x1p-127f : as_type<float>(exponent << 23);
        if (exponent == 255u) scale = NAN;
        const uint i = start + lane;
        sum = fma(ink_mx_e4m3(weights[i]) * scale, input[i], sum);
    }
    return simd_sum(sum);
}

#ifdef INK_SOURCE_SINGLETON_MXFP8
template <uint ValuesPerLane>
inline float ink_mxfp8_qmv_dot(
    device const float *input, device const uchar *weights,
    device const uchar *exponents, uint width, uint lane) {
    float sum = 0.0f;
    for (uint start = 0; start < width; start += ValuesPerLane * 32) {
        const uint begin = start + lane * ValuesPerLane;
        const uint exponent = uint(exponents[begin / 32]);
        const float scale = exponent == 0u ? 0x1p-127f :
                            exponent == 255u ? NAN : as_type<float>(exponent << 23);
        float partial = 0.0f;
        for (uint j = 0; j < ValuesPerLane; ++j) {
            const uint i = begin + j;
            partial += input[i] * ink_mx_e4m3(weights[i]);
        }
        sum += scale * partial;
    }
    return simd_sum(sum);
}
#endif

kernel void ink_mxfp8_up(
    device const float *input [[buffer(0)]], device const uint *routes [[buffer(1)]],
    device const float *route_weights [[buffer(2)]], device const uchar *weights [[buffer(3)]],
    device const uchar *scales [[buffer(4)]], device bfloat *output [[buffer(5)]],
    constant uint2 &shape [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint column = group.x * 4 + sg;
    const uint route = group.y;
    if (column >= shape.y) return;
    const uint row = (routes[route] * shape.y + column) * 2;
    const uint stride = shape.x, scale_stride = stride / 32;
    // The target rounds each projection independently before its nonlinearities.
#ifdef INK_SOURCE_SINGLETON_MXFP8
    const float gate = float(bfloat(ink_mxfp8_qmv_dot<8>(
        input, weights + row * stride, scales + row * scale_stride, stride, lane)));
    const float up = float(bfloat(ink_mxfp8_qmv_dot<8>(
        input, weights + (row + 1) * stride, scales + (row + 1) * scale_stride, stride, lane)));
    if (lane == 0) {
        const float low = 1.0f / (1.0f + metal::precise::exp(abs(gate)));
        const float sigmoid = gate < 0.0f ? low : 1.0f - low;
#else
    const float gate = float(bfloat(ink_mxfp8_dot(input, weights + row * stride,
                                scales + row * scale_stride, stride, lane)));
    const float up = float(bfloat(ink_mxfp8_dot(input, weights + (row + 1) * stride,
                              scales + (row + 1) * scale_stride, stride, lane)));
    if (lane == 0) {
        const float sigmoid = 1.0f / (1.0f + metal::precise::exp(-gate));
#endif
        const float gate_cap = 4.0f * metal::precise::tanh(gate * 0.25f);
        const float up_cap = 25.0f * metal::precise::tanh(up / 25.0f);
        output[route * shape.y + column] = bfloat(((gate_cap * sigmoid) * up_cap) * route_weights[route]);
    }
}

kernel void ink_mxfp8_down(
    device const float *input [[buffer(0)]], device const uint *routes [[buffer(1)]],
    device const uchar *weights [[buffer(2)]], device const uchar *scales [[buffer(3)]],
    device bfloat *output [[buffer(4)]], constant uint2 &shape [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint column = group.x * 4 + sg;
    const uint route = group.y;
    if (column >= shape.x) return;
    const uint row = routes[route] * shape.x + column;
    const uint stride = shape.y, scale_stride = stride / 32;
#ifdef INK_SOURCE_SINGLETON_MXFP8
    const float dot = ink_mxfp8_qmv_dot<4>(input + route * stride, weights + row * stride,
                                         scales + row * scale_stride, stride, lane);
#else
    const float dot = ink_mxfp8_dot(input + route * stride, weights + row * stride,
                                   scales + row * scale_stride, stride, lane);
#endif
    if (lane == 0) output[route * shape.x + column] = bfloat(dot);
}
