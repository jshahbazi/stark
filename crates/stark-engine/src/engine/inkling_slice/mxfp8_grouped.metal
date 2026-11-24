// SPDX-License-Identifier: Apache-2.0

#ifdef INK_GROUP_MXFP8_SOURCE
// Match the pinned MLX gather QMV lane layout. Up uses the aligned fast
// kernel with eight contiguous values per lane. Down uses four. Scale
// each partial dot before adding it to the lane accumulator.
#ifdef INK_MX_SOURCE_VECTOR_LOAD
inline float ink_mxfp8_accumulate_four(float accumulator, float4 input, float4 weight) {
#pragma clang fp contract(off) reassociate(off)
    accumulator += input.x * weight.x;
    accumulator += input.y * weight.y;
    accumulator += input.z * weight.z;
    accumulator += input.w * weight.w;
    return accumulator;
}
#endif

template <uint Width, uint ValuesPerLane, bool Full>
inline float4 ink_mxfp8_source_dot(
    device const float *input, uint4 rows, uint count,
    device const uchar *weights, device const uchar *exponents, uint lane) {
#ifdef INK_MX_SOURCE_VECTOR_LOAD
#pragma clang fp contract(off) reassociate(off)
#endif
    float4 sum = 0.0f;
    const uint4 base = rows * Width;
    for (uint start = 0; start < Width; start += ValuesPerLane * 32) {
        const uint begin = start + lane * ValuesPerLane;
        const uint exponent = uint(exponents[begin / 32]);
        const float scale = exponent == 0u ? 0x1p-127f :
                            exponent == 255u ? NAN : as_type<float>(exponent << 23);
        float4 partial = 0.0f;
#ifdef INK_MX_SOURCE_VECTOR_LOAD
        const device float4 *input4 = reinterpret_cast<const device float4 *>(input);
        const device uchar4 *weight4 = reinterpret_cast<const device uchar4 *>(weights);
        for (uint j = 0; j < ValuesPerLane; j += 4) {
            const uint block = (begin + j) / 4;
            const uchar4 code = weight4[block];
            const float4 weight = float4(ink_mx_e4m3(code.x), ink_mx_e4m3(code.y),
                                         ink_mx_e4m3(code.z), ink_mx_e4m3(code.w));
            partial.x = ink_mxfp8_accumulate_four(partial.x, input4[base.x / 4 + block], weight);
            if (Full || count > 1)
                partial.y = ink_mxfp8_accumulate_four(partial.y, input4[base.y / 4 + block], weight);
            if (Full || count > 2)
                partial.z = ink_mxfp8_accumulate_four(partial.z, input4[base.z / 4 + block], weight);
            if (Full || count > 3)
                partial.w = ink_mxfp8_accumulate_four(partial.w, input4[base.w / 4 + block], weight);
        }
#else
        for (uint j = 0; j < ValuesPerLane; ++j) {
            const uint i = begin + j;
            const float weight = ink_mx_e4m3(weights[i]);
            partial.x += input[base.x + i] * weight;
            if (Full || count > 1) partial.y += input[base.y + i] * weight;
            if (Full || count > 2) partial.z += input[base.z + i] * weight;
            if (Full || count > 3) partial.w += input[base.w + i] * weight;
        }
#endif
        sum += scale * partial;
    }
    return float4(simd_sum(sum.x), simd_sum(sum.y), simd_sum(sum.z), simd_sum(sum.w));
}
#endif

// Share a decoded weight across up to four routes. Each row retains the
// singleton lane's FP32 recurrence and SIMD reduction.
template <uint Width>
inline float4 ink_mxfp8_group_dot(
    device const float *input, uint4 rows, uint count,
    device const uchar *weights, device const uchar *exponents, uint lane) {
#ifdef INK_GROUP_MXFP8_SOURCE
    return ink_mxfp8_source_dot<Width, Width == 2304 ? 8 : 4, false>(
        input, rows, count, weights, exponents, lane);
#else
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
#endif
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

#ifdef INK_GROUP_MXFP8_WIDE
template <uint Width>
__attribute__((always_inline)) inline void ink_mxfp8_group_dot8(
    device const float *input, uint4 first_rows, uint4 second_rows,
    device const uchar *weights, device const uchar *exponents, uint lane,
    thread float4 &first, thread float4 &second) {
#ifdef INK_GROUP_MXFP8_SOURCE
    constexpr uint VALUES = Width == 2304 ? 8 : 4;
    first = ink_mxfp8_source_dot<Width, VALUES, true>(
        input, first_rows, 4, weights, exponents, lane);
    second = ink_mxfp8_source_dot<Width, VALUES, true>(
        input, second_rows, 4, weights, exponents, lane);
#else
    float4 a = 0.0f, b = 0.0f;
    for (uint start = 0; start < Width; start += 32) {
        const uint exponent = uint(exponents[start / 32]);
        float scale = exponent == 0u ? 0x1p-127f : as_type<float>(exponent << 23);
        if (exponent == 255u) scale = NAN;
        const uint i = start + lane;
        const float weight = ink_mx_e4m3(weights[i]) * scale;
        a.x = fma(weight, input[first_rows.x * Width + i], a.x);
        a.y = fma(weight, input[first_rows.y * Width + i], a.y);
        a.z = fma(weight, input[first_rows.z * Width + i], a.z);
        a.w = fma(weight, input[first_rows.w * Width + i], a.w);
        b.x = fma(weight, input[second_rows.x * Width + i], b.x);
        b.y = fma(weight, input[second_rows.y * Width + i], b.y);
        b.z = fma(weight, input[second_rows.z * Width + i], b.z);
        b.w = fma(weight, input[second_rows.w * Width + i], b.w);
    }
    first = float4(simd_sum(a.x), simd_sum(a.y), simd_sum(a.z), simd_sum(a.w));
    second = float4(simd_sum(b.x), simd_sum(b.y), simd_sum(b.z), simd_sum(b.w));
#endif
}

inline uint4 ink_mxfp8_group_routes(device const uint *order, uint start, uint count) {
    uint4 routes = 0;
    for (uint i = 0; i < count; ++i) routes[i] = order[start + i];
    return routes;
}

__attribute__((always_inline)) inline void ink_mxfp8_group_store_up(
    device bfloat *output, device const float *route_weights,
    uint sorted, uint column, uint count, uint4 routes, float4 gates, float4 ups) {
    constexpr uint MID = 1152;
    for (uint i = 0; i < count; ++i) {
        const float gate = float(bfloat(gates[i]));
        const float up = float(bfloat(ups[i]));
#ifdef INK_GROUP_MXFP8_SOURCE
        const float sigmoid_low = 1.0f / (1.0f + metal::precise::exp(abs(gate)));
        const float sigmoid = gate < 0.0f ? sigmoid_low : 1.0f - sigmoid_low;
#else
        const float sigmoid = 1.0f / (1.0f + metal::precise::exp(-gate));
#endif
        const float gate_cap = 4.0f * metal::precise::tanh(gate * 0.25f);
        const float up_cap = 25.0f * metal::precise::tanh(up / 25.0f);
        output[(sorted + i) * MID + column] =
            bfloat(((gate_cap * sigmoid) * up_cap) * route_weights[routes[i]]);
    }
}

kernel void ink_group_mxfp8_wide_up(
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
    uint offset = 0;
    for (; offset + 8 <= rows; offset += 8) {
        const uint sorted = start + offset;
        const uint4 first = ink_mxfp8_group_routes(order, sorted, 4);
        const uint4 second = ink_mxfp8_group_routes(order, sorted + 4, 4);
        float4 gates_first, gates_second, ups_first, ups_second;
        ink_mxfp8_group_dot8<WIDTH>(
            input, first / 8u, second / 8u, weights + weight_row * WIDTH,
            exponents + weight_row * scale_stride, lane, gates_first, gates_second);
        ink_mxfp8_group_dot8<WIDTH>(
            input, first / 8u, second / 8u, weights + (weight_row + 1) * WIDTH,
            exponents + (weight_row + 1) * scale_stride, lane, ups_first, ups_second);
        if (lane == 0) {
            ink_mxfp8_group_store_up(output, route_weights, sorted, column, 4, first, gates_first, ups_first);
            ink_mxfp8_group_store_up(output, route_weights, sorted + 4, column, 4, second, gates_second, ups_second);
        }
    }
    for (; offset < rows; offset += 4) {
        const uint count = min(4u, rows - offset);
        const uint4 routes = ink_mxfp8_group_routes(order, start + offset, count);
        const uint4 inputs = routes / 8u;
        const float4 gates = ink_mxfp8_group_dot<WIDTH>(
            input, inputs, count, weights + weight_row * WIDTH, exponents + weight_row * scale_stride, lane);
        const float4 ups = ink_mxfp8_group_dot<WIDTH>(
            input, inputs, count, weights + (weight_row + 1) * WIDTH, exponents + (weight_row + 1) * scale_stride, lane);
        if (lane == 0) {
            ink_mxfp8_group_store_up(output, route_weights, start + offset, column, count, routes, gates, ups);
        }
    }
}

kernel void ink_group_mxfp8_wide_down(
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
    uint offset = 0;
    for (; offset + 8 <= rows; offset += 8) {
        const uint sorted = start + offset;
        float4 first, second;
        ink_mxfp8_group_dot8<MID>(
            input, sorted + uint4(0, 1, 2, 3), sorted + uint4(4, 5, 6, 7),
            weights + weight_row * MID, exponents + weight_row * (MID / 32), lane, first, second);
        if (lane == 0) {
            for (uint i = 0; i < 4; ++i) {
                output[order[sorted + i] * WIDTH + column] = bfloat(first[i]);
                output[order[sorted + i + 4] * WIDTH + column] = bfloat(second[i]);
            }
        }
    }
    for (; offset < rows; offset += 4) {
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
#endif
