// SPDX-License-Identifier: Apache-2.0
// Single-row NVFP4 expert projections with several output channels per SIMD
// group. Compiled after experts.metal with INK_NVFP4_VECTOR_BLOCK and
// INK_SOURCE_SINGLETON_EPILOGUE, and computing every output exactly as
// ink_nvfp4_vector_up / ink_nvfp4_vector_down do: lane L owns blocks L, L + 32,
// ..., each block dot is the same four-way vector expression, the lane sums
// are the same fused scaled recurrence in block order, and the SIMD reduction
// and epilogue are unchanged. A lane loads its activation blocks once and
// reuses them for every channel. `Compact` keeps those registers as half,
// which is exact only if every activation is a half value; the bitwise
// screen against the incumbent kernels decides whether a variant is usable.

template <bool Compact>
struct ink_dx_activation {
    using type = float4;
};
template <>
struct ink_dx_activation<true> {
    using type = half4;
};

template <uint Width, bool Compact>
struct ink_dx_lane {
    enum : uint { blocks = Width / 16, per_lane = (Width / 16 + 31) / 32 };
    typename ink_dx_activation<Compact>::type activation[per_lane][4];

    void load(device const float *input, uint lane) {
#pragma clang loop unroll(full)
        for (uint i = 0; i < per_lane; ++i) {
            const uint block = lane + i * 32;
            if (block < blocks) {
                const device float4 *v = reinterpret_cast<device const float4 *>(input + block * 16);
#pragma clang loop unroll(full)
                for (uint q = 0; q < 4; ++q) activation[i][q] = typename ink_dx_activation<Compact>::type(v[q]);
            }
        }
    }

    float dot_row(device const uchar *row_codes, device const uchar *row_scales, uint lane) const {
        float sum = 0.0f;
#pragma clang loop unroll(full)
        for (uint i = 0; i < per_lane; ++i) {
            const uint block = lane + i * 32;
            if (block < blocks) {
                const uint2 packed = reinterpret_cast<device const uint2 *>(row_codes)[block];
                const float scale = ink_e4m3_positive(row_scales[block]);
                const float block_dot = (dot(ink_nvfp4_nibbles(packed.x), float4(activation[i][0])) +
                                         dot(ink_nvfp4_nibbles(packed.x >> 16), float4(activation[i][1]))) +
                                        (dot(ink_nvfp4_nibbles(packed.y), float4(activation[i][2])) +
                                         dot(ink_nvfp4_nibbles(packed.y >> 16), float4(activation[i][3])));
                sum = fma(block_dot, scale, sum);
            }
        }
        return sum;
    }
};

template <uint Channels, bool Compact>
inline void ink_dx_up(
    device const float *inner, device const float *amax, device const uint *ids, device const float *route_weights,
    device const uchar *codes, device const uchar *scales, device const float *expert_max, device bfloat *middle,
    constant InkExpertShape &shape, uint2 group, uint lane, uint simd_group) {
    constexpr uint width = 2304;
    const uint first_column = (group.x * 4 + simd_group) * Channels;
    const uint route = group.y;
    if (first_column >= shape.intermediate) return;
    const uint expert = ids[route];
    ink_dx_lane<width, Compact> state;
    state.load(inner, lane);
    float sums[Channels * 2];
#pragma clang loop unroll(full)
    for (uint r = 0; r < Channels * 2; ++r) {
        const uint row = (expert * shape.intermediate + first_column) * 2 + r;
        sums[r] = state.dot_row(codes + row * (width / 2), scales + row * (width / 16), lane);
    }
    const float outer = (amax[0] * expert_max[expert]) / (1536.0f * shape.denominator);
#pragma clang loop unroll(full)
    for (uint c = 0; c < Channels; ++c) {
        const float gate_dot = simd_sum(sums[2 * c]);
        const float up_dot = simd_sum(sums[2 * c + 1]);
        if (lane == 0) {
            const float gate = float(bfloat(gate_dot * outer));
            const float up = float(bfloat(up_dot * outer));
            const float lo = 1.0f / (1.0f + metal::precise::exp(abs(gate)));
            const float sigmoid = gate < 0.0f ? lo : 1.0f - lo;
            const float cap_gate = 4.0f * metal::precise::tanh(gate * 0.25f);
            const float cap_up = 25.0f * metal::precise::tanh(up / 25.0f);
            const float act = (cap_gate * sigmoid) * cap_up;
            middle[route * shape.intermediate + first_column + c] = bfloat(act * route_weights[route]);
        }
    }
}

template <uint Channels, bool Compact>
inline void ink_dx_down(
    device const float *inner, device const float *amax, device const uint *ids, device const uchar *codes,
    device const uchar *scales, device const float *expert_max, device bfloat *down, constant InkExpertShape &shape,
    uint2 group, uint lane, uint simd_group) {
    constexpr uint width = 1152;
    const uint first_column = (group.x * 4 + simd_group) * Channels;
    const uint route = group.y;
    if (first_column >= shape.width) return;
    const uint expert = ids[route];
    ink_dx_lane<width, Compact> state;
    state.load(inner + route * shape.intermediate, lane);
    float sums[Channels];
#pragma clang loop unroll(full)
    for (uint c = 0; c < Channels; ++c) {
        const uint row = expert * shape.width + first_column + c;
        sums[c] = state.dot_row(codes + row * (width / 2), scales + row * (width / 16), lane);
    }
    const float outer = (amax[route] * expert_max[expert]) / (1536.0f * shape.denominator);
#pragma clang loop unroll(full)
    for (uint c = 0; c < Channels; ++c) {
        const float dot = simd_sum(sums[c]);
        if (lane == 0) down[route * shape.width + first_column + c] = bfloat(dot * outer);
    }
}

#define INK_DX_UP(NAME, CHANNELS, COMPACT)                                                                       \
    kernel void NAME(device const float *inner [[buffer(0)]], device const float *amax [[buffer(1)]],           \
                     device const uint *ids [[buffer(2)]], device const float *route_weights [[buffer(3)]],     \
                     device const uchar *codes [[buffer(4)]], device const uchar *scales [[buffer(5)]],         \
                     device const float *expert_max [[buffer(6)]], device bfloat *middle [[buffer(7)]],         \
                     constant InkExpertShape &shape [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]], \
                     uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) { \
        ink_dx_up<CHANNELS, COMPACT>(inner, amax, ids, route_weights, codes, scales, expert_max, middle, shape,  \
                                     group, lane, simd_group);                                                   \
    }
#define INK_DX_DOWN(NAME, CHANNELS, COMPACT)                                                                     \
    kernel void NAME(device const float *inner [[buffer(0)]], device const float *amax [[buffer(1)]],           \
                     device const uint *ids [[buffer(2)]], device const uchar *codes [[buffer(3)]],             \
                     device const uchar *scales [[buffer(4)]], device const float *expert_max [[buffer(5)]],    \
                     device bfloat *down [[buffer(6)]], constant InkExpertShape &shape [[buffer(7)]],           \
                     uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],     \
                     uint simd_group [[simdgroup_index_in_threadgroup]]) {                                      \
        ink_dx_down<CHANNELS, COMPACT>(inner, amax, ids, codes, scales, expert_max, down, shape, group, lane,   \
                                       simd_group);                                                              \
    }

INK_DX_UP(ink_nvfp4_decode_up, 4, false)
INK_DX_DOWN(ink_nvfp4_decode_down, 4, false)
INK_DX_UP(ink_dx_up_c1, 1, false)
INK_DX_DOWN(ink_dx_down_c1, 1, false)
INK_DX_UP(ink_dx_up_c2, 2, false)
INK_DX_DOWN(ink_dx_down_c2, 2, false)
INK_DX_UP(ink_dx_up_c1h, 1, true)
INK_DX_DOWN(ink_dx_down_c1h, 1, true)
INK_DX_UP(ink_dx_up_c2h, 2, true)
INK_DX_DOWN(ink_dx_down_c2h, 2, true)
INK_DX_UP(ink_dx_up_c4h, 4, true)
INK_DX_DOWN(ink_dx_down_c4h, 4, true)

// The incumbent one-channel structure reading a half copy of the activation
// (exact: every activation is a half value), which halves the activation
// traffic per weight byte.
inline float ink_dx_half_lane_dot(
    device const half *input, device const uchar *codes, device const uchar *scales, uint width, uint lane) {
    float sum = 0.0f;
    for (uint block = lane; block < width / 16; block += 32) {
        const uint2 packed = reinterpret_cast<device const uint2 *>(codes)[block];
        const float scale = ink_e4m3_positive(scales[block]);
        const device half4 *v = reinterpret_cast<device const half4 *>(input + block * 16);
        const float block_dot =
            (dot(ink_nvfp4_nibbles(packed.x), float4(v[0])) + dot(ink_nvfp4_nibbles(packed.x >> 16), float4(v[1]))) +
            (dot(ink_nvfp4_nibbles(packed.y), float4(v[2])) + dot(ink_nvfp4_nibbles(packed.y >> 16), float4(v[3])));
        sum = fma(block_dot, scale, sum);
    }
    return sum;
}

kernel void ink_dx_up_hb(
    device const half *inner [[buffer(0)]], device const float *amax [[buffer(1)]],
    device const uint *ids [[buffer(2)]], device const float *route_weights [[buffer(3)]],
    device const uchar *codes [[buffer(4)]], device const uchar *scales [[buffer(5)]],
    device const float *expert_max [[buffer(6)]], device bfloat *middle [[buffer(7)]],
    constant InkExpertShape &shape [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint column = group.x * 4 + simd_group;
    const uint route = group.y;
    if (column >= shape.intermediate) return;
    const uint expert = ids[route];
    const uint row = (expert * shape.intermediate + column) * 2;
    const uint code_stride = shape.width / 2;
    const uint scale_stride = shape.width / 16;
    const float gate_dot = simd_sum(ink_dx_half_lane_dot(
        inner, codes + row * code_stride, scales + row * scale_stride, shape.width, lane));
    const float up_dot = simd_sum(ink_dx_half_lane_dot(
        inner, codes + (row + 1) * code_stride, scales + (row + 1) * scale_stride, shape.width, lane));
    if (lane == 0) {
        const float outer = (amax[0] * expert_max[expert]) / (1536.0f * shape.denominator);
        const float gate = float(bfloat(gate_dot * outer));
        const float up = float(bfloat(up_dot * outer));
        const float lo = 1.0f / (1.0f + metal::precise::exp(abs(gate)));
        const float sigmoid = gate < 0.0f ? lo : 1.0f - lo;
        const float cap_gate = 4.0f * metal::precise::tanh(gate * 0.25f);
        const float cap_up = 25.0f * metal::precise::tanh(up / 25.0f);
        const float act = (cap_gate * sigmoid) * cap_up;
        middle[route * shape.intermediate + column] = bfloat(act * route_weights[route]);
    }
}

kernel void ink_dx_down_hb(
    device const half *inner [[buffer(0)]], device const float *amax [[buffer(1)]],
    device const uint *ids [[buffer(2)]], device const uchar *codes [[buffer(3)]],
    device const uchar *scales [[buffer(4)]], device const float *expert_max [[buffer(5)]],
    device bfloat *down [[buffer(6)]], constant InkExpertShape &shape [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint column = group.x * 4 + simd_group;
    const uint route = group.y;
    if (column >= shape.width) return;
    const uint expert = ids[route];
    const uint row = expert * shape.width + column;
    const uint code_stride = shape.intermediate / 2;
    const uint scale_stride = shape.intermediate / 16;
    const float dot = simd_sum(ink_dx_half_lane_dot(
        inner + route * shape.intermediate, codes + row * code_stride, scales + row * scale_stride,
        shape.intermediate, lane));
    if (lane == 0) {
        const float outer = (amax[route] * expert_max[expert]) / (1536.0f * shape.denominator);
        down[route * shape.width + column] = bfloat(dot * outer);
    }
}

kernel void ink_dx_to_half(
    device const float *input [[buffer(0)]], device half *output [[buffer(1)]], constant uint &count [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i < count) output[i] = half(input[i]);
}
