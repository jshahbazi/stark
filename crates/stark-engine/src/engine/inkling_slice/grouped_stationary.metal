// SPDX-License-Identifier: Apache-2.0
//! Default grouped NVFP4 prefill product, weight-stationary: a workgroup
//! decodes each 64-column by 32-K weight block of one expert once (double
//! buffered) and applies it to up to 256 routed rows of that expert. Each
//! SIMD group owns 64 rows by 32 columns as four 16-row fragments. The tensor
//! operation, K blocks and their order per output are those of
//! grouped_fast.metal, and the gate/up activation or down scale is applied on
//! the accumulator registers with the arithmetic of `ink_group_gate_up`
//! (source epilogue) and `ink_group_down_epilogue`.
//! Requires experts.metal and MetalPerformancePrimitives.

struct ink_stat_params {
    uint width;
    float denominator;
};

constant constexpr uint INK_GS_CHANNELS = 2304;
constant constexpr uint INK_GS_MID = 1152;
constant constexpr uint INK_GS_BM = 256;
constant constexpr uint INK_GS_BN = 64;
constant constexpr uint INK_GS_BK = 32;
constant constexpr uint INK_GS_BKP = INK_GS_BK;
constant constexpr uint INK_GS_WORKERS = 256;
constant constexpr uint INK_GS_TILE = INK_GS_BN * INK_GS_BKP;

inline float4 ink_gs_fp4(uint4 nibble) {
    const uint4 magnitude = nibble & 7u;
    const uint4 normal = (((magnitude >> 1) + 126u) << 23) | ((magnitude & 1u) << 22);
    const uint4 small = select(uint4(0), uint4(0x3f000000u), magnitude == 1u);
    return as_type<float4>(select(small, normal, magnitude >= 2u) | ((nibble & 8u) << 28));
}

// Eight scaled nibbles as four packed BF16 pairs; exact as in grouped_fast.
inline uint4 ink_gs_pairs(uint word, float gain) {
    const uint4 even = as_type<uint4>(ink_gs_fp4((uint4(word) >> uint4(0, 8, 16, 24)) & 15u) * gain);
    const uint4 odd = as_type<uint4>(ink_gs_fp4((uint4(word) >> uint4(4, 12, 20, 28)) & 15u) * gain);
    return (even >> 16) | (odd & 0xffff0000u);
}

// One packed word (eight K values) of one column per task: 64 x 4 = 256 tasks.
inline void ink_gs_decode(
    device const uint *weights, device const uchar *scales, uint expert, uint col0, uint offset, uint width,
    uint worker, threadgroup bfloat *decoded) {
    const uint column = worker / (INK_GS_BK / 8), k = (worker % (INK_GS_BK / 8)) * 8;
    const uint wr = expert * INK_GS_CHANNELS + col0 + column;
    const uint at = offset + k;
    const uint word = weights[wr * (width / 8) + at / 8];
    const float gain = ink_e4m3_positive(scales[wr * (width / 16) + at / 16]);
    *reinterpret_cast<threadgroup uint4 *>(decoded + column * INK_GS_BKP + k) = ink_gs_pairs(word, gain);
}

template <bool GateUp, typename Accumulator>
inline void ink_gs_epilogue(
    thread Accumulator &acc, uint rowBase, uint rows, uint start, uint col0, device const uint *order,
    device const float *amax, device const float *route_weights, float bank_max, float denominator,
    device bfloat *output) {
    for (uint i = 0; i < acc.get_capacity(); ++i) {
        const auto at = acc.get_multidimensional_index(i);
        const uint row = rowBase + at[1];
        const uint column = col0 + at[0];
        if (row >= rows)
            continue;
        const uint sorted = start + row;
        const uint route = order[sorted];
        if (GateUp) {
            // Gate and up are adjacent columns held by the same lane.
            if ((column & 1u) || i + 1 >= acc.get_capacity())
                continue;
            const auto next = acc.get_multidimensional_index(i + 1);
            if (next[1] != at[1] || next[0] != at[0] + 1)
                continue;
            const float outer = (amax[route / 8] * bank_max) / (1536.0f * denominator);
            const float gate = float(bfloat(acc[i] * outer));
            const float up = float(bfloat(acc[i + 1] * outer));
            const float low = 1.0f / (1.0f + precise::exp(abs(gate)));
            const float sigmoid = gate < 0.0f ? low : 1.0f - low;
            const float value =
                (4.0f * precise::tanh(gate * 0.25f)) * sigmoid * (25.0f * precise::tanh(up / 25.0f));
            output[sorted * INK_GS_MID + column / 2] = bfloat(value * route_weights[route]);
        } else {
            const float outer = (amax[sorted] * bank_max) / (1536.0f * denominator);
            output[route * INK_GS_CHANNELS + column] = bfloat(acc[i] * outer);
        }
    }
}

template <bool GateUp>
inline void ink_gs_qmm(
    device const bfloat *activation, device const uint *weights, device const uchar *scales,
    device const uint *runs, device const uint *order, device const float *amax,
    device const float *route_weights, device const float *expert_max, device bfloat *output,
    constant ink_stat_params &params, uint2 group, uint worker, uint simd, threadgroup bfloat *shared) {
    constexpr uint BK = INK_GS_BK;
    const uint tile = group.y, start = runs[tile * 3], rows = runs[tile * 3 + 1], expert = runs[tile * 3 + 2];
    if (rows == 0)
        return;
    const uint width = params.width;
    const uint col0 = group.x * INK_GS_BN;
    const uint row0 = (simd / 2) * 64, tileColumn = (simd % 2) * 32;
    using Shape = metal::dextents<int32_t, 2>;
    using Strides = metal::array<int32_t, 2>;
    using ActivationTile = metal::tensor<device bfloat, Shape, metal::tensor_inline>;
    using WeightTile = metal::tensor<threadgroup bfloat, Shape, metal::tensor_inline>;
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, BK, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> product;
    auto acc0 = product.template get_destination_cooperative_tensor<ActivationTile, WeightTile, float>();
    auto acc1 = product.template get_destination_cooperative_tensor<ActivationTile, WeightTile, float>();
    auto acc2 = product.template get_destination_cooperative_tensor<ActivationTile, WeightTile, float>();
    auto acc3 = product.template get_destination_cooperative_tensor<ActivationTile, WeightTile, float>();
    for (uint i = 0; i < acc0.get_capacity(); ++i) {
        acc0[i] = 0;
        acc1[i] = 0;
        acc2[i] = 0;
        acc3[i] = 0;
    }
    ink_gs_decode(weights, scales, expert, col0, 0, width, worker, shared);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint current = 0;
    for (uint offset = 0; offset < width; offset += BK) {
        if (offset + BK < width)
            ink_gs_decode(weights, scales, expert, col0, offset + BK, width, worker,
                          shared + (current ^ 1u) * INK_GS_TILE);
        threadgroup bfloat *decoded = shared + current * INK_GS_TILE;
        auto rhs = WeightTile(decoded + tileColumn * INK_GS_BKP, Shape(BK, 32), Strides{1, int(INK_GS_BKP)});
#define INK_GS_RUN(f, acc)                                                                                  \
        if (row0 + f * 16 < rows) {                                                                         \
            auto lhs = ActivationTile((device bfloat *)activation + (start + row0 + f * 16) * width + offset, \
                                      Shape(BK, int(min(16u, rows - (row0 + f * 16)))), Strides{1, int(width)}); \
            product.run(lhs, rhs, acc);                                                                     \
        }
        INK_GS_RUN(0, acc0)
        INK_GS_RUN(1, acc1)
        INK_GS_RUN(2, acc2)
        INK_GS_RUN(3, acc3)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        current ^= 1u;
    }
    const float bank_max = expert_max[expert];
    const uint baseColumn = col0 + tileColumn;
    ink_gs_epilogue<GateUp>(acc0, row0, rows, start, baseColumn, order, amax, route_weights, bank_max,
                            params.denominator, output);
    ink_gs_epilogue<GateUp>(acc1, row0 + 16, rows, start, baseColumn, order, amax, route_weights, bank_max,
                            params.denominator, output);
    ink_gs_epilogue<GateUp>(acc2, row0 + 32, rows, start, baseColumn, order, amax, route_weights, bank_max,
                            params.denominator, output);
    ink_gs_epilogue<GateUp>(acc3, row0 + 48, rows, start, baseColumn, order, amax, route_weights, bank_max,
                            params.denominator, output);
}

#define INK_GS_ARGS                                                                                    \
    device const bfloat *activation [[buffer(0)]], device const uint *weights [[buffer(1)]],           \
        device const uchar *scales [[buffer(2)]], device const uint *runs [[buffer(3)]],               \
        device const uint *order [[buffer(4)]], device const float *amax [[buffer(5)]],                \
        device const float *route_weights [[buffer(6)]], device const float *expert_max [[buffer(7)]], \
        device bfloat *output [[buffer(8)]], constant ink_stat_params &params [[buffer(9)]],           \
        uint2 group [[threadgroup_position_in_grid]], uint worker [[thread_index_in_threadgroup]],    \
        uint simd [[simdgroup_index_in_threadgroup]]

kernel void ink_group_stationary_gate_up(INK_GS_ARGS) {
    threadgroup uint4 storage[2 * INK_GS_TILE / 8];
    ink_gs_qmm<true>(activation, weights, scales, runs, order, amax, route_weights, expert_max, output, params,
                     group, worker, simd, reinterpret_cast<threadgroup bfloat *>(storage));
}

kernel void ink_group_stationary_down(INK_GS_ARGS) {
    threadgroup uint4 storage[2 * INK_GS_TILE / 8];
    ink_gs_qmm<false>(activation, weights, scales, runs, order, amax, route_weights, expert_max, output, params,
                      group, worker, simd, reinterpret_cast<threadgroup bfloat *>(storage));
}
