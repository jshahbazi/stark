// SPDX-License-Identifier: Apache-2.0
//! Default grouped NVFP4 prefill product, second design. Structure follows
//! the public description of mlx-serve's `gather_qmm_nax` (ddalcu/mlx-serve
//! at 2fe9be834a601c70ef1a597563026340f0e722fe, itself adapted from MLX
//! steel/gemm/nax.h); no code is copied. Differences from grouped_fast.metal:
//! threadgroup memory holds only the two decoded weight buffers (BK 64, rows
//! padded by eight words), packed words for the next K step are fetched into
//! registers before the tensor work and decoded into the other buffer after
//! it (one barrier per step), and the epilogue runs on the accumulator
//! registers, so no F32 result is staged. The epilogue arithmetic matches
//! `ink_group_gate_up` (source epilogue) and `ink_group_down_epilogue`.
//! Requires experts.metal and MetalPerformancePrimitives.

#ifndef INK_NAX_BM
#define INK_NAX_BM 64
#endif

struct ink_nax_params {
    uint width;
    float denominator;
};

constant constexpr uint INK_NAX_CHANNELS = 2304;
constant constexpr uint INK_NAX_MID = 1152;
constant constexpr uint INK_NAX_BN = 64;
constant constexpr uint INK_NAX_BK = 64;
constant constexpr uint INK_NAX_BKP = INK_NAX_BK + 8;
constant constexpr uint INK_NAX_WORKERS = INK_NAX_BM * 2;
constant constexpr uint INK_NAX_TILE = INK_NAX_BN * INK_NAX_BKP;
// One loader task: 32 K values (four packed words, two E4M3 scales) of one column.
constant constexpr uint INK_NAX_TASKS = INK_NAX_BN * (INK_NAX_BK / 32);

inline float4 ink_nax_fp4(uint4 nibble) {
    const uint4 magnitude = nibble & 7u;
    const uint4 normal = (((magnitude >> 1) + 126u) << 23) | ((magnitude & 1u) << 22);
    const uint4 small = select(uint4(0), uint4(0x3f000000u), magnitude == 1u);
    return as_type<float4>(select(small, normal, magnitude >= 2u) | ((nibble & 8u) << 28));
}

// Eight scaled nibbles as four packed BF16 pairs. E2M1 times E4M3 has at most
// six significant bits, so the F32 product is exact and its upper half is BF16.
inline uint4 ink_nax_pairs(uint word, float gain) {
    const uint4 even = as_type<uint4>(ink_nax_fp4((uint4(word) >> uint4(0, 8, 16, 24)) & 15u) * gain);
    const uint4 odd = as_type<uint4>(ink_nax_fp4((uint4(word) >> uint4(4, 12, 20, 28)) & 15u) * gain);
    return (even >> 16) | (odd & 0xffff0000u);
}

struct ink_nax_fetch {
    uint4 words;
    float gains[2];
};

inline ink_nax_fetch ink_nax_load(
    device const uint *weights, device const uchar *scales, uint expert, uint col0, uint offset, uint width,
    uint task) {
    const uint column = task / (INK_NAX_BK / 32), k = (task % (INK_NAX_BK / 32)) * 32;
    const uint wr = expert * INK_NAX_CHANNELS + col0 + column;
    const uint at = offset + k;
    ink_nax_fetch fetched;
    fetched.words = *reinterpret_cast<device const uint4 *>(weights + wr * (width / 8) + at / 8);
    fetched.gains[0] = ink_e4m3_positive(scales[wr * (width / 16) + at / 16]);
    fetched.gains[1] = ink_e4m3_positive(scales[wr * (width / 16) + at / 16 + 1]);
    return fetched;
}

inline void ink_nax_store(thread const ink_nax_fetch &fetched, uint task, threadgroup bfloat *decoded) {
    const uint column = task / (INK_NAX_BK / 32), k = (task % (INK_NAX_BK / 32)) * 32;
    threadgroup uint4 *out = reinterpret_cast<threadgroup uint4 *>(decoded + column * INK_NAX_BKP + k);
    for (uint w = 0; w < 4; ++w) out[w] = ink_nax_pairs(fetched.words[w], fetched.gains[w / 2]);
}

template <bool GateUp, typename Accumulator>
inline void ink_nax_epilogue(
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
            if (column & 1u)
                continue;
            if (i + 1 >= acc.get_capacity())
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
            output[sorted * INK_NAX_MID + column / 2] = bfloat(value * route_weights[route]);
        } else {
            const float outer = (amax[sorted] * bank_max) / (1536.0f * denominator);
            output[route * INK_NAX_CHANNELS + column] = bfloat(acc[i] * outer);
        }
    }
}

template <bool GateUp>
inline void ink_nax_qmm(
    device const bfloat *activation, device const uint *weights, device const uchar *scales,
    device const uint *runs, device const uint *order, device const float *amax,
    device const float *route_weights, device const float *expert_max, device bfloat *output,
    constant ink_nax_params &params, uint2 group, uint worker, uint simd, threadgroup bfloat *shared) {
    constexpr uint BK = INK_NAX_BK;
    const uint tile = group.y, start = runs[tile * 3], rows = runs[tile * 3 + 1], expert = runs[tile * 3 + 2];
    if (rows == 0)
        return;
    const uint width = params.width;
    const uint col0 = group.x * INK_NAX_BN;
    const uint tileRow = (simd / 2) * 32, tileColumn = (simd % 2) * 32;
    using Shape = metal::dextents<int32_t, 2>;
    using Strides = metal::array<int32_t, 2>;
    using ActivationTile = metal::tensor<device bfloat, Shape, metal::tensor_inline>;
    using WeightTile = metal::tensor<threadgroup bfloat, Shape, metal::tensor_inline>;
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, BK, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> product;
    auto upper = product.template get_destination_cooperative_tensor<ActivationTile, WeightTile, float>();
    auto lower = product.template get_destination_cooperative_tensor<ActivationTile, WeightTile, float>();
    for (uint i = 0; i < upper.get_capacity(); ++i) {
        upper[i] = 0;
        lower[i] = 0;
    }
    const int validUpper = int(min(16u, rows - min(rows, tileRow)));
    const int validLower = int(min(16u, rows - min(rows, tileRow + 16)));
    const bool loader = worker < INK_NAX_TASKS;
    ink_nax_fetch fetched;
    if (loader) {
        fetched = ink_nax_load(weights, scales, expert, col0, 0, width, worker);
        ink_nax_store(fetched, worker, shared);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint current = 0;
    for (uint offset = 0; offset < width; offset += BK) {
        const bool more = offset + BK < width;
        if (loader && more)
            fetched = ink_nax_load(weights, scales, expert, col0, offset + BK, width, worker);
        threadgroup bfloat *decoded = shared + current * INK_NAX_TILE;
        auto rhs = WeightTile(decoded + tileColumn * INK_NAX_BKP, Shape(BK, 32), Strides{1, int(INK_NAX_BKP)});
        if (validUpper > 0) {
            auto lhs = ActivationTile((device bfloat *)activation + (start + tileRow) * width + offset,
                                      Shape(BK, validUpper), Strides{1, int(width)});
            product.run(lhs, rhs, upper);
        }
        if (validLower > 0) {
            auto lhs = ActivationTile((device bfloat *)activation + (start + tileRow + 16) * width + offset,
                                      Shape(BK, validLower), Strides{1, int(width)});
            product.run(lhs, rhs, lower);
        }
        if (loader && more)
            ink_nax_store(fetched, worker, shared + (current ^ 1u) * INK_NAX_TILE);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        current ^= 1u;
    }
    const float bank_max = expert_max[expert];
    const uint baseColumn = col0 + tileColumn;
    ink_nax_epilogue<GateUp>(upper, tileRow, rows, start, baseColumn, order, amax, route_weights, bank_max,
                             params.denominator, output);
    ink_nax_epilogue<GateUp>(lower, tileRow + 16, rows, start, baseColumn, order, amax, route_weights, bank_max,
                             params.denominator, output);
}

#define INK_NAX_ARGS                                                                                   \
    device const bfloat *activation [[buffer(0)]], device const uint *weights [[buffer(1)]],           \
        device const uchar *scales [[buffer(2)]], device const uint *runs [[buffer(3)]],               \
        device const uint *order [[buffer(4)]], device const float *amax [[buffer(5)]],                \
        device const float *route_weights [[buffer(6)]], device const float *expert_max [[buffer(7)]], \
        device bfloat *output [[buffer(8)]], constant ink_nax_params &params [[buffer(9)]],            \
        uint2 group [[threadgroup_position_in_grid]], uint worker [[thread_index_in_threadgroup]],    \
        uint simd [[simdgroup_index_in_threadgroup]]

kernel void ink_group_nax_gate_up(INK_NAX_ARGS) {
    threadgroup uint4 storage[2 * INK_NAX_TILE / 8];
    threadgroup bfloat *shared = reinterpret_cast<threadgroup bfloat *>(storage);
    ink_nax_qmm<true>(activation, weights, scales, runs, order, amax, route_weights, expert_max, output, params,
                      group, worker, simd, shared);
}

kernel void ink_group_nax_down(INK_NAX_ARGS) {
    threadgroup uint4 storage[2 * INK_NAX_TILE / 8];
    threadgroup bfloat *shared = reinterpret_cast<threadgroup bfloat *>(storage);
    ink_nax_qmm<false>(activation, weights, scales, runs, order, amax, route_weights, expert_max, output, params,
                       group, worker, simd, shared);
}
