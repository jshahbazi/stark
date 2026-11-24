// SPDX-License-Identifier: Apache-2.0
//! Default grouped NVFP4 prefill product: tensor-unit F32 accumulation with
//! the gate/up activation or the down scale applied at the store, so that
//! BF16 leaves the kernel directly. The epilogue arithmetic matches the
//! separate `ink_group_gate_up` (source epilogue) and
//! `ink_group_down_epilogue`. Requires experts.metal and
//! MetalPerformancePrimitives. Geometry is chosen by the defines below.

#ifndef INK_FAST_BM
#define INK_FAST_BM 64
#endif
#ifndef INK_FAST_BN
#define INK_FAST_BN 32
#endif
#ifndef INK_FAST_BK
#define INK_FAST_BK 32
#endif
// Rows of the tile owned by one SIMD group: 16 (one fragment) or 32 (two).
#ifndef INK_FAST_SG_ROWS
#define INK_FAST_SG_ROWS 16
#endif
#ifndef INK_FAST_DOUBLE
#define INK_FAST_DOUBLE 0
#endif
#ifndef INK_FAST_PAD
#define INK_FAST_PAD 0
#endif
// Diagnostic ceiling variants (never the default path):
// INK_FAST_CEIL_PREDECODED reads BF16 weights already decoded in device
// memory instead of decoding; INK_FAST_CEIL_NODECODE decodes only the first
// K step; INK_FAST_CEIL_TG_ACTIVATION feeds the activation operand from the
// threadgroup buffer instead of device memory; INK_FAST_CEIL_NOMMA skips the
// tensor product.
#ifndef INK_FAST_CEIL_PREDECODED
#define INK_FAST_CEIL_PREDECODED 0
#endif
#ifndef INK_FAST_CEIL_NODECODE
#define INK_FAST_CEIL_NODECODE 0
#endif
#ifndef INK_FAST_CEIL_TG_ACTIVATION
#define INK_FAST_CEIL_TG_ACTIVATION 0
#endif
#ifndef INK_FAST_CEIL_NOMMA
#define INK_FAST_CEIL_NOMMA 0
#endif

struct ink_fast_params {
    uint width;
    float denominator;
};

constant constexpr uint INK_FAST_CHANNELS = 2304;
constant constexpr uint INK_FAST_MID = 1152;
constant constexpr uint INK_FAST_BKP = INK_FAST_BK + INK_FAST_PAD;
constant constexpr uint INK_FAST_COLUMN_GROUPS = INK_FAST_BN / 32;
constant constexpr uint INK_FAST_WORKERS = (INK_FAST_BM / INK_FAST_SG_ROWS) * INK_FAST_COLUMN_GROUPS * 32;
constant constexpr uint INK_FAST_DECODE = INK_FAST_BN * INK_FAST_BKP * (INK_FAST_DOUBLE ? 2 : 1);
constant constexpr uint INK_FAST_RESULT = INK_FAST_BM * INK_FAST_BN * 2;
// Shared storage in BF16 words: decode buffers, aliased by the F32 result.
constant constexpr uint INK_FAST_SHARED_WORDS = INK_FAST_DECODE > INK_FAST_RESULT ? INK_FAST_DECODE : INK_FAST_RESULT;

// Exact E2M1 nibble values as F32 bits: 0, 0.5, 1, 1.5, 2, 3, 4, 6 with sign.
inline float4 ink_fast_fp4(uint4 nibble) {
    const uint4 magnitude = nibble & 7u;
    const uint4 normal = (((magnitude >> 1) + 126u) << 23) | ((magnitude & 1u) << 22);
    const uint4 small = select(uint4(0), uint4(0x3f000000u), magnitude == 1u);
    return as_type<float4>(select(small, normal, magnitude >= 2u) | ((nibble & 8u) << 28));
}

// Eight scaled nibbles as four packed BF16 pairs, lowest nibble first.
inline uint4 ink_fast_bf16_pairs(uint word, float gain) {
    const uint4 even = as_type<uint4>(ink_fast_fp4((uint4(word) >> uint4(0, 8, 16, 24)) & 15u) * gain);
    const uint4 odd = as_type<uint4>(ink_fast_fp4((uint4(word) >> uint4(4, 12, 20, 28)) & 15u) * gain);
    return (even >> 16) | (odd & 0xffff0000u);
}

// Decode one BN-column by BK block: one packed word of eight values per task.
inline void ink_fast_decode(
    device const uint *weights, device const uchar *scales, uint expert, uint col0, uint offset,
    uint width, uint worker, threadgroup bfloat *decoded) {
    for (uint i = worker; i < INK_FAST_BN * (INK_FAST_BK / 8); i += INK_FAST_WORKERS) {
        const uint column = i / (INK_FAST_BK / 8), k = (i % (INK_FAST_BK / 8)) * 8;
        const uint wr = expert * INK_FAST_CHANNELS + col0 + column;
        const uint at = offset + k;
        const uint word = weights[wr * (width / 8) + at / 8];
        const float gain = ink_e4m3_positive(scales[wr * (width / 16) + at / 16]);
        // E2M1 times an E4M3 scale has at most six significant bits, so the
        // F32 product is exact and its upper half is the BF16 word.
        const uint4 low = ink_fast_bf16_pairs(word, gain);
        *reinterpret_cast<threadgroup uint4 *>(decoded + column * INK_FAST_BKP + k) = low;
    }
}

template <bool GateUp>
inline void ink_fast_qmm(
    device const bfloat *activation, device const uint *weights, device const uchar *scales,
    device const uint *runs, device const uint *order, device const float *amax,
    device const float *route_weights, device const float *expert_max,
    device bfloat *output, constant ink_fast_params &params, uint2 group, uint worker, uint simd,
    threadgroup bfloat *shared) {
    constexpr uint BN = INK_FAST_BN, BK = INK_FAST_BK, SR = INK_FAST_SG_ROWS;
    const uint tile = group.y, start = runs[tile * 3], rows = runs[tile * 3 + 1], expert = runs[tile * 3 + 2];
    if (rows == 0)
        return;
    const uint width = params.width;
    const uint col0 = group.x * BN;
    const uint tileRow = (simd / INK_FAST_COLUMN_GROUPS) * SR, tileColumn = (simd % INK_FAST_COLUMN_GROUPS) * 32;
    threadgroup float *final = reinterpret_cast<threadgroup float *>(shared);
    using Shape = metal::dextents<int32_t, 2>;
    using Strides = metal::array<int32_t, 2>;
#if INK_FAST_CEIL_TG_ACTIVATION
    using ActivationTile = metal::tensor<threadgroup bfloat, Shape, metal::tensor_inline>;
#else
    using ActivationTile = metal::tensor<device bfloat, Shape, metal::tensor_inline>;
#endif
#if INK_FAST_CEIL_PREDECODED
    using WeightTile = metal::tensor<device bfloat, Shape, metal::tensor_inline>;
#else
    using WeightTile = metal::tensor<threadgroup bfloat, Shape, metal::tensor_inline>;
#endif
    using ResultTile = metal::tensor<threadgroup float, Shape, metal::tensor_inline>;
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
    const int validLower = SR == 32 ? int(min(16u, rows - min(rows, tileRow + 16))) : 0;
#if INK_FAST_DOUBLE
#if !INK_FAST_CEIL_PREDECODED
    ink_fast_decode(weights, scales, expert, col0, 0, width, worker, shared);
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint current = 0;
#endif
    for (uint offset = 0; offset < width; offset += BK) {
#if INK_FAST_DOUBLE
        if (offset + BK < width && !INK_FAST_CEIL_NODECODE && !INK_FAST_CEIL_PREDECODED)
            ink_fast_decode(weights, scales, expert, col0, offset + BK, width, worker,
                            shared + (current ^ 1u) * BN * INK_FAST_BKP);
        threadgroup bfloat *decoded = shared + current * BN * INK_FAST_BKP;
#else
        ink_fast_decode(weights, scales, expert, col0, offset, width, worker, shared);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup bfloat *decoded = shared;
#endif
#if INK_FAST_CEIL_PREDECODED
        auto rhs = WeightTile((device bfloat *)weights + (expert * INK_FAST_CHANNELS + col0 + tileColumn) * width + offset,
                              Shape(BK, 32), Strides{1, int(width)});
#else
        auto rhs = WeightTile(decoded + tileColumn * INK_FAST_BKP, Shape(BK, 32), Strides{1, int(INK_FAST_BKP)});
#endif
#if INK_FAST_CEIL_TG_ACTIVATION
#define INK_FAST_LHS(row, valid) ActivationTile(decoded + (row % 32) * INK_FAST_BKP, Shape(BK, valid), Strides{1, int(INK_FAST_BKP)})
#else
#define INK_FAST_LHS(row, valid) \
    ActivationTile((device bfloat *)activation + (start + row) * width + offset, Shape(BK, valid), Strides{1, int(width)})
#endif
        if (validUpper > 0) {
            auto lhs = INK_FAST_LHS(tileRow, validUpper);
            if (!INK_FAST_CEIL_NOMMA)
                product.run(lhs, rhs, upper);
        }
        if (validLower > 0) {
            auto lhs = INK_FAST_LHS(tileRow + 16, validLower);
            if (!INK_FAST_CEIL_NOMMA)
                product.run(lhs, rhs, lower);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#if INK_FAST_DOUBLE
        current ^= 1u;
#endif
    }
    upper.store(ResultTile(final + tileRow * BN + tileColumn, Shape(32, 16), Strides{1, BN}));
    if (SR == 32)
        lower.store(ResultTile(final + (tileRow + 16) * BN + tileColumn, Shape(32, 16), Strides{1, BN}));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float denominator = params.denominator;
    const float bank_max = expert_max[expert];
    if (GateUp) {
        for (uint i = worker; i < rows * (BN / 2); i += INK_FAST_WORKERS) {
            const uint r = i / (BN / 2), pair = i % (BN / 2);
            const uint sorted = start + r;
            const uint route = order[sorted];
            const float outer = (amax[route / 8] * bank_max) / (1536.0f * denominator);
            const float gate = float(bfloat(final[r * BN + 2 * pair] * outer));
            const float up = float(bfloat(final[r * BN + 2 * pair + 1] * outer));
            // MLX's compiled GateUpEpilogue F32 unary operations, as in
            // ink_group_gate_up with INK_GROUP_SOURCE_EPILOGUE.
            const float low = 1.0f / (1.0f + precise::exp(abs(gate)));
            const float sigmoid = gate < 0.0f ? low : 1.0f - low;
            const float value =
                (4.0f * precise::tanh(gate * 0.25f)) * sigmoid * (25.0f * precise::tanh(up / 25.0f));
            output[sorted * INK_FAST_MID + col0 / 2 + pair] = bfloat(value * route_weights[route]);
        }
    } else {
        for (uint i = worker; i < rows * BN; i += INK_FAST_WORKERS) {
            const uint r = i / BN, column = i % BN;
            const uint sorted = start + r;
            const uint route = order[sorted];
            const float outer = (amax[sorted] * bank_max) / (1536.0f * denominator);
            output[route * INK_FAST_CHANNELS + col0 + column] = bfloat(final[r * BN + column] * outer);
        }
    }
}

#define INK_FAST_ARGS                                                                                   \
    device const bfloat *activation [[buffer(0)]], device const uint *weights [[buffer(1)]],            \
        device const uchar *scales [[buffer(2)]], device const uint *runs [[buffer(3)]],                \
        device const uint *order [[buffer(4)]], device const float *amax [[buffer(5)]],                 \
        device const float *route_weights [[buffer(6)]], device const float *expert_max [[buffer(7)]],  \
        device bfloat *output [[buffer(8)]], constant ink_fast_params &params [[buffer(9)]],            \
        uint2 group [[threadgroup_position_in_grid]], uint worker [[thread_index_in_threadgroup]],     \
        uint simd [[simdgroup_index_in_threadgroup]]

kernel void ink_group_fast_gate_up(INK_FAST_ARGS) {
    threadgroup float storage[(INK_FAST_SHARED_WORDS + 1) / 2];
    threadgroup bfloat *shared = reinterpret_cast<threadgroup bfloat *>(storage);
    ink_fast_qmm<true>(activation, weights, scales, runs, order, amax, route_weights, expert_max, output, params,
                       group, worker, simd, shared);
}

kernel void ink_group_fast_down(INK_FAST_ARGS) {
    threadgroup float storage[(INK_FAST_SHARED_WORDS + 1) / 2];
    threadgroup bfloat *shared = reinterpret_cast<threadgroup bfloat *>(storage);
    ink_fast_qmm<false>(activation, weights, scales, runs, order, amax, route_weights, expert_max, output, params,
                        group, worker, simd, shared);
}

// Diagnostic only: decode a whole NVFP4 bank to BF16 rows of `width` values,
// one packed word of eight values per thread.
kernel void ink_ceil_decode_bank(
    device const uint *weights [[buffer(0)]], device const uchar *scales [[buffer(1)]],
    device bfloat *out [[buffer(2)]], constant uint &width [[buffer(3)]],
    uint word [[thread_position_in_grid]]) {
    const uint total = 128 * INK_FAST_CHANNELS * (width / 8);
    if (word >= total) return;
    const uint row = word / (width / 8), k = (word % (width / 8)) * 8;
    const float gain = ink_e4m3_positive(scales[row * (width / 16) + k / 16]);
    *reinterpret_cast<device uint4 *>(out + row * width + k) = ink_fast_bf16_pairs(weights[word], gain);
}
