// SPDX-License-Identifier: Apache-2.0
//! Default prefill dense projection: BF16 operands read directly from device
//! memory as tensor-unit operands, F32 accumulation over the whole K extent
//! in one order, BF16 output. Each output row depends only on its own input
//! row, so the result is the same at every command width.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#ifndef INK_DENSE_BK
#define INK_DENSE_BK 64
#endif
// Rows of the workgroup tile: 32 per pair of SIMD groups.
#ifndef INK_DENSE_BM
#define INK_DENSE_BM 128
#endif

// Workgroup tile INK_DENSE_BM rows x 64 columns; each SIMD group owns 32 x 32 as two
// 16-row fragments. shape = (width, columns, rows, unused).
template <typename Output>
inline void ink_dense_tile(
    device const bfloat *input, device const bfloat *weight, device Output *output, uint4 shape, uint2 group,
    uint simd) {
    constexpr uint BK = INK_DENSE_BK;
    const uint width = shape.x, columns = shape.y, rows = shape.z;
    const uint row0 = group.y * INK_DENSE_BM + (simd / 2) * 32, col0 = group.x * 64 + (simd % 2) * 32;
    if (row0 >= rows || col0 >= columns)
        return;
    const int validCols = int(min(32u, columns - col0));
    const int validUpper = int(min(16u, rows - row0));
    const int validLower = int(min(16u, rows - min(rows, row0 + 16)));
    using Shape = metal::dextents<int32_t, 2>;
    using Strides = metal::array<int32_t, 2>;
    using Tile = metal::tensor<device bfloat, Shape, metal::tensor_inline>;
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, BK, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> product;
    auto upper = product.template get_destination_cooperative_tensor<Tile, Tile, float>();
    auto lower = product.template get_destination_cooperative_tensor<Tile, Tile, float>();
    for (uint i = 0; i < upper.get_capacity(); ++i) {
        upper[i] = 0;
        lower[i] = 0;
    }
    for (uint k = 0; k < width; k += BK) {
        auto rhs = Tile((device bfloat *)weight + col0 * width + k, Shape(BK, validCols), Strides{1, int(width)});
        auto top = Tile((device bfloat *)input + row0 * width + k, Shape(BK, validUpper), Strides{1, int(width)});
        product.run(top, rhs, upper);
        if (validLower > 0) {
            auto bottom =
                Tile((device bfloat *)input + (row0 + 16) * width + k, Shape(BK, validLower), Strides{1, int(width)});
            product.run(bottom, rhs, lower);
        }
    }
    for (uint i = 0; i < upper.get_capacity(); ++i) {
        const auto at = upper.get_multidimensional_index(i);
        const uint column = col0 + at[0], row = row0 + at[1];
        if (column < columns && row < rows)
            output[row * columns + column] = Output(upper[i]);
        if (column < columns && row + 16 < rows)
            output[(row + 16) * columns + column] = Output(lower[i]);
    }
}

// Weight-stationary variant: a workgroup stages each 64-column by 64-K weight
// block in threadgroup memory once (double buffered) and applies it to 256
// rows; each SIMD group owns 64 rows by 32 columns as four 16-row fragments.
// The tensor operation, K blocks and their order per output are those of
// ink_dense_tile, so each output's accumulation is unchanged.
constant constexpr uint INK_STAT_BM = 256, INK_STAT_BN = 64, INK_STAT_BK = 64, INK_STAT_BKP = INK_STAT_BK + 8;

template <typename Output>
inline void ink_dense_stationary_tile(
    device const bfloat *input, device const bfloat *weight, device Output *output, uint4 shape, uint2 group,
    uint simd, uint worker, threadgroup bfloat *staged) {
    constexpr uint BK = INK_STAT_BK;
    const uint width = shape.x, columns = shape.y, rows = shape.z;
    const uint tileCol0 = group.x * INK_STAT_BN;
    const uint row0 = group.y * INK_STAT_BM + (simd / 2) * 64, col0 = tileCol0 + (simd % 2) * 32;
    using Shape = metal::dextents<int32_t, 2>;
    using Strides = metal::array<int32_t, 2>;
    using Tile = metal::tensor<device bfloat, Shape, metal::tensor_inline>;
    using Staged = metal::tensor<threadgroup bfloat, Shape, metal::tensor_inline>;
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, BK, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> product;
    auto acc0 = product.template get_destination_cooperative_tensor<Tile, Staged, float>();
    auto acc1 = product.template get_destination_cooperative_tensor<Tile, Staged, float>();
    auto acc2 = product.template get_destination_cooperative_tensor<Tile, Staged, float>();
    auto acc3 = product.template get_destination_cooperative_tensor<Tile, Staged, float>();
    for (uint i = 0; i < acc0.get_capacity(); ++i) {
        acc0[i] = 0;
        acc1[i] = 0;
        acc2[i] = 0;
        acc3[i] = 0;
    }
    const bool live = row0 < rows && col0 < columns;
    const int validCols = live ? int(min(32u, columns - col0)) : 0;
    // One uint4 (eight BF16) per task: 64 columns by eight K groups.
    auto load = [&](uint k, threadgroup bfloat *buffer) {
        for (uint task = worker; task < INK_STAT_BN * (BK / 8); task += 256) {
            const uint column = task / (BK / 8), kk = (task % (BK / 8)) * 8;
            uint4 value = 0;
            if (tileCol0 + column < columns)
                value = *reinterpret_cast<device const uint4 *>(weight + (tileCol0 + column) * width + k + kk);
            *reinterpret_cast<threadgroup uint4 *>(buffer + column * INK_STAT_BKP + kk) = value;
        }
    };
    load(0, staged);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint current = 0;
    for (uint k = 0; k < width; k += BK) {
        if (k + BK < width)
            load(k + BK, staged + (current ^ 1u) * INK_STAT_BN * INK_STAT_BKP);
        if (live) {
            auto rhs = Staged(staged + current * INK_STAT_BN * INK_STAT_BKP + (simd % 2) * 32 * INK_STAT_BKP,
                              Shape(BK, validCols), Strides{1, int(INK_STAT_BKP)});
#define INK_STAT_RUN(f, acc)                                                                         \
            if (row0 + f * 16 < rows) {                                                              \
                auto lhs = Tile((device bfloat *)input + (row0 + f * 16) * width + k,                \
                                Shape(BK, int(min(16u, rows - (row0 + f * 16)))), Strides{1, int(width)}); \
                product.run(lhs, rhs, acc);                                                          \
            }
            INK_STAT_RUN(0, acc0)
            INK_STAT_RUN(1, acc1)
            INK_STAT_RUN(2, acc2)
            INK_STAT_RUN(3, acc3)
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        current ^= 1u;
    }
    if (!live)
        return;
#define INK_STAT_STORE(f, acc)                                                                       \
    for (uint i = 0; i < acc.get_capacity(); ++i) {                                                  \
        const auto at = acc.get_multidimensional_index(i);                                           \
        const uint column = col0 + at[0], row = row0 + f * 16 + at[1];                               \
        if (column < columns && row < rows)                                                          \
            output[row * columns + column] = Output(acc[i]);                                         \
    }
    INK_STAT_STORE(0, acc0)
    INK_STAT_STORE(1, acc1)
    INK_STAT_STORE(2, acc2)
    INK_STAT_STORE(3, acc3)
}

kernel void ink_dense_stationary(
    device const bfloat *input [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]], constant uint4 &shape [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint worker [[thread_index_in_threadgroup]]) {
    threadgroup uint4 storage[2 * INK_STAT_BN * INK_STAT_BKP / 8];
    ink_dense_stationary_tile<bfloat>(input, weight, output, shape, group, simd, worker,
                                      reinterpret_cast<threadgroup bfloat *>(storage));
}

// Both operands staged in threadgroup memory: a 128-row by 64-column tile
// loads its activation (128 x 64 K) and weight (64 x 64 K) blocks with
// coalesced vector loads once per K block, and eight SIMD groups (32 rows by
// 32 columns each) read them from threadgroup memory. Same tensor operation,
// K blocks and order per output as ink_dense_tile.
template <uint BM, uint BN, uint BK>
inline void ink_dense_staged_tile(
    device const bfloat *input, device const bfloat *weight, device bfloat *output, uint4 shape, uint2 group,
    uint simd, uint worker, threadgroup bfloat *a_tile) {
    constexpr uint BKP = BK + 8, Threads = (BM / 32) * (BN / 32) * 32, ColumnGroups = BN / 32;
    threadgroup bfloat *b_tile = a_tile + BM * BKP;
    const uint width = shape.x, columns = shape.y, rows = shape.z;
    const uint tileRow0 = group.y * BM, tileCol0 = group.x * BN;
    const uint row0 = tileRow0 + (simd / ColumnGroups) * 32, col0 = tileCol0 + (simd % ColumnGroups) * 32;
    using Shape = metal::dextents<int32_t, 2>;
    using Strides = metal::array<int32_t, 2>;
    using Staged = metal::tensor<threadgroup bfloat, Shape, metal::tensor_inline>;
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, BK, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> product;
    auto upper = product.template get_destination_cooperative_tensor<Staged, Staged, float>();
    auto lower = product.template get_destination_cooperative_tensor<Staged, Staged, float>();
    for (uint i = 0; i < upper.get_capacity(); ++i) {
        upper[i] = 0;
        lower[i] = 0;
    }
    const bool live = row0 < rows && col0 < columns;
    const int validCols = live ? int(min(32u, columns - col0)) : 0;
    const int validUpper = live ? int(min(16u, rows - row0)) : 0;
    const int validLower = live ? int(min(16u, rows - min(rows, row0 + 16))) : 0;
    for (uint k = 0; k < width; k += BK) {
        for (uint task = worker; task < (BM + BN) * (BK / 8); task += Threads) {
            const bool is_a = task < BM * (BK / 8);
            const uint local = is_a ? task : task - BM * (BK / 8);
            const uint line = local / (BK / 8), kk = (local % (BK / 8)) * 8;
            uint4 value = 0;
            if (is_a) {
                if (tileRow0 + line < rows)
                    value = *reinterpret_cast<device const uint4 *>(input + (tileRow0 + line) * width + k + kk);
                *reinterpret_cast<threadgroup uint4 *>(a_tile + line * BKP + kk) = value;
            } else {
                if (tileCol0 + line < columns)
                    value = *reinterpret_cast<device const uint4 *>(weight + (tileCol0 + line) * width + k + kk);
                *reinterpret_cast<threadgroup uint4 *>(b_tile + line * BKP + kk) = value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (live) {
            auto rhs = Staged(b_tile + (col0 - tileCol0) * BKP, Shape(BK, validCols), Strides{1, int(BKP)});
            auto top = Staged(a_tile + (row0 - tileRow0) * BKP, Shape(BK, validUpper), Strides{1, int(BKP)});
            product.run(top, rhs, upper);
            if (validLower > 0) {
                auto bottom = Staged(a_tile + (row0 + 16 - tileRow0) * BKP, Shape(BK, validLower), Strides{1, int(BKP)});
                product.run(bottom, rhs, lower);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!live)
        return;
    for (uint i = 0; i < upper.get_capacity(); ++i) {
        const auto at = upper.get_multidimensional_index(i);
        const uint column = col0 + at[0], row = row0 + at[1];
        if (column < columns && row < rows)
            output[row * columns + column] = bfloat(upper[i]);
        if (column < columns && row + 16 < rows)
            output[(row + 16) * columns + column] = bfloat(lower[i]);
    }
}

#define INK_DENSE_STAGED(name, BM, BN, BK)                                                              \
    kernel void name(                                                                                   \
        device const bfloat *input [[buffer(0)]], device const bfloat *weight [[buffer(1)]],            \
        device bfloat *output [[buffer(2)]], constant uint4 &shape [[buffer(3)]],                       \
        uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],    \
        uint worker [[thread_index_in_threadgroup]]) {                                                  \
        threadgroup uint4 storage[(BM + BN) * (BK + 8) / 8];                                            \
        ink_dense_staged_tile<BM, BN, BK>(input, weight, output, shape, group, simd, worker,           \
                                          reinterpret_cast<threadgroup bfloat *>(storage));             \
    }
// Both operands staged in threadgroup memory per tile, with coalesced vector
// loads; SIMD groups of 32 rows by 32 columns read them from threadgroup
// memory. Work per staged operand element is BM x BN / (BM + BN).
INK_DENSE_STAGED(ink_dense_staged, 128, 64, 64)
INK_DENSE_STAGED(ink_dense_staged_256x64, 256, 64, 32)
INK_DENSE_STAGED(ink_dense_staged_256x128, 256, 128, 32)

kernel void ink_dense_fast(
    device const bfloat *input [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]], constant uint4 &shape [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]]) {
    ink_dense_tile<bfloat>(input, weight, output, shape, group, simd);
}

// Count rows whose eight selected experts differ as a set.
kernel void ink_router_compare(
    device const uint *ids [[buffer(0)]], device const uint *reference [[buffer(1)]],
    device atomic_uint *counts [[buffer(2)]], constant uint &rows [[buffer(3)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= rows) return;
    uint missing = 0;
    for (uint i = 0; i < 8; ++i) {
        bool found = false;
        for (uint j = 0; j < 8; ++j) found |= ids[row * 8 + i] == reference[row * 8 + j];
        missing += uint(!found);
    }
    if (missing != 0) atomic_fetch_add_explicit(counts, 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(counts + 1, 1u, memory_order_relaxed);
}
