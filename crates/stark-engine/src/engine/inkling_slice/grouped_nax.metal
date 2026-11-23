// SPDX-License-Identifier: Apache-2.0
//! Grouped NVFP4 prefill product for wide commands (96-row expert runs).
//!
//! Derivative of the sorted gathered matmul `gather_db` schedule in mlx-serve
//! `src/src/kernels/moe/gather_qmm_nax_header.metal` at
//! 2fe9be834a601c70ef1a597563026340f0e722fe (https://github.com/ddalcu/mlx-serve),
//! which is ported from oMLX `omlx/patches/m5_gather_qmm_nax.py` at d6b2b92
//! (Copyright oMLX contributors, Apache License 2.0) and adapted from MLX
//! `steel/gemm/nax.h` (Copyright (c) 2025 Apple Inc., MIT License; see
//! metal_reference/LICENSE). Followed from that source: the 96 x 64 tile of six
//! 32 x 32 SIMD groups, 128 of 192 threads dequantizing a 64 x 64 weight block,
//! double-buffered dequantized weights as the only threadgroup storage, the next
//! K step's packed words fetched into registers before the tensor work and
//! dequantized into the other buffer after it with one barrier per step,
//! register-filled 16 x 32 x 16 tensor operations with activations loaded from
//! device memory as four-element vectors, and the GX = 32 tile order.
//! Stark changes: the E2M1 x E4M3 decode of experts.metal / grouped_fast.metal,
//! Stark's run table, and the fused gate/up activation and down scale epilogues
//! of grouped_fast.metal applied to the accumulator registers (same arithmetic
//! as `ink_group_gate_up` with the source epilogue and `ink_group_down_epilogue`).
//! Requires experts.metal, grouped_fast.metal and MetalPerformancePrimitives.

#define INK_NAX_UNROLL _Pragma("clang loop unroll(full)")

constant constexpr uint INK_NAX_BM = 96;
constant constexpr uint INK_NAX_BN = 64;
constant constexpr uint INK_NAX_BK = 64;
constant constexpr uint INK_NAX_BKP = INK_NAX_BK + 8;
constant constexpr uint INK_NAX_THREADS = 192;
constant constexpr uint INK_NAX_LOADERS = 128;
constant constexpr uint INK_NAX_GX = 32;
constant constexpr uint INK_NAX_TILE = INK_NAX_BN * INK_NAX_BKP;

struct ink_nax_params {
    uint width;
    float denominator;
    uint tiles;
};

// Lane coordinates of a 16 x 16 fragment: (column, row) of its first element.
inline short2 ink_nax_coord(ushort lane) {
    const short quad = short(lane >> 2);
    return short2(((quad & 2) | short(lane & 1)) * 4, (quad & 4) | short((lane >> 1) & 3));
}

// One 16 x 32 x 16 multiply-accumulate on register fragments.
inline void ink_nax_mma(
    thread vec<float, 8> &c0, thread vec<float, 8> &c1, thread const vec<bfloat, 8> &a,
    thread const vec<bfloat, 8> &b0, thread const vec<bfloat, 8> &b1) {
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> op;
    auto left = op.template get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    auto right = op.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto result = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(left)>, metal::remove_addrspace_t<decltype(right)>, float>();
    INK_NAX_UNROLL
    for (short i = 0; i < 8; i++) {
        left[i] = a[i];
        right[i] = b0[i];
        right[8 + i] = b1[i];
        result[i] = c0[i];
        result[8 + i] = c1[i];
    }
    op.run(left, right, result);
    INK_NAX_UNROLL
    for (short i = 0; i < 8; i++) {
        c0[i] = result[i];
        c1[i] = result[8 + i];
    }
}

template <bool GateUp>
inline void ink_nax_qmm(
    device const bfloat *activation, device const uint *weights, device const uchar *scales,
    device const uint *runs, device const uint *order, device const float *amax,
    device const float *route_weights, device const float *expert_max, device bfloat *output,
    constant ink_nax_params &params, uint2 group, uint simd, uint lane, threadgroup bfloat *shared) {
    const uint width = params.width;
    const uint columns = INK_FAST_CHANNELS / INK_NAX_BN;
    const uint tile = (group.y / columns) * INK_NAX_GX + group.x;
    if (tile >= params.tiles)
        return;
    const uint start = runs[tile * 3], tileRows = runs[tile * 3 + 1], expert = runs[tile * 3 + 2];
    if (tileRows == 0)
        return;
    const uint col0 = (group.y % columns) * INK_NAX_BN;
    const uint worker = simd * 32 + lane;
    const bool loads = worker < INK_NAX_LOADERS;

    // Loader: weight row worker / 2, 32 values from column (worker % 2) * 32.
    const uint loadRow = worker / 2, loadCol = (worker % 2) * 32;
    const uint weightRow = expert * INK_FAST_CHANNELS + col0 + loadRow;
    device const uint *words = weights + weightRow * (width / 8) + loadCol / 8;
    device const uchar *blockScales = scales + weightRow * (width / 16) + loadCol / 16;
    uint raw[4];
    float gain[2];
    auto fetch = [&](uint step) {
        INK_NAX_UNROLL
        for (uint i = 0; i < 4; ++i)
            raw[i] = words[step * (INK_NAX_BK / 8) + i];
        INK_NAX_UNROLL
        for (uint g = 0; g < 2; ++g)
            gain[g] = ink_e4m3_positive(blockScales[step * (INK_NAX_BK / 16) + g]);
    };
    auto store = [&](threadgroup bfloat *buffer) {
        threadgroup bfloat *destination = buffer + loadRow * INK_NAX_BKP + loadCol;
        INK_NAX_UNROLL
        for (uint i = 0; i < 4; ++i)
            *reinterpret_cast<threadgroup uint4 *>(destination + i * 8) = ink_fast_bf16_pairs(raw[i], gain[i / 2]);
    };

    const int m0 = int(simd / 2) * 32;
    const int rows = min(32, int(tileRows) - m0);
    device const bfloat *xs = activation + (start + uint(max(0, min(m0, int(tileRows) - 1)))) * width;
    const short2 coord = ink_nax_coord(ushort(lane));
    int xOffset[2][2];
    INK_NAX_UNROLL
    for (short i = 0; i < 2; i++)
        INK_NAX_UNROLL
        for (short h = 0; h < 2; h++)
            xOffset[i][h] = min(int(i * 16 + coord.y + h * 8), max(rows, 1) - 1) * int(width) + coord.x;
    const short fragments = rows > 0 ? short((rows + 15) / 16) : short(0);
    const threadgroup bfloat *simdWeights = shared + (simd % 2) * 32 * INK_NAX_BKP;

    vec<float, 8> acc[2][2];
    INK_NAX_UNROLL
    for (short i = 0; i < 2; i++)
        INK_NAX_UNROLL
        for (short n = 0; n < 2; n++)
            acc[i][n] = vec<float, 8>(0.0f);

    const uint steps = width / INK_NAX_BK;
    if (loads) {
        fetch(0);
        store(shared);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 0; step < steps; step++) {
        const bool more = step + 1 < steps;
        if (more && loads)
            fetch(step + 1);
        const threadgroup bfloat *buffer = simdWeights + (step & 1) * INK_NAX_TILE;
        INK_NAX_UNROLL
        for (short sub = 0; sub < short(INK_NAX_BK); sub += 32) {
            // Weight fragments: 32 output columns (two 16-row blocks) by 32 K.
            vec<bfloat, 8> b[2][2];
            INK_NAX_UNROLL
            for (short n = 0; n < 2; n++)
                INK_NAX_UNROLL
                for (short kk = 0; kk < 2; kk++)
                    INK_NAX_UNROLL
                    for (short j = 0; j < 8; j++)
                        b[n][kk][j] = buffer[(n * 16 + coord.y + (j / 4) * 8) * INK_NAX_BKP + sub + kk * 16 + coord.x + j % 4];
            const uint k = step * INK_NAX_BK + uint(sub);
            INK_NAX_UNROLL
            for (short i = 0; i < 2; i++) {
                if (i < fragments) {
                    vec<bfloat, 8> a[2];
                    INK_NAX_UNROLL
                    for (short h = 0; h < 2; h++) {
                        device const bfloat *pointer = xs + xOffset[i][h] + k;
                        const vec<bfloat, 4> low = *reinterpret_cast<device const vec<bfloat, 4> *>(pointer);
                        const vec<bfloat, 4> high = *reinterpret_cast<device const vec<bfloat, 4> *>(pointer + 16);
                        INK_NAX_UNROLL
                        for (short c = 0; c < 4; c++) {
                            a[0][h * 4 + c] = low[c];
                            a[1][h * 4 + c] = high[c];
                        }
                    }
                    INK_NAX_UNROLL
                    for (short kk = 0; kk < 2; kk++)
                        ink_nax_mma(acc[i][0], acc[i][1], a[kk], b[0][kk], b[1][kk]);
                }
            }
        }
        if (more && loads)
            store(shared + ((step + 1) & 1) * INK_NAX_TILE);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Register epilogue. Element j of fragment (i, n) is row
    // m0 + i * 16 + coord.y + (j / 4) * 8, column n * 16 + coord.x + j % 4.
    const float denominator = params.denominator;
    const float bankMax = expert_max[expert];
    const uint colBase = col0 + (simd % 2) * 32;
    INK_NAX_UNROLL
    for (short i = 0; i < 2; i++) {
        INK_NAX_UNROLL
        for (short n = 0; n < 2; n++) {
            INK_NAX_UNROLL
            for (short j = 0; j < 8; j += GateUp ? 2 : 1) {
                const int r = m0 + i * 16 + coord.y + (j / 4) * 8;
                if (r >= int(tileRows))
                    continue;
                const uint sorted = start + uint(r);
                const uint route = order[sorted];
                const uint column = colBase + n * 16 + coord.x + j % 4;
                if (GateUp) {
                    const float outer = (amax[route / 8] * bankMax) / (1536.0f * denominator);
                    const float gate = float(bfloat(acc[i][n][j] * outer));
                    const float up = float(bfloat(acc[i][n][j + 1] * outer));
                    const float low = 1.0f / (1.0f + precise::exp(abs(gate)));
                    const float sigmoid = gate < 0.0f ? low : 1.0f - low;
                    const float value =
                        (4.0f * precise::tanh(gate * 0.25f)) * sigmoid * (25.0f * precise::tanh(up / 25.0f));
                    output[sorted * INK_FAST_MID + column / 2] = bfloat(value * route_weights[route]);
                } else {
                    const float outer = (amax[sorted] * bankMax) / (1536.0f * denominator);
                    output[route * INK_FAST_CHANNELS + column] = bfloat(acc[i][n][j] * outer);
                }
            }
        }
    }
}

#define INK_NAX_ARGS                                                                                    \
    device const bfloat *activation [[buffer(0)]], device const uint *weights [[buffer(1)]],            \
        device const uchar *scales [[buffer(2)]], device const uint *runs [[buffer(3)]],                \
        device const uint *order [[buffer(4)]], device const float *amax [[buffer(5)]],                 \
        device const float *route_weights [[buffer(6)]], device const float *expert_max [[buffer(7)]],  \
        device bfloat *output [[buffer(8)]], constant ink_nax_params &params [[buffer(9)]],             \
        uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],    \
        uint lane [[thread_index_in_simdgroup]]

[[kernel, max_total_threads_per_threadgroup(192)]] void ink_group_nax_gate_up(INK_NAX_ARGS) {
    threadgroup uint4 storage[2 * INK_NAX_TILE / 8];
    ink_nax_qmm<true>(activation, weights, scales, runs, order, amax, route_weights, expert_max, output, params,
                      group, simd, lane, reinterpret_cast<threadgroup bfloat *>(storage));
}

[[kernel, max_total_threads_per_threadgroup(192)]] void ink_group_nax_down(INK_NAX_ARGS) {
    threadgroup uint4 storage[2 * INK_NAX_TILE / 8];
    ink_nax_qmm<false>(activation, weights, scales, runs, order, amax, route_weights, expert_max, output, params,
                       group, simd, lane, reinterpret_cast<threadgroup bfloat *>(storage));
}
