// SPDX-License-Identifier: Apache-2.0
#if __METAL_VERSION__ >= 400

// Keep the same descriptor, per-projection K order and BF16 cast as the
// separate kernels. Independent products reuse the same threadgroup B tile.
template <uint BM, int SG, typename AT>
static void gemm_q4_grouped_swiglu_rows(
        device const uint* gate_codes, device const bfloat* gate_scales,
        device const bfloat* gate_biases, device const uint* up_codes,
        device const bfloat* up_scales, device const bfloat* up_biases,
        thread AT& ta, device bfloat* c, uint K, uint N, uint GS,
        uint m0, uint b_row0, uint n0, uint m_end,
        threadgroup bfloat* b_tile, uint tid) {
    using namespace mpp::tensor_ops;
    constexpr auto desc = matmul2d_descriptor(
        BM, QNAX_BN, QNAX_BK, /*transpose_left=*/false,
        /*transpose_right=*/true, /*relaxed_precision=*/false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, metal::execution_simdgroups<SG>> op;
    using ASlice = decltype(ta.slice(0, 0));
    auto gate = op.template get_destination_cooperative_tensor<ASlice, QnaxTile, float>();
    auto up = op.template get_destination_cooperative_tensor<ASlice, QnaxTile, float>();
    for (uint16_t i = 0; i < gate.get_capacity(); ++i) {
        if (gate.is_valid_element(i)) {
            gate[i] = 0.0f;
            up[i] = 0.0f;
        }
    }

    const uint words = K / 8, groups = K / GS;
    const uint k_full = K - K % QNAX_BK;
    for (uint k0 = 0; k0 < k_full; k0 += QNAX_BK) {
        gemm_q4_grouped_step<BM, SG, QNAX_BK>(
            gate_codes, gate_scales, gate_biases, ta, gate, words, groups, GS,
            m0, b_row0, k0, b_tile, tid);
        gemm_q4_grouped_step<BM, SG, QNAX_BK>(
            up_codes, up_scales, up_biases, ta, up, words, groups, GS,
            m0, b_row0, k0, b_tile, tid);
    }
    if (k_full < K) {
        gemm_q4_grouped_step<BM, SG, QNAX_BK / 2>(
            gate_codes, gate_scales, gate_biases, ta, gate, words, groups, GS,
            m0, b_row0, k_full, b_tile, tid);
        gemm_q4_grouped_step<BM, SG, QNAX_BK / 2>(
            up_codes, up_scales, up_biases, ta, up, words, groups, GS,
            m0, b_row0, k_full, b_tile, tid);
    }

    for (uint16_t i = 0; i < gate.get_capacity(); ++i) {
        if (gate.is_valid_element(i)) {
            auto ix = gate.get_multidimensional_index(i);
            uint row = m0 + uint(ix[1]);
            uint col = n0 + uint(ix[0]);
            if (row < m_end) {
                float g = float(bfloat(gate[i]));
                float s = g / (1.0f + exp(-g));
                c[(ulong)row * N + col] = bfloat(s * float(bfloat(up[i])));
            }
        }
    }
}

template <uint BM, int SG>
static void gemm_q4_grouped_swiglu_body(
        device const uint* gate_codes, device const bfloat* gate_scales,
        device const bfloat* gate_biases, device const uint* up_codes,
        device const bfloat* up_scales, device const bfloat* up_biases,
        device bfloat* a, device bfloat* c, device const uint4* blocks,
        uint K, uint N, uint GS, threadgroup bfloat* b_tile, uint bid, uint tid) {
    const uint4 blk = blocks[bid];
    if (blk.x >= blk.w) {
        return;
    }
    const uint rows = blk.w - blk.x;
    auto ta = metal::tensor(a, metal::dextents<int32_t, 2>(K, int(blk.w)));
#define STARK_GROUPED_SWIGLU_ROWS(HEIGHT) \
    gemm_q4_grouped_swiglu_rows<HEIGHT, SG>(gate_codes, gate_scales, gate_biases, \
        up_codes, up_scales, up_biases, ta, c, K, N, GS, blk.x, blk.y, blk.z, \
        blk.w, b_tile, tid)
    if (rows <= BM / 4) {
        STARK_GROUPED_SWIGLU_ROWS(BM / 4);
    } else if (rows <= BM / 2) {
        STARK_GROUPED_SWIGLU_ROWS(BM / 2);
    } else {
        STARK_GROUPED_SWIGLU_ROWS(BM);
    }
#undef STARK_GROUPED_SWIGLU_ROWS
}

#define STARK_GROUPED_SWIGLU(NAME, BM)                                         \
    kernel void NAME(device const uint* gate_codes      [[buffer(0)]],        \
                     device const bfloat* gate_scales   [[buffer(1)]],        \
                     device const bfloat* gate_biases   [[buffer(2)]],        \
                     device const uint* up_codes        [[buffer(3)]],        \
                     device const bfloat* up_scales     [[buffer(4)]],        \
                     device const bfloat* up_biases     [[buffer(5)]],        \
                     device bfloat* a                   [[buffer(6)]],        \
                     device bfloat* c                   [[buffer(7)]],        \
                     device const uint4* blocks         [[buffer(8)]],        \
                     constant uint& K                   [[buffer(9)]],        \
                     constant uint& N                   [[buffer(10)]],       \
                     constant uint& GS                  [[buffer(11)]],       \
                     uint bid [[threadgroup_position_in_grid]],               \
                     uint tid [[thread_index_in_threadgroup]]) {              \
        threadgroup bfloat b_tile[QNAX_BN * QNAX_LDT];                          \
        gemm_q4_grouped_swiglu_body<BM, 4>(gate_codes, gate_scales, gate_biases,\
            up_codes, up_scales, up_biases, a, c, blocks, K, N, GS, b_tile, bid, tid); \
    }

STARK_GROUPED_SWIGLU(stark_q4_grouped_swiglu_t32, 32)
STARK_GROUPED_SWIGLU(stark_q4_grouped_swiglu_t64, 64)
#undef STARK_GROUPED_SWIGLU
#endif
