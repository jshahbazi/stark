// SPDX-License-Identifier: MIT
// Adapted from the F32 lane mapping, clamped tails, and K-slice reduction in
// packed_kquant_mma_matvec.metal, itself adapted from ggml-org/llama.cpp
// dde879bc4f954b809892fd0d660515f42536394a,
// ggml/src/ggml-metal/kernels/mul_mv_mma.metal.
// Copyright (c) 2023-2026 The ggml authors. See LICENSE.ggml-hybrid.
// Original Stark adaptations: GPU expert/route maps, gathered token or route
// inputs, IQ2_XS/IQ3_XXS byte-safe hybrid_quant, scattered route-major outputs,
// and the exact vector8 fallback. All fragments and accumulators remain F32.
// Runtime admission requires separately qualified device and matrix shapes.

inline float hybrid_iq_batch_vector8_dot(
    device const uchar *codes, device const float *input,
    ulong row_offset, ulong input_offset, uint columns, uint lane) {
    float sum = 0.0f;
    for (uint col = lane * 8; col < columns; col += 256) {
        device const uchar *block = codes + row_offset + ulong(col >> 8) * hybrid_packed_block_bytes;
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j)
            sum += hybrid_quant(block, hybrid_packed_kind, (col & 255) + j) * input[input_offset + col + j];
    }
    return simd_sum(sum);
}

kernel void hybrid_graph_iq_indexed_batch_mma(
    device const uchar *codes [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint *p [[buffer(3)]],
    device const uint *maps [[buffer(4)]],
    device const uint *descriptors [[buffer(5)]],
    device const uint *descriptor_count [[buffer(6)]],
    device atomic_uint *failure [[buffer(30)]],
    threadgroup float *red [[threadgroup(0)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    if (tile.y >= descriptor_count[0]) return;
    const uint rows = p[0], columns = p[1];
    const uint nsg = hybrid_kquant_mma_nsg;
    const uint expert = descriptors[3 * tile.y];
    const uint first = descriptors[3 * tile.y + 1];
    const uint count = descriptors[3 * tile.y + 2];
    const uint first_row = tile.x * 32;
    const uint fm = hybrid_kquant_mma_fm(lane);
    const uint fn = hybrid_kquant_mma_fn(lane);
    const uint tid = sg * 32 + lane;
    const ulong row_bytes = ulong(columns / 256) * hybrid_packed_block_bytes;
    const uint route0 = maps[first + min(fn, count - 1)];
    const uint route1 = maps[first + min(fn + 1, count - 1)];
    const ulong input0 = ulong(p[5] ? route0 : route0 / p[3]) * columns;
    const ulong input1 = ulong(p[5] ? route1 : route1 / p[3]) * columns;
    threadgroup atomic_uint reference;
    if (tid == 0) atomic_store_explicit(&reference, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 accum[4];
    #pragma clang loop unroll(full)
    for (uint t = 0; t < 4; ++t) accum[t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    bool extreme = false, nonfinite = false;
    for (uint chunk = sg; chunk < columns / 64; chunk += nsg) {
        simdgroup_float8x8 b[8];
        const uint at = chunk * 64 + 16 * (fm / 2) + 8 * (fm % 2);
        #pragma clang loop unroll(full)
        for (uint s = 0; s < 8; ++s) {
            const float x0 = input[input0 + at + s];
            const float x1 = input[input1 + at + s];
            extreme = extreme || abs(x0) > 1e20f || abs(x1) > 1e20f;
            nonfinite = nonfinite || !isfinite(x0) || !isfinite(x1);
            b[s].thread_elements()[0] = x0;
            b[s].thread_elements()[1] = x1;
        }
        const uint ci = chunk * 4 + fn / 2;
        #pragma clang loop unroll(full)
        for (uint t = 0; t < 4; ++t) {
            const uint row = min(first_row + t * 8 + fm, rows - 1);
            const uint first_k = ci * 16;
            device const uchar *block = codes + (ulong(expert) * rows + row) * row_bytes
                                              + ulong(first_k / 256) * hybrid_packed_block_bytes;
            float4x4 weights;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 16; ++j)
                weights[j / 4][j % 4] = hybrid_quant(block, hybrid_packed_kind, first_k % 256 + j);
            #pragma clang loop unroll(full)
            for (uint s = 0; s < 8; ++s) {
                simdgroup_float8x8 a;
                a.thread_elements()[0] = weights[s / 4][s % 4];
                a.thread_elements()[1] = weights[s / 4 + 2][s % 4];
                simdgroup_multiply_accumulate(accum[t], a, b[s], accum[t]);
            }
        }
    }
    if (extreme || nonfinite) atomic_store_explicit(&reference, 1u, memory_order_relaxed);
    if (nonfinite) atomic_store_explicit(failure, 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&reference, memory_order_relaxed)) {
        for (uint flat = sg; flat < 32 * 8; flat += nsg) {
            const uint row = first_row + flat % 32;
            const uint slot = flat / 32;
            if (row < rows && slot < count) {
                const uint route = maps[first + slot];
                const ulong input_offset = ulong(p[5] ? route : route / p[3]) * columns;
                const float total = hybrid_iq_batch_vector8_dot(codes, input,
                    (ulong(expert) * rows + row) * row_bytes, input_offset, columns, lane);
                if (lane == 0) {
                    if (!isfinite(total)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
                    output[ulong(route) * rows + row] = total;
                }
            }
        }
        return;
    }
    #pragma clang loop unroll(full)
    for (uint t = 0; t < 4; ++t) {
        red[(sg * 4 + t) * 64 + 2 * lane] = accum[t].thread_elements()[0];
        red[(sg * 4 + t) * 64 + 2 * lane + 1] = accum[t].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < 4 * 64; index += nsg * 32) {
        float sum = 0.0f;
        for (uint split = 0; split < nsg; ++split) sum += red[split * 4 * 64 + index];
        const uint t = index / 64;
        const uint owner_lane = (index % 64) / 2;
        const uint row = first_row + t * 8 + hybrid_kquant_mma_fm(owner_lane);
        const uint slot = hybrid_kquant_mma_fn(owner_lane) + index % 2;
        if (row < rows && slot < count) {
            if (!isfinite(sum)) atomic_store_explicit(failure, 1u, memory_order_relaxed);
            output[ulong(maps[first + slot]) * rows + row] = sum;
        }
    }
}
