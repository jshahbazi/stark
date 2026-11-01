// SPDX-License-Identifier: MIT
// Based on the Stark Q5 grouped32 adaptation of ggml-org/llama.cpp b11429,
// d81235049384534c167caea52b85a694f6103d14, kernels/mul_mv.metal Q5_K.
// Copyright (c) 2023-2026 The ggml authors. See LICENSE.ggml-hybrid.
// Original Stark multirow extension: share 32 activation loads and four input
// sums, while each aligned row keeps the current grouped32 F32 chronology.
constant uint hybrid_q5_rows_per_simd [[function_constant(74)]];

kernel void hybrid_q5_multirow_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    const uint first_row = tile * (2 * hybrid_q5_rows_per_simd) + simd_group * hybrid_q5_rows_per_simd;
    if (first_row >= p.rows) return;
    const ulong row_bytes = ulong(p.columns >> 8) * hybrid_packed_block_bytes;
    const ulong first_offset = ulong(first_row) * row_bytes;
    const ulong input_offset = p.group ? ulong(first_row / p.group) * p.columns : 0;
    device const uchar *codes = codes_base + offsets.x;
    const uint last_row = min(first_row + hybrid_q5_rows_per_simd, p.rows) - 1;
    if (p.group && first_row / p.group != last_row / p.group) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < hybrid_q5_rows_per_simd; ++r) {
            if (first_row + r >= p.rows) continue;
            const ulong x = ulong((first_row + r) / p.group) * p.columns;
            const float total = hybrid_packed_block_dot(codes, input, first_offset + r * row_bytes, x, p.columns, lane);
            if (lane == 0) output[first_row + r] = total;
        }
        return;
    }
    const uint tid = lane >> 2;
    const uint parity = lane & 3;
    const uint quarter = tid >> 2;
    const uint low_start = (tid & 3) * 8;
    const uint low_offset = 48 + quarter * 32 + low_start;
    const uint high_offset = 16 + low_start;
    const uint input_start = quarter * 64 + low_start;
    const uint g0 = quarter * 2;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    bool reference_required = false;
    for (uint block_index = parity; block_index < (p.columns >> 8); block_index += 4) {
        device const float *x = input + input_offset + block_index * 256 + input_start;
        float4 values[8];
        float4 input_sum = float4(0.0f);
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j) {
            values[j] = float4(x[j], x[j + 32], x[j + 128], x[j + 160]);
            reference_required = reference_required || any(abs(values[j]) > float4(1e20f));
            input_sum += values[j];
        }
        device const uchar *first_block = codes + first_offset + ulong(block_index) * hybrid_packed_block_bytes;
        #pragma clang loop unroll(full)
        for (uint r = 0; r < hybrid_q5_rows_per_simd; ++r) {
            if (first_row + r >= p.rows) continue;
            device const uchar *block = first_block + r * row_bytes;
            float4 lower = float4(0.0f);
            float4 upper = float4(0.0f);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) {
                const uint low0 = block[low_offset + j];
                const uint low1 = block[low_offset + j + 64];
                const uint high = block[high_offset + j];
                lower.x += values[j].x * float(low0 & 15);
                lower.y += values[j].y * float(low0 >> 4);
                lower.z += values[j].z * float(low1 & 15);
                lower.w += values[j].w * float(low1 >> 4);
                upper.x += high & (1u << g0) ? values[j].x : 0.0f;
                upper.y += high & (1u << (g0 + 1)) ? values[j].y : 0.0f;
                upper.z += high & (1u << (g0 + 4)) ? values[j].z : 0.0f;
                upper.w += high & (1u << (g0 + 5)) ? values[j].w : 0.0f;
            }
            const uint2 sm0 = hybrid_scale_min(block + 4, g0);
            const uint2 sm1 = hybrid_scale_min(block + 4, g0 + 1);
            const uint2 sm2 = hybrid_scale_min(block + 4, g0 + 4);
            const uint2 sm3 = hybrid_scale_min(block + 4, g0 + 5);
            const float4 group_dot = lower + 16.0f * upper;
            sums[r] += hybrid_half(block) * (float(sm0.x) * group_dot.x + float(sm1.x) * group_dot.y
                + float(sm2.x) * group_dot.z + float(sm3.x) * group_dot.w)
                - hybrid_half(block + 2) * (input_sum.x * float(sm0.y) + input_sum.y * float(sm1.y)
                + input_sum.z * float(sm2.y) + input_sum.w * float(sm3.y));
        }
    }
    if (simd_any(reference_required)) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < hybrid_q5_rows_per_simd; ++r) {
            if (first_row + r >= p.rows) continue;
            const float total = hybrid_packed_block_dot(codes, input, first_offset + r * row_bytes, input_offset, p.columns, lane);
            if (lane == 0) output[first_row + r] = total;
        }
        return;
    }
    #pragma clang loop unroll(full)
    for (uint r = 0; r < hybrid_q5_rows_per_simd; ++r) {
        const float total = simd_sum(sums[r]);
        if (lane == 0 && first_row + r < p.rows) output[first_row + r] = total;
    }
}
