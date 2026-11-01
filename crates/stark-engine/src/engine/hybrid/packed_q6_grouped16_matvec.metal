// SPDX-License-Identifier: MIT
// Adapted from ggml-org/llama.cpp b11429, commit
// d81235049384534c167caea52b85a694f6103d14,
// ggml/src/ggml-metal/kernels/mul_mv.metal, Q6_K implementation (1756-1850).
// Copyright (c) 2023-2026 The ggml authors. See LICENSE.ggml-hybrid.
// Stark adaptations: checked byte addressing, batch/tail handling, Safe F32
// inputs, explicit original-block fallback and test-only bounded specialization.
// Scale/d factoring and the lane/reduction schedule change F32 association.
constant uint hybrid_q6_grouped16_bounded_only [[function_constant(73)]];

// Explicit-FMA experiments use separate entrypoints. The production wrapper
// instantiates false/false and retains the original arithmetic/fallback.
template<bool InnerFma, bool BlockFma, bool TypedHalf = false>
inline void hybrid_q6_grouped16_matvec_impl(
    device const uchar *codes_base, device const float *input,
    device float *output, constant HybridMatrixParams &p,
    constant uint4 &offsets, uint tile,
    uint lane, uint simd_group) {
    const uint first_row = tile * 4 + simd_group * 2;
    if (first_row >= p.rows) return;
    const ulong row_bytes = ulong(p.columns >> 8) * hybrid_packed_block_bytes;
    const ulong first_offset = ulong(first_row) * row_bytes;
    const ulong input_offset = p.group ? ulong(first_row / p.group) * p.columns : 0;
    device const uchar *codes = codes_base + offsets.x;
    if (p.group && first_row + 1 < p.rows && first_row / p.group != (first_row + 1) / p.group) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 2; ++r) {
            const ulong x = ulong((first_row + r) / p.group) * p.columns;
            const float total = hybrid_packed_block_dot(codes, input, first_offset + r * row_bytes, x, p.columns, lane);
            if (lane == 0) output[first_row + r] = total;
        }
        return;
    }
    const uint tid = lane >> 1;
    const uint parity = lane & 1;
    const uint half_block = tid >> 3;
    const uint low_start = (tid & 7) * 4;
    const uint scale_start = half_block * 8 + (low_start >> 4);
    const uint input_start = half_block * 128 + low_start;
    const uint low_offset = half_block * 64 + low_start;
    const uint high_offset = 128 + half_block * 32 + low_start;
    float sums[2] = {0.0f, 0.0f};
    bool reference_required = false;
    for (uint block_index = parity; block_index < (p.columns >> 8); block_index += 2) {
        float4 values[4];
        device const float *x = input + input_offset + block_index * 256 + input_start;
        #pragma clang loop unroll(full)
        for (uint l = 0; l < 4; ++l) {
            values[l] = float4(x[l], x[l + 32], x[l + 64], x[l + 96]);
            if (!hybrid_q6_grouped16_bounded_only) {
                reference_required = reference_required || any(abs(values[l]) > float4(1e20f));
            }
        }
        device const uchar *first_block = codes + first_offset + ulong(block_index) * hybrid_packed_block_bytes;
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 2; ++r) {
            if (first_row + r >= p.rows) continue;
            device const uchar *block = first_block + r * row_bytes;
            float4 partial = float4(0.0f);
            #pragma clang loop unroll(full)
            for (uint l = 0; l < 4; ++l) {
                const uint low0 = block[low_offset + l];
                const uint low1 = block[low_offset + l + 32];
                const uint high = block[high_offset + l];
                const int q0 = int((low0 & 15) | ((high & 0x03) << 4)) - 32;
                const int q1 = int((low1 & 15) | ((high & 0x0c) << 2)) - 32;
                const int q2 = int((low0 >> 4) | (high & 0x30)) - 32;
                const int q3 = int((low1 >> 4) | ((high & 0xc0) >> 2)) - 32;
                if (InnerFma) {
                    partial.x = fma(values[l].x, float(q0), partial.x);
                    partial.y = fma(values[l].y, float(q1), partial.y);
                    partial.z = fma(values[l].z, float(q2), partial.z);
                    partial.w = fma(values[l].w, float(q3), partial.w);
                } else {
                partial.x += values[l].x * float(q0);
                partial.y += values[l].y * float(q1);
                partial.z += values[l].z * float(q2);
                partial.w += values[l].w * float(q3);
                }
            }
            const float s0 = float(as_type<char>(block[192 + scale_start]));
            const float s1 = float(as_type<char>(block[194 + scale_start]));
            const float s2 = float(as_type<char>(block[196 + scale_start]));
            const float s3 = float(as_type<char>(block[198 + scale_start]));
            if (BlockFma) {
                float local = fma(partial.y, s1, partial.x * s0);
                local = fma(partial.z, s2, local);
                local = fma(partial.w, s3, local);
                sums[r] = fma(hybrid_grouped_half<TypedHalf>(block + 208), local, sums[r]);
            } else {
            sums[r] += hybrid_grouped_half<TypedHalf>(block + 208) * (partial.x * s0 + partial.y * s1 + partial.z * s2 + partial.w * s3);
            }
        }
    }
    // Finite GGML weight magnitudes are below 2.7e8. At |x|<=1e20 all
    // admitted absolute product sums are below 2.2e35. Larger inputs require
    // the old row so grouped algebra cannot conceal an overflowing product.
    if (!hybrid_q6_grouped16_bounded_only && simd_any(reference_required)) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 2; ++r) {
            if (first_row + r >= p.rows) continue;
            const float total = hybrid_packed_block_dot(codes, input, first_offset + r * row_bytes, input_offset, p.columns, lane);
            if (lane == 0) output[first_row + r] = total;
        }
        return;
    }
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 2; ++r) {
        const float total = simd_sum(sums[r]);
        if (lane == 0 && first_row + r < p.rows) output[first_row + r] = total;
    }
}

kernel void hybrid_q6_grouped16_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    hybrid_q6_grouped16_matvec_impl<false, false>(codes_base, input, output, p, offsets, tile, lane, simd_group);
}

kernel void hybrid_q6_grouped16_fma_inner_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    hybrid_q6_grouped16_matvec_impl<true, false>(codes_base, input, output, p, offsets, tile, lane, simd_group);
}

kernel void hybrid_q6_grouped16_fma_block_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    hybrid_q6_grouped16_matvec_impl<true, true>(codes_base, input, output, p, offsets, tile, lane, simd_group);
}


#ifdef STARK_KQUANT_TYPED_HALF_TESTS
// Test-only; typed half loads require host admission of even code offsets.
kernel void hybrid_q6_grouped16_typedhalf_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    hybrid_q6_grouped16_matvec_impl<false, false, true>(codes_base, input, output, p, offsets, tile, lane, simd_group);
}
#endif
