// SPDX-License-Identifier: MIT
// Adapted from ggml-org/llama.cpp b11429, commit
// d81235049384534c167caea52b85a694f6103d14,
// ggml/src/ggml-metal/kernels/mul_mv.metal, Q5_K implementation (1622-1744).
// Copyright (c) 2023-2026 The ggml authors. See LICENSE.ggml-hybrid.
// Stark adaptations: safe byte reads, direct high-nibble extraction, checked
// row/batch views and original-block fallback. Grouped dots/block factors and
// the alternating-block lane schedule change F32 association; codes are intact.
// The Q4_K entrypoint is Stark's specialization of this pinned Q5 schedule:
// its kind constant selects the Q4 payload and removes the high-bit terms.
// That reassociated path is separately qualified against decoded F64 products.

// Explicit-FMA experiments use separate entrypoints. The production wrapper
// instantiates false/false and retains the original arithmetic/fallback.
template<bool InnerFma, bool BlockFma, bool TypedHalf = false>
inline void hybrid_q5_grouped32_matvec_impl(
    device const uchar *codes_base, device const float *input,
    device float *output, constant HybridMatrixParams &p,
    constant uint4 &offsets, uint tile,
    uint lane, uint simd_group) {
    const uint row = tile * 2 + simd_group;
    if (row >= p.rows) return;
    const ulong row_offset = ulong(row) * (p.columns >> 8) * hybrid_packed_block_bytes;
    const ulong input_offset = p.group ? ulong(row / p.group) * p.columns : 0;
    device const uchar *codes = codes_base + offsets.x;
    const uint tid = lane >> 2;
    const uint parity = lane & 3;
    const uint quarter = tid >> 2;
    const uint low_start = (tid & 3) * 8;
    const uint low_offset = (hybrid_packed_kind == 12 ? 16 : 48) + quarter * 32 + low_start;
    const uint high_offset = 16 + low_start;
    const uint input_start = quarter * 64 + low_start;
    const uint g0 = quarter * 2;
    float sum = 0.0f;
    bool reference_required = false;
    for (uint block_index = parity; block_index < (p.columns >> 8); block_index += 4) {
        device const uchar *block = codes + row_offset + ulong(block_index) * hybrid_packed_block_bytes;
        device const float *x = input + input_offset + block_index * 256 + input_start;
        float4 lower = float4(0.0f);
        float4 upper = float4(0.0f);
        float4 input_sum = float4(0.0f);
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j) {
            const float4 values = float4(x[j], x[j + 32], x[j + 128], x[j + 160]);
            reference_required = reference_required || any(abs(values) > float4(1e20f));
            input_sum += values;
            const uint low0 = block[low_offset + j];
            const uint low1 = block[low_offset + j + 64];
            const uint high = hybrid_packed_kind == 12 ? 0 : block[high_offset + j];
            if (InnerFma) {
                lower.x = fma(values.x, float(low0 & 15), lower.x);
                lower.y = fma(values.y, float(low0 >> 4), lower.y);
                lower.z = fma(values.z, float(low1 & 15), lower.z);
                lower.w = fma(values.w, float(low1 >> 4), lower.w);
            } else {
            lower.x += values.x * float(low0 & 15);
            lower.y += values.y * float(low0 >> 4);
            lower.z += values.z * float(low1 & 15);
            lower.w += values.w * float(low1 >> 4);
            }
            upper.x += high & (1u << g0) ? values.x : 0.0f;
            upper.y += high & (1u << (g0 + 1)) ? values.y : 0.0f;
            upper.z += high & (1u << (g0 + 4)) ? values.z : 0.0f;
            upper.w += high & (1u << (g0 + 5)) ? values.w : 0.0f;
        }
        const uint2 sm0 = hybrid_scale_min(block + 4, g0);
        const uint2 sm1 = hybrid_scale_min(block + 4, g0 + 1);
        const uint2 sm2 = hybrid_scale_min(block + 4, g0 + 4);
        const uint2 sm3 = hybrid_scale_min(block + 4, g0 + 5);
        const float4 group_dot = lower + 16.0f * upper;
        if (BlockFma) {
            float qdot = fma(float(sm1.x), group_dot.y, float(sm0.x) * group_dot.x);
            qdot = fma(float(sm2.x), group_dot.z, qdot);
            qdot = fma(float(sm3.x), group_dot.w, qdot);
            float mdot = fma(input_sum.y, float(sm1.y), input_sum.x * float(sm0.y));
            mdot = fma(input_sum.z, float(sm2.y), mdot);
            mdot = fma(input_sum.w, float(sm3.y), mdot);
            sum += fma(hybrid_grouped_half<TypedHalf>(block), qdot, -hybrid_grouped_half<TypedHalf>(block + 2) * mdot);
        } else {
        sum += hybrid_grouped_half<TypedHalf>(block) * (float(sm0.x) * group_dot.x + float(sm1.x) * group_dot.y
            + float(sm2.x) * group_dot.z + float(sm3.x) * group_dot.w)
            - hybrid_grouped_half<TypedHalf>(block + 2) * (input_sum.x * float(sm0.y) + input_sum.y * float(sm1.y)
            + input_sum.z * float(sm2.y) + input_sum.w * float(sm3.y));
        }
    }
    // Finite encoded scales bound weights below 2.7e8. At |x|<=1e20 all
    // admitted absolute product sums stay below 2.2e35. Larger inputs use the
    // old row so grouped algebra cannot conceal an overflowing product.
    if (simd_any(reference_required)) {
        const float total = hybrid_packed_block_dot(codes, input, row_offset, input_offset, p.columns, lane);
        if (lane == 0) output[row] = total;
        return;
    }
    const float total = simd_sum(sum);
    if (lane == 0) output[row] = total;
}

kernel void hybrid_q5_grouped32_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    hybrid_q5_grouped32_matvec_impl<false, false>(codes_base, input, output, p, offsets, tile, lane, simd_group);
}

// Q4_K uses the same checked schedule; its kind constant removes Q5 high bits.
kernel void hybrid_q4_grouped32_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    hybrid_q5_grouped32_matvec_impl<false, false>(codes_base, input, output, p, offsets, tile, lane, simd_group);
}

kernel void hybrid_q5_grouped32_fma_inner_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    hybrid_q5_grouped32_matvec_impl<true, false>(codes_base, input, output, p, offsets, tile, lane, simd_group);
}

kernel void hybrid_q5_grouped32_fma_block_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    hybrid_q5_grouped32_matvec_impl<true, true>(codes_base, input, output, p, offsets, tile, lane, simd_group);
}


#ifdef STARK_KQUANT_TYPED_HALF_TESTS
// Test-only; typed half loads require host admission of even code offsets.
kernel void hybrid_q5_grouped32_typedhalf_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    hybrid_q5_grouped32_matvec_impl<false, false, true>(codes_base, input, output, p, offsets, tile, lane, simd_group);
}
kernel void hybrid_q4_grouped32_typedhalf_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    hybrid_q5_grouped32_matvec_impl<false, false, true>(codes_base, input, output, p, offsets, tile, lane, simd_group);
}
#endif
