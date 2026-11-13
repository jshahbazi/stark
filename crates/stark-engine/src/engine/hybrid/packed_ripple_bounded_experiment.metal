// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 The ggml authors. See LICENSE.ggml-hybrid.
// Test-only wrapper of Stark's Q4 specialization of the GGML grouped schedule
// in packed_q5_grouped32_matvec.metal. NO production caller may use this entry
// without independently proving every input finite and abs(input) <= 1e20.
// This component ceiling experiment omits the cost of that future validation.
kernel void hybrid_ripple_q4_grouped32_bounded_matvec(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    hybrid_q5_grouped32_matvec_impl<false, false, false, true>(codes,input,output,p,offsets,tile,lane,simd_group);
}
