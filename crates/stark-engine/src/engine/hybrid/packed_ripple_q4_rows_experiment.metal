// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 The ggml authors. See LICENSE.ggml-hybrid.
// Adapted through Stark's Q4 specialization from ggml-org/llama.cpp
// d81235049384534c167caea52b85a694f6103d14, mul_mv.metal Q5_K schedule.
// Test-only Stark row-reuse specialization of packed_q5_grouped32_matvec.metal.
// Its GGML origin, copyright and MIT notice remain in that file and
// LICENSE.ggml-hybrid. Stored Q4_K codes remain unchanged. These schedules
// change F32 grouping and must be qualified separately before runtime use.
template<uint BlockGroups>
inline void hybrid_ripple_q4_rows_impl(
    device const uchar *codes_base, device const float *input, device float *output,
    constant HybridMatrixParams &p, constant uint4 &offsets, uint tile,
    uint lane, uint simd_group) {
    const uint row = tile * 4 + simd_group * 2;
    if (row >= p.rows) return;
    const ulong row_bytes = ulong(p.columns >> 8) * 144;
    device const uchar *codes = codes_base + offsets.x;
    const ulong input_offset = p.group ? ulong(row / p.group) * p.columns : 0;
    if (p.group && row + 1 < p.rows && row / p.group != (row + 1) / p.group) {
        for (uint r = 0; r < 2; ++r) {
            const float total = hybrid_packed_block_dot(codes, input,
                ulong(row + r) * row_bytes, ulong((row + r) / p.group) * p.columns, p.columns, lane);
            if (lane == 0) output[row+r] = total;
        }
        return;
    }
    const uint tid = lane / BlockGroups;
    const uint parity = lane % BlockGroups;
    const uint quarter = tid >> 2;
    const uint low_start = (tid & 3) * 8;
    const uint low_offset = 16 + quarter * 32 + low_start;
    const uint input_start = quarter * 64 + low_start;
    const uint g0 = quarter * 2;
    float sums[2] = {0.0f, 0.0f};
    bool reference_required = false;
    for (uint block_index = parity; block_index < (p.columns >> 8); block_index += BlockGroups) {
        device const float *x = input + input_offset + block_index * 256 + input_start;
        float4 values[8];
        float4 input_sum = float4(0.0f);
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 8; ++j) {
            values[j] = float4(x[j], x[j+32], 0.0f, 0.0f);
            if (BlockGroups == 4) {
                values[j].z = x[j+128]; values[j].w = x[j+160];
            }
            reference_required = reference_required || any(abs(values[j]) > float4(1e20f));
            input_sum += values[j];
        }
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 2; ++r) {
            if (row + r >= p.rows) continue;
            device const uchar *block = codes + ulong(row + r) * row_bytes + ulong(block_index) * 144;
            float4 lower = float4(0.0f);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; ++j) {
                const uint low0 = block[low_offset+j];
                lower.x += values[j].x * float(low0 & 15);
                lower.y += values[j].y * float(low0 >> 4);
                if (BlockGroups == 4) {
                    const uint low1 = block[low_offset+j+64];
                    lower.z += values[j].z * float(low1 & 15);
                    lower.w += values[j].w * float(low1 >> 4);
                }
            }
            const uint2 sm0 = hybrid_scale_min(block+4,g0);
            const uint2 sm1 = hybrid_scale_min(block+4,g0+1);
            float code_dot = float(sm0.x)*lower.x + float(sm1.x)*lower.y;
            float min_dot = float(sm0.y)*input_sum.x + float(sm1.y)*input_sum.y;
            if (BlockGroups == 4) {
                const uint2 sm2 = hybrid_scale_min(block+4,g0+4);
                const uint2 sm3 = hybrid_scale_min(block+4,g0+5);
                code_dot += float(sm2.x)*lower.z; code_dot += float(sm3.x)*lower.w;
                min_dot += float(sm2.y)*input_sum.z; min_dot += float(sm3.y)*input_sum.w;
            }
            sums[r] += hybrid_half(block)*code_dot - hybrid_half(block+2)*min_dot;
        }
    }
    // Preserve the original elementwise-overflow contract for extreme inputs.
    if (simd_any(reference_required)) {
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 2; ++r) if (row+r < p.rows) {
            const float total = hybrid_packed_block_dot(codes,input,ulong(row+r)*row_bytes,input_offset,p.columns,lane);
            if (lane == 0) output[row+r] = total;
        }
        return;
    }
    #pragma clang loop unroll(full)
    for (uint r = 0; r < 2; ++r) {
        const float total = simd_sum(sums[r]);
        if (lane == 0 && row+r < p.rows) output[row+r] = total;
    }
}

#define RIPPLE_Q4_ROWS(NAME, B) \
kernel void NAME(device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]], \
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]], \
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) { \
    hybrid_ripple_q4_rows_impl<B>(codes,input,output,p,offsets,tile,lane,simd_group); \
}
RIPPLE_Q4_ROWS(hybrid_ripple_q4_grouped16_r2_matvec, 2)
RIPPLE_Q4_ROWS(hybrid_ripple_q4_grouped32_r2_matvec, 4)
#undef RIPPLE_Q4_ROWS
