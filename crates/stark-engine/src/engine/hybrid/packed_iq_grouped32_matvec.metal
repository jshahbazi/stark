// SPDX-License-Identifier: Apache-2.0
// Stark IQ2_XS/IQ3_XXS vector32 adaptation of the numerical schedules in
// llama.cpp d81235049384534c167caea52b85a694f6103d14 (b11429),
// ggml/src/ggml-metal/kernels/mul_mv.metal, IQ2_XS and IQ3_XXS.
// Copyright (c) 2023-2026 The ggml authors. Upstream MIT terms and table/layout
// provenance are retained in LICENSE.ggml-hybrid. No dependency is added.
// This schedule explicitly reassociates F32 sums and scales. Large finite
// inputs fall back uniformly to Stark's bit-preserving vector8 chronology.

template<bool typed16> inline ushort hybrid_iq_grouped32_u16(device const uchar *p) {
    if (typed16) return ((device const ushort *)p)[0];
    return hybrid_u16(p);
}
template<bool typed16> inline float hybrid_iq_grouped32_half(device const uchar *p) {
    if (typed16) return float(((device const half *)p)[0]);
    return hybrid_half(p);
}
template<bool typed16> inline uint hybrid_iq_grouped32_aux(device const uchar *p) {
    if (typed16) {
        device const ushort *words = (device const ushort *)p;
        return uint(words[0]) | (uint(words[1]) << 16);
    }
    return hybrid_u32(p);
}

template<uint arithmetic> inline float hybrid_iq_grouped32_add(float sum, float value, uchar grid, bool negative) {
    if (arithmetic == 0) return sum + value * float(grid) * (negative ? -1.0f : 1.0f);
    const float signed_grid = negative ? -float(grid) : float(grid);
    if (arithmetic == 1) return sum + value * signed_grid;
    return fma(value, signed_grid, sum);
}

template<bool guard_input, uint rows_per_simd = 4, bool typed16 = false, uint arithmetic = 0>
inline void hybrid_iq_grouped32_rows(
    device const uchar *codes, device const float *input,
    device float *output, constant uint *p,
    device const uint *indices, device atomic_uint *failure,
    uint2 grid, uint lane, uint simd_group,
    threadgroup const uchar *lookup, threadgroup const uchar *sign_table) {
    const uint first_row = grid.x * (2 * rows_per_simd) + simd_group * rows_per_simd;
    const uint batch = grid.y;
    const uint expert = indices[batch];
    if (expert >= p[2]) {
        if (lane == 0) {
            for (uint r = 0; r < rows_per_simd; ++r)
                if (first_row + r < p[0]) output[batch * p[0] + first_row + r] = 0.0f;
            atomic_store_explicit(failure, 1u, memory_order_relaxed);
        }
        return;
    }
    const ulong row_bytes = ulong(p[1] >> 8) * hybrid_packed_block_bytes;
    const ulong input_offset = p[3] ? ulong(batch) * p[1] : 0;
    float sums[rows_per_simd] = {0.0f};
    bool large = false;
    // A lane owns 32 consecutive inputs and advances by 1024 columns. Each
    // preload is consumed by all four independent row accumulators.
    for (uint col = lane * 32; col < p[1]; col += 1024) {
        float values[32];
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 32; ++j) {
            values[j] = input[input_offset + col + j];
            if (guard_input) {
                large = (fabs(values[j]) > 1.0e20f) || large;
                if (!isfinite(values[j])) atomic_store_explicit(failure, 1u, memory_order_relaxed);
            }
        }
        const uint part = (col & 255) >> 5;
        #pragma clang loop unroll(full)
        for (uint r = 0; r < rows_per_simd; ++r) {
            if (first_row + r >= p[0]) continue;
            device const uchar *block = codes + (ulong(expert) * p[0] + first_row + r) * row_bytes
                                       + ulong(col >> 8) * hybrid_packed_block_bytes;
            if (hybrid_packed_kind == 17) {
                const uint scales = block[66 + part];
                const float d1 = hybrid_iq_grouped32_half<typed16>(block) * (0.5f + float(scales & 15));
                const float d2 = hybrid_iq_grouped32_half<typed16>(block) * (0.5f + float(scales >> 4));
                float sum1 = 0.0f, sum2 = 0.0f;
                #pragma clang loop unroll(full)
                for (uint l = 0; l < 2; ++l) {
                    const uint q = hybrid_iq_grouped32_u16<typed16>(block + 2 + part * 8 + l * 2);
                    threadgroup const uchar *entry = lookup + (q & 511) * 8;
                    const uint signs = sign_table[q >> 9];
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 8; ++j)
                        sum1 = hybrid_iq_grouped32_add<arithmetic>(sum1, values[8 * l + j], entry[j], (signs & (1 << j)) != 0);
                }
                #pragma clang loop unroll(full)
                for (uint l = 2; l < 4; ++l) {
                    const uint q = hybrid_iq_grouped32_u16<typed16>(block + 2 + part * 8 + l * 2);
                    threadgroup const uchar *entry = lookup + (q & 511) * 8;
                    const uint signs = sign_table[q >> 9];
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 8; ++j)
                        sum2 = hybrid_iq_grouped32_add<arithmetic>(sum2, values[8 * l + j], entry[j], (signs & (1 << j)) != 0);
                }
                sums[r] += d1 * sum1 + d2 * sum2;
            } else {
                const uint ss = hybrid_iq_grouped32_aux<typed16>(block + 66 + part * 4);
                const float d = hybrid_iq_grouped32_half<typed16>(block) * (0.5f + float(ss >> 28));
                float2 sum = float2(0.0f);
                #pragma clang loop unroll(full)
                for (uint l = 0; l < 4; ++l) {
                    const uint at = 2 + part * 8 + 2 * l;
                    threadgroup const uchar *entry1 = lookup + uint(block[at]) * 4;
                    threadgroup const uchar *entry2 = lookup + uint(block[at + 1]) * 4;
                    const uint signs = sign_table[(ss >> (7 * l)) & 127];
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < 4; ++j) {
                        sum.x = hybrid_iq_grouped32_add<arithmetic>(sum.x, values[8 * l + j], entry1[j], (signs & (1 << j)) != 0);
                        sum.y = hybrid_iq_grouped32_add<arithmetic>(sum.y, values[8 * l + j + 4], entry2[j], (signs & (1 << (j + 4))) != 0);
                    }
                }
                sums[r] += d * (sum.x + sum.y);
            }
        }
    }
    // Both SIMD groups visit the same batch input. Detection piggybacks on
    // its required preload; no extra full-width scan precedes normal inputs.
    if (guard_input && simd_any(large)) {
        hybrid_iq_staged_rows(codes, input, output, p, indices, failure,
                             grid, lane, simd_group, lookup, sign_table);
        return;
    }
    #pragma clang loop unroll(full)
    for (uint r = 0; r < rows_per_simd; ++r) {
        const float total = simd_sum(sums[r]) * (hybrid_packed_kind == 17 ? 0.25f : 0.5f);
        if (lane == 0 && first_row + r < p[0]) output[batch * p[0] + first_row + r] = total;
    }
}

kernel void hybrid_graph_iq_indexed_grouped32_iq2xs(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup ulong lookup[512];
    threadgroup uchar signs[128];
    const uint tid = simd_group * 32 + lane;
    for (uint i = tid; i < 512; i += 64) lookup[i] = iq2xs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    // All 64 threads reach this barrier, including the final row tile and
    // invalid-expert groups. Bounds exits happen only inside the helper.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<true>(codes, input, output, p, indices, failure, grid, lane,
                         simd_group, (threadgroup const uchar *)lookup, signs);
}

kernel void hybrid_graph_iq_indexed_grouped32_iq3xxs(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint lookup[256];
    threadgroup uchar signs[128];
    const uint tid = simd_group * 32 + lane;
    for (uint i = tid; i < 256; i += 64) lookup[i] = iq3xxs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    // All 64 threads reach this barrier, including the final row tile and
    // invalid-expert groups. Bounds exits happen only inside the helper.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<true>(codes, input, output, p, indices, failure, grid, lane,
                         simd_group, (threadgroup const uchar *)lookup, signs);
}

// Test-only diagnostic: caller must first verify every input is finite and
// abs(input) <= 1e20. The false template removes detection and fallback code.

kernel void hybrid_graph_iq_indexed_grouped32_bound_iq2xs(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup ulong lookup[512];
    threadgroup uchar signs[128];
    const uint tid = simd_group * 32 + lane;
    for (uint i = tid; i < 512; i += 64) lookup[i] = iq2xs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    // All 64 threads reach this barrier, including the final row tile and
    // invalid-expert groups. Bounds exits happen only inside the helper.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false>(codes, input, output, p, indices, failure, grid, lane,
                         simd_group, (threadgroup const uchar *)lookup, signs);
}

kernel void hybrid_graph_iq_indexed_grouped32_bound_iq3xxs(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint lookup[256];
    threadgroup uchar signs[128];
    const uint tid = simd_group * 32 + lane;
    for (uint i = tid; i < 256; i += 64) lookup[i] = iq3xxs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    // All 64 threads reach this barrier, including the final row tile and
    // invalid-expert groups. Bounds exits happen only inside the helper.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false>(codes, input, output, p, indices, failure, grid, lane,
                         simd_group, (threadgroup const uchar *)lookup, signs);
}

// One flag per contiguous input row. Zero proves all source values are finite
// and within the grouped schedule's envelope; one requests the old chronology.
// The nonfinite signal is sticky even when a later operator masks its output.
kernel void hybrid_graph_iq_input_bound(
    device const float *input [[buffer(0)]], device uint *bounds [[buffer(1)]],
    constant uint *p [[buffer(2)]], device atomic_uint *failure [[buffer(30)]],
    uint batch [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    bool outside = false, nonfinite = false;
    for (uint d = tid; d < p[0]; d += 256) {
        const float value = input[ulong(batch) * p[0] + d];
        const bool invalid = !isfinite(value);
        outside = outside || invalid || fabs(value) > 1.0e20f;
        nonfinite = nonfinite || invalid;
    }
    if (simd_any(nonfinite) && lane == 0)
        atomic_store_explicit(failure, 1u, memory_order_relaxed);
    threadgroup uint flags[8];
    const bool any_outside = simd_any(outside);
    if (lane == 0) flags[sg] = any_outside ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const bool fallback = simd_any(lane < 8 ? flags[lane] != 0 : false);
        if (lane == 0) bounds[batch] = fallback ? 1u : 0u;
    }
}

// The host binds a flag produced by hybrid_graph_iq_input_bound in this command.
// This single uniform branch retains the established multirow4 fallback.
kernel void hybrid_graph_iq_indexed_gpu_bound_iq2xs(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device const uint *bounds [[buffer(7)]],
    device atomic_uint *failure [[buffer(30)]], uint2 grid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    if (bounds[p[3] ? grid.y : 0] != 0) {
        hybrid_iq_multirow_rows(codes, input, output, p, indices, failure, grid, lane, simd_group);
        return;
    }
    threadgroup ulong lookup[512];
    threadgroup uchar signs[128];
    const uint tid = simd_group * 32 + lane;
    for (uint i = tid; i < 512; i += 64) lookup[i] = iq2xs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false>(codes, input, output, p, indices, failure, grid, lane,
                                   simd_group, (threadgroup const uchar *)lookup, signs);
}

// The host binds a flag produced by hybrid_graph_iq_input_bound in this command.
// This single uniform branch retains the established multirow4 fallback.
kernel void hybrid_graph_iq_indexed_gpu_bound_iq3xxs(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device const uint *bounds [[buffer(7)]],
    device atomic_uint *failure [[buffer(30)]], uint2 grid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    if (bounds[p[3] ? grid.y : 0] != 0) {
        hybrid_iq_multirow_rows(codes, input, output, p, indices, failure, grid, lane, simd_group);
        return;
    }
    threadgroup uint lookup[256];
    threadgroup uchar signs[128];
    const uint tid = simd_group * 32 + lane;
    for (uint i = tid; i < 256; i += 64) lookup[i] = iq3xxs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false>(codes, input, output, p, indices, failure, grid, lane,
                                   simd_group, (threadgroup const uchar *)lookup, signs);
}

// Test-only row/typed16 sweep. Typed entry points require a two-byte-aligned
// code binding; all74/98-byte block and row strides preserve that alignment.

kernel void hybrid_iq_sweep_iq2xs_r1_byte(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup ulong lookup[512];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 512; i += 64) lookup[i] = iq2xs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 1, false>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}

kernel void hybrid_iq_sweep_iq2xs_r1_typed16(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup ulong lookup[512];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 512; i += 64) lookup[i] = iq2xs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 1, true>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}

kernel void hybrid_iq_sweep_iq2xs_r2_byte(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup ulong lookup[512];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 512; i += 64) lookup[i] = iq2xs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 2, false>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}

kernel void hybrid_iq_sweep_iq2xs_r2_typed16(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup ulong lookup[512];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 512; i += 64) lookup[i] = iq2xs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 2, true>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}

kernel void hybrid_iq_sweep_iq2xs_r4_byte(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup ulong lookup[512];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 512; i += 64) lookup[i] = iq2xs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 4, false>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}

kernel void hybrid_iq_sweep_iq2xs_r4_typed16(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup ulong lookup[512];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 512; i += 64) lookup[i] = iq2xs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 4, true>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}

kernel void hybrid_iq_sweep_iq2xs_r8_byte(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup ulong lookup[512];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 512; i += 64) lookup[i] = iq2xs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 8, false>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}

kernel void hybrid_iq_sweep_iq2xs_r8_typed16(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup ulong lookup[512];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 512; i += 64) lookup[i] = iq2xs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 8, true>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}

kernel void hybrid_iq_sweep_iq3xxs_r1_byte(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint lookup[256];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 256; i += 64) lookup[i] = iq3xxs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 1, false>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}

kernel void hybrid_iq_sweep_iq3xxs_r1_typed16(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint lookup[256];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 256; i += 64) lookup[i] = iq3xxs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 1, true>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}

kernel void hybrid_iq_sweep_iq3xxs_r2_byte(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint lookup[256];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 256; i += 64) lookup[i] = iq3xxs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 2, false>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}

kernel void hybrid_iq_sweep_iq3xxs_r2_typed16(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint lookup[256];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 256; i += 64) lookup[i] = iq3xxs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 2, true>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}

kernel void hybrid_iq_sweep_iq3xxs_r4_byte(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint lookup[256];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 256; i += 64) lookup[i] = iq3xxs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 4, false>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}

kernel void hybrid_iq_sweep_iq3xxs_r4_typed16(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint lookup[256];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 256; i += 64) lookup[i] = iq3xxs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 4, true>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}

kernel void hybrid_iq_sweep_iq3xxs_r8_byte(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint lookup[256];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 256; i += 64) lookup[i] = iq3xxs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 8, false>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}

kernel void hybrid_iq_sweep_iq3xxs_r8_typed16(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint lookup[256];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 256; i += 64) lookup[i] = iq3xxs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 8, true>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}

// Production typed loads require host-checked two-byte code alignment.

// The host binds a flag produced by hybrid_graph_iq_input_bound in this command.
// This single uniform branch retains the established multirow4 fallback.
kernel void hybrid_graph_iq_indexed_gpu_bound_typed16_iq2xs(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device const uint *bounds [[buffer(7)]],
    device atomic_uint *failure [[buffer(30)]], uint2 grid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    if (bounds[p[3] ? grid.y : 0] != 0) {
        hybrid_iq_multirow_rows(codes, input, output, p, indices, failure, grid, lane, simd_group);
        return;
    }
    threadgroup ulong lookup[512];
    threadgroup uchar signs[128];
    const uint tid = simd_group * 32 + lane;
    for (uint i = tid; i < 512; i += 64) lookup[i] = iq2xs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 4, true>(codes, input, output, p, indices, failure, grid, lane,
                                   simd_group, (threadgroup const uchar *)lookup, signs);
}

// The host binds a flag produced by hybrid_graph_iq_input_bound in this command.
// This single uniform branch retains the established multirow4 fallback.
kernel void hybrid_graph_iq_indexed_gpu_bound_typed16_iq3xxs(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device const uint *bounds [[buffer(7)]],
    device atomic_uint *failure [[buffer(30)]], uint2 grid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    if (bounds[p[3] ? grid.y : 0] != 0) {
        hybrid_iq_multirow_rows(codes, input, output, p, indices, failure, grid, lane, simd_group);
        return;
    }
    threadgroup uint lookup[256];
    threadgroup uchar signs[128];
    const uint tid = simd_group * 32 + lane;
    for (uint i = tid; i < 256; i += 64) lookup[i] = iq3xxs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 4, true>(codes, input, output, p, indices, failure, grid, lane,
                                   simd_group, (threadgroup const uchar *)lookup, signs);
}

// Opt-in numerically qualified signed-grid reassociation; extreme inputs use the unchanged fallback.
kernel void hybrid_graph_iq_indexed_gpu_bound_factor_iq2xs(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device const uint *bounds [[buffer(7)]],
    device atomic_uint *failure [[buffer(30)]], uint2 grid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    if (bounds[p[3] ? grid.y : 0] != 0) {
        hybrid_iq_multirow_rows(codes, input, output, p, indices, failure, grid, lane, simd_group);
        return;
    }
    threadgroup ulong lookup[512];
    threadgroup uchar signs[128];
    const uint tid = simd_group * 32 + lane;
    for (uint i = tid; i < 512; i += 64) lookup[i] = iq2xs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 4, true, 1>(codes, input, output, p, indices, failure, grid, lane,
                                   simd_group, (threadgroup const uchar *)lookup, signs);
}

// The host binds a flag produced by hybrid_graph_iq_input_bound in this command.
// This single uniform branch retains the established multirow4 fallback.
kernel void hybrid_graph_iq_indexed_gpu_bound_factor_iq3xxs(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device const uint *bounds [[buffer(7)]],
    device atomic_uint *failure [[buffer(30)]], uint2 grid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
    if (bounds[p[3] ? grid.y : 0] != 0) {
        hybrid_iq_multirow_rows(codes, input, output, p, indices, failure, grid, lane, simd_group);
        return;
    }
    threadgroup uint lookup[256];
    threadgroup uchar signs[128];
    const uint tid = simd_group * 32 + lane;
    for (uint i = tid; i < 256; i += 64) lookup[i] = iq3xxs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 4, true, 1>(codes, input, output, p, indices, failure, grid, lane,
                                   simd_group, (threadgroup const uchar *)lookup, signs);
}


kernel void hybrid_iq_arithmetic_iq2xs_factor(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup ulong lookup[512];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 512; i += 64) lookup[i] = iq2xs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 4, true, 1>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}

kernel void hybrid_iq_arithmetic_iq3xxs_factor(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint lookup[256];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 256; i += 64) lookup[i] = iq3xxs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 4, true, 1>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}

kernel void hybrid_iq_arithmetic_iq2xs_fma(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup ulong lookup[512];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 512; i += 64) lookup[i] = iq2xs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 4, true, 2>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}

kernel void hybrid_iq_arithmetic_iq3xxs_fma(
    device const uchar *codes [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint *p [[buffer(3)]],
    device const uint *indices [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint2 grid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint lookup[256];
    threadgroup uchar signs[128];
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < 256; i += 64) lookup[i] = iq3xxs_grid[i];
    for (uint i = tid; i < 128; i += 64) signs[i] = ksigns[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hybrid_iq_grouped32_rows<false, 4, true, 2>(codes,input,output,p,indices,failure,
        grid,lane,sg,(threadgroup const uchar *)lookup,signs);
}
