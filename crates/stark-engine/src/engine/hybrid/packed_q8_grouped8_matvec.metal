// SPDX-License-Identifier: MIT
// Adapted from ggml-org/llama.cpp b11429, commit
// d81235049384534c167caea52b85a694f6103d14,
// ggml/src/ggml-metal/kernels/mul_mv.metal, Q8_0 implementation (514-586).
// Copyright (c) 2023-2026 The ggml authors. See LICENSE.ggml-hybrid.
// Stark adaptations: byte-safe blocks, masked row tails, batch-boundary and
// bounded-input fallback to the unchanged Stark block dot. Opt-in dense schedules.

// Raw projection commands use established finite input/output checks; checked
// test wrappers additionally publish the same failure to a bound status word.
template<bool checked>
inline void hybrid_q8_grouped8_impl(
    device const uchar *codes_base, device const float *input,
    device float *output, constant HybridMatrixParams &p,
    constant uint4 &offsets, device atomic_uint *failure,
    uint tile, uint lane,
    uint sg, uint tid, threadgroup uint *risk,
    threadgroup uint *fallback, threadgroup float *partial) {
    const uint first = tile * 2;
    const ulong input_at = p.group ? ulong(first / p.group) * p.columns : 0;
    bool large = p.group && first + 1 < p.rows && first / p.group != (first + 1) / p.group;
    bool bad = false;
    for (uint d = tid; d < p.columns; d += 128) {
        float value = input[input_at + d];
        large |= !isfinite(value) || fabs(value) > 1e20f;
        bad |= !isfinite(value);
    }
    bool group_bad = simd_any(bad);
    if (checked && group_bad) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
    bool group_large = simd_any(large);
    if (lane == 0) risk[sg] = group_large ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        bool any_large = simd_any(lane < 4 ? risk[lane] != 0u : false);
        if (lane == 0) *fallback = any_large ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const uchar *codes = codes_base + offsets.x;
    const ulong row_bytes = ulong(p.columns / 32) * 34;
    if (*fallback != 0) {
        if (sg < 2 && first + sg < p.rows) {
            uint row = first + sg;
            ulong at = p.group ? ulong(row / p.group) * p.columns : 0;
            float total = hybrid_packed_block_dot(codes, input, ulong(row) * row_bytes, at, p.columns, lane);
            if (lane == 0) {
                output[row] = total;
                if (checked && !isfinite(total)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
            }
        }
        return;
    }
    const uint ix = lane / 4;
    const uint il = lane % 4;
    float sum[2] = {0.0f, 0.0f};
    for (uint block = sg * 8 + ix; block < p.columns / 32; block += 32) {
        float values[8];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) values[i] = input[input_at + block * 32 + il * 8 + i];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 2; ++r) {
            if (first + r >= p.rows) continue;
            device const uchar *encoded = codes + ulong(first + r) * row_bytes + ulong(block) * 34;
            float local = 0.0f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8; ++i)
                local += float(as_type<char>(encoded[2 + il * 8 + i])) * values[i];
            sum[r] += local * hybrid_half(encoded);
        }
    }
    for (uint r = 0; r < 2; ++r) {
        float total = simd_sum(sum[r]);
        if (lane == 0) partial[r * 4 + sg] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) for (uint r = 0; r < 2; ++r) {
        float total = simd_sum(lane < 4 ? partial[r * 4 + lane] : 0.0f);
        if (lane == 0 && first + r < p.rows) {
            output[first + r] = total;
            if (checked && !isfinite(total)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
        }
    }
}

kernel void hybrid_q8_grouped8_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint risk[4];
    threadgroup uint fallback;
    threadgroup float partial[8];
    hybrid_q8_grouped8_impl<true>(codes_base,input,output,p,offsets,failure,tile,lane,sg,tid,risk,&fallback,partial);
}

kernel void hybrid_q8_grouped8_safe_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]],
    uint tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint risk[4];
    threadgroup uint fallback;
    threadgroup float partial[8];
    hybrid_q8_grouped8_impl<false>(codes_base,input,output,p,offsets,nullptr,tile,lane,sg,tid,risk,&fallback,partial);
}

// Tiny-K diagnostic: eight lanes per row, four rows per SIMD group, two
// SIMD groups per threadgroup. The fallback computes four legacy rows per SG.
template<bool checked>
inline void hybrid_q8_segmented8_impl(
    device const uchar *codes_base, device const float *input,
    device float *output, constant HybridMatrixParams &p,
    constant uint4 &offsets, device atomic_uint *failure,
    uint tile, uint lane,
    uint sg, threadgroup uint *risk, threadgroup uint *fallback) {
    uint segment = lane / 8;
    uint local_lane = lane % 8;
    uint row = tile * 8 + sg * 4 + segment;
    ulong input_at = p.group ? ulong(row / p.group) * p.columns : 0;
    bool large = false;
    bool bad = false;
    if (row < p.rows) for (uint col = local_lane; col < p.columns; col += 8) {
        float value = input[input_at + col];
        large |= !isfinite(value) || fabs(value) > 1e20f;
        bad |= !isfinite(value);
    }
    bool group_bad = simd_any(bad);
    if (checked && group_bad) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
    bool group_large = simd_any(large);
    if (lane == 0) risk[sg] = group_large ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        bool any_large = simd_any(lane < 2 ? risk[lane] != 0u : false);
        if (lane == 0) *fallback = any_large ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const uchar *codes = codes_base + offsets.x;
    ulong row_bytes = ulong(p.columns / 32) * 34;
    if (*fallback != 0) {
        for (uint r = 0; r < 4; ++r) {
            uint old_row = tile * 8 + sg * 4 + r;
            if (old_row >= p.rows) continue;
            ulong at = p.group ? ulong(old_row / p.group) * p.columns : 0;
            float total = hybrid_packed_block_dot(codes,input,ulong(old_row)*row_bytes,at,p.columns,lane);
            if (lane == 0) {
                output[old_row] = total;
                if (checked && !isfinite(total)) atomic_fetch_or_explicit(failure,1u,memory_order_relaxed);
            }
        }
        return;
    }
    float sum = 0.0f;
    if (row < p.rows) for (uint block = 0; block < p.columns / 32; ++block) {
        device const uchar *encoded = codes + ulong(row)*row_bytes + ulong(block)*34;
        float local = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i)
            local += float(as_type<char>(encoded[2 + local_lane*4+i])) * input[input_at+block*32+local_lane*4+i];
        sum += local*hybrid_half(encoded);
    }
    sum += simd_shuffle_down(sum,4);
    sum += simd_shuffle_down(sum,2);
    sum += simd_shuffle_down(sum,1);
    if (local_lane == 0 && row < p.rows) {
        output[row] = sum;
        if (checked && !isfinite(sum)) atomic_fetch_or_explicit(failure,1u,memory_order_relaxed);
    }
}

kernel void hybrid_q8_segmented8_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], device atomic_uint *failure [[buffer(30)]],
    uint tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint risk[2];
    threadgroup uint fallback;
    hybrid_q8_segmented8_impl<true>(codes_base,input,output,p,offsets,failure,tile,lane,sg,risk,&fallback);
}

kernel void hybrid_q8_segmented8_safe_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]],
    uint tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint risk[2];
    threadgroup uint fallback;
    hybrid_q8_segmented8_impl<false>(codes_base,input,output,p,offsets,nullptr,tile,lane,sg,risk,&fallback);
}

// Original Stark schedule follow-ups. The measured r4g2 main geometry is
// selected by the existing Q8 opt-in; the segmented4 trial remains unselected.
template<bool checked>
inline void hybrid_q8_grouped8_r4g2_impl(
    device const uchar *codes_base, device const float *input,
    device float *output, constant HybridMatrixParams &p,
    constant uint4 &offsets, device atomic_uint *failure,
    uint tile, uint lane,
    uint sg, uint tid, threadgroup uint *risk,
    threadgroup uint *fallback, threadgroup float *partial) {
    const uint first = tile * 4;
    const ulong input_at = p.group ? ulong(first / p.group) * p.columns : 0;
    bool large = p.group && first / p.group != (min(first + 3, p.rows - 1)) / p.group;
    bool bad = false;
    for (uint d = tid; d < p.columns; d += 64) {
        float value = input[input_at + d];
        large |= !isfinite(value) || fabs(value) > 1e20f;
        bad |= !isfinite(value);
    }
    bool group_bad = simd_any(bad);
    if (checked && group_bad) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
    bool group_large = simd_any(large);
    if (lane == 0) risk[sg] = group_large ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        bool any_large = simd_any(lane < 2 ? risk[lane] != 0u : false);
        if (lane == 0) *fallback = any_large ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const uchar *codes = codes_base + offsets.x;
    const ulong row_bytes = ulong(p.columns / 32) * 34;
    if (*fallback != 0) {
        for (uint r = sg; r < 4; r += 2) {
            uint row = first + r;
            if (row >= p.rows) continue;
            ulong at = p.group ? ulong(row / p.group) * p.columns : 0;
            float total = hybrid_packed_block_dot(codes, input, ulong(row) * row_bytes, at, p.columns, lane);
            if (lane == 0) {
                output[row] = total;
                if (checked && !isfinite(total)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
            }
        }
        return;
    }
    const uint ix = lane / 4;
    const uint il = lane % 4;
    float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint block = sg * 8 + ix; block < p.columns / 32; block += 16) {
        float values[8];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) values[i] = input[input_at + block * 32 + il * 8 + i];
        #pragma clang loop unroll(full)
        for (uint r = 0; r < 4; ++r) {
            if (first + r >= p.rows) continue;
            device const uchar *encoded = codes + ulong(first + r) * row_bytes + ulong(block) * 34;
            float local = 0.0f;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 8; ++i)
                local += float(as_type<char>(encoded[2 + il * 8 + i])) * values[i];
            sum[r] += local * hybrid_half(encoded);
        }
    }
    for (uint r = 0; r < 4; ++r) {
        float total = simd_sum(sum[r]);
        if (lane == 0) partial[r * 2 + sg] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) for (uint r = 0; r < 4; ++r) {
        float total = simd_sum(lane < 2 ? partial[r * 2 + lane] : 0.0f);
        if (lane == 0 && first + r < p.rows) {
            output[first + r] = total;
            if (checked && !isfinite(total)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
        }
    }
}

template<bool checked>
inline void hybrid_q8_segmented4_impl(
    device const uchar *codes_base, device const float *input,
    device float *output, constant HybridMatrixParams &p,
    constant uint4 &offsets, device atomic_uint *failure,
    uint tile, uint lane,
    uint sg, threadgroup uint *risk, threadgroup uint *fallback) {
    uint segment = lane / 4;
    uint local_lane = lane % 4;
    uint row = tile * 16 + sg * 8 + segment;
    ulong input_at = p.group ? ulong(row / p.group) * p.columns : 0;
    bool large = false;
    bool bad = false;
    if (row < p.rows) for (uint col = local_lane; col < p.columns; col += 4) {
        float value = input[input_at + col];
        large |= !isfinite(value) || fabs(value) > 1e20f;
        bad |= !isfinite(value);
    }
    bool group_bad = simd_any(bad);
    if (checked && group_bad) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
    bool group_large = simd_any(large);
    if (lane == 0) risk[sg] = group_large ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        bool any_large = simd_any(lane < 2 ? risk[lane] != 0u : false);
        if (lane == 0) *fallback = any_large ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const uchar *codes = codes_base + offsets.x;
    ulong row_bytes = ulong(p.columns / 32) * 34;
    if (*fallback != 0) {
        for (uint r = 0; r < 8; ++r) {
            uint old_row = tile * 16 + sg * 8 + r;
            if (old_row >= p.rows) continue;
            ulong at = p.group ? ulong(old_row / p.group) * p.columns : 0;
            float total = hybrid_packed_block_dot(codes,input,ulong(old_row)*row_bytes,at,p.columns,lane);
            if (lane == 0) {
                output[old_row] = total;
                if (checked && !isfinite(total)) atomic_fetch_or_explicit(failure,1u,memory_order_relaxed);
            }
        }
        return;
    }
    float sum = 0.0f;
    if (row < p.rows) for (uint block = 0; block < p.columns / 32; ++block) {
        device const uchar *encoded = codes + ulong(row)*row_bytes + ulong(block)*34;
        float local = 0.0f;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i)
            local += float(as_type<char>(encoded[2 + local_lane*8+i])) * input[input_at+block*32+local_lane*8+i];
        sum += local*hybrid_half(encoded);
    }
    sum += simd_shuffle_down(sum,2);
    sum += simd_shuffle_down(sum,1);
    if (local_lane == 0 && row < p.rows) {
        output[row] = sum;
        if (checked && !isfinite(sum)) atomic_fetch_or_explicit(failure,1u,memory_order_relaxed);
    }
}


// The main r4g2 entrypoint is admitted for 16384x1536 only. Tiny-K retains
// the earlier segmented8 schedule; segmented4 below remains a test control.
kernel void hybrid_q8_grouped8_r4g2_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint risk[2];
    threadgroup uint fallback;
    threadgroup float partial[8];
    hybrid_q8_grouped8_r4g2_impl<false>(codes_base,input,output,p,offsets,nullptr,
        tile,lane,sg,tid,risk,&fallback,partial);
}
kernel void hybrid_q8_segmented4_matvec(
    device const uchar *codes_base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridMatrixParams &p [[buffer(3)]],
    constant uint4 &offsets [[buffer(6)]], uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint risk[2];
    threadgroup uint fallback;
    hybrid_q8_segmented4_impl<false>(codes_base,input,output,p,offsets,nullptr,
        tile,lane,sg,risk,&fallback);
}
