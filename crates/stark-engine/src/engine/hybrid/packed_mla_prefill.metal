// SPDX-License-Identifier: MIT
// Original Stark cross-token weight reuse around the existing reduction trees.
// Q5/Q6 arithmetic derives from the attributed grouped kernels in this folder:
// ggml-org/llama.cpp d81235049384534c167caea52b85a694f6103d14,
// Copyright (c) 2023-2026 the ggml authors. See LICENSE.ggml-hybrid.
// No activation cast or tensor-core reassociation is introduced.

struct HybridTokenProjection {
    uint rows, columns, kind, block, block_bytes, group, input_stride, tokens;
    uint codes_offset, input_offset, output_offset;
};

inline ulong hybrid_token_input(constant HybridTokenProjection &p, uint token, uint row) {
    return ulong(p.input_offset) + ulong(token) * p.input_stride
        + (p.group ? ulong(row / p.group) * p.columns : 0);
}

kernel void hybrid_token_projection_generic(
    device const uchar *base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridTokenProjection &p [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]], uint2 grid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = grid.x, first = grid.y * 8;
    const uint shift = hybrid_packed_block == 32 ? 5 : 0;
    const ulong row_at = ulong(row) * (p.columns >> shift) * hybrid_packed_block_bytes;
    device const uchar *codes = base + p.codes_offset;
    float sums[8] = {0.0f};
    for (uint col = tid; col < p.columns; col += 256) {
        const float w = hybrid_quant(codes + row_at + ulong(col >> shift) * hybrid_packed_block_bytes,
            hybrid_packed_kind, col & (hybrid_packed_block - 1));
        // Each encoded weight is decoded once and contributes to eight
        // independent input rows in the old per-lane column chronology.
        for (uint t = 0; t < 8; ++t) if (first + t < p.tokens)
            sums[t] += w * input[hybrid_token_input(p, first + t, row) + col];
    }
    threadgroup float partial[8][8];
    for (uint t = 0; t < 8; ++t) {
        const float total = simd_sum(sums[t]);
        if (lane == 0) partial[t][sg] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) for (uint t = 0; t < 8; ++t) if (first + t < p.tokens) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) total += partial[t][i];
        output[ulong(p.output_offset) + ulong(first + t) * p.rows + row] = total;
        if (!isfinite(total)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
    }
}

kernel void hybrid_token_projection_q8(
    device const uchar *base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridTokenProjection &p [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]], uint2 grid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint first_row = grid.x * 4, first_token = grid.y * 4;
    device const uchar *codes = base + p.codes_offset;
    const ulong row_bytes = ulong(p.columns / 32) * 34;
    threadgroup uint risks[4][2], fallback[4];
    for (uint t = 0; t < 4; ++t) {
        bool large = p.group && first_row / p.group != min(first_row + 3, p.rows - 1) / p.group;
        for (uint col = tid; first_token + t < p.tokens && col < p.columns; col += 64) {
            const float x = input[hybrid_token_input(p, first_token + t, first_row) + col];
            large |= !isfinite(x) || fabs(x) > 1e20f;
        }
        const bool group_large = simd_any(large);
        if (lane == 0) risks[t][sg] = group_large ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) for (uint t = 0; t < 4; ++t) {
        const bool large = simd_any(lane < 2 ? risks[t][lane] != 0 : false);
        if (lane == 0) fallback[t] = large ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint ix = lane / 4, il = lane % 4;
    float sums[4][4] = {{0.0f}};
    for (uint block = sg * 8 + ix; block < p.columns / 32; block += 16) {
        for (uint r = 0; r < 4; ++r) if (first_row + r < p.rows) {
            device const uchar *encoded = codes + ulong(first_row + r) * row_bytes + ulong(block) * 34;
            float coefficients[8];
            for (uint i = 0; i < 8; ++i) coefficients[i] = float(as_type<char>(encoded[2 + il * 8 + i]));
            const float scale = hybrid_half(encoded);
            for (uint t = 0; t < 4; ++t) if (first_token + t < p.tokens && fallback[t] == 0) {
                float local = 0.0f;
                const ulong at = hybrid_token_input(p, first_token + t, first_row) + block * 32 + il * 8;
                for (uint i = 0; i < 8; ++i) local += coefficients[i] * input[at + i];
                sums[t][r] += local * scale;
            }
        }
    }
    threadgroup float partial[4][8];
    for (uint t = 0; t < 4; ++t) for (uint r = 0; r < 4; ++r) {
        const float total = simd_sum(sums[t][r]);
        if (lane == 0) partial[t][r * 2 + sg] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < 4; ++t) if (first_token + t < p.tokens) {
        if (fallback[t] != 0) {
            for (uint r = sg; r < 4; r += 2) if (first_row + r < p.rows) {
                const uint row = first_row + r;
                const float total = hybrid_packed_block_dot(codes, input, ulong(row) * row_bytes,
                    hybrid_token_input(p, first_token + t, row), p.columns, lane);
                if (lane == 0) {
                    output[ulong(p.output_offset) + ulong(first_token + t) * p.rows + row] = total;
                    if (!isfinite(total)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
                }
            }
        } else if (sg == 0) for (uint r = 0; r < 4; ++r) {
            const float total = simd_sum(lane < 2 ? partial[t][r * 2 + lane] : 0.0f);
            if (lane == 0 && first_row + r < p.rows) {
                output[ulong(p.output_offset) + ulong(first_token + t) * p.rows + first_row + r] = total;
                if (!isfinite(total)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
            }
        }
    }
}

kernel void hybrid_token_projection_q5(
    device const uchar *base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridTokenProjection &p [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]], uint2 grid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = grid.x * 2 + sg, first = grid.y * 4;
    if (row >= p.rows) return;
    device const uchar *codes = base + p.codes_offset;
    const ulong row_at = ulong(row) * (p.columns >> 8) * p.block_bytes;
    const uint tid = lane >> 2, parity = lane & 3, quarter = tid >> 2;
    const uint low_start = (tid & 3) * 8;
    const uint low_offset = 48 + quarter * 32 + low_start, high_offset = 16 + low_start;
    const uint input_start = quarter * 64 + low_start, g0 = quarter * 2;
    float sums[4] = {0.0f};
    bool reference_required[4] = {false};
    for (uint bi = parity; bi < (p.columns >> 8); bi += 4) {
        device const uchar *block = codes + row_at + ulong(bi) * p.block_bytes;
        float4 low[8]; uint high[8];
        for (uint j = 0; j < 8; ++j) {
            const uint a = block[low_offset + j], b = block[low_offset + j + 64];
            low[j] = float4(float(a & 15), float(a >> 4), float(b & 15), float(b >> 4));
            high[j] = block[high_offset + j];
        }
        const uint2 sm0 = hybrid_scale_min(block + 4, g0), sm1 = hybrid_scale_min(block + 4, g0 + 1);
        const uint2 sm2 = hybrid_scale_min(block + 4, g0 + 4), sm3 = hybrid_scale_min(block + 4, g0 + 5);
        const float scale = hybrid_half(block), minimum = hybrid_half(block + 2);
        for (uint t = 0; t < 4; ++t) if (first + t < p.tokens) {
            device const float *x = input + hybrid_token_input(p, first + t, row) + bi * 256 + input_start;
            float4 lower = float4(0.0f), upper = float4(0.0f), input_sum = float4(0.0f);
            for (uint j = 0; j < 8; ++j) {
                const float4 v = float4(x[j], x[j + 32], x[j + 128], x[j + 160]);
                reference_required[t] |= any(abs(v) > float4(1e20f));
                input_sum += v;
                lower.x += v.x * low[j].x; lower.y += v.y * low[j].y;
                lower.z += v.z * low[j].z; lower.w += v.w * low[j].w;
                upper.x += high[j] & (1u << g0) ? v.x : 0.0f;
                upper.y += high[j] & (1u << (g0 + 1)) ? v.y : 0.0f;
                upper.z += high[j] & (1u << (g0 + 4)) ? v.z : 0.0f;
                upper.w += high[j] & (1u << (g0 + 5)) ? v.w : 0.0f;
            }
            const float4 dot = lower + 16.0f * upper;
            sums[t] += scale * (float(sm0.x) * dot.x + float(sm1.x) * dot.y + float(sm2.x) * dot.z + float(sm3.x) * dot.w)
                - minimum * (input_sum.x * float(sm0.y) + input_sum.y * float(sm1.y) + input_sum.z * float(sm2.y) + input_sum.w * float(sm3.y));
        }
    }
    for (uint t = 0; t < 4; ++t) if (first + t < p.tokens) {
        const bool fallback = simd_any(reference_required[t]);
        const float total = fallback
            ? hybrid_packed_block_dot(codes, input, row_at, hybrid_token_input(p, first + t, row), p.columns, lane)
            : simd_sum(sums[t]);
        if (lane == 0) {
            output[ulong(p.output_offset) + ulong(first + t) * p.rows + row] = total;
            if (!isfinite(total)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
        }
    }
}

kernel void hybrid_token_projection_q6(
    device const uchar *base [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant HybridTokenProjection &p [[buffer(3)]],
    device atomic_uint *failure [[buffer(4)]], uint2 grid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint first_row = grid.x * 4 + sg * 2, first = grid.y * 4;
    if (first_row >= p.rows) return;
    device const uchar *codes = base + p.codes_offset;
    const ulong row_bytes = ulong(p.columns >> 8) * p.block_bytes;
    const uint tid = lane >> 1, parity = lane & 1, half_block = tid >> 3, low_start = (tid & 7) * 4;
    const uint scale_start = half_block * 8 + (low_start >> 4), input_start = half_block * 128 + low_start;
    const uint low_offset = half_block * 64 + low_start, high_offset = 128 + half_block * 32 + low_start;
    float sums[4][2] = {{0.0f}};
    bool reference_required[4] = {false};
    for (uint bi = parity; bi < (p.columns >> 8); bi += 2) {
        float4 coefficients[2][4], scales[2]; float block_scale[2];
        for (uint r = 0; r < 2; ++r) if (first_row + r < p.rows) {
            device const uchar *block = codes + ulong(first_row + r) * row_bytes + ulong(bi) * p.block_bytes;
            for (uint l = 0; l < 4; ++l) {
                const uint a = block[low_offset + l], b = block[low_offset + l + 32], h = block[high_offset + l];
                coefficients[r][l] = float4(float(int((a & 15) | ((h & 3) << 4)) - 32),
                    float(int((b & 15) | ((h & 12) << 2)) - 32), float(int((a >> 4) | (h & 48)) - 32),
                    float(int((b >> 4) | ((h & 192) >> 2)) - 32));
            }
            scales[r] = float4(float(as_type<char>(block[192 + scale_start])), float(as_type<char>(block[194 + scale_start])),
                float(as_type<char>(block[196 + scale_start])), float(as_type<char>(block[198 + scale_start])));
            block_scale[r] = hybrid_half(block + 208);
        }
        for (uint t = 0; t < 4; ++t) if (first + t < p.tokens) {
            device const float *x = input + hybrid_token_input(p, first + t, first_row) + bi * 256 + input_start;
            float4 values[4];
            for (uint l = 0; l < 4; ++l) {
                values[l] = float4(x[l], x[l + 32], x[l + 64], x[l + 96]);
                reference_required[t] |= any(abs(values[l]) > float4(1e20f));
            }
            for (uint r = 0; r < 2; ++r) if (first_row + r < p.rows) {
                float4 partial = float4(0.0f);
                for (uint l = 0; l < 4; ++l) {
                    partial.x += values[l].x * coefficients[r][l].x;
                    partial.y += values[l].y * coefficients[r][l].y;
                    partial.z += values[l].z * coefficients[r][l].z;
                    partial.w += values[l].w * coefficients[r][l].w;
                }
                sums[t][r] += block_scale[r] * (partial.x * scales[r].x + partial.y * scales[r].y
                    + partial.z * scales[r].z + partial.w * scales[r].w);
            }
        }
    }
    for (uint t = 0; t < 4; ++t) if (first + t < p.tokens) {
        const bool fallback = simd_any(reference_required[t]);
        for (uint r = 0; r < 2; ++r) if (first_row + r < p.rows) {
            const uint row = first_row + r;
            const float total = fallback
                ? hybrid_packed_block_dot(codes, input, ulong(row) * row_bytes, hybrid_token_input(p, first + t, row), p.columns, lane)
                : simd_sum(sums[t][r]);
            if (lane == 0) {
                output[ulong(p.output_offset) + ulong(first + t) * p.rows + row] = total;
                if (!isfinite(total)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
            }
        }
    }
}

kernel void hybrid_mla_index_norm_batch(
    device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *output [[buffer(3)]], constant uint *p [[buffer(4)]],
    device atomic_uint *failure [[buffer(30)]], uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint length = p[0];
    threadgroup float moments[2];
    device const float *row = input + ulong(token) * length;
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < length; ++i) sum += row[i];
        const float mean = sum / float(length);
        float variance = 0.0f;
        for (uint i = 0; i < length; ++i) variance += (row[i] - mean) * (row[i] - mean);
        moments[0] = mean;
        moments[1] = sqrt(variance / float(length) + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < length; i += 256) {
        const float value = (row[i] - moments[0]) / moments[1] * weight[i] + bias[i];
        output[ulong(token) * length + i] = value;
        if (!isfinite(value)) atomic_fetch_or_explicit(failure, 1u, memory_order_relaxed);
    }
}
