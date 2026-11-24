// SPDX-License-Identifier: Apache-2.0
//! Tensor MXFP8 prefill with independent BF16 certification and source replay.

#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

constant constexpr uint MX_COLS = 2304;

inline float mx_scale(uint code) {
    return code == 0 ? 0x1p-127f : code == 255 ? NAN : as_type<float>(code << 23);
}

inline bool mx_admissible(float value) {
    const float magnitude = abs(value);
    return (magnitude == 0 || magnitude >= 0x1p-56f) &&
        magnitude <= 0x1p56f && float(bfloat(value)) == value;
}

kernel void ink_mx_weight_norm(
    device const uchar *codes [[buffer(0)]], device const uchar *exponents [[buffer(1)]],
    device float *norms [[buffer(2)]], constant uint &width [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group * 4 + simd;
    float squares = 0;
    bool valid = true;
    for (uint i = lane; i < width; i += 32) {
        const uint scale = exponents[row * (width / 32) + i / 32];
        const float decoded = ink_mx_e4m3(codes[row * width + i]);
        const float v = decoded * mx_scale(scale);
        valid &= scale != 255u && (decoded == 0.f || (scale != 0u && v != 0.f && mx_admissible(v)));
        squares = fma(v, v, squares);
    }
    const bool all_valid = simd_all(valid);
    const float norm = sqrt(simd_sum(squares));
    if (lane == 0) norms[row] = all_valid ? norm : -1.f;
}

kernel void ink_mx_tensor_ingress(
    device const float *input [[buffer(0)]], device const uint *order [[buffer(1)]],
    device bfloat *packed [[buffer(2)]], device float *norms [[buffer(3)]],
    constant uint2 &shape [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group * 4 + simd;
    if (row >= shape.y) return;
    const uint width = shape.x;
    const uint from = width == MX_COLS ? order[row] / 8 : row;
    float squares = 0;
    bool valid = true;
    for (uint i = lane; i < width; i += 32) {
        const float v = input[from * width + i];
        packed[row * width + i] = bfloat(v);
        valid &= mx_admissible(v);
        squares = fma(v, v, squares);
    }
    const bool all_valid = simd_all(valid);
    const float norm = sqrt(simd_sum(squares));
    if (lane == 0) norms[row] = all_valid ? norm : -1.f;
}

// Compare outward-rounded endpoints in magnitude space and keep the sign
// only if the whole interval stays away from zero. The error model assumes
// binary32 additions in independent K32 tensor partials and compensated FP32 merging.
inline bool mx_certify(float value, float a_norm, float b_norm, uint width) {
    if (!(a_norm >= 0.f && b_norm >= 0.f)) return false;
    const float magnitude = abs(value);
    const float coefficient = (width == MX_COLS ? 88.f : 84.f) * 0x1p-24f;
    const float radius = (a_norm * b_norm) * coefficient + 0x1p-126f;
    if (!(magnitude > radius) || !isfinite(magnitude)) return false;
    const float lo = magnitude - radius, hi = magnitude + radius;
    if (!(lo > 0.f) || !isfinite(hi)) return false;
    const uint low_bits = as_type<uint>(lo) - 1u;
    const uint high_bits = as_type<uint>(hi) + 1u;
    return as_type<ushort>(bfloat(as_type<float>(low_bits))) ==
           as_type<ushort>(bfloat(as_type<float>(high_bits)));
}

kernel void ink_mx_tensor_product(
    device const bfloat *input [[buffer(0)]], device const uint *runs [[buffer(1)]],
    device const uchar *codes [[buffer(2)]], device const uchar *exponents [[buffer(3)]],
    device const float *input_norms [[buffer(4)]], device const float *weight_norms [[buffer(5)]],
    device bfloat *rounded [[buffer(6)]], device uint *masks [[buffer(7)]],
    device atomic_uint *counts [[buffer(8)]], constant uint2 &params [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]], uint worker [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
#pragma clang fp contract(off) reassociate(off)
    constexpr uint BN = 64, BK = 32, BM = 32, Workers = 128;
    const uint start = runs[group.y * 3], rows = runs[group.y * 3 + 1];
    if (rows == 0) return;
    const uint width = params.x;
    const uint expert = runs[group.y * 3 + 2];
    const uint col0 = group.x * BN;
    const uint tile_row = (sg / 2) * 16, tile_col = (sg % 2) * 32;
    threadgroup bfloat decoded[BN * BK];
    threadgroup float final[BM * BN];
    using Shape = metal::dextents<int32_t, 2>;
    using Strides = metal::array<int32_t, 2>;
    using A = metal::tensor<device bfloat, Shape, metal::tensor_inline>;
    using B = metal::tensor<threadgroup bfloat, Shape, metal::tensor_inline>;
    using C = metal::tensor<threadgroup float, Shape, metal::tensor_inline>;
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, BK, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> product;
    auto partial = product.template get_destination_cooperative_tensor<A, B, float>();
    float total[16] = {};
    float error[16] = {};
    for (uint offset = 0; offset < width; offset += BK) {
        for (uint i = worker; i < BN * (BK / 4); i += Workers) {
            const uint wr = expert * MX_COLS + col0 + i / (BK / 4);
            const uint k = offset + (i % (BK / 4)) * 4;
            const float scale = mx_scale(exponents[wr * (width / 32) + k / 32]);
            const uchar4 packed = reinterpret_cast<device const uchar4 *>(codes)[(wr * width + k) / 4];
            for (uint j = 0; j < 4; ++j) decoded[i * 4 + j] = bfloat(ink_mx_e4m3(packed[j]) * scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tile_row < rows) {
            const int valid = int(min(16u, rows - tile_row));
            auto lhs = A((device bfloat *)input + (start + tile_row) * width + offset,
                         Shape(BK, valid), Strides{1, int(width)});
            auto rhs = B(decoded + tile_col * BK, Shape(BK, 32), Strides{1, BK});
            for (uint i = 0; i < partial.get_capacity(); ++i) partial[i] = 0.f;
            product.run(lhs, rhs, partial);
            // TwoSum keeps each rounded K32 merge and its exact residual.
            // Only the sum of residuals and the final addition can lose bits.
            for (uint i = 0; i < partial.get_capacity(); ++i) {
                const float old = total[i], add = partial[i];
                const float combined = old + add;
                const float high = combined - old;
                error[i] += (old - (combined - high)) + (add - high);
                total[i] = combined;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < partial.get_capacity(); ++i) partial[i] = total[i] + error[i];
    partial.store(C(final + tile_row * BN + tile_col, Shape(32, 16), Strides{1, BN}));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each lane keeps the mask for one column of its own 16-row fragment.
    // Pair the two fragments for the 32-row source replay run.
    threadgroup uint lower_masks[BN];
    const uint i = tile_col + lane, col = col0 + i;
    const float b_norm = weight_norms[expert * MX_COLS + col];
    uint mask = 0;
    for (uint r = tile_row; r < min(tile_row + 16, rows); ++r) {
        const uint row = start + r;
        const float value = final[r * BN + i];
        const bool needs_replay = !mx_certify(value, input_norms[row], b_norm, width);
        rounded[row * MX_COLS + col] = bfloat(value);
        mask |= uint(needs_replay) << r;
    }
    if (tile_row == 0) lower_masks[i] = mask;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tile_row != 0) masks[group.y * MX_COLS + col] = lower_masks[i] | mask;
    const uint rejected = popcount(mask);
    if (params.y != 0) {
        const uint simd_rejected = simd_sum(rejected);
        if (lane == 0 && simd_rejected != 0) {
            atomic_fetch_add_explicit(counts + (params.y - 1), simd_rejected, memory_order_relaxed);
        }
    }
}

// Default source prefill: one F32 tensor accumulation across K with the BF16
// projection rounding at the store, no certification or replay.
kernel void ink_mx_tensor_fast(
    device const bfloat *input [[buffer(0)]], device const uint *runs [[buffer(1)]],
    device const uchar *codes [[buffer(2)]], device const uchar *exponents [[buffer(3)]],
    device bfloat *rounded [[buffer(4)]], constant uint &width [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint worker [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint BN = 64, BK = 64, BM = 32, Workers = 128;
    const uint start = runs[group.y * 3], rows = runs[group.y * 3 + 1];
    if (rows == 0) return;
    const uint expert = runs[group.y * 3 + 2];
    const uint col0 = group.x * BN;
    const uint tile_row = (sg / 2) * 16, tile_col = (sg % 2) * 32;
    threadgroup bfloat decoded[BN * BK];
    threadgroup float final[BM * BN];
    using Shape = metal::dextents<int32_t, 2>;
    using Strides = metal::array<int32_t, 2>;
    using A = metal::tensor<device bfloat, Shape, metal::tensor_inline>;
    using B = metal::tensor<threadgroup bfloat, Shape, metal::tensor_inline>;
    using C = metal::tensor<threadgroup float, Shape, metal::tensor_inline>;
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, BK, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroup> product;
    auto partial = product.template get_destination_cooperative_tensor<A, B, float>();
    for (uint i = 0; i < partial.get_capacity(); ++i) partial[i] = 0.f;
    const int valid = int(min(16u, rows - min(rows, tile_row)));
    for (uint offset = 0; offset < width; offset += BK) {
        for (uint i = worker; i < BN * (BK / 4); i += Workers) {
            const uint wr = expert * MX_COLS + col0 + i / (BK / 4);
            const uint k = offset + (i % (BK / 4)) * 4;
            const float scale = mx_scale(exponents[wr * (width / 32) + k / 32]);
            const uchar4 packed = reinterpret_cast<device const uchar4 *>(codes)[(wr * width + k) / 4];
            for (uint j = 0; j < 4; ++j) decoded[i * 4 + j] = bfloat(ink_mx_e4m3(packed[j]) * scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (valid > 0) {
            auto lhs = A((device bfloat *)input + (start + tile_row) * width + offset,
                         Shape(BK, valid), Strides{1, int(width)});
            auto rhs = B(decoded + tile_col * BK, Shape(BK, 32), Strides{1, BK});
            product.run(lhs, rhs, partial);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    partial.store(C(final + tile_row * BN + tile_col, Shape(32, 16), Strides{1, BN}));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = worker; i < rows * BN; i += Workers) {
        rounded[(start + i / BN) * MX_COLS + col0 + i % BN] = bfloat(final[i]);
    }
}

kernel void ink_mx_replay_reset(device atomic_uint *count [[buffer(0)]]) {
    atomic_store_explicit(count, 0u, memory_order_relaxed);
}

template <uint Width, uint ValuesPerLane>
inline void mx_replay_column(
    device const float *input, device const uint *order,
    device const uchar *weights, device const uchar *scales,
    device bfloat *rounded, uint col, uint start, uint projected_rows, uint mask, uint lane,
    device atomic_uint *false_certificates) {
#ifdef INK_MX_AUDIT
    const uint replay_mask = mask;
    mask = ~0u >> (32 - projected_rows);
    uint false_count = 0;
#endif
    while (mask != 0) {
        uint4 sorted = 0, source = 0;
        uint count = 0;
        for (; count < 4 && mask != 0; ++count) {
            sorted[count] = start + ctz(mask);
            source[count] = Width == MX_COLS ? order[sorted[count]] / 8 : sorted[count];
            mask &= mask - 1;
        }
        const float4 values = count == 4 ?
            ink_mxfp8_source_dot<Width, ValuesPerLane, true>(input, source, 4, weights, scales, lane) :
            ink_mxfp8_source_dot<Width, ValuesPerLane, false>(input, source, count, weights, scales, lane);
        if (lane == 0) {
            for (uint i = 0; i < count; ++i) {
                const uint offset = sorted[i] * MX_COLS + col;
                const bfloat source_value = bfloat(values[i]);
#ifdef INK_MX_AUDIT
                if ((replay_mask & (1u << (sorted[i] - start))) == 0) {
                    false_count += uint(as_type<ushort>(rounded[offset]) != as_type<ushort>(source_value));
                    continue;
                }
#endif
                rounded[offset] = source_value;
            }
        }
    }
#ifdef INK_MX_AUDIT
    if (lane == 0 && false_count != 0) {
        atomic_fetch_add_explicit(false_certificates, false_count, memory_order_relaxed);
    }
#endif
}

// The source recurrence of ink_mxfp8_source_dot reading the sorted BF16
// ingress rows. Admissible rows hold exactly the F32 inputs, so every
// product and sum is the same; rows without a norm use the F32 kernel.
constant constexpr uint MX_REPLAY_ROWS = 4;

template <uint Width, uint ValuesPerLane>
inline void ink_mx_source_dot_bf16(
    device const bfloat *input, thread const uint *rows, uint count,
    device const uchar *weights, device const uchar *exponents, uint lane, thread float *result) {
#pragma clang fp contract(off) reassociate(off)
    float sum[MX_REPLAY_ROWS] = {};
    const device bfloat4 *input4 = reinterpret_cast<const device bfloat4 *>(input);
    const device uchar4 *weight4 = reinterpret_cast<const device uchar4 *>(weights);
    for (uint start = 0; start < Width; start += ValuesPerLane * 32) {
        const uint begin = start + lane * ValuesPerLane;
        const uint exponent = uint(exponents[begin / 32]);
        const float scale = exponent == 0u ? 0x1p-127f :
                            exponent == 255u ? NAN : as_type<float>(exponent << 23);
        float partial[MX_REPLAY_ROWS] = {};
        for (uint j = 0; j < ValuesPerLane; j += 4) {
            const uint block = (begin + j) / 4;
            const uchar4 code = weight4[block];
            const float4 weight = float4(ink_mx_e4m3(code.x), ink_mx_e4m3(code.y),
                                         ink_mx_e4m3(code.z), ink_mx_e4m3(code.w));
            for (uint r = 0; r < MX_REPLAY_ROWS; ++r) {
                if (r < count)
                    partial[r] = ink_mxfp8_accumulate_four(
                        partial[r], float4(input4[rows[r] * (Width / 4) + block]), weight);
            }
        }
        for (uint r = 0; r < MX_REPLAY_ROWS; ++r) sum[r] += scale * partial[r];
    }
    for (uint r = 0; r < MX_REPLAY_ROWS; ++r) {
        if (r < count) result[r] = simd_sum(sum[r]);
    }
}

template <uint Width, uint ValuesPerLane>
inline void mx_replay_column_bf16(
    device const float *input, device const bfloat *packed, device const float *norms,
    device const uint *order, device const uchar *weights, device const uchar *scales,
    device bfloat *rounded, uint col, uint start, uint mask, uint lane) {
    while (mask != 0) {
        uint sorted[MX_REPLAY_ROWS] = {};
        float values[MX_REPLAY_ROWS] = {};
        uint count = 0;
        bool admissible = true;
        for (; count < MX_REPLAY_ROWS && mask != 0; ++count) {
            sorted[count] = start + ctz(mask);
            admissible &= norms[sorted[count]] >= 0.f;
            mask &= mask - 1;
        }
        if (admissible) {
            ink_mx_source_dot_bf16<Width, ValuesPerLane>(packed, sorted, count, weights, scales, lane, values);
        } else {
            for (uint first = 0; first < count; first += 4) {
                const uint chunk = min(4u, count - first);
                uint4 source = 0;
                for (uint i = 0; i < chunk; ++i)
                    source[i] = Width == MX_COLS ? order[sorted[first + i]] / 8 : sorted[first + i];
                const float4 four =
                    ink_mxfp8_source_dot<Width, ValuesPerLane, false>(input, source, chunk, weights, scales, lane);
                for (uint i = 0; i < chunk; ++i) values[first + i] = four[i];
            }
        }
        if (lane == 0) {
            for (uint i = 0; i < count; ++i) rounded[sorted[i] * MX_COLS + col] = bfloat(values[i]);
        }
    }
}

kernel void ink_mx_replay_packed(
    device const float *input [[buffer(0)]], device const uint *order [[buffer(1)]],
    device const uint *runs [[buffer(2)]], device const uchar *codes [[buffer(3)]],
    device const uchar *exponents [[buffer(4)]], device bfloat *rounded [[buffer(5)]],
    device const uint *masks [[buffer(6)]], device const bfloat *packed [[buffer(7)]],
    device const float *norms [[buffer(8)]], constant uint &width [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint rows = runs[group.y * 3 + 1];
    if (rows == 0) return;
    const uint col = group.x * 4 + sg;
    const uint mask = masks[group.y * MX_COLS + col];
    if (mask == 0) return;
    const uint start = runs[group.y * 3];
    const uint wr = runs[group.y * 3 + 2] * MX_COLS + col;
    const device uchar *w = codes + wr * width;
    const device uchar *s = exponents + wr * (width / 32);
    if (width == MX_COLS) {
        mx_replay_column_bf16<MX_COLS, 8>(input, packed, norms, order, w, s, rounded, col, start, mask, lane);
    } else {
        mx_replay_column_bf16<MX_COLS / 2, 4>(input, packed, norms, order, w, s, rounded, col, start, mask, lane);
    }
}

kernel void ink_mx_replay_source(
    device const float *input [[buffer(0)]], device const uint *order [[buffer(1)]],
    device const uint *runs [[buffer(2)]], device const uchar *codes [[buffer(3)]],
    device const uchar *exponents [[buffer(4)]], device bfloat *rounded [[buffer(5)]],
    device const uint *masks [[buffer(6)]], device atomic_uint *false_certificates [[buffer(7)]],
    constant uint2 &params [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint rows = runs[group.y * 3 + 1];
    if (rows == 0) return;
    const uint col = group.x * 4 + sg;
    const uint mask = masks[group.y * MX_COLS + col];
#ifndef INK_MX_AUDIT
    if (mask == 0) return;
#endif
    const uint width = params.x;
    const uint start = runs[group.y * 3];
    const uint wr = runs[group.y * 3 + 2] * MX_COLS + col;
    const device uchar *w = codes + wr * width;
    const device uchar *s = exponents + wr * (width / 32);
    if (width == MX_COLS) {
        mx_replay_column<MX_COLS, 8>(input, order, w, s, rounded, col, start, rows, mask, lane,
                                    false_certificates + params.y);
    } else {
        mx_replay_column<MX_COLS / 2, 4>(input, order, w, s, rounded, col, start, rows, mask, lane,
                                        false_certificates + params.y);
    }
}

kernel void ink_mx_tensor_finish_up(
    device const bfloat *rounded [[buffer(0)]], device const uint *order [[buffer(1)]],
    device const float *route_weights [[buffer(2)]], device bfloat *output [[buffer(3)]],
    uint2 at [[thread_position_in_grid]]) {
    if (at.x >= MX_COLS / 2) return;
    const uint i = at.y * MX_COLS + 2 * at.x;
    const float gate = float(rounded[i]), up = float(rounded[i + 1]);
    const float low = 1.0f / (1.0f + metal::precise::exp(abs(gate)));
    const float sigmoid = gate < 0.0f ? low : 1.0f - low;
    const float gate_cap = 4.0f * metal::precise::tanh(gate * 0.25f);
    const float up_cap = 25.0f * metal::precise::tanh(up / 25.0f);
    output[at.y * (MX_COLS / 2) + at.x] = bfloat(((gate_cap * sigmoid) * up_cap) * route_weights[order[at.y]]);
}

kernel void ink_mx_tensor_finish_down(
    device const bfloat *rounded [[buffer(0)]], device const uint *order [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    uint2 at [[thread_position_in_grid]]) {
    if (at.x >= MX_COLS) return;
    output[order[at.y] * MX_COLS + at.x] = rounded[at.y * MX_COLS + at.x];
}
