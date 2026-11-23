// SPDX-License-Identifier: Apache-2.0
// Shared-weight NVFP4 projection for two through eight verifier rows.
// Each output keeps the serial 16-channel blocks, per-lane scaled FMA
// recurrence and final SIMD reduction. Route order is restored by the
// existing grouped gate/up and down epilogues.
#ifdef INK_VERIFY_BYTE_COEFFICIENTS
// Every byte encodes two FP4 coefficients, low nibble first. Generate a
// 1 KiB constant table of exact FP16 bit patterns, including signed zero.
constexpr uint ink_verify_fp4_half_bits(uint code) {
    const uint magnitude = code & 7u;
    const uint positive = magnitude == 0 ? 0 : (magnitude + 28u - uint(magnitude == 1u)) << 9;
    return positive | ((code & 8u) << 12);
}

constexpr uint ink_verify_fp4_pair_bits(uint low, uint high) {
    return ink_verify_fp4_half_bits(low) | (ink_verify_fp4_half_bits(high) << 16);
}

#define INK_FP4_PAIR_ROW(high) \
    ink_verify_fp4_pair_bits(0, high), ink_verify_fp4_pair_bits(1, high), \
    ink_verify_fp4_pair_bits(2, high), ink_verify_fp4_pair_bits(3, high), \
    ink_verify_fp4_pair_bits(4, high), ink_verify_fp4_pair_bits(5, high), \
    ink_verify_fp4_pair_bits(6, high), ink_verify_fp4_pair_bits(7, high), \
    ink_verify_fp4_pair_bits(8, high), ink_verify_fp4_pair_bits(9, high), \
    ink_verify_fp4_pair_bits(10, high), ink_verify_fp4_pair_bits(11, high), \
    ink_verify_fp4_pair_bits(12, high), ink_verify_fp4_pair_bits(13, high), \
    ink_verify_fp4_pair_bits(14, high), ink_verify_fp4_pair_bits(15, high)

constant uint ink_verify_fp4_half_pairs[256] = {
    INK_FP4_PAIR_ROW(0), INK_FP4_PAIR_ROW(1), INK_FP4_PAIR_ROW(2), INK_FP4_PAIR_ROW(3),
    INK_FP4_PAIR_ROW(4), INK_FP4_PAIR_ROW(5), INK_FP4_PAIR_ROW(6), INK_FP4_PAIR_ROW(7),
    INK_FP4_PAIR_ROW(8), INK_FP4_PAIR_ROW(9), INK_FP4_PAIR_ROW(10), INK_FP4_PAIR_ROW(11),
    INK_FP4_PAIR_ROW(12), INK_FP4_PAIR_ROW(13), INK_FP4_PAIR_ROW(14), INK_FP4_PAIR_ROW(15),
};
#undef INK_FP4_PAIR_ROW

inline float4 ink_verify_fp4_coefficients(uint word) {
    const uint2 pairs(
        ink_verify_fp4_half_pairs[word & 255u], ink_verify_fp4_half_pairs[(word >> 8) & 255u]);
    return float4(as_type<half4>(pairs));
}
#else
inline float4 ink_verify_fp4_coefficients(uint word) {
    return ink_nvfp4_nibbles(word);
}
#endif

#ifdef INK_VERIFY_BF16_INPUT
// Unquantized expert inputs read from the BF16 rows, each value widened
// through binary16 exactly as the unquantized half activation pass stores it.
typedef bfloat ink_verify_input_word;
inline float4 ink_verify_widen(vec<bfloat, 4> x) {
    return float4(half4(float4(x)));
}
#else
typedef ink_nvfp4_activation_word ink_verify_input_word;
inline float4 ink_verify_widen(vec<ink_nvfp4_activation_word, 4> x) {
    return float4(x);
}
#endif

template <uint Batch, uint Width>
inline void ink_verify_nvfp4_route_dot(
    device const ink_verify_input_word *input, device const uchar *codes, device const uchar *scales,
    device const uint *order, device float *output,
    uint start, uint count, uint out_col, uint lane) {
    constexpr bool up = Width == 2304;
    uint offsets[Batch];
    float sums[Batch] = {};
    for (uint row = 0; row < Batch; ++row) {
        offsets[row] = up ? order[start + min(row, count - 1)] / 8 * Width : (start + row) * Width;
    }
    for (uint block = lane; block < Width / 16; block += 32) {
        const uint2 packed = reinterpret_cast<device const uint2 *>(codes)[block];
        const float scale = ink_e4m3_positive(scales[block]);
        const float4 w0 = ink_verify_fp4_coefficients(packed.x), w1 = ink_verify_fp4_coefficients(packed.x >> 16);
        const float4 w2 = ink_verify_fp4_coefficients(packed.y), w3 = ink_verify_fp4_coefficients(packed.y >> 16);
#pragma clang loop unroll(full)
        for (uint row = 0; row < Batch; ++row) {
            if (row < count) {
                // The half ingress stores exactly the same decoded FP4 ×
                // E4M3 coefficients. Widen before the unchanged F32 dot.
                const device auto *v = reinterpret_cast<device const vec<ink_verify_input_word, 4> *>(
                    input + offsets[row]) + block * 4;
                const float dot16 = (dot(w0, ink_verify_widen(v[0])) + dot(w1, ink_verify_widen(v[1]))) +
                    (dot(w2, ink_verify_widen(v[2])) + dot(w3, ink_verify_widen(v[3])));
                sums[row] = fma(dot16, scale, sums[row]);
            }
        }
    }
#pragma clang loop unroll(full)
    for (uint row = 0; row < Batch; ++row) {
        if (row < count) {
            const float value = simd_sum(sums[row]);
            if (lane == 0) output[(start + row) * 2304 + out_col] = value;
        }
    }
}

template <uint Width>
inline void ink_verify_nvfp4_route_tile(
    device const ink_verify_input_word *input, device const uchar *codes, device const uchar *scales,
    device const uint *order, device float *output,
    uint start, uint count, uint expert, uint out_col, uint lane) {
    const uint channel = expert * 2304 + out_col;
    const device uchar *weights = codes + channel * (Width / 2);
    const device uchar *gain = scales + channel * (Width / 16);
#ifdef INK_VERIFY_EXACT_RUN_WIDTH
    // Dispatch fixed-size tails so a two- or three-route run does not
    // carry four accumulators and per-route guards into every K16 block.
    for (; count >= 4; count -= 4, start += 4)
        ink_verify_nvfp4_route_dot<4, Width>(input, weights, gain, order, output, start, 4, out_col, lane);
    switch (count) {
        case 3: ink_verify_nvfp4_route_dot<3, Width>(input, weights, gain, order, output, start, 3, out_col, lane); break;
        case 2: ink_verify_nvfp4_route_dot<2, Width>(input, weights, gain, order, output, start, 2, out_col, lane); break;
        case 1: ink_verify_nvfp4_route_dot<1, Width>(input, weights, gain, order, output, start, 1, out_col, lane); break;
        default: break;
    }
#else
    if (count == 1) {
        ink_verify_nvfp4_route_dot<1, Width>(input, weights, gain, order, output, start, 1, out_col, lane);
    } else {
        for (uint offset = 0; offset < count; offset += 4) {
            ink_verify_nvfp4_route_dot<4, Width>(
                input, weights, gain, order, output, start + offset, min(4u, count - offset), out_col, lane);
        }
    }
#endif
}

kernel void ink_verify_nvfp4_group_dot(
    device const ink_verify_input_word *input [[buffer(0)]],
    device const uchar *codes [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device const uint *order [[buffer(3)]],
    device const uint *runs [[buffer(4)]],
    device float *output [[buffer(5)]],
#ifdef INK_VERIFY_BF16_INPUT
    // The epilogues' row scales: the unquantized activation's constant 1536
    // for each of shape.y rows, stored in place of the skipped pass.
    device float *amax [[buffer(6)]],
    constant uint2 &shape [[buffer(7)]],
#else
    constant uint &width [[buffer(6)]],
#endif
    uint2 tile [[threadgroup_position_in_grid]], uint group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
#ifdef INK_VERIFY_BF16_INPUT
    const uint width = shape.x;
    if (tile.x == 0 && tile.y == 0 && group == 0) {
        for (uint i = lane; i < shape.y; i += 32) amax[i] = 1536.0f;
    }
#endif
    const uint count = runs[tile.y * 3 + 1];
    if (count == 0) return;
    const uint start = runs[tile.y * 3], expert = runs[tile.y * 3 + 2];
    const uint out_col = tile.x * 4 + group;
    if (width == 2304) {
        ink_verify_nvfp4_route_tile<2304>(input, codes, scales, order, output, start, count, expert, out_col, lane);
    } else {
        ink_verify_nvfp4_route_tile<1152>(input, codes, scales, order, output, start, count, expert, out_col, lane);
    }
}
