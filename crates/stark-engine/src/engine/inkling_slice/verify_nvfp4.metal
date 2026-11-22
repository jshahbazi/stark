// SPDX-License-Identifier: Apache-2.0
// Shared-weight NVFP4 projection for two through eight verifier rows.
// Each output keeps the serial 16-channel blocks, per-lane scaled FMA
// recurrence and final SIMD reduction. Route order is restored by the
// existing grouped gate/up and down epilogues.
template <uint Batch, uint Width>
inline void ink_verify_nvfp4_route_dot(
    device const ink_nvfp4_activation_word *input, device const uchar *codes, device const uchar *scales,
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
        const float4 w0 = ink_nvfp4_nibbles(packed.x), w1 = ink_nvfp4_nibbles(packed.x >> 16);
        const float4 w2 = ink_nvfp4_nibbles(packed.y), w3 = ink_nvfp4_nibbles(packed.y >> 16);
#pragma clang loop unroll(full)
        for (uint row = 0; row < Batch; ++row) {
            if (row < count) {
                // The half ingress stores exactly the same decoded FP4 ×
                // E4M3 coefficients. Widen before the unchanged F32 dot.
                const device auto *v = reinterpret_cast<device const vec<ink_nvfp4_activation_word, 4> *>(
                    input + offsets[row]) + block * 4;
                const float dot16 = (dot(w0, float4(v[0])) + dot(w1, float4(v[1]))) +
                    (dot(w2, float4(v[2])) + dot(w3, float4(v[3])));
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
    device const ink_nvfp4_activation_word *input, device const uchar *codes, device const uchar *scales,
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
    device const ink_nvfp4_activation_word *input [[buffer(0)]],
    device const uchar *codes [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device const uint *order [[buffer(3)]],
    device const uint *runs [[buffer(4)]],
    device float *output [[buffer(5)]],
    constant uint &width [[buffer(6)]],
    uint2 tile [[threadgroup_position_in_grid]], uint group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
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
