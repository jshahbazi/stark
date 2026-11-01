// SPDX-License-Identifier: Apache-2.0
// Stark exact short-row adaptation of Lily quant.metal GEMV_Q8_BODY.
// Requires quant.metal and official_head.metal. This preserves the generic
// native Q8 dot_word_q8 expression and SIMD reduction, sharing packed-code
// loads across inputs and fusing the qualified integer-RNE BF16 value store.
// Compile with the ordinary generic Q8 pipeline's PlatformDefault math mode.

template <uint M>
inline void official_head_q8_generic_exact_rows_body(
    device const uint* codes,
    device const bfloat* scales,
    device const bfloat* biases,
    device const bfloat* input,
    device float* output,
    constant uint& K,
    constant uint& GS,
    constant uint& N,
    uint row,
    uint lane) {
    const uint words = K / 4;
    const uint blocks = words / 4;
    const uint bpg = GS / 16;
    const uint groups = K / GS;
    device const uint4* wrow = (device const uint4*)(codes + (ulong)row * words);
    float sums[M] = {};
    for (uint i = lane; i < blocks; i += 32) {
        const uint g = i / bpg;
        const float s = float(scales[row * groups + g]);
        const float b = float(biases[row * groups + g]);
        const uint4 w4 = wrow[i];
        #pragma clang loop unroll(full)
        for (uint m = 0; m < M; ++m) {
            device const bfloat4* xv =
                (device const bfloat4*)(input + (ulong)m * K);
            sums[m] += dot_word_q8(w4.x, s, b, xv, 4 * i)
                + dot_word_q8(w4.y, s, b, xv, 4 * i + 1)
                + dot_word_q8(w4.z, s, b, xv, 4 * i + 2)
                + dot_word_q8(w4.w, s, b, xv, 4 * i + 3);
        }
    }
    #pragma clang loop unroll(full)
    for (uint m = 0; m < M; ++m) {
        const float sum = simd_sum(sums[m]);
        if (lane == 0) output[(ulong)m * N + row] = official_head_bf16_value(sum);
    }
}

#define OFFICIAL_HEAD_Q8_GENERIC_EXACT_ROWS(M) \
kernel void official_head_q8_generic_exact_rows##M##_bf16_values_f32( \
    device const uint* codes [[buffer(0)]], \
    device const bfloat* scales [[buffer(1)]], \
    device const bfloat* biases [[buffer(2)]], \
    device const bfloat* input [[buffer(3)]], \
    device float* output [[buffer(4)]], \
    constant uint& K [[buffer(5)]], \
    constant uint& GS [[buffer(6)]], \
    constant uint& N [[buffer(7)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]]) { \
    official_head_q8_generic_exact_rows_body<M>( \
        codes, scales, biases, input, output, K, GS, N, row, lane); \
}
OFFICIAL_HEAD_Q8_GENERIC_EXACT_ROWS(1)
OFFICIAL_HEAD_Q8_GENERIC_EXACT_ROWS(2)
OFFICIAL_HEAD_Q8_GENERIC_EXACT_ROWS(3)
OFFICIAL_HEAD_Q8_GENERIC_EXACT_ROWS(4)
