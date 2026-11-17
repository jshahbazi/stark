// SPDX-License-Identifier: Apache-2.0
// Stark: row-batched entry points for the ordinary decode projection bodies.
// Compiled after quant.metal, which owns the unmodified affine dot helpers.

#define STARK_DECODE_ROWS_KERNEL(NAME, CODE_BODY, CT, STORE)                  \
kernel void NAME(device const uint* codes [[buffer(0)]],                     \
                 device const bfloat* scales [[buffer(1)]],                  \
                 device const bfloat* biases [[buffer(2)]],                  \
                 device const bfloat* input [[buffer(3)]],                   \
                 device CT* output [[buffer(4)]],                            \
                 constant uint& K [[buffer(5)]],                             \
                 constant uint& GS [[buffer(6)]],                            \
                 constant uint& N [[buffer(7)]],                             \
                 uint2 tg [[threadgroup_position_in_grid]],                  \
                 uint lane [[thread_index_in_threadgroup]]) {                \
    const uint row = tg.y;                                                     \
    device const bfloat* x = input + (ulong)tg.x * K;                         \
    device CT* y = output + (ulong)tg.x * N;                                  \
    CODE_BODY(STORE)                                                          \
}

STARK_DECODE_ROWS_KERNEL(stark_decode_rows_q4_bf16, GEMV_Q4_BODY, bfloat, y[row] = bfloat(sum))
STARK_DECODE_ROWS_KERNEL(stark_decode_rows_q4_f32, GEMV_Q4_BODY, float, y[row] = sum)
STARK_DECODE_ROWS_KERNEL(stark_decode_rows_q8_bf16, GEMV_Q8_BODY, bfloat, y[row] = bfloat(sum))
STARK_DECODE_ROWS_KERNEL(stark_decode_rows_q8_f32, GEMV_Q8_BODY, float, y[row] = sum)

kernel void stark_decode_rows_q4_bf16_2row(
    device const uint* codes [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const bfloat* input [[buffer(3)]],
    device bfloat* output [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& GS [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    constexpr uint LANES = 16;
    const uint sublane = lane % LANES;
    const uint row = tg.y * 2 + lane / LANES;
    device const bfloat* x = input + (ulong)tg.x * K;
    const float sum = gemv_q4_packed_sum<2>(codes, scales, biases, x, row, sublane, K, GS);
    if (sublane == 0) {
        output[(ulong)tg.x * N + row] = bfloat(sum);
    }
}

// Reuse packed weights across independent inputs. Q4 uses the existing packed
// helper; Q8 retains GEMV_Q8_BODY's per-input expression and SIMD32 reduction.
template <uint M>
inline void stark_projection_reuse_q4_body(
    device const uint* codes, device const bfloat* scales,
    device const bfloat* biases, device const bfloat* input,
    device bfloat* output, uint K, uint GS, uint N, uint group, uint lane) {
    const uint sublane = lane % 16;
    const uint channel = group * 2 + lane / 16;
    float sums[M];
    stark_gemv_q4_packed_rows_sum<M>(
        codes, scales, biases, input, channel, sublane, K, GS, sums);
    if (sublane == 0) {
        #pragma clang loop unroll(full)
        for (uint m = 0; m < M; ++m)
            output[(ulong)m * N + channel] = bfloat(sums[m]);
    }
}

template <uint M, typename CT>
inline void stark_projection_reuse_q8_body(
    device const uint* codes, device const bfloat* scales,
    device const bfloat* biases, device const bfloat* input,
    device CT* output, uint K, uint GS, uint N, uint row, uint lane) {
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
        if (lane == 0) output[(ulong)m * N + row] = CT(sum);
    }
}

#define STARK_PROJECTION_REUSE_Q4(M) \
kernel void stark_projection_reuse_q4_bf16_m##M( \
    device const uint* codes [[buffer(0)]], \
    device const bfloat* scales [[buffer(1)]], \
    device const bfloat* biases [[buffer(2)]], \
    device const bfloat* input [[buffer(3)]], \
    device bfloat* output [[buffer(4)]], \
    constant uint& K [[buffer(5)]], constant uint& GS [[buffer(6)]], \
    constant uint& N [[buffer(7)]], \
    uint group [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]]) { \
    stark_projection_reuse_q4_body<M>( \
        codes, scales, biases, input, output, K, GS, N, group, lane); \
}

#define STARK_PROJECTION_REUSE_Q8(M, CT, SUFFIX) \
kernel void stark_projection_reuse_q8_##SUFFIX##_m##M( \
    device const uint* codes [[buffer(0)]], \
    device const bfloat* scales [[buffer(1)]], \
    device const bfloat* biases [[buffer(2)]], \
    device const bfloat* input [[buffer(3)]], \
    device CT* output [[buffer(4)]], \
    constant uint& K [[buffer(5)]], constant uint& GS [[buffer(6)]], \
    constant uint& N [[buffer(7)]], \
    uint row [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_threadgroup]]) { \
    stark_projection_reuse_q8_body<M, CT>( \
        codes, scales, biases, input, output, K, GS, N, row, lane); \
}
STARK_PROJECTION_REUSE_Q4(2)
STARK_PROJECTION_REUSE_Q4(3)
STARK_PROJECTION_REUSE_Q4(4)
STARK_PROJECTION_REUSE_Q8(2, bfloat, bf16)
STARK_PROJECTION_REUSE_Q8(3, bfloat, bf16)
STARK_PROJECTION_REUSE_Q8(4, bfloat, bf16)
STARK_PROJECTION_REUSE_Q8(2, float, f32)
STARK_PROJECTION_REUSE_Q8(3, float, f32)
STARK_PROJECTION_REUSE_Q8(4, float, f32)
#undef STARK_PROJECTION_REUSE_Q4
#undef STARK_PROJECTION_REUSE_Q8
