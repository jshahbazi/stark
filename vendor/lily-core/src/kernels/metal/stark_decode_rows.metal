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
