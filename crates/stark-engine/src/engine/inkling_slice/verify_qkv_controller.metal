// SPDX-License-Identifier: Apache-2.0
// Short verifier. Each threadgroup uses the selected controller's original arithmetic.
#if defined(INK_QKV_NORM)
kernel void ink_verify_qkv_norm_pair(
    device const float *x0 [[buffer(0)]],
    device const float *x1 [[buffer(1)]],
    device const float *x2 [[buffer(2)]],
    device const float *x3 [[buffer(3)]],
    device const float *x4 [[buffer(4)]],
    device const float *x5 [[buffer(5)]],
    device const float *x6 [[buffer(6)]],
    device const float *x7 [[buffer(7)]],
    device const float *x8 [[buffer(8)]],
    device const float *x9 [[buffer(9)]],
    device const float *x10 [[buffer(10)]],
    device const float *x11 [[buffer(11)]],
    device const float *m0 [[buffer(12)]],
    device const float *m1 [[buffer(13)]],
    device const float *m2 [[buffer(14)]],
    device const float *m3 [[buffer(15)]],
    device float *raw [[buffer(16)]], device bfloat *normalized [[buffer(17)]],
    constant uint &rows [[buffer(18)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const device float *inputs[] = {x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11};
    const device float *messages[] = {m0, m1, m2, m3};
    const uint slot = group.y / rows, row = group.y % rows;
    const uint index = (slot * 4 + row) * 2 + group.x;
    const device float *input = group.x == 0 ? inputs[slot * 4 + row] : messages[row] + slot * ink_width;
    const float4 value = reinterpret_cast<device const float4 *>(input)[tid];
    reinterpret_cast<device float4 *>(raw + index * ink_width)[tid] = value;
    threadgroup float sums[18];
    threadgroup float scale;
    ink_norm_row(value, nullptr, normalized + index * ink_width, false, sums, scale, tid, simd, lane);
}

kernel void ink_verify_qkv_norm_output(
    device const float *mixed [[buffer(0)]],
    device const bfloat *w0 [[buffer(1)]],
    device const bfloat *w1 [[buffer(2)]],
    device const bfloat *w2 [[buffer(3)]],
    device bfloat *y0 [[buffer(4)]],
    device bfloat *y1 [[buffer(5)]],
    device bfloat *y2 [[buffer(6)]],
    device bfloat *y3 [[buffer(7)]],
    device bfloat *y4 [[buffer(8)]],
    device bfloat *y5 [[buffer(9)]],
    device bfloat *y6 [[buffer(10)]],
    device bfloat *y7 [[buffer(11)]],
    device bfloat *y8 [[buffer(12)]],
    device bfloat *y9 [[buffer(13)]],
    device bfloat *y10 [[buffer(14)]],
    device bfloat *y11 [[buffer(15)]],
    constant uint &rows [[buffer(16)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const device bfloat *weights[] = {w0, w1, w2};
    device bfloat *outputs[] = {y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11};
    const uint slot = group / rows;
    const uint index = slot * 4 + group % rows;
    const float4 value = reinterpret_cast<device const float4 *>(mixed + index * ink_width)[tid];
    threadgroup float sums[18];
    threadgroup float scale;
    ink_norm_row(value, weights[slot], outputs[index], true, sums, scale, tid, simd, lane);
}
#elif defined(INK_QKV_RANK)
kernel void ink_verify_qkv_rank(
    device const bfloat4 *w0 [[buffer(0)]],
    device const bfloat4 *w1 [[buffer(1)]],
    device const bfloat4 *w2 [[buffer(2)]],
    device const bfloat4 *normalized [[buffer(3)]], device bfloat *rank [[buffer(4)]],
    constant uint &rows [[buffer(5)]], uint2 group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint cols = 2 * 2304;
    const device bfloat4 *weights[] = {w0, w1, w2};
    const device bfloat4 *input = normalized + group.y * cols;
    device bfloat *output = rank + group.y * 4 * 64;
    device const bfloat4 *inputs[] = {
        input, input + cols / 4, input + min(2u, rows - 1) * cols / 4, input + (rows - 1) * cols / 4,
    };
    device bfloat *outputs[] = {output, output + 64, output + 128, output + 192};
    ink_matvec_prefill_shared<4>(weights[group.y], inputs, outputs, uint3(cols, 64, rows), group.x, simd, lane);
}
#elif defined(INK_QKV_EXPAND)
kernel void ink_verify_qkv_expand(
    device bfloat *rank [[buffer(0)]],
    device bfloat *g0 [[buffer(1)]],
    device bfloat *g1 [[buffer(2)]],
    device bfloat *g2 [[buffer(3)]],
    device bfloat *f0 [[buffer(4)]],
    device bfloat *f1 [[buffer(5)]],
    device bfloat *f2 [[buffer(6)]],
    device const float *raw [[buffer(7)]], device float *mixed [[buffer(8)]],
    constant uint &rows [[buffer(9)]], uint2 group [[threadgroup_position_in_grid]]) {
    const uint start = group.y * 4;
    device bfloat *gains[] = {g0, g1, g2};
    device bfloat *forgets[] = {f0, f1, f2};
    expand_native<16, 16>(rank + start * 64, gains[group.y], forgets[group.y],
        raw + start * 2 * 2304, mixed + start * 2304, rows, group.x);
}
#endif
