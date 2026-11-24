// SPDX-License-Identifier: Apache-2.0
// Compiled with serial.metal to share the source RMS reduction.

kernel void ink_prefill_controller_norm_pair(
    device const float *x0 [[buffer(0)]], device const float *x1 [[buffer(1)]],
    device const float *x2 [[buffer(2)]], device const float *x3 [[buffer(3)]],
    device const float *m0 [[buffer(4)]], device const float *m1 [[buffer(5)]],
    device const float *m2 [[buffer(6)]], device const float *m3 [[buffer(7)]],
    device bfloat *y0 [[buffer(8)]], device bfloat *y1 [[buffer(9)]],
    device bfloat *y2 [[buffer(10)]], device bfloat *y3 [[buffer(11)]],
    device float *packed [[buffer(12)]],
    uint2 row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    const device float *inputs[] = {x0, x1, x2, x3};
    const device float *messages[] = {m0, m1, m2, m3};
    device bfloat *outputs[] = {y0, y1, y2, y3};
    const device float *x = row.x == 0 ? inputs[row.y] : messages[row.y];
    device float *raw = packed + (row.y * 2 + row.x) * ink_width;
    const float4 value = reinterpret_cast<device const float4 *>(x)[tid];
    reinterpret_cast<device float4 *>(raw)[tid] = value;
    threadgroup float sums[18];
    threadgroup float scale;
    ink_norm_row(value, nullptr, outputs[row.y] + row.x * ink_width, false, sums, scale, tid, simd, lane);
}

kernel void ink_prefill_controller_pack_rank(
    device const bfloat *x0 [[buffer(0)]], device const bfloat *x1 [[buffer(1)]],
    device const bfloat *x2 [[buffer(2)]], device const bfloat *x3 [[buffer(3)]],
    device bfloat *out [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
) {
    const device bfloat *inputs[] = {x0, x1, x2, x3};
    out[row * 64 + tid] = inputs[row][tid];
}

kernel void ink_prefill_controller_norm_output(
    device const float *x [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device bfloat *y0 [[buffer(2)]], device bfloat *y1 [[buffer(3)]],
    device bfloat *y2 [[buffer(4)]], device bfloat *y3 [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    device bfloat *outputs[] = {y0, y1, y2, y3};
    threadgroup float sums[18];
    threadgroup float scale;
    const float4 value = reinterpret_cast<device const float4 *>(x + row * ink_width)[tid];
    ink_norm_row(value, weight, outputs[row], true, sums, scale, tid, simd, lane);
}
