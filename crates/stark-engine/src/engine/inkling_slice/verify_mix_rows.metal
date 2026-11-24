// SPDX-License-Identifier: Apache-2.0
// Each workgroup runs the same source reduction as the single-row kernel.
kernel void ink_verify_mix_rows(
    device const float *b0 [[buffer(0)]], device const float *b1 [[buffer(1)]],
    device const float *b2 [[buffer(2)]], device const float *b3 [[buffer(3)]],
    device const float *p0 [[buffer(4)]], device const float *p1 [[buffer(5)]],
    device const float *p2 [[buffer(6)]], device const float *p3 [[buffer(7)]],
    device float *o0 [[buffer(8)]], device float *o1 [[buffer(9)]],
    device float *o2 [[buffer(10)]], device float *o3 [[buffer(11)]],
    device const float *queries [[buffer(12)]],
    constant uint &blocks [[buffer(13)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    device const float *sources[] = {b0, b1, b2, b3};
    device const float *partial[] = {p0, p1, p2, p3};
    device float *output[] = {o0, o1, o2, o3};
    threadgroup float sums[18];
    threadgroup float dots[18];
    threadgroup float inverse;
    threadgroup float scores[5];
    ink_mix_row(sources[group.y], partial[group.y], queries, output[group.y],
        uint2(blocks, 1), group.x, tid, simd, lane, sums, dots, inverse, scores);
}

kernel void ink_verify_mix_attention(
    device const float *b0 [[buffer(0)]], device const float *b1 [[buffer(1)]],
    device const float *b2 [[buffer(2)]], device const float *b3 [[buffer(3)]],
    device const float *p0 [[buffer(4)]], device const float *p1 [[buffer(5)]],
    device const float *p2 [[buffer(6)]], device const float *p3 [[buffer(7)]],
    device float *o0 [[buffer(8)]], device float *o1 [[buffer(9)]],
    device float *o2 [[buffer(10)]], device float *o3 [[buffer(11)]],
    device const float *queries [[buffer(12)]],
    device const float *v0 [[buffer(13)]], device const float *v1 [[buffer(14)]],
    device const float *v2 [[buffer(15)]], device const float *v3 [[buffer(16)]],
    device float *m0 [[buffer(17)]], device float *m1 [[buffer(18)]],
    device float *m2 [[buffer(19)]], device float *m3 [[buffer(20)]],
    device const float *gate [[buffer(21)]],
    constant uint2 &shape [[buffer(22)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    device const float *sources[] = {b0, b1, b2, b3};
    device const float *partial[] = {p0, p1, p2, p3};
    device float *output[] = {o0, o1, o2, o3};
    device const float *previous[] = {v0, v1, v2, v3};
    device float *messages[] = {m0, m1, m2, m3};
    threadgroup float sums[18];
    threadgroup float dots[18];
    threadgroup float inverse;
    threadgroup float scores[5];
    const uint row = group.y;
    const bool message = group.x >= shape.y;
    ink_mix_row(message ? previous[row] : sources[row], partial[row], message ? gate : queries,
        message ? messages[row] : output[row], message ? uint2(2, 0) : uint2(shape.x, 1),
        message ? group.x - shape.y : group.x, tid, simd, lane, sums, dots, inverse, scores);
}
