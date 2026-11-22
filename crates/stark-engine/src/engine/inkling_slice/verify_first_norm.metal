// SPDX-License-Identifier: Apache-2.0

kernel void ink_verify_first_qkv_norm(
    device const float4 *x0 [[buffer(0)]], device const float4 *x1 [[buffer(1)]],
    device const float4 *x2 [[buffer(2)]], device const float4 *x3 [[buffer(3)]],
    device const bfloat *wq [[buffer(4)]], device const bfloat *wk [[buffer(5)]], device const bfloat *wv [[buffer(6)]],
    device bfloat *q0 [[buffer(7)]], device bfloat *q1 [[buffer(8)]],
    device bfloat *q2 [[buffer(9)]], device bfloat *q3 [[buffer(10)]],
    device bfloat *k0 [[buffer(11)]], device bfloat *k1 [[buffer(12)]],
    device bfloat *k2 [[buffer(13)]], device bfloat *k3 [[buffer(14)]],
    device bfloat *v0 [[buffer(15)]], device bfloat *v1 [[buffer(16)]],
    device bfloat *v2 [[buffer(17)]], device bfloat *v3 [[buffer(18)]],
    constant uint &shared_input [[buffer(19)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const device float4 *inputs[] = {x0, x1, x2, x3};
    const device bfloat *weights[] = {wq, wk, wv};
    device bfloat *outputs[] = {q0,q1,q2,q3,k0,k1,k2,k3,v0,v1,v2,v3};
    threadgroup float sums[18];
    threadgroup float scale;
    const float4 x = inputs[group.y][group.x * (ink_width / 4) + tid];
    ink_norm_row(x, weights[group.x], outputs[group.x * 4 + group.y], true, sums, scale, tid, simd, lane);
    if (shared_input != 0) {
        for (uint i = 0; i < 4; ++i) {
            const uint j = tid * 4 + i;
            const float unit = x[i] * scale;
            outputs[4 + group.y][j] = bfloat(float(wk[j]) * unit);
            outputs[8 + group.y][j] = bfloat(float(wv[j]) * unit);
        }
    }
}

kernel void ink_verify_first_expert_norm(
    device const float4 *x0 [[buffer(0)]], device const float4 *x1 [[buffer(1)]],
    device const float4 *x2 [[buffer(2)]], device const float4 *x3 [[buffer(3)]],
    device const bfloat *w [[buffer(4)]],
    device bfloat *y0 [[buffer(5)]], device bfloat *y1 [[buffer(6)]],
    device bfloat *y2 [[buffer(7)]], device bfloat *y3 [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const device float4 *inputs[] = {x0, x1, x2, x3};
    device bfloat *outputs[] = {y0, y1, y2, y3};
    threadgroup float sums[18];
    threadgroup float scale;
    ink_norm_row(inputs[row][tid], w, outputs[row], true, sums, scale, tid, simd, lane);
}
