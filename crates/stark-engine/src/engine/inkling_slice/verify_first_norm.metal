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

// The two kernels below each replace an `ink_verify_mix_rows` launch and the
// first-pass norm launch that consumed its output. Every workgroup runs the
// unchanged mix reduction, stores the F32 mix as before, reads back the four
// values this thread stored and runs the unchanged norm reduction on them.
// F32 storage is exact, so the norm sees the values the separate launch read.
static inline float4 ink_verify_mixed_value(
    device const float *sources, device const float *partial, device const float *queries,
    device float *mixed, uint blocks, uint q, uint tid, uint simd, uint lane,
    threadgroup float *sums, threadgroup float *dots, threadgroup float &inverse,
    threadgroup float *scores
) {
    ink_mix_row(sources, partial, queries, mixed, uint2(blocks, 1), q, tid, simd, lane,
        sums, dots, inverse, scores);
    const uint offset = blocks + 1 == 1 ? 0 : q * ink_width;
    return *reinterpret_cast<device const float4 *>(mixed + offset + tid * 4);
}

kernel void ink_verify_mix_qkv_norm(
    device const float *b0 [[buffer(0)]], device const float *b1 [[buffer(1)]],
    device const float *b2 [[buffer(2)]], device const float *b3 [[buffer(3)]],
    device const float *p0 [[buffer(4)]], device const float *p1 [[buffer(5)]],
    device const float *p2 [[buffer(6)]], device const float *p3 [[buffer(7)]],
    device float *o0 [[buffer(8)]], device float *o1 [[buffer(9)]],
    device float *o2 [[buffer(10)]], device float *o3 [[buffer(11)]],
    device const float *queries [[buffer(12)]],
    device const bfloat *wq [[buffer(13)]], device const bfloat *wk [[buffer(14)]], device const bfloat *wv [[buffer(15)]],
    device bfloat *q0 [[buffer(16)]], device bfloat *q1 [[buffer(17)]],
    device bfloat *q2 [[buffer(18)]], device bfloat *q3 [[buffer(19)]],
    device bfloat *k0 [[buffer(20)]], device bfloat *k1 [[buffer(21)]],
    device bfloat *k2 [[buffer(22)]], device bfloat *k3 [[buffer(23)]],
    device bfloat *v0 [[buffer(24)]], device bfloat *v1 [[buffer(25)]],
    device bfloat *v2 [[buffer(26)]], device bfloat *v3 [[buffer(27)]],
    constant uint2 &args [[buffer(28)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    device const float *sources[] = {b0, b1, b2, b3};
    device const float *partial[] = {p0, p1, p2, p3};
    device float *mixed[] = {o0, o1, o2, o3};
    const device bfloat *weights[] = {wq, wk, wv};
    device bfloat *outputs[] = {q0,q1,q2,q3,k0,k1,k2,k3,v0,v1,v2,v3};
    threadgroup float sums[18];
    threadgroup float dots[18];
    threadgroup float inverse;
    threadgroup float scores[5];
    threadgroup float norm_sums[18];
    threadgroup float scale;
    const float4 x = ink_verify_mixed_value(sources[group.y], partial[group.y], queries, mixed[group.y],
        args.x, group.x, tid, simd, lane, sums, dots, inverse, scores);
    ink_norm_row(x, weights[group.x], outputs[group.x * 4 + group.y], true, norm_sums, scale, tid, simd, lane);
    if (args.y != 0) {
        for (uint i = 0; i < 4; ++i) {
            const uint j = tid * 4 + i;
            const float unit = x[i] * scale;
            outputs[4 + group.y][j] = bfloat(float(wk[j]) * unit);
            outputs[8 + group.y][j] = bfloat(float(wv[j]) * unit);
        }
    }
}

kernel void ink_verify_mix_expert_norm(
    device const float *b0 [[buffer(0)]], device const float *b1 [[buffer(1)]],
    device const float *b2 [[buffer(2)]], device const float *b3 [[buffer(3)]],
    device const float *p0 [[buffer(4)]], device const float *p1 [[buffer(5)]],
    device const float *p2 [[buffer(6)]], device const float *p3 [[buffer(7)]],
    device float *o0 [[buffer(8)]], device float *o1 [[buffer(9)]],
    device float *o2 [[buffer(10)]], device float *o3 [[buffer(11)]],
    device const float *queries [[buffer(12)]], device const bfloat *w [[buffer(13)]],
    device bfloat *y0 [[buffer(14)]], device bfloat *y1 [[buffer(15)]],
    device bfloat *y2 [[buffer(16)]], device bfloat *y3 [[buffer(17)]],
    constant uint &blocks [[buffer(18)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    device const float *sources[] = {b0, b1, b2, b3};
    device const float *partial[] = {p0, p1, p2, p3};
    device float *mixed[] = {o0, o1, o2, o3};
    device bfloat *outputs[] = {y0, y1, y2, y3};
    threadgroup float sums[18];
    threadgroup float dots[18];
    threadgroup float inverse;
    threadgroup float scores[5];
    threadgroup float norm_sums[18];
    threadgroup float scale;
    const float4 x = ink_verify_mixed_value(sources[group.y], partial[group.y], queries, mixed[group.y],
        blocks, group.x, tid, simd, lane, sums, dots, inverse, scores);
    ink_norm_row(x, w, outputs[group.y], true, norm_sums, scale, tid, simd, lane);
}

// Replaces `ink_verify_finish_attention`, the expert `ink_verify_mix_rows`
// and, on first-pass visits, the expert norm. One workgroup owns one row and
// each thread four channels, so the mix reads only values its own thread
// produced. The causal attention convolution matches the per-channel kernel:
// a lag that falls on an earlier provisional row reads that row's BF16 input,
// which is the value the serial kernel had stored in the history slot, and an
// older lag reads the history as it was before the command. The history slots
// are left unchanged here; `ink_verify_finish_experts_history` stores them.
kernel void ink_verify_attention_mix(
    device const bfloat *weight [[buffer(0)]], device const bfloat *history [[buffer(1)]],
    device const bfloat *x0 [[buffer(2)]], device const bfloat *x1 [[buffer(3)]],
    device const bfloat *x2 [[buffer(4)]], device const bfloat *x3 [[buffer(5)]],
    device float *p0 [[buffer(6)]], device float *p1 [[buffer(7)]],
    device float *p2 [[buffer(8)]], device float *p3 [[buffer(9)]],
    device float *b0 [[buffer(10)]], device float *b1 [[buffer(11)]],
    device float *b2 [[buffer(12)]], device float *b3 [[buffer(13)]],
    device float *o0 [[buffer(14)]], device float *o1 [[buffer(15)]],
    device float *o2 [[buffer(16)]], device float *o3 [[buffer(17)]],
    device const float *queries [[buffer(18)]], device const bfloat *w [[buffer(19)]],
    device bfloat *y0 [[buffer(20)]], device bfloat *y1 [[buffer(21)]],
    device bfloat *y2 [[buffer(22)]], device bfloat *y3 [[buffer(23)]],
    constant uint4 &args [[buffer(24)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    device const bfloat *inputs[] = {x0, x1, x2, x3};
    device float *partials[] = {p0, p1, p2, p3};
    device float *sources[] = {b0, b1, b2, b3};
    device float *mixed[] = {o0, o1, o2, o3};
    device bfloat *outputs[] = {y0, y1, y2, y3};
    const uint row = group.y;
    const uint pos = args.x + row;
    const bool boundary = args.z != 0;
    device float *partial = partials[row];
    device float *block = sources[row] + min(args.y, 3u) * ink_width;
    for (uint k = 0; k < 4; ++k) {
#pragma clang fp contract(off) reassociate(off)
        const uint channel = tid * 4 + k;
        const float current = float(inputs[row][channel]);
        float result = 0.0f;
        for (uint tap = 0; tap < 4; ++tap) {
            const uint lag = 3 - tap;
            if (pos < lag) continue;
            const float value = lag == 0 ? current
                : row >= lag ? float(inputs[row - lag][channel])
                : float(history[((pos - lag) % 3) * ink_width + channel]);
            result += float(weight[channel * 4 + tap]) * value;
        }
        const bfloat attention = bfloat((result + current) * 0.447213595499958f);
        if (boundary) block[channel] = partial[channel];
        partial[channel] = (boundary ? 0.0f : partial[channel]) + float(attention);
    }
    threadgroup float sums[18];
    threadgroup float dots[18];
    threadgroup float inverse;
    threadgroup float scores[5];
    threadgroup float norm_sums[18];
    threadgroup float scale;
    const float4 x = ink_verify_mixed_value(sources[row], partial, queries, mixed[row],
        args.y + uint(boundary), 0, tid, simd, lane, sums, dots, inverse, scores);
    if (args.w != 0) {
        ink_norm_row(x, w, outputs[row], true, norm_sums, scale, tid, simd, lane);
    }
}
