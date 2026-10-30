// Qwen3.8 PLE math adapted into Stark's standard Uzu kernel DSL.
// Algorithm source: lily-qwen3.8-flash-next e22d58fa7cc488493fb1b5d810a198bd29730aeb.
// Copyright 2026 Perplexity AI, Inc.; Copyright 2026 Fabio Greter.
// Apache-2.0: see LICENSE.qwen38-reference and NOTICE.qwen38-reference.
#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/thread_context.h"
using namespace metal;

PUBLIC KERNEL(Qwen38PleGather)(
    const device uchar* codes,
    const device bfloat* scales,
    const device bfloat* biases,
    device bfloat* output,
    const constant uint& head_dim,
    const constant uint& group_size,
    const constant uint& heads,
    const constant uint& rows,
    uint tile GROUPS(head_dim.div_ceil(256)),
    uint head GROUPS(heads * rows),
    uint tid THREADS(256)
) {
    const uint column = tile * 256 + tid;
    if (column >= head_dim) return;
    const ulong code_at = ulong(head) * (head_dim / 2) + column / 2;
    const uint code = (codes[code_at] >> (4 * (column % 2))) & 15;
    const ulong parameter = ulong(head) * (head_dim / group_size) + column / group_size;
    output[ulong(head) * head_dim + column] = bfloat(float(code) * float(scales[parameter]) + float(biases[parameter]));
}

PUBLIC KERNEL(Qwen38PleGate)(
    const device bfloat* key,
    const device bfloat* query,
    const device bfloat* value,
    device bfloat* output,
    const constant uint& width,
    const constant uint& streams,
    const constant uint& rows,
    threadgroup float partial[8],
    threadgroup float gate[1],
    uint segment GROUPS(streams * rows),
    uint tid THREADS(256),
    const ThreadContext thread_context
) {
    const uint lane = thread_context.simd_lane_id;
    const uint sg = thread_context.simdgroup_index;
    const ulong base = ulong(segment) * width;
    float total = 0.0f;
    for (uint i = tid; i < width; i += 256) total += float(key[base + i]) * float(query[base + i]);
    total = simd_sum(total);
    if (lane == 0) partial[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < 8; ++i) sum += partial[i];
        sum *= rsqrt(float(width));
        const float magnitude = sqrt(max(abs(sum), 1e-6f));
        const float signed_root = sum < 0 ? -magnitude : magnitude;
        gate[0] = 1.0f / (1.0f + exp(-signed_root));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong value_base = ulong(segment / streams) * width;
    for (uint i = tid; i < width; i += 256) output[base + i] = bfloat(gate[0] * float(value[value_base + i]));
}

PUBLIC KERNEL(Qwen38PleConv)(
    device bfloat* window,
    const device bfloat* input,
    const device float* weight,
    const device bfloat* base,
    device bfloat* hyper,
    const constant uint& width,
    const constant uint& kernel_size,
    const constant uint& dilation,
    const constant uint& rows,
    uint tile GROUPS(width.div_ceil(256)),
    uint tid THREADS(256)
) {
    const uint column = tile * 256 + tid;
    if (column >= width) return;
    const uint history_size = (kernel_size - 1) * dilation;
    float history[16];
    for (uint i = 0; i < history_size; ++i) history[i] = float(window[ulong(column) * history_size + i]);
    for (uint row = 0; row < rows; ++row) {
        const ulong at = ulong(row) * width + column;
        const float current = float(input[at]);
        float sum = current * weight[ulong(column) * kernel_size + kernel_size - 1];
        for (uint tap = 0; tap + 1 < kernel_size; ++tap)
            sum += history[history_size - (kernel_size - 1 - tap) * dilation] * weight[ulong(column) * kernel_size + tap];
        const float conv = sum / (1.0f + exp(-sum));
        hyper[at] = bfloat(float(hyper[at]) + float(base[at]) + conv);
        for (uint i = 0; i + 1 < history_size; ++i) history[i] = history[i + 1];
        history[history_size - 1] = current;
    }
    for (uint i = 0; i < history_size; ++i) window[ulong(column) * history_size + i] = bfloat(history[i]);
}
