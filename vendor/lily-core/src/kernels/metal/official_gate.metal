// SPDX-License-Identifier: Apache-2.0
// Direct BF16 gates for the official MLX-Serve Qwen checkpoint. Weight
// values are read unchanged; only activations and F32 accumulation occur.
#include <metal_stdlib>
using namespace metal;

template<typename Out>
static inline void official_gate_body(device const bfloat* x,
    device const bfloat* w, device Out* y, uint K, uint N,
    uint2 row, uint lane) {
    float sum = 0.0f;
    for (uint k = lane; k < K; k += 32)
        sum += float(x[ulong(row.y) * K + k]) * float(w[ulong(row.x) * K + k]);
    sum = simd_sum(sum);
    if (lane == 0) y[ulong(row.y) * N + row.x] = Out(sum);
}

#define OFFICIAL_GATE(NAME, OUT) \
kernel void NAME(device const bfloat* x [[buffer(0)]], \
    device const bfloat* w [[buffer(1)]], device OUT* y [[buffer(2)]], \
    constant uint& K [[buffer(3)]], constant uint& N [[buffer(4)]], \
    uint2 row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
    official_gate_body(x, w, y, K, N, row, lane); \
}
OFFICIAL_GATE(official_gate_bf16, bfloat)
OFFICIAL_GATE(official_gate_f32, float)

// Grouped-norm input is kept in F32 as in the fused affine HC read. Each
// simdgroup owns one stream and the final thread sums its normalized dot.
kernel void official_hc_inject(device const bfloat* hyper [[buffer(0)]],
    device const bfloat* norm [[buffer(1)]], device const bfloat* w [[buffer(2)]],
    device bfloat* out [[buffer(3)]], constant uint& H [[buffer(4)]],
    constant uint& G [[buffer(5)]], constant uint& N [[buffer(6)]],
    constant float& eps [[buffer(7)]], constant float& bias [[buffer(8)]],
    uint2 row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint g [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[8];
    const uint K = G * H;
    float dot = 0.0f, squares = 0.0f;
    for (uint k = lane; k < H; k += 32) {
        const uint i = g * H + k;
        const float x = float(hyper[ulong(row.y) * K + i]);
        dot += x * (bias + float(norm[i])) * float(w[ulong(row.x) * K + i]);
        squares += x * x;
    }
    dot = simd_sum(dot);
    squares = simd_sum(squares);
    if (lane == 0) partial[g] = dot * rsqrt(squares / float(H) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint stream = 0; stream < G; ++stream) sum += partial[stream];
        out[ulong(row.y) * N + row.x] = bfloat(sum);
    }
}
