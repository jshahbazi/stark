// SPDX-License-Identifier: Apache-2.0
// Affine 8-bit dense weights for the Inkling 2 Mini affine8 artifact.
//
// The effective weight of every quantized tensor is the BF16 rounding of
// fma(code, scale, bias) in FP32, with an unsigned 8-bit code and BF16 scale
// and bias per row and 64 consecutive inputs (MLX affine, group size 64).
// Prefill reads a BF16 copy holding exactly these values. The decode kernels
// below dequantize to the same BF16 values and then run the unchanged K-lane
// order and FP32 block sums of ink_matvec_source / ink_matvec_prefill_shared,
// so each output equals the BF16 kernel's output on the BF16 copy.
//
// One Metal buffer per tensor: codes, then one uint per row and group holding
// the BF16 scale (low half) and bias (high half). Codes are interleaved per
// row so that each K lane reads its eight vec4 columns of one block as 32
// contiguous bytes: for lanes L, stride S = 8 L vec4 columns and end the
// largest multiple of S not above columns / 4,
//   column c < end:  b = c / S, r = c % S  ->  b S + (r % L) 8 + r / L
//   column c >= end: t = c - end, T = (columns / 4 - end) / L  ->  end + (t % L) T + t / L
// in vec4 units (4 bytes each).

#include <metal_stdlib>
using namespace metal;

constant uint ink_a8_width = 2304;

static inline float4 ink_a8_value(uint word, uint param) {
    const float scale = as_type<float>(param << 16);
    const float bias = as_type<float>(param & 0xffff0000u);
    return float4(bfloat4(fma(float4(as_type<uchar4>(word)), float4(scale), float4(bias))));
}

static inline uint ink_a8_lanes(uint rows) {
    return rows <= 64 ? 32 : 16;
}

// Accumulate Batch dot products of one weight row with the source K-lane order.
template <uint Batch>
static inline void ink_a8_row(
    device const uchar *tensor, thread const device bfloat4 *const *inputs, thread float *sums,
    uint cols, uint stored, uint rows, uint row, uint k_lane, uint lanes) {
    const uint cols4 = cols / 4;
    const uint stride = lanes * 8;
    const uint end = cols4 - cols4 % stride;
    const uint tail = (cols4 - end) / lanes;
    const uint live = min(row, rows - 1);
    device const uchar *codes = tensor + size_t(live) * cols;
    device const uint *params = reinterpret_cast<device const uint *>(tensor + size_t(stored) * cols) + live * (cols / 64);
#pragma clang loop unroll(full)
    for (uint vector = 0; vector < Batch; ++vector) sums[vector] = 0.0f;
    for (uint base = 0; base < end; base += stride) {
        device const uint4 *packed = reinterpret_cast<device const uint4 *>(codes + base * 4 + k_lane * 32);
        const uint4 low = packed[0], high = packed[1];
        const uint words[8] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
        float4 loaded[8];
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) loaded[i] = ink_a8_value(words[i], params[(base + i * lanes + k_lane) / 16]);
#pragma clang loop unroll(full)
        for (uint vector = 0; vector < Batch; ++vector) {
            float block = 0.0f;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8; ++i) {
                const uint col = base + i * lanes + k_lane;
                block += dot(loaded[i], float4(inputs[vector][col]));
            }
            sums[vector] += block;
        }
    }
    device const uint *rest = reinterpret_cast<device const uint *>(codes + end * 4) + k_lane * tail;
    for (uint j = 0; j < tail; ++j) {
        const uint col = end + k_lane + j * lanes;
        const float4 loaded = ink_a8_value(rest[j], params[col / 16]);
#pragma clang loop unroll(full)
        for (uint vector = 0; vector < Batch; ++vector) sums[vector] += dot(loaded, float4(inputs[vector][col]));
    }
#pragma clang loop unroll(full)
    for (uint vector = 0; vector < Batch; ++vector) {
        for (uint delta = lanes / 2; delta > 0; delta >>= 1) {
            sums[vector] += simd_shuffle_down(sums[vector], ushort(delta));
        }
    }
}

// shape: x = input columns, y = output rows, z = live vectors. A threadgroup of
// T threads holds T / lanes output rows.
template <uint Batch>
static inline void ink_a8_matvec_rows(
    device const uchar *tensor, thread const device bfloat4 *const *inputs, thread device bfloat *const *outputs,
    uint3 shape, uint group, uint threads, uint simd, uint lane) {
    const uint lanes = ink_a8_lanes(shape.y);
    const uint row = group * (threads / lanes) + simd * (32 / lanes) + lane / lanes;
    const uint k_lane = lane % lanes;
    float sums[Batch];
    ink_a8_row<Batch>(tensor, inputs, sums, shape.x, shape.y, shape.y, row, k_lane, lanes);
#pragma clang loop unroll(full)
    for (uint vector = 0; vector < Batch; ++vector) {
        if (k_lane == 0 && row < shape.y && vector < shape.z) outputs[vector][row] = bfloat(sums[vector]);
    }
}

kernel void ink_a8_matvec1(
    device const uchar *tensor [[buffer(0)]], device const bfloat4 *x0 [[buffer(1)]],
    device bfloat *y0 [[buffer(2)]], constant uint3 &shape [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]], uint threads [[threads_per_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    device const bfloat4 *inputs[] = {x0};
    device bfloat *outputs[] = {y0};
    ink_a8_matvec_rows<1>(tensor, inputs, outputs, shape, group, threads, simd, lane);
}

kernel void ink_a8_matvec2(
    device const uchar *tensor [[buffer(0)]], device const bfloat4 *x0 [[buffer(1)]],
    device const bfloat4 *x1 [[buffer(2)]], device bfloat *y0 [[buffer(3)]], device bfloat *y1 [[buffer(4)]],
    constant uint3 &shape [[buffer(5)]], uint group [[threadgroup_position_in_grid]],
    uint threads [[threads_per_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    device const bfloat4 *inputs[] = {x0, x1};
    device bfloat *outputs[] = {y0, y1};
    ink_a8_matvec_rows<2>(tensor, inputs, outputs, shape, group, threads, simd, lane);
}

kernel void ink_a8_matvec3(
    device const uchar *tensor [[buffer(0)]], device const bfloat4 *x0 [[buffer(1)]],
    device const bfloat4 *x1 [[buffer(2)]], device const bfloat4 *x2 [[buffer(3)]],
    device bfloat *y0 [[buffer(4)]], device bfloat *y1 [[buffer(5)]], device bfloat *y2 [[buffer(6)]],
    constant uint3 &shape [[buffer(7)]], uint group [[threadgroup_position_in_grid]],
    uint threads [[threads_per_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    device const bfloat4 *inputs[] = {x0, x1, x2};
    device bfloat *outputs[] = {y0, y1, y2};
    ink_a8_matvec_rows<3>(tensor, inputs, outputs, shape, group, threads, simd, lane);
}

kernel void ink_a8_matvec4(
    device const uchar *tensor [[buffer(0)]], device const bfloat4 *x0 [[buffer(1)]],
    device const bfloat4 *x1 [[buffer(2)]], device const bfloat4 *x2 [[buffer(3)]],
    device const bfloat4 *x3 [[buffer(4)]], device bfloat *y0 [[buffer(5)]], device bfloat *y1 [[buffer(6)]],
    device bfloat *y2 [[buffer(7)]], device bfloat *y3 [[buffer(8)]], constant uint3 &shape [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]], uint threads [[threads_per_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    device const bfloat4 *inputs[] = {x0, x1, x2, x3};
    device bfloat *outputs[] = {y0, y1, y2, y3};
    ink_a8_matvec_rows<4>(tensor, inputs, outputs, shape, group, threads, simd, lane);
}

// The single-row attention projections of one visit in one launch, as
// ink_matvec_source_visit: q, k, v, relative, gate, sink and, for global
// layers, the key and value dynamic convolution projections. A workgroup of
// 128 threads holds 128 / lanes rows.
constant uint ink_a8_projection_rows[8] = {3072, 1024, 1024, 384, 3072, 24, 32, 32};
constant uint ink_a8_projection_input[8] = {0, 1, 2, 0, 0, 0, 1, 2};

kernel void ink_a8_matvec_visit(
    device const bfloat4 *x0 [[buffer(0)]], device const bfloat4 *x1 [[buffer(1)]],
    device const bfloat4 *x2 [[buffer(2)]],
    device const uchar *w0 [[buffer(3)]], device const uchar *w1 [[buffer(4)]],
    device const uchar *w2 [[buffer(5)]], device const uchar *w3 [[buffer(6)]],
    device const uchar *w4 [[buffer(7)]], device const uchar *w5 [[buffer(8)]],
    device const uchar *w6 [[buffer(9)]], device const uchar *w7 [[buffer(10)]],
    device bfloat *y0 [[buffer(11)]], device bfloat *y1 [[buffer(12)]], device bfloat *y2 [[buffer(13)]],
    device bfloat *y3 [[buffer(14)]], device bfloat *y4 [[buffer(15)]], device bfloat *y5 [[buffer(16)]],
    device bfloat *y6 [[buffer(17)]], device bfloat *y7 [[buffer(18)]],
    constant uint &segments [[buffer(19)]],
    uint group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    device const bfloat4 *inputs[] = {x0, x1, x2};
    device const uchar *weights[] = {w0, w1, w2, w3, w4, w5, w6, w7};
    device bfloat *outputs[] = {y0, y1, y2, y3, y4, y5, y6, y7};
    uint segment = 0;
    uint first = 0;
    for (; segment < segments; ++segment) {
        const uint rows = ink_a8_projection_rows[segment];
        const uint groups = rows / (rows <= 64 ? 4 : 8);
        if (group < first + groups) break;
        first += groups;
    }
    if (segment >= segments) return;
    const uint rows = ink_a8_projection_rows[segment];
    const uint lanes = ink_a8_lanes(rows);
    const uint row = (group - first) * (128 / lanes) + simd * (32 / lanes) + lane / lanes;
    const uint k_lane = lane % lanes;
    device const bfloat4 *x[] = {inputs[ink_a8_projection_input[segment]]};
    float sums[1];
    ink_a8_row<1>(weights[segment], x, sums, ink_a8_width, rows, rows, row, k_lane, lanes);
    if (k_lane == 0 && row < rows) outputs[segment][row] = bfloat(sums[0]);
}

// Vocabulary readout for up to four live rows, with the logit cap of
// ink_logits_cap applied in place. shape: x = width, y = live vocabulary,
// z = live rows, w = padded output row stride.
template <uint Batch>
static inline void ink_a8_unembed_rows(
    device const uchar *tensor, device const bfloat4 *hidden, device float *logits, uint4 shape, uint group,
    uint threads, uint simd, uint lane) {
    const uint lanes = 16;
    const uint row = group * (threads / lanes) + simd * 2 + lane / lanes;
    const uint k_lane = lane % lanes;
    device const bfloat4 *inputs[Batch];
#pragma clang loop unroll(full)
    for (uint vector = 0; vector < Batch; ++vector) inputs[vector] = hidden + vector * (shape.x / 4);
    float sums[Batch];
    ink_a8_row<Batch>(tensor, inputs, sums, shape.x, shape.w, shape.y, row, k_lane, lanes);
#pragma clang loop unroll(full)
    for (uint vector = 0; vector < Batch; ++vector) {
        if (k_lane == 0 && row < shape.y && vector < shape.z) {
            logits[vector * shape.w + row] = 30.0f * metal::precise::tanh(sums[vector] / 30.0f);
        }
    }
}

#define INK_A8_UNEMBED(N)                                                                                     \
    kernel void ink_a8_unembed##N(                                                                            \
        device const bfloat4 *hidden [[buffer(0)]], device const uchar *tensor [[buffer(1)]],                 \
        device float *logits [[buffer(2)]], constant uint4 &shape [[buffer(3)]],                              \
        uint group [[threadgroup_position_in_grid]], uint threads [[threads_per_threadgroup]],                \
        uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {              \
        ink_a8_unembed_rows<N>(tensor, hidden, logits, shape, group, threads, simd, lane);                    \
    }
INK_A8_UNEMBED(1)
INK_A8_UNEMBED(2)
INK_A8_UNEMBED(3)
INK_A8_UNEMBED(4)

// Interleaved vec4 position of input column c (vec4 units).
static inline uint ink_a8_position(uint c, uint cols, uint rows) {
    const uint lanes = ink_a8_lanes(rows);
    const uint cols4 = cols / 4;
    const uint stride = lanes * 8;
    const uint end = cols4 - cols4 % stride;
    if (c < end) {
        const uint r = c % stride;
        return c - r + (r % lanes) * 8 + r / lanes;
    }
    const uint t = c - end;
    const uint tail = (cols4 - end) / lanes;
    return end + (t % lanes) * tail + t / lanes;
}

// BF16 copy of the effective weights. One thread per vec4. shape: columns, rows.
kernel void ink_a8_dequantize(
    device const uchar *tensor [[buffer(0)]], device bfloat4 *output [[buffer(1)]],
    constant uint2 &shape [[buffer(2)]], uint2 index [[thread_position_in_grid]]) {
    const uint cols4 = shape.x / 4;
    if (index.x >= cols4 || index.y >= shape.y) return;
    device const uint *codes = reinterpret_cast<device const uint *>(tensor + size_t(index.y) * shape.x);
    device const uint *params =
        reinterpret_cast<device const uint *>(tensor + size_t(shape.y) * shape.x) + index.y * (shape.x / 64);
    const uint word = codes[ink_a8_position(index.x, shape.x, shape.y)];
    output[size_t(index.y) * cols4 + index.x] = bfloat4(ink_a8_value(word, params[index.x / 16]));
}

// ink_embedding with the token row read from affine codes.
kernel void ink_a8_embedding(
    device const uchar *tensor [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device float *residual [[buffer(2)]],
#ifdef INK_A8_DRAFT_EMBEDDING
    device const uint *proposals [[buffer(3)]], constant uint3 &row_vocabulary_rows [[buffer(4)]],
#else
    constant uint2 &token_rows [[buffer(3)]],
#endif
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sums[18];
    threadgroup float scale;
    const uint column = tid * 4;
#ifdef INK_A8_DRAFT_EMBEDDING
    const uint proposed = proposals[row_vocabulary_rows.x];
    const uint token = proposed < row_vocabulary_rows.y ? proposed : 0u;
    const uint rows = row_vocabulary_rows.z;
#else
    const uint token = token_rows.x;
    const uint rows = token_rows.y;
#endif
    device const uint *codes = reinterpret_cast<device const uint *>(tensor + size_t(token) * ink_a8_width);
    device const uint *params =
        reinterpret_cast<device const uint *>(tensor + size_t(rows) * ink_a8_width) + token * (ink_a8_width / 64);
    const float4 values = ink_a8_value(codes[ink_a8_position(tid, ink_a8_width, rows)], params[tid / 16]);
    float total = 0;
    for (uint i = 0; i < 4; ++i) total += values[i] * values[i];
    total = simd_sum(total);
    if (lane == 0) sums[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        total = simd_sum(lane < 18 ? sums[lane] : 0.0f);
        if (lane == 0) scale = metal::precise::rsqrt(total / ink_a8_width + 1.0e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
        residual[column + i] = float(bfloat(float(weight[column + i]) * (values[i] * scale)));
    }
}
