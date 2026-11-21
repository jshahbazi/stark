// SPDX-License-Identifier: Apache-2.0 AND MIT
// The controller expansion adapts MLX steel/gemm/nax.h at
// fc344c986c4169d99f76b5d95ae6b4a58a359a63 (Copyright © 2025 Apple Inc., LICENSE.mlx-reference).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

constant uint ink_width = 2304;

// Materialize the FP32 rounding boundaries of the source's RMS, scaled
// query dot, softmax, and weighted residual sum in one dispatch.
kernel void ink_packed_mix(
    device const float *sources [[buffer(0)]], device const float *partial [[buffer(1)]],
    device const float *queries [[buffer(2)]], device float *result [[buffer(3)]],
    constant uint4 &args [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float sums[18];
    threadgroup float dots[18];
    threadgroup float inverse;
    threadgroup float scores[5];
    float4 units[5];
    const uint q = group.x;
    sources += group.y * args.z;
    partial += group.y * ink_width;
    result += group.y * args.w + q * ink_width;
    uint count = args.x + args.y;
    for (uint s = 0; s < count; ++s) {
        device const float *row = s == args.x ? partial : sources + s * ink_width;
        float4 batch = *reinterpret_cast<device const float4 *>(row + tid * 4);
        float norm = 0;
        for (uint i = 0; i < 4; ++i) norm += batch[i] * batch[i];
        norm = simd_sum(norm);
        if (lane == 0) sums[simd] = norm;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0) {
            float total = simd_sum(lane < 18 ? sums[lane] : 0.f);
            if (lane == 0) inverse = metal::precise::rsqrt(total / ink_width + 1e-6f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        units[s] = batch * inverse;
        if (count == 1) {
            *reinterpret_cast<device float4 *>(result + tid * 4) = units[s];
            return;
        }
        {
#pragma clang fp contract(off) reassociate(off)
            float4 query = *reinterpret_cast<device const float4 *>(queries + q * ink_width + tid * 4);
            query = query * 0.020833333333333332f;
            float dot = 0;
            for (uint i = 0; i < 4; ++i) dot += units[s][i] * query[i];
            dot = simd_sum(dot);
            if (lane == 0) dots[simd] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0) {
            float total = simd_sum(lane < 18 ? dots[lane] : 0.f);
            if (lane == 0) scores[s] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float peak = -INFINITY;
        for (uint s = 0; s < count; ++s) peak = max(peak, scores[s]);
        float total = 0;
        for (uint s = 0; s < count; ++s) {
            scores[s] = fast::exp(scores[s] - peak);
            total += scores[s];
        }
        float scale = 1.f / total;
        for (uint s = 0; s < count; ++s) scores[s] *= scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
#pragma clang fp contract(off) reassociate(off)
        float4 out = 0;
        for (uint s = 0; s < count; ++s) out += units[s] * scores[s];
        *reinterpret_cast<device float4 *>(result + tid * 4) = out;
    }
}

// The source RMS reduction reads four adjacent values per thread and reduces
// the 18 SIMD partial sums in one SIMD operation for a 2304-wide input.
static inline void ink_packed_norm_row(
    float4 value, device const bfloat *weight, device bfloat *out, bool weighted,
    threadgroup float *sums, threadgroup float &scale, uint tid, uint simd, uint lane
) {
    float sum = 0;
    for (uint i = 0; i < 4; ++i) sum += value[i] * value[i];
    sum = simd_sum(sum);
    if (lane == 0) sums[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        sum = simd_sum(lane < 18 ? sums[lane] : 0.f);
        if (lane == 0) scale = metal::precise::rsqrt(sum / ink_width + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
        uint j = tid * 4 + i;
        float unit = value[i] * scale;
        out[j] = bfloat(weighted ? float(weight[j]) * unit : unit);
    }
}

kernel void ink_packed_norm(
    device const float *x [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device bfloat *out [[buffer(2)]], constant uint2 &strides [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float sums[18];
    threadgroup float scale;
    float4 value = reinterpret_cast<device const float4 *>(x + row * strides.x)[tid];
    ink_packed_norm_row(value, weight, out + row * strides.y, true, sums, scale, tid, simd, lane);
}

kernel void ink_packed_controller_norm(
    device const float *input [[buffer(0)]], device const float *message [[buffer(1)]],
    device bfloat *out [[buffer(2)]], constant uint2 &strides [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float sums[18];
    threadgroup float scale;
    const device float *x = group.x == 0 ? input + group.y * strides.x : message + group.y * strides.y;
    const float4 value = reinterpret_cast<device const float4 *>(x)[tid];
    ink_packed_norm_row(value, nullptr, out + (2 * group.y + group.x) * ink_width,
                        false, sums, scale, tid, simd, lane);
}

// Each SIMD group expands 32 channels across as many as 16 prompt rows.
kernel void ink_packed_controller_expand(
    device const bfloat *compressed [[buffer(0)]],
    device const bfloat *gain [[buffer(1)]],
    device const bfloat *forget [[buffer(2)]],
    device const float *input [[buffer(3)]],
    device const float *message [[buffer(4)]],
    device float *mixed [[buffer(5)]],
    constant uint4 &args [[buffer(6)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rank = 64;
    constexpr uint columns = 32;
    constexpr uint fragment_elements = 8;
    const uint rows = args.x;
    const uint first = tile.x * columns;
    const uint first_row = tile.y * 16;

    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        16, columns, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroup> mma;
    auto a = mma.get_left_input_cooperative_tensor<float, float, float>();
    auto b = mma.get_right_input_cooperative_tensor<float, float, float>();
    using a_type = remove_addrspace_t<decltype(a)>;
    using b_type = remove_addrspace_t<decltype(b)>;
    auto g = mma.get_destination_cooperative_tensor<a_type, b_type, float>();
    auto f = mma.get_destination_cooperative_tensor<a_type, b_type, float>();
    for (ushort i = 0; i < g.get_capacity(); ++i) {
        g[i] = 0.0f;
        f[i] = 0.0f;
    }

    const uint base_row = ((lane >> 2) & 4) | ((lane >> 1) & 3);
    const uint base_column = (((lane >> 2) & 2) | (lane & 1)) * 4;
    for (uint k = 0; k < rank; k += 16) {
        for (ushort i = 0; i < fragment_elements; ++i) {
            const uint row = base_row + (i >> 2) * 8;
            const uint column = base_column + (i & 3);
            const uint index = (first + row) * rank + k + column;
            const uint prompt_row = first_row + row;
            a[i] = prompt_row < rows ? float(compressed[prompt_row * rank + k + column]) : 0.0f;
            b[i] = float(gain[index]);
            b[fragment_elements + i] = float(gain[index + 16 * rank]);
        }
        mma.run(a, b, g);
        for (ushort i = 0; i < fragment_elements; ++i) {
            const uint row = base_row + (i >> 2) * 8;
            const uint column = base_column + (i & 3);
            const uint index = (first + row) * rank + k + column;
            b[i] = float(forget[index]);
            b[fragment_elements + i] = float(forget[index + 16 * rank]);
        }
        mma.run(a, b, f);
    }

    // The compiled target gate uses separate FP32 products followed by an add.
    {
#pragma clang fp contract(off) reassociate(off)
        for (ushort i = 0; i < g.get_capacity(); ++i) {
            const auto coordinate = g.get_multidimensional_index(i);
            const uint prompt_row = first_row + coordinate[1];
            if (prompt_row < rows) {
                const uint channel = first + coordinate[0];
                const float keep = 1.0f - metal::precise::tanh(f[i]);
                const float admit = metal::precise::tanh(g[i]);
                mixed[prompt_row * args.w + channel] =
                    input[prompt_row * args.y + channel] * keep + message[prompt_row * args.z + channel] * admit;
            }
        }
    }
}

kernel void ink_packed_embedding(
    device const bfloat *embedding [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device float *residual [[buffer(2)]], device const uint *tokens [[buffer(3)]],
    uint token_row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sums[18];
    threadgroup float scale;
    const uint column = tid * 4;
    const uint row = tokens[token_row] * ink_width;
    residual += token_row * ink_width;
    const float4 values = float4(*reinterpret_cast<device const bfloat4 *>(embedding + row + column));
    float total = 0;
    for (uint i = 0; i < 4; ++i) total += values[i] * values[i];
    total = simd_sum(total);
    if (lane == 0) sums[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        total = simd_sum(lane < 18 ? sums[lane] : 0.0f);
        if (lane == 0) scale = metal::precise::rsqrt(total / ink_width + 1.0e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 4; ++i) {
        residual[column + i] = float(bfloat(float(weight[column + i]) * (values[i] * scale)));
    }
}

kernel void ink_packed_after_attention(
    device float *partial [[buffer(0)]], device const bfloat *attn [[buffer(1)]],
    device float *blocks [[buffer(2)]], constant uint2 &args [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]
) {
    const uint column = position.x;
    const uint row = position.y;
    const uint i = row * ink_width + column;
    float previous = partial[i];
    if (args.x != 0) {
        blocks[(row * 4 + args.y) * ink_width + column] = previous;
        previous = 0.0f;
    }
    partial[i] = previous + float(attn[i]);
}

kernel void ink_packed_after_experts(
    device float *partial [[buffer(0)]], device const bfloat *ffn [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]
) {
    const uint i = position.y * ink_width + position.x;
    partial[i] = float(bfloat(partial[i] + float(ffn[i])));
}

// At the first core visit remember the prelude. At each later core pass,
// freeze the prior pass's last block and partial before restoring the prelude.
kernel void ink_packed_prepare_visit(
    device float *partial [[buffer(0)]], device float *prelude [[buffer(1)]],
    device const float *blocks [[buffer(2)]], device float *previous [[buffer(3)]],
    constant uint &visit [[buffer(4)]], uint2 position [[thread_position_in_grid]]
) {
    const uint column = position.x;
    const uint row = position.y;
    const uint i = row * ink_width + column;
    if (visit == 6) {
        prelude[i] = partial[i];
    } else {
        const uint history = row * 2 * ink_width + column;
        previous[history] = blocks[(row * 4 + 2) * ink_width + column];
        previous[history + ink_width] = partial[i];
        partial[i] = prelude[i];
    }
}
