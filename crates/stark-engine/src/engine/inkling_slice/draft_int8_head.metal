// SPDX-License-Identifier: Apache-2.0
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

// Scale each vocabulary row once, from the same padded BF16 matrix as
// the target. Quantization only feeds the multistep proposal head.
kernel void ink_draft_pack_int8_head(
    device const bfloat *source [[buffer(0)]], device int8_t *packed [[buffer(1)]],
    device float *scales [[buffer(2)]], constant uint2 &shape [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    const uint width = shape.x;
    source += size_t(row) * width;
    packed += size_t(row) * width;
    threadgroup float maxima[4];
    float maximum = 0.0f;
    for (uint k = lid; k < width; k += 128) {
        maximum = max(maximum, abs(float(source[k])));
    }
    maximum = simd_max(maximum);
    if (lane == 0) {
        maxima[simd] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = max(max(maxima[0], maxima[1]), max(maxima[2], maxima[3]));
    const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    if (lid == 0) {
        scales[row] = scale;
    }
    for (uint k = lid; k < width; k += 128) {
        packed[k] = int8_t(rint(float(source[k]) / scale));
    }
}

kernel void ink_draft_int8_head(
    device bfloat *hidden [[buffer(0)]], device int8_t *packed [[buffer(1)]],
    device float *scores [[buffer(2)]], constant uint2 &shape [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]]
) {
    constexpr int rows = 8, channels = 32;
    const int width = int(shape.x), columns = int(shape.y);
    const uint channel = group * channels;
    using dims = dextents<int, 2>;
    using strides = array<int, 2>;
    tensor<device bfloat, dims, tensor_inline> a(hidden, dims(width, rows), strides{1, width});
    tensor<device int8_t, dims, tensor_inline> b(
        packed + size_t(channel) * width, dims(width, channels), strides{1, width});
    tensor<device float, dims, tensor_inline> c(scores + channel, dims(channels, rows), strides{1, columns});
    constexpr mpp::tensor_ops::matmul2d_descriptor tile(rows, channels, int(dynamic_extent), false, true, true);
    mpp::tensor_ops::matmul2d<tile, execution_simdgroup> op;
    op.run(a, b, c);
}

kernel void ink_draft_scale_int8_head(
    device float *scores [[buffer(0)]], device const float *scales [[buffer(1)]],
    constant uint2 &shape [[buffer(2)]], uint2 tid [[thread_position_in_grid]]
) {
    const uint columns = shape.x;
    if (tid.x < columns && tid.y < shape.y) {
        scores[tid.y * columns + tid.x] *= scales[tid.x];
    }
}
