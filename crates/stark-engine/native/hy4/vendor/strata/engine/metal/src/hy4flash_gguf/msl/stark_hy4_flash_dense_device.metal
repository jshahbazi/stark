// SPDX-License-Identifier: Apache-2.0
// Half-operand version of Stark's inkling_slice/dense_fast.metal device view
// kernel. Packed weights decode with stark_hy4_flash_dense_packed.metal.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// Pack both complete operands once, using the same casts as the packed
// fragment kernel. These two projections have contiguous input rows.
kernel void stark_hy4_dense_pack_device(
    device const uchar* weights [[buffer(0)]], device const float4* input [[buffer(1)]],
    device half4* packed_weights [[buffer(2)]], device half4* packed_input [[buffer(3)]],
    constant Hy4FlashDenseMmArgs& shape [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    const uint row4 = shape.n_in / 4, weights4 = shape.n_out * row4;
    if (i < weights4) {
        packed_weights[i] = stark_hy4_q8_fragment(weights + ulong(i / row4) * shape.row_bytes, 4 * (i % row4));
    } else if (i - weights4 < shape.n_tok * row4) {
        packed_input[i - weights4] = half4(input[i - weights4]);
    }
}

kernel void stark_hy4_dense_device_views(
    device half* input [[buffer(0)]], device half* weight [[buffer(1)]],
    device float* output [[buffer(2)]], constant Hy4FlashDenseMmArgs& shape [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]]) {
    constexpr uint K = 64, N = 32, M = 16;
    const uint row0 = group.y * 128 + (simd / 2) * 32, col0 = group.x * 64 + (simd % 2) * 32;
    if (row0 >= shape.n_tok) return;
    const int upper_rows = int(min(M, shape.n_tok - row0));
    const int lower_rows = int(min(M, shape.n_tok - min(shape.n_tok, row0 + M)));
    using Shape = dextents<int32_t, 2>;
    using Strides = array<int32_t, 2>;
    using Tile = tensor<device half, Shape, tensor_inline>;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        M, N, K, false, true, true, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> product;
    auto upper = product.template get_destination_cooperative_tensor<Tile, Tile, float>();
    auto lower = product.template get_destination_cooperative_tensor<Tile, Tile, float>();
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < upper.get_capacity(); ++i) {
        upper[i] = 0.0f;
        lower[i] = 0.0f;
    }
    const Strides strides{1, int(shape.n_in)};
    for (uint k = 0; k < shape.n_in; k += K) {
        auto rhs = Tile(weight + ulong(col0) * shape.n_in + k, Shape(K, N), strides);
        auto top = Tile(input + ulong(row0) * shape.n_in + k, Shape(K, upper_rows), strides);
        product.run(top, rhs, upper);
        if (lower_rows > 0) {
            auto bottom = Tile(input + ulong(row0 + M) * shape.n_in + k, Shape(K, lower_rows), strides);
            product.run(bottom, rhs, lower);
        }
    }
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < upper.get_capacity(); ++i) {
        const auto at = upper.get_multidimensional_index(i);
        const uint token = row0 + at[1], channel = col0 + at[0];
        if (token < shape.n_tok) output[ulong(token) * shape.y_stride + channel] = upper[i];
        if (token + M < shape.n_tok) output[ulong(token + M) * shape.y_stride + channel] = lower[i];
    }
}
