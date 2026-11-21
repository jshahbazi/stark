// SPDX-License-Identifier: Apache-2.0
// Tiled BF16 linear transform using Metal 4's tensor operations.
// The FP32 accumulator and BF16 output preserve the source stage boundary.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void inkling_visual_linear_mpp(
    device bfloat *input [[buffer(0)]],
    device bfloat *weight [[buffer(1)]],
    device const bfloat *bias [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant uint4 &p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]]
) {
    constexpr int tile_m = 16;
    constexpr int tile_n = 32;
    constexpr int tile_k = 32;
    int row = int(group.y) * tile_m;
    int column = int(group.x) * tile_n;
    int width = int(p.x), outputs = int(p.y);
    int rows = min(tile_m, int(p.z) - row);
    int columns = min(tile_n, outputs - column);

    using Dims = dextents<int, 2>;
    using Strides = array<int, 2>;
    using Tensor = tensor<device bfloat, Dims, tensor_inline>;
    constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
        tile_m, tile_n, tile_k, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<descriptor, execution_simdgroup> op;
    auto accumulator = op.get_destination_cooperative_tensor<Tensor, Tensor, float>();
    for (ushort i = 0; i < accumulator.get_capacity(); ++i) {
        accumulator[i] = 0.0f;
    }

    for (int k = 0; k < width; k += tile_k) {
        int elements = min(tile_k, width - k);
        Tensor a(input + row * width + k, Dims(elements, rows), Strides{1, width});
        Tensor b(weight + column * width + k, Dims(elements, columns), Strides{1, width});
        op.run(a, b, accumulator);
    }

    for (ushort i = 0; i < accumulator.get_capacity(); ++i) {
        auto coord = accumulator.get_multidimensional_index(i);
        if (coord[0] < columns && coord[1] < rows) {
            int channel = column + coord[0];
            output[(row + coord[1]) * outputs + channel] =
                bfloat(accumulator[i] + (p.w ? float(bias[channel]) : 0.0f));
        }
    }
}
