// SPDX-License-Identifier: MIT
// Copyright (c) Ollama
// Adapted from Ollama 6949a42ad24450cf567d7ce1ff7d278ebb3eb5d8,
// mlx/bf16_matmul.go. Keep the source's padded 16x32 output tile.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

kernel void ink_unembed_tensor(
    device const bfloat *hidden [[buffer(0)]], device const bfloat *weight [[buffer(1)]],
    device float *scores [[buffer(2)]], constant uint2 &shape [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]]) {
    constexpr int rows = 16, channels = 32;
    const uint channel = group * channels;
    using dims = dextents<int, 2>;
    using strides = array<int, 2>;
    const int width = int(shape.x), columns = int(shape.y);
    tensor<device bfloat, dims, tensor_inline> input(
        const_cast<device bfloat *>(hidden), dims(width, rows), strides{1, width});
    tensor<device bfloat, dims, tensor_inline> weights(
        const_cast<device bfloat *>(weight) + size_t(channel) * width, dims(width, channels), strides{1, width});
    tensor<device float, dims, tensor_inline> output(
        scores + channel, dims(channels, rows), strides{1, columns});
    constexpr mpp::tensor_ops::matmul2d_descriptor tile(rows, channels, int(dynamic_extent), false, true, true);
    mpp::tensor_ops::matmul2d<tile, execution_simdgroup> product;
    product.run(input, weights, output);
}
