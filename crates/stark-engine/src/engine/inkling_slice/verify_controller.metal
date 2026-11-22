// SPDX-License-Identifier: MIT
// Copyright © 2025 Apple Inc.
// Keep the gate epilogue from prefill_controller.metal (LICENSE.mlx-reference).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

template <int TileRows, int Columns>
inline void expand_native(
    device bfloat *compressed, device bfloat *gain, device bfloat *forget,
    device const float *input_message, device float *mixed, uint rows, uint tile) {
    constexpr int rank=64, width=2304;
    const int first=int(tile)*Columns;
    using dims=dextents<int,2>;
    using strides=array<int,2>;
    using operand=tensor<device bfloat,dims,tensor_inline>;
    auto a=operand(compressed,dims(rank,int(rows)),strides{1,rank});
    auto gain_tile=operand(gain+first*rank,dims(rank,Columns),strides{1,rank});
    auto forget_tile=operand(forget+first*rank,dims(rank,Columns),strides{1,rank});
    constexpr auto descriptor=mpp::tensor_ops::matmul2d_descriptor(
        TileRows,Columns,int(dynamic_extent),false,true,true);
    mpp::tensor_ops::matmul2d<descriptor,execution_simdgroup> product;
    auto g=product.template get_destination_cooperative_tensor<operand,operand,float>();
    auto f=product.template get_destination_cooperative_tensor<operand,operand,float>();
    product.run(a,gain_tile,g);
    product.run(a,forget_tile,f);
    {
#pragma clang fp contract(off) reassociate(off)
        for (ushort i=0;i<g.get_capacity();++i) {
            const auto coord=g.get_multidimensional_index(i);
            const uint row=coord[1];
            if (row<rows) {
                const uint channel=first+coord[0];
                const float keep=1.0f-metal::precise::tanh(f[i]);
                const float admit=metal::precise::tanh(g[i]);
                const device float *input=input_message+row*2*width;
                mixed[row*width+channel]=input[channel]*keep+input[width+channel]*admit;
            }
        }
    }
}

#define CONTROLLER_NATIVE(KERNEL, ROWS, COLS) \
kernel void KERNEL(device bfloat *compressed [[buffer(0)]], device bfloat *gain [[buffer(1)]], \
    device bfloat *forget [[buffer(2)]], device const float *input [[buffer(3)]], \
    device float *mixed [[buffer(4)]], constant uint &rows [[buffer(5)]], \
    uint tile [[threadgroup_position_in_grid]]) { \
    expand_native<ROWS,COLS>(compressed,gain,forget,input,mixed,rows,tile); \
}

CONTROLLER_NATIVE(ink_verify_controller_native_m16n16,16,16)
