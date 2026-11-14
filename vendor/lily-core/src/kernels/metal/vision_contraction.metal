// Vision-only source-compatible contraction adapted from Stark's main engine and PyTorch
// 5c4886908584029761b579af026dcfb627c84070 BlasKernel.cpp,
// ReducedPrecisionFloatGemvFastPathKernel.cpp and functional_base.h.
// BSD/MIT notices: scripts/licenses/distribution/pytorch-LICENSE and ggml-LICENSE.
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

kernel void vision_contraction_bf16_bias(
    device const bfloat* input [[buffer(0)]],
    device const bfloat* weights [[buffer(1)]],
    device const bfloat* bias [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& streams [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= M * N) return;
    const uint row = index / N;
    const uint column = index % N;
    float result;
    if (streams == 32) {
        float4 sums[8] = {};
        const uint aligned = K / 32 * 32;
        for (uint start = 0; start < aligned; start += 32) {
            for (uint reg = 0; reg < 8; ++reg) {
                for (uint lane = 0; lane < 4; ++lane) {
                    const uint inner = start + reg * 4 + lane;
                    sums[reg][lane] = fma(float(input[row * K + inner]),
                        float(weights[column * K + inner]), sums[reg][lane]);
                }
            }
        }
        for (uint offset = 4; offset != 0; offset >>= 1) {
            for (uint reg = 0; reg < offset; ++reg) sums[reg] = sums[reg] + sums[reg + offset];
        }
        result = (sums[0][0] + sums[0][1]) + (sums[0][2] + sums[0][3]);
        float4 tail = 0.0f;
        const uint vector_aligned = K / 8 * 8;
        for (uint start = aligned; start < vector_aligned; start += 8) {
            for (uint part = 0; part < 2; ++part) {
                for (uint lane = 0; lane < 4; ++lane) {
                    const uint inner = start + part * 4 + lane;
                    tail[lane] = fma(float(input[row * K + inner]),
                        float(weights[column * K + inner]), tail[lane]);
                }
            }
        }
        result += (tail[0] + tail[1]) + (tail[2] + tail[3]);
        for (uint inner = vector_aligned; inner < K; ++inner)
            result = fma(float(input[row * K + inner]), float(weights[column * K + inner]), result);
    } else {
        float4 sums = 0.0f;
        const uint aligned = K / 4 * 4;
        for (uint start = 0; start < aligned; start += 4) {
            for (uint lane = 0; lane < 4; ++lane) {
                const uint inner = start + lane;
                sums[lane] = fma(float(input[row * K + inner]),
                    float(weights[column * K + inner]), sums[lane]);
            }
        }
        for (uint inner = aligned; inner < K; ++inner)
            sums[0] = fma(float(input[row * K + inner]), float(weights[column * K + inner]), sums[0]);
        result = ((sums[0] + sums[1]) + sums[2]) + sums[3];
    }
    output[index] = bfloat(result + float(bias[column]));
}
