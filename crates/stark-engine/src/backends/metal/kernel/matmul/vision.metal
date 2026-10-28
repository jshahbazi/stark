// FP32 reduction schedules adapted from PyTorch commit
// 5c4886908584029761b579af026dcfb627c84070, BlasKernel.cpp,
// ReducedPrecisionFloatGemvFastPathKernel.cpp and functional_base.h.
// See scripts/licenses/distribution/pytorch-LICENSE and ggml-LICENSE for
// upstream BSD/MIT terms and copyrights.

#include <metal_stdlib>
#include "../common/dsl.h"

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

PUBLIC KERNEL(VisionMatmulBF16)(
    const device bfloat* input,
    const device bfloat* weights,
    device float* output,
    constant uint& m,
    constant uint& n,
    constant uint& k,
    constant uint& b_leading_dimension,
    constant uint& transpose_b,
    constant uint& streams,
    const uint index AXIS(m * n, 32)
) {
  const uint row = index / n;
  const uint column = index % n;
  float result;
  if (streams == 32) {
    float4 sums[8] = {};
    const uint aligned = k / 32 * 32;
    for (uint start = 0; start < aligned; start += 32) {
      for (uint reg = 0; reg < 8; ++reg) {
        for (uint lane = 0; lane < 4; ++lane) {
          const uint inner = start + reg * 4 + lane;
          const uint b_index = transpose_b != 0 ? column * b_leading_dimension + inner
                                               : inner * b_leading_dimension + column;
          sums[reg][lane] = fma(float(input[row * k + inner]), float(weights[b_index]), sums[reg][lane]);
        }
      }
    }
    for (uint offset = 4; offset != 0; offset >>= 1) {
      for (uint reg = 0; reg < offset; ++reg) {
        sums[reg] = sums[reg] + sums[reg + offset];
      }
    }
    result = (sums[0][0] + sums[0][1]) + (sums[0][2] + sums[0][3]);
    float4 tail = 0.0f;
    const uint vector_aligned = k / 8 * 8;
    for (uint start = aligned; start < vector_aligned; start += 8) {
      for (uint part = 0; part < 2; ++part) {
        for (uint lane = 0; lane < 4; ++lane) {
          const uint inner = start + part * 4 + lane;
          const uint b_index = transpose_b != 0 ? column * b_leading_dimension + inner
                                               : inner * b_leading_dimension + column;
          tail[lane] = fma(float(input[row * k + inner]), float(weights[b_index]), tail[lane]);
        }
      }
    }
    result += (tail[0] + tail[1]) + (tail[2] + tail[3]);
    for (uint inner = vector_aligned; inner < k; ++inner) {
      const uint b_index = transpose_b != 0 ? column * b_leading_dimension + inner
                                           : inner * b_leading_dimension + column;
      result = fma(float(input[row * k + inner]), float(weights[b_index]), result);
    }
  } else {
    float4 sums = 0.0f;
    const uint aligned = k / 4 * 4;
    for (uint start = 0; start < aligned; start += 4) {
      for (uint lane = 0; lane < 4; ++lane) {
        const uint inner = start + lane;
        const uint b_index = transpose_b != 0 ? column * b_leading_dimension + inner
                                             : inner * b_leading_dimension + column;
        sums[lane] = fma(float(input[row * k + inner]), float(weights[b_index]), sums[lane]);
      }
    }
    for (uint inner = aligned; inner < k; ++inner) {
      const uint b_index = transpose_b != 0 ? column * b_leading_dimension + inner
                                           : inner * b_leading_dimension + column;
      sums[0] = fma(float(input[row * k + inner]), float(weights[b_index]), sums[0]);
    }
    result = ((sums[0] + sums[1]) + sums[2]) + sums[3];
  }
  output[index] = result;
}
