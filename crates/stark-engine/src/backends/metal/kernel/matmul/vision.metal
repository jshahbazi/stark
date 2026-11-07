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

// Source-contraction witness: packed unsigned affine group64 coefficients
// are rounded once to BF16 without allocating a dequantized weight bank.
PUBLIC KERNEL(QwenAffineReferenceMatmulBF16)(
    const device bfloat* input,
    const device uchar* codes,
    const device bfloat* scales,
    const device bfloat* biases,
    device bfloat* output,
    constant uint& m,
    constant uint& n,
    constant uint& k,
    constant uint& bits,
    constant uint& scale_output_stride,
    constant uint& scale_group_stride,
    const uint index AXIS(m * n, 32)
) {
  if ((bits != 4 && bits != 5 && bits != 6 && bits != 8) || k % 64 != 0) {
    output[index] = bfloat(NAN);
    return;
  }
  const uint row = index / n;
  const uint column = index % n;
  const ulong row_bytes = ulong(k) * bits / 8;
  float4 sums[8] = {};
  for (uint start = 0; start < k; start += 32) {
    for (uint reg = 0; reg < 8; ++reg) {
      for (uint lane = 0; lane < 4; ++lane) {
        const uint inner = start + reg * 4 + lane;
        const uint bit = inner * bits;
        const uint shift = bit % 8;
        const ulong offset = ulong(column) * row_bytes + bit / 8;
        uint packed = codes[offset];
        if (shift + bits > 8) {
          packed |= uint(codes[offset + 1]) << 8;
        }
        const uint code = (packed >> shift) & ((1u << bits) - 1u);
        const ulong group = ulong(column) * scale_output_stride + ulong(inner / 64) * scale_group_stride;
        const float weight = float(bfloat(float(code) * float(scales[group]) + float(biases[group])));
        sums[reg][lane] = fma(float(input[ulong(row) * k + inner]), weight, sums[reg][lane]);
      }
    }
  }
  for (uint offset = 4; offset != 0; offset >>= 1) {
    for (uint reg = 0; reg < offset; ++reg) {
      sums[reg] = sums[reg] + sums[reg + offset];
    }
  }
  output[index] = bfloat((sums[0][0] + sums[0][1]) + (sums[0][2] + sums[0][3]));
}
