#include <metal_stdlib>
#include "../common/dsl.h"

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

// SLEEF expf_u10, revision 5a1d179df9cf652951b59010a2d2075372d67f68.
// Copyright Naoki Shibata and contributors, Boost Software License 1.0;
// see scripts/licenses/distribution/sleef-LICENSE.txt.
static float vision_exp(float x) {
  if (x < -104.0f) return 0.0f;
  if (x > 100.0f) return INFINITY;
  const float q = rint(x * 1.4426950408889634f);
  float s = fma(q, -0.693145751953125f, x);
  s = fma(q, -1.428606765330187e-6f, s);
  float u = 0.00019852761761285365f;
  u = fma(u, s, 0.0013930435525253415f);
  u = fma(u, s, 0.008333360776305199f);
  u = fma(u, s, 0.041666485369205475f);
  u = fma(u, s, 0.1666666716337204f);
  u = fma(u, s, 0.5f);
  u = 1.0f + fma(s * s, u, s);
  const int first_exponent = int(q) >> 1;
  const float first = as_type<float>(uint(first_exponent + 127) << 23);
  const float second = as_type<float>(uint(int(q) - first_exponent + 127) << 23);
  return (u * first) * second;
}

// Deliberately materialize BF16 probabilities to match the pinned eager vision
// reference. The language model's fused attention kernels are unchanged.
// FP32 reduction/normalization order follows PyTorch v2.14.1 SoftMaxKernel.cpp
// and functional_base.h; BSD notices are in scripts/licenses/distribution.
PUBLIC KERNEL(VisionSoftmaxBF16)(
    device bfloat* values,
    constant uint& row_length,
    constant uint& rows,
    const uint row_index AXIS(rows, 32)
) {
  device bfloat* row = values + row_index * row_length;
  float maximum = -INFINITY;
  for (uint column = 0; column < row_length; ++column) {
    maximum = max(maximum, float(row[column]));
  }
  float4 partial = 0.0f;
  for (uint column = 0; column < row_length; ++column) {
    partial[column % 4] += vision_exp(float(row[column]) - maximum);
  }
  float total = (partial[0] + partial[2]) + (partial[1] + partial[3]);
  if (row_length < 4) {
    total = partial[0];
    for (uint lane = 1; lane < row_length; ++lane) total += partial[lane];
  }
  const float inverse_total = 1.0f / total;
  for (uint column = 0; column < row_length; ++column) {
    row[column] = bfloat(vision_exp(float(row[column]) - maximum) * inverse_total);
  }
}

PUBLIC KERNEL(VisionHeadCopyBF16)(
    const device bfloat* input,
    device bfloat* output,
    constant uint& length,
    constant uint& head_dim,
    constant uint& model_dim,
    constant uint& head_index,
    const uint index AXIS(length, 32)
) {
  uint destination = (index / head_dim) * model_dim + head_index * head_dim + index % head_dim;
  output[destination] = input[index];
}
