#include <metal_stdlib>
#include "../common/dsl.h"

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

PUBLIC KERNEL(ReferenceCausalScoresBF16)(
    const device float* input,
    device bfloat* output,
    constant uint& key_count,
    constant uint& rows,
    constant uint& prefix_len,
    constant float& scale,
    const uint index AXIS(rows * key_count, 128)
) {
  const uint row = index / key_count;
  const uint column = index % key_count;
  const float dot = float(bfloat(input[index]));
  output[index] = column <= prefix_len + row ? bfloat(dot * scale) : bfloat(as_type<float>(0xff7f0000u));
}

#include "../common/vision_exp.h"

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
