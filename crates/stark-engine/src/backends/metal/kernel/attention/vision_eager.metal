#include <metal_stdlib>
#include "../common/dsl.h"

// Deliberately materialize BF16 probabilities to match the pinned eager vision
// reference. The language model's fused attention kernels are unchanged.
PUBLIC KERNEL(VisionSoftmaxBF16)(
    device bfloat* values,
    constant uint& row_length,
    constant uint& rows,
    threadgroup float shared[32],
    const uint row_index GROUPS(rows),
    const uint tid THREADS(256)
) {
  device bfloat* row = values + row_index * row_length;
  const uint lane = tid % 32;
  const uint group = tid / 32;
  float maximum = -INFINITY;
  for (uint column = tid; column < row_length; column += 256) {
    maximum = max(maximum, float(row[column]));
  }
  maximum = simd_max(maximum);
  if (lane == 0) shared[group] = maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (group == 0) {
    float value = lane < 8 ? shared[lane] : -INFINITY;
    value = simd_max(value);
    if (lane == 0) shared[0] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maximum = shared[0];
  float total = 0.0f;
  for (uint column = tid; column < row_length; column += 256) {
    total += metal::precise::exp(float(row[column]) - maximum);
  }
  total = simd_sum(total);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) shared[group] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (group == 0) {
    float value = lane < 8 ? shared[lane] : 0.0f;
    value = simd_sum(value);
    if (lane == 0) shared[0] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  total = shared[0];
  for (uint column = tid; column < row_length; column += 256) {
    row[column] = bfloat(metal::precise::exp(float(row[column]) - maximum) / total);
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
