#include <metal_stdlib>
#include "../common/dsl.h"

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

PUBLIC KERNEL(SiglipPoolBF16)(
    const device bfloat* input,
    device bfloat* output,
    constant uint& width,
    constant uint& side,
    constant uint& pool,
    const uint index AXIS((side / pool) * (side / pool) * width, 32)
) {
  const uint channel = index % width;
  const uint row = index / width;
  const uint out_side = side / pool;
  const uint y = row / out_side;
  const uint x = row % out_side;
  float sum = 0.0f;
  for (uint dy = 0; dy < pool; ++dy) {
    for (uint dx = 0; dx < pool; ++dx) {
      sum += float(input[((y * pool + dy) * side + x * pool + dx) * width + channel]);
    }
  }
  output[index] = bfloat(sum / float(pool * pool));
}
