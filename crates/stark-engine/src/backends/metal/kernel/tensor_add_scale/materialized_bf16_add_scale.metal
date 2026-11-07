#include <metal_stdlib>
#include "../common/dsl.h"

// Gemma's source profile stores the residual sum in BF16 before scaling it.
// Keep both storage boundaries while removing the intermediate dispatch.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

PUBLIC KERNEL(MaterializedBF16AddScale)(
    const device bfloat* input OPTIONAL(!in_place),
    const device bfloat* bias,
    device bfloat* output,
    constant uint& num_cols,
    constant uint& length,
    constant float& scale,
    const uint position AXIS(length, 32),
    const bool in_place SPECIALIZE
) {
  if (in_place) input = output;
  const bfloat sum = bfloat(float(input[position]) + float(bias[position % num_cols]));
  output[position] = bfloat((float(sum) + 0.0f) * scale);
}
