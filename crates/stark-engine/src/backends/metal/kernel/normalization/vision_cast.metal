#include <metal_stdlib>
#include "../common/dsl.h"

PUBLIC KERNEL(ReferenceRoundBF16F32)(
    const device float* input, device float* output, constant uint& length,
    const uint index AXIS(length, 32)
) {
  output[index] = float(bfloat(input[index]));
}

PUBLIC KERNEL(VisionCastBF16)(
    const device float* input,
    device bfloat* output,
    constant uint& length,
    const uint index AXIS(length, 32)
) {
  output[index] = bfloat(input[index]);
}

PUBLIC KERNEL(VisionBiasCastBF16)(
    const device float* input,
    const device bfloat* bias,
    device bfloat* output,
    constant uint& row_width,
    constant uint& length,
    const uint index AXIS(length, 32)
) {
  output[index] = bfloat(input[index] + float(bias[index % row_width]));
}
