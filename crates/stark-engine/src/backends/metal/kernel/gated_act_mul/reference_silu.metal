#include <metal_stdlib>
#include "../common/dsl.h"

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include "../common/vision_exp.h"

// PyTorch 5c4886908584029761b579af026dcfb627c84070 Activation.cpp/NEON formula;
// BSD terms are preserved in scripts/licenses/distribution/pytorch-LICENSE.
// SLEEF u10 provenance is
// recorded with the shared exponential helper and supplemental licenses.
PUBLIC KERNEL(ReferenceGatedSiLUBF16)(
    const device bfloat* packed,
    device bfloat* output,
    constant uint& gated_dim,
    constant uint& rows,
    const uint index AXIS(rows * gated_dim, 128)
) {
  const uint row = index / gated_dim;
  const uint column = index % gated_dim;
  const uint base = row * 2 * gated_dim;
  const float gate = float(packed[base + gated_dim + column]);
  const bfloat activated = bfloat(gate / (1.0f + vision_exp(-gate)));
  output[index] = bfloat(float(packed[base + column]) * float(activated));
}
