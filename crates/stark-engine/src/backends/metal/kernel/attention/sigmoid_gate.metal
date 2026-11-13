#include <metal_stdlib>
#include "../common/dsl.h"

template <typename T>
VARIANTS(T, float, bfloat)
PUBLIC KERNEL(SigmoidGate)(
    const device T* gate,
    device T* output,
    const constant uint& gate_dim,
    const constant uint& batch_dim,
    const constant uint& gate_row_stride,
    const bool reference_bf16 SPECIALIZE,
    const bool source_bf16 SPECIALIZE,
    const uint gate_idx AXIS(gate_dim, 256),
    const uint batch_idx AXIS(batch_dim, 1)
) {
  const uint output_idx = batch_idx * gate_dim + gate_idx;
  if (source_bf16) {
    // MLX 36969c094354086e15d2a327a9c50feb20b9cf11 unary_ops.h Sigmoid
    // and bf16_math.h, Apple Inc., MIT (scripts/licenses/distribution/mlx-LICENSE).
    // Preserve the BF16 exponential, denominator, reciprocal, complement and
    // sigmoid-product boundaries. Existing precision policies do not opt in.
    const bfloat x = bfloat(gate[batch_idx * gate_row_stride + gate_idx]);
    const bfloat exponential = bfloat(metal::precise::exp(metal::abs(float(x))));
    const bfloat denominator = bfloat(1.0f + float(exponential));
    const bfloat lower = bfloat(1.0f / float(denominator));
    const bfloat sigmoid = x < bfloat(0) ? lower : bfloat(1.0f - float(lower));
    output[output_idx] = T(float(output[output_idx]) * float(sigmoid));
    return;
  }
  float sigmoid = 1.0f / (1.0f + exp(-float(gate[batch_idx * gate_row_stride + gate_idx])));
  if (reference_bf16) {
    sigmoid = float(bfloat(sigmoid));
  }
  output[output_idx] = T(float(output[output_idx]) * sigmoid);
}
