#pragma once

#include <metal_stdlib>
#include "activation_type.h"
#include "erf.h"

using namespace stark::activation_type;

template <typename T>
inline T activate_silu_alpha(T x, float alpha) {
  float xf = float(x);
  float y = 1.0f / (1.0f + fast::exp(-fabs(xf) * alpha));
  float out = (xf < 0.0f) ? (1.0f - y) * xf : y * xf;
  return static_cast<T>(out);
}

template <typename T>
inline T activate_silu(T x) {
  return activate_silu_alpha(x, 1.0);
}

// Independently follows MLX v0.32.3 unary_ops.h Sigmoid and bf16_math.h
// (Copyright 2023-2024 Apple Inc., MIT). Quantized OLMo artifacts opt in;
// existing activation policies retain their float intermediate expression.
template <typename T>
inline T activate_silu_materialized_sigmoid(T x) {
  const T exponential = T(metal::precise::exp(metal::abs(float(x))));
  const T denominator = T(T(1) + exponential);
  const T lower = T(T(1) / denominator);
  const T sigmoid = x < T(0) ? lower : T(T(1) - lower);
  return T(x * sigmoid);
}

template <typename T>
inline T activate_gelu(T x) {
  float xf = float(x);
  float yf = 0.5f * xf * (1.0f + metal::precise::tanh(GELU_K1 * (xf + GELU_K0 * xf * xf * xf)));
  return T(yf);
}

// MLX-LM's BF16 approximate GELU consists of distinct tensor primitives and
// BF16 weak coefficients. Keep this declared source profile separate from the
// canonical FP32 polynomial above. MLX 0.32.3 nn/layers/activations.py,
// Copyright 2023 Apple Inc., MIT; called by MLX-LM 1f7057ca.
template <typename T>
inline T activate_gelu_materialized_polynomial(T x) {
#pragma METAL fp contract(off)
  const float xf = float(x);
  const float cubic = float(T(xf * xf * xf));
  const float term = float(T(float(T(GELU_K0)) * cubic));
  const float inner = float(T(xf + term));
  const float argument = float(T(float(T(GELU_K1)) * inner));
  const float activated = float(T(metal::precise::tanh(argument)));
  const float increment = float(T(1.0f + activated));
  const float half_x = float(T(0.5f * xf));
  return T(half_x * increment);
}
#pragma METAL fp contract(on)

template <typename T>
inline T activate_gelu_exact(T x) {
  constexpr float inv_sqrt_2 = 0.7071067811865475f;
  float xf = float(x);
  float yf = 0.5f * xf * (1.0f + erf_precise(inv_sqrt_2 * xf));
  return T(yf);
}

template <typename T>
inline T activate_leaky_relu(T x, float alpha) {
  const float xf = float(x);
  const float yf = (xf >= 0.0f) ? xf : (alpha * xf);
  return static_cast<T>(yf);
}

template <typename T>
inline T activate_softplus(T x) {
  float xf = static_cast<float>(x);
  if (xf > 20.0f) {
    return x;
  }

  float result = fast::log(1.0f + fast::exp(xf));
  return static_cast<T>(result);
}

template <typename T>
inline T activate(T x, ActivationType type) {
  switch (type) {
  case ActivationType::SILU:
    return activate_silu(x);
  case ActivationType::SILUMaterializedSigmoid:
    return activate_silu_materialized_sigmoid(x);
  case ActivationType::GELUApprox:
    return activate_gelu(x);
  case ActivationType::GELUMaterializedPolynomial:
    return activate_gelu_materialized_polynomial(x);
  case ActivationType::GELUExact:
    return activate_gelu_exact(x);
  case ActivationType::IDENTITY:
    return x;
  case ActivationType::SOFTPLUS:
    return activate_softplus(x);
  }
}
