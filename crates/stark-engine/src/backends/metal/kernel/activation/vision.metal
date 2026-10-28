// Vision-only numerical compatibility with PyTorch v2.14.1 Gelu.h and SLEEF
// 5a1d179df9cf652951b59010a2d2075372d67f68 tanhf_u10/df helpers.
// BSD/Boost notices are retained in scripts/licenses/distribution.

#include <metal_stdlib>
#include "../common/dsl.h"

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

static float2 vision_add2_f(float2 x, float y) {
  const float hi = x.x + y;
  const float v = hi - x.x;
  return float2(hi, ((x.x - (hi - v)) + (y - v)) + x.y);
}

static float2 vision_add2(float2 x, float2 y) {
  const float hi = x.x + y.x;
  const float v = hi - x.x;
  return float2(hi, ((x.x - (hi - v)) + (y.x - v)) + (x.y + y.y));
}

static float2 vision_f_add(float x, float2 y) {
  const float hi = x + y.x;
  return float2(hi, ((x - hi) + y.x) + y.y);
}

static float2 vision_add(float2 x, float2 y) {
  const float hi = x.x + y.x;
  return float2(hi, (((x.x - hi) + y.x) + x.y) + y.y);
}

static float2 vision_mul(float2 x, float2 y) {
  const float hi = x.x * y.x;
  return float2(hi, fma(x.x, y.y, fma(x.y, y.x, fma(x.x, y.x, -hi))));
}

static float2 vision_mul_f(float2 x, float y) {
  const float hi = x.x * y;
  return float2(hi, fma(x.y, y, fma(x.x, y, -hi)));
}

static float2 vision_square(float2 x) {
  const float hi = x.x * x.x;
  return float2(hi, fma(x.x + x.x, x.y, fma(x.x, x.x, -hi)));
}

static float2 vision_reciprocal(float2 d) {
  const float hi = 1.0f / d.x;
  return float2(hi, hi * fma(-d.y, hi, fma(-d.x, hi, 1.0f)));
}

static float2 vision_divide(float2 n, float2 d) {
  const float t = 1.0f / d.x;
  const float hi = n.x * t;
  const float u = fma(t, n.x, -hi);
  const float v = fma(-d.y, t, fma(-d.x, t, 1.0f));
  return float2(hi, fma(hi, v, fma(n.y, t, u)));
}

static float2 vision_exp_pair(float2 d) {
  const float q = rint((d.x + d.y) * 1.4426950408889634f);
  float2 s = vision_add2_f(d, q * -0.693145751953125f);
  s = vision_add2_f(s, q * -1.428606765330187e-6f);
  float u = 0.1980960224e-3f;
  u = fma(u, s.x, 0.1394256484e-2f);
  u = fma(u, s.x, 0.8333456703e-2f);
  u = fma(u, s.x, 0.4166637361e-1f);
  float2 t = vision_add2_f(vision_mul_f(s, u), 0.16666665941423424f);
  t = vision_add2_f(vision_mul(s, t), 0.5f);
  t = vision_add2(s, vision_mul(vision_square(s), t));
  t = vision_f_add(1.0f, t);
  const float scale = as_type<float>(uint(int(q) + 127) << 23);
  return t * scale;
}

static float vision_tanh(float x) {
  if (isnan(x)) return x;
  if (abs(x) > 8.664339742f) return copysign(1.0f, x);
  const float2 d = vision_exp_pair(float2(abs(x), 0.0f));
  const float2 e = vision_reciprocal(d);
  const float2 result = vision_divide(vision_add(d, -e), vision_add(d, e));
  return copysign(result.x + result.y, x);
}

// The pinned NEON erf uses this Arm polynomial, rather than scalar libm erff.
static float vision_erf_exp(float x) {
  if (x < -87.33654f) return 0.0f;
  const float n = round(x * 1.4426950408889634f);
  float r = fma(-n, as_type<float>(0x3f317200u), x);
  r = fma(-n, as_type<float>(0x35bfbe8eu), r);
  const float scale = as_type<float>(uint(int(n) + 127) << 23);
  const float r2 = r * r;
  float p = fma(r, as_type<float>(0x3c072010u), as_type<float>(0x3d2b9f17u));
  float q = fma(r, as_type<float>(0x3e2aaf33u), as_type<float>(0x3efffedbu));
  q = fma(p, r2, q);
  p = as_type<float>(0x3f7ffff6u) * r;
  const float polynomial = fma(q, r2, p);
  return fma(scale, polynomial, scale);
}

static float vision_erf(float x) {
  const float t = 1.0f / fma(0.3275911f, abs(x), 1.0f);
  float r = fma(1.061405429f, t, -1.453152027f);
  r = fma(r, t, 1.421413741f);
  r = fma(r, t, -0.284496736f);
  r = fma(r, t, 0.254829592f);
  const float exponent = vision_erf_exp(-(x * x));
  const float negative_exp_t = t * -exponent;
  return copysign(fma(negative_exp_t, r, 1.0f), x);
}

PUBLIC KERNEL(VisionGeluBF16)(
    device bfloat* values,
    constant uint& length,
    constant uint& approximate,
    const uint index AXIS(length, 32)
) {
  // Metal flushes FP32 subnormal arithmetic. For these BF16 inputs both GELUs
  // round to x/2, so preserve the correctly rounded result directly in bits.
  const ushort input_bits = as_type<ushort>(values[index]);
  const ushort magnitude = input_bits & 0x7fffu;
  if (magnitude <= 0x00ffu) {
    const ushort truncated = magnitude >> 1;
    const ushort rounded = truncated + ((magnitude & 1u) & (truncated & 1u));
    values[index] = as_type<bfloat>(ushort((input_bits & 0x8000u) | rounded));
    return;
  }
  const float x = float(values[index]);
  float cdf;
  if (approximate != 0) {
    const float cube = x * x * x;
    cdf = vision_tanh(0.7978845608028654f * (x + 0.044715f * cube));
  } else {
    cdf = vision_erf(x * 0.7071067811865475f);
  }
  values[index] = bfloat((0.5f * x) * (1.0f + cdf));
}
