// Vision-only binding wrappers around the measured main-engine source-compatible bodies.
// Arithmetic bodies and existing PyTorch/SLEEF attribution are preserved.
// Numerical compatibility implementation of the BF16 RowwiseMoments reduction
// schedule in PyTorch v2.14.1, aten/src/ATen/native/cpu/moments_utils.h and
// layer_norm_kernel.cpp. See scripts/licenses/distribution/pytorch-LICENSE for BSD notices.

#include <metal_stdlib>
using namespace metal;

// Only explicit fma calls in the Welford reduction may contract. The reference
// affine rounds each FP32 multiply/add before its single final BF16 conversion.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

struct VisionMoments {
  uint count;
  float4 mean;
  float4 squared_deviation;
};

static void vision_combine_moments(thread VisionMoments& target, VisionMoments other) {
  const uint old_count = target.count;
  const uint count = old_count + other.count;
  const float fraction = count == 0 ? 0.0f : float(other.count) / float(count);
  const float4 delta = other.mean - target.mean;
  const float4 weighted_delta = fraction * delta;
  const float4 old_delta = delta * float(old_count);
  const float4 deviations = target.squared_deviation + other.squared_deviation;
  target.mean = target.mean + weighted_delta;
  target.squared_deviation = fma(old_delta, weighted_delta, deviations);
  target.count = count;
}

kernel void vision_layernorm_bf16_compat(
    const device bfloat* input [[buffer(0)]],
    const device bfloat* weight [[buffer(1)]],
    const device bfloat* bias [[buffer(2)]],
    device bfloat* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
  if (row >= rows) return;
  const device bfloat* row_input = input + row * width;
  const uint vectors = width / 8;
  const uint chunks = (vectors + 15) / 16;
  uint depth = 0;
  for (uint power = 1; power < chunks; power <<= 1) {
    ++depth;
  }
  // VisionTowerConfig bounds a row below 2^20 elements, hence depth <= 13.
  VisionMoments stack[16] = {};
  for (uint chunk = 0; chunk < chunks; ++chunk) {
    const uint count = min(16u, vectors - chunk * 16);
    VisionMoments halves[2] = {};
    for (uint index = 0; index < count; ++index) {
      const float fraction = 1.0f / float(index + 1);
      for (uint part = 0; part < 2; ++part) {
        const uint start = chunk * 128 + index * 8 + part * 4;
        const float4 value = float4(float(row_input[start]), float(row_input[start + 1]),
                                   float(row_input[start + 2]), float(row_input[start + 3]));
        const float4 delta = value - halves[part].mean;
        halves[part].mean = fma(delta, fraction, halves[part].mean);
        const float4 centered = value - halves[part].mean;
        halves[part].squared_deviation = fma(delta, centered, halves[part].squared_deviation);
      }
    }
    halves[0].count = count;
    halves[1].count = count;
    vision_combine_moments(stack[0], halves[0]);
    vision_combine_moments(stack[0], halves[1]);
    uint mask = chunk + 1;
    uint level = 1;
    while (level < depth && (mask & 1) == 0) {
      vision_combine_moments(stack[level], stack[level - 1]);
      stack[level - 1] = VisionMoments{};
      mask >>= 1;
      ++level;
    }
  }
  for (uint level = 1; level < depth; ++level) {
    vision_combine_moments(stack[0], stack[level]);
  }
  uint count = 0;
  float mean = 0.0f;
  float squared_deviation = 0.0f;
  for (uint index = vectors * 8; index < width; ++index) {
    const float value = float(row_input[index]);
    const float delta = value - mean;
    ++count;
    mean += delta / float(count);
    squared_deviation = fma(delta, value - mean, squared_deviation);
  }
  const uint lane_count = vectors * 2;
  for (uint lane = 0; lane < 4; ++lane) {
    const uint total = count + lane_count;
    const float fraction = total == 0 ? 0.0f : float(lane_count) / float(total);
    const float delta = stack[0].mean[lane] - mean;
    // The pinned scalar AddMoments contracts these expressions, unlike its
    // vector combine mean update and final affine.
    mean = fma(fraction, delta, mean);
    const float weighted_square = delta * delta * fraction;
    const float correction = fma(weighted_square, float(count), stack[0].squared_deviation[lane]);
    squared_deviation += correction;
    count = total;
  }
  const float inverse_std = 1.0f / metal::precise::sqrt(squared_deviation / float(width) + epsilon);
  const float offset = -inverse_std * mean;
  for (uint index = 0; index < width; ++index) {
    const float scaled = float(row_input[index]) * inverse_std;
    const float centered = scaled + offset;
    const float affine = centered * float(weight[index]);
    output[row * width + index] = bfloat(affine + float(bias[index]));
  }
}

// Vision-only numerical compatibility with PyTorch v2.14.1 Gelu.h and SLEEF
// 5a1d179df9cf652951b59010a2d2075372d67f68 tanhf_u10/df helpers.
// BSD/Boost notices are retained in scripts/licenses/distribution.

#include <metal_stdlib>
using namespace metal;

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

kernel void vision_gelu_bf16_compat(
    device bfloat* values [[buffer(0)]],
    constant uint& length [[buffer(1)]],
    constant uint& approximate [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
  if (index >= length) return;
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
