// Pinned PyTorch SumKernel.cpp F32 128-wide reduction and Activation.cpp SiLU.
// Revision 5c4886908584029761b579af026dcfb627c84070; BSD terms/copyrights in
// scripts/licenses/distribution/pytorch-LICENSE. Legacy GDN kernels are separate.
#include <metal_stdlib>
#include "../common/dsl.h"

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include "../common/vision_exp.h"

// Positive SLEEF log1pf_u10 branch; softplus invokes it only for x <= 20.
// SLEEF/df helpers share the retained Boost Software License distribution notice.
static float gdn_log1p_positive(float d) {
  const int exponent = int((as_type<uint>((d + 1.0f) * (1.0f / 0.75f)) >> 23) & 255u) - 127;
  const float t = as_type<float>(uint(127 - exponent) << 23);
  const float m = fma(d, t, t - 1.0f);
  const float sh = 0.6931471824645996f * float(exponent);
  float2 s = float2(sh, fma(-1.904654323148236e-9f, float(exponent), fma(0.6931471824645996f, float(exponent), -sh)));
  const float dh = 2.0f + m;
  const float dl = (2.0f - dh) + m;
  const float reciprocal = 1.0f / dh;
  const float hi = m * reciprocal;
  const float u = fma(reciprocal, m, -hi);
  const float v = fma(-dl, reciprocal, fma(-dh, reciprocal, 1.0f));
  const float2 x = float2(hi, fma(hi, v, fma(0.0f, reciprocal, u)));
  const float x2 = x.x * x.x;
  const float polynomial = fma(fma(0.3027294874f, x2, 0.3996108174f), x2, 0.6666694880f);
  const float2 y = x * 2.0f;
  const float sum = s.x + y.x;
  s = float2(sum, (((s.x - sum) + y.x) + s.y) + y.y);
  const float term = (x2 * x.x) * polynomial;
  const float result_hi = s.x + term;
  const float result_lo = ((s.x - result_hi) + term) + s.y;
  return result_hi + result_lo;
}

PUBLIC KERNEL(ReferenceGDNDecay)(
    const device bfloat* controls, const device float* a_log, const device float* dt_bias, device float* output,
    constant uint& rows, constant uint& heads, constant uint& conv_dim, constant uint& value_dim,
    constant uint& row_stride, constant uint& write_log,
    const uint index AXIS(rows * heads, 128)
) {
  const uint row = index / heads, head = index % heads;
  const float x = float(controls[row * row_stride + conv_dim + value_dim + heads + head]) + dt_bias[head];
  const float softplus = x > 20.0f ? x : gdn_log1p_positive(vision_exp(x));
  const float g = -vision_exp(a_log[head]) * softplus;
  output[index] = write_log != 0 ? g : vision_exp(g);
}

static float gdn_square_sum_128(const device bfloat* input) {
  float4 registers[4] = {};
  for (uint block = 0; block < 8; ++block) {
    for (uint reg = 0; reg < 4; ++reg) {
      const uint index = block * 16 + reg * 4;
      const float4 x = float4(float(input[index]), float(input[index + 1]), float(input[index + 2]), float(input[index + 3]));
      registers[reg] += x * x;
    }
  }
  float4 lanes = registers[0];
  for (uint reg = 1; reg < 4; ++reg) lanes += registers[reg];
  float sum = 0.0f;
  for (uint lane = 0; lane < 4; ++lane) sum += lanes[lane];
  return sum;
}

PUBLIC KERNEL(ReferenceGDNQKL2)(
    const device bfloat* input, device float* q_norm, device float* k_norm,
    constant uint& key_dim, constant uint& rows, constant uint& row_stride,
    const uint index AXIS(rows * (key_dim / 128), 32)
) {
  const uint heads = key_dim / 128;
  const uint row = index / heads;
  const uint head = index % heads;
  const device bfloat* q = input + row * row_stride + head * 128;
  const device bfloat* k = q + key_dim;
  const float q_inverse = 1.0f / metal::precise::sqrt(gdn_square_sum_128(q) + 1e-6f);
  const float k_inverse = 1.0f / metal::precise::sqrt(gdn_square_sum_128(k) + 1e-6f);
  const uint base = row * key_dim + head * 128;
  for (uint channel = 0; channel < 128; ++channel) {
    q_norm[base + channel] = (float(q[channel]) * q_inverse) * 0.0883883461356163f;
    k_norm[base + channel] = float(k[channel]) * k_inverse;
  }
}

PUBLIC KERNEL(ReferenceGDNNormGate)(
    device bfloat* core, const device bfloat* controls, const device float* scales,
    constant uint& rows, constant uint& num_heads, constant uint& value_dim,
    constant uint& conv_dim, constant uint& row_stride, constant float& epsilon,
    const uint index AXIS(rows * num_heads, 32)
) {
  const uint row = index / num_heads;
  const uint head = index % num_heads;
  const uint base = row * value_dim + head * 128;
  const float mean = gdn_square_sum_128(core + base) / 128.0f;
  const float inverse = 1.0f / metal::precise::sqrt(mean + epsilon);
  for (uint channel = 0; channel < 128; ++channel) {
    const float normalized = float(bfloat(float(core[base + channel]) * inverse));
    const float gain = float(bfloat(scales[channel]));
    const float affine = float(bfloat(normalized * gain));
    const float z = float(controls[row * row_stride + conv_dim + head * 128 + channel]);
    const float silu = z / (1.0f + vision_exp(-z));
    core[base + channel] = bfloat(affine * silu);
  }
}
