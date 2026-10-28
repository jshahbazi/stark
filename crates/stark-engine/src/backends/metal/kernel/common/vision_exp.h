#pragma once

// Include after the caller selects safe FP32 math and disables contraction.
// SLEEF expf_u10, revision 5a1d179df9cf652951b59010a2d2075372d67f68.
// Copyright Naoki Shibata and contributors, Boost Software License 1.0;
// see scripts/licenses/distribution/sleef-LICENSE.txt.
static float vision_exp(float x) {
  if (x < -104.0f) return 0.0f;
  if (x > 100.0f) return INFINITY;
  const float q = rint(x * 1.4426950408889634f);
  float s = fma(q, -0.693145751953125f, x);
  s = fma(q, -1.428606765330187e-6f, s);
  float u = 0.00019852761761285365f;
  u = fma(u, s, 0.0013930435525253415f);
  u = fma(u, s, 0.008333360776305199f);
  u = fma(u, s, 0.041666485369205475f);
  u = fma(u, s, 0.1666666716337204f);
  u = fma(u, s, 0.5f);
  u = 1.0f + fma(s * s, u, s);
  const int first_exponent = int(q) >> 1;
  const float first = as_type<float>(uint(first_exponent + 127) << 23);
  const float second = as_type<float>(uint(int(q) - first_exponent + 127) << 23);
  return (u * first) * second;
}
