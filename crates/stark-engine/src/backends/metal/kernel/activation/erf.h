#ifndef STARK_METAL_ERF_H
#define STARK_METAL_ERF_H

#include <metal_stdlib>

inline float erf_precise(float a) {
  // Faithfully rounded erff polynomial from Norbert Juffa, also used by MLX.
  float t = metal::abs(a);
  float s = a * a;

  if (t > 0.927734375f) {
    float r = metal::fma(-1.72853470e-5f, t, 3.83197126e-4f);
    float u = metal::fma(-3.88396438e-3f, t, 2.42546219e-2f);
    r = metal::fma(r, s, u);
    r = metal::fma(r, t, -1.06777877e-1f);
    r = metal::fma(r, t, -6.34846687e-1f);
    r = metal::fma(r, t, -1.28717512e-1f);
    r = metal::fma(r, t, -t);
    r = 1.0f - metal::precise::exp(r);
    return metal::copysign(r, a);
  }

  float r = -5.96761703e-4f;
  r = metal::fma(r, s, 4.99119423e-3f);
  r = metal::fma(r, s, -2.67681349e-2f);
  r = metal::fma(r, s, 1.12819925e-1f);
  r = metal::fma(r, s, -3.76125336e-1f);
  r = metal::fma(r, s, 1.28379166e-1f);
  return metal::fma(r, a, a);
}

#endif
