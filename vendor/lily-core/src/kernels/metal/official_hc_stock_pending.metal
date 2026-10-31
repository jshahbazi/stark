// SPDX-License-Identifier: MIT
// Copyright (c) 2026 David Dalcu
// Ported by Stark from MLX-Serve 046b647686419df06c6c7132fb1d15d76185c3e5,
// src/hc_decode2.zig ND_SOURCE and U2V_SOURCE. See LICENSE.mlx-serve.
// The original comments and arithmetic bodies are preserved verbatim.
// Pending-WR1 specialization; WR0/U2V remain in official_hc_stock.metal.
#include <metal_stdlib>
using namespace metal;
typedef uint uint32_t;
template <typename T, bool DENSE>
inline float hc_val(uint q) {
  if (DENSE) return float(as_type<T>(ushort(q)));
  return float(q);
}
template <int B>
inline ulong hc_pack8(const device uchar* p) {
  ulong r = 0;
  for (int b = 0; b < B; ++b) r |= ulong(p[b]) << (8 * b);
  return r;
}

kernel void official_hc_stock_nd_pending_noinject(
    device const bfloat* x_in [[buffer(0)]],
    device const bfloat* nw [[buffer(1)]],
    device const bfloat* iw [[buffer(2)]],
    device const float* eps [[buffer(3)]],
    device const bfloat* wo_in [[buffer(4)]],
    device const bfloat* wi_in [[buffer(5)]],
    device const uint* dw_q [[buffer(6)]],
    device const bfloat* dw_s [[buffer(7)]],
    device const bfloat* dw_b [[buffer(8)]],
    device bfloat* xn_out [[buffer(9)]],
    device float* parts_out [[buffer(10)]],
    device bfloat* xs_out [[buffer(11)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
    using T = bfloat;
    constexpr int HC=4, H=2560, R=320, GS=64, BITS=8, INJ=0, WR=1, RPS=1, ROWS_TG=8;
const uint t = thread_index_in_threadgroup;
const int row = int(threadgroup_position_in_grid.z);
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int s = int(threadgroup_position_in_grid.y) % HC;
const int g = int(threadgroup_position_in_grid.y) / HC;
constexpr int PER = H / 256;
constexpr int K = HC * H;
// A slot's 16 values are WPL words at 2/4/8 bits, else two 8-value packs of BITS bytes.
constexpr bool WORD = (32 % BITS) == 0;
constexpr bool DENSE = BITS == 16;
constexpr int VPW = WORD ? 32 / BITS : 8;
constexpr int WPL = 16 / VPW;
constexpr int KW = K / VPW;
constexpr int KG = K / GS;
constexpr int NSLOT = H / 16;
constexpr int SITER = (NSLOT + 31) / 32;
constexpr int PR = R + HC;
constexpr int NDG = R / ROWS_TG;
constexpr uint MASK = (1u << BITS) - 1u;
threadgroup T xs[H];
threadgroup float npart[8];
threadgroup float tgi[8 * HC];
const int base = s * H;
const size_t xrow = (size_t)row * K;
const bool down = g < NDG;

// The down weights do not wait for the norm: every load of this simdgroup's rows goes out first.
uint pw[RPS][SITER][WORD ? WPL : 1];
ulong pr[RPS][SITER][WORD ? 1 : WPL];
float ps[RPS][SITER];
float pb[RPS][SITER];
const int n0 = g * ROWS_TG + int(sg) * RPS;
if (down) {
  for (int r = 0; r < RPS; ++r) {
    for (int i = 0; i < SITER; ++i) {
      const int slot = i * 32 + int(lane);
      if (slot < NSLOT) {
        if constexpr (WORD) {
          const device uint32_t* wp = dw_q + (size_t)(n0 + r) * KW + (size_t)(s * (H / VPW) + slot * WPL);
          for (int e = 0; e < WPL; ++e) pw[r][i][e] = wp[e];
        } else {
          const device uchar* wb = (const device uchar*)dw_q + (size_t)(n0 + r) * (size_t)(K * BITS / 8) + (size_t)(base + slot * 16) * BITS / 8;
          for (int e = 0; e < WPL; ++e) pr[r][i][e] = hc_pack8<BITS>(wb + e * BITS);
        }
        const size_t gi = (size_t)(n0 + r) * KG + (size_t)((base + slot * 16) / GS);
        ps[r][i] = DENSE ? 1.0f : float(dw_s[DENSE ? 0 : gi]);
        pb[r][i] = DENSE ? 0.0f : float(dw_b[DENSE ? 0 : gi]);
      }
    }
  }
}

float wv[PER];
for (int i = 0; i < PER; ++i) wv[i] = float(nw[base + int(t) + 256 * i]);
float xv[PER];
T written[PER];
if (WR) {
  // Pending hcWrite: stream' = T(stream + T(out * inj)), the chain's two roundings.
  const float gate = float(wi_in[row * HC + s]);
  for (int i = 0; i < PER; ++i) {
    const int k = int(t) + 256 * i;
    const T v = T(float(x_in[xrow + base + k]) + float(T(float(wo_in[(size_t)row * H + k]) * gate)));
    written[i] = v;
    xv[i] = float(v);
  }
} else {
  for (int i = 0; i < PER; ++i) xv[i] = float(x_in[xrow + base + int(t) + 256 * i]);
}
float a = 0.0f;
for (int i = 0; i < PER; ++i) a += xv[i] * xv[i];
a = simd_sum(a);
if (lane == 0) npart[sg] = a;
threadgroup_barrier(mem_flags::mem_threadgroup);
float tot = 0.0f;
for (int q = 0; q < 8; ++q) tot += npart[q];
const float rsh = rsqrt(tot / float(H) + eps[0]);
T xnv[PER];
for (int i = 0; i < PER; ++i) {
  xnv[i] = T(float(T(xv[i] * rsh)) * wv[i]);
  xs[int(t) + 256 * i] = xnv[i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

device float* pp = parts_out + (size_t)(row * HC + s) * PR;
if (down) {
  float acc[RPS];
  for (int r = 0; r < RPS; ++r) acc[r] = 0.0f;
  for (int i = 0; i < SITER; ++i) {
    const int slot = i * 32 + int(lane);
    if (slot < NSLOT) {
      float xr[16];
      float xsum = 0.0f;
      for (int e = 0; e < 16; ++e) { xr[e] = float(xs[slot * 16 + e]); xsum += xr[e]; }
      for (int r = 0; r < RPS; ++r) {
        float d = 0.0f;
        for (int w = 0; w < WPL; ++w) {
          if constexpr (WORD) {
            for (int j = 0; j < VPW; ++j) d += xr[w * VPW + j] * hc_val<T, DENSE>((pw[r][i][w] >> (j * BITS)) & MASK);
          } else {
            for (int j = 0; j < VPW; ++j) d += xr[w * VPW + j] * float(uint(pr[r][i][w] >> (j * BITS)) & MASK);
          }
        }
        acc[r] += ps[r][i] * d + pb[r][i] * xsum;
      }
    }
  }
  for (int r = 0; r < RPS; ++r) {
    const float v = simd_sum(acc[r]);
    if (lane == 0) pp[n0 + r] = v;
  }
} else if (INJ != 0) {
  // This stream's partial of the inject matvec; the up launch adds the streams.
  float ip[HC];
  for (int c = 0; c < HC; ++c) ip[c] = 0.0f;
  for (int i = 0; i < PER; ++i) {
    const size_t k = (size_t)(base + int(t) + 256 * i);
    for (int c = 0; c < HC; ++c) ip[c] += float(xnv[i]) * float(iw[k * HC + c]);
  }
  for (int c = 0; c < HC; ++c) {
    const float pa = simd_sum(ip[c]);
    if (lane == 0) tgi[sg * HC + c] = pa;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < uint(HC)) {
    float tt = 0.0f;
    for (int q = 0; q < 8; ++q) tt += tgi[q * HC + t];
    pp[R + t] = tt;
  }
}
// Device stores come last: a later device load waits on every earlier store it cannot prove disjoint.
if (g == 0) {
  for (int i = 0; i < PER; ++i) {
    xn_out[xrow + base + int(t) + 256 * i] = xnv[i];
    if (WR) xs_out[xrow + base + int(t) + 256 * i] = written[i];
  }
}
}


kernel void official_hc_stock_nd_pending_inject(
    device const bfloat* x_in [[buffer(0)]],
    device const bfloat* nw [[buffer(1)]],
    device const bfloat* iw [[buffer(2)]],
    device const float* eps [[buffer(3)]],
    device const bfloat* wo_in [[buffer(4)]],
    device const bfloat* wi_in [[buffer(5)]],
    device const uint* dw_q [[buffer(6)]],
    device const bfloat* dw_s [[buffer(7)]],
    device const bfloat* dw_b [[buffer(8)]],
    device bfloat* xn_out [[buffer(9)]],
    device float* parts_out [[buffer(10)]],
    device bfloat* xs_out [[buffer(11)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
    using T = bfloat;
    constexpr int HC=4, H=2560, R=320, GS=64, BITS=8, INJ=1, WR=1, RPS=1, ROWS_TG=8;
const uint t = thread_index_in_threadgroup;
const int row = int(threadgroup_position_in_grid.z);
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int s = int(threadgroup_position_in_grid.y) % HC;
const int g = int(threadgroup_position_in_grid.y) / HC;
constexpr int PER = H / 256;
constexpr int K = HC * H;
// A slot's 16 values are WPL words at 2/4/8 bits, else two 8-value packs of BITS bytes.
constexpr bool WORD = (32 % BITS) == 0;
constexpr bool DENSE = BITS == 16;
constexpr int VPW = WORD ? 32 / BITS : 8;
constexpr int WPL = 16 / VPW;
constexpr int KW = K / VPW;
constexpr int KG = K / GS;
constexpr int NSLOT = H / 16;
constexpr int SITER = (NSLOT + 31) / 32;
constexpr int PR = R + HC;
constexpr int NDG = R / ROWS_TG;
constexpr uint MASK = (1u << BITS) - 1u;
threadgroup T xs[H];
threadgroup float npart[8];
threadgroup float tgi[8 * HC];
const int base = s * H;
const size_t xrow = (size_t)row * K;
const bool down = g < NDG;

// The down weights do not wait for the norm: every load of this simdgroup's rows goes out first.
uint pw[RPS][SITER][WORD ? WPL : 1];
ulong pr[RPS][SITER][WORD ? 1 : WPL];
float ps[RPS][SITER];
float pb[RPS][SITER];
const int n0 = g * ROWS_TG + int(sg) * RPS;
if (down) {
  for (int r = 0; r < RPS; ++r) {
    for (int i = 0; i < SITER; ++i) {
      const int slot = i * 32 + int(lane);
      if (slot < NSLOT) {
        if constexpr (WORD) {
          const device uint32_t* wp = dw_q + (size_t)(n0 + r) * KW + (size_t)(s * (H / VPW) + slot * WPL);
          for (int e = 0; e < WPL; ++e) pw[r][i][e] = wp[e];
        } else {
          const device uchar* wb = (const device uchar*)dw_q + (size_t)(n0 + r) * (size_t)(K * BITS / 8) + (size_t)(base + slot * 16) * BITS / 8;
          for (int e = 0; e < WPL; ++e) pr[r][i][e] = hc_pack8<BITS>(wb + e * BITS);
        }
        const size_t gi = (size_t)(n0 + r) * KG + (size_t)((base + slot * 16) / GS);
        ps[r][i] = DENSE ? 1.0f : float(dw_s[DENSE ? 0 : gi]);
        pb[r][i] = DENSE ? 0.0f : float(dw_b[DENSE ? 0 : gi]);
      }
    }
  }
}

float wv[PER];
for (int i = 0; i < PER; ++i) wv[i] = float(nw[base + int(t) + 256 * i]);
float xv[PER];
T written[PER];
if (WR) {
  // Pending hcWrite: stream' = T(stream + T(out * inj)), the chain's two roundings.
  const float gate = float(wi_in[row * HC + s]);
  for (int i = 0; i < PER; ++i) {
    const int k = int(t) + 256 * i;
    const T v = T(float(x_in[xrow + base + k]) + float(T(float(wo_in[(size_t)row * H + k]) * gate)));
    written[i] = v;
    xv[i] = float(v);
  }
} else {
  for (int i = 0; i < PER; ++i) xv[i] = float(x_in[xrow + base + int(t) + 256 * i]);
}
float a = 0.0f;
for (int i = 0; i < PER; ++i) a += xv[i] * xv[i];
a = simd_sum(a);
if (lane == 0) npart[sg] = a;
threadgroup_barrier(mem_flags::mem_threadgroup);
float tot = 0.0f;
for (int q = 0; q < 8; ++q) tot += npart[q];
const float rsh = rsqrt(tot / float(H) + eps[0]);
T xnv[PER];
for (int i = 0; i < PER; ++i) {
  xnv[i] = T(float(T(xv[i] * rsh)) * wv[i]);
  xs[int(t) + 256 * i] = xnv[i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

device float* pp = parts_out + (size_t)(row * HC + s) * PR;
if (down) {
  float acc[RPS];
  for (int r = 0; r < RPS; ++r) acc[r] = 0.0f;
  for (int i = 0; i < SITER; ++i) {
    const int slot = i * 32 + int(lane);
    if (slot < NSLOT) {
      float xr[16];
      float xsum = 0.0f;
      for (int e = 0; e < 16; ++e) { xr[e] = float(xs[slot * 16 + e]); xsum += xr[e]; }
      for (int r = 0; r < RPS; ++r) {
        float d = 0.0f;
        for (int w = 0; w < WPL; ++w) {
          if constexpr (WORD) {
            for (int j = 0; j < VPW; ++j) d += xr[w * VPW + j] * hc_val<T, DENSE>((pw[r][i][w] >> (j * BITS)) & MASK);
          } else {
            for (int j = 0; j < VPW; ++j) d += xr[w * VPW + j] * float(uint(pr[r][i][w] >> (j * BITS)) & MASK);
          }
        }
        acc[r] += ps[r][i] * d + pb[r][i] * xsum;
      }
    }
  }
  for (int r = 0; r < RPS; ++r) {
    const float v = simd_sum(acc[r]);
    if (lane == 0) pp[n0 + r] = v;
  }
} else if (INJ != 0) {
  // This stream's partial of the inject matvec; the up launch adds the streams.
  float ip[HC];
  for (int c = 0; c < HC; ++c) ip[c] = 0.0f;
  for (int i = 0; i < PER; ++i) {
    const size_t k = (size_t)(base + int(t) + 256 * i);
    for (int c = 0; c < HC; ++c) ip[c] += float(xnv[i]) * float(iw[k * HC + c]);
  }
  for (int c = 0; c < HC; ++c) {
    const float pa = simd_sum(ip[c]);
    if (lane == 0) tgi[sg * HC + c] = pa;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < uint(HC)) {
    float tt = 0.0f;
    for (int q = 0; q < 8; ++q) tt += tgi[q * HC + t];
    pp[R + t] = tt;
  }
}
// Device stores come last: a later device load waits on every earlier store it cannot prove disjoint.
if (g == 0) {
  for (int i = 0; i < PER; ++i) {
    xn_out[xrow + base + int(t) + 256 * i] = xnv[i];
    if (WR) xs_out[xrow + base + int(t) + 256 * i] = written[i];
  }
}
}

// Stark integration: same two BF16 stages as official_hc_write_gate_m1,
// with distinct source/destination for the attention-write/read pair.
kernel void official_hc_write_gate_to_m1(
    device const bfloat* residual [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    device const bfloat* branch [[buffer(2)]],
    device const bfloat* gate [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const uint g = gid / 2560;
    const uint i = gid % 2560;
    const bfloat delta = bfloat(float(branch[i]) * float(gate[g]));
    output[gid] = bfloat(float(residual[gid]) + float(delta));
}
