// SPDX-License-Identifier: MIT
// Copyright (c) 2026 David Dalcu
// Ported by Stark from MLX-Serve 046b647686419df06c6c7132fb1d15d76185c3e5,
// src/hc_decode2.zig ND_SOURCE and U2V_SOURCE. See LICENSE.mlx-serve.
// The original comments and arithmetic bodies are preserved verbatim.
// This private port is uncompiled and not wired into the model graph.
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

kernel void official_hc_stock_nd_noinject(
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
    constexpr int HC=4, H=2560, R=320, GS=64, BITS=8, INJ=0, WR=0, RPS=1, ROWS_TG=8;
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

kernel void official_hc_stock_u2v_noinject(
    device const bfloat* xn_in [[buffer(0)]],
    device const float* parts_in [[buffer(1)]],
    device const uint* uw_q [[buffer(2)]],
    device const bfloat* uw_s [[buffer(3)]],
    device const bfloat* uw_b [[buffer(4)]],
    device bfloat* mixed_out [[buffer(5)]],
    device bfloat* inj_out [[buffer(6)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
    using T = bfloat;
    constexpr int HC=4, H=2560, R=320, GS=64, BITS=8, INJ=0, ROWS=1, SGS=16;
const uint lane = thread_index_in_simdgroup;
const uint tid = thread_index_in_threadgroup;
const uint sgi = simdgroup_index_in_threadgroup;
const uint part = lane & 3u;
const uint h = (lane >> 2) & 3u;
const uint cl = lane >> 4;
const uint j = threadgroup_position_in_grid.x * uint(SGS * 2) + sgi * 2u + cl;
constexpr int VPW = 32 / BITS;
constexpr int R_by_p = R / VPW;
constexpr int R_by_gs = R / GS;
constexpr int PV = R / 4;
constexpr int PU4 = PV / (4 * VPW);
constexpr int PR = R + HC;
threadgroup float act_tg[ROWS * R];
const size_t wrow = (size_t)h * (size_t)H + (size_t)j;
const device uint4* wq4 = (const device uint4*)(uw_q + wrow * (size_t)R_by_p) + part * PU4;
uint4 wv[PU4];
for (int i = 0; i < PU4; ++i) wv[i] = wq4[i];
float sc[PU4], bi[PU4];
for (int i = 0; i < PU4; ++i) {
  const int k0 = int(part) * PV + i * 4 * VPW;
  sc[i] = float(uw_s[wrow * (size_t)R_by_gs + (size_t)(k0 / GS)]);
  bi[i] = float(uw_b[wrow * (size_t)R_by_gs + (size_t)(k0 / GS)]);
}
float xnv[ROWS];
for (int r = 0; r < ROWS; ++r) xnv[r] = float(xn_in[(size_t)r * (size_t)(HC * H) + wrow]);
for (uint e = tid; e < uint(ROWS * R); e += uint(SGS * 32)) {
  const int r = int(e) / R;
  const int k = int(e) % R;
  float tsum = 0.0f;
  for (int s = 0; s < HC; ++s) tsum += parts_in[(size_t)(r * HC + s) * PR + k];
  const T v = T(tsum);
  const T sig = T(1.0f / (1.0f + metal::exp(-float(v))));
  act_tg[e] = float(v * sig);
}
if (INJ != 0 && threadgroup_position_in_grid.x == 0 && tid < uint(ROWS * HC)) {
  const int r = int(tid) / HC;
  const int c = int(tid) % HC;
  float tsum = 0.0f;
  for (int s = 0; s < HC; ++s) tsum += parts_in[(size_t)(r * HC + s) * PR + R + c];
  const T v = T(tsum);
  const T sig = T(1.0f / (1.0f + metal::exp(-float(v))));
  inj_out[tid] = sig * T(2.0f);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float acc[ROWS];
for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
for (int i = 0; i < PU4; ++i) {
  const int k0 = int(part) * PV + i * 4 * VPW;
  const uint words[4] = {wv[i].x, wv[i].y, wv[i].z, wv[i].w};
  for (int wd = 0; wd < 4; ++wd) {
    for (int b = 0; b < VPW; ++b) {
      const float wf = float((words[wd] >> (b * BITS)) & ((1u << BITS) - 1u)) * sc[i] + bi[i];
      const int k = k0 + wd * VPW + b;
      for (int r = 0; r < ROWS; ++r) acc[r] += act_tg[r * R + k] * wf;
    }
  }
}
float mix[ROWS];
for (int r = 0; r < ROWS; ++r) {
  float a = acc[r];
  a += simd_shuffle_xor(a, 1);
  a += simd_shuffle_xor(a, 2);
  const T u = T(a);
  const T sgv = T(1.0f / (1.0f + metal::exp(-float(u))));
  float m = float(T(float(sgv) * xnv[r]));
  m += simd_shuffle_xor(m, 4);
  m += simd_shuffle_xor(m, 8);
  mix[r] = m;
}
if ((lane & 15u) == 0) {
  for (int r = 0; r < ROWS; ++r) mixed_out[(size_t)r * (size_t)H + (size_t)j] = T(float(T(mix[r])) * float(T(1.0f / float(HC))));
}
}

kernel void official_hc_stock_nd_inject(
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
    constexpr int HC=4, H=2560, R=320, GS=64, BITS=8, INJ=1, WR=0, RPS=1, ROWS_TG=8;
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

kernel void official_hc_stock_u2v_inject(
    device const bfloat* xn_in [[buffer(0)]],
    device const float* parts_in [[buffer(1)]],
    device const uint* uw_q [[buffer(2)]],
    device const bfloat* uw_s [[buffer(3)]],
    device const bfloat* uw_b [[buffer(4)]],
    device bfloat* mixed_out [[buffer(5)]],
    device bfloat* inj_out [[buffer(6)]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]]) {
    using T = bfloat;
    constexpr int HC=4, H=2560, R=320, GS=64, BITS=8, INJ=1, ROWS=1, SGS=16;
const uint lane = thread_index_in_simdgroup;
const uint tid = thread_index_in_threadgroup;
const uint sgi = simdgroup_index_in_threadgroup;
const uint part = lane & 3u;
const uint h = (lane >> 2) & 3u;
const uint cl = lane >> 4;
const uint j = threadgroup_position_in_grid.x * uint(SGS * 2) + sgi * 2u + cl;
constexpr int VPW = 32 / BITS;
constexpr int R_by_p = R / VPW;
constexpr int R_by_gs = R / GS;
constexpr int PV = R / 4;
constexpr int PU4 = PV / (4 * VPW);
constexpr int PR = R + HC;
threadgroup float act_tg[ROWS * R];
const size_t wrow = (size_t)h * (size_t)H + (size_t)j;
const device uint4* wq4 = (const device uint4*)(uw_q + wrow * (size_t)R_by_p) + part * PU4;
uint4 wv[PU4];
for (int i = 0; i < PU4; ++i) wv[i] = wq4[i];
float sc[PU4], bi[PU4];
for (int i = 0; i < PU4; ++i) {
  const int k0 = int(part) * PV + i * 4 * VPW;
  sc[i] = float(uw_s[wrow * (size_t)R_by_gs + (size_t)(k0 / GS)]);
  bi[i] = float(uw_b[wrow * (size_t)R_by_gs + (size_t)(k0 / GS)]);
}
float xnv[ROWS];
for (int r = 0; r < ROWS; ++r) xnv[r] = float(xn_in[(size_t)r * (size_t)(HC * H) + wrow]);
for (uint e = tid; e < uint(ROWS * R); e += uint(SGS * 32)) {
  const int r = int(e) / R;
  const int k = int(e) % R;
  float tsum = 0.0f;
  for (int s = 0; s < HC; ++s) tsum += parts_in[(size_t)(r * HC + s) * PR + k];
  const T v = T(tsum);
  const T sig = T(1.0f / (1.0f + metal::exp(-float(v))));
  act_tg[e] = float(v * sig);
}
if (INJ != 0 && threadgroup_position_in_grid.x == 0 && tid < uint(ROWS * HC)) {
  const int r = int(tid) / HC;
  const int c = int(tid) % HC;
  float tsum = 0.0f;
  for (int s = 0; s < HC; ++s) tsum += parts_in[(size_t)(r * HC + s) * PR + R + c];
  const T v = T(tsum);
  const T sig = T(1.0f / (1.0f + metal::exp(-float(v))));
  inj_out[tid] = sig * T(2.0f);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float acc[ROWS];
for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
for (int i = 0; i < PU4; ++i) {
  const int k0 = int(part) * PV + i * 4 * VPW;
  const uint words[4] = {wv[i].x, wv[i].y, wv[i].z, wv[i].w};
  for (int wd = 0; wd < 4; ++wd) {
    for (int b = 0; b < VPW; ++b) {
      const float wf = float((words[wd] >> (b * BITS)) & ((1u << BITS) - 1u)) * sc[i] + bi[i];
      const int k = k0 + wd * VPW + b;
      for (int r = 0; r < ROWS; ++r) acc[r] += act_tg[r * R + k] * wf;
    }
  }
}
float mix[ROWS];
for (int r = 0; r < ROWS; ++r) {
  float a = acc[r];
  a += simd_shuffle_xor(a, 1);
  a += simd_shuffle_xor(a, 2);
  const T u = T(a);
  const T sgv = T(1.0f / (1.0f + metal::exp(-float(u))));
  float m = float(T(float(sgv) * xnv[r]));
  m += simd_shuffle_xor(m, 4);
  m += simd_shuffle_xor(m, 8);
  mix[r] = m;
}
if ((lane & 15u) == 0) {
  for (int r = 0; r < ROWS; ++r) mixed_out[(size_t)r * (size_t)H + (size_t)j] = T(float(T(mix[r])) * float(T(1.0f / float(HC))));
}
}


// Stark integration: U2V already returns the BF16 2*sigmoid gate.
// The write preserves the reference's BF16 product and residual-add stages.
kernel void official_hc_write_gate_m1(
    device bfloat* hyper [[buffer(0)]],
    device const bfloat* branch [[buffer(1)]],
    device const bfloat* gate [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const uint g = gid / 2560;
    const uint i = gid % 2560;
    const bfloat delta = bfloat(float(branch[i]) * float(gate[g]));
    hyper[gid] = bfloat(float(hyper[gid]) + float(delta));
}
