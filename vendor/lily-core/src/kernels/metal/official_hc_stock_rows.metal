// SPDX-License-Identifier: MIT
// Copyright (c) 2026 David Dalcu
// Adapted by Stark from the pinned MLX-Serve U2V_SOURCE port in
// official_hc_stock.metal. The verifier row count is a compile-time constant;
// each row retains the one-row arithmetic and shares loaded up weights.
// Source revision: 046b647686419df06c6c7132fb1d15d76185c3e5.
// See NOTICE and LICENSE.mlx-serve.

template <int Rows, bool WithInject>
inline void official_hc_stock_u2v_exact_rows_body(

    device const bfloat* xn_in,
    device const float* parts_in,
    device const uint* uw_q,
    device const bfloat* uw_s,
    device const bfloat* uw_b,
    device bfloat* mixed_out,
    device bfloat* inj_out,
    threadgroup float* act_tg,
    uint thread_index_in_threadgroup,
    uint3 threadgroup_position_in_grid,
    uint thread_index_in_simdgroup,
    uint simdgroup_index_in_threadgroup) {
    using T = bfloat;
    constexpr int HC=4, H=2560, R=320, GS=64, BITS=8, INJ=WithInject, ROWS=Rows, SGS=16;
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

kernel void official_hc_stock_u2v_rows2_noinject(
    device const bfloat* xn_in [[buffer(0)]],
    device const float* parts_in [[buffer(1)]],
    device const uint* uw_q [[buffer(2)]],
    device const bfloat* uw_s [[buffer(3)]],
    device const bfloat* uw_b [[buffer(4)]],
    device bfloat* mixed_out [[buffer(5)]],
    device bfloat* inj_out [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float act_tg[2 * 320];
    official_hc_stock_u2v_exact_rows_body<2, false>(
        xn_in, parts_in, uw_q, uw_s, uw_b, mixed_out, inj_out, act_tg,
        tid, group, lane, sg);
}

kernel void official_hc_stock_u2v_rows2_inject(
    device const bfloat* xn_in [[buffer(0)]],
    device const float* parts_in [[buffer(1)]],
    device const uint* uw_q [[buffer(2)]],
    device const bfloat* uw_s [[buffer(3)]],
    device const bfloat* uw_b [[buffer(4)]],
    device bfloat* mixed_out [[buffer(5)]],
    device bfloat* inj_out [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float act_tg[2 * 320];
    official_hc_stock_u2v_exact_rows_body<2, true>(
        xn_in, parts_in, uw_q, uw_s, uw_b, mixed_out, inj_out, act_tg,
        tid, group, lane, sg);
}

kernel void official_hc_stock_u2v_rows3_noinject(
    device const bfloat* xn_in [[buffer(0)]],
    device const float* parts_in [[buffer(1)]],
    device const uint* uw_q [[buffer(2)]],
    device const bfloat* uw_s [[buffer(3)]],
    device const bfloat* uw_b [[buffer(4)]],
    device bfloat* mixed_out [[buffer(5)]],
    device bfloat* inj_out [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float act_tg[3 * 320];
    official_hc_stock_u2v_exact_rows_body<3, false>(
        xn_in, parts_in, uw_q, uw_s, uw_b, mixed_out, inj_out, act_tg,
        tid, group, lane, sg);
}

kernel void official_hc_stock_u2v_rows3_inject(
    device const bfloat* xn_in [[buffer(0)]],
    device const float* parts_in [[buffer(1)]],
    device const uint* uw_q [[buffer(2)]],
    device const bfloat* uw_s [[buffer(3)]],
    device const bfloat* uw_b [[buffer(4)]],
    device bfloat* mixed_out [[buffer(5)]],
    device bfloat* inj_out [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float act_tg[3 * 320];
    official_hc_stock_u2v_exact_rows_body<3, true>(
        xn_in, parts_in, uw_q, uw_s, uw_b, mixed_out, inj_out, act_tg,
        tid, group, lane, sg);
}

kernel void official_hc_stock_u2v_rows4_noinject(
    device const bfloat* xn_in [[buffer(0)]],
    device const float* parts_in [[buffer(1)]],
    device const uint* uw_q [[buffer(2)]],
    device const bfloat* uw_s [[buffer(3)]],
    device const bfloat* uw_b [[buffer(4)]],
    device bfloat* mixed_out [[buffer(5)]],
    device bfloat* inj_out [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float act_tg[4 * 320];
    official_hc_stock_u2v_exact_rows_body<4, false>(
        xn_in, parts_in, uw_q, uw_s, uw_b, mixed_out, inj_out, act_tg,
        tid, group, lane, sg);
}

kernel void official_hc_stock_u2v_rows4_inject(
    device const bfloat* xn_in [[buffer(0)]],
    device const float* parts_in [[buffer(1)]],
    device const uint* uw_q [[buffer(2)]],
    device const bfloat* uw_s [[buffer(3)]],
    device const bfloat* uw_b [[buffer(4)]],
    device bfloat* mixed_out [[buffer(5)]],
    device bfloat* inj_out [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float act_tg[4 * 320];
    official_hc_stock_u2v_exact_rows_body<4, true>(
        xn_in, parts_in, uw_q, uw_s, uw_b, mixed_out, inj_out, act_tg,
        tid, group, lane, sg);
}

// U2V provides the finished BF16 gate. Keep the same rounded product
// and rounded residual addition as the ordinary one-row gate write.
kernel void official_hc_write_gate_exact_rows(
    device bfloat* hyper [[buffer(0)]],
    device const bfloat* branch [[buffer(1)]],
    device const bfloat* gate [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
    const uint g = gid.x / 2560;
    const uint i = gid.x % 2560;
    const bfloat delta = bfloat(float(branch[gid.y * 2560 + i]) * float(gate[gid.y * 4 + g]));
    const uint at = gid.y * 10240 + gid.x;
    hyper[at] = bfloat(float(hyper[at]) + float(delta));
}
