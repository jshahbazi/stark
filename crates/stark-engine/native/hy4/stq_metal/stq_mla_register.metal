// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Theinruj Toranavikrai
// Copyright (c) 2026 David Dalcu
// Register-fragment MLA schedule derived from Sushi v1.8.1 at
// b7b7975cc017acba1b4c0877d65543c4c5a4a381, src/hy4flash/hy4_attn.zig
// and src/metal_def.zig. See LICENSE. Stark reads its interleaved FP16 KV
// plane directly and scales the split FP16 query by an exact power of two.
#include <metal_stdlib>
#include <metal_tensor>
using namespace metal;

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
inline short stark_stq_mla_r(ushort lane, int i) { const short qid = short(lane) >> 2; return ((qid & 4) | ((short(lane) >> 1) & 3)) + (i >> 2) * 8; }
inline short stark_stq_mla_c(ushort lane, int i) { const short qid = short(lane) >> 2; return ((qid & 2) | (short(lane) & 1)) * 4 + (i & 3); }
constant constexpr int stark_stq_mla_partial_width = 514;
template <int N, int KK, typename AT, typename BT>
inline void stark_stq_mla_mma(thread float* c, thread const AT* a, thread const BT* b) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, N, KK, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ca = op.template get_left_input_cooperative_tensor<AT, BT, float>();
  auto cb = op.template get_right_input_cooperative_tensor<AT, BT, float>();
  auto cc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, float>();
  #pragma clang loop unroll(full)
  for (int i = 0; i < KK / 2; ++i) ca[i] = a[i];
  #pragma clang loop unroll(full)
  for (int i = 0; i < KK * N / 32; ++i) cb[i] = b[i];
  #pragma clang loop unroll(full)
  for (int i = 0; i < N / 2; ++i) cc[i] = c[i];
  op.run(ca, cb, cc);
  #pragma clang loop unroll(full)
  for (int i = 0; i < N / 2; ++i) c[i] = cc[i];
}


kernel void stark_hy4_stq_register_mla(
    device const float* query [[buffer(0)]], device const half* cache [[buffer(1)]],
    device const float* sink [[buffer(2)]], device const int* picks [[buffer(3)]],
    device const int* n_sel [[buffer(4)]], device const int* positions [[buffer(5)]],
    device float* output [[buffer(6)]], constant Hy4FlashAttnArgs& a [[buffer(7)]],
    constant uint& partitions [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane_id [[thread_index_in_simdgroup]]) {
    constexpr int HB = 16, KT = 32, L = 512, NSG = 9;
    const ushort lane = ushort(lane_id);
    const uint l = group.x;
    const int h0 = int(group.y) * HB;
    const int H = int(a.n_head);
    const int position = positions[l];
    const float scale = a.kq_scale * 16.0f;
    constexpr int NL = L / 64, QS = 2, NK = KT / 16, NJ = KT / 16;
    const bool rsg = int(sg) == NL, work = int(sg) < NSG;
    const int d0 = rsg ? 0 : int(sg) * 64;
    threadgroup float sp[NSG * HB * KT];
    threadgroup half ph[HB * KT], pl[HB * KT];
    threadgroup float cr[HB], rl[HB];
    threadgroup int rsrc[2][KT];
    const int hh = int(tid) / 16, sub = int(tid) % 16;
    const bool soft = int(tid) < HB * 16;
    float m = soft && group.z == 0 ? sink[metal::min(h0 + hh, H - 1)] : -INFINITY;
    float lsum = group.z == 0 ? 1.0f : 0.0f;
    half qh[QS][16], qm[QS][16], qo[QS][16];
#pragma clang loop unroll(full)
    for (int ks = 0; ks < QS; ++ks)
      #pragma clang loop unroll(full)
      for (int i = 0; i < 16; ++i) {
        const int ch = ks * 32 + (i / 8) * 16 + stark_stq_mla_r(lane, i % 8);
        const int head = h0 + stark_stq_mla_c(lane, i % 8);
        float v = 0.0f;
        if (head < H && work) {
          v = query[ulong(l) * a.q_stride + uint(head) * 576u + (rsg ? L + ch : d0 + ch)] * 0.0625f;
        }
        const half hi = half(v);
        const float r1 = v - float(hi);
        const half md = half(r1);
        qh[ks][i] = hi;
        qm[ks][i] = md;
        qo[ks][i] = half(r1 - float(md));
      }
    float o[32];
#pragma clang loop unroll(full)
    for (int i = 0; i < 32; ++i) o[i] = 0.0f;
    const int start = int(group.z) * 256;
    const int lo = hy4_flash_attn_sel || a.window == 0 ? 0 : max(0, position - int(a.window) + 1);
    const int count = hy4_flash_attn_sel ? clamp(n_sel[l], 0, int(a.sel_stride)) : position - lo + 1;
    const int remaining = max(0, count - start);
    const int total = partitions != 0 ? min(256, remaining) : remaining;
    device const int* selected = picks + ulong(l) * a.sel_stride;
#define LOAD_SRC(dst, t) { \
    const int key = (t) < total ? (hy4_flash_attn_sel ? selected[start + (t)] : lo + start + (t)) : -1; \
    dst = key >= 0 && key <= position ? key : -1; \
}
#define TSRC(j) rsrc[bb][(j)]

    const short r0 = stark_stq_mla_r(lane, 0), c0f = stark_stq_mla_c(lane, 0);
    int bb = 0;
    if (tid < KT) LOAD_SRC(rsrc[0][tid], int(tid))
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int t0 = 0; t0 < total; t0 += KT, bb ^= 1) {
      float s[NK][8];
      half kf[NK][QS][16];
      if (work) {
      #pragma clang loop unroll(full)
      for (int kt = 0; kt < NK; ++kt) {
        const int ra = TSRC(kt * 16 + r0), rb = TSRC(kt * 16 + r0 + 8);
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) s[kt][i] = 0.0f;
        #pragma clang loop unroll(full)
        for (int ks = 0; ks < QS; ++ks) {
          #pragma clang loop unroll(full)
          for (int g = 0; g < 4; ++g) {
            const int src = (g & 1) ? rb : ra;
            const int ch0 = ks * 32 + (g >> 1) * 16 + c0f;
            half v[4] = {half(0.0f), half(0.0f), half(0.0f), half(0.0f)};
            if (src >= 0) {
              const int d = (rsg ? L : d0) + ch0;
              const half4 values = *(device const half4*)(cache + ulong(src) * a.kv_stride + d);
              for (int e = 0; e < 4; ++e) v[e] = values[e];
            }
            #pragma clang loop unroll(full)
            for (int e = 0; e < 4; ++e) kf[kt][ks][g * 4 + e] = v[e];
          }
          stark_stq_mla_mma<16, 32>(s[kt], kf[kt][ks], qo[ks]);
          stark_stq_mla_mma<16, 32>(s[kt], kf[kt][ks], qm[ks]);
          stark_stq_mla_mma<16, 32>(s[kt], kf[kt][ks], qh[ks]);
        }
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) sp[(int(sg) * HB + stark_stq_mla_c(lane, i)) * KT + kt * 16 + stark_stq_mla_r(lane, i)] = s[kt][i];
      }
      }
      threadgroup_barrier(metal::mem_flags::mem_threadgroup);
      if (soft) {
        float sv[NJ], tmax = -INFINITY;
        #pragma clang loop unroll(full)
        for (int jj = 0; jj < NJ; ++jj) {
          const int j = sub + jj * 16;
          float v = -INFINITY;
          if (TSRC(j) != -1) {
            float a = 0.0f;
            for (int g = 0; g < NSG; ++g) a += sp[(g * HB + hh) * KT + j];
            v = a * scale;
          }
          sv[jj] = v;
          tmax = metal::max(tmax, v);
        }
        if (tid < KT) LOAD_SRC(rsrc[bb ^ 1][tid], t0 + KT + int(tid))
        for (int off = 8; off > 0; off >>= 1) tmax = metal::max(tmax, metal::simd_shuffle_xor(tmax, ushort(off)));
        const float mn = metal::max(m, tmax);
        const float corr = lsum > 0.0f ? metal::precise::exp(m - mn) : 0.0f;
        float psum = 0.0f;
        #pragma clang loop unroll(full)
        for (int jj = 0; jj < NJ; ++jj) {
          const int j = sub + jj * 16;
          const float p = TSRC(j) >= 0 ? metal::precise::exp(sv[jj] - mn) : 0.0f;
          const half hi = half(p);
          psum += p;
          ph[hh * KT + j] = hi;
          pl[hh * KT + j] = half(p - float(hi));
        }
        for (int off = 8; off > 0; off >>= 1) psum += metal::simd_shuffle_xor(psum, ushort(off));
        lsum = lsum * corr + psum;
        m = mn;
        if (sub == 0) cr[hh] = corr;
      }
      threadgroup_barrier(metal::mem_flags::mem_threadgroup);
      if (work && !rsg) {
        #pragma clang loop unroll(full)
        for (int i = 0; i < 32; ++i) o[i] *= cr[stark_stq_mla_r(lane, i % 8)];
        #pragma clang loop unroll(full)
        for (int kt = 0; kt < NK; ++kt) {
          half ah[8], al[8];
          #pragma clang loop unroll(full)
          for (int i = 0; i < 8; ++i) { ah[i] = ph[stark_stq_mla_r(lane, i) * KT + kt * 16 + stark_stq_mla_c(lane, i)]; al[i] = pl[stark_stq_mla_r(lane, i) * KT + kt * 16 + stark_stq_mla_c(lane, i)]; }
          #pragma clang loop unroll(full)
          for (int g = 0; g < QS; ++g) {
            stark_stq_mla_mma<32, 16>(o + g * 16, al, kf[kt][g]);
            stark_stq_mla_mma<32, 16>(o + g * 16, ah, kf[kt][g]);
          }
        }
      }
    }
    if (soft && sub == 0) {
      const int head = h0 + hh;
      if (partitions != 0 && head < H) {
        device float* partial = output + ((ulong(l) * H + head) * partitions + group.z) * stark_stq_mla_partial_width;
        partial[L] = m;
        partial[L + 1] = lsum;
      }
      rl[hh] = partitions == 0 ? 1.0f / lsum : 1.0f;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (work && !rsg) {
      #pragma clang loop unroll(full)
      for (int hs = 0; hs < 2; ++hs) {
        const int row = stark_stq_mla_r(lane, hs * 4);
        const float inv = rl[row];
        const int head = h0 + row;
        if (head < H) {
          const ulong row = partitions == 0 ? ulong(l) * a.o_stride + head * L
            : ((ulong(l) * H + head) * partitions + group.z) * stark_stq_mla_partial_width;
          device float* dst = output + row + d0 + c0f;
          #pragma clang loop unroll(full)
          for (int f = 0; f < 2 * QS; ++f)
            #pragma clang loop unroll(full)
            for (int j = 0; j < 4; ++j) dst[f * 16 + j] = o[f * 8 + hs * 4 + j] * inv;
        }
      }
    }
#undef LOAD_SRC
#undef TSRC
}

// Merge independent key runs. Only the first run includes the sink.
kernel void stark_hy4_stq_register_mla_finish(
    device const float* partials [[buffer(0)]], device float* output [[buffer(1)]],
    constant Hy4FlashAttnArgs& args [[buffer(2)]], constant uint& partitions [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constexpr uint L = 512;
    device const float* row = partials + (ulong(group.x) * args.n_head + group.y) * partitions * stark_stq_mla_partial_width;
    threadgroup float weights[16];
    threadgroup float inv;
    if (tid == 0) {
        float maximum = -INFINITY;
        for (uint p = 0; p < partitions; ++p) maximum = max(maximum, row[p * stark_stq_mla_partial_width + L]);
        float sum = 0.0f;
        for (uint p = 0; p < partitions; ++p) {
            const uint stat = p * stark_stq_mla_partial_width + L;
            weights[p] = precise::exp(row[stat] - maximum);
            sum += weights[p] * row[stat + 1];
        }
        inv = 1.0f / sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* dst = output + ulong(group.x) * args.o_stride + group.y * L;
    for (uint d = tid; d < L; d += 256) {
        float value = 0.0f;
        for (uint p = 0; p < partitions; ++p) value += weights[p] * row[p * stark_stq_mla_partial_width + d];
        dst[d] = value * inv;
    }
}
