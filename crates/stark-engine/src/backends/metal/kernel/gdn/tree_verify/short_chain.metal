// Copyright (c) 2025 Mirai Tech Inc.
// SPDX-License-Identifier: MIT
//
// Stark adaptation of tree_gram.metal, tree_update_solve.metal, out.metal
// and state_advance.metal from the inherited Uzu-derived engine. The source
// operations and fragment layouts are retained; only dispatch/storage change.
// Baseline source revision: d5ec35466d6a287ed6f9614baed5eefbd637d902.
#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../../common/thread_context.h"
#include "../../matmul/common/fragment.h"
#include "../../matmul/common/mxu_fragment/ops.h"
#include "../../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;
using namespace stark::matmul;

// A threadgroup owns 32 independent value rows of one head. The complete
// 128-key state for those rows is local, so no inter-threadgroup barrier or
// speculative global-state write is needed. Committing an accepted prefix
// continues to use the existing StateAdvance kernel outside this verifier.
KERNEL(DeltaNetShortChainVerify)(
    const device bfloat* q,
    const device bfloat* k,
    const device bfloat* v,
    const device float* log_decay,
    const device float* beta,
    const device float* h0,
    device bfloat* output,
    constant const uint& tree_size,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    threadgroup float local_state[32 * 128],
    threadgroup float local_kh0[32],
    threadgroup float local_u[32],
    threadgroup float local_qkd[1],
    const ThreadContext thread_context,
    const uint hv_idx GROUPS(num_v_heads),
    const uint value_tile_idx GROUPS(4),
    const uint tid THREADS(256)
) {
  constexpr uint HEAD_DIM = 128;
  constexpr uint VALUE_TILE = 32;
  constexpr uint DV_PER_SIMDGROUP = 4;
  using GramOps = MxuFragmentOps<>;
  using GramAcc = Fragment<float, 1, 2, GramOps>;
  using GramLeft = OperandFragment<float, 1, 1, GramOps>;
  using GramRight = OperandFragment<float, 1, 2, GramOps, ReadTranspose>;
  using SolveOps = SimdgroupFragmentOps;
  using SolveAcc = Fragment<float, 2, 4, SolveOps>;
  using SolveInverse = OperandFragment<float, 2, 2, SolveOps>;
  using OutOps = MxuStrictFragmentOps;
  using OutAcc = Fragment<float, 1, 2, OutOps>;
  using OutLeft = OperandFragment<float, 1, 1, OutOps>;
  using OutRight = OperandFragment<float, 1, 2, OutOps>;

  const ushort lane = thread_context.simd_lane_id;
  const uint sg = thread_context.simdgroup_index;
  const uint value_base = value_tile_idx * VALUE_TILE;
  const uint dv_base = sg * DV_PER_SIMDGROUP;
  const uint dk_base = lane * 4;
  const uint hk_idx = hv_idx / (num_v_heads / num_k_heads);
  const uint key_dim = num_k_heads * HEAD_DIM;
  const uint value_dim = num_v_heads * HEAD_DIM;
  const uint local_offset = dv_base * HEAD_DIM + dk_base;
  const uint state_base = (hv_idx * HEAD_DIM + value_base) * HEAD_DIM;
  for (uint offset = tid * 4; offset < VALUE_TILE * HEAD_DIM; offset += 256 * 4) {
    *reinterpret_cast<threadgroup float4*>(local_state + offset) =
        *reinterpret_cast<const device float4*>(h0 + state_base + offset);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint row = 0; row < tree_size; ++row) {
    const uint head_offset = row * num_v_heads + hv_idx;
    const device bfloat* q_row = q + row * key_dim + hk_idx * HEAD_DIM;
    const device bfloat* k_row = k + row * key_dim + hk_idx * HEAD_DIM;
    const device bfloat* v_row = v + row * value_dim + hv_idx * HEAD_DIM + value_base;
    // Preserve BuildTreePrefix's T=1 zero initialization and SIMD reduction.
    float partial = 0.0f;
    if (lane == 0) {
      partial += log_decay[head_offset];
    }
    const float prefix = simd_sum(partial);

    if (sg == 0) {
      GramAcc qk_acc;
      GramAcc kh0_acc;
      qk_acc.clear();
      kh0_acc.clear();
      // Same relaxed MXU 16x32x16 accumulate sequence and padded T=1
      // operands as BuildTreeGram. Repeating qk for each value tile removes
      // a cross-threadgroup dependency without changing its reduction order.
      for (uint kb = 0; kb < HEAD_DIM; kb += GramOps::FRAGMENT_ROWS) {
        GramLeft q_frag;
        GramLeft k_frag;
        GramRight k_right;
        GramRight h_right;
        q_frag.load_from(lane, fragment_source(q_row + kb, int(key_dim)).bounded(1, 16));
        k_frag.load_from(lane, fragment_source(k_row + kb, int(key_dim)).bounded(1, 16));
        k_right.load_from(lane, fragment_source(k_row + kb, int(key_dim)).bounded(1, 16));
        h_right.load_from(lane, fragment_source(local_state + kb, int(HEAD_DIM)).bounded(32, 16));
        fragment_mma(qk_acc, q_frag, k_right);
        fragment_mma(kh0_acc, k_frag, h_right);
      }
      qk_acc.map_coords(lane, [&](short r, short c, float x) {
        return r == 0 && c == 0 ? 1.0f * exp(prefix - prefix) * x : 0.0f;
      });
      qk_acc.store_safe(lane, local_qkd, 1, short2(1, 1));
      kh0_acc.store_safe(lane, local_kh0, int(VALUE_TILE), short2(VALUE_TILE, 1));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
      // Retain the FP32 SIMDgroup identity solve, including the zero-padded
      // second 8-wide depth fragment. A scalar assignment is not substituted.
      SolveAcc acc;
      SolveAcc value_frag;
      acc.load_from(lane, fragment_source(local_kh0, int(VALUE_TILE)).bounded(1, VALUE_TILE));
      value_frag.load_from(lane, fragment_source(v_row, int(value_dim)).bounded(1, VALUE_TILE));
      const float lane_beta = lane == 0 ? beta[head_offset] : 0.0f;
      const float lane_decay = lane == 0 ? exp(prefix) : 0.0f;
      SolveAcc::zip_for_each_coord(
          lane,
          [&](ushort local_row, ushort, thread float& a, thread float& value) {
            const float b = simd_shuffle(lane_beta, local_row);
            const float d = simd_shuffle(lane_decay, local_row);
            a = b * (value - d * a);
          },
          acc,
          value_frag
      );
      SolveInverse inv;
      inv.clear();
      inv.map_coords(lane, [](short r, short c, float) { return r == 0 && c == 0 ? 1.0f : 0.0f; });
      SolveAcc solved;
      solved.clear();
      fragment_mma(solved, inv, acc);
      solved.store_safe(lane, local_u, int(VALUE_TILE), short2(VALUE_TILE, 1));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
      // Output uses strict MXU, unlike Gram. The pre-decay state and the
      // separately solved u must remain independent of StateAdvance's delta.
      OutAcc acc;
      acc.clear();
      for (uint kb = 0; kb < HEAD_DIM; kb += OutOps::FRAGMENT_ROWS) {
        OutLeft q_frag;
        OutRight h_frag;
        q_frag.load_from(lane, fragment_source(q_row + kb, int(key_dim)).bounded(1, 16));
        h_frag.load_from(lane, fragment_source(local_state + kb, 1, int(HEAD_DIM)).bounded(16, VALUE_TILE));
        fragment_mma(acc, q_frag, h_frag);
      }
      acc.map_coords(lane, [&](short r, short, float x) { return x * (r == 0 ? exp(prefix) * 1.0f : 0.0f); });
      OutLeft qkd_frag;
      OutRight u_frag;
      qkd_frag.load_from(lane, fragment_source(local_qkd, 1).bounded(1, 1));
      u_frag.load_from(lane, fragment_source(local_u, int(VALUE_TILE)).bounded(1, VALUE_TILE));
      fragment_mma(acc, qkd_frag, u_frag);
      acc.store_safe(
          lane,
          output + row * value_dim + hv_idx * HEAD_DIM + value_base,
          int(value_dim),
          short2(VALUE_TILE, 1)
      );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row + 1 < tree_size) {
      // Verbatim StateAdvance lane grouping and operation order: decay the
      // state before float4 dot + simd_sum, then form a separate delta. Its
      // fast::exp and retrieval must not reuse the output solve residual.
      // Keep state in threadgroup memory across phases, so its float4
      // register lifetime does not overlap the three MMA fragment layouts.
      float4 state_rows[DV_PER_SIMDGROUP];
      state_rows[0] = *reinterpret_cast<const threadgroup float4*>(local_state + local_offset + 0 * HEAD_DIM);
      state_rows[1] = *reinterpret_cast<const threadgroup float4*>(local_state + local_offset + 1 * HEAD_DIM);
      state_rows[2] = *reinterpret_cast<const threadgroup float4*>(local_state + local_offset + 2 * HEAD_DIM);
      state_rows[3] = *reinterpret_cast<const threadgroup float4*>(local_state + local_offset + 3 * HEAD_DIM);
      const float decay = fast::exp(log_decay[head_offset]);
      const float b = beta[head_offset];
      const float4 key = float4(*reinterpret_cast<const device vec<bfloat, 4>*>(k_row + dk_base));
      state_rows[0] *= decay;
      state_rows[1] *= decay;
      state_rows[2] *= decay;
      state_rows[3] *= decay;
      float4 kv_mem = float4(
          dot(state_rows[0], key), dot(state_rows[1], key), dot(state_rows[2], key), dot(state_rows[3], key)
      );
      kv_mem = simd_sum(kv_mem);
      const device bfloat* values = v_row + dv_base;
      const float4 v_value = float4(float(values[0]), float(values[1]), float(values[2]), float(values[3]));
      const float4 delta = b * (v_value - kv_mem);
      state_rows[0] += key * delta[0];
      state_rows[1] += key * delta[1];
      state_rows[2] += key * delta[2];
      state_rows[3] += key * delta[3];
      *reinterpret_cast<threadgroup float4*>(local_state + local_offset + 0 * HEAD_DIM) = state_rows[0];
      *reinterpret_cast<threadgroup float4*>(local_state + local_offset + 1 * HEAD_DIM) = state_rows[1];
      *reinterpret_cast<threadgroup float4*>(local_state + local_offset + 2 * HEAD_DIM) = state_rows[2];
      *reinterpret_cast<threadgroup float4*>(local_state + local_offset + 3 * HEAD_DIM) = state_rows[3];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
