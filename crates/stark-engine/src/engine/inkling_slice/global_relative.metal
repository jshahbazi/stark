// SPDX-License-Identifier: MIT
// Mask-free Inkling global attention. Derived from `attention_nax` in
// mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h at MLX
// fc344c986c4169d99f76b5d95ae6b4a58a359a63 (Copyright 2024-26 Apple Inc., MIT;
// see metal_reference/LICENSE), replacing the explicit F32 mask
// tensor with the same F32 value computed in the kernel from the BF16
// relative table and sink, exactly as `ink_global_probe_mask` writes it.
// Every other operation is the upstream kernel's.
// Included at the end of metal_reference/attention.metal, which provides
// metal_stdlib, the MLX utilities and the pinned attention header (with its
// function constants), so both kernels share one translation unit.

// relative = (queries, keys, query offset, unused). The value of element
// (row, key) matches the probe mask, and elements outside qL x kL are zero
// as in BaseNAXFrag::load_safe.
METAL_FUNC void ink_relative_bias(
    thread metal::vec<float, 8>& dst,
    const device bfloat* table,
    const device bfloat* sink,
    uint4 relative,
    uint head,
    int qL,
    int kL,
    int row_pos,
    int col_pos) {
  constexpr uint query_heads = 24, distances = 1024;
  const short2 sc = BaseNAXFrag::get_coord();
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < BaseNAXFrag::kElemRows; i++) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < BaseNAXFrag::kElemCols; j++) {
      const int row = row_pos + sc.y + i * BaseNAXFrag::kElemRowsJump;
      const int key = col_pos + sc.x + j;
      float value = 0.0f;
      if (row < qL && key < kL) {
        const int distance = int(relative.y - relative.x) + row - key;
        const uint query_head = (relative.z + uint(row)) * query_heads + head;
        const float bias = distance >= 0 && distance < int(distances)
            ? float(table[query_head * distances + distance])
            : 0.0f;
        value = distance < 0 ? -INFINITY : bias - float(sink[query_head]);
      }
      dst[i * BaseNAXFrag::kElemCols + j] = value;
    }
  }
}

// clang-format off
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int BV,
    int WM,
    int WN,
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void ink_global_relative_attention(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant uint4& relative [[buffer(5)]],
    const device bfloat* table [[buffer(6)]],
    const device T* sinks [[buffer(7), function_constant(has_sinks)]],
    device float* lse [[buffer(8), function_constant(save_lse)]],
    const device bfloat* relative_sink [[buffer(9)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;
  (void)simd_lane_id;

  // Move to correct block
  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      tidl.x * BQ * params->Q_strides[2]; // Sequence

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + // Batch
      kv_head_idx * params->K_strides[1]; // Head

  V += tidl.z * params->V_strides[0] + // Batch
      kv_head_idx * params->V_strides[1]; // Head

  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      tidl.x * BQ * params->O_strides[2]; // Sequence

  if (save_lse) {
    // [B, H, qL], contiguous.
    lse += tidl.z * params->H * params->qL + // Batch
        tidl.y * params->qL + // Head
        tidl.x * BQ; // Sequence
  }


  const metal::uniform<float> scale2 =
      make_uniform(params->scale) * make_uniform(1.44269504089f);

  // Prepare MMA tiles
  constexpr short kU = 16;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kU);
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kU;
  constexpr int TV = BV / kU;
  static_assert(BD % kU == 0 && BV % (2 * kU) == 0, "Invalid head dimensions");
  // KV seq frags per warp
  constexpr short TK = BK / kU;

  static_assert(TQ == 1, "Check TQ");
  using otile_t = NAXTile<AccumType, TQ, TV>;
  otile_t Otile;

  Otile.clear();

  // Prepare mma tile offsets
  const short tm = kU * TQ * simd_group_id;
  Q += tm * int(params->Q_strides[2]);

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  // Init row reduction variables
  constexpr short kRowsPT = otile_t::kRowsPerThread;

  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

  // Init to -Inf
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  if (has_sinks) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = M_LOG2E_F * static_cast<AccumType>(sinks[tidl.y]);
      sum_score[i] = 1;
    }
  }

  int kb_lim = params->NK;
  int kb_min_causal = params->NK;

  // Blocks before this are ignored
  int kb_start = 0;

  // First block with no masked block edge
  int kb_win_lo = 0;
  // Last block with no masked block edge
  int kb_win_hi = params->NK;

  // Query positions covered by this block are [q_min, q_max)
  const int q_max = (tid.x + 1) * BQ + params->qL_off;
  const int q_min = max(0, int(tid.x) * BQ + params->qL_off);

  if (do_causal) {
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);

    kb_min_causal = (q_min / BK);
  } else if (has_window) {
    int k_max = q_max + window_size - 1;
    kb_lim = (k_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);
  }

  if (has_window) {
    // The smallest k this block attends to is decided by the window
    int k_min = max(0, q_min - window_size + 1);
    kb_start = k_min / BK;

    // Move the K and V pointers to the first key block
    K += kb_start * BK * params->K_strides[2];
    V += kb_start * BK * params->V_strides[2];

    // All queries in the block [q_min, q_max) attend to
    // keys [common_k_min, common_k_max)
    auto common_k_min = max(0, q_max - window_size);
    auto common_k_max = q_min + window_size;
    kb_win_lo = (common_k_min + BK - 1) / BK;
    kb_win_hi = common_k_max / BK;
  }

  const bool is_last_bq = int(tid.x) == (params->NQ_aligned);
  // const bool is_last_tq = int(simd_group_id) >= (params->qL_rem / UQ);
  const bool is_last_q = is_last_bq;

  const short lim_rows_q = params->qL_rem - tm;
  const short lim_rows_k = params->kL_rem;

  // Loop over KV seq length
  for (int kb = kb_start; kb < kb_lim; kb++) {
    const int is_last_k = (kb == (params->NK_aligned));

    // Do S = Q @ K.T
    using stile_t = NAXTile<AccumType, TQ, TK>;
    stile_t Stile;

    Stile.clear();

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik += 2) {
        // Unrolling the head-dim loop by 4, rather than fully, potentially
        // lets the compiler interleave the next K-tile loads with the running
        // mma chain instead of hoisting all TD loads up front and is faster
        // for head dim 128.
#pragma clang loop unroll_count(4)
        for (short id = 0; id < TD; id++) {
          NAXTile<T, 1, 1> Qtile;
          NAXTile<T, 2, 1> Ktile;

          const int Q_load_off = iq * kU * int(params->Q_strides[2]) + id * kU;
          const int K_load_off = ik * kU * int(params->K_strides[2]) + id * kU;

          if (!align_Q && is_last_q) {
            Qtile.load_rows(
                Q + Q_load_off,
                int(params->Q_strides[2]),
                lim_rows_q - iq * kU);
          } else {
            Qtile.load(Q + Q_load_off, int(params->Q_strides[2]));
          }

          if (!align_K && is_last_k) {
            Ktile.load_rows(
                K + K_load_off,
                int(params->K_strides[2]),
                lim_rows_k - ik * kU);
          } else {
            Ktile.load(K + K_load_off, int(params->K_strides[2]));
          }

          stile_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik),
              Stile.frag_at(iq, ik + 1),
              Qtile.frag_at(0, 0),
              metal::false_type{},
              Ktile.frag_at(0, 0),
              Ktile.frag_at(1, 0),
              metal::true_type{});
        }
      }
    }

    // Scale S
    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= float(scale2);
    }

    // Mask out length sequence
    if (!align_K && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = ik * kU + sn;

          thread auto& fg = Stile.frag_at(iq, ik);

          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < params->kL_rem) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal or at the window edge
    const bool is_causal_masked = do_causal && (kb >= kb_min_causal);
    const bool is_win_lo_masked = has_window && (kb < kb_win_lo);
    const bool is_win_hi_masked = has_window && !do_causal && (kb >= kb_win_hi);

    if (is_causal_masked || is_win_lo_masked || is_win_hi_masked) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + params->qL_off + tm;
      const int base_col = kb * BK;

      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);

          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto r =
                  base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
              const auto c = base_col + ik * kU + jj + sn;
              const auto loc = ii * stile_t::kFragThrCols + jj;

              if (has_window) {
                if (is_causal_masked) {
                  fg[loc] = (r < c) ? neg_inf : fg[loc];
                }
                if (is_win_lo_masked) {
                  fg[loc] = ((r - c) >= window_size) ? neg_inf : fg[loc];
                }
                if (is_win_hi_masked) {
                  fg[loc] = ((c - r) >= window_size) ? neg_inf : fg[loc];
                }
              } else if (do_causal) {
                fg[loc] = (r < c) ? neg_inf : fg[loc];
              }
            }
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + tm;
      const int base_col = kb * BK;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, AccumType>;
      using mtile_t = NAXTile<melem_t, TQ, TK>;
      using mfrag_t = typename mtile_t::frag_type;

      if (base_row + BQ <= params->qL && base_col + BK <= params->kL) {
        for (short iq = 0; iq < TQ; iq++) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            const int row_pos = base_row + iq * kU;
            const int col_pos = base_col + ik * kU;

            mfrag_t mfrag;
            ink_relative_bias(mfrag, table, relative_sink, relative, tid.y, params->qL, params->kL,
                              row_pos, col_pos);

            thread auto& fg = Stile.frag_at(iq, ik);

            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
              if constexpr (is_bool) {
                fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
              } else {
                fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
              }
            }
          }
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ik++) {
            const int row_pos = base_row + iq * kU;
            const int col_pos = base_col + ik * kU;

            mfrag_t mfrag;
            ink_relative_bias(mfrag, table, relative_sink, relative, tid.y, params->qL, params->kL,
                              row_pos, col_pos);

            thread auto& fg = Stile.frag_at(iq, ik);

            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
              if constexpr (is_bool) {
                fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
              } else {
                fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
              }
            }
          }
        }
      }
    }

    // Do softmax

    // Temp variables
    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }

    // Row Sum
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }

    Stile.template row_reduce<SumOp>(sum_score);

    // Update O
    Otile.template row_bin_op<MulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

    // Do O = P @ V
    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TV; id += 2) {
        if constexpr (BV == 128) {
          if (id == 4) {
            threadgroup_barrier(mem_flags::mem_none);
          }
        }

        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          NAXTile<T, 1, 2> Vtile;

          const int V_load_off = ik * kU * int(params->V_strides[2]) + id * kU;

          if (!align_K && is_last_k) {
            Vtile.load_rows(
                V + V_load_off,
                int(params->V_strides[2]),
                lim_rows_k - ik * kU);
          } else {
            Vtile.load(V + V_load_off, int(params->V_strides[2]));
          }

          otile_t::NAXFrag_t::mma(
              Otile.frag_at(iq, id),
              Otile.frag_at(iq, id + 1),
              Stile.frag_at(iq, ik),
              metal::false_type{},
              Vtile.frag_at(0, 0),
              Vtile.frag_at(0, 1),
              metal::false_type{});
        }
      }
    }

    // Prepare for next iteration
    K += BK * int(params->K_strides[2]);
    V += BK * int(params->V_strides[2]);
  }

  // Normalize output

  threadgroup_barrier(mem_flags::mem_none);

  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }

  Otile.template row_bin_op<MulOp>(rcp);

  if (save_lse && sn == 0) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      const short r = tm + i * otile_t::kFragRowsJump + sm;
      if (align_Q || !is_last_q || r < params->qL_rem) {
        lse[r] = max_score[i] * M_LN2_F + metal::log(sum_score[i]);
      }
    }
  }

  // Store results
  O += tm * int(params->O_strides[2]);

  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;

    Otile.store_rows(O, int(params->O_strides[2]), lim_rows_q);
  } else {
    Otile.store(O, int(params->O_strides[2]));
  }
}

instantiate_kernel("ink_global_relative_attention", ink_global_relative_attention, float, 64, 32, 128, 128, 4, 1, float, float)

// Default prefill: BF16 query, key and value operands read as stored (the F32
// instance widens the same BF16 values), F32 scores, softmax and output
// accumulation, BF16 output.
instantiate_kernel("ink_global_relative_attention_bf16", ink_global_relative_attention, bfloat16_t, 64, 32, 128, 128, 4, 1, float, float)
