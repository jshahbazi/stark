#pragma once

#include <metal_stdlib>

#include "../../../../common/thread_context.h"
#include "../../../../generated/gemm.h"
#include "../../../common/fragment.h"
#include "../../../common/mxu_gemm_loop.h"
#include "../gemm_alignment.h"
#include "../operands.h"
#include "../quant_scale_bias.h"
#include "../quant_scale_zero_point.h"
#include "tile_context.h"

using namespace metal;

namespace stark {
namespace gemm {
namespace schedules {

template <typename Core, typename RightOperand, ushort STAGING_COLS>
static METAL_FUNC auto make_staged_group_loader(
    const typename Core::RightStorage right,
    const constant stark::matmul::GemmParams* params,
    const size_t block_col,
    const uint k_offset,
    threadgroup typename Core::RightElementType* staging,
    const thread ThreadContext& thread_context
) {
  using Element = typename Core::RightElementType;
  const uint row_stride =
      uint(params->K) * uint(get_bytes_per_pack<RightOperand::BITS>()) / uint(get_pack_factor<RightOperand::BITS>());
  const uint first_group = k_offset / uint(RightOperand::GROUP_SIZE);
  const int params_offset =
      int(block_col) * int(params->scale_output_stride) + int(first_group) * int(params->scale_group_stride);
  const device Element* scales = right.scales + params_offset;
  const device uint8_t* values = right.codes + size_t(block_col) * row_stride +
                                 size_t(k_offset) * size_t(get_bytes_per_pack<RightOperand::BITS>()) /
                                     size_t(get_pack_factor<RightOperand::BITS>());
  if constexpr (RightOperand::SCHEME == GemmBPrologueKind::ScaleBiasDequant) {
    const device Element* biases = right.bias() + params_offset;
    using Loader = QuantizedBlockLoaderScaleBias<
        Element,
        Core::THREADGROUP_BLOCK_N,
        STAGING_COLS,
        Core::SHARED_STRIDE_B,
        Core::THREADGROUP_THREADS,
        RightOperand::GROUP_SIZE,
        RightOperand::BITS>;
    return Loader(
        values,
        scales,
        biases,
        right.signed_codes,
        int(params->K),
        int(params->scale_group_stride),
        int(params->scale_output_stride),
        staging,
        thread_context.simdgroup_index,
        thread_context.simd_lane_id
    );
  } else if constexpr (RightOperand::SCHEME == GemmBPrologueKind::ScaleZeroPointDequant) {
    using Loader = QuantizedBlockLoaderScaleZeroPoint<
        Element,
        Core::THREADGROUP_BLOCK_N,
        STAGING_COLS,
        Core::SHARED_STRIDE_B,
        Core::THREADGROUP_THREADS,
        RightOperand::GROUP_SIZE,
        RightOperand::BITS,
        false>;
    return Loader(
        values,
        scales,
        right.zp(),
        right.signed_codes,
        int(params->K),
        int(params->scale_group_stride),
        int(params->scale_output_stride),
        params->zero_point_output_stride,
        params->zero_point_group_stride,
        uint(block_col) * params->zero_point_output_stride + first_group * params->zero_point_group_stride,
        staging,
        thread_context.simdgroup_index,
        thread_context.simd_lane_id
    );
  } else {
    static_assert(
        RightOperand::SCHEME == GemmBPrologueKind::ScaleSymmetricDequant,
        "staged loader requires a quantized weight scheme"
    );
    using Loader = QuantizedBlockLoaderScaleZeroPoint<
        Element,
        Core::THREADGROUP_BLOCK_N,
        STAGING_COLS,
        Core::SHARED_STRIDE_B,
        Core::THREADGROUP_THREADS,
        RightOperand::GROUP_SIZE,
        RightOperand::BITS,
        true>;
    return Loader(
        values,
        scales,
        right.signed_codes,
        int(params->K),
        int(params->scale_group_stride),
        int(params->scale_output_stride),
        staging,
        thread_context.simdgroup_index,
        thread_context.simd_lane_id
    );
  }
}

// Stage independent quantization groups together without changing their
// coefficients, unpack arithmetic, or the order of the MMA operations.
template <typename Loader, ushort GROUP_SIZE>
struct PairedStagedLoader {
  Loader first;
  Loader second;

  void load_unsafe() const {
    first.load_unsafe();
    second.load_unsafe();
  }

  void load_safe(short2 dimensions) const {
    first.load_safe(short2(min(dimensions.x, short(GROUP_SIZE)), dimensions.y));
    second.load_safe(short2(max(short(dimensions.x - GROUP_SIZE), short(0)), dimensions.y));
  }

  void next() {
    first.next();
    first.next();
    second.next();
    second.next();
  }
};

template <typename Core, typename RightOperand>
static METAL_FUNC auto make_staged_loader(
    const typename Core::RightStorage right,
    const constant stark::matmul::GemmParams* params,
    const size_t block_col,
    const uint k_offset,
    threadgroup typename Core::RightElementType* staging,
    const thread ThreadContext& thread_context
) {
  if constexpr (Core::THREADGROUP_BLOCK_K > RightOperand::GROUP_SIZE) {
    static_assert(Core::THREADGROUP_BLOCK_K == 2 * RightOperand::GROUP_SIZE, "paired staging holds exactly two groups");
    auto first = make_staged_group_loader<Core, RightOperand, RightOperand::GROUP_SIZE>(
        right, params, block_col, k_offset, staging, thread_context
    );
    auto second = make_staged_group_loader<Core, RightOperand, RightOperand::GROUP_SIZE>(
        right,
        params,
        block_col,
        k_offset + RightOperand::GROUP_SIZE,
        staging + RightOperand::GROUP_SIZE,
        thread_context
    );
    return PairedStagedLoader<decltype(first), RightOperand::GROUP_SIZE>{first, second};
  } else {
    return make_staged_group_loader<Core, RightOperand, Core::THREADGROUP_BLOCK_K>(
        right, params, block_col, k_offset, staging, thread_context
    );
  }
}

template <bool ALIGNED_N, typename Loader>
static METAL_FUNC void stage(thread Loader& loader, const short tile_block_cols, const short block_k) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if constexpr (ALIGNED_N) {
    loader.load_unsafe();
  } else {
    loader.load_safe(short2(block_k, tile_block_cols));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <typename Core>
static METAL_FUNC auto make_full_precision_loader(
    const typename Core::RightStorage right,
    const constant stark::matmul::GemmParams* params,
    const size_t block_col,
    const uint k_offset,
    threadgroup typename Core::RightElementType* staging,
    const thread ThreadContext& thread_context
) {
  const device typename Core::RightElementType* values =
      right.dense + (Core::TRANSPOSE_RIGHT ? block_col * size_t(params->leading_dimension_b) + k_offset
                                           : block_col + size_t(k_offset) * size_t(params->leading_dimension_b));
  using Loader = typename Core::FullPrecisionRightLoader;
  return Loader(values, params->leading_dimension_b, staging, thread_context);
}

struct StagedSchedule {
  template <typename Core, bool ALIGNED_M, bool ALIGNED_N>
  static METAL_FUNC typename Core::AccumFragment launch(
      typename Core::LeftStorage left,
      typename Core::RightStorage right,
      threadgroup typename Core::RightElementType* staging,
      const constant stark::matmul::GemmParams* params,
      const TileContext tile,
      const GemmAlignment,
      const thread ThreadContext& thread_context
  ) {
    static_assert(!Core::Left::QUANTIZED && Core::Right::QUANTIZED, "staged schedule requires dense A and quantized W");
    static_assert(Core::TRANSPOSE_RIGHT, "quantized staged GEMM requires transposed B");
    using LeftTile =
        stark::matmul::Fragment<typename Core::LeftElementType, Core::TILES_M, Core::TILES_K, typename Core::FragmentOps>;
    using RightTile = stark::matmul::Fragment<
        typename Core::RightElementType,
        Core::TILES_N,
        Core::TILES_K,
        typename Core::FragmentOps,
        stark::matmul::ReadDirect,
        true>;

    auto left_source = stark::matmul::fragment_source(
        left.values + size_t(tile.abs_row_base) * size_t(params->leading_dimension_a) + tile.k_offset,
        int(params->leading_dimension_a)
    );
    if constexpr (!ALIGNED_M) {
      left_source = left_source.bounded(tile.simdgroup_limit_m, Core::SIMDGROUP_BLOCK_K);
    }
    auto right_source = stark::matmul::fragment_source(
        staging + tile.tile_col_offset * Core::SHARED_STRIDE_B,
        int(Core::SHARED_STRIDE_B)
    );
    auto loader = make_staged_loader<Core, typename Core::RightOperand>(
        right,
        params,
        tile.block_col,
        tile.k_offset,
        staging,
        thread_context
    );

    typename Core::AccumFragment accumulator;
    accumulator.clear();

    METAL_PRAGMA_NO_UNROLL
    for (int outer_k = 0; outer_k < int(params->aligned_inner_iterations); ++outer_k) {
      stage<ALIGNED_N>(loader, tile.tile_block_cols, Core::THREADGROUP_BLOCK_K);

      METAL_PRAGMA_NO_UNROLL
      for (ushort inner_k = 0; inner_k < Core::THREADGROUP_BLOCK_K; inner_k += Core::SIMDGROUP_BLOCK_K) {
        LeftTile left_tile;
        RightTile right_tile;
        left_tile.load_from(thread_context.simd_lane_id, left_source.advanced(inner_k));
        right_tile.load_from(thread_context.simd_lane_id, right_source.advanced(inner_k));
        stark::matmul::fragment_mma(accumulator, left_tile, right_tile);
      }

      left_source = left_source.advanced(Core::THREADGROUP_BLOCK_K);
      loader.next();
    }

    return accumulator;
  }
};

} // namespace schedules
} // namespace gemm
} // namespace stark
