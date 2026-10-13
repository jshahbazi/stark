#pragma once

#include <metal_simdgroup>
#include <metal_stdlib>

#include "../../common/defines.h"
#include "../../common/quant_pack.h"
#include "../../common/quant_unpack.h"

using namespace metal;

namespace stark {
namespace gemm {

template <
    typename T,
    short THREADGROUP_TILE_ROWS,
    short THREADGROUP_TILE_COLS,
    short DESTINATION_LEADING_DIMENSION,
    short THREADGROUP_SIZE,
    short GROUP_SIZE,
    short BITS,
    bool SCALE_SYMMETRIC = false>
struct QuantizedBlockLoaderScaleZeroPoint {
  static_assert(
      GROUP_SIZE % THREADGROUP_TILE_COLS == 0 || THREADGROUP_TILE_COLS % GROUP_SIZE == 0,
      "staging tiles and coefficient groups must contain whole multiples of each other"
  );
  static_assert(BITS == 4 || BITS == 8, "Only int4 and int8 supported");

  STARK_CONST short PACK_FACTOR = get_pack_factor<BITS, 8>();
  STARK_CONST short BYTES_PER_PACK = get_bytes_per_pack<BITS>();
  STARK_CONST short THREADGROUP_TILE_COLS_PACKED = THREADGROUP_TILE_COLS / PACK_FACTOR;
  STARK_CONST short READS_PER_THREAD = (THREADGROUP_TILE_COLS_PACKED * THREADGROUP_TILE_ROWS < THREADGROUP_SIZE)
                                         ? 1
                                         : (THREADGROUP_TILE_COLS_PACKED * THREADGROUP_TILE_ROWS) / THREADGROUP_SIZE;
  STARK_CONST short GROUP_STEPS_PER_BLOCK = GROUP_SIZE / THREADGROUP_TILE_COLS;
  STARK_CONST short PARAM_GROUPS_PER_BLOCK =
      THREADGROUP_TILE_COLS > GROUP_SIZE ? THREADGROUP_TILE_COLS / GROUP_SIZE : 1;
  STARK_CONST short READS_PER_SCALE =
      READS_PER_THREAD * PACK_FACTOR > GROUP_SIZE ? GROUP_SIZE / PACK_FACTOR : READS_PER_THREAD;
  STARK_CONST bool TILE_HAS_IDLE_THREADS = THREADGROUP_TILE_COLS_PACKED * THREADGROUP_TILE_ROWS < THREADGROUP_SIZE;
  static_assert(
      GROUP_SIZE % (READS_PER_THREAD * PACK_FACTOR) == 0 || (READS_PER_THREAD * PACK_FACTOR) % GROUP_SIZE == 0,
      "thread read segments and coefficient groups must contain whole multiples of each other"
  );

  const int src_leading_dim;
  const int tile_stride;
  short group_step_counter;
  const int group_stride;
  const uint zero_point_group_stride;

  const short thread_index;
  const short tile_row_index;
  const short tile_col_index;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device uint8_t* zero_points_row_start;
  uint zero_point_index;
  const bool signed_codes;

  QuantizedBlockLoaderScaleZeroPoint(
      const device uint8_t* src_,
      const device T* scales_,
      const device uint8_t* zero_points_row_start_,
      const bool signed_codes_,
      const int src_leading_dim_,
      const int params_group_stride_,
      const int params_output_stride_,
      const uint zero_point_output_stride_,
      const uint zero_point_group_stride_,
      const uint zero_point_index_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  )
      : src_leading_dim(src_leading_dim_), tile_stride(THREADGROUP_TILE_COLS_PACKED * BYTES_PER_PACK),
        group_step_counter(0), group_stride(params_group_stride_), zero_point_group_stride(zero_point_group_stride_),
        thread_index(simd_group_id * 32 + simd_lane_id),
        tile_row_index(READS_PER_THREAD * thread_index / THREADGROUP_TILE_COLS_PACKED),
        tile_col_index((READS_PER_THREAD * thread_index) % THREADGROUP_TILE_COLS_PACKED),
        dst(dst_ + tile_row_index * DESTINATION_LEADING_DIMENSION + tile_col_index * PACK_FACTOR),
        src(src_ + tile_row_index * src_leading_dim_ * BYTES_PER_PACK / PACK_FACTOR + tile_col_index * BYTES_PER_PACK),
        scales(
            scales_ + tile_row_index * params_output_stride_ +
            (tile_col_index * PACK_FACTOR / GROUP_SIZE) * params_group_stride_
        ),
        zero_points_row_start(SCALE_SYMMETRIC ? nullptr : zero_points_row_start_),
        zero_point_index(
            zero_point_index_ + uint(tile_row_index) * zero_point_output_stride_ +
            uint(tile_col_index * PACK_FACTOR / GROUP_SIZE) * zero_point_group_stride_
        ),
        signed_codes(signed_codes_) {}

  QuantizedBlockLoaderScaleZeroPoint(
      const device uint8_t* src_,
      const device T* scales_,
      const bool signed_codes_,
      const int src_leading_dim_,
      const int params_group_stride_,
      const int params_output_stride_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]
  )
      : QuantizedBlockLoaderScaleZeroPoint(
            src_,
            scales_,
            static_cast<const device uint8_t*>(nullptr),
            signed_codes_,
            src_leading_dim_,
            params_group_stride_,
            params_output_stride_,
            0,
            0,
            0,
            dst_,
            simd_group_id,
            simd_lane_id
        ) {
    static_assert(SCALE_SYMMETRIC, "zero-point-free loader construction requires symmetric quantization");
  }

  inline void current_scale_bias(thread T& out_scale, thread T& out_bias, const uint group_offset = 0) const {
    uint zero_point_value;
    T scale_value;
    scale_value = scales[group_offset * uint(group_stride)];
    if constexpr (SCALE_SYMMETRIC) {
      zero_point_value = symmetric_zero_point<ushort(BITS)>();
    } else {
      const uint absolute_group_index = zero_point_index + group_offset * zero_point_group_stride;
      const uint byte_index = absolute_group_index / uint(PACK_FACTOR);
      const uint packed_index = absolute_group_index % uint(PACK_FACTOR);
      zero_point_value = decode_zero_point<ushort(BITS)>(zero_points_row_start[byte_index], packed_index);
    }
    out_scale = scale_value;
    out_bias = static_cast<T>(-scale_value * static_cast<T>(zero_point_value));
  }

  void load_unsafe() const {
    if constexpr (TILE_HAS_IDLE_THREADS) {
      if (tile_row_index >= THREADGROUP_TILE_ROWS) {
        return;
      }
    }

    for (int read_base = 0; read_base < READS_PER_THREAD; read_base += READS_PER_SCALE) {
      T scale;
      T bias;
      current_scale_bias(scale, bias, uint(read_base * PACK_FACTOR / GROUP_SIZE));
      for (int i = read_base; i < read_base + READS_PER_SCALE; i++) {
        dequantize<T, PACK_FACTOR, BITS>(src + i * BYTES_PER_PACK, scale, bias, dst + i * PACK_FACTOR, signed_codes);
      }
    }
  }

  void load_safe(short2 src_tile_dim) const {
    if constexpr (TILE_HAS_IDLE_THREADS) {
      if (tile_row_index >= THREADGROUP_TILE_ROWS) {
        return;
      }
    }

    if (tile_row_index >= src_tile_dim.y) {
      for (int i = 0; i < READS_PER_THREAD * PACK_FACTOR; i++) {
        dst[i] = T(0);
      }
      return;
    }

    const int valid_cols = src_tile_dim.x;
    const int valid_packs = (valid_cols + PACK_FACTOR - 1) / PACK_FACTOR;
    for (int read_base = 0; read_base < READS_PER_THREAD; read_base += READS_PER_SCALE) {
      T scale;
      T bias;
      current_scale_bias(scale, bias, uint(read_base * PACK_FACTOR / GROUP_SIZE));
      for (int i = read_base; i < read_base + READS_PER_SCALE; i++) {
        const int pack_index = tile_col_index + i;
        if (pack_index < valid_packs) {
          dequantize<T, PACK_FACTOR, BITS>(src + i * BYTES_PER_PACK, scale, bias, dst + i * PACK_FACTOR, signed_codes);
          if (pack_index == valid_packs - 1) {
            const int remaining = valid_cols - pack_index * PACK_FACTOR;
            for (int lane = remaining; lane < PACK_FACTOR; ++lane) {
              dst[i * PACK_FACTOR + lane] = T(0);
            }
          }
        } else {
          for (int lane = 0; lane < PACK_FACTOR; ++lane) {
            dst[i * PACK_FACTOR + lane] = T(0);
          }
        }
      }
    }
  }

  void next() {
    src += tile_stride;
    if constexpr (GROUP_STEPS_PER_BLOCK > 1) {
      group_step_counter++;
      if (group_step_counter != GROUP_STEPS_PER_BLOCK) {
        return;
      }
      group_step_counter = 0;
    }
    scales += group_stride * PARAM_GROUPS_PER_BLOCK;
    if constexpr (!SCALE_SYMMETRIC) {
      zero_point_index += zero_point_group_stride * uint(PARAM_GROUPS_PER_BLOCK);
    }
  }
};

} // namespace gemm
} // namespace stark
