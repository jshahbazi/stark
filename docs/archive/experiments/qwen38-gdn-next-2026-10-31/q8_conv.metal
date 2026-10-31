// RESEARCH ONLY: fused ordinary official Q8 GDN input projection + convolution.
// No production selector or caller uses this entrypoint.
// Original QMV tiling, dot expressions, SIMD sums and BF16 rounding are retained.
// The sole changed projection epilogue owns exactly one channel per output row.
// Raw QKV is not materialized unless RESEARCH_CAPTURE_RAW_QKV=1; z/a/b remain.
// ABI buffers0..4=codes,scales,biases,inputBF16,joinedBF16[N16480].
// Buffers5..7=mutableBF16history[10240,3],convBF16weights[4,10240],
// convolvedBF16[10240]. Constants8..9=K2560,N16480.
// Grid (1,N/8,1), threads(32,2,1), Safe Metal4. Inputs/outputs must not alias.
// Source revisions and SHA256:
// official_q8_gdn_decode.metal: 21465bbcf61db5342186429c9c4405693fc2205cb5c8730af9f57296adcdfa44
// official_gdn_k1_helper.metal: 6cfe8926c91f02bf14dbca0c4e68a36f8b27cc381bcb9b23e9ff2185e226b78e
// Original convolution: vendor/lily-core/src/kernels/metal/gdn.metal SHA256
// 78ca01774e89e1fea05190234171704cc6a2bca4b24415cee2aa45cdd5ac6844
#ifndef RESEARCH_CAPTURE_RAW_QKV
#define RESEARCH_CAPTURE_RAW_QKV 0
#endif

// MIT License
// Copyright (c) 2026 David Dalcu
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Source contract: mlx-serve 046b647686419df06c6c7132fb1d15d76185c3e5,
// gdn_decode.zig K1_SOURCE; inline BF16 sigmoid and SiLU. MLX BF16 wrappers
// round each arithmetic result. Stark port uses integer RNE barriers so native
// bfloat promotion/fast-math cannot erase intermediate BF16 boundaries.
// MODE0 keeps the literal non-precise metal::exp(float) intrinsic. MODE1/2
// are explicit diagnostic controls only; they do not select production policy.

#include <metal_stdlib>
using namespace metal;

static inline float official_gdn_k1_round_bf16(float x) {
    const uint bits = as_type<uint>(x);
    // Preserve infinities/NaNs. All oracle inputs are finite; exp can overflow.
    if ((bits & 0x7f800000u) == 0x7f800000u) return x;
    const uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return as_type<float>(rounded & 0xffff0000u);
}

template <uint MODE>
static inline float official_gdn_k1_sigmoid_f32(float x) {
    float ex;
    if constexpr (MODE == 1) ex = metal::precise::exp(metal::abs(x));
    else if constexpr (MODE == 2) ex = metal::fast::exp(metal::abs(x));
    else ex = metal::exp(metal::abs(x));
    const float e = official_gdn_k1_round_bf16(ex);
    const float d = official_gdn_k1_round_bf16(1.0f + e);
    const float y = official_gdn_k1_round_bf16(1.0f / d);
    return x < 0.0f ? y : official_gdn_k1_round_bf16(1.0f - y);
}

template <uint MODE>
static inline float official_gdn_k1_silu_f32(float acc) {
    const float x = official_gdn_k1_round_bf16(acc);
    const float sig = official_gdn_k1_sigmoid_f32<MODE>(x);
    return official_gdn_k1_round_bf16(x * sig);
}


// Copyright © 2023-2024 Apple Inc.
// Source MLX64ea011cb65f14d9ce2737e60db9a4ae91ed7441, MIT; see NOTICE.
// Affine Q8 decode: original helper/loop expressions, BF16 output unchanged.
// Local changes: symbol prefixes, METAL_FUNC spelling, wrapper entrypoint.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;
static constant constexpr const int OFFICIAL_Q8_GDN_SIMD_SIZE = 32;

template <int bits, int wsize = 8>
inline constexpr short official_q8_gdn_get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short official_q8_gdn_get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename T, typename U, int values_per_thread, int bits>
inline U official_q8_gdn_load_vector(const device T* x, thread U* x_thread) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  return sum;
}

template <typename U, int values_per_thread, int bits>
inline U official_q8_gdn_qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename T, int group_size, int bits>
inline void research_q8_gdn_projection_conv_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    device bfloat* window,
    const device bfloat* conv_weights,
    device bfloat* convolved,
    const constant int& in_vec_size,
    const constant int& out_vec_size,
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int packs_per_thread = bits == 2 ? 1 : 2;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int pack_factor = official_q8_gdn_get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = official_q8_gdn_get_bytes_per_pack<bits, 32>();
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * OFFICIAL_Q8_GDN_SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  x += tid.x * in_vec_size + simd_lid * values_per_thread;
  y += tid.x * out_vec_size + out_row;

  for (int k = 0; k < in_vec_size; k += block_size) {
    U sum = official_q8_gdn_load_vector<T, U, values_per_thread, bits>(x, x_thread);

    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = scales + row * in_vec_size_g;
      const device T* bl = biases + row * in_vec_size_g;

      U s = sl[0];
      U b = bl[0];
      result[row] += official_q8_gdn_qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
    }

    ws += block_size * bytes_per_pack / pack_factor;
    scales += block_size / group_size;
    biases += block_size / group_size;
    x += block_size;
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    result[row] = simd_sum(result[row]);
    if (simd_lid == 0) {
      // Preserve the production projection's BF16 boundary before convolution.
      const T projected = static_cast<T>(result[row]);
      const int channel = out_row + row;
      if (channel < 10240) {
#if RESEARCH_CAPTURE_RAW_QKV
        y[row] = projected;
#endif
        float acc = 0.0f;
        for (uint tap = 0; tap < 3; ++tap)
          acc += float(window[channel * 3 + tap]) * float(conv_weights[tap * 10240 + channel]);
        acc += float(projected) * float(conv_weights[3 * 10240 + channel]);
        convolved[channel] = bfloat(official_gdn_k1_silu_f32<0>(acc));
        window[channel * 3] = window[channel * 3 + 1];
        window[channel * 3 + 1] = window[channel * 3 + 2];
        window[channel * 3 + 2] = projected;
      } else {
        // z, a and b retain the ordinary joined output layout and rounding.
        y[row] = projected;
      }
    }
  }
}


kernel void qwen38_q8_projection_conv_research(
    device const uint* codes [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const bfloat* x [[buffer(3)]],
    device bfloat* joined [[buffer(4)]],
    device bfloat* window [[buffer(5)]],
    device const bfloat* conv_weights [[buffer(6)]],
    device bfloat* convolved [[buffer(7)]],
    constant int& K [[buffer(8)]],
    constant int& N [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
    research_q8_gdn_projection_conv_impl<bfloat, 64, 8>(
        codes, scales, biases, x, joined, window, conv_weights, convolved,
        K, N, tid, simd_gid, simd_lid);
}
