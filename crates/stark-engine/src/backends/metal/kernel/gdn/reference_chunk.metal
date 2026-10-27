// Qwen3.5 pinned torch_chunk_gated_delta_rule, revision
// 2e12b325d5c4532f3ec5b0ec54e1ff0b7e0db397. Full F32 scratch and state.
// Transformers Apache-2.0 terms: scripts/licenses/distribution/transformers-LICENSE.
#include <metal_stdlib>
#include "../common/dsl.h"
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#include "../common/vision_exp.h"
#include "reference_cumsum.h"

PUBLIC KERNEL(ReferenceGDNChunkPrepare)(
    const device float* q_input, const device float* k_input,
    const device float* beta, const device float* log_decay, const device bfloat* values,
    device float* q, device float* k, device float* kb, device float* vb, device float* cumulative,
    constant uint& rows, constant uint& heads, constant uint& groups,
    constant uint& key_dim, constant uint& value_dim, constant uint& row_stride,
    const uint plane AXIS(((rows + 63) / 64) * heads, 32)
) {
  const uint chunk = plane / heads, head = plane % heads;
  const uint group = head / (heads / groups);
  ulong sum = 0;
  for (uint i = 0; i < 64; ++i) {
    const uint row = chunk * 64 + i;
    const bool active = row < rows;
    sum = gdn_f64_accumulate(sum, active ? log_decay[row * heads + head] : 0.0f);
    cumulative[plane * 64 + i] = gdn_f64_prefix(sum);
    const float b = active ? beta[row * heads + head] : 0.0f;
    for (uint channel = 0; channel < 128; ++channel) {
      const uint dst = (plane * 64 + i) * 128 + channel;
      const float kval = active ? k_input[row * key_dim + group * 128 + channel] : 0.0f;
      q[dst] = active ? q_input[row * key_dim + group * 128 + channel] : 0.0f;
      k[dst] = kval;
      kb[dst] = kval * b;
      vb[dst] = active ? float(values[row * row_stride + 2 * key_dim + head * 128 + channel]) * b : 0.0f;
    }
  }
}

PUBLIC KERNEL(ReferenceGDNChunkGram)(
    const device float* q, const device float* k, const device float* kb, const device float* cumulative,
    device float* triangular, device float* attention,
    constant uint& planes,
    const uint index AXIS(planes * 64 * 64, 128)
) {
  const uint plane = index / 4096, row = (index / 64) % 64, column = index % 64;
  float kk = 0.0f, qk = 0.0f;
  if (column <= row) {
    for (uint inner = 0; inner < 128; ++inner) {
      const float b = k[(plane * 64 + column) * 128 + inner];
      kk = fma(kb[(plane * 64 + row) * 128 + inner], b, kk);
      qk = fma(q[(plane * 64 + row) * 128 + inner], b, qk);
    }
    const float pairwise = vision_exp(cumulative[plane * 64 + row] - cumulative[plane * 64 + column]);
    kk *= pairwise;
    qk *= pairwise;
  }
  triangular[index] = kk;
  attention[index] = qk;
}

PUBLIC KERNEL(ReferenceGDNChunkSolve)(
    const device float* triangular, const device float* kb, const device float* vb,
    const device float* cumulative, device float* u, device float* w,
    constant uint& planes,
    const uint index AXIS(planes * 128, 128)
) {
  const uint plane = index / 128, channel = index % 128;
  for (uint row = 0; row < 64; ++row) {
    const uint dst = (plane * 64 + row) * 128 + channel;
    float left = vb[dst];
    float right = kb[dst] * vision_exp(cumulative[plane * 64 + row]);
    for (uint column = 0; column < row; ++column) {
      const float coefficient = -triangular[plane * 4096 + row * 64 + column];
      left = fma(coefficient, u[(plane * 64 + column) * 128 + channel], left);
      right = fma(coefficient, w[(plane * 64 + column) * 128 + channel], right);
    }
    u[dst] = left;
    w[dst] = right;
  }
}

PUBLIC KERNEL(ReferenceGDNChunkNewValues)(
    const device float* u, const device float* w, const device float* state, device float* vnew,
    constant uint& heads, constant uint& chunk,
    const uint index AXIS(heads * 64 * 128, 128)
) {
  const uint head = index / 8192, row = (index / 128) % 64, dv = index % 128;
  const uint plane = chunk * heads + head;
  float product = 0.0f;
  for (uint dk = 0; dk < 128; ++dk)
    product = fma(w[(plane * 64 + row) * 128 + dk], state[(head * 128 + dv) * 128 + dk], product);
  vnew[index] = u[(plane * 64 + row) * 128 + dv] - product;
}

PUBLIC KERNEL(ReferenceGDNChunkOutput)(
    const device float* q, const device float* attention, const device float* cumulative,
    const device float* state, const device float* vnew, device bfloat* output,
    constant uint& heads, constant uint& chunk, constant uint& rows,
    const uint index AXIS(heads * 64 * 128, 128)
) {
  const uint head = index / 8192, row = (index / 128) % 64, dv = index % 128;
  if (chunk * 64 + row >= rows) return;
  const uint plane = chunk * heads + head;
  const float decay = vision_exp(cumulative[plane * 64 + row]);
  float inter = 0.0f, intra = 0.0f;
  for (uint dk = 0; dk < 128; ++dk) {
    const float query = q[(plane * 64 + row) * 128 + dk] * decay;
    inter = fma(query, state[(head * 128 + dv) * 128 + dk], inter);
  }
  for (uint inner = 0; inner < 64; ++inner)
    intra = fma(attention[plane * 4096 + row * 64 + inner], vnew[(head * 64 + inner) * 128 + dv], intra);
  output[((chunk * 64 + row) * heads + head) * 128 + dv] = bfloat(inter + intra);
}

PUBLIC KERNEL(ReferenceGDNChunkState)(
    const device float* k, const device float* cumulative, const device float* vnew, device float* state,
    constant uint& heads, constant uint& chunk,
    const uint index AXIS(heads * 128 * 128, 128)
) {
  const uint head = index / 16384, dv = (index / 128) % 128, dk = index % 128;
  const uint plane = chunk * heads + head;
  const float end = cumulative[plane * 64 + 63];
  float update = 0.0f;
  for (uint row = 0; row < 64; ++row) {
    const float key = k[(plane * 64 + row) * 128 + dk] * vision_exp(end - cumulative[plane * 64 + row]);
    update = fma(key, vnew[(head * 64 + row) * 128 + dv], update);
  }
  state[index] = state[index] * vision_exp(end) + update;
}
