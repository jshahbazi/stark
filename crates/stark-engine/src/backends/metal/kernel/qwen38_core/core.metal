// Qwen3.8 architecture operations for Stark's Uzu backend.
// Recurrence/HC arithmetic adapted from Lily e22d58fa7cc488493fb1b5d810a198bd29730aeb
// Apache-2.0; see root LICENSE.qwen38-reference, NOTICE.qwen38-reference and
// docs/QWEN38_UZU_PROVENANCE.md. This file uses the Uzu kernel DSL and Uzu
// buffers; projections use standard Matmul.
// Copyright 2026 Perplexity AI, Inc. Copyright 2026 Fabio Greter.
// Gated delta equations also derive from MLX-LM (MIT):
// Copyright (c) 2023-2025 Apple Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include <metal_stdlib>
#include "../common/dsl.h"

PUBLIC KERNEL(Qwen38HcNorm)(
    const device bfloat* input, const device float* weight, device bfloat* output,
    constant uint& width, constant uint& streams, constant uint& rows, constant float& epsilon,
    threadgroup float partial[8], threadgroup float inverse[1],
    const uint segment GROUPS(rows * streams), const uint tid THREADS(256)
) {
  float sum = 0.0f;
  for (uint i = tid; i < width; i += 256) {
    const float value = float(input[segment * width + i]);
    sum += value * value;
  }
  sum = simd_sum(sum);
  if ((tid % 32) == 0) partial[tid / 32] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i) total += partial[i];
    inverse[0] = rsqrt(total / float(width) + epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < width; i += 256) {
    const uint index = segment * width + i;
    output[index] = bfloat(float(input[index]) * inverse[0] * (1.0f + weight[(segment % streams) * width + i]));
  }
}

PUBLIC KERNEL(Qwen38SiluScaled)(
    const device bfloat* input, device bfloat* output,
    constant uint& count, constant float& scale, const uint i AXIS(count, 128)
) {
  const float value = float(input[i]) * scale;
  output[i] = bfloat(value / (1.0f + exp(-value)));
}

PUBLIC KERNEL(Qwen38HcMix)(
    const device bfloat* normalized, const device bfloat* logits, device bfloat* output,
    constant uint& width, constant uint& streams, constant uint& rows,
    const uint i AXIS(rows * width, 128)
) {
  const uint row = i / width, column = i % width;
  float sum = 0.0f;
  for (uint stream = 0; stream < streams; ++stream) {
    const uint index = (row * streams + stream) * width + column;
    const float gate = 1.0f / (1.0f + exp(-float(logits[index])));
    sum += gate * float(normalized[index]);
  }
  output[i] = bfloat(sum / float(streams));
}

PUBLIC KERNEL(Qwen38HcInject)(
    const device bfloat* input, const device bfloat* branch, const device bfloat* logits,
    device bfloat* output, constant uint& width, constant uint& streams, constant uint& rows,
    const uint i AXIS(rows * width * streams, 128)
) {
  const uint row = i / (width * streams);
  const float logit = float(logits[i / width]) / float(streams);
  const float gate = 2.0f / (1.0f + exp(-logit));
  output[i] = bfloat(float(input[i]) + float(branch[row * width + i % width]) * gate);
}

PUBLIC KERNEL(Qwen38Broadcast)(
    const device bfloat* input, device bfloat* output,
    constant uint& width, constant uint& streams, constant uint& rows,
    const uint i AXIS(rows * width * streams, 128)
) {
  output[i] = input[(i / (width * streams)) * width + i % width];
}

PUBLIC KERNEL(Qwen38SiluMultiply)(
    const device bfloat* gate, const device bfloat* up, device bfloat* output,
    constant uint& count, const uint i AXIS(count, 128)
) {
  const float value = float(gate[i]);
  output[i] = bfloat((value / (1.0f + exp(-value))) * float(up[i]));
}

PUBLIC KERNEL(Qwen38GdnConv)(
    const device bfloat* input, const device float* weight, device bfloat* window,
    device bfloat* output, constant uint& channels, constant uint& taps,
    const uint channel AXIS(channels, 128)
) {
  const uint base = channel * (taps - 1);
  float sum = 0.0f;
  for (uint tap = 0; tap + 1 < taps; ++tap) {
    sum += float(window[base + tap]) * weight[channel * taps + tap];
  }
  sum += float(input[channel]) * weight[channel * taps + taps - 1];
  for (uint tap = 0; tap + 2 < taps; ++tap) window[base + tap] = window[base + tap + 1];
  window[base + taps - 2] = input[channel];
  output[channel] = bfloat(sum / (1.0f + exp(-sum)));
}

PUBLIC KERNEL(Qwen38GdnStep)(
    const device bfloat* qkv, const device bfloat* z, const device bfloat* a,
    const device bfloat* b, const device float* a_log, const device float* dt_bias,
    const device float* norm, device float* state, device bfloat* output,
    constant uint& key_heads, constant uint& value_heads, constant float& epsilon,
    constant uint& sigmoid_gate,
    threadgroup float q[128], threadgroup float k[128],
    threadgroup float partial_q[4], threadgroup float partial_k[4],
    threadgroup float gates[4], threadgroup float partial_norm[4], threadgroup float inverse[1],
    const uint head GROUPS(value_heads), const uint tid THREADS(128)
) {
  const uint key_head = head / (value_heads / key_heads);
  const uint key_dim = key_heads * 128;
  const float qv = float(qkv[key_head * 128 + tid]);
  const float kv = float(qkv[key_dim + key_head * 128 + tid]);
  const float qs = simd_sum(qv * qv), ks = simd_sum(kv * kv);
  if (tid % 32 == 0) { partial_q[tid / 32] = qs; partial_k[tid / 32] = ks; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float tq = 0.0f, tk = 0.0f;
    for (uint i = 0; i < 4; ++i) { tq += partial_q[i]; tk += partial_k[i]; }
    gates[0] = 1.0f / max(sqrt(tq), 1e-12f);
    gates[1] = 1.0f / max(sqrt(tk), 1e-12f);
    const float value = float(a[head]) + dt_bias[head];
    const float softplus = value > 20.0f ? value : log(1.0f + exp(value));
    gates[2] = exp(-exp(a_log[head]) * softplus);
    gates[3] = 1.0f / (1.0f + exp(-float(b[head])));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  q[tid] = qv * gates[0] * 0.08838834764831845f;
  k[tid] = kv * gates[1];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float* column = state + ulong(head) * 128 * 128 + tid;
  float predicted = 0.0f;
  for (uint i = 0; i < 128; ++i) predicted += k[i] * column[i * 128] * gates[2];
  const float delta = (float(qkv[2 * key_dim + head * 128 + tid]) - predicted) * gates[3];
  float result = 0.0f;
  for (uint i = 0; i < 128; ++i) {
    const float updated = column[i * 128] * gates[2] + k[i] * delta;
    column[i * 128] = updated;
    result += q[i] * updated;
  }
  const float rounded = float(bfloat(result));
  const float ss = simd_sum(rounded * rounded);
  if (tid % 32 == 0) partial_norm[tid / 32] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = 0.0f;
    for (uint i = 0; i < 4; ++i) total += partial_norm[i];
    inverse[0] = rsqrt(total / 128.0f + epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float value = float(z[head * 128 + tid]);
  const float sigmoid = 1.0f / (1.0f + exp(-value));
  const float activation = sigmoid_gate != 0 ? sigmoid : value * sigmoid;
  output[head * 128 + tid] = bfloat(norm[tid] * rounded * inverse[0] * activation);
}

PUBLIC KERNEL(Qwen38MoeRoute)(
    const device float* logits, device uint* indices, device float* scores,
    constant uint& experts, constant uint& selected, constant uint& normalize, constant uint& rows,
    threadgroup uint partial_keys[8], threadgroup uint partial_ids[8],
    threadgroup uint ceiling_key[1], threadgroup uint ceiling_id[1],
    const uint row GROUPS(rows), const uint tid THREADS(256)
) {
  logits += row * experts;
  indices += row * selected;
  scores += row * selected;
  // For every finite F32, this key is strictly increasing in numeric value;
  // collapse +/-zero because serial `>` treats them as equal. Integer
  // comparisons also preserve subnormal order under Metal fast-math.
  // A pair (key,id) ranks higher when key is larger, then id is smaller.
  // After selecting its maximum, every unselected pair is strictly below
  // that winner. This cutoff is therefore equivalent to excluding all prior
  // IDs, without rereading them or synchronizing device memory each slot.
  if (tid == 0) { ceiling_key[0] = 0xffffffffu; ceiling_id[0] = 0; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint slot = 0; slot < selected; ++slot) {
    const uint limit_key = ceiling_key[0], limit_id = ceiling_id[0];
    uint best_key = 0x007fffffu; // numeric -infinity sentinel
    uint best_id = 0xffffffffu;
    for (uint expert = tid; expert < experts; expert += 256) {
      const uint bits = as_type<uint>(logits[expert]);
      const uint magnitude = bits & 0x7fffffffu;
      const uint key = magnitude == 0 ? 0x80000000u : ((bits & 0x80000000u) != 0 ? ~bits : bits ^ 0x80000000u);
      const bool below = key < limit_key || (key == limit_key && expert > limit_id);
      // Keep the preceding router's NaN/-infinity exclusion and fallback;
      // +infinity still reaches the unchanged nonfinite softmax/readout fence.
      if (magnitude <= 0x7f800000u && key > 0x007fffffu && below &&
          (key > best_key || (key == best_key && expert < best_id))) {
        best_key = key; best_id = expert;
      }
    }
    for (uint offset = 16; offset > 0; offset >>= 1) {
      const uint other_key = simd_shuffle_xor(best_key, offset);
      const uint other_id = simd_shuffle_xor(best_id, offset);
      if (other_key > best_key || (other_key == best_key && other_id < best_id)) { best_key = other_key; best_id = other_id; }
    }
    if (tid % 32 == 0) {
      partial_keys[tid / 32] = best_key;
      partial_ids[tid / 32] = best_id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
      best_key = tid < 8 ? partial_keys[tid] : 0x007fffffu;
      best_id = tid < 8 ? partial_ids[tid] : 0xffffffffu;
      for (uint offset = 16; offset > 0; offset >>= 1) {
        const uint other_key = simd_shuffle_xor(best_key, offset);
        const uint other_id = simd_shuffle_xor(best_id, offset);
        if (other_key > best_key || (other_key == best_key && other_id < best_id)) { best_key = other_key; best_id = other_id; }
      }
      if (tid == 0) {
        indices[slot] = best_id < experts ? best_id : 0;
        ceiling_key[0] = best_key;
        ceiling_id[0] = best_id;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid == 0) {
    const float maximum = logits[indices[0]];
    float total = 0.0f;
    if (normalize != 0) {
      for (uint slot = 0; slot < selected; ++slot) total += exp(logits[indices[slot]] - maximum);
    } else {
      for (uint expert = 0; expert < experts; ++expert) total += exp(logits[expert] - maximum);
    }
    for (uint slot = 0; slot < selected; ++slot) scores[slot] = exp(logits[indices[slot]] - maximum) / total;
  }
}

PUBLIC KERNEL(Qwen38MoeIndices)(
    const device uint* selected_experts, device uint* gate_indices, device uint* down_indices,
    constant uint& intermediate, constant uint& hidden, constant uint& selected, constant uint& rows,
    constant uint& index_count, const uint i AXIS(index_count, 128)
) {
  if (i < rows * selected * intermediate) gate_indices[i] = selected_experts[i / intermediate] * intermediate + i % intermediate;
  if (i < rows * selected * hidden) down_indices[i] = selected_experts[i / hidden] * hidden + i % hidden;
}

PUBLIC KERNEL(Qwen38MoeCombine)(
    const device bfloat* experts, const device float* scores,
    const device bfloat* shared, const device float* shared_gate, device bfloat* output,
    constant uint& hidden, constant uint& selected, constant uint& rows,
    const uint i AXIS(rows * hidden, 128)
) {
  const uint row = i / hidden, column = i % hidden;
  float sum = 0.0f;
  for (uint slot = 0; slot < selected; ++slot) sum += scores[row * selected + slot] * float(experts[(row * selected + slot) * hidden + column]);
  const float gate = 1.0f / (1.0f + exp(-shared_gate[row]));
  sum += gate * float(shared[i]);
  output[i] = bfloat(sum);
}
