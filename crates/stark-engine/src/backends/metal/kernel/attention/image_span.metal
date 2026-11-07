#include <metal_stdlib>
#include <metal_simdgroup>
#include "../common/dsl.h"
#include "../common/thread_context.h"

// Transformers v4.53.3 Gemma3 same-image OR mask; online, linear memory.
template <typename T, uint HEAD_DIM>
VARIANTS(T, float, bfloat)
VARIANTS(HEAD_DIM, 64, 128, 256, 512)
PUBLIC KERNEL(AttentionImageSpan)(
    const device T* queries, const device T* keys, const device T* values, device T* output,
    const constant uint& gqa_factor, const constant uint& num_heads,
    const constant uint& suffix_length, const constant uint& prefix_length,
    const constant uint& ring_offset, const constant uint& has_ring,
    const constant uint& sliding_window_size, const constant uint& image_start,
    const constant uint& image_end, const constant float& scale,
    const ThreadContext thread_context,
    const uint group GROUPS((suffix_length * num_heads).div_ceil(4)),
    const uint tid THREADS(128)
) {
  const uint row_head = group * 4 + thread_context.simdgroup_index;
  if (row_head >= suffix_length * num_heads) return;
  constexpr uint channels = HEAD_DIM / 32;
  const uint row = row_head / num_heads, head = row_head % num_heads;
  const uint lane = thread_context.simd_lane_id;
  const uint kv_stride = (num_heads / gqa_factor) * HEAD_DIM;
  const uint kv_head = (head / gqa_factor) * HEAD_DIM;
  const uint q_base = (head * suffix_length + row) * HEAD_DIM;
  const uint query_position = prefix_length + row;
  float query[channels], accumulator[channels];
  for (uint dim = 0; dim < channels; dim++) {
    query[dim] = float(queries[q_base + lane + dim * 32]);
    accumulator[dim] = 0;
  }
  float maximum = -INFINITY, denominator = 0;
  for (uint key = 0; key < prefix_length + suffix_length; key++) {
    const uint key_position = (has_ring != 0 && key < prefix_length)
        ? (prefix_length + key - ring_offset) % prefix_length : key;
    const bool causal = key_position <= query_position
        && (sliding_window_size == 0 || query_position - key_position < sliding_window_size);
    const bool same_image = key >= prefix_length && row >= image_start && row < image_end
        && key - prefix_length >= image_start && key - prefix_length < image_end;
    if (!causal && !same_image) continue;
    const uint kv_base = key * kv_stride + kv_head;
    float score = 0;
    for (uint dim = 0; dim < channels; dim++)
      score += query[dim] * float(keys[kv_base + lane + dim * 32]);
    score = simd_sum(score) * scale;
    const float next_max = max(maximum, score);
    const float prior_factor = fast::exp(maximum - next_max), factor = fast::exp(score - next_max);
    denominator = denominator * prior_factor + factor;
    for (uint dim = 0; dim < channels; dim++)
      accumulator[dim] = accumulator[dim] * prior_factor + factor * float(values[kv_base + lane + dim * 32]);
    maximum = next_max;
  }
  const uint out_base = (row * num_heads + head) * HEAD_DIM;
  for (uint dim = 0; dim < channels; dim++)
    output[out_base + lane + dim * 32] = T(accumulator[dim] / denominator);
}
