#include "../common/dsl.h"

// The source coefficient path follows MLX v0.32.3 rope.metal, revision
// 64ea011cb65f14d9ce2737e60db9a4ae91ed7441 (Apple Inc., MIT).
// Copyright (c) 2023-2024 Apple Inc. License retained in LICENSE.mlx-reference.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
METAL_FUNC float source_yarn_frequency(
    float extra,
    uint32_t pair,
    const device float* parameters
) {
  const float inter = parameters[0] * extra;
  const float shifted = float(pair) - parameters[1];
  const float span = parameters[2] - parameters[1];
  const float ramp = metal::clamp(shifted / span, 0.0f, 1.0f);
  const float mask = 1.0f - ramp;
  const float product = inter * extra;
  const float weighted_inter = inter * mask;
  const float complement = 1.0f - mask;
  const float weighted_extra = extra * complement;
  const float denominator = weighted_inter + weighted_extra;
  return product / denominator;
}
#pragma METAL fp contract(on)
METAL_FUNC float source_custom_rope_frequency(
    float base,
    uint32_t pair,
    uint32_t rope_dim,
    uint32_t rotary_dim,
    bool yarn,
    const device float* yarn_parameters
) {
  const float fraction = float(pair) / float(rope_dim / 2);
  // MLX's precompiled array-power kernel uses -fno-fast-math. The Metal
  // standard library chooses its default pow implementation at preprocessing,
  // so a function-scoped Safe pragma alone does not select precise pow here.
  float frequency = 2 * pair < rotary_dim ? metal::precise::pow(base, fraction) : INFINITY;
  if (yarn) {
    frequency = source_yarn_frequency(frequency, pair, yarn_parameters);
  }
  return frequency;
}

template <typename ElementT>
METAL_FUNC ElementT apply_source_rope(
    const device ElementT* head,
    uint32_t position,
    uint32_t head_dim_idx,
    uint32_t rope_dim,
    float base,
    uint32_t rotary_dim,
    bool proportional,
    bool yarn,
    const device float* yarn_parameters
) {
  const uint32_t half_dim = rope_dim / 2;
  const uint32_t pair = head_dim_idx % half_dim;
  const float fraction = float(pair) / float(half_dim);
  float inverse_frequency;
  if (proportional) {
    const float frequency = source_custom_rope_frequency(base, pair, rope_dim, rotary_dim, yarn, yarn_parameters);
    inverse_frequency = 1.0f / frequency;
  } else {
    inverse_frequency = metal::exp2(-fraction * base);
  }
  const float theta = float(position) * inverse_frequency;
  const float cosine = metal::fast::cos(theta);
  const float sine = metal::fast::sin(theta);
  float x1 = float(head[pair]);
  float x2 = float(head[pair + half_dim]);
  if (yarn) {
    // The source weak BF16 scalar multiplies/stores the input before RoPE.
    volatile ElementT first = ElementT(x1 * yarn_parameters[3]);
    volatile ElementT second = ElementT(x2 * yarn_parameters[3]);
    x1 = float(first);
    x2 = float(second);
  }
  // Preserve the source operand order as well as its final BF16 boundary.
  const float value = head_dim_idx < half_dim ? x1 * cosine - x2 * sine : x1 * sine + x2 * cosine;
  return ElementT(value);
}

// Private, aggregate-only coefficient inspection used by ignored unit controls.
KERNEL(SourceRoPEFrequencyProbe)(
    device float* output,
    const device float* yarn_parameters OPTIONAL(source_yarn),
    constant float& base,
    constant uint& rope_dim,
    constant uint& rotary_dim,
    const bool source_yarn SPECIALIZE,
    const uint pair AXIS(rope_dim / 2, 256)
) {
  output[pair] = source_custom_rope_frequency(base, pair, rope_dim, rotary_dim, source_yarn, yarn_parameters);
}
#pragma METAL fp math_mode(fast)
#pragma METAL fp contract(on)

// TODO: very ugly and wasteful, clean up and optimize

template <typename ElementT, typename RopeT>
METAL_FUNC ElementT apply_rope(
    const device ElementT* head,
    const device RopeT* cosines,
    const device RopeT* sines,
    uint32_t batch_idx,
    uint32_t head_dim_idx,
    uint32_t rope_dim,
    bool reference_bf16
) {
  const uint32_t half_rope_dim = rope_dim / 2;
  const uint32_t paired_idx =
      head_dim_idx < half_rope_dim ? head_dim_idx + half_rope_dim : head_dim_idx - half_rope_dim;
  const float input = float(head[head_dim_idx]);
  const float paired = float(head[paired_idx]);
  const float signed_paired = head_dim_idx < half_rope_dim ? -paired : paired;
  const float cos_val = float(cosines[batch_idx * rope_dim + head_dim_idx]);
  const float sin_val = float(sines[batch_idx * rope_dim + head_dim_idx]);

  if (reference_bf16) {
    // Explicit BF16 storage preserves the source dtype boundaries even when
    // the ordinary attention path is compiled with contraction enabled.
    const bfloat cosine = bfloat(cos_val);
    const bfloat sine = bfloat(sin_val);
    volatile bfloat first = bfloat(input * float(cosine));
    volatile bfloat second = bfloat(signed_paired * float(sine));
    return static_cast<ElementT>(float(first) + float(second));
  }

  return static_cast<ElementT>(input * cos_val + signed_paired * sin_val);
}

// Cache-only final-layer preparation keeps the original QKVG row stride and
// head coordinates. Query lanes return before loading, RoPE or storing. K/V
// expressions and the original exact dispatch extents are retained.
template <typename ElementT, typename RopeT>
VARIANTS(ElementT, bfloat)
VARIANTS(RopeT, float)
PUBLIC KERNEL(AttentionPrepareKVOnly) (
  const device ElementT* qkvg,
  device ElementT* keys,
  device ElementT* values,
  const device RopeT* cosines OPTIONAL(has_rope),
  const device RopeT* sines OPTIONAL(has_rope),
  const device uint32_t* source_positions OPTIONAL(source_rope),
  const constant float& source_base OPTIONAL(source_rope),
  const constant uint32_t& source_rotary_dim OPTIONAL(source_rope),
  const device float* source_yarn_parameters OPTIONAL(source_yarn),
  const constant uint32_t& num_q_heads,
  const constant uint32_t& num_kv_heads,
  const constant uint32_t& head_dim,
  const constant uint32_t& rope_dim OPTIONAL(has_rope),
  const constant uint32_t& kv_token_offset,
  const constant uint32_t& input_row_stride,
  const constant uint32_t& batch_dim,
  const bool has_rope SPECIALIZE,
  const bool reference_bf16 SPECIALIZE,
  const bool source_rope SPECIALIZE,
  const bool source_proportional SPECIALIZE,
  const bool source_yarn SPECIALIZE,
  const uint32_t head_dim_idx AXIS(head_dim, 128),
  const uint32_t head_idx AXIS(num_q_heads + num_kv_heads * 2, 1),
  const uint32_t batch_idx AXIS(batch_dim, 1)
) {
  if (head_idx < num_q_heads) {
    return;
  }
  const uint32_t qkvg_head_idx = batch_idx * input_row_stride + head_idx * head_dim;
  const device ElementT* qkvg_head = qkvg + qkvg_head_idx;
  const bool is_key = head_idx < num_q_heads + num_kv_heads;

  ElementT element = qkvg_head[head_dim_idx];
  if (has_rope && head_dim_idx < rope_dim && is_key) {
    element = source_rope
        ? apply_source_rope(qkvg_head, source_positions[batch_idx], head_dim_idx, rope_dim,
                            source_base, source_rotary_dim, source_proportional, source_yarn, source_yarn_parameters)
        : apply_rope(qkvg_head, cosines, sines, batch_idx, head_dim_idx, rope_dim, reference_bf16);
  }
  if (is_key) {
    const uint32_t k_idx =
        (kv_token_offset + batch_idx) * num_kv_heads * head_dim + (head_idx - num_q_heads) * head_dim + head_dim_idx;
    keys[k_idx] = element;
  } else {
    const uint32_t v_idx = (kv_token_offset + batch_idx) * num_kv_heads * head_dim +
                           (head_idx - num_q_heads - num_kv_heads) * head_dim + head_dim_idx;
    values[v_idx] = element;
  }
}

template <typename ElementT, typename RopeT>
VARIANTS(ElementT, bfloat)
VARIANTS(RopeT, float)
PUBLIC KERNEL(AttentionPrepare) (
  const device ElementT* qkvg, // [token, input_row_stride], with an optional trailing gate
  device ElementT* queries, // [head_idx, token, head_dim]
  device ElementT* keys OPTIONAL(has_kv), // [(kv_token_offset + token), head_idx, head_dim]
  device ElementT* values OPTIONAL(has_kv), // [(kv_token_offset + token), head_idx, head_dim]
  const device RopeT* cosines OPTIONAL(has_rope),
  const device RopeT* sines OPTIONAL(has_rope),
  const device uint32_t* source_positions OPTIONAL(source_rope),
  const constant float& source_base OPTIONAL(source_rope),
  const constant uint32_t& source_rotary_dim OPTIONAL(source_rope),
  const device float* source_yarn_parameters OPTIONAL(source_yarn),
  const constant uint32_t& num_q_heads,
  const constant uint32_t& num_kv_heads OPTIONAL(has_kv),
  const constant uint32_t& head_dim,
  const constant uint32_t& rope_dim OPTIONAL(has_rope),
  const constant uint32_t& kv_token_offset OPTIONAL(has_kv),
  const constant uint32_t& input_row_stride,
  const constant uint32_t& batch_dim,
  const bool has_kv SPECIALIZE,
  const bool has_rope SPECIALIZE,
  const bool reference_bf16 SPECIALIZE,
  const bool source_rope SPECIALIZE,
  const bool source_proportional SPECIALIZE,
  const bool source_yarn SPECIALIZE,
  const uint32_t head_dim_idx AXIS(head_dim, 128),
  const uint32_t head_idx AXIS(num_q_heads + num_kv_heads.unwrap_or(0) * 2, 1),
  const uint32_t batch_idx AXIS(batch_dim, 1)
) {
  const uint32_t qkvg_head_idx = batch_idx * input_row_stride + head_idx * head_dim;
  const device ElementT* qkvg_head = qkvg + qkvg_head_idx;
  const bool is_query = !has_kv || head_idx < num_q_heads;
  const bool is_key = has_kv && head_idx >= num_q_heads && head_idx < num_q_heads + num_kv_heads;

  ElementT element = qkvg_head[head_dim_idx];
  if (has_rope && head_dim_idx < rope_dim && (is_query || is_key)) {
    element = source_rope
        ? apply_source_rope(qkvg_head, source_positions[batch_idx], head_dim_idx, rope_dim,
                            source_base, source_rotary_dim, source_proportional, source_yarn, source_yarn_parameters)
        : apply_rope(qkvg_head, cosines, sines, batch_idx, head_dim_idx, rope_dim, reference_bf16);
  }

  if (is_query) {
    const uint32_t q_idx = head_idx * batch_dim * head_dim + batch_idx * head_dim + head_dim_idx;
    queries[q_idx] = element;
  } else if (is_key) {
    const uint32_t k_idx =
        (kv_token_offset + batch_idx) * num_kv_heads * head_dim + (head_idx - num_q_heads) * head_dim + head_dim_idx;

    keys[k_idx] = element;
  } else if (has_kv) {
    const uint32_t v_idx = (kv_token_offset + batch_idx) * num_kv_heads * head_dim +
                           (head_idx - num_q_heads - num_kv_heads) * head_dim + head_dim_idx;

    values[v_idx] = element;
  }
}
