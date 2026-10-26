#include <metal_stdlib>
#include "../common/dsl.h"

// The pinned vision path materializes two FP32 multiplies, then adds them.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

PUBLIC KERNEL(VisionPrepareBF16)(
    const device bfloat* qkv,
    device bfloat* queries,
    device bfloat* keys,
    device bfloat* values,
    const device float* cosines,
    const device float* sines,
    constant uint& rows,
    constant uint& num_heads,
    constant uint& head_dim,
    const uint index AXIS(rows * num_heads * head_dim, 128)
) {
  const uint model_width = num_heads * head_dim;
  const uint row = index / model_width;
  const uint head = (index % model_width) / head_dim;
  const uint channel = index % head_dim;
  const uint paired = channel < head_dim / 2 ? channel + head_dim / 2 : channel - head_dim / 2;
  const uint source = row * 3 * model_width + head * head_dim;
  const float cosine = cosines[row * head_dim + channel];
  const float sine = sines[row * head_dim + channel];
  const float query = float(qkv[source + channel]);
  const float key = float(qkv[source + model_width + channel]);
  const float pair_sign = channel < head_dim / 2 ? -1.0f : 1.0f;
  const float query_pair = pair_sign * float(qkv[source + paired]);
  const float key_pair = pair_sign * float(qkv[source + model_width + paired]);
  const float query_first = query * cosine;
  const float query_second = query_pair * sine;
  const float key_first = key * cosine;
  const float key_second = key_pair * sine;
  queries[head * rows * head_dim + row * head_dim + channel] = bfloat(query_first + query_second);
  keys[index] = bfloat(key_first + key_second);
  values[index] = qkv[source + 2 * model_width + channel];
}
