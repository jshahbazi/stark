#include "../../common/dsl.h"
#include "../../common/thread_context.h"

using namespace metal;

// Match the unsigned BF16 affine arithmetic in common/quant_unpack.h exactly,
// including the separately rounded scale/16 used for the high nibble. The
// caller admits only whole G64 groups and bounds the complete output buffer.
template <typename T>
VARIANTS(T, bfloat)
KERNEL(MaterializeAffineQ4)(
    const device uint8_t* weights,
    const device T* scales,
    const device T* biases,
    device T* output,
    const constant uint& pair_count,
    const constant uint& input_dim,
    const constant uint& scale_output_stride,
    const constant uint& scale_group_stride,
    const constant uint& threadgroup_count,
    const uint group_index GROUPS(threadgroup_count),
    const uint thread_index THREADS(256),
    const ThreadContext thread_context
) {
  (void)group_index;
  (void)thread_index;
  const uint local_thread = thread_context.simdgroup_index * thread_context.simdgroup_size + thread_context.simd_lane_id;
  const uint index = thread_context.threadgroup_position.x * 256u + local_thread;
  if (index >= pair_count) {
    return;
  }
  const uint packed_columns = input_dim / 2u;
  const uint row = index / packed_columns;
  const uint group = ((index % packed_columns) * 2u) / 64u;
  const uint metadata_index = row * scale_output_stride + group * scale_group_stride;
  const T scale = scales[metadata_index];
  const T bias = biases[metadata_index];
  const T s0 = scale;
  const T s1 = scale / static_cast<T>(16.0f);
  const uint8_t word = weights[index];
  output[2u * index] = s0 * (word & 0x0f) + bias;
  output[2u * index + 1u] = s1 * (word & 0xf0) + bias;
}
