#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;

#define BLOCK_SIZE 1024

// Source-only arithmetic is adapted from MLX v0.32.3 rms_norm.metal,
// 64ea011cb65f14d9ce2737e60db9a4ae91ed7441, Copyright 2024 Apple Inc.
// The MIT notice is preserved in LICENSE.mlx-reference.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(on)
METAL_FUNC float source_rms_four(float accum, float x0, float x1, float x2, float x3) {
  accum += x0*x0;
  accum += x1*x1;
  accum += x2*x2;
  accum += x3*x3;
  return accum;
}
METAL_FUNC float source_inverse_rms(float sum, uint count, float epsilon) {
  const float mean = sum / float(count);
  return metal::precise::rsqrt(mean + epsilon);
}
#pragma METAL fp math_mode(fast)
#pragma METAL fp contract(on)

// TODO: Are numerics of subtract_mean fine?

template <typename InputT, typename AffineT, typename OutputT, typename AccumT>
VARIANTS(InputT, float, half, bfloat)
VARIANTS(AffineT, float, half, bfloat)
VARIANTS(OutputT, float, half, bfloat)
VARIANTS(AccumT, float)
PUBLIC KERNEL(Normalization)(
    const device InputT* input OPTIONAL(!in_place),
    const device AffineT* scales OPTIONAL(has_scales),
    const device AffineT* biases OPTIONAL(has_biases),
    device OutputT* output,
    device InputT* shortcut OPTIONAL(copy_to_shortcut),
    const device int32_t* hadamard_factors OPTIONAL(use_hadamard),
    constant uint& batch_size,
    constant uint& element_count,
    constant float& epsilon,
    constant float& scale_offset,
    constant float& post_layer_scalar,
    const bool in_place SPECIALIZE,
    const bool subtract_mean SPECIALIZE,
    const bool full_layer SPECIALIZE,
    const bool copy_to_shortcut SPECIALIZE,
    const bool residual_add SPECIALIZE,
    const bool use_hadamard SPECIALIZE,
    const bool scale_residual_sum SPECIALIZE,
    const bool scale_output SPECIALIZE,
    const bool has_biases SPECIALIZE,
    const bool has_scales SPECIALIZE,
    const bool mlx_bf16_reduction SPECIALIZE,
    threadgroup AccumT shared_sum[METAL_SIMD_SIZE],
    const ThreadContext thread_context,
    const uint batch_idx GROUPS(batch_size),
    const uint thread_in_row THREADS(BLOCK_SIZE)
) {
  if (in_place) {
    input = reinterpret_cast<const device InputT*>(output);
  }

  const uint batch_offset = batch_idx * element_count;
  input += batch_offset;
  output += batch_offset;
  if (copy_to_shortcut) {
    shortcut += batch_offset;
  }

  // Step 1 - threads read from global and accumulate sum / sum of squares
  AccumT thread_sum = static_cast<AccumT>(0.0f);
  AccumT thread_sum_of_squares = static_cast<AccumT>(0.0f);

  if (mlx_bf16_reduction) {
    // Four contiguous values form each source lane subtotal. Inactive groups
    // supply zero to the same 32-lane final reduction as the source kernel.
    for (uint base = thread_in_row * 4; base < element_count; base += BLOCK_SIZE * 4) {
      float x[4] = {0.0f,0.0f,0.0f,0.0f};
      for (uint j = 0; j < 4; ++j) {
        const uint i = base + j;
        if (i < element_count) {
          InputT val = input[i];
          if (copy_to_shortcut) {
            if (residual_add) {
              val += shortcut[i];
              if (scale_residual_sum) val = InputT(float(val) * post_layer_scalar);
            }
            shortcut[i] = val;
          }
          x[j] = float(val);
        }
      }
      thread_sum_of_squares = source_rms_four(thread_sum_of_squares,x[0],x[1],x[2],x[3]);
    }
  } else {
  for (uint i = thread_in_row; i < element_count; i += BLOCK_SIZE) {
    InputT val = input[i];
    // We can also fuse:
    // - TensorCopy (copy_to_shortcut)
    // - TensorAddSwap (copy_to_shortcut + residual_add)
    // Normalization in TensorAddSwap fusion mode operates on input + shortcut
    if (copy_to_shortcut) {
      if (residual_add) {
        val += shortcut[i];
        if (scale_residual_sum) {
          val = static_cast<InputT>(static_cast<float>(val) * post_layer_scalar);
        }
      }
      shortcut[i] = val;
    }
    AccumT val_accum_t = static_cast<AccumT>(val);
    if (subtract_mean) {
      thread_sum += val_accum_t;
    }
    thread_sum_of_squares += val_accum_t * val_accum_t;
  }

  }

  // Step 2 - threads reduce their partial sums / sums of squares
  AccumT total_sum;
  if (subtract_mean) {
    total_sum =
        threadgroup_cooperative_reduce<SimdReduceSum<AccumT>, BLOCK_SIZE>(thread_sum, shared_sum, thread_context);
  }
  AccumT total_sum_of_squares = threadgroup_cooperative_reduce<SimdReduceSum<AccumT>, BLOCK_SIZE>(
      thread_sum_of_squares,
      shared_sum,
      thread_context
  );

  // And calculate mean/var/rms_inv from it
  AccumT mean = static_cast<AccumT>(0.0f);
  if (subtract_mean) {
    mean = total_sum / static_cast<AccumT>(element_count);
  }

  AccumT var = total_sum_of_squares / static_cast<AccumT>(element_count) - mean * mean;

  AccumT rms_inv = mlx_bf16_reduction
      ? source_inverse_rms(total_sum_of_squares,element_count,epsilon)
      : rsqrt(var + static_cast<AccumT>(epsilon));

  // Step 3 - elementwise normalization
  for (uint i = thread_in_row; i < element_count; i += BLOCK_SIZE) {
    AccumT x;
    // If we fuse TensorAddSwap, read shortcut (that now has input + shortcut)
    // No need for memory barrier because each thread only reads what it wrote
    if (residual_add) {
      x = static_cast<AccumT>(shortcut[i]);
    } else {
      x = static_cast<AccumT>(input[i]);
    }

    // If full_layer, normalize and scale in AccumT, cast to OutputT at the end
    // If not, cast to OutputT after normalize, scale in OutputT
    OutputT val;
    if (has_scales) {
      AccumT scale = static_cast<AccumT>(scales[i]) + static_cast<AccumT>(scale_offset);
      if (full_layer) {
        val = static_cast<OutputT>((x - mean) * rms_inv * scale);
      } else {
        val = static_cast<OutputT>((x - mean) * rms_inv) * static_cast<OutputT>(scale);
      }
    } else {
      val = static_cast<OutputT>((x - mean) * rms_inv);
    }

    if (has_biases) {
      val = static_cast<OutputT>(static_cast<AccumT>(val) + static_cast<AccumT>(biases[i]));
    }

    if (use_hadamard) {
      val = static_cast<OutputT>(simdgroup_input_random_hadamard_transform(
          static_cast<ushort>(thread_in_row % METAL_SIMD_SIZE),
          val,
          hadamard_factors[i]
      ));
    }

    if (scale_output) {
      val = static_cast<OutputT>(static_cast<float>(val) * post_layer_scalar);
    }

    output[i] = val;
  }
}
