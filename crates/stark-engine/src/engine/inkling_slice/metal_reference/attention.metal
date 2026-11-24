// SPDX-License-Identifier: Apache-2.0
#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"
instantiate_kernel("steel_attention_float32_bq64_bk32_bd128_bv128_wm4_wn1_maskfloat32", attention_nax, float, 64, 32, 128, 128, 4, 1, float, float)
#include "../global_relative.metal"
