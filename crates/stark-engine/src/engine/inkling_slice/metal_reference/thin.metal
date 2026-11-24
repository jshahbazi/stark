// SPDX-License-Identifier: Apache-2.0
#include <metal_stdlib>
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_thin_nax.h"
instantiate_kernel("steel_gemm_thin_nax_nt_bfloat16_um2_un2_sm1_sn1_ks4", gemm_thin_nax, bfloat, 2, 2, 1, 1, 4, true)
instantiate_kernel("steel_gemm_thin_nax_nt_bfloat16_um2_un2_sm1_sn2_ks2", gemm_thin_nax, bfloat, 2, 2, 1, 2, 2, true)
