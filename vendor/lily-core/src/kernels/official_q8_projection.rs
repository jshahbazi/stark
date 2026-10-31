// SPDX-License-Identifier: Apache-2.0
//! Scoped external-MLX GDN decode projection; all other routes delegate.

use anyhow::{Result, ensure};

use crate::kernels::{quant, u32_bytes};
use crate::metal::{ComputePass, Grid, MetalContext, MslMathMode, MslVersion};
use crate::tensor::{DType, Tensor};
use crate::weights::QuantWeights;

const SOURCE: &str = include_str!("metal/official_q8_gdn_decode.metal");

/// The stock helper uses signed int expressions for row/stride addressing.
/// Bound code bytes, metadata elements, vector/output strides and the
/// pre-division K*bytes_per_pack expression before dispatch.
fn stock_signed_offsets_fit(n: usize, k: usize) -> bool {
    let cap = i32::MAX as usize;
    n > 0
        && k > 0
        && n <= cap
        && k <= cap
        && k.checked_mul(4).is_some_and(|v| v <= cap)
        && n.checked_mul(k).is_some_and(|v| v <= cap)
        && n.checked_mul(k / 64).is_some_and(|v| v <= cap)
}

/// Only the two row-joined ordinary GDN decode shapes qualified by the private
/// stock oracle. No prefill/batch/attention caller changes in this stage.
pub(crate) fn gemv_gdn_decode(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    x: &Tensor,
    y: &Tensor,
    official_mlx_rounding: bool,
) -> Result<()> {
    if !official_mlx_rounding
        || w.bits != 8
        || w.group_size != 64
        || w.codes.shape().len() != 2
        || x.dtype() != DType::BF16
        || y.dtype() != DType::BF16
        || w.scales.dtype() != DType::BF16
        || w.biases.dtype() != DType::BF16
    {
        return quant::gemv_quant(ctx, pass, w, x, y);
    }
    let (n, k) = (w.out_features(), w.in_features());
    if !matches!((n, k), (16480, 2560) | (2560, 6144))
        || !stock_signed_offsets_fit(n, k)
    {
        return quant::gemv_quant(ctx, pass, w, x, y);
    }
    quant::check_quant(w)?;
    ensure!(x.numel() == k && y.numel() == n, "official Q8 decode geometry");
    let pipeline = ctx.pipeline_with_math_mode(
        "official_q8_gdn_decode_qmv_fast_g64_bf16",
        SOURCE,
        MslVersion::V4_0,
        MslMathMode::Safe,
    )?;
    if pipeline.thread_execution_width() != 32 {
        return quant::gemv_quant(ctx, pass, w, x, y);
    }
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            x.binding(),
            y.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n)],
        Grid::Threadgroups { groups: (1, n / 8, 1), threadgroup: (32, 2, 1) },
    )
}

#[cfg(test)]
#[path = "official_q8_projection_tests.rs"]
mod tests;
