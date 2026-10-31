// Modified by Stark: in-process core extraction; see UPSTREAM.md.
//! Normalization kernels: weighted RMSNorm (row-wise) with fp32 reduction.

use anyhow::{Result, ensure};

use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};

/// Threads per row-reduction threadgroup.
const TG: usize = 256;
const RESIDUAL_H: usize = 2048;

const SOURCE: &str = include_str!("metal/norm.metal");

/// Fuses BF16-rounded `x += b` with `out = rmsnorm(x) * (w_bias + w)`.
/// The normalized result may differ from separate kernels by one BF16 ULP.
#[allow(clippy::too_many_arguments)]
pub fn add_rmsnorm_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    b: &Tensor,
    w: &Tensor,
    out: &Tensor,
    eps: f32,
    w_bias: f32,
) -> Result<()> {
    let h =
        *x.shape().last().ok_or_else(|| anyhow::anyhow!("rmsnorm on 0-d tensor"))?;
    let m = x.numel() / h;
    ensure!(b.numel() == x.numel(), "addend numel mismatch");
    ensure!(h == RESIDUAL_H, "residual RMSNorm requires H={RESIDUAL_H}, got {h}");
    ensure!(w.numel() == h, "weight numel {} != H {h}", w.numel());
    ensure!(out.numel() == x.numel(), "output numel mismatch");
    for t in [x, b, w, out] {
        ensure!(t.dtype() == DType::BF16, "rmsnorm expects BF16, got {:?}", t.dtype());
    }
    let pipeline = ctx.pipeline("add_rmsnorm_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), b.binding(), w.binding(), out.binding()],
        &[&eps.to_ne_bytes(), &w_bias.to_ne_bytes()],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (TG, 1, 1) },
    )
}

/// RMSNorm over the last dimension with gain `w_bias + w`.
pub fn rmsnorm_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    w: &Tensor,
    out: &Tensor,
    eps: f32,
    w_bias: f32,
) -> Result<()> {
    let h =
        *x.shape().last().ok_or_else(|| anyhow::anyhow!("rmsnorm on 0-d tensor"))?;
    let m = x.numel() / h;
    ensure!(w.numel() == h, "weight numel {} != H {h}", w.numel());
    ensure!(out.numel() == x.numel(), "output numel mismatch");
    for t in [x, w, out] {
        ensure!(t.dtype() == DType::BF16, "rmsnorm expects BF16, got {:?}", t.dtype());
    }
    let pipeline = ctx.pipeline("rmsnorm_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), w.binding(), out.binding()],
        &[&u32_bytes(h), &eps.to_ne_bytes(), &w_bias.to_ne_bytes()],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (TG, 1, 1) },
    )
}

/// LayerNorm over the last dimension with weight and bias, statistics in
/// f32 (two-pass): `out = (x - mean) / sqrt(var + eps) * w + b`.
pub fn layernorm_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    out: &Tensor,
    eps: f32,
) -> Result<()> {
    let h =
        *x.shape().last().ok_or_else(|| anyhow::anyhow!("layernorm on 0-d tensor"))?;
    let m = x.numel() / h;
    ensure!(w.numel() == h, "weight numel {} != H {h}", w.numel());
    ensure!(b.numel() == h, "bias numel {} != H {h}", b.numel());
    ensure!(out.numel() == x.numel(), "output numel mismatch");
    for t in [x, w, b, out] {
        ensure!(
            t.dtype() == DType::BF16,
            "layernorm expects BF16, got {:?}",
            t.dtype()
        );
    }
    let pipeline = ctx.pipeline("layernorm_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), w.binding(), b.binding(), out.binding()],
        &[&u32_bytes(h), &eps.to_ne_bytes()],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (TG, 1, 1) },
    )
}

const OFFICIAL_SOURCE: &str = include_str!("metal/official_norm.metal");

/// Ordinary BF16 RMS with the admitted external MLX operation policy.
///
/// MLX fast RMS with BF16 gains rounds the normalized activation before gain.
/// Its F32-gain primitive promotes the output to F32; this BF16-only wrapper
/// rejects that case rather than silently narrowing it. Select this policy from
/// Qwen4ExpConfig::official_mlx_rounding(), never from folded gains alone.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_bf16_with_rounding(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    w: &Tensor,
    out: &Tensor,
    eps: f32,
    w_bias: f32,
    official_mlx: bool,
) -> Result<()> {
    if !official_mlx {
        return rmsnorm_bf16(ctx, pass, x, w, out, eps, w_bias);
    }
    ensure!(w_bias == 0.0, "official ordinary RMS requires folded gain");
    for t in [x, w, out] {
        ensure!(
            t.dtype() == DType::BF16,
            "official ordinary RMS requires BF16 activations, gain and output"
        );
    }
    let h = *x
        .shape()
        .last()
        .ok_or_else(|| anyhow::anyhow!("ordinary RMS on 0-d tensor"))?;
    ensure!(
        (1..=4096).contains(&h),
        "official ordinary RMS single-row width outside 1..=4096"
    );
    ensure!(
        x.numel() > 0 && x.numel().is_multiple_of(h),
        "official ordinary RMS input shape"
    );
    ensure!(
        w.shape() == [h].as_slice(),
        "official ordinary RMS gain must be one vector"
    );
    ensure!(out.numel() == x.numel(), "official ordinary RMS output size");
    ensure!(eps.is_finite() && eps >= 0.0, "official ordinary RMS epsilon");
    let rows = x.numel() / h;
    // Pinned rms_single_row N_READS=4; 32-lane SIMD groups, at most32 groups.
    let threads = h.div_ceil(4).div_ceil(32) * 32;
    let pipeline =
        ctx.pipeline("rmsnorm_bf16_mlx", OFFICIAL_SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), w.binding(), out.binding()],
        &[&u32_bytes(h), &eps.to_ne_bytes()],
        Grid::Threadgroups { groups: (rows, 1, 1), threadgroup: (threads, 1, 1) },
    )
}

#[cfg(test)]
#[path = "official_rms_rounding_tests.rs"]
mod official_rounding_tests;
