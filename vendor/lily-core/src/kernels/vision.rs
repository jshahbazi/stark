// Modified by Stark: in-process core extraction; see UPSTREAM.md.
//! The vision tower's kernels: the position-embedding blend, the 2-D rotary
//! over fused qkv rows, and full bidirectional attention over one image
//! (`docs/architecture.md`, "The vision tower"; the chain is
//! `tools/reference/VISION.md`, "The tower").

use anyhow::{Result, ensure};

use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion, Param};
use crate::tensor::{DType, Tensor};

pub const SOURCE: &str = include_str!("metal/vision.metal");

/// Head dimension the attention kernel is compiled for.
pub const ATTN_D: usize = 72;
/// Query rows per threadgroup of the attention kernel.
const ATTN_BQ: usize = 32;
const ATTN_THREADS: usize = 128;

/// Source-compatible contractions for the measured retained tower geometry.
/// Other admitted geometries retain the original dense GEMM. Each dispatch
/// covers at most32 input rows and147456 output elements, with no new buffers.
pub(crate) fn linear_bf16_bias(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    out: &Tensor,
    streams: usize,
) -> Result<()> {
    ensure!(
        input.shape().len() == 2 && weight.shape().len() == 2,
        "vision linear rank"
    );
    let (m, k, n) = (input.shape()[0], input.shape()[1], weight.shape()[0]);
    let elements = m
        .checked_mul(n)
        .ok_or_else(|| anyhow::anyhow!("vision linear element overflow"))?;
    ensure!(m > 0 && m <= 8192 && k > 0 && n > 0, "vision linear row bounds");
    ensure!(
        weight.shape() == [n, k] && bias.numel() == n && out.numel() == elements,
        "vision linear shape"
    );
    for tensor in [input, weight, bias, out] {
        ensure!(tensor.dtype() == DType::BF16, "vision linear dtype");
    }
    let measured = matches!(
        (k, n, streams),
        (1536, 1152, 4)
            | (1152, 3456, 32)
            | (1152, 1152, 32)
            | (1152, 4304, 32)
            | (4304, 1152, 32)
            | (4608, 4608, 32)
            | (4608, 2560, 32)
    );
    if !measured {
        return crate::kernels::gemm::gemm_bf16_nt_bias(
            ctx, pass, input, weight, bias, out,
        );
    }
    let pipeline = ctx.pipeline(
        "vision_contraction_bf16_bias",
        include_str!("metal/vision_contraction.metal"),
        MslVersion::V4_0,
    )?;
    for row in (0..m).step_by(32) {
        let rows = (m - row).min(32);
        let input_tile = input.view(
            row.checked_mul(k)
                .ok_or_else(|| anyhow::anyhow!("vision linear input offset"))?,
            &[rows, k],
        )?;
        let output_tile = out.view(
            row.checked_mul(n)
                .ok_or_else(|| anyhow::anyhow!("vision linear output offset"))?,
            &[rows, n],
        )?;
        let work = rows
            .checked_mul(n)
            .ok_or_else(|| anyhow::anyhow!("vision linear tile overflow"))?;
        ensure!(work <= 147456, "vision linear tile work bound");
        pass.dispatch_with(
            &pipeline,
            &[
                input_tile.binding(),
                weight.binding(),
                bias.binding(),
                output_tile.binding(),
            ],
            &[
                Param::U32(rows as u32),
                Param::U32(n as u32),
                Param::U32(k as u32),
                Param::U32(streams as u32),
            ],
            Grid::Threads { grid: (work, 1, 1), threadgroup: (128, 1, 1) },
        )?;
    }
    Ok(())
}

/// Checks that `(gh, gw)` is a grid of `n` patches in 2 x 2 merge blocks.
fn check_grid(n: usize, gh: usize, gw: usize) -> Result<()> {
    ensure!(
        gh >= 2 && gw >= 2 && gh.is_multiple_of(2) && gw.is_multiple_of(2),
        "grid ({gh}, {gw}) is not made of 2 x 2 merge blocks"
    );
    ensure!(
        gh * gw == n,
        "grid ({gh}, {gw}) holds {} patches, tensor has {n}",
        gh * gw
    );
    Ok(())
}

/// In place over the patch embedding `x` `[N, H]`: adds the `side x side`
/// position `table` `[side^2, H]` resampled bilinearly (align_corners) to the
/// `(gh, gw)` grid, rows in the block-major patch order. Taps and weights are
/// computed in the kernel from the patch's grid coordinates.
pub fn patch_pos_embed_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    table: &Tensor,
    gh: usize,
    gw: usize,
) -> Result<()> {
    ensure!(x.shape().len() == 2, "x must be [N, H], got {:?}", x.shape());
    let (n, h) = (x.shape()[0], x.shape()[1]);
    check_grid(n, gh, gw)?;
    ensure!(
        table.shape().len() == 2 && table.shape()[1] == h,
        "position table {:?} is not [side^2, {h}]",
        table.shape()
    );
    let side = (table.shape()[0] as f64).sqrt() as usize;
    ensure!(side * side == table.shape()[0], "position table rows are not a square");
    for t in [x, table] {
        ensure!(t.dtype() == DType::BF16, "patch_pos_embed expects BF16");
    }
    let pipeline = ctx.pipeline("vision_patch_pos_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), table.binding()],
        &[&u32_bytes(h), &u32_bytes(gh), &u32_bytes(gw), &u32_bytes(side)],
        Grid::Threads { grid: (h, n, 1), threadgroup: (256.min(h), 1, 1) },
    )
}

/// In place over the fused `qkv` `[N, 3H]` projection: applies the 2-D
/// rotary (`heads` heads of `H / heads`, angles from the patch's absolute
/// grid coordinates, `theta`) to the q and k thirds in f32 with
/// `rotate_half` pairing, results back to bf16.
pub fn qkv_rope_2d_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    heads: usize,
    gh: usize,
    gw: usize,
    theta: f32,
) -> Result<()> {
    ensure!(qkv.shape().len() == 2, "qkv must be [N, 3H], got {:?}", qkv.shape());
    let (n, h3) = (qkv.shape()[0], qkv.shape()[1]);
    ensure!(h3.is_multiple_of(3), "qkv width {h3} is not 3H");
    let h = h3 / 3;
    check_grid(n, gh, gw)?;
    ensure!(
        heads > 0 && h.is_multiple_of(heads),
        "H {h} not a multiple of {heads} heads"
    );
    let d = h / heads;
    ensure!(d.is_multiple_of(4), "head dim {d} must split into rotate_half quarters");
    ensure!(qkv.dtype() == DType::BF16, "rope expects BF16");
    let pipeline = ctx.pipeline("vision_qkv_rope_bf16", SOURCE, MslVersion::V3_1)?;
    // One thread per rotate_half pair of q and of k: 2 * H / 2.
    let pairs = h;
    pass.dispatch_with(
        &pipeline,
        &[qkv.binding()],
        &[
            Param::U32(h as u32),
            Param::U32(d as u32),
            Param::U32(gw as u32),
            Param::F32(theta),
        ],
        Grid::Threads { grid: (pairs, n, 1), threadgroup: (256.min(pairs), 1, 1) },
    )
}

/// Full bidirectional attention over all `N` rows of `qkv` `[N, 3H]`
/// (q | k | v, `heads` heads of [`ATTN_D`]), softmax in f32, into `out`
/// `[N, H]`. One tensor-op threadgroup per 32-query tile and head, walking
/// 64-key tiles twice. QK and scaled scores round to BF16; softmax statistics
/// are F32; globally normalized probabilities round to BF16 before PV, matching
/// the pinned eager source without allocating an N x N attention matrix.
pub fn attention_full_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    out: &Tensor,
    heads: usize,
    scale: f32,
) -> Result<()> {
    ensure!(qkv.shape().len() == 2, "qkv must be [N, 3H], got {:?}", qkv.shape());
    let (n, h3) = (qkv.shape()[0], qkv.shape()[1]);
    ensure!(h3.is_multiple_of(3), "qkv width {h3} is not 3H");
    let h = h3 / 3;
    ensure!(n > 0, "attention over zero rows");
    ensure!(
        heads * ATTN_D == h,
        "H {h} is not {heads} heads of the compiled head dim {ATTN_D}"
    );
    ensure!(out.numel() == n * h, "out numel {} != [{n}, {h}]", out.numel());
    ensure!(
        qkv.dtype() == DType::BF16 && out.dtype() == DType::BF16,
        "attention expects BF16"
    );
    let pipeline = ctx.pipeline("vision_attn_nax_d72", SOURCE, MslVersion::V4_0)?;
    pass.dispatch_with(
        &pipeline,
        &[qkv.binding(), out.binding()],
        &[Param::U32(n as u32), Param::U32(h as u32), Param::F32(scale)],
        Grid::Threadgroups {
            groups: (n.div_ceil(ATTN_BQ), heads, 1),
            threadgroup: (ATTN_THREADS, 1, 1),
        },
    )
}

/// Test-only same-input observations over the production arithmetic body.
#[cfg(test)]
pub(crate) fn attention_stages_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    out: &Tensor,
    stages: &Tensor,
    heads: usize,
    scale: f32,
) -> Result<()> {
    const TRACE_SOURCE: &str = concat!(
        include_str!("metal/vision.metal"),
        "\n",
        include_str!("metal/vision_trace.metal")
    );
    let n = qkv.shape()[0];
    let h = qkv.shape()[1] / 3;
    ensure!(n <= 256 && heads * ATTN_D == h, "private attention trace shape");
    ensure!(
        stages.dtype() == DType::F32 && stages.numel() == 3 * heads * n * n,
        "private attention trace size"
    );
    let pipeline =
        ctx.pipeline("vision_attn_trace_nax_d72", TRACE_SOURCE, MslVersion::V4_0)?;
    pass.dispatch_with(
        &pipeline,
        &[qkv.binding(), out.binding(), stages.binding()],
        &[Param::U32(n as u32), Param::U32(h as u32), Param::F32(scale)],
        Grid::Threadgroups {
            groups: (n.div_ceil(ATTN_BQ), heads, 1),
            threadgroup: (ATTN_THREADS, 1, 1),
        },
    )
}

#[cfg(test)]
pub(crate) fn contraction_reference_bf16_bias(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    out: &Tensor,
    streams: usize,
) -> Result<()> {
    const REFERENCE_SOURCE: &str =
        include_str!("metal/vision_contraction_reference.metal");
    let (m, k, n) = (input.shape()[0], input.shape()[1], weight.shape()[0]);
    ensure!(
        m <= 256 && k <= 4608 && n <= 4608 && matches!(streams, 4 | 32),
        "private contraction reference bounds"
    );
    ensure!(
        weight.shape() == [n, k] && bias.numel() == n && out.numel() == m * n,
        "private contraction reference shape"
    );
    for tensor in [input, weight, bias, out] {
        ensure!(tensor.dtype() == DType::BF16, "private contraction reference dtype");
    }
    let pipeline = ctx.pipeline(
        "vision_contraction_reference_bf16_bias",
        REFERENCE_SOURCE,
        MslVersion::V4_0,
    )?;
    pass.dispatch_with(
        &pipeline,
        &[input.binding(), weight.binding(), bias.binding(), out.binding()],
        &[
            Param::U32(m as u32),
            Param::U32(n as u32),
            Param::U32(k as u32),
            Param::U32(streams as u32),
        ],
        Grid::Threads { grid: (m * n, 1, 1), threadgroup: (128, 1, 1) },
    )
}

#[cfg(test)]
pub(crate) fn layernorm_reference_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    out: &Tensor,
    eps: f32,
) -> Result<()> {
    let width = *input.shape().last().unwrap();
    let rows = input.numel() / width;
    ensure!(rows <= 256 && width <= 4608 && width > 0, "private norm reference bounds");
    let pipeline = ctx.pipeline(
        "vision_layernorm_reference_bf16",
        include_str!("metal/vision_nonlinear_reference.metal"),
        MslVersion::V4_0,
    )?;
    pass.dispatch_with(
        &pipeline,
        &[input.binding(), weight.binding(), bias.binding(), out.binding()],
        &[Param::U32(rows as u32), Param::U32(width as u32), Param::F32(eps)],
        Grid::Threads { grid: (rows, 1, 1), threadgroup: (32, 1, 1) },
    )
}

#[cfg(test)]
pub(crate) fn gelu_reference_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    values: &Tensor,
    tanh: bool,
) -> Result<()> {
    ensure!(
        values.numel() <= 256 * 4608 && values.dtype() == DType::BF16,
        "private gelu reference bounds"
    );
    let pipeline = ctx.pipeline(
        "vision_gelu_reference_bf16",
        include_str!("metal/vision_nonlinear_reference.metal"),
        MslVersion::V4_0,
    )?;
    pass.dispatch_with(
        &pipeline,
        &[values.binding()],
        &[Param::U32(values.numel() as u32), Param::U32(u32::from(tanh))],
        Grid::Threads { grid: (values.numel(), 1, 1), threadgroup: (128, 1, 1) },
    )
}
