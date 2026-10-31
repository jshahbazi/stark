// Modified by Stark: in-process core extraction; see UPSTREAM.md.
//! Affine Q4/Q8 kernels with BF16 activations and F32 accumulation.
//! Q4 packs eight low-nibble-first codes per u32; `w = scale*q + bias`.

use anyhow::{Result, ensure};

use crate::kernels::gemm::gemm_bf16_nt;
use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};
use crate::weights::QuantWeights;

const SOURCE: &str = include_str!("metal/quant.metal");
const OFFICIAL_SOURCE: &str = include_str!("metal/official_gate.metal");

#[cfg(test)]
#[path = "official_gate_tests.rs"]
mod official_gate_tests;

/// Rows each thread of the Q4 gather GEMV accumulates.
const Q4_GEMV_ROWS: usize = 2;

/// Row-tile height and execution-simdgroup count of a grouped Q4 GEMM.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MoeTile {
    T32Sg4,
    T64Sg4,
}

impl MoeTile {
    /// Row-tile height T — the block map's granularity.
    pub fn rows(self) -> usize {
        match self {
            Self::T32Sg4 => 32,
            Self::T64Sg4 => 64,
        }
    }

    /// Tensor-op simdgroups; dispatch uses `32 * simdgroups` threads.
    pub fn simdgroups(self) -> usize {
        match self {
            Self::T32Sg4 | Self::T64Sg4 => 4,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::T32Sg4 => "32x4",
            Self::T64Sg4 => "64x4",
        }
    }

    fn nax_kernel(self) -> &'static str {
        match self {
            Self::T32Sg4 => "gemm_q4_nt_nax_grouped_t32x4",
            Self::T64Sg4 => "gemm_q4_nt_nax_grouped_t64x4",
        }
    }
}

/// Row count at which grouped Q4 GEMM switches to the 64x4 tile.
pub const MOE_TILE_MROUTE_DEFAULT: usize = 3072;

/// Largest prefill chunk routed to grouped GEMV instead of tiled GEMM.
const MOE_SMALLM_THRESHOLD: usize = 8;

// The small-M route is limited by per-expert register capacity.
const _: () = assert!(MOE_SMALLM_THRESHOLD > 0);
const _: () = assert!(MOE_SMALLM_THRESHOLD <= crate::kernels::moe::MOE_SMALLM_MAX_M);

/// Grouped GEMM or direct small-M GEMV execution.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MoeRoute {
    Grouped(MoeTile),
    SmallmGemv,
}

impl MoeRoute {
    pub fn label(self) -> &'static str {
        match self {
            Self::Grouped(tile) => tile.label(),
            Self::SmallmGemv => "gemv",
        }
    }
}

/// Fixed per-chunk MoE routing policy.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MoeTilePolicy;

impl MoeTilePolicy {
    /// Selects small-M through 8 rows, 64x4 from 3072, otherwise 32x4.
    pub fn route_for(&self, chunk_m: usize) -> MoeRoute {
        if chunk_m > 0 && chunk_m <= MOE_SMALLM_THRESHOLD {
            MoeRoute::SmallmGemv
        } else {
            MoeRoute::Grouped(self.tile_for(chunk_m))
        }
    }

    fn tile_for(&self, chunk_m: usize) -> MoeTile {
        if chunk_m >= MOE_TILE_MROUTE_DEFAULT {
            MoeTile::T64Sg4
        } else {
            MoeTile::T32Sg4
        }
    }

    /// Upper bound for grouped block-map rows.
    pub fn alloc_tile(&self) -> MoeTile {
        MoeTile::T32Sg4
    }
}

/// Returns the fixed chunk-routing policy.
pub fn moe_tile_mroute_table() -> MoeTilePolicy {
    MoeTilePolicy
}

/// Validates a quantized weight for the GEMV kernels; returns `(out, in)`.
pub(crate) fn check_quant(w: &QuantWeights) -> Result<(usize, usize)> {
    let (n, k) = (w.out_features(), w.in_features());
    ensure!(matches!(w.bits, 4 | 8), "unsupported bit width {}", w.bits);
    // A uint4 block must not cross a quantization group.
    ensure!(
        k.is_multiple_of(w.group_size) && w.group_size.is_multiple_of(128 / w.bits),
        "in_features {k} / group size {} not block-packable at {} bits",
        w.group_size,
        w.bits
    );
    ensure!(w.codes.dtype() == DType::U32, "codes must be U32");
    // uint4 code loads require 16-byte alignment.
    ensure!(
        w.codes.binding().1.is_multiple_of(16),
        "codes byte offset {} not 16-aligned; the kernels read codes as uint4",
        w.codes.binding().1
    );
    ensure!(
        w.scales.shape() == [n, k / w.group_size]
            && w.biases.shape() == [n, k / w.group_size],
        "scales/biases shape mismatch for [{n}, {k}] gs={}",
        w.group_size
    );
    Ok((n, k))
}

pub fn gemv_quant(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    x: &Tensor,
    y: &Tensor,
) -> Result<()> {
    if w.bits == 16 {
        return dense_bf16_project(ctx, pass, x, &w.codes, y);
    }
    let (n, k) = check_quant(w)?;
    ensure!(x.numel() == k, "x numel {} != K {k}", x.numel());
    ensure!(y.numel() == n, "y numel {} != N {n}", y.numel());
    let requested_rows =
        if w.bits == 4 && y.dtype() == DType::BF16 { Q4_GEMV_ROWS } else { 1 };
    let packed_rows = if n.is_multiple_of(requested_rows) { requested_rows } else { 1 };
    let fn_name = match (w.bits, y.dtype(), packed_rows) {
        (4, DType::BF16, 1) => "gemv_q4_bf16",
        (4, DType::BF16, 2) => "gemv_q4_bf16_2row",
        (4, DType::F32, 1) => "gemv_q4_bf16_f32out",
        (8, DType::BF16, 1) => "gemv_q8_bf16",
        (8, DType::F32, 1) => "gemv_q8_bf16_f32out",
        _ => anyhow::bail!("gemv output must be BF16 or F32"),
    };
    let pipeline = ctx.pipeline(fn_name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            x.binding(),
            y.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(w.group_size)],
        Grid::Threadgroups { groups: (n / packed_rows, 1, 1), threadgroup: (32, 1, 1) },
    )
}

/// Q4 two-row GDN input projection with its four-tap causal convolution.
/// The complete BF16 projection is preserved in `raw_out`; only its first
/// `conv_out.numel()` rows additionally update the convolution window/output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemv_q4_gdn_conv_step(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    x: &Tensor,
    raw_out: &Tensor,
    window: &Tensor,
    conv_w: &Tensor,
    conv_out: &Tensor,
) -> Result<()> {
    let (n, k) = check_quant(w)?;
    let c = conv_out.numel();
    ensure!(
        w.bits == 4 && n.is_multiple_of(2),
        "fused GDN projection requires Q4 row pairs"
    );
    ensure!(c > 0 && c <= n, "invalid GDN convolution channel count");
    ensure!(
        x.numel() == k && raw_out.numel() == n,
        "fused GDN projection size mismatch"
    );
    ensure!(
        conv_w.shape() == [4, c] && window.numel() == 3 * c,
        "fused GDN convolution requires four taps"
    );
    ensure!(
        [x, raw_out, window, conv_w, conv_out].iter().all(|t| t.dtype() == DType::BF16),
        "fused GDN projection/convolution expects BF16"
    );
    let pipeline =
        ctx.pipeline("gemv_q4_bf16_2row_gdn_conv", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            x.binding(),
            raw_out.binding(),
            window.binding(),
            conv_w.binding(),
            conv_out.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(w.group_size), &u32_bytes(c), &u32_bytes(4)],
        Grid::Threadgroups { groups: (n / 2, 1, 1), threadgroup: (32, 1, 1) },
    )
}

/// Writes `out[N, K] = dequant(W)` in bf16.
fn dequant_to_bf16(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    out: &Tensor,
) -> Result<()> {
    let (n, k) = check_quant(w)?;
    ensure!(
        out.shape() == [n, k] && out.dtype() == DType::BF16,
        "dequant out must be BF16 [{n}, {k}], got {:?} {:?}",
        out.shape(),
        out.dtype()
    );
    let fn_name = if w.bits == 4 { "dequant_q4_bf16" } else { "dequant_q8_bf16" };
    let pipeline = ctx.pipeline(fn_name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[w.codes.binding(), w.scales.binding(), w.biases.binding(), out.binding()],
        &[&u32_bytes(k), &u32_bytes(w.group_size)],
        Grid::Threads { grid: (k / (32 / w.bits), n, 1), threadgroup: (32, 1, 1) },
    )
}

/// Dequantizes `W` into bf16 scratch, then computes
/// `C[M, N] = A[M, K] . W[N, K]^T`.
pub fn gemm_quant_bf16_nt(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    scratch: &Tensor,
) -> Result<()> {
    if w.bits == 16 {
        return dense_bf16_project(ctx, pass, a, &w.codes, c);
    }
    let (n, k) = check_quant(w)?;
    ensure!(
        scratch.numel() >= n * k && scratch.dtype() == DType::BF16,
        "dequant scratch holds {} bf16 elements, needs {}",
        scratch.numel(),
        n * k
    );
    let staged = scratch.view(0, &[n, k])?;
    dequant_to_bf16(ctx, pass, w, &staged)?;
    // The GEMM reads the staged dequant, and the next projection's dequant
    // must not overwrite the staging while it does: both edges get barriers
    // (no-ops on a serial encoder; the staged path is only taken for chunks
    // past the skinny kernels' row limit, where the GEMMs fill the GPU).
    pass.level_barrier(&[&staged])?;
    gemm_bf16_nt(ctx, pass, a, &staged, c)?;
    pass.level_barrier(&[c])
}

/// Exact BF16 gate weights from the official MLX pack, with F32 accumulation.
/// This narrow projection path leaves every affine Q4/Q8 route unchanged.
pub(crate) fn dense_bf16_project(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    w: &Tensor,
    y: &Tensor,
) -> Result<()> {
    ensure!(
        w.dtype() == DType::BF16 && w.shape().len() == 2 && x.dtype() == DType::BF16,
        "dense BF16 projection dtypes"
    );
    let (n, k) = (w.shape()[0], w.shape()[1]);
    ensure!(k > 0 && x.numel().is_multiple_of(k), "dense BF16 projection input");
    let m = x.numel() / k;
    ensure!(m > 0 && y.numel() == m * n, "dense BF16 projection output");
    let name = match y.dtype() {
        DType::BF16 => "official_gate_bf16",
        DType::F32 => "official_gate_f32",
        _ => anyhow::bail!("dense gate output dtype"),
    };
    let pipeline = ctx.pipeline(name, OFFICIAL_SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), w.binding(), y.binding()],
        &[&u32_bytes(k), &u32_bytes(n)],
        Grid::Threadgroups { groups: (n, m, 1), threadgroup: (32, 1, 1) },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn hc_dense_bf16_inject(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    hyper: &Tensor,
    norm: &Tensor,
    w: &QuantWeights,
    out: &Tensor,
    h: usize,
    groups: usize,
    eps: f32,
    bias: f32,
) -> Result<()> {
    let k = h * groups;
    ensure!(
        groups > 0
            && groups <= 8
            && h > 0
            && w.bits == 16
            && w.in_features() == k
            && w.codes.dtype() == DType::BF16
            && hyper.dtype() == DType::BF16
            && norm.dtype() == DType::BF16
            && norm.numel() == k
            && hyper.numel().is_multiple_of(k),
        "HC BF16 gate geometry"
    );
    let rows = hyper.numel() / k;
    let n = w.out_features();
    ensure!(
        rows > 0 && out.dtype() == DType::BF16 && out.numel() == rows * n,
        "HC BF16 gate output"
    );
    let pipeline =
        ctx.pipeline("official_hc_inject", OFFICIAL_SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[hyper.binding(), norm.binding(), w.codes.binding(), out.binding()],
        &[
            &u32_bytes(h),
            &u32_bytes(groups),
            &u32_bytes(n),
            &eps.to_ne_bytes(),
            &bias.to_ne_bytes(),
        ],
        Grid::Threadgroups { groups: (n, rows, 1), threadgroup: (32 * groups, 1, 1) },
    )
}

/// Block-mapped grouped GEMM with Q4 dequantization fused into B-tile staging.
/// `tile` must match the configuration used to build the block map.
#[allow(clippy::too_many_arguments)]
pub fn gemm_q4_grouped_nt(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &QuantWeights,
    c: &Tensor,
    blocks: &Tensor,
    n_blocks: usize,
    tile: MoeTile,
) -> Result<()> {
    let (_, k) = check_quant(w)?;
    ensure!(
        w.bits == 4 && w.group_size == 64 && k.is_multiple_of(64),
        "grouped NAX GEMM needs 4-bit group-64 weights and K % 64 == 0"
    );
    let (s, ak) = (a.shape()[0], a.shape()[1]);
    ensure!(ak == k, "A K {ak} != weight K {k}");
    let n = c.shape()[1];
    ensure!(c.shape() == [s, n], "C shape {:?} != [{s}, {n}]", c.shape());
    ensure!(n.is_multiple_of(64), "grouped GEMM needs N % 64 == 0 (got {n})");
    ensure!(blocks.dtype() == DType::U32, "block map must be U32");
    ensure!(blocks.numel() >= n_blocks * 4 && n_blocks > 0, "bad block map");
    let pipeline = ctx.pipeline(tile.nax_kernel(), SOURCE, MslVersion::V4_0)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            a.binding(),
            c.binding(),
            blocks.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(n), &u32_bytes(w.group_size)],
        Grid::Threadgroups {
            groups: (n_blocks, 1, 1),
            threadgroup: (32 * tile.simdgroups(), 1, 1),
        },
    )
}

/// `out[M, K]` = dequantized embedding rows for `ids` (batched prefill
/// gather), from a 4-bit table or an 8-bit one (`--q8-embed`).
pub fn gather_rows_quant(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    ids: &Tensor,
    out: &Tensor,
) -> Result<()> {
    let (_, k) = check_quant(w)?;
    let m = ids.numel();
    ensure!(ids.dtype() == DType::U32, "ids must be U32");
    ensure!(
        out.numel() == m * k && out.dtype() == DType::BF16,
        "gather out must be BF16 [{m}, {k}]"
    );
    let fn_name =
        if w.bits == 4 { "gather_rows_q4_bf16" } else { "gather_rows_q8_bf16" };
    let pipeline = ctx.pipeline(fn_name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            ids.binding(),
            out.binding(),
        ],
        &[&u32_bytes(k), &u32_bytes(w.group_size)],
        // One thread per code word: 32 / bits elements each.
        Grid::Threads { grid: (k / (32 / w.bits), m, 1), threadgroup: (32, 1, 1) },
    )
}

/// The physical sampler/scorer buffer stays F32. Official external-MLX head
/// values are BF16-rounded after accumulation; Lily/reference values remain F32.
pub const OFFICIAL_HEAD_VALUE_POLICY: &str = "f32_accumulation_bf16_values_f32_buffer";

const HEAD_SOURCE: &str = concat!(
    include_str!("metal/quant.metal"),
    include_str!("metal/official_head.metal"),
);

pub(crate) fn round_logits_bf16_values(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
) -> Result<()> {
    ensure!(
        logits.dtype() == DType::F32
            && logits.numel() > 0
            && logits.numel() <= u32::MAX as usize,
        "head readout must be a nonempty bounded F32 buffer"
    );
    let pipeline = ctx.pipeline_with_math_mode(
        "official_head_round_bf16_values_f32",
        HEAD_SOURCE,
        MslVersion::V3_1,
        crate::metal::MslMathMode::Safe,
    )?;
    pass.dispatch_at(
        &pipeline,
        &[logits.binding()],
        &[&u32_bytes(logits.numel())],
        Grid::Threads { grid: (logits.numel(), 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// Stock QMV uses signed-int byte row offsets. Bound the complete Q4 code
/// span before selecting its unchecked fast loop; larger valid heads retain
/// the original GEMV fallback. K2560 also makes its 512-value loop exact.
fn official_head_stock_qmv_fast_supported(n: usize, k: usize) -> bool {
    k == 2560
        && n > 0
        && n.is_multiple_of(8)
        && n.checked_mul(k / 2)
            .is_some_and(|code_bytes| code_bytes <= i32::MAX as usize)
}

/// Explicit LM-head policy. Ordinary aligned Q4/group64/H2560 decode uses stock
/// QMV affine expressions/reduction geometry with a fused BF16-value store;
/// valid other quantization/storage geometries retain their projection and add
/// a separate in-place output rounding pass. No weight bytes are changed.
pub fn gemv_quant_logits(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    x: &Tensor,
    y: &Tensor,
    bf16_values: bool,
) -> Result<()> {
    if !bf16_values {
        return gemv_quant(ctx, pass, w, x, y);
    }
    ensure!(
        w.codes.shape().len() == 2 && matches!(w.bits, 4 | 8 | 16),
        "head storage shape/quantization"
    );
    let (n, k) = (w.out_features(), w.in_features());
    ensure!(
        n > 0
            && k > 0
            && x.dtype() == DType::BF16
            && x.numel() == k
            && y.dtype() == DType::F32
            && y.numel() == n,
        "head activation/output dtype or geometry"
    );
    if w.bits != 16 {
        ensure!(
            w.group_size > 0
                && w.scales.dtype() == DType::BF16
                && w.biases.dtype() == DType::BF16,
            "head affine metadata must be BF16 with a positive group size"
        );
        check_quant(w)?;
    }
    if w.bits == 4 && w.group_size == 64 && official_head_stock_qmv_fast_supported(n, k)
    {
        let pipeline = ctx.pipeline_with_math_mode(
            "official_head_qmv_fast_q4_g64_bf16_values_f32",
            HEAD_SOURCE,
            MslVersion::V4_0,
            crate::metal::MslMathMode::Safe,
        )?;
        if pipeline.thread_execution_width() != 32 {
            gemv_quant(ctx, pass, w, x, y)?;
            pass.level_barrier(&[y])?;
            return round_logits_bf16_values(ctx, pass, y);
        }
        return pass.dispatch_at(
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
        );
    }
    gemv_quant(ctx, pass, w, x, y)?;
    pass.level_barrier(&[y])?;
    round_logits_bf16_values(ctx, pass, y)
}

#[cfg(test)]
#[path = "official_head_precision_tests.rs"]
mod official_head_precision_tests;
