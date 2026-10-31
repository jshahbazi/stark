// Modified by Stark: in-process core extraction; see UPSTREAM.md.
//! Gated DeltaNet recurrence, causal depthwise conv1d, and gated RMSNorm.

use anyhow::{Result, ensure};

use crate::kernels::{Pos, u32_bytes};
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion, Param};
use crate::tensor::{DType, Tensor};

/// Compiled GDN key/value head dimension.
pub const GDN_HEAD_DIM: usize = 128;

/// Prefill conv1d tile width; must match `CONV_TILE` in gdn.metal.
pub const CONV1D_PREFILL_TILE: usize = 64;

const SOURCE: &str = include_str!("metal/gdn.metal");

/// Recurrent state is F32 because rounding compounds across tokens.
pub const GDN_STATE_DTYPE: DType = DType::F32;

/// Activation applied to the output gate `z` inside the gated RMSNorm.
/// Qwen3.5 uses SiLU; Qwen3.8-Flash-Next sets `output_gate_type = "sigmoid"`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GdnGate {
    Silu = 0,
    Sigmoid = 1,
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_step_gated_fused(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    z: &Tensor,
    norm_w: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
    eps: f32,
    gate_act: GdnGate,
) -> Result<()> {
    let num_heads = a.numel();
    let dim = GDN_HEAD_DIM;
    ensure!(
        num_k_heads > 0 && num_heads.is_multiple_of(num_k_heads),
        "v-heads {num_heads} not a multiple of k-heads {num_k_heads}"
    );
    let k_part = num_k_heads * dim;
    let part = num_heads * dim;
    ensure!(qkv.numel() == 2 * k_part + part, "bad fused qkv size");
    ensure!(qkv.dtype() == DType::BF16, "qkv must be BF16");
    ensure!(state.numel() == num_heads * dim * dim, "bad state size");
    ensure!(state.dtype() == DType::F32, "GDN state must be F32");
    ensure!(z.numel() == part && z.dtype() == DType::BF16, "z must be BF16 [H, dim]");
    ensure!(
        norm_w.numel() == dim && norm_w.dtype() == DType::F32,
        "norm_w must be F32 [dim]"
    );
    ensure!(out.numel() == part && out.dtype() == DType::BF16, "bad output size");
    gdn_step_gated_named(
        ctx,
        pass,
        "gdn_step_gated",
        qkv,
        a,
        b,
        a_log,
        dt_bias,
        state,
        z,
        norm_w,
        out,
        scale,
        num_k_heads,
        eps,
        gate_act,
    )
}

/// [`gdn_step_gated_fused`] through the kernel `name` (a variant with the
/// shipped kernel's bindings and grid, for the timing harness).
#[allow(clippy::too_many_arguments)]
pub fn gdn_step_gated_named(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    name: &'static str,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    z: &Tensor,
    norm_w: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
    eps: f32,
    gate_act: GdnGate,
) -> Result<()> {
    let num_heads = a.numel();
    let dim = GDN_HEAD_DIM;
    let vpk = num_heads / num_k_heads;
    let k_part = num_k_heads * dim;
    let elem = qkv.dtype().size();
    let (qkv_buf, qkv_off) = qkv.binding();
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            (qkv_buf, qkv_off),
            (qkv_buf, qkv_off + k_part * elem),
            (qkv_buf, qkv_off + 2 * k_part * elem),
            a.binding(),
            b.binding(),
            a_log.binding(),
            dt_bias.binding(),
            state.binding(),
            z.binding(),
            norm_w.binding(),
            out.binding(),
        ],
        &[
            &scale.to_ne_bytes(),
            &u32_bytes(vpk),
            &eps.to_ne_bytes(),
            &u32_bytes(gate_act as usize),
        ],
        Grid::Threadgroups { groups: (num_heads, 1, 1), threadgroup: (dim, 1, 1) },
    )
}

/// Staging for normalized q/k rows and precomputed gates.
pub struct GdnRegscanStaging<'a> {
    /// BF16 `[M, 2*HK*128]` normalized q then k rows.
    pub qk_norm: &'a Tensor,
    /// F32 `[M, H]` decay gates.
    pub decay: &'a Tensor,
    /// F32 `[M, H]` sigmoid(b) gates.
    pub beta: &'a Tensor,
    /// The chunked scan's staging; without it every prefill takes the
    /// token-serial scan.
    pub chunk: Option<GdnChunkStaging<'a>>,
}

/// Tokens per chunk of the chunked prefill scan (`GDN_CHUNK` in gdn.metal).
pub const GDN_CHUNK: usize = 64;
/// State columns per `gdn_chunk_scan` threadgroup (`GDN_CHUNK_NB`).
pub const GDN_CHUNK_NB: usize = 16;
/// Rows from which a prefill takes the chunked scan; shorter batches (the
/// verify passes) stay on the token-serial scan.
pub const GDN_CHUNK_MIN_ROWS: usize = 128;

/// Per-chunk WY staging the chunked scan writes in its first pass and reads
/// in its second: [`gdn_chunk_staging_shapes`] gives the tensors' shapes.
pub struct GdnChunkStaging<'a> {
    /// BF16 `[M, H, 128]` W = T diag(beta exp(G)) K.
    pub w: &'a Tensor,
    /// BF16 `[M, H, 128]` U = T diag(beta) V.
    pub u: &'a Tensor,
    /// BF16 `[M, H, 64]` causal, decayed Q K^T within the chunk.
    pub p: &'a Tensor,
    /// F32 `[M, H]` cumulative log-decay within the chunk.
    pub g: &'a Tensor,
}

/// Shapes and dtypes of the chunk staging tensors (`w`, `u`, `p`, `g`) for
/// `m` rows and `num_heads` value heads.
pub fn gdn_chunk_staging_shapes(
    m: usize,
    num_heads: usize,
) -> [(Vec<usize>, DType); 4] {
    [
        (vec![m, num_heads, GDN_HEAD_DIM], DType::BF16),
        (vec![m, num_heads, GDN_HEAD_DIM], DType::BF16),
        (vec![m, num_heads, GDN_CHUNK], DType::BF16),
        (vec![m, num_heads], DType::F32),
    ]
}

/// L2-normalizes q/k into BF16 `[M, 2*HK*128]`; q is pre-scaled.
#[allow(clippy::too_many_arguments)]
fn gdn_qk_l2norm(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    qk_norm: &Tensor,
    scale: f32,
    num_k_heads: usize,
    num_heads: usize,
) -> Result<()> {
    let dim = GDN_HEAD_DIM;
    ensure!(
        num_k_heads > 0 && num_heads > 0 && num_heads.is_multiple_of(num_k_heads),
        "v-heads {num_heads} not a positive multiple of k-heads {num_k_heads}"
    );
    ensure!(qkv.dtype() == DType::BF16, "qkv must be BF16");
    let row_size = (2 * num_k_heads + num_heads) * dim;
    ensure!(qkv.numel().is_multiple_of(row_size), "qkv not [M, (2*HK+H)*{dim}]");
    let m = qkv.numel() / row_size;
    ensure!(m > 0, "qkv must contain at least one row");
    ensure!(
        qk_norm.numel() == m * 2 * num_k_heads * dim && qk_norm.dtype() == DType::BF16,
        "qk_norm must be BF16 [M, 2*HK*{dim}]"
    );

    let pipeline = ctx.pipeline("gdn_qk_l2norm", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[qkv.binding(), qk_norm.binding()],
        &[&scale.to_ne_bytes(), &u32_bytes(num_k_heads), &u32_bytes(num_heads)],
        Grid::Threads {
            grid: (32, num_k_heads, m),
            threadgroup: (32, 4.min(num_k_heads), 1),
        },
    )
}

/// Computes F32 decay and beta gates from BF16 `[M, H]` inputs.
#[allow(clippy::too_many_arguments)]
fn gdn_gates(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
) -> Result<()> {
    let num_heads = a_log.numel();
    ensure!(num_heads > 0, "a_log must contain at least one head");
    ensure!(a.dtype() == DType::BF16, "a must be BF16");
    ensure!(b.dtype() == DType::BF16, "b must be BF16");
    ensure!(a_log.dtype() == DType::F32, "a_log must be F32");
    ensure!(dt_bias.dtype() == DType::BF16, "dt_bias must be BF16");
    ensure!(a.numel().is_multiple_of(num_heads), "a not [M, H]");
    let n = a.numel();
    ensure!(n > 0, "a must contain at least one row");
    ensure!(b.numel() == n, "b not [M, H]");
    ensure!(dt_bias.numel() == num_heads, "dt_bias not [H]");
    ensure!(
        decay.numel() == n && decay.dtype() == DType::F32,
        "decay must be F32 [M, H]"
    );
    ensure!(beta.numel() == n && beta.dtype() == DType::F32, "beta must be F32 [M, H]");

    let pipeline = ctx.pipeline("gdn_gates", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            a.binding(),
            b.binding(),
            a_log.binding(),
            dt_bias.binding(),
            decay.binding(),
            beta.binding(),
        ],
        &[&u32_bytes(num_heads)],
        Grid::Threads { grid: (n, 1, 1), threadgroup: (256.min(n), 1, 1) },
    )
}

/// Scans `[q | k | v]` rows serially with F32 state held in registers.
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    staging: &GdnRegscanStaging<'_>,
    state: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
) -> Result<()> {
    gdn_prefill_mid(
        ctx,
        pass,
        qkv,
        a,
        b,
        a_log,
        dt_bias,
        staging,
        state,
        out,
        scale,
        num_k_heads,
        None,
    )
}

/// [`gdn_prefill`] that also records the running state after each of the
/// first `mid.shape()[0]` tokens into `mid` (F32 `[count, H, 128, 128]`), for
/// rolling back a speculative batch to an accepted prefix.
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_mid(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    staging: &GdnRegscanStaging<'_>,
    state: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
    mid: Option<&Tensor>,
) -> Result<()> {
    // `LILY_GDN_SCAN_KERNEL` names an alternative scan (timing experiments):
    // a token-serial kernel by name, or `gdn_chunk` for the chunked scan.
    static FORCED: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let forced = FORCED.get_or_init(|| None);
    let m = a.numel() / a_log.numel().max(1);
    let default = if mid.is_none() && staging.chunk.is_some() && m >= GDN_CHUNK_MIN_ROWS
    {
        GDN_CHUNK_SCAN
    } else {
        GDN_SCAN_KERNEL
    };
    let scan_name = forced.as_deref().unwrap_or(default);
    gdn_prefill_scan_named(
        ctx,
        pass,
        qkv,
        a,
        b,
        a_log,
        dt_bias,
        staging,
        state,
        out,
        scale,
        num_k_heads,
        mid,
        scan_name,
    )
}

/// The token-serial prefill scan kernel.
const GDN_SCAN_KERNEL: &str = "gdn_prefill_regscan";
/// The name that selects the chunked scan (`gdn_chunk_wy` + `gdn_chunk_scan`).
pub const GDN_CHUNK_SCAN: &str = "gdn_chunk";

/// [`gdn_prefill_mid`] through the scan kernel given by name (the shipped
/// scan takes four value columns per simdgroup; `_c1` / `_c2` are the
/// one- and two-column instantiations).
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_scan_named(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    staging: &GdnRegscanStaging<'_>,
    state: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
    mid: Option<&Tensor>,
    scan_name: &str,
) -> Result<()> {
    gdn_prefill_scan_named_with_rounding(
        ctx,
        pass,
        qkv,
        a,
        b,
        a_log,
        dt_bias,
        staging,
        state,
        out,
        scale,
        num_k_heads,
        mid,
        scan_name,
        false,
    )
}

/// The chunked scan: `gdn_chunk_wy` over every (chunk, key head), then
/// `gdn_chunk_scan` over every (column block, value head) walking the
/// chunks in order. Same operands, state and output as the serial scan.
#[allow(clippy::too_many_arguments)]
fn gdn_prefill_chunked(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    staging: &GdnRegscanStaging<'_>,
    chunk: &GdnChunkStaging<'_>,
    state: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
) -> Result<()> {
    let num_heads = a_log.numel();
    let dim = GDN_HEAD_DIM;
    let vpk = num_heads / num_k_heads;
    let m = a.numel() / num_heads;
    ensure!(
        m > 0 && m.is_multiple_of(GDN_CHUNK),
        "the chunked scan takes whole chunks of {GDN_CHUNK} rows, got {m}"
    );
    for ((t, name), (shape, dtype)) in
        [(chunk.w, "w"), (chunk.u, "u"), (chunk.p, "p"), (chunk.g, "g")]
            .into_iter()
            .zip(gdn_chunk_staging_shapes(m, num_heads))
    {
        ensure!(
            t.numel() >= shape.iter().product::<usize>() && t.dtype() == dtype,
            "chunk staging {name} must hold {dtype:?} {shape:?}, got {:?} {:?}",
            t.dtype(),
            t.shape()
        );
    }
    ensure!(
        dt_bias.dtype() == DType::BF16
            && a.dtype() == DType::BF16
            && b.dtype() == DType::BF16,
        "a, b and dt_bias must be BF16"
    );
    gdn_qk_l2norm(ctx, pass, qkv, staging.qk_norm, scale, num_k_heads, num_heads)?;
    pass.level_barrier(&[staging.qk_norm])?;
    // The three-heads-at-once pass covers up to three value heads per key
    // head; wider grouping takes the per-head pass.
    let (wy_name, wy_threads) =
        if vpk <= 3 { ("gdn_chunk_wy3", 256) } else { ("gdn_chunk_wy", 128) };
    let wy = ctx.pipeline(wy_name, SOURCE, MslVersion::V4_0)?;
    pass.dispatch_at(
        &wy,
        &[
            qkv.binding(),
            staging.qk_norm.binding(),
            a.binding(),
            b.binding(),
            a_log.binding(),
            dt_bias.binding(),
            chunk.w.binding(),
            chunk.u.binding(),
            chunk.p.binding(),
            chunk.g.binding(),
        ],
        &[&u32_bytes(m), &u32_bytes(num_heads), &u32_bytes(vpk)],
        Grid::Threadgroups {
            groups: (m.div_ceil(GDN_CHUNK), num_k_heads, 1),
            threadgroup: (wy_threads, 1, 1),
        },
    )?;
    pass.level_barrier(&[chunk.w, chunk.u, chunk.p, chunk.g])?;
    let scan = ctx.pipeline("gdn_chunk_scan", SOURCE, MslVersion::V4_0)?;
    pass.dispatch_at(
        &scan,
        &[
            staging.qk_norm.binding(),
            chunk.w.binding(),
            chunk.u.binding(),
            chunk.p.binding(),
            chunk.g.binding(),
            state.binding(),
            out.binding(),
        ],
        &[&u32_bytes(m), &u32_bytes(num_heads), &u32_bytes(vpk)],
        Grid::Threadgroups {
            groups: (dim / GDN_CHUNK_NB, num_heads, 1),
            threadgroup: (32 * 4, 1, 1),
        },
    )
}

/// Rewinds a conv window to the state after only the first `n` rows of `x`
/// followed `window_in`: `window_out` (`[C, S]`, may not alias `window_in`)
/// gets the last `S` entries of `window_in ++ x[..n]`. `x` is `[M, C]` with
/// `n <= M`; `S` is the window length (GDN: `KD-1`, PLE: `(KD-1)*dilation`).
pub fn conv_window_rollback<'t>(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    window_in: &Tensor,
    x: &Tensor,
    window_out: &Tensor,
    n: impl Into<Pos<'t>>,
) -> Result<()> {
    let n = n.into();
    ensure!(
        window_in.shape().len() == 2 && x.shape().len() == 2,
        "window must be [C, S] and x [M, C]"
    );
    let (c, s) = (window_in.shape()[0], window_in.shape()[1]);
    ensure!(
        window_out.shape() == [c, s],
        "window_out shape {:?} != [{c}, {s}]",
        window_out.shape()
    );
    ensure!(
        x.shape()[1] == c && n.max <= x.shape()[0],
        "x {:?} does not hold {} rows of {c} channels",
        x.shape(),
        n.max
    );
    for t in [window_in, x, window_out] {
        ensure!(t.dtype() == DType::BF16, "conv window rollback expects BF16");
    }
    let (in_buf, in_off) = window_in.binding();
    let (out_buf, out_off) = window_out.binding();
    ensure!(
        !(std::ptr::eq(in_buf, out_buf) && in_off == out_off),
        "window buffers must be distinct"
    );
    let pipeline =
        ctx.pipeline("conv_window_rollback_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_with(
        &pipeline,
        &[window_in.binding(), x.binding(), window_out.binding()],
        &[Param::U32(c as u32), Param::U32(s as u32), n.param()],
        Grid::Threads { grid: (c, 1, 1), threadgroup: (256.min(c), 1, 1) },
    )
}

/// Causal conv1d + SiLU over `[M, C]`; separate window buffers avoid tile races.
pub fn conv1d_prefill(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    window_in: &Tensor,
    window_out: &Tensor,
    x: &Tensor,
    w: &Tensor,
    out: &Tensor,
) -> Result<()> {
    ensure!(x.shape().len() == 2, "x must be [M, C]");
    let (m, c) = (x.shape()[0], x.shape()[1]);
    let kd = w.numel() / c;
    ensure!(w.numel() == kd * c, "weight numel {} not a multiple of C {c}", w.numel());
    ensure!((2..=9).contains(&kd), "conv kernel dim {kd} outside register window");
    for (name, t) in [("window_in", window_in), ("window_out", window_out)] {
        ensure!(t.numel() == c * (kd - 1), "{name} numel {} != C*(KD-1)", t.numel());
    }
    let (in_buf, in_off) = window_in.binding();
    let (out_buf, out_off) = window_out.binding();
    ensure!(
        !(std::ptr::eq(in_buf, out_buf) && in_off == out_off),
        "window buffers must be distinct (double-buffered)"
    );
    ensure!(out.numel() == m * c, "out numel {} != M*C", out.numel());
    let pipeline = ctx.pipeline("conv1d_prefill_bf16", SOURCE, MslVersion::V3_1)?;
    let tiles = m.div_ceil(CONV1D_PREFILL_TILE).max(1);
    pass.dispatch_at(
        &pipeline,
        &[
            window_in.binding(),
            window_out.binding(),
            x.binding(),
            w.binding(),
            out.binding(),
        ],
        &[&u32_bytes(c), &u32_bytes(kd), &u32_bytes(m)],
        Grid::Threads { grid: (c, tiles, 1), threadgroup: (256.min(c), 1, 1) },
    )
}

/// One causal conv1d + SiLU decode step. `window` (`[C, KD-1]`, bf16) is
/// shifted in place; `w` is tap-major `[KD, C]`.
pub fn conv1d_step(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    window: &Tensor,
    x: &Tensor,
    w: &Tensor,
    out: &Tensor,
) -> Result<()> {
    let c = x.numel();
    let kd = w.numel() / c;
    ensure!(w.numel() == kd * c, "weight numel {} not a multiple of C {c}", w.numel());
    ensure!(kd >= 2, "conv kernel dim {kd} < 2");
    ensure!(
        window.numel() == c * (kd - 1),
        "window numel {} != C*(KD-1)",
        window.numel()
    );
    ensure!(out.numel() == c, "out numel {} != C {c}", out.numel());
    let pipeline = ctx.pipeline("conv1d_step_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[window.binding(), x.binding(), w.binding(), out.binding()],
        &[&u32_bytes(c), &u32_bytes(kd)],
        Grid::Threads { grid: (c, 1, 1), threadgroup: (256.min(c), 1, 1) },
    )
}

/// Gated RMS norm over rows of size `d` with F32 weights.
#[allow(clippy::too_many_arguments)]
pub fn gated_rmsnorm(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    gate: &Tensor,
    w: &Tensor,
    out: &Tensor,
    eps: f32,
    gate_act: GdnGate,
) -> Result<()> {
    let d = w.numel();
    ensure!(w.dtype() == DType::F32, "gated_rmsnorm weight must be F32");
    ensure!(
        x.numel().is_multiple_of(d),
        "x numel {} not a multiple of D {d}",
        x.numel()
    );
    ensure!(gate.numel() == x.numel() && out.numel() == x.numel(), "size mismatch");
    let rows = x.numel() / d;
    let pipeline = ctx.pipeline("gated_rmsnorm_bf16", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), gate.binding(), w.binding(), out.binding()],
        &[&u32_bytes(d), &eps.to_ne_bytes(), &u32_bytes(gate_act as usize)],
        Grid::Threadgroups { groups: (rows, 1, 1), threadgroup: (256, 1, 1) },
    )
}

#[cfg(test)]
#[path = "gdn_tests.rs"]
mod tests;

// Official external-MLX numerical entry points; legacy calls keep their defaults.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_prefill_with_rounding(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    window_in: &Tensor,
    window_out: &Tensor,
    x: &Tensor,
    w: &Tensor,
    out: &Tensor,
    official: bool,
) -> Result<()> {
    if official {
        ensure!(
            [window_in, window_out, x, w, out].iter().all(|t| t.dtype() == DType::BF16),
            "official GDN conv tensors must be BF16"
        );
    }

    ensure!(x.shape().len() == 2, "x must be [M, C]");
    let (m, c) = (x.shape()[0], x.shape()[1]);
    let kd = w.numel() / c;
    ensure!(w.numel() == kd * c, "weight numel {} not a multiple of C {c}", w.numel());
    ensure!((2..=9).contains(&kd), "conv kernel dim {kd} outside register window");
    for (name, t) in [("window_in", window_in), ("window_out", window_out)] {
        ensure!(t.numel() == c * (kd - 1), "{name} numel {} != C*(KD-1)", t.numel());
    }
    let (in_buf, in_off) = window_in.binding();
    let (out_buf, out_off) = window_out.binding();
    ensure!(
        !(std::ptr::eq(in_buf, out_buf) && in_off == out_off),
        "window buffers must be distinct (double-buffered)"
    );
    ensure!(out.numel() == m * c, "out numel {} != M*C", out.numel());
    let pipeline = ctx.pipeline(
        if official { "conv1d_prefill_bf16_mlx" } else { "conv1d_prefill_bf16" },
        SOURCE,
        MslVersion::V3_1,
    )?;
    let tiles = m.div_ceil(CONV1D_PREFILL_TILE).max(1);
    pass.dispatch_at(
        &pipeline,
        &[
            window_in.binding(),
            window_out.binding(),
            x.binding(),
            w.binding(),
            out.binding(),
        ],
        &[&u32_bytes(c), &u32_bytes(kd), &u32_bytes(m)],
        Grid::Threads { grid: (c, tiles, 1), threadgroup: (256.min(c), 1, 1) },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn conv1d_step_with_rounding(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    window: &Tensor,
    x: &Tensor,
    w: &Tensor,
    out: &Tensor,
    official: bool,
) -> Result<()> {
    if official {
        ensure!(
            [window, x, w, out].iter().all(|t| t.dtype() == DType::BF16),
            "official GDN conv tensors must be BF16"
        );
    }

    let c = x.numel();
    let kd = w.numel() / c;
    ensure!(w.numel() == kd * c, "weight numel {} not a multiple of C {c}", w.numel());
    ensure!(kd >= 2, "conv kernel dim {kd} < 2");
    ensure!(
        window.numel() == c * (kd - 1),
        "window numel {} != C*(KD-1)",
        window.numel()
    );
    ensure!(out.numel() == c, "out numel {} != C {c}", out.numel());
    let pipeline = ctx.pipeline(
        if official { "conv1d_step_bf16_mlx" } else { "conv1d_step_bf16" },
        SOURCE,
        MslVersion::V3_1,
    )?;
    pass.dispatch_at(
        &pipeline,
        &[window.binding(), x.binding(), w.binding(), out.binding()],
        &[&u32_bytes(c), &u32_bytes(kd)],
        Grid::Threads { grid: (c, 1, 1), threadgroup: (256.min(c), 1, 1) },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gated_rmsnorm_with_rounding(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    gate: &Tensor,
    w: &Tensor,
    out: &Tensor,
    eps: f32,
    gate_act: GdnGate,
    official: bool,
) -> Result<()> {
    if official {
        ensure!(
            [x, gate, out].iter().all(|t| t.dtype() == DType::BF16),
            "official GDN norm activations must be BF16"
        );
        ensure!(w.numel() == GDN_HEAD_DIM, "official GDN norm requires D128");
    }

    let d = w.numel();
    ensure!(w.dtype() == DType::F32, "gated_rmsnorm weight must be F32");
    ensure!(
        x.numel().is_multiple_of(d),
        "x numel {} not a multiple of D {d}",
        x.numel()
    );
    ensure!(gate.numel() == x.numel() && out.numel() == x.numel(), "size mismatch");
    let rows = x.numel() / d;
    let pipeline = ctx.pipeline(
        if official { "gated_rmsnorm_bf16_mlx" } else { "gated_rmsnorm_bf16" },
        SOURCE,
        MslVersion::V3_1,
    )?;
    pass.dispatch_at(
        &pipeline,
        &[x.binding(), gate.binding(), w.binding(), out.binding()],
        &[&u32_bytes(d), &eps.to_ne_bytes(), &u32_bytes(gate_act as usize)],
        Grid::Threadgroups { groups: (rows, 1, 1), threadgroup: (256, 1, 1) },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_step_gated_fused_with_rounding(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    z: &Tensor,
    norm_w: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
    eps: f32,
    gate_act: GdnGate,
    official: bool,
) -> Result<()> {
    if official {
        ensure!(
            [a, b, dt_bias].iter().all(|t| t.dtype() == DType::BF16),
            "official GDN gate inputs must be BF16"
        );
        ensure!(
            a_log.dtype() == DType::F32
                && a_log.numel() == a.numel()
                && dt_bias.numel() == a.numel(),
            "official GDN gate parameter shape/dtype mismatch"
        );
    }

    let num_heads = a.numel();
    let dim = GDN_HEAD_DIM;
    ensure!(
        num_k_heads > 0 && num_heads.is_multiple_of(num_k_heads),
        "v-heads {num_heads} not a multiple of k-heads {num_k_heads}"
    );
    let k_part = num_k_heads * dim;
    let part = num_heads * dim;
    ensure!(qkv.numel() == 2 * k_part + part, "bad fused qkv size");
    ensure!(qkv.dtype() == DType::BF16, "qkv must be BF16");
    ensure!(state.numel() == num_heads * dim * dim, "bad state size");
    ensure!(state.dtype() == DType::F32, "GDN state must be F32");
    ensure!(z.numel() == part && z.dtype() == DType::BF16, "z must be BF16 [H, dim]");
    ensure!(
        norm_w.numel() == dim && norm_w.dtype() == DType::F32,
        "norm_w must be F32 [dim]"
    );
    ensure!(out.numel() == part && out.dtype() == DType::BF16, "bad output size");
    gdn_step_gated_named(
        ctx,
        pass,
        if official { "gdn_step_gated_mlx" } else { "gdn_step_gated" },
        qkv,
        a,
        b,
        a_log,
        dt_bias,
        state,
        z,
        norm_w,
        out,
        scale,
        num_k_heads,
        eps,
        gate_act,
    )
}

#[allow(clippy::too_many_arguments)]
fn gdn_qk_rms_mlx(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    qk_norm: &Tensor,
    scale: f32,
    num_k_heads: usize,
    num_heads: usize,
) -> Result<()> {
    let dim = GDN_HEAD_DIM;
    ensure!(
        num_k_heads > 0 && num_heads > 0 && num_heads.is_multiple_of(num_k_heads),
        "v-heads {num_heads} not a positive multiple of k-heads {num_k_heads}"
    );
    ensure!(qkv.dtype() == DType::BF16, "qkv must be BF16");
    let row_size = (2 * num_k_heads + num_heads) * dim;
    ensure!(qkv.numel().is_multiple_of(row_size), "qkv not [M, (2*HK+H)*{dim}]");
    let m = qkv.numel() / row_size;
    ensure!(m > 0, "qkv must contain at least one row");
    ensure!(
        qk_norm.numel() == m * 2 * num_k_heads * dim && qk_norm.dtype() == DType::BF16,
        "qk_norm must be BF16 [M, 2*HK*{dim}]"
    );

    let pipeline = ctx.pipeline("gdn_qk_rms_mlx", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[qkv.binding(), qk_norm.binding()],
        &[&scale.to_ne_bytes(), &u32_bytes(num_k_heads), &u32_bytes(num_heads)],
        Grid::Threads {
            grid: (32, num_k_heads, m),
            threadgroup: (32, 4.min(num_k_heads), 1),
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn gdn_gates_mlx(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
) -> Result<()> {
    let num_heads = a_log.numel();
    ensure!(num_heads > 0, "a_log must contain at least one head");
    ensure!(a.dtype() == DType::BF16, "a must be BF16");
    ensure!(b.dtype() == DType::BF16, "b must be BF16");
    ensure!(a_log.dtype() == DType::F32, "a_log must be F32");
    ensure!(dt_bias.dtype() == DType::BF16, "dt_bias must be BF16");
    ensure!(a.numel().is_multiple_of(num_heads), "a not [M, H]");
    let n = a.numel();
    ensure!(n > 0, "a must contain at least one row");
    ensure!(b.numel() == n, "b not [M, H]");
    ensure!(dt_bias.numel() == num_heads, "dt_bias not [H]");
    ensure!(
        decay.numel() == n && decay.dtype() == DType::F32,
        "decay must be F32 [M, H]"
    );
    ensure!(beta.numel() == n && beta.dtype() == DType::F32, "beta must be F32 [M, H]");

    let pipeline = ctx.pipeline("gdn_gates_mlx", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            a.binding(),
            b.binding(),
            a_log.binding(),
            dt_bias.binding(),
            decay.binding(),
            beta.binding(),
        ],
        &[&u32_bytes(num_heads)],
        Grid::Threads { grid: (n, 1, 1), threadgroup: (256.min(n), 1, 1) },
    )
}

#[allow(clippy::too_many_arguments)]
fn gdn_prefill_scan_named_with_rounding(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    staging: &GdnRegscanStaging<'_>,
    state: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
    mid: Option<&Tensor>,
    scan_name: &str,
    official: bool,
) -> Result<()> {
    let num_heads = a_log.numel();
    let dim = GDN_HEAD_DIM;
    ensure!(
        num_k_heads > 0 && num_heads.is_multiple_of(num_k_heads),
        "v-heads {num_heads} not a multiple of k-heads {num_k_heads}"
    );
    let vpk = num_heads / num_k_heads;
    ensure!(a.numel().is_multiple_of(num_heads), "a not [M, H]");
    let m = a.numel() / num_heads;
    ensure!(b.numel() == m * num_heads, "b not [M, H]");
    ensure!(
        qkv.numel() == m * (2 * num_k_heads + num_heads) * dim,
        "qkv not [M, (2*HK+H)*{dim}]"
    );
    ensure!(out.numel() == m * num_heads * dim, "out not [M, H, {dim}]");
    ensure!(state.numel() == num_heads * dim * dim, "state must be [H, {dim}, {dim}]");
    ensure!(state.dtype() == DType::F32, "GDN state must be F32");
    ensure!(a_log.dtype() == DType::F32, "a_log must be F32");
    ensure!(
        staging.qk_norm.numel() == m * 2 * num_k_heads * dim
            && staging.qk_norm.dtype() == DType::BF16,
        "qk_norm must be BF16 [M, 2*HK*{dim}]"
    );
    ensure!(
        staging.decay.numel() == m * num_heads && staging.decay.dtype() == DType::F32,
        "decay must be F32 [M, H]"
    );
    ensure!(
        staging.beta.numel() == m * num_heads && staging.beta.dtype() == DType::F32,
        "beta must be F32 [M, H]"
    );

    ensure!(
        !official || scan_name == GDN_SCAN_KERNEL,
        "official GDN requires F32 serial recurrence"
    );

    if scan_name == GDN_CHUNK_SCAN {
        ensure!(mid.is_none(), "the chunked scan records no mid states");
        let chunk = staging.chunk.as_ref().ok_or_else(|| {
            anyhow::anyhow!("the chunked scan needs its chunk staging")
        })?;
        // Whole chunks through the tensor kernels; a ragged tail continues
        // from their state through the token-serial scan (its kernels read
        // whole chunks of rows, see gdn.metal).
        let m_main = m - m % GDN_CHUNK;
        if m_main == m {
            return gdn_prefill_chunked(
                ctx,
                pass,
                qkv,
                a,
                b,
                a_log,
                dt_bias,
                staging,
                chunk,
                state,
                out,
                scale,
                num_k_heads,
            );
        }
        let rows = |t: &Tensor, start: usize, len: usize| -> Result<Tensor> {
            let mut shape = t.shape().to_vec();
            let per_row = t.numel() / shape[0];
            shape[0] = len;
            t.view(start * per_row, &shape)
        };
        if m_main > 0 {
            let head = GdnRegscanStaging {
                qk_norm: &rows(staging.qk_norm, 0, m_main)?,
                decay: staging.decay,
                beta: staging.beta,
                chunk: Some(GdnChunkStaging {
                    w: &rows(chunk.w, 0, m_main)?,
                    u: &rows(chunk.u, 0, m_main)?,
                    p: &rows(chunk.p, 0, m_main)?,
                    g: &rows(chunk.g, 0, m_main)?,
                }),
            };
            gdn_prefill_chunked(
                ctx,
                pass,
                &rows(qkv, 0, m_main)?,
                &rows(a, 0, m_main)?,
                &rows(b, 0, m_main)?,
                a_log,
                dt_bias,
                &head,
                head.chunk.as_ref().expect("chunk staging"),
                state,
                &rows(out, 0, m_main)?,
                scale,
                num_k_heads,
            )?;
            pass.level_barrier(&[state])?;
        }
        let tail_len = m - m_main;
        let tail = GdnRegscanStaging {
            qk_norm: &rows(staging.qk_norm, m_main, tail_len)?,
            decay: &rows(staging.decay, m_main, tail_len)?,
            beta: &rows(staging.beta, m_main, tail_len)?,
            chunk: None,
        };
        return gdn_prefill_scan_named(
            ctx,
            pass,
            &rows(qkv, m_main, tail_len)?,
            &rows(a, m_main, tail_len)?,
            &rows(b, m_main, tail_len)?,
            a_log,
            dt_bias,
            &tail,
            state,
            &rows(out, m_main, tail_len)?,
            scale,
            num_k_heads,
            None,
            GDN_SCAN_KERNEL,
        );
    }
    let mid_count = match mid {
        Some(t) => {
            ensure!(
                t.dtype() == DType::F32
                    && t.shape().len() == 4
                    && t.shape()[1..] == [num_heads, dim, dim],
                "mid states must be F32 [count, H, {dim}, {dim}]"
            );
            ensure!(
                t.shape()[0] <= m,
                "more mid states ({}) than tokens ({m})",
                t.shape()[0]
            );
            t.shape()[0]
        }
        None => 0,
    };

    if official {
        gdn_qk_rms_mlx(ctx, pass, qkv, staging.qk_norm, scale, num_k_heads, num_heads)?;
        gdn_gates_mlx(ctx, pass, a, b, a_log, dt_bias, staging.decay, staging.beta)?;
    } else {
        gdn_qk_l2norm(ctx, pass, qkv, staging.qk_norm, scale, num_k_heads, num_heads)?;
        gdn_gates(ctx, pass, a, b, a_log, dt_bias, staging.decay, staging.beta)?;
    }
    // The scan reads the staging the two dispatches above wrote.
    pass.level_barrier(&[staging.qk_norm, staging.decay, staging.beta])?;
    let scan_name: &'static str = if scan_name == GDN_SCAN_KERNEL {
        GDN_SCAN_KERNEL
    } else {
        Box::leak(scan_name.to_string().into_boxed_str())
    };
    // Value columns per simdgroup: the shipped scan takes four, the
    // comparison instantiations carry theirs in the name.
    let cols_per_sg = ["_c16", "_c8", "_c4", "_c2", "_c1"]
        .iter()
        .find(|tag| scan_name.contains(*tag))
        .map_or(4, |tag| tag[2..].parse::<usize>().expect("column tag"));
    let scan = ctx.pipeline(scan_name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &scan,
        &[
            qkv.binding(),
            staging.qk_norm.binding(),
            staging.decay.binding(),
            staging.beta.binding(),
            state.binding(),
            out.binding(),
            // Never written when mid_count is 0; the state buffer stands in.
            mid.unwrap_or(state).binding(),
        ],
        &[&u32_bytes(m), &u32_bytes(num_heads), &u32_bytes(vpk), &u32_bytes(mid_count)],
        Grid::Threadgroups {
            groups: (num_heads, dim / (4 * cols_per_sg), 1),
            threadgroup: (32, 4, 1),
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_mid_with_rounding(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    staging: &GdnRegscanStaging<'_>,
    state: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
    mid: Option<&Tensor>,
    official: bool,
) -> Result<()> {
    if official {
        // >=128 rows must not re-enter the legacy WY lane, whose gates and
        // BF16 W/U/P intermediates follow a different numerical contract.
        gdn_prefill_scan_named_with_rounding(
            ctx,
            pass,
            qkv,
            a,
            b,
            a_log,
            dt_bias,
            staging,
            state,
            out,
            scale,
            num_k_heads,
            mid,
            GDN_SCAN_KERNEL,
            true,
        )
    } else {
        gdn_prefill_mid(
            ctx,
            pass,
            qkv,
            a,
            b,
            a_log,
            dt_bias,
            staging,
            state,
            out,
            scale,
            num_k_heads,
            mid,
        )
    }
}

#[cfg(test)]
#[path = "official_gdn_rounding_tests.rs"]
mod official_rounding_tests;

#[cfg(test)]
#[path = "official_gdn_k1_helper_tests.rs"]
mod official_k1_helper_tests;

use crate::metal::MslMathMode;

/// Physical F32 GDN state order. The equal D128 dimensions do not encode this
/// distinction in a tensor shape; owners and snapshot restores keep it explicit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnStateLayout {
    /// Lily's retained `[H, key, value]` order.
    KeyValue,
    /// The served MLX K1's `[H, value, key]` order.
    ValueKey,
}

/// One materialized decode conv update. Preserve primitive/reference behavior
/// for KeyValue owners; ValueKey owners use the served K1's custom BF16 SiLU.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_step_with_layout(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    window: &Tensor,
    x: &Tensor,
    w: &Tensor,
    out: &Tensor,
    official: bool,
    layout: GdnStateLayout,
) -> Result<()> {
    if layout == GdnStateLayout::KeyValue {
        return conv1d_step_with_rounding(ctx, pass, window, x, w, out, official);
    }
    ensure!(official, "VK conv requires official execution");
    ensure!(
        [window, x, w, out].iter().all(|t| t.dtype() == DType::BF16),
        "VK conv tensors must be BF16"
    );
    ensure!(
        x.numel() == 10240
            && out.numel() == 10240
            && window.numel() == 3 * 10240
            && w.numel() == 4 * 10240,
        "VK conv requires C10240/KD4"
    );
    let pipeline = ctx.pipeline_with_math_mode(
        "conv1d_step_value_key_mlx",
        SOURCE,
        MslVersion::V3_1,
        MslMathMode::Safe,
    )?;
    pass.dispatch_at(
        &pipeline,
        &[window.binding(), x.binding(), w.binding(), out.binding()],
        &[&u32_bytes(10240), &u32_bytes(4)],
        Grid::Threads { grid: (10240, 1, 1), threadgroup: (256, 1, 1) },
    )
}

fn gated_rmsnorm_value_key_decode(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    raw: &Tensor,
    z: &Tensor,
    norm_w: &Tensor,
    out: &Tensor,
    eps: f32,
) -> Result<()> {
    let pipeline = ctx.pipeline_with_math_mode(
        "gdn_normgate_value_key_mlx",
        SOURCE,
        MslVersion::V3_1,
        MslMathMode::Safe,
    )?;
    pass.dispatch_at(
        &pipeline,
        &[raw.binding(), z.binding(), norm_w.binding(), out.binding()],
        &[&eps.to_ne_bytes()],
        Grid::Threadgroups { groups: (48, 1, 1), threadgroup: (32, 1, 1) },
    )
}

impl GdnStateLayout {
    pub fn for_model(
        official: bool,
        key_heads: usize,
        value_heads: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> Self {
        if official
            && (key_heads, value_heads, key_dim, value_dim) == (16, 48, 128, 128)
        {
            Self::ValueKey
        } else {
            Self::KeyValue
        }
    }
}

/// Stock K1's SIMD-distributed recurrence on materialized BF16 conv+SiLU.
/// F32 state is already physically `[48, value=128, key=128]` and is updated
/// in place. Every thread owns disjoint state elements; no convolution window
/// is accessed here. Convolution executes once before this dispatch.
#[allow(clippy::too_many_arguments)]
pub fn gdn_step_raw_value_key_mlx(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    raw: &Tensor,
    scale: f32,
    num_k_heads: usize,
) -> Result<()> {
    ensure!(num_k_heads == 16 && a.numel() == 48, "VK GDN requires Hk16/Hv48/D128");
    ensure!(
        qkv.numel() == 10240 && qkv.dtype() == DType::BF16,
        "VK GDN QKV must be BF16 C10240"
    );
    ensure!(
        b.numel() == 48 && dt_bias.numel() == 48 && a_log.numel() == 48,
        "VK GDN head parameter sizes"
    );
    ensure!(
        [a, b, dt_bias].iter().all(|t| t.dtype() == DType::BF16),
        "VK GDN gates must be BF16"
    );
    ensure!(a_log.dtype() == DType::F32, "VK GDN A_log must be widened F32");
    ensure!(
        state.shape() == [48, 128, 128] && state.dtype() == DType::F32,
        "VK GDN state must be F32 [48,V128,K128]"
    );
    ensure!(
        raw.numel() == 48 * 128 && raw.dtype() == DType::BF16,
        "VK GDN raw output must be BF16 [48,128]"
    );
    ensure!(
        scale.to_bits() == (1.0f32 / 128.0f32.sqrt()).to_bits(),
        "VK GDN K scale mismatch"
    );
    let pipeline = ctx.pipeline_with_math_mode(
        "gdn_step_raw_value_key_mlx",
        SOURCE,
        MslVersion::V3_1,
        MslMathMode::Safe,
    )?;
    pass.dispatch_at(
        &pipeline,
        &[
            qkv.binding(),
            a.binding(),
            b.binding(),
            a_log.binding(),
            dt_bias.binding(),
            state.binding(),
            raw.binding(),
        ],
        &[&scale.to_ne_bytes()],
        Grid::Threadgroups { groups: (48 * 4, 1, 1), threadgroup: (256, 1, 1) },
    )
}

/// Explicit state-policy dispatcher. Legacy callers continue to use their
/// original function/entrypoint, without changed arithmetic or layout.
#[allow(clippy::too_many_arguments)]
pub fn gdn_step_gated_fused_with_layout(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    z: &Tensor,
    norm_w: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
    eps: f32,
    gate_act: GdnGate,
    official: bool,
    layout: GdnStateLayout,
    raw: &Tensor,
) -> Result<()> {
    if layout == GdnStateLayout::KeyValue {
        return gdn_step_gated_fused_with_rounding(
            ctx,
            pass,
            qkv,
            a,
            b,
            a_log,
            dt_bias,
            state,
            z,
            norm_w,
            out,
            scale,
            num_k_heads,
            eps,
            gate_act,
            official,
        );
    }
    ensure!(official, "VK GDN state is only valid for official execution");
    ensure!(gate_act == GdnGate::Sigmoid, "VK GDN requires sigmoid output gate");
    ensure!(
        z.numel() == 48 * 128 && z.dtype() == DType::BF16,
        "VK GDN output gate shape/dtype"
    );
    ensure!(
        norm_w.numel() == 128 && norm_w.dtype() == DType::F32,
        "VK GDN norm weight shape/dtype"
    );
    ensure!(
        out.numel() == 48 * 128 && out.dtype() == DType::BF16,
        "VK GDN output shape/dtype"
    );
    gdn_step_raw_value_key_mlx(
        ctx,
        pass,
        qkv,
        a,
        b,
        a_log,
        dt_bias,
        state,
        raw,
        scale,
        num_k_heads,
    )?;
    pass.level_barrier(&[raw, state])?;
    pass.profile_stage(crate::metal::stage_profile::Phase::GdnNorm)?;
    gated_rmsnorm_value_key_decode(ctx, pass, raw, z, norm_w, out, eps)
}

/// Official ValueKey prefill. Eligible >=64-row spans without state capture
/// use contiguous stock-order Q/K preparation and the source-qualified TB12
/// recurrence. Short/captured/unaligned spans retain their original arithmetic;
/// convolution, gates and norm-gate stay unchanged.
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_mid_with_layout(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    staging: &GdnRegscanStaging<'_>,
    state: &Tensor,
    out: &Tensor,
    scale: f32,
    num_k_heads: usize,
    mid: Option<&Tensor>,
    official: bool,
    layout: GdnStateLayout,
) -> Result<()> {
    if layout == GdnStateLayout::KeyValue {
        return gdn_prefill_mid_with_rounding(
            ctx,
            pass,
            qkv,
            a,
            b,
            a_log,
            dt_bias,
            staging,
            state,
            out,
            scale,
            num_k_heads,
            mid,
            official,
        );
    }
    ensure!(
        official && num_k_heads == 16 && a_log.numel() == 48,
        "VK prefill requires official Hk16/Hv48/D128"
    );
    ensure!(
        a.numel() > 0 && a.numel().is_multiple_of(48),
        "VK prefill A must be nonempty [M,48]"
    );
    let m = a.numel() / 48;
    ensure!(
        [qkv, a, b, dt_bias, out].iter().all(|t| t.dtype() == DType::BF16),
        "VK prefill activations must be BF16"
    );
    ensure!(b.numel() == m * 48 && dt_bias.numel() == 48, "VK prefill gate shapes");
    ensure!(a_log.dtype() == DType::F32, "VK prefill A_log must be F32");
    ensure!(
        qkv.numel() == m * 10240 && out.numel() == m * 48 * 128,
        "VK prefill QKV/readout shapes"
    );
    ensure!(
        state.shape() == [48, 128, 128] && state.dtype() == DType::F32,
        "VK prefill state shape/dtype"
    );
    ensure!(
        scale.to_bits() == (1.0f32 / 128.0f32.sqrt()).to_bits(),
        "VK prefill K scale mismatch"
    );
    ensure!(
        staging.qk_norm.numel() == m * 2 * 16 * 128
            && staging.qk_norm.dtype() == DType::BF16,
        "VK prefill QK staging"
    );
    ensure!(
        [staging.decay, staging.beta]
            .iter()
            .all(|t| t.numel() == m * 48 && t.dtype() == DType::F32),
        "VK prefill gate staging"
    );
    let mid_count = if let Some(t) = mid {
        ensure!(
            t.dtype() == DType::F32
                && t.shape().len() == 4
                && t.shape()[1..] == [48, 128, 128]
                && t.shape()[0] <= m,
            "VK prefill mid states must be F32 [count<=M,48,V128,K128]"
        );
        t.shape()[0]
    } else {
        0
    };
    let pipelined_prefill = value_key_contiguous_prefill_qk_selected(
        official,
        layout,
        m,
        mid.is_some(),
        pipelined_prefill_buffers_aligned(qkv, staging, state, out),
    );
    if pipelined_prefill {
        gdn_prefill_qk_contiguous_value_key_mlx(
            ctx,
            pass,
            qkv,
            staging.qk_norm,
            scale,
        )?;
    } else {
        gdn_qk_rms_mlx(ctx, pass, qkv, staging.qk_norm, scale, 16, 48)?;
    }
    gdn_gates_mlx(ctx, pass, a, b, a_log, dt_bias, staging.decay, staging.beta)?;
    pass.level_barrier(&[staging.qk_norm, staging.decay, staging.beta])?;
    if pipelined_prefill {
        return gdn_prefill_pipelined_value_key_mlx(
            ctx, pass, qkv, staging, state, out, layout,
        );
    }
    let pipeline =
        ctx.pipeline("gdn_prefill_regscan_value_key_mlx", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            qkv.binding(),
            staging.qk_norm.binding(),
            staging.decay.binding(),
            staging.beta.binding(),
            state.binding(),
            out.binding(),
            mid.unwrap_or(state).binding(),
        ],
        &[&u32_bytes(m), &u32_bytes(48), &u32_bytes(3), &u32_bytes(mid_count)],
        Grid::Threadgroups { groups: (48, 8, 1), threadgroup: (32, 4, 1) },
    )
}

#[cfg(test)]
#[path = "official_gdn_value_key_tests.rs"]
mod official_value_key_tests;

/// The stock TB12/DB16 recurrence consumes the existing interleaved staging.
/// Physical state is F32 [48,V128,K128]; no transposition or allocation occurs.
const PIPELINED_PREFILL_SOURCE: &str =
    include_str!("metal/official_gdn_pipelined.metal");

pub(crate) fn value_key_pipelined_prefill_selected(
    official: bool,
    layout: GdnStateLayout,
    rows: usize,
    captures_mid: bool,
) -> bool {
    official && layout == GdnStateLayout::ValueKey && rows >= 64 && !captures_mid
}

#[allow(clippy::too_many_arguments)]
fn gdn_prefill_pipelined_value_key_mlx(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    staging: &GdnRegscanStaging<'_>,
    state: &Tensor,
    out: &Tensor,
    layout: GdnStateLayout,
) -> Result<()> {
    ensure!(
        layout == GdnStateLayout::ValueKey,
        "pipelined GDN requires ValueKey state"
    );
    ensure!(
        qkv.dtype() == DType::BF16 && qkv.numel().is_multiple_of(10240),
        "pipelined GDN QKV shape/dtype"
    );
    let rows = qkv.numel() / 10240;
    let count =
        i32::try_from(rows).map_err(|_| anyhow::anyhow!("pipelined GDN row count"))?;
    ensure!(rows > 0, "pipelined GDN requires nonempty input");
    ensure!(
        state.shape() == [48, 128, 128] && state.dtype() == DType::F32,
        "pipelined GDN state shape/dtype"
    );
    ensure!(
        out.numel() == rows * 48 * 128 && out.dtype() == DType::BF16,
        "pipelined GDN readout shape/dtype"
    );
    ensure!(
        staging.qk_norm.numel() == rows * 4096
            && staging.qk_norm.dtype() == DType::BF16,
        "pipelined GDN QK staging"
    );
    ensure!(
        [staging.decay, staging.beta]
            .iter()
            .all(|t| t.numel() == rows * 48 && t.dtype() == DType::F32),
        "pipelined GDN gate staging"
    );
    ensure!(
        pipelined_prefill_buffers_aligned(qkv, staging, state, out),
        "pipelined GDN vector alignment"
    );
    let pipeline = ctx.pipeline_with_math_mode(
        "gdn_prefill_pipelined_value_key_mlx",
        PIPELINED_PREFILL_SOURCE,
        MslVersion::V3_1,
        crate::metal::MslMathMode::Safe,
    )?;
    ensure!(
        pipeline.thread_execution_width() == 32,
        "pipelined GDN requires SIMD width 32"
    );
    pass.dispatch_at(
        &pipeline,
        &[
            qkv.binding(),
            staging.qk_norm.binding(),
            staging.decay.binding(),
            staging.beta.binding(),
            state.binding(),
            out.binding(),
        ],
        &[&count.to_ne_bytes()],
        Grid::Threadgroups { groups: (8, 48, 1), threadgroup: (128, 1, 1) },
    )
}

fn pipelined_prefill_buffers_aligned(
    qkv: &Tensor,
    staging: &GdnRegscanStaging<'_>,
    state: &Tensor,
    out: &Tensor,
) -> bool {
    [qkv, staging.qk_norm, out].iter().all(|t| t.binding().1.is_multiple_of(8))
        && state.binding().1.is_multiple_of(16)
}

#[cfg(test)]
#[path = "official_gdn_pipelined_tests.rs"]
mod official_pipelined_tests;

#[cfg(test)]
#[path = "official_gdn_dense_diagnostic_tests.rs"]
mod official_dense_diagnostic_tests;

const CONTIGUOUS_PREFILL_QK_SOURCE: &str =
    include_str!("metal/official_gdn_prefill_qk.metal");

fn value_key_contiguous_prefill_qk_selected(
    official: bool,
    layout: GdnStateLayout,
    rows: usize,
    captures_mid: bool,
    aligned: bool,
) -> bool {
    value_key_pipelined_prefill_selected(official, layout, rows, captures_mid)
        && aligned
}

fn gdn_prefill_qk_contiguous_value_key_mlx(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    qkv: &Tensor,
    qk: &Tensor,
    scale: f32,
) -> Result<()> {
    ensure!(
        qkv.dtype() == DType::BF16 && qkv.numel().is_multiple_of(10240),
        "contiguous prefill QKV shape/dtype"
    );
    let rows = qkv.numel() / 10240;
    ensure!(rows > 0 && rows <= u32::MAX as usize, "contiguous prefill row count");
    ensure!(
        qk.dtype() == DType::BF16 && qk.numel() == rows * 4096,
        "contiguous prefill QK shape/dtype"
    );
    ensure!(
        scale.to_bits() == (1.0f32 / 128.0f32.sqrt()).to_bits(),
        "contiguous prefill K scale"
    );
    let pipeline = ctx.pipeline_with_math_mode(
        "gdn_prefill_qk_value_key_contiguous_mlx",
        CONTIGUOUS_PREFILL_QK_SOURCE,
        MslVersion::V3_1,
        crate::metal::MslMathMode::Safe,
    )?;
    ensure!(
        pipeline.thread_execution_width() == 32,
        "contiguous prefill QK requires SIMD32"
    );
    pass.dispatch_at(
        &pipeline,
        &[qkv.binding(), qk.binding()],
        &[&scale.to_ne_bytes()],
        Grid::Threadgroups { groups: (rows, 32, 1), threadgroup: (32, 1, 1) },
    )
}

#[cfg(test)]
#[path = "official_gdn_prefill_qk_tests.rs"]
mod official_prefill_qk_tests;
