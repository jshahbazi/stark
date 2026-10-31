// Modified by Stark: in-process core extraction; see UPSTREAM.md.
//! GPU-resident sparse-MoE routing, expert projection, and combine kernels.

use anyhow::{Result, bail, ensure};

use crate::kernels::u32_bytes;
use crate::metal::{ComputePass, Grid, MetalContext, MslVersion};
use crate::tensor::{DType, Tensor};
use crate::weights::QuantWeights;

const SOURCE: &str = include_str!("metal/moe.metal");

/// Router limits compiled into moe.metal.
const MAX_E: usize = 1024;
const MAX_K: usize = 16;
/// Experts the router top-k kernels select at most (`MOE_ROUTER_MAX_K`):
/// above `MAX_K` only for the prediction log's wider selection.
pub const ROUTER_MAX_K: usize = 32;

/// Optional single-token HC write following the decode MoE projection.
/// The projection's BF16 branch value is retained before the F32 write gate.
#[derive(Clone, Copy)]
pub(crate) struct HcInjection<'a> {
    pub hyper: &'a Tensor,
    pub logits: &'a Tensor,
    pub groups: usize,
}

/// Softmax top-k routing from F32 logits; ties choose the lowest expert id.
pub fn moe_router_topk(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    indices: &Tensor,
    scores: &Tensor,
    renorm: bool,
) -> Result<()> {
    let e = logits.numel();
    let k = indices.numel();
    ensure!(
        e <= MAX_E && logits.dtype() == DType::F32,
        "logits must be F32 [<= {MAX_E}]"
    );
    ensure!(
        k <= ROUTER_MAX_K && k <= e && indices.dtype() == DType::U32,
        "indices must be U32 [<= {ROUTER_MAX_K}]"
    );
    ensure!(
        scores.numel() == k && scores.dtype() == DType::F32,
        "scores must be F32 [k]"
    );
    let pipeline = ctx.pipeline(router_kernel(e, false)?, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[logits.binding(), indices.binding(), scores.binding()],
        &[&u32_bytes(e), &u32_bytes(k), &u32_bytes(renorm as usize)],
        Grid::Threadgroups {
            groups: (1, 1, 1),
            threadgroup: (router_threadgroup(e, renorm), 1, 1),
        },
    )
}

/// The router kernel instantiated for `e` logits: one simdgroup selects, each
/// lane holding `EPL` experts in registers (`E <= 32 * EPL`).
fn router_kernel(e: usize, rows: bool) -> Result<&'static str> {
    Ok(match (e.div_ceil(32), rows) {
        (0..=8, false) => "moe_router_topk_e8",
        (0..=8, true) => "moe_router_topk_rows_e8",
        (9..=16, false) => "moe_router_topk_e16",
        (9..=16, true) => "moe_router_topk_rows_e16",
        (17..=32, false) => "moe_router_topk_e32",
        (17..=32, true) => "moe_router_topk_rows_e32",
        _ => bail!("router top-k supports at most {MAX_E} logits (got {e})"),
    })
}

/// Threads per router threadgroup: the raw-logit (renorm) path is one
/// simdgroup; the full-softmax path keeps its strided reduction over up to
/// 256 threads (its probabilities depend on that reduction order).
fn router_threadgroup(e: usize, renorm: bool) -> usize {
    if renorm { 32 } else { e.clamp(32, 256).next_multiple_of(32) }
}

/// Fused Q4 gate/up gather GEMV with SwiGLU output.
#[allow(clippy::too_many_arguments)]
pub fn moe_gather_gemv_gate_up(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &QuantWeights,
    up: &QuantWeights,
    n_per_expert: usize,
    x: &Tensor,
    indices: &Tensor,
    y: &Tensor,
) -> Result<()> {
    moe_gather_gemv_gate_up_named(
        ctx,
        pass,
        "moe_gather_gemv_q4_gate_up",
        gate,
        up,
        n_per_expert,
        x,
        indices,
        y,
    )
}

/// [`moe_gather_gemv_gate_up`] through the kernel `name` (timing variants
/// share the shipped kernel's bindings and grid).
#[allow(clippy::too_many_arguments)]
pub fn moe_gather_gemv_gate_up_named(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    name: &'static str,
    gate: &QuantWeights,
    up: &QuantWeights,
    n_per_expert: usize,
    x: &Tensor,
    indices: &Tensor,
    y: &Tensor,
) -> Result<()> {
    ensure!(gate.bits == 4 && up.bits == 4, "expert gate/up fusion is Q4 only");
    ensure!(
        (gate.out_features(), gate.in_features(), gate.group_size)
            == (up.out_features(), up.in_features(), up.group_size),
        "expert gate/up shapes or group sizes differ"
    );
    let (rows, k_in) = (gate.out_features(), gate.in_features());
    ensure!(rows.is_multiple_of(n_per_expert), "bad stacked expert rows");
    ensure!(
        k_in.is_multiple_of(gate.group_size) && gate.group_size.is_multiple_of(32),
        "in {k_in} / group size {} not block-packable",
        gate.group_size
    );
    ensure!(indices.dtype() == DType::U32, "indices must be U32");
    let slots = indices.numel();
    ensure!(x.numel() == k_in, "gate/up fusion expects one shared input vector");
    ensure!(
        y.numel() == slots * n_per_expert && y.dtype() == DType::BF16,
        "y must be BF16 [{slots}, {n_per_expert}]"
    );
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            gate.codes.binding(),
            gate.scales.binding(),
            gate.biases.binding(),
            up.codes.binding(),
            up.scales.binding(),
            up.biases.binding(),
            x.binding(),
            indices.binding(),
            y.binding(),
        ],
        &[&u32_bytes(k_in), &u32_bytes(gate.group_size), &u32_bytes(n_per_expert)],
        Grid::Threadgroups {
            groups: (n_per_expert, slots, 1),
            threadgroup: (32, 1, 1),
        },
    )
}

/// Largest chunk row count the fused small-batch expert kernels hold per
/// union expert in registers.
pub const MOE_SMALLM_MAX_M: usize = 8;

/// Pair-list capacity compiled into moe.metal.
const MOE_SMALLM_MAX_S: usize = 256;

/// Pair-parallel simdgroups per column group of the fused down kernel
/// (`MOE_SMALLM_DOWN_PSG` in moe.metal).
const MOE_SMALLM_DOWN_PSG: usize = 4;

/// Shape checks shared by the small-batch expert kernels: Q4 stacked experts
/// with block-packable groups, `S = m * top_k` routed pairs with
/// `m <= MOE_SMALLM_MAX_M`, and `n_per_expert` a multiple of the `2 * r4`
/// rows a threadgroup walks. Returns `(k_in, m, s)`.
fn check_smallm(
    w: &QuantWeights,
    n_per_expert: usize,
    indices: &Tensor,
    top_k: usize,
    r4: usize,
) -> Result<(usize, usize, usize)> {
    let (rows, k_in) = (w.out_features(), w.in_features());
    ensure!(w.bits == 4, "small-m expert kernels are 4-bit only");
    ensure!(
        rows.is_multiple_of(n_per_expert),
        "stacked rows {rows} not a multiple of per-expert rows {n_per_expert}"
    );
    // Each 32-element block must stay within one scale group.
    ensure!(
        k_in.is_multiple_of(w.group_size) && w.group_size.is_multiple_of(32),
        "in {k_in} / group size {} not block-packable",
        w.group_size
    );
    ensure!(top_k > 0 && top_k <= MAX_K, "top_k {top_k} out of 1..={MAX_K}");
    ensure!(indices.dtype() == DType::U32, "indices must be U32 [m, top_k]");
    let s = indices.numel();
    ensure!(s.is_multiple_of(top_k) && s > 0, "pair count {s} not m * {top_k}");
    ensure!(
        s <= MOE_SMALLM_MAX_S,
        "pair count {s} exceeds the staged pair-list capacity"
    );
    let m = s / top_k;
    ensure!(
        m <= MOE_SMALLM_MAX_M,
        "small-m expert kernels hold at most {MOE_SMALLM_MAX_M} rows per expert \
         in registers (got m={m})"
    );
    ensure!(
        n_per_expert.is_multiple_of(2 * r4),
        "small-m N per expert {n_per_expert} must be a multiple of {}",
        2 * r4
    );
    Ok((k_in, m, s))
}

/// Fused small-batch gate + up + SwiGLU over the union of the routed experts:
/// `y[pair, :] = silu(gate_e . x[row]) * (up_e . x[row])` for every routed
/// `(row, k)` pair with `e = indices[row, k]`, `x` `[m, K]`, `y` BF16
/// `[S, N]`. Each union expert's weights are streamed once and applied to
/// all its rows; the grid is `(N / rows, S)` (duplicate pairs exit), so it
/// encodes before the routing is known. Matches `silu_mul_bf16` of the two
/// expert-major GEMVs bit for bit.
#[allow(clippy::too_many_arguments)]
pub fn moe_smallm_gate_up(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &QuantWeights,
    up: &QuantWeights,
    n_per_expert: usize,
    x: &Tensor,
    indices: &Tensor,
    y: &Tensor,
    top_k: usize,
) -> Result<()> {
    ensure!(
        (up.out_features(), up.in_features(), up.group_size, up.bits)
            == (gate.out_features(), gate.in_features(), gate.group_size, gate.bits),
        "expert gate/up shapes, group sizes or bit widths differ"
    );
    let k_in = gate.in_features();
    let m = indices.numel() / top_k.max(1);
    // Rows per simdgroup: 8 on the wide walk (K <= 512), else 4 for the
    // m <= 4 tier and 2 for the 8-row tier (64 accumulators per lane at 4
    // rows changed the compiler's f32 scheduling against the reference).
    let (name, r4) = match (m <= 4, k_in <= 512) {
        (true, false) => ("moe_smallm_q4_gate_up_r4", 4),
        (true, true) => ("moe_smallm_q4_gate_up_r4_w", 8),
        (false, false) => ("moe_smallm_q4_gate_up_r8", 2),
        (false, true) => ("moe_smallm_q4_gate_up_r8_w", 8),
    };
    let (k_in, m, s) = check_smallm(gate, n_per_expert, indices, top_k, r4)?;
    ensure!(
        x.numel() == m * k_in && x.dtype() == DType::BF16,
        "x must be BF16 [{m}, {k_in}]"
    );
    ensure!(
        y.numel() == s * n_per_expert && y.dtype() == DType::BF16,
        "y must be BF16 [{s}, {n_per_expert}]"
    );
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            gate.codes.binding(),
            gate.scales.binding(),
            gate.biases.binding(),
            up.codes.binding(),
            up.scales.binding(),
            up.biases.binding(),
            x.binding(),
            indices.binding(),
            y.binding(),
        ],
        &[
            &u32_bytes(k_in),
            &u32_bytes(gate.group_size),
            &u32_bytes(n_per_expert),
            &u32_bytes(s),
            &u32_bytes(top_k),
        ],
        Grid::Threadgroups {
            groups: (n_per_expert / (2 * r4), s, 1),
            threadgroup: (64, 1, 1),
        },
    )
}

/// Fused small-batch down + combine + shared-expert add:
/// `out[row, :] = bf16(sum_k scores[row, k] * bf16(down_e . x[row, k])) +
/// sigmoid(shared_gate[row]) * shared_out[row, :]` with `e = indices[row,
/// k]`, `x` BF16 `[S, K]` (one activation row per routed pair), `scores` F32
/// `[m, top_k]`, `shared_out`/`out` BF16 `[m, H]`, `shared_gate` BF16 `[m]`.
/// One threadgroup per (column tile, row) walks the row's pairs; grid
/// `(H / rows, m)`. Matches the expert-major down GEMV followed by
/// `moe_combine_rows` and `moe_row_gate_add` bit for bit.
#[allow(clippy::too_many_arguments)]
pub fn moe_smallm_down_combine(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    h: usize,
    x: &Tensor,
    indices: &Tensor,
    scores: &Tensor,
    shared_out: &Tensor,
    shared_gate: &Tensor,
    out: &Tensor,
    top_k: usize,
) -> Result<()> {
    let (name, r4) = if w.in_features() <= 512 {
        ("moe_smallm_q4_down_combine_w", 8)
    } else {
        ("moe_smallm_q4_down_combine", 4)
    };
    let (k_in, m, s) = check_smallm(w, h, indices, top_k, r4)?;
    ensure!(
        x.numel() == s * k_in && x.dtype() == DType::BF16,
        "x must be BF16 [{s}, {k_in}]"
    );
    ensure!(
        scores.numel() == s && scores.dtype() == DType::F32,
        "scores must be F32 [{m}, {top_k}]"
    );
    for (t, what) in [(shared_out, "shared_out"), (out, "out")] {
        ensure!(
            t.numel() == m * h && t.dtype() == DType::BF16,
            "{what} must be BF16 [{m}, {h}]"
        );
    }
    ensure!(
        shared_gate.numel() == m && shared_gate.dtype() == DType::BF16,
        "shared_gate must be BF16 [{m}]"
    );
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            x.binding(),
            indices.binding(),
            scores.binding(),
            shared_out.binding(),
            shared_gate.binding(),
            out.binding(),
        ],
        &[&u32_bytes(k_in), &u32_bytes(w.group_size), &u32_bytes(h), &u32_bytes(top_k)],
        Grid::Threadgroups {
            groups: (h / (2 * r4), m, 1),
            threadgroup: (64 * MOE_SMALLM_DOWN_PSG, 1, 1),
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub fn moe_gather_gemv_down_combine(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    h: usize,
    x: &Tensor,
    indices: &Tensor,
    scores: &Tensor,
    shared: Option<(&Tensor, &Tensor)>,
    out: &Tensor,
) -> Result<()> {
    moe_gather_gemv_down_combine_named(
        ctx,
        pass,
        "moe_gather_gemv_q4_down_combine",
        w,
        h,
        x,
        indices,
        scores,
        shared,
        out,
    )
}

/// [`moe_gather_gemv_down_combine`] through the kernel `name` (timing
/// variants share the shipped kernel's bindings and grid).
#[allow(clippy::too_many_arguments)]
pub fn moe_gather_gemv_down_combine_named(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    name: &'static str,
    w: &QuantWeights,
    h: usize,
    x: &Tensor,
    indices: &Tensor,
    scores: &Tensor,
    shared: Option<(&Tensor, &Tensor)>,
    out: &Tensor,
) -> Result<()> {
    check_down_combine(w, h, x, indices, scores, shared, out)?;
    let k_in = w.in_features();
    let slots = indices.numel();
    let (shared_out, gate) = match shared {
        Some((shared_out, gate)) => (shared_out, gate),
        None => (x, scores),
    };
    let pipeline = ctx.pipeline(name, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            x.binding(),
            indices.binding(),
            scores.binding(),
            shared_out.binding(),
            gate.binding(),
            out.binding(),
        ],
        &[
            &u32_bytes(k_in),
            &u32_bytes(w.group_size),
            &u32_bytes(h),
            &u32_bytes(slots),
            &u32_bytes(shared.is_some() as usize),
        ],
        down_combine_grid(name, h),
    )
}

#[allow(clippy::too_many_arguments)]
fn check_down_combine(
    w: &QuantWeights,
    h: usize,
    x: &Tensor,
    indices: &Tensor,
    scores: &Tensor,
    shared: Option<(&Tensor, &Tensor)>,
    out: &Tensor,
) -> Result<()> {
    let k_in = w.in_features();
    ensure!(w.bits == 4, "fused down/combine requires Q4 weights");
    ensure!(h.is_multiple_of(2), "fused down/combine requires even H");
    ensure!(indices.dtype() == DType::U32, "indices must be U32");
    let slots = indices.numel();
    ensure!(
        w.out_features().is_multiple_of(h),
        "stacked expert rows {} not divisible by H {h}",
        w.out_features()
    );
    ensure!((1..=MAX_K).contains(&slots), "down/combine requires 1..={MAX_K} slots");
    ensure!(scores.numel() == slots && scores.dtype() == DType::F32, "bad scores");
    ensure!(x.numel() == slots * k_in, "x must be BF16 [{slots}, {k_in}]");
    ensure!(x.dtype() == DType::BF16, "x must be BF16");
    ensure!(out.numel() == h && out.dtype() == DType::BF16, "out must be BF16 [{h}]");
    ensure!(
        k_in.is_multiple_of(w.group_size) && w.group_size.is_multiple_of(32),
        "in {k_in} / group size {} not block-packable",
        w.group_size
    );
    if let Some((shared_out, gate)) = shared {
        ensure!(
            shared_out.numel() == h && shared_out.dtype() == DType::BF16,
            "shared_out must be BF16 [{h}]"
        );
        ensure!(
            gate.numel() == 1 && gate.dtype() == DType::F32,
            "gate must be F32 [1]"
        );
    }
    Ok(())
}

/// The original down/combine dot and BF16 branch store followed by the
/// original HC injection expression in the row's owning thread. Each owner
/// writes its column in every stream, so no cross-thread synchronization is
/// required after the existing slot-ordered reduction. The caller orders both
/// `out` and `injection.hyper` before subsequent readers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_gather_gemv_down_combine_inject(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    h: usize,
    x: &Tensor,
    indices: &Tensor,
    scores: &Tensor,
    shared: Option<(&Tensor, &Tensor)>,
    out: &Tensor,
    injection: HcInjection<'_>,
) -> Result<()> {
    check_down_combine(w, h, x, indices, scores, shared, out)?;
    ensure!(injection.groups > 0, "HC injection needs at least one stream");
    ensure!(
        injection.hyper.dtype() == DType::BF16
            && injection.hyper.numel() == h * injection.groups,
        "HC hyper must be BF16 [groups, h]"
    );
    ensure!(
        injection.logits.dtype() == DType::BF16
            && injection.logits.numel() == injection.groups,
        "HC injection logits must be BF16 [groups]"
    );
    let k_in = w.in_features();
    let slots = indices.numel();
    let (shared_out, gate) = match shared {
        Some((shared_out, gate)) => (shared_out, gate),
        None => (x, scores),
    };
    let pipeline = ctx.pipeline(
        "moe_gather_gemv_q4_down_combine_inject",
        SOURCE,
        MslVersion::V3_1,
    )?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            x.binding(),
            indices.binding(),
            scores.binding(),
            shared_out.binding(),
            gate.binding(),
            out.binding(),
            injection.hyper.binding(),
            injection.logits.binding(),
        ],
        &[
            &u32_bytes(k_in),
            &u32_bytes(w.group_size),
            &u32_bytes(h),
            &u32_bytes(slots),
            &u32_bytes(shared.is_some() as usize),
            &u32_bytes(injection.groups),
            &(1.0 / injection.groups as f32).to_ne_bytes(),
        ],
        down_combine_grid("moe_gather_gemv_q4_down_combine_inject", h),
    )
}

/// The down gather's dispatch shape by kernel name: two rows per
/// threadgroup, four simdgroups on them for the shipped kernel and one for
/// the `_sg1` reference form.
fn down_combine_grid(name: &str, h: usize) -> Grid {
    let sgs = if name.ends_with("_sg1") { 1 } else { 4 };
    Grid::Threadgroups { groups: (h / 2, 1, 1), threadgroup: (32 * sgs, 1, 1) }
}

/// Routes each BF16 `[m, E]` row into indices and F32 scores.
pub fn moe_router_topk_rows(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    indices: &Tensor,
    scores: &Tensor,
    renorm: bool,
) -> Result<()> {
    ensure!(logits.shape().len() == 2, "logits must be [m, E]");
    let (m, e) = (logits.shape()[0], logits.shape()[1]);
    ensure!(e <= MAX_E && logits.dtype() == DType::BF16, "logits must be BF16 [m, E]");
    ensure!(
        indices.dtype() == DType::U32 && indices.numel().is_multiple_of(m),
        "indices must be U32 [m, k]"
    );
    let k = indices.numel() / m;
    ensure!(k <= ROUTER_MAX_K && k <= e, "k {k} out of range");
    ensure!(
        scores.numel() == m * k && scores.dtype() == DType::F32,
        "scores must be F32 [m, k]"
    );
    let pipeline = ctx.pipeline(router_kernel(e, true)?, SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[logits.binding(), indices.binding(), scores.binding()],
        &[&u32_bytes(e), &u32_bytes(k), &u32_bytes(renorm as usize)],
        Grid::Threadgroups {
            groups: (m, 1, 1),
            threadgroup: (router_threadgroup(e, renorm), 1, 1),
        },
    )
}

/// Zero-fills a U32 tensor (histogram/cursor reset).
pub fn fill_zero_u32(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    dst: &Tensor,
) -> Result<()> {
    ensure!(dst.dtype() == DType::U32, "fill_zero_u32 needs U32");
    let n = dst.numel();
    let pipeline = ctx.pipeline("fill_zero_u32", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[dst.binding()],
        &[&u32_bytes(n)],
        Grid::Threads { grid: (n, 1, 1), threadgroup: (64, 1, 1) },
    )
}

/// Device buffers for the counting-sort pipeline (`moe_sort_slots`).
pub struct MoeSortBuffers<'a> {
    pub indices: &'a Tensor,
    pub counts: &'a Tensor,
    pub cursors: &'a Tensor,
    pub offsets: &'a Tensor,
    pub tile_offsets: &'a Tensor,
    pub ids_sorted: &'a Tensor,
    pub slot_of: &'a Tensor,
}

/// Sorts routes by expert; scratch must be zeroed and `tile_m` must match the
/// consuming grouped GEMM.
pub fn moe_sort_slots(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    b: &MoeSortBuffers<'_>,
    num_experts: usize,
    top_k: usize,
    tile_m: usize,
) -> Result<()> {
    let MoeSortBuffers {
        indices,
        counts,
        cursors,
        offsets,
        tile_offsets,
        ids_sorted,
        slot_of,
    } = *b;
    let s = indices.numel();
    ensure!(tile_m > 0, "row-tile height must be positive");
    ensure!(indices.dtype() == DType::U32, "indices must be U32");
    ensure!(
        counts.numel() == num_experts && cursors.numel() == num_experts,
        "counts/cursors must be [E]"
    );
    ensure!(
        offsets.numel() == num_experts + 1 && tile_offsets.numel() == num_experts + 1,
        "offsets must be [E+1]"
    );
    ensure!(
        ids_sorted.numel() == s && slot_of.numel() == s,
        "sorted views must be [S]"
    );
    let hist = ctx.pipeline("moe_histogram", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &hist,
        &[indices.binding(), counts.binding()],
        &[&u32_bytes(s)],
        Grid::Threads { grid: (s, 1, 1), threadgroup: (64, 1, 1) },
    )?;
    // Histogram → scans → scatter: each reads what the previous wrote.
    pass.level_barrier(&[b.counts])?;
    let scan = ctx.pipeline("moe_scan_offsets", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &scan,
        &[counts.binding(), offsets.binding(), tile_offsets.binding()],
        &[&u32_bytes(num_experts), &u32_bytes(tile_m)],
        // One threadgroup; the kernel's MOE_SCAN_THREADS.
        Grid::Threadgroups { groups: (1, 1, 1), threadgroup: (256, 1, 1) },
    )?;
    pass.level_barrier(&[b.offsets, b.tile_offsets, b.cursors])?;
    let scatter = ctx.pipeline("moe_scatter_slots", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &scatter,
        &[
            indices.binding(),
            offsets.binding(),
            cursors.binding(),
            ids_sorted.binding(),
            slot_of.binding(),
        ],
        &[&u32_bytes(top_k), &u32_bytes(s)],
        Grid::Threads { grid: (s, 1, 1), threadgroup: (64, 1, 1) },
    )
}

/// Builds a tile-major grouped-GEMM map; `tile_m` must match the offset scan.
/// Unused capacity receives sentinel blocks.
#[allow(clippy::too_many_arguments)]
pub fn moe_build_blocks(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    offsets: &Tensor,
    tile_offsets: &Tensor,
    blocks: &Tensor,
    slot_of: Option<&Tensor>,
    num_experts: usize,
    n_per: usize,
    tile_m: usize,
) -> Result<()> {
    ensure!(n_per.is_multiple_of(64), "N per expert must be a multiple of 64");
    ensure!(tile_m > 0, "row-tile height must be positive");
    let n_tiles = n_per / 64;
    ensure!(
        blocks.dtype() == DType::U32 && blocks.numel().is_multiple_of(n_tiles * 4),
        "block map must be U32 sized as tiles x {n_tiles} x uint4"
    );
    if let Some(t) = slot_of {
        ensure!(
            t.numel() >= num_experts && t.dtype() == DType::U32,
            "slot table must be U32 [E]"
        );
    }
    let tile_capacity = blocks.numel() / (n_tiles * 4);
    let pipeline = ctx.pipeline("moe_build_blocks", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            offsets.binding(),
            tile_offsets.binding(),
            blocks.binding(),
            slot_of.unwrap_or(offsets).binding(),
        ],
        &[
            &u32_bytes(num_experts),
            &u32_bytes(n_per),
            &u32_bytes(tile_m),
            &u32_bytes(0),
            &u32_bytes(usize::from(slot_of.is_some())),
        ],
        Grid::Threads { grid: (tile_capacity, n_tiles, 1), threadgroup: (32, 1, 1) },
    )
}

/// `slots[i] = slot_of[indices[i]]` over `n` routed ids: the expert-cache
/// slots the gather kernels address instead of the expert ids.
pub fn moe_remap_slots(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    indices: &Tensor,
    slot_of: &Tensor,
    slots: &Tensor,
    n: usize,
) -> Result<()> {
    ensure!(
        indices.numel() >= n && slots.numel() >= n,
        "remap of {n} ids over {} indices and {} slots",
        indices.numel(),
        slots.numel()
    );
    ensure!(
        indices.dtype() == DType::U32
            && slots.dtype() == DType::U32
            && slot_of.dtype() == DType::U32,
        "remap tensors must be U32"
    );
    let pipeline = ctx.pipeline("moe_remap_slots", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[indices.binding(), slot_of.binding(), slots.binding()],
        &[&u32_bytes(n)],
        Grid::Threads { grid: (n, 1, 1), threadgroup: (n.min(256), 1, 1) },
    )
}

/// Writes `out[row] = Σ_k scores[row,k] * ed[slots[row,k]]`.
pub fn moe_combine_rows(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    ed: &Tensor,
    slots: &Tensor,
    scores: &Tensor,
    out: &Tensor,
    top_k: usize,
) -> Result<()> {
    ensure!(out.shape().len() == 2, "out must be [m, h]");
    let (m, h) = (out.shape()[0], out.shape()[1]);
    ensure!(h.is_multiple_of(4), "combine needs h % 4 == 0 (got {h})");
    ensure!(
        ed.dtype() == DType::BF16 && ed.numel().is_multiple_of(h),
        "ed must be [S, h]"
    );
    ensure!(
        slots.dtype() == DType::U32 && slots.numel() == m * top_k,
        "slots must be U32 [m, top_k]"
    );
    ensure!(
        scores.dtype() == DType::F32 && scores.numel() == m * top_k,
        "scores must be F32 [m, top_k]"
    );
    let pipeline = ctx.pipeline("moe_combine_rows", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[ed.binding(), slots.binding(), scores.binding(), out.binding()],
        &[&u32_bytes(h), &u32_bytes(top_k)],
        Grid::Threads { grid: (h / 4, m, 1), threadgroup: (32, 1, 1) },
    )
}

/// [`moe_combine_rows`] then [`moe_row_gate_add`] in one dispatch:
/// `out[row] = bf16(bf16(Σ_k scores[row,k] * ed[slots[row,k]]) +
/// sigmoid(gate[row]) * shared[row])`, bit-identical to the two passes.
#[allow(clippy::too_many_arguments)]
pub fn moe_combine_rows_gate_add(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    ed: &Tensor,
    slots: &Tensor,
    scores: &Tensor,
    shared: &Tensor,
    gate: &Tensor,
    out: &Tensor,
    top_k: usize,
) -> Result<()> {
    ensure!(out.shape().len() == 2, "out must be [m, h]");
    let (m, h) = (out.shape()[0], out.shape()[1]);
    ensure!(h.is_multiple_of(4), "combine needs h % 4 == 0 (got {h})");
    ensure!(
        ed.dtype() == DType::BF16 && ed.numel().is_multiple_of(h),
        "ed must be [S, h]"
    );
    ensure!(
        slots.dtype() == DType::U32 && slots.numel() == m * top_k,
        "slots must be U32 [m, top_k]"
    );
    ensure!(
        scores.dtype() == DType::F32 && scores.numel() == m * top_k,
        "scores must be F32 [m, top_k]"
    );
    ensure!(
        shared.numel() == m * h && shared.dtype() == DType::BF16,
        "shared must be BF16 [m, h]"
    );
    ensure!(
        gate.numel() == m && gate.dtype() == DType::BF16,
        "gate logits must be BF16 [m]"
    );
    let pipeline =
        ctx.pipeline("moe_combine_rows_gate_add", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            ed.binding(),
            slots.binding(),
            scores.binding(),
            shared.binding(),
            gate.binding(),
            out.binding(),
        ],
        &[&u32_bytes(h), &u32_bytes(top_k)],
        Grid::Threads { grid: (h / 4, m, 1), threadgroup: (32, 1, 1) },
    )
}

/// `dst[r, :] += sigmoid(gate[r]) * src[r, :]` (prefill shared-expert add).
pub fn moe_row_gate_add(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    src: &Tensor,
    gate: &Tensor,
    dst: &Tensor,
) -> Result<()> {
    ensure!(dst.shape().len() == 2, "dst must be [m, h]");
    let (m, h) = (dst.shape()[0], dst.shape()[1]);
    ensure!(src.numel() == m * h && src.dtype() == DType::BF16, "src must be [m, h]");
    ensure!(
        gate.numel() == m && gate.dtype() == DType::BF16,
        "gate logits must be BF16 [m]"
    );
    let pipeline = ctx.pipeline("moe_row_gate_add", SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[src.binding(), gate.binding(), dst.binding()],
        &[&u32_bytes(h)],
        Grid::Threads { grid: (h, m, 1), threadgroup: (32, 1, 1) },
    )
}

#[cfg(test)]
#[path = "moe_injection_tests.rs"]
mod injection_tests;

// Official-checkpoint candidates use separate source and entry points.
// Legacy SOURCE and all its original Metal bodies remain unchanged.
const OFFICIAL_MOE_SOURCE: &str = include_str!("metal/official_moe.metal");

/// Exact model geometry audited against the served affine4 arm. The policy
/// has not been qualified for other Qwen checkpoints or MoE architectures.
pub(crate) fn official_moe_shape_supported(
    experts: usize,
    top_k: usize,
    hidden: usize,
    intermediate: usize,
) -> bool {
    (experts, top_k, hidden, intermediate) == (512, 10, 2560, 640)
}

/// Build the finite-domain activation table outside the decode loop. This
/// candidate's primitive spelling needs exhaustive stock-MLX table controls
/// before the official policy is activated; source similarity is insufficient.
pub(crate) fn official_moe_sigmoid_table(ctx: &MetalContext) -> Result<Tensor> {
    let table = Tensor::zeros(ctx, &[65536], DType::BF16)?;
    let pass = ctx.begin()?;
    let pipeline = ctx.pipeline(
        "official_moe_sigmoid_table",
        OFFICIAL_MOE_SOURCE,
        MslVersion::V3_1,
    )?;
    pass.dispatch_at(
        &pipeline,
        &[table.binding()],
        &[],
        Grid::Threads { grid: (65536, 1, 1), threadgroup: (256, 1, 1) },
    )?;
    pass.commit_wait()?;
    Ok(table)
}

/// Served fused-router arithmetic over BF16 logits. Output storage remains
/// F32 for the native expert graph; every stored value is rounded BF16.
/// The optional gate reproduces softmax_gate's spare-SIMDgroup reduction.
#[allow(clippy::too_many_arguments)]
pub(crate) fn official_moe_router(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    logits: &Tensor,
    indices: &Tensor,
    scores: &Tensor,
    renorm: bool,
    gate: Option<(&Tensor, &Tensor, &Tensor)>,
) -> Result<()> {
    ensure!(renorm, "official MoE router requires normalized top-k probabilities");
    let e =
        *logits.shape().last().ok_or_else(|| anyhow::anyhow!("empty router shape"))?;
    ensure!(
        e == 512 && logits.dtype() == DType::BF16,
        "official router needs BF16 [..., 512]"
    );
    let m = logits.numel() / e;
    ensure!(m > 0 && indices.numel().is_multiple_of(m), "official router row geometry");
    let k = indices.numel() / m;
    ensure!(
        k > 0 && k <= ROUTER_MAX_K && indices.dtype() == DType::U32,
        "official router indices"
    );
    ensure!(
        scores.dtype() == DType::F32 && scores.numel() == m * k,
        "official router scores"
    );
    let (name, gx, gw, glog, gate_k) = match gate {
        Some((x, w, out)) => {
            ensure!(
                m == 1 && x.dtype() == DType::BF16 && x.numel() == 2560,
                "official folded shared gate needs one BF16 hidden row"
            );
            ensure!(
                w.dtype() == DType::BF16 && w.numel() == 2560,
                "official folded shared gate needs direct BF16 weights"
            );
            ensure!(
                out.dtype() == DType::BF16 && out.numel() == 1,
                "official folded shared gate output"
            );
            ("official_moe_router_gate_bf16", x, w, out, 2560)
        }
        None => ("official_moe_router_bf16", logits, logits, logits, 0),
    };
    let pipeline = ctx.pipeline(name, OFFICIAL_MOE_SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            logits.binding(),
            indices.binding(),
            scores.binding(),
            gx.binding(),
            gw.binding(),
            glog.binding(),
        ],
        &[&u32_bytes(e), &u32_bytes(k), &u32_bytes(gate_k)],
        Grid::Threadgroups { groups: (m, 1, 1), threadgroup: (256, 1, 1) },
    )
}

fn check_official_bank(
    w: &QuantWeights,
    experts: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    ensure!(
        w.bits == 4 && w.group_size == 64,
        "official affine4 decode is unqualified outside Q4/group64"
    );
    ensure!(
        w.out_features() == experts * n && w.in_features() == k,
        "official affine4 decode bank geometry"
    );
    ensure!(
        w.codes.dtype() == DType::U32
            && w.scales.dtype() == DType::BF16
            && w.biases.dtype() == DType::BF16,
        "official affine4 decode bank precision"
    );
    ensure!(
        w.scales.numel() == experts * n * k / 64
            && w.biases.numel() == w.scales.numel(),
        "official affine4 parameter geometry"
    );
    Ok(())
}

pub(crate) fn official_moe_banks_supported(
    gate: &QuantWeights,
    up: &QuantWeights,
    down: &QuantWeights,
) -> Result<()> {
    check_official_bank(gate, 512, 640, 2560)?;
    check_official_bank(up, 512, 640, 2560)?;
    check_official_bank(down, 512, 2560, 640)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn official_moe_gate_up(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &QuantWeights,
    up: &QuantWeights,
    x: &Tensor,
    indices: &Tensor,
    table: &Tensor,
    out: &Tensor,
) -> Result<()> {
    check_official_bank(gate, 512, 640, 2560)?;
    check_official_bank(up, 512, 640, 2560)?;
    ensure!(x.dtype() == DType::BF16 && x.numel() == 2560, "official gate/up input");
    ensure!(
        indices.dtype() == DType::U32 && indices.numel() == 10,
        "official gate/up indices"
    );
    ensure!(
        out.dtype() == DType::BF16 && out.numel() == 10 * 640,
        "official gate/up output"
    );
    ensure!(
        table.dtype() == DType::BF16 && table.numel() == 65536,
        "official sigmoid table"
    );
    let pipeline =
        ctx.pipeline("official_moe_gate_up_a4", OFFICIAL_MOE_SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            gate.codes.binding(),
            gate.scales.binding(),
            gate.biases.binding(),
            up.codes.binding(),
            up.scales.binding(),
            up.biases.binding(),
            x.binding(),
            indices.binding(),
            table.binding(),
            out.binding(),
        ],
        &[&u32_bytes(2560), &u32_bytes(640), &u32_bytes(64)],
        Grid::Threadgroups { groups: (640 / 2, 10, 1), threadgroup: (64, 1, 1) },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn official_moe_down(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    w: &QuantWeights,
    x: &Tensor,
    indices: &Tensor,
    scores: &Tensor,
    shared: Option<(&Tensor, &Tensor)>,
    out: &Tensor,
) -> Result<()> {
    check_official_bank(w, 512, 2560, 640)?;
    ensure!(x.dtype() == DType::BF16 && x.numel() == 10 * 640, "official down input");
    ensure!(
        indices.dtype() == DType::U32 && indices.numel() == 10,
        "official down indices"
    );
    ensure!(
        scores.dtype() == DType::F32 && scores.numel() == 10,
        "official down scores"
    );
    ensure!(out.dtype() == DType::BF16 && out.numel() == 2560, "official down output");
    let (sdown, glog) = match shared {
        Some((sd, gl)) => {
            ensure!(
                sd.dtype() == DType::BF16
                    && sd.numel() == 2560
                    && gl.dtype() == DType::BF16
                    && gl.numel() == 1,
                "official shared operands"
            );
            (sd, gl)
        }
        None => (x, x),
    };
    let pipeline =
        ctx.pipeline("official_moe_down_a4", OFFICIAL_MOE_SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[
            w.codes.binding(),
            w.scales.binding(),
            w.biases.binding(),
            x.binding(),
            indices.binding(),
            scores.binding(),
            sdown.binding(),
            glog.binding(),
            out.binding(),
        ],
        &[
            &u32_bytes(640),
            &u32_bytes(2560),
            &u32_bytes(64),
            &u32_bytes(10),
            &u32_bytes(shared.is_some() as usize),
        ],
        Grid::Threadgroups { groups: (2560 / 2, 1, 1), threadgroup: (64, 1, 1) },
    )
}

pub(crate) fn official_moe_silu_mul(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &Tensor,
    up: &Tensor,
    table: &Tensor,
    out: &Tensor,
) -> Result<()> {
    let n = out.numel();
    ensure!(
        gate.numel() == n
            && up.numel() == n
            && [gate, up, out, table].iter().all(|t| t.dtype() == DType::BF16)
            && table.numel() == 65536,
        "official SwiGLU tensor contract"
    );
    let pipeline =
        ctx.pipeline("official_moe_silu_mul", OFFICIAL_MOE_SOURCE, MslVersion::V3_1)?;
    pass.dispatch_at(
        &pipeline,
        &[gate.binding(), up.binding(), table.binding(), out.binding()],
        &[&u32_bytes(n)],
        Grid::Threads { grid: (n, 1, 1), threadgroup: (256, 1, 1) },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn official_moe_prefill_combine(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    ed: &Tensor,
    slots: &Tensor,
    scores: &Tensor,
    shared: &Tensor,
    gate: &Tensor,
    out: &Tensor,
) -> Result<()> {
    let h = 2560;
    let m = gate.numel();
    ensure!(
        m > 0 && slots.dtype() == DType::U32 && slots.numel() == m * 10,
        "official prefill slots"
    );
    ensure!(
        scores.dtype() == DType::F32 && scores.numel() == m * 10,
        "official prefill scores"
    );
    ensure!(
        [ed, shared, gate, out].iter().all(|t| t.dtype() == DType::BF16)
            && ed.numel() == m * 10 * h
            && shared.numel() == m * h
            && out.numel() == m * h,
        "official prefill combine geometry"
    );
    let pipeline = ctx.pipeline(
        "official_moe_prefill_combine",
        OFFICIAL_MOE_SOURCE,
        MslVersion::V3_1,
    )?;
    pass.dispatch_at(
        &pipeline,
        &[
            ed.binding(),
            slots.binding(),
            scores.binding(),
            shared.binding(),
            gate.binding(),
            out.binding(),
        ],
        &[&u32_bytes(h), &u32_bytes(10)],
        Grid::Threads { grid: (h, m, 1), threadgroup: (256, 1, 1) },
    )
}

#[cfg(test)]
#[path = "official_moe_compat_tests.rs"]
mod official_compat_tests;

#[cfg(test)]
#[path = "official_moe_oracle_tests.rs"]
mod official_oracle_tests;

#[cfg(test)]
#[path = "official_moe_affine4_dot_tests.rs"]
mod official_affine4_dot_tests;
