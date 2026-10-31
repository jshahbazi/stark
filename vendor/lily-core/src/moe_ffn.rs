// Modified by Stark: in-process core extraction; see UPSTREAM.md.
//! The model-agnostic sparse-MoE FFN graph: quantized projection dispatch,
//! the decode gather-GEMV path, and the GPU-resident batched prefill path
//! (router → top-k → counting sort → grouped GEMMs → combine), plus the
//! scratch buffers both paths own.

use anyhow::{Result, ensure};

use crate::kernels::elementwise::{
    gather_rows_bf16, silu_mul_bf16, silu_mul_rows_bf16, split_cols_bf16,
};
use crate::kernels::{moe, quant, skinny};
use crate::metal::{ComputePass, MetalContext};
use crate::tensor::{DType, Tensor};
use crate::weights::{LinearWeights, MoeWeights};

#[cfg(test)]
#[path = "moe_decode_exact_tests.rs"]
mod decode_exact_tests;

/// The FFN shape a model hands the shared graph.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeDims {
    pub num_experts: usize,
    pub top_k: usize,
    pub moe_intermediate: usize,
    pub hidden: usize,
    pub norm_topk_prob: bool,
    /// Strictly admitted external MLX checkpoint operation-rounding policy.
    pub official_mlx_rounding: bool,
}

/// Projects `a` `[m, K]` through an already-stacked weight group as one
/// skinny GEMM into `stack_c` when eligible, returning the `[m, n_total]`
/// view holding the stacked output; `None` when the stack must be projected
/// slice by slice (the caller then dispatches the per-slice projections).
/// `widths` are the slice widths in stack order.
pub(crate) fn project_stack_fused(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    stack_w: &LinearWeights,
    stack_c: &Tensor,
    widths: &[usize],
) -> Result<Option<Tensor>> {
    let m = a.shape()[0];
    let n_total = stack_w.out_features();
    if widths.iter().sum::<usize>() != n_total || !skinny::dense_smallm_routes(m) {
        return Ok(None);
    }
    // Fusing is bit-identical only when the stack and every slice choose
    // the same skinny variant; staged and register-A reduce in different
    // f32 orders.
    let fused = match stack_w.bits {
        4 => {
            let walk_ok =
                skinny::q4_block_walk_ok(stack_w.in_features(), stack_w.group_size);
            skinny::stack_route_uniform(m, n_total, widths, walk_ok)
        }
        // A `--q8-dense` stack: the Q8 kernel's choice for a bf16 output
        // depends on m and the packing only, never on the width, so the
        // stack and its slices always take the same variant (the per-slice
        // `project_mat` route, whose packing condition this repeats).
        8 => {
            stack_w.group_size.is_multiple_of(8)
                && stack_w.in_features().is_multiple_of(8)
                && stack_c.dtype() == DType::BF16
        }
        _ => false,
    };
    if !fused {
        return Ok(None);
    }
    let c = stack_c.view(0, &[m, n_total])?;
    if stack_w.bits == 4 {
        skinny::gemm_skinny_q4_nt(ctx, pass, a, stack_w, &c)?;
    } else {
        skinny::gemm_skinny_q8_nt(ctx, pass, a, stack_w, &c)?;
    }
    Ok(Some(c))
}

/// Projects an already-stacked weight group as one skinny GEMM when eligible
/// and splits the result into the slice outputs; otherwise dispatches the
/// corresponding per-slice projections. Keeping both arms here prevents the
/// fused and fallback projection lists from drifting.
pub(crate) fn project_stack_or_slices<const N: usize>(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    stack_w: &LinearWeights,
    stack_c: &Tensor,
    projections: [(&LinearWeights, &Tensor); N],
    scratch: &Tensor,
) -> Result<()> {
    let m = a.shape()[0];
    let widths: [usize; N] = std::array::from_fn(|i| projections[i].1.numel() / m);
    if let Some(c) = project_stack_fused(ctx, pass, a, stack_w, stack_c, &widths)? {
        pass.level_barrier(&[&c])?;
        let outs: [&Tensor; N] = std::array::from_fn(|i| projections[i].1);
        split_cols_bf16(ctx, pass, &c, &outs)?;
        return Ok(());
    }

    for (w, out) in projections {
        project_mat(ctx, pass, a, w, out, scratch)?;
    }
    Ok(())
}

/// Dispatches a GEMM on the projection's storage precision; quantized weights
/// stage a bf16 dequant into `scratch` first (see `quant::gemm_q4_bf16_nt`),
/// unless the site routes to a skinny small-m kernel (4- and 8-bit). Every
/// route ends with one dispatch writing `c`; ordering after it is the
/// caller's (the dequant staging carries its own internal barrier).
pub(crate) fn project_mat(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &LinearWeights,
    c: &Tensor,
    scratch: &Tensor,
) -> Result<()> {
    let m = a.shape()[0];
    if skinny::dense_smallm_routes(m) {
        if w.bits == 4 {
            return skinny::gemm_skinny_q4_nt(ctx, pass, a, w, c);
        }
        if w.bits == 8
            && w.group_size.is_multiple_of(8)
            && w.in_features().is_multiple_of(8)
        {
            return skinny::gemm_skinny_q8_nt(ctx, pass, a, w, c);
        }
    }
    quant::gemm_quant_bf16_nt(ctx, pass, a, w, c, scratch)
}

/// Verify projection with precisely the ordinary decode GEMV arithmetic.
pub(crate) fn project_mat_decode_exact(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &LinearWeights,
    c: &Tensor,
) -> Result<()> {
    quant::gemv_quant_rows_decode_exact(ctx, pass, w, a, c)
}

/// Decode-side MoE intermediates: the router distribution and the top-k
/// expert projections, all device-resident (no readback per step).
pub(crate) struct MoeScratch {
    pub router_logits: Tensor,
    pub indices: Tensor,
    /// The routed ids as expert-cache slots (`MoeWeights::slot_of`); aliases
    /// `indices` when the experts are resident.
    pub slots: Tensor,
    pub scores: Tensor,
    pub act: Tensor,
    pub shared_out: Tensor,
    pub shared_gate: Tensor,
    /// Startup-only BF16 activation table; absent on legacy paths.
    pub sigmoid_table: Option<Tensor>,
}

impl MoeScratch {
    pub fn new(ctx: &MetalContext, dims: &MoeDims) -> Result<Self> {
        let (e, k, i, h) =
            (dims.num_experts, dims.top_k, dims.moe_intermediate, dims.hidden);
        Ok(Self {
            router_logits: Tensor::zeros(
                ctx,
                &[e],
                if dims.official_mlx_rounding { DType::BF16 } else { DType::F32 },
            )?,
            indices: Tensor::zeros(ctx, &[k], DType::U32)?,
            slots: Tensor::zeros(ctx, &[k], DType::U32)?,
            scores: Tensor::zeros(ctx, &[k], DType::F32)?,
            act: Tensor::zeros(ctx, &[k, i], DType::BF16)?,
            shared_out: Tensor::zeros(ctx, &[h], DType::BF16)?,
            shared_gate: Tensor::zeros(
                ctx,
                &[1],
                if dims.official_mlx_rounding { DType::BF16 } else { DType::F32 },
            )?,
            sigmoid_table: if dims.official_mlx_rounding {
                Some(moe::official_moe_sigmoid_table(ctx)?)
            } else {
                None
            },
        })
    }
}

/// GPU-resident MoE prefill intermediates (capacity-sized, viewed per chunk):
/// the router distribution, the counting-sort state, the sentinel-padded block
/// maps, and the expert-slot activations (`S = m·top_k` rows).
pub(crate) struct PrefillMoeScratch {
    pub router_logits: Tensor,
    pub shared_out: Tensor,
    pub shared_gate: Tensor,
    /// Startup-only BF16 activation table; absent on legacy paths.
    pub sigmoid_table: Option<Tensor>,
    /// Verify-only F32 values; ordinary prefill keeps its original buffers.
    decode_router_logits: Tensor,
    decode_shared_gate: Tensor,
    pub indices: Tensor,
    /// The routed ids as expert-cache slots for the small-m kernels
    /// (`MoeWeights::slot_of`; the grouped route remaps in its block map).
    pub slots: Tensor,
    pub scores: Tensor,
    pub counts: Tensor,
    pub cursors: Tensor,
    pub offsets: Tensor,
    pub tile_offsets: Tensor,
    pub ids_sorted: Tensor,
    pub slot_of: Tensor,
    pub blocks_gu: Tensor,
    pub blocks_dn: Tensor,
    pub gx: Tensor,
    /// `[S, inter]` expert gate / up outputs (grouped route only).
    pub eg: Tensor,
    pub eu: Tensor,
    /// `[S, inter]` expert activations `silu(gate) * up`, one row per
    /// routed pair; both routes' down input.
    pub ea: Tensor,
    /// `[S, h]` expert down outputs (grouped route; the small-m route
    /// combines in-kernel).
    pub ed: Tensor,
    /// Row-tile capacity bound `ceil(S/T) + min(E, S)` at `tile.rows()` —
    /// covers any expert split without a readback; block-map slots past the
    /// real count carry the sentinel the grouped GEMM kernels skip.
    pub tile_capacity: usize,
    /// The grouped-GEMM row-tile config `tile_capacity` was computed at: the
    /// policy's alloc tile for the backing scratch (the sizing bound), the
    /// policy's per-chunk answer for a chunk view.
    pub tile: quant::MoeTile,
    /// The chunk's grouped-MoE execution route: block-mapped grouped GEMM at
    /// `tile`, or the small-m GEMV fast path. Set together with
    /// `tile`/`tile_capacity` in [`Self::chunk`]; `prefill_moe` re-checks it
    /// against the fixed shipped policy (the desync guard).
    pub route: quant::MoeRoute,
    /// Keeps chunk views on a qualified grouped reference route.
    pub official_mlx_rounding: bool,
}

/// A row-prefix view of `t` (same buffer, leading dimension shrunk to `m`).
pub(crate) fn prefix_rows(t: &Tensor, m: usize) -> Result<Tensor> {
    let mut shape = t.shape().to_vec();
    shape[0] = m;
    t.view(0, &shape)
}

impl PrefillMoeScratch {
    /// Capacity-sized buffers for chunks of up to `m` rows.
    pub fn new(ctx: &MetalContext, dims: &MoeDims, m: usize) -> Result<Self> {
        let (e, k, mi, h) =
            (dims.num_experts, dims.top_k, dims.moe_intermediate, dims.hidden);
        let bf = DType::BF16;
        let u32t = DType::U32;
        let moe_policy = quant::moe_tile_mroute_table();
        let s_total = m * k;
        // Sized at the policy's alloc tile: cap is monotone nonincreasing in
        // tile rows, so the smallest tile the policy can emit bounds every
        // m-routed chunk's map.
        let alloc_tile = moe_policy.alloc_tile();
        // Tail chunks have S < E: bounding the partial-tile term by min(E, S)
        // keeps the sentinel share sane on short tails.
        let cap = s_total.div_ceil(alloc_tile.rows()) + e.min(s_total);
        Ok(Self {
            router_logits: Tensor::zeros(ctx, &[m, e], bf)?,
            shared_out: Tensor::zeros(ctx, &[m, h], bf)?,
            shared_gate: Tensor::zeros(ctx, &[m], bf)?,
            sigmoid_table: if dims.official_mlx_rounding {
                Some(moe::official_moe_sigmoid_table(ctx)?)
            } else {
                None
            },
            decode_router_logits: Tensor::zeros(ctx, &[m.min(4), e], DType::F32)?,
            decode_shared_gate: Tensor::zeros(ctx, &[m.min(4)], DType::F32)?,
            indices: Tensor::zeros(ctx, &[m, k], u32t)?,
            slots: Tensor::zeros(ctx, &[m, k], u32t)?,
            scores: Tensor::zeros(ctx, &[m, k], DType::F32)?,
            counts: Tensor::zeros(ctx, &[e], u32t)?,
            cursors: Tensor::zeros(ctx, &[e], u32t)?,
            offsets: Tensor::zeros(ctx, &[e + 1], u32t)?,
            tile_offsets: Tensor::zeros(ctx, &[e + 1], u32t)?,
            ids_sorted: Tensor::zeros(ctx, &[s_total], u32t)?,
            slot_of: Tensor::zeros(ctx, &[m, k], u32t)?,
            blocks_gu: Tensor::zeros(ctx, &[cap * (mi / 64) * 4], u32t)?,
            blocks_dn: Tensor::zeros(ctx, &[cap * (h / 64) * 4], u32t)?,
            gx: Tensor::zeros(ctx, &[s_total, h], bf)?,
            eg: Tensor::zeros(ctx, &[s_total, mi], bf)?,
            eu: Tensor::zeros(ctx, &[s_total, mi], bf)?,
            ea: Tensor::zeros(ctx, &[s_total, mi], bf)?,
            ed: Tensor::zeros(ctx, &[s_total, h], bf)?,
            tile_capacity: cap,
            tile: alloc_tile,
            route: quant::MoeRoute::Grouped(alloc_tile),
            official_mlx_rounding: dims.official_mlx_rounding,
        })
    }

    /// Per-chunk views (`S = m·top_k` slot rows, the chunk's own row-tile
    /// capacity bound). The block-map views are sized to the chunk's bound so
    /// `moe_build_blocks` (grid = view capacity) rewrites exactly the entries
    /// the grouped GEMMs read — stale tiles a larger previous chunk wrote
    /// past this extent are unreachable.
    pub fn chunk(&self, m: usize) -> Result<Self> {
        let e = self.counts.numel();
        let k = self.indices.shape()[1];
        let s_total = m * k;
        // Set route, tile, and capacity together. The GEMV route retains
        // bounded unused block-map views.
        let policy = quant::moe_tile_mroute_table();
        let route = if self.official_mlx_rounding {
            quant::MoeRoute::Grouped(policy.alloc_tile())
        } else {
            policy.route_for(m)
        };
        let tile = match route {
            quant::MoeRoute::Grouped(tile) => tile,
            quant::MoeRoute::SmallmGemv => policy.alloc_tile(),
        };
        let cap = s_total.div_ceil(tile.rows()) + e.min(s_total);
        let slot_rows = |t: &Tensor| prefix_rows(t, s_total);
        let block_view = |t: &Tensor| {
            let words_per_tile = t.numel() / self.tile_capacity;
            t.view(0, &[cap * words_per_tile])
        };
        Ok(Self {
            router_logits: prefix_rows(&self.router_logits, m)?,
            shared_out: prefix_rows(&self.shared_out, m)?,
            shared_gate: prefix_rows(&self.shared_gate, m)?,
            sigmoid_table: self.sigmoid_table.clone(),
            decode_router_logits: prefix_rows(&self.decode_router_logits, m.min(4))?,
            decode_shared_gate: prefix_rows(&self.decode_shared_gate, m.min(4))?,
            indices: prefix_rows(&self.indices, m)?,
            slots: prefix_rows(&self.slots, m)?,
            scores: prefix_rows(&self.scores, m)?,
            counts: self.counts.view(0, self.counts.shape())?,
            cursors: self.cursors.view(0, self.cursors.shape())?,
            offsets: self.offsets.view(0, self.offsets.shape())?,
            tile_offsets: self.tile_offsets.view(0, self.tile_offsets.shape())?,
            ids_sorted: self.ids_sorted.view(0, &[s_total])?,
            slot_of: prefix_rows(&self.slot_of, m)?,
            blocks_gu: block_view(&self.blocks_gu)?,
            blocks_dn: block_view(&self.blocks_dn)?,
            gx: slot_rows(&self.gx)?,
            eg: slot_rows(&self.eg)?,
            eu: slot_rows(&self.eu)?,
            ea: slot_rows(&self.ea)?,
            ed: slot_rows(&self.ed)?,
            tile_capacity: cap,
            tile,
            route,
            official_mlx_rounding: self.official_mlx_rounding,
        })
    }
}

/// The prefill router's projection: `x` `[m, h]` through the router `gate`
/// into `logits` `[m, E]` (bf16), on the route `project_mat` picks for `m`.
/// [`prefill_moe`] and the cross-layer prediction log
/// (`Qwen4ExpModel::enable_expert_prediction`) both route through here and
/// [`router_select_rows`], so a prediction runs the real router's kernels.
pub(crate) fn router_project_rows(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &LinearWeights,
    x: &Tensor,
    logits: &Tensor,
    dequant: &Tensor,
) -> Result<()> {
    project_mat(ctx, pass, x, gate, logits, dequant)
}

/// The prefill routing decision over `logits` `[m, E]`: per-row softmax,
/// top-k into `indices`/`scores` `[m, k]`, renormalised when the model
/// says so.
pub(crate) fn router_select_rows(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    dims: &MoeDims,
    logits: &Tensor,
    indices: &Tensor,
    scores: &Tensor,
) -> Result<()> {
    if dims.official_mlx_rounding {
        return moe::official_moe_router(
            ctx,
            pass,
            logits,
            indices,
            scores,
            dims.norm_topk_prob,
            None,
        );
    }
    moe::moe_router_topk_rows(ctx, pass, logits, indices, scores, dims.norm_topk_prob)
}

/// The decode router's projection: one row `x` `[h]` through the router
/// `gate` into `logits` `[E]` (f32). Shared with the prediction log as
/// [`router_project_rows`] is.
pub(crate) fn router_project_decode(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    gate: &LinearWeights,
    x: &Tensor,
    logits: &Tensor,
) -> Result<()> {
    quant::gemv_quant(ctx, pass, gate, x, logits)
}

/// The decode routing decision over `logits` `[E]` (see
/// [`router_select_rows`]).
pub(crate) fn router_select_decode(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    dims: &MoeDims,
    logits: &Tensor,
    indices: &Tensor,
    scores: &Tensor,
) -> Result<()> {
    if dims.official_mlx_rounding {
        return moe::official_moe_router(
            ctx,
            pass,
            logits,
            indices,
            scores,
            dims.norm_topk_prob,
            None,
        );
    }
    moe::moe_router_topk(ctx, pass, logits, indices, scores, dims.norm_topk_prob)
}

/// The per-chunk tensors the prefill FFN reads and writes.
pub(crate) struct PrefillMoeIo<'a> {
    /// `[m, h]` FFN input.
    pub x: &'a Tensor,
    /// `[m, h]` FFN output (routed experts plus the gated shared expert).
    pub out: &'a Tensor,
    /// Fused-stack GEMM staging (see `project_stack_or_slices`).
    pub stack: &'a Tensor,
    pub mlp_gate: &'a Tensor,
    pub mlp_up: &'a Tensor,
    pub mlp_act: &'a Tensor,
    /// Dequant staging for the batched fallback GEMM.
    pub dequant: &'a Tensor,
}

/// The prefill MoE FFN, GPU-resident end to end: router GEMM → per-row
/// softmax/top-k → then either (grouped route, m > 8) counting sort
/// (histogram, serial scans, atomic scatter) → GPU-built block map
/// (sentinel-padded to a readback-free capacity bound) → grouped expert GEMMs
/// → per-row combine with the gated shared-expert add, or (small-m route, m <= 8)
/// the two fused small-batch expert kernels (gate + up + SwiGLU over the
/// union of the routed experts; down + combine + shared add per row), the
/// batched shape of the decode gathers. No host round-trip and no per-layer
/// allocations, so the whole chunk stays in one command buffer.
pub(crate) fn prefill_moe(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    dims: &MoeDims,
    moe_w: &MoeWeights,
    io: &PrefillMoeIo<'_>,
    ms: &PrefillMoeScratch,
) -> Result<()> {
    let m = io.x.shape()[0];
    let (e, top_k, inter) = (dims.num_experts, dims.top_k, dims.moe_intermediate);
    let h = dims.hidden;
    let policy = quant::moe_tile_mroute_table();
    let expected_route = if dims.official_mlx_rounding {
        ensure!(
            moe::official_moe_shape_supported(e, top_k, h, inter),
            "official MoE prefill geometry is unqualified"
        );
        moe::official_moe_banks_supported(
            &moe_w.expert_gate,
            &moe_w.expert_up,
            &moe_w.expert_down,
        )?;
        ensure!(
            moe_w.slot_of.is_none(),
            "official MoE policy requires resident expert banks"
        );
        quant::MoeRoute::Grouped(policy.alloc_tile())
    } else {
        policy.route_for(m)
    };
    // Desync guard, extended to the route: the chunk view and this function
    // must agree on the policy's answer — a mismatch would consume block maps
    // scanned at another tile (dropping rows SILENTLY) or run the GEMV path
    // over unsorted state the grouped kernels needed.
    let s_slots = m * top_k;
    ensure!(
        ms.route == expected_route
            && ms.official_mlx_rounding == dims.official_mlx_rounding,
        "prefill MoE route desync: chunk m={m} carries route={} but the policy routes {}",
        ms.route.label(),
        expected_route.label(),
    );
    if let quant::MoeRoute::Grouped(tile) = ms.route {
        ensure!(
            ms.tile == tile
                && ms.tile_capacity
                    == s_slots.div_ceil(ms.tile.rows()) + e.min(s_slots),
            "prefill MoE tile desync: chunk m={m} carries tile={} cap={} but the policy routes {}",
            ms.tile.label(),
            ms.tile_capacity,
            tile.label(),
        );
    }
    // Level structure for a concurrent encoder (serial encoders ignore the
    // barriers): the shared expert depends only on `x`, so its projections
    // are encoded alongside the router and the two chains meet at the final
    // gated add.
    let shared = &moe_w.shared;
    router_project_rows(ctx, pass, &moe_w.gate, io.x, &ms.router_logits, io.dequant)?;
    // The shared expert's gate|up as one stacked GEMM when eligible: its
    // SwiGLU then reads the two halves in place (no split copy).
    let inter_s = io.mlp_gate.numel() / m;
    let stacked_gu = if dims.official_mlx_rounding {
        None
    } else {
        project_stack_fused(
            ctx,
            pass,
            io.x,
            &shared.gate_up_proj,
            io.stack,
            &[inter_s, inter_s],
        )?
    };
    if stacked_gu.is_none() {
        project_mat(ctx, pass, io.x, &shared.gate_proj, io.mlp_gate, io.dequant)?;
        project_mat(ctx, pass, io.x, &shared.up_proj, io.mlp_up, io.dequant)?;
    }
    project_mat(ctx, pass, io.x, &moe_w.shared_gate, &ms.shared_gate, io.dequant)?;
    pass.level_barrier(&[
        &ms.router_logits,
        io.stack,
        io.mlp_gate,
        io.mlp_up,
        &ms.shared_gate,
    ])?;
    router_select_rows(ctx, pass, dims, &ms.router_logits, &ms.indices, &ms.scores)?;
    if dims.official_mlx_rounding {
        moe::official_moe_silu_mul(
            ctx,
            pass,
            io.mlp_gate,
            io.mlp_up,
            ms.sigmoid_table
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing official sigmoid table"))?,
            io.mlp_act,
        )?;
    } else {
        match &stacked_gu {
            Some(gu) => silu_mul_rows_bf16(ctx, pass, gu, io.mlp_act)?,
            None => silu_mul_bf16(ctx, pass, io.mlp_gate, io.mlp_up, io.mlp_act)?,
        }
    }
    pass.level_barrier(&[&ms.indices, &ms.scores, io.mlp_act])?;
    let slots = match &moe_w.slot_of {
        Some(table) => {
            resolve_experts(pass, moe_w, &ms.indices, s_slots)?;
            moe::moe_remap_slots(ctx, pass, &ms.indices, table, &ms.slots, s_slots)?;
            pass.level_barrier(&[&ms.slots])?;
            &ms.slots
        }
        None => &ms.indices,
    };
    project_mat(ctx, pass, io.mlp_act, &shared.down_proj, &ms.shared_out, io.dequant)?;

    let tile = match ms.route {
        quant::MoeRoute::Grouped(tile) => tile,
        quant::MoeRoute::SmallmGemv => {
            ensure!(
                moe_w.expert_gate.bits == 4
                    && moe_w.expert_up.bits == 4
                    && moe_w.expert_down.bits == 4,
                "small-m route requires q4 experts"
            );
            // The fused kernels consume the router output directly in
            // natural (row, k) pair order: no union map, no counting sort,
            // no gather, no block maps, and the combine and the shared add
            // ride in the down kernel.
            moe::moe_smallm_gate_up(
                ctx,
                pass,
                &moe_w.expert_gate,
                &moe_w.expert_up,
                inter,
                io.x,
                slots,
                &ms.ea,
                top_k,
            )?;
            pass.level_barrier(&[&ms.ea, &ms.shared_out])?;
            return moe::moe_smallm_down_combine(
                ctx,
                pass,
                &moe_w.expert_down,
                h,
                &ms.ea,
                slots,
                &ms.scores,
                &ms.shared_out,
                &ms.shared_gate,
                io.out,
                top_k,
            );
        }
    };

    moe::fill_zero_u32(ctx, pass, &ms.counts)?;
    moe::fill_zero_u32(ctx, pass, &ms.cursors)?;
    pass.level_barrier(&[&ms.counts, &ms.cursors])?;
    moe::moe_sort_slots(
        ctx,
        pass,
        &moe::MoeSortBuffers {
            indices: &ms.indices,
            counts: &ms.counts,
            cursors: &ms.cursors,
            offsets: &ms.offsets,
            tile_offsets: &ms.tile_offsets,
            ids_sorted: &ms.ids_sorted,
            slot_of: &ms.slot_of,
        },
        e,
        top_k,
        tile.rows(),
    )?;
    pass.level_barrier(&[&ms.ids_sorted, &ms.offsets, &ms.tile_offsets, &ms.slot_of])?;
    gather_rows_bf16(ctx, pass, io.x, &ms.ids_sorted, &ms.gx)?;
    moe::moe_build_blocks(
        ctx,
        pass,
        &ms.offsets,
        &ms.tile_offsets,
        &ms.blocks_gu,
        moe_w.slot_of.as_ref(),
        e,
        inter,
        tile.rows(),
    )?;
    moe::moe_build_blocks(
        ctx,
        pass,
        &ms.offsets,
        &ms.tile_offsets,
        &ms.blocks_dn,
        moe_w.slot_of.as_ref(),
        e,
        h,
        tile.rows(),
    )?;
    pass.level_barrier(&[&ms.gx, &ms.blocks_gu, &ms.blocks_dn])?;

    let nb_gu = ms.tile_capacity * (inter / 64);
    let nb_dn = ms.tile_capacity * (h / 64);
    quant::gemm_q4_grouped_nt(
        ctx,
        pass,
        &ms.gx,
        &moe_w.expert_gate,
        &ms.eg,
        &ms.blocks_gu,
        nb_gu,
        tile,
    )?;
    quant::gemm_q4_grouped_nt(
        ctx,
        pass,
        &ms.gx,
        &moe_w.expert_up,
        &ms.eu,
        &ms.blocks_gu,
        nb_gu,
        tile,
    )?;
    pass.level_barrier(&[&ms.eg, &ms.eu])?;
    if dims.official_mlx_rounding {
        moe::official_moe_silu_mul(
            ctx,
            pass,
            &ms.eg,
            &ms.eu,
            ms.sigmoid_table
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing official sigmoid table"))?,
            &ms.ea,
        )?;
    } else {
        silu_mul_bf16(ctx, pass, &ms.eg, &ms.eu, &ms.ea)?;
    }
    pass.level_barrier(&[&ms.ea])?;
    quant::gemm_q4_grouped_nt(
        ctx,
        pass,
        &ms.ea,
        &moe_w.expert_down,
        &ms.ed,
        &ms.blocks_dn,
        nb_dn,
        tile,
    )?;
    // Both chains are complete: the routed rows are in `ed`, the shared
    // expert in `shared_out`; one pass combines and adds them.
    pass.level_barrier(&[&ms.ed, &ms.shared_out])?;
    if dims.official_mlx_rounding {
        return moe::official_moe_prefill_combine(
            ctx,
            pass,
            &ms.ed,
            &ms.slot_of,
            &ms.scores,
            &ms.shared_out,
            &ms.shared_gate,
            io.out,
        );
    }
    moe::moe_combine_rows_gate_add(
        ctx,
        pass,
        &ms.ed,
        &ms.slot_of,
        &ms.scores,
        &ms.shared_out,
        &ms.shared_gate,
        io.out,
        top_k,
    )
}

/// Row-batched MoE with the exact ordinary-decode arithmetic at every
/// projection, router, SwiGLU and combine boundary. Only addressing and grid
/// rows change; the original decode kernel bodies provide the calculations.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_moe_decode_exact(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    dims: &MoeDims,
    w: &MoeWeights,
    io: &PrefillMoeIo<'_>,
    ms: &PrefillMoeScratch,
    decode_scratch: &MoeScratch,
) -> Result<()> {
    let injected = prefill_moe_decode_exact_with_injection(
        ctx,
        pass,
        dims,
        w,
        io,
        ms,
        decode_scratch,
        None,
    )?;
    ensure!(!injected, "absent HC injection changed passthrough semantics");
    Ok(())
}

/// The exact verifier may fuse its subsequent HC write into the final
/// projection. `None` preserves the branch-only path and returns false.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_moe_decode_exact_with_injection(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    dims: &MoeDims,
    w: &MoeWeights,
    io: &PrefillMoeIo<'_>,
    ms: &PrefillMoeScratch,
    _decode_scratch: &MoeScratch,
    injection: Option<moe::HcInjection<'_>>,
) -> Result<bool> {
    let m = io.x.shape()[0];
    ensure!((1..=4).contains(&m), "decode-exact MoE supports 1..4 rows");
    let h = dims.hidden;
    let top_k = dims.top_k;
    let inter = dims.moe_intermediate;
    let shared_inter = w.shared.gate_proj.out_features();
    let shared_gu = io.stack.view(0, &[m, 2 * shared_inter])?;
    project_mat_decode_exact(ctx, pass, io.x, &w.gate, &ms.decode_router_logits)?;
    project_mat_decode_exact(ctx, pass, io.x, &w.shared_gate, &ms.decode_shared_gate)?;
    project_mat_decode_exact(ctx, pass, io.x, &w.shared.gate_up_proj, &shared_gu)?;
    pass.level_barrier(&[
        &ms.decode_router_logits,
        &ms.decode_shared_gate,
        &shared_gu,
    ])?;
    moe::moe_router_topk_decode_rows(
        ctx,
        pass,
        &ms.decode_router_logits,
        &ms.indices,
        &ms.scores,
        dims.norm_topk_prob,
    )?;
    silu_mul_rows_bf16(ctx, pass, &shared_gu, io.mlp_act)?;
    pass.level_barrier(&[&ms.indices, &ms.scores, io.mlp_act])?;
    let slots = match &w.slot_of {
        Some(table) => {
            resolve_experts(pass, w, &ms.indices, m * top_k)?;
            moe::moe_remap_slots(ctx, pass, &ms.indices, table, &ms.slots, m * top_k)?;
            pass.level_barrier(&[&ms.slots])?;
            &ms.slots
        }
        None => &ms.indices,
    };
    pass.profile_stage(crate::metal::stage_profile::Phase::MoeGateUp)?;
    moe::moe_gate_up_decode_rows(
        ctx,
        pass,
        &w.expert_gate,
        &w.expert_up,
        inter,
        io.x,
        slots,
        &ms.ea,
        top_k,
    )?;
    project_mat_decode_exact(
        ctx,
        pass,
        io.mlp_act,
        &w.shared.down_proj,
        &ms.shared_out,
    )?;
    pass.level_barrier(&[&ms.ea, &ms.shared_out])?;
    pass.profile_stage(crate::metal::stage_profile::Phase::MoeDown)?;
    if let Some(injection) = injection {
        moe::moe_down_combine_inject_decode_rows(
            ctx,
            pass,
            &w.expert_down,
            h,
            &ms.ea,
            slots,
            &ms.scores,
            &ms.shared_out,
            &ms.decode_shared_gate,
            io.out,
            top_k,
            injection,
        )?;
        return Ok(true);
    }
    moe::moe_down_combine_decode_rows(
        ctx,
        pass,
        &w.expert_down,
        h,
        &ms.ea,
        slots,
        &ms.scores,
        &ms.shared_out,
        &ms.decode_shared_gate,
        io.out,
        top_k,
    )?;
    Ok(false)
}

/// With a served expert cache: has the routed ids in `indices` (`count`
/// of them, written by the router) resolved into slots before the pass
/// goes on. The pass signals the cache's `routed` event once the router
/// is done and waits on `ready` for the same sequence number; the service
/// thread loads the missing experts in between (`expert_cache`).
fn resolve_experts(
    pass: &ComputePass<'_>,
    moe_w: &MoeWeights,
    indices: &Tensor,
    count: usize,
) -> Result<()> {
    let _ = (pass, moe_w, indices, count);
    Ok(())
}

/// The single-token tensors the decode FFN reads and writes.
pub(crate) struct DecodeMoeIo<'a> {
    /// `[h]` FFN input.
    pub x: &'a Tensor,
    /// `[h]` FFN output.
    pub out: &'a Tensor,
    /// `[2*inter]` shared-expert gate|up matvec output; `mlp_gate`/`mlp_up`
    /// are its two halves.
    pub mlp_gu: &'a Tensor,
    pub mlp_gate: &'a Tensor,
    pub mlp_up: &'a Tensor,
    pub mlp_act: &'a Tensor,
    /// If present, fuse the final branch's HC write into its projection.
    pub hc_injection: Option<moe::HcInjection<'a>>,
}

/// The decode MoE FFN, fully on-GPU: router GEMV + softmax/top-k kernel,
/// gather-GEMVs over the stacked experts through the device-resident indices,
/// the shared expert through the dense-MLP scratch, and the score-weighted
/// combine into `io.out`. Encoded for a concurrent pass: every inter-level
/// data edge is marked with a barrier except the final write, which the
/// caller orders. Returns whether the optional HC stream write was encoded.
pub(crate) fn decode_moe(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    dims: &MoeDims,
    moe_w: &MoeWeights,
    io: &DecodeMoeIo<'_>,
    ms: &MoeScratch,
) -> Result<bool> {
    let inter = dims.moe_intermediate;
    if dims.official_mlx_rounding {
        ensure!(
            moe::official_moe_shape_supported(
                dims.num_experts,
                dims.top_k,
                dims.hidden,
                inter
            ),
            "official MoE decode geometry is unqualified"
        );
        moe::official_moe_banks_supported(
            &moe_w.expert_gate,
            &moe_w.expert_up,
            &moe_w.expert_down,
        )?;
        ensure!(
            moe_w.slot_of.is_none(),
            "official MoE policy requires resident expert banks"
        );
        ensure!(
            io.hc_injection.is_none(),
            "official HC write must remain a separate qualified operation"
        );
        ensure!(
            moe_w.shared_gate.bits == 16,
            "official folded shared gate needs direct BF16 weights"
        );
    }
    // The shared expert reads only `x` and writes only its own scratch, so
    // its whole chain is independent of the router and of the expert gathers;
    // encoding it interleaved puts the two on the same dependency levels.
    let shared = &moe_w.shared;

    if crate::ablate::on(crate::ablate::Group::Moe) {
        return Ok(false);
    }
    pass.profile_stage(crate::metal::stage_profile::Phase::MoeInputProjections)?;
    // Level 0 — everything whose only input is `x`.
    if !crate::ablate::on(crate::ablate::Group::Router) {
        router_project_decode(ctx, pass, &moe_w.gate, io.x, &ms.router_logits)?;
    }
    if !crate::ablate::on(crate::ablate::Group::Shared) {
        quant::gemv_quant(ctx, pass, &shared.gate_up_proj, io.x, io.mlp_gu)?;
        if !dims.official_mlx_rounding {
            quant::gemv_quant(ctx, pass, &moe_w.shared_gate, io.x, &ms.shared_gate)?;
        }
    }
    pass.level_barrier(&[&ms.router_logits, io.mlp_gu, &ms.shared_gate])?;

    pass.profile_stage(crate::metal::stage_profile::Phase::MoeRouteSharedActivation)?;
    // Level 1 — the routing decision, and the shared expert's activation.
    if !crate::ablate::on(crate::ablate::Group::Topk) {
        if dims.official_mlx_rounding {
            moe::official_moe_router(
                ctx,
                pass,
                &ms.router_logits,
                &ms.indices,
                &ms.scores,
                dims.norm_topk_prob,
                Some((io.x, &moe_w.shared_gate.codes, &ms.shared_gate)),
            )?;
            moe::official_moe_silu_mul(
                ctx,
                pass,
                io.mlp_gate,
                io.mlp_up,
                ms.sigmoid_table
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing official sigmoid table"))?,
                io.mlp_act,
            )?;
        } else {
            router_select_decode(
                ctx,
                pass,
                dims,
                &ms.router_logits,
                &ms.indices,
                &ms.scores,
            )?;
            silu_mul_bf16(ctx, pass, io.mlp_gate, io.mlp_up, io.mlp_act)?;
        }
        if dims.official_mlx_rounding {
            pass.level_barrier(&[
                &ms.indices,
                &ms.scores,
                io.mlp_act,
                &ms.shared_gate,
            ])?;
        } else {
            pass.level_barrier(&[&ms.indices, &ms.scores, io.mlp_act])?;
        }
    }
    let slots = match &moe_w.slot_of {
        Some(table) => {
            resolve_experts(pass, moe_w, &ms.indices, dims.top_k)?;
            moe::moe_remap_slots(ctx, pass, &ms.indices, table, &ms.slots, dims.top_k)?;
            pass.level_barrier(&[&ms.slots])?;
            &ms.slots
        }
        None => &ms.indices,
    };

    if crate::ablate::on(crate::ablate::Group::Gather) {
        return Ok(false);
    }
    // Expert gate/up and the independent shared down projection overlap.
    pass.profile_stage(crate::metal::stage_profile::Phase::MoeExpertGateUpSharedDown)?;
    if dims.official_mlx_rounding {
        moe::official_moe_gate_up(
            ctx,
            pass,
            &moe_w.expert_gate,
            &moe_w.expert_up,
            io.x,
            slots,
            ms.sigmoid_table
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing official sigmoid table"))?,
            &ms.act,
        )?;
    } else {
        moe::moe_gather_gemv_gate_up(
            ctx,
            pass,
            &moe_w.expert_gate,
            &moe_w.expert_up,
            inter,
            io.x,
            slots,
            &ms.act,
        )?;
    }
    if !crate::ablate::on(crate::ablate::Group::Shared) {
        quant::gemv_quant(ctx, pass, &shared.down_proj, io.mlp_act, &ms.shared_out)?;
    }
    pass.level_barrier(&[&ms.act, &ms.shared_out])?;

    if crate::ablate::on(crate::ablate::Group::Down) {
        return Ok(false);
    }
    pass.profile_stage(crate::metal::stage_profile::Phase::MoeExpertDownCombine)?;
    if dims.official_mlx_rounding {
        moe::official_moe_down(
            ctx,
            pass,
            &moe_w.expert_down,
            &ms.act,
            slots,
            &ms.scores,
            Some((&ms.shared_out, &ms.shared_gate)),
            io.out,
        )?;
        return Ok(false);
    }
    if let Some(injection) = io.hc_injection {
        moe::moe_gather_gemv_down_combine_inject(
            ctx,
            pass,
            &moe_w.expert_down,
            dims.hidden,
            &ms.act,
            slots,
            &ms.scores,
            Some((&ms.shared_out, &ms.shared_gate)),
            io.out,
            injection,
        )?;
        return Ok(true);
    }
    moe::moe_gather_gemv_down_combine(
        ctx,
        pass,
        &moe_w.expert_down,
        dims.hidden,
        &ms.act,
        slots,
        &ms.scores,
        Some((&ms.shared_out, &ms.shared_gate)),
        io.out,
    )?;
    Ok(false)
}

/// LM-head-only output policy. F32 is the buffer/accumulation type; official
/// external-MLX values are BF16-rounded before sampling/scoring. Other model
/// projections and the legacy head route remain unchanged.
#[allow(clippy::too_many_arguments)]
pub(crate) fn project_lm_logits(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    a: &Tensor,
    w: &LinearWeights,
    out: &Tensor,
    scratch: &Tensor,
    bf16_values: bool,
) -> Result<()> {
    if !bf16_values {
        return project_mat(ctx, pass, a, w, out, scratch);
    }
    anyhow::ensure!(
        a.shape().len() == 2
            && w.codes.shape().len() == 2
            && matches!(w.bits, 4 | 8 | 16),
        "head matrix shape/storage"
    );
    let (m, k, n) = (a.shape()[0], w.in_features(), w.out_features());
    anyhow::ensure!(
        m > 0
            && k > 0
            && n > 0
            && a.shape()[1] == k
            && a.dtype() == DType::BF16
            && out.dtype() == DType::F32
            && out.numel() == m * n,
        "head matrix activation/output geometry"
    );
    if w.bits != 16 {
        anyhow::ensure!(
            w.group_size > 0
                && w.scales.dtype() == DType::BF16
                && w.biases.dtype() == DType::BF16,
            "head matrix affine metadata must be BF16"
        );
        quant::check_quant(w)?;
    }
    if m == 1 {
        // A one-row scoring/batch readout has exactly the vector-head policy,
        // including the scoped compiler mode and fused store when admitted.
        return quant::gemv_quant_logits(
            ctx,
            pass,
            w,
            &a.view(0, &[k])?,
            &out.view(0, &[n])?,
            true,
        );
    }
    let fast_matrix = skinny::dense_smallm_routes(m)
        && n.is_multiple_of(64)
        && ((w.bits == 4 && w.group_size == 64 && k.is_multiple_of(64))
            || (w.bits == 8 && w.group_size.is_multiple_of(8) && k.is_multiple_of(8)));
    if fast_matrix {
        project_mat(ctx, pass, a, w, out, scratch)?;
    } else {
        // The trunk scratch does not contain a full dequantized vocabulary
        // matrix. Use per-row F32 projection for other valid formats/widths;
        // never allocate/convert a second head or reinterpret BF16 as F32.
        for row in 0..m {
            quant::gemv_quant(
                ctx,
                pass,
                w,
                &a.view(row * k, &[k])?,
                &out.view(row * n, &[n])?,
            )?;
        }
    }
    pass.level_barrier(&[out])?;
    quant::round_logits_bf16_values(ctx, pass, out)
}
