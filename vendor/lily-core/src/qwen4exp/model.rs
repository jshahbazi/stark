// Modified by Stark: in-process core extraction; see UPSTREAM.md.
//! The Qwen3.8-Flash-Next model graph: single-token decode steps plus a
//! batched prefill path that processes `PREFILL_CHUNK` prompt tokens per
//! command buffer.
//!
//! Compared with the Qwen3.5 graph the residual is a `hc_count`-wide stream
//! read through a gated low-rank mixer before every block and written back
//! through per-stream scalar gates (hyper-connections); decoder layer 1 adds
//! the hashed n-gram embedding; and the full-attention layers run Qwen Sparse
//! Attention once the context exceeds the indexer budget.

use std::path::Path;

use anyhow::{Result, ensure};

use std::cell::Cell;
#[cfg(debug_assertions)]
use std::cell::RefCell;
use std::rc::Rc;

use crate::engine::{
    BatchRow, CountsSlot, DecodeStateApi, Draw, LanguageModel, LoadOptions,
    RowsInFlight, RowsStepTiming, ScratchApi, SnapshotApi, VisionMode, VisionTower,
};
use crate::kernels::attention::{
    KvCache, KvFormat, MAX_SEQ, k_norm_rope_scatter_decode_rows,
    k_norm_rope_scatter_decode_with_rounding as k_norm_rope_scatter_decode_with_bias,
    kv_dequant_q8, q_norm_rope_split_decode_rows,
    q_norm_rope_split_decode_with_rounding as q_norm_rope_split_decode_with_bias,
    rope_neox, scatter_kv, sdpa_decode, sdpa_prefill, sdpa_split_scratch_splits,
    split_q_gate,
};
use crate::kernels::elementwise::{
    add_bf16, copy_words, gather_row_bf16,
    sigmoid_mul_bf16_with_rounding as sigmoid_mul_bf16, split_cols_bf16,
};
use crate::kernels::gdn::{
    GDN_HEAD_DIM, GDN_STATE_DTYPE, GdnChunkStaging, GdnGate, GdnRegscanStaging,
    GdnStateLayout, conv1d_prefill_with_rounding as conv1d_prefill,
    conv1d_step_with_layout as conv1d_step,
    gated_rmsnorm_with_rounding as gated_rmsnorm, gdn_chunk_staging_shapes,
    gdn_prefill_mid_with_layout as gdn_prefill_mid,
    gdn_step_gated_fused_with_layout as gdn_step_gated_fused, gdn_verify_gated_exact,
};
use crate::kernels::hc::stock::{
    InjectionInputKind, PendingHcWrite, inject_after_read as hc_inject_bf16,
    write_gate_to as hc_write_gate_to, write_pair_selected,
};
use crate::kernels::hc::{
    HC_FUSED_MAX_ROWS, fused_read_supported, hc_broadcast_bf16,
    hc_mix_bf16_with_rounding as hc_mix_bf16, hc_read_down_q8_decode_rows,
    hc_read_down_q8_rows, hc_read_up_mix_q8_cached, hc_read_up_mix_q8_cached_rows,
    hc_read_up_mix_q8_rows, official_hc_down_act_q8, official_hc_up_mix_q8,
    official_narrow_sigmoid_supported,
    rmsnorm_grouped_bf16_with_rounding as rmsnorm_grouped_bf16,
    silu_scaled_bf16_with_rounding as silu_scaled_bf16,
};
use crate::kernels::norm::rmsnorm_bf16_with_rounding as rmsnorm_bf16;
use crate::kernels::ple;
use crate::kernels::qsa::{self, INDEXER_D, SparseAttnRoute, SparseSplitScratch};
use crate::kernels::sample::{
    DraftDists, SamplerScratch, SamplingParams, sample_f32, sample_spec_f32,
};
use crate::kernels::spec::ctrl_words;
use crate::kernels::{Arg, Pos, Rope};
use crate::kernels::{quant, skinny};
use crate::metal::{
    BlitCopy, ComputePass, EncodedPass, MetalContext, PendingPass, SharedEvent,
};
use crate::moe_ffn::{
    DecodeMoeIo, MoeDims, MoeScratch, PrefillMoeIo, PrefillMoeScratch, decode_moe,
    prefill_moe, prefill_moe_decode_exact_with_injection, prefix_rows, project_mat,
    project_mat_decode_exact, project_stack_or_slices,
};
use crate::tensor::{DType, Tensor};

use super::ExpertCacheStats;
use super::vision::{self, VisionScratch};

use super::config::{GateAct, Qwen4ExpConfig};
use super::ngram::{NgramHasher, NgramStorage, NgramTable, StagedRows};
use super::positions::{ImageSpan, Positions};
use super::weights::{
    self, AttnWeights, GdnWeights, HcWeights, LayerWeights, Mixer, ModelWeights,
    MtpWeights, PleWeights,
};

/// Every RMSNorm in this model is zero-centered: gain = 1 + weight. (The GDN
/// GatedNorm is the exception and uses a plain gain.)

/// Prompt tokens processed by one prefill command buffer.
const PREFILL_CHUNK: usize = 4096;

/// Queries per sparse-attention sub-batch: bounds the `[QB, blocks]` score
/// matrix and the split partials while keeping the GPU busy.
const QSA_QUERY_BATCH: usize = 256;

/// Per-token caches grow in steps of this many tokens (the trunk's K/V take
/// 192 MiB a step in bf16, 102 MiB in q8, plus the indexer keys), so a
/// session's footprint follows its length.
const CAPACITY_STEP: usize = 8192;

/// Most draft tokens a speculative step may verify at once (the spec scratch
/// is sized for it).
pub const MAX_DRAFTS: usize = 3;
/// Prefill passes of more rows than this are gated
/// ([`crate::metal::ComputePass::gate_host_waits`]); the others route as
/// few experts per layer as a verify pass of [`MAX_DRAFTS`] + 1 rows, whose
/// misses take well under the queue wait's time limit even when every one
/// misses (`docs/low-ram-experts.md`, "Timed-out waits").
const GATED_PREFILL_ROWS: usize = MAX_DRAFTS + 1;
// Every verify pass (MAX_DRAFTS + 1 rows) and every draft-head batch takes
// the fused hyper-connection read.
const _: () = assert!(MAX_DRAFTS < HC_FUSED_MAX_ROWS);

/// How the batched (multi-row) graph is encoded: the serial encoder orders
/// every dispatch implicitly; the concurrent one runs a dependency level's
/// dispatches together and relies on the explicit barriers in this file.
/// Both give bit-identical results. Measured on the M5 Max: the concurrent
/// encoder wins for a few rows (53 ms → 22 ms at one row, where 2 229
/// dispatches each waited for the previous one), the serial encoder for big
/// chunks (8 000 tokens: 1 366 against 882 tok/s, where the coarse
/// level barriers stall more than the driver's own hazard tracking).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoder {
    Serial,
    Concurrent,
}

/// Row count up to which the batched graph is encoded concurrently.
const CONCURRENT_ROWS_MAX: usize = 16;

fn encoder_for(rows: usize) -> Encoder {
    if rows <= CONCURRENT_ROWS_MAX { Encoder::Concurrent } else { Encoder::Serial }
}

/// What a batched pass computes after its layers.
#[derive(Clone, Copy)]
pub(super) enum BatchMode<'p> {
    /// Prompt prefill: logits (and a draw) for the last row only, plus the
    /// draft head's catch-up over the chunk when the model has one. `vision`
    /// is set for a prompt with images: per-row rotary positions and the
    /// vision rows that replace the placeholders' embeddings.
    Prefill { draw: Option<Draw<'p>>, vision: Option<&'p PrefillVision<'p>> },
    /// Speculative verification: a draw per row (`step0 + row` indexes the
    /// request's draws), recurrent states after each row recorded for
    /// rollback, no draft-head work (that follows once acceptance is known).
    /// Never parked: committed whole once its inputs are staged (`spec.rs`).
    Verify { params: &'p SamplingParams, step0: usize },
    /// Candidate verifier: complete readouts and recurrent captures only.
    VerifyHost,
}

/// One image of a prompt: its placeholder span and the tower's merged rows.
pub struct ImageEmbeds<'t> {
    pub span: ImageSpan,
    /// bf16 `[span.len, hidden_size]`: the tower's merged output, one row per
    /// placeholder in order (what the reference `masked_scatter`s into the
    /// token embeddings).
    pub rows: &'t Tensor,
}

/// What a prompt with images adds to its tokens for
/// [`Qwen4ExpModel::prefill_with_vision`].
pub struct VisionInput<'t> {
    /// Positions of the whole prompt from sequence index 0, not only of the
    /// tokens being fed: the kernels also rope the head's catch-up row one
    /// before the chunk and the first token of every indexer block the chunk
    /// completes, which can lie before it.
    pub positions: &'t Positions,
    pub images: &'t [ImageEmbeds<'t>],
}

/// A prefill chunk's share of a [`VisionInput`]: the position rows its
/// kernels read (U32 `[rows, 3]`, row `i` for sequence index `base + i`) and
/// the images whose rows may fall into the chunk.
pub(super) struct PrefillVision<'p> {
    positions: Tensor,
    base: usize,
    images: &'p [ImageEmbeds<'p>],
}

impl PrefillVision<'_> {
    fn rope(&self) -> Rope<'_> {
        Rope::Rows { positions: &self.positions, base: self.base }
    }
}

/// Vision rows replacing the embedding rows of one batch: the batch row the
/// run starts at and a `[rows, hidden]` bf16 view of the merged rows.
pub(super) struct RowOverride {
    row: usize,
    rows: Tensor,
}

/// The overrides of a batch whose row `j` holds token `first + j`, `rows`
/// rows long: every image span overlapping that range, clipped to it. Empty
/// for a batch without placeholders.
fn row_overrides(
    images: &[ImageEmbeds<'_>],
    first: usize,
    rows: usize,
    h: usize,
) -> Result<Vec<RowOverride>> {
    let mut out = Vec::new();
    for image in images {
        let a = image.span.start.max(first);
        let b = image.span.end().min(first + rows);
        if a >= b {
            continue;
        }
        ensure!(
            image.rows.shape() == [image.span.len, h]
                && image.rows.dtype() == DType::BF16,
            "image rows are {:?} {:?}, the span at {} needs bf16 [{}, {h}]",
            image.rows.shape(),
            image.rows.dtype(),
            image.span.start,
            image.span.len
        );
        out.push(RowOverride {
            row: a - first,
            rows: image.rows.view((a - image.span.start) * h, &[b - a, h])?,
        });
    }
    Ok(out)
}

/// Writes `overrides` into `x` (`[rows, h]` bf16, the gathered embeddings).
/// The caller orders this after the gather and before the next reader.
fn override_rows(
    ctx: &MetalContext,
    pass: &ComputePass<'_>,
    x: &Tensor,
    overrides: &[RowOverride],
    h: usize,
) -> Result<()> {
    for o in overrides {
        copy_words(ctx, pass, &o.rows, &x.view(o.row * h, o.rows.shape())?)?;
    }
    Ok(())
}

/// Host/GPU handshake for parked passes: a pass committed before its
/// per-token host inputs exist waits for `event >= value`; the host stages
/// the inputs and releases it. Values only grow, one parked pass at a time.
pub(super) struct StepSync {
    pub(super) event: SharedEvent,
    /// Diagnostics (`LILY_PROBE_ARRIVAL`): the GPU raises this to the parked
    /// value right before it waits, so a host poller can time the arrival.
    pub arrival: Option<SharedEvent>,
    /// Diagnostics: raised right after the wait, when the GPU resumed.
    pub resumed: Option<SharedEvent>,
    /// Value the outstanding parked pass waits for; 0 when none.
    armed: Cell<u64>,
    last: Cell<u64>,
    /// Raised by the GPU at the end of every decode, verify and draft pass
    /// (`done_last` is the last value handed out), for polled waits.
    done: SharedEvent,
    done_last: Cell<u64>,
}

impl StepSync {
    fn new(ctx: &MetalContext) -> Result<Self> {
        let probe = false;
        let arrival = if probe { Some(ctx.new_shared_event()?) } else { None };
        let resumed = if probe { Some(ctx.new_shared_event()?) } else { None };
        Ok(Self {
            event: ctx.new_shared_event()?,
            arrival,
            resumed,
            armed: Cell::new(0),
            last: Cell::new(0),
            done: ctx.new_shared_event()?,
            done_last: Cell::new(0),
        })
    }

    /// Encodes the pass's done signal as its last command.
    pub(super) fn signal_done(&self, pass: &ComputePass<'_>) -> Result<()> {
        let value = self.done_last.get() + 1;
        self.done_last.set(value);
        pass.signal_done(&self.done, value)
    }

    /// Encodes the parked wait (and the arrival probe when enabled).
    pub(super) fn encode_wait(&self, pass: &ComputePass<'_>, value: u64) -> Result<()> {
        if let Some(arrival) = &self.arrival {
            pass.signal_event(arrival, value)?;
        }
        pass.wait_event(&self.event, value)?;
        if let Some(resumed) = &self.resumed {
            pass.signal_event(resumed, value)?;
        }
        Ok(())
    }

    /// Claims the value the next parked pass waits for.
    pub(super) fn arm(&self) -> Result<u64> {
        ensure!(self.armed.get() == 0, "a decode step is already parked");
        let value = self.last.get() + 1;
        self.last.set(value);
        self.armed.set(value);
        Ok(value)
    }

    /// The value the outstanding parked pass waits for; 0 when none.
    pub(super) fn armed(&self) -> u64 {
        self.armed.get()
    }

    /// Releases the parked pass, if any.
    pub(super) fn release(&self) -> Result<()> {
        let value = self.armed.take();
        if value != 0 {
            self.event.signal(value);
        }
        Ok(())
    }
}

fn moe_dims(cfg: &Qwen4ExpConfig) -> MoeDims {
    MoeDims {
        num_experts: cfg.num_experts,
        top_k: cfg.num_experts_per_tok,
        moe_intermediate: cfg.moe_intermediate_size,
        hidden: cfg.hidden_size,
        norm_topk_prob: cfg.norm_topk_prob,
        official_mlx_rounding: cfg.official_mlx_rounding(),
    }
}

pub struct Qwen4ExpModel {
    pub config: Qwen4ExpConfig,
    pub(super) weights: ModelWeights,
    pub(super) attn_scale: f32,
    gdn_scale: f32,
    gdn_gate: GdnGate,
    pub(super) hasher: Option<NgramHasher>,
    /// Prompt tokens per batched prefill pass: `PREFILL_CHUNK`, or what
    /// `LILY_PREFILL_CHUNK` says (up to twice that; under an expert cache
    /// the larger chunk streams the cold experts half as often per token,
    /// see docs/low-ram-experts.md for why it is not the default; resident,
    /// 8 192 measured no faster for 2.4 GB more scratch, see
    /// docs/architecture.md, "Prefill").
    prefill_chunk: usize,
    /// Whether a multi-chunk prefill stages chunk k+1's n-gram rows while
    /// the GPU runs chunk k (paged table only; on by default, off for the
    /// bit-identity test of the staging pipeline).
    ngram_ahead: bool,
    mtp_enabled: bool,
    #[cfg(debug_assertions)]
    numerical_trace: RefCell<Option<super::NumericalTrace>>,
    /// The attention K/V caches' element format (`--kv-cache`); states and
    /// scratch made after [`Self::set_kv_format`] follow it.
    kv_format: KvFormat,
}

pub(super) enum LayerState {
    Gdn {
        /// F32 recurrent state. Equal D128 axes are interpreted by `layout`.
        state: Tensor,
        /// Persistent physical order, retained across reset/snapshot/restore.
        layout: GdnStateLayout,
        /// Double-buffered conv window, bf16 `[C, KD-1]` each (see the
        /// Qwen3.5 graph for the buffering contract).
        conv_windows: [Tensor; 2],
    },
    Attn {
        /// `[KVH, max_seq, D]`, bf16 or q8 (`--kv-cache`).
        k_cache: KvCache,
        v_cache: KvCache,
        /// Raw indexer keys, bf16 `[max_seq, INDEXER_D]`.
        idx_keys: Tensor,
        /// Normed, roped block keys, bf16 `[max_seq / ratio, INDEXER_D]`.
        blk_keys: Tensor,
    },
}

/// Per-Layer Embedding recurrent state: the two-token hash history (host
/// side, since the host hashes) and the dilated conv window, double-buffered
/// like the GDN conv windows.
pub(super) struct PleState {
    pub(super) hist: [u32; 2],
    pub(super) conv_windows: [Tensor; 2],
    eos: u32,
}

/// The draft head's per-session state: its attention caches (positions
/// aligned with the trunk's) and the trunk's wide residual for the last fed
/// token, which pairs with the next token to form the head's next input.
pub(super) struct MtpState {
    pub(super) layer: LayerState,
    /// bf16 `[hc_count * h]`.
    pub(super) hidden: Tensor,
}

/// A verify pass whose acceptance is still pending: enough to roll the state
/// back to any accepted prefix of the verified tokens.
pub(super) struct SpecPending {
    pub(super) pos_before: usize,
    pub(super) hist_before: Option<[u32; 2]>,
    /// The verified tokens (pending token first, then the drafts).
    pub(super) tokens: Vec<u32>,
    /// The trunk's draw per row.
    pub(super) sampled: Vec<u32>,
    pub(super) uses_penalties: bool,
    /// Proposals the draft pass behind the verify pass chains.
    pub(super) chain: usize,
}

impl PleState {
    fn reset(&mut self) {
        self.hist = [self.eos, self.eos];
        for w in &self.conv_windows {
            w.zero_fill();
        }
    }
}

pub struct DecodeState {
    pub pos: usize,
    /// What every token fed from now on adds to its sequence index to get
    /// its rotary position on all three axes (VISION.md `rope_deltas`): 0
    /// for a text prompt, negative after an image. Set from the prompt's
    /// [`Positions`] by [`Qwen4ExpModel::prefill_with_vision`] (the engine
    /// sets it when it acquires a session); a pure function of the prompt,
    /// so it is not part of a snapshot or the disk tier and a restore leaves
    /// it alone.
    pub rope_delta: i64,
    /// Tokens the per-token caches hold; grows in `CAPACITY_STEP`s.
    capacity: usize,
    pub(super) layers: Vec<LayerState>,
    /// Which double-buffered window slot is current (all recurrent layers
    /// advance in lockstep): prefill chunks read it, write the other, then
    /// flip; decode steps update it in place.
    pub(super) conv_slot: usize,
    pub(super) ple: Option<PleState>,
    pub(super) mtp: Option<MtpState>,
    pub(super) spec: Option<SpecPending>,
    kv_format: KvFormat,
    kv_heads: usize,
    head_dim: usize,
    ratio: usize,
}

/// The recurrent part of a [`DecodeState`] at one position: GDN states and
/// conv windows, the PLE conv window and hash history, the draft head's
/// last hidden.
pub struct Snapshot {
    pos: usize,
    /// Per GDN layer, in layer order: `(state, conv_window, physical_layout)`.
    gdn: Vec<(Tensor, Tensor, GdnStateLayout)>,
    ple: Option<([u32; 2], Tensor)>,
    mtp_hidden: Option<Tensor>,
}

impl SnapshotApi for Snapshot {
    fn pos(&self) -> usize {
        self.pos
    }

    fn bytes(&self) -> usize {
        self.gdn.iter().map(|(a, b, _)| a.byte_len() + b.byte_len()).sum::<usize>()
            + self.ple.as_ref().map_or(0, |(_, w)| w.byte_len())
            + self.mtp_hidden.as_ref().map_or(0, Tensor::byte_len)
    }
}

/// Rows of an attention layer's block-key store for `capacity` tokens: one
/// per complete indexer block, at least one. Capacities are multiples of the
/// ratio (the config admits only 4), so this also covers the partial block a
/// prefix of up to `capacity` tokens starts ([`attn_prefix_plan`] checks).
fn block_key_rows(capacity: usize, ratio: usize) -> usize {
    (capacity / ratio).max(1)
}

/// Copies of `sources`, in order, made in one GPU pass and one wait. A
/// recurrent snapshot copies 74 tensors with the full model: a pass and a
/// wait per tensor took 17 ms in all, one pass for all of them takes 7.3 ms,
/// most of which is `Tensor::zeros`' allocation and first touch (about
/// 5 ms).
fn clone_tensors(ctx: &MetalContext, sources: &[&Tensor]) -> Result<Vec<Tensor>> {
    let copies = sources
        .iter()
        .map(|t| Tensor::zeros(ctx, t.shape(), t.dtype()))
        .collect::<Result<Vec<_>>>()?;
    let blits: Vec<BlitCopy<'_>> = sources
        .iter()
        .zip(&copies)
        .map(|(src, dst)| BlitCopy {
            src,
            src_offset: 0,
            dst,
            dst_offset: 0,
            len: src.byte_len(),
        })
        .collect();
    ctx.blit_copy(&blits)?;
    Ok(copies)
}

/// Copies the first `rows` rows of every `[heads, cap, d]` head block from
/// `src` to `dst` (one dtype, possibly different capacities).
fn head_block_copies<'t>(
    src: &'t Tensor,
    dst: &'t Tensor,
    rows: usize,
    out: &mut Vec<BlitCopy<'t>>,
) -> Result<()> {
    let (heads, src_cap, d) = (src.shape()[0], src.shape()[1], src.shape()[2]);
    let dst_cap = dst.shape()[1];
    ensure!(
        rows <= src_cap
            && rows <= dst_cap
            && dst.shape()[0] == heads
            && dst.shape()[2] == d,
        "cache copy shape mismatch"
    );
    ensure!(src.dtype() == dst.dtype(), "cache copy between dtypes");
    let row_bytes = d * src.dtype().size();
    for h in 0..heads {
        out.push(BlitCopy {
            src,
            src_offset: h * src_cap * row_bytes,
            dst,
            dst_offset: h * dst_cap * row_bytes,
            len: rows * row_bytes,
        });
    }
    Ok(())
}

/// [`head_block_copies`] of a K or V cache's values and, when q8, scales.
fn kv_copies<'t>(
    src: &'t KvCache,
    dst: &'t KvCache,
    rows: usize,
    out: &mut Vec<BlitCopy<'t>>,
) -> Result<()> {
    ensure!(src.kv().format()? == dst.kv().format()?, "cache copy between K/V formats");
    head_block_copies(&src.values, &dst.values, rows, out)?;
    if let (Some(s), Some(d)) = (&src.scales, &dst.scales) {
        head_block_copies(s, d, rows, out)?;
    }
    Ok(())
}

/// Copies the first `rows` rows of a `[cap, d]` store (both bf16, possibly
/// different capacities). Rows either side lacks are an error, as in
/// [`head_block_copies`], not a shorter copy: a prefix cut short here would
/// leave the destination's tail rows stale without a word.
fn row_copy<'t>(
    src: &'t Tensor,
    dst: &'t Tensor,
    rows: usize,
    out: &mut Vec<BlitCopy<'t>>,
) -> Result<()> {
    ensure!(
        rows <= src.shape()[0]
            && rows <= dst.shape()[0]
            && dst.shape()[1] == src.shape()[1],
        "row copy of {rows} rows between stores of {} and {}",
        src.shape()[0],
        dst.shape()[0]
    );
    let row_bytes = src.shape()[1] * src.dtype().size();
    out.push(BlitCopy {
        src,
        src_offset: 0,
        dst,
        dst_offset: 0,
        len: rows * row_bytes,
    });
    Ok(())
}

impl DecodeState {
    /// Returns the state to position zero. KV and indexer caches need no
    /// clearing: rows past `pos` are never read and get overwritten. The GPU
    /// must be idle on this state's buffers.
    pub fn reset(&mut self) -> Result<()> {
        for lstate in &self.layers {
            if let LayerState::Gdn { state, conv_windows, .. } = lstate {
                state.zero_fill();
                conv_windows[0].zero_fill();
                conv_windows[1].zero_fill();
            }
        }
        if let Some(ple) = &mut self.ple {
            ple.reset();
        }
        if let Some(mtp) = &self.mtp {
            mtp.hidden.zero_fill();
        }
        self.spec = None;
        self.pos = 0;
        self.rope_delta = 0;
        self.conv_slot = 0;
        Ok(())
    }

    fn attn_caches(
        ctx: &MetalContext,
        kv_format: KvFormat,
        kv_heads: usize,
        head_dim: usize,
        ratio: usize,
        capacity: usize,
    ) -> Result<LayerState> {
        Ok(LayerState::Attn {
            k_cache: KvCache::zeros(ctx, kv_format, kv_heads, capacity, head_dim)?,
            v_cache: KvCache::zeros(ctx, kv_format, kv_heads, capacity, head_dim)?,
            idx_keys: Tensor::zeros(ctx, &[capacity, INDEXER_D], DType::BF16)?,
            blk_keys: Tensor::zeros(
                ctx,
                &[block_key_rows(capacity, ratio), INDEXER_D],
                DType::BF16,
            )?,
        })
    }

    /// Per-token cache bytes for `capacity` tokens across all attention
    /// layers, the draft head's included.
    fn cache_bytes(&self, capacity: usize) -> usize {
        let attn_layers =
            self.layers.iter().filter(|l| matches!(l, LayerState::Attn { .. })).count()
                + usize::from(self.mtp.is_some());
        let per_token =
            2 * self.kv_heads * self.kv_format.row_bytes(self.head_dim) + INDEXER_D * 2;
        let blocks = block_key_rows(capacity, self.ratio) * INDEXER_D * 2;
        attn_layers * (capacity * per_token + blocks)
    }

    /// Replaces an attention layer's caches with ones of `capacity` tokens,
    /// copying the first `pos` entries. GPU idle.
    fn grow_attn(
        &self,
        ctx: &MetalContext,
        lstate: &mut LayerState,
        capacity: usize,
    ) -> Result<()> {
        let LayerState::Attn { k_cache, v_cache, idx_keys, blk_keys } = lstate else {
            return Ok(());
        };
        let LayerState::Attn { k_cache: k2, v_cache: v2, idx_keys: i2, blk_keys: b2 } =
            DecodeState::attn_caches(
                ctx,
                self.kv_format,
                self.kv_heads,
                self.head_dim,
                self.ratio,
                capacity,
            )?
        else {
            unreachable!()
        };
        let mut copies = Vec::new();
        kv_copies(k_cache, &k2, self.pos, &mut copies)?;
        kv_copies(v_cache, &v2, self.pos, &mut copies)?;
        row_copy(idx_keys, &i2, self.pos, &mut copies)?;
        row_copy(blk_keys, &b2, self.pos.div_ceil(self.ratio), &mut copies)?;
        ctx.blit_copy(&copies)?;
        drop(copies);
        *k_cache = k2;
        *v_cache = v2;
        *idx_keys = i2;
        *blk_keys = b2;
        Ok(())
    }
}

/// Copies the first `tokens` entries of one attention layer's caches.
fn attn_prefix_copies<'t>(
    dst: &'t LayerState,
    src: &'t LayerState,
    tokens: usize,
    ratio: usize,
    out: &mut Vec<BlitCopy<'t>>,
) -> Result<()> {
    if let (
        LayerState::Attn { k_cache, v_cache, idx_keys, blk_keys },
        LayerState::Attn { k_cache: k0, v_cache: v0, idx_keys: i0, blk_keys: b0 },
    ) = (dst, src)
    {
        kv_copies(k0, k_cache, tokens, out)?;
        kv_copies(v0, v_cache, tokens, out)?;
        row_copy(i0, idx_keys, tokens, out)?;
        row_copy(b0, blk_keys, tokens.div_ceil(ratio), out)?;
    }
    Ok(())
}

/// Sparse-attention scratch shared by decode (one query) and prefill
/// sub-batches (up to `QSA_QUERY_BATCH` queries), viewed per use.
struct QsaScratch {
    /// F32 `[QB, max_blocks]`.
    scores: Tensor,
    /// U32 `[QB, k_max]`.
    sel: Tensor,
    /// U32 `[QB]`.
    n_sel: Tensor,
    /// F32 `[slots * NQ, D]`, indexed `[QB*NQ, splits, D]` per dispatch
    /// (`qsa::split_scratch_slots`).
    partials: Tensor,
    /// F32 `[slots * NQ, 2]`.
    stats: Tensor,
    /// How sub-batches past the dense limit attend (`LILY_QSA_ROUTE`).
    route: SparseAttnRoute,
}

impl QsaScratch {
    fn new(
        ctx: &MetalContext,
        cfg: &Qwen4ExpConfig,
        max_seq: usize,
        qb: usize,
    ) -> Result<Self> {
        let idx = &cfg.indexer;
        let max_blocks = (max_seq / idx.compress_ratio).max(1);
        let k_max = idx.block_topk();
        let slots = qsa::split_scratch_slots(qb, k_max, idx.compress_ratio);
        let nq = cfg.num_attention_heads;
        Ok(Self {
            scores: Tensor::zeros(ctx, &[qb, max_blocks], DType::F32)?,
            sel: Tensor::zeros(ctx, &[qb, k_max], DType::U32)?,
            n_sel: Tensor::zeros(ctx, &[qb], DType::U32)?,
            partials: Tensor::zeros(ctx, &[slots * nq, cfg.head_dim], DType::F32)?,
            stats: Tensor::zeros(ctx, &[slots * nq, 2], DType::F32)?,
            route: SparseAttnRoute::from_env(),
        })
    }

    fn split_scratch(&self) -> SparseSplitScratch<'_> {
        SparseSplitScratch { partials: &self.partials, stats: &self.stats }
    }
}

/// Single-token PLE intermediates.
pub(super) struct PleScratch {
    /// U32 `[rows, heads]`: hashed row ids (resident table) or the
    /// sequential ids of the staged rows (paged table).
    ids: Tensor,
    /// Gathered rows for the paged table, at full capacity.
    stage: Option<Rc<StagedRows>>,
    /// A second staging buffer of the same size (prefill scratch only): a
    /// chunked prefill stages the next chunk's rows here while the GPU reads
    /// the current chunk's from `stage`, alternating per chunk.
    stage_alt: Option<Rc<StagedRows>>,
    emb: Tensor,
    key: Tensor,
    key_n: Tensor,
    query_n: Tensor,
    value: Tensor,
    gated: Tensor,
    gated_n: Tensor,
}

impl PleScratch {
    fn new(
        ctx: &MetalContext,
        cfg: &Qwen4ExpConfig,
        rows: usize,
        table: &NgramTable,
        double_staging: bool,
    ) -> Result<Self> {
        let ple = cfg.ple.as_ref().expect("PLE scratch without PLE config");
        let (h, wide) = (cfg.hidden_size, cfg.hc_width());
        let bf = DType::BF16;
        let heads = ple.ngram_heads();
        let (ids, stage, stage_alt) = match table {
            NgramTable::Resident(_) => {
                (Tensor::zeros(ctx, &[rows, heads], DType::U32)?, None, None)
            }
            NgramTable::Paged(paged) => {
                // Staged rows are gathered with their own sequential ids.
                let seq: Vec<u32> = (0..(rows * heads) as u32).collect();
                let ids = Tensor::from_bytes(
                    ctx,
                    bytemuck::cast_slice(&seq),
                    &[rows, heads],
                    DType::U32,
                )?;
                let stage = |ctx| -> Result<Rc<StagedRows>> {
                    Ok(Rc::new(StagedRows::new(ctx, paged, rows * heads)?))
                };
                let alt = if double_staging { Some(stage(ctx)?) } else { None };
                (ids, Some(stage(ctx)?), alt)
            }
        };
        Ok(Self {
            ids,
            stage,
            stage_alt,
            emb: Tensor::zeros(ctx, &[rows, ple.embed_dim], bf)?,
            key: Tensor::zeros(ctx, &[rows, wide], bf)?,
            key_n: Tensor::zeros(ctx, &[rows, wide], bf)?,
            query_n: Tensor::zeros(ctx, &[rows, wide], bf)?,
            value: Tensor::zeros(ctx, &[rows, h], bf)?,
            gated: Tensor::zeros(ctx, &[rows, wide], bf)?,
            gated_n: Tensor::zeros(ctx, &[rows, wide], bf)?,
        })
    }

    fn rows(&self, m: usize) -> Result<Self> {
        self.rows_on(m, 0)
    }

    /// [`Self::rows`] reading its staged rows from staging buffer `buffer`
    /// (0 is `stage`, 1 is `stage_alt`, which must exist for a paged table).
    /// The view's `stage` is that buffer and it has no second one.
    pub(super) fn rows_on(&self, m: usize, buffer: usize) -> Result<Self> {
        let stage = match buffer {
            0 => self.stage.clone(),
            1 => {
                ensure!(
                    self.stage_alt.is_some() || self.stage.is_none(),
                    "no second n-gram staging buffer"
                );
                self.stage_alt.clone()
            }
            _ => anyhow::bail!("n-gram staging buffer {buffer} of 2"),
        };
        Ok(Self {
            ids: prefix_rows(&self.ids, m)?,
            stage,
            stage_alt: None,
            emb: prefix_rows(&self.emb, m)?,
            key: prefix_rows(&self.key, m)?,
            key_n: prefix_rows(&self.key_n, m)?,
            query_n: prefix_rows(&self.query_n, m)?,
            value: prefix_rows(&self.value, m)?,
            gated: prefix_rows(&self.gated, m)?,
            gated_n: prefix_rows(&self.gated_n, m)?,
        })
    }
}

/// Hyper-connection read intermediates for `rows` tokens. The fused decode
/// and small-batch reads touch `down`, `act`, `inj`, `inv_rms` and `mixed`;
/// the unfused batched read of larger chunks writes every buffer.
pub(super) struct HcScratch {
    /// `[rows, G*H]` normed streams.
    pub(super) hn: Tensor,
    /// `[rows, lowrank]` read-gate logits and activation.
    down: Tensor,
    act: Tensor,
    /// `[rows, G*H]` per-element gate logits.
    pub(super) up: Tensor,
    /// `[rows, H]` the block input.
    pub(super) mixed: Tensor,
    /// `[rows, G]` raw logits, or finished gates after a selected stock read.
    /// The subsequent write must use the matching hc_injection_kind.
    inj: Tensor,
    /// `[rows, G]` F32 stream RMS reciprocals handed from the fused down
    /// kernel to the fused up kernel.
    inv_rms: Tensor,
    /// Stock partials for up to four verifier rows; bounded at prefill capacity.
    stock_parts: Option<Tensor>,
    /// One distinct WR1 residual stream; pairs finish back in canonical hyper.
    stock_write_hyper: Option<Tensor>,
}

impl HcScratch {
    fn new(ctx: &MetalContext, cfg: &Qwen4ExpConfig, rows: usize) -> Result<Self> {
        let (h, wide, g) = (cfg.hidden_size, cfg.hc_width(), cfg.hc_count);
        let bf = DType::BF16;
        Ok(Self {
            hn: Tensor::zeros(ctx, &[rows, wide], bf)?,
            down: Tensor::zeros(ctx, &[rows, cfg.hc_lowrank], bf)?,
            act: Tensor::zeros(ctx, &[rows, cfg.hc_lowrank], bf)?,
            up: Tensor::zeros(ctx, &[rows, wide], bf)?,
            mixed: Tensor::zeros(ctx, &[rows, h], bf)?,
            inj: Tensor::zeros(ctx, &[rows, g], bf)?,
            inv_rms: Tensor::zeros(ctx, &[rows, g], DType::F32)?,
            stock_write_hyper: if rows == 1
                && cfg.official_mlx_rounding()
                && h == 2560
                && g == 4
                && cfg.hc_lowrank == 320
            {
                Some(Tensor::zeros(ctx, &[1, wide], bf)?)
            } else {
                None
            },
            stock_parts: if cfg.official_mlx_rounding()
                && h == 2560
                && g == 4
                && cfg.hc_lowrank == 320
            {
                Some(Tensor::zeros(
                    ctx,
                    &[rows.min(4), crate::kernels::hc::stock::PARTS],
                    DType::F32,
                )?)
            } else {
                None
            },
        })
    }

    fn rows(&self, m: usize) -> Result<Self> {
        Ok(Self {
            hn: prefix_rows(&self.hn, m)?,
            down: prefix_rows(&self.down, m)?,
            act: prefix_rows(&self.act, m)?,
            up: prefix_rows(&self.up, m)?,
            mixed: prefix_rows(&self.mixed, m)?,
            inj: prefix_rows(&self.inj, m)?,
            inv_rms: prefix_rows(&self.inv_rms, m)?,
            stock_parts: self
                .stock_parts
                .as_ref()
                .map(|t| prefix_rows(t, m.min(4)))
                .transpose()?,
            stock_write_hyper: self.stock_write_hyper.clone(),
        })
    }
}

/// Per-step intermediates, allocated once. Projection outputs that share one
/// fused matvec (`gdn_in`, `attn_qkv`, `mlp_gu`) are single buffers whose
/// named segments are views.
pub struct Scratch {
    x: Tensor,
    /// `[G*H]` the hyper-connection residual stream.
    hyper: Tensor,
    hc: HcScratch,
    branch_out: Tensor,
    gdn_in: Tensor,
    attn_qkv: Tensor,
    mlp_gu: Tensor,
    mlp_gate: Tensor,
    mlp_up: Tensor,
    mlp_act: Tensor,
    qkv: Tensor,
    qkv_conv: Tensor,
    z: Tensor,
    a: Tensor,
    b: Tensor,
    gdn_gated: Tensor,
    gdn_raw: Tensor,
    qg: Tensor,
    q: Tensor,
    gate: Tensor,
    k_new: Tensor,
    v_new: Tensor,
    attn_o: Tensor,
    attn_gated: Tensor,
    /// `[(NH+1)*INDEXER_D]` indexer projection and its prepared queries.
    idx_qk: Tensor,
    idx_q: Tensor,
    qsa: QsaScratch,
    ple: Option<PleScratch>,
    logits: Tensor,
    sdpa_partials: Tensor,
    sdpa_stats: Tensor,
    pub(super) sampler: SamplerScratch,
    /// Tokens written by the in-graph sampler (`U32[2]`): two ping-pong
    /// slots so the pipelined loop can host-read step N's token while the
    /// in-flight step N+1 writes the other slot.
    pub next_token: Tensor,
    /// Prefill staging for the largest quantized projection.
    pub(super) dequant: Tensor,
    moe: MoeScratch,
    /// Session-lived batched prefill intermediates, grown on demand and capped
    /// at `PREFILL_CHUNK` rows.
    pub(super) prefill: Option<PrefillScratch>,
    /// Speculative-decoding intermediates (models with a draft head).
    pub(super) spec: Option<SpecScratch>,
    /// Handshake for passes committed ahead of their host inputs.
    pub(super) sync: StepSync,
    /// The vision tower's intermediates, allocated by the first image and
    /// grown to the largest patch count seen; `None` until then.
    pub(super) vision: Option<VisionScratch>,
    /// Batched decode across sessions (`batch`): per-slot samplers and
    /// per-row attention scratch, allocated by the first batched step.
    pub(super) batch: Option<batch::BatchScratch>,
}

impl Scratch {
    /// The arrival probe event (`LILY_PROBE_ARRIVAL`), for benchmarks.
    pub fn arrival_probe(&self) -> Option<&SharedEvent> {
        self.sync.arrival.as_ref()
    }

    /// The resumed probe event (`LILY_PROBE_ARRIVAL`), for benchmarks.
    pub fn resumed_probe(&self) -> Option<&SharedEvent> {
        self.sync.resumed.as_ref()
    }
}

/// What a verify pass records so a draft pass can roll the recurrent state
/// back to the accepted prefix, plus the draft head's small buffers.
pub(super) struct SpecScratch {
    /// Per GDN layer: F32 `[MAX_DRAFTS, H, 128, 128]`, the state after each
    /// verified row but the last.
    pub(super) mid: Vec<Tensor>,
    /// Per GDN layer: bf16 `[MAX_DRAFTS + 1, C]`, the verified rows' conv inputs.
    pub(super) conv_in: Vec<Tensor>,
    /// bf16 `[MAX_DRAFTS + 1, G*H]`: the PLE conv inputs.
    pub(super) ple_conv_in: Option<Tensor>,
    /// F32 `[MAX_DRAFTS + 1, vocab]`.
    pub(super) logits: Tensor,
    /// U32 `[MAX_DRAFTS + 1]`: the trunk's draw per verified row.
    pub(super) verify_tokens: Tensor,
    /// U32 `[MAX_DRAFTS]`: the draft head's proposals.
    pub(super) draft_tokens: Tensor,
    /// The distributions the proposals were drawn from under a sampling
    /// request (one slot per draft), for the verify pass's speculative draws.
    pub(super) dists: DraftDists,
    /// U32 `[MAX_DRAFTS + 1]`: host-written tokens for the head's catch-up rows.
    pub(super) mtp_ids: Tensor,
    /// U32 control block the GPU fills with the accepted count and what
    /// follows from it (`kernels::spec`), read by the draft pass's dispatches.
    pub(super) ctrl: Tensor,
    /// bf16 `[1, G*H]`: the head residual row the chain continues from.
    pub(super) chain_in: Tensor,
}

impl SpecScratch {
    fn new(ctx: &MetalContext, cfg: &Qwen4ExpConfig) -> Result<Self> {
        let heads = cfg.linear_num_value_heads;
        let c = cfg.gdn_conv_channels();
        let gdn_layers = cfg
            .layer_types
            .iter()
            .filter(|t| matches!(t, super::config::LayerType::LinearAttention))
            .count();
        let rows = MAX_DRAFTS + 1;
        Ok(Self {
            mid: (0..gdn_layers)
                .map(|_| {
                    Tensor::zeros(
                        ctx,
                        &[MAX_DRAFTS, heads, GDN_HEAD_DIM, GDN_HEAD_DIM],
                        GDN_STATE_DTYPE,
                    )
                })
                .collect::<Result<_>>()?,
            conv_in: (0..gdn_layers)
                .map(|_| Tensor::zeros(ctx, &[rows, c], DType::BF16))
                .collect::<Result<_>>()?,
            ple_conv_in: cfg
                .ple
                .as_ref()
                .map(|_| Tensor::zeros(ctx, &[rows, cfg.hc_width()], DType::BF16))
                .transpose()?,
            logits: Tensor::zeros(ctx, &[rows, cfg.vocab_size], DType::F32)?,
            verify_tokens: Tensor::zeros(ctx, &[rows], DType::U32)?,
            draft_tokens: Tensor::zeros(ctx, &[MAX_DRAFTS], DType::U32)?,
            dists: DraftDists::new(ctx, MAX_DRAFTS)?,
            mtp_ids: Tensor::zeros(ctx, &[rows], DType::U32)?,
            ctrl: Tensor::zeros(ctx, &[ctrl_words(MAX_DRAFTS)], DType::U32)?,
            chain_in: Tensor::zeros(ctx, &[1, cfg.hc_width()], DType::BF16)?,
        })
    }
}

/// Where a batched attention pass runs: the position of its first row and,
/// when the GPU supplies that position, the indexer block the pass completes
/// (block index and 0/1 count), written earlier in the same pass. A
/// GPU-supplied position is limited to single-row passes whose position range
/// spans less than one indexer block, so at most one block completes.
#[derive(Clone, Copy)]
pub(super) struct AttnPos<'t> {
    pub(super) pos: Pos<'t>,
    block: Option<(&'t Tensor, &'t Tensor)>,
}

impl<'t> AttnPos<'t> {
    pub(super) fn host(pos: usize) -> Self {
        Self { pos: Pos::host(pos), block: None }
    }

    /// `pos` (a U32 word in `min..=max`), `block` and `count`: the words a
    /// `kernels::spec::spec_accept` dispatch wrote for this row.
    pub(super) fn gpu(
        pos: &'t Tensor,
        min: usize,
        max: usize,
        block: &'t Tensor,
        count: &'t Tensor,
    ) -> Self {
        Self { pos: Pos::gpu(pos, min, max), block: Some((block, count)) }
    }
}

/// Batched prefill intermediates at a capacity that only grows; each chunk
/// uses exact row-prefix views.
pub(super) struct PrefillScratch {
    pub(super) m: usize,
    pub(super) ids: Tensor,
    x: Tensor,
    pub(super) hyper: Tensor,
    pub(super) hc: HcScratch,
    branch_out: Tensor,
    mlp_gate: Tensor,
    mlp_up: Tensor,
    mlp_act: Tensor,
    qkv: Tensor,
    qkv_conv: Tensor,
    z: Tensor,
    a: Tensor,
    b: Tensor,
    gdn_out: Tensor,
    gdn_gated: Tensor,
    gdn_stage: GdnStageScratch,
    qg: Tensor,
    q: Tensor,
    gate: Tensor,
    k_new: Tensor,
    v_new: Tensor,
    attn_o: Tensor,
    attn_gated: Tensor,
    idx_qk: Tensor,
    idx_q: Tensor,
    stack: Tensor,
    moe: PrefillMoeScratch,
    qsa: QsaScratch,
    pub(super) ple: Option<PleScratch>,
    /// Draft head: its residual stream and the trunk hiddens feeding it,
    /// bf16 `[m, G*H]` each (models with a draft head only).
    pub(super) mtp_hyper: Option<Tensor>,
    mtp_hidden_in: Option<Tensor>,
    /// U32 `[m + ratio, 3]`: the 3-axis rotary positions a chunk of a prompt
    /// with an image reads (its rows, the head's row before it and the first
    /// token of every block it completes, up to `ratio` rows earlier).
    positions: Tensor,
    /// q8 caches only: K and V of the dense window as bf16
    /// (`[KVH, dense_limit, D]` each), which the dense tensor-op prefill
    /// kernel reads; rows past the indexer's dense limit never go dense.
    kv_staging: Option<[Tensor; 2]>,
}

struct GdnStageScratch {
    qk_norm: Tensor,
    decay: Tensor,
    beta: Tensor,
    /// The chunked scan's per-chunk WY staging (`gdn_chunk_staging_shapes`).
    w: Tensor,
    u: Tensor,
    p: Tensor,
    g: Tensor,
}

impl GdnStageScratch {
    fn new(ctx: &MetalContext, m: usize, hk: usize, heads: usize) -> Result<Self> {
        let [w, u, p, g] = gdn_chunk_staging_shapes(m, heads);
        Ok(Self {
            qk_norm: Tensor::zeros(ctx, &[m, 2 * hk * GDN_HEAD_DIM], DType::BF16)?,
            decay: Tensor::zeros(ctx, &[m, heads], DType::F32)?,
            beta: Tensor::zeros(ctx, &[m, heads], DType::F32)?,
            w: Tensor::zeros(ctx, &w.0, w.1)?,
            u: Tensor::zeros(ctx, &u.0, u.1)?,
            p: Tensor::zeros(ctx, &p.0, p.1)?,
            g: Tensor::zeros(ctx, &g.0, g.1)?,
        })
    }

    fn prefix(&self, m: usize) -> Result<Self> {
        Ok(Self {
            qk_norm: prefix_rows(&self.qk_norm, m)?,
            decay: prefix_rows(&self.decay, m)?,
            beta: prefix_rows(&self.beta, m)?,
            w: prefix_rows(&self.w, m)?,
            u: prefix_rows(&self.u, m)?,
            p: prefix_rows(&self.p, m)?,
            g: prefix_rows(&self.g, m)?,
        })
    }

    fn staging(&self) -> GdnRegscanStaging<'_> {
        GdnRegscanStaging {
            qk_norm: &self.qk_norm,
            decay: &self.decay,
            beta: &self.beta,
            chunk: Some(GdnChunkStaging {
                w: &self.w,
                u: &self.u,
                p: &self.p,
                g: &self.g,
            }),
        }
    }
}

impl PrefillScratch {
    fn new(
        ctx: &MetalContext,
        cfg: &Qwen4ExpConfig,
        capacity: usize,
        max_seq: usize,
        table: Option<&NgramTable>,
        with_mtp: bool,
        kv_format: KvFormat,
    ) -> Result<Self> {
        ensure!(capacity > 0, "prefill scratch capacity must be nonzero");
        let m = capacity;
        let h = cfg.hidden_size;
        let inter = cfg.shared_expert_intermediate_size;
        let dim_v = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_c = cfg.gdn_conv_channels();
        let heads = cfg.linear_num_value_heads;
        let hk = cfg.linear_num_key_heads;
        let (nq, nkv, hd) =
            (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let nh = cfg.indexer.n_heads;
        let bf = DType::BF16;
        Ok(Self {
            m,
            ids: Tensor::zeros(ctx, &[m], DType::U32)?,
            x: Tensor::zeros(ctx, &[m, h], bf)?,
            hyper: Tensor::zeros(ctx, &[m, cfg.hc_width()], bf)?,
            hc: HcScratch::new(ctx, cfg, m)?,
            branch_out: Tensor::zeros(ctx, &[m, h], bf)?,
            mlp_gate: Tensor::zeros(ctx, &[m, inter], bf)?,
            mlp_up: Tensor::zeros(ctx, &[m, inter], bf)?,
            mlp_act: Tensor::zeros(ctx, &[m, inter], bf)?,
            qkv: Tensor::zeros(ctx, &[m, conv_c], bf)?,
            qkv_conv: Tensor::zeros(ctx, &[m, conv_c], bf)?,
            z: Tensor::zeros(ctx, &[m, dim_v], bf)?,
            a: Tensor::zeros(ctx, &[m, heads], bf)?,
            b: Tensor::zeros(ctx, &[m, heads], bf)?,
            gdn_out: Tensor::zeros(ctx, &[m, heads, GDN_HEAD_DIM], bf)?,
            gdn_gated: Tensor::zeros(ctx, &[m, dim_v], bf)?,
            gdn_stage: GdnStageScratch::new(ctx, m, hk, heads)?,
            qg: Tensor::zeros(ctx, &[m, nq * 2 * hd], bf)?,
            q: Tensor::zeros(ctx, &[m, nq, hd], bf)?,
            gate: Tensor::zeros(ctx, &[m, nq, hd], bf)?,
            k_new: Tensor::zeros(ctx, &[m, nkv, hd], bf)?,
            v_new: Tensor::zeros(ctx, &[m, nkv, hd], bf)?,
            attn_o: Tensor::zeros(ctx, &[m, nq, hd], bf)?,
            attn_gated: Tensor::zeros(ctx, &[m, nq * hd], bf)?,
            idx_qk: Tensor::zeros(ctx, &[m, (nh + 1) * INDEXER_D], bf)?,
            idx_q: Tensor::zeros(ctx, &[m, nh, INDEXER_D], bf)?,
            stack: {
                // Rows cap at the largest chunk eligible for fused projection.
                let rows = m.min(skinny::DENSE_SMALLM_THRESHOLD);
                let width = (conv_c + dim_v + 2 * heads)
                    .max((nq * 2 + 2 * nkv) * hd)
                    .max(2 * inter);
                Tensor::zeros(ctx, &[rows, width], bf)?
            },
            moe: PrefillMoeScratch::new(ctx, &moe_dims(cfg), m)?,
            qsa: QsaScratch::new(ctx, cfg, max_seq, QSA_QUERY_BATCH.min(m))?,
            ple: table.map(|t| PleScratch::new(ctx, cfg, m, t, true)).transpose()?,
            mtp_hyper: with_mtp
                .then(|| Tensor::zeros(ctx, &[m, cfg.hc_width()], bf))
                .transpose()?,
            mtp_hidden_in: with_mtp
                .then(|| Tensor::zeros(ctx, &[m, cfg.hc_width()], bf))
                .transpose()?,
            positions: Tensor::zeros(
                ctx,
                &[m + cfg.indexer.compress_ratio, 3],
                DType::U32,
            )?,
            kv_staging: match kv_format {
                KvFormat::Bf16 => None,
                KvFormat::Q8 | KvFormat::Q8Affine64 => {
                    let shape = [nkv, cfg.indexer.dense_limit(), hd];
                    Some([
                        Tensor::zeros(ctx, &shape, bf)?,
                        Tensor::zeros(ctx, &shape, bf)?,
                    ])
                }
            },
        })
    }

    /// Views of the capacity scratch for one chunk of `tokens`, uploading the
    /// token ids (the GPU is idle here: chunks are commit_wait-synchronized).
    pub(super) fn chunk(&self, tokens: &[u32]) -> Result<Self> {
        let chunk = self.rows(tokens.len())?;
        chunk.ids.write_bytes(bytemuck::cast_slice(tokens))?;
        Ok(chunk)
    }

    /// Row-prefix views for `m` rows without touching the ids.
    pub(super) fn rows(&self, m: usize) -> Result<Self> {
        ensure!(m <= self.m, "chunk of {m} tokens exceeds scratch capacity {}", self.m);
        let ids = prefix_rows(&self.ids, m)?;
        Ok(Self {
            m,
            ids,
            x: prefix_rows(&self.x, m)?,
            hyper: prefix_rows(&self.hyper, m)?,
            hc: self.hc.rows(m)?,
            branch_out: prefix_rows(&self.branch_out, m)?,
            mlp_gate: prefix_rows(&self.mlp_gate, m)?,
            mlp_up: prefix_rows(&self.mlp_up, m)?,
            mlp_act: prefix_rows(&self.mlp_act, m)?,
            qkv: prefix_rows(&self.qkv, m)?,
            qkv_conv: prefix_rows(&self.qkv_conv, m)?,
            z: prefix_rows(&self.z, m)?,
            a: prefix_rows(&self.a, m)?,
            b: prefix_rows(&self.b, m)?,
            gdn_out: prefix_rows(&self.gdn_out, m)?,
            gdn_gated: prefix_rows(&self.gdn_gated, m)?,
            gdn_stage: self.gdn_stage.prefix(m)?,
            qg: prefix_rows(&self.qg, m)?,
            q: prefix_rows(&self.q, m)?,
            gate: prefix_rows(&self.gate, m)?,
            k_new: prefix_rows(&self.k_new, m)?,
            v_new: prefix_rows(&self.v_new, m)?,
            attn_o: prefix_rows(&self.attn_o, m)?,
            attn_gated: prefix_rows(&self.attn_gated, m)?,
            idx_qk: prefix_rows(&self.idx_qk, m)?,
            idx_q: prefix_rows(&self.idx_q, m)?,
            // Kept at full extent: `project_stack_or_slices` views the exact
            // [m, n_total] prefix per level and never fuses past the threshold.
            stack: self.stack.view(0, self.stack.shape())?,
            moe: self.moe.chunk(m)?,
            // Sub-batch views are taken per use inside the attention branch.
            qsa: QsaScratch {
                scores: self.qsa.scores.view(0, self.qsa.scores.shape())?,
                sel: self.qsa.sel.view(0, self.qsa.sel.shape())?,
                n_sel: self.qsa.n_sel.view(0, self.qsa.n_sel.shape())?,
                partials: self.qsa.partials.view(0, self.qsa.partials.shape())?,
                stats: self.qsa.stats.view(0, self.qsa.stats.shape())?,
                route: self.qsa.route,
            },
            ple: self.ple.as_ref().map(|p| p.rows(m)).transpose()?,
            mtp_hyper: self
                .mtp_hyper
                .as_ref()
                .map(|t| prefix_rows(t, m))
                .transpose()?,
            mtp_hidden_in: self
                .mtp_hidden_in
                .as_ref()
                .map(|t| prefix_rows(t, m))
                .transpose()?,
            // Full extent: the chunk's slice is viewed at upload.
            positions: self.positions.view(0, self.positions.shape())?,
            kv_staging: self
                .kv_staging
                .as_ref()
                .map(|[k, v]| {
                    Ok::<_, anyhow::Error>([
                        k.view(0, k.shape())?,
                        v.view(0, v.shape())?,
                    ])
                })
                .transpose()?,
        })
    }
}

/// What a verify pass records per layer for rollback: the GDN states after
/// each row but the last, and the rows' conv inputs (GDN and PLE) so the conv
/// windows can be rewound.
pub(super) struct Capture {
    exact: bool,
    pub(super) mid: Option<Tensor>,
    pub(super) conv_in: Tensor,
    pub(super) ple_conv_in: Option<Tensor>,
}

/// Ordinary decode values copied into request-local speculative captures.
struct DecodeCapture<'a> {
    row: usize,
    rows: usize,
    spec: &'a SpecScratch,
    hyper: &'a Tensor,
}

impl Qwen4ExpModel {
    pub fn load(ctx: &MetalContext, dir: impl AsRef<Path>) -> Result<Self> {
        Self::load_with(
            ctx,
            dir,
            NgramStorage::default(),
            false,
            VisionMode::Off,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    }

    /// Loads with the n-gram table `storage` of choice, the draft head when
    /// `with_mtp` and the checkpoint has one, and the vision tower when
    /// `vision` allows it and the checkpoint has one.
    #[allow(clippy::too_many_arguments)]
    pub fn load_with(
        ctx: &MetalContext,
        dir: impl AsRef<Path>,
        storage: NgramStorage,
        with_mtp: bool,
        vision: VisionMode,
        expert_slots: Option<usize>,
        expert_usage: Option<std::path::PathBuf>,
        machine_memory: Option<u64>,
        memory_limit: Option<u64>,
        expert_usage_out: Option<std::path::PathBuf>,
        session_context: Option<weights::SessionContext>,
    ) -> Result<Self> {
        let config = Qwen4ExpConfig::from_model_dir(&dir)?;
        ensure!(
            config.linear_key_head_dim == GDN_HEAD_DIM
                && config.linear_value_head_dim == GDN_HEAD_DIM,
            "GDN head dim {} unsupported (kernel is compiled for {GDN_HEAD_DIM})",
            config.linear_key_head_dim,
        );
        ensure!(
            config.indexer.head_dim == INDEXER_D,
            "indexer head dim {} unsupported (kernels are compiled for {INDEXER_D})",
            config.indexer.head_dim
        );
        let weights = weights::load(
            ctx,
            &dir,
            &config,
            storage,
            with_mtp,
            vision == VisionMode::Auto,
            expert_slots,
            expert_usage,
            machine_memory,
            memory_limit,
            expert_usage_out,
            session_context,
        )?;
        Self::from_weights(config, weights)
    }

    /// Executes a self-contained Uzu converted text artifact through the same
    /// validated BF16 architecture kernels. No native folder is consulted.
    pub fn load_uzu(
        ctx: &MetalContext,
        config: Qwen4ExpConfig,
        checkpoint: crate::safetensors::Checkpoint,
    ) -> Result<Self> {
        ensure!(
            config.linear_key_head_dim == GDN_HEAD_DIM
                && config.linear_value_head_dim == GDN_HEAD_DIM,
            "unsupported Uzu GDN head dimension"
        );
        ensure!(
            config.indexer.head_dim == INDEXER_D,
            "unsupported Uzu indexer head dimension"
        );
        ensure!(
            config.mtp.is_none() && config.vision.is_none(),
            "Uzu Qwen execution is text-only and non-speculative"
        );
        let weights = weights::load_checkpoint(
            ctx,
            checkpoint,
            &config,
            NgramStorage::Paged,
            false,
            false,
            None,
        )?;
        Self::from_weights(config, weights)
    }

    fn from_weights(config: Qwen4ExpConfig, weights: ModelWeights) -> Result<Self> {
        let hasher = match &config.ple {
            Some(p) => Some(NgramHasher::new(
                &p.layer_multipliers,
                &p.head_vocab_sizes,
                &p.head_offsets,
                p.heads_per_ngram,
                p.eos_token_id,
            )?),
            None => None,
        };
        let gdn_gate = match config.gdn_gate {
            GateAct::Silu => GdnGate::Silu,
            GateAct::Sigmoid => GdnGate::Sigmoid,
        };
        let attn_scale = 1.0 / (config.head_dim as f32).sqrt();
        let gdn_scale = 1.0 / (config.linear_key_head_dim as f32).sqrt();
        let prefill_chunk = PREFILL_CHUNK;
        let mtp_enabled = weights.mtp.is_some();
        Ok(Self {
            config,
            weights,
            attn_scale,
            gdn_scale,
            gdn_gate,
            hasher,
            prefill_chunk,
            ngram_ahead: true,
            mtp_enabled,
            #[cfg(debug_assertions)]
            numerical_trace: RefCell::new(None),
            kv_format: kv_format_from_env()?.unwrap_or_default(),
        })
    }

    /// Runs the vision tower over one preprocessed image and returns its
    /// merged rows as an owned bf16 `[grid_h * grid_w / 4, hidden_size]`
    /// tensor (the tower's output lives in the scratch only until the next
    /// image, so the engine's copy is what a prompt with several images
    /// hands to [`Self::prefill_with_vision`]). Waits for the GPU.
    pub fn encode_image(
        &self,
        ctx: &MetalContext,
        s: &mut Scratch,
        pixels: &[f32],
        grid_h: usize,
        grid_w: usize,
    ) -> Result<Tensor> {
        let (Some(config), Some(weights)) = (&self.config.vision, &self.weights.vision)
        else {
            anyhow::bail!("the vision tower is not loaded");
        };
        let out = vision::forward_with(
            ctx,
            config,
            weights,
            &mut s.vision,
            pixels,
            grid_h,
            grid_w,
            config.depth,
        )?;
        let shape = [grid_h * grid_w / 4, self.config.hidden_size];
        ensure!(
            out.merged.shape() == shape,
            "the tower produced {:?} rows for a ({grid_h}, {grid_w}) grid, expected {shape:?}",
            out.merged.shape()
        );
        Tensor::from_bytes(ctx, out.merged.raw_bytes(), &shape, DType::BF16)
    }

    /// Whether the draft head is loaded.
    pub fn has_mtp(&self) -> bool {
        self.weights.mtp.is_some()
    }

    /// Explicit diagnostic scope. GPU tensor copies stay in memory until
    /// the returned trace is compared and dropped; no production flag uses it.
    #[cfg(debug_assertions)]
    pub fn begin_numerical_trace(&self, layers: usize) -> Result<()> {
        let mut trace = self.numerical_trace.borrow_mut();
        ensure!(
            trace.is_none()
                && layers > 0
                && layers <= self.config.num_hidden_layers + 1,
            "invalid numerical trace scope"
        );
        *trace = Some(super::NumericalTrace::new(layers));
        Ok(())
    }

    #[cfg(debug_assertions)]
    pub fn finish_numerical_trace(&self) -> Result<super::NumericalTrace> {
        self.numerical_trace
            .borrow_mut()
            .take()
            .ok_or_else(|| anyhow::anyhow!("no numerical trace scope"))
    }

    #[inline(always)]
    fn trace_stage(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        layer: usize,
        stage: &'static str,
        tensor: &Tensor,
        rows: usize,
    ) -> Result<()> {
        #[cfg(debug_assertions)]
        if let Some(trace) = self.numerical_trace.borrow().as_ref() {
            trace.capture(ctx, pass, layer, stage, tensor, rows)?;
        }
        #[cfg(not(debug_assertions))]
        let _ = (ctx, pass, layer, stage, tensor, rows);
        Ok(())
    }

    /// Whether the vision tower is loaded.
    pub fn has_vision(&self) -> bool {
        self.weights.vision.is_some()
    }

    fn begin_batched<'a>(
        &self,
        ctx: &'a MetalContext,
        rows: usize,
    ) -> Result<ComputePass<'a>> {
        match encoder_for(rows) {
            Encoder::Serial => ctx.begin(),
            Encoder::Concurrent => ctx.begin_concurrent(),
        }
    }

    /// The n-gram table, when the checkpoint has a PLE layer.
    fn ple_table(&self) -> Option<&NgramTable> {
        self.weights.layers.iter().find_map(|l| l.ple.as_ref()).map(|p| &p.table)
    }

    /// Stages the n-gram rows for `tokens` following `hist` into `p`: hashed
    /// ids for a resident table, the rows themselves for a paged one. The GPU
    /// must not be reading `p`.
    pub(super) fn stage_ngram(
        &self,
        w: &PleWeights,
        p: &PleScratch,
        tokens: &[u32],
        hist: [u32; 2],
    ) -> Result<()> {
        let hasher = self.hasher.as_ref().expect("PLE weights without hasher");
        let mut ids = Vec::with_capacity(tokens.len() * hasher.heads());
        hasher.ids(tokens, hist, &mut ids);
        match &w.table {
            NgramTable::Resident(_) => p.ids.write_bytes(bytemuck::cast_slice(&ids)),
            NgramTable::Paged(table) => {
                let stage = p.stage.as_ref().expect("paged table without staging");
                stage.fill(table, &ids)
            }
        }
    }

    /// Encodes the table gather for `rows` tokens into `p.emb`.
    fn gather_ngram(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &PleWeights,
        p: &PleScratch,
        rows: usize,
    ) -> Result<()> {
        let heads = self.config.ple.as_ref().expect("PLE config").ngram_heads();
        match &w.table {
            NgramTable::Resident(table) => {
                ple::ple_gather_q4(ctx, pass, table, &p.ids, heads, &p.emb)
            }
            NgramTable::Paged(_) => {
                let stage = p.stage.as_ref().expect("paged table without staging");
                let staged = stage.as_table(rows * heads)?;
                ple::ple_gather_q4(ctx, pass, &staged, &p.ids, heads, &p.emb)
            }
        }
    }

    pub fn new_state(
        &self,
        ctx: &MetalContext,
        capacity: usize,
    ) -> Result<DecodeState> {
        let capacity = round_capacity(capacity)?;
        let cfg = &self.config;
        let c = cfg.gdn_conv_channels();
        let kd = cfg.linear_conv_kernel_dim;
        let heads = cfg.linear_num_value_heads;
        let ratio = cfg.indexer.compress_ratio;
        let layers = self
            .weights
            .layers
            .iter()
            .map(|layer| match &layer.mixer {
                Mixer::Gdn(_) => Ok(LayerState::Gdn {
                    layout: GdnStateLayout::for_model(
                        cfg.official_mlx_rounding()
                            && self.gdn_gate == GdnGate::Sigmoid,
                        cfg.linear_num_key_heads,
                        cfg.linear_num_value_heads,
                        cfg.linear_key_head_dim,
                        cfg.linear_value_head_dim,
                    ),
                    state: Tensor::zeros(
                        ctx,
                        &[heads, GDN_HEAD_DIM, GDN_HEAD_DIM],
                        GDN_STATE_DTYPE,
                    )?,
                    conv_windows: [
                        Tensor::zeros(ctx, &[c, kd - 1], DType::BF16)?,
                        Tensor::zeros(ctx, &[c, kd - 1], DType::BF16)?,
                    ],
                }),
                Mixer::Attn(_) => DecodeState::attn_caches(
                    ctx,
                    self.kv_format,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    ratio,
                    capacity,
                ),
            })
            .collect::<Result<Vec<_>>>()?;
        let ple = cfg
            .ple
            .as_ref()
            .map(|p| {
                let s = p.conv_state_len();
                Ok::<_, anyhow::Error>(PleState {
                    hist: [p.eos_token_id, p.eos_token_id],
                    conv_windows: [
                        Tensor::zeros(ctx, &[cfg.hc_width(), s], DType::BF16)?,
                        Tensor::zeros(ctx, &[cfg.hc_width(), s], DType::BF16)?,
                    ],
                    eos: p.eos_token_id,
                })
            })
            .transpose()?;
        let mtp = match &self.weights.mtp {
            Some(_) => Some(MtpState {
                layer: DecodeState::attn_caches(
                    ctx,
                    self.kv_format,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    ratio,
                    capacity,
                )?,
                hidden: Tensor::zeros(ctx, &[cfg.hc_width()], DType::BF16)?,
            }),
            None => None,
        };
        Ok(DecodeState {
            pos: 0,
            rope_delta: 0,
            capacity,
            layers,
            conv_slot: 0,
            ple,
            mtp,
            spec: None,
            kv_format: self.kv_format,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            ratio,
        })
    }

    /// Prompt tokens per batched prefill pass (see `prefill_chunk`).
    pub fn prefill_chunk(&self) -> usize {
        self.prefill_chunk
    }

    /// Sets the prompt tokens per batched prefill pass (a power of two up
    /// to twice `PREFILL_CHUNK`; for measurement and tests).
    pub fn set_prefill_chunk(&mut self, chunk: usize) -> Result<()> {
        ensure!(
            chunk.is_power_of_two() && (256..=PREFILL_CHUNK * 2).contains(&chunk),
            "prefill chunk {chunk}: a power of two from 256 to {}",
            PREFILL_CHUNK * 2
        );
        self.prefill_chunk = chunk;
        Ok(())
    }

    /// The expert cache's counters so far (`None` without a cache).
    pub fn expert_cache_stats(&self) -> Option<ExpertCacheStats> {
        None
    }

    /// Tells the expert cache, if any, where the pass about to be encoded
    /// starts (`ExpertCacheLink::set_position`): a decode or verify pass's
    /// first row, `None` for prefill.
    pub(super) fn set_cache_position(&self, pos: Option<usize>) {
        let _ = pos;
    }

    /// Logical head-value precision; the sampler/scorer transport remains F32.
    pub fn head_value_precision_policy(&self) -> &'static str {
        if self.config.official_mlx_rounding() {
            quant::OFFICIAL_HEAD_VALUE_POLICY
        } else {
            "f32_accumulation_f32_values_f32_buffer"
        }
    }

    /// Experts and top-k of the routed FFN (for reading the log).
    pub fn moe_shape(&self) -> (usize, usize, usize) {
        (
            self.weights.layers.len(),
            self.config.num_experts,
            self.config.num_experts_per_tok,
        )
    }

    /// Scratch whose split-decode and sparse-attention buffers are sized for
    /// `capacity_tokens` of context rather than the engine's ceiling.
    pub fn new_scratch_with_capacity(
        &self,
        ctx: &MetalContext,
        capacity_tokens: usize,
    ) -> Result<Scratch> {
        let cfg = &self.config;
        // Dense decode only ever runs up to the indexer's dense limit.
        let splits =
            sdpa_split_scratch_splits(capacity_tokens.min(cfg.indexer.dense_limit()));
        let h = cfg.hidden_size;
        let wide = cfg.hc_width();
        let inter = cfg.shared_expert_intermediate_size;
        let dim_v = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_c = cfg.gdn_conv_channels();
        let heads = cfg.linear_num_value_heads;
        let (nq, nkv, hd) =
            (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let nh = cfg.indexer.n_heads;
        let bf = DType::BF16;
        let gdn_in = Tensor::zeros(ctx, &[conv_c + dim_v + 2 * heads], bf)?;
        let attn_qkv = Tensor::zeros(ctx, &[(nq * 2 + 2 * nkv) * hd], bf)?;
        let mlp_gu = Tensor::zeros(ctx, &[2 * inter], bf)?;
        // Largest dense projection the batched fallback may stage. Expert
        // stacks use native grouped Q4 kernels and never dequant in full.
        let ple_embed = cfg.ple.as_ref().map_or(0, |p| p.embed_dim);
        let dequant_numel = [
            conv_c * h,
            dim_v * h,
            heads * h,
            h * dim_v,
            nq * 2 * hd * h,
            nkv * hd * h,
            h * nq * hd,
            cfg.num_experts * h,
            cfg.moe_intermediate_size * h,
            h * cfg.moe_intermediate_size,
            inter * h,
            h * inter,
            cfg.hc_lowrank * wide,
            cfg.hc_count * wide,
            (nh + 1) * INDEXER_D * h,
            wide * ple_embed,
            h * ple_embed,
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        Ok(Scratch {
            sync: StepSync::new(ctx)?,
            x: Tensor::zeros(ctx, &[h], bf)?,
            hyper: Tensor::zeros(ctx, &[wide], bf)?,
            hc: HcScratch::new(ctx, cfg, 1)?,
            branch_out: Tensor::zeros(ctx, &[h], bf)?,
            mlp_gate: mlp_gu.view(0, &[inter])?,
            mlp_up: mlp_gu.view(inter, &[inter])?,
            mlp_act: Tensor::zeros(ctx, &[inter], bf)?,
            qkv: gdn_in.view(0, &[conv_c])?,
            qkv_conv: Tensor::zeros(ctx, &[conv_c], bf)?,
            z: gdn_in.view(conv_c, &[dim_v])?,
            a: gdn_in.view(conv_c + dim_v, &[heads])?,
            b: gdn_in.view(conv_c + dim_v + heads, &[heads])?,
            gdn_gated: Tensor::zeros(ctx, &[dim_v], bf)?,
            gdn_raw: Tensor::zeros(ctx, &[dim_v], bf)?,
            qg: attn_qkv.view(0, &[nq, 2 * hd])?,
            q: Tensor::zeros(ctx, &[nq, hd], bf)?,
            gate: Tensor::zeros(ctx, &[nq, hd], bf)?,
            k_new: attn_qkv.view(nq * 2 * hd, &[nkv, hd])?,
            v_new: attn_qkv.view((nq * 2 + nkv) * hd, &[nkv, hd])?,
            attn_o: Tensor::zeros(ctx, &[nq, hd], bf)?,
            attn_gated: Tensor::zeros(ctx, &[nq * hd], bf)?,
            idx_qk: Tensor::zeros(ctx, &[(nh + 1) * INDEXER_D], bf)?,
            idx_q: Tensor::zeros(ctx, &[nh, INDEXER_D], bf)?,
            qsa: QsaScratch::new(ctx, cfg, capacity_tokens, 1)?,
            ple: self
                .ple_table()
                .map(|t| PleScratch::new(ctx, cfg, 1, t, false))
                .transpose()?,
            gdn_in,
            attn_qkv,
            mlp_gu,
            logits: Tensor::zeros(ctx, &[cfg.vocab_size], DType::F32)?,
            sdpa_partials: Tensor::zeros(ctx, &[nq, splits, hd], DType::F32)?,
            sdpa_stats: Tensor::zeros(ctx, &[nq, splits, 2], DType::F32)?,
            sampler: SamplerScratch::new(ctx, cfg.vocab_size)?,
            next_token: Tensor::zeros(ctx, &[2], DType::U32)?,
            dequant: Tensor::zeros(ctx, &[dequant_numel], bf)?,
            moe: MoeScratch::new(ctx, &moe_dims(cfg))?,
            prefill: None,
            vision: None,
            batch: None,
            spec: self
                .weights
                .mtp
                .as_ref()
                .map(|_| SpecScratch::new(ctx, cfg))
                .transpose()?,
        })
    }

    /// Makes sure `s.prefill` can hold chunks of `needed` rows.
    pub(super) fn ensure_prefill_scratch(
        &self,
        ctx: &MetalContext,
        s: &mut Scratch,
        needed: usize,
    ) -> Result<()> {
        let needed = needed.min(self.prefill_chunk);
        let have = s.prefill.as_ref().map_or(0, |p| p.m);
        if have < needed {
            let target = needed.next_power_of_two().min(self.prefill_chunk);
            s.prefill = None;
            s.prefill = Some(PrefillScratch::new(
                ctx,
                &self.config,
                target,
                MAX_SEQ,
                self.ple_table(),
                self.weights.mtp.is_some(),
                self.kv_format,
            )?);
        }
        Ok(())
    }

    /// Encodes one decode step reading `next_token[slot_in]` and drawing
    /// into `next_token[slot_out]`, without committing. Without `park` the
    /// caller stages the token's n-gram rows first, then commits and advances
    /// `pos`. With `park` (a value from [`StepSync::arm`]) the pass waits on
    /// the step sync right before the n-gram gather, so it can be committed
    /// at once and released after staging.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_decode_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        s: &Scratch,
        slot_in: usize,
        slot_out: usize,
        draw: Draw<'_>,
        park: Option<u64>,
    ) -> Result<EncodedPass<'a>> {
        ensure!(state.spec.is_none(), "decode step during a pending speculative step");
        let pass = ctx.begin_concurrent()?;
        pass.set_label("decode");
        self.encode_decode_graph(
            ctx, &pass, state, s, slot_in, slot_out, draw, park, None,
        )?;
        pass.end()
    }

    /// The ordinary decode graph with copies for speculative rollback.
    pub(super) fn encode_host_verify_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        s: &Scratch,
        row: usize,
        rows: usize,
    ) -> Result<EncodedPass<'a>> {
        ensure!(
            row < rows && rows <= MAX_DRAFTS + 1,
            "invalid host verify capture row"
        );
        let spec = s
            .spec
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing host verify captures"))?;
        let hyper = &s
            .prefill
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing host hidden captures"))?
            .hyper;
        let pass = ctx.begin_concurrent()?;
        pass.set_label("exact host MTP verify");
        let greedy = SamplingParams::greedy();
        self.encode_decode_graph(
            ctx,
            &pass,
            state,
            s,
            0,
            1,
            Draw { params: &greedy, step: state.pos },
            None,
            Some(DecodeCapture { row, rows, spec, hyper }),
        )?;
        pass.end()
    }

    /// Host work for the step consuming `token`: stage its n-gram rows and
    /// advance the hash history.
    pub fn prepare_step_inputs(
        &self,
        state: &mut DecodeState,
        s: &Scratch,
        token: u32,
    ) -> Result<()> {
        if let (Some(pst), Some(p)) = (&mut state.ple, &s.ple) {
            let w = self
                .weights
                .layers
                .iter()
                .find_map(|l| l.ple.as_deref())
                .expect("PLE state without weights");
            self.stage_ngram(w, p, &[token], pst.hist)?;
            pst.hist = NgramHasher::advance(pst.hist, &[token]);
        }
        Ok(())
    }

    /// Runs the prompt in batches of `PREFILL_CHUNK` tokens, one command
    /// buffer per chunk. With `draw`, the final token's logits are sampled
    /// into `next_token[0]`. Text only: the rotary position of every row is
    /// its sequence index plus the state's `rope_delta`.
    pub fn prefill(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        tokens: &[u32],
        draw: Option<Draw<'_>>,
    ) -> Result<()> {
        self.prefill_with_vision(ctx, state, s, tokens, draw, None)
    }

    /// [`Self::prefill`] for a prompt that may carry images: `tokens` are fed
    /// at `state.pos` as sequence indices into `vision.positions`, every
    /// row takes its own 3-axis rotary position, the placeholders' embedding
    /// rows are replaced by the images' merged rows (in the trunk and in the
    /// draft head's catch-up), and the state's `rope_delta` becomes the
    /// prompt's. Without `vision` this is the text path, untouched.
    pub fn prefill_with_vision(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        tokens: &[u32],
        draw: Option<Draw<'_>>,
        vision: Option<&VisionInput<'_>>,
    ) -> Result<()> {
        self.prefill_chunks(ctx, state, s, tokens, draw, vision, None).map(drop)
    }

    /// [`Self::prefill_with_vision`] without a draw that stops before the
    /// first chunk at which `stop` returns true (checked before every chunk,
    /// the first included). Returns the tokens fed, a whole number of chunks
    /// unless all of them. The state is then at rest at that position, as
    /// after a prefill of exactly those tokens: positions, n-gram history
    /// and convolution slot advance per completed chunk, and a staged-ahead
    /// next chunk only wrote scratch.
    pub fn prefill_until(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        tokens: &[u32],
        vision: Option<&VisionInput<'_>>,
        stop: &dyn Fn() -> bool,
    ) -> Result<usize> {
        self.prefill_chunks(ctx, state, s, tokens, None, vision, Some(stop))
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_chunks(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        s: &mut Scratch,
        tokens: &[u32],
        draw: Option<Draw<'_>>,
        vision: Option<&VisionInput<'_>>,
        stop: Option<&dyn Fn() -> bool>,
    ) -> Result<usize> {
        ensure!(!tokens.is_empty(), "empty prompt");
        ensure!(state.spec.is_none(), "prefill during a pending speculative step");
        let h = self.config.hidden_size;
        if let Some(v) = vision {
            let n = v.positions.len();
            ensure!(
                state.pos + tokens.len() <= n,
                "the prompt's positions cover {n} tokens, the prefill feeds {}..{}",
                state.pos,
                state.pos + tokens.len()
            );
            for image in v.images {
                ensure!(
                    image.span.end() <= n,
                    "image span {}..{} exceeds the prompt of {n} tokens",
                    image.span.start,
                    image.span.end()
                );
                ensure!(
                    image.rows.shape() == [image.span.len, h]
                        && image.rows.dtype() == DType::BF16,
                    "image rows are {:?} {:?}, the span at {} needs bf16 [{}, {h}]",
                    image.rows.shape(),
                    image.rows.dtype(),
                    image.span.start,
                    image.span.len
                );
            }
            state.rope_delta = v.positions.rope_delta;
        }
        let alloc_started = std::time::Instant::now();
        state.ensure_capacity(ctx, state.pos + tokens.len())?;
        self.ensure_prefill_scratch(ctx, s, tokens.len())?;
        let alloc_secs = alloc_started.elapsed().as_secs_f64();
        crate::stats::record(|c| c.prefill.alloc_secs += alloc_secs);
        let s = &*s;
        let capacity = s
            .prefill
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("prefill scratch missing after growth"))?;
        let ple_w = self.weights.layers.iter().find_map(|l| l.ple.as_deref());
        let ratio = self.config.indexer.compress_ratio;
        let mut remaining = tokens.len();
        let profile = false;
        let chunks: Vec<&[u32]> = tokens.chunks(self.prefill_chunk).collect();
        // Staging ahead: while the GPU runs chunk k from one staging buffer,
        // the host stages chunk k+1's rows into the other (a paged table's
        // cold rows are SSD reads, and a chunk's gather is a tenth of its
        // GPU time). Buffer k % 2 serves chunk k; chunk k+1's buffer was
        // last read by chunk k-1, which completed before chunk k was
        // committed. The rows are the same ones the unpipelined loop
        // gathers: chunk k+1's ids follow the history advanced over chunk k.
        let ahead = self.ngram_ahead
            && chunks.len() > 1
            && ple_w.is_some_and(|w| w.table.is_paged())
            && capacity.ple.as_ref().is_some_and(|p| p.stage_alt.is_some());
        let buffer = |k: usize| if ahead { k % 2 } else { 0 };
        let mut staged_ahead = false;
        let mut fed = 0;
        for (k, &chunk) in chunks.iter().enumerate() {
            if stop.is_some_and(|stop| stop()) {
                return Ok(fed);
            }
            let mut ps = capacity.chunk(chunk)?;
            ps.ple = capacity
                .ple
                .as_ref()
                .map(|p| p.rows_on(chunk.len(), buffer(k)))
                .transpose()?;
            let staged = std::time::Instant::now();
            if !staged_ahead
                && let (Some(w), Some(p), Some(pst)) = (ple_w, &ps.ple, &state.ple)
            {
                self.stage_ngram(w, p, chunk, pst.hist)?;
            }
            let ngram_secs = staged.elapsed().as_secs_f64();
            remaining -= chunk.len();
            let chunk_vision = match vision {
                Some(v) => {
                    // The rows this chunk's kernels rope: its own, the head's
                    // catch-up row one before it, and the first token of
                    // every block the chunk or the head completes, which lies
                    // at most `ratio - 1` before that row.
                    let pos = state.pos;
                    let lo = pos.saturating_sub(1) / ratio * ratio;
                    let hi = pos + chunk.len();
                    let positions = ps.positions.view(0, &[hi - lo, 3])?;
                    positions
                        .write_bytes(bytemuck::cast_slice(&v.positions.flat(lo..hi)))?;
                    Some(PrefillVision { positions, base: lo, images: v.images })
                }
                None => None,
            };
            let mode = BatchMode::Prefill {
                draw: if remaining == 0 { draw } else { None },
                vision: chunk_vision.as_ref(),
            };
            let started = std::time::Instant::now();
            let encoded = self.encode_batch(ctx, state, s, &ps, mode)?;
            let encoded_at = std::time::Instant::now();
            let pending = encoded.commit()?;
            // The next chunk's rows, while this one runs.
            let next = chunks.get(k + 1).filter(|_| ahead);
            staged_ahead = false;
            let mut ahead_span = None;
            if let (Some(&next), Some(w), Some(cap), Some(pst)) =
                (next, ple_w, &capacity.ple, &state.ple)
            {
                let from = crate::metal::host_secs();
                let p = cap.rows_on(next.len(), buffer(k + 1))?;
                self.stage_ngram(w, &p, next, NgramHasher::advance(pst.hist, chunk))?;
                ahead_span = Some((from, crate::metal::host_secs()));
                staged_ahead = true;
            }
            let completed = pending.wait_retain()?;
            // The span from the commit feedback, which arrives a few tens of
            // microseconds after completion; the request's timings split the
            // chunk's wall time into host, GPU and waiting with it.
            let timing = completed.timing().ok();
            let gpu_secs = timing.map_or(0.0, |t| t.gpu_end_secs - t.gpu_start_secs);
            // What of the staging ahead outlasted the chunk kept the GPU
            // idle: exposed, and not waiting. Without GPU timestamps the
            // whole of it counts as exposed.
            let (ahead_exposed, ahead_hidden) = match ahead_span {
                Some((from, to)) => {
                    let exposed = match timing {
                        Some(t) => {
                            (to - t.gpu_end_secs.max(from)).clamp(0.0, to - from)
                        }
                        None => to - from,
                    };
                    (exposed, to - from - exposed)
                }
                None => (0.0, 0.0),
            };
            let submitted_secs = encoded_at.elapsed().as_secs_f64();
            let encode_secs = (encoded_at - started).as_secs_f64();
            crate::stats::record(|c| {
                c.prefill.chunks += 1;
                c.prefill.ngram_secs += ngram_secs + ahead_exposed;
                c.gather.hidden_secs += ahead_hidden;
                c.prefill.encode_secs += encode_secs;
                c.prefill.gpu_secs += gpu_secs;
                c.prefill.wait_secs +=
                    (submitted_secs - gpu_secs - ahead_exposed).max(0.0);
            });
            if profile {
                {}
            }
            state.pos += chunk.len();
            fed += chunk.len();
            if let Some(pst) = &mut state.ple {
                pst.hist = NgramHasher::advance(pst.hist, chunk);
            }
            // The chunk's recurrent kernels wrote the other window buffers.
            state.conv_slot = 1 - state.conv_slot;
        }
        Ok(fed)
    }

    /// Whether a multi-chunk prefill stages the next chunk's n-gram rows
    /// while the GPU runs the current one (the default). Off, every chunk
    /// stages its rows before its commit, as before the pipeline; the
    /// gathered rows are the same either way (for tests and measurement).
    /// The format of the attention K/V caches of the states and scratch
    /// made from now on (states and scratch made before keep theirs, and
    /// the kernels refuse to mix them).
    pub fn set_kv_format(&mut self, format: KvFormat) {
        self.kv_format = format;
    }

    pub fn kv_format(&self) -> KvFormat {
        self.kv_format
    }

    pub fn set_ngram_ahead(&mut self, on: bool) {
        self.ngram_ahead = on;
    }

    /// Select head work only while the request is reset.
    pub fn set_mtp_enabled(&mut self, enabled: bool) -> Result<()> {
        ensure!(!enabled || self.weights.mtp.is_some(), "no MTP head loaded");
        self.mtp_enabled = enabled;
        Ok(())
    }

    // --- prefill ------------------------------------------------------------

    /// Encodes the batched graph for the chunk `ps` at `state.pos` without
    /// committing: every layer over all rows, then the head according to
    /// `mode`. Barriers mark every inter-level data edge so the same code runs
    /// on the serial and the concurrent encoder.
    pub(super) fn encode_batch<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        s: &Scratch,
        ps: &PrefillScratch,
        mode: BatchMode<'_>,
    ) -> Result<EncodedPass<'a>> {
        let m = ps.m;
        let pass = self.begin_batched(ctx, m)?;
        ensure!(state.pos + m <= state.capacity, "sequence full ({})", state.capacity);
        let cfg = &self.config;
        let (h, g) = (cfg.hidden_size, cfg.hc_count);
        let pos = state.pos;
        let conv_slot = state.conv_slot;
        let verify = matches!(mode, BatchMode::Verify { .. } | BatchMode::VerifyHost);
        let profile_exact = matches!(mode, BatchMode::VerifyHost);
        pass.set_label(if verify { "verify" } else { "prefill" });
        self.set_cache_position(verify.then_some(pos));
        if !verify && m > GATED_PREFILL_ROWS {
            // Under the expert cache every MoE layer waits for its misses,
            // and a long prefill's waits add up to seconds (5.6 s for 4 096
            // rows at 40 GB), past the 4 s after which Metal releases a
            // queue wait and the pass runs on with experts not loaded. Gated,
            // the pass is submitted layer by layer as the misses are read,
            // so the queue never waits. Without the cache there are no
            // waits and nothing changes.
            pass.gate_host_waits();
        }
        if verify {
            ensure!(
                m <= MAX_DRAFTS + 1,
                "verify batch of {m} rows exceeds {}",
                MAX_DRAFTS + 1
            );
        }
        let spec = if verify { s.spec.as_ref() } else { None };
        let vision = match mode {
            BatchMode::Prefill { vision, .. } => vision,
            BatchMode::Verify { .. } | BatchMode::VerifyHost => None,
        };
        // Rotary positions: each row's own 3-axis position for a chunk of a
        // prompt with an image, otherwise the sequence index plus the
        // state's delta (0 for text).
        let rope = match vision {
            Some(v) => v.rope(),
            None => Rope::Delta(state.rope_delta),
        };

        if profile_exact {
            pass.profile_stage(crate::metal::stage_profile::Phase::Embedding)?;
        }
        quant::gather_rows_quant(
            ctx,
            &pass,
            &self.weights.embed_tokens,
            &ps.ids,
            &ps.x,
        )?;
        if let Some(v) = vision {
            // The images' merged rows replace the placeholders' embeddings
            // before the streams are initialised from them (the reference's
            // `masked_scatter` on `inputs_embeds`).
            let overrides = row_overrides(v.images, pos, m, h)?;
            if !overrides.is_empty() {
                pass.level_barrier(&[&ps.x])?;
                override_rows(ctx, &pass, &ps.x, &overrides, h)?;
            }
        }
        pass.level_barrier(&[&ps.x])?;
        hc_broadcast_bf16(ctx, &pass, &ps.x, &ps.hyper, h, g)?;
        pass.level_barrier(&[&ps.hyper])?;

        let mut gdn_index = 0usize;
        for (layer_index, (layer, lstate)) in
            self.weights.layers.iter().zip(state.layers.iter()).enumerate()
        {
            let ple = match (&layer.ple, &ps.ple, &state.ple) {
                (Some(w), Some(p), Some(pst)) => Some((w.as_ref(), p, pst)),
                _ => None,
            };
            let capture = match (spec, lstate) {
                (Some(sp), LayerState::Gdn { .. }) => {
                    let mid = (m > 1)
                        .then(|| {
                            sp.mid[gdn_index].view(
                                0,
                                &[
                                    m - 1,
                                    cfg.linear_num_value_heads,
                                    GDN_HEAD_DIM,
                                    GDN_HEAD_DIM,
                                ],
                            )
                        })
                        .transpose()?;
                    Some(Capture {
                        exact: matches!(mode, BatchMode::VerifyHost),
                        mid,
                        conv_in: prefix_rows(&sp.conv_in[gdn_index], m)?,
                        ple_conv_in: sp
                            .ple_conv_in
                            .as_ref()
                            .map(|t| prefix_rows(t, m))
                            .transpose()?,
                    })
                }
                (Some(sp), LayerState::Attn { .. }) => Some(Capture {
                    exact: matches!(mode, BatchMode::VerifyHost),
                    mid: None,
                    conv_in: prefix_rows(&sp.conv_in[0], 0)?,
                    ple_conv_in: sp
                        .ple_conv_in
                        .as_ref()
                        .map(|t| prefix_rows(t, m))
                        .transpose()?,
                }),
                _ => None,
            };
            if matches!(lstate, LayerState::Gdn { .. }) {
                gdn_index += 1;
            }
            self.block_batched(
                ctx,
                &pass,
                layer_index,
                layer,
                lstate,
                &ps.hyper,
                pos,
                conv_slot,
                s,
                ps,
                ple,
                capture.as_ref(),
                rope,
            )?;
            // Keep this layer's routing (and, enabled, the predictions from
            // its gate input) before the next layer overwrites both.
        }

        match mode {
            BatchMode::Prefill { draw, vision } => {
                if let Some(draw) = draw {
                    // Only the last prompt token feeds decoding: pull its stream
                    // row into the single-token scratch and reuse the decode
                    // head path.
                    gather_row_bf16(ctx, &pass, &ps.hyper, &s.hyper, m - 1)?;
                    pass.level_barrier(&[&s.hyper])?;
                    self.hc_read_decode(ctx, &pass, &self.weights.final_mixer, s)?;
                    quant::gemv_quant_logits(
                        ctx,
                        &pass,
                        &self.weights.lm_head,
                        &s.hc.mixed,
                        &s.logits,
                        self.config.official_mlx_rounding(),
                    )?;
                    pass.level_barrier(&[&s.logits])?;
                    // Slot 0 by convention: the pipelined loop's first decode step
                    // consumes the prefill token from this slot.
                    let out = s.next_token.view(0, &[1])?;
                    sample_f32(
                        ctx,
                        &pass,
                        &s.logits,
                        &s.sampler,
                        draw.params,
                        draw.step,
                        &out,
                    )?;
                    pass.level_barrier(&[&s.next_token])?;
                }
                if self.mtp_enabled
                    && let (Some(mtp), Some(mst)) = (&self.weights.mtp, &state.mtp)
                {
                    let images = vision.map_or(&[][..], |v| v.images);
                    self.mtp_catch_up(ctx, &pass, mtp, mst, pos, s, ps, rope, images)?;
                }
            }
            BatchMode::Verify { .. } | BatchMode::VerifyHost => {
                let sp = s.spec.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("verify pass without spec scratch")
                })?;
                if matches!(mode, BatchMode::VerifyHost) {
                    pass.profile_stage(crate::metal::stage_profile::Phase::Hc)?;
                    self.hc_read_verify_exact(
                        ctx,
                        &pass,
                        &self.weights.final_mixer,
                        &ps.hyper,
                        ps,
                    )?;
                } else {
                    self.hc_read_batched(
                        ctx,
                        &pass,
                        &self.weights.final_mixer,
                        &ps.hyper,
                        s,
                        ps,
                    )?;
                }
                self.trace_stage(
                    ctx,
                    &pass,
                    cfg.num_hidden_layers,
                    "head_input",
                    &ps.hc.mixed,
                    m,
                )?;
                let logits = prefix_rows(&sp.logits, m)?;
                if matches!(mode, BatchMode::VerifyHost) {
                    pass.profile_stage(crate::metal::stage_profile::Phase::Head)?;
                    quant::gemv_quant_logits_rows_decode_exact(
                        ctx,
                        &pass,
                        &self.weights.lm_head,
                        &ps.hc.mixed,
                        &logits,
                        cfg.official_mlx_rounding(),
                    )?;
                } else {
                    crate::moe_ffn::project_lm_logits(
                        ctx,
                        &pass,
                        &ps.hc.mixed,
                        &self.weights.lm_head,
                        &logits,
                        &s.dequant,
                        self.config.official_mlx_rounding(),
                    )?;
                }
                pass.level_barrier(&[&logits])?;
                self.trace_stage(
                    ctx,
                    &pass,
                    cfg.num_hidden_layers,
                    "logits",
                    &logits,
                    m,
                )?;
                let BatchMode::Verify { params, step0 } = mode else {
                    pass.finish_stage_profile()?;
                    s.sync.signal_done(&pass)?;
                    return pass.end();
                };
                // One draw per row; the rows share the sampler scratch (and
                // the penalty counts, which must see earlier rows' draws).
                // Row j's draw checks proposal j (the row after it in the
                // pass): a speculative draw against the distribution the
                // draft head drew it from. The last row has no proposal.
                for j in 0..m {
                    let row = sp.logits.view(j * cfg.vocab_size, &[cfg.vocab_size])?;
                    let out = sp.verify_tokens.view(j, &[1])?;
                    if j + 1 < m {
                        sample_spec_f32(
                            ctx,
                            &pass,
                            &row,
                            &s.sampler,
                            params,
                            step0 + j,
                            &sp.dists.slot(j)?,
                            &ps.ids.view(j + 1, &[1])?,
                            &out,
                        )?;
                    } else {
                        sample_f32(
                            ctx,
                            &pass,
                            &row,
                            &s.sampler,
                            params,
                            step0 + j,
                            &out,
                        )?;
                    }
                    pass.level_barrier(&[&sp.verify_tokens])?;
                }
            }
        }
        s.sync.signal_done(&pass)?;
        pass.end()
    }

    /// One decoder block over the rows of `ps` on the residual `hyper`
    /// (`[m, G*H]`): the optional PLE, the mixer branch, the MoE, each read
    /// and written through its hyper-connection. `capture` (verify passes)
    /// records what a rollback needs.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn block_batched(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        layer_index: usize,
        layer: &LayerWeights,
        lstate: &LayerState,
        hyper: &Tensor,
        pos: usize,
        conv_slot: usize,
        s: &Scratch,
        ps: &PrefillScratch,
        ple: Option<(&PleWeights, &PleScratch, &PleState)>,
        capture: Option<&Capture>,
        rope: Rope<'_>,
    ) -> Result<()> {
        let cfg = &self.config;
        let (h, g) = (cfg.hidden_size, cfg.hc_count);
        let profile_exact = capture.is_some_and(|c| c.exact);
        // Official exact rows use one stock HC read per row, including its
        // finished gate contract. Injection must follow that scalar policy.
        let hc_policy_rows =
            if profile_exact && cfg.official_mlx_rounding() { 1 } else { ps.m };
        self.trace_stage(ctx, pass, layer_index, "input", hyper, ps.m)?;
        if let Some((w, p, pst)) = ple {
            if profile_exact {
                pass.profile_stage(crate::metal::stage_profile::Phase::Ple)?;
            }
            self.ple_batched(
                ctx,
                pass,
                w,
                p,
                pst,
                conv_slot,
                hyper,
                s,
                ps,
                capture.and_then(|c| c.ple_conv_in.as_ref()),
                capture.is_some_and(|c| c.exact),
            )?;
        }

        self.trace_stage(ctx, pass, layer_index, "post_ple", hyper, ps.m)?;

        if profile_exact {
            pass.profile_stage(crate::metal::stage_profile::Phase::Hc)?;
        }
        if capture.is_some_and(|c| c.exact) {
            self.hc_read_verify_exact(ctx, pass, &layer.attn_hc, hyper, ps)?;
        } else {
            self.hc_read_batched(ctx, pass, &layer.attn_hc, hyper, s, ps)?;
        }
        self.trace_stage(ctx, pass, layer_index, "mixer_input", &ps.hc.mixed, ps.m)?;
        match (&layer.mixer, lstate) {
            (Mixer::Gdn(w), LayerState::Gdn { state, conv_windows, layout }) => {
                if profile_exact {
                    pass.profile_stage(crate::metal::stage_profile::Phase::Gdn)?;
                }
                self.gdn_batched(
                    ctx,
                    pass,
                    w,
                    s,
                    ps,
                    state,
                    *layout,
                    conv_windows,
                    conv_slot,
                    capture,
                    layer_index,
                )?;
            }
            (
                Mixer::Attn(w),
                LayerState::Attn { k_cache, v_cache, idx_keys, blk_keys },
            ) => {
                if capture.is_some_and(|c| c.exact) {
                    pass.profile_stage(crate::metal::stage_profile::Phase::Qsa)?;
                    let Rope::Delta(delta) = rope else {
                        anyhow::bail!("exact verify requires text rotary positions");
                    };
                    self.attn_verify_exact(
                        ctx, pass, w, s, ps, k_cache, v_cache, idx_keys, blk_keys, pos,
                        delta,
                    )?;
                } else {
                    self.attn_batched(
                        ctx, pass, w, s, ps, k_cache, v_cache, idx_keys, blk_keys, pos,
                        rope,
                    )?;
                }
            }
            _ => anyhow::bail!("layer/state kind mismatch"),
        }
        pass.level_barrier(&[&ps.branch_out])?;
        self.trace_stage(ctx, pass, layer_index, "mixer_output", &ps.branch_out, ps.m)?;
        if profile_exact {
            pass.profile_stage(crate::metal::stage_profile::Phase::Hc)?;
        }
        hc_inject_bf16(
            ctx,
            pass,
            hyper,
            &ps.branch_out,
            &ps.hc.inj,
            h,
            g,
            self.config.official_mlx_rounding(),
            self.hc_primitive_sigmoid(&layer.attn_hc, hc_policy_rows),
            self.hc_injection_kind(&layer.attn_hc, hc_policy_rows),
        )?;
        pass.level_barrier(&[hyper])?;

        if capture.is_some_and(|c| c.exact) {
            self.hc_read_verify_exact(ctx, pass, &layer.mlp_hc, hyper, ps)?;
        } else {
            self.hc_read_batched(ctx, pass, &layer.mlp_hc, hyper, s, ps)?;
        }
        self.trace_stage(ctx, pass, layer_index, "moe_input", &ps.hc.mixed, ps.m)?;
        let io = PrefillMoeIo {
            x: &ps.hc.mixed,
            out: &ps.branch_out,
            stack: &ps.stack,
            mlp_gate: &ps.mlp_gate,
            mlp_up: &ps.mlp_up,
            mlp_act: &ps.mlp_act,
            dequant: &s.dequant,
        };
        if profile_exact {
            pass.profile_stage(crate::metal::stage_profile::Phase::Moe)?;
        }
        let injected = if capture.is_some_and(|c| c.exact) {
            prefill_moe_decode_exact_with_injection(
                ctx,
                pass,
                &moe_dims(cfg),
                &layer.ffn,
                &io,
                &ps.moe,
                &s.moe,
                (!cfg.official_mlx_rounding()).then_some(
                    crate::kernels::moe::HcInjection {
                        hyper,
                        logits: &ps.hc.inj,
                        groups: g,
                    },
                ),
            )?
        } else {
            prefill_moe(ctx, pass, &moe_dims(cfg), &layer.ffn, &io, &ps.moe)?;
            false
        };
        if injected {
            pass.level_barrier(&[&ps.branch_out, hyper])?;
        } else {
            pass.level_barrier(&[&ps.branch_out])?;
        }
        self.trace_stage(ctx, pass, layer_index, "moe_output", &ps.branch_out, ps.m)?;
        if !injected {
            hc_inject_bf16(
                ctx,
                pass,
                hyper,
                &ps.branch_out,
                &ps.hc.inj,
                h,
                g,
                self.config.official_mlx_rounding(),
                self.hc_primitive_sigmoid(&layer.mlp_hc, hc_policy_rows),
                self.hc_injection_kind(&layer.mlp_hc, hc_policy_rows),
            )?;
            pass.level_barrier(&[hyper])?;
        }
        self.trace_stage(ctx, pass, layer_index, "output", hyper, ps.m)
    }

    /// `hyper` `[m, G*H]` → `ps.hc.mixed` `[m, H]` and the write-gate input
    /// when the block has an inject weight; hc_injection_kind identifies
    /// raw logits versus a finished gate. Up to `HC_FUSED_MAX_ROWS` rows
    /// (the verify pass, the draft head) take the two fused kernels of the
    /// decode read, so those rows compute the read exactly as decode does;
    /// larger chunks (prefill) take the six-kernel skinny/GEMM chain. Leaves
    /// `mixed`/`inj` ordered.
    pub(super) fn hc_read_batched(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        hc: &HcWeights,
        hyper: &Tensor,
        s: &Scratch,
        ps: &PrefillScratch,
    ) -> Result<()> {
        self.hc_read_rows(ctx, pass, hc, hyper, &ps.hc, ps.m, &s.dequant)
    }

    /// One descriptor/row predicate governs both the read and its subsequent
    /// injection interpretation. A skipped read never returns a finished gate.
    pub(super) fn hc_full_read_selected(&self, hc: &HcWeights, rows: usize) -> bool {
        hc.official_read.as_ref().is_some_and(|read| {
            read.selected_for(
                self.config.official_mlx_rounding(),
                rows,
                !crate::ablate::on(crate::ablate::Group::Hc),
            )
        })
    }

    /// Defer only one attention write into its immediately following MLP read.
    /// Both reads must have produced/return the finished stock gate contract.
    fn hc_write_pair_selected(
        &self,
        first: &HcWeights,
        next: &HcWeights,
        scratch: &HcScratch,
    ) -> bool {
        write_pair_selected(
            self.hc_full_read_selected(first, 1),
            self.hc_full_read_selected(next, 1),
            first.official_read.as_ref().is_some_and(|read| read.has_inject()),
            next.official_read.as_ref().is_some_and(|read| read.has_inject()),
            !crate::ablate::on(crate::ablate::Group::Inject),
            scratch.stock_write_hyper.is_some(),
        )
    }

    pub(super) fn hc_injection_kind(
        &self,
        hc: &HcWeights,
        rows: usize,
    ) -> InjectionInputKind {
        InjectionInputKind::from_full_read(self.hc_full_read_selected(hc, rows))
    }

    /// Follow the sigmoid used by the pinned MLX serving branch, rather than
    /// inferring its narrow fused eligibility solely from the number of rows.
    pub(super) fn hc_primitive_sigmoid(&self, hc: &HcWeights, rows: usize) -> bool {
        self.config.official_mlx_rounding()
            && !official_narrow_sigmoid_supported(
                &hc.norm,
                &hc.down,
                &hc.up,
                hc.inject.as_ref(),
                self.config.hidden_size,
                self.config.hc_count,
                rows,
            )
    }

    fn hc_read_verify_exact(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        hc: &HcWeights,
        hyper: &Tensor,
        ps: &PrefillScratch,
    ) -> Result<()> {
        let (h, g) = (self.config.hidden_size, self.config.hc_count);
        let out = &ps.hc;
        if self.config.official_mlx_rounding() {
            // Keep the stock per-row normalization and reduction order,
            // sharing loaded U2V weights across the short verifier rows.
            ensure!(
                self.hc_full_read_selected(hc, 1),
                "official exact HC rows require an admitted stock read"
            );
            let prepared = hc.official_read.as_ref().expect("selected descriptor");
            let parts = out.stock_parts.as_ref().ok_or_else(|| {
                anyhow::anyhow!("official exact HC rows require stock partials")
            })?;
            prepared.read_rows(
                ctx, pass, hyper, &hc.norm, &out.hn, parts, &out.mixed, &out.inj,
            )?;
            return pass.level_barrier(&[&out.mixed, &out.inj]);
        }
        hc_read_down_q8_decode_rows(
            ctx,
            pass,
            hyper,
            &hc.norm,
            &hc.down,
            hc.inject.as_ref(),
            &out.down,
            &out.inj,
            &out.inv_rms,
            &out.act,
            h,
            g,
            self.config.rms_norm_eps,
            self.config.norm_weight_bias(),
        )?;
        pass.level_barrier(&[&out.down, &out.inj, &out.inv_rms, &out.act])?;
        hc_read_up_mix_q8_cached_rows(
            ctx,
            pass,
            &hc.up,
            &out.act,
            hyper,
            &hc.norm,
            &out.inv_rms,
            &out.mixed,
            h,
            g,
            self.config.norm_weight_bias(),
        )?;
        pass.level_barrier(&[&out.mixed])
    }

    /// [`Self::hc_read_batched`] into any `m`-row hyper-connection scratch
    /// (the prediction log runs a later layer's read into its own).
    #[allow(clippy::too_many_arguments)]
    fn hc_read_rows(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        hc: &HcWeights,
        hyper: &Tensor,
        out: &HcScratch,
        m: usize,
        dequant: &Tensor,
    ) -> Result<()> {
        let cfg = &self.config;
        let (h, g) = (cfg.hidden_size, cfg.hc_count);
        if self.hc_full_read_selected(hc, m) {
            let prepared = hc.official_read.as_ref().expect("selected descriptor");
            let parts = out.stock_parts.as_ref().ok_or_else(|| {
                anyhow::anyhow!("stock HC read selected without its parts scratch")
            })?;
            prepared.read(
                ctx, pass, hyper, &hc.norm, &out.hn, parts, &out.mixed, &out.inj,
            )?;
            return pass.level_barrier(&[&out.mixed, &out.inj]);
        }
        // The external checkpoint's reference rounds normalized streams
        // before either projection. The original fused path retains them
        // in F32; keep it for Lily until a compatible fusion is qualified.
        if !cfg.official_mlx_rounding()
            && m <= HC_FUSED_MAX_ROWS
            && fused_read_supported(&hc.down, &hc.up, hc.inject.as_ref(), h, g)
        {
            hc_read_down_q8_rows(
                ctx,
                pass,
                hyper,
                &hc.norm,
                &hc.down,
                hc.inject.as_ref(),
                &out.down,
                &out.inj,
                &out.inv_rms,
                &out.act,
                h,
                g,
                cfg.rms_norm_eps,
                self.config.norm_weight_bias(),
            )?;
            pass.level_barrier(&[&out.down, &out.inj, &out.inv_rms, &out.act])?;
            hc_read_up_mix_q8_rows(
                ctx,
                pass,
                &hc.up,
                &out.act,
                hyper,
                &hc.norm,
                &out.inv_rms,
                &out.mixed,
                h,
                g,
                self.config.norm_weight_bias(),
            )?;
            return pass.level_barrier(&[&out.mixed]);
        }
        let inv_g = 1.0 / g as f32;
        let primitive_sigmoid = self.hc_primitive_sigmoid(hc, m);
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            hyper,
            &hc.norm,
            &out.hn,
            h,
            g,
            cfg.rms_norm_eps,
            self.config.norm_weight_bias(),
            self.config.official_mlx_rounding(),
        )?;
        pass.level_barrier(&[&out.hn])?;
        // Keep the exact two-stage normalized BF16 stream materialized.
        // Only the admitted one-row custom-narrow geometry fuses the tails.
        if cfg.official_mlx_rounding()
            && m == 1
            && official_narrow_sigmoid_supported(
                &hc.norm,
                &hc.down,
                &hc.up,
                hc.inject.as_ref(),
                h,
                g,
                m,
            )
        {
            official_hc_down_act_q8(ctx, pass, &hc.down, &out.hn, &out.down, &out.act)?;
            if let Some(inject) = &hc.inject {
                project_mat(ctx, pass, &out.hn, inject, &out.inj, dequant)?;
            }
            pass.level_barrier(&[&out.down, &out.act, &out.inj])?;
            official_hc_up_mix_q8(ctx, pass, &hc.up, &out.act, &out.hn, &out.mixed)?;
            return pass.level_barrier(&[&out.mixed]);
        }
        project_mat(ctx, pass, &out.hn, &hc.down, &out.down, dequant)?;
        if let Some(inject) = &hc.inject {
            project_mat(ctx, pass, &out.hn, inject, &out.inj, dequant)?;
        }
        pass.level_barrier(&[&out.down, &out.inj])?;
        silu_scaled_bf16(
            ctx,
            pass,
            &out.down,
            &out.act,
            inv_g,
            self.config.official_mlx_rounding(),
            primitive_sigmoid,
        )?;
        pass.level_barrier(&[&out.act])?;
        project_mat(ctx, pass, &out.act, &hc.up, &out.up, dequant)?;
        pass.level_barrier(&[&out.up])?;
        hc_mix_bf16(
            ctx,
            pass,
            &out.up,
            &out.hn,
            &out.mixed,
            h,
            g,
            self.config.official_mlx_rounding(),
            primitive_sigmoid,
            // Pinned hc_prefill.zig supports one-sequence chunks through
            // 8192 + TAIL_MERGE_MAX(512) - 1, and requires dense inject.
            self.config.official_mlx_rounding()
                && (17..=8703).contains(&m)
                && hc.inject.is_some()
                && official_narrow_sigmoid_supported(
                    &hc.norm,
                    &hc.down,
                    &hc.up,
                    hc.inject.as_ref(),
                    h,
                    g,
                    1,
                ),
        )?;
        pass.level_barrier(&[&out.mixed])
    }

    #[allow(clippy::too_many_arguments)]
    fn ple_batched(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &PleWeights,
        p: &PleScratch,
        pst: &PleState,
        conv_slot: usize,
        hyper: &Tensor,
        s: &Scratch,
        ps: &PrefillScratch,
        conv_capture: Option<&Tensor>,
        exact: bool,
    ) -> Result<()> {
        let cfg = &self.config;
        let ple = cfg.ple.as_ref().expect("PLE weights without PLE config");
        let (h, g, eps) = (cfg.hidden_size, cfg.hc_count, cfg.rms_norm_eps);
        self.gather_ngram(ctx, pass, w, p, ps.m)?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            hyper,
            &w.norm_query,
            &p.query_n,
            h,
            g,
            eps,
            self.config.norm_weight_bias(),
            self.config.official_mlx_rounding(),
        )?;
        pass.level_barrier(&[&p.emb, &p.query_n])?;
        if exact {
            project_mat_decode_exact(ctx, pass, &p.emb, &w.key_proj, &p.key)?;
            project_mat_decode_exact(ctx, pass, &p.emb, &w.value_proj, &p.value)?;
        } else {
            project_mat(ctx, pass, &p.emb, &w.key_proj, &p.key, &s.dequant)?;
            project_mat(ctx, pass, &p.emb, &w.value_proj, &p.value, &s.dequant)?;
        }
        pass.level_barrier(&[&p.key, &p.value])?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &p.key,
            &w.norm_key,
            &p.key_n,
            h,
            g,
            eps,
            self.config.norm_weight_bias(),
            self.config.official_mlx_rounding(),
        )?;
        pass.level_barrier(&[&p.key_n])?;
        ple::ple_gate_value_bf16_with_rounding(
            ctx,
            pass,
            &p.key_n,
            &p.query_n,
            &p.value,
            &p.gated,
            h,
            g,
            self.config.official_mlx_rounding(),
        )?;
        pass.level_barrier(&[&p.gated])?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &p.gated,
            &w.norm_conv,
            &p.gated_n,
            h,
            g,
            eps,
            self.config.norm_weight_bias(),
            self.config.official_mlx_rounding(),
        )?;
        pass.level_barrier(&[&p.gated_n])?;
        if let Some(keep) = conv_capture {
            copy_words(ctx, pass, &p.gated_n, keep)?;
        }
        ple::ple_conv1d_prefill_with_rounding(
            ctx,
            pass,
            &pst.conv_windows[conv_slot],
            &pst.conv_windows[1 - conv_slot],
            &p.gated_n,
            &w.conv_w,
            &p.gated,
            hyper,
            ple.ngram_size,
            self.config.official_mlx_rounding(),
        )?;
        pass.level_barrier(&[hyper, &pst.conv_windows[1 - conv_slot]])
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_batched(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &GdnWeights,
        s: &Scratch,
        ps: &PrefillScratch,
        gdn_state: &Tensor,
        gdn_layout: GdnStateLayout,
        conv_windows: &[Tensor; 2],
        conv_slot: usize,
        capture: Option<&Capture>,
        layer_index: usize,
    ) -> Result<()> {
        let cfg = &self.config;
        if let Some(c) = capture.filter(|c| c.exact && cfg.official_mlx_rounding()) {
            ensure!(
                gdn_layout == GdnStateLayout::ValueKey,
                "official exact GDN layout"
            );
            let projected = ps.stack.view(0, &[ps.m, w.in_proj.out_features()])?;
            crate::kernels::official_q8_projection::gemv_gdn_decode_rows(
                ctx,
                pass,
                &w.in_proj,
                &ps.hc.mixed,
                &projected,
                true,
            )?;
            pass.level_barrier(&[&projected])?;
            split_cols_bf16(ctx, pass, &projected, &[&ps.qkv, &ps.z, &ps.a, &ps.b])?;
            pass.level_barrier(&[&ps.qkv, &ps.z, &ps.a, &ps.b])?;
            copy_words(ctx, pass, &ps.qkv, &c.conv_in)?;
            crate::kernels::gdn::conv1d_verify_value_key_exact(
                ctx,
                pass,
                &conv_windows[conv_slot],
                &conv_windows[1 - conv_slot],
                &ps.qkv,
                &w.conv_w,
                &ps.qkv_conv,
            )?;
            pass.level_barrier(&[
                &ps.qkv_conv,
                &conv_windows[1 - conv_slot],
                &c.conv_in,
            ])?;
            self.trace_stage(ctx, pass, layer_index, "gdn_qkv", &ps.qkv, ps.m)?;
            self.trace_stage(ctx, pass, layer_index, "gdn_z", &ps.z, ps.m)?;
            self.trace_stage(ctx, pass, layer_index, "gdn_a", &ps.a, ps.m)?;
            self.trace_stage(ctx, pass, layer_index, "gdn_b", &ps.b, ps.m)?;
            self.trace_stage(ctx, pass, layer_index, "gdn_conv", &ps.qkv_conv, ps.m)?;
            crate::kernels::gdn::gdn_verify_value_key_exact(
                ctx,
                pass,
                &ps.qkv_conv,
                &ps.a,
                &ps.b,
                &w.a_log,
                &w.dt_bias,
                gdn_state,
                &ps.z,
                &w.norm_w,
                &ps.gdn_out,
                &ps.gdn_gated,
                self.gdn_scale,
                cfg.rms_norm_eps,
                c.mid.as_ref(),
            )?;
            pass.level_barrier(&[&ps.gdn_gated, gdn_state])?;
            self.trace_stage(ctx, pass, layer_index, "gdn_gated", &ps.gdn_gated, ps.m)?;
            return crate::kernels::official_q8_projection::gemv_gdn_decode_rows(
                ctx,
                pass,
                &w.out_proj,
                &ps.gdn_gated,
                &ps.branch_out,
                true,
            );
        }
        let fused_capture = capture.filter(|c| {
            c.exact
                && w.in_proj.bits == 4
                && cfg.linear_conv_kernel_dim == 4
                && w.in_proj.out_features().is_multiple_of(2)
        });
        if let Some(c) = fused_capture {
            quant::gemv_q4_gdn_conv_rows_exact(
                ctx,
                pass,
                &w.in_proj,
                &ps.hc.mixed,
                &ps.qkv,
                &ps.z,
                &ps.a,
                &ps.b,
                &conv_windows[conv_slot],
                &conv_windows[1 - conv_slot],
                &w.conv_w,
                &ps.qkv_conv,
                Some(&c.conv_in),
            )?;
            pass.level_barrier(&[
                &ps.qkv,
                &ps.z,
                &ps.a,
                &ps.b,
                &ps.qkv_conv,
                &conv_windows[1 - conv_slot],
                &c.conv_in,
            ])?;
        } else {
            if capture.is_some_and(|c| c.exact) {
                let width = w.in_proj.out_features();
                let projected = ps.stack.view(0, &[ps.m, width])?;
                project_mat_decode_exact(
                    ctx,
                    pass,
                    &ps.hc.mixed,
                    &w.in_proj,
                    &projected,
                )?;
                pass.level_barrier(&[&projected])?;
                split_cols_bf16(
                    ctx,
                    pass,
                    &projected,
                    &[&ps.qkv, &ps.z, &ps.a, &ps.b],
                )?;
            } else {
                project_stack_or_slices(
                    ctx,
                    pass,
                    &ps.hc.mixed,
                    &w.in_proj,
                    &ps.stack,
                    [
                        (&w.in_proj_qkv, &ps.qkv),
                        (&w.in_proj_z, &ps.z),
                        (&w.in_proj_a, &ps.a),
                        (&w.in_proj_b, &ps.b),
                    ],
                    &s.dequant,
                )?;
            }
            pass.level_barrier(&[&ps.qkv, &ps.z, &ps.a, &ps.b])?;
            if let Some(c) = capture {
                copy_words(ctx, pass, &ps.qkv, &c.conv_in)?;
            }
            conv1d_prefill(
                ctx,
                pass,
                &conv_windows[conv_slot],
                &conv_windows[1 - conv_slot],
                &ps.qkv,
                &w.conv_w,
                &ps.qkv_conv,
                self.config.official_mlx_rounding(),
            )?;
            pass.level_barrier(&[&ps.qkv_conv, &conv_windows[1 - conv_slot]])?;
        }
        self.trace_stage(ctx, pass, layer_index, "gdn_qkv", &ps.qkv, ps.m)?;
        self.trace_stage(ctx, pass, layer_index, "gdn_z", &ps.z, ps.m)?;
        self.trace_stage(ctx, pass, layer_index, "gdn_a", &ps.a, ps.m)?;
        self.trace_stage(ctx, pass, layer_index, "gdn_b", &ps.b, ps.m)?;
        self.trace_stage(ctx, pass, layer_index, "gdn_conv", &ps.qkv_conv, ps.m)?;
        if let Some(capture) = capture.filter(|c| c.exact) {
            gdn_verify_gated_exact(
                ctx,
                pass,
                &ps.qkv_conv,
                &ps.a,
                &ps.b,
                &w.a_log,
                &w.dt_bias,
                gdn_state,
                &ps.z,
                &w.norm_w,
                &ps.gdn_gated,
                self.gdn_scale,
                cfg.linear_num_key_heads,
                cfg.rms_norm_eps,
                self.gdn_gate,
                capture.mid.as_ref(),
            )?;
            pass.level_barrier(&[&ps.gdn_gated, gdn_state])?;
            self.trace_stage(ctx, pass, layer_index, "gdn_gated", &ps.gdn_gated, ps.m)?;
            return project_mat_decode_exact(
                ctx,
                pass,
                &ps.gdn_gated,
                &w.out_proj,
                &ps.branch_out,
            );
        }
        let staging = ps.gdn_stage.staging();
        gdn_prefill_mid(
            ctx,
            pass,
            &ps.qkv_conv,
            &ps.a,
            &ps.b,
            &w.a_log,
            &w.dt_bias,
            &staging,
            gdn_state,
            &ps.gdn_out,
            self.gdn_scale,
            cfg.linear_num_key_heads,
            capture.and_then(|c| c.mid.as_ref()),
            self.config.official_mlx_rounding(),
            gdn_layout,
        )?;
        pass.level_barrier(&[&ps.gdn_out, gdn_state])?;
        gated_rmsnorm(
            ctx,
            pass,
            &ps.gdn_out,
            &ps.z,
            &w.norm_w,
            &ps.gdn_gated,
            cfg.rms_norm_eps,
            self.gdn_gate,
            self.config.official_mlx_rounding(),
        )?;
        pass.level_barrier(&[&ps.gdn_gated])?;
        self.trace_stage(ctx, pass, layer_index, "gdn_gated", &ps.gdn_gated, ps.m)?;
        project_mat(ctx, pass, &ps.gdn_gated, &w.out_proj, &ps.branch_out, &s.dequant)
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_verify_exact(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &AttnWeights,
        s: &Scratch,
        ps: &PrefillScratch,
        k_cache: &KvCache,
        v_cache: &KvCache,
        idx_keys: &Tensor,
        blk_keys: &Tensor,
        pos: usize,
        delta: i64,
    ) -> Result<()> {
        let cfg = &self.config;
        let (m, nq, hd) = (ps.m, cfg.num_attention_heads, cfg.head_dim);
        let idx = &cfg.indexer;
        let (rot, theta, eps) =
            (cfg.rotary_dim(), cfg.rope_parameters.rope_theta, cfg.rms_norm_eps);
        let width = w.qkv_proj.out_features();
        let projected = ps.stack.view(0, &[m, width])?;
        project_mat_decode_exact(ctx, pass, &ps.hc.mixed, &w.qkv_proj, &projected)?;
        project_mat_decode_exact(
            ctx,
            pass,
            &ps.hc.mixed,
            &w.indexer.qk_proj,
            &ps.idx_qk,
        )?;
        pass.level_barrier(&[&projected, &ps.idx_qk])?;
        split_cols_bf16(ctx, pass, &projected, &[&ps.qg, &ps.k_new, &ps.v_new])?;
        pass.level_barrier(&[&ps.qg, &ps.k_new, &ps.v_new])?;
        if cfg.official_mlx_rounding() {
            // These are independent positions: retain the ordinary stock
            // normalization/RoPE and affine-64 cache quantization exactly.
            // The surrounding projections and sparse attention still share
            // the short-batch graph and its weight residency.
            let kvh = cfg.num_key_value_heads;
            for row in 0..m {
                q_norm_rope_split_decode_with_bias(
                    ctx,
                    pass,
                    &ps.qg.view(row * nq * 2 * hd, &[nq, 2 * hd])?,
                    &w.q_norm,
                    &ps.q.view(row * nq * hd, &[nq, hd])?,
                    &ps.gate.view(row * nq * hd, &[nq, hd])?,
                    rot,
                    pos + row,
                    theta,
                    eps,
                    delta,
                    cfg.norm_weight_bias(),
                    true,
                )?;
                k_norm_rope_scatter_decode_with_bias(
                    ctx,
                    pass,
                    &ps.k_new.view(row * kvh * hd, &[kvh, hd])?,
                    &w.k_norm,
                    k_cache,
                    rot,
                    pos + row,
                    theta,
                    eps,
                    delta,
                    cfg.norm_weight_bias(),
                    true,
                )?;
            }
        } else {
            q_norm_rope_split_decode_rows(
                ctx, pass, &ps.qg, &w.q_norm, &ps.q, &ps.gate, rot, pos, theta, eps,
                delta,
            )?;
            k_norm_rope_scatter_decode_rows(
                ctx, pass, &ps.k_new, &w.k_norm, k_cache, rot, pos, theta, eps, delta,
            )?;
        }
        scatter_kv(ctx, pass, v_cache, &ps.v_new, pos)?;
        qsa::qsa_prep_q_with_rounding(
            ctx,
            pass,
            &ps.idx_qk,
            &w.indexer.q_norm,
            &ps.idx_q,
            idx.n_heads,
            rot,
            pos,
            theta,
            eps,
            Rope::Delta(delta),
            cfg.norm_weight_bias(),
            cfg.official_mlx_rounding(),
        )?;
        qsa::qsa_scatter_keys(ctx, pass, &ps.idx_qk, idx_keys, idx.n_heads, pos)?;
        pass.level_barrier(&[
            &ps.q,
            &ps.gate,
            &k_cache.values,
            &v_cache.values,
            &ps.idx_q,
            idx_keys,
        ])?;
        let first_block = pos / idx.compress_ratio;
        let complete = (pos + m) / idx.compress_ratio;
        if complete > first_block {
            qsa::qsa_block_keys_with_rounding(
                ctx,
                pass,
                idx_keys,
                &w.indexer.k_norm,
                blk_keys,
                idx.compress_ratio,
                first_block,
                complete - first_block,
                rot,
                theta,
                eps,
                Rope::Delta(delta),
                cfg.norm_weight_bias(),
                cfg.official_mlx_rounding(),
            )?;
            pass.level_barrier(&[blk_keys])?;
        }
        // Preserve ordinary dense decode arithmetic and its shared partials.
        // Sparse rows use independent prefill scratch, so all short queries
        // can share the same dependency levels without changing split plans.
        let dense_rows = idx.dense_limit().saturating_sub(pos).min(m);
        for row in 0..dense_rows {
            let at = pos + row;
            let query = ps.q.view(row * nq * hd, &[nq, hd])?;
            let output = ps.attn_o.view(row * nq * hd, &[nq, hd])?;
            sdpa_decode(
                ctx,
                pass,
                &query,
                k_cache,
                v_cache,
                &output,
                at + 1,
                self.attn_scale,
                Some((&s.sdpa_partials, &s.sdpa_stats)),
            )?;
            // The following dense row reuses the ordinary partials.
            pass.level_barrier(&[])?;
        }
        let sparse_rows = m - dense_rows;
        if sparse_rows > 0 {
            let base = pos + dense_rows;
            let nb = qsa::visible_blocks(pos + m - 1, idx.compress_ratio);
            let index_queries = ps.idx_q.view(
                dense_rows * idx.n_heads * INDEXER_D,
                &[sparse_rows, idx.n_heads, INDEXER_D],
            )?;
            let queries = ps.q.view(dense_rows * nq * hd, &[sparse_rows, nq, hd])?;
            let outputs =
                ps.attn_o.view(dense_rows * nq * hd, &[sparse_rows, nq, hd])?;
            qsa::qsa_scores(
                ctx,
                pass,
                &index_queries,
                blk_keys,
                &ps.qsa.scores,
                idx.n_heads,
                nb,
                base,
                idx.compress_ratio,
            )?;
            pass.level_barrier(&[&ps.qsa.scores])?;
            qsa::qsa_select_blocks(
                ctx,
                pass,
                &ps.qsa.scores,
                &ps.qsa.sel,
                &ps.qsa.n_sel,
                sparse_rows,
                nb,
                base,
                idx.compress_ratio,
                idx.block_topk(),
            )?;
            pass.level_barrier(&[&ps.qsa.sel, &ps.qsa.n_sel])?;
            // VerifyHost has at most four rows: this is the ordinary split
            // kernel and split/head-group plan for each query, never tensor
            // SDPA or the large-query route.
            qsa::qsa_attention(
                ctx,
                pass,
                &queries,
                k_cache,
                v_cache,
                &ps.qsa.sel,
                &ps.qsa.n_sel,
                &outputs,
                &ps.qsa.split_scratch(),
                sparse_rows,
                idx.block_topk(),
                idx.compress_ratio,
                base,
                self.attn_scale,
            )?;
            pass.level_barrier(&[&outputs])?;
        }
        sigmoid_mul_bf16(
            ctx,
            pass,
            &ps.gate,
            &ps.attn_o,
            &ps.attn_gated,
            ps.moe.sigmoid_table.as_ref(),
            self.config.official_mlx_rounding(),
        )?;
        pass.level_barrier(&[&ps.attn_gated])?;
        project_mat_decode_exact(ctx, pass, &ps.attn_gated, &w.o_proj, &ps.branch_out)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attn_batched(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &AttnWeights,
        s: &Scratch,
        ps: &PrefillScratch,
        k_cache: &KvCache,
        v_cache: &KvCache,
        idx_keys: &Tensor,
        blk_keys: &Tensor,
        pos: usize,
        rope: Rope<'_>,
    ) -> Result<()> {
        self.attn_batched_theta(
            ctx,
            pass,
            w,
            s,
            ps,
            k_cache,
            v_cache,
            idx_keys,
            blk_keys,
            AttnPos::host(pos),
            self.config.rope_parameters.rope_theta,
            rope,
        )
    }

    /// The attention branch over `ps.hc.mixed` at sequence indices
    /// `pos..pos+m` against the given caches, with an explicit RoPE base (the
    /// draft head declares its own) and the rule turning an index into a
    /// rotary position (`rope`). Cache slots and indexer blocks always take
    /// the sequence index.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attn_batched_theta(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &AttnWeights,
        s: &Scratch,
        ps: &PrefillScratch,
        k_cache: &KvCache,
        v_cache: &KvCache,
        idx_keys: &Tensor,
        blk_keys: &Tensor,
        at: AttnPos<'_>,
        theta: f32,
        rope: Rope<'_>,
    ) -> Result<()> {
        let cfg = &self.config;
        let eps = cfg.rms_norm_eps;
        let rot = cfg.rotary_dim();
        let m = ps.m;
        let idx = &cfg.indexer;
        let (nq, hd) = (cfg.num_attention_heads, cfg.head_dim);
        let pos = at.pos;

        project_stack_or_slices(
            ctx,
            pass,
            &ps.hc.mixed,
            &w.qkv_proj,
            &ps.stack,
            [(&w.q_proj, &ps.qg), (&w.k_proj, &ps.k_new), (&w.v_proj, &ps.v_new)],
            &s.dequant,
        )?;
        project_mat(
            ctx,
            pass,
            &ps.hc.mixed,
            &w.indexer.qk_proj,
            &ps.idx_qk,
            &s.dequant,
        )?;
        pass.level_barrier(&[&ps.qg, &ps.k_new, &ps.v_new, &ps.idx_qk])?;
        split_q_gate(ctx, pass, &ps.qg, &ps.q, &ps.gate)?;
        rmsnorm_bf16(
            ctx,
            pass,
            &ps.k_new,
            &w.k_norm,
            &ps.k_new,
            eps,
            self.config.norm_weight_bias(),
            self.config.official_mlx_rounding(),
        )?;
        // Indexer: queries for this chunk, raw keys into the cache.
        qsa::qsa_prep_q_with_rounding(
            ctx,
            pass,
            &ps.idx_qk,
            &w.indexer.q_norm,
            &ps.idx_q,
            idx.n_heads,
            rot,
            pos,
            theta,
            eps,
            rope,
            self.config.norm_weight_bias(),
            self.config.official_mlx_rounding(),
        )?;
        qsa::qsa_scatter_keys(ctx, pass, &ps.idx_qk, idx_keys, idx.n_heads, pos)?;
        pass.level_barrier(&[&ps.q, &ps.gate, &ps.k_new, &ps.idx_q, idx_keys])?;
        rmsnorm_bf16(
            ctx,
            pass,
            &ps.q,
            &w.q_norm,
            &ps.q,
            eps,
            self.config.norm_weight_bias(),
            self.config.official_mlx_rounding(),
        )?;
        rope_neox(
            ctx,
            pass,
            &ps.k_new,
            cfg.num_key_value_heads,
            rot,
            pos,
            theta,
            rope,
        )?;
        // Block keys for every block this chunk completes.
        let ratio = idx.compress_ratio;
        match (pos.arg, at.block) {
            (Arg::Const(p), _) => {
                let first_block = p / ratio;
                let complete = (p + m) / ratio;
                if complete > first_block {
                    qsa::qsa_block_keys_with_rounding(
                        ctx,
                        pass,
                        idx_keys,
                        &w.indexer.k_norm,
                        blk_keys,
                        ratio,
                        first_block,
                        complete - first_block,
                        rot,
                        theta,
                        eps,
                        rope,
                        self.config.norm_weight_bias(),
                        self.config.official_mlx_rounding(),
                    )?;
                }
            }
            (Arg::Gpu(_), Some((block, count))) => {
                // The GPU decided whether this row completes a block; the
                // dispatch is sized for one block and gated by its count word.
                ensure!(
                    m == 1 && pos.max - pos.min < ratio,
                    "a GPU-supplied position needs a single row within one indexer block"
                );
                let first_min = pos.min / ratio;
                if (pos.max + 1) / ratio > first_min {
                    qsa::qsa_block_keys_with_rounding(
                        ctx,
                        pass,
                        idx_keys,
                        &w.indexer.k_norm,
                        blk_keys,
                        ratio,
                        Pos::gpu(block, first_min, pos.max / ratio),
                        Pos::gpu(count, 0, 1),
                        rot,
                        theta,
                        eps,
                        rope,
                        self.config.norm_weight_bias(),
                        self.config.official_mlx_rounding(),
                    )?;
                }
            }
            (Arg::Gpu(_), None) => {
                anyhow::bail!("a GPU-supplied position needs its block words")
            }
        }
        pass.level_barrier(&[&ps.q, &ps.k_new, blk_keys])?;
        rope_neox(ctx, pass, &ps.q, nq, rot, pos, theta, rope)?;
        scatter_kv(ctx, pass, k_cache, &ps.k_new, pos)?;
        scatter_kv(ctx, pass, v_cache, &ps.v_new, pos)?;
        pass.level_barrier(&[&ps.q, &k_cache.values, &v_cache.values])?;
        // The caches the dense kernel reads for a window of `len` rows: the
        // bf16 caches themselves, or a q8 cache's window staged as bf16 (the
        // tensor-op kernel takes bf16 device tensors only).
        let dense_kv = |len: usize| -> Result<(&Tensor, &Tensor)> {
            match &ps.kv_staging {
                None => Ok((&k_cache.values, &v_cache.values)),
                Some([ks, vs]) => {
                    kv_dequant_q8(ctx, pass, k_cache, ks, len)?;
                    kv_dequant_q8(ctx, pass, v_cache, vs, len)?;
                    pass.level_barrier(&[ks, vs])?;
                    Ok((ks, vs))
                }
            }
        };

        if pos.max + m <= idx.dense_limit() {
            // Every query sees at most the budget: the selection is the whole
            // causal window, so the dense kernel is exact.
            let (k_dense, v_dense) = dense_kv(pos.max + m)?;
            sdpa_prefill(
                ctx,
                pass,
                &ps.q,
                k_dense,
                v_dense,
                &ps.attn_o,
                pos,
                self.attn_scale,
            )?;
        } else {
            // Rows whose causal window still fits the budget select every
            // visible block, so the dense kernel is exact for them: the
            // chunk's prefix up to the dense limit (empty when the chunk
            // starts past it). The rest goes through the indexer.
            let dense_rows = idx.dense_limit().saturating_sub(pos.max).min(m);
            if dense_rows > 0 {
                let q = ps.q.view(0, &[dense_rows, nq, hd])?;
                let out = ps.attn_o.view(0, &[dense_rows, nq, hd])?;
                let (k_dense, v_dense) = dense_kv(pos.max + dense_rows)?;
                sdpa_prefill(
                    ctx,
                    pass,
                    &q,
                    k_dense,
                    v_dense,
                    &out,
                    pos,
                    self.attn_scale,
                )?;
            }
            let qb_cap = ps.qsa.n_sel.numel();
            for q0 in (dense_rows..m).step_by(qb_cap) {
                let qb = qb_cap.min(m - q0);
                let base = pos.offset(q0)?;
                let nb_max = qsa::visible_blocks(base.max + qb - 1, idx.compress_ratio);
                let idx_q = ps.idx_q.view(
                    q0 * idx.n_heads * INDEXER_D,
                    &[qb, idx.n_heads, INDEXER_D],
                )?;
                let q = ps.q.view(q0 * nq * hd, &[qb, nq, hd])?;
                let out = ps.attn_o.view(q0 * nq * hd, &[qb, nq, hd])?;
                qsa::qsa_scores(
                    ctx,
                    pass,
                    &idx_q,
                    blk_keys,
                    &ps.qsa.scores,
                    idx.n_heads,
                    nb_max,
                    base,
                    idx.compress_ratio,
                )?;
                pass.level_barrier(&[&ps.qsa.scores])?;
                qsa::qsa_select_blocks(
                    ctx,
                    pass,
                    &ps.qsa.scores,
                    &ps.qsa.sel,
                    &ps.qsa.n_sel,
                    qb,
                    nb_max,
                    base,
                    idx.compress_ratio,
                    idx.block_topk(),
                )?;
                pass.level_barrier(&[&ps.qsa.sel, &ps.qsa.n_sel])?;
                match ps.qsa.route.for_rows(qb) {
                    SparseAttnRoute::Split => qsa::qsa_attention(
                        ctx,
                        pass,
                        &q,
                        k_cache,
                        v_cache,
                        &ps.qsa.sel,
                        &ps.qsa.n_sel,
                        &out,
                        &ps.qsa.split_scratch(),
                        qb,
                        idx.block_topk(),
                        idx.compress_ratio,
                        base,
                        self.attn_scale,
                    )?,
                    SparseAttnRoute::Query => qsa::qsa_attention_query(
                        ctx,
                        pass,
                        &q,
                        k_cache,
                        v_cache,
                        &ps.qsa.sel,
                        &ps.qsa.n_sel,
                        &out,
                        qb,
                        idx.block_topk(),
                        idx.compress_ratio,
                        base,
                        self.attn_scale,
                    )?,
                }
                // The next sub-batch reuses the score and selection scratch.
                pass.level_barrier(&[&out])?;
            }
        }
        pass.level_barrier(&[&ps.attn_o])?;
        sigmoid_mul_bf16(
            ctx,
            pass,
            &ps.gate,
            &ps.attn_o,
            &ps.attn_gated,
            ps.moe.sigmoid_table.as_ref(),
            self.config.official_mlx_rounding(),
        )?;
        pass.level_barrier(&[&ps.attn_gated])?;
        project_mat(ctx, pass, &ps.attn_gated, &w.o_proj, &ps.branch_out, &s.dequant)
    }

    // --- draft head -----------------------------------------------------------

    /// Runs the draft head over the chunk during prefill so its caches keep up
    /// with the trunk: row `j` pairs the trunk hidden of token `pos+j-1` with
    /// token `pos+j` (the previous chunk's last hidden comes from the state;
    /// at position 0 there is no earlier hidden, so that row is skipped).
    /// Afterwards the state's hidden holds the chunk's last row. Row `j`'s
    /// token is `pos+j`; when that is an image placeholder the head is fed
    /// the image's row instead of the placeholder's embedding (`images`), as
    /// the trunk was.
    ///
    /// Only the head's cache writes are ever read again: the state's hidden
    /// is overwritten with the trunk's right after, and the speculative
    /// step's head work runs its own full block. So the catch-up stops after
    /// them ([`Self::mtp_block_caches`]): no query, attention, output
    /// projection, injects or MoE. The caches come out bit-identical to the
    /// full block's (measured 2 to 4 % less prefill wall time, 2026-10).
    #[allow(clippy::too_many_arguments)]
    fn mtp_catch_up(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        mtp: &MtpWeights,
        mst: &MtpState,
        pos: usize,
        s: &Scratch,
        ps: &PrefillScratch,
        rope: Rope<'_>,
        images: &[ImageEmbeds<'_>],
    ) -> Result<()> {
        let wide = self.config.hc_width();
        let m = ps.m;
        let hidden_in = ps
            .mtp_hidden_in
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("draft head without prefill scratch"))?;
        let (rows, pos0) = if pos > 0 { (m, pos - 1) } else { (m - 1, 0) };
        if rows > 0 {
            if pos > 0 {
                copy_words(ctx, pass, &mst.hidden, &hidden_in.view(0, &[wide])?)?;
                if m > 1 {
                    copy_words(
                        ctx,
                        pass,
                        &ps.hyper.view(0, &[m - 1, wide])?,
                        &hidden_in.view(wide, &[m - 1, wide])?,
                    )?;
                }
            } else {
                copy_words(
                    ctx,
                    pass,
                    &ps.hyper.view(0, &[m - 1, wide])?,
                    &hidden_in.view(0, &[m - 1, wide])?,
                )?;
            }
            let (ids, first_token) = if pos > 0 {
                (ps.ids.view(0, &[m])?, pos)
            } else {
                (ps.ids.view(1, &[m - 1])?, 1)
            };
            let overrides =
                row_overrides(images, first_token, rows, self.config.hidden_size)?;
            pass.level_barrier(&[hidden_in])?;
            let ps_rows = ps.rows(rows)?;
            let hidden = hidden_in.view(0, &[rows, wide])?;
            self.mtp_block_caches(
                ctx, pass, mtp, mst, &hidden, &ids, pos0, s, &ps_rows, rope, &overrides,
            )?;
        }
        // The chunk's last trunk hidden pairs with the next token, whenever
        // it arrives. (Ordered after the reads above by the block's barriers,
        // or trivially when no row ran.)
        copy_words(ctx, pass, &ps.hyper.view((m - 1) * wide, &[wide])?, &mst.hidden)?;
        pass.level_barrier(&[&mst.hidden])
    }

    /// The draft head's block over `rows` inputs: `hidden` (`[rows, G*H]`
    /// trunk or head residuals) and `ids` (`[rows]`, the tokens following
    /// them) become the head's residual in `ps.mtp_hyper`, and the block runs
    /// at head positions `pos0..pos0+rows` against the head's caches, roped
    /// by `rope`; `overrides` are the image rows standing in for placeholder
    /// ids among `ids`. Leaves `ps.mtp_hyper` ordered.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mtp_block(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        mtp: &MtpWeights,
        mst: &MtpState,
        hidden: &Tensor,
        ids: &Tensor,
        pos0: AttnPos<'_>,
        s: &Scratch,
        ps: &PrefillScratch,
        rope: Rope<'_>,
        overrides: &[RowOverride],
    ) -> Result<()> {
        let cfg = &self.config;
        let (h, g) = (cfg.hidden_size, cfg.hc_count);
        let hyper = self.mtp_input(ctx, pass, mtp, hidden, ids, s, ps, overrides)?;
        let theta = self.mtp_theta();

        // The block itself: a trunk-style attention layer with its own RoPE base.
        let Mixer::Attn(w) = &mtp.layer.mixer else {
            anyhow::bail!("draft head block is not attention")
        };
        let LayerState::Attn { k_cache, v_cache, idx_keys, blk_keys } = &mst.layer
        else {
            anyhow::bail!("draft head state is not attention")
        };
        self.hc_read_batched(ctx, pass, &mtp.layer.attn_hc, hyper, s, ps)?;
        self.attn_batched_theta(
            ctx, pass, w, s, ps, k_cache, v_cache, idx_keys, blk_keys, pos0, theta,
            rope,
        )?;
        pass.level_barrier(&[&ps.branch_out])?;
        hc_inject_bf16(
            ctx,
            pass,
            hyper,
            &ps.branch_out,
            &ps.hc.inj,
            h,
            g,
            self.config.official_mlx_rounding(),
            self.hc_primitive_sigmoid(&mtp.layer.attn_hc, ps.m),
            self.hc_injection_kind(&mtp.layer.attn_hc, ps.m),
        )?;
        pass.level_barrier(&[hyper])?;
        self.hc_read_batched(ctx, pass, &mtp.layer.mlp_hc, hyper, s, ps)?;
        prefill_moe(
            ctx,
            pass,
            &moe_dims(cfg),
            &mtp.layer.ffn,
            &PrefillMoeIo {
                x: &ps.hc.mixed,
                out: &ps.branch_out,
                stack: &ps.stack,
                mlp_gate: &ps.mlp_gate,
                mlp_up: &ps.mlp_up,
                mlp_act: &ps.mlp_act,
                dequant: &s.dequant,
            },
            &ps.moe,
        )?;
        pass.level_barrier(&[&ps.branch_out])?;
        hc_inject_bf16(
            ctx,
            pass,
            hyper,
            &ps.branch_out,
            &ps.hc.inj,
            h,
            g,
            self.config.official_mlx_rounding(),
            self.hc_primitive_sigmoid(&mtp.layer.mlp_hc, ps.m),
            self.hc_injection_kind(&mtp.layer.mlp_hc, ps.m),
        )?;
        pass.level_barrier(&[hyper])
    }

    /// The draft head's RoPE base (its own when the config declares one).
    pub(super) fn mtp_theta(&self) -> f32 {
        let cfg = &self.config;
        cfg.mtp.map_or(cfg.rope_parameters.rope_theta, |m| m.rope_theta)
    }

    /// The draft head's input over `ps.m` rows: `hidden` (`[rows, G*H]`)
    /// and `ids` become the residual (per stream `fc_hidden(norm(stream))`
    /// plus the broadcast `fc_embedding(norm(embed(id)))`) in
    /// `ps.mtp_hyper`, which is returned ordered. `overrides` replace
    /// placeholder rows' embeddings.
    #[allow(clippy::too_many_arguments)]
    fn mtp_input<'p>(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        mtp: &MtpWeights,
        hidden: &Tensor,
        ids: &Tensor,
        s: &Scratch,
        ps: &'p PrefillScratch,
        overrides: &[RowOverride],
    ) -> Result<&'p Tensor> {
        let cfg = &self.config;
        let (h, g, eps) = (cfg.hidden_size, cfg.hc_count, cfg.rms_norm_eps);
        let rows = ps.m;
        ensure!(
            hidden.shape() == [rows, g * h] && ids.numel() == rows,
            "draft head input shape mismatch"
        );
        let hyper = ps
            .mtp_hyper
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("draft head without scratch"))?;

        // Per stream: fc_hidden(norm(stream)); shared: fc_embedding(norm(emb)).
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            hidden,
            &mtp.norm_hidden,
            &ps.hc.hn,
            h,
            g,
            eps,
            self.config.norm_weight_bias(),
            self.config.official_mlx_rounding(),
        )?;
        quant::gather_rows_quant(ctx, pass, &self.weights.embed_tokens, ids, &ps.x)?;
        if !overrides.is_empty() {
            pass.level_barrier(&[&ps.x])?;
            override_rows(ctx, pass, &ps.x, overrides, h)?;
        }
        pass.level_barrier(&[&ps.hc.hn, &ps.x])?;
        let hn_streams = ps.hc.hn.view(0, &[rows * g, h])?;
        let hyper_streams = hyper.view(0, &[rows * g, h])?;
        project_mat(
            ctx,
            pass,
            &hn_streams,
            &mtp.fc_hidden,
            &hyper_streams,
            &s.dequant,
        )?;
        rmsnorm_bf16(
            ctx,
            pass,
            &ps.x,
            &mtp.norm_embedding,
            &ps.x,
            eps,
            self.config.norm_weight_bias(),
            self.config.official_mlx_rounding(),
        )?;
        pass.level_barrier(&[hyper, &ps.x])?;
        project_mat(ctx, pass, &ps.x, &mtp.fc_embedding, &ps.branch_out, &s.dequant)?;
        pass.level_barrier(&[&ps.branch_out])?;
        hc_broadcast_bf16(ctx, pass, &ps.branch_out, &ps.hc.up, h, g)?;
        pass.level_barrier(&[&ps.hc.up])?;
        add_bf16(ctx, pass, hyper, &ps.hc.up, hyper)?;
        pass.level_barrier(&[hyper])?;
        Ok(hyper)
    }

    /// The catch-up's block: what [`Self::mtp_block`]
    /// writes into the head's caches at sequence indices `pos0..pos0+ps.m`,
    /// and nothing else. The head's input and its attention read run as in
    /// the full block (they produce the attention layer's input), then
    /// [`Self::attn_cache_writes`]. The residual left in `ps.mtp_hyper` is
    /// the block's input, not its output, so only a catch-up, which never
    /// reads it, may call this.
    #[allow(clippy::too_many_arguments)]
    fn mtp_block_caches(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        mtp: &MtpWeights,
        mst: &MtpState,
        hidden: &Tensor,
        ids: &Tensor,
        pos0: usize,
        s: &Scratch,
        ps: &PrefillScratch,
        rope: Rope<'_>,
        overrides: &[RowOverride],
    ) -> Result<()> {
        let hyper = self.mtp_input(ctx, pass, mtp, hidden, ids, s, ps, overrides)?;
        let Mixer::Attn(w) = &mtp.layer.mixer else {
            anyhow::bail!("draft head block is not attention")
        };
        self.hc_read_batched(ctx, pass, &mtp.layer.attn_hc, hyper, s, ps)?;
        self.attn_cache_writes(ctx, pass, w, s, ps, &mst.layer, pos0, rope)
    }

    /// The cache writes of [`Self::attn_batched_theta`] over `ps.hc.mixed`
    /// at sequence indices `pos..pos+m` and the draft head's RoPE base, with
    /// the same kernels on the same inputs: K and V projected (their slices
    /// of `qkv_proj`; the fused stack is taken only when every slice picks
    /// the stack's skinny variant, so a slice alone reduces exactly as in
    /// the stack), K normed and roped, both scattered into the caches, the
    /// indexer's raw keys scattered and the block keys of every block the
    /// rows complete. No query, attention, output gate or output
    /// projection: `ps.branch_out` is not written. Leaves the caches
    /// ordered.
    #[allow(clippy::too_many_arguments)]
    fn attn_cache_writes(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &AttnWeights,
        s: &Scratch,
        ps: &PrefillScratch,
        layer: &LayerState,
        pos: usize,
        rope: Rope<'_>,
    ) -> Result<()> {
        let cfg = &self.config;
        let eps = cfg.rms_norm_eps;
        let rot = cfg.rotary_dim();
        let m = ps.m;
        let idx = &cfg.indexer;
        let theta = self.mtp_theta();
        let LayerState::Attn { k_cache, v_cache, idx_keys, blk_keys } = layer else {
            anyhow::bail!("draft head state is not attention")
        };

        project_mat(ctx, pass, &ps.hc.mixed, &w.k_proj, &ps.k_new, &s.dequant)?;
        project_mat(ctx, pass, &ps.hc.mixed, &w.v_proj, &ps.v_new, &s.dequant)?;
        project_mat(
            ctx,
            pass,
            &ps.hc.mixed,
            &w.indexer.qk_proj,
            &ps.idx_qk,
            &s.dequant,
        )?;
        pass.level_barrier(&[&ps.k_new, &ps.v_new, &ps.idx_qk])?;
        rmsnorm_bf16(
            ctx,
            pass,
            &ps.k_new,
            &w.k_norm,
            &ps.k_new,
            eps,
            self.config.norm_weight_bias(),
            self.config.official_mlx_rounding(),
        )?;
        qsa::qsa_scatter_keys(ctx, pass, &ps.idx_qk, idx_keys, idx.n_heads, pos)?;
        pass.level_barrier(&[&ps.k_new, idx_keys])?;
        rope_neox(
            ctx,
            pass,
            &ps.k_new,
            cfg.num_key_value_heads,
            rot,
            pos,
            theta,
            rope,
        )?;
        // Block keys for every block this chunk completes.
        let ratio = idx.compress_ratio;
        let first_block = pos / ratio;
        let complete = (pos + m) / ratio;
        if complete > first_block {
            qsa::qsa_block_keys_with_rounding(
                ctx,
                pass,
                idx_keys,
                &w.indexer.k_norm,
                blk_keys,
                ratio,
                first_block,
                complete - first_block,
                rot,
                theta,
                eps,
                rope,
                self.config.norm_weight_bias(),
                self.config.official_mlx_rounding(),
            )?;
        }
        pass.level_barrier(&[&ps.k_new, blk_keys])?;
        scatter_kv(ctx, pass, k_cache, &ps.k_new, pos)?;
        scatter_kv(ctx, pass, v_cache, &ps.v_new, pos)?;
        pass.level_barrier(&[&k_cache.values, &v_cache.values])
    }

    // --- decode -------------------------------------------------------------

    /// Encodes one concurrent-dispatch decode graph at `state.pos`. The
    /// concurrent encoder has no implicit dispatch ordering: `level_barrier`
    /// marks every true inter-level data edge.
    #[allow(clippy::too_many_arguments)]
    fn encode_decode_graph(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        state: &DecodeState,
        s: &Scratch,
        slot_in: usize,
        slot_out: usize,
        draw: Draw<'_>,
        park: Option<u64>,
        capture: Option<DecodeCapture<'_>>,
    ) -> Result<()> {
        ensure!(state.pos < state.capacity, "sequence full ({})", state.capacity);
        let cfg = &self.config;
        let (h, g) = (cfg.hidden_size, cfg.hc_count);
        let conv_slot = state.conv_slot;
        self.set_cache_position(Some(state.pos));

        pass.profile_stage(crate::metal::stage_profile::Phase::Embedding)?;
        let ids = s.next_token.view(slot_in, &[1])?;
        quant::gather_rows_quant(ctx, pass, &self.weights.embed_tokens, &ids, &s.x)?;
        pass.level_barrier(&[&s.x])?;
        hc_broadcast_bf16(ctx, pass, &s.x, &s.hyper, h, g)?;
        pass.level_barrier(&[&s.hyper])?;

        let mut gdn_index = 0;
        for (layer_index, (layer, lstate)) in
            self.weights.layers.iter().zip(state.layers.iter()).enumerate()
        {
            self.trace_stage(ctx, pass, layer_index, "input", &s.hyper, 1)?;
            if let (Some(ple_w), Some(ple_s), Some(pst)) =
                (&layer.ple, &s.ple, &state.ple)
            {
                if let Some(value) = park {
                    // First use of host-staged data: the n-gram rows. The
                    // embedding and the layers above ran while the host
                    // read the token and staged them.
                    s.sync.encode_wait(pass, value)?;
                }
                if !crate::ablate::on(crate::ablate::Group::Ple) {
                    pass.profile_stage(crate::metal::stage_profile::Phase::Ple)?;
                    self.ple_decode(ctx, pass, ple_w, ple_s, pst, conv_slot, s)?;
                    if let Some(capture) = &capture
                        && let Some(out) = &capture.spec.ple_conv_in
                    {
                        let width = ple_s.gated_n.numel();
                        copy_words(
                            ctx,
                            pass,
                            &ple_s.gated_n,
                            &out.view(capture.row * width, &[width])?,
                        )?;
                        pass.level_barrier(&[out])?;
                    }
                }
            }

            self.trace_stage(ctx, pass, layer_index, "post_ple", &s.hyper, 1)?;

            pass.profile_stage(crate::metal::stage_profile::Phase::Hc)?;
            self.hc_read_decode(ctx, pass, &layer.attn_hc, s)?;
            self.trace_stage(ctx, pass, layer_index, "mixer_input", &s.hc.mixed, 1)?;
            match (&layer.mixer, lstate) {
                (Mixer::Gdn(w), LayerState::Gdn { state, conv_windows, layout }) => {
                    self.gdn_decode_branch(
                        ctx,
                        pass,
                        w,
                        s,
                        state,
                        *layout,
                        &conv_windows[conv_slot],
                        layer_index,
                    )?;
                    if let Some(capture) = &capture {
                        let raw = &capture.spec.conv_in[gdn_index];
                        let width = s.qkv.numel();
                        copy_words(
                            ctx,
                            pass,
                            &s.qkv,
                            &raw.view(capture.row * width, &[width])?,
                        )?;
                        if capture.row + 1 < capture.rows {
                            let mid = &capture.spec.mid[gdn_index];
                            copy_words(
                                ctx,
                                pass,
                                state,
                                &mid.view(capture.row * state.numel(), state.shape())?,
                            )?;
                            pass.level_barrier(&[mid])?;
                        }
                        pass.level_barrier(&[raw])?;
                    }
                    gdn_index += 1;
                }
                (
                    Mixer::Attn(w),
                    LayerState::Attn { k_cache, v_cache, idx_keys, blk_keys },
                ) => {
                    pass.profile_stage(crate::metal::stage_profile::Phase::Qsa)?;
                    self.attn_decode_branch(
                        ctx,
                        pass,
                        w,
                        s,
                        k_cache,
                        v_cache,
                        idx_keys,
                        blk_keys,
                        state.pos,
                        state.rope_delta,
                    )?;
                }
                _ => anyhow::bail!("layer/state kind mismatch"),
            }
            pass.level_barrier(&[&s.branch_out])?;
            self.trace_stage(ctx, pass, layer_index, "mixer_output", &s.branch_out, 1)?;
            let pair_write =
                self.hc_write_pair_selected(&layer.attn_hc, &layer.mlp_hc, &s.hc);
            if !crate::ablate::on(crate::ablate::Group::Inject) && !pair_write {
                pass.profile_stage(crate::metal::stage_profile::Phase::Hc)?;
                hc_inject_bf16(
                    ctx,
                    pass,
                    &s.hyper,
                    &s.branch_out,
                    &s.hc.inj,
                    h,
                    g,
                    self.config.official_mlx_rounding(),
                    self.hc_primitive_sigmoid(&layer.attn_hc, 1),
                    self.hc_injection_kind(&layer.attn_hc, 1),
                )?;
                pass.level_barrier(&[&s.hyper])?;
            }

            pass.profile_stage(crate::metal::stage_profile::Phase::Hc)?;
            if pair_write {
                let prepared = layer
                    .mlp_hc
                    .official_read
                    .as_ref()
                    .expect("selected pair descriptor");
                let parts = s.hc.stock_parts.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "stock HC write pair selected without parts scratch"
                    )
                })?;
                let written =
                    s.hc.stock_write_hyper.as_ref().expect("selected pair scratch");
                prepared.read_after_write(
                    ctx,
                    pass,
                    &s.hyper,
                    &layer.mlp_hc.norm,
                    &s.hc.hn,
                    parts,
                    &s.hc.mixed,
                    &s.hc.inj,
                    written,
                    PendingHcWrite {
                        branch: &s.branch_out,
                        gate: &s.hc.inj,
                        kind: self.hc_injection_kind(&layer.attn_hc, 1),
                    },
                )?;
                pass.level_barrier(&[&s.hc.mixed, &s.hc.inj, written])?;
            } else {
                self.hc_read_decode(ctx, pass, &layer.mlp_hc, s)?;
            }
            // The official write rounds its branch product before adding
            // the residual; the existing fused MoE write keeps it in F32.
            let fuse_injection = !crate::ablate::on(crate::ablate::Group::Inject)
                && !cfg.official_mlx_rounding();
            self.trace_stage(ctx, pass, layer_index, "moe_input", &s.hc.mixed, 1)?;
            pass.profile_stage(crate::metal::stage_profile::Phase::Moe)?;
            let injected = decode_moe(
                ctx,
                pass,
                &moe_dims(cfg),
                &layer.ffn,
                &DecodeMoeIo {
                    x: &s.hc.mixed,
                    out: &s.branch_out,
                    mlp_gu: &s.mlp_gu,
                    mlp_gate: &s.mlp_gate,
                    mlp_up: &s.mlp_up,
                    mlp_act: &s.mlp_act,
                    hc_injection: fuse_injection.then_some(
                        crate::kernels::moe::HcInjection {
                            hyper: &s.hyper,
                            logits: &s.hc.inj,
                            groups: g,
                        },
                    ),
                },
                &s.moe,
            )?;
            ensure!(
                !pair_write || !injected,
                "stock HC write pair requires the separate official MoE write"
            );
            if injected {
                pass.level_barrier(&[&s.branch_out, &s.hyper])?;
            } else {
                pass.level_barrier(&[&s.branch_out])?;
                if !crate::ablate::on(crate::ablate::Group::Inject) {
                    pass.profile_stage(crate::metal::stage_profile::Phase::Hc)?;
                    if pair_write {
                        // Complete this pair in canonical hyper storage. No
                        // pending branch/gate survives into PLE, MTP or a layer.
                        hc_write_gate_to(
                            ctx,
                            pass,
                            s.hc.stock_write_hyper
                                .as_ref()
                                .expect("selected pair scratch"),
                            &s.hyper,
                            self.config.official_mlx_rounding(),
                            PendingHcWrite {
                                branch: &s.branch_out,
                                gate: &s.hc.inj,
                                kind: self.hc_injection_kind(&layer.mlp_hc, 1),
                            },
                        )?;
                    } else {
                        hc_inject_bf16(
                            ctx,
                            pass,
                            &s.hyper,
                            &s.branch_out,
                            &s.hc.inj,
                            h,
                            g,
                            self.config.official_mlx_rounding(),
                            self.hc_primitive_sigmoid(&layer.mlp_hc, 1),
                            self.hc_injection_kind(&layer.mlp_hc, 1),
                        )?;
                    }
                    pass.level_barrier(&[&s.hyper])?;
                }
            }
            self.trace_stage(ctx, pass, layer_index, "moe_output", &s.branch_out, 1)?;
            self.trace_stage(ctx, pass, layer_index, "output", &s.hyper, 1)?;
        }

        pass.profile_stage(crate::metal::stage_profile::Phase::Hc)?;
        self.hc_read_decode(ctx, pass, &self.weights.final_mixer, s)?;
        self.trace_stage(
            ctx,
            pass,
            cfg.num_hidden_layers,
            "head_input",
            &s.hc.mixed,
            1,
        )?;
        if !crate::ablate::on(crate::ablate::Group::Head) {
            pass.profile_stage(crate::metal::stage_profile::Phase::Head)?;
            quant::gemv_quant_logits(
                ctx,
                pass,
                &self.weights.lm_head,
                &s.hc.mixed,
                &s.logits,
                self.config.official_mlx_rounding(),
            )?;
            pass.level_barrier(&[&s.logits])?;
        }
        self.trace_stage(ctx, pass, cfg.num_hidden_layers, "logits", &s.logits, 1)?;
        if let Some(capture) = &capture {
            let wide = cfg.hc_width();
            copy_words(
                ctx,
                pass,
                &s.hyper,
                &capture.hyper.view(capture.row * wide, &[wide])?,
            )?;
            copy_words(
                ctx,
                pass,
                &s.logits,
                &capture
                    .spec
                    .logits
                    .view(capture.row * cfg.vocab_size, &[cfg.vocab_size])?,
            )?;
            pass.level_barrier(&[capture.hyper, &capture.spec.logits])?;
        }
        pass.profile_stage(crate::metal::stage_profile::Phase::Draw)?;
        let out = s.next_token.view(slot_out, &[1])?;
        sample_f32(ctx, pass, &s.logits, &s.sampler, draw.params, draw.step, &out)?;
        pass.level_barrier(&[&s.next_token])?;
        pass.finish_stage_profile()?;
        s.sync.signal_done(pass)
    }

    /// Single-row hyper-connection read: `s.hyper` → `s.hc.mixed` (+ `inj`).
    /// Two fused dispatches: one-row norm + down/inject projection + SiLU,
    /// then the original up-projection dot/mix body with cached activations.
    /// The activation is computed once and reused by every output column.
    /// The down half keeps the normalized stream in f32 where the unfused
    /// large-batch path rounds it to bf16. Leaves `mixed`/`inj` ordered for
    /// the caller.
    fn hc_read_decode(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        hc: &HcWeights,
        s: &Scratch,
    ) -> Result<()> {
        if crate::ablate::on(crate::ablate::Group::Hc) {
            return Ok(());
        }
        if self.config.official_mlx_rounding() {
            // Materialize the reference's two BF16 normalization stages;
            // do not enter the unrounded cached HC projection path.
            let hyper = s.hyper.view(0, &[1, self.config.hc_width()])?;
            return self.hc_read_rows(ctx, pass, hc, &hyper, &s.hc, 1, &s.dequant);
        }
        self.hc_read_one(ctx, pass, hc, &s.hyper, &s.hc)
    }

    /// [`Self::hc_read_decode`] of the one-row `hyper` into `out` (the
    /// prediction log runs a later layer's read into its own scratch).
    fn hc_read_one(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        hc: &HcWeights,
        hyper: &Tensor,
        out: &HcScratch,
    ) -> Result<()> {
        let cfg = &self.config;
        let (h, g) = (cfg.hidden_size, cfg.hc_count);
        let hyper_rows = hyper.view(0, &[1, g * h])?;
        hc_read_down_q8_rows(
            ctx,
            pass,
            &hyper_rows,
            &hc.norm,
            &hc.down,
            hc.inject.as_ref(),
            &out.down,
            &out.inj,
            &out.inv_rms,
            &out.act,
            h,
            g,
            cfg.rms_norm_eps,
            self.config.norm_weight_bias(),
        )?;
        pass.level_barrier(&[&out.down, &out.inj, &out.inv_rms, &out.act])?;
        hc_read_up_mix_q8_cached(
            ctx,
            pass,
            &hc.up,
            &out.act,
            hyper,
            &hc.norm,
            &out.inv_rms,
            &out.mixed,
            h,
            g,
            self.config.norm_weight_bias(),
        )?;
        pass.level_barrier(&[&out.mixed])
    }

    #[allow(clippy::too_many_arguments)]
    fn ple_decode(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &PleWeights,
        p: &PleScratch,
        pst: &PleState,
        conv_slot: usize,
        s: &Scratch,
    ) -> Result<()> {
        let cfg = &self.config;
        let ple = cfg.ple.as_ref().expect("PLE weights without PLE config");
        let (h, g, eps) = (cfg.hidden_size, cfg.hc_count, cfg.rms_norm_eps);
        self.gather_ngram(ctx, pass, w, p, 1)?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &s.hyper,
            &w.norm_query,
            &p.query_n,
            h,
            g,
            eps,
            self.config.norm_weight_bias(),
            self.config.official_mlx_rounding(),
        )?;
        pass.level_barrier(&[&p.emb, &p.query_n])?;
        quant::gemv_quant(ctx, pass, &w.key_proj, &p.emb, &p.key)?;
        quant::gemv_quant(ctx, pass, &w.value_proj, &p.emb, &p.value)?;
        pass.level_barrier(&[&p.key, &p.value])?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &p.key,
            &w.norm_key,
            &p.key_n,
            h,
            g,
            eps,
            self.config.norm_weight_bias(),
            self.config.official_mlx_rounding(),
        )?;
        pass.level_barrier(&[&p.key_n])?;
        ple::ple_gate_value_bf16_with_rounding(
            ctx,
            pass,
            &p.key_n,
            &p.query_n,
            &p.value,
            &p.gated,
            h,
            g,
            self.config.official_mlx_rounding(),
        )?;
        pass.level_barrier(&[&p.gated])?;
        rmsnorm_grouped_bf16(
            ctx,
            pass,
            &p.gated,
            &w.norm_conv,
            &p.gated_n,
            h,
            g,
            eps,
            self.config.norm_weight_bias(),
            self.config.official_mlx_rounding(),
        )?;
        pass.level_barrier(&[&p.gated_n])?;
        ple::ple_conv1d_step_with_rounding(
            ctx,
            pass,
            &pst.conv_windows[conv_slot],
            &p.gated_n,
            &w.conv_w,
            &p.gated,
            &s.hyper,
            ple.ngram_size,
            self.config.official_mlx_rounding(),
        )?;
        pass.level_barrier(&[&s.hyper, &pst.conv_windows[conv_slot]])
    }

    fn gdn_decode_branch(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &GdnWeights,
        s: &Scratch,
        gdn_state: &Tensor,
        gdn_layout: GdnStateLayout,
        conv_window: &Tensor,
        layer_index: usize,
    ) -> Result<()> {
        let fused_conv = !self.config.official_mlx_rounding()
            && !crate::ablate::on(crate::ablate::Group::GdnProj)
            && !crate::ablate::on(crate::ablate::Group::GdnStep)
            && w.in_proj.bits == 4
            && self.config.linear_conv_kernel_dim == 4
            && w.in_proj.out_features().is_multiple_of(2);
        // The stacked qkv/z/a/b projection is one dependency group.
        pass.profile_stage(if fused_conv {
            crate::metal::stage_profile::Phase::GdnInputProjectionConv
        } else {
            crate::metal::stage_profile::Phase::GdnInputProjections
        })?;
        if !crate::ablate::on(crate::ablate::Group::GdnProj) {
            if fused_conv {
                quant::gemv_q4_gdn_conv_step(
                    ctx,
                    pass,
                    &w.in_proj,
                    &s.hc.mixed,
                    &s.gdn_in,
                    conv_window,
                    &w.conv_w,
                    &s.qkv_conv,
                )?;
                pass.level_barrier(&[&s.gdn_in, &s.qkv_conv, conv_window])?;
            } else {
                crate::kernels::official_q8_projection::gemv_gdn_decode(
                    ctx,
                    pass,
                    &w.in_proj,
                    &s.hc.mixed,
                    &s.gdn_in,
                    self.config.official_mlx_rounding(),
                )?;
                pass.level_barrier(&[&s.gdn_in])?;
            }
        }
        if crate::ablate::on(crate::ablate::Group::GdnStep) {
            return Ok(());
        }
        if !fused_conv {
            pass.profile_stage(crate::metal::stage_profile::Phase::GdnConv)?;
            conv1d_step(
                ctx,
                pass,
                conv_window,
                &s.qkv,
                &w.conv_w,
                &s.qkv_conv,
                self.config.official_mlx_rounding(),
                gdn_layout,
            )?;
            pass.level_barrier(&[&s.qkv_conv, conv_window])?;
        }
        self.trace_stage(ctx, pass, layer_index, "gdn_qkv", &s.qkv, 1)?;
        self.trace_stage(ctx, pass, layer_index, "gdn_z", &s.z, 1)?;
        self.trace_stage(ctx, pass, layer_index, "gdn_a", &s.a, 1)?;
        self.trace_stage(ctx, pass, layer_index, "gdn_b", &s.b, 1)?;
        self.trace_stage(ctx, pass, layer_index, "gdn_conv", &s.qkv_conv, 1)?;
        pass.profile_stage(if gdn_layout == GdnStateLayout::ValueKey {
            crate::metal::stage_profile::Phase::GdnRecurrence
        } else {
            crate::metal::stage_profile::Phase::GdnRecurrenceNorm
        })?;
        gdn_step_gated_fused(
            ctx,
            pass,
            &s.qkv_conv,
            &s.a,
            &s.b,
            &w.a_log,
            &w.dt_bias,
            gdn_state,
            &s.z,
            &w.norm_w,
            &s.gdn_gated,
            self.gdn_scale,
            self.config.linear_num_key_heads,
            self.config.rms_norm_eps,
            self.gdn_gate,
            self.config.official_mlx_rounding(),
            gdn_layout,
            &s.gdn_raw,
        )?;
        pass.level_barrier(&[&s.gdn_gated, gdn_state])?;
        self.trace_stage(ctx, pass, layer_index, "gdn_gated", &s.gdn_gated, 1)?;
        if crate::ablate::on(crate::ablate::Group::GdnProj) {
            return Ok(());
        }
        pass.profile_stage(crate::metal::stage_profile::Phase::GdnOutProjection)?;
        crate::kernels::official_q8_projection::gemv_gdn_decode(
            ctx,
            pass,
            &w.out_proj,
            &s.gdn_gated,
            &s.branch_out,
            self.config.official_mlx_rounding(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_decode_branch(
        &self,
        ctx: &MetalContext,
        pass: &ComputePass<'_>,
        w: &AttnWeights,
        s: &Scratch,
        k_cache: &KvCache,
        v_cache: &KvCache,
        idx_keys: &Tensor,
        blk_keys: &Tensor,
        pos: usize,
        rope_delta: i64,
    ) -> Result<()> {
        let cfg = &self.config;
        let eps = cfg.rms_norm_eps;
        let rot = cfg.rotary_dim();
        let theta = cfg.rope_parameters.rope_theta;
        let idx = &cfg.indexer;
        let len = pos + 1;
        // The token sits at cache slot `pos` and rotates at `pos + delta`
        // (equal for text; VISION.md `rope_deltas` after an image).
        ensure!(
            pos as i64 + rope_delta >= 0,
            "rotary position {} + {rope_delta} is negative",
            pos
        );

        if crate::ablate::on(crate::ablate::Group::Attn) {
            return Ok(());
        }
        quant::gemv_quant(ctx, pass, &w.qkv_proj, &s.hc.mixed, &s.attn_qkv)?;
        quant::gemv_quant(ctx, pass, &w.indexer.qk_proj, &s.hc.mixed, &s.idx_qk)?;
        pass.level_barrier(&[&s.attn_qkv, &s.idx_qk])?;
        q_norm_rope_split_decode_with_bias(
            ctx,
            pass,
            &s.qg,
            &w.q_norm,
            &s.q,
            &s.gate,
            rot,
            pos,
            theta,
            eps,
            rope_delta,
            self.config.norm_weight_bias(),
            self.config.official_mlx_rounding(),
        )?;
        k_norm_rope_scatter_decode_with_bias(
            ctx,
            pass,
            &s.k_new,
            &w.k_norm,
            k_cache,
            rot,
            pos,
            theta,
            eps,
            rope_delta,
            self.config.norm_weight_bias(),
            self.config.official_mlx_rounding(),
        )?;
        scatter_kv(ctx, pass, v_cache, &s.v_new, pos)?;
        qsa::qsa_prep_q_with_rounding(
            ctx,
            pass,
            &s.idx_qk,
            &w.indexer.q_norm,
            &s.idx_q,
            idx.n_heads,
            rot,
            pos,
            theta,
            eps,
            Rope::Delta(rope_delta),
            self.config.norm_weight_bias(),
            self.config.official_mlx_rounding(),
        )?;
        qsa::qsa_scatter_keys(ctx, pass, &s.idx_qk, idx_keys, idx.n_heads, pos)?;
        pass.level_barrier(&[
            &s.q,
            &s.gate,
            &k_cache.values,
            &v_cache.values,
            &s.idx_q,
            idx_keys,
        ])?;
        if len.is_multiple_of(idx.compress_ratio) {
            // This token completes a block: its key becomes selectable from
            // the next position on (and this one, beyond the dense limit).
            let block = len / idx.compress_ratio - 1;
            qsa::qsa_block_keys_with_rounding(
                ctx,
                pass,
                idx_keys,
                &w.indexer.k_norm,
                blk_keys,
                idx.compress_ratio,
                block,
                1,
                rot,
                theta,
                eps,
                Rope::Delta(rope_delta),
                self.config.norm_weight_bias(),
                self.config.official_mlx_rounding(),
            )?;
            pass.level_barrier(&[blk_keys])?;
        }

        if len <= idx.dense_limit() {
            sdpa_decode(
                ctx,
                pass,
                &s.q,
                k_cache,
                v_cache,
                &s.attn_o,
                len,
                self.attn_scale,
                Some((&s.sdpa_partials, &s.sdpa_stats)),
            )?;
        } else {
            let nb = qsa::visible_blocks(pos, idx.compress_ratio);
            qsa::qsa_scores(
                ctx,
                pass,
                &s.idx_q,
                blk_keys,
                &s.qsa.scores,
                idx.n_heads,
                nb,
                pos,
                idx.compress_ratio,
            )?;
            pass.level_barrier(&[&s.qsa.scores])?;
            qsa::qsa_select_blocks(
                ctx,
                pass,
                &s.qsa.scores,
                &s.qsa.sel,
                &s.qsa.n_sel,
                1,
                nb,
                pos,
                idx.compress_ratio,
                idx.block_topk(),
            )?;
            pass.level_barrier(&[&s.qsa.sel, &s.qsa.n_sel])?;
            qsa::qsa_attention(
                ctx,
                pass,
                &s.q,
                k_cache,
                v_cache,
                &s.qsa.sel,
                &s.qsa.n_sel,
                &s.attn_o,
                &s.qsa.split_scratch(),
                1,
                idx.block_topk(),
                idx.compress_ratio,
                pos,
                self.attn_scale,
            )?;
        }
        pass.level_barrier(&[&s.attn_o])?;
        sigmoid_mul_bf16(
            ctx,
            pass,
            &s.gate,
            &s.attn_o,
            &s.attn_gated,
            s.moe.sigmoid_table.as_ref(),
            self.config.official_mlx_rounding(),
        )?;
        pass.level_barrier(&[&s.attn_gated])?;
        quant::gemv_quant(ctx, pass, &w.o_proj, &s.attn_gated, &s.branch_out)
    }
}

/// GPU bytes one session holds with caches for `capacity` tokens and
/// `checkpoints` recurrent snapshots, the draft head's cache and hidden
/// included when `mtp`: what `DecodeState::bytes` and `Snapshot::bytes` add
/// up to, from the config alone, so that the expert cache's plan can reserve
/// a session before any state exists. `warm_up` in the server compares it
/// with a real state and snapshot of the loaded model.
pub(crate) fn session_bytes(
    cfg: &Qwen4ExpConfig,
    mtp: bool,
    kv_format: KvFormat,
    capacity: usize,
    checkpoints: usize,
) -> Result<u64> {
    use super::config::LayerType;
    let capacity = round_capacity(capacity)?;
    let count =
        |kind: LayerType| cfg.layer_types.iter().filter(|t| **t == kind).count();
    // `DecodeState::cache_bytes`: every attention layer, the draft head's too.
    let attn_layers = count(LayerType::FullAttention) + usize::from(mtp);
    let per_token =
        2 * cfg.num_key_value_heads * kv_format.row_bytes(cfg.head_dim) + INDEXER_D * 2;
    let blocks = block_key_rows(capacity, cfg.indexer.compress_ratio) * INDEXER_D * 2;
    let caches = attn_layers * (capacity * per_token + blocks);
    // `Qwen4ExpModel::new_state`: per GDN layer the state and two conv
    // windows (a snapshot keeps one), the PLE's two windows (one), the draft
    // head's hidden.
    let state = cfg.linear_num_value_heads
        * GDN_HEAD_DIM
        * GDN_HEAD_DIM
        * GDN_STATE_DTYPE.size();
    let conv =
        cfg.gdn_conv_channels() * (cfg.linear_conv_kernel_dim - 1) * DType::BF16.size();
    let ple = cfg
        .ple
        .as_ref()
        .map_or(0, |p| cfg.hc_width() * p.conv_state_len() * DType::BF16.size());
    let hidden = if mtp { cfg.hc_width() * DType::BF16.size() } else { 0 };
    let gdn_layers = count(LayerType::LinearAttention);
    let live = caches + gdn_layers * (state + 2 * conv) + 2 * ple + hidden;
    let snapshot = gdn_layers * (state + conv) + ple + hidden;
    Ok((live + checkpoints * snapshot) as u64)
}

/// `LILY_KV_CACHE` (`bf16` or `q8`), which overrides the K/V format of
/// every load (`--kv-cache`, `LoadOptions::kv_format`, the inherent
/// loaders' default): measurement and tests run either format through the
/// same binaries.
fn kv_format_from_env() -> Result<Option<KvFormat>> {
    Ok(None)
}

/// Rounds a requested capacity up to the growth step, within the kernel limit.
fn round_capacity(tokens: usize) -> Result<usize> {
    ensure!(tokens <= MAX_SEQ, "capacity {tokens} exceeds kernel limit {MAX_SEQ}");
    Ok(tokens.max(1).div_ceil(CAPACITY_STEP).saturating_mul(CAPACITY_STEP).min(MAX_SEQ))
}

impl DecodeStateApi for DecodeState {
    fn set_rope_delta(&mut self, delta: i64) -> Result<()> {
        self.rope_delta = delta;
        Ok(())
    }

    type Snapshot = Snapshot;

    fn pos(&self) -> usize {
        self.pos
    }

    fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    fn reset(&mut self) -> Result<()> {
        DecodeState::reset(self)
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn ensure_capacity(&mut self, ctx: &MetalContext, tokens: usize) -> Result<()> {
        if tokens <= self.capacity {
            return Ok(());
        }
        let capacity = round_capacity(tokens)?;
        let mut layers = std::mem::take(&mut self.layers);
        for lstate in &mut layers {
            self.grow_attn(ctx, lstate, capacity)?;
        }
        self.layers = layers;
        if let Some(mut mtp) = self.mtp.take() {
            self.grow_attn(ctx, &mut mtp.layer, capacity)?;
            self.mtp = Some(mtp);
        }
        self.capacity = capacity;
        Ok(())
    }

    fn bytes(&self) -> usize {
        let recurrent: usize = self
            .layers
            .iter()
            .map(|l| match l {
                LayerState::Gdn { state, conv_windows, .. } => {
                    state.byte_len()
                        + conv_windows[0].byte_len()
                        + conv_windows[1].byte_len()
                }
                LayerState::Attn { .. } => 0,
            })
            .sum();
        let ple = self.ple.as_ref().map_or(0, |p| 2 * p.conv_windows[0].byte_len());
        let mtp = self.mtp.as_ref().map_or(0, |m| m.hidden.byte_len());
        recurrent + ple + mtp + self.cache_bytes(self.capacity)
    }

    fn snapshot(&self, ctx: &MetalContext) -> Result<Snapshot> {
        // Mid-step the recurrent state holds rows the draft pass may still
        // roll back, and the position counts them.
        ensure!(self.spec.is_none(), "snapshot while a speculative step is pending");
        let slot = self.conv_slot;
        let mut sources = Vec::new();
        for lstate in &self.layers {
            if let LayerState::Gdn { state, conv_windows, .. } = lstate {
                sources.push(state);
                sources.push(&conv_windows[slot]);
            }
        }
        sources.extend(self.ple.as_ref().map(|p| &p.conv_windows[slot]));
        sources.extend(self.mtp.as_ref().map(|m| &m.hidden));
        let mut copies = clone_tensors(ctx, &sources)?.into_iter();
        let mut next = || copies.next().expect("one copy per source");
        let mut gdn = Vec::new();
        for lstate in &self.layers {
            if let LayerState::Gdn { layout, .. } = lstate {
                gdn.push((next(), next(), *layout));
            }
        }
        let ple = self.ple.as_ref().map(|p| (p.hist, next()));
        let mtp_hidden = self.mtp.as_ref().map(|_| next());
        Ok(Snapshot { pos: self.pos, gdn, ple, mtp_hidden })
    }

    fn restore(&mut self, ctx: &MetalContext, snapshot: &Snapshot) -> Result<()> {
        ensure!(
            snapshot.pos <= self.capacity,
            "snapshot position {} exceeds capacity {}",
            snapshot.pos,
            self.capacity
        );
        let mut copies = Vec::new();
        let mut saved = snapshot.gdn.iter();
        for lstate in &self.layers {
            if let LayerState::Gdn { state, conv_windows, layout } = lstate {
                let (s, w, saved_layout) = saved.next().ok_or_else(|| {
                    anyhow::anyhow!("snapshot has too few GDN layers")
                })?;
                ensure!(
                    layout == saved_layout,
                    "snapshot GDN physical state layout mismatch"
                );
                copies.push(BlitCopy {
                    src: s,
                    src_offset: 0,
                    dst: state,
                    dst_offset: 0,
                    len: state.byte_len(),
                });
                copies.push(BlitCopy {
                    src: w,
                    src_offset: 0,
                    dst: &conv_windows[0],
                    dst_offset: 0,
                    len: w.byte_len(),
                });
            }
        }
        ensure!(saved.next().is_none(), "snapshot has too many GDN layers");
        match (&mut self.ple, &snapshot.ple) {
            (Some(p), Some((hist, w))) => {
                p.hist = *hist;
                copies.push(BlitCopy {
                    src: w,
                    src_offset: 0,
                    dst: &p.conv_windows[0],
                    dst_offset: 0,
                    len: w.byte_len(),
                });
            }
            (None, None) => {}
            _ => anyhow::bail!("snapshot PLE state mismatch"),
        }
        match (&self.mtp, &snapshot.mtp_hidden) {
            (Some(m), Some(h)) => {
                copies.push(BlitCopy {
                    src: h,
                    src_offset: 0,
                    dst: &m.hidden,
                    dst_offset: 0,
                    len: h.byte_len(),
                });
            }
            (None, None) => {}
            _ => anyhow::bail!("snapshot draft-head state mismatch"),
        }
        ctx.blit_copy(&copies)?;
        self.pos = snapshot.pos;
        self.conv_slot = 0;
        self.spec = None;
        Ok(())
    }

    fn copy_prefix_from(
        &mut self,
        ctx: &MetalContext,
        from: &Self,
        tokens: usize,
    ) -> Result<()> {
        ensure!(
            tokens <= from.pos,
            "source state has fed {} tokens, {tokens} requested",
            from.pos
        );
        self.ensure_capacity(ctx, tokens)?;
        let mut copies = Vec::new();
        for (dst, src) in self.layers.iter().zip(&from.layers) {
            attn_prefix_copies(dst, src, tokens, self.ratio, &mut copies)?;
        }
        match (&self.mtp, &from.mtp) {
            (Some(dst), Some(src)) => attn_prefix_copies(
                &dst.layer,
                &src.layer,
                tokens,
                self.ratio,
                &mut copies,
            )?,
            (None, None) => {}
            _ => anyhow::bail!("draft-head state mismatch between sessions"),
        }
        ctx.blit_copy(&copies)
    }
}

impl ScratchApi for Scratch {
    fn next_token(&self) -> &Tensor {
        &self.next_token
    }

    fn arrival_probe(&self) -> Option<&SharedEvent> {
        Scratch::arrival_probe(self)
    }

    fn resumed_probe(&self) -> Option<&SharedEvent> {
        Scratch::resumed_probe(self)
    }

    fn logits(&self) -> &Tensor {
        &self.logits
    }

    fn begin_request(&self) {
        self.sampler.reset_counts();
    }

    /// The block selection of the last sparse-attention decode step (query
    /// row 0): `null` until a step ran past the dense limit.
    fn debug_json(&self) -> Result<serde_json::Value> {
        Ok(serde_json::Value::Null)
    }
}

impl LanguageModel for Qwen4ExpModel {
    type State = DecodeState;
    type Scratch = Scratch;

    const MODEL_ID: &'static str = "Qwen3.8-Flash-Next";

    fn expert_cache_stats(&self) -> Option<ExpertCacheStats> {
        None
    }

    fn expert_cache_slots(&self) -> Option<usize> {
        None
    }

    fn prefill_chunk_tokens(&self) -> Option<usize> {
        Some(self.prefill_chunk)
    }

    fn ngram_table_bytes(&self) -> Option<u64> {
        Some(match self.ple_table()? {
            NgramTable::Paged(table) => table.bytes(),
            NgramTable::Resident(t) => {
                (t.codes.byte_len() + t.scales.byte_len() + t.biases.byte_len()) as u64
            }
        })
    }

    fn load(ctx: &MetalContext, dir: &Path, options: &LoadOptions) -> Result<Self> {
        let expert_slots = options.expert_slots;
        let expert_usage = options.expert_usage.clone();
        let machine_memory = None;
        let expert_usage_out = options.expert_usage_out.clone();
        let kv_format = options.kv_format;
        let session_context = options.session_context;
        Qwen4ExpModel::load_with(
            ctx,
            dir,
            options.ngram_storage,
            options.mtp_drafts > 0,
            options.vision,
            expert_slots,
            expert_usage,
            machine_memory,
            options.memory_limit,
            expert_usage_out,
            session_context,
        )
        .map(|mut model| {
            model.set_kv_format(kv_format);
            model
        })
    }

    fn weight_buffers(&self) -> Vec<crate::metal::Buffer> {
        self.weights.buffers.clone()
    }

    fn planned_memory(&self) -> Option<u64> {
        self.weights.planned_memory
    }

    fn memory_limit(&self) -> Option<u64> {
        self.weights.memory_limit
    }

    fn session_reserve(&self) -> Option<u64> {
        self.weights.session_reserve
    }

    fn session_bytes(&self, capacity: usize, checkpoints: usize) -> Option<u64> {
        session_bytes(
            &self.config,
            self.weights.mtp.is_some(),
            self.kv_format,
            capacity,
            checkpoints,
        )
        .ok()
    }

    fn vision_tower(&self) -> Option<VisionTower> {
        Some(match (&self.config.vision, &self.weights.vision) {
            (None, _) => VisionTower::Absent,
            (Some(_), None) => VisionTower::Off,
            (Some(v), Some(w)) => {
                VisionTower::Loaded { bytes: w.bytes(), blocks: v.depth }
            }
        })
    }

    fn max_position_embeddings(&self) -> usize {
        self.config.max_position_embeddings
    }

    fn eos_token_ids(&self) -> Vec<u32> {
        self.config.eos_token_id.as_vec()
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn kv_format(&self) -> KvFormat {
        self.kv_format
    }

    fn bytes_per_token(&self) -> usize {
        let cfg = &self.config;
        let attn_layers = cfg
            .layer_types
            .iter()
            .filter(|t| matches!(t, super::config::LayerType::FullAttention))
            .count()
            + usize::from(self.weights.mtp.is_some());
        attn_layers
            * (2 * cfg.num_key_value_heads * self.kv_format.row_bytes(cfg.head_dim)
                + INDEXER_D * 2
                + INDEXER_D * 2 / cfg.indexer.compress_ratio)
    }

    fn warm_storage(&self, lock: bool) -> Result<u64> {
        match self.ple_table() {
            Some(NgramTable::Paged(table)) => Ok(table
                .preload(lock, &std::sync::atomic::AtomicBool::new(false))?
                .resident),
            _ => Ok(0),
        }
    }

    fn paged_table(&self) -> Option<std::sync::Arc<super::ngram::PagedTable>> {
        match self.ple_table() {
            Some(NgramTable::Paged(table)) => Some(table.clone()),
            _ => None,
        }
    }

    fn paged_storage_bytes(&self) -> usize {
        match self.ple_table() {
            Some(NgramTable::Paged(table)) => table.bytes() as usize,
            _ => 0,
        }
    }

    fn new_state(&self, ctx: &MetalContext, capacity: usize) -> Result<DecodeState> {
        Qwen4ExpModel::new_state(self, ctx, capacity)
    }

    fn new_scratch_with_capacity(
        &self,
        ctx: &MetalContext,
        capacity_tokens: usize,
    ) -> Result<Scratch> {
        Qwen4ExpModel::new_scratch_with_capacity(self, ctx, capacity_tokens)
    }

    fn prefill(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        scratch: &mut Scratch,
        tokens: &[u32],
        draw: Option<Draw<'_>>,
    ) -> Result<()> {
        Qwen4ExpModel::prefill(self, ctx, state, scratch, tokens, draw)
    }

    fn prefill_with_vision(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        scratch: &mut Scratch,
        tokens: &[u32],
        draw: Option<Draw<'_>>,
        vision: Option<&VisionInput<'_>>,
    ) -> Result<()> {
        Qwen4ExpModel::prefill_with_vision(
            self, ctx, state, scratch, tokens, draw, vision,
        )
    }

    fn prefill_until(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        scratch: &mut Scratch,
        tokens: &[u32],
        vision: Option<&VisionInput<'_>>,
        stop: &dyn Fn() -> bool,
    ) -> Result<usize> {
        Qwen4ExpModel::prefill_until(self, ctx, state, scratch, tokens, vision, stop)
    }

    fn encode_image(
        &self,
        ctx: &MetalContext,
        scratch: &mut Scratch,
        pixels: &[f32],
        grid_h: usize,
        grid_w: usize,
    ) -> Result<Tensor> {
        Qwen4ExpModel::encode_image(self, ctx, scratch, pixels, grid_h, grid_w)
    }

    fn prepare_step_inputs(
        &self,
        state: &mut DecodeState,
        scratch: &Scratch,
        token: u32,
    ) -> Result<()> {
        Qwen4ExpModel::prepare_step_inputs(self, state, scratch, token)
    }

    fn encode_decode_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        scratch: &Scratch,
        slot_in: usize,
        slot_out: usize,
        draw: Draw<'_>,
    ) -> Result<EncodedPass<'a>> {
        Qwen4ExpModel::encode_decode_step(
            self, ctx, state, scratch, slot_in, slot_out, draw, None,
        )
    }

    fn supports_parking(&self) -> bool {
        true
    }

    fn encode_parked_step<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &DecodeState,
        scratch: &Scratch,
        slot_in: usize,
        slot_out: usize,
        draw: Draw<'_>,
    ) -> Result<EncodedPass<'a>> {
        let value = scratch.sync.arm()?;
        match Qwen4ExpModel::encode_decode_step(
            self,
            ctx,
            state,
            scratch,
            slot_in,
            slot_out,
            draw,
            Some(value),
        ) {
            Ok(pass) => Ok(pass),
            Err(e) => {
                // Nothing was committed; free the claimed value.
                scratch.sync.release()?;
                Err(e)
            }
        }
    }

    fn max_batch_rows(&self) -> usize {
        self.batch_rows()
    }

    fn commit_rows<'a>(
        &self,
        ctx: &'a MetalContext,
        scratch: &mut Scratch,
        rows: &mut [BatchRow<'_, DecodeState>],
        timed: bool,
    ) -> Result<RowsInFlight<'a>> {
        self.commit_session_rows(ctx, scratch, rows, timed)
    }

    fn finish_rows(
        &self,
        scratch: &Scratch,
        step: RowsInFlight<'_>,
    ) -> Result<(Vec<u32>, Option<RowsStepTiming>)> {
        self.finish_session_rows(scratch, step)
    }

    fn supports_rows_parking(&self) -> bool {
        true
    }

    fn park_rows<'a>(
        &self,
        ctx: &'a MetalContext,
        scratch: &Scratch,
        rows: &mut [BatchRow<'_, DecodeState>],
        after: &RowsInFlight<'a>,
        timed: bool,
    ) -> Result<RowsInFlight<'a>> {
        self.park_session_rows(ctx, scratch, rows, after, timed)
    }

    fn release_rows(
        &self,
        scratch: &Scratch,
        step: &mut RowsInFlight<'_>,
        rows: &mut [BatchRow<'_, DecodeState>],
    ) -> Result<()> {
        self.release_session_rows(scratch, step, rows)
    }

    fn move_sampler_counts(
        &self,
        ctx: &MetalContext,
        scratch: &mut Scratch,
        from: CountsSlot,
        to: CountsSlot,
    ) -> Result<()> {
        self.move_counts(ctx, scratch, from, to)
    }

    fn release_parked(&self, scratch: &Scratch) -> Result<()> {
        scratch.sync.release()
    }

    fn max_drafts(&self) -> usize {
        Qwen4ExpModel::max_drafts(self)
    }

    fn draft_initial(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        scratch: &mut Scratch,
        first: u32,
        drafts: usize,
        params: &SamplingParams,
        step0: usize,
    ) -> Result<Vec<u32>> {
        Qwen4ExpModel::draft_initial(
            self, ctx, state, scratch, first, drafts, params, step0,
        )
    }

    fn verify<'a>(
        &self,
        ctx: &'a MetalContext,
        state: &mut DecodeState,
        scratch: &mut Scratch,
        pending: u32,
        drafts: &[u32],
        params: &SamplingParams,
        step0: usize,
        next_drafts: usize,
    ) -> Result<(Vec<u32>, PendingPass<'a>)> {
        Qwen4ExpModel::verify(
            self,
            ctx,
            state,
            scratch,
            pending,
            drafts,
            params,
            step0,
            next_drafts,
        )
    }

    fn finish_speculation(
        &self,
        ctx: &MetalContext,
        state: &mut DecodeState,
        scratch: &mut Scratch,
        accepted: usize,
        propose: bool,
        draft: PendingPass<'_>,
    ) -> Result<Vec<u32>> {
        Qwen4ExpModel::finish_speculation(
            self, ctx, state, scratch, accepted, propose, draft,
        )
    }
}

// Batched decode across sessions; a child module for the graph's private
// helpers and scratch.
#[path = "batch.rs"]
mod batch;

#[cfg(test)]
#[path = "official_gdn_state_lifecycle_tests.rs"]
mod official_gdn_state_lifecycle_tests;
