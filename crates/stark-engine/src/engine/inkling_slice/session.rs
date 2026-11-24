// SPDX-License-Identifier: Apache-2.0
//! Serialized target session for the publisher's original Inkling 2 Mini.

use std::{fs::File, ops::Range, path::Path, sync::Arc};

use metal::MTLBlitCommandEncoder;

#[path = "affine8_validation.rs"]
mod affine8_validation;
pub use affine8_validation::{
    QualitySide, measure_affine8_dense, measure_affine8_quality, measure_affine8_timing, measure_quality_compare,
};
#[path = "decode_stages.rs"]
mod decode_stages;
pub(super) use decode_stages::{DECODE_EXTRA, DECODE_VARIANTS};
pub use decode_stages::{
    measure_decode_experts, measure_decode_stages, measure_depth_policies, measure_draft_trace, measure_paired_control,
    measure_round_budget,
};
#[path = "dspark_validation.rs"]
mod dspark_validation;
pub use dspark_validation::{measure_dspark, measure_dspark_depth, measure_source_dspark_depth};

#[path = "draft_overlap.rs"]
mod draft_overlap;
#[path = "packed_prefill.rs"]
mod packed_prefill;
#[path = "prefill.rs"]
mod prefill;
#[path = "prefill_dense.rs"]
mod prefill_dense;
#[path = "prefill_profile.rs"]
pub(super) mod prefill_profile;
#[path = "prefill_relative.rs"]
mod prefill_relative;
#[path = "prefill_trace.rs"]
mod prefill_trace;
#[path = "schedule_validation.rs"]
mod schedule_validation;
#[path = "target_validation.rs"]
mod target_validation;
#[path = "transaction_validation.rs"]
mod transaction_validation;
#[path = "verify_state.rs"]
mod verify_state;
pub use draft_overlap::measure_draft_overlap;
use packed_prefill::PackedPrefill;
use prefill::PrefillLanes;
pub use prefill_profile::{
    measure_dense_ceiling, measure_expert_ceiling, probe_stock_nvfp4, profile_packed_prefill, profile_prefill,
    profile_schedule, profile_stock_4k, profile_worker_sequence,
};
pub use prefill_trace::{PrefillTraceMode, trace_packed_prefill, trace_prefill};
pub use schedule_validation::{
    measure_full_schedule, measure_prefill_tolerance, measure_schedule_replay, measure_source_dspark_depth_schedule,
    measure_split_consistency, measure_verify_transaction_schedule,
};
pub use target_validation::{
    measure_decode_verify, measure_full, measure_full_budget, measure_prefill, measure_prefill_speed,
};
pub use transaction_validation::measure_verify_transaction;
pub use verify_state::{MAX_VERIFY_ROWS, Verification};

use super::{dspark::Dspark, vision::Vision, *};

const STORAGE_VOCABULARY: usize = 201_024;
pub const VOCABULARY: usize = 200_058;
pub(super) const PADDED_VOCABULARY: usize = VOCABULARY.div_ceil(32) * 32;
const MAX_CONTEXT: usize = 32_768;
const PREFILL_COMMAND_ROWS: usize = 8;
const BATCHED_PREFILL_ROWS: usize = 64;
const M5_PREFILL_ROWS: usize = 128;
const DRAFT_TAPS: [usize; 3] = [5, 23, 41];

const PHYSICAL: [usize; 48] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 6, 7, 8,
    9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
];

/// Immutable GPU-only copy of the live keys, values and convolution histories.
/// The generation worker must bind publication to its model, context and request scope.
pub struct Snapshot {
    owner: Arc<()>,
    context: usize,
    position: usize,
    dspark: bool,
    source_text: bool,
    prefill_rows: usize,
    state: Buffer,
}

/// Complete Inkling 2 Mini target with one persistent cache per logical visit.
pub struct Session {
    owner: Arc<()>,
    runtime: Runtime,
    layers: Vec<Layer>,
    vision: Vision,
    visits: Vec<Visit>,
    flow: Flow,
    prefill_lanes: Option<PrefillLanes>,
    packed_prefill: Option<PackedPrefill>,
    source_text_requested: bool,
    embedding: Buffer,
    embed_norm: Buffer,
    output_query: Buffer,
    output_norm: Buffer,
    unembedding: Buffer,
    mixed: Buffer,
    hidden: Buffer,
    logits: Buffer,
    draft_features: Buffer,
    dspark: Option<Dspark>,
    dspark_enabled: bool,
    verifier: Option<verify_state::VerifyWorkspace>,
    source_verifier: bool,
    capture_draft_features: bool,
    features_ready: bool,
    capture_visits: bool,
    context: usize,
    position: usize,
    /// Source prefill command width selected at `set_batched_prefill_width`.
    source_command_rows: usize,
}

impl Session {
    /// Maximum rows in the image-loaded M5 Max source prefill command.
    pub const SOURCE_PREFILL_ROWS: usize = 2048;

    /// Load directly from the original publisher shard. The worker owns package
    /// admission, including tokenizer metadata and source identity checks.
    pub fn load(
        checkpoint: &Path,
        context: usize,
    ) -> Result<Self> {
        Self::load_file(File::open(checkpoint).map_err(|_| "cannot open SafeTensors")?, context)
    }

    /// Consume the same descriptor that the caller verified for admission.
    pub fn load_file(
        checkpoint: File,
        context: usize,
    ) -> Result<Self> {
        let mut runtime = Runtime::new()?;
        Self::check_context(&runtime, context)?;
        let source = Tensors::from_file(checkpoint)?;
        let affine = affine8::present(&source);
        if affine {
            affine8::register(&mut runtime)?;
        }
        let device = &runtime.device;
        let layers = (0..24).map(|physical| Layer::load(&source, device, physical)).collect::<Result<Vec<_>>>()?;
        let vision = Vision::load(&source, device)?;
        let visits = Self::new_visits(&runtime, &layers, context)?;
        let bf16 = |name, shape: &[usize]| source.buffer(device, name, "BF16", shape);
        let (affine_weights, affine_embedding, affine_head) = if affine {
            let (weights, embedding, head) =
                affine8::Affine8Weights::load(&source, device, &layers, STORAGE_VOCABULARY, PADDED_VOCABULARY)?;
            (Some(weights), Some(embedding), Some(head))
        } else {
            (None, None, None)
        };
        let embedding = match affine_embedding {
            Some(copy) => copy,
            None => bf16("model.llm.embed.weight", &[STORAGE_VOCABULARY, WIDTH])?,
        };
        let embed_norm = bf16("model.llm.embed_norm.weight", &[WIDTH])?;
        let output_norm = bf16("model.llm.norm.weight", &[WIDTH])?;
        let output_query = source.buffer(device, "model.llm.attn_res.output_query", "F32", &[WIDTH])?;
        // The vocabulary projection may read up to 16 rows. All rows after
        // the live row stay zero, and the sampler sees only the live vocabulary.
        let logits_rows = if runtime.pinned_m5 {
            16
        } else {
            1
        };
        let unembedding = match affine_head {
            Some(copy) => copy,
            None => {
                let unembedding = zeroed(device, PADDED_VOCABULARY * WIDTH * 2)?;
                source.copy_to(
                    "model.llm.unembed.weight",
                    "BF16",
                    &[STORAGE_VOCABULARY, WIDTH],
                    0,
                    &unembedding,
                    0,
                    PADDED_VOCABULARY * WIDTH * 2,
                )?;
                unembedding
            },
        };
        runtime.affine8 = affine_weights;
        Ok(Self {
            owner: Arc::new(()),
            flow: Flow::new(device)?,
            prefill_lanes: None,
            packed_prefill: None,
            source_text_requested: true,
            mixed: zeroed(device, WIDTH * 4)?,
            hidden: zeroed(device, logits_rows * WIDTH * 2)?,
            logits: zeroed(device, logits_rows * PADDED_VOCABULARY * 4)?,
            draft_features: zeroed(device, 4 * WIDTH * 2)?,
            dspark: None,
            dspark_enabled: false,
            verifier: None,
            source_verifier: false,
            layers,
            vision,
            visits,
            embedding,
            embed_norm,
            output_query,
            output_norm,
            unembedding,
            runtime,
            capture_draft_features: false,
            features_ready: false,
            capture_visits: false,
            context,
            position: 0,
            source_command_rows: Self::SOURCE_PREFILL_ROWS,
        })
    }

    fn check_context(
        runtime: &Runtime,
        context: usize,
    ) -> Result<()> {
        let max_context = if runtime.pinned_m5 {
            MAX_CONTEXT
        } else {
            CAPACITY
        };
        if !(1..=max_context).contains(&context) {
            return Err("Inkling context exceeds the supported Metal layout");
        }
        let persistent =
            Self::context_memory_bytes(context, false).ok_or("Inkling context exceeds the supported range")?;
        if persistent as u64 > runtime.device.recommended_max_working_set_size() {
            return Err("Inkling persistent state exceeds recommended Metal working set");
        }
        Ok(())
    }

    fn new_visits(
        runtime: &Runtime,
        layers: &[Layer],
        context: usize,
    ) -> Result<Vec<Visit>> {
        PHYSICAL
            .iter()
            .map(|&physical| Visit::new_context(&runtime.device, layers[physical].local, context, runtime.pinned_m5))
            .collect()
    }

    /// Maximum supported context, which depends on the selected attention kernel.
    pub fn max_context_length(&self) -> usize {
        if self.runtime.pinned_m5 {
            MAX_CONTEXT
        } else {
            CAPACITY
        }
    }

    /// Resident persistent cache bytes at a given context. Include one full
    /// snapshot when the worker plans to retain a previous prefill boundary.
    /// Weights, Metal scratch and image staging are accounted for separately.
    pub fn context_memory_bytes(
        context: usize,
        prefix: bool,
    ) -> Option<usize> {
        if !(1..=MAX_CONTEXT).contains(&context) {
            return None;
        }
        Some(
            Self::state_bytes(context)
                * if prefix {
                    2
                } else {
                    1
                },
        )
    }

    fn state_bytes(position: usize) -> usize {
        let keys_and_values = (40 * position.min(LOCAL) + 8 * position) * 2 * KV * 2;
        let convolutions = PHYSICAL.len() * 3 * (2 * KV + 2 * WIDTH) * 2;
        keys_and_values + convolutions
    }

    /// Set the worker's context and discard live generation state. A context
    /// change recreates only the visit state and workspaces.
    pub fn set_context(
        &mut self,
        context: usize,
    ) -> Result<()> {
        Self::check_context(&self.runtime, context)?;
        if context != self.context {
            let visits = Self::new_visits(&self.runtime, &self.layers, context)?;
            if let Some(dspark) = &mut self.dspark {
                dspark.set_context(&self.runtime.device, context)?;
            }
            self.visits = visits;
            self.context = context;
            self.apply_prefill_scope();
            self.reset();
        }
        self.ensure_source_workspace()
    }

    /// Size the context-sized source global workspace. A source-scoped
    /// session gets the current context's capacity; any other request keeps
    /// the 4,352 keys that the load-time reserve covers, so only admitted
    /// source requests grow it. Select the scope before the context.
    ///
    /// A failed allocation releases the old buffers first, leaves the
    /// capacity at zero and returns the error, so the worker fails this
    /// request and clears its prefix cache. The source path then stays
    /// inactive (`source_text_ready` is false) until a later request
    /// boundary reallocates successfully; it can never overrun a smaller
    /// workspace.
    fn ensure_source_workspace(&mut self) -> Result<()> {
        let context = if self.source_text_requested {
            self.context
        } else {
            0
        };
        let Some(packed) = self.packed_prefill.as_mut().filter(|packed| packed.context_workspace()) else {
            return Ok(());
        };
        let result = packed.set_source_context(&self.runtime.device, context);
        self.apply_prefill_scope();
        result
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn context(&self) -> usize {
        self.context
    }

    /// Command row bound for worker cancellation and chunk-aligned snapshots.
    pub fn prefill_batch_rows(&self) -> usize {
        if let Some(packed) = &self.packed_prefill
            && (!self.dspark_enabled || self.runtime.source_text_active && packed.captures_all_draft_taps())
        {
            if self.runtime.source_text_active {
                packed.rows()
            } else {
                packed.rows().min(PackedPrefill::EXACT_WORKER_ROWS)
            }
        } else {
            self.image_prefill_batch_rows()
        }
    }

    /// Image embedding rows still use the per-row-input batch.
    pub fn image_prefill_batch_rows(&self) -> usize {
        self.prefill_lanes.as_ref().map_or(PREFILL_COMMAND_ROWS, PrefillLanes::rows)
    }

    pub fn supports_batched_prefill(&self) -> bool {
        self.runtime.pinned_m5
    }

    /// Reserve both the packed and per-row workspaces separately from persistent cache.
    pub fn batched_prefill_memory_bytes(source_allowed: bool) -> usize {
        Self::batched_prefill_memory_bytes_at(source_allowed, Self::SOURCE_PREFILL_ROWS)
    }

    /// Source prefill command widths a worker may select at load. Default
    /// prefill arithmetic does not depend on command width; prefix snapshots
    /// stay aligned to `SOURCE_PREFILL_ROWS` at every width.
    pub const SOURCE_COMMAND_WIDTHS: [usize; 3] = [2048, 4096, 8192];

    /// Reserve for `set_batched_prefill_width` at a source command width.
    /// Measured additional Metal allocation of batched prefill on the M5 Max
    /// (1,133, 2,106 and 4,052 MiB at 2,048, 4,096 and 8,192 rows) plus the
    /// 211 MiB margin of the original 1,344 MiB reserve. Widths outside
    /// `SOURCE_COMMAND_WIDTHS` return the 2,048-row reserve.
    pub fn batched_prefill_memory_bytes_at(
        source_allowed: bool,
        rows: usize,
    ) -> usize {
        const MIB: usize = 1024 * 1024;
        if !source_allowed {
            return 384 * MIB;
        }
        match rows {
            4096 => 2317 * MIB,
            8192 => 4263 * MIB,
            _ => 1344 * MIB,
        }
    }

    /// `set_batched_prefill` with a source command width from
    /// `SOURCE_COMMAND_WIDTHS`. Selecting a different width reallocates the
    /// packed workspace and must happen before ingest.
    pub fn set_batched_prefill_width(
        &mut self,
        enabled: bool,
        source_allowed: bool,
        rows: usize,
    ) -> Result<()> {
        if !Self::SOURCE_COMMAND_WIDTHS.contains(&rows) {
            return Err("Inkling source prefill width must be 2,048, 4,096 or 8,192 rows");
        }
        if enabled && self.prefill_lanes.is_some() && self.source_command_rows != rows {
            if self.position != 0 {
                return Err("Inkling prefill mode must be selected before ingest");
            }
            self.packed_prefill = None;
            self.prefill_lanes = None;
        }
        self.source_command_rows = rows;
        self.set_batched_prefill(enabled, source_allowed)
    }

    /// Select M5 Max batched prefill before ingesting
    /// a prompt. Prefix snapshots stay bound to the selected row geometry.
    pub fn set_batched_prefill(
        &mut self,
        enabled: bool,
        source_allowed: bool,
    ) -> Result<()> {
        if enabled == self.prefill_lanes.is_some() {
            return Ok(());
        }
        if self.position != 0 {
            return Err("Inkling prefill mode must be selected before ingest");
        }
        let (lanes, packed) = if enabled {
            let mut lanes = PrefillLanes::for_m5_worker(&mut self.runtime)?;
            lanes.enable_fused_decode(&mut self.runtime)?;
            let packed = PackedPrefill::for_m5_worker(
                &mut self.runtime,
                &mut self.layers,
                source_allowed,
                self.source_command_rows,
            )?;
            (Some(lanes), Some(packed))
        } else {
            (None, None)
        };
        let tensor = packed.as_ref().is_some_and(PackedPrefill::mxfp8_tensor_enabled);
        self.runtime.mxfp8_tensor_active = tensor;
        self.runtime.mxfp8_tensor_available = tensor;
        self.prefill_lanes = lanes;
        self.packed_prefill = packed;
        self.apply_prefill_scope();
        self.ensure_source_workspace()
    }

    /// Whether this loaded worker can run reference-shaped prefill at a request
    /// context once `set_context` has selected it.
    pub fn supports_source_text_prefill(
        &self,
        context: usize,
    ) -> bool {
        self.packed_prefill.as_ref().is_some_and(|packed| packed.supports_source_context(context))
    }

    /// Whether the allocated source workspace covers the current context.
    fn source_text_ready(&self) -> bool {
        self.packed_prefill.as_ref().is_some_and(|packed| packed.source_ready(self.context))
    }

    /// Smallest packed command. The default path runs every packed command on
    /// the source kernels; shorter chunks use the lane or serial kernels.
    pub const SOURCE_MIN_BULK_ROWS: usize = PackedPrefill::MIN_ROWS;

    /// Largest source-path context with (loaded source global probe) or without the context-sized workspace.
    pub fn source_context_limit(context_workspace: bool) -> usize {
        PackedPrefill::source_context_limit(context_workspace)
    }

    /// Global key capacity the context-sized workspace allocates for a context.
    pub fn source_global_capacity(context: usize) -> usize {
        PackedPrefill::source_global_capacity(context)
    }

    /// Key-extent global workspace bytes of the default path (F32 keys and values).
    pub fn source_global_workspace_bytes(keys: usize) -> usize {
        PackedPrefill::source_global_workspace_bytes(keys)
    }

    /// F32 per-head relative mask bytes at a key capacity.
    pub fn source_global_mask_bytes(keys: usize) -> usize {
        PackedPrefill::source_global_mask_bytes(keys)
    }

    /// Workspace bytes above the load-time source reserve at a context, for a
    /// session with or without the context-sized workspace.
    pub fn source_global_extra_bytes(
        context_workspace: bool,
        context: usize,
    ) -> usize {
        PackedPrefill::source_global_extra_bytes(context_workspace, context)
    }

    /// Additional resident workspace this loaded session allocates at a
    /// context on the source path; zero without the source global probe.
    pub fn source_context_workspace_bytes(
        &self,
        context: usize,
    ) -> usize {
        self.packed_prefill.as_ref().map_or(0, |packed| packed.source_context_extra_bytes(context))
    }

    /// Select the packed text precision policy before ingesting a request.
    pub fn set_text_prefill_scope(
        &mut self,
        source: bool,
    ) {
        debug_assert_eq!(self.position, 0);
        self.source_text_requested = source;
        self.apply_prefill_scope();
    }

    fn apply_prefill_scope(&mut self) {
        let source =
            self.source_text_requested && (!self.dspark_enabled || self.source_verifier) && self.source_text_ready();
        self.runtime.source_text_active = source;
        if let Some(packed) = &mut self.packed_prefill {
            packed.set_source_scope(source);
        }
    }

    /// Reset the live position. Kernels read only committed absolute positions,
    /// so previous request rows are unreachable and overwritten as needed.
    pub fn reset(&mut self) {
        self.position = 0;
        self.features_ready = false;
    }

    /// Choose whether subsequent forwards retain the current token's target taps.
    pub fn capture_draft_features(
        &mut self,
        enable: bool,
    ) {
        self.capture_draft_features = enable;
        self.features_ready = false;
    }

    /// Return the last token's BF16 taps in source order, as 4 × 2304 words.
    /// The worker must avoid logging or retaining these after request cleanup.
    pub fn draft_features(&self) -> Option<Vec<u16>> {
        self.features_ready.then(|| read_shared(&self.draft_features, 4 * WIDTH))
    }

    /// Load the separate original DSpark checkpoint and fold its Markov head
    /// into the target's immutable vocabulary projection.
    pub fn load_dspark(
        &mut self,
        checkpoint: &Path,
    ) -> Result<()> {
        self.load_dspark_file(File::open(checkpoint).map_err(|_| "cannot open SafeTensors")?)
    }

    /// Load the draft through the same descriptor as package admission.
    pub fn load_dspark_file(
        &mut self,
        checkpoint: File,
    ) -> Result<()> {
        if self.position != 0 {
            return Err("Inkling DSpark must be loaded before prefill");
        }
        if self.dspark.is_some() {
            return Err("Inkling DSpark is already loaded");
        }
        self.dspark = Some(Dspark::load(&mut self.runtime, checkpoint, &self.unembedding, self.context)?);
        Ok(())
    }

    pub const DSPARK_INT8_HEAD_BYTES: usize = Dspark::INT8_HEAD_BYTES;

    /// Reserve the optional immutable proposal projection at model admission.
    pub fn prepare_dspark_int8_head(&mut self) -> Result<()> {
        if self.position != 0 || !self.source_verifier {
            return Err("Inkling INT8 draft head requires source verification before prefill");
        }
        self.dspark
            .as_mut()
            .ok_or("Inkling DSpark weights are unavailable")?
            .prepare_int8_head(&mut self.runtime, &self.unembedding)
    }

    pub fn dspark_int8_head_resident_bytes(&self) -> usize {
        self.dspark.as_ref().map_or(0, Dspark::int8_head_resident_bytes)
    }

    pub fn has_dspark(&self) -> bool {
        self.dspark.is_some()
    }

    /// Select whether committed text rows update the independent draft cache.
    pub fn set_dspark_enabled(
        &mut self,
        enabled: bool,
    ) -> Result<()> {
        if enabled == self.dspark_enabled {
            return Ok(());
        }
        if self.position != 0 {
            return Err("Inkling DSpark mode must be selected before prefill");
        }
        if enabled && self.dspark.is_none() {
            return Err("Inkling DSpark weights are unavailable");
        }
        self.dspark_enabled = enabled;
        self.apply_prefill_scope();
        Ok(())
    }

    /// Produce one bounded candidate after the pending anchor. The target
    /// alone determines which tokens are committed.
    pub fn draft_propose(
        &self,
        anchor: u32,
    ) -> Result<Option<u32>> {
        if anchor as usize >= VOCABULARY {
            return Err("Inkling draft anchor is outside the vocabulary");
        }
        let Some(dspark) = self.active_dspark() else {
            return Ok(None);
        };
        if self.position == 0 || self.context - self.position < 2 {
            return Ok(None);
        }
        dspark
            .propose(&self.runtime, anchor, self.position, &self.embedding, &self.embed_norm, &self.unembedding)
            .map(Some)
    }

    /// Conservative additional resident allocation for the original draft and
    /// live state, with a second state reserved when prefix reuse is enabled.
    pub fn dspark_memory_bytes(
        context: usize,
        prefix: bool,
    ) -> Option<usize> {
        if !(1..=MAX_CONTEXT).contains(&context) {
            return None;
        }
        let resident = 736 * 1024 * 1024;
        let states = if prefix {
            2
        } else {
            1
        };
        Some(resident + Dspark::state_bytes(context) * states)
    }

    fn active_dspark(&self) -> Option<&Dspark> {
        self.dspark.as_ref().filter(|_| self.dspark_enabled)
    }

    /// Feed a token whose logits are not needed.
    pub fn ingest(
        &mut self,
        token: u32,
    ) -> Result<()> {
        self.ingest_many(std::slice::from_ref(&token))
    }

    /// Ingest causal rows in bounded command batches. The worker may take a
    /// prefix snapshot at the end of a complete batch.
    pub fn ingest_many(
        &mut self,
        tokens: &[u32],
    ) -> Result<()> {
        self.check_append(tokens.len())?;
        if !tokens.iter().all(|&id| (id as usize) < VOCABULARY) {
            return Err("Inkling token is outside the vocabulary");
        }
        for chunk in tokens.chunks(self.prefill_batch_rows()) {
            self.append(chunk, false)?;
        }
        Ok(())
    }

    /// Decode a single committed token and return its F32 soft-capped logits.
    pub fn step(
        &mut self,
        token: u32,
    ) -> Result<Vec<f32>> {
        self.check_append(1)?;
        if token as usize >= VOCABULARY {
            return Err("Inkling token is outside the vocabulary");
        }
        self.append(std::slice::from_ref(&token), true)?;
        self.read_logits()
    }

    /// Process the prompt suffix and return the final token's logits.
    pub fn prefill(
        &mut self,
        tokens: &[u32],
    ) -> Result<Vec<f32>> {
        let (&last, earlier) = tokens.split_last().ok_or("Inkling prefill is empty")?;
        self.check_append(tokens.len())?;
        if last as usize >= VOCABULARY {
            return Err("Inkling token is outside the vocabulary");
        }
        self.ingest_many(earlier)?;
        self.step(last)
    }

    /// Encode a single image into the source's contiguous spatial token rows.
    pub fn encode_image(
        &self,
        bytes: &[u8],
        available_tokens: usize,
    ) -> Result<ImageEmbeddings> {
        self.vision.encode(&self.runtime, bytes, available_tokens)
    }

    pub fn ingest_image(
        &mut self,
        image: &ImageEmbeddings,
    ) -> Result<()> {
        self.ingest_image_rows(image, 0..image.token_count())
    }

    /// Insert projected rows in their original sequence order. The worker
    /// can check for cancellation between its chosen subranges.
    pub fn ingest_image_rows(
        &mut self,
        image: &ImageEmbeddings,
        rows: Range<usize>,
    ) -> Result<()> {
        self.check_image_mode()?;
        if rows.start > rows.end || rows.end > image.token_count() {
            return Err("Inkling image range falls outside its embeddings");
        }
        self.check_append(rows.len())?;
        let batch_rows = self.image_prefill_batch_rows();
        for start in rows.clone().step_by(batch_rows) {
            self.append_image(image, start..rows.end.min(start + batch_rows), false)?;
        }
        Ok(())
    }

    /// Insert the final image row and return its logits if the prompt ends there.
    pub fn step_image_row(
        &mut self,
        image: &ImageEmbeddings,
        row: usize,
    ) -> Result<Vec<f32>> {
        self.check_image_mode()?;
        self.check_append(1)?;
        if row >= image.token_count() {
            return Err("Inkling image row falls outside its embeddings");
        }
        self.append_image(image, row..row + 1, true)?;
        self.read_logits()
    }

    fn check_append(
        &self,
        count: usize,
    ) -> Result<()> {
        if count > self.context - self.position {
            return Err("Inkling context is full");
        }
        Ok(())
    }

    fn check_image_mode(&self) -> Result<()> {
        if self.dspark_enabled {
            return Err("Inkling image requests require draft mode to be disabled");
        }
        Ok(())
    }

    fn append(
        &mut self,
        tokens: &[u32],
        readout: bool,
    ) -> Result<()> {
        let result = self.encode_append(tokens, readout, None);
        self.finish_append(result, tokens.len())
    }

    fn encode_append(
        &self,
        tokens: &[u32],
        readout: bool,
        profile: Option<&mut PrefillTiming>,
    ) -> Result<Timing> {
        if let Some(packed) = &self.packed_prefill
            && tokens.len() >= packed.min_command_rows(&self.runtime)
            && (self.active_dspark().is_none() || self.runtime.source_text_active && packed.captures_all_draft_taps())
        {
            packed.upload(tokens);
            return execute_with_profile(&self.runtime, profile, |gpu| packed.encode(self, gpu, tokens.len(), readout));
        }
        if let Some(lanes) = &self.prefill_lanes
            && tokens.len() > 1
            && (!self.runtime.source_text_active || tokens.len() > 3)
        {
            let inputs = tokens.iter().copied().map(TokenInput::Text).collect::<Vec<_>>();
            return execute_with_profile(&self.runtime, profile, |gpu| {
                self.encode_prefill_rows(gpu, lanes, &inputs, readout)
            });
        }
        execute_with_profile(&self.runtime, profile, |gpu| {
            let dspark = self.active_dspark();
            for (index, &token) in tokens.iter().enumerate() {
                let last = index + 1 == tokens.len();
                let position = self.position + index;
                self.encode_token(
                    gpu,
                    TokenInput::Text(token),
                    position,
                    readout && last,
                    dspark.is_some() || self.capture_draft_features && last,
                );
                if let Some(draft) = dspark {
                    draft.commit(gpu, &self.draft_features, position);
                }
            }
        })
    }

    fn append_image(
        &mut self,
        image: &ImageEmbeddings,
        rows: Range<usize>,
        readout: bool,
    ) -> Result<()> {
        let embeddings = rows.map(|row| image.row(row)).collect::<Result<Vec<_>>>()?;
        let result = execute(&self.runtime, |gpu| {
            if let Some(lanes) = &self.prefill_lanes
                && embeddings.len() > 1
            {
                let inputs = embeddings.iter().copied().map(TokenInput::Image).collect::<Vec<_>>();
                self.encode_prefill_rows(gpu, lanes, &inputs, readout);
                return;
            }
            for (index, &embedding) in embeddings.iter().enumerate() {
                let last = index + 1 == embeddings.len();
                self.encode_token(gpu, TokenInput::Image(embedding), self.position + index, readout && last, false);
            }
        });
        self.finish_append(result, embeddings.len())?;
        self.features_ready = false;
        Ok(())
    }

    fn finish_append(
        &mut self,
        result: Result<Timing>,
        count: usize,
    ) -> Result<()> {
        if let Err(label) = result {
            self.reset();
            return Err(label);
        }
        self.position += count;
        self.features_ready = self.capture_draft_features;
        Ok(())
    }

    fn read_logits(&mut self) -> Result<Vec<f32>> {
        let logits = read_shared::<f32>(&self.logits, VOCABULARY);
        if logits.iter().any(|value| !value.is_finite()) {
            self.reset();
            return Err("Inkling logits contain a nonfinite value");
        }
        Ok(logits)
    }

    fn encode_token(
        &self,
        gpu: &mut Stream<'_>,
        input: TokenInput<'_>,
        position: usize,
        readout: bool,
        features: bool,
    ) {
        let flow = &self.flow;
        let features = features.then_some(&self.draft_features);
        gpu.phase(PrefillPhase::Other);
        self.encode_embedding(gpu, flow, input);
        let mut blocks = 0;
        for (index, (&physical, visit)) in PHYSICAL.iter().zip(&self.visits).enumerate() {
            let state = self.visit_input(index, physical, &mut blocks);
            gpu.phase(PrefillPhase::AttentionInputs);
            if let Some(lanes) = self.prefill_lanes.as_ref().filter(|lanes| lanes.fuses_decode(gpu, &visit.scratch)) {
                let prepared = verify_state::VerifyWorkspace::prepare_rows(
                    gpu,
                    [flow; MAX_VERIFY_ROWS],
                    [features; MAX_VERIFY_ROWS],
                    index,
                    1,
                );
                if !prepared {
                    Self::prepare_visit(gpu, flow, index, features);
                }
                self.encode_fused_visit(gpu, lanes, &self.layers[physical], visit, state, position);
                continue;
            }
            Self::prepare_visit(gpu, flow, index, features);
            self.layers[physical].encode(gpu, visit, flow, &flow.partial, state, position);
        }
        if readout || features.is_some() {
            gpu.phase(PrefillPhase::Readout);
            self.finish_forward(gpu, flow, blocks, readout, features);
        }
    }

    fn encode_embedding(
        &self,
        gpu: &mut Stream<'_>,
        flow: &Flow,
        input: TokenInput<'_>,
    ) {
        match input {
            TokenInput::Text(token) => {
                if let Some(weights) = gpu.affine8_vocabulary() {
                    let buffers = [&weights.embedding.tensor, &self.embed_norm, &flow.partial];
                    let params = [token, weights.embedding.rows as u32];
                    gpu.one("ink_a8_embedding", &buffers, params, (1, 1), WIDTH / 4);
                    return;
                }
                gpu.one("ink_embedding", &[&self.embedding, &self.embed_norm, &flow.partial], token, (1, 1), WIDTH / 4)
            },
            TokenInput::Image(embedding) => {
                gpu.dispatch("ink_embedding_override", &[embedding, binding(&flow.partial)], (), (WIDTH / 128, 1), 128)
            },
        }
    }

    fn visit_input(
        &self,
        index: usize,
        physical: usize,
        blocks: &mut usize,
    ) -> VisitInput {
        if matches!(index, 18 | 30) {
            *blocks = 1;
        }
        let state = VisitInput {
            blocks: *blocks,
            later: (18..42).contains(&index),
            boundary: physical.is_multiple_of(6),
            capture: self.capture_visits,
        };
        *blocks += usize::from(state.boundary);
        state
    }

    fn prepare_visit(
        gpu: &mut Stream<'_>,
        flow: &Flow,
        index: usize,
        features: Option<&Buffer>,
    ) {
        match index {
            6 => gpu.copy(binding(&flow.partial), binding(&flow.prelude)),
            18 | 30 => {
                gpu.copy((&flow.blocks, 2 * WIDTH * 4), binding(&flow.previous));
                gpu.copy(binding(&flow.partial), (&flow.previous, WIDTH * 4));
                gpu.copy(binding(&flow.prelude), binding(&flow.partial));
            },
            _ => {},
        }
        if let (Some(features), Some(tap)) = (features, DRAFT_TAPS.iter().position(|&value| index == value)) {
            gpu.dispatch(
                "ink_capture_feature",
                &[binding(&flow.partial), (features, tap * WIDTH * 2)],
                (),
                (WIDTH / 128, 1),
                128,
            );
        }
    }

    fn finish_forward(
        &self,
        gpu: &mut Stream<'_>,
        flow: &Flow,
        blocks: usize,
        readout: bool,
        features: Option<&Buffer>,
    ) {
        if !(readout || features.is_some()) {
            return;
        }
        self.finish_hidden(gpu, flow, blocks, binding(&self.hidden), features);
        if readout {
            self.encode_output(gpu);
        }
    }

    fn finish_hidden(
        &self,
        gpu: &mut Stream<'_>,
        flow: &Flow,
        blocks: usize,
        hidden: Binding<'_>,
        features: Option<&Buffer>,
    ) {
        gpu.mix(&flow.blocks, &flow.partial, &self.output_query, &self.mixed, blocks, true, 1);
        gpu.norm(binding(&self.mixed), &self.output_norm, hidden, true);
        if let Some(features) = features {
            gpu.dispatch(
                "ink_trace_copy",
                &[hidden, (features, 3 * WIDTH * 2)],
                (WIDTH / 2) as u32,
                (WIDTH.div_ceil(256), 1),
                128,
            );
        }
    }

    fn encode_output(
        &self,
        gpu: &mut Stream<'_>,
    ) {
        self.encode_output_rows(gpu, 1);
    }

    /// Readout and logit cap of the first `rows` hidden rows (at most four).
    fn encode_output_rows(
        &self,
        gpu: &mut Stream<'_>,
        rows: usize,
    ) {
        if let Some(weights) = gpu.affine8_vocabulary() {
            affine8::encode_readout(gpu, weights, &self.hidden, &self.logits, rows, VOCABULARY, PADDED_VOCABULARY);
            return;
        }
        self.encode_bf16_output(gpu);
        for row in 1..rows {
            self.cap_logits(gpu, row);
        }
    }

    fn encode_bf16_output(
        &self,
        gpu: &mut Stream<'_>,
    ) {
        let buffers = [&self.hidden, &self.unembedding, &self.logits];
        if self.runtime.pinned_m5 {
            gpu.one(
                "ink_unembed_tensor",
                &buffers,
                [WIDTH as u32, PADDED_VOCABULARY as u32],
                (PADDED_VOCABULARY / 32, 1),
                32,
            );
            self.cap_logits(gpu, 0);
        } else {
            gpu.one("ink_unembed_f32", &buffers, VOCABULARY as u32, (VOCABULARY.div_ceil(4), 1), 128);
        }
    }

    fn cap_logits(
        &self,
        gpu: &mut Stream<'_>,
        row: usize,
    ) {
        gpu.dispatch(
            "ink_logits_cap",
            &[(&self.logits, row * PADDED_VOCABULARY * size_of::<f32>())],
            VOCABULARY as u32,
            (VOCABULARY.div_ceil(128), 1),
            128,
        );
    }

    /// Capture the exact persistent state after the last completed command.
    pub fn snapshot(&self) -> Result<Snapshot> {
        if self.position == 0 {
            return Err("Inkling cannot snapshot an empty session");
        }
        let state = self
            .runtime
            .device
            .new_buffer(self.snapshot_bytes(self.position), MTLResourceOptions::STORAGE_MODE_PRIVATE)
            .ok_or("Inkling snapshot allocation failed")?;
        self.transfer_snapshot(&state, self.position, false)?;
        Ok(Snapshot {
            owner: Arc::clone(&self.owner),
            context: self.context,
            position: self.position,
            dspark: self.dspark_enabled,
            source_text: self.runtime.source_text_active,
            prefill_rows: self.prefill_batch_rows(),
            state,
        })
    }

    /// Restore a prefix snapshot bound to this target and context.
    pub fn restore(
        &mut self,
        snapshot: &Snapshot,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.owner, &snapshot.owner)
            || self.context != snapshot.context
            || self.dspark_enabled != snapshot.dspark
            || self.runtime.source_text_active != snapshot.source_text
            || self.prefill_batch_rows() != snapshot.prefill_rows
        {
            return Err("Inkling snapshot belongs to another model, context, draft, or prefill mode");
        }
        if let Err(label) = self.transfer_snapshot(&snapshot.state, snapshot.position, true) {
            self.reset();
            return Err(label);
        }
        self.position = snapshot.position;
        self.features_ready = false;
        Ok(())
    }

    fn snapshot_bytes(
        &self,
        position: usize,
    ) -> usize {
        Self::state_bytes(position) + self.active_dspark().map_or(0, |_| Dspark::state_bytes(position))
    }

    fn transfer_snapshot(
        &self,
        state: &Buffer,
        position: usize,
        restore: bool,
    ) -> Result<()> {
        let command = self.runtime.queue.command_buffer().ok_or("Inkling snapshot command allocation failed")?;
        let encoder = command.blit_command_encoder().ok_or("Inkling snapshot encoder allocation failed")?;
        let mut offset = 0;
        let target_state = self.visits.iter().flat_map(|visit| {
            let kv_bytes = position.min(visit.capacity) * KV * 2;
            [(&visit.keys, kv_bytes), (&visit.values, kv_bytes)]
                .into_iter()
                .chain(visit.conv.iter().map(|buffer| (buffer, buffer.length())))
        });
        let draft_state = self.active_dspark().into_iter().flat_map(|draft| draft.state_buffers(position));
        for (buffer, bytes) in target_state.chain(draft_state) {
            if restore {
                encoder.copy_buffer_to_buffer(state, offset, buffer, 0, bytes);
            } else {
                encoder.copy_buffer_to_buffer(buffer, 0, state, offset, bytes);
            }
            offset += bytes;
        }
        debug_assert_eq!(offset, state.length());
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err("Inkling snapshot command failed");
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum TokenInput<'a> {
    Text(u32),
    Image(Binding<'a>),
}

pub(super) fn read_shared<T: bytemuck::Pod>(
    source: &Buffer,
    count: usize,
) -> Vec<T> {
    debug_assert!(count * size_of::<T>() <= source.length());
    // Session execution and blits are serialized and complete before reading.
    unsafe { std::slice::from_raw_parts(source.contents().as_ptr().cast::<T>(), count) }.to_vec()
}
