// SPDX-License-Identifier: Apache-2.0
//! Serialized target session for the publisher's original Inkling 2 Mini.

use std::{fs::File, ops::Range, path::Path, sync::Arc};

use metal::MTLBlitCommandEncoder;

#[path = "dspark_validation.rs"]
mod dspark_validation;
pub use dspark_validation::measure_dspark;

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
#[path = "target_validation.rs"]
mod target_validation;

use packed_prefill::PackedPrefill;
use prefill::PrefillLanes;
pub use prefill_profile::{profile_packed_prefill, profile_prefill};
pub use prefill_trace::{PrefillTraceMode, trace_prefill};
pub use target_validation::{measure_full, measure_full_budget, measure_prefill, measure_prefill_speed};

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
    capture_draft_features: bool,
    features_ready: bool,
    capture_visits: bool,
    context: usize,
    position: usize,
}

impl Session {
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
        let runtime = Runtime::new()?;
        Self::check_context(&runtime, context)?;
        let source = Tensors::from_file(checkpoint)?;
        let device = &runtime.device;
        let layers = (0..24).map(|physical| Layer::load(&source, device, physical)).collect::<Result<Vec<_>>>()?;
        let vision = Vision::load(&source, device)?;
        let visits = Self::new_visits(&runtime, &layers, context)?;
        let bf16 = |name, shape: &[usize]| source.buffer(device, name, "BF16", shape);
        let embedding = bf16("model.llm.embed.weight", &[STORAGE_VOCABULARY, WIDTH])?;
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
        Ok(Self {
            owner: Arc::new(()),
            flow: Flow::new(device)?,
            prefill_lanes: None,
            packed_prefill: None,
            mixed: zeroed(device, WIDTH * 4)?,
            hidden: zeroed(device, logits_rows * WIDTH * 2)?,
            logits: zeroed(device, logits_rows * PADDED_VOCABULARY * 4)?,
            draft_features: zeroed(device, 4 * WIDTH * 2)?,
            dspark: None,
            dspark_enabled: false,
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
            self.reset();
        }
        Ok(())
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn context(&self) -> usize {
        self.context
    }

    /// Command row bound for worker cancellation and chunk-aligned snapshots.
    pub fn prefill_batch_rows(&self) -> usize {
        if !self.dspark_enabled
            && let Some(packed) = &self.packed_prefill
        {
            packed.rows()
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
    pub const fn batched_prefill_memory_bytes() -> usize {
        384 * 1024 * 1024
    }

    /// Select M5 Max batched prefill before ingesting
    /// a prompt. Prefix snapshots stay bound to the selected row geometry.
    pub fn set_batched_prefill(
        &mut self,
        enabled: bool,
    ) -> Result<()> {
        if enabled == self.prefill_lanes.is_some() {
            return Ok(());
        }
        if self.position != 0 {
            return Err("Inkling prefill mode must be selected before ingest");
        }
        let (lanes, packed) = if enabled {
            (
                Some(PrefillLanes::for_m5_worker(&mut self.runtime)?),
                Some(PackedPrefill::for_m5_worker(&mut self.runtime)?),
            )
        } else {
            (None, None)
        };
        self.prefill_lanes = lanes;
        self.packed_prefill = packed;
        Ok(())
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

    /// Process serial-shape rows in bounded command batches. Every row remains
    /// causal and uses exactly the same kernels as decode. The worker may take
    /// a prefix snapshot at the end of this operation.
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
        if let Some(packed) = &self.packed_prefill
            && tokens.len() >= PackedPrefill::MIN_ROWS
            && self.active_dspark().is_none()
        {
            packed.upload(tokens);
            let result = execute(&self.runtime, |gpu| packed.encode(self, gpu, tokens.len(), readout));
            return self.finish_append(result, tokens.len());
        }
        if let Some(lanes) = &self.prefill_lanes
            && tokens.len() > 1
        {
            let inputs = tokens.iter().copied().map(TokenInput::Text).collect::<Vec<_>>();
            let result = execute(&self.runtime, |gpu| self.encode_prefill_rows(gpu, lanes, &inputs, readout));
            return self.finish_append(result, tokens.len());
        }
        let result = execute(&self.runtime, |gpu| {
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
        });
        self.finish_append(result, tokens.len())
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
        self.encode_embedding(gpu, flow, input);
        let mut blocks = 0;
        for (index, (&physical, visit)) in PHYSICAL.iter().zip(&self.visits).enumerate() {
            let state = self.visit_input(index, physical, &mut blocks);
            Self::prepare_visit(gpu, flow, index, features);
            self.layers[physical].encode(gpu, visit, flow, &flow.partial, state, position);
        }
        self.finish_forward(gpu, flow, blocks, readout, features);
    }

    fn encode_embedding(
        &self,
        gpu: &mut Stream<'_>,
        flow: &Flow,
        input: TokenInput<'_>,
    ) {
        match input {
            TokenInput::Text(token) => {
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
            boundary: physical % 6 == 0,
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
        gpu.mix(&flow.blocks, &flow.partial, &self.output_query, &self.mixed, blocks, true, 1);
        gpu.norm(binding(&self.mixed), &self.output_norm, binding(&self.hidden), true);
        if let Some(features) = features {
            gpu.dispatch(
                "ink_trace_copy",
                &[binding(&self.hidden), (features, 3 * WIDTH * 2)],
                (WIDTH / 2) as u32,
                (WIDTH.div_ceil(256), 1),
                128,
            );
        }
        if readout {
            self.encode_output(gpu);
        }
    }

    fn encode_output(
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
            gpu.one("ink_logits_cap", &[&self.logits], VOCABULARY as u32, (VOCABULARY.div_ceil(128), 1), 128);
        } else {
            gpu.one("ink_unembed_f32", &buffers, VOCABULARY as u32, (VOCABULARY.div_ceil(4), 1), 128);
        }
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

fn read_shared<T: bytemuck::Pod>(
    source: &Buffer,
    count: usize,
) -> Vec<T> {
    debug_assert!(count * size_of::<T>() <= source.length());
    // Session execution and blits are serialized and complete before reading.
    unsafe { std::slice::from_raw_parts(source.contents().as_ptr().cast::<T>(), count) }.to_vec()
}
