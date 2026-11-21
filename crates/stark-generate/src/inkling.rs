// SPDX-License-Identifier: Apache-2.0
//! Original Inkling 2 Mini target through the serialized JSONL worker.

use std::{
    path::Path,
    time::{Duration, Instant},
};

use serde_json::{Value, json};
use stark_engine::{
    backends::metal::Metal,
    engine::{
        Engine,
        hybrid::activity::InferenceActivity,
        inkling_slice::{INKLING_IMAGE_TOKEN as IMAGE_TOKEN, ImageEmbeddings, Session, Snapshot, VOCABULARY as VOCAB},
    },
};
use tokenizers::Tokenizer;

use crate::{
    cancel::CancelSignal,
    generation::{
        FinishReason, GeneratedTokens, GenerationBackend, GenerationOptions, GenerationProgress, ImageArtifactIdentity,
        MtpCapabilities, PreparedImageMetadata, ProgressCallback, TextCallback, admit_request_memory,
        collect_tokens_controlled,
    },
    hybrid::{ProbabilityMapper, sample_with_mapper},
    inkling_artifact::{self, Artifact},
    prefix_cache::{self, BoundaryEntry},
    text_stop::StopFilter,
};

const EOS: &[u32] = &[200_006];
const IMAGE_CONTENT: u32 = 200_005;
const END_MESSAGE: u32 = 200_010;
const CHUNK: usize = prefix_cache::PREFILL_CHUNK_TOKENS;
const INITIAL_CONTEXT: usize = 4352;
const MTP_VERIFICATION: &str = "inkling-dspark-target-draw-v1";

#[derive(Default, serde::Serialize)]
struct DraftStats {
    proposed: usize,
    accepted: usize,
}

pub(crate) struct InklingBackend {
    tokenizer: Tokenizer,
    config_sha256: String,
    target: Session,
    cancellation: Option<CancelSignal>,
    sampler: ProbabilityMapper,
    retained: Option<BoundaryEntry<(bool, Snapshot)>>,
    pending: Option<BoundaryEntry<(bool, Snapshot)>>,
    draft_stats: DraftStats,
    runtime_identity: stark_engine::engine::MetalDeviceIdentity,
}

pub(crate) struct PreparedInklingImage {
    image: ImageEmbeddings,
    input: Vec<u32>,
    slot: usize,
}

fn check_cancelled(
    signal: &mut Option<CancelSignal>,
    request: u64,
) -> Result<bool, &'static str> {
    match signal {
        Some(signal) => signal.check(request),
        None => Ok(false),
    }
}

fn require_active(
    signal: &mut Option<CancelSignal>,
    request: u64,
) -> Result<(), &'static str> {
    if check_cancelled(signal, request)? {
        Err("generation_cancelled")
    } else {
        Ok(())
    }
}

fn admit(bytes: Option<usize>) -> Result<(), &'static str> {
    admit_request_memory(bytes, stark_resource_budget::available_memory_bytes().unwrap_or(0))
}

struct PrefillProgress<'state, 'callback, 'environment> {
    cancellation: &'state mut Option<CancelSignal>,
    request: u64,
    started: Instant,
    last_progress: Duration,
    batch_rows: usize,
    callback: &'state mut Option<&'callback mut ProgressCallback<'environment>>,
}

impl PrefillProgress<'_, '_, '_> {
    fn run(
        &mut self,
        rows: usize,
        mut encode: impl FnMut(std::ops::Range<usize>) -> Result<(), &'static str>,
    ) -> Result<(), &'static str> {
        for start in (0..rows).step_by(self.batch_rows) {
            require_active(self.cancellation, self.request)?;
            encode(start..rows.min(start + self.batch_rows)).map_err(|_| "native_generation_failed")?;
            let now = self.started.elapsed();
            if now.saturating_sub(self.last_progress) >= Duration::from_secs(1) {
                if let Some(callback) = self.callback.as_deref_mut() {
                    callback(GenerationProgress {
                        sampled_tokens: 0,
                        elapsed_seconds: now.as_secs_f64(),
                        prefill_seconds: None,
                    })?;
                }
                self.last_progress = now;
            }
        }
        require_active(self.cancellation, self.request)
    }

    fn text(
        &mut self,
        target: &mut Session,
        tokens: &[u32],
    ) -> Result<(), &'static str> {
        self.run(tokens.len(), |rows| target.ingest_many(&tokens[rows]))
    }
}

impl InklingBackend {
    pub(crate) fn load(
        root: &Path,
        cancellation: Option<CancelSignal>,
    ) -> Result<Self, &'static str> {
        admit(Some(
            inkling_artifact::TARGET_BYTES
                + inkling_artifact::DRAFT_BYTES
                + 1024 * 1024 * 1024
                + Session::batched_prefill_memory_bytes(),
        ))?;
        let artifact = Artifact::load(root)?;
        let mut target =
            Session::load_file(artifact.target.descriptor()?, INITIAL_CONTEXT).map_err(|_| "inkling_load_failed")?;
        artifact.target.verify_unchanged()?;
        let runtime_identity = Engine::<Metal>::new().map_err(|_| "inkling_load_failed")?.metal_device_identity();
        if runtime_identity.device_name == "Apple M5 Max" && runtime_identity.supports_mxu {
            target.load_dspark_file(artifact.draft.descriptor()?).map_err(|_| "inkling_draft_load_failed")?;
            artifact.draft.verify_unchanged()?;
        }
        if target.supports_batched_prefill() {
            target.set_batched_prefill(true).map_err(|_| "inkling_prefill_load_failed")?;
        }
        Ok(Self {
            tokenizer: artifact.tokenizer,
            config_sha256: artifact.config_sha256,
            target,
            cancellation,
            sampler: ProbabilityMapper::default(),
            retained: None,
            pending: None,
            draft_stats: DraftStats::default(),
            runtime_identity,
        })
    }

    fn boundary(
        input_len: usize,
        options: &GenerationOptions,
    ) -> usize {
        options.cache_scope.map_or(0, |_| input_len.saturating_sub(2) / CHUNK * CHUNK)
    }

    fn generate_inner(
        &mut self,
        input: &[u32],
        image: Option<(&ImageEmbeddings, usize)>,
        options: &GenerationOptions,
        on_text: Option<&mut TextCallback<'_>>,
        mut on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        let expanded =
            input.len().checked_add(image.map_or(0, |(image, _)| image.token_count() - 1)).ok_or("context_overflow")?;
        if input.is_empty() || expanded >= options.context_tokens as usize {
            return Err("context_overflow");
        }
        self.draft_stats = DraftStats::default();
        let use_draft = options.mtp_drafts > 0;
        if use_draft && (options.mtp_drafts > 1 || image.is_some() || !self.target.has_dspark()) {
            return Err("mtp_not_supported");
        }
        let boundary = if image.is_none() {
            Self::boundary(input.len(), options)
        } else {
            0
        };
        let reused = if image.is_none() {
            self.prefix_cache_plan(input, options)
        } else {
            0
        };
        let retained = self.retained.take();
        self.pending = None;
        admit(self.context_bytes(options.context_tokens as usize, boundary > reused))?;
        let _activity = InferenceActivity::begin().map_err(|_| "inference_activity_unavailable")?;
        let started = Instant::now();
        require_active(&mut self.cancellation, options.request_id)?;
        self.target.set_context(options.context_tokens as usize).map_err(|_| "native_generation_failed")?;
        self.target.reset();
        self.target.set_dspark_enabled(use_draft).map_err(|_| "mtp_not_supported")?;
        let mut retained = retained.filter(|_| reused > 0);
        if let Some(entry) = retained.as_ref() {
            self.target.restore(&entry.state.1).map_err(|_| "native_generation_failed")?;
        }
        let mut prefill = PrefillProgress {
            cancellation: &mut self.cancellation,
            request: options.request_id,
            started,
            last_progress: Duration::ZERO,
            batch_rows: self.target.prefill_batch_rows(),
            callback: &mut on_progress,
        };
        let last = input.len() - 1;
        let mut snapshot = None;
        let logits = if let Some((image, slot)) = image {
            prefill.text(&mut self.target, &input[..slot])?;
            if slot == last {
                prefill.run(image.token_count() - 1, |rows| self.target.ingest_image_rows(image, rows))?;
                self.target.step_image_row(image, image.token_count() - 1)
            } else {
                prefill.run(image.token_count(), |rows| self.target.ingest_image_rows(image, rows))?;
                prefill.text(&mut self.target, &input[slot + 1..last])?;
                self.target.step(input[last])
            }
        } else {
            if boundary > reused {
                drop(retained.take());
                prefill.text(&mut self.target, &input[reused..boundary])?;
                snapshot = Some(self.target.snapshot().map_err(|_| "native_generation_failed")?);
            } else if boundary > 0 && boundary == reused {
                snapshot = retained.take().map(|entry| entry.state.1);
            }
            prefill.text(&mut self.target, &input[boundary.max(reused)..last])?;
            self.target.step(input[last])
        }
        .map_err(|_| "native_generation_failed")?;
        let result = collect_output(
            &mut self.target,
            &self.tokenizer,
            &mut self.sampler,
            &mut self.cancellation,
            &mut self.draft_stats,
            logits,
            expanded,
            options,
            started,
            on_text,
            on_progress,
        );
        let mut result = result?;
        result.reused_input_tokens = reused;
        if let (Some(state), Some(scope)) = (snapshot, options.cache_scope) {
            self.pending =
                Some(BoundaryEntry::new((use_draft, state), scope, options.context_tokens, &input[..boundary]));
        }
        Ok(result)
    }

    fn context_bytes(
        &self,
        context: usize,
        retain: bool,
    ) -> Option<usize> {
        let target = Session::context_memory_bytes(context, retain)?;
        if self.target.has_dspark() {
            target.checked_add(Session::dspark_memory_bytes(context, retain)?)
        } else {
            Some(target)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_output(
    target: &mut Session,
    tokenizer: &Tokenizer,
    mapper: &mut ProbabilityMapper,
    cancellation: &mut Option<CancelSignal>,
    draft_stats: &mut DraftStats,
    mut logits: Vec<f32>,
    input_tokens: usize,
    options: &GenerationOptions,
    started: Instant,
    mut on_text: Option<&mut TextCallback<'_>>,
    on_progress: Option<&mut ProgressCallback<'_>>,
) -> Result<GeneratedTokens, &'static str> {
    let mut previous = None;
    let mut rng = options.seed;
    let tokens = std::iter::from_fn(|| {
        Some((|| {
            let proposal = if let Some(anchor) = previous {
                let proposal = target.draft_propose(anchor).map_err(|_| "native_generation_failed")?;
                logits = target.step(anchor).map_err(|_| "native_generation_failed")?;
                proposal
            } else {
                None
            };
            let token = sample_with_mapper(&logits, &options.sampling, &mut rng, Some(mapper))?;
            if let Some(proposal) = proposal {
                draft_stats.proposed += 1;
                draft_stats.accepted += usize::from(proposal == token);
            }
            previous = Some(token);
            Ok(u64::from(token))
        })())
    });
    let mut decoder = tokenizer.decode_stream(false);
    let mut filter = StopFilter::new(options.stop.clone());
    let mut raw = String::new();
    let needs_text = on_text.is_some() || !options.stop.is_empty();
    let mut result = collect_tokens_controlled(
        tokens,
        input_tokens,
        options,
        VOCAB,
        started,
        &mut |token| {
            if needs_text && let Some(text) = decoder.step(token).map_err(|_| "decoding_failed")? {
                raw.push_str(&text);
                let (text, stopped) = filter.push(&text);
                if let Some(callback) = on_text.as_deref_mut() {
                    callback(&text)?;
                }
                return Ok(!stopped);
            }
            Ok(true)
        },
        on_progress,
        &mut || check_cancelled(cancellation, options.request_id),
    )?;
    if needs_text {
        let complete = tokenizer.decode(&result.generated_token_ids, false).map_err(|_| "decoding_failed")?;
        let (tail, stopped) = filter.push(complete.strip_prefix(&raw).ok_or("decoding_failed")?);
        if stopped {
            result.finish_reason = FinishReason::StopSequence;
        }
        if let Some(callback) = on_text {
            callback(&(tail + &filter.finish()))?;
        } else {
            filter.finish();
        }
        result.text = Some(filter.text().to_owned());
    }
    Ok(result)
}

impl GenerationBackend for InklingBackend {
    type PreparedImage = PreparedInklingImage;

    fn vocab_size(&self) -> usize {
        VOCAB
    }
    fn max_context_length(&self) -> Option<u32> {
        u32::try_from(self.target.max_context_length()).ok()
    }
    fn eos_token_ids(&self) -> &[u32] {
        EOS
    }

    fn supports_automatic_context_planning(&self) -> bool {
        true
    }
    fn automatic_context_length(
        &self,
        input: u32,
        maximum: u32,
        memory: usize,
    ) -> Option<u32> {
        let minimum = input.checked_add(1)?;
        let mut max = maximum.min(self.max_context_length()?);
        if minimum > max || self.context_bytes(minimum as usize, true)? > memory {
            return None;
        }
        let mut min = minimum;
        while min < max {
            let midpoint = min + (max - min).div_ceil(2);
            if self.context_bytes(midpoint as usize, true).is_some_and(|bytes| bytes <= memory) {
                min = midpoint;
            } else {
                max = midpoint - 1;
            }
        }
        Some(min)
    }

    fn tokenize(
        &mut self,
        text: &str,
        special: bool,
    ) -> Result<Vec<u32>, &'static str> {
        self.tokenizer.encode(text, special).map(|x| x.get_ids().to_vec()).map_err(|_| "tokenization_failed")
    }

    fn decode(
        &mut self,
        ids: &[u32],
        skip_special: bool,
    ) -> Result<String, &'static str> {
        self.tokenizer.decode(ids, skip_special).map_err(|_| "decoding_failed")
    }

    fn generate(
        &mut self,
        input: &[u32],
        options: &GenerationOptions,
        on_text: Option<&mut TextCallback<'_>>,
        on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        if input.contains(&IMAGE_TOKEN) {
            self.clear_prefix_cache();
            return Err("image_placeholder_requires_media");
        }
        let result = self.generate_inner(input, None, options, on_text, on_progress);
        self.target.reset();
        if result.is_err() {
            self.clear_prefix_cache();
        }
        result
    }

    fn cancelled(
        &mut self,
        request_id: u64,
    ) -> Result<bool, &'static str> {
        check_cancelled(&mut self.cancellation, request_id)
    }

    fn prefix_cache_supported(&self) -> bool {
        true
    }
    fn prefix_cache_key(&self) -> &'static str {
        "model-instance+context-tokens+cache-scope+mtp-enabled"
    }
    fn prefix_cache_plan(
        &self,
        input: &[u32],
        options: &GenerationOptions,
    ) -> usize {
        self.retained
            .as_ref()
            .filter(|entry| entry.state.0 == (options.mtp_drafts > 0))
            .map_or(0, |entry| entry.reusable(input, options.context_tokens, options.cache_scope.as_ref()))
    }
    fn publish_prefix_cache(&mut self) {
        self.retained = self.pending.take();
    }
    fn clear_prefix_cache(&mut self) {
        self.retained = None;
        self.pending = None;
    }
    fn prefix_cache_retained(&self) -> bool {
        self.retained.is_some() || self.pending.is_some()
    }

    fn supports_images(&self) -> bool {
        true
    }
    fn mtp_capabilities(&self) -> Option<MtpCapabilities> {
        self.target.has_dspark().then_some(MtpCapabilities {
            max_drafts: 1,
            recommended_drafts: 1,
            sampling: "native-target-draw-match-v1",
            verification: MTP_VERIFICATION,
        })
    }
    fn mtp_metadata(&self) -> Option<Value> {
        self.target.has_dspark().then(|| json!({"stats":self.draft_stats}))
    }
    fn image_geometry(&self) -> (u32, u32) {
        (20, 2)
    }
    fn image_family_profile(&self) -> Option<(&str, u32)> {
        Some(("inkling2-mini-native-v1", 1))
    }
    fn image_source_identity(&self) -> Option<(&str, Option<&str>)> {
        Some((inkling_artifact::SOURCE, Some(inkling_artifact::REVISION)))
    }

    fn image_artifact_identity(&self) -> Option<ImageArtifactIdentity<'_>> {
        Some(ImageArtifactIdentity {
            schema_version: 7,
            artifact_sha256: inkling_artifact::TARGET_SHA,
            config_sha256: &self.config_sha256,
            tokenizer_sha256: inkling_artifact::TOKENIZER_SHA,
            processor_sha256: inkling_artifact::PROCESSOR_SHA,
            template_sha256: inkling_artifact::TEMPLATE_SHA,
        })
    }

    fn runtime_identity(&self) -> Option<Value> {
        let identity = &self.runtime_identity;
        Some(json!({"backend":identity.backend, "device_name":identity.device_name,
            "apple_gpu_family":identity.apple_gpu_family, "gpu_core_count":identity.gpu_core_count, "supports_mxu":identity.supports_mxu}))
    }

    fn prepare_image(
        &mut self,
        input: Vec<u32>,
        encoded: &[u8],
        context: u32,
        request: u64,
    ) -> Result<(PreparedInklingImage, PreparedImageMetadata), &'static str> {
        self.clear_prefix_cache();
        require_active(&mut self.cancellation, request)?;
        self.target.reset();
        self.target.set_dspark_enabled(false).map_err(|_| "native_generation_failed")?;
        let mut slots = input.iter().enumerate().filter_map(|(i, &id)| (id == IMAGE_TOKEN).then_some(i));
        let slot = slots.next().ok_or("invalid_image_request")?;
        if slots.next().is_some()
            || slot == 0
            || input.get(slot - 1) != Some(&IMAGE_CONTENT)
            || input.get(slot + 1) != Some(&END_MESSAGE)
        {
            return Err("invalid_image_request");
        }
        let remaining = (context as usize).checked_sub(input.len()).ok_or("context_overflow")?;
        let image = self.target.encode_image(encoded, remaining).map_err(|_| "invalid_image_request")?;
        require_active(&mut self.cancellation, request)?;
        let grid = image.grid();
        let image_tokens = u32::try_from(image.token_count()).map_err(|_| "invalid_image_request")?;
        let metadata = PreparedImageMetadata {
            input_tokens: input.len() + image.token_count() - 1,
            image_tokens,
            grid,
            processed_dimensions: [grid[2] * 20, grid[1] * 20],
        };
        Ok((
            PreparedInklingImage {
                image,
                input,
                slot,
            },
            metadata,
        ))
    }

    fn generate_image(
        &mut self,
        prepared: PreparedInklingImage,
        options: &GenerationOptions,
        on_text: Option<&mut TextCallback<'_>>,
        on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        self.clear_prefix_cache();
        let result =
            self.generate_inner(&prepared.input, Some((&prepared.image, prepared.slot)), options, on_text, on_progress);
        self.target.reset();
        result
    }
}
