use std::{
    collections::{HashSet, VecDeque},
    fs::{File, OpenOptions},
    io::Read,
    os::unix::fs::OpenOptionsExt,
    path::Path,
    time::{Duration, Instant},
};

use serde_json::Value;
use stark_engine::{
    backends::metal::Metal,
    engine::{
        Engine,
        language_model::{
            LanguageModel,
            dense_mtp::{DenseMtpError, DenseMtpRound, DenseMtpSession},
            gemma4_assistant::Gemma4AssistantRetainedPrefix,
            state::LanguageModelState,
            stream::{PREFILL_CHUNK_TOKENS, SamplingMethod},
        },
        vision::{
            NativeVisionErrorKind, PreparedVisionImage, VisionLanguageModel,
            gemma3::{Gemma3VisionLanguageModel, PreparedGemma3Image},
            gemma4::{Gemma4Assistant, Gemma4VisionLanguageModel, PreparedGemmaImage},
        },
    },
};

mod gemma4_mtp;

use crate::{
    cancel::CancelSignal,
    generation::{
        ContextPlanningRequest, GeneratedTokens, GenerationBackend, GenerationOptions, ImageArtifactIdentity,
        PreparedImageMetadata, ProgressCallback, Sampling, TextCallback, admit_request_memory,
        collect_tokens_controlled, native_result_after_cleanup,
    },
    prefix_cache::{self, PrefixEntry},
    text_stop::StopFilter,
};

// Reuse must follow the engine's prefill chunk grid exactly.
const _: () = assert!(prefix_cache::PREFILL_CHUNK_TOKENS == PREFILL_CHUNK_TOKENS);
const GEMMA4_ASSISTANT_VERIFICATION_POLICY: &str = "gemma4-e2b-target-verify-v1";

type RetainedState = PrefixEntry<LanguageModelState<Metal>>;
type TextSnapshotState = prefix_cache::BoundaryEntry<(LanguageModelState<Metal>, &'static str)>;
type DenseRetainedState = prefix_cache::BoundaryEntry<(DenseMtpSession<Metal>, u32, &'static str)>;
type GemmaRetainedState = prefix_cache::BoundaryEntry<(Gemma4AssistantRetainedPrefix<Metal>, u32, &'static str)>;

fn admit_generation_memory(
    model: &LanguageModel<Metal>,
    input_tokens: usize,
    options: &GenerationOptions,
) -> Result<(), &'static str> {
    let input_tokens = u32::try_from(input_tokens).map_err(|_| "context_overflow")?;
    admit_memory_bytes_with_reclaim(
        model.generation_memory_bytes_for_request(
            input_tokens,
            options.context_tokens,
            options.max_new_tokens,
            options.mtp_drafts > 0,
        ),
        || model.reclaim_cached_memory().map_err(|_| "native_generation_failed"),
    )
}

fn admit_memory_bytes_with_reclaim(
    request_bytes: Option<usize>,
    reclaim: impl FnOnce() -> Result<usize, &'static str>,
) -> Result<(), &'static str> {
    let bytes = request_bytes.ok_or("resource_budget_exceeded")?;
    if admit_memory_bytes(Some(bytes)).is_ok() {
        return Ok(());
    }
    reclaim()?;
    // Only the observed headroom after owners have actually retired is used.
    admit_memory_bytes(Some(bytes))
}

fn admit_memory_bytes(request_bytes: Option<usize>) -> Result<(), &'static str> {
    let available = stark_resource_budget::available_memory_bytes().unwrap_or(0);
    admit_request_memory(request_bytes, available)
}

fn current_memory_budget() -> usize {
    stark_resource_budget::inference_memory_budget(stark_resource_budget::available_memory_bytes().unwrap_or(0))
}

fn record_accepted_prefix_round(
    counts: &mut [u64; 4],
    accepted: usize,
) -> bool {
    let Some(count) = counts.get_mut(accepted) else {
        return false;
    };
    *count += 1;
    true
}

fn disabled_dense_mtp_stats() -> Value {
    serde_json::json!({
        "enabled":false,"drafts":0,"rounds":0,"proposed":0,"accepted":0,
        "rounds_by_accepted_drafts":[0,0,0,0],"verified_rows":0,
        "draft_seconds":0.0,"verify_seconds":0.0,"rollback_seconds":0.0,
        "target_encode_seconds":0.0,"target_gpu_seconds":0.0,"target_readout_seconds":0.0,
    })
}

pub(crate) struct NativeBackend {
    model: NativeModel,
    idle_engine: Option<std::sync::Arc<Engine<Metal>>>,
    vocab_size: usize,
    eos_token_ids: Vec<u32>,
    cancellation: Option<CancelSignal>,
    runtime_identity: stark_engine::engine::MetalDeviceIdentity,
    /// Reusable state of the previous generation whose reply was delivered.
    prefix_cache: Option<RetainedState>,
    /// State of the last generation, published once its reply is flushed.
    pending_prefix: Option<RetainedState>,
    last_mtp: Option<Value>,
    dense_prefix_cache: Option<DenseRetainedState>,
    dense_pending_prefix: Option<DenseRetainedState>,
    text_snapshot_prefix_cache: Option<TextSnapshotState>,
    text_snapshot_pending_prefix: Option<TextSnapshotState>,
    gemma4_assistant: Option<Gemma4Assistant<Metal>>,
    gemma_prefix_cache: Option<GemmaRetainedState>,
    gemma_pending_prefix: Option<GemmaRetainedState>,
}

enum NativeModel {
    Text(Box<LanguageModel<Metal>>),
    Vision(Box<VisionLanguageModel<Metal>>),
    Gemma4Vision(Box<Gemma4VisionLanguageModel<Metal>>),
    Gemma3Vision(Box<Gemma3VisionLanguageModel<Metal>>),
}

macro_rules! dense_model_call {
    ($this:expr, |$model:ident| $operation:expr) => {
        match $this {
            NativeModel::Text($model) => $operation,
            NativeModel::Vision($model) => $operation,
            _ => Err(DenseMtpError::Invalid),
        }
    };
}

impl NativeModel {
    fn text_execution_graph(&self) -> &'static str {
        match self {
            Self::Vision(model) => model.text_execution_graph(),
            Self::Text(_) => "ordinary-text-v1",
            Self::Gemma3Vision(_) => "gemma3-reference-bf16-text-v1",
            Self::Gemma4Vision(_) => "gemma4-reference-bf16-text-v1",
        }
    }
    fn has_dense_mtp(&self) -> bool {
        match self {
            Self::Text(model) => model.has_dense_mtp(),
            Self::Vision(model) => model.has_dense_mtp(),
            _ => false,
        }
    }

    fn dense_generation_memory_bytes(
        &self,
        input: u32,
        context: u32,
        output: u32,
    ) -> Option<usize> {
        match self {
            Self::Text(model) => model.generation_memory_bytes(input, context, output),
            Self::Vision(model) => model.text_generation_memory_bytes(input, context, output),
            _ => None,
        }
    }

    fn generation_memory_bytes_reusing_dense_mtp(
        &self,
        session: &DenseMtpSession<Metal>,
        input: u32,
        context: u32,
        output: u32,
        reused: u32,
    ) -> Option<usize> {
        match self {
            Self::Text(model) => {
                model.generation_memory_bytes_reusing_dense_mtp(session, input, context, output, reused)
            },
            Self::Vision(model) => {
                model.generation_memory_bytes_reusing_dense_mtp(session, input, context, output, reused)
            },
            _ => None,
        }
    }

    fn reclaim_cached_memory(&self) -> Result<usize, &'static str> {
        match self {
            Self::Text(model) => model.reclaim_cached_memory(),
            Self::Vision(model) => model.reclaim_decoder_cached_memory(),
            Self::Gemma3Vision(model) => model.reclaim_decoder_cached_memory(),
            Self::Gemma4Vision(model) => model.reclaim_decoder_cached_memory(),
        }
        .map_err(|_| "native_generation_failed")
    }

    fn prefill_dense_mtp_controlled(
        &mut self,
        input: &[u32],
        capacity: u32,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<DenseMtpSession<Metal>, DenseMtpError<Metal>> {
        dense_model_call!(self, |model| model.prefill_dense_mtp_controlled(input, capacity, cancelled))
    }

    fn restore_dense_mtp_prefix(
        &mut self,
        session: &mut DenseMtpSession<Metal>,
        position: usize,
    ) -> Result<(), DenseMtpError<Metal>> {
        dense_model_call!(self, |model| model.restore_dense_mtp_prefix(session, position))
    }

    fn continue_prefill_dense_mtp_controlled(
        &mut self,
        session: &mut DenseMtpSession<Metal>,
        input: &[u32],
        capture: usize,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<(), DenseMtpError<Metal>> {
        dense_model_call!(self, |model| model.continue_prefill_dense_mtp_controlled(session, input, capture, cancelled))
    }

    fn dense_mtp_round(
        &mut self,
        session: &mut DenseMtpSession<Metal>,
        first: u32,
        drafts: usize,
        sampler: &mut dyn stark_engine::engine::hybrid::MtpSampler,
    ) -> Result<DenseMtpRound, DenseMtpError<Metal>> {
        dense_model_call!(self, |model| model.dense_mtp_round(session, first, drafts, sampler))
    }
}

enum PreparedImagePayload {
    Qwen(PreparedVisionImage),
    Gemma4(PreparedGemmaImage),
    Gemma3(PreparedGemma3Image),
}

pub(crate) struct NativePreparedImage {
    image: PreparedImagePayload,
    tokens: Box<[u64]>,
    token_start: usize,
}

impl NativeBackend {
    fn reclaim_text_cache(&self) -> Result<(), &'static str> {
        let NativeModel::Text(model) = &self.model else {
            return Ok(());
        };
        model.reclaim_cached_memory().map_err(|_| "native_generation_failed")?;
        Ok(())
    }

    fn retained_payload_bytes(&self) -> Option<usize> {
        let mut bytes = 0usize;
        for entry in [&self.prefix_cache, &self.pending_prefix].into_iter().flatten() {
            bytes = bytes.checked_add(entry.state.resident_payload_bytes()?)?;
        }
        for entry in [&self.dense_prefix_cache, &self.dense_pending_prefix].into_iter().flatten() {
            bytes = bytes.checked_add(entry.state.0.resident_payload_bytes()?)?;
        }
        for entry in [&self.text_snapshot_prefix_cache, &self.text_snapshot_pending_prefix].into_iter().flatten() {
            bytes = bytes.checked_add(entry.state.0.resident_payload_bytes()?)?;
        }
        Some(bytes)
    }

    fn warm_planning_candidate(
        &self,
        request: &ContextPlanningRequest<'_>,
    ) -> Option<(u32, usize)> {
        let input = request.input_token_ids?;
        let context = if request.options.mtp_drafts > 0 {
            self.dense_prefix_cache.as_ref()?.context_tokens()
        } else if let Some(entry) = &self.text_snapshot_prefix_cache {
            entry.context_tokens()
        } else {
            self.prefix_cache.as_ref()?.context_tokens()
        };
        if context <= request.input_tokens || context > request.maximum {
            return None;
        }
        let options = GenerationOptions {
            context_tokens: context,
            max_new_tokens: context - request.input_tokens,
            ..request.options.clone()
        };
        let reused = self.prefix_cache_plan(input, &options);
        (reused > 0).then_some((context, reused))
    }

    fn choose_context_plan(
        &self,
        request: &ContextPlanningRequest<'_>,
        budget: usize,
    ) -> Option<u32> {
        let NativeModel::Text(model) = &self.model else {
            return None;
        };
        let cold = model.automatic_context_length_for_request(
            request.input_tokens,
            request.maximum,
            budget,
            request.options.mtp_drafts > 0,
        );
        let warm = self.warm_planning_candidate(request).and_then(|(context, reused)| {
            let maximum = context - request.input_tokens;
            let bytes = if request.options.mtp_drafts > 0 {
                model.generation_memory_bytes_reusing_dense_mtp(
                    &self.dense_prefix_cache.as_ref()?.state.0,
                    request.input_tokens,
                    context,
                    maximum,
                    reused as u32,
                )
            } else if let Some(entry) = &self.text_snapshot_prefix_cache {
                model.generation_memory_bytes_reusing_text_prefix(
                    &entry.state.0,
                    request.input_tokens,
                    context,
                    maximum,
                    request.options.cache_scope.is_some(),
                    reused as u32,
                )
            } else {
                model.generation_memory_bytes_reusing_state(
                    &self.prefix_cache.as_ref()?.state,
                    request.input_tokens,
                    context,
                    maximum,
                    reused as u32,
                )
            }?;
            (bytes <= budget).then_some(context)
        });
        cold.max(warm)
    }

    pub(crate) fn load(
        path: &Path,
        cancellation: Option<CancelSignal>,
    ) -> Result<Self, &'static str> {
        Self::load_model(path, cancellation, false, false)
    }

    pub(crate) fn load_vision(
        path: &Path,
        cancellation: Option<CancelSignal>,
    ) -> Result<Self, &'static str> {
        Self::load_model(path, cancellation, true, false)
    }

    pub(crate) fn load_gemma4_text_validation(
        path: &Path,
        cancellation: Option<CancelSignal>,
    ) -> Result<Self, &'static str> {
        Self::load_model(path, cancellation, true, true)
    }

    fn load_model(
        path: &Path,
        cancellation: Option<CancelSignal>,
        vision: bool,
        validation_text_only: bool,
    ) -> Result<Self, &'static str> {
        if path.join("speculator").exists() {
            return Err("speculator_not_supported");
        }
        // The vision loader owns its bounded, no-follow selected configuration.
        // Do not reopen it here before admission or mix EOS/vocabulary from a
        // different path object into the model that was actually admitted.
        let config: Option<Value> = if vision {
            None
        } else {
            Some(
                serde_json::from_reader(File::open(path.join("config.json")).map_err(|_| "model_config_unavailable")?)
                    .map_err(|_| "invalid_model_config")?,
            )
        };
        let text_vocab = config
            .as_ref()
            .map(|config| {
                config
                    .get("decoder_config")
                    .and_then(|decoder| decoder.get("vocab_size"))
                    .and_then(Value::as_u64)
                    .filter(|size| *size > 0 && *size <= u32::MAX as u64)
                    .map(|size| size as usize)
                    .ok_or("invalid_model_config")
            })
            .transpose()?;
        let engine = Engine::<Metal>::new().map_err(|_| "metal_initialization_failed")?;
        let runtime_identity = engine.metal_device_identity();
        let model = if vision {
            // This bounded dispatch hint grants no metadata capability. Each
            // family loader reopens and admits its complete selected artifact.
            let mut selected = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_NOFOLLOW)
                .open(path.join("config.json"))
                .map_err(|_| "invalid_model_config")?;
            let mut bytes = Vec::new();
            Read::by_ref(&mut selected).take(1_048_577).read_to_end(&mut bytes).map_err(|_| "invalid_model_config")?;
            if bytes.len() > 1_048_576 {
                return Err("invalid_model_config");
            }
            let hint: Value = serde_json::from_slice(&bytes).map_err(|_| "invalid_model_config")?;
            if validation_text_only
                && hint.get("type").and_then(Value::as_str) != Some("Gemma4VisionLanguageModelConfig")
            {
                return Err("validation_family_not_supported");
            }
            if hint.get("type").and_then(Value::as_str) == Some("Gemma3VisionLanguageModelConfig") {
                NativeModel::Gemma3Vision(Box::new(
                    engine.load_gemma3_vision_language_model(path).map_err(|_| "model_load_failed")?,
                ))
            } else if hint.get("type").and_then(Value::as_str) == Some("Gemma4VisionLanguageModelConfig") {
                NativeModel::Gemma4Vision(Box::new(
                    if validation_text_only {
                        engine.load_gemma4_text_validation_model(path)
                    } else {
                        engine.load_gemma4_vision_language_model(path)
                    }
                    .map_err(|_| "model_load_failed")?,
                ))
            } else {
                let budget =
                    stark_resource_budget::available_memory_bytes().unwrap_or(0).saturating_sub(256 * 1024 * 1024);
                NativeModel::Vision(Box::new(
                    engine
                        .load_vision_language_model_with_parameter_budget(path, budget)
                        .map_err(|_| "model_load_failed")?,
                ))
            }
        } else {
            NativeModel::Text(Box::new(engine.load_language_model(path).map_err(|_| "model_load_failed")?))
        };
        let (tokenizer, stops) = match &model {
            NativeModel::Text(model) => (model.tokenizer(), &model.generation_config().stop_token_ids),
            NativeModel::Vision(model) => (model.tokenizer(), &model.generation_config().stop_token_ids),
            NativeModel::Gemma4Vision(model) => (model.tokenizer(), &model.generation_config().stop_token_ids),
            NativeModel::Gemma3Vision(model) => (model.tokenizer(), &model.generation_config().stop_token_ids),
        };
        let (vocab_size, codec_eos) = match &model {
            NativeModel::Vision(model) => (model.vocab_size() as usize, model.codec_eos_token()),
            NativeModel::Gemma4Vision(model) => (model.vocab_size() as usize, model.codec_eos_token()),
            NativeModel::Gemma3Vision(model) => (model.vocab_size() as usize, model.codec_eos_token()),
            NativeModel::Text(_) => (
                text_vocab.ok_or("invalid_model_config")?,
                config
                    .as_ref()
                    .and_then(|config| config.get("token_codec_config"))
                    .and_then(|codec| codec.get("eos_token"))
                    .and_then(Value::as_str),
            ),
        };
        let tokenizer_ids: HashSet<u32> = tokenizer.get_vocab(true).into_values().collect();
        if tokenizer_ids.iter().any(|id| *id as usize >= vocab_size) {
            return Err("invalid_model_config");
        }
        let eos_token_ids = crate::model_stops::model_stop_token_ids(
            stops,
            codec_eos,
            vocab_size,
            |token| tokenizer.token_to_id(token),
            |id| tokenizer_ids.contains(&id),
        )?;
        // Staging exercises the ordinary worker protocol before default
        // serving qualification. Exact sidecar/artifact admission is retained.
        #[cfg(feature = "gemma4-assistant-staging")]
        let gemma4_assistant = match &model {
            NativeModel::Gemma4Vision(model) if path.join("assistant").exists() => {
                admit_memory_bytes(Some(320 * 1024 * 1024))?;
                Some(model.load_assistant_component(&path.join("assistant")).map_err(|_| "assistant_load_failed")?)
            },
            _ => None,
        };
        #[cfg(not(feature = "gemma4-assistant-staging"))]
        let gemma4_assistant = None;
        Ok(Self {
            idle_engine: matches!(&model, NativeModel::Text(_)).then(|| engine.clone()),
            model,
            vocab_size,
            eos_token_ids,
            cancellation,
            runtime_identity,
            prefix_cache: None,
            pending_prefix: None,
            last_mtp: None,
            dense_prefix_cache: None,
            dense_pending_prefix: None,
            text_snapshot_prefix_cache: None,
            text_snapshot_pending_prefix: None,
            gemma4_assistant,
            gemma_prefix_cache: None,
            gemma_pending_prefix: None,
        })
    }
}

impl GenerationBackend for NativeBackend {
    type PreparedImage = NativePreparedImage;

    fn idle_maintenance_interval(&self) -> Option<Duration> {
        self.idle_engine.as_ref().map(|_| Duration::from_millis(500))
    }

    fn maintain_idle_queue(&mut self) -> Result<(), &'static str> {
        if let Some(engine) = &self.idle_engine {
            engine.maintain_idle_queue().map_err(|_| "native_idle_maintenance_failed")?;
        }
        Ok(())
    }

    fn mtp_capabilities(&self) -> Option<crate::generation::MtpCapabilities> {
        if self.gemma4_assistant.is_some() {
            return Some(crate::generation::MtpCapabilities {
                max_drafts: 3,
                recommended_drafts: 3,
                sampling: "exact-rejection-v1",
                verification: GEMMA4_ASSISTANT_VERIFICATION_POLICY,
            });
        }
        match self.model.has_dense_mtp() {
            true => Some(crate::generation::MtpCapabilities {
                max_drafts: 3,
                recommended_drafts: 3,
                sampling: "exact-rejection-v1",
                verification: "dense-qwen35-target-verify-v1",
            }),
            false => None,
        }
    }

    fn mtp_metadata(&self) -> Option<Value> {
        self.last_mtp.clone()
    }

    fn loaded_memory_metadata(&self) -> Option<Value> {
        if !cfg!(feature = "gemma4-assistant-staging") {
            return None;
        }
        let NativeModel::Gemma4Vision(model) = &self.model else {
            return None;
        };
        let head = self.gemma4_assistant.as_ref();
        let identity = head.and_then(|head| model.assistant_artifact_identity(head));
        Some(serde_json::json!({
            "image_tower_resident":model.image_tower_resident(),
            "logical_vision_parameter_bytes":model.logical_vision_parameter_bytes(),
            "resident_model_parameter_bytes":model.resident_parameter_bytes(),
            "metal_allocated_bytes":model.metal_allocated_bytes(),
            "target_config_sha256":model.artifact_identity().config_sha256,
            "target_weights_sha256":model.artifact_identity().artifact_sha256,
            "assistant_sha256":identity.map(|value| value.0),
            "assistant_file_bytes":identity.map(|value| value.1),
            "assistant_resident_storage_bytes":head.map(Gemma4Assistant::resident_storage_bytes),
            "assistant_billed_resident_bytes":head.and_then(|head| model.assistant_resident_memory_bytes(head)),
            "assistant_peak_loading_bytes":head.map(Gemma4Assistant::peak_loading_bytes),
            "assistant_numerical_policy":head.map(Gemma4Assistant::numerical_policy),
        }))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn max_context_length(&self) -> Option<u32> {
        match &self.model {
            NativeModel::Text(model) => model.max_context_length(),
            NativeModel::Vision(model) => model.max_context_length(),
            NativeModel::Gemma4Vision(model) => model.max_context_length(),
            NativeModel::Gemma3Vision(model) => model.max_context_length(),
        }
    }
    fn automatic_context_length(
        &self,
        input_tokens: u32,
        maximum: u32,
        memory_budget_bytes: usize,
    ) -> Option<u32> {
        match &self.model {
            NativeModel::Text(model) => model.automatic_context_length(input_tokens, maximum, memory_budget_bytes),
            // Vision uses its declared finite capacity and prepared image
            // accounting; this text-only planner does not model its encoder.
            NativeModel::Vision(_) | NativeModel::Gemma4Vision(_) | NativeModel::Gemma3Vision(_) => None,
        }
    }

    fn plan_context_length(
        &mut self,
        request: &ContextPlanningRequest<'_>,
        memory_budget_bytes: usize,
    ) -> Result<Option<u32>, &'static str> {
        if !matches!(self.model, NativeModel::Text(_)) {
            return Ok(None);
        }
        let mut budget = memory_budget_bytes;
        let warm = self.warm_planning_candidate(request);
        // A cache is only creditable after checking the actual IDs, scope,
        // context and selected MTP path. An inapplicable owner is released
        // before a fresh plan measures its available headroom.
        if self.prefix_cache_retained() && warm.is_none() {
            self.clear_prefix_cache();
            self.reclaim_text_cache()?;
            budget = current_memory_budget();
        }
        let mut best = self.choose_context_plan(request, budget);
        // Below the caller's maximum, reclaim free scratch pages (including
        // owners whose GPU retirement callback is still pending) before
        // deciding physical headroom is the limiting resource. A complete
        // warm allowance never flushes useful allocator cache pages.
        if best.is_none_or(|capacity| capacity < request.maximum) {
            self.reclaim_text_cache()?;
            budget = current_memory_budget();
            best = self.choose_context_plan(request, budget);
        }
        // Warm capacities are isolated points: a matching retained state may
        // fit where a smaller cold allocation does not. Compare that candidate
        // explicitly, rather than breaking the cold planner's monotonic search.
        let releaseable = self.retained_payload_bytes().unwrap_or(0);
        let NativeModel::Text(model) = &self.model else {
            unreachable!()
        };
        let preview = model.automatic_context_length_for_request(
            request.input_tokens,
            request.maximum,
            budget.saturating_add(releaseable),
            request.options.mtp_drafts > 0,
        );
        if releaseable > 0 && preview > best {
            self.clear_prefix_cache();
            self.reclaim_text_cache()?;
            // Speculative release bytes are never used for final admission.
            best = self.choose_context_plan(request, current_memory_budget());
        }
        Ok(best)
    }
    fn supports_automatic_context_planning(&self) -> bool {
        match &self.model {
            NativeModel::Text(model) => model.supports_automatic_context_planning(),
            NativeModel::Vision(_) | NativeModel::Gemma4Vision(_) | NativeModel::Gemma3Vision(_) => false,
        }
    }
    fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }
    fn tokenize(
        &mut self,
        text: &str,
        special: bool,
    ) -> Result<Vec<u32>, &'static str> {
        let tokenizer = match &self.model {
            NativeModel::Text(model) => model.tokenizer(),
            NativeModel::Vision(model) => model.tokenizer(),
            NativeModel::Gemma4Vision(model) => model.tokenizer(),
            NativeModel::Gemma3Vision(model) => model.tokenizer(),
        };
        tokenizer.encode(text, special).map(|encoding| encoding.get_ids().to_vec()).map_err(|_| "tokenization_failed")
    }
    fn decode(
        &mut self,
        ids: &[u32],
        skip_special: bool,
    ) -> Result<String, &'static str> {
        match &self.model {
            NativeModel::Text(model) => model.tokenizer(),
            NativeModel::Vision(model) => model.tokenizer(),
            NativeModel::Gemma4Vision(model) => model.tokenizer(),
            NativeModel::Gemma3Vision(model) => model.tokenizer(),
        }
        .decode(ids, skip_special)
        .map_err(|_| "decoding_failed")
    }
    fn generate(
        &mut self,
        input: &[u32],
        options: &GenerationOptions,
        on_text: Option<&mut TextCallback<'_>>,
        on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        self.last_mtp = None;
        if self.gemma4_assistant.is_some() && options.mtp_drafts > 0 {
            return self.generate_gemma4_assistant(input, options, on_text, on_progress);
        }
        if self.model.has_dense_mtp() && options.mtp_drafts > 0 {
            return self.generate_dense_mtp(input, options, on_text, on_progress);
        }
        self.last_mtp = (self.model.has_dense_mtp() || self.gemma4_assistant.is_some()).then(disabled_dense_mtp_stats);
        self.generate_inner(input, None, options, on_text, on_progress)
    }

    fn supports_images(&self) -> bool {
        match &self.model {
            NativeModel::Gemma4Vision(model) => model.image_tower_resident(),
            NativeModel::Vision(_) | NativeModel::Gemma3Vision(_) => true,
            NativeModel::Text(_) => false,
        }
    }

    fn image_geometry(&self) -> (u32, u32) {
        if matches!(self.model, NativeModel::Gemma3Vision(_)) {
            (14, 4)
        } else if matches!(self.model, NativeModel::Gemma4Vision(_)) {
            (16, 3)
        } else {
            (16, 2)
        }
    }

    fn runtime_identity(&self) -> Option<Value> {
        Some(serde_json::json!({
            "backend": self.runtime_identity.backend,
            "device_name": self.runtime_identity.device_name,
            "apple_gpu_family": self.runtime_identity.apple_gpu_family,
            "gpu_core_count": self.runtime_identity.gpu_core_count,
            "supports_mxu": self.runtime_identity.supports_mxu,
        }))
    }

    fn image_artifact_identity(&self) -> Option<ImageArtifactIdentity<'_>> {
        let identity = match &self.model {
            NativeModel::Vision(model) => model.artifact_identity(),
            NativeModel::Gemma4Vision(model) => model.artifact_identity(),
            NativeModel::Gemma3Vision(model) => model.artifact_identity(),
            NativeModel::Text(_) => return None,
        };
        Some(ImageArtifactIdentity {
            schema_version: identity.schema_version,
            artifact_sha256: &identity.artifact_sha256,
            config_sha256: &identity.config_sha256,
            tokenizer_sha256: &identity.tokenizer_sha256,
            processor_sha256: &identity.processor_sha256,
            template_sha256: &identity.template_sha256,
        })
    }

    fn image_family_profile(&self) -> Option<(&str, u32)> {
        match &self.model {
            NativeModel::Vision(model) => Some(model.family_profile()),
            NativeModel::Gemma4Vision(model) => Some(model.family_profile()),
            NativeModel::Gemma3Vision(model) => Some(model.family_profile()),
            NativeModel::Text(_) => None,
        }
    }

    fn image_source_identity(&self) -> Option<(&str, Option<&str>)> {
        match &self.model {
            NativeModel::Vision(model) => Some(model.source_identity()),
            NativeModel::Gemma4Vision(model) => Some(model.source_identity()),
            NativeModel::Gemma3Vision(model) => Some(model.source_identity()),
            NativeModel::Text(_) => None,
        }
    }

    fn cancelled(
        &mut self,
        request_id: u64,
    ) -> Result<bool, &'static str> {
        match self.cancellation.as_mut() {
            Some(signal) => signal.check(request_id),
            None => Ok(false),
        }
    }

    fn prepare_image(
        &mut self,
        input_token_ids: Vec<u32>,
        encoded: &[u8],
        context_tokens: u32,
        request_id: u64,
    ) -> Result<(NativePreparedImage, PreparedImageMetadata), &'static str> {
        if self.gemma4_assistant.is_some() {
            self.clear_prefix_cache();
        }
        let tokens = input_token_ids.into_iter().map(u64::from).collect::<Vec<_>>().into_boxed_slice();
        let mut channel_error = false;
        let cancellation = &mut self.cancellation;
        macro_rules! prepare_for_model {
            ($model:expr, $payload:path) => {{
                let model = $model;
                let image = match model.prepare_image_controlled(encoded, || match cancellation.as_mut() {
                    Some(signal) => match signal.check(request_id) {
                        Ok(cancelled) => cancelled,
                        Err(_) => {
                            channel_error = true;
                            true
                        },
                    },
                    None => false,
                }) {
                    Ok(image) => image,
                    Err(error) => {
                        return Err(if channel_error {
                            "cancel_channel_failed"
                        } else if error.kind() == NativeVisionErrorKind::Cancelled {
                            "generation_cancelled"
                        } else if model.is_fenced() {
                            "native_generation_failed"
                        } else {
                            "invalid_image_request"
                        });
                    },
                };
                let (tokens, token_start) =
                    model.prepare_prompt(tokens, &image, context_tokens).map_err(|_| "invalid_image_request")?;
                let metadata = PreparedImageMetadata {
                    input_tokens: tokens.len(),
                    image_tokens: image.image_token_count(),
                    grid: image.grid(),
                    processed_dimensions: image.processed_dimensions(),
                };
                Ok((
                    NativePreparedImage {
                        image: $payload(image),
                        tokens,
                        token_start,
                    },
                    metadata,
                ))
            }};
        }
        match &mut self.model {
            NativeModel::Vision(model) => prepare_for_model!(model, PreparedImagePayload::Qwen),
            NativeModel::Gemma4Vision(model) => prepare_for_model!(model, PreparedImagePayload::Gemma4),
            NativeModel::Gemma3Vision(model) => prepare_for_model!(model, PreparedImagePayload::Gemma3),
            NativeModel::Text(_) => Err("images_not_supported"),
        }
    }

    fn generate_image(
        &mut self,
        image: NativePreparedImage,
        options: &GenerationOptions,
        on_text: Option<&mut TextCallback<'_>>,
        on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        self.last_mtp = (self.model.has_dense_mtp() || self.gemma4_assistant.is_some()).then(disabled_dense_mtp_stats);
        self.generate_inner(&[], Some(image), options, on_text, on_progress)
    }

    fn prefix_cache_supported(&self) -> bool {
        match &self.model {
            NativeModel::Text(model) => model.has_dense_mtp() || model.prefix_reuse_supported(),
            NativeModel::Vision(model) => model.has_dense_mtp() || model.prefix_reuse_supported(),
            NativeModel::Gemma4Vision(model) => model.prefix_reuse_supported(),
            NativeModel::Gemma3Vision(model) => model.prefix_reuse_supported(),
        }
    }

    fn prefix_cache_key(&self) -> &'static str {
        if self.model.has_dense_mtp() || self.gemma4_assistant.is_some() {
            "model-instance+context-tokens+cache-scope+mtp-selection+execution-graph+precision"
        } else if matches!(
            &self.model,
            NativeModel::Vision(_) | NativeModel::Gemma3Vision(_) | NativeModel::Gemma4Vision(_)
        ) {
            "model-instance+context-tokens+cache-scope+execution-graph+precision"
        } else {
            prefix_cache::KEY
        }
    }

    fn prefix_cache_plan(
        &self,
        input: &[u32],
        options: &GenerationOptions,
    ) -> usize {
        if self.gemma4_assistant.is_some() && options.mtp_drafts > 0 {
            return self.gemma_prefix_cache.as_ref().map_or(0, |entry| {
                if entry.state.1 == options.mtp_drafts
                    && self.gemma4_assistant.as_ref().is_some_and(|head| entry.state.2 == head.numerical_policy())
                {
                    entry.reusable(input, options.context_tokens, options.cache_scope.as_ref())
                } else {
                    0
                }
            });
        }
        if self.model.has_dense_mtp() && options.mtp_drafts > 0 {
            return self.dense_prefix_cache.as_ref().map_or(0, |entry| {
                if entry.state.1 == options.mtp_drafts && entry.state.2 == self.model.text_execution_graph() {
                    entry.reusable(input, options.context_tokens, options.cache_scope.as_ref())
                } else {
                    0
                }
            });
        }
        // MTP0 uses the ordinary stream and its own target-only snapshot.
        // Predictor sessions cannot be restored into that execution path.
        if self.dense_prefix_cache.is_some() || self.gemma_prefix_cache.is_some() {
            return 0;
        }
        if let Some(entry) = &self.text_snapshot_prefix_cache {
            return if entry.state.1 == self.model.text_execution_graph() {
                entry.reusable(input, options.context_tokens, options.cache_scope.as_ref())
            } else {
                0
            };
        }
        if options.mtp_drafts > 0 {
            return 0;
        }
        self.prefix_cache
            .as_ref()
            .map_or(0, |entry| entry.reusable(input, options.context_tokens, options.cache_scope.as_ref()))
    }

    fn publish_prefix_cache(&mut self) {
        if let Some(entry) = self.gemma_pending_prefix.take() {
            self.gemma_prefix_cache = Some(entry);
        }
        if let Some(entry) = self.text_snapshot_pending_prefix.take() {
            self.text_snapshot_prefix_cache = Some(entry);
        }
        if let Some(entry) = self.dense_pending_prefix.take() {
            self.dense_prefix_cache = Some(entry);
        }
        if let Some(entry) = self.pending_prefix.take() {
            self.prefix_cache = Some(entry);
        }
    }

    fn clear_prefix_cache(&mut self) {
        self.gemma_prefix_cache = None;
        self.gemma_pending_prefix = None;
        self.prefix_cache = None;
        self.pending_prefix = None;
        self.dense_prefix_cache = None;
        self.dense_pending_prefix = None;
        self.text_snapshot_prefix_cache = None;
        self.text_snapshot_pending_prefix = None;
    }

    fn prefix_cache_retained(&self) -> bool {
        self.gemma_prefix_cache.is_some()
            || self.gemma_pending_prefix.is_some()
            || self.prefix_cache.is_some()
            || self.pending_prefix.is_some()
            || self.dense_prefix_cache.is_some()
            || self.dense_pending_prefix.is_some()
            || self.text_snapshot_prefix_cache.is_some()
            || self.text_snapshot_pending_prefix.is_some()
    }
}

impl NativeBackend {
    fn generate_dense_mtp(
        &mut self,
        input: &[u32],
        options: &GenerationOptions,
        mut on_text: Option<&mut TextCallback<'_>>,
        on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        let reused = self.prefix_cache_plan(input, options);
        let retained = self.dense_prefix_cache.take();
        self.prefix_cache = None;
        self.pending_prefix = None;
        self.dense_pending_prefix = None;
        self.text_snapshot_prefix_cache = None;
        self.text_snapshot_pending_prefix = None;
        let model = &mut self.model;
        if !model.has_dense_mtp() || input.is_empty() {
            return Err("mtp_not_supported");
        }
        let started = Instant::now();
        let cancellation = &mut self.cancellation;
        let mut cancelled = || match cancellation.as_mut() {
            Some(signal) => signal.check(options.request_id),
            None => Ok(false),
        };
        if cancelled()? {
            return Err("generation_cancelled");
        }
        let mut control_error = None;
        let mut control = || match cancelled() {
            Ok(false) => false,
            Ok(true) => {
                control_error = Some("generation_cancelled");
                true
            },
            Err(error) => {
                control_error = Some(error);
                true
            },
        };
        let capture = prefix_cache::reusable_prefill_rows(input.len());
        let mut session = if let Some(entry) = retained.filter(|_| reused > 0) {
            // A reused sparse cache can grow well beyond its previous mapped
            // rows when this request asks for more output. Undeclared contexts
            // were already admitted for their entire unchanged capacity.
            admit_memory_bytes_with_reclaim(
                model.generation_memory_bytes_reusing_dense_mtp(
                    &entry.state.0,
                    input.len() as u32,
                    options.context_tokens,
                    options.max_new_tokens,
                    reused as u32,
                ),
                || model.reclaim_cached_memory(),
            )?;
            let mut session = entry.state.0;
            model.restore_dense_mtp_prefix(&mut session, reused).map_err(|_| "native_generation_failed")?;
            model
                .continue_prefill_dense_mtp_controlled(&mut session, &input[reused..], capture, &mut control)
                .map_err(|_| control_error.unwrap_or("native_generation_failed"))?;
            session
        } else {
            admit_memory_bytes_with_reclaim(
                model.dense_generation_memory_bytes(
                    u32::try_from(input.len()).map_err(|_| "context_overflow")?,
                    options.context_tokens,
                    options.max_new_tokens,
                ),
                || model.reclaim_cached_memory(),
            )?;
            model
                .prefill_dense_mtp_controlled(input, options.context_tokens, &mut control)
                .map_err(|_| control_error.unwrap_or("native_generation_failed"))?
        };
        if cancelled()? {
            return Err("generation_cancelled");
        }
        // A reused prefix must prefill precisely the remaining suffix. Check
        // actual engine position before accepting reported cache reuse.
        if session.position() != input.len() {
            return Err("native_generation_failed");
        }
        let mut rng = options.seed;
        let mut mapper = crate::hybrid::ProbabilityMapper::default();
        let mut sampler = crate::hybrid::mtp::ExactMtpSampler::new(&options.sampling, &mut rng, &mut mapper);
        let first = sampler.sample_initial(session.logits())?;
        let mut root = first;
        let budget = (options.context_tokens as usize).saturating_sub(input.len()).min(options.max_new_tokens as usize);
        let mut remaining = budget;
        let mut pending = VecDeque::from([first]);
        let mut rounds = 0usize;
        let mut proposed = 0usize;
        let mut accepted = 0usize;
        // Exact aggregate accepted-prefix histogram: index j counts rounds
        // with exactly j accepted drafts. Dense MTP supports at most 3 drafts.
        let mut rounds_by_accepted_drafts = [0u64; 4];
        let mut verified_rows = 0usize;
        let mut draft_seconds = 0.0f64;
        let mut verify_seconds = 0.0f64;
        let mut rollback_seconds = 0.0f64;
        let mut target_encode_seconds = 0.0f64;
        let mut target_gpu_seconds = 0.0f64;
        let mut target_readout_seconds = 0.0f64;
        let mut failed = false;
        let tokenizer = match model {
            NativeModel::Text(model) => model.tokenizer().clone(),
            NativeModel::Vision(model) => model.tokenizer().clone(),
            _ => return Err("mtp_not_supported"),
        };
        let tokens = std::iter::from_fn(|| {
            if remaining == 0 || failed {
                return None;
            }
            if let Err(error) = cancelled().and_then(|stop| {
                if stop {
                    Err("generation_cancelled")
                } else {
                    Ok(())
                }
            }) {
                failed = true;
                return Some(Err(error));
            }
            if pending.is_empty() {
                let drafts = (options.mtp_drafts as usize).min(remaining.saturating_sub(1));
                match model.dense_mtp_round(&mut session, root, drafts, &mut sampler) {
                    Ok(round) => {
                        rounds += 1;
                        proposed += round.proposed;
                        accepted += round.accepted;
                        if !record_accepted_prefix_round(&mut rounds_by_accepted_drafts, round.accepted) {
                            failed = true;
                            return Some(Err("native_generation_failed"));
                        }
                        verified_rows += round.verified_rows;
                        draft_seconds += round.draft_seconds;
                        verify_seconds += round.verify_seconds;
                        rollback_seconds += round.rollback_seconds;
                        target_encode_seconds += round.target_encode_seconds;
                        target_gpu_seconds += round.target_gpu_seconds;
                        target_readout_seconds += round.target_readout_seconds;
                        pending.extend(round.accepted_drafts);
                        root = round.next_token;
                        pending.push_back(root);
                    },
                    Err(_) => {
                        failed = true;
                        return Some(Err("native_generation_failed"));
                    },
                }
            }
            // Cancellation may arrive while the synchronous native round is
            // executing. Suppress even its first queued token in that case.
            if let Err(error) = cancelled().and_then(|stop| {
                if stop {
                    Err("generation_cancelled")
                } else {
                    Ok(())
                }
            }) {
                failed = true;
                return Some(Err(error));
            }
            remaining -= 1;
            pending.pop_front().map(|token| Ok(u64::from(token)))
        });
        let mut decoder = tokenizer.decode_stream(false);
        let mut filter = StopFilter::new(options.stop.clone());
        let mut raw = String::new();
        let needs_text = on_text.is_some() || !options.stop.is_empty();
        // The iterator checks cancellation before every token, including queued
        // accepted drafts, and before starting another native round.
        let mut result = collect_tokens_controlled(
            tokens,
            input.len(),
            options,
            self.vocab_size,
            started,
            &mut |token| {
                if needs_text && let Some(text) = decoder.step(token).map_err(|_| "decoding_failed")? {
                    raw.push_str(&text);
                    let (text, stopped) = filter.push(&text);
                    if let Some(callback) = on_text.as_deref_mut() {
                        callback(&text)?
                    }
                    return Ok(!stopped);
                }
                Ok(true)
            },
            on_progress,
            &mut || Ok(false),
        )?;
        if cancelled()? {
            return Err("generation_cancelled");
        }
        result.reused_input_tokens = reused;
        if needs_text {
            let full = tokenizer.decode(&result.generated_token_ids, false).map_err(|_| "decoding_failed")?;
            let suffix = full.strip_prefix(&raw).ok_or("decoding_failed")?;
            let (tail, stopped) = filter.push(suffix);
            if stopped {
                result.finish_reason = crate::generation::FinishReason::StopSequence
            }
            let tail = tail + &filter.finish();
            if let Some(callback) = on_text {
                callback(&tail)?
            }
            result.text = Some(filter.text().to_owned());
        }
        self.last_mtp = Some(serde_json::json!({
            "enabled": options.mtp_drafts > 0, "drafts": options.mtp_drafts,
            "rounds": rounds, "proposed": proposed, "accepted": accepted,
            "rounds_by_accepted_drafts": rounds_by_accepted_drafts,
            "verified_rows": verified_rows,
            "draft_seconds": draft_seconds, "verify_seconds": verify_seconds,
            "rollback_seconds": rollback_seconds,
            "target_encode_seconds": target_encode_seconds,
            "target_gpu_seconds": target_gpu_seconds,
            "target_readout_seconds": target_readout_seconds,
        }));
        if let (Some(scope), Some(position)) = (options.cache_scope, session.prefix_position())
            && position == capture
            && position > 0
        {
            self.dense_pending_prefix = Some(prefix_cache::BoundaryEntry::new(
                (session, options.mtp_drafts, model.text_execution_graph()),
                scope,
                options.context_tokens,
                &input[..position],
            ));
        }
        Ok(result)
    }

    fn generate_inner(
        &mut self,
        input: &[u32],
        image: Option<NativePreparedImage>,
        options: &GenerationOptions,
        mut on_text: Option<&mut TextCallback<'_>>,
        on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        // Include native state allocation, stream setup and the first GPU wait.
        // The engine's iterator can encode one lookahead decode before returning
        // its first token, so this is a wall-clock prefill proxy, not GPU time.
        let started = Instant::now();
        let text_request = image.is_none();
        let reused = if text_request {
            self.prefix_cache_plan(input, options)
        } else {
            0
        };
        let cancellation = &mut self.cancellation;
        let mut cancelled = || match cancellation.as_mut() {
            Some(signal) => signal.check(options.request_id),
            None => Ok(false),
        };
        if cancelled()? {
            return Err("generation_cancelled");
        }
        let tokenizer = match &self.model {
            NativeModel::Text(model) => model.tokenizer().clone(),
            NativeModel::Vision(model) => model.tokenizer().clone(),
            NativeModel::Gemma4Vision(model) => model.tokenizer().clone(),
            NativeModel::Gemma3Vision(model) => model.tokenizer().clone(),
        };
        // Take the retained entry. A request that cannot reuse it drops it
        // before allocating, so two request states never coexist.
        let retained = self.prefix_cache.take();
        let snapshot_retained = self.text_snapshot_prefix_cache.take();
        self.gemma_prefix_cache = None;
        self.gemma_pending_prefix = None;
        self.dense_prefix_cache = None;
        self.dense_pending_prefix = None;
        self.pending_prefix = None;
        self.text_snapshot_pending_prefix = None;
        // Only one model variant is loaded. Drop an unusable boundary before
        // allocating a fresh state, including on image transactions.
        let snapshot_retained = snapshot_retained.filter(|_| reused > 0 && text_request);
        // Prepared tokens include the expanded image placeholder. Image
        // generation enters with an empty ordinary input slice.
        let admission_input_tokens = u32::try_from(image.as_ref().map_or(input.len(), |image| image.tokens.len()))
            .map_err(|_| "context_overflow")?;
        let mut state = match (&mut self.model, retained) {
            (NativeModel::Text(model), retained) if model.has_dense_mtp() => {
                drop(retained);
                let capture_prefix =
                    options.cache_scope.is_some() && prefix_cache::reusable_prefill_rows(input.len()) > 0;
                let memory = if let Some(entry) = snapshot_retained.as_ref() {
                    model.generation_memory_bytes_reusing_text_prefix(
                        &entry.state.0,
                        admission_input_tokens,
                        options.context_tokens,
                        options.max_new_tokens,
                        capture_prefix,
                        reused as u32,
                    )
                } else if capture_prefix {
                    model.generation_memory_bytes_with_text_prefix(
                        admission_input_tokens,
                        options.context_tokens,
                        options.max_new_tokens,
                    )
                } else {
                    model.ordinary_text_generation_memory_bytes(
                        admission_input_tokens,
                        options.context_tokens,
                        options.max_new_tokens,
                    )
                };
                admit_memory_bytes_with_reclaim(memory, || {
                    model.reclaim_cached_memory().map_err(|_| "native_generation_failed")
                })?;
                if let Some(entry) = snapshot_retained {
                    let mut state = entry.state.0;
                    model
                        .restore_text_prefix(&mut state, reused, Some(options.context_tokens), options.seed)
                        .map_err(|_| "native_generation_failed")?;
                    state
                } else {
                    model
                        .create_empty_state(Some(options.context_tokens), options.seed)
                        .map_err(|_| "native_generation_failed")?
                }
            },
            (NativeModel::Text(model), Some(entry)) if reused > 0 => {
                // Finite sparse admission may previously have covered only a
                // short request; an unchanged undeclared capacity was charged
                // in full when its retained state was created.
                admit_memory_bytes_with_reclaim(
                    model.generation_memory_bytes_reusing_state(
                        &entry.state,
                        input.len() as u32,
                        options.context_tokens,
                        options.max_new_tokens,
                        reused as u32,
                    ),
                    || model.reclaim_cached_memory().map_err(|_| "native_generation_failed"),
                )?;
                let mut state = entry.state;
                model
                    .rewind_state(&mut state, reused, Some(options.context_tokens), options.seed)
                    .map_err(|_| "native_generation_failed")?;
                state
            },
            (NativeModel::Text(model), retained) => {
                drop(retained);
                if image.is_none() {
                    admit_generation_memory(model, input.len(), options)?;
                }
                model
                    .create_empty_state(Some(options.context_tokens), options.seed)
                    .map_err(|_| "native_generation_failed")?
            },
            (NativeModel::Vision(model), retained) => {
                drop(retained);
                let capture_prefix = text_request
                    && options.cache_scope.is_some()
                    && prefix_cache::reusable_prefill_rows(input.len()) > 0
                    && model.prefix_reuse_supported();
                let memory_bytes = if let Some(entry) = snapshot_retained.as_ref() {
                    model.decoder_reused_text_memory_bytes(
                        &entry.state.0,
                        admission_input_tokens,
                        options.context_tokens,
                        options.max_new_tokens,
                        capture_prefix,
                        reused as u32,
                    )
                } else if capture_prefix {
                    model.decoder_text_prefix_memory_bytes(
                        admission_input_tokens,
                        options.context_tokens,
                        options.max_new_tokens,
                    )
                } else if text_request {
                    model.decoder_text_generation_memory_bytes(
                        admission_input_tokens,
                        options.context_tokens,
                        options.max_new_tokens,
                    )
                } else {
                    model.decoder_generation_memory_bytes(
                        admission_input_tokens,
                        options.context_tokens,
                        options.max_new_tokens,
                    )
                };
                admit_memory_bytes_with_reclaim(memory_bytes, || {
                    model.reclaim_decoder_cached_memory().map_err(|_| "native_generation_failed")
                })?;
                if let Some(entry) = snapshot_retained {
                    let mut state = entry.state.0;
                    model
                        .rewind_state(&mut state, reused, Some(options.context_tokens), options.seed)
                        .map_err(|_| "native_generation_failed")?;
                    state
                } else {
                    let created = if text_request {
                        model.create_text_state(Some(options.context_tokens), options.seed)
                    } else {
                        model.create_empty_state(Some(options.context_tokens), options.seed)
                    };
                    created.map_err(|_| "native_generation_failed")?
                }
            },
            (NativeModel::Gemma4Vision(model), retained) => {
                drop(retained);
                let capture_prefix = text_request
                    && options.cache_scope.is_some()
                    && prefix_cache::reusable_prefill_rows(input.len()) > 0
                    && model.prefix_reuse_supported();
                let memory_bytes = if let Some(entry) = snapshot_retained.as_ref() {
                    model.decoder_reused_text_memory_bytes(
                        &entry.state.0,
                        admission_input_tokens,
                        options.context_tokens,
                        options.max_new_tokens,
                        capture_prefix,
                        reused as u32,
                    )
                } else if capture_prefix {
                    model.decoder_text_prefix_memory_bytes(
                        admission_input_tokens,
                        options.context_tokens,
                        options.max_new_tokens,
                    )
                } else if text_request {
                    model.decoder_text_generation_memory_bytes(
                        admission_input_tokens,
                        options.context_tokens,
                        options.max_new_tokens,
                    )
                } else {
                    model.decoder_generation_memory_bytes(
                        admission_input_tokens,
                        options.context_tokens,
                        options.max_new_tokens,
                    )
                };
                let memory_bytes = if let Some(head) = &self.gemma4_assistant {
                    memory_bytes.and_then(|bytes| {
                        bytes
                            .checked_add(model.resident_parameter_bytes())?
                            .checked_add(model.assistant_resident_memory_bytes(head)?)
                    })
                } else {
                    memory_bytes
                };
                admit_memory_bytes_with_reclaim(memory_bytes, || {
                    model.reclaim_decoder_cached_memory().map_err(|_| "native_generation_failed")
                })?;
                if let Some(entry) = snapshot_retained {
                    let mut state = entry.state.0;
                    model
                        .rewind_state(&mut state, reused, Some(options.context_tokens), options.seed)
                        .map_err(|_| "native_generation_failed")?;
                    state
                } else {
                    model
                        .create_empty_state(Some(options.context_tokens), options.seed)
                        .map_err(|_| "native_generation_failed")?
                }
            },
            (NativeModel::Gemma3Vision(model), retained) => {
                drop(retained);
                let capture_prefix = text_request
                    && options.cache_scope.is_some()
                    && prefix_cache::reusable_prefill_rows(input.len()) > 0
                    && model.prefix_reuse_supported();
                let memory_bytes = if let Some(entry) = snapshot_retained.as_ref() {
                    model.decoder_reused_text_memory_bytes(
                        &entry.state.0,
                        admission_input_tokens,
                        options.context_tokens,
                        options.max_new_tokens,
                        capture_prefix,
                        reused as u32,
                    )
                } else if capture_prefix {
                    model.decoder_text_prefix_memory_bytes(
                        admission_input_tokens,
                        options.context_tokens,
                        options.max_new_tokens,
                    )
                } else if text_request {
                    model.decoder_text_generation_memory_bytes(
                        admission_input_tokens,
                        options.context_tokens,
                        options.max_new_tokens,
                    )
                } else {
                    model.decoder_generation_memory_bytes(
                        admission_input_tokens,
                        options.context_tokens,
                        options.max_new_tokens,
                    )
                };
                admit_memory_bytes_with_reclaim(memory_bytes, || {
                    model.reclaim_decoder_cached_memory().map_err(|_| "native_generation_failed")
                })?;
                if let Some(entry) = snapshot_retained {
                    let mut state = entry.state.0;
                    model
                        .rewind_state(&mut state, reused, Some(options.context_tokens), options.seed)
                        .map_err(|_| "native_generation_failed")?;
                    state
                } else {
                    model
                        .create_empty_state(Some(options.context_tokens), options.seed)
                        .map_err(|_| "native_generation_failed")?
                }
            },
        };
        if text_request && options.cache_scope.is_some() {
            let position = prefix_cache::reusable_prefill_rows(input.len());
            match &mut self.model {
                NativeModel::Text(model) if model.has_dense_mtp() => {
                    model.prepare_text_prefix_capture(&mut state, position).map_err(|_| "native_generation_failed")?
                },
                NativeModel::Vision(model) if model.prefix_reuse_supported() => {
                    model.prepare_text_prefix_capture(&mut state, position).map_err(|_| "native_generation_failed")?
                },
                _ => {},
            }
        }
        if text_request
            && options.cache_scope.is_some()
            && let NativeModel::Gemma4Vision(model) = &mut self.model
            && model.prefix_reuse_supported()
        {
            model
                .prepare_text_prefix_capture(&mut state, prefix_cache::reusable_prefill_rows(input.len()))
                .map_err(|_| "native_generation_failed")?;
        }
        if text_request
            && options.cache_scope.is_some()
            && let NativeModel::Gemma3Vision(model) = &mut self.model
            && model.prefix_reuse_supported()
        {
            model
                .prepare_text_prefix_capture(&mut state, prefix_cache::reusable_prefill_rows(input.len()))
                .map_err(|_| "native_generation_failed")?;
        }
        let prompt_ids = input;
        let input: Vec<u64> = input.iter().map(|id| u64::from(*id)).collect();
        let input_tokens = image.as_ref().map_or(input.len(), |image| image.tokens.len());
        let mut stream_options = match &self.model {
            NativeModel::Text(model) => model.default_stream_options(),
            NativeModel::Vision(model) => model.default_stream_options(),
            NativeModel::Gemma4Vision(model) => model.default_stream_options(),
            NativeModel::Gemma3Vision(model) => model.default_stream_options(),
        };
        stream_options.sampling_method = match &options.sampling {
            Sampling::Greedy {} => SamplingMethod::Greedy,
            Sampling::Stochastic {
                temperature,
                top_k,
                top_p,
                min_p,
            } => SamplingMethod::Stochastic {
                temperature: Some(*temperature),
                top_k: *top_k,
                top_p: *top_p,
                min_p: *min_p,
                repetition_penalty: None,
                suffix_repetition_length: None,
            },
        };
        let stream = match (&mut self.model, image) {
            (NativeModel::Text(model), None) => if model.has_dense_mtp() {
                model.stream_text(&input[reused..], &mut state, stream_options)
            } else {
                model.stream(&input[reused..], &mut state, stream_options)
            }
            .map_err(|_| "native_generation_failed")?,
            (NativeModel::Vision(model), None) => model
                .stream_text(&input[reused..], &mut state, stream_options)
                .map_err(|_| "native_generation_failed")?,
            (NativeModel::Gemma4Vision(model), None) => {
                model.stream(&input[reused..], &mut state, stream_options).map_err(|_| "native_generation_failed")?
            },
            (NativeModel::Gemma3Vision(model), None) => {
                model.stream(&input[reused..], &mut state, stream_options).map_err(|_| "native_generation_failed")?
            },
            (NativeModel::Vision(model), Some(image)) => {
                let PreparedImagePayload::Qwen(prepared) = image.image else {
                    return Err("invalid_image_request");
                };
                let mut channel_error = false;
                model
                    .stream_prepared_image_controlled(
                        image.tokens,
                        prepared,
                        image.token_start,
                        options.context_tokens,
                        &mut state,
                        stream_options,
                        || match cancelled() {
                            Ok(cancelled) => cancelled,
                            Err(_) => {
                                channel_error = true;
                                true
                            },
                        },
                    )
                    .map_err(|error| {
                        if channel_error {
                            "cancel_channel_failed"
                        } else if error.kind() == NativeVisionErrorKind::Cancelled {
                            "generation_cancelled"
                        } else {
                            "native_generation_failed"
                        }
                    })?
            },
            (NativeModel::Gemma4Vision(model), Some(image)) => {
                let PreparedImagePayload::Gemma4(prepared) = image.image else {
                    return Err("invalid_image_request");
                };
                let mut channel_error = false;
                model
                    .stream_prepared_image_controlled(
                        image.tokens,
                        prepared,
                        image.token_start,
                        options.context_tokens,
                        &mut state,
                        stream_options,
                        || match cancelled() {
                            Ok(value) => value,
                            Err(_) => {
                                channel_error = true;
                                true
                            },
                        },
                    )
                    .map_err(|error| {
                        if channel_error {
                            "cancel_channel_failed"
                        } else if error.kind() == NativeVisionErrorKind::Cancelled {
                            "generation_cancelled"
                        } else {
                            "native_generation_failed"
                        }
                    })?
            },
            (NativeModel::Gemma3Vision(model), Some(image)) => {
                let PreparedImagePayload::Gemma3(prepared) = image.image else {
                    return Err("invalid_image_request");
                };
                let mut channel_error = false;
                model
                    .stream_prepared_image_controlled(
                        image.tokens,
                        prepared,
                        image.token_start,
                        options.context_tokens,
                        &mut state,
                        stream_options,
                        || match cancelled() {
                            Ok(value) => value,
                            Err(_) => {
                                channel_error = true;
                                true
                            },
                        },
                    )
                    .map_err(|error| {
                        if channel_error {
                            "cancel_channel_failed"
                        } else if error.kind() == NativeVisionErrorKind::Cancelled {
                            "generation_cancelled"
                        } else {
                            "native_generation_failed"
                        }
                    })?
            },
            (NativeModel::Text(_), Some(_)) => return Err("images_not_supported"),
        };
        // Verify actual engine work, including restored-prefix suffix dispatch.
        if text_request && input_tokens.checked_sub(reused) != Some(stream.metrics().num_tokens_prefilled) {
            drop(stream);
            return Err("native_generation_failed");
        }
        let tokens = stream.map(|token| token.map_err(|_| "native_generation_failed"));
        let mut decoder = tokenizer.decode_stream(false);
        let mut filter = StopFilter::new(options.stop.clone());
        let mut raw = String::new();
        let needs_text = on_text.is_some() || !options.stop.is_empty();
        let result = collect_tokens_controlled(
            tokens,
            input_tokens,
            options,
            self.vocab_size,
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
            &mut cancelled,
        );
        // collect_tokens has dropped and drained the stream, including its lookahead.
        let fenced = match &self.model {
            NativeModel::Vision(model) => model.is_fenced(),
            NativeModel::Gemma4Vision(model) => model.is_fenced(),
            NativeModel::Gemma3Vision(model) => model.is_fenced(),
            NativeModel::Text(_) => false,
        };
        let mut result = native_result_after_cleanup(result, fenced)?;
        result.reused_input_tokens = reused;
        if needs_text {
            let full = tokenizer.decode(&result.generated_token_ids, false).map_err(|_| "decoding_failed")?;
            let suffix = full.strip_prefix(&raw).ok_or("decoding_failed")?;
            let (tail, stopped) = filter.push(suffix);
            if stopped {
                result.finish_reason = crate::generation::FinishReason::StopSequence;
            }
            let tail = tail + &filter.finish();
            if let Some(callback) = on_text {
                callback(&tail)?;
            }
            result.text = Some(filter.text().to_owned());
        }
        // Image transactions must never publish image-dependent decoder state.
        // Ordinary text on a vision artifact uses the same bounded prefix entry.
        if text_request && let Some(scope) = options.cache_scope {
            match &self.model {
                NativeModel::Text(model) if model.has_dense_mtp() => {
                    if let Some(position) = state.text_prefix_position()
                        && position > 0
                    {
                        self.text_snapshot_pending_prefix = Some(prefix_cache::BoundaryEntry::new(
                            (state, "ordinary-text-v1"),
                            scope,
                            options.context_tokens,
                            &prompt_ids[..position],
                        ));
                    }
                },
                NativeModel::Text(_) => {
                    self.pending_prefix = Some(PrefixEntry::new(state, scope, options.context_tokens, prompt_ids));
                },
                NativeModel::Vision(_) | NativeModel::Gemma4Vision(_) | NativeModel::Gemma3Vision(_) => {
                    if let Some(position) = state.text_prefix_position()
                        && position > 0
                    {
                        self.text_snapshot_pending_prefix = Some(prefix_cache::BoundaryEntry::new(
                            (state, self.model.text_execution_graph()),
                            scope,
                            options.context_tokens,
                            &prompt_ids[..position],
                        ));
                    }
                },
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod dense_mtp_stats_tests {
    use super::*;

    #[test]
    fn reclamation_failure_is_fatal_and_invalid_inventory_never_reclaims() {
        let calls = std::cell::Cell::new(0);
        assert_eq!(
            admit_memory_bytes_with_reclaim(None, || {
                calls.set(calls.get() + 1);
                Ok(0)
            }),
            Err("resource_budget_exceeded")
        );
        assert_eq!(calls.get(), 0);
        assert_eq!(
            admit_memory_bytes_with_reclaim(Some(0), || {
                calls.set(calls.get() + 1);
                Ok(0)
            }),
            Ok(())
        );
        assert_eq!(calls.get(), 0);
        assert_eq!(
            admit_memory_bytes_with_reclaim(Some(usize::MAX), || Err("native_generation_failed")),
            Err("native_generation_failed")
        );
        assert_eq!(
            admit_memory_bytes_with_reclaim(Some(usize::MAX), || {
                calls.set(calls.get() + 1);
                Ok(0)
            }),
            Err("resource_budget_exceeded")
        );
        assert_eq!(calls.get(), 1);
    }

    #[test]
    #[ignore = "requires an existing private converted full-attention model"]
    fn retained_native_plan_admits_warm_capacity_when_cold_minimum_does_not_fit() {
        let path = std::path::PathBuf::from(std::env::var_os("STARK_TEST_MODEL_PATH").expect("set private model path"));
        let mut backend = NativeBackend::load(&path, None).unwrap();
        let input = vec![2; 1025];
        let options = GenerationOptions {
            context_tokens: 4096,
            max_new_tokens: 1,
            seed: 42,
            sampling: Sampling::Greedy {},
            mtp_drafts: 0,
            eos_token_ids: backend.eos_token_ids.clone(),
            extra_stop_token_ids: Vec::new(),
            stop: Vec::new(),
            request_id: 1,
            cache_scope: Some([9; 16]),
        };
        backend.generate(&input, &options, None, None).unwrap();
        backend.publish_prefix_cache();
        let mut next = input.clone();
        next.push(2);
        let NativeModel::Text(model) = &backend.model else {
            panic!("expected text model")
        };
        let warm = model
            .generation_memory_bytes_reusing_state(
                &backend.prefix_cache.as_ref().unwrap().state,
                next.len() as u32,
                4096,
                4096 - next.len() as u32,
                1024,
            )
            .unwrap();
        assert!(model.automatic_context_length_for_request(next.len() as u32, 4096, warm, false).is_none());
        let request = ContextPlanningRequest {
            input_tokens: next.len() as u32,
            maximum: 4096,
            input_token_ids: Some(&next),
            options: &options,
        };
        assert_eq!(backend.plan_context_length(&request, warm).unwrap(), Some(4096));
        assert!(backend.prefix_cache_retained());
        let mismatch = GenerationOptions {
            cache_scope: Some([8; 16]),
            ..options.clone()
        };
        let request = ContextPlanningRequest {
            options: &mismatch,
            ..request
        };
        assert!(backend.plan_context_length(&request, current_memory_budget()).unwrap().is_some());
        assert!(!backend.prefix_cache_retained());
    }

    #[test]
    fn accepted_prefix_histogram_counts_rounds_by_exact_prefix_length() {
        let accepted_per_round = [0usize, 2, 1, 3, 2, 0];
        let mut rounds_by_accepted_drafts = [0u64; 4];
        for accepted in accepted_per_round {
            assert!(super::record_accepted_prefix_round(&mut rounds_by_accepted_drafts, accepted));
        }
        assert_eq!(rounds_by_accepted_drafts, [2, 1, 2, 1]);
        assert!(!super::record_accepted_prefix_round(&mut rounds_by_accepted_drafts, 4));
    }
}
