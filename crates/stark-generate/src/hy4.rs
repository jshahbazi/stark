// SPDX-License-Identifier: Apache-2.0
//! AngelSlim Q3_K through Stark's serialized generation worker.

mod source;
mod vision;

use std::{
    fs::File,
    io::{self, BufReader},
    path::Path,
    time::{Duration, Instant},
};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use stark_engine::{
    backends::metal::Metal,
    engine::{
        Engine,
        hy4::{DraftStats, ExpertStorage, ImageInput, MAX_DRAFTS, Options, PREFILL_CHUNK, SHARD, Session, VOCAB},
        hybrid::activity::InferenceActivity,
    },
};
use tokenizers::Tokenizer;

use self::vision::Hy4PreparedImage;
use crate::{
    cancel::CancelSignal,
    generation::{
        FinishReason, GeneratedTokens, GenerationBackend, GenerationOptions, HY4_VISION_PROFILE, ImageArtifactIdentity,
        MtpCapabilities, PreparedImageMetadata, ProgressCallback, Sampling, TextCallback, collect_tokens_controlled,
    },
    hybrid::{ProbabilityMapper, report_prefill_progress, sample_with_mapper},
    prefix_cache::{self, PrefixEntry},
    text_stop::StopFilter,
};

const EOS: &[u32] = &[120_025];
const TOKENIZER_SHA: &str = "2e3f3d22065c2be5765ce99ab2410411ae338b32ce809ea20dcdb61d00198e83";
const CONFIG_SHA: &str = "1f56ac1a3063f926e91ec43842bfdc0fd6d4401ef91bcdecd1fc27dd506ea968";
const PROCESSOR_SHA: &str = "cc5cc63da33e0ebf53b8ac035879982ba18c47965f82a5f341bc82d98759dc1d";
const TEMPLATE_SHA: &str = "11b12531403da5726b7e1f6003f479043b05999e6746d39af949cbcdfd715c35";
const ENGRAM_MAP_SHA: &str = "99ed7bbd800d9d9c087aa2cbbf0fcc497839c519045f6f3bf9aca177cc1769b5";

pub(crate) fn recognizes(path: &Path) -> bool {
    path.join(SHARD).is_file()
}

fn verify(
    path: &Path,
    expected: &str,
) -> Result<(), &'static str> {
    let artifact = File::open(path).map_err(|_| "incomplete_hy4_artifact")?;
    let mut digest = Sha256::new();
    io::copy(&mut BufReader::with_capacity(1024 * 1024, artifact), &mut digest).map_err(|_| "invalid_hy4_artifact")?;
    (format!("{:x}", digest.finalize()) == expected).then_some(()).ok_or("invalid_hy4_artifact")
}

fn check_cancelled(
    signal: &mut Option<CancelSignal>,
    request: u64,
) -> Result<bool, &'static str> {
    signal.as_mut().map_or(Ok(false), |signal| signal.check(request))
}

fn verify_switch(
    name: &str,
    error: &'static str,
    default: bool,
) -> Result<bool, &'static str> {
    match std::env::var(name).as_deref() {
        Ok("1") => Ok(true),
        Ok("0") => Ok(false),
        Err(std::env::VarError::NotPresent) => Ok(default),
        _ => Err(error),
    }
}

fn optional_bound(
    name: &str,
    bounds: std::ops::RangeInclusive<u32>,
    error: &'static str,
) -> Result<Option<u32>, &'static str> {
    match std::env::var(name) {
        Ok(value) => value.parse().ok().filter(|value| bounds.contains(value)).map(Some).ok_or(error),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => Err(error),
    }
}

pub(crate) struct Hy4Backend {
    native: Session,
    tokenizer: Tokenizer,
    cancellation: Option<CancelSignal>,
    mapper: ProbabilityMapper,
    retained: Option<PrefixEntry<u32>>,
    pending: Option<PrefixEntry<u32>>,
    stats: DraftStats,
    runtime: Value,
    chunk: usize,
}

impl Hy4Backend {
    pub(crate) fn load(
        path: &Path,
        cancellation: Option<CancelSignal>,
    ) -> Result<Self, &'static str> {
        let source = source::VerifiedSource::open(path)?;
        for (name, digest) in [
            ("config.json", CONFIG_SHA),
            ("tokenizer.json", TOKENIZER_SHA),
            ("processor_config.json", PROCESSOR_SHA),
            ("chat_template.jinja", TEMPLATE_SHA),
            ("engram_token_map.bin", ENGRAM_MAP_SHA),
        ] {
            verify(&path.join(name), digest)?;
        }
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json")).map_err(|_| "invalid_hy4_tokenizer")?;
        if tokenizer.get_vocab_size(true) != VOCAB {
            return Err("invalid_hy4_tokenizer");
        }
        let identity = Engine::<Metal>::new().map_err(|_| "native_backend_unavailable")?.metal_device_identity();
        let storage = match std::env::var("STARK_HY4_EXPERT_STORAGE").as_deref() {
            Ok("streamed") | Err(std::env::VarError::NotPresent) => ExpertStorage::Streamed,
            Ok("mapped") => ExpertStorage::Mapped,
            Ok("auto") => ExpertStorage::Auto,
            _ => return Err("invalid_hy4_expert_storage"),
        };
        let grouped = verify_switch("STARK_HY4_GROUPED_PREFILL", "invalid_hy4_grouped_prefill", true)?;
        let gpu_plan = verify_switch("STARK_HY4_GPU_PLAN", "invalid_hy4_gpu_plan", true)?;
        let chunk = match std::env::var("STARK_HY4_PREFILL_CHUNK").as_deref() {
            Ok("1024") | Err(std::env::VarError::NotPresent) => PREFILL_CHUNK,
            Ok("4096") => 4096,
            _ => return Err("invalid_hy4_prefill_chunk"),
        };
        let profile_prefill = verify_switch("STARK_HY4_PROFILE_PREFILL", "invalid_hy4_prefill_profile", false)?;
        let larger_arena = verify_switch("STARK_HY4_STREAM_ARENA_36GB", "invalid_hy4_stream_arena", false)?;
        let arena_gb = optional_bound("STARK_HY4_STREAM_ARENA_GB", 1..=128, "invalid_hy4_stream_arena")?;
        if larger_arena && arena_gb.is_some() {
            return Err("invalid_hy4_stream_arena");
        }
        let arena_bytes_requested = u64::from(arena_gb.unwrap_or(if larger_arena {
            36
        } else {
            0
        })) * 1_000_000_000;
        let context_limit = optional_bound("STARK_HY4_CONTEXT_LIMIT", 2..=u32::MAX, "invalid_hy4_context_limit")?;
        let demand_retention =
            verify_switch("STARK_HY4_STREAM_DEMAND_RETENTION", "invalid_hy4_stream_demand_retention", false)?;
        let routed_half = verify_switch("STARK_HY4_PREFILL_ROUTED_HALF", "invalid_hy4_prefill_routed_half", false)?;
        let dense_packed = verify_switch("STARK_HY4_PREFILL_DENSE_PACKED", "invalid_hy4_prefill_dense_packed", false)?;
        let resident_token_embeddings =
            verify_switch("STARK_HY4_EMBEDDINGS_RESIDENT", "invalid_hy4_embeddings_resident", false)?;
        if routed_half && !grouped {
            return Err("invalid_hy4_prefill_routed_half");
        }
        if demand_retention && !matches!(storage, ExpertStorage::Streamed) {
            return Err("invalid_hy4_stream_demand_retention");
        }
        if arena_bytes_requested != 0 && !matches!(storage, ExpertStorage::Streamed) {
            return Err("invalid_hy4_stream_arena");
        }
        let native = Session::load(
            path,
            path,
            Options {
                chunk,
                storage,
                grouped,
                gpu_plan,
                profile_prefill,
                arena_bytes_requested,
                demand_retention,
                routed_half,
                dense_packed,
                resident_token_embeddings,
                context_limit,
            },
        )?;
        source.verify_unchanged()?;
        let runtime = json!({
            "backend": identity.backend,
            "device_name": identity.device_name,
            "apple_gpu_family": identity.apple_gpu_family,
            "gpu_core_count": identity.gpu_core_count,
            "supports_mxu": identity.supports_mxu,
            "expert_storage": if native.mapped_experts() { "mapped" } else { "streamed" },
            "grouped_prefill_requested": grouped,
            "grouped_prefill_operands": if routed_half { "f16" } else if grouped { "f32" } else { "off" },
            "gpu_expert_plan": gpu_plan && native.mapped_experts(),
            "prefill_chunk_tokens": chunk,
            "prefill_stage_profile_requested": profile_prefill,
            "prefill_dense_packed_requested": dense_packed,
            "token_embeddings_resident": resident_token_embeddings,
            "streamed_expert_demand_retention": demand_retention,
            "streamed_expert_arena_bytes": native.arena_bytes(),
            "streamed_expert_arena_requested_bytes": arena_bytes_requested,
        });
        Ok(Self {
            native,
            tokenizer,
            cancellation,
            mapper: ProbabilityMapper::default(),
            retained: None,
            pending: None,
            stats: DraftStats::default(),
            runtime,
            chunk,
        })
    }

    fn generate_inner(
        &mut self,
        input: &[u32],
        options: &GenerationOptions,
        image: Option<&ImageInput<'_>>,
        mut on_text: Option<&mut TextCallback<'_>>,
        mut on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        let limit = options.context_tokens as usize;
        if input.is_empty()
            || input.len() >= limit
            || options.context_tokens > self.native.context_sizing().context_tokens
        {
            return Err("context_overflow");
        }
        let _activity = InferenceActivity::begin().map_err(|_| "inference_activity_unavailable")?;
        let started = Instant::now();
        let reused = if image.is_none() {
            self.prefix_cache_plan(input, options)
        } else {
            0
        };
        self.retained = None;
        self.pending = None;
        self.stats = DraftStats::default();
        if self.cancelled(options.request_id)? {
            return Err("generation_cancelled");
        }
        self.native.begin(input, options.context_tokens, reused, options.mtp_drafts, image)?;
        let mut last_reported = Duration::ZERO;
        for rows in input[reused..].chunks(self.chunk) {
            if self.cancelled(options.request_id)? {
                return Err("generation_cancelled");
            }
            self.native.prefill(rows.len())?;
            report_prefill_progress(started.elapsed(), &mut last_reported, on_progress.as_deref_mut())?;
        }

        let budget = (options.max_new_tokens as usize).min(limit - input.len());
        let mut previous = None;
        let mut verified = [0; MAX_DRAFTS + 1];
        let mut next_verified = 0;
        let mut verified_len = 0;
        let mut sampled = 0;
        let mut rng = options.seed;
        let native = &mut self.native;
        let mapper = &mut self.mapper;
        let greedy = matches!(options.sampling, Sampling::Greedy {});
        let drafts = options.mtp_drafts as usize;
        let tokens = std::iter::from_fn(|| {
            Some((|| {
                if let Some(token) = previous.take() {
                    native.append(token)?;
                }
                let token = if drafts == 0 || sampled == 0 {
                    let token = if greedy {
                        native.greedy()?
                    } else {
                        sample_with_mapper(native.logits()?, &options.sampling, &mut rng, Some(mapper))?
                    };
                    previous = Some(token);
                    token
                } else {
                    if next_verified == verified_len {
                        verified_len = native.verify((drafts + 1).min(budget - sampled), &mut verified)?;
                        next_verified = 0;
                    }
                    let token = verified[next_verified];
                    next_verified += 1;
                    token
                };
                sampled += 1;
                Ok(u64::from(token))
            })())
        });

        let mut decoder = self.tokenizer.decode_stream(false);
        let mut stop = StopFilter::new(options.stop.clone());
        let mut raw = String::new();
        let needs_text = on_text.is_some() || !options.stop.is_empty();
        let mut result = collect_tokens_controlled(
            tokens,
            input.len(),
            options,
            VOCAB,
            started,
            &mut |token| {
                if needs_text && let Some(text) = decoder.step(token).map_err(|_| "decoding_failed")? {
                    raw.push_str(&text);
                    let (text, stopped) = stop.push(&text);
                    if let Some(callback) = on_text.as_deref_mut() {
                        callback(&text)?;
                    }
                    return Ok(!stopped);
                }
                Ok(true)
            },
            on_progress,
            &mut || check_cancelled(&mut self.cancellation, options.request_id),
        )?;
        if needs_text {
            let full = self.tokenizer.decode(&result.generated_token_ids, false).map_err(|_| "decoding_failed")?;
            let (tail, stopped) = stop.push(full.strip_prefix(&raw).ok_or("decoding_failed")?);
            if stopped {
                result.finish_reason = FinishReason::StopSequence;
            }
            let tail = tail + &stop.finish();
            if let Some(callback) = on_text {
                callback(&tail)?;
            }
            result.text = Some(stop.text().to_owned());
        }
        self.stats = self.native.stats();
        result.reused_input_tokens = reused;
        let boundary = prefix_cache::chunk_boundary(input.len(), self.chunk);
        if let Some(scope) = options.cache_scope.filter(|_| image.is_none() && boundary > 0) {
            self.native.retain_prefix(boundary)?;
            self.pending =
                Some(PrefixEntry::with_chunk(options.mtp_drafts, scope, options.context_tokens, input, self.chunk));
        } else {
            self.native.clear();
        }
        Ok(result)
    }
}

impl GenerationBackend for Hy4Backend {
    type PreparedImage = Hy4PreparedImage;

    fn vocab_size(&self) -> usize {
        VOCAB
    }
    fn max_context_length(&self) -> Option<u32> {
        Some(self.native.context_sizing().context_tokens)
    }
    fn context_sizing_metadata(&self) -> Option<Value> {
        Some(json!(self.native.context_sizing()))
    }
    fn eos_token_ids(&self) -> &[u32] {
        EOS
    }
    fn tokenize(
        &mut self,
        text: &str,
        special: bool,
    ) -> Result<Vec<u32>, &'static str> {
        self.tokenizer
            .encode(text, special)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|_| "tokenization_failed")
    }
    fn decode(
        &mut self,
        ids: &[u32],
        skip_special: bool,
    ) -> Result<String, &'static str> {
        self.tokenizer.decode(ids, skip_special).map_err(|_| "decoding_failed")
    }
    fn cancelled(
        &mut self,
        request_id: u64,
    ) -> Result<bool, &'static str> {
        check_cancelled(&mut self.cancellation, request_id)
    }

    fn generate(
        &mut self,
        input: &[u32],
        options: &GenerationOptions,
        on_text: Option<&mut TextCallback<'_>>,
        on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        if input.contains(&self.native.vision_geometry().image_token) {
            self.clear_prefix_cache();
            return Err("image_placeholder_requires_media");
        }
        let result = self.generate_inner(input, options, None, on_text, on_progress);
        if result.is_err() {
            self.clear_prefix_cache();
        }
        result
    }

    fn supports_images(&self) -> bool {
        true
    }
    fn image_geometry(&self) -> (u32, u32) {
        let geometry = self.native.vision_geometry();
        (geometry.patch, geometry.merge)
    }
    fn image_family_profile(&self) -> Option<(&str, u32)> {
        Some((HY4_VISION_PROFILE, 1))
    }
    fn image_artifact_identity(&self) -> Option<ImageArtifactIdentity<'_>> {
        Some(ImageArtifactIdentity {
            schema_version: 7,
            artifact_sha256: source::ARTIFACT_SHA256,
            config_sha256: CONFIG_SHA,
            tokenizer_sha256: TOKENIZER_SHA,
            processor_sha256: PROCESSOR_SHA,
            template_sha256: TEMPLATE_SHA,
        })
    }
    fn image_source_identity(&self) -> Option<(&str, Option<&str>)> {
        Some((source::REPOSITORY, Some(source::REVISION)))
    }
    fn runtime_identity(&self) -> Option<Value> {
        Some(self.runtime.clone())
    }

    fn prepare_image(
        &mut self,
        input: Vec<u32>,
        encoded: &[u8],
        context: u32,
        request: u64,
    ) -> Result<(Self::PreparedImage, PreparedImageMetadata), &'static str> {
        self.clear_prefix_cache();
        if self.cancelled(request)? {
            return Err("generation_cancelled");
        }
        let result = Hy4PreparedImage::prepare(input, encoded, context, self.native.vision_geometry())?;
        if self.cancelled(request)? {
            return Err("generation_cancelled");
        }
        Ok(result)
    }

    fn generate_image(
        &mut self,
        image: Self::PreparedImage,
        options: &GenerationOptions,
        on_text: Option<&mut TextCallback<'_>>,
        on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        self.clear_prefix_cache();
        let result = self.generate_inner(&image.tokens, options, Some(&image.image_input()), on_text, on_progress);
        if result.is_err() {
            self.clear_prefix_cache();
        }
        result
    }

    fn mtp_capabilities(&self) -> Option<MtpCapabilities> {
        Some(MtpCapabilities {
            max_drafts: MAX_DRAFTS,
            recommended_drafts: MAX_DRAFTS,
            sampling: "greedy-target-match-v1",
            verification: "hy4-nextn-greedy-prefix-match-v1",
        })
    }
    fn effective_mtp_drafts(
        &self,
        options: &GenerationOptions,
    ) -> Result<u32, &'static str> {
        Ok(if matches!(options.sampling, Sampling::Greedy {}) {
            options.mtp_drafts
        } else {
            0
        })
    }
    fn mtp_metadata(&self) -> Option<Value> {
        Some(json!({"stats": self.stats}))
    }
    fn prefix_cache_supported(&self) -> bool {
        true
    }
    fn prefix_cache_chunk_tokens(&self) -> usize {
        self.chunk
    }
    fn prefix_cache_key(&self) -> &'static str {
        "model-instance+context-tokens+cache-scope+mtp-drafts"
    }
    fn prefix_cache_plan(
        &self,
        input: &[u32],
        options: &GenerationOptions,
    ) -> usize {
        self.retained
            .as_ref()
            .filter(|entry| entry.state == options.mtp_drafts)
            .map_or(0, |entry| entry.reusable(input, options.context_tokens, options.cache_scope.as_ref()))
    }
    fn publish_prefix_cache(&mut self) {
        self.retained = self.pending.take();
    }
    fn clear_prefix_cache(&mut self) {
        self.pending = None;
        self.retained = None;
        self.native.clear();
    }
    fn prefix_cache_retained(&self) -> bool {
        self.retained.is_some() || self.pending.is_some()
    }
}
