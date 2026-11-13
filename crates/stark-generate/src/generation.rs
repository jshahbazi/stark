use std::{
    cell::{Cell, RefCell},
    collections::HashSet,
    io::{BufRead, Read, Write},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
#[cfg(any(feature = "metal", test))]
pub(crate) use stark_resource_budget::admit_inference_memory as admit_request_memory;
use stark_resource_budget::{Budget, inference_memory_budget};

use crate::{
    media::{ImageLease, ImageReference, MediaStore, opaque_id, valid_id},
    prefix_cache::{self, CacheScope, RequestInput},
};

#[cfg(test)]
const TEST_WIRE_BYTES: usize = 8 * 1024 * 1024;
const MAX_EXTRA_STOPS: usize = 256;

/// Protocol policy independent of the optional native/Metal dependency.
/// Native backends translate their loaded model's capabilities at this boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MtpCapabilities {
    pub max_drafts: usize,
    pub recommended_drafts: usize,
    pub sampling: &'static str,
    pub verification: &'static str,
}

/// An omitted setting uses its default, while an explicitly present null is
/// invalid. Serde's ordinary Option<T> would otherwise conflate the two.
fn present_value<'de, T, D>(deserializer: D) -> Result<Option<T>, D::Error>
where
    T: Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    T::deserialize(deserializer).map(Some)
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Sampling {
    Greedy {},
    Stochastic {
        temperature: f32,
        #[serde(default)]
        top_k: Option<u32>,
        #[serde(default)]
        top_p: Option<f32>,
        #[serde(default)]
        min_p: Option<f32>,
    },
}

impl Default for Sampling {
    fn default() -> Self {
        Self::Greedy {}
    }
}

impl Sampling {
    fn validate(
        &self,
        vocab_size: usize,
    ) -> Result<(), &'static str> {
        if let Self::Stochastic {
            temperature,
            top_k,
            top_p,
            min_p,
        } = self
            && (!temperature.is_finite()
                || *temperature <= 0.0
                || !temperature.recip().is_finite()
                || *temperature > 100.0
                || top_k.is_some_and(|k| k == 0 || k as usize > vocab_size)
                || top_p.is_some_and(|p| !p.is_finite() || p <= 0.0 || p > 1.0)
                || min_p.is_some_and(|p| !p.is_finite() || !(0.0..=1.0).contains(&p)))
        {
            return Err("invalid_sampling_request");
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Tokenize {
        text: String,
        #[serde(default)]
        add_special_tokens: bool,
    },
    Decode {
        token_ids: Vec<u32>,
        #[serde(default)]
        skip_special_tokens: bool,
    },
    PlanGeneration {
        input_tokens: u32,
        context_tokens: u32,
        #[serde(default, deserialize_with = "present_value")]
        input_token_ids: Option<Vec<u32>>,
        #[serde(default, deserialize_with = "present_value")]
        cache_scope: Option<String>,
        #[serde(default, deserialize_with = "present_value")]
        mtp: Option<bool>,
        #[serde(default, deserialize_with = "present_value")]
        mtp_drafts: Option<u32>,
        #[serde(default)]
        sampling: Sampling,
    },
    Generate {
        #[serde(default)]
        rendered_prompt: Option<String>,
        #[serde(default)]
        input_token_ids: Option<Vec<u32>>,
        context_tokens: u32,
        max_new_tokens: u32,
        #[serde(default)]
        seed: u64,
        #[serde(default)]
        sampling: Sampling,
        #[serde(default, deserialize_with = "present_value")]
        mtp: Option<bool>,
        #[serde(default, deserialize_with = "present_value")]
        mtp_drafts: Option<u32>,
        #[serde(default)]
        stop_token_ids: Vec<u32>,
        #[serde(default)]
        stream: bool,
        #[serde(default)]
        stop: Vec<String>,
        #[serde(default)]
        request_id: u64,
        #[serde(default, deserialize_with = "present_value")]
        cache_scope: Option<String>,
    },
    ResetCache {},
    PrepareImage {
        rendered_prompt: String,
        image: ImageReference,
        context_tokens: u32,
        max_new_tokens: u32,
        #[serde(default)]
        request_id: u64,
    },
    GenerateImage {
        prepared_id: String,
        context_tokens: u32,
        max_new_tokens: u32,
        #[serde(default)]
        seed: u64,
        #[serde(default)]
        sampling: Sampling,
        #[serde(default)]
        stop_token_ids: Vec<u32>,
        #[serde(default)]
        stream: bool,
        #[serde(default)]
        stop: Vec<String>,
        #[serde(default)]
        request_id: u64,
    },
    DiscardImage {
        #[serde(default)]
        prepared_id: Option<String>,
    },
}

pub(crate) struct PreparedImageMetadata {
    pub input_tokens: usize,
    pub image_tokens: u32,
    pub grid: [u32; 3],
    pub processed_dimensions: [u32; 2],
}

pub(crate) struct ImageArtifactIdentity<'a> {
    pub schema_version: u32,
    pub artifact_sha256: &'a str,
    pub config_sha256: &'a str,
    pub tokenizer_sha256: &'a str,
    pub processor_sha256: &'a str,
    pub template_sha256: &'a str,
}

impl ImageArtifactIdentity<'_> {
    fn valid(&self) -> bool {
        matches!(self.schema_version, 2..=6)
            && [
                self.artifact_sha256,
                self.config_sha256,
                self.tokenizer_sha256,
                self.processor_sha256,
                self.template_sha256,
            ]
            .iter()
            .all(|value| {
                value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            })
    }
}

/// Bind each implementation profile to its artifact schema and actual geometry.
/// Capability recognition never implies that an artifact has been qualified.
fn image_family_limits(
    profile: &str,
    version: u32,
    schema: u32,
    geometry: (u32, u32),
) -> Option<(u32, u32, u32)> {
    let (expected_geometry, patches, tokens) = match (profile, version, schema) {
        ("qwen35-dense-bf16-merged-v1", 1, 2 | 3)
        | ("qwen35-dense-bf16-merged-v2", 2, 3)
        | ("qwen35-dense-affine-merged-v3", 3, 3)
        | ("qwen35-moe-affine-merged-v4", 4, 3)
        | ("qwen35-moe-bf16-merged-v5", 5, 3) => ((16, 2), 4096, 1024),
        ("qwen38-lily-retained-bf16-vision-v1", 1, 6) => ((16, 2), 65_536, 16_384),
        (
            "gemma4-e2b-bf16-v1"
            | "gemma4-e4b-bf16-v1"
            | "gemma4-31b-bf16-v1"
            | "gemma4-12b-bf16-v1"
            | "gemma4-26b-a4b-bf16-v1",
            1,
            4,
        ) => ((16, 3), 2520, 280),
        ("gemma3-siglip-bf16-v1", 1, 5) => ((14, 4), 4096, 256),
        _ => return None,
    };
    (geometry == expected_geometry).then_some((patches, tokens, patches * geometry.0 * geometry.0))
}

#[cfg(test)]
mod image_family_tests {
    use super::image_family_limits;

    #[test]
    fn family_handshakes_require_matching_schema_version_and_geometry() {
        for (profile, version, schema, geometry, expected) in [
            ("qwen35-dense-bf16-merged-v2", 2, 3, (16, 2), (4096, 1024, 1048576)),
            ("qwen35-dense-affine-merged-v3", 3, 3, (16, 2), (4096, 1024, 1048576)),
            ("qwen35-moe-affine-merged-v4", 4, 3, (16, 2), (4096, 1024, 1048576)),
            ("qwen35-moe-bf16-merged-v5", 5, 3, (16, 2), (4096, 1024, 1048576)),
            ("qwen38-lily-retained-bf16-vision-v1", 1, 6, (16, 2), (65_536, 16_384, 16_777_216)),
            ("gemma4-e2b-bf16-v1", 1, 4, (16, 3), (2520, 280, 645120)),
            ("gemma4-e4b-bf16-v1", 1, 4, (16, 3), (2520, 280, 645120)),
            ("gemma4-31b-bf16-v1", 1, 4, (16, 3), (2520, 280, 645120)),
            ("gemma4-12b-bf16-v1", 1, 4, (16, 3), (2520, 280, 645120)),
            ("gemma4-26b-a4b-bf16-v1", 1, 4, (16, 3), (2520, 280, 645120)),
            ("gemma3-siglip-bf16-v1", 1, 5, (14, 4), (4096, 256, 802816)),
        ] {
            assert_eq!(image_family_limits(profile, version, schema, geometry), Some(expected));
            assert_eq!(image_family_limits(profile, version + 1, schema, geometry), None);
            assert_eq!(image_family_limits(profile, version, 1, geometry), None);
            assert_eq!(image_family_limits(profile, version, schema, (1, 1)), None);
        }
        assert_eq!(image_family_limits("unimplemented", 1, 5, (14, 4)), None);
    }
}

struct PreparedRequest<P> {
    id: String,
    image: P,
    metadata: PreparedImageMetadata,
    context_tokens: u32,
    _lease: ImageLease,
}

fn valid_image_metadata(
    metadata: &PreparedImageMetadata,
    geometry: (u32, u32),
) -> bool {
    let (patch_size, merge_size) = geometry;
    let (max_patches, max_image_tokens) = match geometry {
        (16, 2) => (65_536, 16_384),
        (16, 3) => (2520, 280),
        (14, 4) if metadata.grid == [1, 64, 64] && metadata.processed_dimensions == [896, 896] => (4096, 256),
        _ => return false,
    };
    let [temporal, height, width] = metadata.grid;
    temporal == 1
        && height > 0
        && width > 0
        && height.is_multiple_of(merge_size)
        && width.is_multiple_of(merge_size)
        && height.checked_mul(width).is_some_and(|patches| {
            patches <= max_patches && patches / (merge_size * merge_size) == metadata.image_tokens
        })
        && metadata.processed_dimensions[0].checked_div(patch_size) == Some(width)
        && metadata.processed_dimensions[1].checked_div(patch_size) == Some(height)
        && metadata.processed_dimensions.iter().all(|dimension| dimension.is_multiple_of(patch_size))
        && metadata.input_tokens >= metadata.image_tokens as usize + 2
        && metadata.image_tokens > 0
        && metadata.image_tokens <= max_image_tokens
}

#[derive(Clone)]
pub(crate) struct GenerationOptions {
    pub context_tokens: u32,
    pub max_new_tokens: u32,
    pub seed: u64,
    pub sampling: Sampling,
    pub mtp_drafts: u32,
    pub eos_token_ids: Vec<u32>,
    pub extra_stop_token_ids: Vec<u32>,
    pub stop: Vec<String>,
    pub request_id: u64,
    /// Present only when prefix reuse is enabled and the host supplied a scope.
    pub cache_scope: Option<CacheScope>,
}

/// Actual request identity needed to inventory retained state safely. Legacy
/// count-only plans cannot establish a reusable token prefix.
pub(crate) struct ContextPlanningRequest<'a> {
    pub input_tokens: u32,
    pub maximum: u32,
    pub input_token_ids: Option<&'a [u32]>,
    pub options: &'a GenerationOptions,
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FinishReason {
    Eos,
    StopToken,
    StopSequence,
    Length,
    Context,
}

#[derive(Debug, Serialize)]
pub(crate) struct GeneratedTokens {
    pub generated_token_ids: Vec<u32>,
    pub sampled_tokens: usize,
    pub finish_reason: FinishReason,
    pub stop_token_id: Option<u32>,
    pub prefill_seconds: Option<f64>,
    pub decode_seconds: Option<f64>,
    pub text: Option<String>,
    /// Prompt tokens served from the retained prefix instead of prefill.
    pub reused_input_tokens: usize,
}

/// Content-free wall-clock observations of native token sampling. The first
/// sample belongs to prefill, including when that sample is a terminal token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GenerationProgress {
    pub sampled_tokens: usize,
    pub elapsed_seconds: f64,
    pub prefill_seconds: Option<f64>,
}

fn tokens_per_second(
    tokens: usize,
    seconds: Option<f64>,
) -> Option<f64> {
    seconds.filter(|seconds| *seconds > 0.0 && seconds.is_finite()).and_then(|seconds| {
        let rate = tokens as f64 / seconds;
        (tokens > 0 && rate.is_finite()).then_some(rate)
    })
}

/// Own the stream so its Drop drains native work before the result is observable.
pub(crate) fn collect_tokens_controlled(
    stream: impl Iterator<Item = Result<u64, &'static str>>,
    input_tokens: usize,
    options: &GenerationOptions,
    vocab_size: usize,
    started: Instant,
    on_token: &mut dyn FnMut(u32) -> Result<bool, &'static str>,
    on_progress: Option<&mut ProgressCallback<'_>>,
    cancelled: &mut dyn FnMut() -> Result<bool, &'static str>,
) -> Result<GeneratedTokens, &'static str> {
    collect_tokens_with_clock_and_callback(
        stream,
        input_tokens,
        options,
        vocab_size,
        || started.elapsed(),
        on_token,
        on_progress,
        cancelled,
    )
}

/// A stream's Drop can discover an error in its owned lookahead submission.
/// Check its model fence after draining, including when collection cancelled.
pub(crate) fn native_result_after_cleanup<T>(
    result: Result<T, &'static str>,
    fenced: bool,
) -> Result<T, &'static str> {
    if fenced {
        Err("native_generation_failed")
    } else {
        result
    }
}

#[cfg(test)]
fn collect_tokens_with_clock(
    stream: impl Iterator<Item = Result<u64, &'static str>>,
    input_tokens: usize,
    options: &GenerationOptions,
    vocab_size: usize,
    elapsed: impl FnMut() -> Duration,
) -> Result<GeneratedTokens, &'static str> {
    collect_tokens_with_clock_and_callback(
        stream,
        input_tokens,
        options,
        vocab_size,
        elapsed,
        &mut |_| Ok(true),
        None,
        &mut || Ok(false),
    )
}

fn collect_tokens_with_clock_and_callback(
    mut stream: impl Iterator<Item = Result<u64, &'static str>>,
    input_tokens: usize,
    options: &GenerationOptions,
    vocab_size: usize,
    mut elapsed: impl FnMut() -> Duration,
    on_token: &mut dyn FnMut(u32) -> Result<bool, &'static str>,
    mut on_progress: Option<&mut ProgressCallback<'_>>,
    cancelled: &mut dyn FnMut() -> Result<bool, &'static str>,
) -> Result<GeneratedTokens, &'static str> {
    let capacity = options.context_tokens as usize - input_tokens;
    let budget = capacity.min(options.max_new_tokens as usize);
    let mut result = GeneratedTokens {
        generated_token_ids: Vec::with_capacity(budget),
        sampled_tokens: 0,
        finish_reason: if capacity < options.max_new_tokens as usize {
            FinishReason::Context
        } else {
            FinishReason::Length
        },
        stop_token_id: None,
        prefill_seconds: None,
        decode_seconds: None,
        text: None,
        reused_input_tokens: 0,
    };
    let mut first_sample = None;
    let mut last_sample = None;
    for _ in 0..budget {
        if cancelled()? {
            return Err("generation_cancelled");
        }
        let token = stream.next().ok_or("invalid_generation_output")??;
        if cancelled()? {
            return Err("generation_cancelled");
        }
        let sample_time = elapsed();
        if token >= vocab_size as u64 {
            return Err("invalid_generation_output");
        }
        let token = token as u32;
        first_sample.get_or_insert(sample_time);
        last_sample = Some(sample_time);
        result.sampled_tokens += 1;
        if let Some(callback) = on_progress.as_deref_mut() {
            callback(GenerationProgress {
                sampled_tokens: result.sampled_tokens,
                elapsed_seconds: sample_time.as_secs_f64(),
                prefill_seconds: first_sample.map(|time| time.as_secs_f64()),
            })?;
        }
        if options.eos_token_ids.contains(&token) {
            result.finish_reason = FinishReason::Eos;
            result.stop_token_id = Some(token);
            break;
        }
        if options.extra_stop_token_ids.contains(&token) {
            result.finish_reason = FinishReason::StopToken;
            result.stop_token_id = Some(token);
            break;
        }
        result.generated_token_ids.push(token);
        if !on_token(token)? {
            result.finish_reason = FinishReason::StopSequence;
            break;
        }
    }
    result.prefill_seconds = first_sample.map(|time| time.as_secs_f64());
    if result.sampled_tokens > 1 {
        result.decode_seconds =
            last_sample.zip(first_sample).map(|(last, first)| last.saturating_sub(first).as_secs_f64());
    }
    drop(stream);
    Ok(result)
}

pub(crate) trait GenerationBackend {
    type PreparedImage;

    /// Optional owner-thread work between complete requests. No inference or
    /// prefix state may be changed by this callback.
    fn idle_maintenance_interval(&self) -> Option<Duration> {
        None
    }
    fn maintain_idle_queue(&mut self) -> Result<(), &'static str> {
        Ok(())
    }
    fn vocab_size(&self) -> usize;
    fn max_context_length(&self) -> Option<u32>;
    /// Select a safe context capacity for text generation. Backends with a
    /// finite model limit can use that limit; unknown unbounded models must
    /// provide a memory-aware implementation.
    fn automatic_context_length(
        &self,
        input_tokens: u32,
        maximum: u32,
        _memory_budget_bytes: usize,
    ) -> Option<u32> {
        let minimum = input_tokens.checked_add(1)?;
        let capacity = self.max_context_length()?.min(maximum);
        (capacity >= minimum).then_some(capacity)
    }
    fn plan_context_length(
        &mut self,
        request: &ContextPlanningRequest<'_>,
        memory_budget_bytes: usize,
    ) -> Result<Option<u32>, &'static str> {
        Ok(self.automatic_context_length(request.input_tokens, request.maximum, memory_budget_bytes))
    }
    fn supports_automatic_context_planning(&self) -> bool {
        false
    }
    fn eos_token_ids(&self) -> &[u32];
    fn tokenize(
        &mut self,
        text: &str,
        special: bool,
    ) -> Result<Vec<u32>, &'static str>;
    fn decode(
        &mut self,
        ids: &[u32],
        skip_special: bool,
    ) -> Result<String, &'static str>;
    fn generate(
        &mut self,
        input: &[u32],
        options: &GenerationOptions,
        on_text: Option<&mut TextCallback<'_>>,
        on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str>;
    fn supports_images(&self) -> bool {
        false
    }
    fn mtp_capabilities(&self) -> Option<MtpCapabilities> {
        None
    }
    fn default_mtp_drafts(&self) -> u32 {
        0
    }
    fn mtp_metadata(&self) -> Option<Value> {
        None
    }
    fn loaded_memory_metadata(&self) -> Option<Value> {
        None
    }
    /// Resolve sampling support before prefix planning and generation. Backends
    /// may fall back to serial, and the reply reports the effective selection.
    fn effective_mtp_drafts(
        &self,
        options: &GenerationOptions,
    ) -> Result<u32, &'static str> {
        Ok(options.mtp_drafts)
    }
    fn image_geometry(&self) -> (u32, u32) {
        (16, 2)
    }
    fn image_artifact_identity(&self) -> Option<ImageArtifactIdentity<'_>> {
        None
    }
    fn image_family_profile(&self) -> Option<(&str, u32)> {
        None
    }
    fn image_source_identity(&self) -> Option<(&str, Option<&str>)> {
        None
    }
    fn runtime_identity(&self) -> Option<Value> {
        None
    }
    fn cancelled(
        &mut self,
        _request_id: u64,
    ) -> Result<bool, &'static str> {
        Ok(false)
    }
    fn prepare_image(
        &mut self,
        _input_token_ids: Vec<u32>,
        _encoded: &[u8],
        _context_tokens: u32,
        _request_id: u64,
    ) -> Result<(Self::PreparedImage, PreparedImageMetadata), &'static str> {
        Err("images_not_supported")
    }
    fn generate_image(
        &mut self,
        _image: Self::PreparedImage,
        _options: &GenerationOptions,
        _on_text: Option<&mut TextCallback<'_>>,
        _on_progress: Option<&mut ProgressCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        Err("images_not_supported")
    }
    /// Whether the loaded model's text states can be rewound for reuse.
    fn prefix_cache_supported(&self) -> bool {
        false
    }
    /// Reused prompt tokens are a multiple of this prefill chunk length, itself
    /// a multiple of `PREFILL_CHUNK_TOKENS`.
    fn prefix_cache_chunk_tokens(&self) -> usize {
        prefix_cache::PREFILL_CHUNK_TOKENS
    }
    /// What a retained state must match, as reported in readiness.
    fn prefix_cache_key(&self) -> &'static str {
        prefix_cache::KEY
    }
    /// Prompt tokens the next text `generate` will reuse for `input`. A text
    /// generation with a scope retains its state; any other drops the entry.
    fn prefix_cache_plan(
        &self,
        _input: &[u32],
        _options: &GenerationOptions,
    ) -> usize {
        0
    }
    /// Make the state retained by the last successful generation reusable.
    /// Called only after its terminal reply has been written and flushed.
    fn publish_prefix_cache(&mut self) {}
    /// Drop retained and published state.
    fn clear_prefix_cache(&mut self) {}
    fn prefix_cache_retained(&self) -> bool {
        false
    }
}

pub(crate) type TextCallback<'a> = dyn FnMut(&str) -> Result<(), &'static str> + 'a;
pub(crate) type ProgressCallback<'a> = dyn FnMut(GenerationProgress) -> Result<(), &'static str> + 'a;

pub(crate) struct GenerationWorker<B: GenerationBackend> {
    backend: B,
    mtp: Option<MtpCapabilities>,
    default_mtp_drafts: u32,
    media: Option<MediaStore>,
    prepared: Option<PreparedRequest<B::PreparedImage>>,
    /// `Some(enabled)` when the host requested prefix reuse at startup.
    prefix_cache: Option<bool>,
    budget: Budget,
}

fn valid_ids(
    ids: &[u32],
    vocab_size: usize,
) -> bool {
    ids.iter().all(|id| (*id as usize) < vocab_size)
}

fn write_reply(
    writer: &mut impl Write,
    value: &impl Serialize,
) -> Result<(), &'static str> {
    serde_json::to_writer(&mut *writer, value).map_err(|_| "protocol_write_failed")?;
    writer.write_all(b"\n").map_err(|_| "protocol_write_failed")?;
    writer.flush().map_err(|_| "protocol_write_failed")
}

impl<B: GenerationBackend> GenerationWorker<B> {
    #[cfg(test)]
    pub(crate) fn new(backend: B) -> Result<Self, &'static str> {
        Self::new_with_options(backend, None, false)
    }

    pub(crate) fn new_with_options(
        backend: B,
        media: Option<MediaStore>,
        prefix_cache: bool,
    ) -> Result<Self, &'static str> {
        let mtp = backend.mtp_capabilities();
        let budget = Budget::from_environment()?;
        let default_drafts = backend.default_mtp_drafts();
        if backend.vocab_size() == 0
            || backend.vocab_size() > u32::MAX as usize
            || backend.max_context_length() == Some(0)
            || !valid_ids(backend.eos_token_ids(), backend.vocab_size())
            || (default_drafts > 0 && mtp.is_none_or(|caps| default_drafts as usize > caps.max_drafts))
            || mtp.is_some_and(|capabilities| {
                capabilities.max_drafts == 0
                    || capabilities.max_drafts > u32::MAX as usize
                    || !(1..=capabilities.max_drafts).contains(&capabilities.recommended_drafts)
                    || !matches!(capabilities.sampling, "exact-rejection-v1" | "native-target-draw-match-v1")
                    || capabilities.verification.is_empty()
                    || capabilities.verification.len() > 128
                    || !capabilities
                        .verification
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
            })
        {
            return Err("invalid_model_config");
        }
        let prefix_cache = prefix_cache.then(|| {
            let chunk = backend.prefix_cache_chunk_tokens();
            backend.prefix_cache_supported()
                && chunk >= prefix_cache::PREFILL_CHUNK_TOKENS
                && chunk.is_multiple_of(prefix_cache::PREFILL_CHUNK_TOKENS)
        });
        Ok(Self {
            backend,
            mtp,
            default_mtp_drafts: default_drafts,
            media,
            prepared: None,
            prefix_cache,
            budget,
        })
    }

    fn prefix_cache_enabled(&self) -> bool {
        self.prefix_cache == Some(true)
    }

    fn resolve_cache_scope(
        &self,
        scope: Option<String>,
    ) -> Result<Option<CacheScope>, &'static str> {
        match scope {
            None => Ok(None),
            Some(scope) if self.prefix_cache_enabled() => {
                prefix_cache::parse_scope(&scope).map(Some).ok_or("invalid_generation_request")
            },
            Some(_) => Err("invalid_generation_request"),
        }
    }

    fn resolve_mtp(
        &self,
        enabled: Option<bool>,
        drafts: Option<u32>,
    ) -> Result<u32, &'static str> {
        if self.backend.mtp_capabilities() != self.mtp || self.backend.default_mtp_drafts() != self.default_mtp_drafts {
            return Err("invalid_generation_output");
        }
        if enabled.is_some() && drafts.is_some() {
            return Err("invalid_generation_request");
        }
        let drafts = match (enabled, drafts) {
            (Some(true), None) => self.mtp.ok_or("mtp_not_supported")?.recommended_drafts as u32,
            (Some(false), None) => 0,
            (None, None) => self.default_mtp_drafts,
            (None, Some(drafts)) => drafts,
            (Some(_), Some(_)) => unreachable!("conflicting MTP settings rejected"),
        };
        if drafts > 0 {
            let capabilities = self.mtp.ok_or("mtp_not_supported")?;
            if drafts as usize > capabilities.max_drafts {
                return Err("invalid_generation_request");
            }
        }
        Ok(drafts)
    }

    fn validate_options(
        &self,
        options: &GenerationOptions,
    ) -> Result<(), &'static str> {
        if options.context_tokens == 0
            || options.max_new_tokens == 0
            || options.extra_stop_token_ids.len() > MAX_EXTRA_STOPS
            || options.stop.len() > 4
            || options.stop.iter().any(|s| s.is_empty() || s.chars().count() > 1024)
            || !valid_ids(&options.extra_stop_token_ids, self.backend.vocab_size())
            || options.extra_stop_token_ids.iter().collect::<HashSet<_>>().len() != options.extra_stop_token_ids.len()
        {
            return Err("invalid_generation_request");
        }
        if options.max_new_tokens as usize > self.budget.input_bytes / std::mem::size_of::<u32>() {
            return Err("resource_budget_exceeded");
        }
        self.resolve_mtp(None, Some(options.mtp_drafts))?;
        if self.backend.max_context_length().is_some_and(|max| options.context_tokens > max) {
            return Err("context_overflow");
        }
        options.sampling.validate(self.backend.vocab_size())
    }

    fn generate_completion(
        &mut self,
        input: &[u32],
        image: Option<B::PreparedImage>,
        input_tokens: usize,
        options: &GenerationOptions,
        stream: bool,
        output: &mut impl Write,
    ) -> Result<Value, &'static str> {
        let started = Instant::now();
        let requested_drafts = options.mtp_drafts;
        let effective_drafts = self.backend.effective_mtp_drafts(options)?;
        if effective_drafts > requested_drafts || (effective_drafts != 0 && effective_drafts != requested_drafts) {
            return Err("invalid_generation_output");
        }
        let effective_options;
        let options = if effective_drafts != requested_drafts {
            effective_options = GenerationOptions {
                mtp_drafts: effective_drafts,
                ..options.clone()
            };
            &effective_options
        } else {
            options
        };
        let output_token_budget = (options.context_tokens as usize - input_tokens).min(options.max_new_tokens as usize);
        let reused = match image {
            None => self.backend.prefix_cache_plan(input, options),
            Some(_) => 0,
        };
        if reused >= input_tokens || !reused.is_multiple_of(self.backend.prefix_cache_chunk_tokens()) {
            return Err("invalid_generation_output");
        }
        let report_reuse = self.prefix_cache_enabled();
        let progress = |sampled_tokens: usize, elapsed_seconds: f64, prefill_seconds: Option<f64>| {
            let mut frame = json!({"event":"progress","input_tokens":input_tokens,"sampled_tokens":sampled_tokens,
                "output_token_budget":output_token_budget,"elapsed_seconds":elapsed_seconds,
                "prefill_seconds":prefill_seconds});
            if report_reuse {
                frame["reused_input_tokens"] = json!(reused);
            }
            frame
        };
        if stream {
            write_reply(output, &progress(0, 0.0, None))?;
        }
        let mut emitted = String::new();
        let mut emitted_bytes = 0usize;
        let sampled_tokens = Cell::new(0usize);
        let output_byte_budget = self.budget.input_bytes;
        let generated = {
            // Text and progress callbacks share the same framed output sink.
            // Neither callback retains a borrow across the native invocation.
            let output = RefCell::new(&mut *output);
            let mut on_text = |text: &str| {
                if !text.is_empty() {
                    let reserved_ids = sampled_tokens
                        .get()
                        .min(output_byte_budget / std::mem::size_of::<u32>())
                        .saturating_mul(std::mem::size_of::<u32>());
                    if emitted_bytes
                        .checked_add(text.len())
                        .is_none_or(|size| size > output_byte_budget.saturating_sub(reserved_ids))
                    {
                        return Err("resource_budget_exceeded");
                    }
                    if stream {
                        write_reply(&mut **output.borrow_mut(), &json!({"event":"delta","text":text}))?;
                    }
                    emitted_bytes += text.len();
                    if stream {
                        emitted.push_str(text);
                    }
                }
                Ok(())
            };
            let mut on_progress = |value: GenerationProgress| {
                sampled_tokens.set(value.sampled_tokens);
                if stream {
                    let frame = progress(value.sampled_tokens, value.elapsed_seconds, value.prefill_seconds);
                    write_reply(&mut **output.borrow_mut(), &frame)?;
                }
                Ok(())
            };
            let on_text = Some(&mut on_text as &mut TextCallback<'_>);
            let on_progress = Some(&mut on_progress as &mut ProgressCallback<'_>);
            match image {
                Some(image) => self.backend.generate_image(image, options, on_text, on_progress)?,
                None => self.backend.generate(input, options, on_text, on_progress)?,
            }
        };
        if !valid_ids(&generated.generated_token_ids, self.backend.vocab_size())
            || generated.generated_token_ids.len() > self.budget.input_bytes / std::mem::size_of::<u32>()
            || generated.sampled_tokens > options.max_new_tokens as usize
            || input_tokens + generated.sampled_tokens > options.context_tokens as usize
            || generated.sampled_tokens
                != generated.generated_token_ids.len() + usize::from(generated.stop_token_id.is_some())
            || generated.reused_input_tokens != reused
        {
            return Err("invalid_generation_output");
        }
        let text = match &generated.text {
            Some(text) => text.clone(),
            None => self.backend.decode(&generated.generated_token_ids, false)?,
        };
        if text.len() > self.budget.input_bytes {
            return Err("resource_budget_exceeded");
        }
        if text
            .len()
            .checked_add(generated.generated_token_ids.len() * std::mem::size_of::<u32>())
            .is_none_or(|size| size > self.budget.input_bytes)
        {
            return Err("resource_budget_exceeded");
        }
        if stream {
            let suffix = text.strip_prefix(&emitted).ok_or("decoding_failed")?;
            if !suffix.is_empty() {
                write_reply(output, &json!({"event":"delta","text":suffix}))?;
            }
        }
        let mut reply = json!({"generated_token_ids":generated.generated_token_ids,"text":text,
            "input_tokens":input_tokens,"output_tokens":generated.generated_token_ids.len(),
            "sampled_tokens":generated.sampled_tokens,"finish_reason":generated.finish_reason,
            "stop_token_id":generated.stop_token_id,"generation_seconds":started.elapsed().as_secs_f64(),
            "prefill_seconds":generated.prefill_seconds,"decode_seconds":generated.decode_seconds,
            "input_tokens_per_second":tokens_per_second(input_tokens - reused, generated.prefill_seconds),
            "output_tokens_per_second":tokens_per_second(generated.sampled_tokens.saturating_sub(1), generated.decode_seconds),
            "timing_method":"first-token-wall-clock-v1"});
        if report_reuse {
            reply["reused_input_tokens"] = json!(reused);
        }
        if self.backend.mtp_capabilities() != self.mtp {
            return Err("invalid_generation_output");
        }
        if let Some(capabilities) = self.mtp {
            let mut metadata = self.backend.mtp_metadata().unwrap_or_else(|| json!({}));
            let fields = metadata.as_object_mut().ok_or("invalid_generation_output")?;
            let enabled = options.mtp_drafts > 0;
            for (name, expected) in [
                ("enabled", json!(enabled)),
                ("drafts", json!(options.mtp_drafts)),
                ("sampling", json!(capabilities.sampling)),
                ("verification", json!(capabilities.verification)),
            ] {
                if let Some(value) = fields.get(name) {
                    if *value != expected {
                        return Err("invalid_generation_output");
                    }
                } else {
                    fields.insert(name.to_owned(), expected);
                }
            }
            if effective_drafts != requested_drafts {
                fields.insert("requested_drafts".to_owned(), json!(requested_drafts));
                fields.insert("fallback_reason".to_owned(), json!("unsupported_sampling"));
            }
            reply["mtp"] = metadata;
        }
        Ok(reply)
    }

    fn tokenize(
        &mut self,
        text: &str,
        special: bool,
    ) -> Result<Vec<u32>, &'static str> {
        let ids = self.backend.tokenize(text, special)?;
        if ids.len() > self.budget.input_bytes / std::mem::size_of::<u32>() {
            return Err("resource_budget_exceeded");
        }
        if !valid_ids(&ids, self.backend.vocab_size()) {
            return Err("invalid_tokenization_output");
        }
        Ok(ids)
    }

    fn handle(
        &mut self,
        request: Request,
        output: &mut impl Write,
    ) -> Result<Value, &'static str> {
        match request {
            Request::Tokenize {
                text,
                add_special_tokens,
            } => {
                if text.len() > self.budget.input_bytes {
                    return Err("resource_budget_exceeded");
                }
                let ids = self.tokenize(&text, add_special_tokens)?;
                Ok(json!({"token_ids":ids,"token_count":ids.len()}))
            },
            Request::Decode {
                token_ids,
                skip_special_tokens,
            } => {
                if token_ids.len() > self.budget.input_bytes / std::mem::size_of::<u32>() {
                    return Err("resource_budget_exceeded");
                }
                if !valid_ids(&token_ids, self.backend.vocab_size()) {
                    return Err("invalid_decoding_request");
                }
                let text = self.backend.decode(&token_ids, skip_special_tokens)?;
                if text.len() > self.budget.input_bytes {
                    return Err("resource_budget_exceeded");
                }
                Ok(json!({"text":text,"token_count":token_ids.len()}))
            },
            Request::PlanGeneration {
                input_tokens,
                context_tokens,
                input_token_ids,
                cache_scope,
                mtp,
                mtp_drafts,
                sampling,
            } => {
                if !self.backend.supports_automatic_context_planning() {
                    return Err("context_planning_unavailable");
                }
                let token_limit = (self.budget.input_bytes / std::mem::size_of::<u32>()).min(u32::MAX as usize) as u32;
                let maximum = input_tokens
                    .saturating_add(token_limit)
                    .min(context_tokens)
                    .min(self.backend.max_context_length().unwrap_or(u32::MAX));
                if input_tokens == 0 || input_tokens > token_limit || maximum <= input_tokens {
                    return Err("invalid_generation_request");
                }
                if input_token_ids
                    .as_ref()
                    .is_some_and(|ids| ids.len() != input_tokens as usize || !valid_ids(ids, self.backend.vocab_size()))
                {
                    return Err("invalid_generation_request");
                }
                let mut options = GenerationOptions {
                    context_tokens: maximum,
                    max_new_tokens: maximum - input_tokens,
                    seed: 0,
                    sampling,
                    mtp_drafts: self.resolve_mtp(mtp, mtp_drafts)?,
                    eos_token_ids: self.backend.eos_token_ids().to_vec(),
                    extra_stop_token_ids: Vec::new(),
                    stop: Vec::new(),
                    request_id: 0,
                    cache_scope: self.resolve_cache_scope(cache_scope)?,
                };
                self.validate_options(&options)?;
                let effective = self.backend.effective_mtp_drafts(&options)?;
                if effective > options.mtp_drafts || (effective != 0 && effective != options.mtp_drafts) {
                    return Err("invalid_generation_output");
                }
                options.mtp_drafts = effective;
                let available = stark_resource_budget::available_memory_bytes().unwrap_or(0);
                // Use the same remaining-memory contract as execution's
                // allocation recheck, including its transient host reserve.
                let memory_budget = inference_memory_budget(available);
                let capacity = self
                    .backend
                    .plan_context_length(
                        &ContextPlanningRequest {
                            input_tokens,
                            maximum,
                            input_token_ids: input_token_ids.as_deref(),
                            options: &options,
                        },
                        memory_budget,
                    )?
                    .ok_or("resource_budget_exceeded")?;
                if capacity <= input_tokens || capacity > maximum {
                    return Err("resource_budget_exceeded");
                }
                Ok(json!({"context_tokens":capacity,
                    "max_new_tokens":capacity - input_tokens,
                    "context_capacity_known":self.backend.max_context_length().is_some(),
                    "budget_kind":"memory"}))
            },
            Request::Generate {
                rendered_prompt,
                input_token_ids,
                context_tokens,
                max_new_tokens,
                seed,
                sampling,
                mtp,
                mtp_drafts,
                stop_token_ids,
                stream,
                stop,
                request_id,
                cache_scope,
            } => {
                let cache_scope = self.resolve_cache_scope(cache_scope)?;
                let options = GenerationOptions {
                    context_tokens,
                    max_new_tokens,
                    seed,
                    sampling,
                    mtp_drafts: self.resolve_mtp(mtp, mtp_drafts)?,
                    eos_token_ids: self.backend.eos_token_ids().to_vec(),
                    extra_stop_token_ids: stop_token_ids,
                    stop,
                    request_id,
                    cache_scope,
                };
                self.validate_options(&options)?;
                let input = match (rendered_prompt, input_token_ids) {
                    (Some(prompt), None) => {
                        if prompt.len() > self.budget.input_bytes {
                            return Err("resource_budget_exceeded");
                        }
                        self.tokenize(&prompt, false)?
                    },
                    (None, Some(ids)) => {
                        if ids.len() > self.budget.input_bytes / std::mem::size_of::<u32>() {
                            return Err("resource_budget_exceeded");
                        }
                        ids
                    },
                    _ => return Err("invalid_generation_request"),
                };
                if input.is_empty() || !valid_ids(&input, self.backend.vocab_size()) {
                    return Err("invalid_generation_request");
                }
                if input.len() >= context_tokens as usize {
                    return Err("context_overflow");
                }
                if input.len().checked_add(max_new_tokens as usize).is_none_or(|total| total > context_tokens as usize)
                {
                    return Err("context_overflow");
                }
                self.generate_completion(&input, None, input.len(), &options, stream, output)
            },
            Request::PrepareImage {
                rendered_prompt,
                image,
                context_tokens,
                max_new_tokens,
                request_id,
            } => {
                if rendered_prompt.len() > self.budget.input_bytes {
                    return Err("resource_budget_exceeded");
                }
                if !self.backend.supports_images() || self.media.is_none() {
                    return Err("images_not_supported");
                }
                let options = GenerationOptions {
                    context_tokens,
                    max_new_tokens,
                    seed: 0,
                    sampling: Sampling::default(),
                    mtp_drafts: 0,
                    eos_token_ids: self.backend.eos_token_ids().to_vec(),
                    extra_stop_token_ids: Vec::new(),
                    stop: Vec::new(),
                    request_id,
                    cache_scope: None,
                };
                self.validate_options(&options)?;
                if self.backend.cancelled(request_id)? {
                    return Err("generation_cancelled");
                }
                let tokens = self.tokenize(&rendered_prompt, false)?;
                if tokens.is_empty() {
                    return Err("invalid_image_request");
                }
                // The one source image placeholder cannot shrink. Reject a
                // known overflow before opening bytes or decoding the image.
                if tokens.len().checked_add(max_new_tokens as usize).is_none_or(|total| total > context_tokens as usize)
                {
                    return Err("context_overflow");
                }
                let staged = self.media.as_ref().ok_or("images_not_supported")?.read(&image)?;
                if self.backend.cancelled(request_id)? {
                    return Err("generation_cancelled");
                }
                let (image, metadata) =
                    self.backend.prepare_image(tokens, &staged.bytes, context_tokens, request_id)?;
                if self.backend.cancelled(request_id)? {
                    return Err("generation_cancelled");
                }
                let limits = self.backend.image_family_profile().and_then(|(profile, version)| {
                    image_family_limits(
                        profile,
                        version,
                        self.backend.image_artifact_identity()?.schema_version,
                        self.backend.image_geometry(),
                    )
                });
                if !valid_image_metadata(&metadata, self.backend.image_geometry())
                    || limits.is_none_or(|(patches, tokens, _)| {
                        metadata.grid[1].checked_mul(metadata.grid[2]).is_none_or(|count| count > patches)
                            || metadata.image_tokens > tokens
                    })
                {
                    return Err("invalid_image_preparation_output");
                }
                if metadata
                    .input_tokens
                    .checked_add(max_new_tokens as usize)
                    .is_none_or(|total| total > context_tokens as usize)
                {
                    return Err("context_overflow");
                }
                let id = opaque_id()?;
                let reply = json!({"prepared_id":id,"input_tokens":metadata.input_tokens,
                    "image_tokens":metadata.image_tokens,"grid":metadata.grid,
                    "processed_dimensions":metadata.processed_dimensions});
                self.prepared = Some(PreparedRequest {
                    id,
                    image,
                    metadata,
                    context_tokens,
                    _lease: staged.lease,
                });
                Ok(reply)
            },
            Request::GenerateImage {
                prepared_id,
                context_tokens,
                max_new_tokens,
                seed,
                sampling,
                stop_token_ids,
                stream,
                stop,
                request_id,
            } => {
                let prepared = self.prepared.take().ok_or("invalid_prepared_image")?;
                if !valid_id(&prepared_id) || prepared.id != prepared_id {
                    return Err("invalid_prepared_image");
                }
                if prepared.context_tokens != context_tokens {
                    return Err("invalid_generation_request");
                }
                let options = GenerationOptions {
                    context_tokens,
                    max_new_tokens,
                    seed,
                    sampling,
                    mtp_drafts: 0,
                    eos_token_ids: self.backend.eos_token_ids().to_vec(),
                    extra_stop_token_ids: stop_token_ids,
                    stop,
                    request_id,
                    cache_scope: None,
                };
                self.validate_options(&options)?;
                if prepared
                    .metadata
                    .input_tokens
                    .checked_add(max_new_tokens as usize)
                    .is_none_or(|total| total > context_tokens as usize)
                {
                    return Err("context_overflow");
                }
                if self.backend.cancelled(request_id)? {
                    return Err("generation_cancelled");
                }
                self.generate_completion(
                    &[],
                    Some(prepared.image),
                    prepared.metadata.input_tokens,
                    &options,
                    stream,
                    output,
                )
            },
            Request::ResetCache {} => {
                if !self.prefix_cache_enabled() {
                    return Err("invalid_request");
                }
                self.backend.clear_prefix_cache();
                Ok(json!({"cache_reset":true}))
            },
            Request::DiscardImage {
                prepared_id,
            } => {
                let prepared = self.prepared.take();
                if prepared_id
                    .as_ref()
                    .is_some_and(|id| !valid_id(id) || prepared.as_ref().is_some_and(|value| value.id != *id))
                {
                    return Err("invalid_prepared_image");
                }
                Ok(json!({"discarded":prepared.is_some()}))
            },
        }
    }

    fn process_line(
        &mut self,
        bytes: &[u8],
        output: &mut impl Write,
    ) -> Result<(), &'static str> {
        let request = stark_resource_budget::check_json_structure(bytes, self.budget)
            .and_then(|()| serde_json::from_slice::<Request>(bytes).map_err(|_| "invalid_request"));
        if !matches!(&request, Ok(Request::GenerateImage { .. } | Request::DiscardImage { .. })) {
            self.prepared = None;
        }
        // Image transactions never use or populate the retained text prefix.
        if !matches!(
            &request,
            Ok(Request::Tokenize { .. }
                | Request::Decode { .. }
                | Request::Generate { .. }
                | Request::PlanGeneration { .. })
        ) {
            self.backend.clear_prefix_cache();
        }
        let generate = matches!(&request, Ok(Request::Generate { .. }));
        let result = request.and_then(|request| self.handle(request, output));
        if result.is_err() {
            self.prepared = None;
            self.backend.clear_prefix_cache();
        }
        match result {
            Ok(value) => {
                let result = write_reply(output, &value);
                if result.is_err() {
                    self.prepared = None;
                    self.backend.clear_prefix_cache();
                } else if generate {
                    // Publish only state whose terminal reply reached the host.
                    self.backend.publish_prefix_cache();
                }
                result
            },
            Err(code) => {
                if code == "protocol_write_failed" {
                    return Err(code);
                }
                write_reply(output, &json!({"error":code}))?;
                if matches!(
                    code,
                    "native_generation_failed"
                        | "invalid_generation_output"
                        | "invalid_tokenization_output"
                        | "invalid_image_preparation_output"
                        | "decoding_failed"
                        | "cancel_channel_failed"
                ) {
                    Err("native_operation_failed")
                } else {
                    Ok(())
                }
            },
        }
    }
}

/// Wait without extending the retained prefix's original idle deadline on
/// maintenance ticks. Returning input/EOF immediately avoids a poll-sized
/// request delay; backends without maintenance retain their blocking behavior.
fn wait_for_request<B: GenerationBackend>(
    backend: &mut B,
    input: &mut impl RequestInput,
) -> Result<(), &'static str> {
    wait_for_request_with_clock(backend, input, Instant::now)
}

fn wait_for_request_with_clock<B: GenerationBackend>(
    backend: &mut B,
    input: &mut impl RequestInput,
    mut now: impl FnMut() -> Instant,
) -> Result<(), &'static str> {
    let interval = backend.idle_maintenance_interval().filter(|interval| !interval.is_zero());
    let cache_limit = backend.prefix_cache_retained().then_some(Duration::from_secs(prefix_cache::IDLE_SECONDS));
    let started = now();
    let mut cache_remaining = cache_limit;
    loop {
        let timeout = match (interval, cache_remaining) {
            (Some(tick), Some(cache)) => tick.min(cache),
            (Some(tick), None) => tick,
            (None, Some(cache)) => cache,
            (None, None) => return Ok(()),
        };
        if input.wait_readable(timeout).map_err(|_| "protocol_read_failed")? {
            if cache_remaining.is_some()
                && cache_limit.is_some_and(|limit| now().saturating_duration_since(started) >= limit)
            {
                backend.clear_prefix_cache();
            }
            return Ok(());
        }
        if let Some(remaining) = cache_remaining {
            let elapsed = now().saturating_duration_since(started);
            if remaining <= timeout || cache_limit.is_some_and(|limit| elapsed >= limit) {
                backend.clear_prefix_cache();
                cache_remaining = None;
            } else {
                cache_remaining = cache_limit.map(|limit| limit.saturating_sub(elapsed));
            }
        }
        if interval.is_some() {
            backend.maintain_idle_queue()?;
            // Maintenance itself can take time, including a slow device wake.
            // Recompute before waiting again, even if input is already ready.
            if cache_remaining.is_some() {
                cache_remaining =
                    cache_limit.map(|limit| limit.saturating_sub(now().saturating_duration_since(started)));
                if cache_remaining.is_some_and(|remaining| remaining.is_zero()) {
                    backend.clear_prefix_cache();
                    cache_remaining = None;
                }
            }
        }
    }
}

pub(crate) fn run_lines<B: GenerationBackend>(
    worker: &mut GenerationWorker<B>,
    input: &mut impl RequestInput,
    output: &mut impl Write,
) -> Result<(), &'static str> {
    let mut readiness = json!({"ready":true,"engine":"stark","version":env!("CARGO_PKG_VERSION"),
        "mode":"text-generation-v1","operations":["tokenize","decode","generate"],
        "state_policy":"fresh-per-request","vocab_size":worker.backend.vocab_size(),
        "max_context_tokens":worker.backend.max_context_length().unwrap_or(u32::MAX),
        "max_new_tokens":worker.backend.max_context_length().unwrap_or(u32::MAX).saturating_sub(1)
            .min((worker.budget.input_bytes / std::mem::size_of::<u32>()).min(u32::MAX as usize) as u32),
        "max_request_bytes":worker.budget.wire_bytes,"request_budget_bytes":worker.budget.input_bytes,
        "eos_token_ids":worker.backend.eos_token_ids(),
        "sampling_modes":["greedy","stochastic"],"streaming":"text-delta-v1",
        "generation_progress":"token-counts-v1",
        "stop_sequences":true,"cancellation":"pipe-v1"});
    if let Some(enabled) = worker.prefix_cache {
        readiness["prefix_cache"] = json!({"supported":enabled,"enabled":enabled,
            "chunk_tokens":worker.backend.prefix_cache_chunk_tokens(),"idle_seconds":prefix_cache::IDLE_SECONDS,
            "key":worker.backend.prefix_cache_key()});
        if enabled {
            readiness["state_policy"] = json!(prefix_cache::POLICY);
            readiness["operations"] = json!(["tokenize", "decode", "generate", "reset_cache"]);
        }
    }
    if worker.backend.supports_automatic_context_planning() {
        let operations = if worker.prefix_cache == Some(true) {
            json!(["tokenize", "decode", "generate", "plan_generation", "reset_cache"])
        } else {
            json!(["tokenize", "decode", "generate", "plan_generation"])
        };
        readiness["operations"] = operations;
        readiness["generation_budget"] = json!("memory-context-v1");
        readiness["generation_budget_inputs"] = json!("token-ids-v1");
    }
    if let Some(memory) = worker.backend.loaded_memory_metadata() {
        readiness["loaded_memory"] = memory;
    }
    readiness["mtp"] = match worker.mtp {
        Some(capabilities) => json!({"supported":true,"max_drafts":capabilities.max_drafts,
            "recommended_drafts":capabilities.recommended_drafts,"default_drafts":worker.default_mtp_drafts,
            "sampling":capabilities.sampling,"verification":capabilities.verification,
            "operations":["generate"]}),
        None => json!({"supported":false,"max_drafts":0,"recommended_drafts":0,
            "default_drafts":0,"operations":[]}),
    };
    if worker.media.is_some() && worker.backend.supports_images() {
        let identity = worker
            .backend
            .image_artifact_identity()
            .filter(ImageArtifactIdentity::valid)
            .ok_or("invalid_model_config")?;
        let runtime_identity = worker.backend.runtime_identity().ok_or("invalid_model_config")?;
        let (family_profile, family_version) = worker.backend.image_family_profile().ok_or("invalid_model_config")?;
        let (model_source, model_revision) = worker.backend.image_source_identity().ok_or("invalid_model_config")?;
        let (max_patches, max_image_tokens, max_processed_pixels) = image_family_limits(
            family_profile,
            family_version,
            identity.schema_version,
            worker.backend.image_geometry(),
        )
        .ok_or("invalid_model_config")?;
        readiness["image_input"] = json!({
            "version":"mediaref-v1","transport":"owner-private-file-v1","qualified":false,"qualification":"unverified",
            "operations":["prepare_image","generate_image","discard_image"],
            "formats":["image/jpeg","image/png"],"max_images":1,
            "max_compressed_bytes":crate::media::MAX_COMPRESSED_BYTES,"max_source_pixels":32_000_000,
            "max_source_dimension":16_384,
            "max_processed_pixels":max_processed_pixels,"max_patches":max_patches,"max_image_tokens":max_image_tokens,
            "artifact_schema_version":identity.schema_version,"model_source":model_source,
            "model_revision":model_revision,"family_profile":family_profile,"family_version":family_version,
            "artifact_sha256":identity.artifact_sha256,"config_sha256":identity.config_sha256,
            "tokenizer_sha256":identity.tokenizer_sha256,"processor_sha256":identity.processor_sha256,
            "template_sha256":identity.template_sha256,
            "runtime_identity":runtime_identity,
            "prepared_policy":"single-use-next-operation"
        });
    }
    write_reply(output, &readiness)?;
    loop {
        if let Err(code) = wait_for_request(&mut worker.backend, input) {
            worker.backend.clear_prefix_cache();
            worker.prepared = None;
            return Err(code);
        }
        let mut bytes = Vec::new();
        let count = match Read::by_ref(input).take(worker.budget.wire_bytes as u64 + 1).read_until(b'\n', &mut bytes) {
            Ok(count) => count,
            Err(_) => {
                worker.prepared = None;
                return Err("protocol_read_failed");
            },
        };
        if count == 0 {
            worker.prepared = None;
            return Ok(());
        }
        if bytes.len() > worker.budget.wire_bytes {
            worker.prepared = None;
            worker.backend.clear_prefix_cache();
            // Drain the rest of this frame so the next request remains independent.
            if bytes.last() != Some(&b'\n') {
                loop {
                    let chunk = input.fill_buf().map_err(|_| "protocol_read_failed")?;
                    if chunk.is_empty() {
                        break;
                    }
                    let newline = chunk.iter().position(|byte| *byte == b'\n');
                    let count = newline.map_or(chunk.len(), |position| position + 1);
                    input.consume(count);
                    if newline.is_some() {
                        break;
                    }
                }
            }
            write_reply(output, &json!({"error":"request_too_large"}))?;
            continue;
        }
        worker.process_line(&bytes, output)?;
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        io,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
    };

    use super::*;

    struct TestStream {
        tokens: VecDeque<Result<u64, &'static str>>,
        drained: Arc<AtomicBool>,
    }
    impl Iterator for TestStream {
        type Item = Result<u64, &'static str>;
        fn next(&mut self) -> Option<Self::Item> {
            self.tokens.pop_front()
        }
    }
    impl Drop for TestStream {
        fn drop(&mut self) {
            self.drained.store(true, Ordering::SeqCst);
        }
    }

    #[derive(Debug, PartialEq)]
    struct RecordedPlan {
        input_tokens: u32,
        maximum: u32,
        input_len: Option<usize>,
        cache_scope: Option<CacheScope>,
        mtp_drafts: u32,
        stochastic: bool,
    }

    struct FakeBackend {
        idle_interval: Option<Duration>,
        idle_calls: usize,
        fail_idle: bool,
        max_context: Option<u32>,
        tokens: Vec<Result<u64, &'static str>>,
        eos: Vec<u32>,
        generation_calls: usize,
        tokenization_calls: usize,
        inputs: Vec<Vec<u32>>,
        seeds: Vec<u64>,
        sampling: Vec<Sampling>,
        drained: Arc<AtomicBool>,
        invalid_tokenization: bool,
        images: bool,
        mtp: Option<MtpCapabilities>,
        default_mtp_drafts: u32,
        omit_mtp_metadata: bool,
        mtp_reply_override: Option<Value>,
        mtp_requests: Vec<u32>,
        mtp_effective_override: Option<u32>,
        image_schema: u32,
        family_version: u32,
        family_profile: &'static str,
        image_preparations: usize,
        image_generations: usize,
        prepared_drops: Arc<AtomicUsize>,
        cancel_image: bool,
        cancel_after_prepare: bool,
        fail_image_prepare: bool,
        prepared_metadata_override: Option<PreparedImageMetadata>,
        prefix_supported: bool,
        prefix_entry: Option<prefix_cache::PrefixEntry<usize>>,
        prefix_pending: Option<prefix_cache::PrefixEntry<usize>>,
        state_allocations: usize,
        prefix_chunk: usize,
        prefix_reuse_override: Option<usize>,
        plans: Vec<RecordedPlan>,
        planning_allowed: bool,
    }

    struct FakePreparedImage {
        dropped: Arc<AtomicUsize>,
    }

    impl Drop for FakePreparedImage {
        fn drop(&mut self) {
            self.dropped.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl GenerationBackend for FakeBackend {
        type PreparedImage = FakePreparedImage;

        fn idle_maintenance_interval(&self) -> Option<Duration> {
            self.idle_interval
        }
        fn maintain_idle_queue(&mut self) -> Result<(), &'static str> {
            self.idle_calls += 1;
            if self.fail_idle {
                Err("native_idle_maintenance_failed")
            } else {
                Ok(())
            }
        }
        fn vocab_size(&self) -> usize {
            8
        }
        fn max_context_length(&self) -> Option<u32> {
            self.max_context
        }
        fn automatic_context_length(
            &self,
            input_tokens: u32,
            maximum: u32,
            _memory_budget_bytes: usize,
        ) -> Option<u32> {
            if !self.planning_allowed {
                return None;
            }
            if self.max_context.is_some() {
                let minimum = input_tokens.checked_add(1)?;
                let capacity = self.max_context?.min(maximum);
                return (capacity >= minimum).then_some(capacity);
            }
            input_tokens.checked_add(4096).map(|capacity| capacity.min(maximum))
        }
        fn supports_automatic_context_planning(&self) -> bool {
            self.max_context.is_none() || self.max_context == Some(0)
        }
        fn plan_context_length(
            &mut self,
            request: &ContextPlanningRequest<'_>,
            memory_budget_bytes: usize,
        ) -> Result<Option<u32>, &'static str> {
            self.plans.push(RecordedPlan {
                input_tokens: request.input_tokens,
                maximum: request.maximum,
                input_len: request.input_token_ids.map(<[u32]>::len),
                cache_scope: request.options.cache_scope,
                mtp_drafts: request.options.mtp_drafts,
                stochastic: matches!(request.options.sampling, Sampling::Stochastic { .. }),
            });
            Ok(self.automatic_context_length(request.input_tokens, request.maximum, memory_budget_bytes))
        }
        fn eos_token_ids(&self) -> &[u32] {
            &self.eos
        }
        fn mtp_capabilities(&self) -> Option<MtpCapabilities> {
            self.mtp
        }
        fn default_mtp_drafts(&self) -> u32 {
            self.default_mtp_drafts
        }
        fn effective_mtp_drafts(
            &self,
            options: &GenerationOptions,
        ) -> Result<u32, &'static str> {
            Ok(self.mtp_effective_override.unwrap_or(options.mtp_drafts))
        }
        fn mtp_metadata(&self) -> Option<Value> {
            if self.omit_mtp_metadata {
                return None;
            }
            if let Some(metadata) = &self.mtp_reply_override {
                return Some(metadata.clone());
            }
            self.mtp.map(|_| {
                let drafts = self.mtp_requests.last().copied().unwrap_or(0);
                json!({"enabled":drafts > 0,"drafts":drafts,"sampling":"exact-rejection-v1"})
            })
        }
        fn tokenize(
            &mut self,
            _: &str,
            special: bool,
        ) -> Result<Vec<u32>, &'static str> {
            self.tokenization_calls += 1;
            Ok(if self.invalid_tokenization {
                vec![8]
            } else if special {
                vec![0, 2]
            } else {
                vec![2]
            })
        }
        fn decode(
            &mut self,
            ids: &[u32],
            skip_special: bool,
        ) -> Result<String, &'static str> {
            assert!(self.drained.load(Ordering::SeqCst));
            Ok(ids
                .iter()
                .map(|id| match id {
                    3 => "ré",
                    4 => "sultat\n",
                    1 if !skip_special => "<eos>",
                    7 if !skip_special => "<tool_call>",
                    _ => "",
                })
                .collect())
        }
        fn generate(
            &mut self,
            input: &[u32],
            options: &GenerationOptions,
            mut on_text: Option<&mut TextCallback<'_>>,
            on_progress: Option<&mut ProgressCallback<'_>>,
        ) -> Result<GeneratedTokens, &'static str> {
            self.generation_calls += 1;
            self.inputs.push(input.to_vec());
            self.seeds.push(options.seed);
            self.sampling.push(options.sampling.clone());
            self.mtp_requests.push(options.mtp_drafts);
            self.drained.store(false, Ordering::SeqCst);
            // Mirror the native backend: take the entry, reuse or drop it, and
            // retain a completed scoped state as pending until publication.
            let reused = self.prefix_cache_plan(input, options);
            let retained = self.prefix_entry.take();
            self.prefix_pending = None;
            let state = match retained {
                Some(entry) if reused > 0 => entry.state,
                _ => {
                    self.state_allocations += 1;
                    self.state_allocations
                },
            };
            let mut on_token = |token| {
                if let Some(callback) = on_text.as_deref_mut() {
                    callback(match token {
                        3 => "ré",
                        4 => "sultat\n",
                        7 => "<tool_call>",
                        _ => "",
                    })?;
                }
                Ok(true)
            };
            let mut result = collect_tokens_controlled(
                TestStream {
                    tokens: self.tokens.clone().into(),
                    drained: self.drained.clone(),
                },
                input.len(),
                options,
                self.vocab_size(),
                Instant::now(),
                &mut on_token,
                on_progress,
                &mut || Ok(false),
            )?;
            result.reused_input_tokens = reused;
            if let Some(scope) = options.cache_scope {
                self.prefix_pending = Some(prefix_cache::PrefixEntry::new(state, scope, options.context_tokens, input));
            }
            Ok(result)
        }

        fn prefix_cache_supported(&self) -> bool {
            self.prefix_supported
        }

        fn prefix_cache_chunk_tokens(&self) -> usize {
            self.prefix_chunk
        }

        fn prefix_cache_plan(
            &self,
            input: &[u32],
            options: &GenerationOptions,
        ) -> usize {
            if let Some(reused) = self.prefix_reuse_override {
                return reused;
            }
            self.prefix_entry
                .as_ref()
                .map_or(0, |entry| entry.reusable(input, options.context_tokens, options.cache_scope.as_ref()))
        }

        fn publish_prefix_cache(&mut self) {
            if let Some(entry) = self.prefix_pending.take() {
                self.prefix_entry = Some(entry);
            }
        }

        fn clear_prefix_cache(&mut self) {
            self.prefix_entry = None;
            self.prefix_pending = None;
        }

        fn prefix_cache_retained(&self) -> bool {
            self.prefix_entry.is_some() || self.prefix_pending.is_some()
        }

        fn supports_images(&self) -> bool {
            self.images
        }

        fn image_artifact_identity(&self) -> Option<ImageArtifactIdentity<'_>> {
            self.images.then_some(ImageArtifactIdentity {
                schema_version: self.image_schema,
                artifact_sha256: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                config_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                tokenizer_sha256: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                processor_sha256: "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                template_sha256: "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            })
        }

        fn image_family_profile(&self) -> Option<(&str, u32)> {
            self.images.then_some((self.family_profile, self.family_version))
        }

        fn image_geometry(&self) -> (u32, u32) {
            if self.family_profile == "gemma4-e2b-bf16-v1" {
                (16, 3)
            } else {
                (16, 2)
            }
        }

        fn image_source_identity(&self) -> Option<(&str, Option<&str>)> {
            self.images.then_some(if self.image_schema == 4 {
                ("google/gemma-4-E2B-it", Some("3e22461f65e89153144f8adb70e3b8c2cc9845a7"))
            } else if self.image_schema == 3 {
                ("local:synthetic-merged-model", None)
            } else {
                ("Qwen/Qwen3.5-0.8B", Some("2fc06364715b967f1860aea9cf38778875588b17"))
            })
        }

        fn runtime_identity(&self) -> Option<Value> {
            self.images.then(|| {
                json!({
                    "backend":"metal",
                    "device_name":"Apple Synthetic GPU",
                    "apple_gpu_family":1008,
                    "gpu_core_count":32,
                    "supports_mxu":false
                })
            })
        }

        fn cancelled(
            &mut self,
            _: u64,
        ) -> Result<bool, &'static str> {
            Ok(self.cancel_image)
        }

        fn prepare_image(
            &mut self,
            _: Vec<u32>,
            _: &[u8],
            _: u32,
            _: u64,
        ) -> Result<(FakePreparedImage, PreparedImageMetadata), &'static str> {
            self.image_preparations += 1;
            if self.fail_image_prepare {
                return Err("invalid_image_request");
            }
            self.cancel_image = self.cancel_after_prepare;
            Ok((
                FakePreparedImage {
                    dropped: self.prepared_drops.clone(),
                },
                self.prepared_metadata_override.take().unwrap_or(PreparedImageMetadata {
                    input_tokens: 7,
                    image_tokens: if self.image_schema == 4 {
                        1
                    } else {
                        4
                    },
                    grid: if self.image_schema == 4 {
                        [1, 3, 3]
                    } else {
                        [1, 4, 4]
                    },
                    processed_dimensions: if self.image_schema == 4 {
                        [48, 48]
                    } else {
                        [64, 64]
                    },
                }),
            ))
        }

        fn generate_image(
            &mut self,
            _image: FakePreparedImage,
            options: &GenerationOptions,
            on_text: Option<&mut TextCallback<'_>>,
            on_progress: Option<&mut ProgressCallback<'_>>,
        ) -> Result<GeneratedTokens, &'static str> {
            self.image_generations += 1;
            self.generate(&[2; 7], options, on_text, on_progress)
        }
    }
    fn worker() -> GenerationWorker<FakeBackend> {
        GenerationWorker::new(FakeBackend {
            idle_interval: None,
            idle_calls: 0,
            fail_idle: false,
            max_context: Some(1024),
            tokens: vec![Ok(3), Ok(4), Ok(1), Ok(7)],
            eos: vec![1],
            generation_calls: 0,
            tokenization_calls: 0,
            inputs: Vec::new(),
            seeds: Vec::new(),
            sampling: Vec::new(),
            drained: Arc::new(AtomicBool::new(true)),
            invalid_tokenization: false,
            images: false,
            mtp: None,
            default_mtp_drafts: 0,
            omit_mtp_metadata: false,
            mtp_reply_override: None,
            mtp_requests: Vec::new(),
            mtp_effective_override: None,
            image_schema: 2,
            family_version: 1,
            family_profile: "qwen35-dense-bf16-merged-v1",
            image_preparations: 0,
            image_generations: 0,
            prepared_drops: Arc::new(AtomicUsize::new(0)),
            cancel_image: false,
            cancel_after_prepare: false,
            fail_image_prepare: false,
            prepared_metadata_override: None,
            prefix_supported: false,
            prefix_entry: None,
            prefix_pending: None,
            state_allocations: 0,
            prefix_chunk: prefix_cache::PREFILL_CHUNK_TOKENS,
            prefix_reuse_override: None,
            plans: Vec::new(),
            planning_allowed: true,
        })
        .unwrap()
    }
    fn standard() -> Value {
        json!({"operation":"generate","input_token_ids":[2],"context_tokens":16,"max_new_tokens":8})
    }

    fn capabilities(
        max_drafts: usize,
        recommended_drafts: usize,
    ) -> MtpCapabilities {
        MtpCapabilities {
            max_drafts,
            recommended_drafts,
            sampling: "exact-rejection-v1",
            verification: "fake-model-exact-v1",
        }
    }

    #[test]
    fn mtp_sampling_fallback_reports_effective_selection() {
        let mut worker = worker();
        let mut native = capabilities(3, 3);
        native.sampling = "native-target-draw-match-v1";
        worker.backend.mtp = Some(native);
        worker.mtp = Some(native);
        worker.backend.omit_mtp_metadata = true;
        worker.backend.mtp_effective_override = Some(0);
        let mut request = standard();
        request["mtp_drafts"] = json!(3);
        let reply = post(&mut worker, request);
        assert_eq!(worker.backend.mtp_requests, vec![0]);
        assert_eq!(reply["mtp"]["enabled"], false);
        assert_eq!(reply["mtp"]["drafts"], 0);
        assert_eq!(reply["mtp"]["requested_drafts"], 3);
        assert_eq!(reply["mtp"]["sampling"], "native-target-draw-match-v1");
        assert_eq!(reply["mtp"]["fallback_reason"], "unsupported_sampling");
    }

    #[test]
    fn native_target_sampling_policy_is_admitted_at_startup() {
        let mut backend = worker().backend;
        let mut native = capabilities(3, 3);
        native.sampling = "native-target-draw-match-v1";
        backend.mtp = Some(native);
        assert!(GenerationWorker::new(backend).is_ok());
    }

    #[test]
    fn selected_backend_default_is_resolved_and_can_be_disabled() {
        let mut backend = worker().backend;
        backend.mtp = Some(capabilities(3, 3));
        backend.default_mtp_drafts = 3;
        let mut worker = GenerationWorker::new(backend).unwrap();
        let reply = post(&mut worker, standard());
        assert_eq!(reply["mtp"]["drafts"], 3);
        assert_eq!(worker.backend.mtp_requests, vec![3]);
        let mut request = standard();
        request["mtp"] = json!(false);
        let reply = post(&mut worker, request);
        assert_eq!(reply["mtp"]["drafts"], 0);
        let mut request = standard();
        request["mtp_drafts"] = json!(0);
        assert_eq!(post(&mut worker, request)["mtp"]["drafts"], 0);
    }

    #[test]
    fn invalid_or_changing_backend_default_is_rejected() {
        let mut backend = worker().backend;
        backend.default_mtp_drafts = 3;
        assert!(GenerationWorker::new(backend).is_err());
        let mut worker = mtp_worker(3, 3);
        worker.backend.default_mtp_drafts = 3;
        assert_eq!(worker.resolve_mtp(None, None), Err("invalid_generation_output"));
    }

    #[test]
    fn backend_cannot_silently_change_requested_draft_depth() {
        for effective in [1, 2, 4] {
            let mut worker = worker();
            worker.backend.mtp = Some(capabilities(3, 3));
            worker.mtp = worker.backend.mtp;
            worker.backend.mtp_effective_override = Some(effective);
            let mut request = standard();
            request["mtp_drafts"] = json!(3);
            let mut output = Vec::new();
            assert_eq!(
                worker.process_line(&serde_json::to_vec(&request).unwrap(), &mut output),
                Err("native_operation_failed")
            );
            let reply: Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(reply["error"], "invalid_generation_output");
            assert_eq!(worker.backend.generation_calls, 0);
        }
    }

    fn mtp_worker(
        max_drafts: usize,
        recommended_drafts: usize,
    ) -> GenerationWorker<FakeBackend> {
        let mut backend = worker().backend;
        backend.mtp = Some(capabilities(max_drafts, recommended_drafts));
        GenerationWorker::new(backend).unwrap()
    }

    #[test]
    fn mtp_requests_are_opt_in_bounded_and_require_model_support() {
        let mut worker = worker();
        let mut request = standard();
        request["mtp_drafts"] = json!(1);
        assert_eq!(post(&mut worker, request.clone())["error"], "mtp_not_supported");
        assert_eq!(worker.backend.generation_calls, 0);
        worker = mtp_worker(3, 2);
        request["mtp_drafts"] = json!(4);
        assert_eq!(post(&mut worker, request.clone())["error"], "invalid_generation_request");
        assert_eq!(worker.backend.generation_calls, 0);
        for drafts in 0..=3 {
            request["mtp_drafts"] = json!(drafts);
            let reply = post(&mut worker, request.clone());
            assert!(reply.get("error").is_none());
            assert_eq!(reply["mtp"]["drafts"], drafts);
            assert_eq!(reply["mtp"]["enabled"], drafts > 0);
        }
        assert!(post(&mut worker, standard()).get("error").is_none());
        assert!(worker.backend.mtp_requests == [0, 1, 2, 3, 0]);
        for invalid in [json!(-1), json!(1.5), json!(true), json!(null)] {
            request["mtp_drafts"] = invalid;
            assert_eq!(post(&mut worker, request.clone())["error"], "invalid_request");
        }
        let mut output = Vec::new();
        run_lines(&mut worker, &mut std::io::Cursor::new(Vec::<u8>::new()), &mut output).unwrap();
        let readiness: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(readiness["mtp"]["supported"], true);
        assert_eq!(readiness["mtp"]["default_drafts"], 0);
        assert_eq!(readiness["mtp"]["max_drafts"], 3);
        assert_eq!(readiness["mtp"]["recommended_drafts"], 2);
        assert_eq!(readiness["mtp"]["operations"], json!(["generate"]));
    }

    #[test]
    fn mtp_toggle_uses_each_models_recommendation_and_does_not_persist() {
        for (maximum, recommended) in [(1, 1), (3, 2), (5, 4)] {
            let mut worker = mtp_worker(maximum, recommended);
            let mut output = Vec::new();
            run_lines(&mut worker, &mut io::Cursor::new(Vec::<u8>::new()), &mut output).unwrap();
            let readiness: Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(readiness["mtp"]["max_drafts"], maximum);
            assert_eq!(readiness["mtp"]["recommended_drafts"], recommended);
            let mut request = standard();
            request["sampling"] = json!({"mode":"stochastic","temperature":1.0,"top_p":0.95});
            request["seed"] = json!(42);
            for enabled in [true, false, true] {
                request["mtp"] = json!(enabled);
                let reply = post(&mut worker, request.clone());
                assert!(reply.get("error").is_none());
                assert_eq!(reply["mtp"]["enabled"], enabled);
                assert_eq!(
                    reply["mtp"]["drafts"],
                    if enabled {
                        recommended
                    } else {
                        0
                    }
                );
                assert_eq!(reply["mtp"]["sampling"], "exact-rejection-v1");
                assert_eq!(reply["mtp"]["verification"], "fake-model-exact-v1");
            }
            request.as_object_mut().unwrap().remove("mtp");
            assert!(post(&mut worker, request.clone()).get("error").is_none());
            request["mtp_drafts"] = json!(maximum);
            assert!(post(&mut worker, request.clone()).get("error").is_none());
            request["mtp_drafts"] = json!(maximum + 1);
            assert_eq!(post(&mut worker, request)["error"], "invalid_generation_request");
            assert_eq!(worker.backend.mtp_requests, [recommended as u32, 0, recommended as u32, 0, maximum as u32]);
            assert_eq!(worker.backend.seeds, [42; 5]);
            assert!(worker.backend.sampling.iter().all(|sampling| {
                *sampling
                    == Sampling::Stochastic {
                        temperature: 1.0,
                        top_k: None,
                        top_p: Some(0.95),
                        min_p: None,
                    }
            }));
        }
    }

    #[test]
    fn unsupported_mtp_toggle_off_is_valid_and_errors_do_not_touch_model() {
        let mut worker = worker();
        let mut request = standard();
        request.as_object_mut().unwrap().remove("input_token_ids");
        request["rendered_prompt"] = json!("synthetic prompt");
        request["mtp"] = json!(true);
        assert_eq!(post(&mut worker, request.clone())["error"], "mtp_not_supported");
        assert_eq!(worker.backend.tokenization_calls, 0);
        assert_eq!(worker.backend.generation_calls, 0);
        request["mtp"] = json!(false);
        assert!(post(&mut worker, request).get("error").is_none());
        assert_eq!(worker.backend.mtp_requests, [0]);
        let mut output = Vec::new();
        run_lines(&mut worker, &mut io::Cursor::new(Vec::<u8>::new()), &mut output).unwrap();
        let readiness: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(
            readiness["mtp"],
            json!({"supported":false,"max_drafts":0,
            "recommended_drafts":0,"default_drafts":0,"operations":[]})
        );
    }

    #[test]
    fn conflicting_or_malformed_mtp_settings_are_recoverable_before_inference() {
        let mut worker = mtp_worker(5, 4);
        for enabled in [true, false] {
            for drafts in [0, 1, 5] {
                let mut request = standard();
                request.as_object_mut().unwrap().remove("input_token_ids");
                request["rendered_prompt"] = json!("synthetic prompt");
                request["mtp"] = json!(enabled);
                request["mtp_drafts"] = json!(drafts);
                assert_eq!(post(&mut worker, request)["error"], "invalid_generation_request");
            }
        }
        for malformed in [json!(null), json!(0), json!(1), json!("true"), json!({}), json!([])] {
            let mut request = standard();
            request["mtp"] = malformed;
            assert_eq!(post(&mut worker, request)["error"], "invalid_request");
        }
        assert_eq!(worker.backend.tokenization_calls, 0);
        assert_eq!(worker.backend.generation_calls, 0);
        assert!(post(&mut worker, standard()).get("error").is_none());
    }

    #[test]
    fn invalid_model_mtp_descriptors_fail_startup() {
        let mut invalid = vec![
            capabilities(0, 0),
            capabilities(3, 0),
            capabilities(3, 4),
            MtpCapabilities {
                sampling: "",
                ..capabilities(3, 2)
            },
            MtpCapabilities {
                sampling: "approximate-v1",
                ..capabilities(3, 2)
            },
            MtpCapabilities {
                verification: " ",
                ..capabilities(3, 2)
            },
            MtpCapabilities {
                verification: "unknown:policy",
                ..capabilities(3, 2)
            },
            MtpCapabilities {
                verification: "éxact-v1",
                ..capabilities(3, 2)
            },
            MtpCapabilities {
                verification: "01234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890",
                ..capabilities(3, 2)
            },
        ];
        if let Ok(maximum) = usize::try_from(u64::from(u32::MAX) + 1) {
            invalid.push(capabilities(maximum, 1));
        }
        for capabilities in invalid {
            let mut backend = worker().backend;
            backend.mtp = Some(capabilities);
            assert!(matches!(GenerationWorker::new(backend), Err("invalid_model_config")));
        }
    }

    #[test]
    fn changing_backend_mtp_capabilities_during_a_worker_lifetime_fences_output() {
        let mut worker = mtp_worker(3, 2);
        worker.backend.mtp = Some(capabilities(5, 4));
        let mut output = Vec::new();
        assert_eq!(
            worker.process_line(&serde_json::to_vec(&standard()).unwrap(), &mut output),
            Err("native_operation_failed")
        );
        assert_eq!(serde_json::from_slice::<Value>(&output).unwrap()["error"], "invalid_generation_output");
        assert_eq!(worker.backend.generation_calls, 0);
    }

    #[test]
    fn mtp_acknowledgement_does_not_require_backend_specific_statistics() {
        let mut worker = mtp_worker(5, 4);
        worker.backend.omit_mtp_metadata = true;
        let mut request = standard();
        request["mtp"] = json!(true);
        let reply = post(&mut worker, request);
        assert_eq!(
            reply["mtp"],
            json!({"enabled":true,"drafts":4,
            "sampling":"exact-rejection-v1","verification":"fake-model-exact-v1"})
        );
    }

    #[test]
    fn mtp_execution_method_is_separate_from_stable_verification_contract() {
        for method in ["transactional-greedy-batch", "exact-rejection-serial"] {
            let mut worker = mtp_worker(3, 2);
            worker.backend.mtp_reply_override = Some(json!({"verification_method":method}));
            let mut request = standard();
            request["mtp"] = json!(true);
            let reply = post(&mut worker, request);
            assert_eq!(reply["mtp"]["verification"], "fake-model-exact-v1");
            assert_eq!(reply["mtp"]["verification_method"], method);
        }
    }

    #[test]
    fn contradictory_mtp_backend_acknowledgements_fence_output() {
        for metadata in [
            json!(null),
            json!([]),
            json!({"enabled":false}),
            json!({"drafts":1}),
            json!({"sampling":"approximate"}),
            json!({"verification":"other-policy"}),
        ] {
            let mut worker = mtp_worker(3, 2);
            worker.backend.mtp_reply_override = Some(metadata);
            let mut request = standard();
            request["mtp"] = json!(true);
            let mut output = Vec::new();
            assert_eq!(
                worker.process_line(&serde_json::to_vec(&request).unwrap(), &mut output),
                Err("native_operation_failed")
            );
            assert_eq!(serde_json::from_slice::<Value>(&output).unwrap()["error"], "invalid_generation_output");
        }
    }
    fn post(
        worker: &mut GenerationWorker<FakeBackend>,
        request: Value,
    ) -> Value {
        let mut output = Vec::new();
        worker.process_line(&serde_json::to_vec(&request).unwrap(), &mut output).unwrap();
        serde_json::from_slice(output.split(|byte| *byte == b'\n').rfind(|line| !line.is_empty()).unwrap()).unwrap()
    }

    fn image_worker(root: &crate::media::tests::PrivateRoot) -> GenerationWorker<FakeBackend> {
        let mut worker = worker();
        worker.backend.images = true;
        worker.media = Some(MediaStore::open(&root.0).unwrap());
        worker
    }

    fn image_prepare(root: &crate::media::tests::PrivateRoot) -> Value {
        let image = root.image();
        json!({"operation":"prepare_image","rendered_prompt":"fixture",
            "image":{"id":image.id,"byte_length":image.byte_length,"media_type":image.media_type},
            "context_tokens":16,"max_new_tokens":8,"request_id":10})
    }

    fn image_generate(prepared: &Value) -> Value {
        json!({"operation":"generate_image","prepared_id":prepared["prepared_id"],
            "context_tokens":16,"max_new_tokens":8,"request_id":11})
    }

    #[test]
    fn image_preparation_enforces_each_loaded_familys_advertised_ceiling() {
        for (profile, version, schema, side, expected_patches, accepted) in [
            ("qwen38-lily-retained-bf16-vision-v1", 1, 6, [256, 256], 65_536, true),
            ("qwen38-lily-retained-bf16-vision-v1", 1, 6, [256, 258], 65_536, false),
            ("qwen35-dense-bf16-merged-v2", 2, 3, [64, 64], 4096, true),
            ("qwen35-dense-bf16-merged-v2", 2, 3, [64, 66], 4096, false),
            ("qwen35-dense-bf16-merged-v2", 2, 3, [256, 256], 4096, false),
        ] {
            let root = crate::media::tests::PrivateRoot::new();
            let mut worker = image_worker(&root);
            worker.backend.family_profile = profile;
            worker.backend.family_version = version;
            worker.backend.image_schema = schema;
            worker.backend.max_context = Some(65_536);
            let image_tokens = side[0] * side[1] / 4;
            worker.backend.prepared_metadata_override = Some(PreparedImageMetadata {
                input_tokens: image_tokens as usize + 3,
                image_tokens,
                grid: [1, side[0], side[1]],
                processed_dimensions: [side[1] * 16, side[0] * 16],
            });
            assert_eq!(readiness(&mut worker)["image_input"]["max_patches"], expected_patches);
            let mut request = image_prepare(&root);
            request["context_tokens"] = json!(65_536);
            let mut output = Vec::new();
            let result = worker.process_line(&serde_json::to_vec(&request).unwrap(), &mut output);
            let reply: Value = serde_json::from_slice(&output).unwrap();
            if accepted {
                assert!(result.is_ok());
                assert!(reply.get("prepared_id").is_some());
                assert_eq!(reply["image_tokens"], image_tokens);
            } else {
                assert_eq!(result, Err("native_operation_failed"));
                assert_eq!(reply["error"], "invalid_image_preparation_output");
                assert!(worker.prepared.is_none());
                assert_eq!(worker.backend.prepared_drops.load(Ordering::SeqCst), 1);
            }
        }
    }

    #[test]
    fn image_prepare_reports_expanded_counts_and_generation_consumes_once() {
        let root = crate::media::tests::PrivateRoot::new();
        let mut worker = image_worker(&root);
        let prepared = post(&mut worker, image_prepare(&root));
        assert!(valid_id(prepared["prepared_id"].as_str().unwrap()));
        assert_eq!(prepared["input_tokens"], 7);
        assert_eq!(prepared["image_tokens"], 4);
        assert!(prepared.get("token_ids").is_none());
        assert_eq!(worker.backend.prepared_drops.load(Ordering::SeqCst), 0);
        let request = image_generate(&prepared);
        let reply = post(&mut worker, request.clone());
        assert_eq!(reply["input_tokens"], 7);
        assert_eq!(reply["finish_reason"], "eos");
        assert_eq!(worker.backend.image_preparations, 1);
        assert_eq!(worker.backend.image_generations, 1);
        assert_eq!(worker.backend.prepared_drops.load(Ordering::SeqCst), 1);
        assert_eq!(post(&mut worker, request)["error"], "invalid_prepared_image");
        assert_eq!(post(&mut worker, json!({"operation":"discard_image"}))["discarded"], false);
        assert_eq!(post(&mut worker, standard())["finish_reason"], "eos");
    }

    #[test]
    fn image_requests_keep_drafts_zero_between_mtp_text_requests() {
        let root = crate::media::tests::PrivateRoot::new();
        let mut backend = worker().backend;
        backend.images = true;
        backend.mtp = Some(capabilities(3, 2));
        backend.default_mtp_drafts = 2;
        let mut worker = GenerationWorker::new(backend).unwrap();
        worker.media = Some(MediaStore::open(&root.0).unwrap());
        assert_eq!(post(&mut worker, standard())["mtp"]["drafts"], 2);
        let prepared = post(&mut worker, image_prepare(&root));
        let mut forbidden = image_generate(&prepared);
        forbidden["mtp_drafts"] = json!(2);
        assert_eq!(post(&mut worker, forbidden)["error"], "invalid_request");
        let prepared = post(&mut worker, image_prepare(&root));
        let reply = post(&mut worker, image_generate(&prepared));
        assert!(reply.get("error").is_none());
        assert_eq!(reply["mtp"]["drafts"], 0);
        assert_eq!(worker.backend.image_generations, 1);
        assert_eq!(post(&mut worker, standard())["mtp"]["drafts"], 2);
        assert_eq!(worker.backend.mtp_requests, [2, 0, 2]);
    }

    #[test]
    fn image_memory_rejection_releases_prepared_state_and_accepts_same_worker_retry() {
        let root = crate::media::tests::PrivateRoot::new();
        let mut worker = image_worker(&root);
        let prepared = post(&mut worker, image_prepare(&root));
        worker.backend.tokens = vec![Err("resource_budget_exceeded")];
        assert_eq!(post(&mut worker, image_generate(&prepared))["error"], "resource_budget_exceeded");
        assert_eq!(worker.backend.prepared_drops.load(Ordering::SeqCst), 1);
        assert_eq!(post(&mut worker, json!({"operation":"discard_image"}))["discarded"], false);
        worker.backend.tokens = vec![Ok(3), Ok(1)];
        let prepared = post(&mut worker, image_prepare(&root));
        let retry = post(&mut worker, image_generate(&prepared));
        assert_eq!(retry["input_tokens"], 7);
        assert_eq!(retry["finish_reason"], "eos");
        assert_eq!(worker.backend.prepared_drops.load(Ordering::SeqCst), 2);
        assert_eq!(worker.backend.image_generations, 2);
    }

    #[test]
    fn image_preparation_is_opt_in_and_unknown_fields_remain_rejected() {
        let root = crate::media::tests::PrivateRoot::new();
        let mut vision_worker = image_worker(&root);
        let mut text = worker();
        assert_eq!(post(&mut text, image_prepare(&root))["error"], "images_not_supported");
        assert_eq!(text.backend.image_preparations, 0);
        let mut request = image_prepare(&root);
        request["image"]["path"] = json!("not-allowed");
        assert_eq!(post(&mut vision_worker, request)["error"], "invalid_request");
        let mut request = image_prepare(&root);
        request["input_token_ids"] = json!([2]);
        assert_eq!(post(&mut vision_worker, request)["error"], "invalid_request");
        assert_eq!(vision_worker.backend.image_preparations, 0);
    }

    #[test]
    fn another_request_or_invalid_frame_releases_the_pending_image() {
        let root = crate::media::tests::PrivateRoot::new();
        let mut worker = image_worker(&root);
        let first = post(&mut worker, image_prepare(&root));
        let second = post(&mut worker, image_prepare(&root));
        assert_ne!(first["prepared_id"], second["prepared_id"]);
        assert_eq!(worker.backend.prepared_drops.load(Ordering::SeqCst), 1);
        worker.process_line(b"{", &mut Vec::new()).unwrap();
        assert!(worker.prepared.is_none());
        assert_eq!(worker.backend.prepared_drops.load(Ordering::SeqCst), 2);
        let pending = post(&mut worker, image_prepare(&root));
        post(&mut worker, json!({"operation":"tokenize","text":"fixture"}));
        assert_eq!(post(&mut worker, image_generate(&pending))["error"], "invalid_prepared_image");
        assert_eq!(worker.backend.prepared_drops.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn prepared_handles_are_worker_scoped_and_generation_uses_actual_context_budget() {
        let root = crate::media::tests::PrivateRoot::new();
        let mut first = image_worker(&root);
        let mut second = image_worker(&root);
        let prepared = post(&mut first, image_prepare(&root));
        post(&mut second, image_prepare(&root));
        assert_eq!(post(&mut second, image_generate(&prepared))["error"], "invalid_prepared_image");
        assert!(second.prepared.is_none());
        let mut fitting = image_generate(&prepared);
        fitting["max_new_tokens"] = json!(9);
        assert_eq!(post(&mut first, fitting)["finish_reason"], "eos");
        assert_eq!(first.backend.image_generations, 1);

        let prepared = post(&mut first, image_prepare(&root));
        let mut changed = image_generate(&prepared);
        changed["max_new_tokens"] = json!(10);
        assert_eq!(post(&mut first, changed)["error"], "context_overflow");
        assert!(first.prepared.is_none());
        assert_eq!(post(&mut first, image_generate(&prepared))["error"], "invalid_prepared_image");
        assert_eq!(first.backend.image_generations, 1);
        assert_eq!(second.backend.image_generations, 0);
    }

    #[test]
    fn preparation_failure_cancellation_and_context_overflow_allow_request_b() {
        let root = crate::media::tests::PrivateRoot::new();
        let mut worker = image_worker(&root);
        worker.backend.fail_image_prepare = true;
        assert_eq!(post(&mut worker, image_prepare(&root))["error"], "invalid_image_request");
        worker.backend.fail_image_prepare = false;
        worker.backend.cancel_after_prepare = true;
        assert_eq!(post(&mut worker, image_prepare(&root))["error"], "generation_cancelled");
        assert!(worker.prepared.is_none());
        assert_eq!(worker.backend.prepared_drops.load(Ordering::SeqCst), 1);
        worker.backend.cancel_after_prepare = false;
        worker.backend.cancel_image = false;
        let mut overflow = image_prepare(&root);
        overflow["context_tokens"] = json!(14);
        assert_eq!(post(&mut worker, overflow)["error"], "context_overflow");
        assert_eq!(worker.backend.prepared_drops.load(Ordering::SeqCst), 2);
        let prepared = post(&mut worker, image_prepare(&root));
        assert_eq!(post(&mut worker, image_generate(&prepared))["finish_reason"], "eos");
        assert_eq!(worker.backend.prepared_drops.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn known_text_and_output_overflow_is_rejected_before_opening_image_bytes() {
        let root = crate::media::tests::PrivateRoot::new();
        let mut worker = image_worker(&root);
        let mut request = image_prepare(&root);
        std::fs::remove_file(root.0.join(request["image"]["id"].as_str().unwrap())).unwrap();
        request["context_tokens"] = json!(8);
        assert_eq!(post(&mut worker, request)["error"], "context_overflow");
        assert_eq!(worker.backend.image_preparations, 0);
    }

    #[test]
    fn image_streaming_uses_the_existing_delta_and_terminal_contract() {
        let root = crate::media::tests::PrivateRoot::new();
        let mut worker = image_worker(&root);
        let prepared = post(&mut worker, image_prepare(&root));
        let mut request = image_generate(&prepared);
        request["stream"] = json!(true);
        let mut output = Vec::new();
        worker.process_line(&serde_json::to_vec(&request).unwrap(), &mut output).unwrap();
        let frames = output
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(frames.len(), 7);
        let progress: Vec<_> = frames.iter().filter(|frame| frame["event"] == "progress").collect();
        assert_eq!(progress.len(), 4);
        for (sampled_tokens, frame) in progress.into_iter().enumerate() {
            assert_eq!(frame["input_tokens"], 7);
            assert_eq!(frame["output_token_budget"], 8);
            assert_eq!(frame["sampled_tokens"], sampled_tokens);
        }
        let deltas: Vec<_> = frames.iter().filter(|frame| frame["event"] == "delta").collect();
        assert_eq!(deltas.len(), 2);
        assert_eq!(frames.last().unwrap()["input_tokens"], 7);
        assert_eq!(frames.last().unwrap()["finish_reason"], "eos");
        assert!(worker.backend.drained.load(Ordering::SeqCst));
        assert!(worker.prepared.is_none());
    }

    #[test]
    fn image_native_and_writer_failures_release_prepared_state_after_cleanup() {
        let root = crate::media::tests::PrivateRoot::new();
        let mut worker = image_worker(&root);
        let prepare = image_prepare(&root);
        assert_eq!(
            worker.process_line(
                &serde_json::to_vec(&prepare).unwrap(),
                &mut FailingWriter {
                    fail_flush: false
                }
            ),
            Err("protocol_write_failed")
        );
        assert!(worker.prepared.is_none());
        assert_eq!(worker.backend.prepared_drops.load(Ordering::SeqCst), 1);
        let prepared = post(&mut worker, image_prepare(&root));
        worker.backend.tokens = vec![Err("native_generation_failed")];
        let mut output = Vec::new();
        assert_eq!(
            worker.process_line(&serde_json::to_vec(&image_generate(&prepared)).unwrap(), &mut output),
            Err("native_operation_failed")
        );
        assert!(worker.backend.drained.load(Ordering::SeqCst));
        assert!(worker.prepared.is_none());
        assert_eq!(worker.backend.prepared_drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn family_readiness_preserves_artifact_identity_and_never_self_qualifies() {
        let root = crate::media::tests::PrivateRoot::new();
        let mut worker = image_worker(&root);
        worker.backend.image_schema = 3;
        let mut output = Vec::new();
        run_lines(&mut worker, &mut io::Cursor::new(Vec::<u8>::new()), &mut output).unwrap();
        let ready: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(ready["image_input"]["artifact_schema_version"], 3);
        assert_eq!(ready["image_input"]["family_profile"], "qwen35-dense-bf16-merged-v1");
        assert_eq!(ready["image_input"]["family_version"], 1);
        assert_eq!(ready["image_input"]["model_source"], "local:synthetic-merged-model");
        assert!(ready["image_input"]["model_revision"].is_null());
        assert_eq!(ready["image_input"]["qualified"], false);
        assert_eq!(ready["image_input"]["qualification"], "unverified");
        worker.backend.family_profile = "qwen35-dense-bf16-merged-v2";
        worker.backend.family_version = 2;
        let mut dynamic_output = Vec::new();
        run_lines(&mut worker, &mut io::Cursor::new(Vec::<u8>::new()), &mut dynamic_output).unwrap();
        let dynamic_ready: Value = serde_json::from_slice(&dynamic_output).unwrap();
        assert_eq!(dynamic_ready["image_input"]["family_profile"], "qwen35-dense-bf16-merged-v2");
        assert_eq!(dynamic_ready["image_input"]["family_version"], 2);
        assert_eq!(dynamic_ready["image_input"]["qualification"], "unverified");
        worker.backend.family_profile = "qwen35-dense-bf16-merged-v1";
        for (schema, version) in [(4, 1), (3, 2)] {
            worker.backend.image_schema = schema;
            worker.backend.family_version = version;
            let mut output = Vec::new();
            assert_eq!(
                run_lines(&mut worker, &mut io::Cursor::new(Vec::<u8>::new()), &mut output),
                Err("invalid_model_config")
            );
            assert!(output.is_empty());
        }
    }

    #[test]
    fn gemma_image_readiness_and_preparation_use_three_by_three_pooling() {
        let root = crate::media::tests::PrivateRoot::new();
        let mut worker = image_worker(&root);
        worker.backend.image_schema = 4;
        worker.backend.family_profile = "gemma4-e2b-bf16-v1";
        let mut output = Vec::new();
        run_lines(&mut worker, &mut io::Cursor::new(Vec::<u8>::new()), &mut output).unwrap();
        let ready: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(ready["image_input"]["artifact_schema_version"], 4);
        assert_eq!(ready["image_input"]["max_image_tokens"], 280);
        assert_eq!(ready["image_input"]["max_patches"], 2520);
        assert_eq!(ready["image_input"]["max_processed_pixels"], 645120);
        assert_eq!(ready["image_input"]["qualification"], "unverified");
        let prepared = post(&mut worker, image_prepare(&root));
        assert_eq!(prepared["image_tokens"], 1);
        assert_eq!(prepared["grid"], json!([1, 3, 3]));
        assert!(prepared.get("prepared_id").is_some());
        let metadata = PreparedImageMetadata {
            input_tokens: 7,
            image_tokens: 1,
            grid: [1, 3, 3],
            processed_dimensions: [48, 48],
        };
        assert!(valid_image_metadata(&metadata, (16, 3)));
        assert!(!valid_image_metadata(&metadata, (16, 2)));
        assert!(!valid_image_metadata(&metadata, (16, 0)));
        worker.backend.image_schema = 3;
        assert_eq!(
            run_lines(&mut worker, &mut io::Cursor::new(Vec::<u8>::new()), &mut Vec::new()),
            Err("invalid_model_config")
        );
    }

    #[test]
    fn image_readiness_is_separate_and_oversized_frames_and_eof_clear_preparation() {
        let root = crate::media::tests::PrivateRoot::new();
        let mut worker = image_worker(&root);
        worker.budget = Budget::for_mib(1).unwrap();
        post(&mut worker, image_prepare(&root));
        let mut output = Vec::new();
        run_lines(&mut worker, &mut io::Cursor::new(vec![b'x'; TEST_WIRE_BYTES + 1]), &mut output).unwrap();
        let frames = output
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(frames[0]["mode"], "text-generation-v1");
        assert_eq!(frames[0]["operations"], json!(["tokenize", "decode", "generate"]));
        assert_eq!(frames[0]["image_input"]["version"], "mediaref-v1");
        assert_eq!(frames[0]["image_input"]["artifact_schema_version"], 2);
        assert_eq!(frames[0]["image_input"]["family_profile"], "qwen35-dense-bf16-merged-v1");
        assert_eq!(frames[0]["image_input"]["family_version"], 1);
        assert_eq!(frames[0]["image_input"]["runtime_identity"]["backend"], "metal");
        assert_eq!(frames[0]["image_input"]["runtime_identity"]["device_name"], "Apple Synthetic GPU");
        assert_eq!(frames[0]["image_input"]["runtime_identity"]["apple_gpu_family"], 1008);
        assert_eq!(frames[0]["image_input"]["runtime_identity"]["gpu_core_count"], 32);
        assert_eq!(frames[0]["image_input"]["runtime_identity"]["supports_mxu"], false);
        for (name, expected) in [
            ("artifact_sha256", "b"),
            ("config_sha256", "a"),
            ("tokenizer_sha256", "c"),
            ("processor_sha256", "d"),
            ("template_sha256", "e"),
        ] {
            assert_eq!(frames[0]["image_input"][name], expected.repeat(64));
        }
        assert_eq!(frames[0]["image_input"]["qualified"], false);
        assert_eq!(frames[0]["image_input"]["qualification"], "unverified");
        assert_eq!(frames[1]["error"], "request_too_large");
        assert!(worker.prepared.is_none());
        post(&mut worker, image_prepare(&root));
        run_lines(&mut worker, &mut io::Cursor::new(Vec::<u8>::new()), &mut Vec::new()).unwrap();
        assert!(worker.prepared.is_none());
        let mut text = self::worker();
        let mut output = Vec::new();
        run_lines(&mut text, &mut io::Cursor::new(Vec::<u8>::new()), &mut output).unwrap();
        let ready: Value = serde_json::from_slice(&output).unwrap();
        assert!(ready.get("image_input").is_none());
    }

    #[test]
    fn stage_timing_separates_first_sample_from_subsequent_samples_and_cleanup() {
        let options = GenerationOptions {
            context_tokens: 16,
            max_new_tokens: 8,
            seed: 0,
            sampling: Sampling::Greedy {},
            mtp_drafts: 0,
            eos_token_ids: vec![1],
            extra_stop_token_ids: Vec::new(),
            stop: Vec::new(),
            request_id: 0,
            cache_scope: None,
        };
        let drained = Arc::new(AtomicBool::new(false));
        let mut clock = [Duration::from_secs(2), Duration::from_secs(3), Duration::from_secs(6)].into_iter();
        let result = collect_tokens_with_clock(
            TestStream {
                tokens: vec![Ok(3), Ok(4), Ok(1), Ok(7)].into(),
                drained: drained.clone(),
            },
            4,
            &options,
            8,
            || clock.next().unwrap(),
        )
        .unwrap();
        assert_eq!(result.prefill_seconds, Some(2.0));
        assert_eq!(result.decode_seconds, Some(4.0));
        assert_eq!(result.sampled_tokens, 3);
        assert_eq!(result.generated_token_ids.len(), 2);
        assert_eq!(tokens_per_second(4, result.prefill_seconds), Some(2.0));
        // The first sample belongs to prefill; terminal EOS is a decode sample.
        assert_eq!(tokens_per_second(result.sampled_tokens - 1, result.decode_seconds), Some(0.5));
        assert!(drained.load(Ordering::SeqCst));
        assert!(clock.next().is_none());
    }

    #[test]
    fn progress_observes_terminal_samples_with_the_same_stage_clock_as_the_result() {
        let options = GenerationOptions {
            context_tokens: 16,
            max_new_tokens: 8,
            seed: 0,
            sampling: Sampling::Greedy {},
            mtp_drafts: 0,
            eos_token_ids: vec![1],
            extra_stop_token_ids: Vec::new(),
            stop: Vec::new(),
            request_id: 0,
            cache_scope: None,
        };
        let mut clock = [Duration::from_secs(2), Duration::from_secs(3), Duration::from_secs(6)].into_iter();
        let mut progress = Vec::new();
        let mut emitted = Vec::new();
        let result = collect_tokens_with_clock_and_callback(
            vec![Ok(3), Ok(4), Ok(1), Ok(7)].into_iter(),
            4,
            &options,
            8,
            || clock.next().unwrap(),
            &mut |token| {
                emitted.push(token);
                Ok(true)
            },
            Some(&mut |sample| {
                progress.push(sample);
                Ok(())
            }),
            &mut || Ok(false),
        )
        .unwrap();
        assert_eq!(emitted, vec![3, 4]);
        assert_eq!(
            progress,
            vec![
                GenerationProgress {
                    sampled_tokens: 1,
                    elapsed_seconds: 2.0,
                    prefill_seconds: Some(2.0)
                },
                GenerationProgress {
                    sampled_tokens: 2,
                    elapsed_seconds: 3.0,
                    prefill_seconds: Some(2.0)
                },
                GenerationProgress {
                    sampled_tokens: 3,
                    elapsed_seconds: 6.0,
                    prefill_seconds: Some(2.0)
                },
            ]
        );
        assert_eq!(result.sampled_tokens, progress.last().unwrap().sampled_tokens);
        assert_eq!(result.prefill_seconds, progress[0].prefill_seconds);
        assert_eq!(result.decode_seconds, Some(4.0));
    }

    #[test]
    fn progress_reports_requested_budget_and_precedes_empty_eos_reply() {
        let mut worker = worker();
        worker.backend.tokens = vec![Ok(1)];
        let mut request = standard();
        request["context_tokens"] = json!(4);
        request["max_new_tokens"] = json!(3);
        request["stream"] = json!(true);
        let mut output = Vec::new();
        worker.process_line(&serde_json::to_vec(&request).unwrap(), &mut output).unwrap();
        let frames: Vec<Value> = output
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect();
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0]["event"], "progress");
        assert_eq!(frames[0]["sampled_tokens"], 0);
        assert_eq!(frames[0]["output_token_budget"], 3);
        assert_eq!(frames[1]["sampled_tokens"], 1);
        assert_eq!(frames[1]["output_token_budget"], 3);
        assert_eq!(frames[1]["elapsed_seconds"], frames[1]["prefill_seconds"]);
        assert_eq!(frames[2]["text"], "");
        assert_eq!(frames[2]["output_tokens"], 0);
        assert_eq!(frames[2]["sampled_tokens"], 1);
        assert_eq!(frames[2]["finish_reason"], "eos");
    }

    #[test]
    fn progress_is_stream_only_and_invalid_requests_do_not_begin_progress() {
        let mut worker = worker();
        let mut output = Vec::new();
        worker.process_line(&serde_json::to_vec(&standard()).unwrap(), &mut output).unwrap();
        let frame: Value = serde_json::from_slice(&output).unwrap();
        assert!(frame.get("event").is_none());
        assert_eq!(frame["finish_reason"], "eos");
        let mut request = standard();
        request["stream"] = json!(true);
        request["context_tokens"] = json!(1);
        output.clear();
        worker.process_line(&serde_json::to_vec(&request).unwrap(), &mut output).unwrap();
        let frame: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(frame, json!({"error":"context_overflow"}));
    }

    #[test]
    fn progress_stream_continues_beyond_former_cumulative_frame_budget() {
        let mut worker = worker();
        worker.budget = Budget {
            mib: 1,
            input_bytes: 16 * 1024,
            wire_bytes: 64 * 1024,
            template_live_bytes: 4096,
            template_cumulative_bytes: 4096,
        };
        worker.backend.tokens = vec![Ok(2); 128];
        let mut request = standard();
        request["context_tokens"] = json!(256);
        request["max_new_tokens"] = json!(128);
        request["stream"] = json!(true);
        let mut output = Vec::new();
        worker.process_line(&serde_json::to_vec(&request).unwrap(), &mut output).unwrap();
        let frames: Vec<Value> = output
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect();
        let progress: Vec<&Value> = frames.iter().filter(|frame| frame["event"] == "progress").collect();
        assert_eq!(progress.last().unwrap()["sampled_tokens"], 128);
        // The former cumulative cap was input_bytes / 16 = 1 KiB. The progress
        // frames for this request exceed that total, so reaching the final
        // sample proves late updates are still delivered.
        let progress_bytes: usize = frames
            .iter()
            .filter(|frame| frame["event"] == "progress")
            .map(|frame| serde_json::to_vec(frame).unwrap().len() + 1)
            .sum();
        assert!(progress_bytes > worker.budget.input_bytes / 16);
        assert_eq!(frames.last().unwrap()["finish_reason"], "length");
    }

    #[test]
    fn non_streamed_output_and_requested_token_budget_obey_resource_budget() {
        let mut worker = worker();
        worker.budget = Budget {
            mib: 1,
            input_bytes: 4,
            wire_bytes: 4096,
            template_live_bytes: 4096,
            template_cumulative_bytes: 4096,
        };
        let mut request = standard();
        request["max_new_tokens"] = json!(1);
        let reply = post(&mut worker, request);
        assert_eq!(reply["error"], "resource_budget_exceeded");
        assert_eq!(worker.backend.generation_calls, 1);

        let mut request = standard();
        request["max_new_tokens"] = json!(2);
        let reply = post(&mut worker, request);
        assert_eq!(reply["error"], "resource_budget_exceeded");
        assert_eq!(worker.backend.generation_calls, 1);
    }

    #[test]
    fn progress_callback_failure_drains_without_returning_a_successful_result() {
        let drained = Arc::new(AtomicBool::new(false));
        let options = GenerationOptions {
            context_tokens: 8,
            max_new_tokens: 4,
            seed: 0,
            sampling: Sampling::default(),
            mtp_drafts: 0,
            eos_token_ids: vec![1],
            extra_stop_token_ids: Vec::new(),
            stop: Vec::new(),
            request_id: 0,
            cache_scope: None,
        };
        let mut samples = Vec::new();
        let result = collect_tokens_controlled(
            TestStream {
                tokens: vec![Ok(3), Ok(4), Ok(1)].into(),
                drained: drained.clone(),
            },
            1,
            &options,
            8,
            Instant::now(),
            &mut |_| panic!("text callback must not run after progress callback failure"),
            Some(&mut |sample| {
                samples.push(sample.sampled_tokens);
                Err("protocol_write_failed")
            }),
            &mut || Ok(false),
        );
        assert!(result.is_err_and(|code| code == "protocol_write_failed"));
        assert_eq!(samples, vec![1]);
        assert!(drained.load(Ordering::SeqCst));
    }

    #[test]
    fn cleanup_failure_fences_success_and_cancellation_before_the_result_is_visible() {
        struct LookaheadStream {
            fence: Arc<AtomicBool>,
            fail_on_drop: bool,
        }
        impl Iterator for LookaheadStream {
            type Item = Result<u64, &'static str>;
            fn next(&mut self) -> Option<Self::Item> {
                Some(Ok(3))
            }
        }
        impl Drop for LookaheadStream {
            fn drop(&mut self) {
                self.fence.store(self.fail_on_drop, Ordering::SeqCst);
            }
        }
        for cancelled in [false, true] {
            for fail_on_drop in [false, true] {
                let fence = Arc::new(AtomicBool::new(false));
                let options = GenerationOptions {
                    context_tokens: 8,
                    max_new_tokens: 1,
                    seed: 0,
                    sampling: Sampling::default(),
                    mtp_drafts: 0,
                    eos_token_ids: vec![1],
                    extra_stop_token_ids: Vec::new(),
                    stop: Vec::new(),
                    request_id: 1,
                    cache_scope: None,
                };
                let result = collect_tokens_controlled(
                    LookaheadStream {
                        fence: fence.clone(),
                        fail_on_drop,
                    },
                    1,
                    &options,
                    8,
                    Instant::now(),
                    &mut |_| Ok(true),
                    None,
                    &mut || Ok(cancelled),
                );
                let result = native_result_after_cleanup(result, fence.load(Ordering::SeqCst));
                if fail_on_drop {
                    assert!(result.is_err_and(|code| code == "native_generation_failed"));
                } else if cancelled {
                    assert!(result.is_err_and(|code| code == "generation_cancelled"));
                } else {
                    assert!(result.is_ok());
                }
            }
        }
    }

    #[test]
    fn one_sample_and_unmeasurable_durations_do_not_invent_decode_throughput() {
        let mut worker = worker();
        worker.backend.tokens = vec![Ok(1)];
        let value = post(&mut worker, standard());
        assert_eq!(value["timing_method"], "first-token-wall-clock-v1");
        assert!(value["prefill_seconds"].is_number());
        assert!(value["decode_seconds"].is_null());
        assert!(value["output_tokens_per_second"].is_null());
        for seconds in [None, Some(0.0), Some(-1.0), Some(f64::NAN), Some(f64::INFINITY)] {
            assert_eq!(tokens_per_second(2, seconds), None);
        }
        assert_eq!(tokens_per_second(0, Some(1.0)), None);
    }

    #[test]
    fn stops_exclude_terminal_token_and_preserve_exact_text_after_cleanup() {
        let mut worker = worker();
        let value = post(&mut worker, standard());
        assert_eq!(value["generated_token_ids"], json!([3, 4]));
        assert_eq!(value["text"], "résultat\n");
        assert_eq!(value["finish_reason"], "eos");
        assert_eq!(value["stop_token_id"], 1);
        assert_eq!(value["output_tokens"], 2);
        assert_eq!(value["sampled_tokens"], 3);
        assert!(worker.backend.drained.load(Ordering::SeqCst));
        let mut request = standard();
        request["stop_token_ids"] = json!([4]);
        let value = post(&mut worker, request);
        assert_eq!(value["generated_token_ids"], json!([3]));
        assert_eq!(value["finish_reason"], "stop_token");
        assert_eq!(value["stop_token_id"], 4);
        assert_eq!(value["sampled_tokens"], 2);
    }

    #[test]
    fn streaming_flushes_deltas_before_native_cleanup_and_preserves_final_reply() {
        struct LiveWriter {
            pending: Vec<u8>,
            frames: Vec<Value>,
            drained: Arc<AtomicBool>,
        }
        impl Write for LiveWriter {
            fn write(
                &mut self,
                bytes: &[u8],
            ) -> io::Result<usize> {
                self.pending.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                let value: Value = serde_json::from_slice(&self.pending).unwrap();
                if value["event"] == "progress" && value["sampled_tokens"] == 0 {
                    assert!(self.drained.load(Ordering::SeqCst));
                } else if value.get("event").is_some() {
                    assert!(!self.drained.load(Ordering::SeqCst));
                } else {
                    assert!(self.drained.load(Ordering::SeqCst));
                }
                self.frames.push(value);
                self.pending.clear();
                Ok(())
            }
        }
        let mut worker = worker();
        let mut output = LiveWriter {
            pending: Vec::new(),
            frames: Vec::new(),
            drained: worker.backend.drained.clone(),
        };
        let mut request = standard();
        request["stream"] = json!(true);
        worker.process_line(&serde_json::to_vec(&request).unwrap(), &mut output).unwrap();
        assert_eq!(output.frames.len(), 7);
        let progress: Vec<_> = output.frames.iter().filter(|frame| frame["event"] == "progress").collect();
        assert_eq!(progress.len(), 4);
        assert_eq!(progress[0]["input_tokens"], 1);
        assert_eq!(progress[0]["sampled_tokens"], 0);
        assert_eq!(progress[0]["output_token_budget"], 8);
        assert_eq!(progress[0]["elapsed_seconds"], 0.0);
        assert!(progress[0]["prefill_seconds"].is_null());
        for (sampled_tokens, frame) in progress.iter().enumerate().skip(1) {
            assert_eq!(frame["sampled_tokens"], sampled_tokens);
            assert_eq!(frame["prefill_seconds"], progress[1]["prefill_seconds"]);
            assert!(frame["elapsed_seconds"].as_f64().unwrap() >= frame["prefill_seconds"].as_f64().unwrap());
            assert_eq!(frame.as_object().unwrap().len(), 6);
        }
        let deltas: Vec<_> = output.frames.iter().filter(|frame| frame["event"] == "delta").collect();
        assert_eq!(*deltas[0], json!({"event":"delta","text":"ré"}));
        assert_eq!(*deltas[1], json!({"event":"delta","text":"sultat\n"}));
        let terminal = output.frames.last().unwrap();
        assert_eq!(terminal["text"], "résultat\n");
        assert_eq!(terminal["generated_token_ids"], json!([3, 4]));
        assert_eq!(terminal["finish_reason"], "eos");
        assert_eq!(terminal["sampled_tokens"], progress.last().unwrap()["sampled_tokens"]);
        assert_eq!(terminal["prefill_seconds"], progress[1]["prefill_seconds"]);
    }

    #[test]
    fn streaming_excludes_stops_and_observer_failures_drain_native_work() {
        let mut worker = worker();
        let mut request = standard();
        request["stream"] = json!(true);
        request["stop_token_ids"] = json!([4]);
        let mut output = Vec::new();
        worker.process_line(&serde_json::to_vec(&request).unwrap(), &mut output).unwrap();
        let frames: Vec<Value> = output
            .split(|b| *b == b'\n')
            .filter(|b| !b.is_empty())
            .map(|b| serde_json::from_slice(b).unwrap())
            .collect();
        assert_eq!(frames.len(), 5);
        assert_eq!(frames[2], json!({"event":"delta","text":"ré"}));
        assert_eq!(frames[3]["sampled_tokens"], 2);
        assert_eq!(frames[4]["finish_reason"], "stop_token");
        for fail_flush in [false, true] {
            let mut output = FailingWriter {
                fail_flush,
            };
            assert_eq!(
                worker.process_line(&serde_json::to_vec(&request).unwrap(), &mut output),
                Err("protocol_write_failed")
            );
            assert!(worker.backend.drained.load(Ordering::SeqCst));
        }
        worker.backend.tokens = vec![Ok(1)];
        let empty = post(&mut worker, request);
        assert_eq!(empty["text"], "");
        assert_eq!(empty["finish_reason"], "eos");
    }

    #[test]
    fn model_eos_cannot_be_overridden_and_eos_at_first_token_is_empty_text() {
        let mut worker = worker();
        worker.backend.tokens = vec![Ok(1), Ok(3)];
        let mut request = standard();
        request["stop_token_ids"] = json!([1, 4]);
        let value = post(&mut worker, request);
        assert_eq!(value["generated_token_ids"], json!([]));
        assert_eq!(value["text"], "");
        assert_eq!(value["finish_reason"], "eos");
        assert_eq!(value["sampled_tokens"], 1);
    }

    #[test]
    fn length_and_context_limits_never_consume_an_extra_output_token() {
        let mut worker = worker();
        for (context, maximum, reason) in [(16, 1, "length"), (2, 1, "length")] {
            let mut request = standard();
            request["context_tokens"] = json!(context);
            request["max_new_tokens"] = json!(maximum);
            let value = post(&mut worker, request);
            assert_eq!(value["generated_token_ids"], json!([3]));
            assert_eq!(value["sampled_tokens"], 1);
            assert_eq!(value["finish_reason"], reason);
            assert!(worker.backend.drained.load(Ordering::SeqCst));
        }
        let mut request = standard();
        request["context_tokens"] = json!(2);
        request["max_new_tokens"] = json!(8);
        let value = post(&mut worker, request);
        assert_eq!(value["error"], "context_overflow");
        assert_eq!(worker.backend.generation_calls, 2);
    }

    #[test]
    fn each_generation_dispatches_fresh_input_seed_and_sampling() {
        let mut worker = worker();
        let mut request = standard();
        request["seed"] = json!(41);
        request["sampling"] = json!({"mode":"stochastic","temperature":0.6,"top_k":4,"top_p":0.95});
        let first = post(&mut worker, request.clone());
        let second = post(&mut worker, request);
        assert_eq!(first["generated_token_ids"], second["generated_token_ids"]);
        assert_eq!(worker.backend.generation_calls, 2);
        assert_eq!(worker.backend.inputs, vec![vec![2], vec![2]]);
        assert_eq!(worker.backend.seeds, vec![41, 41]);
        assert_eq!(worker.backend.sampling[0], worker.backend.sampling[1]);
        let prompt =
            json!({"operation":"generate","rendered_prompt":"rendered text", "context_tokens":16,"max_new_tokens":8});
        post(&mut worker, prompt);
        assert_eq!(worker.backend.inputs.last(), Some(&vec![2]));
        assert_eq!(worker.backend.seeds.last(), Some(&0));
        assert_eq!(worker.backend.sampling.last(), Some(&Sampling::Greedy {}));
    }

    #[test]
    fn image_metadata_accepts_long_text_contexts() {
        assert!(valid_image_metadata(
            &PreparedImageMetadata {
                input_tokens: 16_385,
                image_tokens: 4,
                grid: [1, 4, 4],
                processed_dimensions: [64, 64],
            },
            (16, 2)
        ));
    }

    #[test]
    fn gemma3_preparation_requires_its_complete_fixed_grid() {
        let mut metadata = PreparedImageMetadata {
            input_tokens: 273,
            image_tokens: 256,
            grid: [1, 64, 64],
            processed_dimensions: [896, 896],
        };
        assert!(valid_image_metadata(&metadata, (14, 4)));
        assert!(!valid_image_metadata(&metadata, (16, 2)));
        metadata.grid = [1, 32, 64];
        metadata.processed_dimensions = [896, 448];
        metadata.image_tokens = 128;
        assert!(!valid_image_metadata(&metadata, (14, 4)));
        metadata.grid = [1, 64, 64];
        metadata.processed_dimensions = [896, 896];
        assert!(!valid_image_metadata(&metadata, (14, 4)));
    }

    #[test]
    fn context_capacity_uses_the_loaded_model_maximum() {
        let mut worker = worker();
        worker.backend.max_context = Some(262_144);
        let mut output = Vec::new();
        run_lines(&mut worker, &mut std::io::Cursor::new(Vec::<u8>::new()), &mut output).unwrap();
        let ready: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(ready["max_context_tokens"], 262_144);

        let mut request = standard();
        request["context_tokens"] = json!(262_144);
        // Exercise an actual prompt beyond the former 16K boundary.
        request["input_token_ids"] = json!(vec![2; 16_385]);
        assert_eq!(post(&mut worker, request.clone())["finish_reason"], "eos");
        assert_eq!(worker.backend.generation_calls, 1);
        request["context_tokens"] = json!(262_145);
        assert_eq!(post(&mut worker, request)["error"], "context_overflow");
        assert_eq!(worker.backend.generation_calls, 1);
        assert_eq!(post(&mut worker, standard())["finish_reason"], "eos");
    }

    #[test]
    fn context_without_a_finite_model_maximum_uses_the_protocol_integer_range() {
        let mut worker = worker();
        worker.backend.max_context = None;
        let mut output = Vec::new();
        run_lines(&mut worker, &mut std::io::Cursor::new(Vec::<u8>::new()), &mut output).unwrap();
        let ready: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(ready["max_context_tokens"], u32::MAX);
        let mut request = standard();
        request["context_tokens"] = json!(u32::MAX);
        assert_eq!(post(&mut worker, request)["finish_reason"], "eos");
    }

    #[test]
    fn bad_request_does_not_dispatch_native_generation_or_poison_following_request() {
        let mut worker = worker();
        let cases = [
            json!({"context_tokens":0}),
            json!({"context_tokens":16385}),
            json!({"context_tokens":1025}),
            json!({"max_new_tokens":0}),
            json!({"input_token_ids":[]}),
            json!({"input_token_ids":[8]}),
            json!({"input_token_ids":[2,3],"context_tokens":2}),
            json!({"stop_token_ids":[8]}),
            json!({"stop_token_ids":[1,1]}),
            json!({"rendered_prompt":"both sources"}),
            json!({"input_token_ids":null}),
            json!({"sampling":{"mode":"stochastic","temperature":0.0}}),
            json!({"sampling":{"mode":"stochastic","temperature":1e-40}}),
            json!({"sampling":{"mode":"stochastic","temperature":0.6,"top_k":0}}),
            json!({"sampling":{"mode":"stochastic","temperature":0.6,"top_k":9}}),
            json!({"sampling":{"mode":"stochastic","temperature":0.6,"top_p":0.0}}),
            json!({"sampling":{"mode":"stochastic","temperature":0.6,"top_p":1.1}}),
            json!({"sampling":{"mode":"stochastic","temperature":0.6,"min_p":-0.1}}),
            json!({"sampling":{"mode":"stochastic","temperature":0.6,"min_p":1.1}}),
        ];
        for case in cases {
            let mut request = standard();
            request.as_object_mut().unwrap().extend(case.as_object().unwrap().clone());
            assert!(post(&mut worker, request).get("error").is_some());
        }
        assert_eq!(worker.backend.generation_calls, 0);
        assert_eq!(post(&mut worker, standard())["finish_reason"], "eos");
        assert_eq!(worker.backend.generation_calls, 1);
    }

    #[test]
    fn unknown_fields_and_wrong_types_are_rejected_at_every_level() {
        for patch in [
            json!({"private":"content"}),
            json!({"input_token_ids":[-1]}),
            json!({"input_token_ids":[1.5]}),
            json!({"seed":-1}),
            json!({"sampling":{"mode":"greedy","temperature":0.6}}),
            json!({"sampling":{"mode":"stochastic","temperature":0.6,"repetition_penalty":1.1}}),
        ] {
            let mut value = standard();
            value.as_object_mut().unwrap().extend(patch.as_object().unwrap().clone());
            assert!(serde_json::from_value::<Request>(value).is_err());
        }
        let request: Request = serde_json::from_value(standard()).unwrap();
        assert!(matches!(
            request,
            Request::Generate {
                seed: 0,
                sampling: Sampling::Greedy {},
                ..
            }
        ));
        assert!(
            Sampling::Stochastic {
                temperature: f32::NAN,
                top_k: None,
                top_p: None,
                min_p: None
            }
            .validate(8)
            .is_err()
        );
        assert!(
            Sampling::Stochastic {
                temperature: 0.6,
                top_k: None,
                top_p: Some(f32::INFINITY),
                min_p: None
            }
            .validate(8)
            .is_err()
        );
    }

    #[test]
    fn native_errors_invalid_ids_and_early_exhaustion_drain_then_fence_worker() {
        for tokens in [vec![Err("native_generation_failed")], vec![Ok(8)], vec![]] {
            let mut worker = worker();
            worker.backend.tokens = tokens;
            let mut output = Vec::new();
            assert_eq!(
                worker.process_line(&serde_json::to_vec(&standard()).unwrap(), &mut output),
                Err("native_operation_failed")
            );
            assert!(serde_json::from_slice::<Value>(&output).unwrap().get("error").is_some());
            assert!(worker.backend.drained.load(Ordering::SeqCst));
        }
        let mut worker = worker();
        worker.backend.invalid_tokenization = true;
        assert_eq!(
            worker.process_line(br#"{"operation":"tokenize","text":"secret"}"#, &mut Vec::new()),
            Err("native_operation_failed")
        );
    }

    #[test]
    fn framed_protocol_recovers_from_malformed_and_oversized_lines() {
        let mut worker = worker();
        worker.budget = Budget::for_mib(1).unwrap();
        let mut bytes = b"{\n".to_vec();
        bytes.extend(vec![b'x'; TEST_WIRE_BYTES + 30]);
        bytes.extend(b"\n");
        bytes.extend(serde_json::to_vec(&standard()).unwrap());
        bytes.extend(b"\n");
        let mut output = Vec::new();
        run_lines(&mut worker, &mut io::Cursor::new(bytes), &mut output).unwrap();
        let replies: Vec<Value> = output
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect();
        assert_eq!(replies.len(), 4);
        assert_eq!(replies[0]["ready"], true);
        assert_eq!(replies[1], json!({"error":"invalid_request"}));
        assert_eq!(replies[2], json!({"error":"request_too_large"}));
        assert_eq!(replies[3]["finish_reason"], "eos");
        assert_eq!(worker.backend.generation_calls, 1);
    }

    #[test]
    fn tokenize_and_decode_helpers_preserve_specials_and_support_empty_vectors() {
        let mut worker = worker();
        assert_eq!(post(&mut worker, json!({"operation":"tokenize","text":""}))["token_ids"], json!([2]));
        assert_eq!(
            post(&mut worker, json!({"operation":"tokenize","text":"x","add_special_tokens":true}))["token_ids"],
            json!([0, 2])
        );
        assert_eq!(
            post(&mut worker, json!({"operation":"decode","token_ids":[7,3,4,1]}))["text"],
            "<tool_call>résultat\n<eos>"
        );
        assert_eq!(
            post(&mut worker, json!({"operation":"decode","token_ids":[7,3,4,1],"skip_special_tokens":true}))["text"],
            "résultat\n"
        );
        assert_eq!(post(&mut worker, json!({"operation":"decode","token_ids":[]}))["text"], "");
        assert_eq!(
            post(&mut worker, json!({"operation":"decode","token_ids":[8]})),
            json!({"error":"invalid_decoding_request"})
        );
    }

    struct FailingWriter {
        fail_flush: bool,
    }
    impl Write for FailingWriter {
        fn write(
            &mut self,
            bytes: &[u8],
        ) -> io::Result<usize> {
            if self.fail_flush {
                Ok(bytes.len())
            } else {
                Err(io::Error::other("synthetic failure"))
            }
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("synthetic failure"))
        }
    }
    #[test]
    fn output_write_and_flush_failures_are_fatal_after_stream_cleanup() {
        for fail_flush in [false, true] {
            let mut worker = worker();
            assert_eq!(
                worker.process_line(
                    &serde_json::to_vec(&standard()).unwrap(),
                    &mut FailingWriter {
                        fail_flush
                    }
                ),
                Err("protocol_write_failed")
            );
            assert!(worker.backend.drained.load(Ordering::SeqCst));
        }
    }

    const SCOPE: &str = "0123456789abcdef0123456789abcdef";
    const OTHER_SCOPE: &str = "fedcba9876543210fedcba9876543210";

    fn prefix_worker(supported: bool) -> GenerationWorker<FakeBackend> {
        let mut backend = worker().backend;
        backend.max_context = Some(16_384);
        backend.prefix_supported = supported;
        GenerationWorker::new_with_options(backend, None, true).unwrap()
    }

    fn cached_request(
        length: usize,
        scope: Option<&str>,
    ) -> Value {
        let mut request = json!({"operation":"generate","input_token_ids":vec![2; length],
            "context_tokens":8192,"max_new_tokens":8});
        if let Some(scope) = scope {
            request["cache_scope"] = json!(scope);
        }
        request
    }

    fn reused(
        worker: &mut GenerationWorker<FakeBackend>,
        length: usize,
    ) -> Value {
        post(worker, cached_request(length, Some(SCOPE)))["reused_input_tokens"].clone()
    }

    fn readiness(worker: &mut GenerationWorker<FakeBackend>) -> Value {
        let mut output = Vec::new();
        run_lines(worker, &mut io::Cursor::new(Vec::<u8>::new()), &mut output).unwrap();
        serde_json::from_slice(&output).unwrap()
    }

    #[test]
    fn prefix_cache_readiness_reports_policy_only_when_requested() {
        let plain = readiness(&mut worker());
        assert_eq!(plain["state_policy"], "fresh-per-request");
        assert!(plain.get("prefix_cache").is_none());
        assert_eq!(plain["operations"], json!(["tokenize", "decode", "generate"]));

        let unsupported = readiness(&mut prefix_worker(false));
        assert_eq!(unsupported["state_policy"], "fresh-per-request");
        assert_eq!(unsupported["prefix_cache"]["supported"], false);
        assert_eq!(unsupported["prefix_cache"]["enabled"], false);
        assert_eq!(unsupported["operations"], json!(["tokenize", "decode", "generate"]));

        let enabled = readiness(&mut prefix_worker(true));
        assert_eq!(enabled["state_policy"], "consecutive-prefix-v1");
        assert_eq!(
            enabled["prefix_cache"],
            json!({"supported":true,"enabled":true,"chunk_tokens":1024,"idle_seconds":600,
                "key":"model-instance+context-tokens+cache-scope"})
        );
        assert_eq!(enabled["operations"], json!(["tokenize", "decode", "generate", "reset_cache"]));
    }

    #[test]
    fn planning_and_execution_share_available_memory_after_host_reserve() {
        const MIB: usize = 1024 * 1024;
        let available = 1024 * MIB;
        let planned = inference_memory_budget(available);
        assert_eq!(planned, 768 * MIB);
        // A valid allocation above the former quarter-RAM threshold fits.
        assert_eq!(admit_request_memory(Some(512 * MIB), available), Ok(()));
        assert_eq!(admit_request_memory(Some(planned), available), Ok(()));
        assert_eq!(admit_request_memory(Some(planned + 1), available), Err("resource_budget_exceeded"));
        assert_eq!(inference_memory_budget(0), 0);
        assert_eq!(inference_memory_budget(256 * MIB), 0);
        assert_eq!(admit_request_memory(Some(1), 256 * MIB), Err("resource_budget_exceeded"));
        assert_eq!(admit_request_memory(None, available), Err("resource_budget_exceeded"));
        assert_eq!(admit_request_memory(Some(usize::MAX), usize::MAX), Err("resource_budget_exceeded"));
        assert_eq!(inference_memory_budget(usize::MAX), usize::MAX - 256 * MIB);
    }

    #[test]
    fn count_only_context_planning_remains_bounded_and_compatible() {
        let mut worker = prefix_worker(true);
        worker.backend.max_context = None;
        let reply = post(&mut worker, json!({"operation":"plan_generation","input_tokens":8,"context_tokens":128}));
        assert_eq!(reply["context_tokens"], 128);
        assert_eq!(reply["max_new_tokens"], 120);
        assert_eq!(reply["context_capacity_known"], false);
        assert_eq!(reply["budget_kind"], "memory");
        assert_eq!(worker.backend.plans[0].input_len, None);

        let invalid = post(&mut worker, json!({"operation":"plan_generation","input_tokens":0,"context_tokens":128}));
        assert_eq!(invalid["error"], "invalid_generation_request");
        assert_eq!(worker.backend.plans.len(), 1);
    }

    #[test]
    fn unknown_context_planning_rejects_unrepresentable_limits_and_no_memory_fit() {
        let mut worker = worker();
        worker.backend.max_context = None;
        let overflow = post(
            &mut worker,
            json!({"operation":"plan_generation","input_tokens":4294967295u64,"context_tokens":4294967295u64}),
        );
        assert_eq!(overflow["error"], "invalid_generation_request");
        worker.backend.planning_allowed = false;
        let reply = post(&mut worker, json!({"operation":"plan_generation","input_tokens":1,"context_tokens":16}));
        assert_eq!(reply["error"], "resource_budget_exceeded");
    }

    #[test]
    fn planning_receives_validated_prefix_identity_and_effective_mtp_selection() {
        let mut worker = prefix_worker(true);
        worker.backend.max_context = None;
        worker.backend.mtp = Some(capabilities(3, 2));
        worker.mtp = worker.backend.mtp;
        let request = json!({"operation":"plan_generation","input_tokens":2,"context_tokens":128,
            "input_token_ids":[2,3],"cache_scope":SCOPE,"mtp_drafts":2,
            "sampling":{"mode":"stochastic","temperature":0.7,"top_p":0.9}});
        assert_eq!(post(&mut worker, request.clone())["context_tokens"], 128);
        assert_eq!(
            worker.backend.plans,
            vec![RecordedPlan {
                input_tokens: 2,
                maximum: 128,
                input_len: Some(2),
                cache_scope: prefix_cache::parse_scope(SCOPE),
                mtp_drafts: 2,
                stochastic: true,
            }]
        );
        worker.backend.mtp_effective_override = Some(0);
        assert_eq!(post(&mut worker, request)["context_tokens"], 128);
        assert_eq!(worker.backend.plans[1].mtp_drafts, 0);
        let advertised = readiness(&mut worker);
        assert_eq!(advertised["generation_budget_inputs"], "token-ids-v1");
    }

    #[test]
    fn malformed_planning_identity_is_rejected_before_backend_inventory() {
        for extra in [
            json!({"input_token_ids":[2]}),
            json!({"input_token_ids":[2,8]}),
            json!({"input_token_ids":null}),
            json!({"cache_scope":"invalid"}),
            json!({"mtp":false,"mtp_drafts":0}),
            json!({"sampling":{"mode":"stochastic","temperature":0.0}}),
        ] {
            let mut worker = prefix_worker(true);
            worker.backend.max_context = None;
            let mut request = json!({"operation":"plan_generation","input_tokens":2,"context_tokens":128});
            request.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            assert!(post(&mut worker, request).get("error").is_some());
            assert!(worker.backend.plans.is_empty());
        }
        let mut worker = worker();
        worker.backend.max_context = None;
        let reply = post(
            &mut worker,
            json!({"operation":"plan_generation","input_tokens":2,
            "context_tokens":128,"input_token_ids":[2,3],"cache_scope":SCOPE}),
        );
        assert_eq!(reply["error"], "invalid_generation_request");
        assert!(worker.backend.plans.is_empty());
    }

    #[test]
    fn automatic_context_planning_honors_larger_current_allowance() {
        let mut worker = worker();
        worker.backend.max_context = None;
        let first = post(&mut worker, json!({"operation":"plan_generation","input_tokens":8,"context_tokens":128}));
        assert_eq!(first["context_tokens"], 128);
        // The previous plan still exceeds half of the newly admitted span.
        // Reuse preferences must not silently retain that smaller allowance.
        let larger = post(&mut worker, json!({"operation":"plan_generation","input_tokens":8,"context_tokens":200}));
        assert_eq!(larger["context_tokens"], 200);
        assert_eq!(larger["max_new_tokens"], 192);
    }

    #[test]
    fn prefix_cache_reports_the_backend_chunk_and_rejects_misaligned_reuse() {
        let coarse = || {
            let mut backend = prefix_worker(true).backend;
            backend.prefix_chunk = 4096;
            GenerationWorker::new_with_options(backend, None, true).unwrap()
        };
        assert_eq!(readiness(&mut coarse())["prefix_cache"]["chunk_tokens"], 4096);
        let mut aligned = coarse();
        aligned.backend.prefix_reuse_override = Some(4096);
        assert_eq!(reused(&mut aligned, 8_000), 4096);
        // A backend contradicting its own grid or the input length fences the worker.
        for reuse in [2048, 8_000] {
            let mut worker = coarse();
            worker.backend.prefix_reuse_override = Some(reuse);
            let mut output = Vec::new();
            let request = serde_json::to_vec(&cached_request(8_000, Some(SCOPE))).unwrap();
            assert_eq!(worker.process_line(&request, &mut output), Err("native_operation_failed"), "{reuse}");
            assert_eq!(serde_json::from_slice::<Value>(&output).unwrap()["error"], "invalid_generation_output");
            assert_eq!(worker.backend.generation_calls, 0);
            assert!(!worker.backend.prefix_cache_retained());
        }
        for chunk in [0, 512, 1536] {
            let mut backend = prefix_worker(true).backend;
            backend.prefix_chunk = chunk;
            let mut invalid = GenerationWorker::new_with_options(backend, None, true).unwrap();
            assert_eq!(readiness(&mut invalid)["prefix_cache"]["enabled"], false, "{chunk}");
        }
    }

    #[test]
    fn prefix_cache_fields_are_rejected_unless_enabled() {
        let mut plain = worker();
        plain.backend.max_context = Some(16_384);
        for mut disabled in [plain, prefix_worker(false)] {
            assert_eq!(post(&mut disabled, cached_request(4, Some(SCOPE)))["error"], "invalid_generation_request");
            assert_eq!(post(&mut disabled, json!({"operation":"reset_cache"}))["error"], "invalid_request");
            let reply = post(&mut disabled, cached_request(4, None));
            assert!(reply.get("error").is_none());
            assert!(reply.get("reused_input_tokens").is_none());
            assert_eq!(disabled.backend.generation_calls, 1);
        }
        let mut enabled = prefix_worker(true);
        for invalid in [json!(null), json!(7), json!(""), json!("0123456789ABCDEF0123456789ABCDEF"), json!("00")] {
            let mut request = cached_request(4, None);
            request["cache_scope"] = invalid;
            let error = post(&mut enabled, request)["error"].clone();
            assert!(error == "invalid_generation_request" || error == "invalid_request", "{error}");
        }
        assert_eq!(post(&mut enabled, json!({"operation":"reset_cache","extra":1}))["error"], "invalid_request");
        assert_eq!(enabled.backend.generation_calls, 0);
    }

    #[test]
    fn prefix_cache_reuses_aligned_rows_and_reports_processed_rate() {
        let mut worker = prefix_worker(true);
        assert_eq!(reused(&mut worker, 3_000), 0);
        assert_eq!(worker.backend.state_allocations, 1);
        let mut request = cached_request(3_500, Some(SCOPE));
        request["stream"] = json!(true);
        let mut output = Vec::new();
        worker.process_line(&serde_json::to_vec(&request).unwrap(), &mut output).unwrap();
        let frames: Vec<Value> = output
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect();
        let reply = frames.last().unwrap();
        assert_eq!(reply["reused_input_tokens"], 2048);
        assert_eq!(reply["input_tokens"], 3_500);
        let rate = reply["input_tokens_per_second"].as_f64();
        let expected = tokens_per_second(3_500 - 2_048, reply["prefill_seconds"].as_f64());
        // serde_json may parse a float one ulp off; counting one reused token would move the rate by 1/1452.
        assert!(
            match (rate, expected) {
                (Some(rate), Some(expected)) => (rate - expected).abs() < expected * 1e-12,
                _ => rate == expected,
            },
            "the input rate counts processed tokens only: {rate:?} != {expected:?}"
        );
        let progress: Vec<&Value> = frames.iter().filter(|frame| frame["event"] == "progress").collect();
        assert!(!progress.is_empty());
        assert!(progress.iter().all(|frame| frame["reused_input_tokens"] == 2048));
        assert_eq!(worker.backend.state_allocations, 1, "a hit reuses the retained state");
        // Each request keeps at least its own final chunk to prefill.
        assert_eq!(reused(&mut worker, 3_072), 2048);
        assert_eq!(reused(&mut worker, 2_048), 1024);
        assert_eq!(reused(&mut worker, 1_024), 0);
    }

    #[test]
    fn prefix_cache_key_mismatches_drop_the_entry_before_allocating() {
        let mut worker = prefix_worker(true);
        assert_eq!(reused(&mut worker, 3_000), 0);
        let mut other_context = cached_request(3_000, Some(SCOPE));
        other_context["context_tokens"] = json!(4096);
        assert_eq!(post(&mut worker, other_context)["reused_input_tokens"], 0);
        assert_eq!(reused(&mut worker, 3_000), 0, "a context change replaced the entry");
        assert_eq!(post(&mut worker, cached_request(3_000, Some(OTHER_SCOPE)))["reused_input_tokens"], 0);
        assert_eq!(reused(&mut worker, 3_000), 0, "a scope change replaced the entry");
        assert_eq!(reused(&mut worker, 3_000), 2048);
        let unscoped = post(&mut worker, cached_request(3_000, None));
        assert_eq!(unscoped["reused_input_tokens"], 0);
        assert!(!worker.backend.prefix_cache_retained(), "unscoped requests retain nothing");
        assert_eq!(reused(&mut worker, 3_000), 0);
        let mut diverged = cached_request(3_000, Some(SCOPE));
        diverged["input_token_ids"][1500] = json!(3);
        assert_eq!(post(&mut worker, diverged)["reused_input_tokens"], 1024);
        // Only the two hits reused a state; every other request allocated.
        assert_eq!(worker.backend.state_allocations, 7);
    }

    #[test]
    fn prefix_cache_is_cleared_by_errors_resets_and_image_operations() {
        let invalidations: [(&str, Value); 6] = [
            (
                "context_overflow",
                json!({"operation":"generate","input_token_ids":[2,2,2],"context_tokens":2,
                "max_new_tokens":1,"cache_scope":SCOPE}),
            ),
            ("invalid_request", json!({"operation":"generate","unknown":true})),
            ("invalid_decoding_request", json!({"operation":"decode","token_ids":[99]})),
            ("cache_reset", json!({"operation":"reset_cache"})),
            ("discarded", json!({"operation":"discard_image"})),
            (
                "images_not_supported",
                json!({"operation":"prepare_image","rendered_prompt":"x",
                "image":{"id":"00000000000000000000000000000000","media_type":"image/png","byte_length":1},
                "context_tokens":64,"max_new_tokens":1}),
            ),
        ];
        for (expected, request) in invalidations {
            let mut worker = prefix_worker(true);
            assert_eq!(reused(&mut worker, 3_000), 0);
            assert_eq!(reused(&mut worker, 3_000), 2048);
            let reply = post(&mut worker, request);
            assert!(reply["error"] == expected || reply.get(expected).is_some(), "{expected}: {reply}");
            assert!(!worker.backend.prefix_cache_retained(), "{expected}");
            assert_eq!(reused(&mut worker, 3_000), 0, "{expected}");
        }
    }

    #[test]
    fn prefix_cache_is_cleared_by_cancellation_and_native_errors() {
        for error in ["generation_cancelled", "native_generation_failed"] {
            let mut worker = prefix_worker(true);
            assert_eq!(reused(&mut worker, 3_000), 0);
            worker.backend.tokens = vec![Err(error)];
            let mut output = Vec::new();
            let _ = worker.process_line(&serde_json::to_vec(&cached_request(3_000, Some(SCOPE))).unwrap(), &mut output);
            assert!(String::from_utf8_lossy(&output).contains(error));
            assert!(!worker.backend.prefix_cache_retained(), "{error}");
            worker.backend.tokens = vec![Ok(3), Ok(1)];
            assert_eq!(reused(&mut worker, 3_000), 0, "{error}");
        }
    }

    #[test]
    fn prefix_cache_keeps_its_entry_across_tokenize_and_decode() {
        let mut worker = prefix_worker(true);
        assert_eq!(reused(&mut worker, 3_000), 0);
        assert!(post(&mut worker, json!({"operation":"tokenize","text":"fixture"})).get("error").is_none());
        assert!(post(&mut worker, json!({"operation":"decode","token_ids":[3]})).get("error").is_none());
        assert_eq!(reused(&mut worker, 3_000), 2048);
    }

    struct FailingOutput;

    impl Write for FailingOutput {
        fn write(
            &mut self,
            _: &[u8],
        ) -> io::Result<usize> {
            Err(io::Error::other("closed"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("closed"))
        }
    }

    #[test]
    fn prefix_cache_publishes_only_after_the_reply_is_written() {
        let mut worker = prefix_worker(true);
        assert_eq!(
            worker.process_line(&serde_json::to_vec(&cached_request(3_000, Some(SCOPE))).unwrap(), &mut FailingOutput),
            Err("protocol_write_failed")
        );
        assert!(!worker.backend.prefix_cache_retained());
        assert_eq!(reused(&mut worker, 3_000), 0);
        assert_eq!(reused(&mut worker, 3_000), 2048);
    }

    struct IdleInput {
        input: io::Cursor<Vec<u8>>,
        waits: Vec<bool>,
        expected_timeouts: Vec<Duration>,
        fail_wait: bool,
    }

    impl Read for IdleInput {
        fn read(
            &mut self,
            buffer: &mut [u8],
        ) -> io::Result<usize> {
            self.input.read(buffer)
        }
    }

    impl BufRead for IdleInput {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            self.input.fill_buf()
        }
        fn consume(
            &mut self,
            amount: usize,
        ) {
            self.input.consume(amount);
        }
    }

    impl RequestInput for IdleInput {
        fn wait_readable(
            &mut self,
            timeout: Duration,
        ) -> io::Result<bool> {
            if self.fail_wait {
                return Err(io::Error::other("failed wait"));
            }
            assert_eq!(timeout, self.expected_timeouts.remove(0));
            Ok(self.waits.remove(0))
        }
    }

    fn run_requests(
        worker: &mut GenerationWorker<FakeBackend>,
        lines: &[Vec<u8>],
        waits: Vec<bool>,
    ) -> Vec<Value> {
        let mut input = IdleInput {
            input: io::Cursor::new(lines.iter().flat_map(|line| line.iter().copied().chain(*b"\n")).collect()),
            expected_timeouts: vec![Duration::from_secs(600); waits.len()],
            waits,
            fail_wait: false,
        };
        let mut output = Vec::new();
        run_lines(worker, &mut input, &mut output).unwrap();
        assert!(input.waits.is_empty(), "waited only while state was retained");
        output
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .skip(1)
            .collect()
    }

    #[test]
    fn prefix_cache_expires_while_idle_and_on_oversized_lines() {
        let mut prefix = prefix_worker(true);
        prefix.budget = Budget::for_mib(1).unwrap();
        let wire_bytes = prefix.budget.wire_bytes;
        let request = serde_json::to_vec(&cached_request(3_000, Some(SCOPE))).unwrap();
        let replies = run_requests(&mut prefix, &[request.clone(), request.clone()], vec![true, true]);
        assert_eq!(replies[1]["reused_input_tokens"], 2048);
        let mut prefix = prefix_worker(true);
        prefix.budget = Budget::for_mib(1).unwrap();
        let replies = run_requests(&mut prefix, &[request.clone(), request.clone()], vec![false, true]);
        assert_eq!(replies[1]["reused_input_tokens"], 0, "idle expiry released the entry");
        let oversized = vec![b' '; wire_bytes + 1];
        let mut prefix = prefix_worker(true);
        prefix.budget = Budget::for_mib(1).unwrap();
        let replies = run_requests(&mut prefix, &[request.clone(), oversized, request.clone()], vec![true, true]);
        assert_eq!(replies[1]["error"], "request_too_large");
        assert_eq!(replies[2]["reused_input_tokens"], 0);
    }

    fn idle_input(
        waits: Vec<bool>,
        timeouts: Vec<Duration>,
    ) -> IdleInput {
        IdleInput {
            input: io::Cursor::new(Vec::new()),
            waits,
            expected_timeouts: timeouts,
            fail_wait: false,
        }
    }

    #[test]
    fn idle_maintenance_runs_without_a_retained_prefix_and_stops_at_eof() {
        let mut worker = worker();
        worker.backend.idle_interval = Some(Duration::from_millis(500));
        let mut input = idle_input(vec![false, false, true], vec![Duration::from_millis(500); 3]);
        let mut output = Vec::new();
        assert_eq!(run_lines(&mut worker, &mut input, &mut output), Ok(()));
        assert_eq!(worker.backend.idle_calls, 2);
        assert_eq!(worker.backend.generation_calls, 0);
        assert!(input.waits.is_empty());
        assert_eq!(output.split(|byte| *byte == b'\n').filter(|line| !line.is_empty()).count(), 1);
    }

    #[test]
    fn idle_maintenance_preserves_ready_input_and_prefix_state() {
        let mut worker = prefix_worker(true);
        assert_eq!(reused(&mut worker, 3_000), 0);
        worker.backend.idle_interval = Some(Duration::from_millis(500));
        let mut input = idle_input(vec![false, true], vec![Duration::from_millis(500); 2]);
        assert_eq!(wait_for_request(&mut worker.backend, &mut input), Ok(()));
        assert_eq!(worker.backend.idle_calls, 1);
        assert_eq!(reused(&mut worker, 3_000), 2048);
    }

    #[test]
    fn idle_maintenance_does_not_extend_prefix_expiry() {
        let mut worker = prefix_worker(true);
        assert_eq!(reused(&mut worker, 3_000), 0);
        worker.backend.idle_interval = Some(Duration::from_millis(500));
        let mut input = idle_input(
            vec![false, false, false, true],
            vec![
                Duration::from_millis(500),
                Duration::from_millis(500),
                Duration::from_millis(250),
                Duration::from_millis(500),
            ],
        );
        let start = Instant::now();
        let mut times =
            [0, 500, 500, 599_750, 599_750, 600_000].into_iter().map(|ms| start + Duration::from_millis(ms));
        assert_eq!(wait_for_request_with_clock(&mut worker.backend, &mut input, || times.next().unwrap()), Ok(()));
        assert!(!worker.backend.prefix_cache_retained());
        assert_eq!(worker.backend.idle_calls, 3);
    }

    #[test]
    fn idle_failures_exit_and_clear_retained_state() {
        for fail_wait in [false, true] {
            let mut worker = prefix_worker(true);
            assert_eq!(reused(&mut worker, 3_000), 0);
            worker.backend.idle_interval = Some(Duration::from_millis(500));
            worker.backend.fail_idle = !fail_wait;
            let mut input = idle_input(vec![false], vec![Duration::from_millis(500)]);
            input.fail_wait = fail_wait;
            assert_eq!(
                run_lines(&mut worker, &mut input, &mut Vec::new()),
                Err(if fail_wait {
                    "protocol_read_failed"
                } else {
                    "native_idle_maintenance_failed"
                })
            );
            assert!(!worker.backend.prefix_cache_retained());
        }
    }

    #[test]
    fn ready_input_cannot_reuse_a_prefix_after_idle_expiry() {
        for callback_crosses_deadline in [false, true] {
            let mut worker = prefix_worker(true);
            assert_eq!(reused(&mut worker, 3_000), 0);
            worker.backend.idle_interval = Some(Duration::from_millis(500));
            let mut input = idle_input(
                if callback_crosses_deadline {
                    vec![false, true]
                } else {
                    vec![true]
                },
                vec![
                    Duration::from_millis(500);
                    if callback_crosses_deadline {
                        2
                    } else {
                        1
                    }
                ],
            );
            let start = Instant::now();
            let mut times = if callback_crosses_deadline {
                vec![0, 500, 600_001]
            } else {
                vec![0, 600_001]
            }
            .into_iter()
            .map(|ms| start + Duration::from_millis(ms));
            assert_eq!(wait_for_request_with_clock(&mut worker.backend, &mut input, || times.next().unwrap()), Ok(()));
            assert!(!worker.backend.prefix_cache_retained());
        }
    }
}

#[cfg(all(test, feature = "metal"))]
mod native_sampling_recovery_tests {
    use super::*;

    #[test]
    #[ignore = "requires a private converted dense artifact with ordinary streaming"]
    fn accepted_tiny_temperature_keeps_the_loaded_worker_usable() {
        let path = std::path::PathBuf::from(
            std::env::var_os("STARK_DENSE_RECOVERY_MODEL_PATH").expect("set private converted model path"),
        );
        let backend = crate::native::NativeBackend::load(&path, None).unwrap();
        let mut worker = GenerationWorker::new(backend).unwrap();
        let mut request = json!({
            "operation":"generate", "input_token_ids":[2,3,4,5],
            "context_tokens":512,"max_new_tokens":4,"seed":42,"mtp":false,
            "sampling":{"mode":"stochastic","temperature":1e-38,"top_k":20,"top_p":0.9}
        });
        for stochastic in [true, false] {
            if !stochastic {
                request["sampling"] = json!({"mode":"greedy"});
            }
            let mut output = Vec::new();
            let result = worker.process_line(&serde_json::to_vec(&request).unwrap(), &mut output);
            assert!(result.is_ok(), "accepted temperature request or following request stopped the worker");
            let reply: Value = serde_json::from_slice(&output).unwrap();
            assert!(reply.get("error").is_none(), "native sampling request returned an error");
            assert!(
                reply["generated_token_ids"].as_array().is_some_and(|ids| !ids.is_empty()),
                "native sampling request did not emit output"
            );
        }
    }
}
