use std::{
    collections::HashSet,
    io::{BufRead, Read, Write},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::media::{ImageLease, ImageReference, MediaStore, opaque_id, valid_id};

pub(crate) const MAX_CONTEXT_TOKENS: u32 = 16_384;
pub(crate) const MAX_NEW_TOKENS: u32 = 4_096;
const MAX_REQUEST_BYTES: usize = 1_048_576;
const MAX_EXTRA_STOPS: usize = 256;

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
        #[serde(default)]
        mtp_drafts: u32,
        #[serde(default)]
        stop_token_ids: Vec<u32>,
        #[serde(default)]
        stream: bool,
        #[serde(default)]
        stop: Vec<String>,
        #[serde(default)]
        request_id: u64,
    },
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
        matches!(self.schema_version, 2 | 3)
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

struct PreparedRequest<P> {
    id: String,
    image: P,
    metadata: PreparedImageMetadata,
    context_tokens: u32,
    max_new_tokens: u32,
    _lease: ImageLease,
}

fn valid_image_metadata(metadata: &PreparedImageMetadata) -> bool {
    let [temporal, height, width] = metadata.grid;
    temporal == 1
        && height > 0
        && width > 0
        && height.is_multiple_of(2)
        && width.is_multiple_of(2)
        && height.checked_mul(width).is_some_and(|patches| patches <= 4096 && patches / 4 == metadata.image_tokens)
        && metadata.processed_dimensions[0].checked_div(16) == Some(width)
        && metadata.processed_dimensions[1].checked_div(16) == Some(height)
        && metadata.processed_dimensions.iter().all(|dimension| dimension.is_multiple_of(16))
        && metadata.input_tokens >= metadata.image_tokens as usize + 2
        && metadata.input_tokens <= MAX_CONTEXT_TOKENS as usize
        && metadata.image_tokens > 0
        && metadata.image_tokens <= 1024
}

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

#[cfg(test)]
pub(crate) fn collect_tokens_observed(
    stream: impl Iterator<Item = Result<u64, &'static str>>,
    input_tokens: usize,
    options: &GenerationOptions,
    vocab_size: usize,
    started: Instant,
    on_token: &mut dyn FnMut(u32) -> Result<(), &'static str>,
) -> Result<GeneratedTokens, &'static str> {
    collect_tokens_controlled(
        stream,
        input_tokens,
        options,
        vocab_size,
        started,
        &mut |token| on_token(token).map(|()| true),
        &mut || Ok(false),
    )
}

/// Own the stream so its Drop drains native work before the result is observable.
pub(crate) fn collect_tokens_controlled(
    stream: impl Iterator<Item = Result<u64, &'static str>>,
    input_tokens: usize,
    options: &GenerationOptions,
    vocab_size: usize,
    started: Instant,
    on_token: &mut dyn FnMut(u32) -> Result<bool, &'static str>,
    cancelled: &mut dyn FnMut() -> Result<bool, &'static str>,
) -> Result<GeneratedTokens, &'static str> {
    collect_tokens_with_clock_and_callback(
        stream,
        input_tokens,
        options,
        vocab_size,
        || started.elapsed(),
        on_token,
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

    fn vocab_size(&self) -> usize;
    fn max_context_length(&self) -> Option<u32>;
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
    ) -> Result<GeneratedTokens, &'static str>;
    fn supports_images(&self) -> bool {
        false
    }
    fn supports_mtp(&self) -> bool {
        false
    }
    fn mtp_metadata(&self) -> Option<Value> {
        None
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
    ) -> Result<GeneratedTokens, &'static str> {
        Err("images_not_supported")
    }
}

pub(crate) type TextCallback<'a> = dyn FnMut(&str) -> Result<(), &'static str> + 'a;

pub(crate) struct GenerationWorker<B: GenerationBackend> {
    backend: B,
    media: Option<MediaStore>,
    prepared: Option<PreparedRequest<B::PreparedImage>>,
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
        Self::new_with_media(backend, None)
    }

    pub(crate) fn new_with_media(
        backend: B,
        media: Option<MediaStore>,
    ) -> Result<Self, &'static str> {
        if backend.vocab_size() == 0
            || backend.vocab_size() > u32::MAX as usize
            || !valid_ids(backend.eos_token_ids(), backend.vocab_size())
        {
            return Err("invalid_model_config");
        }
        Ok(Self {
            backend,
            media,
            prepared: None,
        })
    }

    fn validate_options(
        &self,
        options: &GenerationOptions,
    ) -> Result<(), &'static str> {
        if options.context_tokens == 0
            || options.context_tokens > MAX_CONTEXT_TOKENS
            || options.max_new_tokens == 0
            || options.max_new_tokens > MAX_NEW_TOKENS
            || options.mtp_drafts > 3
            || options.extra_stop_token_ids.len() > MAX_EXTRA_STOPS
            || options.stop.len() > 4
            || options.stop.iter().any(|s| s.is_empty() || s.chars().count() > 1024)
            || !valid_ids(&options.extra_stop_token_ids, self.backend.vocab_size())
            || options.extra_stop_token_ids.iter().collect::<HashSet<_>>().len() != options.extra_stop_token_ids.len()
        {
            return Err("invalid_generation_request");
        }
        if options.mtp_drafts > 0 && !self.backend.supports_mtp() {
            return Err("mtp_not_supported");
        }
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
        let mut emitted = String::new();
        let generated = {
            let mut on_text = |text: &str| {
                if !text.is_empty() {
                    write_reply(output, &json!({"event":"delta","text":text}))?;
                    emitted.push_str(text);
                }
                Ok(())
            };
            let on_text = if stream {
                Some(&mut on_text as &mut TextCallback<'_>)
            } else {
                None
            };
            match image {
                Some(image) => self.backend.generate_image(image, options, on_text)?,
                None => self.backend.generate(input, options, on_text)?,
            }
        };
        if !valid_ids(&generated.generated_token_ids, self.backend.vocab_size())
            || generated.sampled_tokens > options.max_new_tokens as usize
            || input_tokens + generated.sampled_tokens > options.context_tokens as usize
            || generated.sampled_tokens
                != generated.generated_token_ids.len() + usize::from(generated.stop_token_id.is_some())
        {
            return Err("invalid_generation_output");
        }
        let text = match &generated.text {
            Some(text) => text.clone(),
            None => self.backend.decode(&generated.generated_token_ids, false)?,
        };
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
            "input_tokens_per_second":tokens_per_second(input_tokens, generated.prefill_seconds),
            "output_tokens_per_second":tokens_per_second(generated.sampled_tokens.saturating_sub(1), generated.decode_seconds),
            "timing_method":"first-token-wall-clock-v1"});
        if let Some(metadata) = self.backend.mtp_metadata() {
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
                let ids = self.tokenize(&text, add_special_tokens)?;
                Ok(json!({"token_ids":ids,"token_count":ids.len()}))
            },
            Request::Decode {
                token_ids,
                skip_special_tokens,
            } => {
                if !valid_ids(&token_ids, self.backend.vocab_size()) {
                    return Err("invalid_decoding_request");
                }
                let text = self.backend.decode(&token_ids, skip_special_tokens)?;
                Ok(json!({"text":text,"token_count":token_ids.len()}))
            },
            Request::Generate {
                rendered_prompt,
                input_token_ids,
                context_tokens,
                max_new_tokens,
                seed,
                sampling,
                mtp_drafts,
                stop_token_ids,
                stream,
                stop,
                request_id,
            } => {
                let options = GenerationOptions {
                    context_tokens,
                    max_new_tokens,
                    seed,
                    sampling,
                    mtp_drafts,
                    eos_token_ids: self.backend.eos_token_ids().to_vec(),
                    extra_stop_token_ids: stop_token_ids,
                    stop,
                    request_id,
                };
                self.validate_options(&options)?;
                let input = match (rendered_prompt, input_token_ids) {
                    (Some(prompt), None) => self.tokenize(&prompt, false)?,
                    (None, Some(ids)) => ids,
                    _ => return Err("invalid_generation_request"),
                };
                if input.is_empty() || !valid_ids(&input, self.backend.vocab_size()) {
                    return Err("invalid_generation_request");
                }
                if input.len() >= context_tokens as usize {
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
                if !valid_image_metadata(&metadata) {
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
                    max_new_tokens,
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
                if prepared.context_tokens != context_tokens || prepared.max_new_tokens != max_new_tokens {
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
                };
                self.validate_options(&options)?;
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
        let request = serde_json::from_slice::<Request>(bytes).map_err(|_| "invalid_request");
        if !matches!(&request, Ok(Request::GenerateImage { .. } | Request::DiscardImage { .. })) {
            self.prepared = None;
        }
        let result = request.and_then(|request| self.handle(request, output));
        if result.is_err() {
            self.prepared = None;
        }
        match result {
            Ok(value) => {
                let result = write_reply(output, &value);
                if result.is_err() {
                    self.prepared = None;
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

pub(crate) fn run_lines<B: GenerationBackend>(
    worker: &mut GenerationWorker<B>,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<(), &'static str> {
    let mut readiness = json!({"ready":true,"engine":"stark","version":env!("CARGO_PKG_VERSION"),
        "mode":"text-generation-v1","operations":["tokenize","decode","generate"],
        "state_policy":"fresh-per-request","vocab_size":worker.backend.vocab_size(),
        "max_context_tokens":MAX_CONTEXT_TOKENS.min(worker.backend.max_context_length().unwrap_or(MAX_CONTEXT_TOKENS)),
        "max_new_tokens":MAX_NEW_TOKENS,"eos_token_ids":worker.backend.eos_token_ids(),
        "sampling_modes":["greedy","stochastic"],"streaming":"text-delta-v1",
        "stop_sequences":true,"cancellation":"pipe-v1"});
    if worker.backend.supports_mtp() {
        readiness["mtp"] = json!({"supported":true,"max_drafts":3,
            "default_drafts":0,"sampling":"exact-rejection-v1"});
    }
    if worker.media.is_some() && worker.backend.supports_images() {
        let identity = worker
            .backend
            .image_artifact_identity()
            .filter(ImageArtifactIdentity::valid)
            .ok_or("invalid_model_config")?;
        let runtime_identity = worker.backend.runtime_identity().ok_or("invalid_model_config")?;
        let (family_profile, family_version) = worker
            .backend
            .image_family_profile()
            .filter(|(profile, version)| {
                matches!((*profile, *version), ("qwen35-dense-bf16-merged-v1", 1) | ("qwen35-dense-bf16-merged-v2", 2))
            })
            .ok_or("invalid_model_config")?;
        let (model_source, model_revision) = worker.backend.image_source_identity().ok_or("invalid_model_config")?;
        readiness["image_input"] = json!({
            "version":"mediaref-v1","transport":"owner-private-file-v1","qualified":false,"qualification":"unverified",
            "operations":["prepare_image","generate_image","discard_image"],
            "formats":["image/jpeg","image/png"],"max_images":1,
            "max_compressed_bytes":crate::media::MAX_COMPRESSED_BYTES,"max_source_pixels":32_000_000,
            "max_source_dimension":16_384,
            "max_processed_pixels":1_048_576,"max_patches":4096,"max_image_tokens":1024,
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
        let mut bytes = Vec::new();
        let count = match Read::by_ref(input).take(MAX_REQUEST_BYTES as u64 + 1).read_until(b'\n', &mut bytes) {
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
        if bytes.len() > MAX_REQUEST_BYTES {
            worker.prepared = None;
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

    struct FakeBackend {
        tokens: Vec<Result<u64, &'static str>>,
        eos: Vec<u32>,
        generation_calls: usize,
        inputs: Vec<Vec<u32>>,
        seeds: Vec<u64>,
        sampling: Vec<Sampling>,
        drained: Arc<AtomicBool>,
        invalid_tokenization: bool,
        images: bool,
        mtp: bool,
        mtp_requests: Vec<u32>,
        image_schema: u32,
        family_version: u32,
        family_profile: &'static str,
        image_preparations: usize,
        image_generations: usize,
        prepared_drops: Arc<AtomicUsize>,
        cancel_image: bool,
        cancel_after_prepare: bool,
        fail_image_prepare: bool,
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

        fn vocab_size(&self) -> usize {
            8
        }
        fn max_context_length(&self) -> Option<u32> {
            Some(1024)
        }
        fn eos_token_ids(&self) -> &[u32] {
            &self.eos
        }
        fn supports_mtp(&self) -> bool {
            self.mtp
        }
        fn mtp_metadata(&self) -> Option<Value> {
            self.mtp.then(|| {
                let drafts = self.mtp_requests.last().copied().unwrap_or(0);
                json!({"enabled":drafts > 0,"drafts":drafts,"sampling":"exact-rejection-v1"})
            })
        }
        fn tokenize(
            &mut self,
            _: &str,
            special: bool,
        ) -> Result<Vec<u32>, &'static str> {
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
        ) -> Result<GeneratedTokens, &'static str> {
            self.generation_calls += 1;
            self.inputs.push(input.to_vec());
            self.seeds.push(options.seed);
            self.sampling.push(options.sampling.clone());
            self.mtp_requests.push(options.mtp_drafts);
            self.drained.store(false, Ordering::SeqCst);
            let mut on_token = |token| {
                if let Some(callback) = on_text.as_deref_mut() {
                    callback(match token {
                        3 => "ré",
                        4 => "sultat\n",
                        7 => "<tool_call>",
                        _ => "",
                    })?;
                }
                Ok(())
            };
            collect_tokens_observed(
                TestStream {
                    tokens: self.tokens.clone().into(),
                    drained: self.drained.clone(),
                },
                input.len(),
                options,
                self.vocab_size(),
                Instant::now(),
                &mut on_token,
            )
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

        fn image_source_identity(&self) -> Option<(&str, Option<&str>)> {
            self.images.then_some(if self.image_schema == 3 {
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
                PreparedImageMetadata {
                    input_tokens: 7,
                    image_tokens: 4,
                    grid: [1, 4, 4],
                    processed_dimensions: [64, 64],
                },
            ))
        }

        fn generate_image(
            &mut self,
            _image: FakePreparedImage,
            options: &GenerationOptions,
            on_text: Option<&mut TextCallback<'_>>,
        ) -> Result<GeneratedTokens, &'static str> {
            self.image_generations += 1;
            self.generate(&[2; 7], options, on_text)
        }
    }
    fn worker() -> GenerationWorker<FakeBackend> {
        GenerationWorker::new(FakeBackend {
            tokens: vec![Ok(3), Ok(4), Ok(1), Ok(7)],
            eos: vec![1],
            generation_calls: 0,
            inputs: Vec::new(),
            seeds: Vec::new(),
            sampling: Vec::new(),
            drained: Arc::new(AtomicBool::new(true)),
            invalid_tokenization: false,
            images: false,
            mtp: false,
            mtp_requests: Vec::new(),
            image_schema: 2,
            family_version: 1,
            family_profile: "qwen35-dense-bf16-merged-v1",
            image_preparations: 0,
            image_generations: 0,
            prepared_drops: Arc::new(AtomicUsize::new(0)),
            cancel_image: false,
            cancel_after_prepare: false,
            fail_image_prepare: false,
        })
        .unwrap()
    }
    fn standard() -> Value {
        json!({"operation":"generate","input_token_ids":[2],"context_tokens":16,"max_new_tokens":8})
    }

    #[test]
    fn mtp_requests_are_opt_in_bounded_and_require_model_support() {
        let mut worker = worker();
        let mut request = standard();
        request["mtp_drafts"] = json!(1);
        assert_eq!(post(&mut worker, request.clone())["error"], "mtp_not_supported");
        assert_eq!(worker.backend.generation_calls, 0);
        worker.backend.mtp = true;
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
    }
    fn post(
        worker: &mut GenerationWorker<FakeBackend>,
        request: Value,
    ) -> Value {
        let mut output = Vec::new();
        worker.process_line(&serde_json::to_vec(&request).unwrap(), &mut output).unwrap();
        serde_json::from_slice(&output).unwrap()
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
    fn prepared_handles_are_worker_scoped_and_bound_to_the_reserved_budget() {
        let root = crate::media::tests::PrivateRoot::new();
        let mut first = image_worker(&root);
        let mut second = image_worker(&root);
        let prepared = post(&mut first, image_prepare(&root));
        post(&mut second, image_prepare(&root));
        assert_eq!(post(&mut second, image_generate(&prepared))["error"], "invalid_prepared_image");
        assert!(second.prepared.is_none());
        let mut changed = image_generate(&prepared);
        changed["max_new_tokens"] = json!(7);
        assert_eq!(post(&mut first, changed)["error"], "invalid_generation_request");
        assert!(first.prepared.is_none());
        assert_eq!(post(&mut first, image_generate(&prepared))["error"], "invalid_prepared_image");
        assert_eq!(first.backend.image_generations, 0);
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
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0]["event"], "delta");
        assert_eq!(frames[1]["event"], "delta");
        assert_eq!(frames[2]["input_tokens"], 7);
        assert_eq!(frames[2]["finish_reason"], "eos");
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
    fn image_readiness_is_separate_and_oversized_frames_and_eof_clear_preparation() {
        let root = crate::media::tests::PrivateRoot::new();
        let mut worker = image_worker(&root);
        post(&mut worker, image_prepare(&root));
        let mut output = Vec::new();
        run_lines(&mut worker, &mut io::Cursor::new(vec![b'x'; MAX_REQUEST_BYTES + 1]), &mut output).unwrap();
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
                if value.get("event").is_some() {
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
        assert_eq!(output.frames.len(), 3);
        assert_eq!(output.frames[0], json!({"event":"delta","text":"ré"}));
        assert_eq!(output.frames[1], json!({"event":"delta","text":"sultat\n"}));
        assert_eq!(output.frames[2]["text"], "résultat\n");
        assert_eq!(output.frames[2]["generated_token_ids"], json!([3, 4]));
        assert_eq!(output.frames[2]["finish_reason"], "eos");
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
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], json!({"event":"delta","text":"ré"}));
        assert_eq!(frames[1]["finish_reason"], "stop_token");
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
        for (context, maximum, reason) in [(16, 1, "length"), (2, 8, "context"), (2, 1, "length")] {
            let mut request = standard();
            request["context_tokens"] = json!(context);
            request["max_new_tokens"] = json!(maximum);
            let value = post(&mut worker, request);
            assert_eq!(value["generated_token_ids"], json!([3]));
            assert_eq!(value["sampled_tokens"], 1);
            assert_eq!(value["finish_reason"], reason);
            assert!(worker.backend.drained.load(Ordering::SeqCst));
        }
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
    fn bad_request_does_not_dispatch_native_generation_or_poison_following_request() {
        let mut worker = worker();
        let cases = [
            json!({"context_tokens":0}),
            json!({"context_tokens":16385}),
            json!({"context_tokens":1025}),
            json!({"max_new_tokens":0}),
            json!({"max_new_tokens":4097}),
            json!({"input_token_ids":[]}),
            json!({"input_token_ids":[8]}),
            json!({"input_token_ids":[2,3],"context_tokens":2}),
            json!({"stop_token_ids":[8]}),
            json!({"stop_token_ids":[1,1]}),
            json!({"rendered_prompt":"both sources"}),
            json!({"input_token_ids":null}),
            json!({"sampling":{"mode":"stochastic","temperature":0.0}}),
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
        let mut bytes = b"{\n".to_vec();
        bytes.extend(vec![b'x'; MAX_REQUEST_BYTES + 30]);
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
}
