use std::{collections::HashSet, fs::File, path::Path, time::Instant};

use serde_json::Value;
use stark_engine::{
    backends::metal::Metal,
    engine::{
        Engine,
        language_model::{LanguageModel, stream::SamplingMethod},
        vision::{NativeVisionErrorKind, PreparedVisionImage, VisionLanguageModel},
    },
};

use crate::{
    cancel::CancelSignal,
    generation::{
        GeneratedTokens, GenerationBackend, GenerationOptions, ImageArtifactIdentity, PreparedImageMetadata, Sampling,
        TextCallback, collect_tokens_controlled, native_result_after_cleanup,
    },
    text_stop::StopFilter,
};

pub(crate) struct NativeBackend {
    model: NativeModel,
    vocab_size: usize,
    eos_token_ids: Vec<u32>,
    cancellation: Option<CancelSignal>,
    runtime_identity: stark_engine::engine::MetalDeviceIdentity,
}

enum NativeModel {
    Text(Box<LanguageModel<Metal>>),
    Vision(Box<VisionLanguageModel<Metal>>),
}

pub(crate) struct NativePreparedImage {
    image: PreparedVisionImage,
    tokens: Box<[u64]>,
    token_start: usize,
}

impl NativeBackend {
    pub(crate) fn load(
        path: &Path,
        cancellation: Option<CancelSignal>,
    ) -> Result<Self, &'static str> {
        Self::load_model(path, cancellation, false)
    }

    pub(crate) fn load_vision(
        path: &Path,
        cancellation: Option<CancelSignal>,
    ) -> Result<Self, &'static str> {
        Self::load_model(path, cancellation, true)
    }

    fn load_model(
        path: &Path,
        cancellation: Option<CancelSignal>,
        vision: bool,
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
                    .pointer("/decoder_config/vocab_size")
                    .or_else(|| config.pointer("/decoder_config/architecture/vocab_size"))
                    .and_then(Value::as_u64)
                    .filter(|size| *size > 0 && *size <= u32::MAX as u64)
                    .map(|size| size as usize)
                    .ok_or("invalid_model_config")
            })
            .transpose()?;
        let engine = Engine::<Metal>::new().map_err(|_| "metal_initialization_failed")?;
        let runtime_identity = engine.metal_device_identity();
        let model = if vision {
            NativeModel::Vision(Box::new(engine.load_vision_language_model(path).map_err(|_| "model_load_failed")?))
        } else {
            NativeModel::Text(Box::new(engine.load_language_model(path).map_err(|_| "model_load_failed")?))
        };
        let (tokenizer, stops) = match &model {
            NativeModel::Text(model) => (model.tokenizer(), &model.generation_config().stop_token_ids),
            NativeModel::Vision(model) => (model.tokenizer(), &model.generation_config().stop_token_ids),
        };
        let (vocab_size, codec_eos) = match &model {
            NativeModel::Vision(model) => (model.vocab_size() as usize, model.codec_eos_token()),
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
        Ok(Self {
            model,
            vocab_size,
            eos_token_ids,
            cancellation,
            runtime_identity,
        })
    }
}

impl GenerationBackend for NativeBackend {
    type PreparedImage = NativePreparedImage;

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn max_context_length(&self) -> Option<u32> {
        match &self.model {
            NativeModel::Text(model) => model.max_context_length(),
            NativeModel::Vision(model) => model.max_context_length(),
        }
    }
    fn request_context_limit(&self) -> u32 {
        match &self.model {
            NativeModel::Text(model) if model.is_qwen38() => 65_536,
            _ => crate::generation::MAX_CONTEXT_TOKENS,
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
        }
        .decode(ids, skip_special)
        .map_err(|_| "decoding_failed")
    }
    fn generate(
        &mut self,
        input: &[u32],
        options: &GenerationOptions,
        on_text: Option<&mut TextCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        self.generate_inner(input, None, options, on_text)
    }

    fn supports_images(&self) -> bool {
        matches!(self.model, NativeModel::Vision(_))
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
        let NativeModel::Vision(model) = &self.model else {
            return None;
        };
        let identity = model.artifact_identity();
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
            NativeModel::Text(_) => None,
        }
    }

    fn image_source_identity(&self) -> Option<(&str, Option<&str>)> {
        match &self.model {
            NativeModel::Vision(model) => Some(model.source_identity()),
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
        let NativeModel::Vision(model) = &mut self.model else {
            return Err("images_not_supported");
        };
        let tokens = input_token_ids.into_iter().map(u64::from).collect::<Vec<_>>().into_boxed_slice();
        let mut channel_error = false;
        let cancellation = &mut self.cancellation;
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
                image,
                tokens,
                token_start,
            },
            metadata,
        ))
    }

    fn generate_image(
        &mut self,
        image: NativePreparedImage,
        options: &GenerationOptions,
        on_text: Option<&mut TextCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        self.generate_inner(&[], Some(image), options, on_text)
    }
}

impl NativeBackend {
    fn generate_inner(
        &mut self,
        input: &[u32],
        image: Option<NativePreparedImage>,
        options: &GenerationOptions,
        mut on_text: Option<&mut TextCallback<'_>>,
    ) -> Result<GeneratedTokens, &'static str> {
        // Include native state allocation, stream setup and the first GPU wait.
        // The engine's iterator can encode one lookahead decode before returning
        // its first token, so this is a wall-clock prefill proxy, not GPU time.
        let started = Instant::now();
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
        };
        let mut state = match &mut self.model {
            NativeModel::Text(model) => model
                .create_empty_state(Some(options.context_tokens), options.seed)
                .map_err(|_| "native_generation_failed")?,
            NativeModel::Vision(model) => model
                .create_empty_state(Some(options.context_tokens), options.seed)
                .map_err(|_| "native_generation_failed")?,
        };
        let input: Vec<u64> = input.iter().map(|id| u64::from(*id)).collect();
        let input_tokens = image.as_ref().map_or(input.len(), |image| image.tokens.len());
        let mut stream_options = match &self.model {
            NativeModel::Text(model) => model.default_stream_options(),
            NativeModel::Vision(model) => model.default_stream_options(),
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
            (NativeModel::Text(model), None) => {
                model.stream(&input, &mut state, stream_options).map_err(|_| "native_generation_failed")?
            },
            (NativeModel::Vision(model), None) => {
                model.stream(&input, &mut state, stream_options).map_err(|_| "native_generation_failed")?
            },
            (NativeModel::Vision(model), Some(image)) => {
                let mut channel_error = false;
                model
                    .stream_prepared_image_controlled(
                        image.tokens,
                        image.image,
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
            (NativeModel::Text(_), Some(_)) => return Err("images_not_supported"),
        };
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
            &mut cancelled,
        );
        // collect_tokens has dropped and drained the stream, including its lookahead.
        drop(state);
        let fenced = matches!(&self.model, NativeModel::Vision(model) if model.is_fenced());
        let mut result = native_result_after_cleanup(result, fenced)?;
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
        Ok(result)
    }
}
