//! Exact-artifact native Qwen3.5 image preparation and request execution.
//!
//! Compressed images become opaque, model-bound CPU patch storage. Projected
//! embeddings remain private and can only be produced by the admitted tower.

use std::{
    fmt,
    fs::File,
    io::{self, Read, Seek},
    path::Path,
    sync::Arc,
};

use half::bf16;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, InvalidInput,
    },
    config::{
        decoder::DecoderConfig,
        model::{generation::GenerationConfig, language_model::LanguageModelConfig},
        token_codec::AnyTokenCodecConfig,
    },
    data_type::DataType,
    engine::{
        Engine,
        language_model::{
            EngineLoadLanguageModelError, LanguageModel, SpeculatorPolicy,
            multimodal::{ImageGrid, ImageTokenIds, MultimodalPrefill, ProjectedImage},
            state::{LanguageModelCreateEmptyStateError, LanguageModelState},
            stream::{LanguageModelStream, LanguageModelStreamError, LanguageModelStreamOptions},
        },
    },
    parameters::{NumericalProfile, ParameterLoader, ParameterLoaderError},
};

#[cfg(test)]
mod logit_tests;
#[cfg(test)]
mod matched_input_tests;
#[cfg(test)]
mod parity_tests;
pub(crate) mod preprocess;
#[cfg(test)]
mod reference_tests;
#[cfg(test)]
mod tests;
pub(crate) mod tower;

use preprocess::{PreparedImage, PreprocessLimits, RgbImage, decode_image, prepare_image as prepare_patches};
use tower::{VisionTower, VisionTowerConfig, VisionTowerError};

// Exact deterministic Stage 1 artifact. Admission is deliberately bound to this
// experimental tuple; metadata or a model name cannot enable another artifact.
const ARTIFACT_FILES: [(&str, &str); 5] = [
    ("config.json", "3f0cecb7cb3633bc271eabf69b06ac9f74b377b9ec17b441f5ceac331a71f42a"),
    ("model.safetensors", "7c5960e7225b3b36844fa397a85c5e20d281d732e382475062f552c0737c79d4"),
    ("tokenizer.json", "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"),
    ("preprocessor_config.json", "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"),
    ("chat_template.jinja", "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"),
];
const ARTIFACT_SCHEMA_VERSION: u32 = 2;

/// Immutable identity of the complete artifact admitted by this loaded model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisionArtifactIdentity {
    pub schema_version: u32,
    pub config_sha256: &'static str,
    pub artifact_sha256: &'static str,
    pub tokenizer_sha256: &'static str,
    pub processor_sha256: &'static str,
    pub template_sha256: &'static str,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactConfig {
    #[serde(rename = "type")]
    kind: String,
    decoder_config: DecoderConfig,
    token_codec_config: AnyTokenCodecConfig,
    generation_config: GenerationConfig,
    artifact_contract: serde_json::Value,
}

#[derive(Debug, Error)]
enum NativeVisionFailure<B: Backend> {
    #[error("Native vision request cancelled")]
    Cancelled,
    #[error(transparent)]
    InvalidInput(#[from] InvalidInput),
    #[error("Native vision artifact I/O failed")]
    IO(#[from] io::Error),
    #[error("Native vision configuration failed")]
    Configuration(#[from] serde_json::Error),
    #[error("Native vision language model construction failed")]
    LanguageModel(#[from] EngineLoadLanguageModelError<B>),
    #[error("Native vision parameter loading failed")]
    Parameters(#[from] ParameterLoaderError<B>),
    #[error("Native vision tower failed")]
    Tower(#[from] VisionTowerError<B>),
    #[error("Native vision backend failed")]
    Backend(#[source] B::Error),
    #[error("Native vision stream failed")]
    Stream(#[from] LanguageModelStreamError<B>),
    #[error("Native vision state creation failed")]
    State(#[from] LanguageModelCreateEmptyStateError<B>),
}

/// Content-free error category. Instance fencing is reported by the loaded
/// model because a validation error after native execution can also fence it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeVisionErrorKind {
    Cancelled,
    InvalidInput,
    Artifact,
    Backend,
    Stream,
    State,
}

/// A public error that keeps backend internals and private tower types hidden.
pub struct NativeVisionError<B: Backend> {
    failure: NativeVisionFailure<B>,
}

impl<B: Backend> NativeVisionError<B> {
    pub fn kind(&self) -> NativeVisionErrorKind {
        match &self.failure {
            NativeVisionFailure::Cancelled => NativeVisionErrorKind::Cancelled,
            NativeVisionFailure::InvalidInput(_) => NativeVisionErrorKind::InvalidInput,
            NativeVisionFailure::IO(_)
            | NativeVisionFailure::Configuration(_)
            | NativeVisionFailure::LanguageModel(_)
            | NativeVisionFailure::Parameters(_) => NativeVisionErrorKind::Artifact,
            NativeVisionFailure::Tower(_) | NativeVisionFailure::Backend(_) => NativeVisionErrorKind::Backend,
            NativeVisionFailure::Stream(_) => NativeVisionErrorKind::Stream,
            NativeVisionFailure::State(_) => NativeVisionErrorKind::State,
        }
    }

    fn backend(error: B::Error) -> Self {
        Self {
            failure: NativeVisionFailure::Backend(error),
        }
    }

    fn cancelled() -> Self {
        Self {
            failure: NativeVisionFailure::Cancelled,
        }
    }
}

impl<B: Backend> fmt::Display for NativeVisionError<B> {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        fmt::Display::fmt(&self.failure, formatter)
    }
}

impl<B: Backend> fmt::Debug for NativeVisionError<B> {
    fn fmt(
        &self,
        formatter: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        // Never include source exceptions, paths, request data or backend dumps.
        formatter.debug_struct("NativeVisionError").field("kind", &self.kind()).finish()
    }
}

impl<B: Backend> std::error::Error for NativeVisionError<B> {}

macro_rules! native_vision_error_from {
    ($variant:ident, $source:ty) => {
        impl<B: Backend> From<$source> for NativeVisionError<B> {
            fn from(error: $source) -> Self {
                Self {
                    failure: NativeVisionFailure::$variant(error),
                }
            }
        }
    };
}

native_vision_error_from!(InvalidInput, InvalidInput);
native_vision_error_from!(IO, io::Error);
native_vision_error_from!(Configuration, serde_json::Error);
native_vision_error_from!(LanguageModel, EngineLoadLanguageModelError<B>);
native_vision_error_from!(Parameters, ParameterLoaderError<B>);
native_vision_error_from!(Tower, VisionTowerError<B>);
native_vision_error_from!(Stream, LanguageModelStreamError<B>);
native_vision_error_from!(State, LanguageModelCreateEmptyStateError<B>);

/// Request-owned CPU patches, produced once by the admitted model's processor.
/// This object has no serialization, cloning, Debug, or embedding accessors.
pub struct PreparedVisionImage {
    model_identity: Arc<()>,
    prepared: PreparedImage,
    image_token_count: u32,
}

impl PreparedVisionImage {
    fn from_prepared(
        model_identity: Arc<()>,
        prepared: PreparedImage,
    ) -> Result<Self, InvalidInput> {
        if prepared.grid[0] != 1 || !prepared.grid[1].is_multiple_of(2) || !prepared.grid[2].is_multiple_of(2) {
            return Err(InvalidInput("native image requires one spatially merged static grid"));
        }
        let image_token_count = prepared.grid[1]
            .checked_mul(prepared.grid[2])
            .filter(|count| *count <= 4096)
            .and_then(|count| count.checked_div(4))
            .filter(|count| *count != 0 && *count <= 1024)
            .ok_or(InvalidInput("native image token count exceeds processor bounds"))?;
        Ok(Self {
            model_identity,
            prepared,
            image_token_count,
        })
    }

    pub fn grid(&self) -> [u32; 3] {
        self.prepared.grid
    }

    pub fn image_token_count(&self) -> u32 {
        self.image_token_count
    }

    /// Processed width and height, in pixels.
    pub fn processed_dimensions(&self) -> [u32; 2] {
        [self.prepared.resized_width, self.prepared.resized_height]
    }
}

fn expand_image_placeholder(
    tokens: Box<[u64]>,
    image_token_count: u32,
    context_capacity: u32,
) -> Result<(Box<[u64]>, usize), InvalidInput> {
    if !(1..=1024).contains(&image_token_count) {
        return Err(InvalidInput("native image token count exceeds processor bounds"));
    }
    let start = tokens
        .iter()
        .position(|&token| token == 248056)
        .ok_or(InvalidInput("native image prompt requires one image placeholder"))?;
    if tokens[start + 1..].contains(&248056) {
        return Err(InvalidInput("native image prompt requires exactly one unexpanded placeholder"));
    }
    let count = image_token_count as usize;
    let expanded_len =
        tokens.len().checked_add(count - 1).ok_or(InvalidInput("native image prompt token count overflow"))?;
    if expanded_len >= context_capacity as usize {
        return Err(InvalidInput("native image prompt exceeds requested context"));
    }
    let mut expanded = Vec::new();
    expanded.try_reserve_exact(expanded_len).map_err(|_| InvalidInput("native image prompt allocation failed"))?;
    expanded.extend_from_slice(&tokens[..start]);
    expanded.extend(std::iter::repeat_n(248056, count));
    expanded.extend_from_slice(&tokens[start + 1..]);
    Ok((expanded.into_boxed_slice(), start))
}

pub struct VisionLanguageModel<B: Backend> {
    engine: Arc<Engine<B>>,
    language_model: LanguageModel<B>,
    tower: VisionTower<B>,
    limits: PreprocessLimits,
    failed: bool,
}

fn file_matches(
    file: &mut File,
    expected: &str,
) -> io::Result<bool> {
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65_536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()) == expected)
}

fn official_tower_config() -> VisionTowerConfig {
    VisionTowerConfig {
        depth: 12,
        hidden_size: 768,
        intermediate_size: 3072,
        num_heads: 12,
        num_position_embeddings: 2304,
        out_hidden_size: 1024,
        patch_size: 16,
        spatial_merge_size: 2,
        temporal_patch_size: 2,
        in_channels: 3,
        layer_norm_eps: 1e-6,
    }
}

impl<B: Backend> Engine<B> {
    /// Load only the pinned BF16 image artifact. This explicit loader does not
    /// broaden the text resolver or establish service/hardware qualification.
    pub fn load_vision_language_model(
        self: &Arc<Self>,
        model_path: &Path,
    ) -> Result<VisionLanguageModel<B>, NativeVisionError<B>> {
        if self.capture_manager.is_some() || model_path.join("speculator").exists() {
            return Err(InvalidInput("native vision requires capture-free non-speculative loading").into());
        }
        // Verify before tensor allocation. The weights stay on this opened file
        // descriptor throughout loading, so path replacement cannot switch it.
        let mut verified = std::collections::HashMap::new();
        for (name, expected) in ARTIFACT_FILES {
            let mut file = File::open(model_path.join(name))?;
            if !file.metadata()?.is_file() || !file_matches(&mut file, expected)? {
                return Err(InvalidInput("native vision artifact identity mismatch").into());
            }
            file.rewind()?;
            verified.insert(name, file);
        }
        let config: ArtifactConfig = serde_json::from_reader(
            verified.get_mut("config.json").ok_or(InvalidInput("missing native vision configuration"))?,
        )?;
        if config.kind != "Qwen35VisionLanguageModelConfig"
            || config.artifact_contract["source"]["repo"] != "Qwen/Qwen3.5-0.8B"
            || config.artifact_contract["source"]["revision"] != "2fc06364715b967f1860aea9cf38778875588b17"
            || config.artifact_contract["schema_version"] != ARTIFACT_SCHEMA_VERSION
            || config.artifact_contract["transport"] != "mediaref-v1"
            || config.artifact_contract["runtime_supported"] != true
        {
            return Err(InvalidInput("unsupported native vision artifact contract").into());
        }
        let language_config: LanguageModelConfig = serde_json::from_value(serde_json::json!({
            "type": "LanguageModelConfig", "decoder_config": config.decoder_config,
            "token_codec_config": config.token_codec_config, "generation_config": config.generation_config,
        }))?;
        let mut tokenizer_bytes = Vec::new();
        verified
            .get_mut("tokenizer.json")
            .ok_or(InvalidInput("missing native vision tokenizer"))?
            .read_to_end(&mut tokenizer_bytes)?;
        let tokenizer = Arc::new(
            tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).map_err(EngineLoadLanguageModelError::Tokenizer)?,
        );
        let weights = verified.get("model.safetensors").ok_or(InvalidInput("missing native vision weights"))?;
        let loader = ParameterLoader::new(weights, &*self.context)?
            .with_numerical_profile(NumericalProfile::Qwen35ReferenceBF16);
        let tower = VisionTower::new(&*self.context, &official_tower_config(), &loader.tree().subtree("vision"))?;
        let language_model = self.language_model_from_config(
            model_path,
            language_config,
            &loader,
            Some(tokenizer),
            SpeculatorPolicy::Disabled,
        )?;
        loader.tree().assert_all_tensors_validated()?;
        drop(loader);
        // Recheck the same selected objects before publishing loaded buffers.
        // Ordinary concurrent in-place artifact changes cannot silently acquire
        // the source identity established by the initial admission pass.
        for (name, expected) in ARTIFACT_FILES {
            let file = verified.get_mut(name).ok_or(InvalidInput("missing native vision artifact component"))?;
            file.rewind()?;
            if !file_matches(file, expected)? {
                return Err(InvalidInput("native vision artifact changed while loading").into());
            }
        }
        for (name, id) in [
            ("<|vision_start|>", 248053),
            ("<|vision_end|>", 248054),
            ("<|image_pad|>", 248056),
            ("<|video_pad|>", 248057),
        ] {
            if language_model.tokenizer().token_to_id(name) != Some(id) {
                return Err(InvalidInput("native vision tokenizer identity mismatch").into());
            }
        }
        Ok(VisionLanguageModel {
            engine: self.clone(),
            language_model,
            tower,
            limits: PreprocessLimits::default(),
            failed: false,
        })
    }
}

impl<B: Backend> VisionLanguageModel<B> {
    pub fn artifact_identity(&self) -> VisionArtifactIdentity {
        VisionArtifactIdentity {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            config_sha256: ARTIFACT_FILES[0].1,
            artifact_sha256: ARTIFACT_FILES[1].1,
            tokenizer_sha256: ARTIFACT_FILES[2].1,
            processor_sha256: ARTIFACT_FILES[3].1,
            template_sha256: ARTIFACT_FILES[4].1,
        }
    }

    pub fn tokenizer(&self) -> &Arc<tokenizers::Tokenizer> {
        self.language_model.tokenizer()
    }

    pub fn generation_config(&self) -> &GenerationConfig {
        self.language_model.generation_config()
    }

    pub fn max_context_length(&self) -> Option<u32> {
        self.language_model.max_context_length()
    }

    pub fn recommended_context_length(&self) -> Option<u32> {
        self.language_model.recommended_context_length()
    }

    pub fn is_fenced(&self) -> bool {
        self.failed
    }

    pub fn default_stream_options(&self) -> LanguageModelStreamOptions {
        self.language_model.default_stream_options()
    }

    pub fn create_empty_state(
        &mut self,
        context_capacity: Option<u32>,
        seed: u64,
    ) -> Result<LanguageModelState<B>, NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("native vision model is fenced").into());
        }
        match self.language_model.create_empty_state(context_capacity, seed) {
            Ok(state) => Ok(state),
            Err(error) => {
                if matches!(&error, LanguageModelCreateEmptyStateError::Backend(_)) {
                    self.failed = true;
                }
                Err(error.into())
            },
        }
    }

    /// Decode and preprocess once, before prompt expansion or device work.
    /// The returned object is request-owned and bound to this loaded instance.
    pub fn prepare_image(
        &mut self,
        encoded_image: &[u8],
    ) -> Result<PreparedVisionImage, NativeVisionError<B>> {
        self.prepare_image_controlled(encoded_image, || false)
    }

    /// Observe cancellation between CPU stages. The callback is not retained.
    pub fn prepare_image_controlled<F: FnMut() -> bool>(
        &mut self,
        encoded_image: &[u8],
        mut should_cancel: F,
    ) -> Result<PreparedVisionImage, NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("native vision model is fenced").into());
        }
        if should_cancel() {
            return Err(NativeVisionError::cancelled());
        }
        let decoded = decode_image(encoded_image, &self.limits);
        if should_cancel() {
            return Err(NativeVisionError::cancelled());
        }
        self.prepare_rgb_image_controlled(&decoded?, &mut should_cancel)
    }

    fn prepare_rgb_image_controlled<F: FnMut() -> bool>(
        &self,
        image: &RgbImage,
        should_cancel: &mut F,
    ) -> Result<PreparedVisionImage, NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("native vision model is fenced").into());
        }
        if should_cancel() {
            return Err(NativeVisionError::cancelled());
        }
        let prepared = prepare_patches(image, &self.limits);
        if should_cancel() {
            return Err(NativeVisionError::cancelled());
        }
        Ok(PreparedVisionImage::from_prepared(self.language_model.model_identity.clone(), prepared?)?)
    }

    /// Expand exactly one template image placeholder, then validate its complete
    /// token/grid/span/context contract before any tower or decoder dispatch.
    pub fn prepare_prompt(
        &self,
        tokens: Box<[u64]>,
        image: &PreparedVisionImage,
        context_capacity: u32,
    ) -> Result<(Box<[u64]>, usize), NativeVisionError<B>> {
        self.validate_prepared_identity(image)?;
        let (expanded, start) = expand_image_placeholder(tokens, image.image_token_count, context_capacity)?;
        let input = self.preflight_prepared_prefill(expanded, image, start, context_capacity)?;
        Ok((input.tokens, start))
    }

    fn validate_prepared_identity(
        &self,
        image: &PreparedVisionImage,
    ) -> Result<(), NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("native vision model is fenced").into());
        }
        if !Arc::ptr_eq(&image.model_identity, &self.language_model.model_identity) {
            return Err(InvalidInput("prepared image belongs to a different loaded model").into());
        }
        Ok(())
    }

    #[cfg(all(test, backend = "metal"))]
    pub(crate) fn prepare_prefill(
        &mut self,
        tokens: Box<[u64]>,
        encoded_image: &[u8],
        token_start: usize,
        context_capacity: u32,
    ) -> Result<MultimodalPrefill, NativeVisionError<B>> {
        let prepared = self.prepare_image(encoded_image)?;
        self.prefill_prepared(tokens, prepared, token_start, context_capacity)
    }

    fn preflight_prepared_prefill(
        &self,
        tokens: Box<[u64]>,
        image: &PreparedVisionImage,
        token_start: usize,
        context_capacity: u32,
    ) -> Result<MultimodalPrefill, NativeVisionError<B>> {
        self.validate_prepared_identity(image)?;
        let prepared = &image.prepared;
        let grid = ImageGrid {
            temporal: prepared.grid[0],
            height: prepared.grid[1],
            width: prepared.grid[2],
            merge_size: 2,
        };
        let count = prepared.grid[1]
            .checked_mul(prepared.grid[2])
            .and_then(|n| n.checked_div(4))
            .and_then(|n| n.checked_mul(1024))
            .ok_or(InvalidInput("native projected image shape overflow"))?;
        let capacity = context_capacity.min(self.language_model.max_context_length().unwrap_or(context_capacity));
        // Validate the complete token/span/grid contract before any GPU work.
        // A temporary zero buffer supplies shape only; no zero features escape.
        let input = MultimodalPrefill::try_new(
            tokens,
            ProjectedImage {
                model_identity: self.language_model.model_identity.clone(),
                model_dim: 1024,
                data_type: DataType::BF16,
                grid,
                token_start,
                features: vec![bf16::ZERO; count as usize].into_boxed_slice(),
            },
            ImageTokenIds {
                image: 248056,
                vision_start: 248053,
                vision_end: 248054,
                video: 248057,
            },
            &self.language_model.model_identity,
            1024,
            DataType::BF16,
            248320,
            capacity,
        )?;
        Ok(input)
    }

    #[cfg(all(test, backend = "metal"))]
    fn prefill_prepared(
        &mut self,
        tokens: Box<[u64]>,
        image: PreparedVisionImage,
        token_start: usize,
        context_capacity: u32,
    ) -> Result<MultimodalPrefill, NativeVisionError<B>> {
        self.prefill_prepared_controlled(tokens, image, token_start, context_capacity, &mut || false)
    }

    fn prefill_prepared_controlled<F: FnMut() -> bool>(
        &mut self,
        tokens: Box<[u64]>,
        image: PreparedVisionImage,
        token_start: usize,
        context_capacity: u32,
        should_cancel: &mut F,
    ) -> Result<MultimodalPrefill, NativeVisionError<B>> {
        let mut input = self.preflight_prepared_prefill(tokens, &image, token_start, context_capacity)?;
        if should_cancel() {
            return Err(NativeVisionError::cancelled());
        }
        let count =
            image.image_token_count.checked_mul(1024).ok_or(InvalidInput("native projected image shape overflow"))?;
        input.image.features = self.project(&image.prepared, count as usize)?;
        // project() has waited for GPU completion and released all submissions.
        // Cancellation here leaves the decoder state fresh and the model usable.
        if should_cancel() {
            return Err(NativeVisionError::cancelled());
        }
        Ok(input)
    }

    fn project(
        &mut self,
        prepared: &PreparedImage,
        count: usize,
    ) -> Result<Box<[bf16]>, NativeVisionError<B>> {
        // Backend errors fence the loaded instance. Validation errors above are
        // recoverable and never enter this section or mutate native state.
        self.failed = true;
        let context = &self.engine.context;
        let mut commands = context.create_command_buffer(Some("vision"), None).map_err(NativeVisionError::backend)?;
        let output = self.tower.encode(&prepared.patches, prepared.grid, &mut commands)?;
        let bytes = count.checked_mul(size_of::<bf16>()).ok_or(InvalidInput("native vision readback size overflow"))?;
        if (&output).size() != bytes {
            return Err(InvalidInput("native vision output shape mismatch").into());
        }
        let mut readback = context.create_buffer(bytes).map_err(NativeVisionError::backend)?;
        commands.encode_copy(&output, &mut readback);
        commands.end_encoding().submit().wait_until_completed().map_err(NativeVisionError::backend)?;
        let features: Box<[bf16]> = (&readback).as_slice::<bf16>().into();
        if features.iter().any(|value| !value.is_finite()) {
            return Err(InvalidInput("nonfinite native vision output").into());
        }
        self.failed = false;
        Ok(features)
    }

    /// The exclusive model borrow lasts through stream completion, covering
    /// decode/preprocessing, tower execution, language prefill and generation.
    #[cfg(all(test, backend = "metal"))]
    pub(crate) fn stream_image<'a>(
        &'a mut self,
        tokens: Box<[u64]>,
        encoded_image: &[u8],
        token_start: usize,
        context_capacity: u32,
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<LanguageModelStream<'a, B>, NativeVisionError<B>> {
        self.preflight_image_stream(&tokens, context_capacity, state, &options)?;
        let prepared = self.prepare_image(encoded_image)?;
        self.stream_prepared_image(tokens, prepared, token_start, context_capacity, state, options)
    }

    fn preflight_image_stream(
        &self,
        tokens: &[u64],
        context_capacity: u32,
        state: &LanguageModelState<B>,
        options: &LanguageModelStreamOptions,
    ) -> Result<(), NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("native vision model is fenced").into());
        }
        self.language_model.preflight_multimodal_stream(tokens, state, options)?;
        if state.context_capacity() != Some(context_capacity) {
            return Err(InvalidInput("native image context must match the allocated request state").into());
        }
        if tokens.len() >= context_capacity as usize {
            return Err(InvalidInput("native image prompt exceeds requested context").into());
        }
        Ok(())
    }

    /// Consume the model-bound image after prompt expansion. Preprocessing is
    /// never repeated, and validation finishes before tower allocation/dispatch.
    /// Dropping the stream drains outstanding work before its storage releases.
    pub fn stream_prepared_image<'a>(
        &'a mut self,
        tokens: Box<[u64]>,
        image: PreparedVisionImage,
        token_start: usize,
        context_capacity: u32,
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<LanguageModelStream<'a, B>, NativeVisionError<B>> {
        self.stream_prepared_image_controlled(tokens, image, token_start, context_capacity, state, options, || false)
    }

    /// Cancellation is observed before tower submission and after its completed
    /// drain, before decoder prefill. The callback is not retained by the stream.
    pub fn stream_prepared_image_controlled<'a, F: FnMut() -> bool>(
        &'a mut self,
        tokens: Box<[u64]>,
        image: PreparedVisionImage,
        token_start: usize,
        context_capacity: u32,
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
        mut should_cancel: F,
    ) -> Result<LanguageModelStream<'a, B>, NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("native vision model is fenced").into());
        }
        if should_cancel() {
            return Err(NativeVisionError::cancelled());
        }
        self.preflight_image_stream(&tokens, context_capacity, state, &options)?;
        let input =
            self.prefill_prepared_controlled(tokens, image, token_start, context_capacity, &mut should_cancel)?;
        match self.language_model.stream_multimodal(&input, state, options) {
            Ok(stream) => Ok(stream.with_failure_fence(&mut self.failed)),
            Err(error) => {
                self.failed = true;
                Err(error.into())
            },
        }
    }

    /// Fresh-request text execution using the same exact loaded artifact.
    /// The underlying language model is not exposed as a mutable escape hatch.
    pub fn stream<'a>(
        &'a mut self,
        tokens: &[u64],
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<LanguageModelStream<'a, B>, NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("native vision model is fenced").into());
        }
        self.language_model.preflight_multimodal_stream(tokens, state, &options)?;
        match self.language_model.stream(tokens, state, options) {
            Ok(stream) => Ok(stream.with_failure_fence(&mut self.failed)),
            Err(error) => {
                self.failed = true;
                Err(error.into())
            },
        }
    }
}
