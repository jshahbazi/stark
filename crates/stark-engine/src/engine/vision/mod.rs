//! Admitted native Qwen3.5 image preparation and request execution.
//!
//! Compressed images become opaque, model-bound CPU patch storage. Projected
//! embeddings remain private and can only be produced by the admitted tower.

use std::{
    collections::BTreeMap,
    fmt,
    fs::{File, OpenOptions},
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
        Backend, BufferCpuAccessible, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
        Context, InvalidInput,
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
            dense_mtp::{DenseMtpError, DenseMtpRound, DenseMtpSession},
            multimodal::{ImageGrid, ImageTokenIds, MultimodalPrefill, ProjectedImage},
            state::{
                LanguageModelCreateEmptyStateError, LanguageModelRewindStateError, LanguageModelState,
                TextPrefixRestoreError,
            },
            stream::{LanguageModelStream, LanguageModelStreamError, LanguageModelStreamOptions},
        },
    },
    parameters::{NumericalProfile, ParameterLoader, ParameterLoaderError},
};

mod family;
mod fast_text;
pub mod gemma3;
pub mod gemma4;

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
#[cfg(all(test, backend = "metal"))]
mod text_parity_tests;
pub(crate) mod tower;

use preprocess::{PreparedImage, PreprocessLimits, RgbImage, decode_image, prepare_image as prepare_patches};
use tower::{VisionTower, VisionTowerConfig, VisionTowerError};

// The exact deterministic schema-2 artifact remains bound to its original
// tuple. Schema 3 has a separate structural family contract below.
const ARTIFACT_FILES: [(&str, &str); 5] = [
    ("config.json", "3f0cecb7cb3633bc271eabf69b06ac9f74b377b9ec17b441f5ceac331a71f42a"),
    ("model.safetensors", "7c5960e7225b3b36844fa397a85c5e20d281d732e382475062f552c0737c79d4"),
    ("tokenizer.json", "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"),
    ("preprocessor_config.json", "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"),
    ("chat_template.jinja", "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"),
];
const ARTIFACT_SCHEMA_VERSION: u32 = 2;

/// Immutable identity of the complete artifact admitted by this loaded model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionArtifactIdentity {
    pub schema_version: u32,
    pub config_sha256: String,
    pub artifact_sha256: String,
    pub tokenizer_sha256: String,
    pub processor_sha256: String,
    pub template_sha256: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactConfig {
    #[serde(rename = "type")]
    kind: String,
    decoder_config: DecoderConfig,
    token_codec_config: AnyTokenCodecConfig,
    generation_config: GenerationConfig,
    #[serde(default)]
    mtp_config: Option<crate::config::qwen_mtp::QwenMtpConfig>,
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
    #[error("Native vision prefix state is unsupported")]
    Rewind(LanguageModelRewindStateError),
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
            NativeVisionFailure::State(_) | NativeVisionFailure::Rewind(_) => NativeVisionErrorKind::State,
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
native_vision_error_from!(Rewind, LanguageModelRewindStateError);

fn finish_text_prefix_restore<B: Backend>(
    result: Result<(), TextPrefixRestoreError>,
    failed: &mut bool,
) -> Result<(), NativeVisionError<B>> {
    result.map_err(|error| {
        if matches!(error, TextPrefixRestoreError::RestoreFailed) {
            *failed = true;
        }
        LanguageModelRewindStateError.into()
    })
}

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
    tokens: &[u64],
    image_token_count: u32,
    context_capacity: u32,
    image_token: u64,
) -> Result<(Box<[u64]>, usize), InvalidInput> {
    if !(1..=1024).contains(&image_token_count) {
        return Err(InvalidInput("native image token count exceeds processor bounds"));
    }
    let start = tokens
        .iter()
        .position(|&token| token == image_token)
        .ok_or(InvalidInput("native image prompt requires one image placeholder"))?;
    if tokens[start + 1..].contains(&image_token) {
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
    expanded.extend(std::iter::repeat_n(image_token, count));
    expanded.extend_from_slice(&tokens[start + 1..]);
    Ok((expanded.into_boxed_slice(), start))
}

pub struct VisionLanguageModel<B: Backend> {
    engine: Arc<Engine<B>>,
    language_model: LanguageModel<B>,
    text_language_model: Option<LanguageModel<B>>,
    parameter_storage: crate::parameters::ParameterStorageStatistics,
    tower: Option<VisionTower<B>>,
    limits: PreprocessLimits,
    failed: bool,
    identity: VisionArtifactIdentity,
    source: String,
    revision: Option<String>,
    image_tokens: ImageTokenIds,
    model_dim: u32,
    vocab_size: u32,
    codec_eos_token: Option<String>,
    family_profile: family::FamilyProfile,
}

fn file_matches(
    file: &mut File,
    expected: &str,
) -> io::Result<bool> {
    Ok(file_digest(file)? == expected)
}

fn file_digest(file: &mut File) -> io::Result<String> {
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65_536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn open_artifact_component(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC);
    }
    #[cfg(not(unix))]
    if !path.symlink_metadata()?.is_file() {
        return Err(io::Error::other("native vision artifact requires regular components"));
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err(io::Error::other("native vision artifact requires regular components"));
    }
    Ok(file)
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
    /// Load the pinned artifact or the explicitly supported merged BF16 family.
    /// Admission does not establish reference parity or local execution evidence.
    pub fn load_vision_language_model(
        self: &Arc<Self>,
        model_path: &Path,
    ) -> Result<VisionLanguageModel<B>, NativeVisionError<B>> {
        self.load_vision_language_model_inner(model_path, true, None)
    }

    /// Product worker admission including both resident execution graphs.
    #[doc(hidden)]
    pub fn load_vision_language_model_with_parameter_budget(
        self: &Arc<Self>,
        model_path: &Path,
        parameter_budget: usize,
    ) -> Result<VisionLanguageModel<B>, NativeVisionError<B>> {
        self.load_vision_language_model_inner(model_path, true, Some(parameter_budget))
    }

    /// Validation-only tower residency control. The complete image artifact is
    /// admitted and rehashed identically. This preserves the reference image
    /// decoder, so it is a matched residency control only for profiles without
    /// a separate optimized text graph. Product loading always retains the tower.
    #[doc(hidden)]
    pub fn load_vision_text_validation_model(
        self: &Arc<Self>,
        model_path: &Path,
    ) -> Result<LanguageModel<B>, NativeVisionError<B>> {
        Ok(self.load_vision_language_model_inner(model_path, false, None)?.language_model)
    }

    fn load_vision_language_model_inner(
        self: &Arc<Self>,
        model_path: &Path,
        load_tower: bool,
        parameter_budget: Option<usize>,
    ) -> Result<VisionLanguageModel<B>, NativeVisionError<B>> {
        if self.capture_manager.is_some() || model_path.join("speculator").exists() {
            return Err(InvalidInput("native vision requires capture-free non-speculative loading").into());
        }
        // Verify before tensor allocation. The weights stay on this opened file
        // descriptor throughout loading, so path replacement cannot switch it.
        let mut verified = BTreeMap::new();
        let mut hashes = BTreeMap::new();
        let mut components: BTreeMap<&str, Vec<u8>> = BTreeMap::new();
        let mut family_profile = family::FamilyProfile::FixedV1;
        for (name, _) in ARTIFACT_FILES {
            let mut file = open_artifact_component(&model_path.join(name))?;
            let length = file.metadata()?.len();
            let maximum = match name {
                "config.json" => 1_048_576,
                "tokenizer.json" => 67_108_864,
                "preprocessor_config.json" => 65_536,
                "chat_template.jinja" => 262_144,
                _ => family_profile.max_weight_bytes(),
            };
            if !file.metadata()?.is_file() || length == 0 || length > maximum {
                return Err(InvalidInput("native vision artifact exceeds component bounds").into());
            }
            let hash = if name == "model.safetensors" {
                file_digest(&mut file)?
            } else {
                // Parse precisely the owned bytes that established admission.
                // In-place metadata edits cannot swap a reviewed tokenizer or
                // template program between its hash and parser.
                let mut bytes = Vec::new();
                Read::by_ref(&mut file).take(maximum + 1).read_to_end(&mut bytes)?;
                if bytes.len() as u64 != length || bytes.len() as u64 > maximum {
                    return Err(InvalidInput("native vision component changed while selecting bytes").into());
                }
                let digest = format!("{:x}", Sha256::digest(&bytes));
                components.insert(name, bytes);
                digest
            };
            if name == "config.json" && hash != ARTIFACT_FILES[0].1 {
                let bytes = components.get(name).ok_or(InvalidInput("missing native vision configuration"))?;
                let candidate: serde_json::Value = serde_json::from_slice(bytes)
                    .map_err(|_| InvalidInput("unsupported native vision artifact contract"))?;
                if candidate["type"] != "Qwen35VisionLanguageModelConfig"
                    || candidate["artifact_contract"]["schema_version"] != 3
                {
                    return Err(InvalidInput("unsupported native vision artifact contract").into());
                }
                family_profile = family::FamilyProfile::parse(&candidate["artifact_contract"]["family"])
                    .ok_or(InvalidInput("unsupported native vision artifact family"))?;
            }
            hashes.insert(name, hash);
            file.rewind()?;
            verified.insert(name, file);
        }
        let component =
            |name| components.get(name).map(Vec::as_slice).ok_or(InvalidInput("missing native vision component"));
        let raw_config: serde_json::Value = serde_json::from_slice(component("config.json")?)?;
        let schema = raw_config["artifact_contract"]["schema_version"].as_u64();
        // No arbitrary tokenizer regex/decoder pipeline is parsed by family v1.
        // Its reviewed processing identity must hold before construction.
        let pinned = match schema {
            Some(2) => &ARTIFACT_FILES[..],
            Some(3) => &ARTIFACT_FILES[..0],
            _ => return Err(InvalidInput("unsupported native vision artifact schema").into()),
        };
        if pinned.iter().any(|(name, expected)| hashes.get(name).map(String::as_str) != Some(*expected))
            || schema == Some(3)
                && !hashes.get("tokenizer.json").is_some_and(|digest| family_profile.accepts_tokenizer(digest))
        {
            return Err(InvalidInput("native vision artifact processing identity mismatch").into());
        }
        let tokenizer = Arc::new(
            tokenizers::Tokenizer::from_bytes(component("tokenizer.json")?)
                .map_err(EngineLoadLanguageModelError::Tokenizer)?,
        );
        let admission = match schema {
            Some(version) if version == u64::from(ARTIFACT_SCHEMA_VERSION) => {
                if ARTIFACT_FILES.iter().any(|(name, expected)| hashes.get(name).map(String::as_str) != Some(*expected))
                {
                    return Err(InvalidInput("native vision artifact identity mismatch").into());
                }
                if raw_config["type"] != "Qwen35VisionLanguageModelConfig"
                    || raw_config["artifact_contract"]["source"]["repo"] != "Qwen/Qwen3.5-0.8B"
                    || raw_config["artifact_contract"]["source"]["revision"]
                        != "2fc06364715b967f1860aea9cf38778875588b17"
                    || raw_config["artifact_contract"]["transport"] != "mediaref-v1"
                    || raw_config["artifact_contract"]["runtime_supported"] != true
                {
                    return Err(InvalidInput("unsupported native vision artifact contract").into());
                }
                family::FamilyAdmission {
                    tower: official_tower_config(),
                    tokens: ImageTokenIds {
                        image: 248056,
                        vision_start: 248053,
                        vision_end: 248054,
                        video: 248057,
                    },
                    source: "Qwen/Qwen3.5-0.8B".into(),
                    revision: Some("2fc06364715b967f1860aea9cf38778875588b17".into()),
                    profile: family::FamilyProfile::FixedV1,
                    tensors: None,
                }
            },
            Some(3) => {
                let admission = family::admit(
                    &raw_config,
                    &hashes,
                    component("preprocessor_config.json")?,
                    component("chat_template.jinja")?,
                    &tokenizer,
                )?;
                family::validate_weights(
                    verified.get_mut("model.safetensors").ok_or(InvalidInput("missing native vision weights"))?,
                    &admission,
                )?;
                admission
            },
            _ => return Err(InvalidInput("unsupported native vision artifact schema").into()),
        };
        for (name, id) in [
            ("<|vision_start|>", admission.tokens.vision_start),
            ("<|vision_end|>", admission.tokens.vision_end),
            ("<|image_pad|>", admission.tokens.image),
            ("<|video_pad|>", admission.tokens.video),
        ] {
            if tokenizer.token_to_id(name).map(u64::from) != Some(id) {
                return Err(InvalidInput("native vision tokenizer identity mismatch").into());
            }
        }
        let config: ArtifactConfig = serde_json::from_value(raw_config)?;
        if config.kind != "Qwen35VisionLanguageModelConfig"
            || config.artifact_contract["schema_version"].as_u64() != schema
        {
            return Err(InvalidInput("unsupported native vision artifact contract").into());
        }
        let codec_eos_token = match &config.token_codec_config {
            AnyTokenCodecConfig::ChatCodecConfig(codec) => codec.eos_token.clone(),
            _ => None,
        };
        let language_config: LanguageModelConfig = serde_json::from_value(serde_json::json!({
            "type": "LanguageModelConfig", "decoder_config": config.decoder_config,
            "token_codec_config": config.token_codec_config, "generation_config": config.generation_config,
            "mtp_config": config.mtp_config,
        }))?;
        let model_dim = language_config.decoder_config.transformer_config.model_dim;
        let vocab_size = language_config.decoder_config.vocab_size;
        let weights = verified.get("model.safetensors").ok_or(InvalidInput("missing native vision weights"))?;
        let numerical_profile =
            if matches!(admission.profile, family::FamilyProfile::AffineV3 | family::FamilyProfile::MoeV4) {
                NumericalProfile::Qwen35AffineBF16
            } else {
                NumericalProfile::Qwen35ReferenceBF16
            };
        let mut loader = ParameterLoader::new(weights, &*self.context)?.with_numerical_profile(numerical_profile);
        // The ordinary text conversion uses the same raw parameter transforms
        // for dense BF16/affine and affine MoE. Expert banks load through
        // WeightMatrix too, so both graphs share the full immutable bank.
        // BF16 MoE has no admitted ordinary-text baseline yet; its reference
        // graph and expert-predictor work remain separately bounded.
        let fast_text = load_tower
            && matches!(
                admission.profile,
                family::FamilyProfile::FixedV1
                    | family::FamilyProfile::DynamicV2
                    | family::FamilyProfile::AffineV3
                    | family::FamilyProfile::MoeV4
            );
        let extra = if fast_text {
            loader.qwen_text_graph_extra_bytes()
        } else {
            Some(0)
        };
        if let Some(budget) = parameter_budget {
            let required = usize::try_from(weights.metadata()?.len()).ok().and_then(|bytes| bytes.checked_add(extra?));
            if required.is_none_or(|required| required > budget) {
                return Err(InvalidInput("native vision resident parameter budget exceeded").into());
            }
        }
        let text_config = fast_text.then(|| fast_text::config(&language_config)).transpose()?;
        if fast_text {
            loader.enable_weight_sharing();
        }
        let tower = load_tower
            .then(|| VisionTower::new(&*self.context, &admission.tower, &loader.tree().subtree("vision")))
            .transpose()?;
        let mut image_config = language_config;
        if fast_text {
            image_config.mtp_config = None;
        }
        let language_model = self.language_model_from_config(
            model_path,
            image_config,
            &loader,
            Some(tokenizer.clone()),
            SpeculatorPolicy::Disabled,
        )?;
        let text_language_model = if let Some(config) = text_config {
            loader.set_numerical_profile(NumericalProfile::Legacy);
            Some(self.language_model_from_config(
                model_path,
                config,
                &loader,
                Some(tokenizer),
                SpeculatorPolicy::Disabled,
            )?)
        } else {
            None
        };
        if load_tower {
            loader.tree().assert_all_tensors_validated()?;
        } else {
            // Full header admission above includes the omitted tower. Every
            // allocated decoder/predictor parameter must still be consumed.
            loader.tree().subtree("decoder").assert_all_tensors_validated()?;
            loader.tree().subtree("mtp").assert_all_tensors_validated()?;
        }
        let parameter_storage = loader.storage_statistics();
        drop(loader);
        // Recheck the same selected objects before publishing loaded buffers.
        // Ordinary concurrent in-place artifact changes cannot silently acquire
        // the source identity established by the initial admission pass.
        for (name, expected) in &hashes {
            let file = verified.get_mut(name).ok_or(InvalidInput("missing native vision artifact component"))?;
            file.rewind()?;
            if !file_matches(file, expected)? {
                return Err(InvalidInput("native vision artifact changed while loading").into());
            }
        }
        let hash = |name| hashes.get(name).cloned().ok_or(InvalidInput("missing native vision identity"));
        Ok(VisionLanguageModel {
            engine: self.clone(),
            language_model,
            text_language_model,
            parameter_storage,
            tower,
            limits: PreprocessLimits::default(),
            failed: false,
            identity: VisionArtifactIdentity {
                schema_version: schema.ok_or(InvalidInput("missing native vision schema"))? as u32,
                config_sha256: hash("config.json")?,
                artifact_sha256: hash("model.safetensors")?,
                tokenizer_sha256: hash("tokenizer.json")?,
                processor_sha256: hash("preprocessor_config.json")?,
                template_sha256: hash("chat_template.jinja")?,
            },
            source: admission.source,
            revision: admission.revision,
            image_tokens: admission.tokens,
            model_dim,
            vocab_size,
            codec_eos_token,
            family_profile: admission.profile,
        })
    }
}

impl<B: Backend> VisionLanguageModel<B> {
    fn text_model(&self) -> &LanguageModel<B> {
        self.text_language_model.as_ref().unwrap_or(&self.language_model)
    }

    pub fn text_execution_graph(&self) -> &'static str {
        if self.text_language_model.is_some() {
            match self.family_profile {
                family::FamilyProfile::AffineV3 | family::FamilyProfile::MoeV4 => "qwen-affine-legacy-text-v1",
                _ => "qwen-bf16-legacy-text-v1",
            }
        } else {
            "qwen-reference-bf16-text-v1"
        }
    }

    pub fn parameter_storage_statistics(&self) -> crate::parameters::ParameterStorageStatistics {
        self.parameter_storage
    }
    pub fn artifact_identity(&self) -> &VisionArtifactIdentity {
        &self.identity
    }

    pub fn family_profile(&self) -> (&'static str, u32) {
        self.family_profile.identity()
    }

    pub fn source_identity(&self) -> (&str, Option<&str>) {
        (&self.source, self.revision.as_deref())
    }

    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    pub fn codec_eos_token(&self) -> Option<&str> {
        self.codec_eos_token.as_deref()
    }

    pub fn tokenizer(&self) -> &Arc<tokenizers::Tokenizer> {
        self.language_model.tokenizer()
    }

    pub fn generation_config(&self) -> &GenerationConfig {
        self.language_model.generation_config()
    }

    /// Text speculation remains available with the image tower resident.
    /// Image execution never enters these predictor methods.
    pub fn has_dense_mtp(&self) -> bool {
        !self.failed && self.text_model().has_dense_mtp()
    }

    pub fn max_context_length(&self) -> Option<u32> {
        self.language_model.max_context_length()
    }

    /// Decoder-only admission for plain text or a prepared image prompt.
    /// The bounded image encoder is qualified independently.
    pub fn decoder_generation_memory_bytes(
        &self,
        input_tokens: u32,
        context_tokens: u32,
        max_new_tokens: u32,
    ) -> Option<usize> {
        self.language_model.ordinary_text_generation_memory_bytes(input_tokens, context_tokens, max_new_tokens)
    }

    pub fn text_generation_memory_bytes(
        &self,
        input: u32,
        context: u32,
        output: u32,
    ) -> Option<usize> {
        self.text_model().generation_memory_bytes(input, context, output)
    }

    /// Ordinary text keeps the loaded head but allocates no predictor session.
    pub fn ordinary_text_generation_memory_bytes(
        &self,
        input: u32,
        context: u32,
        output: u32,
    ) -> Option<usize> {
        self.text_model().ordinary_text_generation_memory_bytes(input, context, output)
    }

    /// Decoder admission when a scoped text request also retains one
    /// recurrent-state snapshot. Image and unscoped requests do not allocate it.
    pub fn decoder_text_prefix_memory_bytes(
        &self,
        input_tokens: u32,
        context_tokens: u32,
        max_new_tokens: u32,
    ) -> Option<usize> {
        self.text_model().generation_memory_bytes_with_text_prefix(input_tokens, context_tokens, max_new_tokens)
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

    /// Whether text-only requests on this loaded image artifact can restore a
    /// retained decoder prefix. Image-prefilled states are never reusable.
    pub fn prefix_reuse_supported(&self) -> bool {
        !self.failed && self.text_model().text_prefix_snapshot_supported()
    }

    /// Retain the last non-final text prefill boundary for the next request.
    /// A zero position disables capture for prompts shorter than one chunk.
    pub fn prepare_text_prefix_capture(
        &mut self,
        state: &mut LanguageModelState<B>,
        position: usize,
    ) -> Result<(), NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("invalid text prefix capture state").into());
        }
        self.text_model().prepare_text_prefix_capture(state, position)?;
        Ok(())
    }

    /// Rewind a completed text-only request state through the guarded wrapper.
    /// The decoder performs loaded-instance, cache-layout, context and chunk
    /// boundary checks; this wrapper additionally rejects image-prefilled state.
    pub fn rewind_state(
        &mut self,
        state: &mut LanguageModelState<B>,
        prefix_length: usize,
        max_context_length: Option<u32>,
        sampling_seed: u64,
    ) -> Result<(), NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("native vision model is fenced").into());
        }
        let result = self.text_model().restore_text_prefix(state, prefix_length, max_context_length, sampling_seed);
        finish_text_prefix_restore(result, &mut self.failed)
    }

    /// Ordinary text state has a distinct model identity from image state.
    pub fn create_text_state(
        &mut self,
        capacity: Option<u32>,
        seed: u64,
    ) -> Result<LanguageModelState<B>, NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("native vision model is fenced").into());
        }
        match self.text_model().create_empty_state(capacity, seed) {
            Ok(state) => Ok(state),
            Err(error) => {
                if matches!(&error, LanguageModelCreateEmptyStateError::Backend(_)) {
                    self.failed = true;
                }
                Err(error.into())
            },
        }
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
        let (expanded, start) =
            expand_image_placeholder(&tokens, image.image_token_count, context_capacity, self.image_tokens.image)?;
        drop(tokens);
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
            .and_then(|n| n.checked_mul(self.model_dim))
            .ok_or(InvalidInput("native projected image shape overflow"))?;
        let capacity = context_capacity.min(self.language_model.max_context_length().unwrap_or(context_capacity));
        // Validate the complete token/span/grid contract before any GPU work.
        // A temporary zero buffer supplies shape only; no zero features escape.
        let input = MultimodalPrefill::try_new(
            tokens,
            ProjectedImage {
                model_identity: self.language_model.model_identity.clone(),
                model_dim: self.model_dim,
                data_type: DataType::BF16,
                grid,
                token_start,
                features: vec![bf16::ZERO; count as usize].into_boxed_slice(),
            },
            self.image_tokens,
            &self.language_model.model_identity,
            self.model_dim,
            DataType::BF16,
            self.vocab_size,
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
        let count = image
            .image_token_count
            .checked_mul(self.model_dim)
            .ok_or(InvalidInput("native projected image shape overflow"))?;
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
        let tower = self.tower.as_ref().ok_or(InvalidInput("native vision tower is unavailable"))?;
        let output = tower.encode(&prepared.patches, prepared.grid, &mut commands)?;
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

    /// Reference image-decoder text execution for numerical validation. Product
    /// text requests use `create_text_state` and `stream_text`.
    pub fn stream<'a>(
        &'a mut self,
        tokens: &[u64],
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<LanguageModelStream<'a, B>, NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("native vision model is fenced").into());
        }
        if state.tokens().is_empty() {
            self.language_model.preflight_multimodal_stream(tokens, state, &options)?;
        } else {
            self.language_model.preflight_restored_text_stream(tokens, state, &options)?;
        }
        match self.language_model.stream(tokens, state, options) {
            Ok(stream) => Ok(stream.with_failure_fence(&mut self.failed)),
            Err(error) => {
                self.failed = true;
                Err(error.into())
            },
        }
    }

    /// Ordinary optimized text execution while the image tower remains loaded.
    pub fn stream_text<'a>(
        &'a mut self,
        tokens: &[u64],
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<LanguageModelStream<'a, B>, NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("native vision model is fenced").into());
        }
        let model = self.text_language_model.as_ref().unwrap_or(&self.language_model);
        if state.tokens().is_empty() {
            model.preflight_plain_text_stream(tokens, state, &options)?;
        } else {
            model.preflight_restored_text_stream(tokens, state, &options)?;
        }
        match model.stream(tokens, state, options) {
            Ok(stream) => Ok(stream.with_failure_fence(&mut self.failed)),
            Err(error) => {
                self.failed = true;
                Err(error.into())
            },
        }
    }
}

impl<B: Backend> VisionLanguageModel<B>
where
    B::ScratchBuffer: BufferCpuAccessible,
{
    fn dense_text_operation<T>(
        &mut self,
        operation: impl FnOnce(&LanguageModel<B>) -> Result<T, DenseMtpError<B>>,
    ) -> Result<T, DenseMtpError<B>> {
        if self.failed || !self.has_dense_mtp() {
            return Err(DenseMtpError::Invalid);
        }
        let result = operation(self.text_model());
        // Drained cancellation discards its request-owned session. Invalid
        // errors are resolved using the session's mutation fence below; other
        // native failures fence the loaded instance immediately.
        if result.as_ref().is_err_and(|error| !matches!(error, DenseMtpError::Invalid | DenseMtpError::Cancelled)) {
            self.failed = true;
        }
        result
    }

    pub fn prefill_dense_mtp_controlled(
        &mut self,
        input: &[u32],
        capacity: u32,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<DenseMtpSession<B>, DenseMtpError<B>> {
        if !self.has_dense_mtp()
            || input.is_empty()
            || capacity == 0
            || input.len() >= capacity as usize
            || self.max_context_length().is_some_and(|n| capacity > n)
            || input.iter().any(|&token| token >= self.vocab_size)
        {
            return Err(DenseMtpError::Invalid);
        }
        let result = self.dense_text_operation(|model| model.prefill_dense_mtp_controlled(input, capacity, cancelled));
        self.failed |= result.as_ref().is_err_and(|error| !matches!(error, DenseMtpError::Cancelled));
        result
    }

    pub fn restore_dense_mtp_prefix(
        &mut self,
        session: &mut DenseMtpSession<B>,
        position: usize,
    ) -> Result<(), DenseMtpError<B>> {
        let was_fenced = session.is_fenced();
        let result = self.dense_text_operation(|model| model.restore_dense_mtp_prefix(session, position));
        self.failed |= !was_fenced && session.is_fenced() && !matches!(&result, Err(DenseMtpError::Cancelled));
        result
    }

    pub fn continue_prefill_dense_mtp_controlled(
        &mut self,
        session: &mut DenseMtpSession<B>,
        input: &[u32],
        capture_position: usize,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<(), DenseMtpError<B>> {
        let was_fenced = session.is_fenced();
        let result = self.dense_text_operation(|model| {
            model.continue_prefill_dense_mtp_controlled(session, input, capture_position, cancelled)
        });
        self.failed |= !was_fenced && session.is_fenced() && !matches!(&result, Err(DenseMtpError::Cancelled));
        result
    }

    pub fn dense_mtp_round(
        &mut self,
        session: &mut DenseMtpSession<B>,
        first: u32,
        drafts: usize,
        sampler: &mut dyn crate::engine::hybrid::MtpSampler,
    ) -> Result<DenseMtpRound, DenseMtpError<B>> {
        let was_fenced = session.is_fenced();
        let result = self.dense_text_operation(|model| model.dense_mtp_round(session, first, drafts, sampler));
        self.failed |= !was_fenced && session.is_fenced() && !matches!(&result, Err(DenseMtpError::Cancelled));
        result
    }
}
