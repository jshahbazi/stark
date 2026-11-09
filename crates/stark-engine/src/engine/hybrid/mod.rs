//! Architecture-specific execution for GLM5-Next, Qwen4-Exp and Recursive Flint.
//!
//! These models carry distinct recurrent state, sparse attention and residual
//! streams. Qwen3.8 also admits the typed Uzu converted architecture extension.

pub mod activity;
pub mod finite_guard;
pub mod flint;
pub mod glm;
pub mod guided_admission;
mod image;
pub use image::PreparedHybridImage;
#[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
mod glm_exl3;
#[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
mod glm_exl3_ffi;
#[cfg(feature = "metal")]
mod glm_metal;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod lily;
pub mod math;
mod memory;
pub mod profile;
pub mod qwen;
#[cfg(feature = "metal")]
mod qwen_metal;
mod uzu;
#[cfg(feature = "metal")]
mod weight_metal;
pub mod weights;

use std::{fs::File, io::Read, path::Path, sync::Arc};

use serde_json::Value;
use thiserror::Error;
use tokenizers::Tokenizer;

#[derive(Debug, Error)]
pub enum Error {
    #[error("native hybrid image request cancelled")]
    Cancelled,
    #[error("invalid native hybrid model: {0}")]
    Invalid(&'static str),
    #[error("unsupported native hybrid model: {0}")]
    Unsupported(&'static str),
    #[error("native hybrid model I/O failed")]
    Io(#[from] std::io::Error),
    #[error("native hybrid model metadata is invalid")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Backend-owned recurrent state at a prompt prefix boundary. It remains
/// private to one loaded model instance (see [`Model::restore_prefix`]).
pub struct PrefixSnapshot {
    inner: PrefixState,
}

enum PrefixState {
    Glm(glm::PrefixSnapshot),
    #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
    GlmExl3(glm_exl3::PrefixSnapshot),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Lily(lily::PrefixSnapshot),
}

impl PrefixSnapshot {
    pub fn position(&self) -> usize {
        match &self.inner {
            PrefixState::Glm(snapshot) => snapshot.position(),
            #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
            PrefixState::GlmExl3(snapshot) => snapshot.position(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            PrefixState::Lily(snapshot) => snapshot.position(),
        }
    }

    pub fn mtp_enabled(&self) -> bool {
        match &self.inner {
            PrefixState::Glm(_) => false,
            #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
            PrefixState::GlmExl3(snapshot) => snapshot.mtp_enabled(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            PrefixState::Lily(snapshot) => snapshot.mtp_enabled(),
        }
    }
}

/// Loaded-model MTP policy. Recommendations select the depth when a caller
/// enables MTP without choosing an explicit draft count; MTP remains opt-in.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtpCapabilities {
    pub max_drafts: usize,
    pub recommended_drafts: usize,
    pub sampling: &'static str,
    pub verification: &'static str,
}

/// Caller-owned complete-vocabulary sampling. A round records draft
/// distributions before target verification consumes them in prefix order.
pub trait MtpSampler {
    fn begin_round(&mut self);
    fn draft(
        &mut self,
        logits: &[f32],
    ) -> Result<u32>;
    fn target(
        &mut self,
        logits: &[f32],
        proposal: Option<u32>,
    ) -> Result<u32>;
}

/// Aggregate request counters. No token, logit or probability data is retained.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct MtpStats {
    pub rounds: usize,
    pub proposed: usize,
    pub accepted: usize,
    pub verified_rows: usize,
    pub draft_seconds: f64,
    pub verify_seconds: f64,
    pub rollback_seconds: f64,
}

/// Explicit device target-draw policy for the staged GLM native stream.
/// Its position-keyed RNG is distinct from the caller-owned Rust sampler.
#[derive(Clone, Copy)]
pub enum NativeSampling {
    Greedy {
        seed: u64,
    },
    Stochastic {
        temperature: f32,
        top_p: f32,
        seed: u64,
    },
}

impl NativeSampling {
    pub fn is_supported(self) -> bool {
        match self {
            Self::Greedy {
                ..
            } => true,
            Self::Stochastic {
                temperature,
                top_p,
                ..
            } => {
                temperature.is_finite()
                    && (0.01..=100.0).contains(&temperature)
                    && top_p.is_finite()
                    && top_p > 0.0
                    && top_p <= 1.0
            },
        }
    }
}

/// Native request aggregates. `verified_drafts` counts gate-admitted drafts;
/// it does not count every head proposal computed before the confidence gate.
#[derive(Clone, Copy, Default, serde::Serialize)]
pub struct NativeDecodeStats {
    pub rounds: usize,
    pub verified_drafts: usize,
    pub accepted: usize,
    pub committed: usize,
}

fn json_file(
    path: &Path,
    limit: u64,
) -> Result<Value> {
    let file = File::open(path)?;
    if !file.metadata()?.is_file() || file.metadata()?.len() > limit {
        return Err(Error::Invalid("configuration exceeds size limit"));
    }
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(Error::Invalid("configuration exceeds size limit"));
    }
    Ok(serde_json::from_slice(&bytes)?)
}

fn config(path: &Path) -> Result<Value> {
    json_file(&path.join("config.json"), 8 * 1024 * 1024)
}

fn extend_eos(
    value: &Value,
    eos: &mut Vec<u32>,
    vocab_size: usize,
) -> Result<()> {
    let values: Vec<&Value> = match value {
        Value::Array(values) => values.iter().collect(),
        _ => vec![value],
    };
    for value in values {
        let id = value.as_u64().filter(|id| *id < vocab_size as u64).ok_or(Error::Invalid("EOS IDs"))? as u32;
        if !eos.contains(&id) {
            eos.push(id);
        }
    }
    Ok(())
}

/// Detect only the explicitly implemented architectures; never infer one
/// from a repository name or load repository-provided Python code.
pub fn recognizes(path: &Path) -> Result<bool> {
    let value = config(path)?;
    Ok(value.get("type").and_then(Value::as_str) == Some(uzu::CONFIG_TYPE)
        || matches!(value.get("model_type").and_then(Value::as_str), Some("glm5_next" | "qwen4_exp" | "flint")))
}

enum Forward {
    Flint(flint::FlintModel),
    Glm(glm::GlmModel),
    #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
    GlmExl3(Box<glm_exl3::GlmExl3Model>),
    Qwen(qwen::QwenModel),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Lily(Box<lily::LilyModel>),
}

fn glm_exl3_checkpoint(config: &Value) -> bool {
    config.get("model_type").and_then(Value::as_str) == Some("glm5_next")
        && config.pointer("/expert_streaming/family").and_then(Value::as_str) == Some("mlx-stream")
        && config.pointer("/expert_streaming/format").and_then(Value::as_str) == Some("mtplx-expert-manifest-v2")
        && config.pointer("/expert_streaming/manifest").and_then(Value::as_str) == Some("expert-manifest-v2.json")
        && config.pointer("/expert_streaming/runtime_manifest").and_then(Value::as_str) == Some("expert-manifest.json")
        && config.pointer("/expert_streaming/quant").and_then(Value::as_str) == Some("exl3-k3-mul1")
}

fn mixed_precision_checkpoint(config: &Value) -> bool {
    config.get("model_type").and_then(Value::as_str) == Some("qwen4_exp")
        && (config.pointer("/lily/format").and_then(Value::as_str) == Some("qwen4_exp-affine-v1")
            || (config.pointer("/ngram_table/file").and_then(Value::as_str) == Some("ngram_table.bin")
                && config.pointer("/ngram_table/bits").and_then(Value::as_u64) == Some(4)
                && config.pointer("/ngram_table/group_size").and_then(Value::as_u64) == Some(32)
                && config.pointer("/quantization/mode").and_then(Value::as_str) == Some("affine")
                && config.pointer("/quantization/bits").and_then(Value::as_u64) == Some(4)
                && config.pointer("/quantization/group_size").and_then(Value::as_u64) == Some(64)))
}

pub struct Model {
    forward: Forward,
    tokenizer: Arc<Tokenizer>,
    vocab_size: usize,
    max_context: u32,
    eos: Vec<u32>,
    image_artifact: Option<image::artifact::ImageArtifact>,
}

impl Model {
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_with_mtp(path, false)
    }

    /// Loads the native head when the supported checkpoint declares one.
    /// Request-level execution remains disabled until explicitly selected.
    pub fn load_with_mtp(
        path: &Path,
        with_mtp: bool,
    ) -> Result<Self> {
        Self::load_with_policy(path, with_mtp, false, false)
    }

    /// Explicit source-bound native profile or private validation stage. The
    /// ordinary model loader retains its serial EXL3 reference behavior.
    /// Other model families, and builds without the stage, keep their loader.
    pub fn load_with_native_decode(
        path: &Path,
        with_mtp: bool,
    ) -> Result<Self> {
        Self::load_with_policy(path, with_mtp, Self::native_decode_stage_available(), false)
    }

    /// Load the original-format Qwen tower explicitly. Converted text exports
    /// and other hybrid architectures cannot acquire image capability here.
    pub fn load_with_image_support(
        path: &Path,
        with_mtp: bool,
    ) -> Result<Self> {
        Self::load_with_policy(path, with_mtp, false, true)
    }

    pub const fn native_decode_stage_available() -> bool {
        cfg!(all(
            feature = "glm-exl3",
            target_os = "macos",
            target_arch = "aarch64",
            stark_glm_exl3_native_decode_available
        ))
    }

    /// A model-specific default established by the explicitly selected stage.
    pub fn default_mtp_drafts(&self) -> u32 {
        #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
        if let Forward::GlmExl3(model) = &self.forward {
            return model.default_mtp_drafts();
        }
        0
    }

    fn load_with_policy(
        path: &Path,
        with_mtp: bool,
        native_decode_stage: bool,
        with_images: bool,
    ) -> Result<Self> {
        let _ = native_decode_stage;
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let _ = (with_mtp, with_images);
        let outer = config(path)?;
        if with_images && !mixed_precision_checkpoint(&outer) {
            return Err(Error::Unsupported("retained original-format Qwen image tower required"));
        }
        let image_artifact = if with_images {
            Some(image::artifact::ImageArtifact::admit(path)?)
        } else {
            None
        };
        if outer.get("type").and_then(Value::as_str) == Some(uzu::CONFIG_TYPE) {
            return Self::load_uzu(path);
        }
        let text = outer.get("text_config").unwrap_or(&outer);
        let vocab_size = text
            .get("vocab_size")
            .and_then(Value::as_u64)
            .filter(|n| *n > 0 && *n <= 1_000_000)
            .ok_or(Error::Invalid("vocabulary size"))? as usize;
        let max_context = text
            .get("max_position_embeddings")
            .and_then(Value::as_u64)
            .filter(|n| *n > 0 && *n <= u32::MAX as u64)
            .ok_or(Error::Invalid("context size"))? as u32;
        let tokenizer =
            Arc::new(Tokenizer::from_file(path.join("tokenizer.json")).map_err(|_| Error::Invalid("tokenizer"))?);
        let vocab = tokenizer.get_vocab(true);
        if vocab.is_empty() || vocab.values().any(|&id| id as usize >= vocab_size) {
            return Err(Error::Invalid("tokenizer vocabulary"));
        }
        let mut eos = Vec::new();
        if let Some(value) = text.get("eos_token_id") {
            extend_eos(value, &mut eos, vocab_size)?;
        }
        let generation_path = path.join("generation_config.json");
        if generation_path.exists() {
            let generation = json_file(&generation_path, 1024 * 1024)?;
            if let Some(value) = generation.get("eos_token_id") {
                extend_eos(value, &mut eos, vocab_size)?;
            }
        }
        let tokenizer_path = path.join("tokenizer_config.json");
        if tokenizer_path.exists() {
            let metadata = json_file(&tokenizer_path, 2 * 1024 * 1024)?;
            if let Some(value) = metadata.get("eos_token").filter(|value| !value.is_null()) {
                let token = value
                    .as_str()
                    .or_else(|| value.get("content").and_then(Value::as_str))
                    .filter(|token| !token.is_empty())
                    .ok_or(Error::Invalid("EOS token"))?;
                let id = tokenizer.token_to_id(token).ok_or(Error::Invalid("EOS token"))?;
                extend_eos(&Value::from(id), &mut eos, vocab_size)?;
            }
        }
        if eos.iter().any(|id| !vocab.values().any(|value| value == id)) {
            return Err(Error::Invalid("EOS IDs"));
        }
        if eos.is_empty() {
            return Err(Error::Invalid("EOS IDs"));
        }
        let (forward, max_context) = if outer.get("expert_streaming").is_some() {
            if !glm_exl3_checkpoint(&outer) {
                return Err(Error::Unsupported("GLM streaming pack format"));
            }
            #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
            {
                let context = max_context.min(32_768);
                let canonical = path.canonicalize()?;
                let mut options = glm_exl3::LoadOptions::new(context as usize);
                options.prefix_reuse = cfg!(stark_glm_exl3_prefix_available);
                options.native_decode_stage = native_decode_stage;
                options.load_mtp_head = native_decode_stage && with_mtp;
                let model = glm_exl3::GlmExl3Model::load(&canonical, vocab_size, options)?;
                if model.vocab_size() != vocab_size
                    || model.max_context_length() != context as usize
                    || model.position() != 0
                {
                    return Err(Error::Invalid("GLM EXL3 loaded model metadata"));
                }
                (Forward::GlmExl3(Box::new(model)), context)
            }
            #[cfg(not(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64")))]
            {
                return Err(Error::Unsupported("GLM EXL3 requires the optional macOS arm64 backend"));
            }
        } else if mixed_precision_checkpoint(&outer) {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                let model = lily::LilyModel::load_with_options(path, with_mtp, with_images)?;
                if with_images {
                    model.validate_image_tokenizer(&tokenizer)?;
                }
                if model.vocab_size() != vocab_size || model.position() != 0 {
                    return Err(Error::Invalid("mixed precision metadata mismatch"));
                }
                let context = max_context.min(model.max_context() as u32);
                (Forward::Lily(Box::new(model)), context)
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                return Err(Error::Unsupported("mixed precision checkpoint requires macOS Metal"));
            }
        } else {
            #[cfg(feature = "metal")]
            weight_metal::initialize()?;
            let weights = weights::Weights::open(path)?;
            let forward = match outer.get("model_type").and_then(Value::as_str) {
                Some("flint") => Forward::Flint(flint::FlintModel::new(weights, &outer)?),
                Some("glm5_next") => Forward::Glm(glm::GlmModel::new(weights, &outer)?),
                Some("qwen4_exp") => Forward::Qwen(qwen::QwenModel::new(weights, &outer)?),
                _ => return Err(Error::Unsupported("architecture")),
            };
            (forward, max_context)
        };
        if let Some(artifact) = &image_artifact {
            artifact.unchanged(path)?;
        }
        Ok(Self {
            forward,
            tokenizer,
            vocab_size,
            max_context,
            eos,
            image_artifact,
        })
    }

    pub fn supports_images(&self) -> bool {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Forward::Lily(model) = &self.forward {
            return self.image_artifact.is_some() && model.supports_images();
        }
        false
    }

    pub fn image_artifact_identity(&self) -> Option<&crate::engine::vision::VisionArtifactIdentity> {
        self.image_artifact.as_ref().map(|artifact| &artifact.identity)
    }

    pub fn image_source_identity(&self) -> Option<(&str, Option<&str>)> {
        self.image_artifact.as_ref().map(|artifact| (artifact.source.as_str(), Some(artifact.revision.as_str())))
    }

    pub fn image_family_profile(&self) -> Option<(&'static str, u32)> {
        self.image_artifact.as_ref().map(|_| (image::artifact::PROFILE, 1))
    }

    pub fn prepare_image(
        &self,
        encoded: &[u8],
    ) -> Result<PreparedHybridImage> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Forward::Lily(model) = &self.forward {
            return model.prepare_image(encoded);
        }
        let _ = encoded;
        Err(Error::Unsupported("native image preparation unavailable"))
    }

    pub fn prepare_image_prompt(
        &self,
        tokens: &[u32],
        image: &PreparedHybridImage,
        context_capacity: u32,
    ) -> Result<Box<[u32]>> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Forward::Lily(model) = &self.forward {
            return model.prepare_image_prompt(tokens, image, context_capacity as usize);
        }
        let _ = (tokens, image, context_capacity);
        Err(Error::Unsupported("native image prompt unavailable"))
    }

    /// Consume an expanded image prompt and return its first target logits.
    /// Image state never participates in generation prefix reuse.
    pub fn prefill_image(
        &mut self,
        tokens: &[u32],
        image: PreparedHybridImage,
    ) -> Result<Vec<f32>> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Forward::Lily(model) = &mut self.forward {
            return model.prefill_image(tokens, image);
        }
        let _ = (tokens, image);
        Err(Error::Unsupported("native image prefill unavailable"))
    }

    pub fn prefill_image_controlled<F: FnMut() -> bool>(
        &mut self,
        tokens: &[u32],
        image: PreparedHybridImage,
        should_cancel: F,
    ) -> Result<Vec<f32>> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Forward::Lily(model) = &mut self.forward {
            return model.prefill_image_controlled(tokens, image, should_cancel);
        }
        let _ = (tokens, image, should_cancel);
        Err(Error::Unsupported("native image prefill unavailable"))
    }

    fn load_uzu(path: &Path) -> Result<Self> {
        let config = uzu::read_config(path)?;
        let architecture = &config.decoder_config.architecture;
        let vocab_size = architecture.vocab_size as usize;
        let tokenizer = Arc::new(
            Tokenizer::from_file(path.join("tokenizer.json")).map_err(|_| Error::Invalid("Uzu Qwen tokenizer"))?,
        );
        let vocab = tokenizer.get_vocab(true);
        if vocab.is_empty() || vocab.values().any(|&id| id as usize >= vocab_size) {
            return Err(Error::Invalid("Uzu Qwen tokenizer vocabulary"));
        }
        let mut eos = architecture.eos_token_id.to_vec();
        for &stop in &config.generation_config.stop_token_ids {
            let id = u32::try_from(stop).map_err(|_| Error::Invalid("Uzu Qwen stop identifiers"))?;
            if !eos.contains(&id) {
                eos.push(id);
            }
        }
        if let crate::config::token_codec::AnyTokenCodecConfig::ChatCodecConfig(codec) = &config.token_codec_config
            && let Some(token) = &codec.eos_token
        {
            let id = tokenizer.token_to_id(token).ok_or(Error::Invalid("Uzu Qwen EOS token"))?;
            if !eos.contains(&id) {
                eos.push(id);
            }
        }
        if eos.iter().any(|id| *id as usize >= vocab_size || !vocab.values().any(|value| value == id)) {
            return Err(Error::Invalid("Uzu Qwen EOS identifiers"));
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let model = lily::LilyModel::load_uzu(path, &config.decoder_config)?;
            if model.vocab_size() != vocab_size || model.position() != 0 {
                return Err(Error::Invalid("Uzu Qwen architecture metadata mismatch"));
            }
            let max_context = architecture.max_position_embeddings.min(model.max_context() as u32);
            Ok(Self {
                forward: Forward::Lily(Box::new(model)),
                tokenizer,
                vocab_size,
                max_context,
                eos,
                image_artifact: None,
            })
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            Err(Error::Unsupported("Uzu Qwen converted execution requires macOS Metal"))
        }
    }

    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn max_context_length(&self) -> u32 {
        self.max_context
    }

    /// Allocation admission uses the loaded execution family and cache policy.
    /// The returned context remains bounded by the model's declared capacity.
    pub fn automatic_context_length(
        &self,
        input_tokens: u32,
        maximum: u32,
        memory_budget_bytes: usize,
    ) -> Option<u32> {
        self.automatic_context_length_with_prefix(input_tokens, maximum, memory_budget_bytes, true)
    }

    pub fn automatic_context_length_with_prefix(
        &self,
        input_tokens: u32,
        maximum: u32,
        memory_budget_bytes: usize,
        capture_prefix: bool,
    ) -> Option<u32> {
        self.automatic_context_length_for_request(input_tokens, maximum, memory_budget_bytes, capture_prefix, false)
    }

    pub fn automatic_context_length_for_request(
        &self,
        input_tokens: u32,
        maximum: u32,
        memory_budget_bytes: usize,
        capture_prefix: bool,
        greedy_without_mtp: bool,
    ) -> Option<u32> {
        crate::encodable_block::memory::automatic_context(
            input_tokens,
            maximum.min(self.max_context),
            memory_budget_bytes,
            |context| {
                self.generation_memory_bytes_for_request(context, input_tokens, capture_prefix, greedy_without_mtp)
                    .map(crate::encodable_block::memory::MemoryEstimate::state)
            },
        )
    }

    pub fn supports_automatic_context_planning(&self) -> bool {
        self.request_memory_bytes(2.min(self.max_context), 1).is_some()
    }

    /// Release only per-request owners which cannot participate in prefix
    /// reuse, before the caller samples current OS headroom. Persistent native
    /// recurrent/cache owners and retained prefix snapshots are preserved.
    pub fn prepare_request_memory(&mut self) -> Result<bool> {
        match &mut self.forward {
            Forward::Qwen(model) => model.prepare_request_memory(),
            _ => Ok(false),
        }
    }

    /// Additional state and transient workspace for generation. Persistent
    /// family owners are charged only when they need allocation or growth.
    pub fn request_memory_bytes(
        &self,
        context_tokens: u32,
        input_tokens: u32,
    ) -> Option<usize> {
        self.generation_memory_bytes(context_tokens, input_tokens, true)
    }

    /// Exact scoring does not capture a generation prefix or reserve output.
    pub fn scoring_memory_bytes(
        &self,
        input_tokens: u32,
    ) -> Option<usize> {
        let additional_device_readout = match &self.forward {
            Forward::Flint(_) => cfg!(feature = "metal"),
            Forward::Glm(model) => model.additional_scoring_device_readout(),
            Forward::Qwen(model) => model.additional_scoring_device_readout(),
            // These backends retain/prepay their device readout buffers.
            #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
            Forward::GlmExl3(_) => false,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Forward::Lily(_) => false,
        };
        self.request_state_memory_bytes(input_tokens, input_tokens, false)?
            .checked_add(memory::scoring_host_workspace(self.vocab_size, additional_device_readout)?)
    }

    pub fn generation_memory_bytes(
        &self,
        context_tokens: u32,
        input_tokens: u32,
        capture_prefix: bool,
    ) -> Option<usize> {
        self.generation_memory_bytes_for_request(context_tokens, input_tokens, capture_prefix, false)
    }

    /// Only greedy requests with MTP disabled omit stochastic sampler arrays.
    /// Count-only/legacy callers must retain the conservative default above.
    pub fn generation_memory_bytes_for_request(
        &self,
        context_tokens: u32,
        input_tokens: u32,
        capture_prefix: bool,
        greedy_without_mtp: bool,
    ) -> Option<usize> {
        self.request_state_memory_bytes(context_tokens, input_tokens, capture_prefix)?.checked_add(
            memory::host_workspace_for_request(context_tokens, input_tokens, self.vocab_size, greedy_without_mtp)?,
        )
    }

    fn request_state_memory_bytes(
        &self,
        context_tokens: u32,
        input_tokens: u32,
        capture_prefix: bool,
    ) -> Option<usize> {
        if input_tokens == 0 || context_tokens < input_tokens || context_tokens > self.max_context {
            return None;
        }
        let state = match &self.forward {
            Forward::Flint(model) => model.request_memory_bytes(context_tokens)?,
            Forward::Glm(model) => model.request_memory_bytes(context_tokens, input_tokens, capture_prefix)?,
            Forward::Qwen(model) => model.request_memory_bytes(context_tokens)?,
            #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
            // The bridge's loader bills all device phases at its installed
            // full context, including MTP and prefix-copy peak, before load.
            Forward::GlmExl3(_) => 0,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Forward::Lily(model) => model.request_memory_bytes(context_tokens, input_tokens)?,
        };
        Some(state)
    }
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos
    }

    /// Each worker request starts from reset state, optionally resumed at a
    /// prefix snapshot. No prompt or recurrent state is persisted, and
    /// failures invalidate the entire current sequence.
    pub fn reset(&mut self) {
        match &mut self.forward {
            Forward::Flint(model) => model.reset(),
            Forward::Glm(model) => model.reset(),
            #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
            Forward::GlmExl3(model) => model.reset(),
            Forward::Qwen(model) => model.reset(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Forward::Lily(model) => model.reset(),
        }
    }

    fn reset_after_error(&mut self) {
        #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
        if let Forward::GlmExl3(model) = &mut self.forward {
            // Admission can fail before dispatch reaches the backend. Clear
            // snapshot ownership as well as draining/resetting native state.
            let _ = model.cancel();
            return;
        }
        self.reset();
    }

    /// Expensive aggregate-only native-state digest for explicit artifact
    /// qualification. This API is absent from ordinary engine builds.
    #[cfg(all(feature = "glm-exl3", stark_glm_exl3_prefix_validation, target_os = "macos", target_arch = "aarch64"))]
    pub fn validation_state_sha256(&mut self) -> Result<[u8; 32]> {
        match &mut self.forward {
            Forward::GlmExl3(model) => model.validation_state_sha256(),
            _ => Err(Error::Unsupported("GLM EXL3 state validation unavailable")),
        }
    }

    pub fn step(
        &mut self,
        token: u32,
    ) -> Result<Vec<f32>> {
        let result = if token as usize >= self.vocab_size {
            Err(Error::Invalid("token ID"))
        } else {
            match &mut self.forward {
                Forward::Flint(model) => model.step(token),
                Forward::Glm(model) => model.step(token),
                #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
                Forward::GlmExl3(model) => model.step(token),
                Forward::Qwen(model) => model.step(token),
                #[cfg(all(feature = "metal", target_os = "macos"))]
                Forward::Lily(model) => model.step(token),
            }
            .and_then(|logits| {
                if logits.len() != self.vocab_size || logits.iter().any(|x| !x.is_finite()) {
                    Err(Error::Invalid("nonfinite or invalid readout"))
                } else {
                    Ok(logits)
                }
            })
        };
        if result.is_err() {
            self.reset_after_error();
        }
        result
    }

    /// Advance a prefill prefix without projecting vocabulary logits that the
    /// caller will discard. The final input token must still use `step`.
    pub fn ingest(
        &mut self,
        token: u32,
    ) -> Result<()> {
        let result = if token as usize >= self.vocab_size {
            Err(Error::Invalid("token ID"))
        } else {
            match &mut self.forward {
                Forward::Flint(model) => model.ingest(token),
                Forward::Glm(model) => model.ingest(token),
                #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
                Forward::GlmExl3(model) => model.ingest(token),
                Forward::Qwen(model) => model.ingest(token),
                #[cfg(all(feature = "metal", target_os = "macos"))]
                Forward::Lily(model) => model.ingest(token),
            }
        };
        if result.is_err() {
            self.reset_after_error();
        }
        result
    }

    /// Ingest a known prefix, leaving its final readout token for `step`.
    /// Returns only after every submitted state writer is checked and fenced.
    /// The control callback may run more than once per token; an error stops
    /// submission, drains in-flight work, and resets the model before return.
    pub fn ingest_prefix(
        &mut self,
        tokens: &[u32],
        mut before_token: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
        if let Forward::GlmExl3(model) = &mut self.forward {
            let result = model.ingest_prefix(tokens, &mut before_token);
            if result.is_err() {
                self.reset_after_error();
            }
            return result;
        }
        #[cfg(feature = "metal")]
        if let Forward::Glm(model) = &mut self.forward {
            let result = model.ingest_prefix(tokens, &mut before_token);
            if result.is_err() {
                self.reset_after_error();
            }
            return result;
        }
        for &token in tokens {
            if let Err(error) = before_token().and_then(|_| self.ingest(token)) {
                self.reset_after_error();
                return Err(error);
            }
        }
        Ok(())
    }

    /// Native greedy continuation is available for selected Metal backends.
    pub fn supports_greedy_readout(&self) -> bool {
        #[cfg(feature = "metal")]
        if let Forward::Glm(model) = &self.forward {
            return model.supports_greedy_readout();
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if matches!(self.forward, Forward::Lily(_)) {
            return true;
        }
        false
    }

    /// Opt-in aggregate GPU stage timestamps for the BF16 backend. Profiling
    /// perturbs execution and is never enabled for production rate comparisons.
    pub fn set_bf16_stage_profiling(
        &mut self,
        enabled: bool,
    ) -> Result<()> {
        match &mut self.forward {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Forward::Lily(model) => model.set_stage_profiling(enabled),
            _ => {
                let _ = enabled;
                Err(Error::Unsupported("BF16 stage profiling unavailable"))
            },
        }
    }

    /// The mixed-precision backend can overlap encoding around a host sampler.
    pub fn supports_sampled_decode(&self) -> bool {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if matches!(self.forward, Forward::Lily(_)) {
            return true;
        }
        false
    }

    pub fn supports_mtp(&self) -> bool {
        #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
        if let Forward::GlmExl3(model) = &self.forward {
            return model.mtp_capabilities().is_some();
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Forward::Lily(model) = &self.forward {
            return model.supports_mtp();
        }
        false
    }

    pub fn mtp_capabilities(&self) -> Option<MtpCapabilities> {
        #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
        if let Forward::GlmExl3(model) = &self.forward {
            return model.mtp_capabilities();
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Forward::Lily(model) = &self.forward {
            return model.mtp_capabilities();
        }
        None
    }

    pub fn mtp_verification(&self) -> &'static str {
        #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
        if let Forward::GlmExl3(model) = &self.forward {
            return model.mtp_verification();
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Forward::Lily(model) = &self.forward {
            return model.mtp_verification();
        }
        "unavailable"
    }

    pub fn set_mtp_enabled(
        &mut self,
        enabled: bool,
    ) -> Result<()> {
        match &mut self.forward {
            #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
            Forward::GlmExl3(model) => model.set_mtp_enabled(enabled),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Forward::Lily(model) => model.set_mtp_enabled(enabled),
            _ if !enabled => Ok(()),
            _ => Err(Error::Unsupported("MTP unavailable")),
        }
    }

    pub fn mtp_stats(&self) -> MtpStats {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Forward::Lily(model) = &self.forward {
            return model.mtp_stats();
        }
        MtpStats::default()
    }

    /// Complete a pending GLM prompt and draw its first token without a host
    /// vocabulary copy. The explicit native stage has its own sampler policy.
    pub fn step_native_initial(
        &mut self,
        token: u32,
        sampling: NativeSampling,
    ) -> Result<u32> {
        match &mut self.forward {
            #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
            Forward::GlmExl3(model) => model.step_native_initial(token, sampling),
            _ => {
                let _ = (token, sampling);
                Err(Error::Unsupported("native target draw unavailable"))
            },
        }
    }

    /// Tail after the already-published first token. `total_budget` counts
    /// committed native tokens including that first token, not visible tail IDs.
    pub fn native_decode(
        &mut self,
        first: u32,
        total_budget: usize,
        drafts: usize,
    ) -> Result<Box<dyn Iterator<Item = Result<u32>> + '_>> {
        match &mut self.forward {
            #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
            Forward::GlmExl3(model) => Ok(Box::new(model.native_decode(first, total_budget, drafts)?)),
            _ => {
                let _ = (first, total_budget, drafts);
                Err(Error::Unsupported("native target stream unavailable"))
            },
        }
    }

    /// Idempotent post-iterator cleanup acknowledgement. Early Drop drains
    /// without sampling; a deferred end failure is returned here and fences
    /// the model, so a successful collector cannot hide cleanup failure.
    pub fn finish_native_request(&mut self) -> Result<()> {
        match &mut self.forward {
            #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
            Forward::GlmExl3(model) => model.finish_native_request(),
            _ => Err(Error::Unsupported("native target stream unavailable")),
        }
    }

    /// Cancellation or a host callback/transport failure invalidates every
    /// request snapshot, even if the iterator already drained successfully.
    pub fn abort_native_request(&mut self) -> Result<()> {
        match &mut self.forward {
            #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
            Forward::GlmExl3(model) => model.cancel(),
            _ => Err(Error::Unsupported("native target stream unavailable")),
        }
    }

    pub fn native_decode_stats(&self) -> NativeDecodeStats {
        #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
        if let Forward::GlmExl3(model) = &self.forward {
            return model.native_decode_stats();
        }
        NativeDecodeStats::default()
    }

    pub fn mtp_decode<'a>(
        &'a mut self,
        first: u32,
        remaining: usize,
        drafts: usize,
        sampler: &'a mut dyn MtpSampler,
    ) -> Result<Box<dyn Iterator<Item = Result<u32>> + 'a>> {
        match &mut self.forward {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Forward::Lily(model) => Ok(Box::new(model.mtp_decode(first, remaining, drafts, sampler)?)),
            _ => {
                let _ = (first, remaining, drafts, sampler);
                Err(Error::Unsupported("MTP unavailable"))
            },
        }
    }

    /// Continue a validated full-logit readout using the caller's sampler.
    /// The callback runs only for requested draws; cleanup never samples.
    pub fn sampled_decode<'a>(
        &'a mut self,
        first: u32,
        remaining: usize,
        sample: impl FnMut(&[f32]) -> Result<u32> + 'a,
    ) -> Result<Box<dyn Iterator<Item = Result<u32>> + 'a>> {
        match &mut self.forward {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Forward::Lily(model) => Ok(Box::new(model.sampled_decode(first, remaining, sample)?)),
            _ => {
                let _ = (first, remaining, sample);
                Err(Error::Unsupported("native sampled stream unavailable"))
            },
        }
    }

    pub fn step_greedy(
        &mut self,
        token: u32,
    ) -> Result<u32> {
        let result = if token as usize >= self.vocab_size {
            Err(Error::Invalid("token ID"))
        } else {
            match &mut self.forward {
                #[cfg(feature = "metal")]
                Forward::Glm(model) => model.step_greedy(token),
                #[cfg(all(feature = "metal", target_os = "macos"))]
                Forward::Lily(model) => model.step_greedy(token),
                _ => Err(Error::Unsupported("native greedy readout unavailable")),
            }
        };
        if result.is_err() {
            self.reset_after_error();
        }
        result
    }
    /// Continue an already validated greedy draw within one serialized request.
    /// Dropping the iterator drains submitted work before the model is reusable.
    pub fn greedy_decode(
        &mut self,
        first: u32,
        remaining: usize,
    ) -> Result<Box<dyn Iterator<Item = Result<u32>> + '_>> {
        match &mut self.forward {
            #[cfg(feature = "metal")]
            Forward::Glm(model) => Ok(Box::new(model.greedy_decode(first, remaining)?)),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Forward::Lily(model) => Ok(Box::new(model.greedy_decode(first, remaining)?)),
            _ => {
                let _ = (first, remaining);
                Err(Error::Unsupported("native greedy stream unavailable"))
            },
        }
    }

    /// Owner-thread maintenance while the generation worker waits for input.
    /// Other backends retain their existing idle behavior.
    pub fn idle_maintenance_interval(&self) -> Option<std::time::Duration> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Forward::Lily(model) = &self.forward
            && model.needs_idle_queue_maintenance()
        {
            return Some(std::time::Duration::from_millis(500));
        }
        None
    }

    /// Submit only an empty queue signal; caches, sampling and model positions
    /// survive unchanged. Failures require the owning worker to exit/reload.
    pub fn maintain_idle_queue(&self) -> Result<()> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Forward::Lily(model) = &self.forward
            && model.needs_idle_queue_maintenance()
        {
            return model.maintain_idle_queue();
        }
        Ok(())
    }

    /// The prefill chunk length when this backend can snapshot and resume a
    /// prompt prefix, which then happens only at multiples of it.
    pub fn prefix_chunk_tokens(&self) -> Option<usize> {
        #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
        if let Forward::GlmExl3(model) = &self.forward {
            return model.prefix_chunk_tokens();
        }
        if let Forward::Glm(model) = &self.forward {
            return Some(model.prefill_chunk_tokens());
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Forward::Lily(model) = &self.forward {
            return Some(model.prefill_chunk_tokens());
        }
        None
    }

    /// Capture the recurrent state when the next prefill reaches `position`.
    /// Call after `reset` (and any `restore_prefix`), before the prompt.
    pub fn capture_prefix_at(
        &mut self,
        position: usize,
    ) -> Result<()> {
        match &mut self.forward {
            Forward::Glm(model) => model.capture_prefix_at(position),
            #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
            Forward::GlmExl3(model) => model.capture_prefix_at(position),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Forward::Lily(model) => model.capture_prefix_at(position),
            _ => {
                let _ = position;
                Err(Error::Unsupported("prefix snapshots unavailable"))
            },
        }
    }

    /// The snapshot captured by this request's prefill. Take it before the
    /// next `reset`.
    pub fn take_prefix_snapshot(&mut self) -> Option<PrefixSnapshot> {
        #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
        if let Forward::GlmExl3(model) = &mut self.forward {
            return model.take_prefix_snapshot().map(|snapshot| PrefixSnapshot {
                inner: PrefixState::GlmExl3(snapshot),
            });
        }
        if let Forward::Glm(model) = &mut self.forward {
            return model.take_prefix_snapshot().map(|snapshot| PrefixSnapshot {
                inner: PrefixState::Glm(snapshot),
            });
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Forward::Lily(model) = &mut self.forward {
            return model.take_prefix_snapshot().map(|snapshot| PrefixSnapshot {
                inner: PrefixState::Lily(snapshot),
            });
        }
        None
    }

    /// Resume a reset request at `snapshot`'s position. Only the prompt
    /// prefix the snapshot was captured from may follow, and every request
    /// since the capture must have kept that prefix. On failure the model is
    /// reset.
    pub fn restore_prefix(
        &mut self,
        snapshot: &PrefixSnapshot,
    ) -> Result<()> {
        let result = match (&mut self.forward, &snapshot.inner) {
            (Forward::Glm(model), PrefixState::Glm(snapshot)) => model.restore_prefix(snapshot),
            #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
            (Forward::GlmExl3(model), PrefixState::GlmExl3(snapshot)) => model.restore_prefix(snapshot),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            (Forward::Lily(model), PrefixState::Lily(snapshot)) => model.restore_prefix(snapshot),
            _ => Err(Error::Unsupported("prefix snapshots unavailable")),
        };
        if result.is_err() {
            self.reset_after_error();
        }
        result
    }

    /// Includes failures found while draining an early-dropped greedy stream.
    pub fn is_fenced(&self) -> bool {
        #[cfg(all(feature = "glm-exl3", target_os = "macos", target_arch = "aarch64"))]
        if let Forward::GlmExl3(model) = &self.forward {
            return model.is_fenced();
        }
        #[cfg(feature = "metal")]
        if let Forward::Glm(model) = &self.forward {
            return model.is_fenced();
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Forward::Lily(model) = &self.forward {
            return model.is_fenced();
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn glm_exl3_selection_requires_the_explicit_streaming_pack_contract() {
        let config = serde_json::json!({
            "model_type": "glm5_next",
            "expert_streaming": {
                "family": "mlx-stream", "format": "mtplx-expert-manifest-v2",
                "manifest": "expert-manifest-v2.json", "runtime_manifest": "expert-manifest.json",
                "quant": "exl3-k3-mul1"
            }
        });
        assert!(glm_exl3_checkpoint(&config));
        assert!(!glm_exl3_checkpoint(&serde_json::json!({"model_type": "glm5_next"})));
        for (pointer, replacement) in [
            ("/model_type", "qwen4_exp"),
            ("/expert_streaming/family", "other"),
            ("/expert_streaming/format", "other"),
            ("/expert_streaming/manifest", "../expert-manifest-v2.json"),
            ("/expert_streaming/runtime_manifest", "../expert-manifest.json"),
            ("/expert_streaming/quant", "other"),
        ] {
            let mut invalid = config.clone();
            *invalid.pointer_mut(pointer).unwrap() = Value::from(replacement);
            assert!(!glm_exl3_checkpoint(&invalid), "{pointer}");
        }
    }

    #[stark_test]
    fn mixed_precision_selection_requires_explicit_architecture_and_format() {
        assert!(mixed_precision_checkpoint(&serde_json::json!({
            "model_type": "qwen4_exp", "lily": {"format": "qwen4_exp-affine-v1"}
        })));
        assert!(!mixed_precision_checkpoint(&serde_json::json!({"model_type": "qwen4_exp"})));
        assert!(!mixed_precision_checkpoint(&serde_json::json!({
            "model_type": "flint", "lily": {"format": "qwen4_exp-affine-v1"}
        })));
    }

    #[stark_test]
    fn mixed_precision_selection_admits_exact_external_affine_contract() {
        let config = serde_json::json!({
            "model_type": "qwen4_exp",
            "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
            "ngram_table": {"file": "ngram_table.bin", "bits": 4, "group_size": 32}
        });
        assert!(mixed_precision_checkpoint(&config));
        for (pointer, replacement) in [
            ("/model_type", serde_json::json!("flint")),
            ("/quantization/mode", serde_json::json!("nvfp4")),
            ("/quantization/bits", serde_json::json!(8)),
            ("/quantization/group_size", serde_json::json!(32)),
            ("/ngram_table/file", serde_json::json!("../ngram_table.bin")),
            ("/ngram_table/bits", serde_json::json!(8)),
            ("/ngram_table/group_size", serde_json::json!(64)),
        ] {
            let mut invalid = config.clone();
            *invalid.pointer_mut(pointer).unwrap() = replacement;
            assert!(!mixed_precision_checkpoint(&invalid), "{pointer}");
        }
    }

    #[stark_test]
    fn generation_eos_supplements_primary_end_of_text_without_duplicates() {
        let mut ids = Vec::new();
        extend_eos(&serde_json::json!(248044), &mut ids, 248320).unwrap();
        extend_eos(&serde_json::json!([248046, 248044]), &mut ids, 248320).unwrap();
        assert_eq!(ids, [248044, 248046]);
        assert!(extend_eos(&serde_json::json!(-1), &mut ids, 248320).is_err());
        assert!(extend_eos(&serde_json::json!([248320]), &mut ids, 248320).is_err());
    }
}
