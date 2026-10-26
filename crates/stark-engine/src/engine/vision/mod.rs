//! Private native Qwen3.5 image execution. Public workers remain text-only.

use std::{
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
            EngineLoadLanguageModelError, LanguageModel,
            multimodal::{ImageGrid, ImageTokenIds, MultimodalPrefill, ProjectedImage},
            state::{LanguageModelCreateEmptyStateError, LanguageModelState},
            stream::{LanguageModelStream, LanguageModelStreamError, LanguageModelStreamOptions},
        },
    },
    parameters::{ParameterLoader, ParameterLoaderError},
};

#[cfg(test)]
mod parity_tests;
pub(crate) mod preprocess;
#[cfg(test)]
mod reference_tests;
#[cfg(test)]
mod tests;
pub(crate) mod tower;

use preprocess::{PreparedImage, PreprocessLimits, RgbImage, decode_image, prepare_image};
use tower::{VisionTower, VisionTowerConfig, VisionTowerError};

// Exact deterministic Stage 1 artifact. Admission is deliberately bound to this
// experimental tuple; metadata or a model name cannot enable another artifact.
const ARTIFACT_FILES: [(&str, &str); 5] = [
    ("config.json", "26583eb8a5d2a86aac80d9205eceda6133f0e63ac9ff7085364f8be65fd6139b"),
    ("model.safetensors", "7c5960e7225b3b36844fa397a85c5e20d281d732e382475062f552c0737c79d4"),
    ("tokenizer.json", "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"),
    ("preprocessor_config.json", "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"),
    ("chat_template.jinja", "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"),
];

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
pub(crate) enum NativeVisionError<B: Backend> {
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

pub(crate) struct VisionLanguageModel<B: Backend> {
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
    /// Experimental exact-artifact loader. Public model resolution and worker
    /// capability negotiation intentionally do not dispatch to this method.
    pub(crate) fn load_vision_language_model(
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
            || config.artifact_contract["runtime_supported"] != false
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
        let loader = ParameterLoader::new(weights, &*self.context)?;
        let tower = VisionTower::new(&*self.context, &official_tower_config(), &loader.tree().subtree("vision"))?;
        let language_model = self.language_model_from_config(model_path, language_config, &loader, Some(tokenizer))?;
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
    pub(crate) fn default_stream_options(&self) -> LanguageModelStreamOptions {
        self.language_model.default_stream_options()
    }

    pub(crate) fn create_empty_state(
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

    pub(crate) fn prepare_prefill(
        &mut self,
        tokens: Box<[u64]>,
        encoded_image: &[u8],
        token_start: usize,
        context_capacity: u32,
    ) -> Result<MultimodalPrefill, NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("native vision model is fenced").into());
        }
        let image = decode_image(encoded_image, &self.limits)?;
        self.prepare_rgb_prefill(tokens, &image, token_start, context_capacity)
    }

    pub(crate) fn prepare_rgb_prefill(
        &mut self,
        tokens: Box<[u64]>,
        image: &RgbImage,
        token_start: usize,
        context_capacity: u32,
    ) -> Result<MultimodalPrefill, NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("native vision model is fenced").into());
        }
        let prepared = prepare_image(image, &self.limits)?;
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
        let mut input = MultimodalPrefill::try_new(
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
        input.image.features = self.project(&prepared, count as usize)?;
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
        let mut commands = context.create_command_buffer(Some("vision"), None).map_err(NativeVisionError::Backend)?;
        let output = self.tower.encode(&prepared.patches, prepared.grid, &mut commands)?;
        let bytes = count.checked_mul(size_of::<bf16>()).ok_or(InvalidInput("native vision readback size overflow"))?;
        if (&output).size() != bytes {
            return Err(InvalidInput("native vision output shape mismatch").into());
        }
        let mut readback = context.create_buffer(bytes).map_err(NativeVisionError::Backend)?;
        commands.encode_copy(&output, &mut readback);
        commands.end_encoding().submit().wait_until_completed().map_err(NativeVisionError::Backend)?;
        let features: Box<[bf16]> = (&readback).as_slice::<bf16>().into();
        if features.iter().any(|value| !value.is_finite()) {
            return Err(InvalidInput("nonfinite native vision output").into());
        }
        self.failed = false;
        Ok(features)
    }

    /// The exclusive model borrow lasts through stream completion, covering
    /// decode/preprocessing, tower execution, language prefill and generation.
    pub(crate) fn stream_image<'a>(
        &'a mut self,
        tokens: Box<[u64]>,
        encoded_image: &[u8],
        token_start: usize,
        context_capacity: u32,
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<LanguageModelStream<'a, B>, NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("native vision model is fenced").into());
        }
        self.language_model.preflight_multimodal_stream(&tokens, state, &options)?;
        if state.context_capacity() != Some(context_capacity) {
            return Err(InvalidInput("native image context must match the allocated request state").into());
        }
        if tokens.len() >= context_capacity as usize {
            return Err(InvalidInput("native image prompt exceeds requested context").into());
        }
        let input = self.prepare_prefill(tokens, encoded_image, token_start, context_capacity)?;
        match self.language_model.stream_multimodal(&input, state, options) {
            Ok(stream) => Ok(stream.with_failure_fence(&mut self.failed)),
            Err(error) => {
                self.failed = true;
                Err(error.into())
            },
        }
    }
}
