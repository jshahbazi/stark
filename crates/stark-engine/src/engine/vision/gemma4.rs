//! Native, model-bound Gemma 4 E2B/E4B image execution. Static image decoding,
//! preprocessing, vision inference and text generation do not invoke Python.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek},
    path::Path,
    sync::Arc,
};

use half::bf16;

use super::{ArtifactConfig, NativeVisionError, VisionArtifactIdentity, file_digest, open_artifact_component};
use crate::{
    backends::common::{
        Backend, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, InvalidInput,
    },
    config::{model::language_model::LanguageModelConfig, token_codec::AnyTokenCodecConfig},
    data_type::DataType,
    engine::{
        Engine,
        language_model::{
            LanguageModel, SpeculatorPolicy,
            multimodal::{ImageGrid, ImageTokenIds, MultimodalPrefill, ProjectedImage},
            state::LanguageModelState,
            stream::{LanguageModelStream, LanguageModelStreamOptions},
        },
    },
    parameters::ParameterLoader,
};

#[cfg(all(test, backend = "cpu"))]
mod nople_diagnostics;
mod preprocess;
#[cfg(all(test, backend = "metal"))]
mod reference_tests;
mod tower;
mod unified;

use preprocess::GemmaPatches;
use tower::Gemma4VisionTower;

const REPO: &str = "google/gemma-4-E2B-it";
const REVISION: &str = "3e22461f65e89153144f8adb70e3b8c2cc9845a7";
const TOKENIZER: &str = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f";
const PROCESSOR: &str = "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c";
const TEMPLATE: &str = "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5";
const IMAGE: u64 = 258880;
const START: u64 = 255999;
const END: u64 = 258882;
const VIDEO: u64 = 258884;
const AUDIO: u64 = 258881;
const MODEL: u32 = 1536;
const VOCAB: u32 = 262144;

#[derive(Clone, Copy)]
struct ImageProfile {
    repo: &'static str,
    revision: &'static str,
    family: &'static str,
    tokenizer: &'static str,
    template: &'static str,
    model_dim: u32,
    bidirectional: bool,
    unified: bool,
    processor: &'static str,
    weight_limit_gib: u64,
}

const PROFILES: [ImageProfile; 5] = [
    ImageProfile {
        repo: REPO,
        revision: REVISION,
        family: "gemma4-e2b-bf16-v1",
        tokenizer: TOKENIZER,
        template: TEMPLATE,
        model_dim: MODEL,
        bidirectional: false,
        unified: false,
        processor: PROCESSOR,
        weight_limit_gib: 12,
    },
    ImageProfile {
        repo: "google/gemma-4-E4B-it",
        revision: "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
        family: "gemma4-e4b-bf16-v1",
        tokenizer: "91d9cf86906f78e197021fda88214d448e5c120665d0243a8e776e046986c6bc",
        template: TEMPLATE,
        model_dim: 2560,
        bidirectional: false,
        unified: false,
        processor: PROCESSOR,
        weight_limit_gib: 20,
    },
    ImageProfile {
        repo: "google/gemma-4-31B-it",
        revision: "842da3794eaa0b77d5f08bae87a17459d91ff475",
        family: "gemma4-31b-bf16-v1",
        tokenizer: TOKENIZER,
        template: "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
        model_dim: 5376,
        bidirectional: true,
        unified: false,
        processor: PROCESSOR,
        weight_limit_gib: 72,
    },
    ImageProfile {
        repo: "google/gemma-4-26B-A4B-it",
        revision: "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
        family: "gemma4-26b-a4b-bf16-v1",
        tokenizer: TOKENIZER,
        template: "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
        model_dim: 2816,
        bidirectional: true,
        unified: false,
        processor: PROCESSOR,
        weight_limit_gib: 60,
    },
    ImageProfile {
        repo: "google/gemma-4-12B-it",
        revision: "43c9d014b231474fa7212dbb48a762566c2b255d",
        family: "gemma4-12b-bf16-v1",
        tokenizer: "9788bdb6a021b3170357cd3587b5d622b1177561ad3c5bf43d73728fd71aeacb",
        template: "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
        model_dim: 3840,
        bidirectional: true,
        unified: true,
        processor: "6b938e76555b3e9946890770e1abcd442a4718f34041a58e8139dc8ad34545c9",
        weight_limit_gib: 32,
    },
];

/// Opaque CPU patches bound to the loaded model that prepared them.
pub struct PreparedGemmaImage {
    model_identity: Arc<()>,
    patches: GemmaPatches,
}

impl PreparedGemmaImage {
    pub fn grid(&self) -> [u32; 3] {
        self.patches.grid
    }
    pub fn image_token_count(&self) -> u32 {
        self.patches.grid[1] * self.patches.grid[2] / 9
    }
    pub fn processed_dimensions(&self) -> [u32; 2] {
        [self.patches.width, self.patches.height]
    }
}

pub struct Gemma4VisionLanguageModel<B: Backend> {
    engine: Arc<Engine<B>>,
    language_model: LanguageModel<B>,
    tower: ImageTower<B>,
    failed: bool,
    identity: VisionArtifactIdentity,
    codec_eos_token: Option<String>,
    profile: ImageProfile,
}

enum ImageTower<B: Backend> {
    Encoder(Gemma4VisionTower<B>),
    Unified(unified::UnifiedVisionEmbedder<B>),
}

impl<B: Backend> ImageTower<B> {
    fn encode(
        &self,
        patches: &[bf16],
        grid: [u32; 3],
        commands: &mut <B::CommandBuffer as crate::backends::common::CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, super::tower::VisionTowerError<B>> {
        match self {
            Self::Encoder(tower) => tower.encode(patches, grid, commands),
            Self::Unified(embedder) => embedder.encode(patches, grid, commands),
        }
    }
}

fn validate_contract(raw: &serde_json::Value) -> Result<ImageProfile, InvalidInput> {
    let contract = &raw["artifact_contract"];
    let profile = PROFILES
        .iter()
        .copied()
        .find(|profile| {
            contract["family"] == serde_json::json!({"id": profile.family, "version": 1})
                && contract["source"]["repo"] == profile.repo
                && contract["source"]["revision"] == profile.revision
                && raw["decoder_config"]["transformer_config"]["model_dim"] == profile.model_dim
        })
        .ok_or(InvalidInput("unsupported Gemma image source and decoder profile"))?;
    if raw["type"] != "Gemma4VisionLanguageModelConfig"
        || contract["schema_version"] != 4
        || contract["transport"] != "mediaref-v1"
        || contract["runtime_supported"] != true
        || contract["input_modalities"] != serde_json::json!(["text", "image"])
        || raw["decoder_config"]["gemma4_image_embeddings"] != !profile.bidirectional
        || (profile.bidirectional
            && (raw["decoder_config"]["scalar_image_embeddings"] != true
                || raw["decoder_config"]["bidirectional_image_attention"] != true
                || !raw["decoder_config"]["ple_model_config"].is_null()))
        || raw["decoder_config"]["vocab_size"] != VOCAB
        || contract["token_ids"]
            != serde_json::json!({"image":IMAGE,"vision_start":START,"vision_end":END,"video":VIDEO,"audio":AUDIO})
    {
        return Err(InvalidInput("unsupported Gemma image artifact contract"));
    }
    if profile.bidirectional
        && raw["decoder_config"]["transformer_config"]["layer_configs"].as_array().is_none_or(|layers| {
            layers.is_empty()
                || layers.iter().any(|layer| {
                    let mixer = &layer["mixer_config"];
                    let image_policy = if mixer["sliding_window_size"].is_null() {
                        "causal"
                    } else {
                        "windowed_same_image"
                    };
                    layer["materialize_post_layer_scalar"] != true
                        || mixer["materialize_rope_bf16"] != true
                        || mixer["image_attention_policy"] != image_policy
                })
        })
    {
        return Err(InvalidInput("Gemma no-PLE image decoder requires its BF16 residual and rotary policy"));
    }
    let vision = &contract["vision_config"];
    if profile.unified {
        if *vision
            != serde_json::json!({"model_type":"gemma4_unified_vision", "mm_embed_dim":3840,
            "mm_posemb_size":1120,"model_patch_size":48,"num_soft_tokens":280,"output_proj_dims":3840,
            "patch_size":16,"pooling_kernel_size":3,"rms_norm_eps":1e-6})
        {
            return Err(InvalidInput("unsupported unified Gemma image geometry"));
        }
        return Ok(profile);
    }
    if vision.as_object().is_none_or(|fields| fields.len() != 18) {
        return Err(InvalidInput("Gemma vision declares unsupported configuration fields"));
    }
    for (key, value) in [
        (
            "hidden_size",
            if profile.bidirectional {
                1152
            } else {
                768
            },
        ),
        (
            "num_hidden_layers",
            if profile.bidirectional {
                27
            } else {
                16
            },
        ),
        (
            "num_attention_heads",
            if profile.bidirectional {
                16
            } else {
                12
            },
        ),
        (
            "num_key_value_heads",
            if profile.bidirectional {
                16
            } else {
                12
            },
        ),
        (
            "head_dim",
            if profile.bidirectional {
                72
            } else {
                64
            },
        ),
        (
            "global_head_dim",
            if profile.bidirectional {
                72
            } else {
                64
            },
        ),
        (
            "intermediate_size",
            if profile.bidirectional {
                4304
            } else {
                3072
            },
        ),
        ("patch_size", 16),
        ("pooling_kernel_size", 3),
        ("position_embedding_size", 10240),
        ("default_output_length", 280),
    ] {
        if vision[key] != value {
            return Err(InvalidInput("unsupported Gemma vision geometry"));
        }
    }
    if vision["model_type"] != "gemma4_vision"
        || vision["rms_norm_eps"] != 1e-6
        || vision["standardize"] != profile.bidirectional
        || vision["attention_bias"] != false
        || vision["use_clipped_linears"] != !profile.bidirectional
        || vision["hidden_activation"] != "gelu_pytorch_tanh"
        || vision["rope_parameters"] != serde_json::json!({"rope_theta":100.0,"rope_type":"default"})
    {
        return Err(InvalidInput("unsupported Gemma vision operations"));
    }
    Ok(profile)
}

impl<B: Backend> Engine<B> {
    pub fn load_gemma4_vision_language_model(
        self: &Arc<Self>,
        path: &Path,
    ) -> Result<Gemma4VisionLanguageModel<B>, NativeVisionError<B>> {
        if self.capture_manager.is_some() || path.join("speculator").exists() {
            return Err(InvalidInput("Gemma image loading requires capture-free non-speculative execution").into());
        }
        let mut selected: BTreeMap<&str, File> = BTreeMap::new();
        let mut hashes = BTreeMap::new();
        let mut components = BTreeMap::new();
        for (name, limit) in [
            ("config.json", 1_048_576),
            ("model.safetensors", 72 * 1024 * 1024 * 1024u64),
            ("tokenizer.json", 67_108_864),
            ("preprocessor_config.json", 65_536),
            ("chat_template.jinja", 262_144),
        ] {
            let mut file = open_artifact_component(&path.join(name))?;
            let length = file.metadata()?.len();
            if length == 0 || length > limit {
                return Err(InvalidInput("Gemma image component exceeds bounds").into());
            }
            let hash = file_digest(&mut file)?;
            file.rewind()?;
            if name != "model.safetensors" {
                let mut bytes = Vec::new();
                Read::by_ref(&mut file).take(limit + 1).read_to_end(&mut bytes)?;
                if bytes.len() as u64 != length {
                    return Err(InvalidInput("Gemma image metadata changed during admission").into());
                }
                components.insert(name, bytes);
                file.rewind()?;
            }
            hashes.insert(name, hash);
            selected.insert(name, file);
        }
        let raw: serde_json::Value = serde_json::from_slice(&components["config.json"])?;
        let profile = validate_contract(&raw)?;
        // E2B retains its original admission cap. E4B's larger PLE decoder
        // needs its own bound; neither profile permits unbounded components.
        if selected["model.safetensors"].metadata()?.len() > profile.weight_limit_gib * 1024 * 1024 * 1024u64 {
            return Err(InvalidInput("Gemma image weights exceed the selected profile bound").into());
        }
        for (name, expected) in [
            ("tokenizer.json", profile.tokenizer),
            ("preprocessor_config.json", profile.processor),
            ("chat_template.jinja", profile.template),
        ] {
            if hashes.get(name).map(String::as_str) != Some(expected) {
                return Err(InvalidInput("Gemma image processing identity mismatch").into());
            }
        }
        let declared = raw["artifact_contract"]["component_hashes"]
            .as_object()
            .ok_or(InvalidInput("Gemma artifact component hashes are missing"))?;
        if declared.len() != 4
            || ["model.safetensors", "tokenizer.json", "preprocessor_config.json", "chat_template.jinja"].iter().any(
                |name| declared.get(*name).and_then(serde_json::Value::as_str) != hashes.get(*name).map(String::as_str),
            )
        {
            return Err(InvalidInput("Gemma artifact component hash mismatch").into());
        }
        let processor: serde_json::Value = serde_json::from_slice(&components["preprocessor_config.json"])?;
        if raw["artifact_contract"]["processor"] != processor["image_processor"] {
            return Err(InvalidInput("Gemma artifact processing contract differs from its selected bytes").into());
        }
        let config: ArtifactConfig = serde_json::from_value(raw)?;
        let tokenizer = Arc::new(
            tokenizers::Tokenizer::from_bytes(&components["tokenizer.json"])
                .map_err(crate::engine::language_model::EngineLoadLanguageModelError::Tokenizer)?,
        );
        for (name, id) in [("<|image|>", IMAGE), ("<|image>", START), ("<image|>", END)] {
            if tokenizer.token_to_id(name).map(u64::from) != Some(id) {
                return Err(InvalidInput("Gemma image tokenizer special IDs mismatch").into());
            }
        }
        let codec_eos_token = match &config.token_codec_config {
            AnyTokenCodecConfig::ChatCodecConfig(codec) => codec.eos_token.clone(),
            _ => None,
        };
        let language: LanguageModelConfig = serde_json::from_value(serde_json::json!({"type":"LanguageModelConfig",
            "decoder_config":config.decoder_config,"token_codec_config":config.token_codec_config,"generation_config":config.generation_config}))?;
        let loader = ParameterLoader::new(&selected["model.safetensors"], &*self.context)?;
        let tower = if profile.unified {
            ImageTower::Unified(unified::UnifiedVisionEmbedder::new(&*self.context, &loader.tree().subtree("vision"))?)
        } else {
            ImageTower::Encoder(Gemma4VisionTower::new(
                &*self.context,
                &loader.tree().subtree("vision"),
                profile.model_dim,
            )?)
        };
        let language_model =
            self.language_model_from_config(path, language, &loader, Some(tokenizer), SpeculatorPolicy::Disabled)?;
        loader.tree().assert_all_tensors_validated()?;
        drop(loader);
        for (name, expected) in &hashes {
            let file = selected.get_mut(name).ok_or(InvalidInput("missing Gemma selected component"))?;
            file.rewind()?;
            if file_digest(file)? != *expected {
                return Err(InvalidInput("Gemma artifact changed while loading").into());
            }
        }
        let hash = |name| hashes.get(name).cloned().ok_or(InvalidInput("missing Gemma artifact identity"));
        Ok(Gemma4VisionLanguageModel {
            engine: self.clone(),
            language_model,
            tower,
            failed: false,
            codec_eos_token,
            profile,
            identity: VisionArtifactIdentity {
                schema_version: 4,
                config_sha256: hash("config.json")?,
                artifact_sha256: hash("model.safetensors")?,
                tokenizer_sha256: hash("tokenizer.json")?,
                processor_sha256: hash("preprocessor_config.json")?,
                template_sha256: hash("chat_template.jinja")?,
            },
        })
    }
}

impl<B: Backend> Gemma4VisionLanguageModel<B> {
    pub fn artifact_identity(&self) -> &VisionArtifactIdentity {
        &self.identity
    }
    pub fn family_profile(&self) -> (&'static str, u32) {
        (self.profile.family, 1)
    }
    pub fn source_identity(&self) -> (&str, Option<&str>) {
        (self.profile.repo, Some(self.profile.revision))
    }
    pub fn vocab_size(&self) -> u32 {
        VOCAB
    }
    pub fn codec_eos_token(&self) -> Option<&str> {
        self.codec_eos_token.as_deref()
    }
    pub fn tokenizer(&self) -> &Arc<tokenizers::Tokenizer> {
        self.language_model.tokenizer()
    }
    pub fn generation_config(&self) -> &crate::config::model::generation::GenerationConfig {
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
        capacity: Option<u32>,
        seed: u64,
    ) -> Result<LanguageModelState<B>, NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("Gemma vision model is fenced").into());
        }
        match self.language_model.create_empty_state(capacity, seed) {
            Ok(state) => Ok(state),
            Err(error) => {
                if matches!(
                    &error,
                    crate::engine::language_model::state::LanguageModelCreateEmptyStateError::Backend(_)
                ) {
                    self.failed = true;
                }
                Err(error.into())
            },
        }
    }
    pub fn prepare_image(
        &mut self,
        encoded: &[u8],
    ) -> Result<PreparedGemmaImage, NativeVisionError<B>> {
        self.prepare_image_controlled(encoded, || false)
    }
    pub fn prepare_image_controlled<F: FnMut() -> bool>(
        &mut self,
        encoded: &[u8],
        mut cancelled: F,
    ) -> Result<PreparedGemmaImage, NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("Gemma vision model is fenced").into());
        }
        if cancelled() {
            return Err(NativeVisionError::cancelled());
        }
        let patches = if self.profile.unified {
            preprocess::prepare_unified(encoded)?
        } else {
            preprocess::prepare(encoded)?
        };
        if cancelled() {
            return Err(NativeVisionError::cancelled());
        }
        Ok(PreparedGemmaImage {
            model_identity: self.language_model.model_identity.clone(),
            patches,
        })
    }
    fn validate_image(
        &self,
        image: &PreparedGemmaImage,
    ) -> Result<(), NativeVisionError<B>> {
        if self.failed || !Arc::ptr_eq(&image.model_identity, &self.language_model.model_identity) {
            return Err(InvalidInput("Gemma prepared image belongs to another or fenced model").into());
        }
        Ok(())
    }
    fn preflight(
        &self,
        tokens: Box<[u64]>,
        image: &PreparedGemmaImage,
        start: usize,
        capacity: u32,
    ) -> Result<MultimodalPrefill, NativeVisionError<B>> {
        self.validate_image(image)?;
        if tokens.iter().any(|token| matches!(*token, AUDIO | VIDEO)) {
            return Err(InvalidInput("audio and video are outside Gemma image execution").into());
        }
        let projected = ProjectedImage {
            model_identity: self.language_model.model_identity.clone(),
            model_dim: self.profile.model_dim,
            data_type: DataType::BF16,
            grid: ImageGrid {
                temporal: 1,
                height: image.patches.grid[1],
                width: image.patches.grid[2],
                merge_size: 3,
            },
            token_start: start,
            features: vec![bf16::ZERO; (image.image_token_count() * self.profile.model_dim) as usize]
                .into_boxed_slice(),
        };
        let limit = capacity.min(self.max_context_length().unwrap_or(capacity));
        let input = if self.profile.bidirectional {
            MultimodalPrefill::try_new_scalar_bidirectional(
                tokens,
                projected,
                IMAGE,
                START,
                END,
                &self.language_model.model_identity,
                self.profile.model_dim,
                DataType::BF16,
                VOCAB,
                limit,
            )
        } else {
            MultimodalPrefill::try_new_gemma4(
                tokens,
                projected,
                ImageTokenIds {
                    image: IMAGE,
                    vision_start: START,
                    vision_end: END,
                    video: VIDEO,
                },
                AUDIO,
                &self.language_model.model_identity,
                self.profile.model_dim,
                DataType::BF16,
                VOCAB,
                limit,
            )
        };
        Ok(input?)
    }
    pub fn prepare_prompt(
        &self,
        tokens: Box<[u64]>,
        image: &PreparedGemmaImage,
        capacity: u32,
    ) -> Result<(Box<[u64]>, usize), NativeVisionError<B>> {
        self.validate_image(image)?;
        let index = tokens
            .iter()
            .position(|&token| token == IMAGE)
            .ok_or(InvalidInput("Gemma prompt requires one image placeholder"))?;
        if tokens[index + 1..].contains(&IMAGE) {
            return Err(InvalidInput("Gemma prompt requires exactly one image placeholder").into());
        }
        let wrapped = index > 0 && tokens[index - 1] == START && tokens.get(index + 1) == Some(&END);
        let count = image.image_token_count() as usize;
        let extra = count.saturating_sub(1)
            + if wrapped {
                0
            } else {
                2
            };
        if count == 0 || count > 280 || tokens.len().checked_add(extra).is_none_or(|n| n >= capacity as usize) {
            return Err(InvalidInput("Gemma image prompt exceeds requested context").into());
        }
        let mut expanded = Vec::with_capacity(tokens.len() + count + 2);
        expanded.extend_from_slice(&tokens[..index]);
        if !wrapped {
            expanded.push(START);
        }
        let start = expanded.len();
        expanded.extend(std::iter::repeat_n(IMAGE, count));
        if !wrapped {
            expanded.push(END);
        }
        expanded.extend_from_slice(&tokens[index + 1..]);
        let input = self.preflight(expanded.into_boxed_slice(), image, start, capacity)?;
        Ok((input.tokens, start))
    }
    fn project(
        &mut self,
        image: &PreparedGemmaImage,
    ) -> Result<Box<[bf16]>, NativeVisionError<B>> {
        self.failed = true;
        let context = &self.engine.context;
        let mut commands =
            context.create_command_buffer(Some("Gemma vision"), None).map_err(NativeVisionError::backend)?;
        let output = self.tower.encode(&image.patches.patches, image.patches.grid, &mut commands)?;
        let count = (image.image_token_count() * self.profile.model_dim) as usize;
        let bytes = count * size_of::<bf16>();
        if (&output).size() != bytes {
            return Err(InvalidInput("Gemma vision output shape mismatch").into());
        }
        let mut readback = context.create_buffer(bytes).map_err(NativeVisionError::backend)?;
        commands.encode_copy(&output, &mut readback);
        commands.end_encoding().submit().wait_until_completed().map_err(NativeVisionError::backend)?;
        let features: Box<[bf16]> = (&readback).as_slice::<bf16>().into();
        if features.iter().any(|value| !value.is_finite()) {
            return Err(InvalidInput("nonfinite Gemma projected image").into());
        }
        self.failed = false;
        Ok(features)
    }
    pub fn stream_prepared_image_controlled<'a, F: FnMut() -> bool>(
        &'a mut self,
        tokens: Box<[u64]>,
        image: PreparedGemmaImage,
        start: usize,
        capacity: u32,
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
        mut cancelled: F,
    ) -> Result<LanguageModelStream<'a, B>, NativeVisionError<B>> {
        self.validate_image(&image)?;
        self.language_model.preflight_multimodal_stream(&tokens, state, &options)?;
        if state.context_capacity() != Some(capacity) {
            return Err(InvalidInput("Gemma context differs from allocated request state").into());
        }
        let mut input = self.preflight(tokens, &image, start, capacity)?;
        if cancelled() {
            return Err(NativeVisionError::cancelled());
        }
        input.image.features = self.project(&image)?;
        if cancelled() {
            return Err(NativeVisionError::cancelled());
        }
        match self.language_model.stream_multimodal(&input, state, options) {
            Ok(stream) => Ok(stream.with_failure_fence(&mut self.failed)),
            Err(error) => {
                self.failed = true;
                Err(error.into())
            },
        }
    }
    pub fn stream<'a>(
        &'a mut self,
        tokens: &[u64],
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<LanguageModelStream<'a, B>, NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("Gemma vision model is fenced").into());
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

#[cfg(test)]
mod contract_tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[cfg(any(backend = "cpu", backend = "metal"))]
    fn run_nople_canonical_fixture<B: Backend>(backend: &str)
    where
        B::ScratchBuffer: crate::backends::common::BufferCpuAccessible,
    {
        let root = std::path::PathBuf::from(
            std::env::var_os("STARK_GEMMA4_DENSE_FIXTURE_PATH").expect("set private no-PLE fixture"),
        );
        let path = std::path::PathBuf::from(
            std::env::var_os("STARK_TEST_MODEL_PATH").expect("set private no-PLE native model"),
        );
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
        assert!(
            manifest["reference_profile"] == "transformers-canonical"
                && manifest["source_sha256"] == "4f874549f79deda4bb9ce7fb7d7b8e7349122be1711f733f7a4157e46ed10cc3",
            "private reference identity changed"
        );
        let engine = Engine::<B>::new().unwrap();
        let model = engine.load_language_model(&path).unwrap_or_else(|_| panic!("private no-PLE model loading failed"));
        let mut cases = Vec::new();
        let mut passed = true;
        let compare = |actual: &[f32], reference: &[f32]| {
            assert!(actual.len() == reference.len() && !actual.is_empty());
            let n = actual.len() as f64;
            let mean = actual.iter().zip(reference).map(|(a, b)| f64::from(*a) - f64::from(*b)).sum::<f64>() / n;
            let ref_mean = reference.iter().map(|v| f64::from(*v)).sum::<f64>() / n;
            let centered = (actual
                .iter()
                .zip(reference)
                .map(|(a, b)| (f64::from(*a) - f64::from(*b) - mean).powi(2))
                .sum::<f64>()
                / n)
                .sqrt();
            let ref_rms = (reference.iter().map(|v| (f64::from(*v) - ref_mean).powi(2)).sum::<f64>() / n).sqrt();
            let max = actual
                .iter()
                .zip(reference)
                .map(|(a, b)| (f64::from(*a) - f64::from(*b) - mean).abs())
                .fold(0.0, f64::max);
            let greedy = |values: &[f32]| values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            let greedy_agreement = greedy(actual) == greedy(reference);
            let accepted = actual.iter().all(|v| v.is_finite())
                && centered <= 1.0 / 128.0 + 0.08 * ref_rms
                && max <= 0.1
                && greedy_agreement;
            (
                accepted,
                serde_json::json!({"centered_rms":centered,"reference_centered_rms":ref_rms,
                "centered_maximum_absolute_error":max,"greedy_agreement":greedy_agreement}),
            )
        };
        for (index, case) in manifest["cases"].as_array().unwrap().iter().enumerate() {
            let tokens: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
            let bytes = std::fs::read(root.join(case["logits"].as_str().unwrap())).unwrap();
            let reference =
                bytes.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
            let actual =
                model.score_next_token(&tokens, 128).unwrap_or_else(|_| panic!("private no-PLE scoring failed"));
            let (accepted, metrics) = compare(&actual.logits, &reference);
            passed &= accepted;
            cases.push(serde_json::json!({"case_index":index,"passed":accepted,"metrics":metrics}));
        }
        drop(model);
        let mut negative_rejected = None;
        if let Some(control) = std::env::var_os("STARK_GEMMA4_CONTROL_MODEL_PATH") {
            let model = engine
                .load_language_model(Path::new(&control))
                .unwrap_or_else(|_| panic!("private control loading failed"));
            let mut rejected = false;
            for case in manifest["cases"].as_array().unwrap() {
                let tokens: Vec<u32> = serde_json::from_value(case["tokens"].clone()).unwrap();
                let bytes = std::fs::read(root.join(case["logits"].as_str().unwrap())).unwrap();
                let reference =
                    bytes.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
                let actual = model.score_next_token(&tokens, 128).unwrap();
                rejected |= !compare(&actual.logits, &reference).0;
            }
            negative_rejected = Some(rejected);
            passed &= rejected;
        }
        let report = serde_json::json!({"passed":passed,"backend":backend,"scope":manifest["scope"],"cases":cases,
            "legacy_distributed_scalar_control_rejected":negative_rejected});
        std::fs::write(
            root.join(format!("native-{backend}-metrics.json")),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
        println!("{report}");
        assert!(passed, "private no-PLE canonical gates failed");
    }

    #[cfg(backend = "cpu")]
    #[stark_test]
    #[ignore]
    fn gemma_nople_canonical_cpu_fixture() {
        run_nople_canonical_fixture::<crate::backends::cpu::Cpu>("cpu");
    }

    #[cfg(backend = "metal")]
    #[stark_test]
    #[ignore]
    fn gemma_nople_canonical_metal_fixture() {
        run_nople_canonical_fixture::<crate::backends::metal::Metal>("metal");
    }

    fn config() -> serde_json::Value {
        serde_json::json!({
            "type":"Gemma4VisionLanguageModelConfig",
            "decoder_config":{"gemma4_image_embeddings":true,"vocab_size":VOCAB,
                "transformer_config":{"model_dim":MODEL}},
            "artifact_contract":{"schema_version":4,"transport":"mediaref-v1","runtime_supported":true,
                "input_modalities":["text","image"],"family":{"id":"gemma4-e2b-bf16-v1","version":1},
                "source":{"repo":REPO,"revision":REVISION},
                "token_ids":{"image":IMAGE,"vision_start":START,"vision_end":END,"video":VIDEO,"audio":AUDIO},
                "vision_config":{"model_type":"gemma4_vision","hidden_size":768,"num_hidden_layers":16,
                    "num_attention_heads":12,"num_key_value_heads":12,"head_dim":64,"global_head_dim":64,
                    "intermediate_size":3072,"patch_size":16,"pooling_kernel_size":3,
                    "position_embedding_size":10240,"default_output_length":280,"rms_norm_eps":1e-6,
                    "attention_bias":false,"standardize":false,"use_clipped_linears":true,
                    "hidden_activation":"gelu_pytorch_tanh","rope_parameters":{"rope_theta":100.0,"rope_type":"default"}}}
        })
    }

    #[stark_test]
    fn gemma_image_contract_rejects_semantic_drift_before_allocation() {
        let original = config();
        assert!(validate_contract(&original).is_ok());
        let mut changed = original.clone();
        changed["artifact_contract"]["vision_config"]["attention_dropout"] = 0.1.into();
        assert!(validate_contract(&changed).is_err());
        let mut changed = original.clone();
        changed["artifact_contract"]["vision_config"]["pooling_kernel_size"] = 2.into();
        assert!(validate_contract(&changed).is_err());
        let mut changed = original.clone();
        changed["decoder_config"]["gemma4_image_embeddings"] = false.into();
        assert!(validate_contract(&changed).is_err());
        let mut changed = original.clone();
        changed["artifact_contract"]["source"]["revision"] = "unreviewed".into();
        assert!(validate_contract(&changed).is_err());
        let mut changed = original.clone();
        changed["artifact_contract"]["token_ids"]["image"] = 1.into();
        assert!(validate_contract(&changed).is_err());
    }

    #[stark_test]
    fn gemma_e4b_contract_requires_matching_decoder_and_source() {
        let mut e4b = config();
        let profile = PROFILES[1];
        e4b["artifact_contract"]["family"]["id"] = profile.family.into();
        e4b["artifact_contract"]["source"]["repo"] = profile.repo.into();
        e4b["artifact_contract"]["source"]["revision"] = profile.revision.into();
        e4b["decoder_config"]["transformer_config"]["model_dim"] = profile.model_dim.into();
        let selected = validate_contract(&e4b).expect("matching E4B profile rejected");
        assert_eq!(selected.model_dim, 2560);
        assert_eq!(selected.family, profile.family);
        e4b["decoder_config"]["transformer_config"]["model_dim"] = MODEL.into();
        assert!(validate_contract(&e4b).is_err());
        e4b["decoder_config"]["transformer_config"]["model_dim"] = profile.model_dim.into();
        e4b["artifact_contract"]["source"]["revision"] = REVISION.into();
        assert!(validate_contract(&e4b).is_err());
    }

    #[stark_test]
    fn gemma_31b_contract_requires_standardized_tower_and_bidirectional_no_ple_decoder() {
        let mut larger = config();
        let profile = PROFILES[2];
        larger["artifact_contract"]["family"]["id"] = profile.family.into();
        larger["artifact_contract"]["source"]["repo"] = profile.repo.into();
        larger["artifact_contract"]["source"]["revision"] = profile.revision.into();
        larger["decoder_config"]["transformer_config"]["model_dim"] = profile.model_dim.into();
        larger["decoder_config"]["gemma4_image_embeddings"] = false.into();
        larger["decoder_config"]["scalar_image_embeddings"] = true.into();
        larger["decoder_config"]["bidirectional_image_attention"] = true.into();
        larger["decoder_config"]["transformer_config"]["layer_configs"] = serde_json::json!([{
            "materialize_post_layer_scalar":true,"mixer_config":{"materialize_rope_bf16":true,"image_attention_policy":"causal"}
        }]);
        for (name, value) in [
            ("hidden_size", 1152),
            ("num_hidden_layers", 27),
            ("num_attention_heads", 16),
            ("num_key_value_heads", 16),
            ("head_dim", 72),
            ("global_head_dim", 72),
            ("intermediate_size", 4304),
        ] {
            larger["artifact_contract"]["vision_config"][name] = value.into();
        }
        larger["artifact_contract"]["vision_config"]["standardize"] = true.into();
        larger["artifact_contract"]["vision_config"]["use_clipped_linears"] = false.into();
        assert!(validate_contract(&larger).is_ok());
        let mut changed = larger.clone();
        changed["decoder_config"]["bidirectional_image_attention"] = false.into();
        assert!(validate_contract(&changed).is_err());
        let mut changed = larger.clone();
        changed["decoder_config"]["ple_model_config"] = serde_json::json!({"ple_dim":256});
        assert!(validate_contract(&changed).is_err());
        larger["artifact_contract"]["vision_config"]["standardize"] = false.into();
        assert!(validate_contract(&larger).is_err());
    }

    #[stark_test]
    fn unified_contract_requires_encoder_free_geometry_and_explicit_no_ple_precision() {
        let mut raw = config();
        let profile = PROFILES[4];
        raw["artifact_contract"]["family"]["id"] = profile.family.into();
        raw["artifact_contract"]["source"] = serde_json::json!({"repo":profile.repo,"revision":profile.revision});
        raw["decoder_config"]["transformer_config"]["model_dim"] = profile.model_dim.into();
        raw["decoder_config"]["transformer_config"]["layer_configs"] = serde_json::json!([{
            "materialize_post_layer_scalar":true,"mixer_config":{"materialize_rope_bf16":true,"image_attention_policy":"causal"}
        }]);
        raw["decoder_config"]["gemma4_image_embeddings"] = false.into();
        raw["decoder_config"]["scalar_image_embeddings"] = true.into();
        raw["decoder_config"]["bidirectional_image_attention"] = true.into();
        raw["artifact_contract"]["vision_config"] = serde_json::json!({
            "model_type":"gemma4_unified_vision", "mm_embed_dim":3840,"mm_posemb_size":1120,
            "model_patch_size":48,"num_soft_tokens":280,"output_proj_dims":3840,
            "patch_size":16,"pooling_kernel_size":3,"rms_norm_eps":1e-6
        });
        assert!(validate_contract(&raw).is_ok());
        let mut changed = raw.clone();
        changed["artifact_contract"]["vision_config"]["num_hidden_layers"] = 16.into();
        assert!(validate_contract(&changed).is_err());
        let mut changed = raw.clone();
        changed["decoder_config"]["transformer_config"]["layer_configs"][0]["mixer_config"]["sliding_window_size"] =
            1024.into();
        assert!(validate_contract(&changed).is_err());
        changed["decoder_config"]["transformer_config"]["layer_configs"][0]["mixer_config"]["image_attention_policy"] =
            "windowed_same_image".into();
        assert!(validate_contract(&changed).is_ok());
        raw["decoder_config"]["transformer_config"]["layer_configs"][0]["materialize_post_layer_scalar"] = false.into();
        assert!(validate_contract(&raw).is_err());
    }
}
