//! Native, model-bound Gemma 4 E2B image execution. Static image decoding,
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

mod preprocess;
#[cfg(all(test, backend = "metal"))]
mod reference_tests;
mod tower;

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
    tower: Gemma4VisionTower<B>,
    failed: bool,
    identity: VisionArtifactIdentity,
    codec_eos_token: Option<String>,
}

fn validate_contract(raw: &serde_json::Value) -> Result<(), InvalidInput> {
    let contract = &raw["artifact_contract"];
    if raw["type"] != "Gemma4VisionLanguageModelConfig"
        || contract["schema_version"] != 4
        || contract["transport"] != "mediaref-v1"
        || contract["runtime_supported"] != true
        || contract["input_modalities"] != serde_json::json!(["text", "image"])
        || contract["family"] != serde_json::json!({"id":"gemma4-e2b-bf16-v1","version":1})
        || contract["source"]["repo"] != REPO
        || contract["source"]["revision"] != REVISION
        || raw["decoder_config"]["gemma4_image_embeddings"] != true
        || raw["decoder_config"]["vocab_size"] != VOCAB
        || raw["decoder_config"]["transformer_config"]["model_dim"] != MODEL
        || contract["token_ids"]
            != serde_json::json!({"image":IMAGE,"vision_start":START,"vision_end":END,"video":VIDEO,"audio":AUDIO})
    {
        return Err(InvalidInput("unsupported Gemma image artifact contract"));
    }
    let vision = &contract["vision_config"];
    if vision.as_object().is_none_or(|fields| fields.len() != 18) {
        return Err(InvalidInput("Gemma vision declares unsupported configuration fields"));
    }
    for (key, value) in [
        ("hidden_size", 768),
        ("num_hidden_layers", 16),
        ("num_attention_heads", 12),
        ("num_key_value_heads", 12),
        ("head_dim", 64),
        ("global_head_dim", 64),
        ("intermediate_size", 3072),
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
        || vision["standardize"] != false
        || vision["attention_bias"] != false
        || vision["use_clipped_linears"] != true
        || vision["hidden_activation"] != "gelu_pytorch_tanh"
        || vision["rope_parameters"] != serde_json::json!({"rope_theta":100.0,"rope_type":"default"})
    {
        return Err(InvalidInput("unsupported Gemma vision operations"));
    }
    Ok(())
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
            ("model.safetensors", 12 * 1024 * 1024 * 1024u64),
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
        for (name, expected) in
            [("tokenizer.json", TOKENIZER), ("preprocessor_config.json", PROCESSOR), ("chat_template.jinja", TEMPLATE)]
        {
            if hashes.get(name).map(String::as_str) != Some(expected) {
                return Err(InvalidInput("Gemma image processing identity mismatch").into());
            }
        }
        let raw: serde_json::Value = serde_json::from_slice(&components["config.json"])?;
        validate_contract(&raw)?;
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
        let tower = Gemma4VisionTower::new(&*self.context, &loader.tree().subtree("vision"))?;
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
        ("gemma4-e2b-bf16-v1", 1)
    }
    pub fn source_identity(&self) -> (&str, Option<&str>) {
        (REPO, Some(REVISION))
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
        let patches = preprocess::prepare(encoded)?;
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
        Ok(MultimodalPrefill::try_new_gemma4(
            tokens,
            ProjectedImage {
                model_identity: self.language_model.model_identity.clone(),
                model_dim: MODEL,
                data_type: DataType::BF16,
                grid: ImageGrid {
                    temporal: 1,
                    height: image.patches.grid[1],
                    width: image.patches.grid[2],
                    merge_size: 3,
                },
                token_start: start,
                features: vec![bf16::ZERO; (image.image_token_count() * MODEL) as usize].into_boxed_slice(),
            },
            ImageTokenIds {
                image: IMAGE,
                vision_start: START,
                vision_end: END,
                video: VIDEO,
            },
            AUDIO,
            &self.language_model.model_identity,
            MODEL,
            DataType::BF16,
            VOCAB,
            capacity.min(self.max_context_length().unwrap_or(capacity)),
        )?)
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
        let count = (image.image_token_count() * MODEL) as usize;
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
}
