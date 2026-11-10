//! Model-bound native Gemma 3 static-image execution.
//!
//! Original HF BF16 Gemma 3 4B/12B/27B geometry is admitted structurally and
//! bound to all five selected artifact descriptors. Image features stay private;
//! each image request owns fresh decoder state and clears generation reuse.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek},
    path::Path,
    sync::Arc,
};

use half::bf16;

use super::{
    ArtifactConfig, ComponentSignature, NativeVisionError, VisionArtifactIdentity, file_digest, open_artifact_component,
};
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
            multimodal::{ImageGrid, MultimodalPrefill, ProjectedImage},
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
use preprocess::Gemma3Patches;
use tower::Gemma3VisionTower;

const IMAGE: u64 = 262144;
const START: u64 = 255999;
const END: u64 = 256000;
const VOCAB: u32 = 262208;
const TOKENS: u32 = 256;
const PROFILE: &str = "gemma3-siglip-bf16-v1";
const TEMPLATE: &str = "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4";

/// Opaque request-owned patches, bound to the loaded model that prepared them.
pub struct PreparedGemma3Image {
    model_identity: Arc<()>,
    patches: Gemma3Patches,
}

impl PreparedGemma3Image {
    pub fn grid(&self) -> [u32; 3] {
        [1, 64, 64]
    }
    pub fn image_token_count(&self) -> u32 {
        TOKENS
    }
    pub fn processed_dimensions(&self) -> [u32; 2] {
        [896, 896]
    }
}

pub struct Gemma3VisionLanguageModel<B: Backend> {
    engine: Arc<Engine<B>>,
    language_model: LanguageModel<B>,
    tower: Gemma3VisionTower<B>,
    failed: bool,
    identity: VisionArtifactIdentity,
    codec_eos_token: Option<String>,
    source: String,
    revision: Option<String>,
    model_dim: u32,
}

fn processor() -> serde_json::Value {
    serde_json::json!({"do_convert_rgb":true,"do_normalize":true,"do_pan_and_scan":false,
        "do_rescale":true,"do_resize":true,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],
        "image_processor_type":"Gemma3ImageProcessor","image_seq_length":256,"resample":2,
        "rescale_factor":1.0/255.0,"size":{"height":896,"width":896}})
}

fn hash_string(value: &serde_json::Value) -> bool {
    value.as_str().is_some_and(|value| {
        value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn source_label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && !value.starts_with(['/', '.'])
        && value.split('/').all(|part| !part.is_empty() && part != "..")
        && value.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"_./-".contains(&byte))
}

fn validate_contract(raw: &serde_json::Value) -> Result<(u32, String, Option<String>), InvalidInput> {
    let contract = &raw["artifact_contract"];
    let model_dim = raw["decoder_config"]["transformer_config"]["model_dim"]
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or(0);
    if raw["type"] != "Gemma3VisionLanguageModelConfig"
        || contract["type"] != "Gemma3VisionLanguageArtifact"
        || contract["schema_version"] != 5
        || contract["transport"] != "mediaref-v1"
        || contract["runtime_supported"] != true
        || contract["input_modalities"] != serde_json::json!(["text", "image"])
        || contract["dtype"] != "bfloat16"
        || contract["family"] != serde_json::json!({"profile":PROFILE,"version":1})
        || contract["position_contract"] != serde_json::json!({"type":"scalar","bidirectional_image_attention":true})
        || contract["tensor_layout"] != "hf-gemma3-siglip-v1"
        || contract["qualification"] != "unverified"
        || contract["decoder_model_dim"] != model_dim
        || ![2560, 3840, 5376].contains(&model_dim)
        || raw["decoder_config"]["scalar_image_embeddings"] != true
        || raw["decoder_config"]["bidirectional_image_attention"] != true
        || raw["decoder_config"]["vocab_size"] != VOCAB
        || contract["special_token_ids"] != serde_json::json!({"image":IMAGE,"vision_start":START,"vision_end":END})
        || contract["vision_config"]
            != serde_json::json!({"model_type":"siglip_vision_model","hidden_size":1152,
            "image_size":896,"intermediate_size":4304,"num_attention_heads":16,"num_hidden_layers":27,
            "patch_size":14,"vision_use_head":false,"num_channels":3,"layer_norm_eps":1e-6,
            "hidden_act":"gelu_pytorch_tanh","attention_dropout":0.0})
        || contract["processor_config"] != processor()
    {
        return Err(InvalidInput("unsupported Gemma 3 image artifact contract"));
    }
    let expected_keys = [
        "type",
        "schema_version",
        "family",
        "input_modalities",
        "dtype",
        "source",
        "special_token_ids",
        "vision_config",
        "processor_config",
        "decoder_model_dim",
        "tensor_layout",
        "position_contract",
        "transport",
        "qualification",
        "runtime_supported",
        "component_hashes",
    ];
    if contract.as_object().is_none_or(|object| {
        object.len() != expected_keys.len() || object.keys().any(|key| !expected_keys.contains(&key.as_str()))
    }) {
        return Err(InvalidInput("Gemma 3 image contract contains unsupported fields"));
    }
    let source = &contract["source"];
    let repo = source["repo"]
        .as_str()
        .filter(|value| source_label(value))
        .ok_or(InvalidInput("invalid Gemma 3 source identity"))?;
    let revision = if source["revision"].is_null() {
        None
    } else {
        Some(
            source["revision"]
                .as_str()
                .filter(|value| {
                    (40..=64).contains(&value.len())
                        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                })
                .ok_or(InvalidInput("invalid Gemma 3 source revision"))?
                .to_owned(),
        )
    };
    if source.as_object().is_none_or(|fields| fields.len() != 4) {
        return Err(InvalidInput("invalid Gemma 3 source provenance fields"));
    }
    let hashes = source["hashes"].as_object().ok_or(InvalidInput("Gemma 3 source hashes are missing"))?;
    if !(5..=16).contains(&hashes.len())
        || hashes.iter().any(|(name, value)| !source_label(name) || !hash_string(value))
        || ["config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.jinja"]
            .iter()
            .any(|name| !hashes.contains_key(*name))
    {
        return Err(InvalidInput("invalid Gemma 3 source hashes"));
    }
    let weights = source["weight_files"].as_object().ok_or(InvalidInput("Gemma 3 source weights are missing"))?;
    if weights.is_empty()
        || weights.len() > 256
        || !weights.values().any(|value| value["role"] == "weights")
        || weights.iter().any(|(name, value)| {
            !source_label(name)
                || !hash_string(&value["sha256"])
                || value["size"].as_u64().is_none_or(|size| size == 0)
                || ![Some("weights"), Some("weight_index")].contains(&value["role"].as_str())
                || value.as_object().is_none_or(|fields| fields.len() != 3)
        })
    {
        return Err(InvalidInput("invalid Gemma 3 weight provenance"));
    }
    validate_decoder(&raw["decoder_config"], model_dim)?;
    Ok((model_dim, repo.to_owned(), revision))
}

fn validate_decoder(
    decoder: &serde_json::Value,
    model_dim: u32,
) -> Result<(), InvalidInput> {
    let (layers, hidden, heads, groups, head_dim, scalar) = match model_dim {
        2560 => (34, 10240, 8, 4, 256, 256.0f64),
        3840 => (48, 15360, 16, 8, 256, 256.0),
        5376 => (62, 21504, 32, 16, 128, 168.0),
        _ => return Err(InvalidInput("unsupported Gemma 3 decoder width")),
    };
    let transformer = &decoder["transformer_config"];
    let configs = transformer["layer_configs"].as_array().ok_or(InvalidInput("Gemma 3 decoder layers are missing"))?;
    let norm = serde_json::json!({"epsilon":1e-6,"scale_offset":1.0,"upcast_mode":"full_layer",
        "subtract_mean":false,"has_scale":true,"has_biases":false});
    let mlp = serde_json::json!({"type":"DenseMLPConfig","linear_config":{},"activation":{"type":"GELU","approximate":true},
        "has_up_biases":false,"has_down_biases":false,"gate_clipping":null,"up_clipping":null});
    if configs.len() != layers
        || transformer["hidden_dim"] != hidden
        || decoder["embedding_config"]["type"] != "TiedEmbeddingConfig"
        || decoder["embedding_config"]["input_scale"]
            .as_f64()
            .is_none_or(|scale| (scale - f64::from(model_dim).sqrt()).abs() > 1e-8)
        || !decoder["embedding_norm_config"].is_null()
        || !decoder["ple_model_config"].is_null()
        || !decoder["embedding_config"]["logit_soft_cap"].is_null()
        || !decoder["embedding_config"]["logit_scale"].is_null()
        || transformer["output_norm_config"] != norm
    {
        return Err(InvalidInput("unsupported Gemma 3 image decoder configuration"));
    }
    for (index, layer) in configs.iter().enumerate() {
        let mixer = &layer["mixer_config"];
        let rope = &layer["rope_config"];
        let local = (index + 1) % 6 != 0;
        if mixer["type"] != "AttentionConfig"
            || mixer["num_heads"] != heads
            || mixer["num_groups"] != groups
            || mixer["head_dim"] != head_dim
            || mixer["is_causal"] != true
            // Legacy same-image local masks are equivalent for this fixed
            // 256-token block within the pinned 1024-token sliding window.
            || !(mixer["image_attention_policy"].is_null()
                || mixer["image_attention_policy"] == "same_image"
                || (local && mixer["image_attention_policy"] == "windowed_same_image"))
            || mixer["scale"].as_f64().is_none_or(|scale| (scale - scalar.sqrt().recip()).abs() > 1e-12)
            || mixer["sliding_window_size"]
                != if local {
                    serde_json::json!(1024)
                } else {
                    serde_json::Value::Null
                }
            || rope["head_dim"] != head_dim
            || rope["max_sequence_length"] != 131072
            || rope["base"]
                != if local {
                    serde_json::json!(10000.0)
                } else {
                    serde_json::json!(1000000.0)
                }
            || rope["type"]
                != if local {
                    "UnscaledRoPEConfig"
                } else {
                    "LinearScalingRoPEConfig"
                }
            || (!local && rope["scaling_factor"] != 8.0)
            || mixer["query_norm_config"] != norm
            || mixer["key_norm_config"] != norm
            || !mixer["logit_soft_cap"].is_null()
            || ["has_sinks", "has_qkvg_biases", "has_out_biases", "has_gate", "normalize_values", "is_kv_sharing"]
                .iter()
                .any(|&key| mixer[key] != false)
            || ["pre_mixer_norm_config", "post_mixer_norm_config", "pre_mlp_norm_config", "post_mlp_norm_config"]
                .iter()
                .any(|&key| layer[key] != norm)
            || ["mixer_conv_config", "mlp_conv_config", "hidden_dim", "ple_config", "kv_source_layer_index"]
                .iter()
                .any(|&key| !layer[key].is_null())
            || layer["has_post_layer_scalar"] != false
            || layer["mlp_config"] != mlp
        {
            return Err(InvalidInput("Gemma 3 image decoder attention differs from its profile"));
        }
    }
    Ok(())
}

impl<B: Backend> Engine<B> {
    pub fn load_gemma3_vision_language_model(
        self: &Arc<Self>,
        path: &Path,
    ) -> Result<Gemma3VisionLanguageModel<B>, NativeVisionError<B>> {
        if self.capture_manager.is_some() || path.join("speculator").exists() {
            return Err(InvalidInput("Gemma 3 image loading requires capture-free non-speculative execution").into());
        }
        let mut selected: BTreeMap<&str, File> = BTreeMap::new();
        let mut hashes = BTreeMap::new();
        let mut signatures = BTreeMap::new();
        let mut components = BTreeMap::new();
        for (name, limit) in [
            ("config.json", 1_048_576u64),
            ("model.safetensors", 80 * 1024 * 1024 * 1024),
            ("tokenizer.json", 67_108_864),
            ("preprocessor_config.json", 65_536),
            ("chat_template.jinja", 262_144),
        ] {
            let mut file = open_artifact_component(&path.join(name))?;
            let signature = ComponentSignature::read(&file)?;
            let length = file.metadata()?.len();
            if length == 0 || length > limit {
                return Err(InvalidInput("Gemma 3 image component exceeds bounds").into());
            }
            hashes.insert(name, file_digest(&mut file)?);
            file.rewind()?;
            if name != "model.safetensors" {
                let mut bytes = Vec::new();
                Read::by_ref(&mut file).take(limit + 1).read_to_end(&mut bytes)?;
                if bytes.len() as u64 != length {
                    return Err(InvalidInput("Gemma 3 metadata changed during admission").into());
                }
                components.insert(name, bytes);
                file.rewind()?;
            }
            if !signature.unchanged(&file)? {
                return Err(InvalidInput("image component changed during admission").into());
            }
            signatures.insert(name, signature);
            selected.insert(name, file);
        }
        let raw: serde_json::Value = serde_json::from_slice(&components["config.json"])?;
        let (model_dim, source, revision) = validate_contract(&raw)?;
        let declared = raw["artifact_contract"]["component_hashes"]
            .as_object()
            .ok_or(InvalidInput("Gemma 3 component hashes missing"))?;
        if declared.len() != 4
            || ["model.safetensors", "tokenizer.json", "preprocessor_config.json", "chat_template.jinja"].iter().any(
                |name| declared.get(*name).and_then(serde_json::Value::as_str) != hashes.get(*name).map(String::as_str),
            )
        {
            return Err(InvalidInput("Gemma 3 artifact component hash mismatch").into());
        }
        let actual_processor: serde_json::Value = serde_json::from_slice(&components["preprocessor_config.json"])?;
        if actual_processor != processor()
            || hashes.get("chat_template.jinja").map(String::as_str) != Some(TEMPLATE)
            || raw["artifact_contract"]["source"]["hashes"]["chat_template.jinja"].as_str()
                != hashes.get("chat_template.jinja").map(String::as_str)
        {
            return Err(InvalidInput("Gemma 3 processing bytes differ from their contract").into());
        }
        let config: ArtifactConfig = serde_json::from_value(raw)?;
        let tokenizer = Arc::new(
            tokenizers::Tokenizer::from_bytes(&components["tokenizer.json"])
                .map_err(crate::engine::language_model::EngineLoadLanguageModelError::Tokenizer)?,
        );
        for (name, id) in [("<image_soft_token>", IMAGE), ("<start_of_image>", START), ("<end_of_image>", END)] {
            if tokenizer.token_to_id(name).map(u64::from) != Some(id) {
                return Err(InvalidInput("Gemma 3 image tokenizer special IDs mismatch").into());
            }
        }
        let codec_eos_token = match &config.token_codec_config {
            AnyTokenCodecConfig::ChatCodecConfig(codec) => codec.eos_token.clone(),
            _ => None,
        };
        let language: LanguageModelConfig = serde_json::from_value(serde_json::json!({"type":"LanguageModelConfig",
            "decoder_config":config.decoder_config,"token_codec_config":config.token_codec_config,"generation_config":config.generation_config}))?;
        let loader = ParameterLoader::new(&selected["model.safetensors"], &*self.context)?;
        let tower = Gemma3VisionTower::new(&*self.context, &loader.tree().subtree("vision"), model_dim)?;
        let language_model =
            self.language_model_from_config(path, language, &loader, Some(tokenizer), SpeculatorPolicy::Disabled)?;
        loader.tree().assert_all_tensors_validated()?;
        drop(loader);
        for (name, signature) in &signatures {
            let file = selected.get(name).ok_or(InvalidInput("missing selected image component"))?;
            if !signature.unchanged(file)? {
                return Err(InvalidInput("image artifact changed while loading").into());
            }
        }
        let hash = |name| hashes.get(name).cloned().ok_or(InvalidInput("Gemma 3 artifact identity missing"));
        Ok(Gemma3VisionLanguageModel {
            engine: self.clone(),
            language_model,
            tower,
            failed: false,
            codec_eos_token,
            source,
            revision,
            model_dim,
            identity: VisionArtifactIdentity {
                schema_version: 5,
                config_sha256: hash("config.json")?,
                artifact_sha256: hash("model.safetensors")?,
                tokenizer_sha256: hash("tokenizer.json")?,
                processor_sha256: hash("preprocessor_config.json")?,
                template_sha256: hash("chat_template.jinja")?,
            },
        })
    }
}

impl<B: Backend> Gemma3VisionLanguageModel<B> {
    pub fn artifact_identity(&self) -> &VisionArtifactIdentity {
        &self.identity
    }
    pub fn family_profile(&self) -> (&'static str, u32) {
        (PROFILE, 1)
    }
    pub fn source_identity(&self) -> (&str, Option<&str>) {
        (&self.source, self.revision.as_deref())
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

    /// Decoder-only admission; image-tower execution is separately qualified.
    pub fn image_workspace_bytes(
        &self,
        image: &PreparedGemma3Image,
    ) -> Option<usize> {
        self.validate_image(image).ok()?;
        // Each SigLIP block is submitted and waited before the next block.
        super::memory::gemma3_workspace_bytes(self.model_dim)
    }

    pub fn decoder_generation_memory_bytes(
        &self,
        input_tokens: u32,
        context_tokens: u32,
        max_new_tokens: u32,
    ) -> Option<usize> {
        self.language_model.generation_memory_bytes_for_image(input_tokens, context_tokens, max_new_tokens)
    }

    pub fn decoder_text_generation_memory_bytes(
        &self,
        input_tokens: u32,
        context_tokens: u32,
        max_new_tokens: u32,
    ) -> Option<usize> {
        self.language_model.generation_memory_bytes_for_request(input_tokens, context_tokens, max_new_tokens, false)
    }
    pub fn decoder_reused_text_memory_bytes(
        &self,
        state: &LanguageModelState<B>,
        input_tokens: u32,
        context_tokens: u32,
        max_new_tokens: u32,
        capture_prefix: bool,
        reused_input_tokens: u32,
    ) -> Option<usize> {
        self.language_model.generation_memory_bytes_reusing_text_prefix(
            state,
            input_tokens,
            context_tokens,
            max_new_tokens,
            capture_prefix,
            reused_input_tokens,
        )
    }

    pub fn reclaim_decoder_cached_memory(&self) -> Result<usize, B::Error> {
        self.language_model.reclaim_cached_memory()
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
    /// Plain-text requests restore owned ring payload and shared-KV owners.
    /// Image-prefilled and failed states remain excluded by the decoder guards.
    pub fn prefix_reuse_supported(&self) -> bool {
        !self.failed && self.language_model.text_prefix_snapshot_supported()
    }

    pub fn decoder_text_prefix_memory_bytes(
        &self,
        input_tokens: u32,
        context_tokens: u32,
        max_new_tokens: u32,
    ) -> Option<usize> {
        self.language_model.generation_memory_bytes_with_text_prefix(input_tokens, context_tokens, max_new_tokens)
    }

    pub fn prepare_text_prefix_capture(
        &mut self,
        state: &mut LanguageModelState<B>,
        position: usize,
    ) -> Result<(), NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("Gemma 3 vision model is fenced").into());
        }
        self.language_model.prepare_text_prefix_capture(state, position)?;
        Ok(())
    }

    pub fn rewind_state(
        &mut self,
        state: &mut LanguageModelState<B>,
        prefix_length: usize,
        max_context_length: Option<u32>,
        sampling_seed: u64,
    ) -> Result<(), NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("Gemma 3 vision model is fenced").into());
        }
        super::finish_text_prefix_restore(
            self.language_model.restore_text_prefix(state, prefix_length, max_context_length, sampling_seed),
            &mut self.failed,
        )
    }
    pub fn create_empty_state(
        &mut self,
        capacity: Option<u32>,
        seed: u64,
    ) -> Result<LanguageModelState<B>, NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("Gemma 3 vision model is fenced").into());
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
    ) -> Result<PreparedGemma3Image, NativeVisionError<B>> {
        self.prepare_image_controlled(encoded, || false)
    }
    pub fn prepare_image_controlled<F: FnMut() -> bool>(
        &mut self,
        encoded: &[u8],
        mut cancelled: F,
    ) -> Result<PreparedGemma3Image, NativeVisionError<B>> {
        if self.failed {
            return Err(InvalidInput("Gemma 3 vision model is fenced").into());
        }
        if cancelled() {
            return Err(NativeVisionError::cancelled());
        }
        let patches = preprocess::prepare(encoded)?;
        if cancelled() {
            return Err(NativeVisionError::cancelled());
        }
        Ok(PreparedGemma3Image {
            model_identity: self.language_model.model_identity.clone(),
            patches,
        })
    }
    fn validate_image(
        &self,
        image: &PreparedGemma3Image,
    ) -> Result<(), NativeVisionError<B>> {
        if self.failed || !Arc::ptr_eq(&image.model_identity, &self.language_model.model_identity) {
            return Err(InvalidInput("Gemma 3 prepared image belongs to another or fenced model").into());
        }
        Ok(())
    }
    fn preflight(
        &self,
        tokens: Box<[u64]>,
        image: &PreparedGemma3Image,
        start: usize,
        capacity: u32,
    ) -> Result<MultimodalPrefill, NativeVisionError<B>> {
        self.validate_image(image)?;
        Ok(MultimodalPrefill::try_new_gemma3(
            tokens,
            ProjectedImage {
                model_identity: self.language_model.model_identity.clone(),
                model_dim: self.model_dim,
                data_type: DataType::BF16,
                grid: ImageGrid {
                    temporal: 1,
                    height: 64,
                    width: 64,
                    merge_size: 4,
                },
                token_start: start,
                features: vec![bf16::ZERO; (TOKENS * self.model_dim) as usize].into_boxed_slice(),
            },
            IMAGE,
            START,
            END,
            &self.language_model.model_identity,
            self.model_dim,
            DataType::BF16,
            VOCAB,
            capacity.min(self.max_context_length().unwrap_or(capacity)),
        )?)
    }
    pub fn prepare_prompt(
        &self,
        tokens: Box<[u64]>,
        image: &PreparedGemma3Image,
        capacity: u32,
    ) -> Result<(Box<[u64]>, usize), NativeVisionError<B>> {
        self.validate_image(image)?;
        let (expanded, start) = expand_prompt(self.tokenizer(), &tokens, capacity)?;
        let input = self.preflight(expanded, image, start, capacity)?;
        Ok((input.tokens, start))
    }
    fn project<F: FnMut() -> bool>(
        &mut self,
        image: &PreparedGemma3Image,
        cancelled: &mut F,
    ) -> Result<Box<[bf16]>, NativeVisionError<B>> {
        self.failed = true;
        let context = &self.engine.context;
        let mut hidden = context.create_buffer((4096 * 1152 * 2) as usize).map_err(NativeVisionError::backend)?;
        let mut commands =
            context.create_command_buffer(Some("Gemma 3 vision patches"), None).map_err(NativeVisionError::backend)?;
        let output = self.tower.encode_initial(&image.patches.patches, &mut commands)?;
        commands.encode_copy(&output, &mut hidden);
        commands.end_encoding().submit().wait_until_completed().map_err(NativeVisionError::backend)?;
        for index in 0..self.tower.block_count() {
            if cancelled() {
                // Every submitted patch/block command has completed. Tower
                // weights are immutable and no decoder state was touched.
                self.failed = false;
                return Err(NativeVisionError::cancelled());
            }
            let mut commands = context
                .create_command_buffer(Some("Gemma 3 vision block"), None)
                .map_err(NativeVisionError::backend)?;
            let output = self.tower.encode_block(index, &hidden, &mut commands)?;
            commands.encode_copy(&output, &mut hidden);
            commands.end_encoding().submit().wait_until_completed().map_err(NativeVisionError::backend)?;
        }
        if cancelled() {
            self.failed = false;
            return Err(NativeVisionError::cancelled());
        }
        let mut commands = context
            .create_command_buffer(Some("Gemma 3 vision projector"), None)
            .map_err(NativeVisionError::backend)?;
        let output = self.tower.encode_finish(&hidden, &mut commands)?;
        let count = (TOKENS * self.model_dim) as usize;
        if (&output).size() != count * size_of::<bf16>() {
            return Err(InvalidInput("Gemma 3 projected image shape mismatch").into());
        }
        let mut readback = context.create_buffer(count * size_of::<bf16>()).map_err(NativeVisionError::backend)?;
        commands.encode_copy(&output, &mut readback);
        commands.end_encoding().submit().wait_until_completed().map_err(NativeVisionError::backend)?;
        let features: Box<[bf16]> = (&readback).as_slice::<bf16>().into();
        if features.iter().any(|value| !value.is_finite()) {
            return Err(InvalidInput("nonfinite Gemma 3 projected image").into());
        }
        self.failed = false;
        Ok(features)
    }
    pub fn stream_prepared_image<'a>(
        &'a mut self,
        tokens: Box<[u64]>,
        image: PreparedGemma3Image,
        start: usize,
        capacity: u32,
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
    ) -> Result<LanguageModelStream<'a, B>, NativeVisionError<B>> {
        self.stream_prepared_image_controlled(tokens, image, start, capacity, state, options, || false)
    }
    pub fn stream_prepared_image_controlled<'a, F: FnMut() -> bool>(
        &'a mut self,
        tokens: Box<[u64]>,
        image: PreparedGemma3Image,
        start: usize,
        capacity: u32,
        state: &'a mut LanguageModelState<B>,
        options: LanguageModelStreamOptions,
        mut cancelled: F,
    ) -> Result<LanguageModelStream<'a, B>, NativeVisionError<B>> {
        self.validate_image(&image)?;
        self.language_model.preflight_multimodal_stream(&tokens, state, &options)?;
        if state.context_capacity() != Some(capacity) {
            return Err(InvalidInput("Gemma 3 request context differs from allocated state").into());
        }
        let mut input = self.preflight(tokens, &image, start, capacity)?;
        if cancelled() {
            return Err(NativeVisionError::cancelled());
        }
        input.image.features = self.project(&image, &mut cancelled)?;
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
            return Err(InvalidInput("Gemma 3 vision model is fenced").into());
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
}

fn expand_prompt(
    tokenizer: &tokenizers::Tokenizer,
    tokens: &[u64],
    capacity: u32,
) -> Result<(Box<[u64]>, usize), InvalidInput> {
    if tokens.is_empty() || tokens.len() >= capacity as usize {
        return Err(InvalidInput("Gemma 3 raw prompt exceeds requested context"));
    }
    // The reviewed source template renders a single BOI marker. The HF
    // processor replaces that marker with BOI + 256 soft-image slots + EOI.
    if tokens.iter().filter(|&&token| token == START).count() != 1 || tokens.contains(&IMAGE) || tokens.contains(&END) {
        return Err(InvalidInput("Gemma 3 prompt contains surplus or already-expanded image markers"));
    }
    let ids: Vec<u32> = tokens
        .iter()
        .map(|&token| u32::try_from(token))
        .collect::<Result<_, _>>()
        .map_err(|_| InvalidInput("Gemma 3 prompt token exceeds tokenizer range"))?;
    let rendered = tokenizer.decode(&ids, false).map_err(|_| InvalidInput("Gemma 3 prompt decoding failed"))?;
    let roundtrip =
        tokenizer.encode(rendered.as_str(), false).map_err(|_| InvalidInput("Gemma 3 prompt retokenization failed"))?;
    if roundtrip.get_ids() != ids.as_slice() {
        return Err(InvalidInput("Gemma 3 prompt cannot preserve source processing"));
    }
    let full_image_sequence =
        format!("\n\n<start_of_image>{}<end_of_image>\n\n", "<image_soft_token>".repeat(TOKENS as usize));
    let expanded = rendered.replace("<start_of_image>", &full_image_sequence);
    let expanded = tokenizer
        .encode(expanded.as_str(), false)
        .map_err(|_| InvalidInput("Gemma 3 image prompt tokenization failed"))?;
    if expanded.len() >= capacity as usize {
        return Err(InvalidInput("Gemma 3 image prompt exceeds requested context"));
    }
    let expanded: Box<[u64]> = expanded.get_ids().iter().copied().map(u64::from).collect();
    let start = expanded
        .iter()
        .position(|&token| token == IMAGE)
        .ok_or(InvalidInput("Gemma 3 processor produced no image slots"))?;
    if expanded.iter().filter(|&&token| token == IMAGE).count() != TOKENS as usize {
        return Err(InvalidInput("Gemma 3 processor image span differs from its profile"));
    }
    Ok((expanded, start))
}

#[cfg(test)]
mod tests {
    use stark_engine_macros::stark_test;

    use super::*;

    #[stark_test]
    fn gemma3_processor_rejects_surplus_and_expanded_markers_before_decoding() {
        let tokenizer = tokenizers::Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default());
        for malformed in [vec![2, 17], vec![START, START], vec![START, IMAGE], vec![START, END]] {
            assert!(expand_prompt(&tokenizer, &malformed, 1024).is_err());
        }
    }
}
