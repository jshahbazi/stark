//! Native protocol-v4 Decider inference. Every call owns fresh sequence state.
//!
//! Implements Perplexity's Decider v1.3 full-attention/readout contract:
//! perplexity-ai/pplx-decider-flash-v1.3, revision
//! d06fb3f2d97e0a52325a0d463c450cf0338f44fe, source/src/pplx_decider/model.py
//! and protocol.py (Apache-2.0). Linear-attention layers retain their causal recurrence; full
//! attention sees the entire request. No prefix state is retained or reused.

use std::{fs::File, io::BufReader, path::Path, sync::Arc, time::Instant};

use half::bf16;
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::{
    backends::common::{
        Backend, BufferCpuAccessible, BufferRef, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending,
        Context, gpu_types::trie::TrieNode,
    },
    config::{decoder::DecoderConfig, token_mixer::AnyTokenMixerConfig, weight_matrix::AnyWeightMatrixSpec},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        decoder::{Decoder, EmbeddingReplacement},
        linear::{Linear, LinearMatmul},
    },
    engine::{
        Engine,
        language_model::multimodal::{ImageGrid, ImageTokenIds, MultimodalPrefill, ProjectedImage},
        vision::{
            preprocess::{self, PreprocessLimits},
            tower::{VisionTower, VisionTowerConfig},
        },
    },
    parameters::{NumericalProfile, ParameterLoader},
};

pub const MAX_DECIDER_INPUT_TOKENS: u32 = 4096;

/// Content-free load/request failure; never contains request data or backend dumps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeciderError(pub &'static str);

impl std::fmt::Display for DeciderError {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        formatter.write_str(self.0)
    }
}
impl std::error::Error for DeciderError {}

#[derive(Deserialize)]
struct Config {
    #[serde(rename = "type")]
    kind: String,
    decoder_config: DecoderConfig,
    vision_config: serde_json::Value,
    decision_config: serde_json::Value,
}

pub struct DeciderScores {
    pub logits: Vec<f32>,
    pub input_tokens: usize,
    pub state_creation_seconds: f64,
    pub prefill_seconds: f64,
    pub readout_seconds: f64,
}

pub struct DeciderModel<B: Backend> {
    engine: Arc<Engine<B>>,
    identity: Arc<()>,
    decoder: Decoder<B>,
    readout: LinearMatmul<B>,
    tower: Option<VisionTower<B>>,
    tokenizer: Arc<Tokenizer>,
    config: DecoderConfig,
    marker: u32,
    options: u32,
    model_dim: u32,
    limits: PreprocessLimits,
}

fn integer(
    config: &serde_json::Value,
    name: &str,
) -> Result<u32, DeciderError> {
    config[name]
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or(DeciderError("invalid_decider_config"))
}

fn tower_config(value: &serde_json::Value) -> Result<VisionTowerConfig, DeciderError> {
    let config = VisionTowerConfig {
        depth: integer(value, "depth")?,
        hidden_size: integer(value, "hidden_size")?,
        intermediate_size: integer(value, "intermediate_size")?,
        num_heads: integer(value, "num_heads")?,
        num_position_embeddings: integer(value, "num_position_embeddings")?,
        out_hidden_size: integer(value, "out_hidden_size")?,
        patch_size: integer(value, "patch_size")?,
        spatial_merge_size: integer(value, "spatial_merge_size")?,
        temporal_patch_size: integer(value, "temporal_patch_size")?,
        in_channels: integer(value, "in_channels")?,
        layer_norm_eps: 1e-6,
    };
    config.validate().map_err(|_| DeciderError("invalid_decider_vision_config"))?;
    if config.depth > 32
        || config.hidden_size > 2048
        || config.intermediate_size > 8192
        || config.patch_size != 16
        || config.spatial_merge_size != 2
        || config.temporal_patch_size != 2
    {
        return Err(DeciderError("unsupported_decider_vision_config"));
    }
    Ok(config)
}

impl<B: Backend> Engine<B> {
    pub fn load_decider_model(
        self: &Arc<Self>,
        path: &Path,
    ) -> Result<DeciderModel<B>, DeciderError> {
        let configuration = File::open(path.join("config.json")).map_err(|_| DeciderError("decider_config_io"))?;
        let config: Config = serde_json::from_reader(BufReader::new(configuration))
            .map_err(|_| DeciderError("invalid_decider_config"))?;
        if config.kind != "DeciderModelConfig" || self.capture_manager.is_some() || path.join("speculator").exists() {
            return Err(DeciderError("unsupported_decider_config"));
        }
        let marker = integer(&config.decision_config, "marker_token_id")?;
        let options = integer(&config.decision_config, "num_options")?;
        let model_dim = config.decoder_config.transformer_config.model_dim;
        let layers = &config.decoder_config.transformer_config.layer_configs;
        let mut full = 0;
        let mut recurrent = 0;
        for layer in layers {
            match &layer.mixer_config {
                AnyTokenMixerConfig::AttentionConfig(attention)
                    if !attention.is_causal
                        && attention.reference_bf16
                        && !attention.is_kv_sharing
                        && attention.sliding_window_size.is_none() =>
                {
                    full += 1
                },
                AnyTokenMixerConfig::DeltaNetConfig(_) => recurrent += 1,
                _ => return Err(DeciderError("unsupported_decider_attention")),
            }
        }
        if options != 255
            || marker >= config.decoder_config.vocab_size
            || model_dim == 0
            || model_dim > 8192
            || layers.len() > 128
            || full == 0
            || recurrent == 0
            || config.decoder_config.ple_model_config.is_some()
            || config.decoder_config.embedding_norm_config.is_some()
            || config.decoder_config.separate_prefill_last_token
        {
            return Err(DeciderError("unsupported_decider_geometry"));
        }
        let tower_config = tower_config(&config.vision_config)?;
        if tower_config.out_hidden_size != model_dim {
            return Err(DeciderError("invalid_decider_vision_width"));
        }
        let max_pixels = u64::from(integer(&config.decision_config, "max_pixels")?);
        if max_pixels != 1_638_400 {
            return Err(DeciderError("unsupported_decider_max_pixels"));
        }
        let tokenizer = Arc::new(
            Tokenizer::from_file(path.join("tokenizer.json")).map_err(|_| DeciderError("invalid_decider_tokenizer"))?,
        );
        if tokenizer.get_vocab(true).values().any(|&id| id >= config.decoder_config.vocab_size)
            || tokenizer.token_to_id("<|fim_suffix|>") != Some(marker)
        {
            return Err(DeciderError("invalid_decider_tokenizer"));
        }
        let weights = File::open(path.join("model.safetensors")).map_err(|_| DeciderError("decider_weights_io"))?;
        let loader = ParameterLoader::new(&weights, self.context.as_ref())
            .map_err(|_| DeciderError("invalid_decider_weights"))?
            .with_numerical_profile(NumericalProfile::Qwen35ReferenceBF16);
        let decoder = Decoder::new(
            self.context.as_ref(),
            &config.decoder_config,
            &loader.tree().subtree("decoder"),
            DataType::BF16,
        )
        .map_err(|_| DeciderError("decider_decoder_load_failed"))?;
        let tree = loader.tree().subtree("decision.readout");
        let spec: AnyWeightMatrixSpec = tree.metadata("spec").map_err(|_| DeciderError("invalid_decider_readout"))?;
        let readout = LinearMatmul::load(
            self.context.as_ref(),
            spec,
            model_dim,
            options,
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
            &tree,
            None,
            None,
        )
        .map_err(|_| DeciderError("decider_readout_load_failed"))?;
        let tower = VisionTower::new(self.context.as_ref(), &tower_config, &loader.tree().subtree("vision"))
            .map_err(|_| DeciderError("decider_tower_load_failed"))?;
        loader.tree().assert_all_tensors_validated().map_err(|_| DeciderError("unvalidated_decider_weights"))?;
        let limits = PreprocessLimits {
            processor_max_pixels: max_pixels,
            max_processed_pixels: max_pixels,
            max_encoder_patches: 6400,
            ..Default::default()
        };
        Ok(DeciderModel {
            engine: self.clone(),
            identity: Arc::new(()),
            decoder,
            readout,
            tower: Some(tower),
            tokenizer,
            config: config.decoder_config,
            marker,
            options,
            model_dim,
            limits,
        })
    }

    /// Diagnostic loading arm for matched text measurements. Admission still
    /// validates the complete image artifact before releasing the tower.
    #[doc(hidden)]
    pub fn load_decider_text_validation_model(
        self: &Arc<Self>,
        path: &Path,
    ) -> Result<DeciderModel<B>, DeciderError> {
        let mut model = self.load_decider_model(path)?;
        model.tower = None;
        self.context.reclaim_cached_memory().map_err(|_| DeciderError("decider_backend_failed"))?;
        Ok(model)
    }
}

impl<B: Backend> DeciderModel<B>
where
    B::ScratchBuffer: BufferCpuAccessible,
{
    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }
    pub fn vocab_size(&self) -> u32 {
        self.config.vocab_size
    }
    pub fn max_context_length(&self) -> Option<u32> {
        Some(MAX_DECIDER_INPUT_TOKENS)
    }
    pub fn num_options(&self) -> usize {
        self.options as usize
    }
    pub fn has_images(&self) -> bool {
        self.tower.is_some()
    }

    /// Conservative full-sequence workspace allowance. Ordinary generation's
    /// 1024-row scratch estimate is scaled quadratically for eager attention.
    pub fn memory_bytes(
        &self,
        rows: u32,
        context: u32,
    ) -> Option<usize> {
        if rows == 0 || rows > MAX_DECIDER_INPUT_TOKENS || context < rows || context > MAX_DECIDER_INPUT_TOKENS {
            return None;
        }
        let estimate = self.decoder.scoring_memory_estimate(context, rows)?;
        let scale = (rows as usize).div_ceil(1024).max(1).checked_pow(2)?;
        let scratch = estimate.scratch_bytes.checked_mul(scale)?;
        if self
            .engine
            .context
            .max_buffer_length()
            .is_some_and(|limit| estimate.max_buffer_bytes.checked_mul(scale).is_none_or(|size| size > limit))
        {
            return None;
        }
        estimate.state_bytes.checked_add(scratch)
    }

    pub fn score(
        &self,
        input: &[u32],
        marker: u32,
        context: u32,
    ) -> Result<DeciderScores, DeciderError> {
        self.score_images(input, marker, &[], context)
    }

    /// Raw protocol tokens have one placeholder per image. One static image is
    /// admitted; placeholder expansion and marker adjustment happen natively.
    pub fn score_images(
        &self,
        input: &[u32],
        marker: u32,
        images: &[&[u8]],
        context: u32,
    ) -> Result<DeciderScores, DeciderError> {
        if input.is_empty()
            || input.len() > MAX_DECIDER_INPUT_TOKENS as usize
            || images.len() > 1
            || (!images.is_empty() && self.tower.is_none())
            || input.iter().any(|&id| id >= self.vocab_size())
            || input.get(marker as usize) != Some(&self.marker)
            || input.iter().filter(|&&id| id == self.marker).count() != 1
        {
            return Err(DeciderError("invalid_decider_input"));
        }
        let image_token =
            self.tokenizer.token_to_id("<|image_pad|>").ok_or(DeciderError("invalid_decider_image_token"))?;
        if input.iter().filter(|&&id| id == image_token).count() != images.len() {
            return Err(DeciderError("invalid_decider_image_placeholders"));
        }
        let special = |name| self.tokenizer.token_to_id(name).ok_or(DeciderError("invalid_decider_image_token"));
        let start_token = special("<|vision_start|>")?;
        let end_token = special("<|vision_end|>")?;
        let video_token = special("<|video_pad|>")?;
        if input.contains(&video_token)
            || input.iter().filter(|&&id| id == start_token).count() != images.len()
            || input.iter().filter(|&&id| id == end_token).count() != images.len()
            || input.iter().position(|&id| id == image_token).is_some_and(|start| {
                start == 0 || input.get(start - 1) != Some(&start_token) || input.get(start + 1) != Some(&end_token)
            })
        {
            return Err(DeciderError("invalid_decider_image_layout"));
        }
        let prepared = if let Some(bytes) = images.first() {
            let decoded = preprocess::decode_decider_image(bytes, &self.limits)
                .map_err(|_| DeciderError("invalid_decider_image"))?;
            Some(
                preprocess::prepare_image(&decoded, &self.limits)
                    .map_err(|_| DeciderError("decider_image_preparation_failed"))?,
            )
        } else {
            None
        };
        let image_rows = prepared.as_ref().map_or(0, |image| image.grid[1] * image.grid[2] / 4);
        let image_start = input.iter().position(|&id| id == image_token);
        let mut expanded = Vec::with_capacity(input.len() + image_rows.saturating_sub(1) as usize);
        for &id in input {
            if id == image_token {
                expanded.extend(std::iter::repeat_n(id, image_rows as usize));
            } else {
                expanded.push(id);
            }
        }
        let marker = marker
            + if image_start.is_some_and(|start| start < marker as usize) {
                image_rows - 1
            } else {
                0
            };
        let rows = u32::try_from(expanded.len()).map_err(|_| DeciderError("decider_context_overflow"))?;
        let decoder_bytes = self.memory_bytes(rows, context).ok_or(DeciderError("decider_resource_limit"))?;
        let image_bytes = match &prepared {
            Some(image) => self
                .tower
                .as_ref()
                .and_then(|tower| tower.workspace_bytes(image.grid))
                .ok_or(DeciderError("decider_resource_limit"))?,
            None => 0,
        };
        stark_resource_budget::admit_inference_memory(
            decoder_bytes.checked_add(image_bytes),
            stark_resource_budget::available_memory_bytes().unwrap_or(0),
        )
        .map_err(|_| DeciderError("decider_resource_limit"))?;
        let started = Instant::now();
        let mut state = self
            .decoder
            .create_empty_state(Some(context), self.engine.context.as_ref())
            .map_err(|_| DeciderError("decider_state_failed"))?;
        state.prepare(0, rows).map_err(|_| DeciderError("decider_state_failed"))?;
        let state_creation_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let mut multimodal = None;
        if let Some(prepared) = prepared {
            let pool = self.engine.context.create_allocation_pool();
            let mut command = self
                .engine
                .context
                .create_command_buffer(Some("decider vision"), Some(pool))
                .map_err(|_| DeciderError("decider_backend_failed"))?;
            let tower = self.tower.as_ref().ok_or(DeciderError("decider_images_unavailable"))?;
            let features = tower
                .encode(&prepared.patches, prepared.grid, &mut command)
                .map_err(|_| DeciderError("decider_vision_failed"))?;
            command
                .end_encoding()
                .submit()
                .wait_until_completed()
                .map_err(|_| DeciderError("decider_backend_failed"))?;
            let features = (&features).as_slice::<bf16>().into();
            let projected = ProjectedImage {
                model_identity: self.identity.clone(),
                model_dim: self.model_dim,
                data_type: DataType::BF16,
                grid: ImageGrid {
                    temporal: prepared.grid[0],
                    height: prepared.grid[1],
                    width: prepared.grid[2],
                    merge_size: 2,
                },
                token_start: image_start.ok_or(DeciderError("invalid_decider_image_placeholders"))?,
                features,
            };
            let id = |name| {
                self.tokenizer.token_to_id(name).map(u64::from).ok_or(DeciderError("invalid_decider_image_token"))
            };
            multimodal = Some(
                MultimodalPrefill::try_new(
                    expanded.iter().map(|&id| u64::from(id)).collect(),
                    projected,
                    ImageTokenIds {
                        image: u64::from(image_token),
                        vision_start: id("<|vision_start|>")?,
                        vision_end: id("<|vision_end|>")?,
                        video: id("<|video_pad|>")?,
                    },
                    &self.identity,
                    self.model_dim,
                    DataType::BF16,
                    self.vocab_size(),
                    context,
                )
                .map_err(|_| DeciderError("invalid_decider_image_layout"))?,
            );
        }
        let nodes: Vec<_> = (0..rows)
            .map(|index| TrieNode {
                trie_start: index,
                trie_end: rows - 1,
                height: index,
            })
            .collect();
        let batch = if let Some(image) = &multimodal {
            BatchTopology::with_rotary_positions(&nodes, true, &image.positions)
                .map_err(|_| DeciderError("invalid_decider_positions"))?
        } else {
            BatchTopology::new(&nodes, true)
        };
        let replacement = multimodal.as_ref().map(|image| EmbeddingReplacement {
            token_start: image.image.token_start as u32,
            features: &image.image.features,
        });
        let pool = self.engine.context.create_allocation_pool();
        let mut command = self
            .engine
            .context
            .create_command_buffer(Some("decider full sequence"), Some(pool))
            .map_err(|_| DeciderError("decider_backend_failed"))?;
        let tokens =
            command.allocate_constant_from_slice(&expanded).map_err(|_| DeciderError("decider_backend_failed"))?;
        let hidden = self
            .decoder
            .encode_decision_hidden(&tokens, replacement, &batch, marker, &mut state, &mut command)
            .map_err(|_| DeciderError("decider_forward_failed"))?;
        let logits =
            self.readout.encode(hidden, 1, &mut command).map_err(|_| DeciderError("decider_readout_failed"))?;
        // The single submission is always waited before state or scratch drops.
        command.end_encoding().submit().wait_until_completed().map_err(|_| DeciderError("decider_backend_failed"))?;
        let prefill_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let logits: Vec<_> = (&logits).as_slice::<bf16>().iter().map(|value| value.to_f32()).collect();
        if logits.len() != self.options as usize || logits.iter().any(|value| !value.is_finite()) {
            return Err(DeciderError("invalid_decider_output"));
        }
        Ok(DeciderScores {
            logits,
            input_tokens: rows as usize,
            state_creation_seconds,
            prefill_seconds,
            readout_seconds: started.elapsed().as_secs_f64(),
        })
    }
}
