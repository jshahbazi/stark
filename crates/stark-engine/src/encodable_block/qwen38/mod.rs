//! Full Qwen3.8 architecture execution on Stark's Uzu backend.
//!
//! The converted MLX planes remain in canonical Uzu layout. Every projection,
//! including routed expert banks, uses the ordinary backend matmul dispatcher.
//! Qwen-specific recurrence, sparse attention, hyper-connections and hashed
//! PLE are encodable blocks; this module has no Lily runtime dependency.
use std::range::Range;

use derive_more::Debug;
use parking_lot::Mutex;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferRef, CommandBuffer, CommandBufferEncoding, InvalidInput, Kernels, StageProfileCategory,
        kernel::{
            Qwen38BroadcastKernel,
            matmul::{MatmulA, MatmulArguments, MatmulDOps, MatmulKernel},
        },
    },
    config::model::qwen38_language_model::{Qwen38DecoderConfig, Qwen38LayerType},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        decoder::DecoderEncodeOutput,
        embedding_table::{EmbeddingTable, EmbeddingTableError},
        linear::{LinearMatmul, LinearMatmulError},
        weight_matrix::WeightMatrixError,
    },
    parameters::{ParameterLoader, ParameterLoaderError},
};

#[cfg(test)]
mod execution_tests;
mod gdn;
mod hc;
mod moe;
pub(crate) mod ple;
pub(crate) mod qsa;
pub mod state;
pub(crate) mod weights;

use gdn::Gdn;
use hc::HyperConnection;
use moe::Moe;
use ple::Ple;
use qsa::Qsa;
pub use state::Qwen38State;
use state::{LayerState, MixerState};
use weights::Qwen38WeightLoader;

pub(crate) type Encoding<B> = <<B as Backend>::CommandBuffer as CommandBuffer>::Encoding;

#[derive(Debug, Error)]
pub enum Qwen38Error<B: Backend> {
    #[error("Qwen backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Qwen parameters error: {0}")]
    Parameters(#[from] ParameterLoaderError<B>),
    #[error("Qwen linear error: {0}")]
    Linear(#[from] LinearMatmulError<B>),
    #[error("Qwen matrix error: {0}")]
    Matrix(#[from] WeightMatrixError<B>),
    #[error("Qwen embedding error: {0}")]
    Embedding(#[from] EmbeddingTableError<B>),
    #[error("Qwen parameter I/O failed")]
    Io(#[from] std::io::Error),
    #[error("invalid or unsupported Qwen architecture: {0}")]
    Invalid(&'static str),
}

pub(super) fn project_batch<B: Backend>(
    linear: &LinearMatmul<B>,
    input: impl BufferRef<Backend = B>,
    rows: u32,
    commands: &mut Encoding<B>,
) -> Result<B::ScratchBuffer, B::Error> {
    linear.encode_with_a(
        MatmulA::FullPrecision {
            values: input,
            offset: 0,
        },
        rows,
        None::<crate::encodable_block::linear::Gather<&B::ScratchBuffer>>,
        commands,
    )
}

enum Mixer<B: Backend> {
    Gdn(Gdn<B>),
    Qsa(Qsa<B>),
}
struct Layer<B: Backend> {
    hc_mixer: HyperConnection<B>,
    mixer: Mixer<B>,
    hc_mlp: HyperConnection<B>,
    moe: Moe<B>,
    ple: Option<Ple<B>>,
}

pub struct Qwen38Decoder<B: Backend> {
    embedding: EmbeddingTable<B>,
    head: Option<LinearMatmul<B>>,
    tied_head: Mutex<<B::Kernels as Kernels>::MatmulKernel>,
    final_mixer: HyperConnection<B>,
    layers: Vec<Layer<B>>,
    broadcast: <B::Kernels as Kernels>::Qwen38BroadcastKernel,
    hidden: u32,
    streams: u32,
    vocabulary: u32,
    maximum_context: u32,
}

impl<B: Backend> Qwen38Decoder<B> {
    pub fn new(
        context: &B::Context,
        config: &Qwen38DecoderConfig,
        parameters: &ParameterLoader<B>,
    ) -> Result<Self, Qwen38Error<B>> {
        let cfg = &config.architecture;
        if cfg.hidden_size == 0
            || cfg.hc_count == 0
            || cfg.hc_lowrank == 0
            || cfg.vocab_size == 0
            || cfg.num_hidden_layers == 0
            || cfg.layer_types.len() != cfg.num_hidden_layers as usize
            || cfg.max_position_embeddings == 0
            || !cfg.rms_norm_eps.is_finite()
            || cfg.rms_norm_eps <= 0.0
            || cfg.hidden_act != "silu"
            || cfg.output_gate_type.as_deref().is_some_and(|gate| !matches!(gate, "sigmoid" | "silu"))
        {
            return Err(Qwen38Error::Invalid("model geometry or activation"));
        }
        let loader = Qwen38WeightLoader::new(context, config, parameters)?;
        let embedding = loader.embedding("model.language_model.embed_tokens", cfg.vocab_size, cfg.hidden_size)?;
        let head = (!cfg.tie_word_embeddings)
            .then(|| loader.linear("lm_head", cfg.hidden_size, cfg.vocab_size, DataType::BF16, DataType::F32))
            .transpose()?;
        let final_mixer =
            HyperConnection::new(context, cfg, "model.language_model.hyper_connection_mixer", false, &loader)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers as usize);
        for (index, kind) in cfg.layer_types.iter().enumerate() {
            let index = index as u32;
            let prefix = format!("model.language_model.layers.{index}");
            layers.push(Layer {
                hc_mixer: HyperConnection::new(
                    context,
                    cfg,
                    &format!("{prefix}.attn_hyper_connection"),
                    true,
                    &loader,
                )?,
                mixer: match kind {
                    Qwen38LayerType::LinearAttention => Mixer::Gdn(Gdn::new(context, cfg, index, &loader)?),
                    Qwen38LayerType::FullAttention => Mixer::Qsa(Qsa::new(context, cfg, index, &loader)?),
                },
                hc_mlp: HyperConnection::new(context, cfg, &format!("{prefix}.mlp_hyper_connection"), true, &loader)?,
                moe: Moe::new(context, cfg, index, &loader)?,
                ple: cfg
                    .ple_layer_ids
                    .contains(&(index + 1))
                    .then(|| Ple::new(context, cfg, index, &loader))
                    .transpose()?,
            });
        }
        loader.finish()?;
        Ok(Self {
            embedding,
            head,
            final_mixer,
            layers,
            tied_head: Mutex::new(
                <B::Kernels as Kernels>::MatmulKernel::new(context, DataType::BF16, DataType::BF16, DataType::F32)
                    .map_err(Qwen38Error::Backend)?,
            ),
            broadcast: <B::Kernels as Kernels>::Qwen38BroadcastKernel::new(context).map_err(Qwen38Error::Backend)?,
            hidden: cfg.hidden_size,
            streams: cfg.hc_count,
            vocabulary: cfg.vocab_size,
            maximum_context: qsa::context_limit(cfg.max_position_embeddings, cfg.indexer_compress_ratio),
        })
    }

    pub fn vocab_size(&self) -> u32 {
        self.vocabulary
    }
    pub fn model_dim(&self) -> u32 {
        self.hidden
    }
    pub fn max_context_length(&self) -> Option<u32> {
        Some(self.maximum_context)
    }
    pub fn speculation_supported(&self) -> bool {
        false
    }
    pub fn requires_host_tokens(&self) -> bool {
        true
    }
    /// Bound layer-batched prefill's command/scratch working set.
    pub fn prefill_chunk_size(&self) -> u32 {
        16
    }

    pub fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<Qwen38State<B>, B::Error> {
        let capacity = max_context_length.unwrap_or(self.maximum_context);
        if capacity == 0 || capacity > self.maximum_context {
            return Err(InvalidInput("invalid Qwen context capacity").into());
        }
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            layers.push(LayerState {
                mixer: match &layer.mixer {
                    Mixer::Gdn(mixer) => MixerState::Gdn(mixer.create_state(context)?),
                    Mixer::Qsa(mixer) => MixerState::Qsa(mixer.create_state(context, capacity)?),
                },
                ple: layer.ple.as_ref().map(|ple| ple.create_state(context)).transpose()?,
            });
        }
        Ok(Qwen38State::new(layers, capacity))
    }

    fn readout(
        &self,
        hidden: impl BufferRef<Backend = B>,
        rows: u32,
        commands: &mut Encoding<B>,
    ) -> Result<B::ScratchBuffer, B::Error> {
        if let Some(head) = &self.head {
            return project_batch(head, hidden, rows, commands);
        }
        let matrix = self.embedding.as_matrix().ok_or(InvalidInput("tied Qwen embedding lacks a matrix"))?;
        let mut output = commands.allocate_scratch_for_shape(&[rows, self.vocabulary], DataType::F32)?;
        self.tied_head.lock().encode(
            MatmulArguments {
                a: MatmulA::FullPrecision {
                    values: hidden,
                    offset: 0,
                },
                b: matrix.matmul_b(),
                b_leading_dimension: None,
                b_transpose: true,
                d: &mut output,
                d_transform: MatmulDOps::none(),
                gather_indices: None::<&B::ScratchBuffer>,
                m: rows,
                n: self.vocabulary,
                k: self.hidden,
            },
            commands,
        )?;
        Ok(output)
    }

    /// A complete flat suffix is encoded layer by layer. Each layer batches its
    /// projections while recurrent kernels and sparse caches retain token order.
    /// Host token IDs are supplied by the caller; unfinished GPU buffers are
    /// never read by the CPU for ngram hashing.
    pub fn encode(
        &self,
        _token_ids: impl BufferRef<Backend = B>,
        host_token_ids: &[u32],
        batch: &BatchTopology,
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        state: &mut Qwen38State<B>,
        commands: &mut Encoding<B>,
    ) -> Result<DecoderEncodeOutput<B>, Qwen38Error<B>> {
        let count = batch.size();
        if !batch.is_flat()
            || !batch.full_accept()
            || batch.rotary_positions().is_some()
            || host_token_ids.len() != count as usize
            || host_token_ids.iter().any(|&id| id >= self.vocabulary)
            || hidden_feature_layer_indices.is_some_and(|indices| !indices.is_empty())
            || state.layers.len() != self.layers.len()
        {
            return Err(Qwen38Error::Invalid("Qwen execution requires a complete flat text suffix"));
        }
        let range = output_range.map(|range| (range.start, range.end));
        if range.is_some_and(|(start, end)| start >= end || end > count) {
            return Err(Qwen38Error::Invalid("Qwen readout range"));
        }
        state.begin_encode(count).map_err(Qwen38Error::Backend)?;
        let base_position = state.context_length();
        if count == 1 {
            commands.begin_single_token_profile();
        }
        // The ordinary embedding kernel consumes U32 IDs. Constants belong to
        // this command buffer; PLE also uses these already-known host tokens.
        let token_buffer = commands.allocate_constant_from_slice(host_token_ids).map_err(Qwen38Error::Backend)?;
        let mut embedded =
            commands.allocate_scratch_for_shape(&[count, self.hidden], DataType::BF16).map_err(Qwen38Error::Backend)?;
        self.embedding.encode_lookup(&token_buffer, &mut embedded, count, 1.0, commands);
        let mut hyper = commands
            .allocate_scratch_for_shape(&[count, self.streams, self.hidden], DataType::BF16)
            .map_err(Qwen38Error::Backend)?;
        self.broadcast.encode(&embedded, &mut hyper, self.hidden, self.streams, count, commands);
        commands.stage_profile_checkpoint(StageProfileCategory::Embedding);
        for (layer, layer_state) in self.layers.iter().zip(&mut state.layers) {
            if let (Some(ple), Some(ple_state)) = (&layer.ple, &mut layer_state.ple) {
                hyper = ple.encode(&hyper, host_token_ids, ple_state, commands).map_err(Qwen38Error::Backend)?;
                commands.stage_profile_checkpoint(StageProfileCategory::Ple);
            }
            let read = layer.hc_mixer.read_batch(&hyper, count, commands).map_err(Qwen38Error::Backend)?;
            commands.stage_profile_checkpoint(StageProfileCategory::HcRead);
            let mixer_category = match &layer.mixer {
                Mixer::Gdn(_) => StageProfileCategory::Gdn,
                Mixer::Qsa(_) => StageProfileCategory::Qsa,
            };
            let branch = match (&layer.mixer, &mut layer_state.mixer) {
                (Mixer::Gdn(mixer), MixerState::Gdn(mixer_state)) => {
                    mixer.encode_batch(&read.mixed, count, mixer_state, commands)
                },
                (Mixer::Qsa(mixer), MixerState::Qsa(mixer_state)) => {
                    mixer.encode(&read.mixed, count, base_position, mixer_state, commands)
                },
                _ => return Err(Qwen38Error::Invalid("Qwen mixer state kind")),
            }
            .map_err(Qwen38Error::Backend)?;
            commands.stage_profile_checkpoint(mixer_category);
            hyper = layer
                .hc_mixer
                .inject_batch(
                    &hyper,
                    &branch,
                    read.injection.as_ref().ok_or(Qwen38Error::Invalid("Qwen mixer injection"))?,
                    count,
                    commands,
                )
                .map_err(Qwen38Error::Backend)?;
            commands.stage_profile_checkpoint(StageProfileCategory::HcInject);
            let read = layer.hc_mlp.read_batch(&hyper, count, commands).map_err(Qwen38Error::Backend)?;
            commands.stage_profile_checkpoint(StageProfileCategory::HcRead);
            let branch = layer.moe.encode_batch(&read.mixed, count, commands).map_err(Qwen38Error::Backend)?;
            commands.stage_profile_checkpoint(StageProfileCategory::Moe);
            hyper = layer
                .hc_mlp
                .inject_batch(
                    &hyper,
                    &branch,
                    read.injection.as_ref().ok_or(Qwen38Error::Invalid("Qwen expert injection"))?,
                    count,
                    commands,
                )
                .map_err(Qwen38Error::Backend)?;
            commands.stage_profile_checkpoint(StageProfileCategory::HcInject);
        }
        let logits = if let Some((start, end)) = range {
            let stride = self.streams as usize * self.hidden as usize * 2;
            let requested = (&hyper).subrange(start as usize * stride..end as usize * stride);
            let mixed = self.final_mixer.read_batch(requested, end - start, commands).map_err(Qwen38Error::Backend)?;
            commands.stage_profile_checkpoint(StageProfileCategory::FinalHc);
            let logits = self.readout(&mixed.mixed, end - start, commands).map_err(Qwen38Error::Backend)?;
            commands.stage_profile_checkpoint(StageProfileCategory::Head);
            Some(logits)
        } else {
            None
        };
        state.finish_encode();
        Ok(DecoderEncodeOutput {
            logits,
            hidden_features: None,
            final_hidden: None,
        })
    }
}
