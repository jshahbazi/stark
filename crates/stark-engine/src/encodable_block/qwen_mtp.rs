//! Qwen3.5-family MTP graph. Graph provenance: Apache-2.0 oMLX
//! `qwen35_model.py` at 2165e75726ba83d9651dbaced61c97728da77bba.
//! The pinned driver's predictor consumes the target's output-normalized hidden
//! state and the next token embedding; the fusion layout is [e, h]. Drafts
//! chain on the predictor's output-normalized hidden state.
use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding},
    config::qwen_mtp::QwenMtpConfig,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::{Linear, LinearBlockError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        transformer::{Transformer, TransformerNewError, TransformerState},
    },
    parameters::ParameterTree,
};

#[derive(Debug, Error)]
pub enum QwenMtpError<B: Backend> {
    #[error("invalid Qwen MTP graph")]
    Invalid,
    #[error("Qwen MTP backend failure: {0}")]
    Backend(#[source] B::Error),
    #[error("Qwen MTP normalization failure: {0}")]
    Norm(#[from] NormalizationNewError<B>),
    #[error("Qwen MTP projection failure: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Qwen MTP transformer failure: {0}")]
    Transformer(#[from] TransformerNewError<B>),
}

pub struct QwenMtp<B: Backend> {
    pub(crate) target_layer: u32,
    model_dim: u32,
    embedding_norm: Normalization<B>,
    hidden_norm: Normalization<B>,
    fusion: Box<dyn Linear<B>>,
    transformer: Transformer<B>,
}

pub struct QwenMtpOutput<B: Backend> {
    pub normalized: B::ScratchBuffer,
}

impl<B: Backend> QwenMtp<B> {
    pub fn new(
        config: &QwenMtpConfig,
        target_layers: usize,
        tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<Self, QwenMtpError<B>> {
        config.validate_predictor().map_err(|_| QwenMtpError::Invalid)?;
        if config.model_dim == 0
            || config.model_dim != config.transformer_config.model_dim
            || config.target_layer as usize + 1 != target_layers
            || config.transformer_config.layer_configs.len() != 1
            || !matches!(
                config.transformer_config.layer_configs[0].mixer_config,
                crate::config::token_mixer::AnyTokenMixerConfig::AttentionConfig(_)
            )
        {
            return Err(QwenMtpError::Invalid);
        }
        let norm = |name: &str| {
            Normalization::new(
                config.model_dim,
                None,
                ShortcutMode::None,
                PostLayerScalar::None,
                DataType::BF16,
                &config.input_norm_config,
                &tree.subtree(name),
                context,
            )
        };
        Ok(Self {
            target_layer: config.target_layer,
            model_dim: config.model_dim,
            embedding_norm: norm("embedding_norm")?,
            hidden_norm: norm("hidden_norm")?,
            fusion: <dyn Linear<B>>::new(
                config.model_dim.checked_mul(2).ok_or(QwenMtpError::Invalid)?,
                [config.model_dim],
                false,
                context,
                DataType::BF16,
                &tree.subtree("fusion"),
            )?,
            transformer: Transformer::new(
                context,
                None,
                DataType::BF16,
                &config.transformer_config,
                &tree.subtree("transformer"),
            )?,
        })
    }

    pub fn create_state(
        &self,
        capacity: u32,
        context: &B::Context,
    ) -> Result<TransformerState<B>, B::Error> {
        self.transformer.create_empty_state(Some(capacity), context)
    }

    pub fn encode(
        &self,
        embedding: impl BufferRef<Backend = B>,
        target_hidden: impl BufferRef<Backend = B>,
        topology: &BatchTopology,
        state: &mut TransformerState<B>,
        command: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<QwenMtpOutput<B>, QwenMtpError<B>> {
        let count = topology.size();
        let row_bytes = self.model_dim as usize * DataType::BF16.size_in_bytes();
        if count == 0
            || embedding.size() != count as usize * row_bytes
            || target_hidden.size() != count as usize * row_bytes
        {
            return Err(QwenMtpError::Invalid);
        }
        let e = self
            .embedding_norm
            .encode(embedding, 0, count, None::<&mut B::ScratchBuffer>, command)
            .map_err(QwenMtpError::Backend)?;
        let h = self
            .hidden_norm
            .encode(target_hidden, 0, count, None::<&mut B::ScratchBuffer>, command)
            .map_err(QwenMtpError::Backend)?;
        let mut joined = command.allocate_scratch(count as usize * row_bytes * 2).map_err(QwenMtpError::Backend)?;
        for row in 0..count as usize {
            command.encode_copy(
                e.subrange(row * row_bytes..(row + 1) * row_bytes),
                joined.subrange_mut(2 * row * row_bytes..(2 * row + 1) * row_bytes),
            );
            command.encode_copy(
                h.subrange(row * row_bytes..(row + 1) * row_bytes),
                joined.subrange_mut((2 * row + 1) * row_bytes..(2 * row + 2) * row_bytes),
            );
        }
        let fused = self.fusion.encode(joined, count, command).map_err(QwenMtpError::Backend)?;
        let encoded = self
            .transformer
            .encode(fused, None::<&B::ScratchBuffer>, topology, Some((0..count).into()), None, Some(state), command)
            .map_err(QwenMtpError::Backend)?;
        Ok(QwenMtpOutput {
            normalized: encoded.output.ok_or(QwenMtpError::Invalid)?,
        })
    }
}
