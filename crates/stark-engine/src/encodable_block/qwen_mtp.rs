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
    #[cfg(test)]
    validation_reference_fusion: Option<Box<dyn Linear<B>>>,
}

#[cfg(test)]
struct ValidationReferenceFusion<B: Backend> {
    kernel: crate::encodable_block::linear::ReferenceBF16Matmul<B>,
    weights: std::sync::Arc<B::GlobalBuffer>,
    model_dim: u32,
}

#[cfg(test)]
impl<B: Backend> Linear<B> for ValidationReferenceFusion<B> {
    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        self.kernel.encode(&input, 0, self.weights.as_ref(), batch_dim, self.model_dim, 2 * self.model_dim, command)
    }
}

pub struct QwenMtpOutput<B: Backend> {
    pub normalized: B::ScratchBuffer,
}

impl<B: Backend> QwenMtp<B> {
    pub(crate) fn context_cache_buffer_bytes(
        &self,
        context_tokens: u32,
    ) -> Option<usize> {
        self.transformer.context_cache_buffer_bytes(context_tokens)
    }

    pub(crate) fn memory_estimate(
        &self,
        context_tokens: u32,
        input_tokens: u32,
    ) -> Option<crate::encodable_block::memory::MemoryEstimate> {
        let mut estimate = self.transformer.memory_estimate(context_tokens, input_tokens.max(4))?;
        for rows in [input_tokens.clamp(4, 1024), 1] {
            // Embedding/hidden norms, fused projection output and the [e,h]
            // concatenation use BF16, matching fuse_inputs and predictor pass.
            estimate.add_scratch(&[rows, self.model_dim], 2, 4)?;
            estimate.add_scratch(&[rows, self.model_dim, 2], 2, 1)?;
        }
        Some(estimate)
    }

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
        #[cfg(test)]
        let validation_transformer_config = if std::env::var_os("STARK_TEST_HEAD_MLX_OPERATORS").is_some() {
            // This causal experiment combines reference head linears with the
            // source's BF16 sigmoid/SiLU boundaries. Never alter target policy,
            // normalization, attention or artifact metadata to repair a gate.
            if std::env::var_os("STARK_TEST_HEAD_REFERENCE_LINEARS").is_none()
                || std::env::var_os("STARK_TEST_HEAD_REFERENCE_FUSION").is_some()
                || tree.numerical_profile() != crate::parameters::NumericalProfile::Legacy
            {
                return Err(QwenMtpError::Invalid);
            }
            let mut transformer = config.transformer_config.clone();
            let crate::config::mlp::AnyMLPConfig::DenseMLPConfig(mlp) = &mut transformer.layer_configs[0].mlp_config
            else {
                return Err(QwenMtpError::Invalid);
            };
            let crate::config::activation::AnyActivation::SiLU(silu) = &mut mlp.activation else {
                return Err(QwenMtpError::Invalid);
            };
            if silu.alpha != 1.0 {
                return Err(QwenMtpError::Invalid);
            }
            silu.materialize_sigmoid = true;
            Some(transformer)
        } else {
            None
        };
        let transformer_config = &config.transformer_config;
        #[cfg(test)]
        let transformer_config = validation_transformer_config.as_ref().unwrap_or(transformer_config);
        #[cfg(test)]
        let validation_tree = if std::env::var_os("STARK_TEST_HEAD_REFERENCE_LINEARS").is_some() {
            if std::env::var_os("STARK_TEST_HEAD_REFERENCE_FUSION").is_some()
                || tree.numerical_profile() != crate::parameters::NumericalProfile::Legacy
            {
                return Err(QwenMtpError::Invalid);
            }
            Some(tree.validation_with_numerical_profile(crate::parameters::NumericalProfile::QwenMtpReferenceLinears))
        } else {
            None
        };
        #[cfg(test)]
        let tree = validation_tree.as_ref().unwrap_or(tree);
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
        #[cfg(test)]
        let validation_reference_fusion = {
            use crate::config::weight_matrix::AnyWeightMatrixSpec;
            let weights = tree.subtree("fusion").subtree("weights");
            let spec = weights.metadata::<AnyWeightMatrixSpec>("spec").map_err(LinearBlockError::from)?;
            if matches!(spec, AnyWeightMatrixSpec::FullPrecisionSpec(_))
                && std::env::var_os("STARK_TEST_HEAD_REFERENCE_FUSION").is_some()
            {
                let weights = weights
                    .leaf("weights")
                    .and_then(|leaf| leaf.validate(&[config.model_dim, 2 * config.model_dim], DataType::BF16))
                    .and_then(|leaf| leaf.read_shared_weight())
                    .map_err(LinearBlockError::from)?;
                Some(Box::new(ValidationReferenceFusion {
                    kernel: crate::encodable_block::linear::ReferenceBF16Matmul::new(context)
                        .map_err(QwenMtpError::Backend)?,
                    weights,
                    model_dim: config.model_dim,
                }) as Box<dyn Linear<B>>)
            } else {
                None
            }
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
                transformer_config,
                &tree.subtree("transformer"),
            )?,
            #[cfg(test)]
            validation_reference_fusion,
        })
    }

    #[cfg(test)]
    pub(crate) fn validation_use_reference_fusion(&mut self) -> Result<(), QwenMtpError<B>> {
        self.fusion = self.validation_reference_fusion.take().ok_or(QwenMtpError::Invalid)?;
        Ok(())
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
        let fused = self.fuse_inputs(embedding, target_hidden, count, row_bytes, command)?;
        let encoded = self
            .transformer
            .encode(fused, None::<&B::ScratchBuffer>, topology, Some((0..count).into()), None, Some(state), command)
            .map_err(QwenMtpError::Backend)?;
        Ok(QwenMtpOutput {
            normalized: encoded.output.ok_or(QwenMtpError::Invalid)?,
        })
    }

    /// Populate predictor attention KV while skipping attention output and the
    /// unused feed-forward/output-normalization work.
    pub fn encode_cache_only(
        &self,
        embedding: impl BufferRef<Backend = B>,
        target_hidden: impl BufferRef<Backend = B>,
        topology: &BatchTopology,
        state: &mut TransformerState<B>,
        command: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), QwenMtpError<B>> {
        let count = topology.size();
        let row_bytes = self.model_dim as usize * DataType::BF16.size_in_bytes();
        if count == 0
            || embedding.size() != count as usize * row_bytes
            || target_hidden.size() != count as usize * row_bytes
        {
            return Err(QwenMtpError::Invalid);
        }
        let fused = self.fuse_inputs(embedding, target_hidden, count, row_bytes, command)?;
        self.transformer.encode_cache_only(fused, topology, state, command).map_err(QwenMtpError::Backend)
    }

    fn fuse_inputs(
        &self,
        embedding: impl BufferRef<Backend = B>,
        target_hidden: impl BufferRef<Backend = B>,
        count: u32,
        row_bytes: usize,
        command: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, QwenMtpError<B>> {
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
        self.fusion.encode(joined, count, command).map_err(QwenMtpError::Backend)
    }

    /// Private first-divergence diagnostic. No tensor is printed or published.
    #[cfg(test)]
    pub(crate) fn validation_fusion_stages(
        &self,
        embedding: impl BufferRef<Backend = B>,
        hidden: impl BufferRef<Backend = B>,
        count: u32,
        context: &B::Context,
        command: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Vec<(&'static str, B::GlobalBuffer)>, QwenMtpError<B>> {
        use crate::backends::common::{Buffer, Context};
        let row_bytes = self.model_dim as usize * DataType::BF16.size_in_bytes();
        let e = self
            .embedding_norm
            .encode(embedding, 0, count, None::<&mut B::ScratchBuffer>, command)
            .map_err(QwenMtpError::Backend)?;
        let h = self
            .hidden_norm
            .encode(hidden, 0, count, None::<&mut B::ScratchBuffer>, command)
            .map_err(QwenMtpError::Backend)?;
        let fused = self.fuse_inputs(embedding, hidden, count, row_bytes, command)?;
        let mut result = Vec::new();
        let copy = |value: &B::ScratchBuffer,
                    command: &mut <B::CommandBuffer as CommandBuffer>::Encoding|
         -> Result<B::GlobalBuffer, QwenMtpError<B>> {
            let mut saved = context.create_buffer(value.size()).map_err(QwenMtpError::Backend)?;
            command.encode_copy(value, &mut saved);
            Ok(saved)
        };
        result.push(("embedding_norm", copy(&e, command)?));
        result.push(("hidden_norm", copy(&h, command)?));
        result.push(("fusion", copy(&fused, command)?));
        if let Some((layer, rope)) = self.transformer.validation_first_layer() {
            use crate::{
                backends::common::gpu_types::trie::TrieNode, encodable_block::mixer::attention::rope::PrecalculatedRoPE,
            };
            let nodes = (0..count)
                .map(|height| TrieNode {
                    height,
                    trie_start: height,
                    trie_end: count - 1,
                })
                .collect::<Vec<_>>();
            let topology = BatchTopology::new(&nodes, true);
            let positions = (0..count).collect::<Vec<_>>();
            let rope = rope
                .map(|config| PrecalculatedRoPE::<B>::precalculate(config, &positions, command))
                .transpose()
                .map_err(QwenMtpError::Backend)?;
            let mut shortcut = command.allocate_scratch(fused.size()).map_err(QwenMtpError::Backend)?;
            let pre = layer
                .pre_mixer_norm
                .as_ref()
                .ok_or(QwenMtpError::Invalid)?
                .encode(&fused, 0, count, Some(&mut shortcut), command)
                .map_err(QwenMtpError::Backend)?;
            result.push(("pre_attention", copy(&pre, command)?));
            let attention =
                layer.mixer.encode(pre, rope.as_ref(), &topology, None, command).map_err(QwenMtpError::Backend)?;
            result.push(("attention_output", copy(&attention, command)?));
            let pre_mlp = layer
                .pre_mlp_norm
                .encode(&attention, 0, count, Some(&mut shortcut), command)
                .map_err(QwenMtpError::Backend)?;
            result.push(("attention_residual", copy(&shortcut, command)?));
            result.push(("pre_mlp", copy(&pre_mlp, command)?));
            let mlp = layer.mlp.encode(pre_mlp, count, command).map_err(QwenMtpError::Backend)?;
            result.push(("mlp_output", copy(&mlp, command)?));
            let final_hidden = self
                .transformer
                .validation_output_norm()
                .encode(&mlp, 0, count, Some(&mut shortcut), command)
                .map_err(QwenMtpError::Backend)?;
            result.push(("final_hidden", copy(&final_hidden, command)?));
        }
        Ok(result)
    }

    #[cfg(test)]
    pub(crate) fn validation_primitive_stages(
        &self,
        captured: &[(&'static str, B::GlobalBuffer)],
        rows: u32,
        context: &B::Context,
        commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Vec<(&'static str, B::GlobalBuffer, Vec<usize>)>, QwenMtpError<B>> {
        use std::any::Any;

        use crate::{
            backends::common::{Buffer, Context, gpu_types::trie::TrieNode},
            encodable_block::mixer::attention::{Attention, rope::PrecalculatedRoPE},
        };
        if rows == 0 || rows > 16 {
            return Err(QwenMtpError::Invalid);
        }
        let (layer, rope) = self.transformer.validation_first_layer().ok_or(QwenMtpError::Invalid)?;
        let positions = (0..rows).collect::<Vec<_>>();
        let rope = rope
            .map(|config| PrecalculatedRoPE::<B>::precalculate(config, &positions, commands))
            .transpose()
            .map_err(QwenMtpError::Backend)?;
        let nodes = (0..rows)
            .map(|height| TrieNode {
                height,
                trie_start: height,
                trie_end: rows - 1,
            })
            .collect::<Vec<_>>();
        let topology = BatchTopology::new(&nodes, true);
        let input = |name: &str,
                     commands: &mut <B::CommandBuffer as CommandBuffer>::Encoding|
         -> Result<B::ScratchBuffer, QwenMtpError<B>> {
            let value = captured.iter().find(|(stage, _)| *stage == name).ok_or(QwenMtpError::Invalid)?;
            let mut copy = commands.allocate_scratch(value.1.size()).map_err(QwenMtpError::Backend)?;
            commands.encode_copy(&value.1, &mut copy);
            Ok(copy)
        };
        let attention =
            (layer.mixer.as_ref() as &dyn Any).downcast_ref::<Attention<B>>().ok_or(QwenMtpError::Invalid)?;
        let stages = attention
            .validation_stateless_stages(input("pre_attention", commands)?, rope.as_ref(), &topology, commands)
            .map_err(QwenMtpError::Backend)?;
        let mut result = Vec::new();
        for (name, data, shape) in stages {
            let mut saved = context.create_buffer(data.size()).map_err(QwenMtpError::Backend)?;
            commands.encode_copy(&data, &mut saved);
            result.push((name, saved, shape));
        }
        let stages = layer
            .mlp
            .validation_trace_dense(input("pre_mlp", commands)?, rows, commands)
            .map_err(QwenMtpError::Backend)?
            .ok_or(QwenMtpError::Invalid)?;
        for (name, data) in stages {
            if data.size() % (rows as usize * 2) != 0 {
                return Err(QwenMtpError::Invalid);
            }
            let columns = data.size() / (rows as usize * 2);
            let mut saved = context.create_buffer(data.size()).map_err(QwenMtpError::Backend)?;
            commands.encode_copy(&data, &mut saved);
            result.push((name, saved, vec![1, rows as usize, columns]));
        }
        Ok(result)
    }
}
