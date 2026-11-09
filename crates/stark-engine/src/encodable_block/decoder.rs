use std::range::Range;

use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{Backend, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, InvalidInput},
    config::{decoder::DecoderConfig, mlp::AnyMLPConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        embedding::{Embedding, EmbeddingError},
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        per_layer_embedding::{PerLayerEmbedding, PerLayerEmbeddingError},
        transformer::{Transformer, TransformerNewError, TransformerState},
    },
    parameters::ParameterTree,
};

#[derive(Debug, Error)]
pub enum DecoderError<B: Backend> {
    #[error(transparent)]
    InvalidInput(#[from] crate::backends::common::InvalidInput),
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Embedding error: {0}")]
    EmbeddingError(#[from] EmbeddingError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
    #[error("Per-layer embedding error: {0}")]
    PerLayerEmbedding(#[from] PerLayerEmbeddingError<B>),
    #[error("Transformer error: {0}")]
    Transformer(#[from] TransformerNewError<B>),
}

pub struct Decoder<B: Backend> {
    embedding: Embedding<B>,
    embedding_norm: Option<Normalization<B>>,
    per_layer_embedding: Option<PerLayerEmbedding<B>>,
    transformer: Transformer<B>,
    data_type: DataType,
    gemma4_image_embeddings: bool,
    scalar_image_embeddings: bool,
    bidirectional_image_attention: bool,
    separate_prefill_last_token: bool,
    qwen_routed_prefill: bool,
}

/// A validated chunk of projected image rows. No public transport accepts this.
pub(crate) struct EmbeddingReplacement<'a> {
    pub(crate) token_start: u32,
    pub(crate) features: &'a [half::bf16],
}

fn mask_image_ple_tokens<B: Backend>(
    token_ids: impl BufferRef<Backend = B>,
    span: std::ops::Range<usize>,
    batch_size: u32,
    command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
) -> Result<B::ScratchBuffer, B::Error> {
    if span.start >= span.end || span.end > batch_size as usize {
        return Err(InvalidInput("invalid image PLE token mask span").into());
    }
    let mut ple_tokens = command_buffer.allocate_scratch_for_shape(&[batch_size], DataType::U32)?;
    command_buffer.encode_copy(token_ids, &mut ple_tokens);
    let token_bytes = size_of::<u32>();
    command_buffer.encode_fill((&mut ple_tokens).subrange_mut(span.start * token_bytes..span.end * token_bytes), 0);
    Ok(ple_tokens)
}

pub struct DecoderEncodeOutput<B: Backend> {
    pub logits: Option<B::ScratchBuffer>,
    pub hidden_features: Option<Box<[B::ScratchBuffer]>>,
    pub final_hidden: Option<B::ScratchBuffer>,
}

impl<B: Backend> Decoder<B> {
    pub(crate) fn embedding(&self) -> &Embedding<B> {
        &self.embedding
    }

    pub(crate) fn normalize_captured_output(
        &self,
        hidden: impl BufferRef<Backend = B>,
        rows: u32,
        command: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        self.transformer.normalize_captured_output(hidden, rows, command)
    }

    pub fn new(
        context: &B::Context,
        config: &DecoderConfig,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<Self, DecoderError<B>> {
        if (config.gemma4_image_embeddings && config.scalar_image_embeddings)
            || (config.bidirectional_image_attention && !config.scalar_image_embeddings)
        {
            return Err(InvalidInput("conflicting scalar image embedding modes").into());
        }
        crate::config::validation::validate_ple_layout(
            config.ple_model_config.as_ref(),
            &config.transformer_config.layer_configs,
        )?;
        let (embedding, readout_input_hadamard_factors) = Embedding::new(
            context,
            config.vocab_size,
            config.transformer_config.model_dim,
            &config.embedding_config,
            &parameter_tree.subtree("embedding"),
            data_type,
        )?;

        let embedding_norm = config
            .embedding_norm_config
            .as_ref()
            .map(|norm_config| {
                Normalization::new(
                    config.transformer_config.model_dim,
                    None,
                    ShortcutMode::None,
                    PostLayerScalar::None,
                    data_type,
                    norm_config,
                    &parameter_tree.subtree("embedding_norm"),
                    context,
                )
            })
            .transpose()?;

        let per_layer_embedding = if let Some(ple_config) = &config.ple_model_config {
            assert_eq!(
                ple_config.num_layers,
                config.transformer_config.layer_configs.len() as u32,
                "per-layer embedding num_layers must match transformer layer count"
            );
            Some(PerLayerEmbedding::new(
                context,
                ple_config,
                config.transformer_config.model_dim,
                data_type,
                &parameter_tree.subtree("per_layer_embedding"),
            )?)
        } else {
            None
        };

        let transformer = Transformer::new(
            context,
            readout_input_hadamard_factors,
            data_type,
            &config.transformer_config,
            &parameter_tree.subtree("transformer"),
        )?;

        Ok(Self {
            embedding,
            embedding_norm,
            per_layer_embedding,
            transformer,
            data_type,
            gemma4_image_embeddings: config.gemma4_image_embeddings,
            scalar_image_embeddings: config.scalar_image_embeddings,
            bidirectional_image_attention: config.bidirectional_image_attention,
            separate_prefill_last_token: config.separate_prefill_last_token,
            // This typed variant exclusively constructs QwenMoe in Mlp::new.
            // Gemma4MoeConfig and dense targets retain their existing schedule.
            qwen_routed_prefill: config
                .transformer_config
                .layer_configs
                .iter()
                .any(|layer| matches!(&layer.mlp_config, AnyMLPConfig::MixtureOfExpertsConfig(_))),
        })
    }

    pub(crate) fn ordinary_prefill_execution_cap(&self) -> Option<usize> {
        // A fixed model policy, never selected from current free memory. This
        // keeps full chunks on the existing >=256-token chunked GDN path.
        self.qwen_routed_prefill.then_some(256)
    }

    #[cfg(test)]
    pub(crate) fn validation_set_qwen_routed_prefill(
        &mut self,
        enabled: bool,
    ) {
        self.qwen_routed_prefill = enabled;
    }

    #[cfg(test)]
    pub(crate) fn validation_set_separate_prefill_last_token(
        &mut self,
        enabled: bool,
    ) {
        self.separate_prefill_last_token = enabled;
    }

    pub fn speculation_supported(&self) -> bool {
        !self.gemma4_image_embeddings && !self.scalar_image_embeddings && self.transformer.speculation_supported()
    }

    pub(crate) fn multimodal_supported(&self) -> bool {
        self.qwen35_multimodal_supported() || self.gemma4_multimodal_supported() || self.scalar_multimodal_supported()
    }

    pub(crate) fn qwen35_multimodal_supported(&self) -> bool {
        self.data_type == DataType::BF16
            && !self.gemma4_image_embeddings
            && !self.scalar_image_embeddings
            && self.embedding_norm.is_none()
            && self.per_layer_embedding.is_none()
            && self.transformer.multimodal_positions_supported()
    }

    pub(crate) fn gemma4_multimodal_supported(&self) -> bool {
        self.data_type == DataType::BF16
            && self.gemma4_image_embeddings
            && self.embedding_norm.is_none()
            && self.transformer.scalar_image_positions_supported()
    }

    pub(crate) fn scalar_multimodal_supported(&self) -> bool {
        self.data_type == DataType::BF16
            && self.scalar_image_embeddings
            && self.embedding_norm.is_none()
            && self.per_layer_embedding.is_none()
            && self.transformer.scalar_image_positions_supported()
    }

    pub fn max_context_length(&self) -> Option<u32> {
        self.transformer.max_context_length()
    }

    pub(crate) fn context_cache_buffer_bytes(
        &self,
        context_tokens: u32,
    ) -> Option<usize> {
        self.transformer.context_cache_buffer_bytes(context_tokens)
    }

    fn workspace_memory_estimate(
        &self,
        context_tokens: u32,
        input_tokens: u32,
    ) -> Option<crate::encodable_block::memory::MemoryEstimate> {
        let mut estimate = self.transformer.memory_estimate(context_tokens, input_tokens)?;
        // The engine's u64 history can grow through a doubled Vec capacity;
        // prefill also owns temporary u64 and retained u32 prompt copies.
        estimate.state_bytes = estimate
            .state_bytes
            .checked_add((context_tokens as usize).checked_mul(16)?)?
            .checked_add((input_tokens as usize).checked_mul(12)?)?;
        for rows in [input_tokens.clamp(1, 1024), 1] {
            estimate.add_scratch(&[rows, self.embedding.model_dim()], 4, 3)?;
            if let Some(embedding) = &self.per_layer_embedding {
                embedding.add_memory_scratch(rows, &mut estimate)?;
            }
        }
        Some(estimate)
    }

    pub(crate) fn memory_estimate(
        &self,
        context_tokens: u32,
        input_tokens: u32,
    ) -> Option<crate::encodable_block::memory::MemoryEstimate> {
        let mut estimate = self.workspace_memory_estimate(context_tokens, input_tokens)?;
        // Generation reads one final logit row during prefill, not a
        // prefill_rows*vocabulary matrix. Include sampler/readback workspace.
        estimate.add_scratch(&[self.embedding.vocab_size()], 4, 4)?;
        Some(estimate)
    }

    pub(crate) fn scoring_memory_estimate(
        &self,
        context_tokens: u32,
        input_tokens: u32,
    ) -> Option<crate::encodable_block::memory::MemoryEstimate> {
        let mut estimate = self.workspace_memory_estimate(context_tokens, input_tokens)?;
        let row = crate::encodable_block::memory::checked_bytes(&[self.embedding.vocab_size()], 4)?;
        // Only the final chunk produces a vocabulary row. Keep its device
        // readout, CPU conversion, candidate arrays and logits cache in one
        // request-wide allowance, outside the per-chunk scratch multiplier.
        estimate.add_state_buffers(row.checked_mul(16)?, row)?;
        Some(estimate)
    }

    pub fn prefill_cache_skips_trailing_layers(&self) -> bool {
        self.transformer.prefill_cache_skips_trailing_layers()
    }

    pub(crate) fn prefix_reuse_supported(&self) -> bool {
        self.transformer.prefix_reuse_supported()
    }

    pub(crate) fn prefix_snapshot_supported(&self) -> bool {
        self.transformer.prefix_snapshot_supported()
    }

    pub fn prefill_requires_separate_logits_row(&self) -> bool {
        self.separate_prefill_last_token || self.prefill_cache_skips_trailing_layers()
    }

    pub fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<TransformerState<B>, B::Error> {
        self.transformer.create_empty_state(max_context_length, context)
    }

    pub fn encode(
        &self,
        token_ids: impl BufferRef<Backend = B>,
        batch_dim: &BatchTopology,
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<DecoderEncodeOutput<B>, DecoderError<B>> {
        self.encode_with_features(
            token_ids,
            None,
            batch_dim,
            output_range,
            hidden_feature_layer_indices,
            state,
            command_buffer,
        )
    }

    pub(crate) fn encode_with_features(
        &self,
        token_ids: impl BufferRef<Backend = B>,
        replacement: Option<EmbeddingReplacement<'_>>,
        batch_dim: &BatchTopology,
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        state: &mut TransformerState<B>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<DecoderEncodeOutput<B>, DecoderError<B>> {
        // Validate byte spans before lookup, upload or transformer dispatch.
        if batch_dim.image_span().is_some() && (!self.bidirectional_image_attention || replacement.is_none()) {
            return Err(InvalidInput("image attention block lacks admitted embedding replacement").into());
        }
        let scalar_images = self.gemma4_image_embeddings || self.scalar_image_embeddings;
        let replacement = replacement
            .map(|replacement| {
                let width = self.embedding.model_dim() as usize;
                if !self.multimodal_supported()
                    || replacement.features.is_empty()
                    || !replacement.features.len().is_multiple_of(width)
                    || replacement.features.iter().any(|value| !value.is_finite())
                    || (scalar_images && batch_dim.rotary_positions().is_some())
                    || (!scalar_images && batch_dim.rotary_positions().is_none())
                    || !batch_dim.full_accept()
                    || !batch_dim.is_flat()
                {
                    return Err(InvalidInput("invalid multimodal embedding replacement"));
                }
                let row_bytes = width
                    .checked_mul(self.data_type.size_in_bytes())
                    .ok_or(InvalidInput("embedding row size overflow"))?;
                let start = (replacement.token_start as usize)
                    .checked_mul(row_bytes)
                    .ok_or(InvalidInput("embedding span offset overflow"))?;
                let bytes = replacement
                    .features
                    .len()
                    .checked_mul(self.data_type.size_in_bytes())
                    .ok_or(InvalidInput("embedding span size overflow"))?;
                let end = start.checked_add(bytes).ok_or(InvalidInput("embedding span end overflow"))?;
                let total = (batch_dim.size() as usize)
                    .checked_mul(row_bytes)
                    .ok_or(InvalidInput("embedding batch size overflow"))?;
                if end > total {
                    return Err(InvalidInput("embedding replacement exceeds batch"));
                }
                if self.bidirectional_image_attention {
                    let rows = u32::try_from(replacement.features.len() / width)
                        .map_err(|_| InvalidInput("image attention row count overflow"))?;
                    let span_end = replacement
                        .token_start
                        .checked_add(rows)
                        .ok_or(InvalidInput("image attention span overflow"))?;
                    if batch_dim.image_span() != Some([replacement.token_start, span_end]) {
                        return Err(InvalidInput("image attention block differs from projected image rows"));
                    }
                }
                Ok((replacement, start..end))
            })
            .transpose()?;
        let ple_image_span = replacement.as_ref().filter(|_| self.gemma4_image_embeddings).map(|(input, _)| {
            let first = input.token_start as usize;
            let rows = input.features.len() / self.embedding.model_dim() as usize;
            first..first + rows
        });
        command_buffer.push_debug_group("decoder");

        let mut embedded = self.embedding.encode_lookup(token_ids, batch_dim.size(), command_buffer)?;
        if let Some((replacement, range)) = replacement {
            // Owned command-buffer constants remain alive until the submitted
            // GPU work completes; no borrowed/no-copy image storage is retained.
            let features =
                command_buffer.allocate_constant_from_slice(replacement.features).map_err(DecoderError::Backend)?;
            command_buffer.encode_copy(&features, (&mut embedded).subrange_mut(range));
        }
        let embedded = if let Some(embedding_norm) = &self.embedding_norm {
            embedding_norm
                .encode(&embedded, 0, batch_dim.size(), None::<&mut B::ScratchBuffer>, command_buffer)
                .map_err(DecoderError::Backend)?
        } else {
            embedded
        };

        let per_layer_inputs = if let Some(per_layer_embedding) = &self.per_layer_embedding {
            Some(
                if let Some(span) = ple_image_span {
                    // Gemma's PLE table uses token 0 for image slots. The real
                    // token IDs remain untouched for embedding lookup, masks,
                    // physical KV offsets and subsequent text continuation.
                    // Behavior reference: MLX-VLM 53ebd1647cb83549b482a031ea70ba9d9ea60a04,
                    // gemma4/Model.get_input_embeddings and project_per_layer_inputs.
                    let ple_tokens = mask_image_ple_tokens(token_ids, span, batch_dim.size(), command_buffer)
                        .map_err(DecoderError::Backend)?;
                    per_layer_embedding.encode(&ple_tokens, &embedded, batch_dim.size(), command_buffer)
                } else {
                    per_layer_embedding.encode(token_ids, &embedded, batch_dim.size(), command_buffer)
                }
                .map_err(DecoderError::Backend)?,
            )
        } else {
            None
        };

        let transformer_output = self
            .transformer
            .encode(
                embedded,
                per_layer_inputs.as_ref(),
                batch_dim,
                output_range,
                hidden_feature_layer_indices,
                Some(state),
                command_buffer,
            )
            .map_err(DecoderError::Backend)?;

        let logits = if let Some(output_range) = output_range {
            let output = transformer_output.output.as_ref().expect("decoder output range requires transformer output");
            Some(self.embedding.encode_readout(
                output_range.end - output_range.start,
                output,
                self.embedding.vocab_size(),
                None::<&B::ScratchBuffer>,
                true,
                command_buffer,
            )?)
        } else {
            None
        };
        let final_hidden = if hidden_feature_layer_indices.is_none() {
            None
        } else {
            transformer_output.output
        };

        command_buffer.pop_debug_group();

        Ok(DecoderEncodeOutput {
            logits,
            hidden_features: transformer_output.hidden_features,
            final_hidden,
        })
    }
}

#[cfg(test)]
mod gemma_image_ple_tests {
    use stark_engine_macros::stark_test;

    use super::mask_image_ple_tokens;
    use crate::{
        backends::common::{
            Backend, Buffer, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
        },
        tests::helpers::{buffer_to_vec, create_buffer_with_data, create_context, for_each_backend},
    };

    fn check_image_only_mask<B: Backend>() {
        let context = create_context::<B>();
        let original = [7u32, 91, 90, 90, 90, 92, 8];
        let input = create_buffer_with_data::<B, u32>(&context, &original);
        let mut command = context.create_command_buffer(None, None).unwrap();
        let masked = mask_image_ple_tokens(&input, 2..5, original.len() as u32, &mut command).unwrap();
        assert!(mask_image_ple_tokens(&input, 2..8, original.len() as u32, &mut command).is_err());
        assert!(mask_image_ple_tokens(&input, 4..4, original.len() as u32, &mut command).is_err());
        let mut readback = context.create_buffer(masked.size()).unwrap();
        command.encode_copy(&masked, &mut readback);
        command.end_encoding().submit().wait_until_completed().unwrap();
        assert_eq!(buffer_to_vec::<B, u32>(&readback), [7, 91, 0, 0, 0, 92, 8]);
        assert_eq!(buffer_to_vec::<B, u32>(&input), original);
    }

    #[stark_test]
    fn gemma_image_ple_masks_only_image_rows_without_changing_decoder_tokens() {
        for_each_backend!(|B| check_image_only_mask::<B>());
    }
}
