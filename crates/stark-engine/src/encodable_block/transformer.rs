use std::range::Range;

use derive_more::Debug;
use thiserror::Error;

#[cfg(not(feature = "rope-cache"))]
use crate::encodable_block::mixer::attention::rope::PrecalculatedRoPE;
use crate::{
    backends::common::{
        Backend, Buffer, BufferRef, CommandBuffer, CommandBufferEncoding, InvalidInput, Kernels,
        kernel::TensorAddScaleKernel,
    },
    config::{rope::AnyRoPEConfig, transformer::TransformerConfig},
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        mixer::MixerState,
        normalization::{Normalization, NormalizationNewError, PostLayerScalar, ShortcutMode},
        transformer_layer::{TransformerLayer, TransformerLayerError},
    },
    parameters::ParameterTree,
    utils::maybe_mut::MaybeMut,
};

enum TransformerLayerStateType<B: Backend> {
    Owned(Box<dyn MixerState<B>>),
    Shared(u32),
}

pub struct TransformerState<B: Backend> {
    layer_states: Box<[TransformerLayerStateType<B>]>,
    context_length: u32,
    // Request-owned rotary continuation offset. Physical cache indices remain
    // context_length + trie height, including after spatial image positions.
    rotary_position_delta: Option<i64>,
    #[cfg(feature = "scoring-last-layer-pruning")]
    scoring_last_layer_pruning: bool,
}

pub struct TransformerEncodeOutput<B: Backend> {
    pub output: Option<B::ScratchBuffer>,
    pub hidden_features: Option<Box<[B::ScratchBuffer]>>,
}

/// Shift rotary positions after image prefill without changing physical KV indices.
pub(crate) fn scalar_continuation_position(
    cache_start: u32,
    relative_height: u32,
    delta: i64,
) -> Result<u32, InvalidInput> {
    let physical = cache_start.checked_add(relative_height).ok_or(InvalidInput("rotary cache position overflow"))?;
    i64::from(physical)
        .checked_add(delta)
        .and_then(|position| u32::try_from(position).ok())
        .ok_or(InvalidInput("rotary continuation position overflow"))
}

#[cfg(feature = "scoring-last-layer-pruning")]
fn scoring_cache_only_requested(
    output_range: Option<Range<u32>>,
    hidden_feature_layer_indices: Option<&[u32]>,
    batch_is_flat: bool,
    layer_index: u32,
    layer_count: usize,
    scoring_enabled: bool,
) -> bool {
    scoring_enabled
        && output_range.is_none()
        && hidden_feature_layer_indices.is_none()
        && batch_is_flat
        && layer_index as usize + 1 == layer_count
}

impl<B: Backend> TransformerState<B> {
    #[cfg(test)]
    pub(crate) fn attention_state_for_test(
        &self,
        layer_index: usize,
    ) -> Option<&crate::encodable_block::mixer::attention::AttentionState<B>> {
        match self.layer_states.get(layer_index)? {
            TransformerLayerStateType::Owned(state) => (state.as_ref() as &dyn std::any::Any).downcast_ref(),
            TransformerLayerStateType::Shared(source) => self.attention_state_for_test(*source as usize),
        }
    }

    /// Allow the final attention layer to skip its query path for prompt
    /// chunks whose logits are not read. Only `score_next_token` opts a state
    /// in; the pruning additionally requires a flat batch, no requested output
    /// range or hidden features, and an owned, non-ring full-attention final
    /// layer (see `scoring_cache_only_requested` and
    /// `TransformerLayer::supports_scoring_cache_only`).
    #[cfg(feature = "scoring-last-layer-pruning")]
    pub(crate) fn enable_scoring_last_layer_pruning(&mut self) {
        self.scoring_last_layer_pruning = true;
    }

    pub fn context_length(&self) -> u32 {
        self.context_length
    }

    pub(crate) fn set_rotary_position_delta(
        &mut self,
        delta: Option<i64>,
    ) {
        self.rotary_position_delta = delta;
    }

    pub(crate) fn rotary_position_delta(&self) -> Option<i64> {
        self.rotary_position_delta
    }

    pub fn prepare(
        &mut self,
        context_length: u32,
        suffix_length: u32,
    ) -> Result<(), B::Error> {
        for layer_state in &mut self.layer_states {
            let TransformerLayerStateType::Owned(layer_state) = layer_state else {
                continue;
            };

            layer_state.prepare(context_length, suffix_length)?;
        }

        Ok(())
    }

    pub fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        command_buffer.push_debug_group("transformer accept");

        for layer_state in &mut self.layer_states {
            let TransformerLayerStateType::Owned(layer_state) = layer_state else {
                continue;
            };

            layer_state.encode_accept(accepted_indices, command_buffer)?;
        }

        self.context_length += accepted_indices.len() as u32;

        command_buffer.pop_debug_group();

        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum TransformerNewError<B: Backend> {
    #[error(transparent)]
    InvalidInput(#[from] crate::backends::common::InvalidInput),
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Layer error: {0}")]
    Layer(#[from] TransformerLayerError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] NormalizationNewError<B>),
}

pub struct Transformer<B: Backend> {
    #[cfg(not(feature = "rope-cache"))]
    ropes: Box<[AnyRoPEConfig]>,
    #[cfg(feature = "rope-cache")]
    rope_tables: Box<[super::mixer::attention::rope::RoPETableCache]>,
    layers: Box<[(TransformerLayer<B>, Option<usize>)]>,
    output_norm: Normalization<B>,
    model_dim: u32,
    multimodal_positions_supported: bool,
    scalar_image_positions_supported: bool,
    residual_add: <B::Kernels as Kernels>::TensorAddScaleKernel,
}

impl<B: Backend> Transformer<B> {
    pub fn new(
        context: &B::Context,
        output_norm_hadamard_factors: Option<B::GlobalBuffer>,
        data_type: DataType,
        transformer_config: &TransformerConfig,
        parameter_tree: &ParameterTree<B>,
    ) -> Result<Self, TransformerNewError<B>> {
        crate::config::validation::validate_transformer(transformer_config)?;
        let mut ropes: Vec<AnyRoPEConfig> = Vec::new();

        let num_layers = transformer_config.layer_configs.len() as u32;
        let layers = transformer_config
            .layer_configs
            .iter()
            .zip(0u32..)
            .map(|(layer_config, layer_index)| {
                let layer_loader = parameter_tree.subtree(&format!("layers.{}", layer_index));

                let rope = layer_config.rope_config.as_ref().map(|layer_rope_config| {
                    ropes.iter().position(|rope_config| rope_config == layer_rope_config).unwrap_or_else(|| {
                        ropes.push(layer_rope_config.clone());
                        ropes.len() - 1
                    })
                });

                let layer = TransformerLayer::new(
                    context,
                    transformer_config.model_dim,
                    transformer_config.hidden_dim,
                    num_layers,
                    layer_config,
                    layer_index,
                    &layer_loader,
                    data_type,
                )?;

                Ok((layer, rope))
            })
            .collect::<Result<Box<[_]>, TransformerNewError<B>>>()?;

        let output_norm = Normalization::new(
            transformer_config.model_dim,
            output_norm_hadamard_factors,
            ShortcutMode::Add,
            PostLayerScalar::None,
            data_type,
            &transformer_config.output_norm_config,
            &parameter_tree.subtree("output_norm"),
            context,
        )?;

        let residual_add = <B::Kernels as Kernels>::TensorAddScaleKernel::new(context, data_type, false)
            .map_err(TransformerNewError::Backend)?;

        Ok(Self {
            multimodal_positions_supported: !ropes.is_empty()
                && ropes.iter().all(|rope| matches!(rope, AnyRoPEConfig::Qwen35MRoPEConfig(_))),
            scalar_image_positions_supported: !ropes.is_empty()
                && ropes.iter().all(|rope| {
                    matches!(rope, AnyRoPEConfig::UnscaledRoPEConfig(_) | AnyRoPEConfig::ProportionalRoPEConfig(_))
                })
                && transformer_config.layer_configs.iter().all(|layer| {
                    matches!(
                        &layer.mixer_config,
                        crate::config::token_mixer::AnyTokenMixerConfig::AttentionConfig(attention)
                            if attention.is_causal
                    ) && layer.rope_config.is_some()
                }),
            #[cfg(feature = "rope-cache")]
            rope_tables: ropes.iter().map(super::mixer::attention::rope::RoPETableCache::new).collect(),
            #[cfg(not(feature = "rope-cache"))]
            ropes: ropes.into_boxed_slice(),
            layers,
            output_norm,
            model_dim: transformer_config.model_dim,
            residual_add,
        })
    }

    fn capture_residual(
        &self,
        shortcut: impl BufferRef<Backend = B>,
        hidden: impl BufferRef<Backend = B>,
        batch_size: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let mut output = command_buffer.allocate_scratch(hidden.size())?;
        let elements = batch_size * self.model_dim;
        self.residual_add.encode(Some(shortcut), hidden, &mut output, elements, elements, 1.0, command_buffer);
        Ok(output)
    }

    pub fn speculation_supported(&self) -> bool {
        !self.multimodal_positions_supported
            && self.layers.iter().all(|(layer, _rope)| layer.mixer.speculation_supported())
    }

    pub(crate) fn multimodal_positions_supported(&self) -> bool {
        self.multimodal_positions_supported
    }

    pub(crate) fn scalar_image_positions_supported(&self) -> bool {
        self.scalar_image_positions_supported
    }

    pub fn max_context_length(&self) -> Option<u32> {
        self.layers.iter().map(|(layer, _rope_index)| layer.mixer.max_context_length()).fold(None, |acc, el| {
            match (acc, el) {
                (Some(a), Some(b)) => Some(u32::min(a, b)),
                (Some(x), None) | (None, Some(x)) => Some(x),
                (None, None) => None,
            }
        })
    }

    pub fn prefill_cache_layer_count(&self) -> usize {
        let num_layers = self.layers.len();
        let Some(last_owned_kv_layer_index) =
            self.layers.iter().rposition(|(layer, _rope_index)| layer.kv_source_layer_index.is_none())
        else {
            return num_layers;
        };

        last_owned_kv_layer_index + 1
    }

    pub fn prefill_cache_skips_trailing_layers(&self) -> bool {
        self.prefill_cache_layer_count() < self.layers.len()
    }

    pub fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<TransformerState<B>, B::Error> {
        let layer_states = self
            .layers
            .iter()
            .map(|(layer, _rope)| match layer.kv_source_layer_index {
                None => {
                    layer.mixer.create_empty_state(max_context_length, context).map(TransformerLayerStateType::Owned)
                },
                Some(kv_source_layer_index) => Ok(TransformerLayerStateType::Shared(kv_source_layer_index)),
            })
            .collect::<Result<_, B::Error>>()?;

        let context_length = 0;

        Ok(TransformerState {
            layer_states,
            context_length,
            rotary_position_delta: None,
            #[cfg(feature = "scoring-last-layer-pruning")]
            scoring_last_layer_pruning: false,
        })
    }

    pub fn encode(
        &self,
        input: B::ScratchBuffer,
        per_layer_inputs: Option<impl BufferRef<Backend = B>>,
        batch_dim: &BatchTopology,
        output_range: Option<Range<u32>>,
        hidden_feature_layer_indices: Option<&[u32]>,
        mut state: Option<&mut TransformerState<B>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<TransformerEncodeOutput<B>, B::Error> {
        let context_length = state.as_ref().map(|state| state.context_length).unwrap_or(0);
        let delta = state.as_ref().and_then(|state| state.rotary_position_delta);
        let explicit_positions = batch_dim.rotary_positions();
        if (explicit_positions.is_some() || delta.is_some())
            && (!self.multimodal_positions_supported || !batch_dim.is_flat() || !batch_dim.full_accept())
        {
            return Err(crate::backends::common::InvalidInput(
                "multimodal execution requires supported flat non-speculative positions",
            )
            .into());
        }
        let token_positions = batch_dim
            .heights()
            .map(|relative| {
                scalar_continuation_position(
                    context_length,
                    relative,
                    if explicit_positions.is_some() {
                        0
                    } else {
                        delta.unwrap_or(0)
                    },
                )
            })
            .collect::<Result<Box<[_]>, _>>()?;
        let mut hidden = input;
        let layer_count = if output_range.is_none() && hidden_feature_layer_indices.is_none() {
            self.prefill_cache_layer_count()
        } else {
            self.layers.len()
        };

        let mut shortcut = command_buffer.allocate_scratch(hidden.size())?;
        let mut hidden_features =
            hidden_feature_layer_indices.map(|indices| (0..indices.len()).map(|_| None).collect::<Vec<_>>());

        #[cfg(not(feature = "rope-cache"))]
        let precalculated_ropes = self
            .ropes
            .iter()
            .map(|rope_config| match explicit_positions {
                Some(positions) => PrecalculatedRoPE::precalculate_multimodal(rope_config, positions, command_buffer),
                None => PrecalculatedRoPE::precalculate(rope_config, &token_positions, command_buffer),
            })
            .collect::<Result<Box<[_]>, B::Error>>()?;

        #[cfg(feature = "rope-cache")]
        let precalculated_ropes = self
            .rope_tables
            .iter()
            .map(|cache| match explicit_positions {
                Some(positions) => cache.precalculate_multimodal::<B>(positions, command_buffer),
                None => cache.precalculate::<B>(&token_positions, command_buffer),
            })
            .collect::<Result<Box<[_]>, B::Error>>()?;

        for (layer, layer_rope_index) in self.layers.iter().take(layer_count) {
            let precalculated_rope = layer_rope_index.map(|i| &precalculated_ropes[i]);

            // Earlier layers must produce every row for the following layer's
            // state. Only the actual last layer can stop immediately after KV
            // preparation, and only when no output or hidden feature is read.
            #[cfg(feature = "scoring-last-layer-pruning")]
            if scoring_cache_only_requested(
                output_range,
                hidden_feature_layer_indices,
                batch_dim.is_flat(),
                layer.layer_index,
                self.layers.len(),
                state.as_ref().is_some_and(|state| state.scoring_last_layer_pruning),
            ) && layer.supports_scoring_cache_only()
                && let Some(state) = state.as_deref_mut()
                && let TransformerLayerStateType::Owned(layer_state) =
                    &mut state.layer_states[layer.layer_index as usize]
            {
                layer.encode_scoring_cache_only(
                    hidden,
                    &mut shortcut,
                    precalculated_rope,
                    batch_dim,
                    layer_state.as_mut(),
                    command_buffer,
                )?;
                return Ok(TransformerEncodeOutput {
                    output: None,
                    hidden_features: None,
                });
            }

            let layer_state = if let Some(state) = &mut state {
                Some(match &mut state.layer_states[layer.layer_index as usize] {
                    TransformerLayerStateType::Owned(layer_state) => MaybeMut::Mut(layer_state.as_mut()),
                    TransformerLayerStateType::Shared(owned_layer_index) => {
                        let TransformerLayerStateType::Owned(owned_layer) =
                            &state.layer_states[*owned_layer_index as usize]
                        else {
                            panic!("shared layer doesn't point to an owned layer");
                        };
                        MaybeMut::Const(owned_layer.as_ref())
                    },
                })
            } else {
                None
            };

            hidden = layer.encode(
                hidden,
                &mut shortcut,
                per_layer_inputs,
                precalculated_rope,
                batch_dim,
                layer_state,
                command_buffer,
            )?;

            if let (Some(hidden_features), Some(indices)) = (&mut hidden_features, hidden_feature_layer_indices) {
                for (feature_index, &layer_index) in indices.iter().enumerate() {
                    if layer_index == layer.layer_index {
                        let feature = self.capture_residual(&shortcut, &hidden, batch_dim.size(), command_buffer)?;
                        hidden_features[feature_index] = Some(feature);
                    }
                }
            }
        }

        let hidden_features = hidden_features.map(|hidden_features| {
            hidden_features
                .into_iter()
                .enumerate()
                .map(|(feature_index, feature)| {
                    feature.unwrap_or_else(|| {
                        panic!(
                            "requested hidden feature for missing layer {}",
                            hidden_feature_layer_indices.unwrap()[feature_index]
                        )
                    })
                })
                .collect::<Box<[_]>>()
        });

        let Some(output_range) = output_range else {
            return Ok(TransformerEncodeOutput {
                output: None,
                hidden_features,
            });
        };

        let output_normalized = self.output_norm.encode(
            &hidden,
            output_range.start,
            output_range.end - output_range.start,
            Some(&mut shortcut),
            command_buffer,
        )?;

        Ok(TransformerEncodeOutput {
            output: Some(output_normalized),
            hidden_features,
        })
    }
}

#[cfg(test)]
mod rotary_position_tests {
    use stark_engine_macros::stark_test;

    use super::scalar_continuation_position;

    #[stark_test]
    fn continuation_checks_signed_positions_without_changing_cache_coordinates() {
        let positions = [0, 2, 1].map(|height| scalar_continuation_position(12, height, -4).unwrap());
        assert_eq!(positions, [8, 10, 9]);
        let positions = [0, 1].map(|height| scalar_continuation_position(0, height, 3).unwrap());
        assert_eq!(positions, [3, 4]);
        assert_eq!(scalar_continuation_position(0, 0, 0).unwrap(), 0);
        assert_eq!(scalar_continuation_position(1, 0, -1).unwrap(), 0);
        assert_eq!(scalar_continuation_position(u32::MAX, 0, 0).unwrap(), u32::MAX);
        assert!(scalar_continuation_position(2, 0, -3).is_err());
        assert!(scalar_continuation_position(u32::MAX, 1, -1).is_err());
        assert!(scalar_continuation_position(u32::MAX, 0, 1).is_err());
        assert!(scalar_continuation_position(1, 0, i64::MAX).is_err());
        assert!(scalar_continuation_position(0, 0, i64::MIN).is_err());
    }
}

#[cfg(all(test, feature = "scoring-last-layer-pruning"))]
mod scoring_cache_only_tests {
    use stark_engine_macros::stark_test;

    use super::scoring_cache_only_requested;

    #[stark_test]
    fn cache_only_requires_an_unobserved_actual_final_layer() {
        assert!(scoring_cache_only_requested(None, None, true, 31, 32, true));
        // A final-layer output must be computed even if just one row is read.
        assert!(!scoring_cache_only_requested(Some((1023..1024).into()), None, true, 31, 32, true));
        assert!(!scoring_cache_only_requested(None, Some(&[31]), true, 31, 32, true));
        // Even an explicitly empty feature request conservatively disables it.
        assert!(!scoring_cache_only_requested(None, Some(&[]), true, 31, 32, true));
        // A last owned cache layer followed by shared-KV layers is not eligible.
        assert!(!scoring_cache_only_requested(None, None, true, 30, 32, true));
        assert!(!scoring_cache_only_requested(None, None, false, 31, 32, true));
        // Ordinary generation leaves the fresh-state opt-in flag clear.
        assert!(!scoring_cache_only_requested(None, None, true, 31, 32, false));
    }
}
