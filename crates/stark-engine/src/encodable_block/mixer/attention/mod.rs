use derive_more::Debug;
use thiserror::Error;

#[cfg(feature = "attention-kv-only")]
use crate::backends::common::kernel::AttentionPrepareKVOnlyKernel;
use crate::{
    backends::common::{
        Backend, CommandBuffer, CommandBufferEncoding, Context, DeviceCapabilities, Kernels,
        kernel::{AttentionKernel, AttentionKernelConfig, AttentionPrepareKernel, SigmoidGateKernel},
    },
    config::{
        rope::AnyRoPEConfig,
        token_mixer::attention::{AttentionConfig, ImageAttentionPolicy},
        weight_matrix::AnyWeightMatrixSpec,
    },
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::{Linear, LinearBlockError},
        mixer::{
            Mixer, MixerState,
            attention::{
                mode::LinearProjection,
                qkv_norm::{QKVNorm, QKVNormError},
                rope::PrecalculatedRoPE,
            },
        },
    },
    parameters::{ParameterLoaderError, ParameterTree},
    utils::maybe_mut::MaybeMut,
};

mod committed;
mod mode;
mod qkv_norm;
pub(crate) mod reference;
mod state;

pub(crate) use committed::AssistantAttentionHandoff;
pub use committed::CommittedAttentionKV;
pub use state::{ATTENTION_SUFFIX_CAPACITY, AttentionState, KVCacheView};

pub mod rope;

pub struct Attention<B: Backend> {
    sparse_page_bytes: Option<usize>,
    image_attention_policy: ImageAttentionPolicy,
    input_dim: u32,
    head_dim: u32,
    num_q_heads: u32,
    num_groups: u32,
    num_kv_heads: Option<u32>,
    ring_capacity: Option<u32>,
    max_rope_length: Option<u32>,
    data_type: DataType,
    projection_dim: u32,
    projection: LinearProjection<B>,
    source_rope: bool,
    source_proportional: bool,
    source_yarn: bool,
    prepare: <B::Kernels as Kernels>::AttentionPrepareKernel,
    #[cfg(feature = "attention-kv-only")]
    prepare_kv_only: Option<<B::Kernels as Kernels>::AttentionPrepareKVOnlyKernel>,
    sinks: Option<B::GlobalBuffer>,
    kernel: <B::Kernels as Kernels>::AttentionKernel,
    reference: Option<reference::ReferenceBF16Attention<B>>,
    gate_kernel: Option<<B::Kernels as Kernels>::SigmoidGateKernel>,
    out_projection: Box<dyn Linear<B>>,
}

#[derive(Debug, Error)]
pub enum AttentionNewError<B: Backend> {
    #[error(transparent)]
    InvalidInput(#[from] crate::backends::common::InvalidInput),
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("QKVNorm error: {0}")]
    QKVNorm(#[from] QKVNormError<B>),
}

impl<B: Backend> Attention<B> {
    pub fn new(
        hidden_dim: u32,
        data_type: DataType,
        rope_config: Option<&AnyRoPEConfig>,
        config: &AttentionConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<(Self, Option<B::GlobalBuffer>), AttentionNewError<B>> {
        crate::config::validation::validate_attention(config, rope_config)?;
        if config.reference_bf16 && data_type != DataType::BF16 {
            return Err(crate::backends::common::InvalidInput("reference_bf16 requires BF16 activations").into());
        }
        if config.materialize_rope_bf16 && (data_type != DataType::BF16 || rope_config.is_none()) {
            return Err(
                crate::backends::common::InvalidInput("BF16 rotary materialization requires BF16 and RoPE").into()
            );
        }
        if config.materialize_prefill_scores && (config.reference_bf16 || config.logit_soft_cap.is_some()) {
            return Err(
                crate::backends::common::InvalidInput("incompatible source prefill attention configuration").into()
            );
        }
        let reference = config
            .reference_bf16
            .then(|| reference::ReferenceBF16Attention::<B>::new(context, config))
            .transpose()
            .map_err(AttentionNewError::Backend)?;
        let is_kv_sharing = config.is_kv_sharing;

        let head_dim = config.head_dim;
        let num_groups = config.num_groups;
        let num_q_heads = config.num_heads;
        let num_kv_heads = (!is_kv_sharing).then_some(num_groups);

        let is_causal = config.is_causal;
        let sliding_window_size = config.sliding_window_size;
        let max_rope_length = rope_config.map(|rope_config| *rope_config.max_sequence_length());

        let q_dim = crate::config::validation::checked_product(&[num_q_heads, head_dim])?;

        let projection_tree = parameter_tree.subtree("qkvg_projection");
        let source_profile = data_type == DataType::BF16
            && !config.reference_bf16
            && matches!(projection_tree.subtree("weights").metadata::<AnyWeightMatrixSpec>("spec")?,
                AnyWeightMatrixSpec::MLXSpec(spec) if spec.materialize_affine_input_sum);
        let source_rope = source_profile
            && matches!(
                rope_config,
                Some(
                    AnyRoPEConfig::UnscaledRoPEConfig(_)
                        | AnyRoPEConfig::ProportionalRoPEConfig(_)
                        | AnyRoPEConfig::YARNRoPEConfig(_)
                )
            );
        let source_yarn = source_rope && matches!(rope_config, Some(AnyRoPEConfig::YARNRoPEConfig(_)));
        let source_proportional =
            source_rope && (source_yarn || matches!(rope_config, Some(AnyRoPEConfig::ProportionalRoPEConfig(_))));
        if source_rope && config.materialize_rope_bf16 {
            return Err(crate::backends::common::InvalidInput("conflicting rotary precision policies").into());
        }
        let qkv_dim = if let Some(num_kv_heads) = num_kv_heads {
            let kv_dim = crate::config::validation::checked_product(&[num_kv_heads, head_dim])?;
            q_dim
                .checked_add(kv_dim)
                .and_then(|dim| dim.checked_add(kv_dim))
                .ok_or(crate::backends::common::InvalidInput("attention projection overflow"))?
        } else {
            q_dim
        };
        let projection_dim = if config.has_gate {
            qkv_dim
                .checked_add(q_dim)
                .ok_or(crate::backends::common::InvalidInput("attention gate projection overflow"))?
        } else {
            qkv_dim
        };
        let mut projection_widths = vec![q_dim];
        if let Some(num_kv_heads) = num_kv_heads {
            let kv_dim = crate::config::validation::checked_product(&[num_kv_heads, head_dim])?;
            projection_widths.extend([kv_dim, kv_dim]);
        }
        if config.has_gate {
            projection_widths.push(q_dim);
        }
        let (projection, in_projection_input_hadamard_factors) = <dyn Linear<B>>::new_with_input_rht(
            hidden_dim,
            &projection_widths,
            config.has_qkvg_biases,
            context,
            data_type,
            &projection_tree,
        )?;

        let query_norm_config = config.query_norm_config.clone();
        // TODO: Fix lalamo config, those two must be None if kv sharing.
        let key_norm_config = (!is_kv_sharing).then(|| config.key_norm_config.clone()).flatten();
        let value_norm_config = (!is_kv_sharing).then(|| config.value_norm_config()).flatten();
        let qkv_norm = (query_norm_config.is_some() || key_norm_config.is_some() || value_norm_config.is_some())
            .then(|| {
                QKVNorm::new(
                    context,
                    data_type,
                    query_norm_config,
                    key_norm_config,
                    value_norm_config,
                    parameter_tree,
                    config.num_heads,
                    num_kv_heads.unwrap_or(0), // TODO: should take option
                    projection_dim,
                    config.head_dim,
                    config.qk_norm_full_width,
                )
            })
            .transpose()?;

        let prepare = <B::Kernels as Kernels>::AttentionPrepareKernel::new(
            context,
            data_type,
            DataType::F32,
            !is_kv_sharing,
            rope_config.is_some(),
            config.reference_bf16 || config.materialize_rope_bf16,
            source_rope,
            source_proportional,
            source_yarn,
        )
        .map_err(AttentionNewError::Backend)?;
        let sinks = config
            .has_sinks
            .then(|| parameter_tree.leaf("sinks")?.validate(&[num_q_heads], data_type)?.read_buffer())
            .transpose()?;

        assert!(sliding_window_size.is_none_or(|size| size > 0), "zero sliding window size");
        let ring_capacity = sliding_window_size;
        let is_kv_cache_ring = ring_capacity.is_some();

        #[cfg(feature = "attention-kv-only")]
        let prepare_kv_only = (!is_kv_sharing && !is_kv_cache_ring)
            .then(|| {
                <B::Kernels as Kernels>::AttentionPrepareKVOnlyKernel::new(
                    context,
                    data_type,
                    DataType::F32,
                    rope_config.is_some(),
                    config.reference_bf16 || config.materialize_rope_bf16,
                    source_rope,
                    source_proportional,
                    source_yarn,
                )
            })
            .transpose()
            .map_err(AttentionNewError::Backend)?;

        let kernel = <B::Kernels as Kernels>::AttentionKernel::new(
            context,
            AttentionKernelConfig {
                head_dim,
                num_groups,
                num_q_heads,
                has_sinks: sinks.is_some(),
                is_kv_cache_ring,
                is_causal,
                sliding_window_size,
                scale: config.scale,
                data_type,
                materialize_prefill_scores: config.materialize_prefill_scores,
                source_bf16_prefill: source_profile && matches!(head_dim, 64 | 128),
            },
        )
        .map_err(AttentionNewError::Backend)?;

        let source_bf16_gate =
            parameter_tree.numerical_profile() == crate::parameters::NumericalProfile::QwenMtpSourceBF16;
        #[cfg(test)]
        let source_bf16_gate = source_bf16_gate
            || (parameter_tree.numerical_profile() == crate::parameters::NumericalProfile::QwenMtpReferenceLinears
                && std::env::var_os("STARK_TEST_HEAD_MLX_OPERATORS").is_some());
        if source_bf16_gate && (data_type != DataType::BF16 || config.reference_bf16) {
            return Err(crate::backends::common::InvalidInput("conflicting head gate experiment policies").into());
        }
        let gate_kernel = config
            .has_gate
            .then(|| {
                <B::Kernels as Kernels>::SigmoidGateKernel::new(
                    context,
                    data_type,
                    config.reference_bf16,
                    source_bf16_gate,
                )
            })
            .transpose()
            .map_err(AttentionNewError::Backend)?;

        let out_projection = <dyn Linear<B>>::new(
            q_dim,
            [hidden_dim],
            config.has_out_biases,
            context,
            data_type,
            &parameter_tree.subtree("out_projection"),
        )?;

        Ok((
            Self {
                sparse_page_bytes: context
                    .device_capabilities()
                    .contains(DeviceCapabilities::SPARSE_BUFFERS)
                    .then(|| context.sparse_page_size())
                    .flatten(),
                image_attention_policy: config.image_attention_policy,
                input_dim: hidden_dim,
                head_dim,
                num_q_heads,
                num_kv_heads,
                num_groups,
                ring_capacity,
                max_rope_length,
                data_type,
                projection_dim,
                source_rope,
                source_proportional,
                source_yarn,
                projection: LinearProjection {
                    lin: projection,
                    norm: qkv_norm,
                },
                prepare,
                #[cfg(feature = "attention-kv-only")]
                prepare_kv_only,
                sinks,
                kernel,
                reference,
                gate_kernel,
                out_projection,
            },
            in_projection_input_hadamard_factors,
        ))
    }
}

impl<B: Backend> Mixer<B> for Attention<B> {
    fn speculation_supported(&self) -> bool {
        true
    }

    fn max_context_length(&self) -> Option<u32> {
        self.max_rope_length
    }

    fn memory_estimate(
        &self,
        context_tokens: u32,
        input_tokens: u32,
    ) -> Option<crate::encodable_block::memory::MemoryEstimate> {
        self.memory_estimate_with_prefix(context_tokens, input_tokens, input_tokens, false)
    }

    fn memory_estimate_with_prefix(
        &self,
        context_tokens: u32,
        input_tokens: u32,
        sequence_tokens: u32,
        irregular_prefill: bool,
    ) -> Option<crate::encodable_block::memory::MemoryEstimate> {
        if self.max_rope_length.is_some_and(|maximum| context_tokens > maximum) {
            return None;
        }
        let mut estimate = crate::encodable_block::memory::attention_estimate_with_prefix(
            context_tokens,
            input_tokens,
            sequence_tokens,
            self.head_dim,
            self.num_q_heads,
            self.num_kv_heads,
            self.ring_capacity,
            self.projection_dim,
            self.data_type.size_in_bytes(),
            self.reference.is_some(),
        )?;
        if let Some(page_bytes) = self.sparse_page_bytes {
            let physical = crate::encodable_block::memory::attention_physical_state_bytes(
                context_tokens,
                input_tokens,
                sequence_tokens,
                irregular_prefill,
                self.head_dim,
                self.num_kv_heads,
                self.ring_capacity,
                self.data_type.size_in_bytes(),
                page_bytes,
            )?;
            estimate.set_state_buffers(physical, physical / 2);
        }
        Some(estimate)
    }

    fn context_cache_buffer_bytes(
        &self,
        context_tokens: u32,
    ) -> Option<usize> {
        if self.max_rope_length.is_some_and(|maximum| context_tokens > maximum) {
            return None;
        }
        crate::encodable_block::memory::attention_cache_buffer_bytes(
            context_tokens,
            self.head_dim,
            self.num_kv_heads,
            self.ring_capacity,
            self.data_type.size_in_bytes(),
        )
    }

    fn prefix_reuse_supported(&self) -> bool {
        self.num_kv_heads.is_some() && self.ring_capacity.is_none()
    }

    fn prefix_snapshot_supported(&self) -> bool {
        self.num_kv_heads.is_some()
    }

    fn create_empty_state(
        &self,
        max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<Box<dyn MixerState<B>>, B::Error> {
        Ok(Box::new(AttentionState::create_empty(self, max_context_length, context)?))
    }

    fn encode(
        &self,
        hidden: B::ScratchBuffer,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        command_buffer.push_debug_group("attention");

        assert_eq!(precalculated_rope.is_some(), self.max_rope_length.is_some(), "precalculated rope mismatch");

        let state =
            state.map(|state| state.downcast::<AttentionState<B>>().expect("incorrect type of attention state"));
        let output = self.attend(hidden, precalculated_rope, batch_dim, state, command_buffer)?;

        command_buffer.pop_debug_group();

        Ok(output)
    }
}
