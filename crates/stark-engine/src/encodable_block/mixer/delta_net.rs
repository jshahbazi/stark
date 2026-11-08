use derive_more::Debug;
use thiserror::Error;

#[cfg(feature = "direct-delta-conv")]
use crate::backends::common::kernel::DeltaNetConvDirectKernel;
#[cfg(not(feature = "deferred-state-init"))]
use crate::backends::common::{CommandBufferExecutable, CommandBufferPending};
use crate::{
    array::size_for_shape,
    backends::common::{
        Backend, Buffer, BufferMut, BufferRef, CommandBuffer, CommandBufferEncoding, Context, Kernels,
        kernel::{
            Conv1dPackKernel, ConvTreeScanKernel, DeltaNetConvScanKernel, DeltaNetConvUpdateKernel,
            DeltaNetNormGateKernel, DeltaNetPrefillKernel, DeltaNetPrefillPrepKernel, DeltaNetUpdateKernel,
            ReferenceGDNConvBF16Kernel, ReferenceGDNDecayKernel, ReferenceGDNNormGateKernel, ReferenceGDNQKL2Kernel,
            StateAdvanceKernel,
            delta_net_chunked_prefill::{DeltaNetChunkedPrefill, DeltaNetChunkedPrefillArgs},
            delta_net_tree_verify::DeltaNetTreeVerify as DeltaNetTreeVerifyTrait,
        },
    },
    config::token_mixer::delta_net::DeltaNetConfig,
    data_type::DataType,
    encodable_block::{
        batch_topology::BatchTopology,
        linear::{Linear, LinearBlockError},
        mixer::{
            Mixer, MixerState, MixerStateSnapshot,
            attention::rope::PrecalculatedRoPE,
            delta_net::tree_verify::{TreeVerifyEncodeArguments, TreeVerifyNewArguments},
        },
    },
    parameters::{ParameterLoaderError, ParameterTree},
    utils::maybe_mut::MaybeMut,
};

#[cfg(test)]
mod prefix_tests;
mod reference_chunk;
#[cfg(test)]
mod reference_tests;
#[cfg(all(test, backend = "metal"))]
mod short_chain_tests;
pub(crate) mod tree_verify;

const INNER_DATA_TYPE: DataType = DataType::F32;

enum DeltaNetSuffixStatus<B: Backend> {
    Flat {
        suffix_length: u32,
    },
    Tree {
        conv_states: B::ScratchBuffer,
        k: B::ScratchBuffer,
        v: B::ScratchBuffer,
        log_decay: B::ScratchBuffer,
        beta: B::ScratchBuffer,
        parents: Box<[i32]>,
    },
}

pub struct DeltaNetState<B: Backend> {
    conv_state: B::GlobalBuffer,
    ssm_state: B::GlobalBuffer,
    suffix_status: Option<DeltaNetSuffixStatus<B>>,
    #[cfg(feature = "deferred-state-init")]
    needs_zero_initialization: bool,
    state_advance: <B::Kernels as Kernels>::StateAdvanceKernel,
}

struct DeltaNetPrefix<B: Backend> {
    conv: B::GlobalBuffer,
    ssm: B::GlobalBuffer,
}

impl<B: Backend> MixerStateSnapshot<B> for DeltaNetPrefix<B> {
    fn can_restore(
        &self,
        state: &dyn MixerState<B>,
    ) -> bool {
        (state as &dyn std::any::Any).downcast_ref::<DeltaNetState<B>>().is_some_and(|state| {
            state.conv_state.size() == self.conv.size() && state.ssm_state.size() == self.ssm.size()
        })
    }
    fn restore(
        &self,
        state: &mut dyn MixerState<B>,
        command: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> bool {
        let Some(state) = (state as &mut dyn std::any::Any).downcast_mut::<DeltaNetState<B>>() else {
            return false;
        };
        command.encode_copy(&self.conv, &mut state.conv_state);
        command.encode_copy(&self.ssm, &mut state.ssm_state);
        state.suffix_status = None;
        #[cfg(feature = "deferred-state-init")]
        {
            state.needs_zero_initialization = false;
        }
        true
    }
    fn payload_bytes(&self) -> usize {
        self.conv.size() + self.ssm.size()
    }
}

impl<B: Backend> MixerState<B> for DeltaNetState<B> {
    fn capture_prefix(
        &self,
        _position: u32,
        context: &B::Context,
        command: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Option<Box<dyn MixerStateSnapshot<B>>>, B::Error> {
        // Only a committed state can be retained. Tree scratch contains
        // speculative rows and must have been consumed by encode_accept.
        if self.suffix_status.is_some() {
            return Ok(None);
        }
        #[cfg(feature = "deferred-state-init")]
        if self.needs_zero_initialization {
            return Ok(None);
        }
        let mut conv = context.create_buffer(self.conv_state.size())?;
        let mut ssm = context.create_buffer(self.ssm_state.size())?;
        command.encode_copy(&self.conv_state, &mut conv);
        command.encode_copy(&self.ssm_state, &mut ssm);
        Ok(Some(Box::new(DeltaNetPrefix {
            conv,
            ssm,
        })))
    }
    fn prepare(
        &mut self,
        _context_length: u32,
        _suffix_length: u32,
    ) -> Result<(), B::Error> {
        Ok(())
    }

    fn encode_accept(
        &mut self,
        accepted_indices: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let suffix_status = self.suffix_status.take().expect("delta net state has no suffix to accept");
        let accepted_index = *accepted_indices.last().expect("delta net state attempted to accept zero indices");

        match suffix_status {
            DeltaNetSuffixStatus::Flat {
                suffix_length,
            } => {
                assert!(accepted_index == suffix_length - 1, "attempted a partial flat delta net accept");
            },
            DeltaNetSuffixStatus::Tree {
                conv_states,
                k,
                v,
                log_decay,
                beta,
                parents,
            } => {
                assert!(accepted_indices.iter().all(|&index| (index as usize) < parents.len()));
                assert_eq!(parents[accepted_indices[0] as usize], -1);
                assert!(accepted_indices.windows(2).all(|edge| parents[edge[1] as usize] == edge[0] as i32));

                let conv_state_size = self.conv_state.size();
                let accepted_offset = accepted_index as usize * conv_state_size;
                command_buffer.encode_copy(
                    conv_states.subrange(accepted_offset..accepted_offset + conv_state_size),
                    &mut self.conv_state,
                );

                let accepted_indices_buffer = command_buffer.allocate_constant_from_slice(accepted_indices)?;
                self.state_advance.encode(
                    &k,
                    &v,
                    &log_decay,
                    &beta,
                    &accepted_indices_buffer,
                    &mut self.ssm_state,
                    accepted_indices.len() as u32,
                    command_buffer,
                );
            },
        }
        Ok(())
    }
}

pub struct DeltaNet<B: Backend> {
    num_heads: u32,
    head_dim: u32,
    num_groups: u32,
    value_head_dim: u32,
    key_dim: u32,
    value_dim: u32,
    conv_dim: u32,
    total_proj_dim: u32,
    kernel_size: u32,
    outer_data_type: DataType,
    reference_bf16: bool,
    reference_l2: Option<<B::Kernels as Kernels>::ReferenceGDNQKL2Kernel>,
    reference_norm_gate: Option<<B::Kernels as Kernels>::ReferenceGDNNormGateKernel>,
    reference_chunk: Option<reference_chunk::ReferenceChunk<B>>,
    reference_log_prep: Option<<B::Kernels as Kernels>::DeltaNetPrefillPrepKernel>,
    reference_decay: Option<<B::Kernels as Kernels>::ReferenceGDNDecayKernel>,
    reference_conv: Option<<B::Kernels as Kernels>::ReferenceGDNConvBF16Kernel>,
    in_projection: Box<dyn Linear<B>>,
    conv_weight: B::GlobalBuffer,
    conv_bias: Option<B::GlobalBuffer>,
    conv_update: <B::Kernels as Kernels>::DeltaNetConvUpdateKernel,
    // Retained and initialized for the original indirect-kernel lifetime even
    // when the direct route is selected; the optimized route bypasses encoding.
    #[cfg_attr(feature = "direct-delta-conv", allow(dead_code))]
    conv_pack: <B::Kernels as Kernels>::Conv1dPackKernel,
    #[cfg_attr(feature = "direct-delta-conv", allow(dead_code))]
    conv_scan: <B::Kernels as Kernels>::DeltaNetConvScanKernel,
    #[cfg(feature = "direct-delta-conv")]
    conv_direct: <B::Kernels as Kernels>::DeltaNetConvDirectKernel,
    conv_tree_scan: <B::Kernels as Kernels>::ConvTreeScanKernel,
    a_log: B::GlobalBuffer,
    dt_bias: B::GlobalBuffer,
    norm_weight: B::GlobalBuffer,
    norm_epsilon: f32,
    delta_net_update: <B::Kernels as Kernels>::DeltaNetUpdateKernel,
    delta_net_prefill_prep: <B::Kernels as Kernels>::DeltaNetPrefillPrepKernel,
    delta_net_tree_prep: <B::Kernels as Kernels>::DeltaNetPrefillPrepKernel,
    delta_net_prefill: <B::Kernels as Kernels>::DeltaNetPrefillKernel,
    delta_net_norm_gate: <B::Kernels as Kernels>::DeltaNetNormGateKernel,
    tree_verify: Option<<B::Kernels as Kernels>::DeltaNetTreeVerify>,
    chunked: Option<<B::Kernels as Kernels>::DeltaNetChunkedPrefill>,
    out_projection: Box<dyn Linear<B>>,
}

#[derive(Debug, Error)]
pub enum DeltaNetNewError<B: Backend> {
    #[error("Backend error: {0}")]
    Backend(#[source] B::Error),
    #[error("Parameter loader error: {0}")]
    ParameterLoader(#[from] ParameterLoaderError<B>),
    #[error("Linear error: {0}")]
    Linear(#[from] LinearBlockError<B>),
    #[error("Unsupported configuration: {0}")]
    UnsupportedConfiguration(String),
}

impl<B: Backend> DeltaNet<B> {
    pub fn new(
        hidden_dim: u32,
        outer_data_type: DataType,
        config: &DeltaNetConfig,
        parameter_tree: &ParameterTree<B>,
        context: &B::Context,
    ) -> Result<(Self, Option<B::GlobalBuffer>), DeltaNetNewError<B>> {
        if config.reference_bf16 && outer_data_type != DataType::BF16 {
            return Err(DeltaNetNewError::UnsupportedConfiguration("reference_bf16 requires BF16 activations".into()));
        }
        if config.reference_bf16 && config.conv_config.has_biases {
            return Err(DeltaNetNewError::UnsupportedConfiguration(
                "reference BF16 DeltaNet requires unbiased convolution".into(),
            ));
        }
        if config.reference_bf16 && (config.head_dim != 128 || config.value_head_dim != 128) {
            return Err(DeltaNetNewError::UnsupportedConfiguration(
                "reference BF16 DeltaNet requires 128-wide key/value heads".into(),
            ));
        }
        let reference_l2 = config
            .reference_bf16
            .then(|| <B::Kernels as Kernels>::ReferenceGDNQKL2Kernel::new(context))
            .transpose()
            .map_err(DeltaNetNewError::Backend)?;
        let reference_norm_gate = config
            .reference_bf16
            .then(|| <B::Kernels as Kernels>::ReferenceGDNNormGateKernel::new(context))
            .transpose()
            .map_err(DeltaNetNewError::Backend)?;
        let reference_chunk = config
            .reference_bf16
            .then(|| reference_chunk::ReferenceChunk::new(context))
            .transpose()
            .map_err(DeltaNetNewError::Backend)?;
        let reference_log_prep = config
            .reference_bf16
            .then(|| {
                <B::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
                    context,
                    outer_data_type,
                    DataType::F32,
                    config.head_dim,
                    true,
                    false,
                )
            })
            .transpose()
            .map_err(DeltaNetNewError::Backend)?;
        let reference_decay = config
            .reference_bf16
            .then(|| <B::Kernels as Kernels>::ReferenceGDNDecayKernel::new(context))
            .transpose()
            .map_err(DeltaNetNewError::Backend)?;
        let reference_conv = config
            .reference_bf16
            .then(|| <B::Kernels as Kernels>::ReferenceGDNConvBF16Kernel::new(context))
            .transpose()
            .map_err(DeltaNetNewError::Backend)?;
        crate::config::validation::validate_heads(config.num_heads, config.num_groups, config.head_dim)
            .map_err(|_| DeltaNetNewError::UnsupportedConfiguration("invalid head/group geometry".into()))?;
        if config.kernel_size < 2 {
            return Err(DeltaNetNewError::UnsupportedConfiguration(format!(
                "kernel_size must be >= 2, got {}",
                config.kernel_size
            )));
        }
        if config.head_dim != 128 {
            return Err(DeltaNetNewError::UnsupportedConfiguration(format!(
                "head_dim must be 128, got {}",
                config.head_dim
            )));
        }
        if config.value_head_dim != 128 {
            return Err(DeltaNetNewError::UnsupportedConfiguration(format!(
                "value_head_dim must be 128, got {}",
                config.value_head_dim
            )));
        }

        let geometry_error = || DeltaNetNewError::UnsupportedConfiguration("projection dimension overflow".into());
        let key_dim = config.num_groups.checked_mul(config.head_dim).ok_or_else(geometry_error)?;
        let value_dim = config.num_heads.checked_mul(config.value_head_dim).ok_or_else(geometry_error)?;
        let conv_dim = key_dim.checked_mul(2).and_then(|dim| dim.checked_add(value_dim)).ok_or_else(geometry_error)?;
        let total_proj_dim = conv_dim
            .checked_add(value_dim)
            .and_then(|dim| dim.checked_add(config.num_heads))
            .and_then(|dim| dim.checked_add(config.num_heads))
            .ok_or_else(geometry_error)?;

        let (in_projection, in_projection_input_hadamard_factors) =
            <dyn Linear<B>>::new_with_input_rht_mixed_precision(
                hidden_dim,
                [total_proj_dim],
                false,
                context,
                outer_data_type,
                outer_data_type,
                outer_data_type,
                &parameter_tree.subtree("in_proj"),
            )?;

        let conv_config = &config.conv_config;
        let conv_tree = parameter_tree.subtree("conv");

        let conv_weight =
            conv_tree.leaf("weights")?.validate(&[conv_dim, config.kernel_size], INNER_DATA_TYPE)?.read_buffer()?;
        let conv_bias = if conv_config.has_biases {
            Some(conv_tree.leaf("biases")?.validate(&[conv_dim], INNER_DATA_TYPE)?.read_buffer()?)
        } else {
            None
        };
        let conv_update =
            <B::Kernels as Kernels>::DeltaNetConvUpdateKernel::new(context, outer_data_type, conv_config.has_biases)
                .map_err(DeltaNetNewError::Backend)?;
        let conv_pack = <B::Kernels as Kernels>::Conv1dPackKernel::new(context, INNER_DATA_TYPE, outer_data_type)
            .map_err(DeltaNetNewError::Backend)?;
        let conv_scan =
            <B::Kernels as Kernels>::DeltaNetConvScanKernel::new(context, outer_data_type, conv_config.has_biases)
                .map_err(DeltaNetNewError::Backend)?;
        #[cfg(feature = "direct-delta-conv")]
        let conv_direct =
            <B::Kernels as Kernels>::DeltaNetConvDirectKernel::new(context, outer_data_type, conv_config.has_biases)
                .map_err(DeltaNetNewError::Backend)?;
        let conv_tree_scan = <B::Kernels as Kernels>::ConvTreeScanKernel::new(
            context,
            outer_data_type,
            config.kernel_size,
            conv_config.has_biases,
        )
        .map_err(DeltaNetNewError::Backend)?;

        let a_log = parameter_tree.leaf("a_log")?.validate(&[config.num_heads], INNER_DATA_TYPE)?.read_buffer()?;
        let dt_bias = parameter_tree.leaf("dt_bias")?.validate(&[config.num_heads], INNER_DATA_TYPE)?.read_buffer()?;
        let norm_weight =
            parameter_tree.leaf("norm.scales")?.validate(&[config.value_head_dim], INNER_DATA_TYPE)?.read_buffer()?;
        let norm_epsilon = config.norm_config.epsilon;
        let delta_net_update =
            <B::Kernels as Kernels>::DeltaNetUpdateKernel::new(context, outer_data_type, config.head_dim)
                .map_err(DeltaNetNewError::Backend)?;
        let delta_net_prefill_prep = <B::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
            context,
            outer_data_type,
            INNER_DATA_TYPE,
            config.head_dim,
            false,
            false,
        )
        .map_err(DeltaNetNewError::Backend)?;
        let delta_net_tree_prep = <B::Kernels as Kernels>::DeltaNetPrefillPrepKernel::new(
            context,
            outer_data_type,
            outer_data_type,
            config.head_dim,
            true,
            true,
        )
        .map_err(DeltaNetNewError::Backend)?;
        let delta_net_prefill =
            <B::Kernels as Kernels>::DeltaNetPrefillKernel::new(context, outer_data_type, config.head_dim)
                .map_err(DeltaNetNewError::Backend)?;
        let delta_net_norm_gate = <B::Kernels as Kernels>::DeltaNetNormGateKernel::new(context, outer_data_type)
            .map_err(DeltaNetNewError::Backend)?;
        let tree_verify = if !config.reference_bf16
            && <<B::Kernels as Kernels>::DeltaNetTreeVerify as DeltaNetTreeVerifyTrait>::is_supported(context)
        {
            Some(
                <<B::Kernels as Kernels>::DeltaNetTreeVerify as DeltaNetTreeVerifyTrait>::new(
                    context,
                    &TreeVerifyNewArguments {
                        data_type: outer_data_type,
                        num_k_heads: config.num_groups,
                        num_v_heads: config.num_heads,
                        head_k_dim: config.head_dim,
                        head_v_dim: config.value_head_dim,
                    },
                )
                .map_err(DeltaNetNewError::Backend)?,
            )
        } else {
            None
        };

        // The optimized chunk solver intentionally uses extra BF16 scratch
        // planes. The reference policy keeps the recurrent arithmetic in F32.
        let chunked = if config.reference_bf16 {
            None
        } else {
            <B::Kernels as Kernels>::DeltaNetChunkedPrefill::new(context, outer_data_type, config.head_dim)
                .map_err(DeltaNetNewError::Backend)?
        };

        let out_projection = <dyn Linear<B>>::new_mixed_precision(
            value_dim,
            [hidden_dim],
            false,
            context,
            outer_data_type,
            outer_data_type,
            outer_data_type,
            &parameter_tree.subtree("out_proj"),
        )?;

        Ok((
            Self {
                num_heads: config.num_heads,
                head_dim: config.head_dim,
                num_groups: config.num_groups,
                value_head_dim: config.value_head_dim,
                key_dim,
                value_dim,
                conv_dim,
                total_proj_dim,
                kernel_size: config.kernel_size,
                outer_data_type,
                reference_bf16: config.reference_bf16,
                reference_l2,
                reference_norm_gate,
                reference_chunk,
                reference_log_prep,
                reference_decay,
                reference_conv,
                in_projection,
                conv_weight,
                conv_bias,
                conv_update,
                conv_pack,
                conv_scan,
                #[cfg(feature = "direct-delta-conv")]
                conv_direct,
                conv_tree_scan,
                a_log,
                dt_bias,
                norm_weight,
                norm_epsilon,
                delta_net_update,
                delta_net_prefill_prep,
                delta_net_tree_prep,
                delta_net_prefill,
                delta_net_norm_gate,
                tree_verify,
                chunked,
                out_projection,
            },
            in_projection_input_hadamard_factors,
        ))
    }

    fn encode_tree_verify(
        &self,
        in_projected: impl BufferRef<Backend = B>,
        batch_dim: &BatchTopology,
        state: &mut DeltaNetState<B>,
        project_output: bool,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let tree_verify = self.tree_verify.as_ref().expect("DeltaNet tree verification is unsupported");
        let tree_size = batch_dim.size();
        let parents = command_buffer.allocate_constant_from_slice(batch_dim.parents())?;
        let trie = command_buffer.allocate_constant_from_slice(batch_dim.nodes())?;

        let mut conv_states = command_buffer
            .allocate_scratch_for_shape(&[tree_size, self.conv_dim, self.kernel_size - 1], INNER_DATA_TYPE)?;
        let mut k = command_buffer.allocate_scratch_for_shape(&[tree_size, self.key_dim], self.outer_data_type)?;
        let mut v = command_buffer.allocate_scratch_for_shape(&[tree_size, self.value_dim], self.outer_data_type)?;
        let mut beta = command_buffer.allocate_scratch_for_shape(&[tree_size, self.num_heads], INNER_DATA_TYPE)?;
        let mut log_decay = command_buffer.allocate_scratch_for_shape(&[tree_size, self.num_heads], INNER_DATA_TYPE)?;

        let mut tree_projected =
            command_buffer.allocate_scratch_for_shape(&[tree_size, self.total_proj_dim], self.outer_data_type)?;
        let mut q = command_buffer.allocate_scratch_for_shape(&[tree_size, self.key_dim], self.outer_data_type)?;

        self.conv_tree_scan.encode(
            in_projected,
            &self.conv_weight,
            self.conv_bias.as_ref(),
            &state.conv_state,
            &parents,
            &mut tree_projected,
            &mut conv_states,
            tree_size,
            self.total_proj_dim,
            self.conv_dim,
            command_buffer,
        );

        self.delta_net_tree_prep.encode(
            &tree_projected,
            &tree_projected,
            &self.a_log,
            &self.dt_bias,
            &mut q,
            &mut k,
            Some(&mut v),
            &mut beta,
            &mut log_decay,
            self.num_heads,
            self.num_groups,
            self.key_dim,
            self.value_dim,
            tree_size,
            self.total_proj_dim,
            u32::from(self.reference_bf16),
            command_buffer,
        );

        let mut delta_output = tree_verify.encode(
            TreeVerifyEncodeArguments {
                q: q.subrange(..),
                k: k.subrange(..),
                v: v.subrange(..),
                trie: &trie,
                log_decay: log_decay.subrange(..),
                beta: beta.subrange(..),
                h0: &state.ssm_state,
                tree_size,
            },
            command_buffer,
        )?;
        self.delta_net_norm_gate.encode(
            &mut delta_output,
            &tree_projected,
            &self.norm_weight,
            self.num_heads,
            self.value_head_dim,
            self.value_dim,
            self.conv_dim,
            self.total_proj_dim,
            self.norm_epsilon,
            tree_size,
            u32::from(self.reference_bf16),
            command_buffer,
        );

        let output = if project_output {
            self.out_projection.encode(delta_output, tree_size, command_buffer)?
        } else {
            delta_output
        };
        state.suffix_status = Some(DeltaNetSuffixStatus::Tree {
            conv_states,
            k,
            v,
            log_decay,
            beta,
            parents: batch_dim.parents().into(),
        });
        Ok(output)
    }

    /// A tiny flat speculative chain uses the identical T=1 solve and state
    /// advance arithmetic as serial target decoding. Closed-form multi-row
    /// solves reorder FP32 operations and can change BF16 target decisions.
    /// Projection and output projection still process all rows together.
    fn encode_short_chain_verify(
        &self,
        projected: impl BufferRef<Backend = B>,
        batch: &BatchTopology,
        state: &mut DeltaNetState<B>,
        command: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        self.encode_short_chain_verify_with_mode(projected, batch, state, true, command)
    }

    fn encode_short_chain_verify_with_mode(
        &self,
        projected: impl BufferRef<Backend = B>,
        batch: &BatchTopology,
        state: &mut DeltaNetState<B>,
        use_fused: bool,
        command: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        let count = batch.size();
        let tree_verify = self.tree_verify.as_ref().expect("DeltaNet tree verification is unsupported");
        // Verification must leave committed state untouched. Convolution
        // history is a read-only tree scan with identical per-channel math.
        let mut all_conv =
            command.allocate_scratch_for_shape(&[count, self.conv_dim, self.kernel_size - 1], INNER_DATA_TYPE)?;
        let mut all_q = command.allocate_scratch_for_shape(&[count, self.key_dim], self.outer_data_type)?;
        let mut all_k = command.allocate_scratch_for_shape(&[count, self.key_dim], self.outer_data_type)?;
        let mut all_v = command.allocate_scratch_for_shape(&[count, self.value_dim], self.outer_data_type)?;
        let mut all_beta = command.allocate_scratch_for_shape(&[count, self.num_heads], INNER_DATA_TYPE)?;
        let mut all_decay = command.allocate_scratch_for_shape(&[count, self.num_heads], INNER_DATA_TYPE)?;
        let mut tree_projected =
            command.allocate_scratch_for_shape(&[count, self.total_proj_dim], self.outer_data_type)?;
        let parents = command.allocate_constant_from_slice(batch.parents())?;
        self.conv_tree_scan.encode(
            projected,
            &self.conv_weight,
            self.conv_bias.as_ref(),
            &state.conv_state,
            &parents,
            &mut tree_projected,
            &mut all_conv,
            count,
            self.total_proj_dim,
            self.conv_dim,
            command,
        );
        self.delta_net_tree_prep.encode(
            &tree_projected,
            &tree_projected,
            &self.a_log,
            &self.dt_bias,
            &mut all_q,
            &mut all_k,
            Some(&mut all_v),
            &mut all_beta,
            &mut all_decay,
            self.num_heads,
            self.num_groups,
            self.key_dim,
            self.value_dim,
            count,
            self.total_proj_dim,
            u32::from(self.reference_bf16),
            command,
        );
        let nodes = [crate::backends::common::gpu_types::trie::TrieNode {
            trie_start: 0,
            trie_end: 0,
            height: 0,
        }];
        let trie = command.allocate_constant_from_slice(&nodes)?;
        let fused_output = if use_fused && batch.is_flat() && !self.reference_bf16 {
            tree_verify.encode_short_chain(
                TreeVerifyEncodeArguments {
                    q: all_q.subrange(..),
                    k: all_k.subrange(..),
                    v: all_v.subrange(..),
                    trie: &trie,
                    beta: all_beta.subrange(..),
                    log_decay: all_decay.subrange(..),
                    h0: &state.ssm_state,
                    tree_size: count,
                },
                command,
            )?
        } else {
            None
        };
        let mut all_output = if let Some(output) = fused_output {
            output
        } else {
            // Keep the exact serial T=1 verifier as the fallback and reference.
            let mut working_ssm = command.allocate_scratch(state.ssm_state.size())?;
            command.encode_copy(&state.ssm_state, &mut working_ssm);
            let mut all_output = command.allocate_scratch_for_shape(&[count, self.value_dim], self.outer_data_type)?;
            let accepted = command.allocate_constant_from_slice(&[0u32])?;
            let key_bytes = self.key_dim as usize * self.outer_data_type.size_in_bytes();
            let output_bytes = self.value_dim as usize * self.outer_data_type.size_in_bytes();
            let scalar_bytes = self.num_heads as usize * INNER_DATA_TYPE.size_in_bytes();
            for row in 0..count as usize {
                let q = all_q.subrange(row * key_bytes..(row + 1) * key_bytes);
                let k = all_k.subrange(row * key_bytes..(row + 1) * key_bytes);
                let v = all_v.subrange(row * output_bytes..(row + 1) * output_bytes);
                let beta = all_beta.subrange(row * scalar_bytes..(row + 1) * scalar_bytes);
                let log_decay = all_decay.subrange(row * scalar_bytes..(row + 1) * scalar_bytes);
                let output = tree_verify.encode(
                    TreeVerifyEncodeArguments {
                        q,
                        k,
                        v,
                        trie: &trie,
                        beta,
                        log_decay,
                        h0: &working_ssm,
                        tree_size: 1,
                    },
                    command,
                )?;
                command.encode_copy(&output, all_output.subrange_mut(row * output_bytes..(row + 1) * output_bytes));
                // The last speculative state is never consumed. Committing any
                // accepted prefix later still uses the existing StateAdvance path.
                if row + 1 < count as usize {
                    state.state_advance.encode(k, v, log_decay, beta, &accepted, &mut working_ssm, 1, command);
                }
            }
            all_output
        };
        self.delta_net_norm_gate.encode(
            &mut all_output,
            &tree_projected,
            &self.norm_weight,
            self.num_heads,
            self.value_head_dim,
            self.value_dim,
            self.conv_dim,
            self.total_proj_dim,
            self.norm_epsilon,
            count,
            u32::from(self.reference_bf16),
            command,
        );
        state.suffix_status = Some(DeltaNetSuffixStatus::Tree {
            conv_states: all_conv,
            k: all_k,
            v: all_v,
            beta: all_beta,
            log_decay: all_decay,
            parents: batch.parents().into(),
        });
        self.out_projection.encode(all_output, count, command)
    }
}

impl<B: Backend> Mixer<B> for DeltaNet<B> {
    fn speculation_supported(&self) -> bool {
        self.tree_verify.is_some()
    }

    fn max_context_length(&self) -> Option<u32> {
        None
    }

    fn memory_estimate(
        &self,
        _context_tokens: u32,
        input_tokens: u32,
    ) -> Option<crate::encodable_block::memory::MemoryEstimate> {
        use crate::encodable_block::memory::{MemoryEstimate, checked_bytes};
        let inner = INNER_DATA_TYPE.size_in_bytes();
        let outer = self.outer_data_type.size_in_bytes();
        let state_rows = self.kernel_size.checked_sub(1)?;
        let conv = checked_bytes(&[self.conv_dim, state_rows], inner)?;
        let ssm = checked_bytes(&[self.num_heads, self.value_head_dim, self.head_dim], inner)?;
        let mut estimate = MemoryEstimate::state(conv.checked_add(ssm)?);
        estimate.prefix_snapshot_bytes = conv.checked_add(ssm)?;
        for rows in [input_tokens.clamp(1, 1024), 1] {
            let mut scratch = MemoryEstimate::default();
            scratch.add_scratch(&[rows, self.total_proj_dim], outer, 2)?;
            scratch.add_scratch(&[rows.checked_add(state_rows)?, self.total_proj_dim], inner, 1)?;
            scratch.add_scratch(&[rows, self.value_dim], outer, 2)?;
            scratch.add_scratch(&[rows, self.key_dim], inner, 2)?;
            scratch.add_scratch(&[rows, self.num_heads], inner, 3)?;
            scratch.add_scratch(&[self.conv_dim, state_rows], inner, 1)?;
            if rows > 1 && (self.chunked.is_some() || self.reference_chunk.is_some()) {
                let chunks = rows.div_ceil(64);
                if self.chunked.is_some() {
                    scratch.add_scratch(&[chunks, self.num_groups, 64, 64], inner, 1)?;
                    // QK, packed blocks and inverse blocks are each no larger
                    // than this full per-head chunk plane; plus BF16 T.
                    scratch.add_scratch(&[chunks, self.num_heads, 64, 64], inner, 3)?;
                    scratch.add_scratch(&[chunks, self.num_heads, 64, 64], 2, 1)?;
                }
                if self.reference_chunk.is_some() {
                    scratch.add_scratch(&[chunks, self.num_heads, 64, 128], 4, 6)?;
                    scratch.add_scratch(&[chunks, self.num_heads, 64, 64], 4, 2)?;
                    scratch.add_scratch(&[chunks, self.num_heads, 64], 4, 1)?;
                    scratch.add_scratch(&[self.num_heads, 64, 128], 4, 1)?;
                }
            }
            estimate = estimate.sequential(scratch)?;
        }
        Some(estimate)
    }

    fn create_empty_state(
        &self,
        _max_context_length: Option<u32>,
        context: &B::Context,
    ) -> Result<Box<dyn MixerState<B>>, B::Error> {
        let conv_state =
            context.create_buffer(size_for_shape(&[self.conv_dim, self.kernel_size - 1], INNER_DATA_TYPE))?;

        let ssm_state = context
            .create_buffer(size_for_shape(&[self.num_heads, self.value_head_dim, self.head_dim], INNER_DATA_TYPE))?;

        #[cfg(not(feature = "deferred-state-init"))]
        let (conv_state, ssm_state) = {
            let mut conv_state = conv_state;
            let mut ssm_state = ssm_state;
            let mut zero_command_buffer = context.create_command_buffer(None, None)?;
            zero_command_buffer.encode_fill(&mut conv_state, 0);
            zero_command_buffer.encode_fill(&mut ssm_state, 0);
            zero_command_buffer.end_encoding().submit().wait_until_completed()?;
            (conv_state, ssm_state)
        };

        let state_advance = <B::Kernels as Kernels>::StateAdvanceKernel::new(
            context,
            self.outer_data_type,
            self.head_dim,
            self.num_heads,
            self.num_groups,
        )?;

        Ok(Box::new(DeltaNetState {
            conv_state,
            ssm_state,
            suffix_status: None,
            #[cfg(feature = "deferred-state-init")]
            needs_zero_initialization: true,
            state_advance,
        }))
    }

    fn encode(
        &self,
        hidden: B::ScratchBuffer,
        precalculated_rope: Option<&PrecalculatedRoPE<B>>,
        batch_dim: &BatchTopology,
        state: Option<MaybeMut<dyn MixerState<B>>>,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        command_buffer.push_debug_group("delta net");

        assert!(precalculated_rope.is_none(), "unexpected rope for delta net mixer");

        let state = state.expect("delta net requires state");
        let state = state.downcast::<DeltaNetState<B>>().expect("incorrect type of delta net state");
        let MaybeMut::Mut(state) = state else {
            panic!("delta net doesn't support immutable state");
        };

        assert!(state.suffix_status.is_none(), "delta net called with state with an unaccepted suffix");

        #[cfg(feature = "deferred-state-init")]
        if state.needs_zero_initialization {
            // Order initialization before every state consumer, including tree
            // verification. Later chunks in this command buffer must preserve
            // accumulated state, so clear the flag after encoding both fills.
            // This flag records encoding, not GPU completion. Discard this state
            // if encoding/submission fails or the command buffer is abandoned.
            // The isolated scorer owns fresh state until completion and drops
            // it on every error; this feature does not enable state reuse.
            command_buffer.encode_fill(&mut state.conv_state, 0);
            command_buffer.encode_fill(&mut state.ssm_state, 0);
            state.needs_zero_initialization = false;
        }

        let mut in_projected = self.in_projection.encode(hidden, batch_dim.size(), command_buffer)?;

        if !batch_dim.full_accept() {
            let output = if batch_dim.is_flat() && (2..=4).contains(&batch_dim.size()) {
                self.encode_short_chain_verify(&in_projected, batch_dim, state, command_buffer)?
            } else {
                self.encode_tree_verify(&in_projected, batch_dim, state, true, command_buffer)?
            };

            command_buffer.pop_debug_group();

            return Ok(output);
        }

        let mut delta_output =
            command_buffer.allocate_scratch_for_shape(&[batch_dim.size(), self.value_dim], self.outer_data_type)?;
        if batch_dim.size() == 1 && !self.reference_bf16 {
            self.conv_update.encode(
                &self.conv_weight,
                self.conv_bias.as_ref(),
                &mut in_projected,
                &mut state.conv_state,
                self.kernel_size,
                self.conv_dim,
                self.kernel_size - 1,
                u32::from(self.reference_bf16),
                command_buffer,
            );

            self.delta_net_update.encode(
                &in_projected,
                &self.a_log,
                &self.dt_bias,
                &self.norm_weight,
                &mut state.ssm_state,
                &mut delta_output,
                self.num_heads,
                self.num_groups,
                self.value_head_dim,
                self.key_dim,
                self.value_dim,
                self.norm_epsilon,
                u32::from(self.reference_bf16),
                command_buffer,
            );
        } else {
            #[cfg(feature = "compact-delta-conv")]
            let qkv_stride = self.conv_dim;
            #[cfg(not(feature = "compact-delta-conv"))]
            let qkv_stride = self.total_proj_dim;
            // Keep the original gate/control channels without copying them.
            // The pool remains owned by the command/pending submission.
            #[cfg(feature = "compact-delta-conv")]
            let controls;
            if let Some(reference_conv) = &self.reference_conv {
                let mut history = command_buffer.allocate_scratch(state.conv_state.size())?;
                command_buffer.encode_copy(&state.conv_state, &mut history);
                let mut convolved =
                    command_buffer.allocate_scratch_for_shape(&[batch_dim.size(), qkv_stride], self.outer_data_type)?;
                reference_conv.encode(
                    &in_projected,
                    &history,
                    &self.conv_weight,
                    &mut convolved,
                    &mut state.conv_state,
                    batch_dim.size(),
                    self.kernel_size,
                    self.total_proj_dim,
                    qkv_stride,
                    self.kernel_size - 1,
                    self.conv_dim,
                    command_buffer,
                );
                #[cfg(feature = "compact-delta-conv")]
                {
                    controls = in_projected;
                }
                in_projected = convolved;
            } else {
                #[cfg(feature = "direct-delta-conv")]
                {
                    // Preserve immutable original samples until every output has
                    // been encoded. A history snapshot prevents cross-thread races
                    // between early-token reads and next-state writes.
                    let mut history = command_buffer.allocate_scratch(state.conv_state.size())?;
                    command_buffer.encode_copy(&state.conv_state, &mut history);
                    #[cfg(feature = "compact-delta-conv")]
                    let mut convolved = command_buffer
                        .allocate_scratch_for_shape(&[batch_dim.size(), qkv_stride], self.outer_data_type)?;
                    #[cfg(not(feature = "compact-delta-conv"))]
                    let mut convolved = command_buffer.allocate_scratch(in_projected.size())?;
                    self.conv_direct.encode(
                        &in_projected,
                        &history,
                        &self.conv_weight,
                        self.conv_bias.as_ref(),
                        &mut convolved,
                        &mut state.conv_state,
                        batch_dim.size(),
                        self.kernel_size,
                        self.total_proj_dim,
                        qkv_stride,
                        self.kernel_size - 1,
                        self.conv_dim,
                        u32::from(self.reference_bf16),
                        command_buffer,
                    );
                    #[cfg(feature = "compact-delta-conv")]
                    {
                        controls = in_projected;
                    }
                    in_projected = convolved;
                }
                #[cfg(not(feature = "direct-delta-conv"))]
                {
                    let mut padded = command_buffer.allocate_scratch_for_shape(
                        &[batch_dim.size() + self.kernel_size - 1, self.total_proj_dim],
                        INNER_DATA_TYPE,
                    )?;
                    self.conv_pack.encode(
                        &state.conv_state,
                        &in_projected,
                        &mut padded,
                        self.kernel_size - 1,
                        self.total_proj_dim,
                        batch_dim.size(),
                        self.conv_dim,
                        command_buffer,
                    );
                    self.conv_scan.encode(
                        &padded,
                        &self.conv_weight,
                        self.conv_bias.as_ref(),
                        &mut in_projected,
                        &mut state.conv_state,
                        batch_dim.size(),
                        self.kernel_size,
                        self.total_proj_dim,
                        self.kernel_size - 1,
                        self.conv_dim,
                        self.total_proj_dim,
                        u32::from(self.reference_bf16),
                        command_buffer,
                    );
                }
            }
            #[cfg(feature = "compact-delta-conv")]
            let controls = &controls;
            #[cfg(not(feature = "compact-delta-conv"))]
            let controls = &in_projected;
            if let Some(chunked) = self.chunked.as_ref().filter(|chunked| chunked.should_use(batch_dim.size())) {
                chunked.encode(
                    DeltaNetChunkedPrefillArgs {
                        in_projected: &in_projected,
                        controls,
                        qkv_stride,
                        a_log: &self.a_log,
                        dt_bias: &self.dt_bias,
                        ssm_state: &mut state.ssm_state,
                        delta_output: &mut delta_output,
                        num_heads: self.num_heads,
                        num_groups: self.num_groups,
                        value_head_dim: self.value_head_dim,
                        key_dim: self.key_dim,
                        value_dim: self.value_dim,
                        suffix_len: batch_dim.size(),
                    },
                    command_buffer,
                )?;
            } else {
                let mut prep_q_norm =
                    command_buffer.allocate_scratch_for_shape(&[batch_dim.size(), self.key_dim], INNER_DATA_TYPE)?;
                let mut prep_k_norm =
                    command_buffer.allocate_scratch_for_shape(&[batch_dim.size(), self.key_dim], INNER_DATA_TYPE)?;
                let mut prep_beta =
                    command_buffer.allocate_scratch_for_shape(&[batch_dim.size(), self.num_heads], INNER_DATA_TYPE)?;
                let mut prep_decay =
                    command_buffer.allocate_scratch_for_shape(&[batch_dim.size(), self.num_heads], INNER_DATA_TYPE)?;
                let preparation = if batch_dim.size() > 1 {
                    self.reference_log_prep.as_ref().unwrap_or(&self.delta_net_prefill_prep)
                } else {
                    &self.delta_net_prefill_prep
                };
                preparation.encode(
                    &in_projected,
                    controls,
                    &self.a_log,
                    &self.dt_bias,
                    &mut prep_q_norm,
                    &mut prep_k_norm,
                    None::<&mut B::ScratchBuffer>,
                    &mut prep_beta,
                    &mut prep_decay,
                    self.num_heads,
                    self.num_groups,
                    self.key_dim,
                    self.value_dim,
                    batch_dim.size(),
                    qkv_stride,
                    u32::from(self.reference_bf16),
                    command_buffer,
                );
                if let Some(reference_decay) = &self.reference_decay {
                    reference_decay.encode(
                        controls,
                        &self.a_log,
                        &self.dt_bias,
                        &mut prep_decay,
                        batch_dim.size(),
                        self.num_heads,
                        self.conv_dim,
                        self.value_dim,
                        self.total_proj_dim,
                        u32::from(batch_dim.size() > 1),
                        command_buffer,
                    );
                }
                if let Some(reference_l2) = &self.reference_l2 {
                    reference_l2.encode(
                        &in_projected,
                        &mut prep_q_norm,
                        &mut prep_k_norm,
                        self.key_dim,
                        batch_dim.size(),
                        qkv_stride,
                        command_buffer,
                    );
                }
                if batch_dim.size() > 1
                    && let Some(reference_chunk) = &self.reference_chunk
                {
                    reference_chunk.encode(
                        &prep_q_norm,
                        &prep_k_norm,
                        &prep_beta,
                        &prep_decay,
                        &in_projected,
                        &mut state.ssm_state,
                        &mut delta_output,
                        batch_dim.size(),
                        self.num_heads,
                        self.num_groups,
                        self.key_dim,
                        self.value_dim,
                        qkv_stride,
                        command_buffer,
                    )?;
                } else {
                    self.delta_net_prefill.encode(
                        &prep_q_norm,
                        &prep_k_norm,
                        &prep_beta,
                        &prep_decay,
                        &in_projected,
                        &mut state.ssm_state,
                        &mut delta_output,
                        self.num_heads,
                        self.num_groups,
                        self.value_head_dim,
                        self.key_dim,
                        self.value_dim,
                        batch_dim.size(),
                        self.value_head_dim.div_ceil(16),
                        qkv_stride,
                        command_buffer,
                    );
                }
            }
            if let Some(reference_norm_gate) = &self.reference_norm_gate {
                reference_norm_gate.encode(
                    &mut delta_output,
                    controls,
                    &self.norm_weight,
                    batch_dim.size(),
                    self.num_heads,
                    self.value_dim,
                    self.conv_dim,
                    self.total_proj_dim,
                    self.norm_epsilon,
                    command_buffer,
                );
            } else {
                self.delta_net_norm_gate.encode(
                    &mut delta_output,
                    controls,
                    &self.norm_weight,
                    self.num_heads,
                    self.value_head_dim,
                    self.value_dim,
                    self.conv_dim,
                    self.total_proj_dim,
                    self.norm_epsilon,
                    batch_dim.size(),
                    u32::from(self.reference_bf16),
                    command_buffer,
                );
            }
        }

        state.suffix_status = Some(DeltaNetSuffixStatus::Flat {
            suffix_length: batch_dim.size(),
        });

        let output = self.out_projection.encode(delta_output, batch_dim.size(), command_buffer)?;

        command_buffer.pop_debug_group();

        Ok(output)
    }
}
