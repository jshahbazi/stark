use parking_lot::Mutex;

use crate::{
    backends::{
        common::{
            Backend, BufferRef, CommandBufferEncoding, Kernels,
            kernel::{
                AttentionArguments, AttentionKernelConfig, SoftmaxKernel, VisionCastBF16Kernel,
                matmul::{MatmulA, MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            },
        },
        metal::{
            Metal,
            command_buffer::MetalCommandBufferEncoding,
            context::MetalContext,
            error::MetalError,
            kernel::{
                AttentionFallbackScatterScoresMetalKernel, AttentionFallbackScatterValuesMetalKernel, MetalKernels,
            },
        },
    },
    data_type::DataType,
};

const HEAD_DIM: u32 = 512;

struct SourceQK {
    config: AttentionKernelConfig,
    matmul: Mutex<<MetalKernels as Kernels>::MatmulKernel>,
    cast: <MetalKernels as Kernels>::VisionCastBF16Kernel,
}

pub struct AttentionFallback {
    head_dim: u32,
    num_groups: u32,
    num_q_heads: u32,
    sliding_window_size: Option<u32>,
    scale: Option<f32>,
    data_type: DataType,
    scatter_scores: AttentionFallbackScatterScoresMetalKernel,
    scatter_values: AttentionFallbackScatterValuesMetalKernel,
    softmax: <MetalKernels as Kernels>::SoftmaxKernel,
    source_softmax: Option<super::source_softmax::SourceBF16Softmax>,
    source_qk: Option<SourceQK>,
    matmul: Mutex<<MetalKernels as Kernels>::MatmulKernel>,
}

impl AttentionFallback {
    pub fn is_supported(config: &AttentionKernelConfig) -> bool {
        (config.head_dim == HEAD_DIM || (config.materialize_prefill_scores && config.head_dim == 256))
            && matches!(config.data_type, DataType::BF16 | DataType::F32)
    }

    pub fn new(
        config: &AttentionKernelConfig,
        context: &MetalContext,
    ) -> Result<Self, MetalError> {
        let scatter_scores = AttentionFallbackScatterScoresMetalKernel::new(
            context,
            config.data_type,
            config.is_kv_cache_ring,
            config.is_causal,
            false,
            config.sliding_window_size.is_some(),
        )?;
        let scatter_values = AttentionFallbackScatterValuesMetalKernel::new(context, config.data_type)?;
        let softmax = <<MetalKernels as Kernels>::SoftmaxKernel as SoftmaxKernel>::new(
            context,
            config.data_type,
            config.has_sinks,
        )?;
        let matmul = Mutex::new(<<MetalKernels as Kernels>::MatmulKernel as MatmulKernel>::new(
            context,
            config.data_type,
            config.data_type,
            config.data_type,
        )?);
        let source_softmax = config
            .materialize_prefill_scores
            .then(|| super::source_softmax::SourceBF16Softmax::new(context))
            .transpose()?;
        let source_qk = if config.materialize_prefill_scores
            && config.data_type == DataType::BF16
            && config.head_dim == 512
            && config.num_q_heads == 8
            && config.num_groups == 1
            && context.supports_mxu
        {
            Some(SourceQK {
                config: *config,
                matmul: Mutex::new(<<MetalKernels as Kernels>::MatmulKernel as MatmulKernel>::new(
                    context,
                    DataType::BF16,
                    DataType::BF16,
                    DataType::F32,
                )?),
                cast: <<MetalKernels as Kernels>::VisionCastBF16Kernel as VisionCastBF16Kernel>::new(context)?,
            })
        } else {
            None
        };
        Ok(Self {
            head_dim: config.head_dim,
            num_groups: config.num_groups,
            num_q_heads: config.num_q_heads,
            sliding_window_size: config.sliding_window_size,
            scale: config.scale,
            data_type: config.data_type,
            scatter_scores,
            scatter_values,
            softmax,
            source_softmax,
            source_qk,
            matmul,
        })
    }

    pub fn encode(
        &self,
        arguments: AttentionArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        self.encode_inner(
            arguments,
            command_buffer,
            #[cfg(test)]
            None,
        )
    }

    #[cfg(test)]
    pub(crate) fn encode_trace(
        &self,
        arguments: AttentionArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        command_buffer: &mut MetalCommandBufferEncoding,
    ) -> Result<(<Metal as Backend>::ScratchBuffer, Vec<(&'static str, <Metal as Backend>::ScratchBuffer)>), MetalError>
    {
        if self.num_groups != 1 {
            return Err(crate::backends::common::InvalidInput("trace requires one KV group").into());
        }
        let mut trace = Vec::new();
        let output = self.encode_inner(arguments, command_buffer, Some(&mut trace))?;
        Ok((output, trace))
    }

    fn encode_inner(
        &self,
        arguments: AttentionArguments<
            '_,
            Metal,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
            impl BufferRef<Backend = Metal>,
        >,
        command_buffer: &mut MetalCommandBufferEncoding,
        #[cfg(test)] mut trace: Option<&mut Vec<(&'static str, <Metal as Backend>::ScratchBuffer)>>,
    ) -> Result<<Metal as Backend>::ScratchBuffer, MetalError> {
        assert!(arguments.trie.is_none(), "fallback does not support trie");
        let suffix_length = arguments.suffix_length;
        let sequence_length = arguments.cache.prefix_len() + suffix_length;
        let gqa_factor = self.num_q_heads / self.num_groups;
        let scale = self.scale.unwrap_or(1.0 / (self.head_dim as f32).sqrt());
        let dt_bytes = self.data_type.size_in_bytes();
        let head_dim_bytes = self.head_dim as usize * dt_bytes;
        let group_rows = (gqa_factor * suffix_length) as usize;
        let mut output = command_buffer
            .allocate_scratch_for_shape(&[suffix_length, self.num_q_heads, self.head_dim], self.data_type)?;
        let mut scores = command_buffer
            .allocate_scratch_for_shape(&[self.num_q_heads, suffix_length, sequence_length], self.data_type)?;
        let mut group_scores = command_buffer
            .allocate_scratch_for_shape(&[gqa_factor * suffix_length, sequence_length], self.data_type)?;

        for group_index in 0..self.num_groups {
            if let Some(source) = &self.source_qk
                && source.config.source_nax_qk_split_parts(suffix_length, sequence_length, true).is_some()
            {
                // MLX NAX split-K keeps dense BF16 operands, F32 partials and
                // reduction, then materializes each score once as BF16.
                let mut partials = command_buffer
                    .allocate_scratch_for_shape(&[gqa_factor * suffix_length, sequence_length], DataType::F32)?;
                source.matmul.lock().encode_source_qk_split(
                    MatmulArguments {
                        a: MatmulA::FullPrecision {
                            values: arguments.queries,
                            offset: 0,
                        },
                        b: MatmulB::FullPrecision {
                            b: arguments.keys,
                        },
                        b_leading_dimension: Some(self.head_dim),
                        b_transpose: true,
                        d: &mut partials,
                        d_transform: MatmulDOps::none(),
                        gather_indices: None::<&<Metal as Backend>::ScratchBuffer>,
                        m: gqa_factor * suffix_length,
                        n: sequence_length,
                        k: self.head_dim,
                    },
                    command_buffer,
                )?;
                source.cast.encode(
                    &partials,
                    &mut group_scores,
                    gqa_factor * suffix_length * sequence_length,
                    command_buffer,
                );
            } else {
                self.matmul.lock().encode(
                    MatmulArguments {
                        a: MatmulA::FullPrecision {
                            values: arguments.queries,
                            offset: group_index as usize * group_rows * head_dim_bytes,
                        },
                        b: MatmulB::FullPrecision {
                            b: arguments.keys.subrange(group_index as usize * head_dim_bytes..),
                        },
                        b_leading_dimension: Some(self.num_groups * self.head_dim),
                        b_transpose: true,
                        d: &mut group_scores,
                        d_transform: MatmulDOps {
                            ab_scale: scale,
                            ..MatmulDOps::none()
                        },
                        gather_indices: None::<&<Metal as Backend>::ScratchBuffer>,
                        m: gqa_factor * suffix_length,
                        n: sequence_length,
                        k: self.head_dim,
                    },
                    command_buffer,
                )?;
            }
            self.scatter_scores.encode(
                &group_scores,
                &mut scores,
                arguments.cache.ring_params(),
                None::<&<Metal as Backend>::ConstantBuffer>,
                self.sliding_window_size,
                group_index,
                gqa_factor,
                sequence_length,
                suffix_length,
                gqa_factor * suffix_length * sequence_length,
                command_buffer,
            );
        }

        #[cfg(test)]
        if let Some(trace) = trace.as_deref_mut() {
            let mut snapshot = command_buffer
                .allocate_scratch_for_shape(&[self.num_q_heads, suffix_length, sequence_length], self.data_type)?;
            command_buffer.encode_copy(&scores, &mut snapshot);
            trace.push(("scores", snapshot));
        }
        if let Some(source_softmax) = &self.source_softmax {
            source_softmax.encode(&mut scores, sequence_length, self.num_q_heads * suffix_length, command_buffer)?;
        } else {
            self.softmax.encode(
                &mut scores,
                arguments.sinks,
                sequence_length,
                self.num_q_heads,
                suffix_length,
                command_buffer,
            );
        }
        #[cfg(test)]
        if let Some(trace) = trace.as_deref_mut() {
            let mut snapshot = command_buffer
                .allocate_scratch_for_shape(&[self.num_q_heads, suffix_length, sequence_length], self.data_type)?;
            command_buffer.encode_copy(&scores, &mut snapshot);
            trace.push(("probabilities", snapshot));
        }
        let mut group_output =
            command_buffer.allocate_scratch_for_shape(&[gqa_factor * suffix_length, self.head_dim], self.data_type)?;
        for group_index in 0..self.num_groups {
            self.matmul.lock().encode(
                MatmulArguments {
                    a: MatmulA::FullPrecision {
                        values: &scores,
                        offset: group_index as usize * group_rows * sequence_length as usize * dt_bytes,
                    },
                    b: MatmulB::FullPrecision {
                        b: arguments.values.subrange(group_index as usize * head_dim_bytes..),
                    },
                    b_leading_dimension: Some(self.num_groups * self.head_dim),
                    b_transpose: false,
                    d: &mut group_output,
                    d_transform: MatmulDOps::none(),
                    gather_indices: None::<&<Metal as Backend>::ScratchBuffer>,
                    m: gqa_factor * suffix_length,
                    n: self.head_dim,
                    k: sequence_length,
                },
                command_buffer,
            )?;
            #[cfg(test)]
            if let Some(trace) = trace.as_deref_mut() {
                let mut snapshot = command_buffer
                    .allocate_scratch_for_shape(&[gqa_factor * suffix_length, self.head_dim], self.data_type)?;
                command_buffer.encode_copy(&group_output, &mut snapshot);
                trace.push(("pv", snapshot));
            }
            self.scatter_values.encode(
                &group_output,
                &mut output,
                group_index,
                gqa_factor,
                suffix_length,
                self.num_q_heads,
                self.head_dim,
                gqa_factor * suffix_length * self.head_dim,
                command_buffer,
            );
        }
        Ok(output)
    }
}
