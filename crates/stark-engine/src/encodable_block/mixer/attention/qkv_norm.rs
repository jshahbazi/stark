use derive_more::Debug;
use thiserror::Error;

use crate::{
    backends::common::{
        Backend, BufferMut, CommandBuffer, CommandBufferEncoding,
        kernel::{Kernels, QKVNormKernel, ReferenceRMSBF16Kernel},
    },
    config::normalization::{NormalizationConfig, UpcastMode},
    data_type::DataType,
    parameters::{NumericalProfile, ParameterLoaderError, ParameterTree},
};

#[derive(Debug, Error)]
pub enum QKVNormError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Parameter loading error: {0}")]
    ParameterError(#[from] ParameterLoaderError<B>),
}

struct Head<B: Backend> {
    kernel: <B::Kernels as Kernels>::QKVNormKernel,
    reference: Option<<B::Kernels as Kernels>::ReferenceRMSBF16Kernel>,
    scales: Option<B::GlobalBuffer>,
    config: NormalizationConfig,
}

pub struct QKVNorm<B: Backend> {
    query: Option<Head<B>>,
    key: Option<Head<B>>,
    value: Option<Head<B>>,
    num_q_heads: u32,
    num_kv_heads: u32,
    projection_row_stride: u32,
    head_dim: u32,
}

impl<B: Backend> QKVNorm<B> {
    pub fn new(
        context: &B::Context,
        intermediate_data_type: DataType,
        query_config: Option<NormalizationConfig>,
        key_config: Option<NormalizationConfig>,
        value_config: Option<NormalizationConfig>,
        parameter_tree: &ParameterTree<B>,
        num_q_heads: u32,
        num_kv_heads: u32,
        projection_row_stride: u32,
        head_dim: u32,
    ) -> Result<Self, QKVNormError<B>> {
        let reference_profile = parameter_tree.numerical_profile() == NumericalProfile::Qwen35ReferenceBF16;
        let query = query_config
            .map(|cfg| {
                Self::build_head(
                    context,
                    intermediate_data_type,
                    cfg,
                    Some(&parameter_tree.subtree("query_norm")),
                    head_dim,
                    reference_profile,
                )
            })
            .transpose()?;
        let key = key_config
            .map(|cfg| {
                Self::build_head(
                    context,
                    intermediate_data_type,
                    cfg,
                    Some(&parameter_tree.subtree("key_norm")),
                    head_dim,
                    reference_profile,
                )
            })
            .transpose()?;
        let value = value_config
            .map(|cfg| Self::build_head(context, intermediate_data_type, cfg, None, head_dim, reference_profile))
            .transpose()?;

        Ok(Self {
            query,
            key,
            value,
            num_q_heads,
            num_kv_heads,
            projection_row_stride,
            head_dim,
        })
    }

    fn build_head(
        context: &B::Context,
        intermediate_data_type: DataType,
        config: NormalizationConfig,
        parameter_tree: Option<&ParameterTree<B>>,
        head_dim: u32,
        reference_profile: bool,
    ) -> Result<Head<B>, QKVNormError<B>> {
        let reference = if reference_profile {
            if intermediate_data_type != DataType::BF16
                || config.subtract_mean
                || config.has_biases
                || !config.has_scale
                || config.upcast_mode != UpcastMode::FullLayer
                || head_dim < 16
                || !head_dim.is_multiple_of(16)
            {
                return Err(QKVNormError::BackendError(
                    crate::backends::common::InvalidInput("unsupported reference BF16 Q/K RMS configuration").into(),
                ));
            }
            Some(
                <B::Kernels as Kernels>::ReferenceRMSBF16Kernel::new(context, true, false, false)
                    .map_err(QKVNormError::BackendError)?,
            )
        } else {
            None
        };
        let scales = if config.has_scale {
            Some(
                parameter_tree
                    .expect("scaled norm requires parameter tree")
                    .leaf("scales")?
                    .validate(&[head_dim], DataType::F32)?
                    .read_buffer()?,
            )
        } else {
            None
        };
        let kernel = <B::Kernels as Kernels>::QKVNormKernel::new(
            context,
            intermediate_data_type,
            DataType::F32,
            intermediate_data_type,
            DataType::F32,
            true,
            scales.is_some(),
        )
        .map_err(QKVNormError::BackendError)?;
        Ok(Head {
            kernel,
            reference,
            scales,
            config,
        })
    }

    pub fn encode(
        &self,
        qkvg: impl BufferMut<Backend = B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        self.encode_packed(qkvg, batch_dim, self.num_q_heads, self.projection_row_stride, false, command_buffer)
    }

    pub fn encode_key_value(
        &self,
        key_value: impl BufferMut<Backend = B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        self.encode_packed(key_value, batch_dim, 0, 2 * self.num_kv_heads * self.head_dim, false, command_buffer)
    }

    /// Normalize K/V in the original full QKVG layout, retaining query offsets.
    #[cfg(feature = "attention-kv-only")]
    pub(super) fn encode_projected_key_value(
        &self,
        qkvg: impl BufferMut<Backend = B>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        self.encode_packed(qkvg, batch_dim, self.num_q_heads, self.projection_row_stride, true, command_buffer)
    }

    fn encode_packed(
        &self,
        mut buffer: impl BufferMut<Backend = B>,
        batch_dim: u32,
        q_heads: u32,
        input_row_stride: u32,
        skip_queries: bool,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<(), B::Error> {
        let packed_row_width = (q_heads + 2 * self.num_kv_heads) * self.head_dim;
        assert!(
            input_row_stride >= packed_row_width,
            "QKV norm input row stride ({input_row_stride}) is smaller than its packed row width ({packed_row_width})"
        );

        command_buffer.push_debug_group("qkv norm");

        let kv = self.num_kv_heads;
        let query = if skip_queries {
            &None
        } else {
            &self.query
        };
        let heads = [(query, 0, q_heads), (&self.key, q_heads, kv), (&self.value, q_heads + kv, kv)];
        for (head, head_offset, head_count) in heads {
            let Some(head) = head else {
                continue;
            };
            if head_count == 0 {
                continue;
            }
            if let Some(reference) = &head.reference {
                reference.encode(
                    None::<&B::ScratchBuffer>,
                    head.scales.as_ref().unwrap(),
                    buffer.reborrow(),
                    None::<&mut B::ScratchBuffer>,
                    batch_dim,
                    self.head_dim,
                    input_row_stride,
                    head.config.epsilon,
                    head.config.scale_offset.unwrap_or(0.0),
                    head_offset,
                    head_count,
                    command_buffer,
                );
                continue;
            }
            head.kernel.encode(
                None::<&B::ScratchBuffer>,
                head.scales.as_ref(),
                buffer.reborrow(),
                batch_dim,
                input_row_stride,
                self.head_dim,
                head.config.epsilon,
                head.config.scale_offset.unwrap_or(0.0),
                head_offset,
                head_count,
                head.config.upcast_mode == UpcastMode::FullLayer,
                command_buffer,
            );
        }

        command_buffer.pop_debug_group();

        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../../unit/encodable_block/attention/qkv_norm_test.rs"]
mod tests;
