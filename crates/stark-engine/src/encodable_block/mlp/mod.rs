mod dense;
mod gate_act_mul;
mod gemma4_moe;
mod qwen_moe;
pub(crate) mod routed;

pub use dense::DenseMlp;
use derive_more::Debug;
use gate_act_mul::MlpGateActMulEncodable;
use thiserror::Error;

use crate::{
    backends::common::{Backend, CommandBuffer},
    config::mlp::AnyMLPConfig,
    data_type::DataType,
    encodable_block::linear::{Linear, LinearBlockError},
    parameters::{NumericalProfile, ParameterTree},
};

pub trait Mlp<B: Backend>: Send + Sync {
    fn requires_residual_input(&self) -> bool {
        false
    }

    fn encode_with_residual(
        &self,
        input: B::ScratchBuffer,
        _residual: Option<B::ScratchBuffer>,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error> {
        self.encode(input, batch_dim, command_buffer)
    }

    fn encode(
        &self,
        input: B::ScratchBuffer,
        batch_dim: u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<B::ScratchBuffer, B::Error>;
}

#[derive(Debug, Error)]
pub enum MlpBlockError<B: Backend> {
    #[error("Backend error: {0}")]
    BackendError(#[source] B::Error),
    #[error("Linear block error: {0}")]
    LinearBlockError(#[from] LinearBlockError<B>),
    #[error("Normalization error: {0}")]
    Normalization(#[from] crate::encodable_block::normalization::NormalizationNewError<B>),
    #[error("Mixture of experts is not supported")]
    UnsupportedMixtureOfExperts,
}

impl<B: Backend> dyn Mlp<B> {
    pub fn new(
        config: &AnyMLPConfig,
        model_dimension: u32,
        hidden_dimension: u32,
        context: &B::Context,
        parameter_tree: &ParameterTree<B>,
        data_type: DataType,
    ) -> Result<(Box<dyn Mlp<B>>, Option<B::GlobalBuffer>), MlpBlockError<B>> {
        match config {
            AnyMLPConfig::Gemma4MoeConfig(config) => Ok((
                Box::new(gemma4_moe::Gemma4Moe::new(
                    config,
                    model_dimension,
                    hidden_dimension,
                    context,
                    parameter_tree,
                    data_type,
                )?),
                None,
            )),
            AnyMLPConfig::DenseMLPConfig(dense_config) => {
                let (up_projection, up_input_hadamard_factors) = <dyn Linear<B>>::new_with_input_rht(
                    model_dimension,
                    [hidden_dimension, hidden_dimension],
                    dense_config.has_up_biases,
                    context,
                    data_type,
                    &parameter_tree.subtree("up_projection"),
                )?;

                let (down_projection, down_input_preparation) = <dyn Linear<B>>::new_for_fused_input(
                    hidden_dimension,
                    [model_dimension],
                    dense_config.has_down_biases,
                    context,
                    data_type,
                    &parameter_tree.subtree("down_projection"),
                )?;

                let gate = MlpGateActMulEncodable::new(
                    context,
                    data_type,
                    dense_config.activation.clone(),
                    dense_config.gate_clipping,
                    dense_config.up_clipping,
                    hidden_dimension,
                    down_input_preparation,
                    parameter_tree.numerical_profile() == NumericalProfile::Qwen35ReferenceBF16,
                )
                .map_err(MlpBlockError::BackendError)?;

                Ok((Box::new(DenseMlp::new(up_projection, gate, down_projection)), up_input_hadamard_factors))
            },
            AnyMLPConfig::MixtureOfExpertsConfig(config) => Ok((
                Box::new(qwen_moe::QwenMoe::new(
                    config,
                    model_dimension,
                    hidden_dimension,
                    context,
                    parameter_tree,
                    data_type,
                )?),
                None,
            )),
        }
    }
}
