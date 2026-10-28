use stark_engine_macros::stark_config_abstract;

use crate::backends::common::InvalidInput;

pub mod linear_scaling_rope;
pub mod llama_rope;
pub mod longrope;
pub mod proportional_rope;
pub mod qwen35_mrope;
pub mod unscaled_rope;
pub mod yarn_rope;

#[stark_config_abstract(
    unscaled_rope::UnscaledRoPEConfig,
    llama_rope::LlamaRoPEConfig,
    yarn_rope::YARNRoPEConfig,
    linear_scaling_rope::LinearScalingRoPEConfig,
    longrope::LongRoPEConfig,
    proportional_rope::ProportionalRoPEConfig,
    qwen35_mrope::Qwen35MRoPEConfig
)]
pub struct RoPEConfig {
    pub base: f32,
    pub max_sequence_length: u32,
    pub head_dim: u32,
}

impl AnyRoPEConfig {
    pub fn validate(&self) -> Result<(), InvalidInput> {
        if !self.base().is_finite() || *self.base() <= 0.0 {
            return Err(InvalidInput("RoPE base must be positive and finite"));
        }
        if *self.head_dim() == 0 || !self.head_dim().is_multiple_of(2) || *self.max_sequence_length() == 0 {
            return Err(InvalidInput("RoPE width must be positive and even with nonzero context capacity"));
        }
        if let Self::Qwen35MRoPEConfig(config) = self {
            config.validate()?;
        }
        if let Self::ProportionalRoPEConfig(config) = self
            && (config.rotary_dim == 0 || !config.rotary_dim.is_multiple_of(2) || config.rotary_dim > config.head_dim)
        {
            return Err(InvalidInput("proportional rotary width must be positive, even and fit the complete head"));
        }
        Ok(())
    }
}
