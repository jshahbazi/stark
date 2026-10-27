use stark_engine_macros::stark_config;

use crate::config::{
    linear::LinearConfig, normalization::NormalizationConfig, token_mixer::convolutions::SeparableCausalConvConfig,
};

#[stark_config(super::TokenMixerConfig)]
pub struct DeltaNetConfig {
    pub in_proj_config: LinearConfig,
    pub conv_config: SeparableCausalConvConfig,
    pub out_proj_config: LinearConfig,
    pub norm_config: NormalizationConfig,
    /// Match the pinned BF16 Qwen3.5 eager operation boundaries. Existing text
    /// artifacts retain their established fused precision when this is absent.
    #[serde(default)]
    pub reference_bf16: bool,

    pub num_heads: u32,
    pub num_groups: u32,
    pub head_dim: u32,
    pub value_head_dim: u32,
    pub kernel_size: u32,
}
